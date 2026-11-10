#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/drawStateObserver.h"
#include "graphics/host_gpu/renderer/cleanDraws.h"

#include "common/assert.h"
#include "common/envDefaults.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/liveAb.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"

#include <chrono>
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"
#include "graphics/host_gpu/renderer/virtualGpuClock.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "kernel/memory.h"
#include "libs/agc.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <map>
#include <mutex>
#include <unordered_map>
#include <cstddef>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

namespace Libs::Graphics {

// A compute queue that the GPU thread has processed for at least ProducerFrames presented frames without one of its
// slices ending blocked (a WAIT_REG_MEM that did not pass, a cede) is a producer: it never waits for another queue or for
// the host GPU, so nothing it does depends on when it runs. The GPU thread takes such a queue only when no other queue
// can run; the queues that wait (for labels, for host GPU results) get their work into the one Vulkan queue first, ahead
// of the producers' work, so their waits do not also wait for it. The order of runnable queues is free (the console runs
// them at once; the guest orders them with its labels, and a queue waiting for a producer's label blocks, which lets the
// producer run). GPU thread only.
constexpr uint32_t ProducerFrames = 16;
struct QueueHistory {
	uint32_t first_frame = 0; // presented frame + 1 of the queue's first slice (0: never processed)
	uint32_t block_frame = 0; // presented frame + 1 of its last slice that ended blocked (0: none)
};
static std::array<QueueHistory, 64> g_queue_history {}; // >= GuestGpu::QueueCount (checked at the selection)

static bool IsProducerQueue(uint32_t id, uint32_t frame) {
	if (id == 0) return false; // the graphics ring
	const auto& h = g_queue_history[id];
	if (h.first_frame == 0 || frame + 1u - h.first_frame < ProducerFrames) return false;
	return h.block_frame == 0 || frame + 1u - h.block_frame >= ProducerFrames;
}

// Indirect draws run on the GPU (hardware behaviour since D6b)
// with vkCmdDraw[Indexed]Indirect[Count] on the cached buffers of the guest argument records and count, so the command
// processor no longer reads GPU-written arguments back (every such read submitted and waited for the GPU). The vertex
// shader reads the SGPRs the packet names from the SPIR-V draw parameters
// (ShaderIndirectDrawInfo). The state such a packet leaves on the hardware for later draws - the named SGPRs hold its
// last record's offsets, NUM_INSTANCES its last record's instance count - is not written here; D6b showed the game
// never needs it (KYTY_UPSTREAM_GPU_INDIRECT_RECORDS=2 at the gate, open world, night, night RT, the Atsu house and a
// run from the start: stale_sgpr_draws 0 - no shader is handed a named register before the game rewrites it - and every
// inherited NUM_INSTANCES equal to the last record's; =1 reads the last record's count anyway, same result, dearer).
// Draws the GPU path does not support stay on the CPU path; each reason is logged once.
// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS (D6b; default 0): the state a GPU-executed indirect packet leaves behind.
// 1: NUM_INSTANCES after such a packet is its last record's instance count (upstream c513a4a13: a pointer to the last
//    record, read only when a later draw inherits the count; here from the cached GPU copy when there is one, else
//    guest memory), instead of the count before the packet.
// 2: draws as 0 and counts (PERF indirect_records every 5 s): packets with no SGPR destination (sh_nop) and with named
//    ones; draws whose GS-stage shader is handed a register a GPU-executed packet named and nobody rewrote since
//    (stale_sgpr_draws: on the hardware it holds the packet's last record's offset, here the older value); draws that
//    inherit NUM_INSTANCES after a GPU-executed packet, and of those the ones whose count differs from the last
//    record's (GPU copy). INDIRECT_RECORDS_VERIFY lines (first 32) name them.
uint32_t UpstreamIndirectRecordsMode() {
	static constinit Common::OnceValue<uint32_t> once;
	return once.Get([] {
		const char*    value = std::getenv("KYTY_UPSTREAM_GPU_INDIRECT_RECORDS");
		const uint32_t m     = value != nullptr && (value[0] == '1' || value[0] == '2') ? static_cast<uint32_t>(value[0] - '0')
		                                                                                    : 0u;
		if (m != 0) std::fprintf(stderr, "UPSTREAM_GPU_INDIRECT_RECORDS mode=%u\n", m);
		return m;
	});
}

namespace {

struct GpuIndirectStats {
	std::atomic<uint64_t> packets {0};
	std::atomic<uint64_t> max_draws {0};
	std::atomic<uint64_t> fallbacks {0};
	// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS=2 (totals since the start)
	std::atomic<uint64_t> sh_nop_packets {0};
	std::atomic<uint64_t> named_packets {0};
	std::atomic<uint64_t> stale_sgpr_draws {0};
	std::atomic<uint64_t> inherit_draws {0};
	std::atomic<uint64_t> inherit_differs {0};
	std::atomic<uint32_t> logged {0};
};

GpuIndirectStats& GpuIndirectCounters() {
	static GpuIndirectStats stats;
	return stats;
}

void NoteGpuIndirectFallback(const char* reason, bool indexed, uint32_t max_count,
                             uint64_t count_addr) {
	GpuIndirectCounters().fallbacks.fetch_add(1, std::memory_order_relaxed);
	static std::mutex               mutex;
	static std::vector<std::string> logged;
	std::scoped_lock                lock(mutex);
	if (logged.size() >= 32u || std::ranges::find(logged, std::string(reason)) != logged.end()) {
		return;
	}
	logged.emplace_back(reason);
	std::fprintf(stderr,
	             "DIAG_GPU_INDIRECT CPU expansion: %s (indexed=%d max=%u count_addr=0x%llx), "
	             "logged once per reason\n",
	             reason, indexed ? 1 : 0, max_count, static_cast<unsigned long long>(count_addr));
}

} // namespace

// KYTY_TRACE_MEM_CENSUS=1 only (5.11: with KYTY_TRACE_PERF it cost 7-11% fps and most 250+ ms hitches, so PERF no
// longer runs it): every ~5 s one MEM_CENSUS line with what
// holds memory, by kind, and the collectors' work since the previous line. Sizes in MiB; "a/b" = count/MiB.
static bool MemCensusRequested() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_MEM_CENSUS");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

static void MemCensusLine(RenderContext& renderer) {
	const auto mib = [](uint64_t bytes) { return static_cast<unsigned long long>(bytes >> 20u); };
	const auto num = [](uint64_t value) { return static_cast<unsigned long long>(value); };
	const auto& graphics = renderer.GetGraphics();
	uint64_t    usage = 0, budget = 0, blocks = 0, allocated = 0;
	graphics.GetDeviceLocalMemoryStats(usage, budget, blocks, allocated);
	uint64_t block_count = 0, allocation_count = 0, host_blocks = 0;
	graphics.GetMemoryCounts(block_count, allocation_count, host_blocks);
	TextureCache::Census images;
	renderer.GetTextureCache().TakeCensus(images);
	BufferCache::Census buffers;
	renderer.GetBufferCache().TakeCensus(buffers);
	auto&    table = renderer.GetBindlessTable();
	uint64_t heaps = 0;
	for (const auto& heap: table.Heaps()) {
		if (heap.entries != 0) heaps++;
	}
	const auto     host    = renderer.GetPipelineCache().TakeHostCensus();
	const uint64_t empty   = blocks > allocated ? blocks - allocated : 0;
	const uint64_t non_vma = usage > blocks ? usage - blocks : 0;
	// vram: driver usage and budget (device-local heaps), VMA blocks (count) and allocations (count), the empty
	// space inside the blocks, device memory outside VMA (driver objects), and VMA blocks in host heaps.
	std::fprintf(stderr,
	             "MEM_CENSUS frame=%u vram usage=%llu budget=%llu blocks=%llu(%llu) alloc=%llu(%llu) empty=%llu "
	             "non_vma=%llu host_blocks=%llu",
	             DebugPresentCount().load(), mib(usage), mib(budget), mib(blocks), num(block_count), mib(allocated),
	             num(allocation_count), mib(empty), mib(non_vma), mib(host_blocks));
	// vm: the process's committed host memory by kind (private allocations >= 1 GiB / 64 MiB+ / 1 MiB+ / smaller, mapped,
	// modules), MiB; what grows over a long session (24.10: 18.7 -> 28.2 GB in 24 minutes).
	const auto vm = Common::StartupTrace::ProcessVmSummary();
	std::fprintf(stderr, " | vm huge=%llu large=%llu medium=%llu small=%llu mapped=%llu image=%llu", mib(vm.private_huge),
	             mib(vm.private_large), mib(vm.private_medium), mib(vm.private_small), mib(vm.mapped), mib(vm.image));
	// img: all images, registered, bindless-pinned, GPU-written (tiled / linear / with metadata), depth, render
	// targets, unused for 160+ collector ticks, sharing a start address with another image (@ addresses).
	std::fprintf(stderr,
	             " | img all=%llu/%llu reg=%llu/%llu pinned=%llu/%llu gpu_tiled=%llu/%llu gpu_linear=%llu/%llu "
	             "gpu_meta=%llu/%llu depth=%llu/%llu",
	             num(images.all.count), mib(images.all.bytes), num(images.registered.count),
	             mib(images.registered.bytes), num(images.pinned.count), mib(images.pinned.bytes),
	             num(images.gpu_tiled.count), mib(images.gpu_tiled.bytes), num(images.gpu_linear.count),
	             mib(images.gpu_linear.bytes), num(images.gpu_metadata.count), mib(images.gpu_metadata.bytes),
	             num(images.depth.count), mib(images.depth.bytes));
	std::fprintf(stderr, " rt=%llu/%llu old160=%llu/%llu same_addr=%llu/%llu@%llu unreg=%llu/%llu assoc=%llu scratch=%llu",
	             num(images.render_target.count), mib(images.render_target.bytes), num(images.lru_old.count),
	             mib(images.lru_old.bytes), num(images.same_address.count), mib(images.same_address.bytes),
	             num(images.same_address_groups), num(images.unregistered.count), mib(images.unregistered.bytes),
	             num(images.associations), mib(images.scratch_bytes));
	// buf: cached guest-memory buffers, those with GPU-written bytes, mapped (BAR), largest.
	std::fprintf(stderr, " | buf n=%llu/%llu gpu_dirty=%llu/%llu bar=%llu/%llu max=%llu",
	             num(buffers.count), mib(buffers.bytes),
	             num(buffers.gpu_dirty), mib(buffers.gpu_dirty_bytes), num(buffers.bar), mib(buffers.bar_bytes),
	             mib(buffers.largest));
	std::fprintf(stderr, " | fixed bda_table=%llu staging=%llu stream=%llu download=%llu device=%llu other=%llu bindless=%llu",
	             mib(buffers.bda_table_bytes), mib(buffers.staging_bytes), mib(buffers.stream_bytes),
	             mib(buffers.download_bytes), mib(buffers.device_bytes), mib(buffers.other_fixed_bytes),
	             mib(table.BufferBytes()));
	const auto& gc = images.gc;
	std::fprintf(stderr,
	             " | gc_tex runs=%llu examined=%llu freed=%llu/%llu downloaded=%llu/%llu skip_unreg=%llu skip_assoc=%llu "
	             "skip_pinned=%llu skip_tiled=%llu skip_nopress=%llu skip_nodl=%llu skip_dlfail=%llu",
	             num(gc.runs), num(gc.examined), num(gc.freed), mib(gc.freed_bytes), num(gc.downloaded),
	             mib(gc.downloaded_bytes), num(gc.skip_unregistered), num(gc.skip_association), num(gc.skip_pinned),
	             num(gc.skip_gpu_tiled), num(gc.skip_not_pressured), num(gc.skip_no_download),
	             num(gc.skip_download_failed));
	// KYTY_VRAM_GC_FIX: downloads waiting to be published, evictions cancelled by a use, pinned images freed,
	// depth/stencil images kept, downloads left for a later run (per-run byte limit).
	std::fprintf(stderr, " pending=%llu cancelled=%llu freed_pinned=%llu skip_stencil=%llu skip_dl_budget=%llu",
	             num(images.pending_evictions), num(gc.evict_cancelled), num(gc.freed_pinned), num(gc.skip_stencil),
	             num(gc.skip_download_budget));
	std::fprintf(stderr, " | gc_buf runs=%llu retired=%llu/%llu kept_bda=%llu dirty_skip=%llu", num(buffers.gc_runs),
	             num(buffers.gc_retired), mib(buffers.gc_retired_bytes), num(buffers.gc_kept_bda),
	             num(buffers.gc_dirty_skipped));
	// KYTY_VRAM_BUDGET (last texture collector run): the target, the usage the collectors counted, memory freed in the
	// last ticks (images, buffers) and evictions made before new images/buffers were created.
	std::fprintf(stderr,
	             " | budget target=%llu used=%llu freed_recent=%llu/%llu alloc_evicted=%llu/%llu alloc_retired=%llu/%llu",
	             mib(images.budget_target), mib(images.budget_used), mib(images.recently_freed),
	             mib(buffers.recently_retired), num(images.gc.alloc_evicted), mib(images.gc.alloc_evicted_bytes),
	             num(buffers.alloc_retired), mib(buffers.alloc_retired_bytes));
	// host: shader modules kept for the session, pipelines, driver pipeline cache loaded / last saved, bindless
	// descriptor slots (pinned images) and live heaps.
	std::fprintf(stderr,
	             " | host modules=%llu gfx_pipes=%llu cs_pipes=%llu pcache_loaded=%llu pcache_saved=%llu "
	             "bindless_slots=%zu heaps=%llu\n",
	             num(host.shader_modules), num(host.graphics_pipelines), num(host.compute_pipelines),
	             mib(host.cache_loaded), mib(host.cache_saved),
	             table.descriptor_slots.size(), num(heaps));
}

// KYTY_TRACE_MEM_CENSUS=1 without KYTY_TRACE_PERF: the census line on its own every ~5 s.
static void MemCensusReport(RenderContext& renderer) {
	if (PerfEnabled() || !MemCensusRequested()) return;
	static auto window_start = std::chrono::steady_clock::now();
	const auto  now          = std::chrono::steady_clock::now();
	if (now - window_start < std::chrono::seconds(5)) return;
	window_start = now;
	MemCensusLine(renderer);
}

static void ReportGpuStageCheck();

static void PerfReport(RenderContext& renderer) {
	if (!PerfEnabled()) return;
	static auto window_start = std::chrono::steady_clock::now();
	const auto  now          = std::chrono::steady_clock::now();
	if (now - window_start < std::chrono::seconds(5)) return;
	const auto window_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start).count();
	window_start         = now;
	static const char* names[] = {"process", "fence_wait", "read_memory", "eager_hit", "eager_wait", "sync_read"};
	std::fprintf(stderr, "PERF window_ms=%lld", static_cast<long long>(window_ms));
	for (uint32_t i = 0; i < static_cast<uint32_t>(PerfKind::Count); i++) {
		std::fprintf(stderr, " %s=%llums/%llu", names[i],
		             static_cast<unsigned long long>(PerfNanos()[i].exchange(0) / 1000000u),
		             static_cast<unsigned long long>(PerfCalls()[i].exchange(0)));
	}
	std::fprintf(stderr, "\n");
	CpuProbeReport();
	BufferCache::ReportBarCopyPerf();
	{
		// Device-local memory of this process: driver-reported usage, the share VMA allocated (blocks)
		// and the budget, in MiB (KYTY_TRACE_PERF). images/buffers: guest bytes of the registered images and
		// cached buffers (their collectors replace UsedMemory() with the process' device usage at every run).
		const auto& graphics = renderer.GetGraphics();
		uint64_t    usage = 0, budget = 0, blocks = 0, allocated = 0;
		graphics.GetDeviceLocalMemoryStats(usage, budget, blocks, allocated);
		std::fprintf(stderr, "PERF vram usage=%lluMiB vma_blocks=%lluMiB vma_alloc=%lluMiB budget=%lluMiB images=%lluMiB buffers=%lluMiB\n",
		             static_cast<unsigned long long>(usage >> 20u), static_cast<unsigned long long>(blocks >> 20u),
		             static_cast<unsigned long long>(allocated >> 20u), static_cast<unsigned long long>(budget >> 20u),
		             static_cast<unsigned long long>(renderer.GetTextureCache().RegisteredBytes() >> 20u),
		             static_cast<unsigned long long>(renderer.GetBufferCache().RegisteredBytes() >> 20u));
		std::fprintf(stderr, "PERF device_memory %s\n", TakeDeviceMemoryCallStats().c_str());
		{
			auto&             s     = MasterSemaphore::GetStats();
			const char* const n[4] = {"refresh", "wait_before", "wait_after", "other"};
			std::fprintf(stderr, "PERF semaphore");
			for (uint32_t i = 0; i < 4; i++) {
				std::fprintf(stderr, " %s=%llu/%.1fms", n[i], static_cast<unsigned long long>(s.queries[i].exchange(0)),
				             static_cast<double>(s.query_ns[i].exchange(0)) / 1e6);
			}
			std::fprintf(stderr, " blocks=%llu/%.1fms\n", static_cast<unsigned long long>(s.blocks.exchange(0)),
			             static_cast<double>(s.block_ns.exchange(0)) / 1e6);
		}
		std::fprintf(stderr, "PERF image_unregister %s\n", TakeImageUnregisterStats().c_str());
	}
	{
		static const char* reasons[] = {"other",    "fault_game",    "fault_gpu", "srt_walker",
		                                "dcc_meta", "indirect_args", "invalidate"};
		std::fprintf(stderr, "PERF readers");
		for (uint32_t i = 0; i < static_cast<uint32_t>(PerfReason::Count); i++) {
			auto& stats = PerfReasons()[i];
			std::fprintf(stderr, " %s=%llu/%llu/%llums", reasons[i], static_cast<unsigned long long>(stats.calls.exchange(0)),
			             static_cast<unsigned long long>(stats.drains.exchange(0)),
			             static_cast<unsigned long long>(stats.drain_nanos.exchange(0) / 1000000u));
		}
		std::fprintf(stderr, "\n");
		// Per reader: calls run on the GPU thread itself / ms callers waited for the GPU thread to pick the
		// command up / ms the GPU thread spent running it (drains included).
		std::fprintf(stderr, "PERF sync_time");
		for (uint32_t i = 0; i < static_cast<uint32_t>(PerfReason::Count); i++) {
			auto& stats = PerfReasons()[i];
			std::fprintf(stderr, " %s=%llu/%llums/%llums", reasons[i],
			             static_cast<unsigned long long>(stats.gpu_thread_calls.exchange(0)),
			             static_cast<unsigned long long>(stats.queue_nanos.exchange(0) / 1000000u),
			             static_cast<unsigned long long>(stats.exec_nanos.exchange(0) / 1000000u));
		}
		std::fprintf(stderr, "\n");
		// Contended tracker region locks and tracker protections, GPU thread / other threads: count/ms.
		auto& locks = PerfLocks();
		std::fprintf(stderr, "PERF locks region_wait_gpu=%llu/%llums region_wait_other=%llu/%llums protect_gpu=%llu/%llums "
		                     "protect_other=%llu/%llums\n",
		             static_cast<unsigned long long>(locks.region_waits[0].exchange(0)),
		             static_cast<unsigned long long>(locks.region_wait_nanos[0].exchange(0) / 1000000u),
		             static_cast<unsigned long long>(locks.region_waits[1].exchange(0)),
		             static_cast<unsigned long long>(locks.region_wait_nanos[1].exchange(0) / 1000000u),
		             static_cast<unsigned long long>(locks.protects[0].exchange(0)),
		             static_cast<unsigned long long>(locks.protect_nanos[0].exchange(0) / 1000000u),
		             static_cast<unsigned long long>(locks.protects[1].exchange(0)),
		             static_cast<unsigned long long>(locks.protect_nanos[1].exchange(0) / 1000000u));
		// The same contended waits by lock (GPU thread / other threads: count/ms).
		std::fprintf(stderr, "PERF lock_sites");
		constexpr const char* lock_sites[PerfLockSiteCount] = {"region", "buffer_lookup", "texture_cache"};
		for (uint32_t site = 0; site < PerfLockSiteCount; site++) {
			std::fprintf(stderr, " %s=%llu/%llums,%llu/%llums", lock_sites[site],
			             static_cast<unsigned long long>(locks.site_waits[site][0].exchange(0)),
			             static_cast<unsigned long long>(locks.site_wait_nanos[site][0].exchange(0) / 1000000u),
			             static_cast<unsigned long long>(locks.site_waits[site][1].exchange(0)),
			             static_cast<unsigned long long>(locks.site_wait_nanos[site][1].exchange(0) / 1000000u));
		}
		std::fprintf(stderr, "\n");
		PerfReportTcLockPcs();
		renderer.GetCommandScheduler().ReportSubmitPerf();
		renderer.GetBufferCache().ReportCallerWaitPerf();
		renderer.GetBufferCache().ReportCpWriteThroughPerf();
		renderer.GetBufferCache().ReportProtectPerf();
		renderer.GetTextureCache().ReportDccGpuAllPerf();
		ReportShaderReusePerf();
		ShaderRecompiler::IR::ReportNullShareJournalPerf(); // the null share journal
		renderer.GetBindlessTable().ReportTablesPerf();
		renderer.GetBindlessTable().ReportClassesPerf(); // KYTY_PERF_BINDLESS_CLASSES
		StageF::ReportLookupPerf(); // the resolve thread's lookups
		Common::WorkaroundHits::Report(); // stage 2 pack 1: does the game reach these approximations?
	}
	auto& stats = GpuIndirectCounters();
	std::fprintf(stderr, "PERF gpu_indirect packets=%llu max_draws=%llu cpu_fallback=%llu\n",
	             static_cast<unsigned long long>(stats.packets.exchange(0)),
	             static_cast<unsigned long long>(stats.max_draws.exchange(0)),
	             static_cast<unsigned long long>(stats.fallbacks.exchange(0)));
	if (UpstreamIndirectRecordsMode() == 2u) {
		std::fprintf(stderr,
		             "PERF indirect_records sh_nop=%llu named=%llu stale_sgpr_draws=%llu inherit=%llu inherit_differs=%llu\n",
		             static_cast<unsigned long long>(stats.sh_nop_packets.load()),
		             static_cast<unsigned long long>(stats.named_packets.load()),
		             static_cast<unsigned long long>(stats.stale_sgpr_draws.load()),
		             static_cast<unsigned long long>(stats.inherit_draws.load()),
		             static_cast<unsigned long long>(stats.inherit_differs.load()));
	}
	std::vector<std::pair<uint64_t, uint32_t>> pages;
	{
		std::scoped_lock lock(PerfPageMutex());
		pages.assign(PerfReadPages().begin(), PerfReadPages().end());
		PerfReadPages().clear();
	}
	std::ranges::sort(pages, [](const auto& a, const auto& b) { return a.second > b.second; });
	std::fprintf(stderr, "PERF read_pages distinct=%zu", pages.size());
	for (size_t i = 0; i < pages.size() && i < 10u; i++) {
		std::fprintf(stderr, " [0x%llx0000 size=0x%llx x%u]",
		             static_cast<unsigned long long>(pages[i].first & 0xffffffffffffull),
		             static_cast<unsigned long long>(pages[i].first >> 48u), pages[i].second);
	}
	std::fprintf(stderr, "\n");
	ReportGpuStageCheck();
	if (MemCensusRequested()) MemCensusLine(renderer);
}

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
// The thread that owns the renderer (GuestGpu::IsGpuThread), and its GPU.
static thread_local GuestGpu*         g_gpu_state         = nullptr;
// The command processor thread.
static thread_local bool              g_cp_thread         = false;
// The GPU whose stream this thread makes operations for. Off it (the tests drive a CommandProcessor on their own thread)
// operations are not numbered, there is no journal and labels land when their packet is translated.
static thread_local GuestGpu*         g_stream_gpu        = nullptr;

static GuestGpu* StreamGpu() noexcept {
	return g_stream_gpu;
}


// KYTY_TRACE_PERF: what the pipeline did and where its threads waited. Several threads: atomic.
namespace {
struct GpuStageStats {
	std::atomic<uint64_t> operations {0}, draws {0}, dispatches {0}, snapshot_waits {0}, copied_bytes {0}, journal_max {0};
	// Where the threads wait (count, ns).
	std::array<std::atomic<uint64_t>, static_cast<size_t>(GpuDrain::Count)> drains {}, drain_ns {};
	std::atomic<uint64_t> journal_full {0}, journal_full_ns {0}, journal_reads {0}, journal_scanned {0};
	std::atomic<uint64_t> ring_full {0}, ring_full_ns {0}, slot_waits_ns {0}, resolved_full {0}, resolved_full_ns {0}, resolve_idle_ns {0};
	std::atomic<uint64_t> execution_idle {0}, execution_idle_ns {0}, execution_sleeps {0}, commands {0};
};
GpuStageStats g_stage_stats;
} // namespace

static uint64_t StageNowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Every ~5 s under KYTY_TRACE_PERF. copied_kb: register bytes the snapshots copied. The second line: where the threads
// waited, as count/ms: the command processor for the execution thread by reason, the journal, the ring and a snapshot slot;
// the execution thread for work.
static void ReportGpuStageCheck() {
	if (!g_gpu_stage_stats) return;
	auto& s = g_stage_stats;
	std::fprintf(stderr,
	             "PERF gpu_pipeline frame=%u operations=%llu draws=%llu dispatches=%llu snapshot_waits=%llu copied_kb=%llu "
	             "journal_max=%llu\n",
	             DebugPresentCount().load(std::memory_order_relaxed),
	             static_cast<unsigned long long>(s.operations.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.draws.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.dispatches.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.snapshot_waits.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(s.copied_bytes.exchange(0, std::memory_order_relaxed) / 1024u),
	             static_cast<unsigned long long>(s.journal_max.exchange(0, std::memory_order_relaxed)));
	const auto take    = [](std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.exchange(0, std::memory_order_relaxed)); };
	const auto take_ms = [](std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.exchange(0, std::memory_order_relaxed) / 1000000u); };
	static constexpr const char* reasons[] = {"fault", "predication", "cond_guard", "direct", "journal", "idle"};
	static_assert(std::size(reasons) == static_cast<size_t>(GpuDrain::Count));
	std::fprintf(stderr, "PERF gpu_pipeline_waits drains");
	for (size_t i = 0; i < s.drains.size(); i++) {
		std::fprintf(stderr, " %s=%llu/%llums", reasons[i], take(s.drains[i]), take_ms(s.drain_ns[i]));
	}
	std::fprintf(stderr,
	             " journal_full=%llu/%llums journal_reads=%llu journal_scanned=%llu ring_full=%llu/%llums slot_wait=%llums "
	             "resolved_full=%llu/%llums resolve_idle=%llums execution_idle=%llu/%llums execution_sleeps=%llu commands=%llu\n",
	             take(s.journal_full), take_ms(s.journal_full_ns), take(s.journal_reads), take(s.journal_scanned), take(s.ring_full),
	             take_ms(s.ring_full_ns), take_ms(s.slot_waits_ns), take(s.resolved_full), take_ms(s.resolved_full_ns),
	             take_ms(s.resolve_idle_ns), take(s.execution_idle), take_ms(s.execution_idle_ns), take(s.execution_sleeps),
	             take(s.commands));
}

// Reads memory other threads may write, with the widest aligned accesses.
// Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
static void CopyFromGuest(void* destination, const volatile void* source, size_t size) {
	auto*      out     = static_cast<uint8_t*>(destination);
	const auto address = reinterpret_cast<uint64_t>(source);
	if ((address & 7u) == 0 && (size & 7u) == 0) {
		const auto* words = static_cast<const volatile uint64_t*>(source);
		for (size_t i = 0; i < size / 8u; i++) {
			const uint64_t word = words[i];
			std::memcpy(out + i * 8u, &word, 8u);
		}
		return;
	}
	if ((address & 3u) == 0 && (size & 3u) == 0) {
		const auto* words = static_cast<const volatile uint32_t*>(source);
		for (size_t i = 0; i < size / 4u; i++) {
			const uint32_t word = words[i];
			std::memcpy(out + i * 4u, &word, 4u);
		}
		return;
	}
	const auto* bytes = static_cast<const volatile uint8_t*>(source);
	for (size_t i = 0; i < size; i++) {
		out[i] = bytes[i];
	}
}

// labels at completion (default on, =0 off, live A/B; A1 stage 1).
// Live A/B 20261014-182251 (82 cycles): fps -0.01% (se 0.21%), same GPU-thread busy and fence time; on-from-start run
// 20261014-182029: loads and plays, kimono correct; labels reach the guest ~4 ms (max ~15 ms) after recording. An end-of-pipe label
// (RELEASE_MEM / EVENT_WRITE_EOP data) reaches guest memory when the host GPU has finished the work recorded before it,
// as on the console, instead of when the GPU thread translates the packet. The guest CPU then never sees "done" for work
// the host GPU has not done (needed once the host GPU reads guest memory directly). How:
//   - the command processor records the value as pending (GuestGpu::NoteDeferredWrite) and the GPU thread queues a
//     priority operation on the open tick;
//   - the priority thread, once the tick is finished, sends the publish to the GPU thread (commands run at every packet
//     boundary and wake an idle GPU thread); it writes guest memory exactly as the immediate path did, then raises the
//     label's interrupt, so an interrupt never precedes its label;
//   - the GPU thread's own WAIT_REG_MEM reads pending values (all guest queues record into one in-order host queue, so a
//     wait recorded after the label is ordered after its work on the host): cross-queue waits pass as before.
// Every slice that made progress ends with a submit, so a pending label's work is always submitted. Only the GPU thread
// touches the journal. Measured lag label -> host completion before this: p50 ~2 ms,
// p99 ~12 ms in play.
namespace {
struct LabelStats {
	std::atomic<uint64_t> deferred {0}, published {0}, overlay_hits {0}, lag_ns {0}, max_lag_ns {0}, max_pending {0};
};
LabelStats g_label_stats;

int64_t LabelNowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}


void ReportLabelStats() {
	if (!Common::EnvRead::PerfLines()) return;
	static int64_t window_start = 0;
	const auto     now          = LabelNowNs();
	if (window_start == 0) window_start = now;
	if (now - window_start < 5'000'000'000ll) return;
	window_start         = now;
	auto&          s     = g_label_stats;
	const uint64_t pub   = s.published.exchange(0, std::memory_order_relaxed);
	const uint64_t lag   = s.lag_ns.exchange(0, std::memory_order_relaxed);
	std::fprintf(stderr,
	             "PERF label_at_completion deferred=%llu published=%llu overlay_hits=%llu lag_ms_avg=%.2f lag_ms_max=%.2f "
	             "max_pending=%llu\n",
	             static_cast<unsigned long long>(s.deferred.exchange(0, std::memory_order_relaxed)),
	             static_cast<unsigned long long>(pub),
	             static_cast<unsigned long long>(s.overlay_hits.exchange(0, std::memory_order_relaxed)),
	             pub == 0 ? 0.0 : static_cast<double>(lag) / static_cast<double>(pub) / 1e6,
	             static_cast<double>(s.max_lag_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
	             static_cast<unsigned long long>(s.max_pending.exchange(0, std::memory_order_relaxed)));
}

// Defers an end-of-pipe label (labels at completion); the journal holds it since its packet. Only on the GPU thread.
void DeferLabel(CommandScheduler& scheduler, RenderContext& renderer, int event_id, uint64_t sequence, uint64_t dst,
                uint64_t value, uint32_t size, bool with_interrupt, uint32_t interrupt_context_id) {
	GuestGpu* gpu = g_gpu_state;
	g_label_stats.deferred.fetch_add(1, std::memory_order_relaxed);
	const int64_t recorded_ns = LabelNowNs();
	// Priority thread, once the host GPU finished the open tick: hand the write to the GPU thread.
	scheduler.DeferPriorityOperation([gpu, &renderer, event_id, sequence, dst, value, size, with_interrupt,
	                                  interrupt_context_id, recorded_ns] {
		if (gpu->IsStopping()) return;
		const auto publish = [gpu, &renderer, event_id, sequence, dst, value, size, with_interrupt, interrupt_context_id,
		                      recorded_ns] {
			// The immediate path's write (the write-through or a plain store). In a direct range
			// (KYTY_PERF_DIRECT_RANGES) a plain store: the work before the label is done, which is when the console's CP
			// writes it, and an update recorded now would reach guest memory only with a later submit.
			if (renderer.GetBufferCache().InDirectRange(dst, size) ||
			    !renderer.GetBufferCache().TryCpWriteThrough(dst, &value, size)) {
				std::memcpy(reinterpret_cast<void*>(dst), &value, size);
			}
			if (sequence != 0) gpu->MarkPublished(sequence); // (0: no journal, a test driving a processor here)
			PublishCpuDirtyRange(dst, size); // BDA sync epoch
			const auto lag = static_cast<uint64_t>(std::max<int64_t>(LabelNowNs() - recorded_ns, 0));
			g_label_stats.published.fetch_add(1, std::memory_order_relaxed);
			g_label_stats.lag_ns.fetch_add(lag, std::memory_order_relaxed);
			if (lag > g_label_stats.max_lag_ns.load(std::memory_order_relaxed)) {
				g_label_stats.max_lag_ns.store(lag, std::memory_order_relaxed);
			}
			ReportLabelStats();
			if (with_interrupt) renderer.TriggerInterrupt(event_id, interrupt_context_id);
		};
		gpu->SendCommand(publish);
	});
}
} // namespace

// The guest's indirect draw records match Vulkan's vk::DrawIndirectCommand / vk::DrawIndexedIndirectCommand byte for
// byte (upstream moved the local record structs out; render.h's DrawIndirectArgs is the GPU-read record range).

// The draw-state observer's shadows are kept while its one reader is on: KYTY_TRACE_CLEAN_DRAWS (its observer_clean
// column). Read once.
static bool DrawStateObserverOn() {
	static const bool on = CleanDraws::On();
	return on;
}

static bool GraphicsRunDebugDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() &&
	       Config::GetPrintfDirection() != Config::LogDirection::Silent;
}

GuestGpu::GuestGpu(RenderContext& renderer)
    : m_renderer(renderer), m_operations(OperationQueueSize), m_resolved(ResolvedQueueSize),
      m_neutral_state {&m_neutral_context, &m_neutral_user_config, &m_neutral_shaders} {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GraphicsInitJmpTables();
	m_pending_writes = std::make_unique<GuestWrite[]>(PendingWriteCapacity);
	m_neutral_context.Reset();
	m_neutral_user_config.Reset();
	m_neutral_shaders.Reset();
	g_gpu_stage_stats = PerfEnabled();
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	m_executor = std::jthread([this] { ExecutionThread(); });
	m_resolver = std::jthread([this] { ResolveThread(); });
	m_thread   = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_epoch.fetch_add(1, std::memory_order_release);
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		if (DrawStateObserverOn()) DrawStateObserver::Invalidate(DrawStateObserver::Cause(DrawStateObserver::CauseInvalidate, DrawStateObserver::InvalidateCommand)); // the draw-state observer (DrawStateObserverOn)
		command();
		return;
	}
	if (g_cp_thread) {
		// In operation order: the operations emitted so far may write what the command reads.
		m_gfx_cp->EmitCallback([operation = std::move(command)]() mutable { operation(); });
		return;
	}
	PostCommand(std::move(command));
}

void GuestGpu::PostCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_acq_rel);
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	RingExecution();
}

void GuestGpu::Wake() {
	Common::LockGuard lock(m_queue_mutex);
	m_wake_pending = true;
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread());
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
			m_commands_running = true;
		}
		if (DrawStateObserverOn()) DrawStateObserver::Invalidate(DrawStateObserver::Cause(DrawStateObserver::CauseInvalidate, DrawStateObserver::InvalidateCommand)); // the draw-state observer (DrawStateObserverOn)
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	if (g_cp_thread) {
		// A read for a page the command processor faulted on: the operations emitted so far may write that memory.
		ExecuteSync(std::move(command));
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}


void GuestGpu::Submit(std::span<const uint32_t> draw_commands,
                      std::span<const uint32_t> constant_commands) {
	if (draw_commands.empty()) {
		return;
	}
	Submission submission;
	submission.type              = SubmissionType::Graphics;
	submission.queue_id          = 0;
	submission.commands          = draw_commands;
	submission.constant_commands = constant_commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Compute;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	{
		SyncTimeline::WaitScope timeline(SyncTimeline::Site::Suspend, 0, 0, 0); // KYTY_TRACE_SYNC_TIMELINE
		m_suspend_point_ready->acquire();
	}
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	EXIT_IF(queue_id >= QueueCount);
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void GuestGpu::WaitForSnapshot(uint64_t sequence) {
	if (sequence <= m_executed.Load()) {
		return;
	}
	if (g_gpu_stage_stats) g_stage_stats.snapshot_waits.fetch_add(1, std::memory_order_relaxed);
	const auto start = g_gpu_stage_stats ? StageNowNs() : 0;
	m_executed.WaitFor(sequence);
	if (g_gpu_stage_stats) g_stage_stats.slot_waits_ns.fetch_add(StageNowNs() - start, std::memory_order_relaxed);
}

// Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
void GuestGpu::EmitOperation(GpuOperation&& operation) {
	if (g_gpu_stage_stats) {
		if (!m_operations.TryPush(operation)) {
			const auto start = StageNowNs();
			m_operations.Push(std::move(operation));
			g_stage_stats.ring_full.fetch_add(1, std::memory_order_relaxed);
			g_stage_stats.ring_full_ns.fetch_add(StageNowNs() - start, std::memory_order_relaxed);
		}
	} else {
		m_operations.Push(std::move(operation));
	}
}

void GuestGpu::RingExecution() noexcept {
	// After the store that makes the work visible, a seq_cst fence or RMW: see WaitForOperations.
	if (m_execution_sleeping.load(std::memory_order_seq_cst)) {
		m_execution_bell.fetch_add(1, std::memory_order_seq_cst);
		m_execution_bell.notify_one();
	}
}

void GuestGpu::ExecuteSync(Common::UniqueFunction<void>&& command) {
	m_gfx_cp->EmitCallback([operation = std::move(command)]() mutable { operation(); });
	WaitForExecution(GpuDrain::Fault);
}

void GuestGpu::WaitForExecution(GpuDrain reason) {
	if (!g_gpu_stage_stats || m_executed.Load() >= m_emitted) {
		m_executed.WaitFor(m_emitted);
		return;
	}
	const auto start = StageNowNs();
	m_executed.WaitFor(m_emitted);
	const auto index = static_cast<size_t>(reason);
	g_stage_stats.drains[index].fetch_add(1, std::memory_order_relaxed);
	g_stage_stats.drain_ns[index].fetch_add(StageNowNs() - start, std::memory_order_relaxed);
}

void GuestGpu::BecomeRendererOwner() {
	g_gpu_thread      = true;
	g_gpu_state       = this;
	PerfIsGpuThread() = true;
	PerfSamplerStartForCurrentThread();
	Common::StartupTrace::NoteGpuThread(); // KYTY_TRACE_STARTUP: its CPU time per second
}

void GuestGpu::RunCommands() {
	if (g_gpu_stage_stats) g_stage_stats.commands.fetch_add(1, std::memory_order_relaxed);
	ProcessCommands();
	Common::LockGuard lock(m_queue_mutex);
	m_commands_running = false;
	// A command may have written what a blocked queue waits for (a deferred label's publish, labels at completion, is
	// one): the queues retry now. A wait still unmet blocks the queue again.
	for (auto& queue: m_queues) {
		if (!queue.empty()) {
			queue.front().blocked = false;
		}
	}
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
	if (m_commands.empty() && m_submission_count == 0 && !m_processing) {
		m_idle.SignalAll();
	}
}

bool GuestGpu::WaitForOperations() {
	m_executed.Notify();
	Common::SpinWait spin;
	for (;;) {
		if (!m_resolved.Empty()) {
			return true;
		}
		if (m_pending_commands.load(std::memory_order_acquire) != 0) {
			RunCommands();
			continue;
		}
		if (m_execution_stop.load(std::memory_order_acquire)) {
			return false;
		}
		if (spin.Spin()) {
			continue;
		}
		// The producers store their work, then read the flag; this thread stores the flag, then reads their work.
		if (g_gpu_stage_stats) g_stage_stats.execution_sleeps.fetch_add(1, std::memory_order_relaxed);
		m_execution_sleeping.store(true, std::memory_order_seq_cst);
		const auto bell = m_execution_bell.load(std::memory_order_seq_cst);
		if (m_resolved.Empty() && m_pending_commands.load(std::memory_order_seq_cst) == 0 &&
		    !m_execution_stop.load(std::memory_order_seq_cst)) {
			m_execution_bell.wait(bell, std::memory_order_seq_cst);
		}
		m_execution_sleeping.store(false, std::memory_order_relaxed);
		spin = Common::SpinWait {};
	}
}

// Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
void GuestGpu::ExecutionThread() {
	KYTY_PROFILER_THREAD("Thread_GpuExecute");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Execution);
	BecomeRendererOwner();
	// Waking the command processor costs a fence: publish every operation, but only look for a waiter now and then and
	// before waiting.
	constexpr uint32_t NotifyInterval = 16;
	uint32_t           unnotified     = 0;
	// The renderer's busy time, in batches.
	constexpr uint32_t                     BusyBatch = 256;
	uint32_t                               in_batch  = 0;
	std::optional<PerfScope>               perf_process;
	std::optional<Common::LiveAb::BusyScope> live_busy; // KYTY_LIVE_AB: GPU-thread busy time of the phase
	for (;;) {
		auto operation = m_resolved.TryPop();
		if (!operation) {
			live_busy.reset();
			perf_process.reset();
			in_batch = 0;
			const auto idle_start = g_gpu_stage_stats ? StageNowNs() : 0;
			if (!WaitForOperations()) {
				break;
			}
			if (g_gpu_stage_stats) {
				g_stage_stats.execution_idle.fetch_add(1, std::memory_order_relaxed);
				g_stage_stats.execution_idle_ns.fetch_add(StageNowNs() - idle_start, std::memory_order_relaxed);
			}
			continue;
		}
		if (!live_busy) {
			perf_process.emplace(PerfKind::Process);
			live_busy.emplace();
		}
		operation->processor->Execute(*operation);
		{
			// KYTY_TEST_PAD_RECORD / KYTY_TEST_PAD_REPLAY: the gameplay-start anchor (debugSample.h GameplayStartNs).
			static uint32_t last_present = 0, draws_in_frame = 0, frames_in_row = 0;
			const auto      kind         = operation->kind;
			if (kind == GpuOperationKind::DrawIndex || kind == GpuOperationKind::DrawAuto || kind == GpuOperationKind::DrawIndirect) {
				draws_in_frame++;
			}
			const uint32_t present = DebugPresentCount().load(std::memory_order_relaxed);
			if (present != last_present) {
				frames_in_row  = draws_in_frame > 3000u ? frames_in_row + 1u : 0u;
				draws_in_frame = 0;
				last_present   = present;
				if (frames_in_row == 8u && GameplayStartNs().load(std::memory_order_relaxed) == 0) {
					GameplayStartNs().store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_release);
					std::fprintf(stderr, "GAMEPLAY_START frame=%u\n", present);
				}
			}
		}
		const auto sequence = operation->sequence;
		// Destroy the callback before the command processor may reuse what it references.
		operation.reset();
		m_executed.Publish(sequence);
		if (m_pending_commands.load(std::memory_order_acquire) != 0) {
			RunCommands();
		}
		if (++unnotified >= NotifyInterval) {
			m_executed.Notify();
			unnotified = 0;
		}
		if (++in_batch >= BusyBatch) {
			live_busy.reset();
			perf_process.reset();
			in_batch = 0;
		}
	}
	live_busy.reset();
	perf_process.reset();
	m_executed.Notify();
	g_gpu_state  = nullptr;
	g_gpu_thread = false;
}

static const char* IndirectDrawLayout(const IndirectDrawRegisters& registers, ShaderIndirectDrawInfo& sgprs);

// Based on TheCruZ/KytyPS5-GTA daaf45201 (graphicsRun.cpp), GPL-2.0.
void GuestGpu::ResolveThread() {
	KYTY_PROFILER_THREAD("Thread_GpuResolve");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Resolve);
	auto& executor = m_renderer.GetRenderExecutor();
	for (;;) {
		auto operation = m_operations.TryPop();
		if (!operation) {
			const auto idle_start = g_gpu_stage_stats ? StageNowNs() : 0;
			operation             = m_operations.Pop();
			if (!operation) {
				break;
			}
			if (g_gpu_stage_stats) g_stage_stats.resolve_idle_ns.fetch_add(StageNowNs() - idle_start, std::memory_order_relaxed);
		}
		// The register snapshots stay valid until the operation executed.
		const auto& state = operation->state;
		switch (operation->kind) {
			case GpuOperationKind::DrawIndex:
			case GpuOperationKind::DrawAuto:
				operation->resolved = executor.ResolveDrawAhead(*state.context, *state.user_config, *state.shaders, {});
				break;
			case GpuOperationKind::DrawIndirect: {
				// The layout a GPU indirect draw gets (none: the packet expands on the CPU, with patched registers).
				ShaderIndirectDrawInfo indirect_draw;
				if (IndirectDrawLayout(operation->draw_indirect.registers, indirect_draw) != nullptr) indirect_draw = {};
				operation->resolved = executor.ResolveDrawAhead(*state.context, *state.user_config, *state.shaders, indirect_draw);
				break;
			}
			case GpuOperationKind::DispatchDirect:
			case GpuOperationKind::DispatchIndirect: {
				const auto& dispatch = operation->dispatch;
				operation->resolved  = executor.ResolveDispatchAhead(
				    *state.shaders, dispatch.thread_group_x, dispatch.thread_group_y, dispatch.thread_group_z, dispatch.mode,
				    operation->kind == GpuOperationKind::DispatchIndirect, operation->processor->IsAsyncComputeQueue());
				break;
			}
			default: break;
		}
		if (g_gpu_stage_stats) {
			if (!m_resolved.TryPush(*operation)) {
				const auto start = StageNowNs();
				m_resolved.Push(std::move(*operation));
				g_stage_stats.resolved_full.fetch_add(1, std::memory_order_relaxed);
				g_stage_stats.resolved_full_ns.fetch_add(StageNowNs() - start, std::memory_order_relaxed);
			}
		} else {
			m_resolved.Push(std::move(*operation));
		}
		RingExecution();
	}
}

void GuestGpu::StopExecution() {
	WaitForExecution(GpuDrain::Idle);
	m_operations.Stop();
	if (m_resolver.joinable()) {
		m_resolver.join();
	}
	m_execution_stop.store(true, std::memory_order_seq_cst);
	RingExecution();
	if (m_executor.joinable()) {
		m_executor.join();
	}
}

void GuestGpu::RunCommandsWhileWaiting() {
	auto* const gpu = g_gpu_state;
	if (gpu == nullptr || !IsGpuThread()) {
		return;
	}
	while (gpu->m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			if (gpu->m_commands.empty()) {
				return;
			}
			command = std::move(gpu->m_commands.front());
			gpu->m_commands.pop_front();
			EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
		gpu->m_work_epoch.fetch_add(1, std::memory_order_release);
	}
}

bool GuestGpu::CommandsWaiting() noexcept {
	auto* const gpu = g_gpu_state;
	return gpu != nullptr && gpu->m_pending_commands.load(std::memory_order_acquire) != 0;
}

bool GuestGpu::WaitsServeCommands() noexcept {
	return g_gpu_state != nullptr;
}

// The journal of pending writes: based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
bool GuestGpu::WriteLanded(GuestWrite& write, uint64_t executed, uint64_t published, uint64_t known_tick) {
	if (write.at_publish) {
		return write.sequence <= published;
	}
	if (write.sequence > executed) {
		return false;
	}
	if (!write.stream) {
		return true;
	}
	// Executed: recorded no later than now.
	if (write.tick == 0) {
		write.tick = m_renderer.GetCommandScheduler().CurrentTick();
	}
	return write.tick <= known_tick;
}

void GuestGpu::ForgetExecutedWrites() {
	if (m_pending_count == 0) {
		return;
	}
	auto&      master    = m_renderer.GetCommandScheduler().GetMasterSemaphore();
	const auto executed  = m_executed.Load();
	const auto published = m_published.load(std::memory_order_acquire);
	master.Refresh();
	const auto known_tick = master.KnownGpuTick();
	// Every write executed since the last look gets its tick now, not when it reaches the front: a tick taken there is
	// the one open then, and the journal would drain one write per submission.
	const auto current_tick = m_renderer.GetCommandScheduler().CurrentTick();
	while (m_pending_ticked < m_pending_count) {
		auto& write = m_pending_writes[(m_pending_first + m_pending_ticked) % PendingWriteCapacity];
		if (write.sequence > executed) {
			break;
		}
		if (write.stream && write.tick == 0) {
			write.tick = current_tick;
		}
		m_pending_ticked++;
	}
	while (m_pending_count != 0 && WriteLanded(m_pending_writes[m_pending_first], executed, published, known_tick)) {
		m_pending_first = (m_pending_first + 1) % PendingWriteCapacity;
		m_pending_count--;
		if (m_pending_ticked != 0) m_pending_ticked--;
	}
}

GuestGpu::GuestWrite& GuestGpu::NewWrite() {
	ForgetExecutedWrites();
	if (m_pending_count == PendingWriteCapacity) {
		WaitForOldestWrite();
	}
	auto& write = m_pending_writes[(m_pending_first + m_pending_count) % PendingWriteCapacity];
	write       = {};
	m_pending_count++;
	if (m_pending_count > g_label_stats.max_pending.load(std::memory_order_relaxed)) {
		g_label_stats.max_pending.store(m_pending_count, std::memory_order_relaxed);
	}
	if (g_gpu_stage_stats && m_pending_count > g_stage_stats.journal_max.load(std::memory_order_relaxed)) {
		g_stage_stats.journal_max.store(m_pending_count, std::memory_order_relaxed);
	}
	return write;
}

void GuestGpu::WaitForOldestWrite() {
	// The oldest waits for the host GPU and, a label, then for its publish command. Submitting is enough: waiting for
	// the host GPU on the renderer's thread would stop the rest of the frame too.
	const auto start = g_gpu_stage_stats ? StageNowNs() : 0;
	m_gfx_cp->EmitCallback([this] { m_renderer.GetCommandScheduler().Flush(); });
	while (m_pending_count == PendingWriteCapacity) {
		ForgetExecutedWrites();
		Common::SpinPause();
	}
	if (g_gpu_stage_stats) {
		g_stage_stats.journal_full.fetch_add(1, std::memory_order_relaxed);
		g_stage_stats.journal_full_ns.fetch_add(StageNowNs() - start, std::memory_order_relaxed);
	}
}

void GuestGpu::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value) {
	if (sequence <= m_executed.Load()) {
		return; // executed writes are in guest memory
	}
	auto& write    = NewWrite();
	write.sequence = sequence;
	write.address  = address;
	write.size     = size;
	if (value != nullptr && size <= sizeof(write.value)) {
		write.known = true;
		std::memcpy(write.value, value, size);
	}
}

void GuestGpu::NoteDeferredWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value) {
	auto& write      = NewWrite();
	write.sequence   = sequence;
	write.address    = address;
	write.size       = size;
	write.known      = true;
	write.at_publish = true;
	std::memcpy(write.value, value, size);
}

void GuestGpu::NoteStreamWrite(uint64_t sequence, uint64_t address, uint32_t value) {
	auto& write    = NewWrite();
	write.sequence = sequence;
	write.address  = address;
	write.size     = sizeof(value);
	write.stream   = true;
	write.known    = true;
	std::memcpy(write.value, &value, sizeof(value));
}

void GuestGpu::WaitForPendingWrites(uint64_t address, uint64_t size, bool unknown_only) {
	const auto executed = m_executed.Load();
	uint64_t   last     = 0;
	for (size_t i = 0; i < m_pending_count; i++) {
		const auto& write = m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write.at_publish || write.sequence <= executed || write.address >= address + size ||
		    write.address + write.size <= address || (unknown_only && write.known)) {
			continue;
		}
		last = std::max(last, write.sequence);
	}
	if (last != 0) {
		// The wait acquires the writes.
		if (!g_gpu_stage_stats || m_executed.Load() >= last) {
			m_executed.WaitFor(last);
			return;
		}
		const auto start = StageNowNs();
		m_executed.WaitFor(last);
		const auto index = static_cast<size_t>(GpuDrain::Journal);
		g_stage_stats.drains[index].fetch_add(1, std::memory_order_relaxed);
		g_stage_stats.drain_ns[index].fetch_add(StageNowNs() - start, std::memory_order_relaxed);
	}
}

void GuestGpu::ApplyPendingWrites(const volatile void* address, void* value, size_t size, DecisionRead scope,
                                  bool* covered) {
	EXIT_IF(size > 8u);
	ForgetExecutedWrites();
	const uint64_t known_tick = m_renderer.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick();
	const uint64_t executed   = m_executed.Load();
	const uint64_t published  = m_published.load(std::memory_order_acquire);
	const auto     begin      = reinterpret_cast<uint64_t>(address);
	const auto     end        = begin + size;
	const uint32_t all        = (1u << size) - 1u;
	uint32_t       decided    = 0; // bytes a newer write decided
	uint32_t       pending    = 0; // of those, bytes a pending write holds
	if (g_gpu_stage_stats) {
		g_stage_stats.journal_reads.fetch_add(1, std::memory_order_relaxed);
		g_stage_stats.journal_scanned.fetch_add(m_pending_count, std::memory_order_relaxed);
	}
	// The newest write of a byte decides, also when it is in guest memory already.
	for (size_t i = m_pending_count; i-- > 0 && decided != all;) {
		auto& write = m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write.address >= end || write.address + write.size <= begin) {
			continue;
		}
		const uint64_t lo   = std::max(write.address, begin);
		const uint64_t hi   = std::min(write.address + write.size, end);
		const uint32_t mask = ((1u << (hi - lo)) - 1u) << (lo - begin);
		if ((mask & ~decided) == 0) {
			continue;
		}
		const bool landed  = WriteLanded(write, executed, published, known_tick);
		const bool visible = !landed && (scope == DecisionRead::Overlay || (write.sequence > executed && !write.at_publish));
		if (visible && write.known) {
			for (uint64_t at = lo; at < hi; at++) {
				if ((decided & (1u << (at - begin))) == 0) {
					static_cast<uint8_t*>(value)[at - begin] = write.value[at - write.address];
				}
			}
			pending |= mask & ~decided;
			g_label_stats.overlay_hits.fetch_add(1, std::memory_order_relaxed);
		}
		decided |= mask;
	}
	if (covered != nullptr) *covered = pending == all;
}

void GuestGpu::ReadDecisionMemory(const volatile void* address, void* value, size_t size, DecisionRead scope,
                                  bool* covered) {
	// A value known only when its operation executes is read from guest memory then.
	WaitForPendingWrites(reinterpret_cast<uint64_t>(address), size, true);
	CopyFromGuest(value, address, size);
	ApplyPendingWrites(address, value, size, scope, covered);
}

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_ctx_dirty    = true;
	m_ucfg_dirty   = true;
	m_sh_ctx_dirty = true;
	m_saved_ctx.Reset();
	m_context_state_pushed             = false;
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances.value              = 1;
	m_num_instances.sequence++;
	m_predicate_skip                   = false;
	m_user_data_marker                 = HW::UserSgprType::Unknown;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;

	EmitCallback([this] { std::memset(m_const_ram, 0, sizeof(m_const_ram)); });
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	m_ctx_dirty = true;
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_context_state_pushed);
			m_ctx                  = m_saved_ctx;
			m_saved_ctx            = {};
			m_context_state_pushed = false;
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_context_state_pushed);
			m_saved_ctx            = m_ctx;
			m_context_state_pushed = true;
			m_ctx.Reset();
			break;
		default: EXIT("unknown context state operation: %u\n", static_cast<uint32_t>(operation));
	}
}

// The operations layer: based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
GpuOperation CommandProcessor::MakeOperation(GpuOperationKind kind) {
	GpuOperation operation;
	operation.kind          = kind;
	operation.processor     = this;
	operation.submit_id     = m_submit_id;
	auto* const stream      = StreamGpu();
	operation.sequence      = stream != nullptr ? stream->TakeSequence() : 0;
	operation.num_instances = m_num_instances;
	// Only draws, dispatches and state callbacks read registers.
	if (kind != GpuOperationKind::Callback) {
		operation.state = SnapshotState(operation.sequence);
	}
	return operation;
}

template <typename T>
T* CommandProcessor::Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty, uint64_t sequence) {
	static_assert(std::is_trivially_copyable_v<T>);
	if (dirty || ring.current == SnapshotRing<T>::Size) {
		if (ring.slots == nullptr) {
			ring.slots    = std::make_unique<typename SnapshotRing<T>::Slot[]>(SnapshotRing<T>::Size);
			ring.last_use = std::make_unique<uint64_t[]>(SnapshotRing<T>::Size);
		}
		auto& slot = ring.slots[ring.next];
		if (auto* const stream = StreamGpu(); stream != nullptr) stream->WaitForSnapshot(ring.last_use[ring.next]);
		std::memcpy(static_cast<void*>(&slot.value), &live, sizeof(T));
		if (g_gpu_stage_stats) g_stage_stats.copied_bytes.fetch_add(sizeof(T), std::memory_order_relaxed);
		ring.current = ring.next;
		ring.next    = (ring.next + 1) % SnapshotRing<T>::Size;
		dirty        = false;
	}
	ring.last_use[ring.current] = sequence;
	return &ring.slots[ring.current].value;
}

GpuRegisterState CommandProcessor::SnapshotState(uint64_t sequence) {
	return {Snapshot(m_ctx_snapshots, m_ctx, m_ctx_dirty, sequence),
	        Snapshot(m_ucfg_snapshots, m_ucfg, m_ucfg_dirty, sequence),
	        Snapshot(m_sh_ctx_snapshots, m_sh_ctx, m_sh_ctx_dirty, sequence)};
}

uint64_t CommandProcessor::Emit(GpuOperation&& operation) {
	const auto sequence = operation.sequence;
	if (g_cp_thread) {
		g_stream_gpu->EmitOperation(std::move(operation));
	} else {
		Execute(operation);
	}
	return sequence;
}

void CommandProcessor::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value) {
	if (auto* const stream = StreamGpu(); size != 0 && stream != nullptr) stream->NoteGuestWrite(sequence, address, size, value);
}

void CommandProcessor::ReadDecisionMemory(const volatile void* address, void* value, size_t size) {
	if (auto* const stream = StreamGpu(); stream != nullptr) {
		stream->ReadDecisionMemory(address, value, size, GuestGpu::DecisionRead::Stream);
	} else {
		CopyFromGuest(value, address, size);
	}
}

void CommandProcessor::WaitForPendingWrites(const void* address, uint64_t size) {
	if (auto* const stream = StreamGpu(); stream != nullptr) {
		stream->WaitForPendingWrites(reinterpret_cast<uint64_t>(address), size, false);
	}
}

uint64_t CommandProcessor::EmitCallback(GpuCallback&& callback) {
	auto operation     = MakeOperation(GpuOperationKind::Callback);
	operation.callback = std::move(callback);
	return Emit(std::move(operation));
}

uint64_t CommandProcessor::EmitStateCallback(GpuCallback&& callback) {
	auto operation     = MakeOperation(GpuOperationKind::StateCallback);
	operation.callback = std::move(callback);
	return Emit(std::move(operation));
}

void CommandProcessor::RunOnExecution(GpuDrain reason, GpuCallback&& callback) {
	EmitCallback(std::move(callback));
	if (g_cp_thread) g_stream_gpu->WaitForExecution(reason);
}

void CommandProcessor::PublishDirtyRegions() {
	// Nothing deferred before this point: the operation would have nothing to publish for it.
	if (HasDeferredCpuDirtyRegions()) {
		EmitCallback([] { PublishCpuDirtyRegions(); });
	}
}

void CommandProcessor::Execute(GpuOperation& operation) {
	EXIT_IF(operation.processor != this);
	if (g_gpu_stage_stats) {
		g_stage_stats.operations.fetch_add(1, std::memory_order_relaxed);
		const auto kind = operation.kind;
		if (kind == GpuOperationKind::DrawIndex || kind == GpuOperationKind::DrawAuto || kind == GpuOperationKind::DrawIndirect) {
			g_stage_stats.draws.fetch_add(1, std::memory_order_relaxed);
		} else if (kind == GpuOperationKind::DispatchDirect || kind == GpuOperationKind::DispatchIndirect) {
			g_stage_stats.dispatches.fetch_add(1, std::memory_order_relaxed);
		}
	}
	// Operations read their registers from their snapshot, through the accessors of every CommandBuffer. What the
	// scheduler keeps bound is read only between operations (a commit outside the operation reads the registers of now).
	struct StateScope {
		explicit StateScope(const GpuRegisterState& state): previous(t_draw_regs) {
			t_draw_regs = {state.context, state.user_config, state.shaders};
		}
		~StateScope() { t_draw_regs = previous; }
		DrawRegsOverride previous;
	};
	std::optional<StateScope> state_scope;
	if (operation.kind != GpuOperationKind::Callback) {
		state_scope.emplace(operation.state);
	}
	auto& executor = m_renderer.GetRenderExecutor();
	if (operation.resolved != nullptr) executor.UseResolvedDraw(operation.resolved);
	switch (operation.kind) {
		case GpuOperationKind::DrawIndex: ExecuteDrawIndex(operation, operation.draw_index); break;
		case GpuOperationKind::DrawAuto: ExecuteDrawAuto(operation, operation.draw_auto); break;
		case GpuOperationKind::DrawIndirect: ExecuteDrawIndirect(operation); break;
		case GpuOperationKind::DispatchDirect: {
			const auto& dispatch = operation.dispatch;
			ExecuteDispatchDirect(operation, dispatch.thread_group_x, dispatch.thread_group_y, dispatch.thread_group_z,
			                      dispatch.mode);
			break;
		}
		case GpuOperationKind::DispatchIndirect: ExecuteDispatchIndirect(operation); break;
		case GpuOperationKind::Callback:
		case GpuOperationKind::StateCallback:
			if (operation.callback) operation.callback();
			break;
	}
	if (operation.resolved != nullptr) {
		executor.UseResolvedDraw(nullptr);
		RenderExecutor::ReleaseResolvedDraw(operation.resolved);
	}
}

void CommandProcessor::BufferInit() {
	// The execution thread never reads the live registers: what is bound is neutral.
	const auto* const stream = StreamGpu();
	const auto bound = stream != nullptr ? stream->NeutralState() : GpuRegisterState {&m_ctx, &m_ucfg, &m_sh_ctx};
	EmitCallback([this, bound] { GetScheduler().Begin(*bound.context, *bound.user_config, *bound.shaders); });
}

void CommandProcessor::BufferFlush() {
	EmitCallback([this] { GetScheduler().Flush(); });
}

void CommandProcessor::BufferFlushForInterrupt() {
	EmitCallback([this] { GetScheduler().FlushForInterrupt(); });
}

void CommandProcessor::BufferWait() {
	BufferInit();
	EmitCallback([this] { GetScheduler().Finish(); });
}

void CommandProcessor::RunGarbageCollector() {
	EmitCallback([this] { m_renderer.RunGarbageCollector(); });
}

void CommandProcessor::ResetDeCe() {
	m_de_count    = 0;
	m_ce_count    = 0;
	m_ce_complete = false;
}

void CommandProcessor::WaitCe() {
	if (m_ce_count <= m_de_count && !m_ce_complete) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitDeDiff(uint32_t diff) {
	EXIT_IF(m_de_count > m_ce_count);
	if (m_ce_count - m_de_count >= diff) {
		SuspendPm4();
	}
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
		return;
	}
	// BDA sync epoch: the guest appended the commands that follow, and their data, after the
	// submission.
	PublishDirtyRegions();
}

void CommandProcessor::IncrementDe() {
	m_de_count++;
}

void CommandProcessor::IncrementCe() {
	m_ce_count++;
}

// The constant RAM is the execution side's: a dump reads it in the stream's order. The source dwords are in the command
// buffer (see WriteData).
void CommandProcessor::WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num) {
	EmitCallback([this, offset, src, dw_num] { memcpy(m_const_ram + offset / 4, src, static_cast<size_t>(dw_num) * 4); });
}

void CommandProcessor::DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num) {
	const auto sequence = EmitCallback([this, dst, offset, dw_num] {
		memcpy(dst, m_const_ram + offset / 4, static_cast<size_t>(dw_num) * 4);
		// BDA sync epoch: the commands after this one see the GPU thread's own write.
		PublishCpuDirtyRange(reinterpret_cast<uint64_t>(dst), uint64_t {dw_num} * 4u);
	});
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), uint64_t {dw_num} * 4u);
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

static int64_t SteadyNowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// Upstream 1b7093805: the async memory semaphores of AgcAcbMemSemaphore.
void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
	} else if (*counter == 0) {
		SuspendPm4();
	} else {
		(*counter)--;
	}
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll,
                                  uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	(void)poll;
	// labels at completion / KYTY_PERF_DIRECT_RANGES: labels and CP writes recorded but not yet in guest memory;
	// other GPU writes into a direct range are waited for unless the recorded values cover the whole word.
	bool       covered = false;
	T          value {};
	const auto read = [&] {
		if (auto* const stream = StreamGpu(); stream != nullptr) {
			stream->ReadDecisionMemory(addr, &value, sizeof(T), GuestGpu::DecisionRead::Overlay, &covered);
		} else {
			CopyFromGuest(&value, addr, sizeof(T));
		}
	};
	read();
	if (!covered && m_renderer.GetBufferCache().InDirectRange(reinterpret_cast<uint64_t>(addr), sizeof(T))) {
		RunOnExecution(GpuDrain::DirectWrites, [this, addr] {
			m_renderer.GetBufferCache().WaitDirectGpuWrites(reinterpret_cast<uint64_t>(addr), sizeof(T));
		});
		read();
	}
	const bool passed = TestWaitRegMemValue(value, ref, mask, func);
	if (SyncTimeline::Enabled()) {
		SyncTimeline::GuestWaitRegMem(reinterpret_cast<uint64_t>(addr), value, ref, passed,
		                              static_cast<uint32_t>(m_interrupt_event_id));
	}
	if (!passed) {
		// The queue waits: a wait of a block entered unread holds the stream (liveness guard, the stuck line).
		if (auto& block = g_current_execution->m_cond; block.active && !block.waited) {
			block.waited  = true;
			block.wait_ns = SteadyNowNs();
		}
		SuspendPm4();
		return;
	}
	// BDA sync epoch: the guest may have written the data this wait guards before its label.
	PublishDirtyRegions();
	// The GPU half of the wait: the writer of the label recorded its work earlier into the one host queue (labels are
	// written when their packet is translated), and this barrier puts that work before everything recorded from here
	// on.
	EmitGlobalBarrier();
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t, uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t, uint32_t);

// KYTY_TRACE_CS_WRITER_ADDR also names CP packets (WRITE_DATA, DMA_DATA/COPY_DATA) writing it.
static void TraceCpWriter(const char* kind, uint64_t dst, uint64_t bytes, uint64_t src, uint32_t sel) {
	static constinit Common::OnceValue<uint64_t> writer_addr_once;
	const uint64_t writer_addr = writer_addr_once.Get([] { return Common::EnvRead::CsWriterAddr(); });
	if (writer_addr == 0 || writer_addr < dst || writer_addr >= dst + bytes) return;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1) < 40u) {
		std::fprintf(stderr, "CP_WRITER %s addr=0x%llx dst=0x%llx bytes=0x%llx src=0x%llx src_sel=%u\n", kind,
		             static_cast<unsigned long long>(writer_addr), static_cast<unsigned long long>(dst),
		             static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(src), sel);
	}
}

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;
	TraceCpWriter("write_data", reinterpret_cast<uint64_t>(dst), uint64_t {dw_num} * 4u, 0, dst_sel);

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	// The source dwords are in the command buffer, which stays valid until the guest sees a later label.
	auto* const stream   = StreamGpu();
	const auto  sequence = stream != nullptr ? stream->NextSequence() : 0;
	// KYTY_PERF_DIRECT_RANGES: the reads after this one (WAIT_REG_MEM) see a write into a direct range at its point of the
	// stream without waiting for the host GPU to run it; larger writes: readers wait for the host GPU instead.
	const bool direct = stream != nullptr && !write_one_address && dw_num <= 16u &&
	                    m_renderer.GetBufferCache().InDirectRange(reinterpret_cast<uint64_t>(dst),
	                                                              uint64_t {dw_num} * sizeof(uint32_t));
	if (direct) {
		for (uint32_t i = 0; i < dw_num; i++) {
			stream->NoteStreamWrite(sequence, reinterpret_cast<uint64_t>(dst) + i * 4u, src[i]);
		}
	}
	EmitCallback([this, dst, src, dw_num, write_one_address] {
		// Completion labels can be reused immediately; memcpy may write a destination twice. The guest-memory stores
		// below write each word exactly once, in order (upstream 099de8dea).
		if (write_one_address) {
			auto* output = static_cast<volatile uint32_t*>(dst);
			for (uint32_t i = 0; i < dw_num; i++) {
				*output = src[i];
			}
		} else if (!m_renderer.GetBufferCache().TryCpWriteThrough(reinterpret_cast<uint64_t>(dst), src,
		                                                          uint64_t {dw_num} * sizeof(uint32_t))) {
			auto* output = static_cast<volatile uint32_t*>(dst);
			for (uint32_t i = 0; i < dw_num; i++) {
				output[i] = src[i];
			}
		}
		// BDA sync epoch: the commands after this one see the GPU thread's own write.
		PublishCpuDirtyRange(reinterpret_cast<uint64_t>(dst),
		                     write_one_address ? sizeof(uint32_t) : uint64_t {dw_num} * sizeof(uint32_t));
	});
	if (write_one_address) {
		NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), sizeof(uint32_t), src + dw_num - 1);
	} else if (!direct) {
		NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), uint64_t {dw_num} * sizeof(uint32_t), src);
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto sequence = EmitCallback([this, dst_address, num_bytes] {
		const auto value = Sync::ReadReferenceClock();
		// (The low bytes of the clock on little-endian, as the memcpy.)
		if (!m_renderer.GetBufferCache().TryCpWriteThrough(dst_address, &value, num_bytes)) {
			std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
		}
		PublishCpuDirtyRange(dst_address, num_bytes); // BDA sync epoch, as WriteData
		static std::atomic<uint32_t> clock_log_count {0};
		if (clock_log_count.fetch_add(1) < 64) {
			LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
			     " size=%u\n",
			     dst_address, value, num_bytes);
		}
	});
	NoteGuestWrite(sequence, dst_address, num_bytes);
}

void CommandProcessor::DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
                               uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint8_t  src_cache_policy,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
                               uint8_t wait_for_previous, uint8_t write_confirm,
                               uint8_t block_engine) {
	EXIT_NOT_IMPLEMENTED(engine > 1);
	if (num_bytes == 0) {
		return;
	}
	TraceCpWriter("dma_data", dst_address_or_offset, num_bytes, src_address_or_offset_or_immediate,
	              src_sel | (static_cast<uint32_t>(dst_sel) << 8u));
	EXIT_NOT_IMPLEMENTED(dst_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(src_cache_policy > 3);
	EXIT_NOT_IMPLEMENTED(wait_for_previous > 1);
	EXIT_NOT_IMPLEMENTED(write_confirm > 1);
	EXIT_NOT_IMPLEMENTED(block_engine > 1);
	if (static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	if (DebugCaptureFrame().load() && (src_sel == 1 || dst_sel == 1)) {
		std::fprintf(stderr,
		             "CAPTURE_GDS_DMA cp=%p index=%u dst=0x%llx dst_sel=%u src=0x%llx src_sel=%u bytes=%u marker=[%s]\n",
		             static_cast<void*>(this), DebugCaptureIndex().load(),
		             static_cast<unsigned long long>(dst_address_or_offset), dst_sel,
		             static_cast<unsigned long long>(src_address_or_offset_or_immediate), src_sel, num_bytes,
		             DebugMarkerPath().c_str());
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	bool src_gds = false;
	if (src_sel != 2) {
		if (!decode_gds(src_sel, src_gds)) {
			EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
		}
		if (src_gds && dst_gds) {
			EXIT("unsupported dmaData GDS-to-GDS copy\n");
		}
	}
	const auto sequence = EmitCallback([=, this] {
		ExecuteDmaData(dst_gds, dst_address_or_offset, src_sel, src_gds, src_address_or_offset_or_immediate, num_bytes);
	});
	if (!dst_gds) {
		NoteGuestWrite(sequence, dst_address_or_offset, num_bytes);
	}
}

void CommandProcessor::ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset, uint8_t src_sel, bool src_gds,
                                      uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes) {
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
	} else {
		buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
		                        dst_gds, src_gds);
	}
	// BDA sync epoch: the commands after this one see the fill or copy, also where the GPU thread wrote
	// guest memory itself (the CPU paths).
	if (!dst_gds) {
		PublishCpuDirtyRange(dst_address_or_offset, num_bytes);
	}
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_commands_running || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

// KYTY_TRACE_PERF=1: every ~5 s, how long the GPU thread slept with every queue blocked.
static void PerfBlockedWait(std::chrono::steady_clock::duration waited) {
	static const bool enabled = Common::EnvRead::Perf();
	if (!enabled) return;
	static auto     window_start = std::chrono::steady_clock::now();
	static uint64_t count        = 0;
	static std::chrono::steady_clock::duration total {};
	count++;
	total += waited;
	const auto now = std::chrono::steady_clock::now();
	if (now - window_start >= std::chrono::seconds(5)) {
		std::fprintf(stderr, "PERF blocked_waits=%llu blocked_ms=%lld window_ms=%lld\n",
		             static_cast<unsigned long long>(count),
		             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(total).count()),
		             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start).count()));
		window_start = now;
		count        = 0;
		total        = {};
	}
}

// The register structures (render.h RegsWritten*) a packet's handler may write. Register packets
// write their own class; draws, index/instance setup, SET_BASE, buffer jumps and plain/marker NOPs write none; every
// other packet counts as writing all three.
static uint32_t RegistersWrittenBy(uint32_t opcode, uint32_t r) {
	switch (opcode) {
		case Pm4::IT_SET_CONTEXT_REG:
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT: return RegsWrittenCtx;
		case Pm4::IT_SET_SH_REG:
		case Pm4::IT_SET_SH_REG_INDIRECT: return RegsWrittenSh;
		case Pm4::IT_SET_UCONFIG_REG:
		case Pm4::IT_SET_UCONFIG_REG_INDEX:
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return RegsWrittenUcfg;
		case Pm4::IT_DRAW_INDEX_2:
		case Pm4::IT_DRAW_INDEX_OFFSET_2:
		case Pm4::IT_DRAW_INDEX_AUTO:
		case Pm4::IT_INDEX_BASE:
		case Pm4::IT_INDEX_BUFFER_SIZE:
		case Pm4::IT_INDEX_TYPE:
		case Pm4::IT_NUM_INSTANCES:
		case Pm4::IT_SET_BASE:
		case Pm4::IT_INDIRECT_BUFFER: return 0u;
		case Pm4::IT_NOP: return r == 0x00 || r == 0x0b || r == 0x0c ? 0u : RegsWrittenAll;
		default: return RegsWrittenAll;
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	// The stage's core and priority (without its core the thread wandered over all 16 logical processors and an SMT
	// sibling ran another thread of the process in ~57% of its samples, Nsight Systems run nsys4; one core alone gave
	// 9.23 -> 9.54 fps). Nothing the emulator computes changes, only where the threads run.
	Config::ConfigureGpuStageThread(Config::GpuStageThread::CommandProcessor);
	g_stream_gpu = gpu;
	g_cp_thread  = true;

	// When every queue waits (WAIT_REG_MEM), the command processor retries without sleeping until the spin budget
	// is spent. Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
	constexpr auto                                       BlockedRetryInterval = std::chrono::microseconds(20);
	std::optional<std::chrono::steady_clock::time_point> blocked_since;
	bool                                                 spin_blocked = false;

	for (;;) {
		Submission                   submission;
		bool                         should_stop = false;
		bool                         cond_guard  = false; // The queue waits: the submission goes to the guard
		if (spin_blocked) {
			spin_blocked      = false;
			const auto epoch  = gpu->m_work_epoch.load(std::memory_order_acquire);
			const auto target = std::chrono::steady_clock::now() + BlockedRetryInterval;
			while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch && std::chrono::steady_clock::now() < target) {
				Common::SpinPause();
			}
		}
		// The GPU is idle once the execution thread ran every operation too. Only this thread emits operations, so none can
		// follow the wait until it finds work.
		bool no_work = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			no_work = gpu->m_submission_count == 0 && !gpu->m_stopping;
		}
		if (no_work) {
			gpu->WaitForExecution(GpuDrain::Idle);
			uint64_t epoch = 0;
			{
				Common::LockGuard lock(gpu->m_queue_mutex);
				epoch = gpu->m_work_epoch.load(std::memory_order_acquire);
				if (gpu->m_submission_count == 0) {
					gpu->m_processing = false;
					gpu->m_idle.SignalAll();
				}
			}
			// Spin for the next submission; sleep below only when none came.
			Common::SpinWait spin;
			while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch && spin.Spin()) {
			}
		}
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else {
				int selected_queue = -1;
				static_assert(QueueCount <= 64); // g_queue_history
				const uint32_t frame_now = DebugPresentCount().load(std::memory_order_relaxed);
				int            producer  = -1; // the first runnable producer in round-robin order
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						if (IsProducerQueue(id, frame_now)) {
							if (producer < 0) producer = static_cast<int>(id);
							continue;
						}
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) selected_queue = producer;
				// the queue waits liveness guard: no queue can run and a stream waits in a block of waits whose
				// predicate was never read.
				if (selected_queue < 0) {
					selected_queue = gpu->CondGuardQueue();
					cond_guard     = selected_queue >= 0;
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					const auto blocked_start = std::chrono::steady_clock::now();
					// A wake that came before this wait must not be lost (GuestGpu::Wake; only KYTY_UPSTREAM_FLIP_SUSPEND
					// wakes the thread this way, so by default the flag is never set).
					const auto now = std::chrono::steady_clock::now();
					if (!blocked_since) {
						blocked_since = now;
					}
					if (std::exchange(gpu->m_wake_pending, false)) {
						// Retry at once.
					} else if (now - *blocked_since < Common::SpinBudget()) {
						spin_blocked = true;
					} else {
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					}
					PerfBlockedWait(std::chrono::steady_clock::now() - blocked_start);
					if (GpuTimer::Enabled() && GpuTimer::TimelineCpuActive()) GpuTimer::NoteCpuWait('b', blocked_start); // KYTY_TRACE_GPU_TIMELINE
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			gpu->StopExecution();
			if (PerfEnabled()) ReportGpuStageCheck(); // a run shorter than the report's window
			g_stream_gpu = nullptr;
			g_cp_thread  = false;
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			return;
		}
		if (cond_guard) {
			// Outside the queue lock: the exact read can wait for the GPU while other threads keep submitting. The
			// submission goes back to the front of its queue, runnable when the guard skipped the rest of its block.
			const bool skipped = gpu->ResolveCondGuard(submission);
			Common::LockGuard lock(gpu->m_queue_mutex);
			submission.blocked = !skipped;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
			gpu->m_processing = false;
			continue;
		}

		const bool complete = gpu->Process(submission);
		if (complete || gpu->m_slice_progressed) {
			gpu->m_last_progress = std::chrono::steady_clock::now(); // the queue waits guard
			blocked_since.reset();
		}
		if (static const bool reports = PerfEnabled() || MemCensusRequested() || SyncTimeline::Enabled(); reports) {
			gpu->m_gfx_cp->EmitCallback([gpu] {
				PerfReport(gpu->m_renderer);
				SyncTimeline::Report(); // KYTY_TRACE_SYNC_TIMELINE
				MemCensusReport(gpu->m_renderer);
			});
		}

		{ // The queue's history (IsProducerQueue).
			const uint32_t frame = DebugPresentCount().load(std::memory_order_relaxed);
			auto&          h     = g_queue_history[submission.queue_id];
			if (h.first_frame == 0) h.first_frame = frame + 1u;
			if (!complete) h.block_frame = frame + 1u;
		}
		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		// Processing until the operations executed: the idle wait above.
	}
}

bool GuestGpu::Process(Submission& submission) {
	const bool first_slice = !submission.started;
	auto&      cp          = GetProcessor(submission.queue_id);
	// BDA sync epoch: a submission starting or resuming sees the guest writes made before it.
	cp.PublishDirtyRegions();

	if (first_slice) {
		submission.started = true;
		cp.SetSubmitId(++m_submit_id);
		cp.ResetDeCe();
		cp.SetFlip({});
	}

	cp.BufferInit();
	bool complete      = true;
	m_slice_progressed = false; // the queue waits guard: whether this slice ran a packet

	switch (submission.type) {
		case SubmissionType::Graphics: {
			bool progressed = false;
			submission.constant_complete |= submission.constant_commands.empty();
			for (;;) {
				bool round_progress = false;
				if (!submission.constant_complete) {
					submission.constant_complete =
					    cp.Process(submission.constant_execution, submission.constant_commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.constant_execution.MadeProgress();
				}
				cp.SetCeComplete(submission.constant_complete);
				if (!submission.command_complete) {
					submission.command_complete =
					    cp.Process(submission.command_execution, submission.commands) ==
					    Pm4ProcessResult::Complete;
					round_progress |= submission.command_execution.MadeProgress();
				}
				progressed |= round_progress;
				complete = submission.command_complete && submission.constant_complete;
				if (complete || !round_progress) {
					break;
				}
			}
			m_slice_progressed = progressed;
			if (progressed) {
				cp.BufferFlush();
			}
			break;
		}
		case SubmissionType::Compute: {
			const auto      num_dw = static_cast<uint32_t>(submission.commands.size());
			const auto*     buffer = submission.commands.data();
			static uint32_t compute_batch_log_count = 0;
			if (first_slice && num_dw <= 128 && compute_batch_log_count++ < 32) {
				LOGF("compute direct batch: data=0x%016" PRIx64 ", num_dw=%" PRIu32 "\n",
				     reinterpret_cast<uint64_t>(buffer), num_dw);
				for (uint32_t i = 0; i < std::min<uint32_t>(num_dw, 16); i++) {
					LOGF("\t compute[%02" PRIu32 "] = 0x%08" PRIx32 "\n", i, buffer[i]);
				}
			}
			if (first_slice) {
				GraphicsDbgDumpDcb("cc", num_dw, buffer);
			}
			complete = cp.Process(submission.command_execution, submission.commands) ==
			           Pm4ProcessResult::Complete;
			m_slice_progressed = submission.command_execution.MadeProgress();
			if (m_slice_progressed) {
				cp.BufferFlush();
			}
			break;
		}
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			// Age caches at the frame boundary, not once per command list.
			cp.RunGarbageCollector();
			// Kept (upstream 70668fbd9 drops the PM4 global barriers): in our renderer these barriers order the game's
			// work on the host GPU.
			cp.EmitGlobalBarrier();
			cp.EmitCallback([this, ready = m_suspend_point_ready] {
				m_renderer.GetCommandScheduler().DeferPriorityOperation([ready] { ready->release(); });
			});
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		ExecutionScope(CommandProcessor& processor, Pm4Execution& execution)
		    : previous_processor(g_current_processor), previous_execution(g_current_execution),
		      previous_stream(DebugMarkerStream()) {
			g_current_processor  = &processor;
			g_current_execution  = &execution;
			DebugMarkerStream() = &processor;
		}
		~ExecutionScope() {
			g_current_processor  = previous_processor;
			g_current_execution  = previous_execution;
			DebugMarkerStream() = previous_stream;
		}

		CommandProcessor* previous_processor;
		Pm4Execution*     previous_execution;
		const void*       previous_stream;
	} execution_scope(*this, execution);

	if (DrawStateObserverOn()) DrawStateObserver::Invalidate(DrawStateObserver::Cause(DrawStateObserver::CauseInvalidate, DrawStateObserver::InvalidateSlice)); // the draw-state observer (DrawStateObserverOn): another processor's slice may have run
	ProcessPm4(execution);
	if (DrawStateObserverOn()) DrawStateObserver::Invalidate(DrawStateObserver::Cause(DrawStateObserver::CauseInvalidate, DrawStateObserver::InvalidateSlice));
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}


void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	WaitForPendingWrites(commands.data(), commands.size_bytes()); // the stream may have written the buffer
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool observer_on = DrawStateObserverOn(); // the draw-state observer (DrawStateObserverOn)
	// KYTY_DIAG_GPU_BURN_AT=pm4 (measurement only): one GPU-thread burn per packet handled (renderDraw.cpp GpuBurnAt).
	const bool pm4_burn =
	    RenderExecutor::GpuBurnSelected() == RenderExecutor::GpuBurnSite::Pm4 && GuestGpu::IsGpuThread();
	while (!execution.m_buffer_stack.empty()) {
		// The queue waits: checked before every packet, so every way of advancing (packets, fillers, predicated
		// skips, the end of the buffer) leaves the block of waits once the stream is past it.
		if (execution.m_cond.active) {
			LeaveCondBlockIfPassed(execution);
		}
		auto& cursor = execution.m_buffer_stack.back();
		EXIT_IF(cursor.offset_dw > cursor.commands.size());
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;
		EXIT_NOT_IMPLEMENTED(remaining_dw > total_dw);

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (GraphicsRunDebugDumpEnabled()) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     total_dw - remaining_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			if (DrawStateObserverOn()) DrawStateObserver::Dirty(DrawStateObserver::Cause(DrawStateObserver::CausePredicated, opcode)); // the draw-state observer (DrawStateObserverOn) (as chenxiao: unknown)
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw == 0 || packet_dw > remaining_dw);
			static std::atomic<uint32_t> skip_log_count {0};
			if (skip_log_count.fetch_add(1) < 2048) {
				LOGF("\t predicated skip: op=0x%02" PRIx32 ", r=0x%02" PRIx32 ", len=%" PRIu32
				     ", packet=0x%016" PRIx64 ", cmd_id=0x%08" PRIx32 "\n",
				     opcode, KYTY_PM4_R(packet_header), packet_dw,
				     reinterpret_cast<uint64_t>(packet), packet_header);
			}
			if (opcode == Pm4::IT_NOP && KYTY_PM4_R(packet_header) == Pm4::R_RELEASE_MEM &&
			    packet_dw >= 7) {
				static std::atomic<uint32_t> log_count {0};
				if (log_count.fetch_add(1) < 128) {
					const auto dst = packet[3] | (static_cast<uint64_t>(packet[4]) << 32u);
					const auto val = packet[5] | (static_cast<uint64_t>(packet[6]) << 32u);
					LOGF("\t predicated skip: R_RELEASE_MEM dst=0x%016" PRIx64
					     ", value=0x%016" PRIx64 ", action=0x%08" PRIx32
					     ", gcr/data/int=0x%08" PRIx32 "\n",
					     dst, val, packet[1], packet[2]);
				}
			}
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		if (pm4_burn) RenderExecutor::GpuBurnAt(RenderExecutor::GpuBurnSite::Pm4); // KYTY_DIAG_GPU_BURN_AT=pm4 (measurement only)
		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = total_dw - remaining_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			// The same dump on stderr (runs silence LOGF): which buffer, where, and the words around the bad header
			// (an intermittent corrupt stream, 20261013-122251 / 20261013-172443 / 20261015-014455).
			std::fprintf(stderr, "PM4_UNKNOWN_OP data=0x%016" PRIx64 " num_dw=%" PRIu32 " offset=0x%05" PRIx32 " frame=%u\n",
			             reinterpret_cast<uint64_t>(base), total_dw, offset, DebugPresentCount().load());
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				std::fprintf(stderr, "PM4_UNKNOWN_OP %05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			std::fflush(stderr);
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n",
			     total_dw - remaining_dw, packet_header);
		}

		// The draw-state observer sees the packet before its handler (a draw's handler reads its answer).
		if (observer_on) {
			DrawStateObserver::ObservePacket(opcode, packet, std::min<uint32_t>(KYTY_PM4_LEN(packet_header), remaining_dw),
			                                 RegistersWrittenBy(opcode, KYTY_PM4_R(packet_header)));
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			return; // the packet runs again from its start when the stream resumes
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (m_renderer.GetCommandScheduler().SubmitDeferred()) {
			// A deferred interrupt is submitted once the GPU is idle, or at its bound.
			EmitCallback([this] { GetScheduler().NoteDeferredPacket(); });
		}
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
	// The queue waits: a block of waits at the very end of the submission.
	if (execution.m_cond.active) {
		LeaveCondBlockIfPassed(execution);
	}
}

void CommandProcessor::LeaveCondBlockIfPassed(Pm4Execution& execution) {
	auto&       block = execution.m_cond;
	const auto& stack = execution.m_buffer_stack;
	// A block of waits calls no buffer and does not chain, so the stream leaves it in the same buffer at its end; a
	// popped or replaced buffer also means the block is behind the stream.
	const bool inside = !stack.empty() && stack.size() == block.depth && stack.back().commands.data() == block.commands &&
	                    stack.back().offset_dw < block.end_dw;
	if (inside) {
		return;
	}
	block = {};
}

void CommandProcessor::EnterAssumedCondBlock(uint64_t predicate, uint32_t end_dw) {
	EXIT_IF(g_current_execution == nullptr || g_current_execution->m_buffer_stack.empty());
	auto&       execution = *g_current_execution;
	const auto& cursor    = execution.m_buffer_stack.back();
	EXIT_IF(end_dw > cursor.commands.size() || end_dw <= cursor.offset_dw);
	auto& block     = execution.m_cond;
	block           = {};
	block.active    = true;
	block.predicate = predicate;
	block.commands  = cursor.commands.data();
	block.depth     = execution.m_buffer_stack.size();
	block.end_dw    = end_dw;
	block.frame     = DebugPresentCount().load(std::memory_order_relaxed);
}

bool CommandProcessor::ResolveAssumedCondBlock(Pm4Execution& execution, uint32_t queue_id) {
	auto& block = execution.m_cond;
	if (!block.active || block.checked) {
		return false;
	}
	// The predicate's writer precedes the COND_EXEC in this stream, which is suspended inside the block, so this read
	// (a fault that downloads once the GPU finished the writer, as the plain path of CpOpCondExec) sees this frame's value.
	uint32_t value = 0;
	RunOnExecution(GpuDrain::CondGuard, [&value, predicate = block.predicate] {
		Libs::LibKernel::Memory::ReadGuestPlain(&value, predicate, sizeof(value));
	});
	// BDA sync epoch: the stream decides on guest memory, as CpOpCondExec does.
	PublishDirtyRegions();
	static std::atomic<uint32_t> logged {0};
	if (const auto n = logged.fetch_add(1); n < 64u || n % 256u == 0u) {
		std::fprintf(stderr, "QUEUE_WAITS guard queue=0x%02x addr=0x%llx value=%u block_frame=%u frame=%u %s\n", queue_id,
		             static_cast<unsigned long long>(block.predicate), value, block.frame, DebugPresentCount().load(),
		             value == 0 ? "skipped" : "waits");
	}
	if (value != 0) {
		// The console waits for the block too.
		block.checked    = true;
		block.checked_ns = SteadyNowNs();
		return false;
	}
	auto& cursor = execution.m_buffer_stack.back();
	EXIT_IF(execution.m_buffer_stack.size() != block.depth || cursor.commands.data() != block.commands ||
	        cursor.offset_dw >= block.end_dw);
	cursor.offset_dw = block.end_dw;
	block            = {};
	return true;
}

bool CommandProcessor::IsCondGuardCandidate(const Pm4Execution& execution) {
	return execution.m_cond.active && !execution.m_cond.checked;
}

void CommandProcessor::NoteStuckCondBlock(Pm4Execution& execution, uint32_t queue_id, int64_t now_ns) {
	auto& block = execution.m_cond;
	if (!block.active || !block.checked || block.stuck_logged || now_ns - block.checked_ns < 1'000'000'000ll) {
		return;
	}
	block.stuck_logged = true;
	std::fprintf(stderr, "QUEUE_WAITS genuine wait stuck queue=0x%02x addr=0x%llx block_frame=%u frame=%u waited_ms=%lld\n",
	             queue_id, static_cast<unsigned long long>(block.predicate), block.frame, DebugPresentCount().load(),
	             static_cast<long long>((now_ns - block.wait_ns) / 1000000));
}

int GuestGpu::CondGuardQueue() {
	const auto now    = std::chrono::steady_clock::now();
	const auto now_ns = SteadyNowNs();
	int        found  = -1;
	for (uint32_t index = 0; index < QueueCount; index++) {
		auto& queue = m_queues[index];
		if (queue.empty()) {
			continue;
		}
		auto& front = queue.front();
		CommandProcessor::NoteStuckCondBlock(front.command_execution, GuestQueueId(index), now_ns);
		CommandProcessor::NoteStuckCondBlock(front.constant_execution, GuestQueueId(index), now_ns);
		if (found < 0 && (CommandProcessor::IsCondGuardCandidate(front.command_execution) ||
		                  CommandProcessor::IsCondGuardCandidate(front.constant_execution))) {
			found = static_cast<int>(index);
		}
	}
	// How long no slice may run a packet before the liveness guard reads the predicate of a block of waits a stream is
	// suspended in.
	constexpr std::chrono::milliseconds GuardDelay {250};
	return found >= 0 && now - m_last_progress >= GuardDelay ? found : -1;
}

bool GuestGpu::ResolveCondGuard(Submission& submission) {
	auto& cp = GetProcessor(submission.queue_id);
	// The exact read may download through the scheduler, which records with a processor's registers bound.
	cp.BufferInit();
	// The next candidate, if any, gets a delay of its own.
	m_last_progress  = std::chrono::steady_clock::now();
	const auto queue = GuestQueueId(submission.queue_id);
	return cp.ResolveAssumedCondBlock(submission.command_execution, queue) ||
	       cp.ResolveAssumedCondBlock(submission.constant_execution, queue);
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances.value = num_instances;
	m_num_instances.sequence++;
}

// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS: the last record's instance count of the last GPU-executed indirect packet (its
// count and record from the cached GPU copy when there is one, else guest memory); nullopt: none, or no records.
static std::optional<uint32_t> LastRecordInstances(BufferCache& buffers, uint64_t args, uint64_t count_addr,
                                                   uint32_t max_count, uint32_t stride, bool indexed) {
	uint32_t count = max_count;
	if (count_addr != 0) {
		uint32_t value = 0;
		if (!buffers.ReadGpuCopy(count_addr, 4, &value)) Libs::LibKernel::Memory::ReadGuestPlain(&value, count_addr, sizeof(value));
		count = std::min(value, max_count);
	}
	if (count == 0) return std::nullopt;
	const uint64_t field = args + uint64_t {count - 1u} * stride +
	                       (indexed ? offsetof(vk::DrawIndexedIndirectCommand, instanceCount)
	                                : offsetof(vk::DrawIndirectCommand, instanceCount));
	uint32_t instances = 0;
	if (!buffers.ReadGpuCopy(field, 4, &instances)) Libs::LibKernel::Memory::ReadGuestPlain(&instances, field, sizeof(instances));
	return instances;
}

// Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
uint32_t CommandProcessor::ResolveNumInstances(const GpuOperation& operation) {
	// A CPU-expanded indirect draw executed after the last NUM_INSTANCES packet of the operation's stream position set
	// the persistent count.
	const uint64_t sequence = operation.num_instances.sequence;
	const uint32_t current  = m_has_indirect_instances && m_indirect_instances_sequence == sequence
	                              ? m_indirect_instances
	                              : operation.num_instances.value;
	const auto&    r        = m_num_instances_records;
	if (r.args == 0 || r.sequence != sequence) return current;
	const uint32_t mode = UpstreamIndirectRecordsMode();
	const auto     last = LastRecordInstances(m_renderer.GetBufferCache(), r.args, r.count_addr, r.max_count, r.stride,
	                                          r.indexed);
	if (mode == 2u) {
		auto& stats = GpuIndirectCounters();
		stats.inherit_draws.fetch_add(1, std::memory_order_relaxed);
		if (last.has_value() && *last != current) {
			stats.inherit_differs.fetch_add(1, std::memory_order_relaxed);
			if (stats.logged.fetch_add(1, std::memory_order_relaxed) < 32u) {
				std::fprintf(stderr,
				             "INDIRECT_RECORDS_VERIFY inherit frame=%u args=0x%llx count_addr=0x%llx max=%u last_record=%u "
				             "num_instances=%u\n",
				             m_renderer.GetGpu().GetFrameNum(), static_cast<unsigned long long>(r.args),
				             static_cast<unsigned long long>(r.count_addr), r.max_count, *last, current);
			}
		}
	}
	return mode == 1u && last.has_value() ? *last : current;
}

void CommandProcessor::CheckStaleIndirectSgprs(const GpuOperation& operation, uint32_t named, const char* kind) {
	const auto& vs       = operation.state.shaders->GetVs();
	const auto  handed   = vs.gs_regs.rsrc2.user_sgpr;
	const auto  received = handed >= 32u ? ~0u : (1u << handed) - 1u;
	const auto  stale    = vs.gs_user_sgpr.indirect_stale & received & ~named;
	if (stale == 0) return;
	auto& stats = GpuIndirectCounters();
	stats.stale_sgpr_draws.fetch_add(1, std::memory_order_relaxed);
	if (stats.logged.fetch_add(1, std::memory_order_relaxed) < 32u) {
		std::fprintf(stderr, "INDIRECT_RECORDS_VERIFY stale_sgpr frame=%u kind=%s gs=0x%llx user_sgpr=%u stale=0x%08x\n",
		             m_renderer.GetGpu().GetFrameNum(), kind, static_cast<unsigned long long>(vs.gs_regs.data_addr),
		             static_cast<unsigned>(handed), stale);
	}
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address, uint32_t count_in_dwords) {
	(void)count_in_dwords;
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// One begin/end pair per DB; bit 63 marks each counter ready. Read at this point of the stream: the backing
			// reads are the execution side's.
			constexpr uint64_t       ready_bit = 1ull << 63u;
			std::array<uint64_t, 32> pairs {};
			RunOnExecution(GpuDrain::Predication, [&pairs, results = reinterpret_cast<uint64_t>(address)] {
				for (uint32_t i = 0; i < pairs.size(); i++) {
					Libs::LibKernel::Memory::ReadGuestPlain(&pairs[i], results + i * 8u, sizeof(pairs[i]));
				}
			});
			for (uint32_t db = 0; db < 16u; db++) {
				const uint64_t begin = pairs[db * 2u];
				const uint64_t end   = pairs[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03: {
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			// The predicate read goes through the backing store: a predicate the GPU wrote (the tiled-lighting class,
			// CS 85312e99) is synced first; the plain read sees the guest memory from before that write, skips the
			// predicated dispatch, and the tiles of that class stay black.
			RunOnExecution(GpuDrain::Predication, [&value, addr = reinterpret_cast<uint64_t>(address)] {
				uint64_t raw = 0;
				Libs::LibKernel::Memory::ReadGuestPlain(&raw, addr, sizeof(raw));
				value           = raw;
				uint64_t synced = 0;
				const bool ok   = Libs::LibKernel::Memory::TryReadBufferBacking(addr, &synced, sizeof(synced));
				if (ok) value = synced;
			});
		} break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	// BDA sync epoch: the predicated commands depend on guest memory the guest may have written.
	PublishDirtyRegions();
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
	if (op == 0x03) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1) < 128) {
			LOGF("\t bool predication: addr=0x%016" PRIx64 ", value=0x%016" PRIx64
			     ", condition=%" PRIu32 ", skip=%u, wait_op=%" PRIu32 "\n",
			     reinterpret_cast<uint64_t>(address), value, condition,
			     m_predicate_skip ? 1u : 0u, wait_op);
		}
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	auto operation           = MakeOperation(GpuOperationKind::DrawIndex);
	operation.draw_index     = args;
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = ResolveNumInstances(operation);
	}
	if (UpstreamIndirectRecordsMode() == 2u && args.offset_source != DrawOffsetSource::IndirectArgs) {
		CheckStaleIndirectSgprs(operation, 0, "draw_index");
	}
	m_renderer.GetRenderExecutor().DrawIndex(operation.submit_id, CurrentBuffer(), args);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	uint64_t index_size = 0;
	switch (m_index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", m_index_type_and_size);
	}
	return reinterpret_cast<const void*>(m_index_base_addr + first_index * index_size);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

// Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.cpp), GPL-2.0.
static void PatchUserSgpr(HW::Shader& shaders, uint32_t location, uint32_t value, HW::UserSgprType type) {
	const auto in = [location](uint32_t base) { return location >= base && location < base + 32u; };
	if (in(Pm4::SPI_SHADER_USER_DATA_GS_0)) {
		shaders.SetGsUserSgpr(location - Pm4::SPI_SHADER_USER_DATA_GS_0, value, type);
	} else if (in(Pm4::SPI_SHADER_USER_DATA_HS_0)) {
		shaders.SetHsUserSgpr(location - Pm4::SPI_SHADER_USER_DATA_HS_0, value, type);
	} else if (in(Pm4::SPI_SHADER_USER_DATA_PS_0)) {
		shaders.SetPsUserSgpr(location - Pm4::SPI_SHADER_USER_DATA_PS_0, value, type);
	} else {
		EXIT("unsupported indirect draw offset register 0x%" PRIx32 "\n", location);
	}
}

// Writes the offsets of one draw of a CPU-expanded indirect packet to the user SGPRs the packet names, as the CP does
// before the draw: the operation's snapshot is shared with other operations, so a copy is patched and this draw reads it.
void CommandProcessor::WriteIndirectOffsets(const GpuOperation& operation, uint32_t vertex_offset, uint32_t instance_offset,
                                            uint32_t first_index) {
	const auto& registers = operation.draw_indirect.registers;
	if (registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP) {
		return;
	}
	if (m_indirect_shaders == nullptr) {
		m_indirect_shaders = std::make_unique<HW::Shader>();
	}
	std::memcpy(static_cast<void*>(m_indirect_shaders.get()), operation.state.shaders, sizeof(HW::Shader));
	const auto type  = static_cast<HW::UserSgprType>(operation.draw_indirect.user_data_marker);
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location != Pm4::SH_NOP) PatchUserSgpr(*m_indirect_shaders, location, value, type);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
	t_draw_regs.sh = m_indirect_shaders.get();
	// The packet's writes stay in the live registers while they are this thread's.
	if (g_gpu_state == nullptr) {
		const auto live = [&](uint32_t location, uint32_t value) {
			if (location == Pm4::SH_NOP) return;
			EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM || g_hw_sh_indirect_func[location] == nullptr);
			g_hw_sh_indirect_func[location](*this, location, value);
		};
		live(registers.vertex_offset, vertex_offset);
		live(registers.instance_offset, instance_offset);
		live(registers.index_offset, first_index);
	}
}

// The SGPRs of a GPU indirect draw's vertex variant (ShaderIndirectDrawInfo); the reason when the packet cannot have one.
static const char* IndirectDrawLayout(const IndirectDrawRegisters& registers, ShaderIndirectDrawInfo& sgprs) {
	if (registers.index_offset != Pm4::SH_NOP) {
		return "first-index SGPR (no SPIR-V draw parameter)";
	}
	const auto user_data = [](uint32_t location, uint8_t& index) {
		index = ShaderIndirectDrawInfo::NoSgpr;
		if (location == Pm4::SH_NOP) {
			return true;
		}
		if (location < Pm4::SPI_SHADER_USER_DATA_GS_0 || location >= Pm4::SPI_SHADER_USER_DATA_GS_0 + 32u) {
			return false;
		}
		index = static_cast<uint8_t>(location - Pm4::SPI_SHADER_USER_DATA_GS_0);
		return true;
	};
	sgprs.enabled = true;
	if (!user_data(registers.vertex_offset, sgprs.base_vertex) || !user_data(registers.instance_offset, sgprs.base_instance) ||
	    !user_data(registers.draw_index, sgprs.draw_index)) {
		return "SGPR outside the vertex-stage user data";
	}
	const auto aliased = [](uint8_t a, uint8_t b) { return a != ShaderIndirectDrawInfo::NoSgpr && a == b; };
	if (aliased(sgprs.base_vertex, sgprs.base_instance) || aliased(sgprs.base_vertex, sgprs.draw_index) ||
	    aliased(sgprs.base_instance, sgprs.draw_index)) {
		return "aliased SGPR destinations";
	}
	return nullptr;
}

// Hands the packet to the GPU path. False
// (logged once per reason) leaves it to the CPU expansion; nothing was recorded for it then.
bool CommandProcessor::DrawIndirectGpu(const GpuOperation& operation) {
	const auto&    indirect           = operation.draw_indirect;
	const uint32_t max_count_or_count = indirect.max_count;
	const uint32_t stride_in_bytes    = indirect.stride;
	const bool     indexed            = indirect.indexed;
	const auto&    registers          = indirect.registers;
	const auto     count_address      = reinterpret_cast<uint64_t>(indirect.count_addr);
	const auto     fallback           = [&](const char* reason) {
		NoteGpuIndirectFallback(reason, indexed, max_count_or_count, count_address);
		return false;
	};
	DrawIndirectGpuArgs args {
	    .args_addr           = indirect.args_base + indirect.data_offset,
	    .count_addr          = count_address,
	    .max_count           = max_count_or_count,
	    .stride              = stride_in_bytes,
	    .index_base_addr     = indexed ? indirect.index_base : 0,
	    .index_buffer_size   = indexed ? indirect.index_buffer_size : 0,
	    .index_type_and_size = indirect.index_type,
	    .indexed             = indexed,
	};
	if (const char* reason = IndirectDrawLayout(registers, args.indirect_draw); reason != nullptr) {
		return fallback(reason);
	}
	if (const char* reason =
	        m_renderer.GetRenderExecutor().DrawIndirectGpu(operation.submit_id, CurrentBuffer(), args);
	    reason != nullptr) {
		return fallback(reason);
	}
	static std::atomic_bool logged = false;
	if (!logged.exchange(true, std::memory_order_relaxed)) {
		std::fprintf(stderr,
		             "DIAG_GPU_INDIRECT first GPU indirect draw: indexed=%d max=%u count_addr=0x%llx "
		             "stride=%u sgprs=0x%08x\n",
		             indexed ? 1 : 0, max_count_or_count,
		             static_cast<unsigned long long>(count_address), stride_in_bytes,
		             args.indirect_draw.Key());
	}
	auto& stats = GpuIndirectCounters();
	stats.packets.fetch_add(1, std::memory_order_relaxed);
	stats.max_draws.fetch_add(max_count_or_count, std::memory_order_relaxed);
	if (const uint32_t records_mode = UpstreamIndirectRecordsMode(); records_mode != 0u) {
		// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS: NUM_INSTANCES is now the last record's; =2 also the named registers.
		m_num_instances_records = {.args       = args.args_addr,
		                           .count_addr = count_address,
		                           .max_count  = max_count_or_count,
		                           .stride     = stride_in_bytes,
		                           .sequence   = operation.num_instances.sequence,
		                           .indexed    = indexed};
		if (records_mode == 2u) {
			const auto& sgprs = args.indirect_draw;
			uint32_t    named = 0;
			for (const uint8_t index: {sgprs.base_vertex, sgprs.base_instance, sgprs.draw_index}) {
				if (sgprs.enabled && index != ShaderIndirectDrawInfo::NoSgpr) named |= 1u << index;
			}
			(named == 0 ? stats.sh_nop_packets : stats.named_packets).fetch_add(1, std::memory_order_relaxed);
			CheckStaleIndirectSgprs(operation, named, "gpu_indirect");
			// (Marks the snapshot: what a later draw shares of it.)
			operation.state.shaders->MarkGsIndirectStale(named);
		}
	}
	return true;
}

void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);

	auto operation          = MakeOperation(GpuOperationKind::DrawIndirect);
	operation.draw_indirect = {.args_base         = m_draw_indirect_args_base_addr,
	                           .data_offset       = data_offset,
	                           .max_count         = max_count_or_count,
	                           .count_addr        = count_addr,
	                           .stride            = stride_in_bytes,
	                           .draw_initiator    = draw_initiator,
	                           .index_type        = m_index_type_and_size,
	                           .index_buffer_size = m_index_buffer_size,
	                           .user_data_marker  = static_cast<uint32_t>(m_user_data_marker),
	                           .index_base        = m_index_base_addr,
	                           .indexed           = indexed,
	                           .multi             = true,
	                           .registers         = registers};
	Emit(std::move(operation));
}

// The arguments are read when the draw executes: earlier work may produce them. The registers the CPU expansion writes
// hold for the draw only (the same as the GPU path: D6b, see above).
void CommandProcessor::ExecuteDrawIndirect(const GpuOperation& operation) {
	if (DrawIndirectGpu(operation)) {
		return;
	}

	const auto&    indirect           = operation.draw_indirect;
	const uint32_t max_count_or_count = indirect.max_count;
	const auto*    count_addr         = indirect.count_addr;
	const uint32_t stride_in_bytes    = indirect.stride;
	const bool     indexed            = indirect.indexed;
	const auto&    registers          = indirect.registers;
	const auto     first_record       = indirect.args_base + indirect.data_offset;

	uint32_t draw_count = max_count_or_count;
	if (count_addr != nullptr) {
		draw_count = *count_addr;
		if (draw_count > max_count_or_count) {
			draw_count = max_count_or_count;
		}
	}
	// KYTY_TRACE_INDIRECT_DRAWS[=<frame>]: GPU-generated draw counts/arguments as the CP reads them (the first 400 from
	// that frame on, with the debug marker path).
	static constinit Common::OnceValue<bool> trace_indirect_once;
	const bool trace_indirect = trace_indirect_once.Get([&] { return std::getenv("KYTY_TRACE_INDIRECT_DRAWS") != nullptr; });
	static const uint32_t trace_indirect_from = [] {
		const char* value = std::getenv("KYTY_TRACE_INDIRECT_DRAWS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (trace_indirect) {
		static std::atomic<uint32_t> logged {0};
		static std::atomic<uint32_t> packets {0};
		static std::atomic<uint32_t> zero_packets {0};
		static std::atomic<uint64_t> draws {0};
		static std::atomic<uint32_t> last_frame {0};
		const auto frame = m_renderer.GetGpu().GetFrameNum();
		if (last_frame.exchange(frame) != frame && frame % 30u == 0u) {
			std::fprintf(stderr, "INDIRECT_DRAWS frame=%u packets=%u zero=%u draws=%llu\n", frame,
			             packets.exchange(0), zero_packets.exchange(0),
			             static_cast<unsigned long long>(draws.exchange(0)));
		}
		packets.fetch_add(1);
		draws.fetch_add(draw_count);
		if (draw_count == 0) zero_packets.fetch_add(1);
		if (frame >= trace_indirect_from && logged.fetch_add(1) < 400u) {
			auto&      cache    = m_renderer.GetBufferCache();
			const auto args     = first_record;
			const bool count_gpu = count_addr != nullptr &&
			                       cache.IsRegionGpuModified(reinterpret_cast<uint64_t>(count_addr), 4);
			const bool args_gpu = cache.IsRegionGpuModified(args, uint64_t {stride_in_bytes} *
			                                                          std::max(draw_count, 1u));
			const auto* first = reinterpret_cast<const uint32_t*>(args);
			ShaderMappedData shader_data {};
			uint64_t         gs_hash = 0, ps_hash = 0;
			(void)ShaderLookupMapped(operation.state.shaders->GetVs().gs_regs.data_addr, &shader_data, &gs_hash);
			(void)ShaderLookupMapped(operation.state.shaders->GetPs().ps_regs.data_addr, &shader_data, &ps_hash);
			std::fprintf(stderr, "INDIRECT_DRAW_SHADERS gs=0x%016llx ps=0x%016llx\n", static_cast<unsigned long long>(gs_hash),
			             static_cast<unsigned long long>(ps_hash));
			std::fprintf(stderr,
			             "INDIRECT_DRAW frame=%u indexed=%d count_addr=0x%llx count_gpu=%d max=%u count=%u "
			             "args=0x%llx args_gpu=%d stride=%u first=%u,%u,%u,%u,%u marker=[%s]\n",
			             frame, indexed ? 1 : 0, static_cast<unsigned long long>(reinterpret_cast<uint64_t>(count_addr)),
			             count_gpu ? 1 : 0, max_count_or_count, draw_count, static_cast<unsigned long long>(args),
			             args_gpu ? 1 : 0, stride_in_bytes, first[0], first[1], first[2], first[3], first[4],
			             DebugMarkerPath().c_str());
		}
	}

	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	// The guest records match Vulkan's indirect commands byte for byte.
	const auto args_size =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	EXIT_NOT_IMPLEMENTED(stride_in_bytes < args_size);

	// Upstream c513a4a13, 949de19bb and 86fed10b8 (draws from GPU-read records through RenderExecutor::DrawIndex/DrawAuto
	// with DrawIndirectArgs) are not taken: they need upstream's renderer path, which ours does not have. Our GPU path is
	// The GPU path above; the rest is the CPU expansion.

	uint64_t index_size = 0;
	if (indexed) {
		switch (indirect.index_type) {
			case 0: index_size = 2; break;
			case 1: index_size = 4; break;
			case 2: index_size = 1; break;
			default: EXIT("unknown index_type_and_size: %u\n", indirect.index_type);
		}
	}

	// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS: NUM_INSTANCES and the named registers are written below, record by record.
	m_num_instances_records = {};
	if (UpstreamIndirectRecordsMode() == 2u) {
		uint32_t named = 0;
		for (const uint32_t location: {registers.vertex_offset, registers.instance_offset, registers.index_offset}) {
			if (location >= Pm4::SPI_SHADER_USER_DATA_GS_0 && location < Pm4::SPI_SHADER_USER_DATA_GS_0 + 32u) {
				named |= 1u << (location - Pm4::SPI_SHADER_USER_DATA_GS_0);
			}
		}
		CheckStaleIndirectSgprs(operation, named, "cpu_indirect");
	}
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto args_addr = first_record + static_cast<uint64_t>(i) * stride_in_bytes;

		if (!indexed) {
			const auto& args = *reinterpret_cast<const vk::DrawIndirectCommand*>(args_addr);
			WriteIndirectOffsets(operation, args.firstVertex, args.firstInstance, 0);
			m_indirect_instances          = args.instanceCount;
			m_indirect_instances_sequence = operation.num_instances.sequence;
			m_has_indirect_instances      = true;
			ExecuteDrawAuto(operation, {.vertex_count   = args.vertexCount,
			                            .instance_count = args.instanceCount,
			                            .first_vertex   = args.firstVertex,
			                            .first_instance = args.firstInstance,
			                            .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		const auto& args = *reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(args_addr);
		WriteIndirectOffsets(operation, static_cast<uint32_t>(args.vertexOffset), args.firstInstance, args.firstIndex);

		auto* index_addr = reinterpret_cast<const void*>(
		    indirect.index_base + static_cast<uint64_t>(args.firstIndex) * index_size);

		const uint32_t index_count =
		    (indirect.index_buffer_size != 0 ? std::min(args.indexCount, indirect.index_buffer_size)
		                                     : args.indexCount);
		if (GraphicsRunDebugDumpEnabled() && index_count != args.indexCount) {
			static std::atomic<uint32_t> log_count {0};
			if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("\t DrawIndexIndirectMulti: clamped index_count from %" PRIu32 " to %" PRIu32
				     " using INDEX_BUFFER_SIZE\n",
				     args.indexCount, index_count);
			}
		}

		const uint32_t instance_count = args.instanceCount;
		if (instance_count == 0u && index_count != 0u) {
			static std::atomic<uint32_t> zero_instances {0};
			static std::atomic<uint32_t> zero_frame {0};
			const auto frame = m_renderer.GetGpu().GetFrameNum();
			if (trace_indirect && zero_frame.exchange(frame) != frame && frame % 30u == 0u) {
				std::fprintf(stderr, "INDIRECT_ZERO_INSTANCES frame=%u count=%u\n", frame, zero_instances.exchange(0));
			}
			zero_instances.fetch_add(1);
			if (DebugRtCollecting()) {
				const auto       path = DebugMarkerPath();
				std::scoped_lock lock(DebugRtMutex());
				auto&            zero = DebugZeroDraws();
				auto it = std::ranges::find_if(zero, [&](const auto& entry) { return entry.first == path; });
				if (it != zero.end()) {
					it->second++;
				} else if (zero.size() < 128u) {
					zero.emplace_back(path, 1u);
				}
			}
		}
		m_indirect_instances          = instance_count;
		m_indirect_instances_sequence = operation.num_instances.sequence;
		m_has_indirect_instances      = true;
		ExecuteDrawIndex(operation, {.index_count         = index_count,
		                             .index_addr          = index_addr,
		                             .instance_count      = instance_count,
		                             .index_type_and_size = indirect.index_type,
		                             .base_vertex         = static_cast<int32_t>(args.vertexOffset),
		                             .first_instance      = args.firstInstance,
		                             .offset_source       = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::SetCsWaveSize(uint32_t mode) {
	const auto wave_size = Pm4::ComputeWaveSize(mode);
	if (m_sh_ctx.GetCs().cs_regs.wave_size != wave_size) GetShCtx().SetCsWaveSize(wave_size);
}

// KYTY_TRACE_PASS_CENSUS: the dispatch's compute shader for its pass's shader list.
static void NoteCensusShader(const GpuOperation& operation) {
	if (!PassCensus::Enabled()) return;
	ShaderMappedData data {};
	uint64_t         hash = 0;
	if (ShaderLookupMapped(operation.state.shaders->GetCs().cs_regs.data_addr, &data, &hash)) PassCensus::NoteShader(hash);
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	SetCsWaveSize(mode);
	auto operation     = MakeOperation(GpuOperationKind::DispatchDirect);
	NoteCensusShader(operation);
	operation.dispatch = {.thread_group_x = thread_group_x,
	                      .thread_group_y = thread_group_y,
	                      .thread_group_z = thread_group_z,
	                      .mode           = mode};
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x,
                                             uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode) {
	if (GraphicsRunDebugDumpEnabled()) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 1024) {
			const auto  frame_num = m_renderer.GetGpu().GetFrameNum();
			const auto& ucfg      = *operation.state.user_config;
			const auto& cs        = operation.state.shaders->GetCs().cs_regs;
			const auto& oa        = ucfg.GetGdsOaCounter(ucfg.GetGdsOaState().GetIndex());
			LOGF("QueuePoint DispatchDirect: frame=%u submit=%" PRIu64
			     " groups=%ux%ux%u local=%ux%ux%u mode=0x%08" PRIx32 " wave=%u cs=0x%016" PRIx64
			     " oa_index=%u oa_enabled=%s oa_addr=0x%04" PRIx32 " oa_space=0x%08" PRIx32
			     "\n",
			     frame_num, operation.submit_id, thread_group_x, thread_group_y, thread_group_z,
			     std::max(cs.num_thread_x, 1u), std::max(cs.num_thread_y, 1u),
			     std::max(cs.num_thread_z, 1u), mode, static_cast<uint32_t>(cs.wave_size),
			     cs.data_addr, ucfg.GetGdsOaState().GetIndex(),
			     oa.IsCounterEnabled() ? "true" : "false", oa.GetAddressBytes(),
			     oa.GetSpaceAvailable());
		}
	}

	m_renderer.GetRenderExecutor().DispatchDirect(operation.submit_id, CurrentBuffer(), thread_group_x,
	                                              thread_group_y, thread_group_z, mode, IsAsyncComputeQueue());
}


void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	SetCsWaveSize(mode);
	auto operation     = MakeOperation(GpuOperationKind::DispatchIndirect);
	NoteCensusShader(operation);
	operation.dispatch = {.mode = mode, .args_addr = args_addr};
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDispatchIndirect(const GpuOperation& operation) {
	const auto& dispatch = operation.dispatch;
	if ((dispatch.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		// The arguments are read when the dispatch executes: earlier work may produce them.
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(dispatch.args_addr);
		ExecuteDispatchDirect(operation, args->x, args->y, args->z, dispatch.mode);
		return;
	}
	m_renderer.GetRenderExecutor().DispatchIndirect(operation.submit_id, CurrentBuffer(), dispatch.args_addr, dispatch.mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	auto operation      = MakeOperation(GpuOperationKind::DrawAuto);
	operation.draw_auto = args;
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDrawAuto(const GpuOperation& operation, DrawAutoArgs args) {
	if (args.instance_count == 0) {
		args.instance_count = ResolveNumInstances(operation);
	}
	if (UpstreamIndirectRecordsMode() == 2u && args.offset_source != DrawOffsetSource::IndirectArgs) {
		CheckStaleIndirectSgprs(operation, 0, "draw_auto");
	}
	m_renderer.GetRenderExecutor().DrawAuto(operation.submit_id, CurrentBuffer(), args);
}

// KYTY_UPSTREAM_FLIP_SUSPEND=1 (upstream a358b847c, default off): waitUntilSafeForRendering suspends only this command
// processor until the display buffer is neither on screen nor waiting for a flip (the flip queue wakes the GPU thread),
// instead of blocking the GPU thread until no flip of that buffer is pending. Off (default): our blocking wait; the
// presenter copies a flipped buffer when it prepares the flip, so rendering into the buffer on screen changes no image.
static bool UpstreamFlipSuspendEnabled() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] {
		const char* value = std::getenv("KYTY_UPSTREAM_FLIP_SUSPEND");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) std::fprintf(stderr, "UPSTREAM_FLIP_SUSPEND enabled\n");
		return on;
	});
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	// (Not on the command processor thread: the flips it ran ahead of have not made the buffer busy yet. The execution
	// thread waits in operation order.)
	if (UpstreamFlipSuspendEnabled() && !g_cp_thread) {
		// Stall only this command processor, not the host command lane or other queues.
		if (!m_renderer.GetVideoOut().IsSafeForRendering(static_cast<int>(video_out_handle),
		                                                 static_cast<int>(display_buffer_index))) {
			SuspendPm4();
		}
		return;
	}
	EmitCallback([this, video_out_handle, display_buffer_index] {
		GetScheduler().Flush();

		SyncTimeline::WaitScope timeline(SyncTimeline::Site::FlipDone, 0, 0, 0); // KYTY_TRACE_SYNC_TIMELINE
		m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
		                                      static_cast<int>(display_buffer_index));
	});
}

template <typename T>
void CommandProcessor::WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest,
                                        uint32_t eop_event_type, uint32_t cache_action,
                                        uint32_t event_index, uint32_t event_write_source,
                                        void* dst_gpu_addr, T value, uint32_t interrupt_selector,
                                        uint32_t interrupt_context_id) {
	static_assert(sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t));

	if (GraphicsRunDebugDumpEnabled()) {
		const auto bits      = static_cast<unsigned>(sizeof(T) * 8u);
		const auto log_width = static_cast<int>(sizeof(T) * 2u);

		LOGF("CommandProcessor::WriteAtEndOfPipe%u()\n"
		     "\t cache_policy        = 0x%08" PRIx32 "\n"
		     "\t event_write_dest    = 0x%08" PRIx32 "\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t event_index         = 0x%08" PRIx32 "\n"
		     "\t event_write_source  = 0x%08" PRIx32 "\n"
		     "\t interrupt_selector  = 0x%08" PRIx32 "\n"
		     "\t interrupt_context   = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%0*" PRIx64 "\n",
		     bits, cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
		     event_write_source, interrupt_selector, interrupt_context_id,
		     reinterpret_cast<uint64_t>(dst_gpu_addr), log_width, static_cast<uint64_t>(value));
	}

	EXIT_NOT_IMPLEMENTED(cache_policy != 0x00000000);
	EXIT_NOT_IMPLEMENTED(event_write_dest != 0x00000000);

	bool with_interrupt = false;
	switch (interrupt_selector) {
		case 0x00:
		case 0x03: with_interrupt = false; break;
		case 0x01:
			if (!IsAsyncComputeQueue()) {
				TriggerEopEventAtEndOfPipe(interrupt_context_id);
				return;
			}
			with_interrupt = true;
			break;
		case 0x02: with_interrupt = true; break;
		default: EXIT("unknown interrupt_selector\n");
	}

	const auto write =
	    ClassifyEndOfPipe<T>(event_write_source, eop_event_type, cache_action, event_index, with_interrupt);
	if (write == EndOfPipeWrite::None) {
		EXIT("unknown event type\n");
	}
	const bool gpu_clock = event_write_source == 0x04;
	uint64_t   data      = value;
	if (write == EndOfPipeWrite::Qword && gpu_clock) {
		data = Sync::ReadReferenceClock();
	}
	auto* const gpu         = StreamGpu();
	const auto  sequence    = gpu != nullptr ? gpu->NextSequence() : 0;
	const auto  destination = reinterpret_cast<uint64_t>(dst_gpu_addr);
	// A label lands when the host GPU got here; its value is known now (a virtual GPU clock's value only when it lands).
	const bool stamped = write == EndOfPipeWrite::Qword && gpu_clock && VirtualGpuClock::Enabled();
	if (gpu != nullptr && write == EndOfPipeWrite::Dword) {
		const auto label = static_cast<uint32_t>(data);
		gpu->NoteDeferredWrite(sequence, destination, sizeof(label), &label);
	} else if (gpu != nullptr && write == EndOfPipeWrite::Qword && !stamped) {
		gpu->NoteDeferredWrite(sequence, destination, sizeof(data), &data);
	}
	EmitCallback([=, this] {
		ExecuteEndOfPipe(write, sequence, dst_gpu_addr, data, gpu_clock, with_interrupt, interrupt_context_id);
	});
	if (write == EndOfPipeWrite::Gds) {
		NoteGuestWrite(sequence, destination, uint64_t {static_cast<uint32_t>(data) >> 16u} * sizeof(uint32_t));
	}
}

// What the packet writes; None: the combination is not implemented.
template <typename T>
CommandProcessor::EndOfPipeWrite CommandProcessor::ClassifyEndOfPipe(uint32_t event_write_source, uint32_t eop_event_type,
                                                                     uint32_t cache_action, uint32_t event_index,
                                                                     bool with_interrupt) {
	switch (event_write_source) {
		case 0x01:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (eop_event_type == 0x2f && cache_action == 0x00 && event_index == 0x06) return EndOfPipeWrite::Gds;
			} else if (eop_event_type == 0x04 && cache_action == 0x00 && event_index == 0x05) {
				return EndOfPipeWrite::Dword;
			}
			break;
		case 0x02:
		case 0x04:
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				if (event_write_source == 0x02 && eop_event_type == 0x2f && event_index == 0x06 &&
				    (cache_action == 0x00 || cache_action == 0x38)) {
					return EndOfPipeWrite::Dword;
				}
			} else {
				switch (cache_action) {
					case 0x00:
						switch (eop_event_type) {
							case 0x04:
								if (event_index == 0x05) return EndOfPipeWrite::Qword;
								break;
							case 0x14:
							case 0x28:
							case 0x2f:
								if (event_index == 0x00) return EndOfPipeWrite::Qword;
								break;
							case 0x2b:
							case 0x2d:
							case 0x30:
								if (event_index == 0x00 && !with_interrupt) return EndOfPipeWrite::Qword;
								break;
							default: break;
						}
						break;
					case 0x38:
						switch (eop_event_type) {
							case 0x04:
							case 0x14:
							case 0x28:
								if (((eop_event_type == 0x04 || eop_event_type == 0x28) && event_index == 0x05) ||
								    event_index == 0x00) {
									return EndOfPipeWrite::Qword;
								}
								break;
							case 0x2b:
							case 0x2d:
								if (event_index == 0x00 && !with_interrupt) return EndOfPipeWrite::Qword;
								break;
							case 0x2f:
								if (event_index == 0x06 && !with_interrupt) return EndOfPipeWrite::Qword;
								break;
							default: break;
						}
						break;
					case 0x3b:
						if (eop_event_type == 0x04 && event_index == 0x05 && with_interrupt) return EndOfPipeWrite::Qword;
						break;
					default: break;
				}
			}
			break;
		default: break;
	}
	return EndOfPipeWrite::None;
}

void CommandProcessor::ExecuteEndOfPipe(EndOfPipeWrite write, uint64_t sequence, void* dst_gpu_addr, uint64_t value,
                                        bool gpu_clock, bool with_interrupt, uint32_t interrupt_context_id) {
	auto& command = CurrentBuffer();
	switch (write) {
		case EndOfPipeWrite::Gds: {
			auto* dst       = static_cast<uint32_t*>(dst_gpu_addr);
			const auto data = static_cast<uint32_t>(value);
			GetScheduler().Finish();
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(), dst, data & 0xffffu, data >> 16u);
			PublishCpuDirtyRange(reinterpret_cast<uint64_t>(dst), // BDA sync epoch
			                     uint64_t {data >> 16u} * sizeof(uint32_t));
			if (with_interrupt) {
				m_renderer.TriggerInterrupt(m_interrupt_event_id, interrupt_context_id);
			}
			return;
		}
		case EndOfPipeWrite::Dword: {
			const auto data = static_cast<uint32_t>(value);
			EXIT_IF(dst_gpu_addr == nullptr);
			(void)command.Handle();
			if (g_gpu_state == nullptr) {
				WriteLabelNow(dst_gpu_addr, &data, sizeof(data), with_interrupt, interrupt_context_id);
				return;
			}
			// Labels at completion: the label (and its interrupt) once the host GPU got here.
			DeferLabel(GetScheduler(), m_renderer, m_interrupt_event_id, sequence, reinterpret_cast<uint64_t>(dst_gpu_addr),
			           data, sizeof(data), with_interrupt, interrupt_context_id);
			return;
		}
		case EndOfPipeWrite::Qword: {
			EXIT_IF(dst_gpu_addr == nullptr);
			(void)command.Handle();
			// KYTY_FIX_VIRTUAL_GPU_CLOCK: the GPU clock value lands once the GPU got here (virtualGpuClock.h).
			if (gpu_clock && VirtualGpuClock::Stamp(GetScheduler(), command.Handle(), reinterpret_cast<uint64_t>(dst_gpu_addr),
			                                        with_interrupt)) {
				if (with_interrupt) {
					DeferEopInterrupt(interrupt_context_id);
				}
				return;
			}
			if (g_gpu_state == nullptr) {
				WriteLabelNow(dst_gpu_addr, &value, sizeof(value), with_interrupt, interrupt_context_id);
				return;
			}
			// labels at completion (as the dword; a clock the virtual GPU clock refused after all has no journal entry: a
			// wait for it passes when it lands).
			DeferLabel(GetScheduler(), m_renderer, m_interrupt_event_id, sequence, reinterpret_cast<uint64_t>(dst_gpu_addr),
			           value, sizeof(value), with_interrupt, interrupt_context_id);
			return;
		}
		case EndOfPipeWrite::None: break;
	}
}

// Off the GPU thread (the tests): the label lands when its packet is translated.
void CommandProcessor::WriteLabelNow(void* dst_gpu_addr, const void* value, uint32_t size, bool with_interrupt,
                                     uint32_t interrupt_context_id) {
	const auto dst = reinterpret_cast<uint64_t>(dst_gpu_addr);
	// A GPU-owned page takes the label without a fault when the GPU owns none of its bytes.
	if (!m_renderer.GetBufferCache().TryCpWriteThrough(dst, value, size)) {
		std::memcpy(dst_gpu_addr, value, size);
	}
	if (SyncTimeline::Enabled()) {
		uint64_t label = 0;
		std::memcpy(&label, value, size);
		SyncTimeline::GuestLabel(dst, label, IsAsyncComputeQueue(), GetScheduler().CurrentTick(),
		                         GetScheduler().GetMasterSemaphore().KnownGpuTick());
	}
	PublishCpuDirtyRange(dst, size); // BDA sync epoch
	if (with_interrupt) {
		DeferEopInterrupt(interrupt_context_id);
	}
}

void CommandProcessor::WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint32_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
                                          uint32_t eop_event_type, uint32_t cache_action,
                                          uint32_t event_index, uint32_t event_write_source,
                                          void* dst_gpu_addr, uint64_t value,
                                          uint32_t interrupt_selector,
                                          uint32_t interrupt_context_id) {
	WriteAtEndOfPipe(cache_policy, event_write_dest, eop_event_type, cache_action, event_index,
	                 event_write_source, dst_gpu_addr, value, interrupt_selector,
	                 interrupt_context_id);
}

void CommandProcessor::EmitGlobalBarrier() {
	EmitCallback([this] { RecordGlobalBarrier(); });
}

void CommandProcessor::RecordGlobalBarrier() {
	Common::LockGuard lock(m_renderer.GetMutex());

	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;

	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	GetScheduler().EndRendering();
	CurrentBuffer().Handle().pipelineBarrier2(dependency);
}

void CommandProcessor::TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id) {
	EmitCallback([this, interrupt_context_id] { DeferEopInterrupt(interrupt_context_id); });
}

void CommandProcessor::DeferEopInterrupt(uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
		    renderer.TriggerInterrupt(event_id, interrupt_context_id);
	    });
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto sequence = EmitCallback([this, event_type, event_index, event_address] {
		ExecuteTriggerEvent(event_type, event_index, event_address);
	});
	if (event_type == 0x00000039) {
		// Occlusion counter pairs of 16 DBs.
		NoteGuestWrite(sequence, event_address, 16u * 2u * sizeof(uint64_t));
	}
}

void CommandProcessor::ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address) {
	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks record a full barrier here (upstream 70668fbd9 relies on the renderer's
	// resource barriers alone; in our renderer these barriers order the game's work: consecutive dispatches have no other
	// barrier).
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010:
			RecordGlobalBarrier();
			break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			RecordGlobalBarrier();
			break;
		// DbDataWritebackInvalidate, DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			RecordGlobalBarrier();
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			auto& occlusion = m_renderer.GetOcclusionQueries();
			// KYTY_FIX_OCCLUSION_QUERIES=1: host occlusion queries count the samples and the GPU writes the
			// counters here, in command order.
			if (OcclusionQueriesRequested()) {
				Common::LockGuard lock(m_renderer.GetMutex());
				if (occlusion.Dump(CurrentBuffer(), event_address)) {
					break;
				}
			}
			// Which guest queries are open (draw-drop diagnostics), also for the synthetic counters below.
			occlusion.NoteSyntheticDump(event_address);
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are currently treated as "
				            "always visible; GPU usage may be higher and FPS may be lower.\n");
			});

			// Until host occlusion queries are implemented, publish an always-visible result. The
			// PS5 layout contains one interleaved begin/end pair per DB, and bit 63 marks a result
			// ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			PublishCpuDirtyRange(event_address, 32u * sizeof(uint64_t)); // BDA sync epoch
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::WriteLodStats(void* dst, uint32_t size) {
	const auto sequence = EmitCallback([dst, size] {
		memset(dst, 0, size);
		// Hack?
		if (size >= sizeof(uint32_t)) {
			auto* label = static_cast<uint32_t*>(dst);
			*label      = 1;
		}
		// BDA sync epoch: the commands after this one see the GPU thread's own write.
		PublishCpuDirtyRange(reinterpret_cast<uint64_t>(dst), size);
	});
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), size);
}

// KYTY_LIVE_AB: each flip ends with a submission (Flush), which also resets the scheduler's batch state; right after
// it is the frame boundary where live A/B phases change, so no frame runs partly in one phase and partly in another.
static void LiveAbFrameBoundary() {
	if (GuestGpu::IsGpuThread()) Common::LiveAb::OnFrameBoundary(DebugPresentCount().load());
	// KYTY_DIAG_GPU_BURN_FLIP_US=<us> (measurement only, default 0; KYTY_LIVE_AB toggles 0/5000): the GPU thread spins that
	// long after each flip, where no other stage is waited for: how much frame time GPU-thread work costs.
	static const uint32_t burn_fixed = [] {
		const char* value = std::getenv("KYTY_DIAG_GPU_BURN_FLIP_US");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	if (const uint32_t burn = burn_fixed; burn != 0 && GuestGpu::IsGpuThread()) {
		const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(burn);
		while (std::chrono::steady_clock::now() < until) {
		}
	}
}

void CommandProcessor::Flip() {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	EmitCallback([this, flip = m_flip] {
		auto& command = CurrentBuffer();
		auto  request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index, flip.flip_mode, flip.flip_arg);
		(void)command.Handle();
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
		GetScheduler().Flush();
		LiveAbFrameBoundary();
	});
}

void CommandProcessor::Flip(void* dst_gpu_addr, uint32_t value) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::Flip()\n"
		     "\t dst_gpu_addr = 0x%016" PRIx64 "\n"
		     "\t value        = 0x%08" PRIx32 "\n",
		     reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	EXIT_IF(dst_gpu_addr == nullptr);
	const auto sequence = EmitCallback([this, flip = m_flip, dst_gpu_addr, value] {
		std::memcpy(dst_gpu_addr, &value, sizeof(value));
		PublishCpuDirtyRange(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value)); // BDA sync epoch
		auto& command = CurrentBuffer();
		auto  request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index, flip.flip_mode, flip.flip_arg);
		(void)command.Handle();
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
		GetScheduler().Flush();
		LiveAbFrameBoundary();
	});
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value), &value);
}

void CommandProcessor::FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action,
                                         void* dst_gpu_addr, uint32_t value) {
	if (GraphicsRunDebugDumpEnabled()) {
		LOGF("CommandProcessor::FlipWithInterrupt()\n"
		     "\t eop_event_type      = 0x%08" PRIx32 "\n"
		     "\t cache_action        = 0x%08" PRIx32 "\n"
		     "\t dst_gpu_addr        = 0x%016" PRIx64 "\n"
		     "\t value               = 0x%08" PRIx32 "\n",
		     eop_event_type, cache_action, reinterpret_cast<uint64_t>(dst_gpu_addr), value);
	}

	if (eop_event_type != 0x00000004 || cache_action != 0x00000038) {
		EXIT("unknown event type\n");
	}
	EXIT_IF(dst_gpu_addr == nullptr);
	const auto sequence = EmitCallback([this, flip = m_flip, dst_gpu_addr, value] {
		std::memcpy(dst_gpu_addr, &value, sizeof(value));
		PublishCpuDirtyRange(reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value)); // BDA sync epoch
		auto& command = CurrentBuffer();
		auto  request = Sync::PrepareVideoOutFlip(command, flip.handle, flip.index, flip.flip_mode, flip.flip_arg);
		(void)command.Handle();
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, event_id = m_interrupt_event_id, request] {
			    renderer.GetVideoOut().CompleteFlip(request);
			    renderer.TriggerInterrupt(event_id, 0);
		    });
		GetScheduler().Flush();
		LiveAbFrameBoundary();
	});
	NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst_gpu_addr), sizeof(value), &value);
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	EmitCallback([this, request_id] {
		struct ProcessorScope {
			explicit ProcessorScope(CommandProcessor& processor) { g_current_processor = &processor; }
			~ProcessorScope() { g_current_processor = nullptr; }
		};
		ProcessorScope processor_scope(*this);

		m_renderer.GetVideoOut().PrepareFlip(request_id, CurrentBuffer());
		GetScheduler().DeferPriorityOperation(
		    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
		GetScheduler().Flush();
		LiveAbFrameBoundary();
	});
}

void CommandProcessor::SynchronizeGpu() {
	EmitCallback([this] { GetScheduler().Finish(); });
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}


} // namespace Libs::Graphics
