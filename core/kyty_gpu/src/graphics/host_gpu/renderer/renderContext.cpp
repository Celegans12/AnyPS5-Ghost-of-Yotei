#include "common/profiler.h"
#include "graphics/host_gpu/renderer/indirectLoadFast.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"

#include "common/assert.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

namespace {

// The BDA sync epoch: PrepareBda's pass runs only when guest writes became visible to the GPU work that follows, not
// whenever any guest thread wrote anywhere. A guest CPU write is visible only to GPU work the guest ordered after it (a
// later submission, or a label a WAIT_REG_MEM or a packet deciding on guest memory read): the regions a guest thread's
// write fault dirties are deferred (regionManager.h) and the GPU thread publishes them at those points (GuestGpu::Process,
// WaitRegMem, COND_EXEC, the conditional INDIRECT_BUFFER, SET_PREDICATION, REWIND); the GPU thread's own writes and
// asynchronous completions stay visible at once, and CP packets writing guest memory publish their range. Bound buffers
// still synchronize their exact ranges at every draw.

} // namespace

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      // No image heap source: upstream's image heap needs upstream's MaterializeIndirectDescriptor, not taken -
      // SESJA-SYNC1 (m_image_heap).
      m_pipeline_cache(graphics, nullptr), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_bindless_table(graphics, m_command_scheduler) {
	m_texture_cache.on_bindless_unregister = [this](ImageId id, const char* reason) {
		m_bindless_table.OnImageUnregistered(id, reason);
	};
	// The list path of bindless tables over 1 MiB: their left-out LRU touches, answered for the cache (none without the
	// tables).
	if (m_bindless_table.TablesEnabled()) {
		m_texture_cache.bindless_fast_keep = [this](ImageId id, uint64_t age) {
			return m_bindless_table.FastTouchKeeps(id, age);
		};
	}
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	// Background shader compiles may use the bindless table, which is destroyed before the pipeline cache.
	m_pipeline_cache.StopBackgroundCompiles();
	m_command_scheduler.Shutdown();
	// The bindless table is destroyed before the texture cache, whose destructor unregisters images.
	m_texture_cache.on_bindless_unregister = nullptr;
	m_texture_cache.bindless_fast_keep     = nullptr;
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// Image read coherence (textureCache.cpp): only a guest thread's read is served; a guest thread is the one the guest
	// kernel knows, HLE code it runs included. The access runs again when this returns.
	const bool is_write = access == PageFaultAccess::Write;
	if (m_page_manager.ImageReadArmed(fault_vaddr)) {
		const bool guest_thread = LibKernel::PthreadSelfOrNull() != nullptr;
		if (guest_thread && !is_write) {
			ServeImageRead(fault_vaddr);
		} else {
			TextureCache::NoteImageReadLift(fault_vaddr, !guest_thread);
			LiftImageRead(fault_vaddr, fault_size, is_write);
		}
		return true;
	}
	// Another thread opened the page first: the fault would not have happened now.
	if (m_page_manager.WatchersAllow(fault_vaddr, is_write)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		KYTY_PROFILER_BLOCK("Fault::Write");
		if (BufferCache::UploadClobberCheck()) m_buffer_cache.NoteGuestWriteFault(fault_vaddr); // KYTY_D6_C0=2/3
		{
			// The BDA sync epoch: a guest thread's write waits for the next point that orders it before GPU work; the GPU
			// thread's own writes and asynchronous completions stay visible at once.
			CpuDirtyDeferScope defer(!GuestGpu::IsGpuThread() && !CommandScheduler::InDeferredOperation());
			m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		}
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		// (Upstream also invalidates its image heap here; not in use, see m_image_heap.)
	} else {
		// Read faults that reach the synchronizing path (counted).
		static std::atomic_uint read_faults {0};
		if (((read_faults.fetch_add(1) + 1) & 0xfffu) == 0 && Common::EnvRead::PerfLines()) {
			std::fprintf(stderr, "FAULT_SYNC reads=%u last=0x%llx\n", read_faults.load(),
			             static_cast<unsigned long long>(fault_vaddr));
		}
		KYTY_PROFILER_BLOCK("Fault::Read");
		PerfReasonScope reason(GuestGpu::IsGpuThread() ? PerfReason::FaultGpu : PerfReason::FaultGame);
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

void RenderContext::ServeImageRead(uint64_t vaddr) {
	if (m_gpu != nullptr && m_gpu->IsStopping()) {
		m_page_manager.DisarmImageRead(vaddr, 1); // nothing serves any more
		return;
	}
	TextureCache::CpuReadServe serve;
	const auto                 request = [this, &serve, vaddr] { serve = m_texture_cache.BeginCpuRead(vaddr); };
	if (m_gpu == nullptr) {
		request();
	} else {
		m_gpu->SendCommandSync(request);
	}
	const auto start = std::chrono::steady_clock::now();
	if (serve.tick != 0) {
		m_command_scheduler.GetMasterSemaphore().Wait(serve.tick);
		// An older publication of the same bytes (an eviction's download) must not land after this one.
		m_command_scheduler.WaitPriorityOperations(serve.tick);
	}
	m_texture_cache.FinishCpuRead(serve);
	TextureCache::NoteImageReadWait(static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()));
}

void RenderContext::LiftImageRead(uint64_t vaddr, uint64_t size, bool is_write) {
	// The bit goes at once (the region's lock only: the caller may hold anything, the GPU thread included); the images, under
	// m_lock, at the GPU thread's next packet boundary, never in this fault.
	m_page_manager.DisarmImageRead(vaddr, size);
	const auto settle = [this, vaddr, size, is_write] { m_texture_cache.SettleImageReadLift(vaddr, size, is_write); };
	if (m_gpu == nullptr) {
		settle();
	} else if (!m_gpu->IsStopping()) {
		m_gpu->PostCommand(settle);
	}
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	// An emulator write may run in kernel mode (a file read into guest memory), where a no-access page fails the call instead
	// of faulting: the image read watch is lifted first.
	if (m_page_manager.ImageReadArmedInRange(vaddr, size)) {
		TextureCache::NoteImageReadLift(vaddr, false);
		LiftImageRead(vaddr, size, true);
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	m_buffer_cache.QueueFallbackMap(vaddr, size); // the BDA fallback (M06)
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	// Upstream queues the range for its BDA sync here (d58aefde4, BufferCache::SynchronizeBda); PrepareBda takes the
	// CPU-dirty tracker regions instead (the BDA region pass, the BDA sync epoch).
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto              tick = m_command_scheduler.CurrentTick();
			SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Unmap); // KYTY_TRACE_SYNC_TIMELINE
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		if (m_command_scheduler.Active()) m_buffer_cache.UnmapDirect(vaddr, size); // KYTY_PERF_DIRECT_RANGES
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_buffer_cache.ClearFallback(vaddr, size); // the BDA fallback (M06)
		// No image read watch outlives the mapping (lifted while it is still mapped).
		if (GuestRange {vaddr, size}.Valid()) {
			m_page_manager.DisarmImageRead(vaddr, size);
		}
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	// KYTY_TRACE_LONG_PREP: the parts' times and the region pass's counts (t_long_prep_parts 21-29).
	const bool parts_on  = LongPrepPartsOn();
	auto       part_last = parts_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const auto part_mark = [&](size_t part) {
		if (!parts_on) return;
		const auto now = std::chrono::steady_clock::now();
		t_long_prep_parts[part] +=
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - part_last).count());
		part_last = now;
	};
	m_buffer_cache.ApplyFallbackMaps(); // the BDA fallback (M06)
	part_mark(21);
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	// Nothing became CPU-modified and no buffer was created since the last pass: every buffer is
	// already synchronized (each check below would find no upload range).
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	if (epoch != m_bda_synced_epoch) {
		// The epoch is read before the pending regions are taken: a region dirtied after that bumps
		// the epoch again (and sets its pending bit again), so the next call handles it.
		KYTY_PROFILER_BLOCK("PrepareBda::RegionPass");
		std::shared_lock lock(m_mapped_ranges_mutex);
		BufferSyncStats  stats;
		uint64_t         run_start = 0;
		uint64_t         run_end   = 0;
		// Only mapped memory (a buffer can outlive the mapping of its range). Buffers that start
		// before the slice are found by SynchronizeBuffersInRange itself.
		const auto flush_run = [&] {
			if (run_end != run_start) {
				m_mapped_ranges.ForEachInRange(run_start, run_end - run_start,
				                               [&](uint64_t begin, uint64_t end) {
					                               m_buffer_cache.SynchronizeBuffersInRange(
					                                   begin, end - begin, &stats);
				                               });
			}
		};
		{
			// The upload batch: the pass's read-only synchronizations write-protect their pages with one
			// update per tracker region at the end of the pass (before any of their bytes is copied).
			const BufferCache::UploadBatchScope upload_batch(m_buffer_cache);
			TakeCpuDirtyRegions([&](uint64_t region) {
				const uint64_t start = region * TRACKER_REGION_SIZE;
				m_bda_counters.regions++;
				if (parts_on) t_long_prep_parts[26]++;
				if (start != run_end) {
					flush_run();
					run_start = start;
				}
				run_end = start + TRACKER_REGION_SIZE;
			});
			flush_run();
		}
		m_bda_counters.passes++;
		m_bda_counters.buffers_visited += stats.buffers;
		if (parts_on) {
			t_long_prep_parts[27] += stats.buffers;
			t_long_prep_parts[28] += stats.uploads;
			t_long_prep_parts[29] += stats.bytes;
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - m_bda_log_time >= std::chrono::seconds(10) && Common::EnvRead::PerfLines()) {
			m_bda_log_time = now;
			std::fprintf(stderr, "BDA_REGION_STATS passes=%llu regions=%llu buffers_visited=%llu\n",
			             static_cast<unsigned long long>(m_bda_counters.passes),
			             static_cast<unsigned long long>(m_bda_counters.regions),
			             static_cast<unsigned long long>(m_bda_counters.buffers_visited));
		}
		m_bda_synced_epoch = epoch;
		m_bda_epoch_counters.passes++;
	}
	part_mark(24);
	ReportBdaSyncEpoch();
	part_mark(25);
	m_fault_process_pending = true;
}

// The PERF line of the BDA sync epoch about every 10 s.
void RenderContext::ReportBdaSyncEpoch() {
	auto& counters = m_bda_epoch_counters;
	counters.calls++;
	const auto now = std::chrono::steady_clock::now();
	if (now - m_bda_epoch_log_time < std::chrono::seconds(10) || !Common::EnvRead::PerfLines()) {
		return;
	}
	m_bda_epoch_log_time = now;
	const auto& deferred = g_cpu_deferred_counters;
	std::fprintf(stderr,
	             "PERF bda_sync_epoch calls=%llu passes=%llu deferred_marks=%llu publishes=%llu "
	             "published_regions=%llu write_regions=%llu\n",
	             static_cast<unsigned long long>(counters.calls), static_cast<unsigned long long>(counters.passes),
	             static_cast<unsigned long long>(deferred.deferred_marks.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(deferred.publishes.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(deferred.published_regions.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(deferred.write_regions.load(std::memory_order_relaxed)));
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	// Live shader flags (KYTY_PERF_INDIRECT_LOAD_FAST=3): this frame boundary's live A/B phase.
	SpirvLive::Update(m_command_scheduler, *m_buffer_cache.GetFaultBuffer(),
	                  BufferCache::FAULT_DIAG_BASE + BufferCache::FAULT_DIAG_WORDS);
	m_texture_cache.RunGarbageCollector();
	m_texture_cache.RunTextureMove(m_bindless_table); // texture relocation (textureMove.cpp)
	m_texture_cache.ReportImageReadServe(); // KYTY_TRACE_READBACK
	m_buffer_cache.RunGarbageCollector();
	m_pipeline_cache.EvictIdlePipelines(m_command_scheduler);
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
