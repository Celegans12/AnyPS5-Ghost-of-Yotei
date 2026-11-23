#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/emulatorConfig.h"
#include "common/hackNotice.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/recompiler/frontend/decode/RecordReach.h"
#include "graphics/shader/shaderBindings.h"
#include "kernel/memory.h"
#include "libs/errno.h"
#include "loader/demonsSoulsPageFree.h"

#include <atomic>
#include <span>
#include <bit>
#include <algorithm>
#include <array>
#include <bit>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint64_t VsharpBaseReach = uint64_t {32} << 20u;

}

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_render_executor(*this),
      m_command_scheduler(*this, graphics,
                          (Config::GpuPipelineStages() & Config::GPU_PIPELINE_RECORDING) != 0),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache, this),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager), m_pipeline_cache(graphics, &m_image_heap, shader_store) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_ray_tracing = std::make_unique<RayTracing>(*this);
#if defined(ANYPS5_KYTY)
	m_pipeline_cache.SetRenderContext(this);
#endif
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
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
	constexpr uint64_t fault_size = 1;
	if (access == PageFaultAccess::Execute && LibKernel::Memory::IsGuestExecutable(fault_vaddr) &&
	    m_page_manager.AllowExecute(fault_vaddr)) {
		return true;
	}
	if (!IsMapped(fault_vaddr, fault_size)) {
		if (!GuestRange {fault_vaddr, fault_size}.Valid() || !m_page_manager.IsWatched(fault_vaddr)) {
			return false;
		}
	}
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	if (access == PageFaultAccess::Write && GuestGpu::IsGpuThread()) {
		m_bda_generation++;
	}
	if (access == PageFaultAccess::Write) {
		const auto owner   = m_buffer_cache.InvalidateFaultedPage(fault_vaddr);
		const bool texture = m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		const bool heap    = m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
		m_strict_watch.InvalidateMemory(fault_vaddr, fault_size);
		if (!GuestGpu::IsGpuThread() && !GuestGpu::IsCommandProcessorThread()) {
			m_page_manager.NoteWriteFault(fault_vaddr);
			BufferCache::CountWriteFault(owner, texture || heap);
		}
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size, false, true);
		m_texture_cache.FlushLazyReadback(fault_vaddr, fault_size);
	}
	return true;
}

void RenderContext::CollectHotPageWrites(uint64_t vaddr, uint64_t size) {
	m_page_manager.CollectHotPageWrites(
	    [](void* context, uint64_t begin, uint64_t end) {
		    auto& self = *static_cast<RenderContext*>(context);
		    self.m_buffer_cache.InvalidateMemory(begin, end - begin, true);
		    self.m_texture_cache.InvalidateMemory(begin, end - begin);
		    self.m_image_heap.InvalidateMemory(begin, end - begin);
		    self.m_strict_watch.InvalidateMemory(begin, end - begin);
	    },
	    this, vaddr, size);
}

static bool UrgentWriteBackWait() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_URGENT_WRITEBACK_WAIT");
		return value == nullptr || value[0] != '0';
	}();
	return on;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		if (UrgentWriteBackWait() && !GuestGpu::IsCommandProcessorThread()) {
			GetGpu().SendUrgentCommandSync(
			    [this, tick] {
				    if (tick == m_command_scheduler.CurrentTick()) {
					    m_command_scheduler.Flush();
				    }
			    },
			    false);
			m_command_scheduler.GetMasterSemaphore().Wait(tick);
		} else {
			GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
		}
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

void RenderContext::WriteFromCompletion(void* destination, const void* data,
                                        uint64_t size) noexcept {
	const auto vaddr = reinterpret_cast<uint64_t>(destination);
	if (GuestRange {vaddr, size}.Valid() && LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		const auto page_size = TRACKER_PAGE_SIZE;
		for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
			if (m_page_manager.IsWatched(page)) {
				m_buffer_cache.InvalidateUnlessGpuModified(vaddr, size);
				m_texture_cache.InvalidateMemory(vaddr, size);
				m_image_heap.InvalidateMemory(vaddr, size);
				break;
			}
		}
		return;
	}
	if (size == sizeof(uint32_t)) {
		uint32_t value = 0;
		std::memcpy(&value, data, sizeof(value));
		*static_cast<volatile uint32_t*>(destination) = value;
	} else if (size == sizeof(uint64_t)) {
		uint64_t value = 0;
		std::memcpy(&value, data, sizeof(value));
		*static_cast<volatile uint64_t*>(destination) = value;
	} else {
		std::memcpy(destination, data, size);
	}
	LibKernel::Memory::NoteEmulatorWrite(vaddr, size);
}

void RenderContext::WriteQwordsFromCompletion(void* destination, const uint64_t* values,
                                              uint32_t count, uint64_t stride) noexcept {
	const auto vaddr = reinterpret_cast<uint64_t>(destination);
	if (count == 0) {
		return;
	}
	const auto size = uint64_t {count - 1u} * stride + sizeof(uint64_t);
	if (GuestRange {vaddr, size}.Valid() &&
	    LibKernel::Memory::TryWriteBackingQwords(vaddr, values, count, stride)) {
		for (uint32_t i = 0; i < count; i++) {
			const auto address = vaddr + i * stride;
			if (m_page_manager.IsWatched(address & ~(TRACKER_PAGE_SIZE - 1))) {
				m_buffer_cache.InvalidateUnlessGpuModified(address, sizeof(uint64_t));
				m_texture_cache.InvalidateMemory(address, sizeof(uint64_t));
				m_image_heap.InvalidateMemory(address, sizeof(uint64_t));
			}
		}
		return;
	}
	for (uint32_t i = count; i-- > 0;) {
		WriteFromCompletion(static_cast<uint8_t*>(destination) + i * stride, &values[i],
		                    sizeof(uint64_t));
	}
}

void RenderContext::PrepareGpuWrite(const void* destination, uint64_t size) noexcept {
	const auto vaddr     = reinterpret_cast<uint64_t>(destination);
	const auto page_size = TRACKER_PAGE_SIZE;
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	for (auto page = vaddr & ~(page_size - 1); page < vaddr + size; page += page_size) {
		if (m_page_manager.IsWatched(page)) {
			(void)HandleFault(PageFaultAccess::Write, std::max(page, vaddr));
		}
	}
}

bool RenderContext::WriteAroundGpuWrites(void* destination, const void* data, uint64_t size) {
	const auto vaddr = reinterpret_cast<uint64_t>(destination);
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	if (m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		if (!m_buffer_cache.WriteAroundGpuWrites(vaddr, data, size)) {
			return false;
		}
		m_texture_cache.InvalidateMemory(vaddr, size);
		return true;
	}
	return m_buffer_cache.WriteAroundGpuWrites(vaddr, data, size);
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	m_strict_watch.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size) {
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemoryForWrite(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	m_strict_watch.InvalidateMemory(vaddr, size);
	return true;
}

void RenderContext::MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_mapped_ranges.ForEachInRange(vaddr, size, [&](uint64_t begin, uint64_t end) { out.Add(begin, end - begin); });
}

void RenderContext::FreedByGame(uint64_t vaddr, uint64_t size) {
	if (m_gpu == nullptr || m_gpu->IsStopping() || !GuestRange {vaddr, size}.Valid()) return;
	m_gpu->SendCommand([this, vaddr, size] {
		if (!IsMapped(vaddr, size)) return;
		const auto start = std::chrono::steady_clock::now();
		static BufferCache::FreeBlockSplits splits;
		static uint64_t                     blocks = 0, bytes = 0, let_go = 0, ns = 0, max_ns = 0;
		static auto                         next = start + std::chrono::seconds(30);
		const auto                          lists = splits.lists;
		const auto freed = m_buffer_cache.DropFreedRange(vaddr, size, &Loader::DemonsSoulsPageFree::FreeBlocksIn, &splits);
		const auto took  = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
		ns += took;
		max_ns = std::max(max_ns, took);
		if (freed != 0) {
			blocks++;
			bytes += size;
			let_go += freed;
		}
		if ((freed == 0 && splits.lists == lists) || start < next) return;
		std::printf("Engine frees: %llu freed blocks (%.0f MiB) had buffers over them: %.0f MiB of buffers let go (%.0f MiB cut off "
		            "buffers over other free blocks too, the allocator's lists read %llu times), %.1f ms on the GPU thread (at most "
		            "%.2f ms a block)\n",
		            static_cast<unsigned long long>(blocks), static_cast<double>(bytes) / 1048576.0,
		            static_cast<double>(let_go) / 1048576.0, static_cast<double>(splits.cut) / 1048576.0,
		            static_cast<unsigned long long>(splits.lists), static_cast<double>(ns) / 1e6,
		            static_cast<double>(max_ns) / 1e6);
		std::fflush(stdout);
		blocks = bytes = let_go = ns = max_ns = 0;
		splits = {};
		next   = start + std::chrono::seconds(30);
	});
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

uint64_t RenderContext::MappedRangesVersion() const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges_version;
}

bool RenderContext::UnmappedSince(uint64_t version, uint64_t vaddr, uint64_t size) const {
	std::shared_lock lock(m_mapped_ranges_mutex);
	if (version == m_mapped_ranges_version) {
		return false;
	}
	if (version < m_unmap_log_floor) {
		return true;
	}
	for (auto it = m_unmap_log.rbegin(); it != m_unmap_log.rend() && it->version > version; ++it) {
		if (vaddr < it->vaddr + it->size && it->vaddr < vaddr + size) {
			return true;
		}
	}
	return false;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	{
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Add(vaddr, size);
		m_mapped_ranges_version++;
		m_mapping_epoch.store(m_mapped_ranges_version, std::memory_order_release);
		m_bda_mapped_since_sync.emplace_back(vaddr, size);
	}
	if (m_texture_cache.HasPartialCandidates()) {
		const auto map = [this, vaddr, size] { m_texture_cache.MapMemory(vaddr, size); };
		if (m_gpu == nullptr || m_gpu->IsStopping()) {
			map();
		} else {
			m_gpu->SendUrgentCommandSync(map);
		}
	}
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		m_page_manager.ForgetHotPages(vaddr, size);
		if (m_command_scheduler.Active()) {
			m_command_scheduler.PopPendingOperations();
		}
		const bool drain = m_command_scheduler.Active() &&
		                   (m_buffer_cache.IsRegionGpuModified(vaddr, size) ||
		                    m_texture_cache.HasGpuWrittenImages(vaddr, size) ||
		                    m_command_scheduler.HasPendingGuestOperations(vaddr, size));
		if (drain) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		const bool registered = size != 0 && m_buffer_cache.IsRegionRegistered(vaddr, size);
		m_buffer_cache.ForgetCommandBuffers(vaddr, size);
		m_buffer_cache.InvalidateMemory(vaddr, size);
		if (registered) {
			m_buffer_cache.DeleteUnmappedBuffers(vaddr, size);
			m_buffer_cache.SplitUnmappedBuffers(vaddr, size);
		}
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		m_strict_watch.UnmapMemory(vaddr, size);
		m_occlusion_queries.OnUnmap(vaddr, size);
		if (size != 0 && vaddr <= UINT64_MAX - size) {
			m_guest_bvh_memory.erase(m_guest_bvh_memory.lower_bound(vaddr), m_guest_bvh_memory.lower_bound(vaddr + size));
			std::erase_if(m_hle_tlases, [&](const HleTlas& tlas) { return tlas.header >= vaddr && tlas.header - vaddr < size; });
		}
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		m_mapped_ranges_version++;
		m_mapping_epoch.store(m_mapped_ranges_version, std::memory_order_release);
		if (m_unmap_log.size() >= 4096) {
			m_unmap_log_floor = m_unmap_log.front().version;
			m_unmap_log.pop_front();
		}
		m_unmap_log.push_back({m_mapped_ranges_version, vaddr, size});
	};
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendUrgentCommandSync(unmap);
}

void RenderContext::CacheDmaBases(const ShaderStageRuntime& runtime) {
	const auto&      program   = runtime.module->info;
	const auto&      user_data = runtime.resources->user_data;
	std::shared_lock lock(m_mapped_ranges_mutex);

	const auto cache_after = [&](const std::array<uint32_t, 2>& pair, uint64_t window,
	                             bool contiguous) {
		const auto [low_reg, high_reg] = pair;
		const auto low  = static_cast<uint64_t>(low_reg) - program.user_data_base;
		const auto high = static_cast<uint64_t>(high_reg) - program.user_data_base;
		if (low_reg < program.user_data_base || high_reg < program.user_data_base ||
		    low >= user_data.size() || high >= user_data.size()) {
			return;
		}
		const auto base =
		    user_data[low] | (static_cast<uint64_t>(user_data[high] & 0xffffu) << 32u);
		if (base > UINT64_MAX - window) {
			return;
		}
		m_mapped_ranges.ForEachInRange(base, window, [&](uint64_t start, uint64_t end) {
			if (!contiguous || start == base) {
				(void)m_buffer_cache.FindBuffer(start, end - start);
			}
		});
	};
	for (const auto& pair: program.info.dma_base_registers) {
		cache_after(pair, BufferCache::CACHING_PAGESIZE, false);
	}
	for (const auto& pair: program.info.vsharp_base_registers) {
		cache_after(pair, VsharpBaseReach, true);
	}
	CacheTlasMemory(runtime);
}

bool RenderContext::CacheSelfSizedBuffers(const ShaderStageRuntime& runtime) {
	constexpr uint64_t CachedBytes = 4 * BufferCache::CACHING_PAGESIZE;
	const auto&        program     = runtime.module->info;
	const auto&        user_data   = runtime.resources->user_data;
	const auto         value       = [&](uint32_t reg, uint32_t& result) {
		const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
		if (reg < program.user_data_base || index >= user_data.size()) {
			return false;
		}
		result = user_data[index];
		return true;
	};
	bool             written = false;
	std::shared_lock lock(m_mapped_ranges_mutex);
	for (const auto& buffer: program.info.self_sized_buffers) {
		uint32_t low  = 0;
		uint32_t high = 0;
		if (!value(buffer.base_low_register, low) || !value(buffer.base_high_register, high)) {
			continue;
		}
		written |= buffer.written;
		const auto base = low | (uint64_t {high & 0xffffu} << 32u);
		if (base == 0) {
			continue;
		}
		m_mapped_ranges.ForEachInRange(base, CachedBytes, [this](uint64_t start, uint64_t end) {
			if (const auto id = m_buffer_cache.FindBuffer(start, end - start)) {
				m_buffer_cache.GetBuffer(id).device_address_pinned = true;
			}
		});
	}
	return written;
}

void RenderContext::CacheTlasMemory(const ShaderStageRuntime& runtime) {
	const auto&                  program = runtime.module->info;
	RayTracing::TlasSource source;
	if (!RayTracing::FindGuestTlasSource(program.info, source)) {
		return;
	}
	const auto& resources = *runtime.resources;
	const auto readable = [this](uint64_t address, uint64_t size) {
		return GuestRange {address, size}.Valid() && m_mapped_ranges.Contains(address, size) &&
		       !(m_buffer_cache.MayBeGpuModified(address, size) &&
		         m_buffer_cache.IsRegionGpuModified(address, size));
	};
	uint64_t    header = 0;
	const auto& srt    = resources.flattened_srt;
	const auto  in_srt = [&](uint32_t slot) {
		return slot != ShaderRecompiler::IR::ShaderInfo::NoRayQuerySlot && slot < srt.size();
	};
	if (in_srt(source.header_slots[0]) && in_srt(source.header_slots[1])) {
		header = srt[source.header_slots[0]] | (uint64_t {srt[source.header_slots[1]]} << 32u);
	} else {
		const auto& user_data = resources.user_data;
		const auto  index     = uint64_t {source.user_sgpr} - program.user_data_base;
		if (source.user_sgpr < program.user_data_base || index + 1u >= user_data.size()) {
			return;
		}
		const auto pointer = RayTracing::UserPointer(
		    source, user_data[index] | (uint64_t {user_data[index + 1u]} << 32u), readable);
		if (RayTracing::HeaderInUserData(source)) {
			header = RayTracing::UserDataHeader(pointer);
		} else if (pointer == 0 || pointer > UINT64_MAX - source.header_offset ||
		           !readable(pointer + source.header_offset, sizeof(header))) {
			return;
		} else {
			std::memcpy(&header, reinterpret_cast<const void*>(pointer + source.header_offset), sizeof(header));
		}
	}
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	const auto tlas = m_ray_tracing->ReadGuestTlas(header, source, readable, ranges);
	if (tlas.end != 0) {
		ranges.emplace_back(header, tlas.end);
	}
	for (const auto& [first, last]: ranges) {
		m_mapped_ranges.ForEachInRange(first, last - first, [this](uint64_t start, uint64_t stop) {
			if (const auto id = m_buffer_cache.FindBuffer(start, stop - start)) {
				m_buffer_cache.GetBuffer(id).device_address_pinned = true;
			}
		});
	}
	if (program.info.ray_query && m_ray_tracing->Enabled()) {
		m_ray_tracing->NoteTlasInstances(tlas.instances);
	}
}

bool RenderContext::SynchronizeDmaFootprint(std::span<const ShaderStageRuntime* const> stages) {
	constexpr int JoinedRuns = 6;
	for (const auto* stage: stages) {
		const auto& program = stage->module->info;
		if (!program.info.uses_dma || program.info.dma_bounded || program.info.ray_query_reach.empty()) {
			continue;
		}
		auto& traced  = m_traced_programs[program.shader_hash];
		traced.recent = static_cast<uint8_t>((traced.recent << 1u) |
		                                     (traced.unbounded_ops != m_bda_unbounded_ops ? 1u : 0u));
		traced.unbounded_ops = m_bda_unbounded_ops;
		if (std::popcount(traced.recent) >= JoinedRuns) {
			return false;
		}
	}
	if (!SynchronizeDmaReach(stages)) {
		m_bda_unbounded_ops++;
		return false;
	}
	return true;
}

bool RenderContext::SynchronizeDmaReach(std::span<const ShaderStageRuntime* const> stages) {
	constexpr uint64_t MaxBufferBytes = uint64_t {256} * 1024 * 1024;
	constexpr uint64_t WaveRecords = 64;
	m_dma_spans.clear();
	m_dma_sets.clear();
	std::shared_lock lock(m_mapped_ranges_mutex);
	const auto add = [&](uint64_t address, uint64_t size) {
		if (size != 0 && address <= UINT64_MAX - size) {
			m_mapped_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
				m_dma_spans.emplace_back(start, end);
			});
		}
	};
	for (const auto* stage: stages) {
		const auto& program = stage->module->info;
		if (!program.info.uses_dma) {
			continue;
		}
		if (!program.info.dma_bounded && program.info.ray_query_reach.empty()) {
			if (!AddRecordReach(*stage)) {
				return false;
			}
			continue;
		}
		if (!program.info.dma_bounded) {
			if (!AddRayTracingReach(*stage)) {
				return false;
			}
		}
		const auto& user_data = stage->resources->user_data;
		const auto  base_of   = [&](uint32_t reg, uint64_t& base) {
			const auto index = static_cast<uint64_t>(reg) - program.user_data_base;
			if (reg < program.user_data_base || index + 1u >= user_data.size()) {
				return false;
			}
			base = user_data[index] | (static_cast<uint64_t>(user_data[index + 1u]) << 32u);
			return true;
		};
		for (const auto& window: program.info.dma_windows) {
			uint64_t base = 0;
			if (!base_of(window.base_register, base)) {
				return false;
			}
			const auto before = static_cast<uint64_t>(-window.first) + 4u;
			const auto start  = base > before ? base - before : 0;
			add(start, std::min(base, UINT64_MAX - window.last) + window.last - start);
		}
		for (const auto& table: program.info.dma_tables) {
			uint64_t base = 0;
			if (!base_of(table.base_register, base)) {
				return false;
			}
			const auto entries = (base & ~uint64_t {3}) + static_cast<uint64_t>(table.immediate);
			Common::HackNotice::Report("DMA upload assumes V# indices stay within 64 records",
			                           "RenderContext::SynchronizeDmaFootprint",
			                           program.shader_hash, table.base_register);
			for (uint32_t offset = table.offset_bits;; offset = (offset - 1u) & table.offset_bits) {
				const auto address = entries + offset;
				if (!m_mapped_ranges.Contains(address, 12) ||
				    m_buffer_cache.IsRegionGpuModified(address, 12)) {
					return false;
				}
				ShaderBufferResource descriptor;
				std::memcpy(descriptor.fields, reinterpret_cast<const void*>(address), 12);
				const auto stride = uint64_t {descriptor.Stride()};
				if (descriptor.Base48() != 0 && descriptor.NumRecords() != 0) {
					const auto wanted =
					    descriptor.GetSize() + std::max<uint64_t>(stride, 4u) * WaveRecords;
					if (wanted > MaxBufferBytes) {
						Common::HackNotice::Report("DMA upload of a V# range clipped to 256 MiB",
						                           "RenderContext::SynchronizeDmaFootprint",
						                           program.shader_hash, table.base_register);
					}
					add(descriptor.Base48(), std::min(wanted, MaxBufferBytes));
				}
				if (offset == 0) {
					break;
				}
			}
		}
		for (const auto& buffer: program.info.self_sized_buffers) {
			uint64_t low  = 0;
			uint64_t high = 0;
			if (!base_of(buffer.base_low_register, low) || !base_of(buffer.base_high_register, high)) {
				return false;
			}
			const auto base = (low & 0xffffffffu) | ((high & 0xffffu) << 32u);
			if (base == 0) {
				continue;
			}
			auto memory = m_guest_bvh_memory.upper_bound(base);
			if (memory == m_guest_bvh_memory.begin() || base - (--memory)->first >= memory->second) {
				return false;
			}
			add(base, memory->first + memory->second - base);
		}
		for (const auto& table: program.info.dma_record_tables) {
			const auto& buffers = stage->resources->buffers;
			if (table.buffer >= buffers.size()) {
				return false;
			}
			const auto records = DecodeNativeDescriptor<ShaderBufferResource>(buffers[table.buffer]);
			const auto first   = records.Base48();
			const auto bytes   = records.GetSize();
			if (first == 0 || bytes == 0) {
				continue;
			}
			if (table.stride == sizeof(records.fields) && table.offset == 0) {
				if (!AddBufferTable(records)) {
					return false;
				}
				continue;
			}
			if (bytes > MaxBufferBytes || !m_mapped_ranges.Contains(first, bytes) ||
			    m_buffer_cache.IsRegionGpuModified(first, bytes)) {
				return false;
			}
			for (uint64_t offset = table.offset; offset + 12u <= bytes; offset += table.stride) {
				ShaderBufferResource descriptor {};
				std::memcpy(descriptor.fields, reinterpret_cast<const void*>(first + offset), 12);
				if (descriptor.Base48() == 0 || descriptor.NumRecords() == 0) {
					continue;
				}
				if (descriptor.GetSize() > MaxBufferBytes) {
					return false;
				}
				add(descriptor.Base48(), descriptor.GetSize());
			}
		}
	}
	const auto reach_layout = ReachLayout();
	lock.unlock();
	m_buffer_cache.SynchronizeBuffersInSpans(m_dma_spans);
	for (auto* table: m_dma_sets) {
		if (table->synced_in == reach_layout) {
			continue;
		}
		table->synced_in = reach_layout;
		m_buffer_cache.SynchronizeBuffersInSet(table->ranges, table->regions);
	}
	return true;
}

bool RenderContext::AddRecordReach(const ShaderStageRuntime& runtime) {
	using Reach = ShaderRecompiler::Decoder::RecordReach;
	constexpr uint64_t MaxRecords = 4096;
	constexpr uint32_t MaxIndex   = 1u << 20u;
	constexpr uint32_t MaxOffset  = 1u << 28u;
	const auto&        program    = runtime.module->info;
	Reach              reach;
	if (program.info.dma_record_reach.empty() || !Reach::Unpack(program.info.dma_record_reach, reach)) {
		return false;
	}
	const auto readable = [this](uint64_t address, uint64_t size) {
		return GuestRange {address, size}.Valid() && m_mapped_ranges.Contains(address, size) &&
		       !(m_buffer_cache.MayBeGpuModified(address, size) &&
		         m_buffer_cache.IsRegionGpuModified(address, size));
	};
	const auto& user_data  = runtime.resources->user_data;
	const auto  descriptor = [&](const Reach::Descriptor& source, ShaderBufferResource& result) {
		const auto index = uint64_t {source.user_sgpr} - program.user_data_base;
		if (source.user_sgpr < program.user_data_base || index + 1u >= user_data.size()) {
			return false;
		}
		const auto pointer = user_data[index] | (uint64_t {user_data[index + 1u]} << 32u);
		if (pointer > UINT64_MAX - source.offset || !readable(pointer + source.offset, sizeof(result.fields))) {
			return false;
		}
		std::memcpy(result.fields, reinterpret_cast<const void*>(pointer + source.offset), sizeof(result.fields));
		return true;
	};
	const auto load = [&](const ShaderBufferResource& buffer, uint64_t offset, void* data, uint64_t size) {
		std::memset(data, 0, size);
		const auto bytes = buffer.GetSize();
		if (buffer.Base48() == 0 || offset > bytes || size > bytes - offset) {
			return true;
		}
		if (!readable(buffer.Base48() + offset, size)) {
			return false;
		}
		std::memcpy(data, reinterpret_cast<const void*>(buffer.Base48() + offset), size);
		return true;
	};
	const auto first_span = m_dma_spans.size();
	const auto add        = [&](uint64_t address, uint64_t size) {
		if (size != 0 && address <= UINT64_MAX - size) {
			m_mapped_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
				const std::pair span {start, end};
				if (std::find(m_dma_spans.begin() + static_cast<ptrdiff_t>(first_span),
				              m_dma_spans.end(), span) == m_dma_spans.end()) {
					m_dma_spans.push_back(span);
				}
			});
		}
	};
	ShaderBufferResource    header;
	ShaderBufferResource    records;
	ShaderBufferResource    pointers;
	std::array<uint32_t, 2> range {};
	if (!descriptor(reach.header, header) || !descriptor(reach.records, records) ||
	    !descriptor(reach.pointers, pointers) || records.Stride() != reach.record_size ||
	    pointers.Stride() != sizeof(pointers.fields) || reach.record_size > 0x1000u ||
	    !load(header, reach.header_offset, range.data(), sizeof(range)) || range[1] > MaxRecords ||
	    range[0] > MaxIndex) {
		return false;
	}
	std::array<uint8_t, 0x1000> record {};
	const auto                  field = [&](uint32_t offset) {
        uint32_t value = 0;
        std::memcpy(&value, record.data() + offset, sizeof(value));
        return value;
	};
	for (uint64_t index = range[0]; index < uint64_t {range[0]} + range[1]; index++) {
		if (!load(records, index * reach.record_size, record.data(), reach.record_size)) {
			return false;
		}
		const auto threads = field(reach.threads_field);
		const auto first   = field(reach.first_field);
		if (static_cast<uint32_t>(first + threads) <= first) {
			continue;
		}
		const auto stride = (field(reach.stride_field) >> (reach.stride_byte * 8u)) & 0xffu;
		const auto last   = uint64_t {threads - 1u} * stride;
		const auto entry  = field(reach.pointer_field);
		std::array<uint32_t, 2> pointer {};
		if (last > MaxOffset || entry > MaxIndex ||
		    !load(pointers, uint64_t {entry} * sizeof(pointers.fields), pointer.data(), sizeof(pointer))) {
			return false;
		}
		const auto high = static_cast<int16_t>(pointer[1]);
		uint64_t   low_offset  = UINT64_MAX;
		uint64_t   high_offset = 0;
		for (const auto& attribute: reach.attributes) {
			uint32_t value = 0;
			if (attribute.field != Reach::None) {
				value = field(attribute.field);
				if (attribute.byte < 4u) {
					value = (value >> (attribute.byte * 8u)) & 0xffu;
				}
			}
			if (value > MaxOffset) {
				return false;
			}
			low_offset  = std::min(low_offset, uint64_t {value} & ~uint64_t {3});
			high_offset = std::max(high_offset, ((value + last) & ~uint64_t {3}) + attribute.end);
		}
		if (high >= 0 && low_offset < high_offset) {
			add((uint64_t {static_cast<uint16_t>(high)} << 32u | pointer[0]) + low_offset,
			    high_offset - low_offset);
		}
		for (const auto store: reach.store_fields) {
			const auto           target_entry = field(store);
			ShaderBufferResource target;
			if (target_entry > MaxIndex ||
			    !load(pointers, uint64_t {target_entry} * sizeof(target.fields), target.fields, sizeof(target.fields))) {
				return false;
			}
			const auto written = IndirectWriteTables::TargetRange(target);
			add(written.address, written.size);
		}
	}
	return true;
}

void RenderContext::NoteHleTlas(uint64_t header, uint64_t size, std::vector<uint64_t> blases) {
	constexpr size_t MaxTlases = 16;
	for (auto& tlas: m_hle_tlases) {
		if (tlas.header == header) {
			tlas.size   = size;
			tlas.blases = std::move(blases);
			return;
		}
	}
	if (m_hle_tlases.size() >= MaxTlases) {
		m_hle_tlases.erase(m_hle_tlases.begin());
	}
	m_hle_tlases.push_back({header, size, std::move(blases)});
}

void RenderContext::ForgetHleTlas(uint64_t header) {
	std::erase_if(m_hle_tlases, [&](const HleTlas& tlas) { return header == 0 || tlas.header == header; });
}

void RenderContext::NoteGuestBvhMemory(uint64_t address, uint64_t size) {
	constexpr size_t MaxCaptures = 65536;
	if (m_guest_bvh_memory.size() >= MaxCaptures) {
		m_guest_bvh_memory.clear();
	}
	m_guest_bvh_memory[address] = size;
}

bool RenderContext::GuestTlasHeader(const ShaderStageRuntime& runtime, const RayTracing::TlasSource& source,
                                    uint64_t& header, uint64_t& header_at) {
	const auto& program   = runtime.module->info;
	const auto& resources = *runtime.resources;
	const auto readable = [this](uint64_t address, uint64_t size) {
		return GuestRange {address, size}.Valid() && m_mapped_ranges.Contains(address, size) &&
		       !(m_buffer_cache.MayBeGpuModified(address, size) &&
		         m_buffer_cache.IsRegionGpuModified(address, size));
	};
	header_at             = 0;
	const auto& user_data = resources.user_data;
	const auto  index     = uint64_t {source.user_sgpr} - program.user_data_base;
	uint64_t    pointer   = 0;
	if (source.user_sgpr >= program.user_data_base && index + 1u < user_data.size()) {
		pointer = RayTracing::UserPointer(
		    source, user_data[index] | (uint64_t {user_data[index + 1u]} << 32u), readable);
	}
	if (RayTracing::HeaderInUserData(source)) {
		header = RayTracing::UserDataHeader(pointer);
		return header != 0;
	}
	if (pointer != 0 && pointer <= UINT64_MAX - source.header_offset) {
		header_at = pointer + source.header_offset;
	}
	const auto& srt    = resources.flattened_srt;
	const auto  in_srt = [&](uint32_t slot) {
		return slot != ShaderRecompiler::IR::ShaderInfo::NoRayQuerySlot && slot < srt.size();
	};
	if (in_srt(source.header_slots[0]) && in_srt(source.header_slots[1])) {
		header = srt[source.header_slots[0]] | (uint64_t {srt[source.header_slots[1]]} << 32u);
		return true;
	}
	if (header_at == 0 || !readable(header_at, sizeof(header))) {
		return false;
	}
	std::memcpy(&header, reinterpret_cast<const void*>(header_at), sizeof(header));
	return true;
}

bool RenderContext::GuestBlasMemory(uint64_t blas, uint64_t& size) {
	constexpr uint64_t HeaderSize = 256;
	constexpr uint64_t SizeField  = 16;
	constexpr uint64_t MaxSize    = uint64_t {256} * 1024 * 1024;
	size                          = 0;
	if (!m_mapped_ranges.Contains(blas, HeaderSize)) {
		return true;
	}
	std::array<uint32_t, 2> magic {};
	uint32_t                field = 0;
	if (!(m_buffer_cache.MayBeGpuModified(blas, HeaderSize) && m_buffer_cache.IsRegionGpuModified(blas, HeaderSize))) {
		std::memcpy(magic.data(), reinterpret_cast<const void*>(blas), sizeof(magic));
		std::memcpy(&field, reinterpret_cast<const void*>(blas + SizeField), sizeof(field));
		if (magic[0] != 0x5f525350u || magic[1] != 0x4c485642u) {
			return false;
		}
	} else if (const auto captured = m_guest_bvh_memory.find(blas); captured != m_guest_bvh_memory.end()) {
		field = static_cast<uint32_t>(std::min<uint64_t>(captured->second, UINT32_MAX));
	} else if (!Libs::LibKernel::Memory::TryReadAroundGpuWrites(blas + SizeField, &field, sizeof(field))) {
		return false;
	}
	size = field;
	return field >= HeaderSize && field <= MaxSize;
}

bool RenderContext::AddBufferTable(const ShaderBufferResource& table) {
	constexpr uint64_t MaxTableBytes  = uint64_t {16} * 1024 * 1024;
	constexpr size_t   MaxTables      = 4;
	constexpr uint64_t EntryBytes     = sizeof(table.fields);
	const auto         address        = table.Base48();
	const auto         size           = table.GetSize() / EntryBytes * EntryBytes;
	if (address == 0 || size == 0) {
		return true;
	}
	if (table.Stride() != EntryBytes || size > MaxTableBytes || !m_mapped_ranges.Contains(address, size) ||
	    (m_buffer_cache.MayBeGpuModified(address, size) && m_buffer_cache.IsRegionGpuModified(address, size))) {
		return false;
	}
	auto cached = std::ranges::find_if(m_buffer_tables, [&](const auto& known) { return known->address == address; });
	if (cached == m_buffer_tables.end()) {
		if (m_buffer_tables.size() >= MaxTables) {
			m_buffer_tables.erase(m_buffer_tables.begin());
		}
		m_buffer_tables.push_back(std::make_unique<BufferTable>());
		cached            = std::prev(m_buffer_tables.end());
		(*cached)->address = address;
	}
	auto&       entries = **cached;
	const auto* memory  = reinterpret_cast<const uint8_t*>(address);
	const auto layout = ReachLayout();
	bool       same   = entries.copy.size() == size;
	m_table_runs.clear();
	if (entries.compared_in != layout || !same) {
		entries.compared_in = layout;
		const auto compare  = [&](uint64_t run, uint64_t bytes) {
			const auto begin = std::max(run, address);
			const auto end   = std::min(run + bytes, address + size);
			if (entries.copy.size() == size && begin < end &&
			    std::memcmp(entries.copy.data() + (begin - address), memory + (begin - address), end - begin) != 0) {
				same = false;
				m_table_runs.emplace_back(begin - address, end - begin);
			}
		};
		m_buffer_cache.ConsumeCpuWrites(CpuWriteConsumer::BufferTables, address, size, [&](uint64_t run, uint64_t bytes) {
			compare(run, bytes);
			const auto first = run & ~(TRACKER_PAGE_SIZE - 1);
			const auto last  = (run + bytes + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
			for (const auto& other: m_buffer_tables) {
				if (other.get() != &entries && other->address < last &&
				    first < other->address + other->copy.size()) {
					other->written.emplace_back(first, last - first);
				}
			}
		});
		for (const auto& [run, bytes]: entries.written) {
			compare(run, bytes);
		}
		entries.written.clear();
	}
	if (!same) {
		const auto range_of = [](const uint8_t* entry) {
			ShaderBufferResource buffer;
			std::memcpy(buffer.fields, entry, EntryBytes);
			auto       bytes = buffer.GetSize();
			const auto base  = buffer.Base48();
			if (base < LOWER_ADDRESS_SIZE) {
				bytes = std::min(bytes, LOWER_ADDRESS_SIZE - base);
			} else if (base >= Libs::LibKernel::Memory::kExtendedMemoryBase &&
			           base - Libs::LibKernel::Memory::kExtendedMemoryBase < Libs::LibKernel::Memory::kExtendedMemorySize) {
				bytes = std::min(bytes, Libs::LibKernel::Memory::kExtendedMemorySize -
				                            (base - Libs::LibKernel::Memory::kExtendedMemoryBase));
			} else {
				bytes = 0;
			}
			return std::pair {base, base != 0 ? bytes : 0};
		};
		const auto count = static_cast<uint32_t>(size / EntryBytes);
		if (entries.copy.size() == size && entries.replaced <= count / 8) {
			for (const auto& [offset, bytes]: m_table_runs) {
				for (auto at = offset / EntryBytes * EntryBytes; at < offset + bytes; at += EntryBytes) {
					if (std::memcmp(entries.copy.data() + at, memory + at, EntryBytes) == 0) {
						continue;
					}
					std::memcpy(entries.copy.data() + at, memory + at, EntryBytes);
					entries.replaced++;
					if (const auto [base, length] = range_of(entries.copy.data() + at); length != 0) {
						m_buffer_cache.AddToSet(entries.ranges, entries.regions, base, length);
					}
				}
			}
		} else {
			entries.copy.assign(memory, memory + size);
			entries.ranges.Clear();
			entries.regions  = {};
			entries.replaced = 0;
			for (uint64_t offset = 0; offset < size; offset += EntryBytes) {
				if (const auto [base, length] = range_of(entries.copy.data() + offset); length != 0) {
					entries.ranges.Add(base, length);
				}
			}
		}
		entries.synced_in = {};
	}
	m_mapped_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) { m_dma_spans.emplace_back(start, end); });
	if (std::ranges::find(m_dma_sets, &entries) == m_dma_sets.end()) {
		m_dma_sets.push_back(&entries);
	}
	return true;
}

bool RenderContext::RayQueryTableVsharp(const ShaderStageRuntime& runtime, uint32_t table, ShaderBufferResource& vsharp) {
	const auto& buffers = runtime.resources->buffers;
	if ((table & ShaderRecompiler::IR::ShaderInfo::RayQuerySrtVsharp) == 0) {
		if (table >= buffers.size()) {
			return false;
		}
		vsharp = DecodeNativeDescriptor<ShaderBufferResource>(buffers[table]);
		return true;
	}
	const auto& srt   = runtime.resources->flattened_srt;
	const auto  first = table & ~ShaderRecompiler::IR::ShaderInfo::RayQuerySrtVsharp;
	if (uint64_t {first} + 4u > srt.size()) {
		return false;
	}
	for (uint32_t dword = 0; dword < 4u; dword++) {
		vsharp.fields[dword] = srt[first + dword];
	}
	return true;
}

bool RenderContext::AddRayTracingReach(const ShaderStageRuntime& runtime) {
	const auto&                  program = runtime.module->info;
	const auto&                  reach   = program.info.ray_query_reach;
	RayTracing::TlasSource source;
	uint64_t                     header    = 0;
	uint64_t                     header_at = 0;
	if (reach.empty() || reach.size() != 1u + uint64_t {reach[0]} * 2u || !m_ray_tracing->Enabled() ||
	    !RayTracing::FindGuestTlasSource(program.info, source) ||
	    !GuestTlasHeader(runtime, source, header, header_at)) {
		return false;
	}
	const auto tlas = std::ranges::find_if(m_hle_tlases, [&](const HleTlas& known) { return known.header == header; });
	if (tlas == m_hle_tlases.end()) {
		return false;
	}
	const auto add = [&](uint64_t address, uint64_t size) {
		if (size != 0 && address <= UINT64_MAX - size) {
			m_mapped_ranges.ForEachInRange(address, size,
			                               [&](uint64_t start, uint64_t end) { m_dma_spans.emplace_back(start, end); });
		}
	};
	if (header_at != 0) {
		add(header_at, sizeof(header));
	}
	add(header, tlas->size);
	m_walk_blases.clear();
	m_ray_tracing->PendingWalks(m_walk_blases);
	for (const auto* list: {&tlas->blases, &m_walk_blases}) {
		for (const auto blas: *list) {
			uint64_t size = 0;
			if (!GuestBlasMemory(blas, size)) {
				return false;
			}
			add(blas, size);
		}
	}
	const auto& buffers = runtime.resources->buffers;
	for (uint32_t test = 0; test < reach[0]; test++) {
		const auto material = reach[1u + test * 2u];
		const auto table    = reach[2u + test * 2u];
		ShaderBufferResource table_vsharp;
		if (material >= buffers.size() || !RayQueryTableVsharp(runtime, table, table_vsharp)) {
			return false;
		}
		const auto records = DecodeNativeDescriptor<ShaderBufferResource>(buffers[material]);
		add(records.Base48(), records.GetSize());
		if (!AddBufferTable(table_vsharp)) {
			return false;
		}
	}
	m_ray_tracing->AnnounceWalks(m_walk_blases);
	return true;
}

void RenderContext::PrepareBda(bool may_write, bool full_sync) {
	[[maybe_unused]] static const bool logged = [] {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		return true;
	}();
	m_buffer_cache.PrepareFaultBuffer();
	if (may_write) {
		m_buffer_cache.NoteDeviceAddressUse();
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	const std::array<uint64_t, 3> epochs {RegionManager::CpuDirtyEpoch(),
	                                      m_buffer_cache.RegisterEpoch(), m_mapped_ranges_version};
	const bool same_layout = epochs[1] == m_bda_sync_epochs[1] && epochs[2] == m_bda_sync_epochs[2];
	if (full_sync && (epochs != m_bda_sync_epochs || m_buffer_cache.HasKeptPages()) &&
	    (!same_layout || m_bda_synced_generation != m_bda_generation)) {
		m_bda_synced_generation = m_bda_generation;
		for (const auto& [vaddr, size]: m_bda_mapped_since_sync) {
			m_buffer_cache.ForceSynchronization(vaddr, size);
		}
		m_bda_mapped_since_sync.clear();
		m_buffer_cache.SynchronizeBuffersInRanges(m_mapped_ranges, false);
		m_bda_sync_epochs = epochs;
	}
	m_fault_process_pending = true;
	m_use_scan_pending      = true;
}

void RenderContext::RunGarbageCollector() {
	m_frame_number++;
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	if (m_use_scan_countdown != 0) {
		m_use_scan_countdown--;
	} else if (m_use_scan_pending) {
		m_use_scan_pending   = false;
		m_use_scan_countdown = BufferCache::USE_SCAN_PERIOD;
		m_buffer_cache.ProcessUseBuffer();
	}
	if (m_indirect_write_tables.TakePendingWrites()) {
		m_buffer_cache.ProcessWriteBuffer([this](std::span<const uint64_t> pages) {
			m_indirect_write_tables.NoteWrites(pages);
		});
		m_command_scheduler.DeferOperation([this] { m_buffer_cache.EndIndirectWrites(); });
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	m_ray_tracing->RunGarbageCollector();
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

}
