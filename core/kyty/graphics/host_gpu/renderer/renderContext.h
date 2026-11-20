#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/indirectWriteTables.h"
#include "graphics/host_gpu/renderer/lodStats.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/strictWatch.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "kernel/eventQueue.h"

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <shared_mutex>
#include <span>
#include <deque>
#include <utility>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

struct ShaderBufferResource;

class GuestGpu;

class RenderContext {
public:
	explicit RenderContext(GraphicContext&              graphics,
	                       const std::filesystem::path& shader_store = ShaderStorePath());
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	ImageHeap&          GetImageHeap() { return m_image_heap; }
	StrictWatch&        GetStrictWatch() { return m_strict_watch; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	IndirectWriteTables& GetIndirectWriteTables() { return m_indirect_write_tables; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }
	OcclusionQueries&   GetOcclusionQueries() { return m_occlusion_queries; }
	RayTracing*         GetRayTracing() { return m_ray_tracing.get(); }
	LodStatsCounter&    GetLodStats() { return m_lod_stats; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	void PrepareGpuWrite(const void* destination, uint64_t size) noexcept;
	void WriteFromCompletion(void* destination, const void* data, uint64_t size) noexcept;
	void WriteQwordsFromCompletion(void* destination, const uint64_t* values, uint32_t count,
	                               uint64_t stride) noexcept;
	void NoteOperation(uint64_t submit_id, bool after_wait) noexcept {
		if (submit_id != m_bda_submit || after_wait) {
			m_bda_submit = submit_id;
			m_bda_generation++;
			BdaDirtyRegions::Publish();
		}
	}
	[[nodiscard]] bool WriteAroundGpuWrites(void* destination, const void* data, uint64_t size);
	void               CollectHotPageWrites(uint64_t vaddr = 0, uint64_t size = 0);
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	void               MappedRanges(uint64_t vaddr, uint64_t size, RangeSet& out) const;
	void               FreedByGame(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] uint64_t MappedRangesVersion() const noexcept;
	[[nodiscard]] uint64_t MappingEpoch() const noexcept { return m_mapping_epoch.load(std::memory_order_acquire); }
	[[nodiscard]] bool HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
		return m_page_manager.HasReadWatchers(vaddr, size);
	}
	[[nodiscard]] const std::atomic<uint64_t>* ReadGranules() const noexcept { return m_page_manager.ReadGranules(); }
	[[nodiscard]] bool WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
		return m_page_manager.WatchedSpan(vaddr, size, first, end);
	}
	[[nodiscard]] uint64_t FrameNumber() const noexcept { return m_frame_number; }
#if defined(ANYPS5_KYTY)
	[[nodiscard]] uint64_t FlipNumber() const noexcept { return m_flip_number.load(std::memory_order_relaxed); }
	void NoteFlip() noexcept { m_flip_number.fetch_add(1, std::memory_order_relaxed); }
#endif
	[[nodiscard]] bool UnmappedSince(uint64_t version, uint64_t vaddr, uint64_t size) const;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               CacheDmaBases(const ShaderStageRuntime& runtime);
	bool               CacheSelfSizedBuffers(const ShaderStageRuntime& runtime);
	[[nodiscard]] bool SynchronizeDmaFootprint(std::span<const ShaderStageRuntime* const> stages);
	void               PrepareBda(bool may_write, bool full_sync = true);
	void               RunGarbageCollector();
	void NoteHleTlas(uint64_t header, uint64_t size, std::vector<uint64_t> blases);
	void ForgetHleTlas(uint64_t header);
	void NoteGuestBvhMemory(uint64_t address, uint64_t size);

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	void CacheTlasMemory(const ShaderStageRuntime& runtime);
	[[nodiscard]] bool SynchronizeDmaReach(std::span<const ShaderStageRuntime* const> stages);
	[[nodiscard]] bool AddRecordReach(const ShaderStageRuntime& runtime);
	[[nodiscard]] bool AddRayTracingReach(const ShaderStageRuntime& runtime);
	[[nodiscard]] bool RayQueryTableVsharp(const ShaderStageRuntime& runtime, uint32_t table,
	                                       ShaderBufferResource& vsharp);
	[[nodiscard]] bool GuestTlasHeader(const ShaderStageRuntime& runtime, const RayTracing::TlasSource& source,
	                                   uint64_t& header, uint64_t& header_at);
	[[nodiscard]] bool GuestBlasMemory(uint64_t blas, uint64_t& size);
	[[nodiscard]] bool AddBufferTable(const ShaderBufferResource& table);

	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	bool WaitWriteBacks(uint64_t vaddr, uint64_t size);

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	ImageHeap                 m_image_heap;
	PipelineCache             m_pipeline_cache;
	StrictWatch               m_strict_watch {m_page_manager};
	IndirectWriteTables       m_indirect_write_tables {*this};
	OcclusionQueries          m_occlusion_queries {*this};
	LodStatsCounter           m_lod_stats {*this};
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	uint64_t                  m_mapped_ranges_version = 0;
	std::atomic<uint64_t>     m_mapping_epoch {0};
	std::vector<std::pair<uint64_t, uint64_t>> m_bda_mapped_since_sync;
	struct UnmapEntry {
		uint64_t version = 0, vaddr = 0, size = 0;
	};
	std::deque<UnmapEntry> m_unmap_log;
	uint64_t               m_unmap_log_floor = 0;
	uint64_t               m_frame_number    = 0;
#if defined(ANYPS5_KYTY)
	std::atomic<uint64_t> m_flip_number {0};
#endif
	std::vector<std::pair<uint64_t, uint64_t>> m_dma_spans;
	std::vector<std::pair<uint64_t, uint64_t>> m_table_runs;
	struct HleTlas {
		uint64_t              header = 0;
		uint64_t              size   = 0;
		std::vector<uint64_t> blases;
	};
	std::vector<HleTlas>         m_hle_tlases;
	std::map<uint64_t, uint64_t> m_guest_bvh_memory;
	struct BufferTable {
		uint64_t                address = 0;
		std::vector<uint8_t>    copy;
		RangeSet                ranges;
		BufferCache::SetRegions regions;
		std::array<uint64_t, 3> compared_in {};
		std::array<uint64_t, 3> synced_in {};
		uint32_t                replaced = 0;
		std::vector<std::pair<uint64_t, uint64_t>> written;
	};
	[[nodiscard]] std::array<uint64_t, 3> ReachLayout() const {
		return {m_bda_generation, m_buffer_cache.RegisterEpoch(), m_mapped_ranges_version};
	}
	std::vector<std::unique_ptr<BufferTable>> m_buffer_tables;
	std::vector<BufferTable*> m_dma_sets;
	std::vector<uint64_t>     m_walk_blases;
	std::array<uint64_t, 3> m_bda_sync_epochs {UINT64_MAX, UINT64_MAX, UINT64_MAX};
	uint64_t m_bda_generation        = 1;
	uint64_t m_bda_synced_generation = 0;
	uint64_t m_bda_unbounded_ops = 0;
	struct TracedProgram {
		uint64_t unbounded_ops = 0;
		uint8_t  recent        = 0;
	};
	std::map<uint64_t, TracedProgram> m_traced_programs;
	uint64_t m_bda_submit            = UINT64_MAX;
	std::unique_ptr<GuestGpu> m_gpu;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_use_scan_pending      = false;
	uint32_t                  m_use_scan_countdown    = 0;

	std::unique_ptr<RayTracing> m_ray_tracing;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

}

#endif
