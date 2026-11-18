#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_

#if defined(ANYPS5_KYTY)

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

enum class PipelineLibraryPart : uint32_t {
	VertexInput      = 0,
	PreRasterization = 1,
	FragmentShader   = 2,
	FragmentOutput   = 3,
};

[[nodiscard]] bool PipelineLibrariesWanted();
[[nodiscard]] bool PipelineLibrariesAhead();

void       CreateGraphicsPipelineLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                        const GraphicsPipelineKey&                 key,
                                        const std::array<const ShaderModule*, 4>& stages);
vk::Result CreateGraphicsPipelineWhole(GraphicContext& graphics, const PipelineCache::Pipeline& layout,
                                       const GraphicsPipelineKey&                 key,
                                       const std::array<const ShaderModule*, 4>& stages,
                                       vk::PipelineCache cache, vk::PipelineCreateFlags flags,
                                       vk::Pipeline& out);
vk::Pipeline CreateGraphicsPipelineLibrary(GraphicContext& graphics, vk::PipelineLayout layout,
                                           bool native_bindings, const GraphicsPipelineKey& key,
                                           const std::array<const ShaderModule*, 4>& stages,
                                           PipelineLibraryPart                         part);
vk::Pipeline LinkGraphicsPipeline(GraphicContext& graphics, vk::PipelineLayout layout,
                                  std::span<const vk::Pipeline> libraries);

void PlaceBackgroundCompileThread(bool low_priority);

[[nodiscard]] const char*& PipelineRouteNote();

class PipelineLibraries {
public:
	using Stages = std::array<const ShaderModule*, 4>;

	using Optimized = OptimizedGraphicsPipeline;

	PipelineLibraries(GraphicContext& graphics, vk::PipelineCache driver_cache,
	                  std::function<void(std::chrono::steady_clock::time_point)> on_optimized);
	~PipelineLibraries();
	KYTY_CLASS_NO_COPY(PipelineLibraries);

	[[nodiscard]] static bool Eligible(const Stages& stages);
	bool Link(PipelineCache::Pipeline& pipeline, const GraphicsPipelineKey& key, const Stages& stages);
	void Prepare(const GraphicsPipelineKey& key, const Stages& stages, bool native);
	void PrepareNow(const GraphicsPipelineKey& key, const Stages& stages, bool native);
	void QueueOptimized(const PipelineCache::Pipeline& linked, const GraphicsPipelineKey& key,
	                    const Stages& stages);
	[[nodiscard]] uint64_t OptimizedReady() const {
		return m_optimized_ready.load(std::memory_order_acquire);
	}
	std::vector<Optimized> TakeOptimized();
	void                   Stop();
	void                   LogStats();
	void                   Tick();
	[[nodiscard]] bool     Active() const { return m_active.load(std::memory_order_relaxed); }

private:
	struct Library;
	struct OptimizedJob {
		GraphicsPipelineKey key {};
		Stages              stages {};
		bool                native = false;
		vk::PipelineLayout  layout = nullptr;
	};

	std::shared_ptr<Library> Request(PipelineLibraryPart part, std::string&& key,
	                                 const GraphicsPipelineKey& pipeline_key, const Stages& stages,
	                                 bool native, vk::PipelineLayout layout, bool urgent, bool ahead);
	vk::Pipeline             Await(const std::shared_ptr<Library>& library);
	void                     BuildIfQueued(const std::shared_ptr<Library>& library);
	void                     Build(const std::shared_ptr<Library>& library);
	vk::PipelineLayout       LibraryLayout(const std::string& key, const GraphicsPipelineKey& pipeline_key,
	                                       const Stages& stages, bool native);
	void                     Worker();
	void                     OptimizedWorker();

	GraphicContext&                                            m_graphics;
	vk::PipelineCache                                          m_driver_cache;
	std::function<void(std::chrono::steady_clock::time_point)> m_on_optimized;

	std::mutex                                                              m_mutex;
	std::condition_variable                                                 m_available;
	std::condition_variable                                                 m_done;
	std::array<std::unordered_map<std::string, std::shared_ptr<Library>>, 4> m_parts;
	std::unordered_map<std::string, PipelineCache::Pipeline>                 m_layouts;
	std::deque<std::shared_ptr<Library>>                                     m_queue;
	bool                                                                     m_stopping = false;
	std::vector<std::jthread>                                                m_workers;

	std::mutex                m_optimized_mutex;
	std::condition_variable   m_optimized_available;
	std::deque<OptimizedJob>  m_optimized_jobs;
	std::vector<Optimized>    m_optimized;
	bool                      m_optimized_stopping = false;
	std::atomic<uint64_t>     m_optimized_ready {0};
	std::vector<std::jthread> m_optimized_workers;

	std::array<std::atomic<uint64_t>, 4> m_built {};
	std::array<std::atomic<uint64_t>, 4> m_build_ns {};
	std::atomic<uint64_t>                m_linked {0};
	std::atomic<uint64_t>                m_link_ns {0};
	std::atomic<uint64_t>                m_wait_ns {0};
	std::atomic<uint64_t>                m_ahead_requested {0};
	std::atomic<uint64_t>                m_ahead_used {0};
	std::atomic<uint64_t>                m_claimed {0};
	std::atomic<uint64_t>                m_failed {0};
	std::atomic<uint64_t>                m_optimized_built {0};
	std::atomic<uint64_t>                m_optimized_ns {0};
	std::atomic<int64_t>                 m_last_link {0};
	std::atomic<bool>                    m_active {false};
	std::atomic<uint64_t>                m_built_ahead {0};
};

}

#endif

#endif
