#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECHAINTHREADS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECHAINTHREADS_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {

inline constexpr uint32_t DefaultPipelineChainThreads = 4;
inline constexpr uint32_t MaxPipelineChainThreads     = 16;

struct PipelineChainThreadSetting {
	uint32_t threads   = DefaultPipelineChainThreads;
	uint64_t requested = DefaultPipelineChainThreads;
	bool     capped    = false;
};

[[nodiscard]] inline PipelineChainThreadSetting PipelineChainThreadCount(const char* value) {
	if (value == nullptr) {
		return {};
	}
	const auto requested = std::strtoull(value, nullptr, 10);
	return {.threads   = static_cast<uint32_t>(std::clamp<unsigned long long>(requested, 1ull, MaxPipelineChainThreads)),
	        .requested = requested,
	        .capped    = requested > MaxPipelineChainThreads};
}

}

#endif
