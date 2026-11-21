#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CHECK_OUTPUTS_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CHECK_OUTPUTS_H_

#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics {

class RenderContext;
struct PreparedBindings;

struct CheckOutput {
	uint64_t   key    = 0;
	uint64_t   size   = 0;
	uint32_t   kind   = 0;
	vk::Buffer buffer = nullptr;
	uint64_t   offset = 0;
	uint32_t   image  = 0;
	uint64_t   at     = 0;
};

struct CheckComparePush {
	uint32_t count = 0;
	uint32_t slot  = 0;
	uint32_t fetch = 0;
};

struct CheckComparePipeline {
	vk::DescriptorSetLayout descriptor_layout = nullptr;
	vk::PipelineLayout      pipeline_layout   = nullptr;
	vk::Pipeline            pipeline          = nullptr;
};

[[nodiscard]] inline auto CheckOutputPlace(const CheckOutput& output) {
	return std::tuple {output.key, output.kind};
}

[[nodiscard]] uint64_t CheckOutputsBytes(const std::vector<CheckOutput>& outputs);
[[nodiscard]] bool     CollectCheckOutputs(RenderContext& context, const PreparedBindings& bindings,
                                           std::vector<CheckOutput>& outputs, std::string& reason);
void CopyCheckOutputs(RenderContext& context, const CommandRecorder& command, vk::Buffer scratch,
                      const std::vector<CheckOutput>& outputs, uint64_t region, bool restore);
void CheckBarrier(const CommandRecorder& command);
void CreateCheckComparePipeline(vk::Device device, const char* check, const char* name, CheckComparePipeline& compare);
void DestroyCheckComparePipeline(vk::Device device, CheckComparePipeline& compare);

}

#endif
