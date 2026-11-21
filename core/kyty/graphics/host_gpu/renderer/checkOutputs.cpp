#include "graphics/host_gpu/renderer/checkOutputs.h"

#include "gpu_tiler_shaders/check_compare_spv.h"

#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <fmt/format.h>

#include <algorithm>

namespace Libs::Graphics {

namespace {

constexpr uint64_t RegionBytes = uint64_t {128} << 20u;
constexpr uint64_t OutputAlign = 256;

uint64_t Aligned(uint64_t size) {
	return (size + OutputAlign - 1) / OutputAlign * OutputAlign;
}

auto OutputOrder(const CheckOutput& output) {
	return std::tuple {output.key, output.kind, output.size};
}

}

uint64_t CheckOutputsBytes(const std::vector<CheckOutput>& outputs) {
	return outputs.empty() ? 0 : outputs.back().at + Aligned(outputs.back().size);
}

bool CollectCheckOutputs(RenderContext& context, const PreparedBindings& bindings, std::vector<CheckOutput>& outputs,
                         std::string& reason) {
	const auto& program  = bindings.runtime->module->info;
	auto&       textures = context.GetTextureCache();
	outputs.clear();
	for (const auto& resource: program.info.buffers) {
		if (!resource.written) {
			continue;
		}
		const auto index = resource.descriptor_index;
		if (index >= bindings.buffers.size() || index >= bindings.buffer_sources.size()) {
			reason = "writes a buffer without a binding";
			return false;
		}
		const auto& source = bindings.buffer_sources[index];
		const auto& view   = bindings.buffers[index];
		if (source.address == 0 || view.buffer == nullptr || view.range == 0) {
			continue;
		}
		if (view.range == VK_WHOLE_SIZE || view.range < source.size) {
			reason = "writes a buffer without a bounded view";
			return false;
		}
		outputs.push_back({.key = source.address, .size = source.size, .kind = 0, .buffer = view.buffer,
		                   .offset = view.offset + (view.range - source.size)});
	}
	for (uint32_t i = 0; i < program.info.images.size() && i < bindings.images.size(); i++) {
		const auto& binding = bindings.images[i];
		if (!program.info.images[i].written || !binding.image_id) {
			continue;
		}
		auto& image = textures.GetImage(binding.image_id);
		if (image.info.IsDepth() || image.info.samples != 1) {
			reason = "writes a depth or multisampled image";
			return false;
		}
		outputs.push_back({.key = image.info.data.address, .size = image.info.LinearSize(), .kind = 1,
		                   .image = binding.image_id.index});
	}
	std::sort(outputs.begin(), outputs.end(),
	          [](const CheckOutput& a, const CheckOutput& b) { return OutputOrder(a) < OutputOrder(b); });
	outputs.erase(std::unique(outputs.begin(), outputs.end(),
	                          [](const CheckOutput& a, const CheckOutput& b) { return OutputOrder(a) == OutputOrder(b); }),
	              outputs.end());
	uint64_t at = 0;
	for (auto& output: outputs) {
		output.at = at;
		at += Aligned(output.size);
	}
	if (at == 0) {
		reason = "writes no output";
		return false;
	}
	if (at > RegionBytes) {
		reason = fmt::format("writes {} MiB, more than the {} MiB the check keeps", (at + (1u << 20u) - 1) >> 20u,
		                     RegionBytes >> 20u);
		return false;
	}
	return true;
}

void CopyCheckOutputs(RenderContext& context, const CommandRecorder& command, vk::Buffer scratch,
                      const std::vector<CheckOutput>& outputs, uint64_t region, bool restore) {
	auto& textures = context.GetTextureCache();
	for (const auto& output: outputs) {
		const auto at = region + output.at;
		if (output.kind == 0) {
			if (restore) {
				const vk::BufferCopy copy {at, output.offset, output.size};
				command.copyBuffer(scratch, output.buffer, 1, &copy);
			} else {
				const vk::BufferCopy copy {output.offset, at, output.size};
				command.copyBuffer(output.buffer, scratch, 1, &copy);
			}
			continue;
		}
		auto&      image  = textures.GetImage(ImageId {output.image});
		const auto copies = image.info.BufferCopies(at);
		if (restore) {
			image.Upload(copies, scratch, at, output.size);
		} else {
			image.Download(copies, scratch, at, output.size);
		}
	}
}

void CheckBarrier(const CommandRecorder& command) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eHost;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
	                        vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

void CreateCheckComparePipeline(vk::Device device, const char* check, const char* name, CheckComparePipeline& compare) {
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = static_cast<uint32_t>(std::size(bindings));
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(device.createDescriptorSetLayout(&layout_info, nullptr, &compare.descriptor_layout),
	                     fmt::format("create {} descriptor layout", check).c_str());
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(CheckComparePush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &compare.descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push;
	RequireVulkanSuccess(device.createPipelineLayout(&pipeline_layout_info, nullptr, &compare.pipeline_layout),
	                     fmt::format("create {} pipeline layout", check).c_str());
	compare.pipeline = CreateComputePipeline(device, CHECK_COMPARE_SPV, compare.pipeline_layout,
	                                         fmt::format("create {} pipeline", check).c_str());
	SetVulkanObjectNameF(device, compare.pipeline, "{}", name);
}

void DestroyCheckComparePipeline(vk::Device device, CheckComparePipeline& compare) {
	if (compare.pipeline) {
		device.destroyPipeline(compare.pipeline, nullptr);
	}
	if (compare.pipeline_layout) {
		device.destroyPipelineLayout(compare.pipeline_layout, nullptr);
	}
	if (compare.descriptor_layout) {
		device.destroyDescriptorSetLayout(compare.descriptor_layout, nullptr);
	}
	compare = {};
}

}
