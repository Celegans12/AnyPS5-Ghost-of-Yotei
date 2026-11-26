#include "gpu_tiler_shaders/gpu_tiler_color_clear_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_color_transform_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_demote_d16_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_depth_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_promote_d16_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_prt_3d_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_prt_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_render_target_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard256_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard4_3d_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard4_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard64_3d_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard64_spv.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

constexpr const char* Name = "kyty_shim_tiler_shaders";

constexpr uint32_t FamilyCount          = static_cast<uint32_t>(TileBlockFamily::Count);
constexpr uint32_t BytesPerElementCount = 5u;

struct NamedShader {
	const char*               name;
	std::span<const uint32_t> code;
};

constexpr std::array<NamedShader, FamilyCount> FamilyShaders {{
    {"standard256", GPU_TILER_STANDARD256_SPV},
    {"standard4", GPU_TILER_STANDARD4_SPV},
    {"standard4_3d", GPU_TILER_STANDARD4_3D_SPV},
    {"standard64", GPU_TILER_STANDARD64_SPV},
    {"standard64_3d", GPU_TILER_STANDARD64_3D_SPV},
    {"prt", GPU_TILER_PRT_SPV},
    {"prt_3d", GPU_TILER_PRT_3D_SPV},
    {"render_target", GPU_TILER_RENDER_TARGET_SPV},
    {"depth", GPU_TILER_DEPTH_SPV},
}};

constexpr std::array<NamedShader, 4> AuxShaders {{
    {"promote_d16", GPU_TILER_PROMOTE_D16_SPV},
    {"demote_d16", GPU_TILER_DEMOTE_D16_SPV},
    {"color_transform", GPU_TILER_COLOR_TRANSFORM_SPV},
    {"color_clear", GPU_TILER_COLOR_CLEAR_SPV},
}};

struct alignas(16) MipParams {
	uint32_t src_base;
	uint32_t dst_base;
	uint32_t width;
	uint32_t height;
	uint32_t pitch_bytes;
	uint32_t slice_bytes;
	uint32_t tiled_slice_bytes;
	uint32_t blocks_per_row;
	uint32_t surface_z;
	uint32_t tail_x;
	uint32_t tail_y;
	uint32_t num_texels;
};

struct TilingParamsBuffer {
	uint32_t                  num_texels      = 0;
	uint32_t                  color_transform = 0;
	std::array<uint32_t, 2>   padding {};
	std::array<MipParams, 16> mips {};
};

static_assert(sizeof(MipParams) == 48);
static_assert(sizeof(TilingParamsBuffer) == 784);

void Require(bool condition, const std::string& message) {
	if (condition) return;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, message.c_str());
	std::exit(1);
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess,
	        std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess,
	        std::string(what) + " failed: " + vk::to_string(result));
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0,
	        std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s_gpu: skipped, %s\n", Name, why);
	return 77;
}

void VerifySpirvModule(const char* label, std::span<const uint32_t> code) {
	Require(code.size() >= 5u && code[0] == spv::MagicNumber,
	        std::string(label) + ": invalid SPIR-V header");

	spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
	std::string          errors;
	validator.SetMessageConsumer(
	    [&](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		    if (!errors.empty()) errors += "; ";
		    errors += message;
	    });
	const bool valid = validator.Validate(code.data(), code.size());
	Require(valid, std::string(label) + ": spirv-val failed: " + errors);

	std::set<uint32_t> vector_types;
	size_t             offset = 5u;
	while (offset < code.size()) {
		const uint32_t word  = code[offset];
		const uint32_t count = word >> 16u;
		const auto     op    = static_cast<spv::Op>(word & 0xffffu);
		Require(count != 0u && offset + count <= code.size(),
		        std::string(label) + ": truncated SPIR-V instruction");
		if (op == spv::OpTypeVector && count >= 4u) {
			vector_types.insert(code[offset + 1u]);
		} else if (op == spv::OpSpecConstantOp && count >= 4u) {
			const uint32_t result_type = code[offset + 1u];
			const auto     sub_op      = static_cast<spv::Op>(code[offset + 3u]);
			Require(!(sub_op == spv::OpSelect && vector_types.contains(result_type)),
			        std::string(label) + ": contains vector OpSpecConstantOp Select");
		}
		offset += count;
	}
}

void CheckAllSpirvModules() {
	for (const auto& shader : FamilyShaders) {
		VerifySpirvModule(shader.name, shader.code);
	}
	for (const auto& shader : AuxShaders) {
		VerifySpirvModule(shader.name, shader.code);
	}
}

uint32_t FindMemoryType(const vk::PhysicalDeviceMemoryProperties& props, uint32_t bits,
                        vk::MemoryPropertyFlags flags) {
	for (uint32_t i = 0; i < props.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) != 0u && (props.memoryTypes[i].propertyFlags & flags) == flags) {
			return i;
		}
	}
	Require(false, "no suitable Vulkan memory type");
	return 0;
}

struct HostBuffer {
	vk::Device       device = nullptr;
	vk::Buffer       buffer = nullptr;
	vk::DeviceMemory memory = nullptr;
	void*            mapped = nullptr;
	vk::DeviceSize   size   = 0;

	HostBuffer(vk::Device dev, const vk::PhysicalDeviceMemoryProperties& mem_props,
	           vk::DeviceSize bytes, vk::BufferUsageFlags usage)
	    : device(dev), size(bytes) {
		vk::BufferCreateInfo info {};
		info.size        = bytes;
		info.usage       = usage;
		info.sharingMode = vk::SharingMode::eExclusive;
		buffer           = Checked(device.createBuffer(info), "createBuffer");
		const auto req   = device.getBufferMemoryRequirements(buffer);
		vk::MemoryAllocateInfo alloc {};
		alloc.allocationSize = req.size;
		alloc.memoryTypeIndex = FindMemoryType(
		    mem_props, req.memoryTypeBits,
		    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		memory = Checked(device.allocateMemory(alloc), "allocateMemory");
		Checked(device.bindBufferMemory(buffer, memory, 0), "bindBufferMemory");
		mapped = Checked(device.mapMemory(memory, 0, bytes), "mapMemory");
	}

	HostBuffer(const HostBuffer&)            = delete;
	HostBuffer& operator=(const HostBuffer&) = delete;

	~HostBuffer() {
		if (mapped != nullptr) device.unmapMemory(memory);
		if (buffer != nullptr) device.destroyBuffer(buffer);
		if (memory != nullptr) device.freeMemory(memory);
	}
};

vk::Pipeline CreateTilerPipeline(vk::Device device, vk::PipelineLayout layout,
                                 std::span<const uint32_t> code, uint32_t element_bytes,
                                 uint32_t direction) {
	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = code.size() * sizeof(uint32_t);
	module_info.pCode    = code.data();
	const auto module    = Checked(device.createShaderModule(module_info), "createShaderModule");

	const std::array<uint32_t, 2>                  values {element_bytes, direction};
	const std::array<vk::SpecializationMapEntry, 2> entries {{{0, 0, 4}, {1, 4, 4}}};
	const vk::SpecializationInfo                   spec {2, entries.data(), sizeof(values),
                                       values.data()};

	vk::ComputePipelineCreateInfo create {};
	create.stage.stage               = vk::ShaderStageFlagBits::eCompute;
	create.stage.module              = module;
	create.stage.pName               = "main";
	create.stage.pSpecializationInfo = &spec;
	create.layout                    = layout;

	vk::Pipeline pipeline = nullptr;
	const auto   result   = device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
	device.destroyShaderModule(module);
	Checked(result, "createComputePipelines");
	return pipeline;
}

void RunTilerDispatch(vk::Queue queue, vk::CommandBuffer cmd, vk::PipelineLayout layout,
                      vk::DescriptorSet set, vk::Pipeline pipeline, uint32_t num_texels) {
	Checked(cmd.reset({}), "resetCommandBuffer");
	const vk::CommandBufferBeginInfo begin {.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
	Checked(cmd.begin(begin), "beginCommandBuffer");
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &set, 0, nullptr);
	cmd.dispatch((num_texels + 63u) / 64u, 1, 1);
	const vk::MemoryBarrier barrier {
	    .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
	    .dstAccessMask = vk::AccessFlagBits::eHostRead,
	};
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                    vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0, nullptr, 0, nullptr);
	Checked(cmd.end(), "endCommandBuffer");
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &cmd;
	Checked(queue.submit(1, &submit, nullptr), "queueSubmit");
	Checked(queue.waitIdle(), "queueWaitIdle");
}

void UpdateBuffers(vk::Device device, vk::DescriptorSet set, vk::Buffer src, vk::Buffer dst,
                   vk::Buffer ubo, vk::DeviceSize block_bytes) {
	const std::array<vk::DescriptorBufferInfo, 3> infos {{
	    {src, 0, block_bytes},
	    {dst, 0, block_bytes},
	    {ubo, 0, sizeof(TilingParamsBuffer)},
	}};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t i = 0; i < 3u; ++i) {
		writes[i].dstSet          = set;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType =
		    i == 2u ? vk::DescriptorType::eUniformBuffer : vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo = &infos[i];
	}
	device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

int RunGpuChecks() {
	vk::detail::DynamicLoader loader;
	if (!loader.success()) return NoDevice("Vulkan loader unavailable");
	const auto get_instance =
	    loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	if (get_instance == nullptr) return NoDevice("vkGetInstanceProcAddr unavailable");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_instance);

	const vk::ApplicationInfo app {.pApplicationName = Name,
	                               .apiVersion       = VULKAN_TARGET_API_VERSION};
	const vk::InstanceCreateInfo instance_info {.pApplicationInfo = &app};
	vk::Instance                 instance = nullptr;
	if (vk::createInstance(&instance_info, nullptr, &instance) != vk::Result::eSuccess) {
		return NoDevice("could not create a Vulkan 1.3 instance");
	}
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

	struct InstanceGuard {
		vk::Instance handle;
		~InstanceGuard() {
			if (handle != nullptr) handle.destroy();
		}
	} instance_guard {instance};

	const auto devices = instance.enumeratePhysicalDevices();
	if (devices.result != vk::Result::eSuccess || devices.value.empty()) {
		return NoDevice("no Vulkan physical device");
	}

	vk::PhysicalDevice chosen      = nullptr;
	uint32_t           queue_index = UINT32_MAX;
	for (const auto physical : devices.value) {
		const auto props = physical.getProperties();
		if (props.apiVersion < VULKAN_TARGET_API_VERSION) continue;

		vk::PhysicalDeviceVulkan12Features supported12 {};
		vk::PhysicalDeviceVulkan11Features supported11 {.pNext = &supported12};
		vk::PhysicalDeviceFeatures2        supported {.pNext = &supported11};
		physical.getFeatures2(&supported);
		if (!supported11.storageBuffer16BitAccess || !supported12.storageBuffer8BitAccess) {
			continue;
		}

		const auto families = physical.getQueueFamilyProperties();
		for (uint32_t i = 0; i < families.size(); ++i) {
			if ((families[i].queueFlags & vk::QueueFlagBits::eCompute) != vk::QueueFlags {}) {
				chosen      = physical;
				queue_index = i;
				break;
			}
		}
		if (chosen != nullptr) break;
	}
	if (chosen == nullptr) {
		return NoDevice("no Vulkan 1.3 device with 8-bit and 16-bit storage buffer access");
	}

	vk::PhysicalDeviceVulkan12Features features12 {.storageBuffer8BitAccess = VK_TRUE};
	vk::PhysicalDeviceVulkan11Features features11 {.pNext                    = &features12,
	                                               .storageBuffer16BitAccess = VK_TRUE};
	const float                        priority = 1.0f;
	const vk::DeviceQueueCreateInfo    queue_info {
        .queueFamilyIndex = queue_index, .queueCount = 1, .pQueuePriorities = &priority};
	const vk::DeviceCreateInfo device_info {.pNext                = &features11,
	                                        .queueCreateInfoCount = 1,
	                                        .pQueueCreateInfos    = &queue_info};
	vk::Device                 device = nullptr;
	Checked(chosen.createDevice(&device_info, nullptr, &device), "createDevice");
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

	struct DeviceGuard {
		vk::Device handle;
		~DeviceGuard() {
			if (handle != nullptr) handle.destroy();
		}
	} device_guard {device};

	const auto queue     = device.getQueue(queue_index, 0);
	const auto mem_props = chosen.getMemoryProperties();

	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	bindings[0] = {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute};
	bindings[1] = {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute};
	bindings[2] = {2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute};
	const vk::DescriptorSetLayoutCreateInfo dsl_info {
	    .bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()};
	const auto dsl = Checked(device.createDescriptorSetLayout(dsl_info), "createDescriptorSetLayout");
	const vk::PipelineLayoutCreateInfo pl_info {.setLayoutCount = 1, .pSetLayouts = &dsl};
	const auto pl = Checked(device.createPipelineLayout(pl_info), "createPipelineLayout");

	const std::array<vk::DescriptorPoolSize, 2> pool_sizes {{
	    {vk::DescriptorType::eStorageBuffer, 2},
	    {vk::DescriptorType::eUniformBuffer, 1},
	}};
	const vk::DescriptorPoolCreateInfo pool_info {.maxSets       = 1,
	                                              .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
	                                              .pPoolSizes    = pool_sizes.data()};
	const auto pool = Checked(device.createDescriptorPool(pool_info), "createDescriptorPool");
	const vk::DescriptorSetAllocateInfo set_alloc {
	    .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
	const auto sets = Checked(device.allocateDescriptorSets(set_alloc), "allocateDescriptorSets");
	const auto set  = sets[0];

	const vk::CommandPoolCreateInfo cp_info {
	    .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer, .queueFamilyIndex = queue_index};
	const auto cmd_pool = Checked(device.createCommandPool(cp_info), "createCommandPool");
	const vk::CommandBufferAllocateInfo cb_alloc {.commandPool        = cmd_pool,
	                                              .level              = vk::CommandBufferLevel::ePrimary,
	                                              .commandBufferCount = 1};
	const auto cmds = Checked(device.allocateCommandBuffers(cb_alloc), "allocateCommandBuffers");
	const auto cmd  = cmds[0];

	constexpr uint32_t BlockBytes = 65536u;
	HostBuffer linear_buf(device, mem_props, BlockBytes, vk::BufferUsageFlagBits::eStorageBuffer);
	HostBuffer tiled_buf(device, mem_props, BlockBytes, vk::BufferUsageFlagBits::eStorageBuffer);
	HostBuffer roundtrip_buf(device, mem_props, BlockBytes, vk::BufferUsageFlagBits::eStorageBuffer);
	HostBuffer ubo_buf(device, mem_props, sizeof(TilingParamsBuffer),
	                   vk::BufferUsageFlagBits::eUniformBuffer);

	constexpr std::array<TileBlockFamily, 3> VerifiedFamilies {
	    TileBlockFamily::Standard64KB3D,
	    TileBlockFamily::Prt64KB,
	    TileBlockFamily::Prt64KB3D,
	};

	std::vector<uint8_t> linear_src(BlockBytes);
	std::vector<uint8_t> expected_tiled(BlockBytes);
	for (uint32_t i = 0; i < BlockBytes; ++i) {
		linear_src[i] = static_cast<uint8_t>((i * 131u + (i >> 8u) * 17u + 29u) & 0xffu);
	}

	for (uint32_t family = 0; family < FamilyCount; ++family) {
		const auto& shader = FamilyShaders[family];
		for (uint32_t elem_idx = 0; elem_idx < BytesPerElementCount; ++elem_idx) {
			const uint32_t bytes_per_element = 1u << elem_idx;
			vk::Pipeline   detile_pipe = CreateTilerPipeline(device, pl, shader.code, bytes_per_element, 0u);
			vk::Pipeline   tile_pipe   = CreateTilerPipeline(device, pl, shader.code, bytes_per_element, 1u);

			const auto block_family = static_cast<TileBlockFamily>(family);
			bool       verify       = false;
			for (const auto target : VerifiedFamilies) {
				if (target == block_family) verify = true;
			}
			if (verify) {
				TileBlockLayout layout {};
				Require(TileGetBlockLayout(block_family, bytes_per_element, layout),
				        "TileGetBlockLayout failed");
				Require(layout.block_size == BlockBytes, "unexpected block size");

				const uint32_t pitch_bytes = layout.block_width * bytes_per_element;
				const uint32_t slice_bytes = pitch_bytes * layout.block_height;
				const uint32_t num_texels  = layout.block_width * layout.block_height * layout.block_depth;

				TilingParamsBuffer params {};
				params.num_texels      = num_texels;
				params.color_transform = 0u;
				params.mips[0]         = {
                    .src_base          = 0u,
                    .dst_base          = 0u,
                    .width             = layout.block_width,
                    .height            = layout.block_height,
                    .pitch_bytes       = pitch_bytes,
                    .slice_bytes       = slice_bytes,
                    .tiled_slice_bytes = BlockBytes,
                    .blocks_per_row    = 1u,
                    .surface_z         = 0u,
                    .tail_x            = 0u,
                    .tail_y            = 0u,
                    .num_texels        = num_texels,
                };
				std::memcpy(ubo_buf.mapped, &params, sizeof(params));
				std::memcpy(linear_buf.mapped, linear_src.data(), BlockBytes);
				std::memset(tiled_buf.mapped, 0, BlockBytes);
				std::memset(roundtrip_buf.mapped, 0, BlockBytes);

				std::fill(expected_tiled.begin(), expected_tiled.end(), uint8_t {0});
				for (uint32_t z = 0; z < layout.block_depth; ++z) {
					for (uint32_t y = 0; y < layout.block_height; ++y) {
						for (uint32_t x = 0; x < layout.block_width; ++x) {
							uint32_t tiled_off = 0;
							Require(TileGetBlockOffset(layout, x, y, z, tiled_off),
							        "TileGetBlockOffset failed");
							const uint32_t linear_off =
							    z * slice_bytes + y * pitch_bytes + x * bytes_per_element;
							std::memcpy(expected_tiled.data() + tiled_off,
							            linear_src.data() + linear_off, bytes_per_element);
						}
					}
				}

				UpdateBuffers(device, set, linear_buf.buffer, tiled_buf.buffer, ubo_buf.buffer,
				              BlockBytes);
				RunTilerDispatch(queue, cmd, pl, set, tile_pipe, num_texels);
				Require(std::memcmp(tiled_buf.mapped, expected_tiled.data(), BlockBytes) == 0,
				        std::string(shader.name) + " bpe=" + std::to_string(bytes_per_element) +
				            ": GPU tiled output differs from TileGetBlockOffset");

				UpdateBuffers(device, set, tiled_buf.buffer, roundtrip_buf.buffer, ubo_buf.buffer,
				              BlockBytes);
				RunTilerDispatch(queue, cmd, pl, set, detile_pipe, num_texels);
				Require(std::memcmp(roundtrip_buf.mapped, linear_src.data(), BlockBytes) == 0,
				        std::string(shader.name) + " bpe=" + std::to_string(bytes_per_element) +
				            ": GPU detile round-trip mismatch");
			}

			device.destroyPipeline(tile_pipe);
			device.destroyPipeline(detile_pipe);
		}
	}

	device.destroyCommandPool(cmd_pool);
	device.destroyDescriptorPool(pool);
	device.destroyPipelineLayout(pl);
	device.destroyDescriptorSetLayout(dsl);
	std::printf("%s_gpu: ok\n", Name);
	return 0;
}

}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "gpu") == 0) {
		return RunGpuChecks();
	}
	CheckAllSpirvModules();
	std::printf("%s: ok\n", Name);
	return 0;
}
