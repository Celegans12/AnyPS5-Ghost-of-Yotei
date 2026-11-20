#include "common/assert.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderBindings.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
using BindingKind    = Recompiler::IR::DescriptorBindingKind;

constexpr const char* Name           = "kyty_shim_compute_execution";
constexpr uint32_t    WaveSize       = 64u;
constexpr uint32_t    HostSubgroup   = 32u;
constexpr int         SkippedExit    = 77;
constexpr uint32_t    UserDataDwords = 16u;

uint32_t g_failures = 0;

void Require(bool value, const std::string& what) {
	if (value) return;
	std::fprintf(stderr, "%s: %s\n", Name, what.c_str());
	std::abort();
}

void Check(bool value, const std::string& what) {
	if (value) return;
	if (g_failures++ < 16) std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

std::string Hex(uint32_t value) {
	char text[16];
	std::snprintf(text, sizeof(text), "0x%08x", value);
	return text;
}

struct Gpu {
	vk::Instance                       instance;
	vk::PhysicalDevice                 physical;
	vk::Device                         device;
	vk::Queue                          queue;
	uint32_t                           family = 0;
	vk::PhysicalDeviceMemoryProperties memory;
	uint32_t                           min_subgroup = 0;
	uint32_t                           max_subgroup = 0;
	bool                               subgroup_control = false;
	std::string                        name;
};

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0, std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("%s: skipped, %s\n", Name, why);
	return SkippedExit;
}

std::optional<Gpu> OpenGpu(const char*& why) {
	static const vk::detail::DynamicLoader loader;
	if (!loader.success()) {
		why = "the Vulkan loader is missing";
		return std::nullopt;
	}
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) {
		why = "no Vulkan 1.3 instance";
		return std::nullopt;
	}
	Gpu gpu;
	gpu.instance = created.value;
	dispatch.init(gpu.instance);

	for (const auto candidate: Checked(gpu.instance.enumeratePhysicalDevices(), "device enumeration")) {
		if (candidate.getProperties().apiVersion < VK_API_VERSION_1_3) continue;
		const auto families = candidate.getQueueFamilyProperties();
		const auto compute  = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);
		});
		if (compute == families.end()) continue;
		const bool software = candidate.getProperties().deviceType == vk::PhysicalDeviceType::eCpu;
		if (gpu.physical && !software) continue;
		gpu.physical = candidate;
		gpu.family   = static_cast<uint32_t>(compute - families.begin());
		if (software) break;
	}
	if (!gpu.physical) {
		gpu.instance.destroy();
		why = "no Vulkan 1.3 device with a compute queue";
		return std::nullopt;
	}

	vk::PhysicalDeviceVulkan13Features features13;
	vk::PhysicalDeviceVulkan12Features features12;
	vk::PhysicalDeviceVulkan11Features features11;
	vk::PhysicalDeviceFeatures2        features;
	features12.pNext = &features13;
	features11.pNext = &features12;
	features.pNext   = &features11;
	gpu.physical.getFeatures2(&features);
	features13.robustImageAccess = VK_FALSE;
	features.features.robustBufferAccess = VK_FALSE;

	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = gpu.family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	vk::DeviceCreateInfo device_info;
	device_info.pNext                = &features;
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos    = &queue_info;
	gpu.device                       = Checked(gpu.physical.createDevice(device_info), "device creation");
	dispatch.init(gpu.device);
	gpu.queue  = gpu.device.getQueue(gpu.family, 0);
	gpu.memory = gpu.physical.getMemoryProperties();

	vk::PhysicalDeviceVulkan13Properties properties13;
	vk::PhysicalDeviceDriverProperties   driver;
	vk::PhysicalDeviceProperties2        properties;
	driver.pNext     = &properties13;
	properties.pNext = &driver;
	gpu.physical.getProperties2(&properties);
	gpu.min_subgroup = properties13.minSubgroupSize;
	gpu.max_subgroup = properties13.maxSubgroupSize;
	gpu.subgroup_control = features13.subgroupSizeControl == VK_TRUE &&
	                       (properties13.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute);
	gpu.name = std::string(properties.properties.deviceName.data()) + " (" + driver.driverName.data() + ", " +
	           driver.driverInfo.data() + ")";
	return gpu;
}

void CloseGpu(Gpu& gpu) {
	gpu.device.destroy();
	gpu.instance.destroy();
}

struct HostBuffer {
	vk::Buffer       buffer;
	vk::DeviceMemory memory;
	void*            mapped = nullptr;
	uint64_t         bytes  = 0;
};

HostBuffer CreateHostBuffer(const Gpu& gpu, uint64_t bytes) {
	HostBuffer           result;
	vk::BufferCreateInfo info;
	result.bytes = std::max<uint64_t>((bytes + 15u) & ~uint64_t {15u}, 16u);
	info.size    = result.bytes;
	info.usage   = vk::BufferUsageFlagBits::eStorageBuffer;
	result.buffer = Checked(gpu.device.createBuffer(info), "buffer creation");
	const auto needs = gpu.device.getBufferMemoryRequirements(result.buffer);
	const auto flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	for (uint32_t type = 0; type < gpu.memory.memoryTypeCount && !result.memory; type++) {
		if ((needs.memoryTypeBits >> type & 1u) == 0u || (gpu.memory.memoryTypes[type].propertyFlags & flags) != flags)
			continue;
		vk::MemoryAllocateInfo allocate;
		allocate.allocationSize  = needs.size;
		allocate.memoryTypeIndex = type;
		result.memory            = Checked(gpu.device.allocateMemory(allocate), "memory allocation");
	}
	Require(static_cast<bool>(result.memory), "the device has no host-visible memory for a buffer");
	Checked(gpu.device.bindBufferMemory(result.buffer, result.memory, 0), "buffer binding");
	result.mapped = Checked(gpu.device.mapMemory(result.memory, 0, VK_WHOLE_SIZE), "buffer mapping");
	std::memset(result.mapped, 0, result.bytes);
	return result;
}

void DestroyHostBuffer(const Gpu& gpu, HostBuffer& buffer) {
	gpu.device.unmapMemory(buffer.memory);
	gpu.device.destroyBuffer(buffer.buffer);
	gpu.device.freeMemory(buffer.memory);
}

void RequireValid(const std::vector<uint32_t>& spirv, const std::string& what) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	Require(tools.Validate(spirv), what + ": the program is not valid SPIR-V: " + messages);
}

std::array<uint32_t, 4> BufferDescriptor(const void* data, uint32_t records, uint32_t stride, uint32_t word3) {
	const auto address = reinterpret_cast<uintptr_t>(data);
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu) | (stride << 16u), records,
	        word3};
}

struct Dispatch {
	const char*                        name = nullptr;
	std::span<const uint32_t>          code;
	std::array<uint32_t, UserDataDwords> user_data {};
	uint32_t                           threads = 0;
};

bool Run(const Gpu& gpu, const Dispatch& dispatch) {
	const std::string what = dispatch.name;
	ShaderComputeInputInfo compute;
	compute.threads_num[0]     = dispatch.threads;
	compute.threads_num[1]     = 1u;
	compute.threads_num[2]     = 1u;
	compute.wave_size          = WaveSize;
	compute.host_subgroup_size = HostSubgroup;
	compute.thread_ids_num     = 1;
	Recompiler::CompileOptions options;
	options.stage                 = ShaderType::Compute;
	options.wave_size             = WaveSize;
	options.user_data_base        = 0u;
	options.shader_hash           = 0x5ad0u;
	options.dump_ir               = false;
	options.user_data             = dispatch.user_data;
	options.input_info.compute    = &compute;
	const auto source             = Recompiler::PrepareShaderSource(dispatch.code, options);
	auto       translated         = Recompiler::TranslateProgram(source.decoded, options);
	const bool lane_local         = translated.program.lane_local;
	const auto plan               = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime       runtime {.user_data = dispatch.user_data};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	Require(Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
	        what + ": the program's resources were not materialized");
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
	RequireValid(compiled.spirv, what);
	const auto info = std::move(compiled.program).TakeCompiledInfo();
	if (!lane_local && gpu.max_subgroup < HostSubgroup) {
		std::printf("%s: %s: skipped, the program is not lane-local and the device's subgroups hold %u lanes\n", Name,
		            dispatch.name, gpu.max_subgroup);
		return false;
	}
	const auto& layout = info.bindings;
	Require(layout.UsesPushData(), what + ": the shader data does not fit the push constants");
	for (auto mask = layout.descriptor_mask; mask != 0; mask &= mask - 1u) {
		const auto kind = static_cast<BindingKind>(std::countr_zero(mask));
		Require(kind == BindingKind::Buffers, what + ": descriptor binding kind " +
		                                          std::to_string(static_cast<uint32_t>(kind)) + " is not served by this test");
	}

	const auto buffer_count = layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)];
	std::vector<HostBuffer>      buffers(buffer_count);
	std::vector<void*>           targets(buffer_count, nullptr);
	std::vector<bool>            written(buffer_count, false);
	std::vector<uint32_t>        shader_data(layout.ShaderDataDwords(), 0u);
	for (size_t i = 0; i < info.info.user_data_registers.size(); i++) {
		shader_data[i] = dispatch.user_data[info.info.user_data_registers[i] - info.user_data_base];
	}
	for (uint32_t i = 0; i < info.info.buffers.size(); i++) {
		const auto& resource = info.info.buffers[i];
		const auto  index    = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		Require(index < buffer_count && !buffers[index].buffer, what + ": buffer descriptor " + std::to_string(index) +
		                                                            " is out of range or bound twice");
		ShaderBufferResource descriptor;
		std::copy_n(snapshot.buffers[i].dwords.begin(), std::size(descriptor.fields), descriptor.fields);
		auto* host        = reinterpret_cast<void*>(static_cast<uintptr_t>(descriptor.Base48()));
		const auto bytes  = descriptor.GetSize();
		buffers[index]    = CreateHostBuffer(gpu, bytes);
		targets[index]    = host;
		written[index]    = resource.written;
		std::memcpy(buffers[index].mapped, host, bytes);
		shader_data[layout.MemorySizeDword() + index] = static_cast<uint32_t>(bytes);
		if (index < layout.stride_count && Recompiler::IR::HasRuntimeStride(resource.packed_stride)) {
			shader_data[layout.StrideDword() + index] = descriptor.PackedStride() & Recompiler::IR::BufferStrideMask;
		}
	}
	for (uint32_t index = 0; index < buffer_count; index++) {
		Require(static_cast<bool>(buffers[index].buffer), what + ": buffer descriptor " + std::to_string(index) + " has no resource");
	}
	if (layout.dispatch_thread_dword != Recompiler::IR::PushData::NoStart) {
		shader_data[layout.dispatch_thread_dword]      = dispatch.threads;
		shader_data[layout.dispatch_thread_dword + 1u] = 1u;
		shader_data[layout.dispatch_thread_dword + 2u] = 1u;
	}
	Recompiler::IR::PushData push;
	Require(layout.push_data_start_dword + shader_data.size() <= push.dwords.size(), what + ": the push data overflows");
	std::ranges::copy(shader_data, push.dwords.begin() + layout.push_data_start_dword);

	std::vector<vk::DescriptorSetLayoutBinding> set_bindings;
	if (buffer_count != 0) {
		set_bindings.push_back({Recompiler::IR::NativeBinding(ShaderType::Compute, BindingKind::Buffers),
		                        vk::DescriptorType::eStorageBuffer, buffer_count, vk::ShaderStageFlagBits::eCompute, nullptr});
	}
	vk::DescriptorSetLayoutCreateInfo set_layout_info;
	set_layout_info.bindingCount = static_cast<uint32_t>(set_bindings.size());
	set_layout_info.pBindings    = set_bindings.data();
	const auto set_layout        = Checked(gpu.device.createDescriptorSetLayout(set_layout_info), "set layout creation");
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0u, Recompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo pipeline_layout_info;
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	const auto pipeline_layout = Checked(gpu.device.createPipelineLayout(pipeline_layout_info), "pipeline layout creation");
	vk::ShaderModuleCreateInfo module_info;
	module_info.codeSize = compiled.spirv.size() * sizeof(uint32_t);
	module_info.pCode    = compiled.spirv.data();
	const auto module    = Checked(gpu.device.createShaderModule(module_info), "module creation");
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup;
	vk::ComputePipelineCreateInfo                         pipeline_info;
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	if (gpu.subgroup_control && HostSubgroup >= gpu.min_subgroup && HostSubgroup <= gpu.max_subgroup) {
		subgroup.requiredSubgroupSize = HostSubgroup;
		pipeline_info.stage.pNext     = &subgroup;
	}
	pipeline_info.layout = pipeline_layout;
	vk::Pipeline pipeline;
	Checked(gpu.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline), "pipeline creation");

	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eStorageBuffer, std::max(buffer_count, 1u)};
	vk::DescriptorPoolCreateInfo pool_info;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	const auto pool         = Checked(gpu.device.createDescriptorPool(pool_info), "descriptor pool creation");
	vk::DescriptorSetAllocateInfo set_info;
	set_info.descriptorPool     = pool;
	set_info.descriptorSetCount = 1;
	set_info.pSetLayouts        = &set_layout;
	vk::DescriptorSet set;
	Checked(gpu.device.allocateDescriptorSets(&set_info, &set), "descriptor set allocation");
	std::vector<vk::DescriptorBufferInfo> buffer_infos;
	for (const auto& buffer: buffers) buffer_infos.push_back({buffer.buffer, 0u, VK_WHOLE_SIZE});
	if (buffer_count != 0) {
		vk::WriteDescriptorSet write;
		write.dstSet          = set;
		write.dstBinding      = set_bindings[0].binding;
		write.descriptorCount = buffer_count;
		write.descriptorType  = vk::DescriptorType::eStorageBuffer;
		write.pBufferInfo     = buffer_infos.data();
		gpu.device.updateDescriptorSets(1, &write, 0, nullptr);
	}

	vk::CommandPoolCreateInfo command_pool_info;
	command_pool_info.queueFamilyIndex = gpu.family;
	const auto command_pool            = Checked(gpu.device.createCommandPool(command_pool_info), "command pool creation");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = command_pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command;
	Checked(gpu.device.allocateCommandBuffers(&command_info, &command), "command buffer allocation");
	vk::CommandBufferBeginInfo begin;
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	Checked(command.begin(begin), "command buffer begin");
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &set, 0, nullptr);
	command.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	command.dispatch(1, 1, 1);
	vk::MemoryBarrier barrier;
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0,
	                        nullptr, 0, nullptr);
	Checked(command.end(), "command buffer end");
	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Checked(gpu.queue.submit(1, &submit, nullptr), "submission");
	Checked(gpu.queue.waitIdle(), "queue wait");

	for (uint32_t index = 0; index < buffer_count; index++) {
		if (written[index]) std::memcpy(targets[index], buffers[index].mapped, buffers[index].bytes);
		DestroyHostBuffer(gpu, buffers[index]);
	}
	gpu.device.destroyCommandPool(command_pool);
	gpu.device.destroyDescriptorPool(pool);
	gpu.device.destroyPipeline(pipeline);
	gpu.device.destroyShaderModule(module);
	gpu.device.destroyPipelineLayout(pipeline_layout);
	gpu.device.destroyDescriptorSetLayout(set_layout);
	return true;
}

std::array<uint32_t, UserDataDwords> TwoBuffers(const std::array<uint32_t, 4>& input, const std::array<uint32_t, 4>& output) {
	std::array<uint32_t, UserDataDwords> user {};
	std::ranges::copy(input, user.begin());
	std::ranges::copy(output, user.begin() + 4);
	return user;
}

constexpr uint32_t Threads = 64u;
constexpr uint32_t Inputs  = 4u;
constexpr uint32_t Results = 16u;
constexpr uint32_t StrideWord3 = 0x11016facu;

alignas(256) std::array<uint32_t, Threads * Inputs>  g_input {};
alignas(256) std::array<uint32_t, Threads * Results> g_output {};

void FillInput(uint64_t state, std::span<const std::array<uint32_t, 4>> edges, bool mask_odd) {
	g_input.fill(0u);
	for (uint32_t tid = 0; tid < Threads; ++tid) {
		auto* words = &g_input[tid * Inputs];
		for (uint32_t j = 0; j < 3u; ++j) {
			state    = state * 6364136223846793005ull + 1442695040888963407ull;
			words[j] = tid < edges.size() ? edges[tid][j] : static_cast<uint32_t>(state >> 32u);
		}
		if (mask_odd && tid >= edges.size() && (tid & 1u) != 0u) {
			words[1] &= (tid & 2u) != 0u ? 0x00ffff00u : 0xff0000ffu;
		}
	}
}

bool RunInputOutput(const Gpu& gpu, const char* name, std::span<const uint32_t> code, uint32_t threads) {
	g_output.fill(0xdeadbeefu);
	Dispatch dispatch;
	dispatch.name      = name;
	dispatch.code      = code;
	dispatch.threads   = threads;
	dispatch.user_data = TwoBuffers(BufferDescriptor(g_input.data(), threads * Inputs, 4u, StrideWord3),
	                                BufferDescriptor(g_output.data(), threads * Results, 4u, StrideWord3));
	return Run(gpu, dispatch);
}

void CheckResult(const char* test, const char* what, uint32_t tid, const uint32_t* in, uint32_t actual, uint32_t expected) {
	Check(actual == expected, std::string(test) + ": " + what + ": thread " + std::to_string(tid) + " (" + Hex(in[0]) + ", " +
	                              Hex(in[1]) + ", " + Hex(in[2]) + ") is " + Hex(actual) + ", expected " + Hex(expected));
}

constexpr uint32_t SadLiteral = 0x8000ffffu;
alignas(256) constexpr std::array<uint32_t, 51> SadCode {
    0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601,
    0xbf8c3f70, 0xd55a000a, 0x041a0b04, 0xd55b000b, 0x041a0b04, 0xd55c000c, 0x041a0b04, 0xd571000d,
    0x041a0b04, 0xd55d000e, 0x041a0b04, 0xd55a000f, 0x04190104, 0xd55a0010, 0x041a0ac1, 0xd55b0011,
    0x042a0905, 0xd55c0012, 0x0419ff04, SadLiteral, 0xd5710013, 0x041a0905, 0xe0702000, 0x80010a03,
    0xe0702004, 0x80010b03, 0xe0702008, 0x80010c03, 0xe070200c, 0x80010d03, 0xe0702010, 0x80010e03,
    0xe0702014, 0x80010f03, 0xe0702018, 0x80011003, 0xe070201c, 0x80011103, 0xe0702020, 0x80011203,
    0xe0702024, 0x80011303, 0xbf810000,
};

constexpr std::array<std::array<uint32_t, 4>, 12> SadEdges {{
    {0xffffffffu, 0u, 0u, 0u}, {0u, 0xffffffffu, 0xffffffffu, 0u}, {0xff00ff00u, 0x00ff00ffu, 0xfffffc04u, 0u},
    {0x80000000u, 0x7fffffffu, 1u, 0u}, {0x7f80ff00u, 0x807f00ffu, 0xffff0000u, 0u}, {0x01020304u, 0x04030201u, 0x10u, 0u},
    {0x12345678u, 0x12345678u, 0x9abcdef0u, 0u}, {0x80808080u, 0x7f7f7f7fu, 0u, 0u}, {0x8000ffffu, 0x7fff0000u, 0xfffe0002u, 0u},
    {0x00ff0001u, 0x000000ffu, 0x7fffffffu, 0u}, {0xffffffffu, 0x00ff0000u, 0x03fcu, 0u}, {0u, 0u, 0u, 0u},
}};

uint32_t Sad(uint32_t lhs, uint32_t rhs, uint32_t field_bits, bool masked) {
	const uint32_t field_mask = (1u << field_bits) - 1u;
	uint32_t       sum        = 0u;
	for (uint32_t offset = 0; offset < 32u; offset += field_bits) {
		const uint32_t left  = (lhs >> offset) & field_mask;
		const uint32_t right = (rhs >> offset) & field_mask;
		if (masked && right == 0u) continue;
		sum += left > right ? left - right : right - left;
	}
	return sum;
}

void SumOfAbsoluteDifferences(const Gpu& gpu) {
	constexpr const char* Test = "sum of absolute differences";
	FillInput(0x9e3779b97f4a7c15ull, SadEdges, true);
	if (!RunInputOutput(gpu, Test, SadCode, Threads)) return;
	constexpr std::array<const char*, 10> Names {
	    "v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_msad_u8", "v_sad_u32", "v_sad_u8 with a zero reference",
	    "v_sad_u8 with an inline constant", "v_sad_hi_u8 accumulating into a v_sad_u8 result", "v_sad_u16 with a literal",
	    "v_msad_u8 with swapped sources",
	};
	for (uint32_t tid = 0; tid < Threads; ++tid) {
		const auto*    in    = &g_input[tid * Inputs];
		const uint32_t a     = in[0];
		const uint32_t b     = in[1];
		const uint32_t c     = in[2];
		const uint32_t bytes = Sad(a, b, 8u, false) + c;
		const std::array<uint32_t, 10> expected {
		    bytes,
		    (Sad(a, b, 8u, false) << 16u) + c,
		    Sad(a, b, 16u, false) + c,
		    Sad(a, b, 8u, true) + c,
		    (a > b ? a - b : b - a) + c,
		    Sad(a, 0u, 8u, false) + c,
		    Sad(0xffffffffu, b, 8u, false) + c,
		    (Sad(b, a, 8u, false) << 16u) + bytes,
		    Sad(a, SadLiteral, 16u, false) + c,
		    Sad(b, a, 8u, true) + c,
		};
		for (uint32_t j = 0; j < expected.size(); ++j) {
			CheckResult(Test, Names[j], tid, in, g_output[tid * Results + j], expected[j]);
		}
	}
}

constexpr uint32_t LerpLiteral = 0x01fe01feu;
alignas(256) constexpr std::array<uint32_t, 44> LerpCode {
    0x34020082, 0x34060084, 0xe0302000, 0x80000401, 0xe0302004, 0x80000501, 0xe0302008, 0x80000601,
    0xbf8c3f70, 0xd54d000a, 0x041a0b04, 0xd54d000b, 0x041a0905, 0xd54d000c, 0x02020b04, 0xd54d000d,
    0x03060b04, 0xd54d000e, 0x03fe0b04, LerpLiteral, 0xd54d000f, 0x041a0ac1, 0xd54d0010, 0x041a0904,
    0x7e220304, 0xd54d0011, 0x041a0b11, 0xe0702000, 0x80010a03, 0xe0702004, 0x80010b03, 0xe0702008,
    0x80010c03, 0xe070200c, 0x80010d03, 0xe0702010, 0x80010e03, 0xe0702014, 0x80010f03, 0xe0702018,
    0x80011003, 0xe070201c, 0x80011103, 0xbf810000,
};

constexpr std::array<std::array<uint32_t, 4>, 12> LerpEdges {{
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0u, 0x01010101u, 0x80808080u},
    {0xffffffffu, 0u, 0u, 0x7f7f7f7fu}, {0u, 0u, 0xffffffffu, 0u}, {0x01010101u, 0u, 0x01010101u, 0x01010101u},
    {0x01010101u, 0u, 0xfefefefeu, 0u}, {0xff00ff00u, 0x00ff00ffu, 0x01000100u, 0x807f807fu},
    {0x80808080u, 0x7f7f7f7fu, 0x00010001u, 0x7f807f80u}, {0x000000ffu, 0x0000ff00u, 0x00010000u, 0x00007f7fu},
    {0x12345678u, 0x9abcdef0u, 0xffffffffu, 0x56789ab4u}, {0xfe01fe01u, 0x01fe01feu, 0x80808080u, 0x7f7f7f7fu},
    {0u, 0u, 0u, 0u},
}};

uint32_t Lerp(uint32_t lhs, uint32_t rhs, uint32_t rounding) {
	uint32_t result = 0u;
	for (uint32_t offset = 0; offset < 32u; offset += 8u) {
		const uint32_t sum = ((lhs >> offset) & 0xffu) + ((rhs >> offset) & 0xffu) + ((rounding >> offset) & 1u);
		result |= (sum >> 1u) << offset;
	}
	return result;
}

void ByteAverage(const Gpu& gpu) {
	constexpr const char* Test = "byte average";
	FillInput(0xd1b54a32d192ed03ull, LerpEdges, false);
	if (!RunInputOutput(gpu, Test, LerpCode, Threads)) return;
	constexpr std::array<const char*, 8> Names {
	    "v_lerp_u8", "v_lerp_u8 with swapped sources", "v_lerp_u8 truncating", "v_lerp_u8 rounding every byte",
	    "v_lerp_u8 with a literal round mode", "v_lerp_u8 with an inline constant", "v_lerp_u8 of a value with itself",
	    "v_lerp_u8 into its first source",
	};
	for (uint32_t tid = 0; tid < Threads; ++tid) {
		const auto*    in = &g_input[tid * Inputs];
		const uint32_t a  = in[0];
		const uint32_t b  = in[1];
		const uint32_t c  = in[2];
		const std::array<uint32_t, 8> expected {
		    tid < LerpEdges.size() ? LerpEdges[tid][3] : Lerp(a, b, c),
		    Lerp(a, b, c),
		    Lerp(a, b, 0u),
		    Lerp(a, b, 0xffffffffu),
		    Lerp(a, b, LerpLiteral),
		    Lerp(0xffffffffu, b, c),
		    a,
		    Lerp(a, b, c),
		};
		for (uint32_t j = 0; j < expected.size(); ++j) {
			CheckResult(Test, Names[j], tid, in, g_output[tid * Results + j], expected[j]);
		}
	}
}

}

int main() {
	const char* why = nullptr;
	auto        gpu = OpenGpu(why);
	if (!gpu) return NoDevice(why);
	SumOfAbsoluteDifferences(*gpu);
	ByteAverage(*gpu);
	const auto name = gpu->name;
	CloseGpu(*gpu);
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u checks failed on %s\n", Name, g_failures, name.c_str());
		return 1;
	}
	std::printf("%s: ok on %s\n", Name, name.c_str());
	return 0;
}
