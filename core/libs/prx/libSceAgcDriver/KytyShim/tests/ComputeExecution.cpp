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

constexpr uint32_t QuadThreads = 32u;
alignas(256) constexpr std::array<uint32_t, 50> QuadSadCode {
    0x34020084, 0x34060086, 0xe0381000, 0x80000401, 0xbf8c3f70, 0xd572000a, 0x041a0d04, 0xd573000c,
    0x041a0d04, 0xd575000e, 0x04120d04, 0xd5720012, 0x02020906, 0xd5750014, 0x04120b06, 0xd5730018,
    0x04120b06, 0xe0701000, 0x80010a03, 0xe0701004, 0x80010b03, 0xe0701008, 0x80010c03, 0xe070100c,
    0x80010d03, 0xe0701010, 0x80010e03, 0xe0701014, 0x80010f03, 0xe0701018, 0x80011003, 0xe070101c,
    0x80011103, 0xe0701020, 0x80011203, 0xe0701024, 0x80011303, 0xe0701028, 0x80011403, 0xe070102c,
    0x80011503, 0xe0701030, 0x80011603, 0xe0701034, 0x80011703, 0xe0701038, 0x80011803, 0xe070103c,
    0x80011903, 0xbf810000,
};

constexpr uint32_t QuadSadRows[QuadThreads][4] = {
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0xffffffffu, 0xffffffffu, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
    {0x01020304u, 0x05060708u, 0x00ff00ffu, 0xfffefffdu},
    {0xff00ff00u, 0x00ff00ffu, 0x00000000u, 0xffffffffu},
    {0x12345678u, 0x9abcdef0u, 0x00010203u, 0xffff0000u},
    {0xffffffffu, 0x00000000u, 0x00ff0000u, 0xfffffff0u},
    {0x80808080u, 0x7f7f7f7fu, 0x01000001u, 0x00010000u},
    {0xfffffff0u, 0xfffffff0u, 0xffffffffu, 0xffff0001u},
    {0xecc3f80cu, 0xc7859faeu, 0xf2d7d40fu, 0x4a37fa2du},
    {0xe47682e6u, 0xd46375dcu, 0x156393d8u, 0x045f21dau},
    {0x978f18a7u, 0x4e86c4fau, 0x6c7ab5c9u, 0x611244c0u},
    {0x87b3d90eu, 0x5bab1eecu, 0x215b8892u, 0xb9d8249eu},
    {0x3126b9c3u, 0x44732494u, 0x71cf92e3u, 0x039a7b88u},
    {0x389605feu, 0x9ec2d776u, 0x747d0a2bu, 0xe6d30f0au},
    {0x065720ceu, 0x1b1c3f27u, 0x15f54a5cu, 0xfb34ccc5u},
    {0xb5c318e9u, 0x7f8cb6d1u, 0x6bd6eed6u, 0x05032a7eu},
    {0x833bfa03u, 0xa0aeb4e5u, 0xbbf5204au, 0x6e6944d3u},
    {0x5be72f6eu, 0xc55176d5u, 0x0ebbe4d0u, 0x34a3f451u},
    {0xc8796571u, 0x090a0e01u, 0x96d0cbffu, 0x5e7db530u},
    {0xa86cf7b4u, 0x2c03a513u, 0x9ef94bdau, 0x374ebe5au},
    {0xf8821a96u, 0xbc36c297u, 0x5cdff86du, 0x90c68e93u},
    {0xc2816017u, 0xdc3260fdu, 0x5e620fe9u, 0xf7f3a683u},
    {0x9271a86fu, 0x4801920eu, 0xa918a1dcu, 0xf9810e12u},
    {0x507bef95u, 0x92af2f03u, 0x15f2cd71u, 0xd0745cc1u},
    {0x795c6ad3u, 0x2d794610u, 0xb7c93640u, 0x7bab67bcu},
    {0x97323268u, 0xca9f211eu, 0x728c2f7bu, 0xacfc4403u},
    {0xa472ea5fu, 0x29e0b1f5u, 0x2e99777fu, 0xcc8ba3e3u},
    {0x2db4cb47u, 0x8899230bu, 0x52cb26d8u, 0xac61e3a6u},
    {0xd0dd15d6u, 0x85232575u, 0x32135b34u, 0x951179a0u},
    {0x1d42d810u, 0x15136429u, 0x193e1ff7u, 0x6d17abfbu}
};

constexpr uint32_t QuadSadExpected[QuadThreads][16] = {
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x03fc03fcu, 0x03fc03fcu, 0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu, 0x00000000u, 0x00000000u, 0x03fc03fcu, 0x03fc03fcu, 0x000003fbu, 0x000003fbu, 0x000003fcu, 0x000003fcu, 0x03fb03fbu, 0x03fb03fbu},
    {0x03fb03fbu, 0x03fb03fbu, 0x03fb03fbu, 0x03fb03fbu, 0x000003fcu, 0x000003fcu, 0x000003fbu, 0x000003fbu, 0x03fc03fcu, 0x03fc03fcu, 0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu, 0x00000000u, 0x00000000u},
    {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
    {0x030302fbu, 0x020201f9u, 0x02f902f7u, 0x01f401f1u, 0x010204fcu, 0x05060902u, 0x00ff02f3u, 0xffff01f3u, 0x01fe01fcu, 0x02f802f7u, 0x01020500u, 0x05060906u, 0x00ff03eeu, 0xffff02edu, 0x03000500u, 0x07f609f7u},
    {0x02fd01feu, 0x02fc01fdu, 0x00000000u, 0xffffffffu, 0xff00ff00u, 0x00ff00ffu, 0x00000000u, 0xffffffffu, 0x00ff01feu, 0x00ff01feu, 0xff0100feu, 0x00ff02fdu, 0x000000ffu, 0x000000feu, 0x00fe00feu, 0x01fe01feu},
    {0x01870311u, 0x0295020eu, 0x009702ffu, 0x01d90130u, 0x12345774u, 0x9abcdf86u, 0x00010333u, 0xffff01dau, 0x0111010eu, 0x01ef0113u, 0x12345996u, 0x9abce211u, 0x00010526u, 0xffff02efu, 0x15555996u, 0x9dabe213u},
    {0x02fd02fdu, 0x01fd02edu, 0x00ff0000u, 0x00fe00efu, 0xffffffffu, 0x00000000u, 0x00ff00ffu, 0x000000efu, 0x020d02fdu, 0x010e010eu, 0xffffffffu, 0x00000000u, 0x00ff0000u, 0xfffffff0u, 0xffffffffu, 0x00000000u},
    {0x02fd01ffu, 0x01fc01fcu, 0x01fd00ffu, 0x00fe00fdu, 0x8080817eu, 0x7f7f807cu, 0x010000feu, 0x000100fdu, 0x01ff01feu, 0x01fe01ffu, 0x8080827au, 0x7f7f817au, 0x010001fcu, 0x000101fau, 0x827b827au, 0x8179817au},
    {0x000e000eu, 0x000e0010u, 0x000e000eu, 0x000e0010u, 0xffffffffu, 0xffffffffu, 0x0000000eu, 0xffff0010u, 0x010d000fu, 0x020c020cu, 0xffffffffu, 0x000000fdu, 0x0000020bu, 0xffff020du, 0x010cffffu, 0x020b01fcu},
    {0xf42ad450u, 0x4bdffb75u, 0xf42ad450u, 0x4bdffb75u, 0xecc3f84du, 0xc785a101u, 0xf2d7d557u, 0x4a37fbd5u, 0x01d70041u, 0x029d0175u, 0xecc3f95du, 0xc785a113u, 0xf2d7d516u, 0x4a37fbe8u, 0xee28f95du, 0xc940a0b5u},
    {0x171e94d9u, 0x05142366u, 0x171e94d9u, 0x05142366u, 0xe47683e7u, 0xd4637797u, 0x15639564u, 0x045f228fu, 0x00dd0101u, 0x02030217u, 0xe47683c7u, 0xd463768bu, 0x156395dbu, 0x045f23bdu, 0xe52583c7u, 0xd64677dfu},
    {0x6dfcb6c8u, 0x61ed45f0u, 0x6dfcb6c8u, 0x61ed45f0u, 0x978f19a6u, 0x4e86c67cu, 0x6c7ab6f9u, 0x6112459bu, 0x00bc00ffu, 0x01b30105u, 0x978f1911u, 0x4e86c615u, 0x6c7ab6e5u, 0x611245d0u, 0x98aa1911u, 0x4f96c616u},
    {0x22c48a25u, 0xbb0e2554u, 0x22c48a25u, 0xbb0e2554u, 0x87b3daa1u, 0x5bab2055u, 0x215b8948u, 0xb9d825d4u, 0x01a10193u, 0x012e017du, 0x87b3da5cu, 0x5bab205au, 0x215b896au, 0xb9d826edu, 0x8921da5cu, 0x5dfa1fc4u},
    {0x73269413u, 0x04fb7d2eu, 0x73269413u, 0x04fb7d2eu, 0x3126baf3u, 0x447325ebu, 0x71cf9489u, 0x039a7ce9u, 0x00e90130u, 0x01410100u, 0x3126bb09u, 0x44732587u, 0x71cf93b7u, 0x039a7c6du, 0x3219bb09u, 0x45582568u},
    {0x75760b58u, 0xe7f4100du, 0x75760b58u, 0xe7f4100du, 0x3896072bu, 0x9ec2d86fu, 0x747d0b2eu, 0xe6d3102bu, 0x01bc012du, 0x01b101a5u, 0x38960785u, 0x9ec2d91eu, 0x747d0bdcu, 0xe6d310c1u, 0x3a3e0785u, 0xa079d927u},
    {0x173f4ba5u, 0xfc6ace06u, 0x173f4ba5u, 0xfc6ace06u, 0x06572217u, 0x1b1c4071u, 0x15f54b9du, 0xfb34cdfbu, 0x025a0149u, 0x02010166u, 0x065721edu, 0x1b1c40b1u, 0x15f54caeu, 0xfb34ce26u, 0x07e121edu, 0x1c7d4179u},
    {0x6d46f01cu, 0x05822b1au, 0x6d46f01cu, 0x05822b1au, 0xb5c31a2fu, 0x7f8cb841u, 0x6bd6ef72u, 0x05032afdu, 0x01520146u, 0x022f0136u, 0xb5c31984u, 0x7f8cb730u, 0x6bd6ef89u, 0x05032bfau, 0xb6221984u, 0x8108b784u},
    {0xbd5c225du, 0x6fb5455cu, 0xbd5c225du, 0x6fb5455cu, 0x833bfc16u, 0xa0aeb64cu, 0xbbf520d3u, 0x6e69461fu, 0x00f20213u, 0x01020208u, 0x833bfb94u, 0xa0aeb62bu, 0xbbf520e2u, 0x6e6945bdu, 0x8481fb94u, 0xa198b57du},
    {0x1086e660u, 0x35aff573u, 0x1086e660u, 0x35aff573u, 0x5be730feu, 0xc55178a0u, 0x0ebbe5f2u, 0x34a3f55du, 0x01e50190u, 0x00d7019du, 0x5be73102u, 0xc55177e0u, 0x0ebbe581u, 0x34a3f602u, 0x5cf23102u, 0xc7027786u},
    {0x9859cd7cu, 0x60ccb710u, 0x9859cd7cu, 0x60ccb710u, 0xc87966eeu, 0x090a0f8au, 0x96d0cddfu, 0x5e7db77fu, 0x017a017du, 0x00e100ecu, 0xc879687fu, 0x090a1040u, 0x96d0ce28u, 0x5e7db706u, 0xcab8687fu, 0x0ae0102au},
    {0xa0134d43u, 0x38a7c012u, 0xa0134d43u, 0x38a7c012u, 0xa86cf91du, 0x2c03a62du, 0x9ef94d92u, 0x374ebfb3u, 0x00eb0169u, 0x015f00c6u, 0xa86cfa3du, 0x2c03a668u, 0x9ef94db0u, 0x374ec00du, 0xa9c1fa3du, 0x2db6a6e9u},
    {0x5dfcfa6du, 0x91f58f56u, 0x5dfcfa6du, 0x91f58f56u, 0xf8821c96u, 0xbc36c3b4u, 0x5cdff930u, 0x90c68fc2u, 0x01b20200u, 0x00f10106u, 0xf8821bffu, 0xbc36c364u, 0x5cdff9a6u, 0x90c68f5fu, 0xf94f1bffu, 0xbd02c3d0u},
    {0x605c118fu, 0xf936a83bu, 0x605c118fu, 0xf936a83bu, 0xc28161bdu, 0xdc3262f7u, 0x5e6211a1u, 0xf7f3a7c6u, 0x006c01a6u, 0x00c0006bu, 0xc281612au, 0xdc326272u, 0x5e62110du, 0xf7f3a7d0u, 0xc3f6612au, 0xdd7f6221u},
    {0xaa91a2c0u, 0xfb800eadu, 0xaa91a2c0u, 0xfb800eadu, 0x9271a953u, 0x48019387u, 0xa918a277u, 0xf9811011u, 0x017a00e4u, 0x0144013bu, 0x9271a9c4u, 0x480193f9u, 0xa918a248u, 0xf9810f73u, 0x945ca9c4u, 0x4962927au},
    {0x1776ce69u, 0xd2bc5e51u, 0x1776ce69u, 0xd2bc5e51u, 0x507bf08du, 0x92af3087u, 0x15f2cf01u, 0xd0745f09u, 0x011200f8u, 0x00f10189u, 0x507bf161u, 0x92af3159u, 0x15f2cec2u, 0xd0745dd6u, 0x52d1f161u, 0x93c43054u},
    {0xb91037b2u, 0x7ccb6945u, 0xb91037b2u, 0x7ccb6945u, 0x795c6c45u, 0x2d794757u, 0xb7c937c9u, 0x7bab68dcu, 0x019a0172u, 0x00ab00c9u, 0x795c6bedu, 0x2d794786u, 0xb7c937e7u, 0x7bab6969u, 0x7ad26bedu, 0x2f2647b7u},
    {0x73373010u, 0xadc14573u, 0x73373010u, 0xadc14573u, 0x973232fdu, 0xca9f21c9u, 0x728c30ebu, 0xacfc44c8u, 0x01670095u, 0x00b000e6u, 0x9732333eu, 0xca9f228eu, 0x728c315cu, 0xacfc4502u, 0x98a2333eu, 0xcb9e22ffu},
    {0x2fdb78afu, 0xcdf8a4fcu, 0x2fdb78afu, 0xcdf8a4fcu, 0xa472eb8fu, 0x29e0b337u, 0x2e997898u, 0xcc8ba550u, 0x00ec0130u, 0x00820168u, 0xa472eb5bu, 0x29e0b3f7u, 0x2e9978dbu, 0xcc8ba57bu, 0xa674eb5bu, 0x2b78b351u},
    {0x544b284au, 0xae16e4c0u, 0x544b284au, 0xae16e4c0u, 0x2db4ccb9u, 0x8899248bu, 0x52cb27f2u, 0xac61e55bu, 0x00fc0172u, 0x009301c1u, 0x2db4cc7fu, 0x88992433u, 0x52cb282fu, 0xac61e4e1u, 0x2edccc7fu, 0x89d42462u},
    {0x33b45d84u, 0x95e87b2du, 0x33b45d84u, 0x95e87b2du, 0xd0dd1826u, 0x85232716u, 0x32135cc1u, 0x95117a77u, 0x01580250u, 0x02520174u, 0xd0dd16b0u, 0x852325cbu, 0x32135c2cu, 0x95117b28u, 0xd13316b0u, 0x86ab266du},
    {0x19b1219fu, 0x6e27ad12u, 0x19b1219fu, 0x6e27ad12u, 0x1d42d9b8u, 0x1513649cu, 0x193e210eu, 0x6d17ad0bu, 0x01b001a8u, 0x009b0234u, 0x1d42d952u, 0x15136545u, 0x193e21d5u, 0x6d17ad3cu, 0x1e5ed952u, 0x16546607u}
};

void QuadSumOfAbsoluteDifferences(const Gpu& gpu) {
	constexpr const char* Test = "quad sum of absolute differences";
	g_input.fill(0u);
	for (uint32_t tid = 0; tid < QuadThreads; ++tid) std::ranges::copy(QuadSadRows[tid], &g_input[tid * Inputs]);
	g_output.fill(0xdeadbeefu);
	Dispatch dispatch;
	dispatch.name      = Test;
	dispatch.code      = QuadSadCode;
	dispatch.threads   = QuadThreads;
	dispatch.user_data = TwoBuffers(BufferDescriptor(g_input.data(), QuadThreads * Inputs * 4u, 0u, 0x31016facu),
	                                BufferDescriptor(g_output.data(), QuadThreads * Results * 4u, 0u, 0x31016facu));
	if (!Run(gpu, dispatch)) return;
	constexpr std::array<const char*, 16> Names {
	    "v_qsad_pk_u16_u8 low", "v_qsad_pk_u16_u8 high", "v_mqsad_pk_u16_u8 low", "v_mqsad_pk_u16_u8 high",
	    "v_mqsad_u32_u8 0", "v_mqsad_u32_u8 1", "v_mqsad_u32_u8 2", "v_mqsad_u32_u8 3",
	    "v_qsad_pk_u16_u8 with a constant low", "v_qsad_pk_u16_u8 with a constant high", "v_mqsad_u32_u8 b0",
	    "v_mqsad_u32_u8 b1", "v_mqsad_u32_u8 b2", "v_mqsad_u32_u8 b3", "v_mqsad_pk_u16_u8 b low", "v_mqsad_pk_u16_u8 b high",
	};
	for (uint32_t tid = 0; tid < QuadThreads; ++tid) {
		const auto* in = &g_input[tid * Inputs];
		for (uint32_t j = 0; j < Names.size(); ++j) {
			CheckResult(Test, Names[j], tid, in, g_output[tid * Results + j], QuadSadExpected[tid][j]);
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
	QuadSumOfAbsoluteDifferences(*gpu);
	const auto name = gpu->name;
	CloseGpu(*gpu);
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u checks failed on %s\n", Name, g_failures, name.c_str());
		return 1;
	}
	std::printf("%s: ok on %s\n", Name, name.c_str());
	return 0;
}
