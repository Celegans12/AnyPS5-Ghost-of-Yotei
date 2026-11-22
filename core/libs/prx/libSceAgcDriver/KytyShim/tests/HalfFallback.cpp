#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace Emit = Recompiler::Spirv::Emitter;
constexpr const char* Name = "kyty_shim_half_fallback";
constexpr int SkippedExit = 77;
constexpr uint32_t Samples = 131072;
constexpr uint32_t Inputs = 4;
constexpr uint32_t Results = 24;
constexpr uint32_t Stride = Inputs + Results;

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

	const auto lower = [](std::string value) {
		std::ranges::transform(value, value.begin(), [](unsigned char letter) { return static_cast<char>(std::tolower(letter)); });
		return value;
	};
	const char* requested = std::getenv("ANYPS5_GPU");
	const auto filter = lower(requested != nullptr ? requested : "");
	for (const auto candidate: Checked(gpu.instance.enumeratePhysicalDevices(), "device enumeration")) {
		const auto properties = candidate.getProperties();
		if (properties.apiVersion < VK_API_VERSION_1_3 || lower(properties.deviceName.data()).find(filter) == std::string::npos) continue;
		const auto families = candidate.getQueueFamilyProperties();
		const auto compute  = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);
		});
		if (compute == families.end()) continue;
		const bool software = properties.deviceType == vk::PhysicalDeviceType::eCpu;
		if (gpu.physical && software) continue;
		gpu.physical = candidate;
		gpu.family   = static_cast<uint32_t>(compute - families.begin());
		if (!software || !filter.empty()) break;
	}
	if (!gpu.physical) {
		gpu.instance.destroy();
		why = "no matching Vulkan 1.3 device with a compute queue";
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
	features12.shaderFloat16 = VK_FALSE;
	features.features.shaderInt16 = VK_FALSE;
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

	vk::PhysicalDeviceDriverProperties   driver;
	vk::PhysicalDeviceProperties2        properties;
	properties.pNext = &driver;
	gpu.physical.getProperties2(&properties);
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

uint32_t DecodeBits(uint32_t half) {
	const auto sign = (half & 0x8000u) << 16u;
	const auto exponent = (half >> 10u) & 31u;
	const auto mantissa = half & 1023u;
	if (exponent == 31u) return sign | 0x7f800000u | (mantissa << 13u);
	if (exponent != 0u) return sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
	if (mantissa == 0u) return sign;
	const auto highest = std::bit_width(mantissa) - 1u;
	return sign | ((highest + 103u) << 23u) | ((mantissa << (23u - highest)) & 0x7fffffu);
}

uint32_t EncodeBits(uint32_t bits) {
	const uint32_t sign = (bits >> 16u) & 0x8000u;
	const uint32_t magnitude = bits & 0x7fffffffu;
	if (magnitude >= 0x7f800000u) return sign | (magnitude == 0x7f800000u ? 0x7c00u : 0x7e00u);
	if (magnitude < 0x33000000u) return sign;
	if (magnitude >= 0x477ff000u) return sign | 0x7c00u;
	const auto exponent = magnitude >> 23u;
	const auto significand = (magnitude & 0x7fffffu) | 0x800000u;
	const auto shift = exponent < 113u ? 126u - exponent : 13u;
	const auto remainder = significand & ((1u << shift) - 1u);
	auto rounded = significand >> shift;
	const auto halfway = 1u << (shift - 1u);
	rounded += remainder > halfway || (remainder == halfway && (rounded & 1u));
	return sign | (exponent < 113u ? rounded : ((exponent - 113u) << 10u) + rounded);
}

bool IsNan(uint32_t half) {
	return (half & 0x7fffu) > 0x7c00u;
}

__int128 HalfInteger(uint32_t half) {
	const auto exponent = (half >> 10u) & 31u;
	__int128 value = ((half & 1023u) | (exponent != 0u ? 1024u : 0u));
	value <<= std::max(exponent, 1u) - 1u;
	return (half & 0x8000u) != 0u ? -value : value;
}

uint32_t RoundInteger(__int128 value, uint32_t zero_sign) {
	if (value == 0) return zero_sign;
	const auto sign = value < 0 ? 0x8000u : 0u;
	const auto magnitude = static_cast<unsigned __int128>(value < 0 ? -value : value);
	auto top = magnitude;
	uint32_t highest = 0;
	while ((top >>= 1u) != 0) ++highest;
	auto exponent = highest > 33u ? highest - 33u : 0u;
	const auto shift = exponent == 0u ? 24u : exponent + 23u;
	const auto halfway = static_cast<unsigned __int128>(1) << (shift - 1u);
	const auto remainder = magnitude & ((halfway << 1u) - 1u);
	auto rounded = static_cast<uint32_t>(magnitude >> shift);
	rounded += remainder > halfway || (remainder == halfway && (rounded & 1u));
	if (exponent == 0u) return sign | rounded;
	if (rounded == 2048u) {
		++exponent;
		rounded = 1024u;
	}
	return sign | (exponent >= 31u ? 0x7c00u : (exponent << 10u) | (rounded - 1024u));
}

uint32_t Fma(uint32_t a, uint32_t b, uint32_t c) {
	if ((a & 0x7c00u) == 0x7c00u || (b & 0x7c00u) == 0x7c00u || (c & 0x7c00u) == 0x7c00u) {
		const auto value = std::fma(std::bit_cast<float>(DecodeBits(a)), std::bit_cast<float>(DecodeBits(b)),
		                            std::bit_cast<float>(DecodeBits(c)));
		return EncodeBits(std::bit_cast<uint32_t>(value));
	}
	const auto product = HalfInteger(a) * HalfInteger(b);
	const auto addend = HalfInteger(c) * (static_cast<__int128>(1) << 24u);
	const auto zero_sign = product == 0 && addend == 0 ? ((a ^ b) & c & 0x8000u) : 0u;
	return RoundInteger(product + addend, zero_sign);
}

uint32_t MinMax(uint32_t a, uint32_t b, bool maximum) {
	if (IsNan(b)) return a;
	if (IsNan(a)) return b;
	const auto lhs = std::bit_cast<float>(DecodeBits(a));
	const auto rhs = std::bit_cast<float>(DecodeBits(b));
	if (lhs == rhs) return maximum ? a & b : a | b;
	return (maximum ? lhs > rhs : lhs < rhs) ? a : b;
}

std::vector<uint32_t> BuildModule(bool native) {
	Recompiler::IR::Program program;
	program.stage = ShaderType::Compute;
	const ShaderComputeInputInfo compute {};
	Recompiler::Spirv::Profile profile;
	profile.native_float16 = native;
	Emit::EmitterState state(program, {.compute = &compute}, profile);
	auto& builder = state.builder;
	builder.RequireCapability(spv::CapabilityShader);
	builder.RequireCapability(spv::CapabilitySignedZeroInfNanPreserve);
	builder.RequireExtension("SPV_KHR_float_controls");
	if (native) builder.RequireCapability(spv::CapabilityFloat16);
	builder.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
	const auto function = builder.AllocateId();
	const auto global_type = Emit::TypeU32Vector(state, 3);
	const auto input_pointer = builder.Type(spv::OpTypePointer, spv::StorageClassInput, global_type);
	const auto global_id = builder.DefineGlobalVariable(input_pointer, spv::StorageClassInput);
	builder.AddAnnotation(spv::OpDecorate, global_id, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
	const auto array_type = builder.Type(spv::OpTypeRuntimeArray, Emit::TypeU32(state));
	builder.AddAnnotation(spv::OpDecorate, array_type, spv::DecorationArrayStride, 4u);
	const auto buffer_type = builder.Type(spv::OpTypeStruct, array_type);
	builder.AddAnnotation(spv::OpDecorate, buffer_type, spv::DecorationBlock);
	builder.AddAnnotation(spv::OpMemberDecorate, buffer_type, 0u, spv::DecorationOffset, 0u);
	const auto buffer_pointer = builder.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, buffer_type);
	const auto buffer = builder.DefineGlobalVariable(buffer_pointer, spv::StorageClassStorageBuffer);
	builder.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationDescriptorSet, 0u);
	builder.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationBinding, 0u);
	const auto word_pointer = builder.Type(spv::OpTypePointer, spv::StorageClassStorageBuffer, Emit::TypeU32(state));
	builder.AddEntryPoint(spv::ExecutionModelGLCompute, function, "main", {global_id});
	builder.AddExecutionMode(function, spv::ExecutionModeLocalSize, 64u, 1u, 1u);
	builder.AddExecutionMode(function, spv::ExecutionModeSignedZeroInfNanPreserve, 32u);
	builder.AddFunction(spv::OpFunction, Emit::TypeVoid(state), function, spv::FunctionControlMaskNone, Emit::TypeFunction(state));
	builder.AddFunction(spv::OpLabel, builder.AllocateId());
	const auto invocation = Emit::Unary(state, spv::OpLoad, global_type, global_id);
	const auto index = builder.AllocateId();
	builder.AddFunction(spv::OpCompositeExtract, Emit::TypeU32(state), index, invocation, 0u);
	const auto base = Emit::Binary(state, spv::OpIMul, Emit::TypeU32(state), index, Emit::ConstantU32(state, Stride));
	const auto address = [&](uint32_t offset) {
		const auto position = Emit::Binary(state, spv::OpIAdd, Emit::TypeU32(state), base, Emit::ConstantU32(state, offset));
		const auto pointer = builder.AllocateId();
		builder.AddFunction(spv::OpAccessChain, word_pointer, pointer, buffer, Emit::ConstantU32(state, 0), position);
		return pointer;
	};
	std::array<uint32_t, Inputs> in {};
	for (uint32_t i = 0; i < Inputs; ++i) in[i] = Emit::Unary(state, spv::OpLoad, Emit::TypeU32(state), address(i));
	const Recompiler::IR::Value low(0u), high(1u);
	const auto a = Emit::EmitBitCastF16U16(state, in[0]);
	const auto b = Emit::EmitBitCastF16U16(state, in[1]);
	const auto c = Emit::EmitBitCastF16U16(state, in[2]);
	const auto to_bits = [&](uint32_t value) { return Emit::EmitBitCastU16F16(state, value); };
	const auto decoded = Emit::EmitConvertF32F16(state, a);
	const auto f32_bits = Emit::Unary(state, spv::OpBitcast, Emit::TypeF32(state), in[3]);
	const auto float_pair = builder.AllocateId();
	builder.AddFunction(spv::OpCompositeConstruct, Emit::TypeF32Vector(state, 2), float_pair, decoded, f32_bits);
	const std::array<uint32_t, Results> out {
	    Emit::Unary(state, spv::OpBitcast, Emit::TypeU32(state), decoded),
	    to_bits(Emit::EmitConvertF16F32(state, decoded)),
	    to_bits(a),
	    to_bits(Emit::EmitExtractF16Lane(state, in[0], low)),
	    to_bits(Emit::EmitExtractF16Lane(state, in[0], high)),
	    Emit::EmitInsertF16Lane(state, in[0], b, low),
	    Emit::EmitInsertF16Lane(state, in[0], b, high),
	    to_bits(Emit::EmitFPAbs16(state, a)),
	    to_bits(Emit::EmitFPNeg16(state, a)),
	    to_bits(Emit::EmitFPAdd16(state, a, b)),
	    to_bits(Emit::EmitFPSub16(state, a, b)),
	    to_bits(Emit::EmitFPMul16(state, a, b)),
	    to_bits(Emit::EmitFPFma16(state, a, b, c)),
	    to_bits(Emit::EmitFPMin16(state, a, b)),
	    to_bits(Emit::EmitFPMax16(state, a, b)),
	    Emit::EmitPackedFPAdd16x2(state, in[0], in[1]),
	    Emit::EmitPackedFPMul16x2(state, in[0], in[1]),
	    Emit::EmitPackedFPFma16x2(state, in[0], in[1], in[2]),
	    Emit::EmitPackedFPMin16x2(state, in[0], in[1]),
	    Emit::EmitPackedFPMax16x2(state, in[0], in[1]),
	    to_bits(Emit::EmitConvertF16F32(state, f32_bits)),
	    Emit::Unary(state, spv::OpBitcast, Emit::TypeU32(state), Emit::EmitConvertF32F16Lane(state, in[0], low)),
	    Emit::Unary(state, spv::OpBitcast, Emit::TypeU32(state), Emit::EmitConvertF32F16Lane(state, in[0], high)),
	    Emit::EmitPackHalf2x16(state, float_pair),
	};
	for (uint32_t i = 0; i < Results; ++i) builder.AddFunction(spv::OpStore, address(Inputs + i), out[i]);
	builder.AddFunction(spv::OpReturn);
	builder.AddFunction(spv::OpFunctionEnd);
	auto words = builder.Build();
	RequireValid(words, native ? "native half operations" : "software half operations");
	for (size_t at = 5; at < words.size(); at += words[at] >> 16u) {
		const auto opcode = static_cast<spv::Op>(words[at] & 0xffffu);
		if (!native && opcode == spv::OpTypeFloat) Check(words[at + 2] != 16u, "fallback has no Float16 type");
		if (!native && opcode == spv::OpCapability) {
			const auto capability = words[at + 1];
			Check(capability != spv::CapabilityFloat16 && capability != spv::CapabilityFloat16Buffer &&
			          capability != spv::CapabilityStorageBuffer16BitAccess && capability != spv::CapabilityUniformAndStorageBuffer16BitAccess &&
			          capability != spv::CapabilityStoragePushConstant16 && capability != spv::CapabilityStorageInputOutput16,
			      "fallback has no half capability");
		}
	}
	return words;
}
void ValidateIrHalfProgram(bool native, bool dispatcher) {
	namespace IR = Recompiler::IR;
	IR::Program program;
	program.stage = ShaderType::Compute;
	program.wave_size = 32;
	program.lane_local = true;
	program.dispatcher_fallback = dispatcher;
	program.srt_plan_complete = true;
	program.resource_tracking_complete = true;
	program.shader_info_complete = true;
	program.info.float16 = true;
	for (uint32_t i = 0; i < 4; ++i) {
		program.block_storage.push_back(std::make_unique<IR::Block>());
		auto* block = program.block_storage.back().get();
		block->id = i;
		program.blocks.push_back(block);
	}
	auto& entry = *program.blocks[0];
	auto& left = *program.blocks[1];
	auto& right = *program.blocks[2];
	auto& merge = *program.blocks[3];
	entry.AddBranch(&left);
	entry.AddBranch(&right);
	left.AddBranch(&merge);
	right.AddBranch(&merge);
	entry.terminator.kind = Recompiler::CFG::TerminatorKind::ConditionalBranch;
	entry.terminator.true_block = &left;
	entry.terminator.false_block = &right;
	entry.terminator.merge_block = &merge;
	entry.condition = IR::Value(true);
	left.terminator.kind = Recompiler::CFG::TerminatorKind::Branch;
	left.terminator.true_block = &merge;
	right.terminator.kind = Recompiler::CFG::TerminatorKind::Branch;
	right.terminator.true_block = &merge;
	auto& sum = left.AppendNewInst(IR::ValueOpcode::FPAdd16, {IR::Value::F16(0x3c00), IR::Value::F16(0x0001)});
	auto& negated = right.AppendNewInst(IR::ValueOpcode::FPNeg16, {IR::Value::F16(0x8000)});
	auto& phi = merge.AppendNewInst(IR::ValueOpcode::Phi, {}, static_cast<uint64_t>(IR::Type::F16));
	phi.AddPhiOperand(&left, IR::Value(&sum));
	phi.AddPhiOperand(&right, IR::Value(&negated));
	auto& result = merge.AppendNewInst(IR::ValueOpcode::FPAdd16, {IR::Value(&phi), IR::Value::F16(0x0400)});
	auto& bits = merge.AppendNewInst(IR::ValueOpcode::BitCastU16F16, {IR::Value(&result)});
	auto& word = merge.AppendNewInst(IR::ValueOpcode::ConvertU32U16, {IR::Value(&bits)});
	merge.AppendNewInst(IR::ValueOpcode::SelectU32, {entry.condition, IR::Value(&word), IR::Value(uint32_t {0})});
	ShaderComputeInputInfo compute;
	compute.threads_num[0] = 32;
	compute.threads_num[1] = 1;
	compute.threads_num[2] = 1;
	compute.wave_size = 32;
	compute.host_subgroup_size = 32;
	Recompiler::Spirv::Profile profile;
	profile.native_float16 = native;
	const auto words = Recompiler::Spirv::EmitProgram(program, {.compute = &compute}, 0, profile);
	RequireValid(words, std::string(native ? "native" : "software") + (dispatcher ? " dispatcher IR" : " structured IR"));
	bool half_type = false;
	bool half_capability = false;
	for (size_t at = 5; at < words.size(); at += words[at] >> 16u) {
		const auto opcode = static_cast<spv::Op>(words[at] & 0xffffu);
		half_type |= opcode == spv::OpTypeFloat && words[at + 2] == 16u;
		half_capability |= opcode == spv::OpCapability && words[at + 1] == spv::CapabilityFloat16;
	}
	Check(half_type == native && half_capability == native, "IR half types and capabilities follow the profile");
}

void Execute(const Gpu& gpu, const std::vector<uint32_t>& words, HostBuffer& buffer) {
	const vk::DescriptorSetLayoutBinding binding {0u, vk::DescriptorType::eStorageBuffer, 1u, vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo set_info;
	set_info.bindingCount = 1;
	set_info.pBindings = &binding;
	const auto set_layout = Checked(gpu.device.createDescriptorSetLayout(set_info), "set layout creation");
	vk::PipelineLayoutCreateInfo layout_info;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &set_layout;
	const auto layout = Checked(gpu.device.createPipelineLayout(layout_info), "pipeline layout creation");
	vk::ShaderModuleCreateInfo module_info;
	module_info.codeSize = words.size() * sizeof(uint32_t);
	module_info.pCode = words.data();
	const auto module = Checked(gpu.device.createShaderModule(module_info), "module creation");
	vk::ComputePipelineCreateInfo pipeline_info;
	pipeline_info.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName = "main";
	pipeline_info.layout = layout;
	vk::Pipeline pipeline;
	Checked(gpu.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline), "pipeline creation");
	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eStorageBuffer, 1u};
	vk::DescriptorPoolCreateInfo pool_info;
	pool_info.maxSets = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes = &pool_size;
	const auto pool = Checked(gpu.device.createDescriptorPool(pool_info), "descriptor pool creation");
	vk::DescriptorSetAllocateInfo allocate;
	allocate.descriptorPool = pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts = &set_layout;
	vk::DescriptorSet set;
	Checked(gpu.device.allocateDescriptorSets(&allocate, &set), "descriptor set allocation");
	const vk::DescriptorBufferInfo buffer_info {buffer.buffer, 0u, VK_WHOLE_SIZE};
	vk::WriteDescriptorSet write;
	write.dstSet = set;
	write.dstBinding = 0;
	write.descriptorCount = 1;
	write.descriptorType = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo = &buffer_info;
	gpu.device.updateDescriptorSets(1, &write, 0, nullptr);
	vk::CommandPoolCreateInfo command_pool_info;
	command_pool_info.queueFamilyIndex = gpu.family;
	const auto command_pool = Checked(gpu.device.createCommandPool(command_pool_info), "command pool creation");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool = command_pool;
	command_info.level = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command;
	Checked(gpu.device.allocateCommandBuffers(&command_info, &command), "command buffer allocation");
	vk::CommandBufferBeginInfo begin;
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	Checked(command.begin(begin), "command buffer begin");
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, layout, 0, 1, &set, 0, nullptr);
	command.dispatch(Samples / 64u, 1, 1);
	vk::MemoryBarrier barrier;
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0,
	                        nullptr, 0, nullptr);
	Checked(command.end(), "command buffer end");
	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &command;
	Checked(gpu.queue.submit(1, &submit, nullptr), "submission");
	Checked(gpu.queue.waitIdle(), "queue wait");
	gpu.device.destroyCommandPool(command_pool);
	gpu.device.destroyDescriptorPool(pool);
	gpu.device.destroyPipeline(pipeline);
	gpu.device.destroyShaderModule(module);
	gpu.device.destroyPipelineLayout(layout);
	gpu.device.destroyDescriptorSetLayout(set_layout);
}

void Fill(uint32_t* words) {
	uint64_t random = 0x22516f00dull;
	const auto next = [&]() {
		random = random * 6364136223846793005ull + 1442695040888963407ull;
		return static_cast<uint32_t>(random >> 32u);
	};
	for (uint32_t i = 0; i < Samples; ++i) {
		auto* row = words + i * Stride;
		row[0] = i < 65536u ? i | ((i ^ 0xffffu) << 16u) : next();
		row[1] = next();
		row[2] = next();
		row[3] = i < 65536u ? DecodeBits(i) : next();
	}
	constexpr std::array<std::array<uint32_t, 3>, 26> edges {{
	    {0x5200, 0x5001, 0x8400}, {0xd200, 0x5001, 0x0400},
	    {0x0000, 0x3c00, 0x0000}, {0x8000, 0x3c00, 0x8000},
	    {0x8000, 0x3c00, 0x0000}, {0x0000, 0xbc00, 0x8000},
	    {0x3c00, 0x3c00, 0xbc00}, {0xbc00, 0x3c00, 0x3c00},
	    {0x7c00, 0x0000, 0x3c00}, {0x7c00, 0x3c00, 0xfc00},
	    {0xfc00, 0x3c00, 0x7c00}, {0x3c00, 0x3c00, 0x7c00},
	    {0x7e01, 0x3c00, 0x0000}, {0x3c00, 0x7d01, 0x0000},
	    {0x3c00, 0x3c00, 0xfe01}, {0x7c00, 0xfc00, 0xfc00},
	    {0x0001, 0x3800, 0x0000}, {0x8001, 0x3800, 0x8000},
	    {0x0003, 0x3800, 0x0000}, {0x03ff, 0x3c00, 0x0001},
	    {0x7bff, 0x3c00, 0x4c00}, {0x7bff, 0x4000, 0xfbff},
	    {0x0400, 0x3800, 0x0001}, {0x3c01, 0x3e00, 0x1000},
	    {0x8000, 0x0000, 0x8000}, {0x7d01, 0xfe02, 0x0000},
	}};
	constexpr std::array<uint32_t, 16> boundaries {
	    0x33000000u, 0x33000001u, 0xb3000001u, 0x32ffffffu,
	    0x387fe000u, 0x387fdfffu, 0x387fe001u, 0x477ff000u,
	    0x477fefffu, 0xc77ff000u, 0x7f800000u, 0xff800000u,
	    0x7fc00000u, 0xff800001u, 0x00000001u, 0x80000001u,
	};
	for (uint32_t i = 0; i < edges.size(); ++i) {
		auto* row = words + (65536u + i) * Stride;
		for (uint32_t j = 0; j < 3; ++j) row[j] = edges[i][j] | (edges[edges.size() - 1u - i][j] << 16u);
		row[3] = boundaries[i % boundaries.size()];
	}
}

void CheckHalf(uint32_t actual, uint32_t expected, const std::string& what) {
	Check(actual == expected || (IsNan(actual) && IsNan(expected)), what + ": got " + Hex(actual) + ", expected " + Hex(expected));
}

void Verify(const uint32_t* words) {
	for (uint32_t i = 0; i < Samples; ++i) {
		const auto* in = words + i * Stride;
		const auto* out = in + Inputs;
		const auto a = in[0] & 0xffffu;
		const auto b = in[1] & 0xffffu;
		const auto c = in[2] & 0xffffu;
		const auto what = std::string("sample ") + std::to_string(i);
		const auto decoded = DecodeBits(a);
		Check(out[0] == decoded || (IsNan(a) && std::isnan(std::bit_cast<float>(out[0]))), what + ": half to float");
		CheckHalf(out[1], a, what + ": conversion round trip");
		const std::array<uint32_t, 7> exact {
		    a, a, in[0] >> 16u, (in[0] & 0xffff0000u) | b, (in[0] & 0xffffu) | (b << 16u), a & 0x7fffu, a ^ 0x8000u,
		};
		for (uint32_t j = 0; j < exact.size(); ++j) Check(out[2 + j] == exact[j], what + ": bits operation " + std::to_string(j));
		const std::array<uint32_t, 6> scalar {
		    Fma(a, 0x3c00u, b), Fma(a, 0x3c00u, b ^ 0x8000u), Fma(a, b, (a ^ b) & 0x8000u),
		    Fma(a, b, c), MinMax(a, b, false), MinMax(a, b, true),
		};
		for (uint32_t j = 0; j < scalar.size(); ++j) CheckHalf(out[9 + j], scalar[j], what + ": scalar operation " + std::to_string(j));
		for (uint32_t lane = 0; lane < 2; ++lane) {
			const auto lhs = (in[0] >> (lane * 16u)) & 0xffffu;
			const auto rhs = (in[1] >> (lane * 16u)) & 0xffffu;
			const auto add = (in[2] >> (lane * 16u)) & 0xffffu;
			const std::array<uint32_t, 5> packed {
			    Fma(lhs, 0x3c00u, rhs), Fma(lhs, rhs, (lhs ^ rhs) & 0x8000u), Fma(lhs, rhs, add),
			    MinMax(lhs, rhs, false), MinMax(lhs, rhs, true),
			};
			for (uint32_t j = 0; j < packed.size(); ++j) {
				CheckHalf((out[15 + j] >> (lane * 16u)) & 0xffffu, packed[j], what + ": packed operation " + std::to_string(j));
			}
		}
		CheckHalf(out[20], EncodeBits(in[3]), what + ": float to half");
		CheckHalf(out[23] & 0xffffu, a, what + ": float pair low lane");
		CheckHalf(out[23] >> 16u, EncodeBits(in[3]), what + ": float pair high lane");
		for (uint32_t lane = 0; lane < 2; ++lane) {
			const auto half = (in[0] >> (lane * 16u)) & 0xffffu;
			Check(out[21 + lane] == DecodeBits(half) || (IsNan(half) && std::isnan(std::bit_cast<float>(out[21 + lane]))),
			      what + ": packed conversion lane " + std::to_string(lane));
		}
	}
}

}

int main() {
	Check(Fma(0x5200, 0x5001, 0x8400) == 0x6601u, "the exact oracle catches double rounding");
	for (const bool native: {false, true}) {
		for (const bool dispatcher: {false, true}) ValidateIrHalfProgram(native, dispatcher);
	}
	const auto words = BuildModule(false);
	BuildModule(true);
	const char* why = nullptr;
	auto opened = OpenGpu(why);
	if (!opened) return NoDevice(why);
	auto& gpu = *opened;
	std::printf("%s: %s; shaderFloat16 disabled\n", Name, gpu.name.c_str());
	auto buffer = CreateHostBuffer(gpu, static_cast<uint64_t>(Samples) * Stride * sizeof(uint32_t));
	Fill(static_cast<uint32_t*>(buffer.mapped));
	Execute(gpu, words, buffer);
	Verify(static_cast<const uint32_t*>(buffer.mapped));
	DestroyHostBuffer(gpu, buffer);
	CloseGpu(gpu);
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %u checks failed\n", Name, g_failures);
		return 1;
	}
	std::printf("%s: ok, all 65536 half encodings and 131072 scalar/packed samples\n", Name);
	return 0;
}
