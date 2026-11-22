#include "Dispatch.hpp"

#include "common/assert.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <spirv-tools/libspirv.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <utility>

namespace KytyOracle {
namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace Decoder = Recompiler::Decoder;
using BindingKind = Recompiler::IR::DescriptorBindingKind;

constexpr std::uint32_t BufferWord3 = 0x31016facu;
constexpr std::uint32_t AmdVendor = 0x1002u;
constexpr std::uint64_t ShaderHash = 0x0a11c1e5u;
constexpr std::uint32_t RowBytes = InputDwords * 4u;
constexpr std::uint32_t OutputBytes = OutputDwords * 4u;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Checked(vk::Result result, const char* what) {
    if (result != vk::Result::eSuccess) throw std::runtime_error(std::string(what) + " failed: " + vk::to_string(result));
}

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
    Checked(result.result, what);
    return std::move(result.value);
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, BufferWord3};
}

void RequireValid(const std::vector<std::uint32_t>& spirv) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
    std::string messages;
    tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
        messages += message;
        messages += '\n';
    });
    Require(tools.Validate(spirv), "the recompiled program is not valid SPIR-V: " + messages);
}

void RequireBodyWithoutUserSgprs(const Decoder::Program& decoded, const Program& program) {
    const auto begin = program.body_begin * 4u;
    const auto end = program.body_end * 4u;
    for (const auto& inst : decoded.instructions) {
        if (inst.pc < begin || inst.pc >= end) continue;
        for (const auto* operand : {&inst.dst, &inst.dst2, &inst.src0, &inst.src1, &inst.src2, &inst.src3}) {
            if (operand->kind == Decoder::OperandKind::Sgpr && operand->reg < UserSgprs) {
                throw std::runtime_error("the body names s" + std::to_string(operand->reg) + " at byte " + std::to_string(inst.pc) +
                                         "; s0-s" + std::to_string(UserSgprs - 1u) +
                                         " hold the runner's buffer descriptors, so such a body is refused");
            }
        }
    }
}

std::string Lower(std::string_view text) {
    std::string result(text);
    std::ranges::transform(result, result.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

struct Candidate {
    vk::PhysicalDevice physical;
    std::uint32_t family = 0;
    std::string name;
    bool software = false;
};

vk::Instance CreateInstance() {
    static const vk::detail::DynamicLoader loader;
    if (!loader.success()) throw NoDevice("the Vulkan loader is missing");
    auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
    dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
    vk::ApplicationInfo application;
    application.apiVersion = VK_API_VERSION_1_3;
    vk::InstanceCreateInfo instance_info;
    instance_info.pApplicationInfo = &application;
    const auto created = vk::createInstance(instance_info);
    if (created.result != vk::Result::eSuccess) throw NoDevice("no Vulkan 1.3 instance");
    dispatch.init(created.value);
    return created.value;
}

std::vector<Candidate> Candidates(vk::Instance instance) {
    std::vector<Candidate> result;
    for (const auto physical : Checked(instance.enumeratePhysicalDevices(), "device enumeration")) {
        if (physical.getProperties().apiVersion < VK_API_VERSION_1_3) continue;
        const auto families = physical.getQueueFamilyProperties();
        const auto compute = std::ranges::find_if(families, [](const auto& queue) {
            return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eCompute);
        });
        if (compute == families.end()) continue;
        vk::PhysicalDeviceDriverProperties driver;
        vk::PhysicalDeviceProperties2 properties;
        properties.pNext = &driver;
        physical.getProperties2(&properties);
        result.push_back({physical, static_cast<std::uint32_t>(compute - families.begin()),
                          std::string(properties.properties.deviceName.data()) + " (" + driver.driverName.data() + ", " +
                              driver.driverInfo.data() + ")",
                          properties.properties.deviceType == vk::PhysicalDeviceType::eCpu});
    }
    return result;
}

std::string Listing(const std::vector<Candidate>& candidates) {
    std::string text;
    for (std::size_t i = 0; i < candidates.size(); ++i) text += "\n  " + std::to_string(i) + ": " + candidates[i].name;
    return text;
}

const Candidate& Choose(const std::vector<Candidate>& candidates, std::string_view selector, Choice choice) {
    if (candidates.empty()) throw NoDevice("no Vulkan 1.3 device with a compute queue");
    if (!selector.empty()) {
        if (std::ranges::all_of(selector, [](unsigned char c) { return std::isdigit(c) != 0; })) {
            const auto index = ParseCount(selector, "HW_ORACLE_VULKAN_DEVICE");
            if (index < candidates.size()) return candidates[index];
            throw AmbiguousDevice("HW_ORACLE_VULKAN_DEVICE=" + std::string(selector) + " is not a device index:" + Listing(candidates));
        }
        const auto wanted = Lower(selector);
        const Candidate* match = nullptr;
        for (const auto& candidate : candidates) {
            if (Lower(candidate.name).find(wanted) == std::string::npos) continue;
            if (match != nullptr) {
                throw AmbiguousDevice("HW_ORACLE_VULKAN_DEVICE=" + std::string(selector) + " names more than one device:" +
                                      Listing(candidates));
            }
            match = &candidate;
        }
        if (match == nullptr) {
            throw AmbiguousDevice("HW_ORACLE_VULKAN_DEVICE=" + std::string(selector) + " names no device:" + Listing(candidates));
        }
        return *match;
    }
    if (candidates.size() == 1) return candidates.front();
    if (choice == Choice::PreferCpu) {
        const auto software = std::ranges::find_if(candidates, [](const Candidate& candidate) { return candidate.software; });
        return software != candidates.end() ? *software : candidates.front();
    }
    throw AmbiguousDevice("several Vulkan devices; choose one with HW_ORACLE_VULKAN_DEVICE=<index or part of the name>:" +
                          Listing(candidates));
}

struct HostBuffer {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    void* mapped = nullptr;
    std::uint64_t bytes = 0;
};

class Cleanup {
public:
    Cleanup() = default;
    Cleanup(const Cleanup&) = delete;
    Cleanup& operator=(const Cleanup&) = delete;

    ~Cleanup() {
        for (auto step = steps.rbegin(); step != steps.rend(); ++step) (*step)();
    }

    void Add(std::function<void()> step) {
        steps.push_back(std::move(step));
    }

private:
    std::vector<std::function<void()>> steps;
};

std::uint32_t LdsDwords(const Kernel& kernel) {
    return static_cast<std::uint32_t>((std::uint64_t {kernel.lds_bytes} + 3u) / 4u);
}

}

struct Device::State {
    vk::Instance instance;
    vk::PhysicalDevice physical;
    vk::Device device;
    vk::Queue queue;
    std::uint32_t family = 0;
    vk::PhysicalDeviceMemoryProperties memory;
    std::uint32_t min_subgroup = 0;
    std::uint32_t max_subgroup = 0;
    std::uint32_t max_shared_bytes = 0;
    std::uint32_t max_workgroup_threads = 0;
    std::uint32_t max_workgroup_subgroups = 0;
    bool subgroup_control = false;
    std::string name;
    Recompiler::Spirv::Profile profile;

    void Open(std::string_view selector, Choice choice);

    bool Enforces(std::uint32_t host_subgroup) const {
        return subgroup_control && host_subgroup >= min_subgroup && host_subgroup <= max_subgroup;
    }

    HostBuffer CreateHostBuffer(std::uint64_t bytes, Cleanup& cleanup) const {
        HostBuffer result;
        vk::BufferCreateInfo info;
        result.bytes = std::max<std::uint64_t>((bytes + 15u) & ~std::uint64_t {15u}, 16u);
        info.size = result.bytes;
        info.usage = vk::BufferUsageFlagBits::eStorageBuffer;
        result.buffer = Checked(device.createBuffer(info), "buffer creation");
        cleanup.Add([this, buffer = result.buffer] { device.destroyBuffer(buffer); });
        const auto needs = device.getBufferMemoryRequirements(result.buffer);
        const auto flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        for (std::uint32_t type = 0; type < memory.memoryTypeCount && !result.memory; type++) {
            if ((needs.memoryTypeBits >> type & 1u) == 0u || (memory.memoryTypes[type].propertyFlags & flags) != flags) continue;
            vk::MemoryAllocateInfo allocate;
            allocate.allocationSize = needs.size;
            allocate.memoryTypeIndex = type;
            result.memory = Checked(device.allocateMemory(allocate), "memory allocation");
            cleanup.Add([this, memory = result.memory] { device.freeMemory(memory); });
        }
        Require(static_cast<bool>(result.memory), "the device has no host-visible memory for a buffer");
        Checked(device.bindBufferMemory(result.buffer, result.memory, 0), "buffer binding");
        result.mapped = Checked(device.mapMemory(result.memory, 0, VK_WHOLE_SIZE), "buffer mapping");
        cleanup.Add([this, memory = result.memory] { device.unmapMemory(memory); });
        std::memset(result.mapped, 0, result.bytes);
        return result;
    }
};

std::vector<std::uint32_t> UserData(std::span<const std::uint32_t> input, std::span<const std::uint32_t> output) {
    std::vector<std::uint32_t> user(UserSgprs, 0u);
    std::ranges::copy(BufferDescriptor(output.data(), static_cast<std::uint32_t>(output.size_bytes())), user.begin() + OutputDescriptorSgpr);
    std::ranges::copy(BufferDescriptor(input.data(), static_cast<std::uint32_t>(input.size_bytes())), user.begin() + InputDescriptorSgpr);
    return user;
}

Module Compile(const Kernel& kernel, const Program& program, std::uint32_t threads, std::uint32_t host_subgroup,
               const Recompiler::Spirv::Profile& profile, std::span<const std::uint32_t> user_data) {
    Require(user_data.size() == UserSgprs, "the user data must be " + std::to_string(UserSgprs) + " dwords");
    ShaderComputeInputInfo compute;
    compute.threads_num[0] = threads;
    compute.threads_num[1] = 1u;
    compute.threads_num[2] = 1u;
    compute.lds_size_dwords = LdsDwords(kernel);
    compute.host_subgroup_size = host_subgroup;
    compute.wave_size = kernel.wave_size;
    compute.float_mode = static_cast<std::uint8_t>(kernel.float_mode);
    std::ranges::copy(kernel.group_id, compute.group_id);
    compute.thread_ids_num = static_cast<int>(kernel.thread_id_components);
    compute.workgroup_register = static_cast<int>(UserSgprs);
    compute.tg_size_en = kernel.tg_size;
    Recompiler::CompileOptions options;
    options.stage = ShaderType::Compute;
    options.wave_size = kernel.wave_size;
    options.user_data_base = 0u;
    options.shader_hash = ShaderHash;
    options.dump_ir = false;
    options.profile = &profile;
    options.user_data = user_data;
    options.input_info.compute = &compute;
    const Common::SoftExitScope soft_exit;
    const auto source = Recompiler::PrepareShaderSource(program.code, options);
    RequireBodyWithoutUserSgprs(source.decoded, program);
    auto translated = Recompiler::TranslateProgram(source.decoded, options);
    Module module;
    module.lane_local = translated.program.lane_local;
    const auto plan = Recompiler::IR::ExtractResourcePlan(translated.program);
    const Recompiler::IR::SrtRuntime runtime {.user_data = user_data};
    Recompiler::IR::ResourceSpecialization specialization;
    Require(Recompiler::IR::MaterializeResources(plan, runtime, module.snapshot, specialization),
            "the program's resources were not materialized");
    auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
    RequireValid(compiled.spirv);
    module.spirv = std::move(compiled.spirv);
    module.info = std::move(compiled.program).TakeCompiledInfo();
    return module;
}

std::string DescribeModes(const Kernel& kernel) {
    char text[160];
    std::snprintf(text, sizeof(text), "FLOAT_MODE 0x%02x (round32 %u, round16 %u, denorm32 %u, denorm16 %u), IEEE %u, DX10_CLAMP %u, FP16_OVFL %u",
                  kernel.float_mode, kernel.float_mode & 3u, (kernel.float_mode >> 2u) & 3u, (kernel.float_mode >> 4u) & 3u,
                  (kernel.float_mode >> 6u) & 3u, kernel.ieee_mode ? 1u : 0u, kernel.dx10_clamp ? 1u : 0u, kernel.fp16_overflow ? 1u : 0u);
    return text;
}

std::vector<std::string> Device::List() {
    const auto instance = CreateInstance();
    std::vector<std::string> names;
    try {
        for (const auto& candidate : Candidates(instance)) names.push_back(candidate.name);
    } catch (...) {
        instance.destroy();
        throw;
    }
    instance.destroy();
    return names;
}

void Device::State::Open(std::string_view selector, Choice choice) {
    const auto candidates = Candidates(instance);
    const auto& chosen = Choose(candidates, selector, choice);
    physical = chosen.physical;
    family = chosen.family;
    name = chosen.name;

    vk::PhysicalDeviceVulkan13Features features13;
    vk::PhysicalDeviceVulkan12Features features12;
    vk::PhysicalDeviceVulkan11Features features11;
    vk::PhysicalDeviceFeatures2 features;
    features12.pNext = &features13;
    features11.pNext = &features12;
    features.pNext = &features11;
    physical.getFeatures2(&features);
    features13.robustImageAccess = VK_FALSE;
    features.features.robustBufferAccess = VK_FALSE;

    const float priority = 1.0f;
    vk::DeviceQueueCreateInfo queue_info;
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    vk::DeviceCreateInfo device_info;
    device_info.pNext = &features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device = Checked(physical.createDevice(device_info), "device creation");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
    queue = device.getQueue(family, 0);
    memory = physical.getMemoryProperties();

    vk::PhysicalDeviceVulkan13Properties properties13;
    vk::PhysicalDeviceProperties2 properties;
    properties.pNext = &properties13;
    physical.getProperties2(&properties);
    const auto& limits = properties.properties.limits;
    min_subgroup = properties13.minSubgroupSize;
    max_subgroup = properties13.maxSubgroupSize;
    max_shared_bytes = limits.maxComputeSharedMemorySize;
    max_workgroup_threads = std::min(limits.maxComputeWorkGroupInvocations, limits.maxComputeWorkGroupSize[0]);
    max_workgroup_subgroups = properties13.maxComputeWorkgroupSubgroups;
    subgroup_control = features13.subgroupSizeControl == VK_TRUE &&
                       (properties13.requiredSubgroupSizeStages & vk::ShaderStageFlagBits::eCompute);
    profile.native_float_min_max = properties.properties.vendorID == AmdVendor;
    profile.native_float16 = features12.shaderFloat16 == VK_TRUE;
}

Device::Device(std::string_view selector, Choice choice) : state(std::make_unique<State>()) {
    auto& s = *state;
    s.instance = CreateInstance();
    try {
        s.Open(selector, choice);
    } catch (...) {
        if (s.device) s.device.destroy();
        s.instance.destroy();
        throw;
    }
}

Device::~Device() {
    state->device.destroy();
    state->instance.destroy();
}

const std::string& Device::Name() const {
    return state->name;
}

const Recompiler::Spirv::Profile& Device::Profile() const {
    return state->profile;
}

std::uint32_t Device::HostSubgroup(std::uint32_t wave_size) const {
    const auto& s = *state;
    if (s.Enforces(wave_size)) return wave_size;
    return std::min<std::uint32_t>(wave_size, 32u);
}

std::vector<std::uint32_t> Device::Run(const Kernel& kernel, const Program& program, std::uint32_t threads,
                                       std::span<const std::uint32_t> input) const {
    const auto& s = *state;
    Require(threads >= 1u && threads <= MaxThreads, "threads must be 1 to " + std::to_string(MaxThreads));
    Require(threads % kernel.wave_size == 0u, "threads must be a multiple of the wave size " + std::to_string(kernel.wave_size));
    Require(input.size() >= std::size_t {threads} * InputDwords, "the input holds fewer than threads * 16 bytes");
    Require(input.size() == std::size_t {threads} * InputDwords, "the input has extra words after the rows; extras need BDA and are not supported yet");
    Require(std::uint64_t {LdsDwords(kernel)} * 4u <= s.max_shared_bytes, "the kernel's LDS (" + std::to_string(kernel.lds_bytes) +
                                                                                " bytes) exceeds the device's maxComputeSharedMemorySize");
    const auto host_subgroup = HostSubgroup(kernel.wave_size);
    const bool enforced = s.Enforces(host_subgroup);
    if (threads > s.max_workgroup_threads || (enforced && threads > s.max_workgroup_subgroups * host_subgroup)) {
        throw Refusal(std::to_string(threads) + " threads exceed the device's workgroup of " +
                      std::to_string(enforced ? std::min(s.max_workgroup_threads, s.max_workgroup_subgroups * host_subgroup)
                                              : s.max_workgroup_threads) +
                      " invocations; run fewer rows per dispatch");
    }

    std::vector<std::uint32_t> in(input.begin(), input.end());
    std::vector<std::uint32_t> out(std::size_t {threads} * OutputDwords, UnwrittenOutput);
    const auto user_data = UserData(in, out);
    const auto module = Compile(kernel, program, threads, host_subgroup, s.profile, user_data);
    const bool fixed = s.min_subgroup == host_subgroup && s.max_subgroup == host_subgroup;
    if (!module.lane_local && !enforced && !fixed) {
        const auto widths = s.min_subgroup == s.max_subgroup
                                ? std::to_string(s.min_subgroup)
                                : std::to_string(s.min_subgroup) + " to " + std::to_string(s.max_subgroup);
        throw NotLaneLocal("the program is not lane-local and the device cannot run it on " + std::to_string(host_subgroup) +
                           "-lane subgroups: its subgroups hold " + widths + " lanes" +
                           (s.subgroup_control ? "" : " and their size cannot be chosen"));
    }

    const auto& layout = module.info.bindings;
    Require(layout.UsesPushData(), "the shader data does not fit the push constants");
    for (auto mask = layout.descriptor_mask; mask != 0; mask &= mask - 1u) {
        const auto kind = static_cast<BindingKind>(std::countr_zero(mask));
        Require(kind == BindingKind::Buffers,
                "descriptor binding kind " + std::to_string(static_cast<std::uint32_t>(kind)) + " is not served by the runner");
    }

    Cleanup cleanup;
    const auto buffer_count = layout.descriptor_counts[static_cast<std::size_t>(BindingKind::Buffers)];
    std::vector<HostBuffer> buffers(buffer_count);
    std::vector<void*> targets(buffer_count, nullptr);
    std::vector<bool> written(buffer_count, false);
    std::vector<std::uint64_t> sizes(buffer_count, 0u);
    std::vector<std::uint32_t> shader_data(layout.ShaderDataDwords(), 0u);
    for (std::size_t i = 0; i < module.info.info.user_data_registers.size(); i++) {
        shader_data[i] = user_data[module.info.info.user_data_registers[i] - module.info.user_data_base];
    }
    for (std::uint32_t i = 0; i < module.info.info.buffers.size(); i++) {
        const auto& resource = module.info.info.buffers[i];
        const auto index = resource.descriptor_index;
        if (index == UINT32_MAX) continue;
        Require(index < buffer_count && !buffers[index].buffer,
                "buffer descriptor " + std::to_string(index) + " is out of range or bound twice");
        ShaderBufferResource descriptor;
        std::copy_n(module.snapshot.buffers[i].dwords.begin(), std::size(descriptor.fields), descriptor.fields);
        auto* host = reinterpret_cast<void*>(static_cast<std::uintptr_t>(descriptor.Base48()));
        const auto bytes = descriptor.GetSize();
        buffers[index] = s.CreateHostBuffer(bytes, cleanup);
        targets[index] = host;
        written[index] = resource.written;
        sizes[index] = bytes;
        std::memcpy(buffers[index].mapped, host, bytes);
        shader_data[layout.MemorySizeDword() + index] = static_cast<std::uint32_t>(bytes);
        if (index < layout.stride_count && Recompiler::IR::HasRuntimeStride(resource.packed_stride)) {
            shader_data[layout.StrideDword() + index] = descriptor.PackedStride() & Recompiler::IR::BufferStrideMask;
        }
    }
    for (std::uint32_t index = 0; index < buffer_count; index++) {
        Require(static_cast<bool>(buffers[index].buffer), "buffer descriptor " + std::to_string(index) + " has no resource");
    }
    if (layout.dispatch_thread_dword != Recompiler::IR::PushData::NoStart) {
        shader_data[layout.dispatch_thread_dword] = threads;
        shader_data[layout.dispatch_thread_dword + 1u] = 1u;
        shader_data[layout.dispatch_thread_dword + 2u] = 1u;
    }
    Recompiler::IR::PushData push;
    Require(layout.push_data_start_dword + shader_data.size() <= push.dwords.size(), "the push data overflows");
    std::ranges::copy(shader_data, push.dwords.begin() + layout.push_data_start_dword);

    std::vector<vk::DescriptorSetLayoutBinding> set_bindings;
    if (buffer_count != 0) {
        set_bindings.push_back({Recompiler::IR::NativeBinding(ShaderType::Compute, BindingKind::Buffers),
                                vk::DescriptorType::eStorageBuffer, buffer_count, vk::ShaderStageFlagBits::eCompute, nullptr});
    }
    vk::DescriptorSetLayoutCreateInfo set_layout_info;
    set_layout_info.bindingCount = static_cast<std::uint32_t>(set_bindings.size());
    set_layout_info.pBindings = set_bindings.data();
    const auto set_layout = Checked(s.device.createDescriptorSetLayout(set_layout_info), "set layout creation");
    cleanup.Add([&s, set_layout] { s.device.destroyDescriptorSetLayout(set_layout); });
    const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0u, Recompiler::IR::NativePushConstantSize};
    vk::PipelineLayoutCreateInfo pipeline_layout_info;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &set_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
    const auto pipeline_layout = Checked(s.device.createPipelineLayout(pipeline_layout_info), "pipeline layout creation");
    cleanup.Add([&s, pipeline_layout] { s.device.destroyPipelineLayout(pipeline_layout); });
    vk::ShaderModuleCreateInfo module_info;
    module_info.codeSize = module.spirv.size() * sizeof(std::uint32_t);
    module_info.pCode = module.spirv.data();
    const auto shader = Checked(s.device.createShaderModule(module_info), "module creation");
    cleanup.Add([&s, shader] { s.device.destroyShaderModule(shader); });
    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup;
    vk::ComputePipelineCreateInfo pipeline_info;
    pipeline_info.stage.stage = vk::ShaderStageFlagBits::eCompute;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    if (enforced) {
        subgroup.requiredSubgroupSize = host_subgroup;
        pipeline_info.stage.pNext = &subgroup;
    }
    pipeline_info.layout = pipeline_layout;
    vk::Pipeline pipeline;
    Checked(s.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline), "pipeline creation");
    cleanup.Add([&s, pipeline] { s.device.destroyPipeline(pipeline); });

    const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eStorageBuffer, std::max(buffer_count, 1u)};
    vk::DescriptorPoolCreateInfo pool_info;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    const auto pool = Checked(s.device.createDescriptorPool(pool_info), "descriptor pool creation");
    cleanup.Add([&s, pool] { s.device.destroyDescriptorPool(pool); });
    vk::DescriptorSetAllocateInfo set_info;
    set_info.descriptorPool = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &set_layout;
    vk::DescriptorSet set;
    Checked(s.device.allocateDescriptorSets(&set_info, &set), "descriptor set allocation");
    std::vector<vk::DescriptorBufferInfo> buffer_infos;
    for (const auto& buffer : buffers) buffer_infos.push_back({buffer.buffer, 0u, VK_WHOLE_SIZE});
    if (buffer_count != 0) {
        vk::WriteDescriptorSet write;
        write.dstSet = set;
        write.dstBinding = set_bindings[0].binding;
        write.descriptorCount = buffer_count;
        write.descriptorType = vk::DescriptorType::eStorageBuffer;
        write.pBufferInfo = buffer_infos.data();
        s.device.updateDescriptorSets(1, &write, 0, nullptr);
    }

    vk::CommandPoolCreateInfo command_pool_info;
    command_pool_info.queueFamilyIndex = s.family;
    const auto command_pool = Checked(s.device.createCommandPool(command_pool_info), "command pool creation");
    cleanup.Add([&s, command_pool] { s.device.destroyCommandPool(command_pool); });
    vk::CommandBufferAllocateInfo command_info;
    command_info.commandPool = command_pool;
    command_info.level = vk::CommandBufferLevel::ePrimary;
    command_info.commandBufferCount = 1;
    vk::CommandBuffer command;
    Checked(s.device.allocateCommandBuffers(&command_info, &command), "command buffer allocation");
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
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0, nullptr, 0,
                            nullptr);
    Checked(command.end(), "command buffer end");
    vk::SubmitInfo submit;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    Checked(s.queue.submit(1, &submit, nullptr), "submission");
    Checked(s.queue.waitIdle(), "queue wait");

    for (std::uint32_t index = 0; index < buffer_count; index++) {
        if (written[index]) std::memcpy(targets[index], buffers[index].mapped, sizes[index]);
    }
    return out;
}

}
