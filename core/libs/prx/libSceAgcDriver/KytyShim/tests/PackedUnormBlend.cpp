#include "graphics/shader/recompiler/backend/spirv/packedUnormBlendProbe.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

constexpr uint32_t RecordWords = 5u;
constexpr uint32_t Records     = 1024u;

enum Factor : uint32_t {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    DstColor,
    OneMinusDstColor,
    SrcAlphaSaturate,
    FactorCount,
};

enum Op : uint32_t { Add, Subtract, Min, Max, ReverseSubtract, OpCount };

struct Blend {
    uint32_t src      = One;
    uint32_t dst      = Zero;
    uint32_t op       = Add;
    uint32_t mask     = 7u;
    bool     blend    = true;
    bool     clamp    = true;
    bool     truncate = false;

    [[nodiscard]] uint32_t Encode() const {
        return src | (dst << 5u) | (op << 10u) | (static_cast<uint32_t>(blend) << 13u) |
               (static_cast<uint32_t>(clamp) << 14u) | (mask << 16u);
    }
};

struct Record {
    std::array<float, 4> source {};
    uint32_t             destination = 0;
};

float Saturate(float value) {
    if (std::isnan(value)) return 0.0f;
    return std::fmin(std::fmax(value, 0.0f), 1.0f);
}

float Weighted(uint32_t factor, float operand, float src, float alpha, float dst) {
    switch (factor) {
        case Zero: return 0.0f;
        case One: return operand;
        case SrcColor: return operand * src;
        case OneMinusSrcColor: return operand * (1.0f - src);
        case SrcAlpha: return operand * alpha;
        case OneMinusSrcAlpha: return operand * (1.0f - alpha);
        case DstAlpha: return operand * 1.0f;
        case OneMinusDstAlpha: return operand * 0.0f;
        case DstColor: return operand * dst;
        case OneMinusDstColor: return operand * (1.0f - dst);
        case SrcAlphaSaturate: return operand * std::fmin(alpha, 0.0f);
        default: std::abort();
    }
}

uint32_t Reference(const Blend& blend, const Record& record) {
    constexpr std::array<uint32_t, 3> bits {11u, 11u, 10u};
    constexpr std::array<uint32_t, 3> offsets {0u, 11u, 22u};
    float alpha = record.source[3];
    if (blend.blend && blend.clamp) alpha = Saturate(alpha);
    uint32_t packed = 0;
    for (uint32_t component = 0; component < 3u; ++component) {
        const uint32_t field = (1u << bits[component]) - 1u;
        if ((blend.mask & (1u << component)) == 0u) {
            packed |= record.destination & (field << offsets[component]);
            continue;
        }
        float src    = record.source[component];
        float result = src;
        if (blend.blend) {
            if (blend.clamp) src = Saturate(src);
            const uint32_t stored = (record.destination >> offsets[component]) & field;
            const float    dst    = static_cast<float>(stored) * std::nextafter(1.0f / static_cast<float>(field), 1.0f);
            const float    s      = Weighted(blend.src, src, src, alpha, dst);
            const float    d      = Weighted(blend.dst, dst, src, alpha, dst);
            switch (blend.op) {
                case Add: result = s + d; break;
                case Subtract: result = s - d; break;
                case ReverseSubtract: result = d - s; break;
                case Min: result = std::fmin(src, dst); break;
                case Max: result = std::fmax(src, dst); break;
                default: std::abort();
            }
        }
        float scaled = Saturate(result) * static_cast<float>(field);
        if (!blend.truncate) scaled = scaled + 0.5f;
        packed |= static_cast<uint32_t>(scaled) << offsets[component];
    }
    return packed;
}

std::vector<Record> MakeRecords(bool with_nan) {
    std::vector<float> values {0.0f, 1.0f, 0.5f, 0.25f, 0.75f, 0.1f, 0.9f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f / 2047.0f,
                               0.5f / 2047.0f, 1.5f / 2047.0f, 1023.5f / 2047.0f, 511.5f / 1023.0f, -0.25f, -1.0f, 1.25f,
                               2.0f, 1.0e30f, -1.0e30f, 0.999f, 0.001f};
    if (with_nan) values.push_back(std::numeric_limits<float>::quiet_NaN());
    uint64_t state = 0x9e3779b97f4a7c15ull;
    const auto next = [&] {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(state >> 32u);
    };
    std::vector<Record> records(Records);
    for (uint32_t index = 0; index < Records; ++index) {
        auto& record = records[index];
        for (auto& value : record.source) {
            const uint32_t choice = next();
            if ((choice & 3u) == 0u) {
                value = static_cast<float>(next() & 0xffffffu) / static_cast<float>(0xffffffu) * 2.0f - 0.5f;
            } else {
                value = values[(choice >> 2u) % values.size()];
            }
        }
        record.destination = next();
        if ((index & 7u) == 0u) record.destination = 0u;
        if ((index & 7u) == 1u) record.destination = 0xffffffffu;
    }
    return records;
}

std::vector<Blend> MakeCases() {
    std::vector<Blend> cases;
    for (const bool clamp : {true, false}) {
        for (uint32_t src = 0; src < FactorCount; ++src) {
            for (uint32_t dst = 0; dst < FactorCount; ++dst) {
                cases.push_back({.src = src, .dst = dst, .op = Add, .clamp = clamp});
            }
        }
        constexpr std::array<std::array<uint32_t, 2>, 6> pairs {{{SrcAlpha, OneMinusSrcAlpha},
                                                                 {One, One},
                                                                 {SrcColor, OneMinusDstColor},
                                                                 {DstColor, Zero},
                                                                 {SrcAlphaSaturate, OneMinusSrcColor},
                                                                 {OneMinusDstAlpha, DstAlpha}}};
        for (uint32_t op = 0; op < OpCount; ++op) {
            for (const auto& pair : pairs) {
                cases.push_back({.src = pair[0], .dst = pair[1], .op = op, .clamp = clamp});
            }
        }
        cases.push_back({.src = SrcAlpha, .dst = OneMinusSrcAlpha, .clamp = clamp, .truncate = true});
        cases.push_back({.src = One, .dst = One, .clamp = clamp, .truncate = true});
        cases.push_back({.src = DstColor, .dst = SrcColor, .op = Subtract, .clamp = clamp, .truncate = true});
        for (uint32_t mask = 1; mask < 7u; ++mask) {
            cases.push_back({.src = SrcAlpha, .dst = OneMinusSrcAlpha, .mask = mask, .clamp = clamp});
            cases.push_back({.mask = mask, .blend = false, .clamp = clamp});
            cases.push_back({.mask = mask, .blend = false, .clamp = clamp, .truncate = true});
        }
    }
    return cases;
}

std::string Describe(const Blend& blend) {
    return "src=" + std::to_string(blend.src) + " dst=" + std::to_string(blend.dst) + " op=" + std::to_string(blend.op) +
           " mask=" + std::to_string(blend.mask) + " blend=" + std::to_string(blend.blend) +
           " clamp=" + std::to_string(blend.clamp) + " truncate=" + std::to_string(blend.truncate);
}

struct Loader {
    void*                     library = nullptr;
    PFN_vkGetInstanceProcAddr get     = nullptr;

    bool Open() {
#if defined(_WIN32)
        library = reinterpret_cast<void*>(LoadLibraryA("vulkan-1.dll"));
        if (library != nullptr) {
            get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), "vkGetInstanceProcAddr")));
        }
#else
        library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (library != nullptr) get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
        return get != nullptr;
    }
};

#define KYTY_VK_INSTANCE(name) const auto name = reinterpret_cast<PFN_##name>(loader.get(instance, #name))
#define KYTY_VK_DEVICE(name) const auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name))

bool Check(VkResult result, const char* what) {
    if (result == VK_SUCCESS) return true;
    std::fprintf(stderr, "kyty_shim_packed_unorm_blend: %s failed: %d\n", what, static_cast<int>(result));
    return false;
}

int RunDevice(const Loader& loader, VkInstance instance, VkPhysicalDevice physical, const std::vector<Blend>& cases,
              const std::vector<Record>& with_nan, const std::vector<Record>& without_nan) {
    KYTY_VK_INSTANCE(vkGetPhysicalDeviceProperties2);
    KYTY_VK_INSTANCE(vkGetPhysicalDeviceQueueFamilyProperties);
    KYTY_VK_INSTANCE(vkGetPhysicalDeviceMemoryProperties);
    KYTY_VK_INSTANCE(vkCreateDevice);
    KYTY_VK_INSTANCE(vkGetDeviceProcAddr);
    VkPhysicalDeviceFloatControlsProperties float_controls {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceProperties2             properties {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &float_controls};
    vkGetPhysicalDeviceProperties2(physical, &properties);
    const char* name = properties.properties.deviceName;
    if (float_controls.shaderSignedZeroInfNanPreserveFloat32 != VK_TRUE) {
        std::printf("%s: skipped, no shaderSignedZeroInfNanPreserveFloat32\n", name);
        return -1;
    }
    uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count && family == UINT32_MAX; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0u) family = i;
    }
    if (family == UINT32_MAX) {
        std::printf("%s: skipped, no compute queue\n", name);
        return -1;
    }
    const float             priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount       = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos    = &queue_info;
    VkDevice device                  = VK_NULL_HANDLE;
    if (!Check(vkCreateDevice(physical, &device_info, nullptr, &device), "vkCreateDevice")) return 1;
    KYTY_VK_DEVICE(vkDestroyDevice);
    KYTY_VK_DEVICE(vkGetDeviceQueue);
    KYTY_VK_DEVICE(vkCreateBuffer);
    KYTY_VK_DEVICE(vkDestroyBuffer);
    KYTY_VK_DEVICE(vkGetBufferMemoryRequirements);
    KYTY_VK_DEVICE(vkAllocateMemory);
    KYTY_VK_DEVICE(vkFreeMemory);
    KYTY_VK_DEVICE(vkBindBufferMemory);
    KYTY_VK_DEVICE(vkMapMemory);
    KYTY_VK_DEVICE(vkCreateDescriptorSetLayout);
    KYTY_VK_DEVICE(vkDestroyDescriptorSetLayout);
    KYTY_VK_DEVICE(vkCreatePipelineLayout);
    KYTY_VK_DEVICE(vkDestroyPipelineLayout);
    KYTY_VK_DEVICE(vkCreateDescriptorPool);
    KYTY_VK_DEVICE(vkDestroyDescriptorPool);
    KYTY_VK_DEVICE(vkAllocateDescriptorSets);
    KYTY_VK_DEVICE(vkUpdateDescriptorSets);
    KYTY_VK_DEVICE(vkCreateShaderModule);
    KYTY_VK_DEVICE(vkDestroyShaderModule);
    KYTY_VK_DEVICE(vkCreateComputePipelines);
    KYTY_VK_DEVICE(vkDestroyPipeline);
    KYTY_VK_DEVICE(vkCreateCommandPool);
    KYTY_VK_DEVICE(vkDestroyCommandPool);
    KYTY_VK_DEVICE(vkAllocateCommandBuffers);
    KYTY_VK_DEVICE(vkBeginCommandBuffer);
    KYTY_VK_DEVICE(vkEndCommandBuffer);
    KYTY_VK_DEVICE(vkResetCommandBuffer);
    KYTY_VK_DEVICE(vkCmdBindPipeline);
    KYTY_VK_DEVICE(vkCmdBindDescriptorSets);
    KYTY_VK_DEVICE(vkCmdDispatch);
    KYTY_VK_DEVICE(vkQueueSubmit);
    KYTY_VK_DEVICE(vkQueueWaitIdle);

    VkPhysicalDeviceMemoryProperties memory_properties {};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    const VkDeviceSize                sizes[2] {sizeof(uint32_t) * RecordWords * Records, sizeof(uint32_t) * Records};
    std::array<VkBuffer, 2>           buffers {};
    std::array<VkDeviceMemory, 2>     memories {};
    std::array<uint32_t*, 2>          mapped {};
    int                               failures = 0;
    for (uint32_t i = 0; i < 2u; ++i) {
        VkBufferCreateInfo buffer_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size  = sizes[i];
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (!Check(vkCreateBuffer(device, &buffer_info, nullptr, &buffers[i]), "vkCreateBuffer")) return 1;
        VkMemoryRequirements requirements {};
        vkGetBufferMemoryRequirements(device, buffers[i], &requirements);
        uint32_t type = UINT32_MAX;
        constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t t = 0; t < memory_properties.memoryTypeCount && type == UINT32_MAX; ++t) {
            if ((requirements.memoryTypeBits & (1u << t)) != 0u && (memory_properties.memoryTypes[t].propertyFlags & wanted) == wanted) type = t;
        }
        if (type == UINT32_MAX) {
            std::fprintf(stderr, "kyty_shim_packed_unorm_blend: %s has no host-visible coherent memory\n", name);
            return 1;
        }
        VkMemoryAllocateInfo allocate {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize  = requirements.size;
        allocate.memoryTypeIndex = type;
        if (!Check(vkAllocateMemory(device, &allocate, nullptr, &memories[i]), "vkAllocateMemory")) return 1;
        if (!Check(vkBindBufferMemory(device, buffers[i], memories[i], 0), "vkBindBufferMemory")) return 1;
        void* pointer = nullptr;
        if (!Check(vkMapMemory(device, memories[i], 0, VK_WHOLE_SIZE, 0, &pointer), "vkMapMemory")) return 1;
        mapped[i] = static_cast<uint32_t*>(pointer);
    }
    std::array<VkDescriptorSetLayoutBinding, 2> bindings {};
    for (uint32_t i = 0; i < 2u; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_layout_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_layout_info.bindingCount = 2;
    set_layout_info.pBindings    = bindings.data();
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    if (!Check(vkCreateDescriptorSetLayout(device, &set_layout_info, nullptr, &set_layout), "vkCreateDescriptorSetLayout")) return 1;
    VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts    = &set_layout;
    VkPipelineLayout layout    = VK_NULL_HANDLE;
    if (!Check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout), "vkCreatePipelineLayout")) return 1;
    VkDescriptorPoolSize       pool_size {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool_info {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets       = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes    = &pool_size;
    VkDescriptorPool pool   = VK_NULL_HANDLE;
    if (!Check(vkCreateDescriptorPool(device, &pool_info, nullptr, &pool), "vkCreateDescriptorPool")) return 1;
    VkDescriptorSetAllocateInfo set_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool     = pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts        = &set_layout;
    VkDescriptorSet set         = VK_NULL_HANDLE;
    if (!Check(vkAllocateDescriptorSets(device, &set_info, &set), "vkAllocateDescriptorSets")) return 1;
    std::array<VkDescriptorBufferInfo, 2> buffer_infos {};
    std::array<VkWriteDescriptorSet, 2>   writes {};
    for (uint32_t i = 0; i < 2u; ++i) {
        buffer_infos[i]           = {buffers[i], 0, VK_WHOLE_SIZE};
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(device, 2, writes.data(), 0, nullptr);
    VkCommandPoolCreateInfo command_pool_info {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool_info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = family;
    VkCommandPool command_pool         = VK_NULL_HANDLE;
    if (!Check(vkCreateCommandPool(device, &command_pool_info, nullptr, &command_pool), "vkCreateCommandPool")) return 1;
    VkCommandBufferAllocateInfo command_info {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool        = command_pool;
    command_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command         = VK_NULL_HANDLE;
    if (!Check(vkAllocateCommandBuffers(device, &command_info, &command), "vkAllocateCommandBuffers")) return 1;
    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, family, 0, &queue);

    const auto dispatch = [&](const Blend& blend, const std::vector<Record>& records) {
        for (uint32_t i = 0; i < Records; ++i) {
            std::memcpy(mapped[0] + i * RecordWords, records[i].source.data(), sizeof(float) * 4u);
            mapped[0][i * RecordWords + 4u] = records[i].destination;
            mapped[1][i]                    = 0xdeadbeefu;
        }
        const auto code = Libs::Graphics::ShaderRecompiler::Spirv::EmitPackedUnormBlendProbe(blend.Encode(), blend.truncate);
        VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = code.size() * sizeof(uint32_t);
        module_info.pCode    = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (!Check(vkCreateShaderModule(device, &module_info, nullptr, &module), "vkCreateShaderModule")) return false;
        VkComputePipelineCreateInfo pipeline_info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module = module;
        pipeline_info.stage.pName  = "main";
        pipeline_info.layout       = layout;
        VkPipeline pipeline        = VK_NULL_HANDLE;
        if (!Check(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline), "vkCreateComputePipelines")) return false;
        VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(command, 0);
        vkBeginCommandBuffer(command, &begin);
        vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(command, Records / 64u, 1, 1);
        vkEndCommandBuffer(command);
        VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers    = &command;
        if (!Check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit")) return false;
        if (!Check(vkQueueWaitIdle(queue), "vkQueueWaitIdle")) return false;
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, module, nullptr);
        return true;
    };

    uint32_t checked = 0;
    for (const auto& blend : cases) {
        const auto& records = blend.clamp || !blend.blend ? with_nan : without_nan;
        if (!dispatch(blend, records)) return 1;
        int case_failures = 0;
        for (uint32_t i = 0; i < Records; ++i) {
            const uint32_t expected = Reference(blend, records[i]);
            const uint32_t actual   = mapped[1][i];
            ++checked;
            if (actual == expected) continue;
            if (case_failures++ < 4) {
                std::fprintf(stderr,
                             "%s: %s record %u: source=(%a, %a, %a, %a) destination=0x%08x expected 0x%08x got 0x%08x\n", name,
                             Describe(blend).c_str(), i, static_cast<double>(records[i].source[0]),
                             static_cast<double>(records[i].source[1]), static_cast<double>(records[i].source[2]),
                             static_cast<double>(records[i].source[3]), records[i].destination, expected, actual);
            }
        }
        failures += case_failures;
    }

    uint32_t kept = 0;
    for (const bool clamp : {true, false}) {
        for (const bool truncate : {true, false}) {
            const Blend keep {.src = Zero, .dst = One, .clamp = clamp, .truncate = truncate};
            for (uint32_t base = 0; base < 2048u; base += Records) {
                std::vector<Record> records = clamp ? with_nan : without_nan;
                for (uint32_t i = 0; i < Records; ++i) {
                    const uint32_t code    = base + i;
                    records[i].destination = code | ((code ^ 0x7ffu) << 11u) | ((code & 0x3ffu) << 22u);
                }
                if (!dispatch(keep, records)) return 1;
                for (uint32_t i = 0; i < Records; ++i) {
                    const uint32_t actual = mapped[1][i];
                    ++kept;
                    if (actual == records[i].destination) continue;
                    if (failures++ < 8) {
                        std::fprintf(stderr, "%s: %s destination 0x%08x came back as 0x%08x\n", name,
                                     Describe(keep).c_str(), records[i].destination, actual);
                    }
                }
            }
        }
    }
    vkDestroyCommandPool(device, command_pool, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
    for (uint32_t i = 0; i < 2u; ++i) {
        vkDestroyBuffer(device, buffers[i], nullptr);
        vkFreeMemory(device, memories[i], nullptr);
    }
    vkDestroyDevice(device, nullptr);
    std::printf("%s: %u blend states, %u texels, %u kept destinations, %d mismatches\n", name,
                static_cast<uint32_t>(cases.size()), checked, kept, failures);
    return failures == 0 ? 0 : 1;
}

int Unavailable(const char* why) {
    const char* require = std::getenv("ANYPS5_REQUIRE_VULKAN");
    std::printf("kyty_shim_packed_unorm_blend: %s\n", why);
    return require != nullptr && std::strcmp(require, "1") == 0 ? 1 : 0;
}

}

int main() {
    const auto cases       = MakeCases();
    const auto with_nan    = MakeRecords(true);
    const auto without_nan = MakeRecords(false);
    {
        const Blend   over {.src = SrcAlpha, .dst = OneMinusSrcAlpha};
        const Record  half {.source = {1.0f, 0.0f, 2.0f, 0.5f}, .destination = 0u};
        const uint32_t word = Reference(over, half);
        if ((word & 0x7ffu) != 1024u || ((word >> 11u) & 0x7ffu) != 0u || (word >> 22u) != 512u) {
            std::fprintf(stderr, "kyty_shim_packed_unorm_blend: reference over blend gave 0x%08x\n", word);
            return 1;
        }
        const Blend   unclamped {.src = SrcAlpha, .dst = OneMinusSrcAlpha, .clamp = false};
        const Record  bright {.source = {4.0f, -4.0f, 0.25f, 0.5f}, .destination = 0x3ffu << 22u};
        const uint32_t clamped_word = Reference(over, bright);
        const uint32_t free_word    = Reference(unclamped, bright);
        if ((free_word & 0x7ffu) != 2047u || ((free_word >> 11u) & 0x7ffu) != 0u || (clamped_word & 0x7ffu) != 1024u) {
            std::fprintf(stderr, "kyty_shim_packed_unorm_blend: reference clamp gave 0x%08x and 0x%08x\n", clamped_word, free_word);
            return 1;
        }
    }
    Loader loader;
    if (!loader.Open()) return Unavailable("skipped, no Vulkan loader");
    const auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(loader.get(VK_NULL_HANDLE, "vkCreateInstance"));
    VkApplicationInfo application {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "kyty_shim_packed_unorm_blend";
    application.apiVersion       = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    VkInstance instance            = VK_NULL_HANDLE;
    if (vkCreateInstance == nullptr || vkCreateInstance(&instance_info, nullptr, &instance) != VK_SUCCESS) {
        return Unavailable("skipped, no Vulkan 1.2 instance");
    }
    KYTY_VK_INSTANCE(vkEnumeratePhysicalDevices);
    KYTY_VK_INSTANCE(vkDestroyInstance);
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    int tested = 0;
    int failed = 0;
    for (const auto physical : devices) {
        const int result = RunDevice(loader, instance, physical, cases, with_nan, without_nan);
        if (result < 0) continue;
        ++tested;
        failed += result;
    }
    vkDestroyInstance(instance, nullptr);
    if (tested == 0) return Unavailable("skipped, no Vulkan device ran the probe");
    return failed == 0 ? 0 : 1;
}
