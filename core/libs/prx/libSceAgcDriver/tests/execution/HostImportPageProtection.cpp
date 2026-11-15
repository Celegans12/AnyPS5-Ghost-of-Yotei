#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestPageTrack.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
namespace {

constexpr std::size_t PageBytes = 4096;
constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t RegionBytes = 16 * BlockBytes;
constexpr VkBufferUsageFlags Usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
constexpr std::uint32_t ShaderBase = 0xc0de0000u;

const std::uint32_t StoreShader[] = {
    0x07230203u, 0x00010000u, 0u, 23u, 0u,
    (2u << 16) | 17u, 1u,
    (3u << 16) | 14u, 0u, 1u,
    (6u << 16) | 15u, 5u, 1u, 0x6e69616du, 0u, 2u,
    (6u << 16) | 16u, 1u, 17u, 64u, 1u, 1u,
    (4u << 16) | 71u, 2u, 11u, 28u,
    (4u << 16) | 71u, 3u, 6u, 4u,
    (5u << 16) | 72u, 4u, 0u, 35u, 0u,
    (3u << 16) | 71u, 4u, 3u,
    (4u << 16) | 71u, 5u, 34u, 0u,
    (4u << 16) | 71u, 5u, 33u, 0u,
    (2u << 16) | 19u, 6u,
    (3u << 16) | 33u, 7u, 6u,
    (4u << 16) | 21u, 8u, 32u, 0u,
    (4u << 16) | 23u, 9u, 8u, 3u,
    (4u << 16) | 32u, 10u, 1u, 9u,
    (4u << 16) | 59u, 10u, 2u, 1u,
    (3u << 16) | 29u, 3u, 8u,
    (3u << 16) | 30u, 4u, 3u,
    (4u << 16) | 32u, 11u, 2u, 4u,
    (4u << 16) | 59u, 11u, 5u, 2u,
    (4u << 16) | 21u, 12u, 32u, 1u,
    (4u << 16) | 43u, 12u, 13u, 0u,
    (4u << 16) | 32u, 14u, 1u, 8u,
    (4u << 16) | 43u, 8u, 15u, 0u,
    (4u << 16) | 43u, 8u, 16u, ShaderBase,
    (4u << 16) | 32u, 17u, 2u, 8u,
    (5u << 16) | 54u, 6u, 1u, 0u, 7u,
    (2u << 16) | 248u, 18u,
    (5u << 16) | 65u, 14u, 19u, 2u, 15u,
    (4u << 16) | 61u, 8u, 20u, 19u,
    (5u << 16) | 128u, 8u, 21u, 16u, 20u,
    (6u << 16) | 65u, 17u, 22u, 5u, 13u, 20u,
    (3u << 16) | 62u, 22u, 21u,
    (1u << 16) | 253u,
    (1u << 16) | 56u,
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void Check(VkResult result, const char* what) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed with VkResult " + std::to_string(static_cast<int>(result)));
}

std::atomic<std::uintptr_t> watchedBase{0};
std::atomic<std::uint64_t> faults{0};

LONG CALLBACK HandleFault(EXCEPTION_POINTERS* exception) {
    const auto* record = exception->ExceptionRecord;
    if (record->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record->NumberParameters < 2 || record->ExceptionInformation[0] != 1) return EXCEPTION_CONTINUE_SEARCH;
    const auto base = watchedBase.load(std::memory_order_acquire);
    const auto address = static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
    if (GuestArena::GuestPageTrackHandleWrite_nid_no_patch(address)) return EXCEPTION_CONTINUE_EXECUTION;
    if (base == 0 || address < base || address >= base + RegionBytes) return EXCEPTION_CONTINUE_SEARCH;
    DWORD previous;
    if (!VirtualProtect(reinterpret_cast<void*>(address & ~(BlockBytes - 1)), BlockBytes, PAGE_READWRITE, &previous)) return EXCEPTION_CONTINUE_SEARCH;
    faults.fetch_add(1, std::memory_order_relaxed);
    return EXCEPTION_CONTINUE_EXECUTION;
}

class Vulkan {
public:
    Vulkan() {
        library = LoadLibraryA("vulkan-1.dll");
        if (library == nullptr) return;
        instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
        if (instanceProc == nullptr) return;
        const VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "host import page protection", 1, nullptr, 0, VK_API_VERSION_1_1};
        const VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &application, 0, nullptr, 0, nullptr};
        if (Global<PFN_vkCreateInstance>("vkCreateInstance")(&instanceInfo, nullptr, &instance) != VK_SUCCESS) return;
        std::uint32_t count = 0;
        Instance<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices")(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        Instance<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices")(instance, &count, devices.data());
        for (const bool discrete : {true, false}) {
            for (const auto candidate : devices) {
                if (physical != VK_NULL_HANDLE) break;
                VkPhysicalDeviceProperties candidateProperties{};
                Instance<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(candidate, &candidateProperties);
                if (discrete != (candidateProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) continue;
                std::uint32_t extensionCount = 0;
                Instance<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties")(candidate, nullptr, &extensionCount, nullptr);
                std::vector<VkExtensionProperties> extensions(extensionCount);
                Instance<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties")(candidate, nullptr, &extensionCount, extensions.data());
                if (std::none_of(extensions.begin(), extensions.end(), [](const VkExtensionProperties& extension) { return std::strcmp(extension.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0; })) continue;
                physical = candidate;
                properties = candidateProperties;
            }
        }
        if (physical == VK_NULL_HANDLE) return;
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT host{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT, nullptr, 0};
        VkPhysicalDeviceProperties2 extended{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &host, {}};
        Instance<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(physical, &extended);
        alignment = host.minImportedHostPointerAlignment;
        std::uint32_t familyCount = 0;
        Instance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties")(physical, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        Instance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties")(physical, &familyCount, families.data());
        const auto found = std::find_if(families.begin(), families.end(), [](const VkQueueFamilyProperties& candidate) { return (candidate.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0; });
        if (found == families.end()) return;
        family = static_cast<std::uint32_t>(found - families.begin());
        Instance<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(physical, &memory);
        const float priority = 1.0f;
        const VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &priority};
        const char* const extension = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
        const VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &queueInfo, 0, nullptr, 1, &extension, nullptr};
        if (Instance<PFN_vkCreateDevice>("vkCreateDevice")(physical, &deviceInfo, nullptr, &device) != VK_SUCCESS) return;
        deviceProc = Instance<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
        Device<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(device, family, 0, &queue);
        const VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
        Check(Device<PFN_vkCreateCommandPool>("vkCreateCommandPool")(device, &poolInfo, nullptr, &pool), "vkCreateCommandPool");
        const VkCommandBufferAllocateInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        Check(Device<PFN_vkAllocateCommandBuffers>("vkAllocateCommandBuffers")(device, &commandInfo, &commands), "vkAllocateCommandBuffers");
        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        const VkDescriptorSetLayoutCreateInfo setLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &binding};
        Check(Device<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(device, &setLayoutInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
        const VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &setLayout, 0, nullptr};
        Check(Device<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
        const VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, sizeof(StoreShader), StoreShader};
        Check(Device<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        const VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        const VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage, layout, VK_NULL_HANDLE, 0};
        Check(Device<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines");
        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        const VkDescriptorPoolCreateInfo descriptorPoolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &poolSize};
        Check(Device<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(device, &descriptorPoolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");
        const VkDescriptorSetAllocateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, descriptorPool, 1, &setLayout};
        Check(Device<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(device, &setInfo, &set), "vkAllocateDescriptorSets");
        ready = true;
    }

    Vulkan(const Vulkan&) = delete;
    Vulkan& operator=(const Vulkan&) = delete;

    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkResult result = VK_SUCCESS;
        const char* failed = "";
    };

    Buffer Staging(std::size_t bytes) {
        Buffer staging;
        const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes, Usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        Check(Device<PFN_vkCreateBuffer>("vkCreateBuffer")(device, &info, nullptr, &staging.buffer), "vkCreateBuffer");
        VkMemoryRequirements requirements{};
        Device<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(device, staging.buffer, &requirements);
        const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, requirements.size, MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        Check(Device<PFN_vkAllocateMemory>("vkAllocateMemory")(device, &allocation, nullptr, &staging.memory), "vkAllocateMemory of a staging buffer");
        Check(Device<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device, staging.buffer, staging.memory, 0), "vkBindBufferMemory");
        Check(Device<PFN_vkMapMemory>("vkMapMemory")(device, staging.memory, 0, VK_WHOLE_SIZE, 0, &staging.mapped), "vkMapMemory");
        return staging;
    }

    Buffer Import(void* pointer, std::size_t bytes) {
        Buffer imported;
        VkMemoryHostPointerPropertiesEXT host{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT, nullptr, 0};
        imported.result = Device<PFN_vkGetMemoryHostPointerPropertiesEXT>("vkGetMemoryHostPointerPropertiesEXT")(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, pointer, &host);
        if (imported.result != VK_SUCCESS) {
            imported.failed = "vkGetMemoryHostPointerPropertiesEXT";
            return imported;
        }
        const VkExternalMemoryBufferCreateInfo external{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
        const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &external, 0, bytes, Usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr};
        Check(Device<PFN_vkCreateBuffer>("vkCreateBuffer")(device, &info, nullptr, &imported.buffer), "vkCreateBuffer of an import");
        VkMemoryRequirements requirements{};
        Device<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(device, imported.buffer, &requirements);
        const VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, nullptr, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, pointer};
        const VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import, bytes, MemoryType(requirements.memoryTypeBits & host.memoryTypeBits, 0)};
        imported.result = Device<PFN_vkAllocateMemory>("vkAllocateMemory")(device, &allocation, nullptr, &imported.memory);
        if (imported.result != VK_SUCCESS) {
            imported.failed = "vkAllocateMemory";
            return imported;
        }
        imported.result = Device<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device, imported.buffer, imported.memory, 0);
        if (imported.result != VK_SUCCESS) imported.failed = "vkBindBufferMemory";
        return imported;
    }

    void Destroy(Buffer& buffer) {
        Device<PFN_vkDeviceWaitIdle>("vkDeviceWaitIdle")(device);
        if (buffer.buffer != VK_NULL_HANDLE) Device<PFN_vkDestroyBuffer>("vkDestroyBuffer")(device, buffer.buffer, nullptr);
        if (buffer.memory != VK_NULL_HANDLE) Device<PFN_vkFreeMemory>("vkFreeMemory")(device, buffer.memory, nullptr);
        buffer = {};
    }

    VkResult Copy(VkBuffer source, VkBuffer destination, std::size_t bytes) {
        Begin();
        const VkBufferCopy region{0, 0, bytes};
        Device<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(commands, source, destination, 1, &region);
        return Submit();
    }

    VkResult Store(VkBuffer destination, std::size_t bytes) {
        const VkDescriptorBufferInfo bufferInfo{destination, 0, bytes};
        const VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bufferInfo, nullptr};
        Device<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(device, 1, &write, 0, nullptr);
        Begin();
        Device<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        Device<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        Device<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, static_cast<std::uint32_t>(bytes / 4 / 64), 1, 1);
        return Submit();
    }

    bool ready = false;
    VkPhysicalDeviceProperties properties{};
    VkDeviceSize alignment = 0;

private:
    template <class TFunction>
    TFunction Global(const char* name) const { return reinterpret_cast<TFunction>(instanceProc(VK_NULL_HANDLE, name)); }
    template <class TFunction>
    TFunction Instance(const char* name) const { return reinterpret_cast<TFunction>(instanceProc(instance, name)); }
    template <class TFunction>
    TFunction Device(const char* name) const {
        const auto function = reinterpret_cast<TFunction>(deviceProc(device, name));
        if (function == nullptr) throw std::runtime_error(std::string("the device has no ") + name);
        return function;
    }

    std::uint32_t MemoryType(std::uint32_t bits, VkMemoryPropertyFlags wanted) const {
        for (std::uint32_t type = 0; type < memory.memoryTypeCount; ++type) {
            if ((bits & (1u << type)) != 0 && (memory.memoryTypes[type].propertyFlags & wanted) == wanted) return type;
        }
        throw std::runtime_error("no Vulkan memory type fits the request");
    }

    void Begin() {
        const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
        Check(Device<PFN_vkBeginCommandBuffer>("vkBeginCommandBuffer")(commands, &begin), "vkBeginCommandBuffer");
    }

    VkResult Submit() {
        Check(Device<PFN_vkEndCommandBuffer>("vkEndCommandBuffer")(commands), "vkEndCommandBuffer");
        const VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &commands, 0, nullptr};
        if (const auto result = Device<PFN_vkQueueSubmit>("vkQueueSubmit")(queue, 1, &submit, VK_NULL_HANDLE); result != VK_SUCCESS) return result;
        return Device<PFN_vkQueueWaitIdle>("vkQueueWaitIdle")(queue);
    }

    HMODULE library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    PFN_vkGetDeviceProcAddr deviceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

class GuestRegion {
public:
    explicit GuestRegion(bool resident) : resident(resident) {
        data = static_cast<unsigned char*>(GuestArena::GuestArenaAllocate_nid_postfix(RegionBytes, 0x200000));
        Require(data != nullptr, "cannot allocate a guest arena range");
        if (resident) {
            section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, RegionBytes, nullptr);
            Require(section != nullptr, "cannot reserve the backing of resident guest memory");
            GuestArena::GuestArenaMapDirect_nid_postfix(data, RegionBytes, section, 0, PAGE_READWRITE);
        } else {
            GuestArena::GuestArenaCommit_nid_postfix(data, RegionBytes, PAGE_READWRITE, RegionBytes);
        }
        std::memset(data, 0x11, RegionBytes);
        Require(ResetWriteWatch(data, RegionBytes) == 0, "guest arena memory is not write-watched");
        watchedBase.store(reinterpret_cast<std::uintptr_t>(data), std::memory_order_release);
    }

    ~GuestRegion() {
        watchedBase.store(0, std::memory_order_release);
        DWORD previous;
        VirtualProtect(data, RegionBytes, PAGE_READWRITE, &previous);
        GuestArena::GuestArenaReset_nid_postfix(data, RegionBytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, RegionBytes);
        if (section != nullptr) {
            GuestArena::GuestArenaForgetSection_nid_postfix(section);
            CloseHandle(section);
        }
    }

    GuestRegion(const GuestRegion&) = delete;
    GuestRegion& operator=(const GuestRegion&) = delete;

    bool Protect(DWORD protection) const {
        DWORD previous;
        return VirtualProtect(data, RegionBytes, protection, &previous) != 0;
    }

    DWORD Protection(std::size_t offset = 0) const {
        MEMORY_BASIC_INFORMATION memory{};
        Require(VirtualQuery(data + offset, &memory, sizeof(memory)) == sizeof(memory), "cannot query guest memory");
        return memory.Protect;
    }

    std::size_t Watched() const {
        std::vector<void*> written(RegionBytes / PageBytes);
        ULONG_PTR count = written.size();
        DWORD granularity = 0;
        Require(GetWriteWatch(WRITE_WATCH_FLAG_RESET, data, RegionBytes, written.data(), &count, &granularity) == 0, "GetWriteWatch failed");
        return count;
    }

    bool Holds(unsigned char value) const {
        return std::all_of(data, data + RegionBytes, [value](unsigned char byte) { return byte == value; });
    }

    bool HoldsShaderStore() const {
        for (std::uint32_t index = 0; index < RegionBytes / 4; ++index) {
            std::uint32_t value = 0;
            std::memcpy(&value, data + index * 4, 4);
            if (value != ShaderBase + index) return false;
        }
        return true;
    }

    const char* Kind() const { return resident ? "resident" : "private"; }

    unsigned char* data = nullptr;

private:
    HANDLE section = nullptr;
    bool resident;
};

struct Findings {
    bool readsArmed = true;
    bool writesLandArmed = true;
    bool writesFault = false;
    bool writesWatched = false;
    bool survives = true;
    bool importsArmed = false;
    bool armedImportWrites = false;
};

const char* Named(DWORD protection) {
    return protection == PAGE_READONLY ? "read-only" : protection == PAGE_READWRITE ? "read-write" : "another protection";
}

void ProtectAfterImport(Vulkan& vulkan, bool resident, Findings& findings) {
    GuestRegion region(resident);
    const auto kind = region.Kind();
    auto staging = vulkan.Staging(RegionBytes);
    auto imported = vulkan.Import(region.data, RegionBytes);
    if (imported.result != VK_SUCCESS) {
        std::printf("[import %s] importing read-write memory: %s failed with VkResult %d\n", kind, imported.failed, static_cast<int>(imported.result));
        findings.survives = false;
        vulkan.Destroy(imported);
        vulkan.Destroy(staging);
        return;
    }
    const auto readBack = [&](unsigned char expected) {
        std::memset(staging.mapped, 0, RegionBytes);
        const auto result = vulkan.Copy(imported.buffer, staging.buffer, RegionBytes);
        const auto* bytes = static_cast<const unsigned char*>(staging.mapped);
        return result == VK_SUCCESS && std::all_of(bytes, bytes + RegionBytes, [expected](unsigned char byte) { return byte == expected; });
    };
    const auto copyIn = [&](unsigned char value) {
        std::memset(staging.mapped, value, RegionBytes);
        return vulkan.Copy(staging.buffer, imported.buffer, RegionBytes);
    };
    const auto importWatched = region.Watched();
    const bool baselineRead = readBack(0x11);
    const auto baselineWrite = copyIn(0x22);
    const bool baselineLanded = region.Holds(0x22);
    const auto baselineWatched = region.Watched();
    std::printf("[import %s] imported while read-write: the import itself put %zu of %zu pages on the write watch; GPU read %s; GPU copy into the import returned %d and %s; the write watch saw %zu pages of that copy\n", kind, importWatched, RegionBytes / PageBytes, baselineRead ? "matches" : "DIFFERS", static_cast<int>(baselineWrite), baselineLanded ? "landed" : "did NOT land", baselineWatched);
    if (baselineWatched != 0) findings.writesWatched = true;

    const bool armed = region.Protect(PAGE_READONLY);
    std::printf("[import %s] VirtualProtect of the imported range to read-only: %s, protection now %s\n", kind, armed ? "succeeded" : "FAILED", Named(region.Protection()));
    if (!armed) findings.survives = false;
    const bool armedRead = readBack(0x22);
    std::printf("[import %s] GPU read of the armed range: %s\n", kind, armedRead ? "matches the CPU bytes" : "DIFFERS from the CPU bytes");
    if (!armedRead) findings.readsArmed = false;

    auto before = faults.load();
    const auto armedCopy = copyIn(0x33);
    const bool copyLanded = region.Holds(0x33);
    const auto copyFaults = faults.load() - before;
    const auto copyWatched = region.Watched();
    std::printf("[import %s] GPU copy into the armed range: returned %d, bytes %s, %llu CPU faults, protection still %s, the write watch saw %zu pages\n", kind, static_cast<int>(armedCopy), copyLanded ? "landed" : "did NOT land", static_cast<unsigned long long>(copyFaults), Named(region.Protection()), copyWatched);
    if (!copyLanded) findings.writesLandArmed = false;
    if (copyFaults != 0) findings.writesFault = true;
    if (copyWatched != 0) findings.writesWatched = true;

    before = faults.load();
    const auto armedStore = vulkan.Store(imported.buffer, RegionBytes);
    const bool storeLanded = region.HoldsShaderStore();
    const auto storeFaults = faults.load() - before;
    const auto storeWatched = region.Watched();
    std::printf("[import %s] shader store into the armed range: returned %d, values %s, %llu CPU faults, protection still %s, the write watch saw %zu pages\n", kind, static_cast<int>(armedStore), storeLanded ? "landed" : "did NOT land", static_cast<unsigned long long>(storeFaults), Named(region.Protection()), storeWatched);
    if (!storeLanded) findings.writesLandArmed = false;
    if (storeFaults != 0) findings.writesFault = true;
    if (storeWatched != 0) findings.writesWatched = true;

    before = faults.load();
    for (std::size_t offset = 0; offset < RegionBytes; offset += PageBytes) std::memset(region.data + offset, 0x44, PageBytes);
    const auto cpuFaults = faults.load() - before;
    const bool afterFault = readBack(0x44);
    std::printf("[import %s] CPU writes to the armed range: %llu faults reopened %zu blocks; the GPU then reads %s\n", kind, static_cast<unsigned long long>(cpuFaults), RegionBytes / BlockBytes, afterFault ? "the new bytes" : "STALE bytes");
    if (!afterFault) findings.survives = false;

    bool cycles = true;
    for (unsigned char cycle = 0; cycle < 8; ++cycle) {
        if (!region.Protect(PAGE_READONLY) || !region.Protect(PAGE_READWRITE)) cycles = false;
        std::memset(region.data, 0x50 + cycle, RegionBytes);
        if (!readBack(0x50 + cycle)) cycles = false;
        if (copyIn(0x60 + cycle) != VK_SUCCESS || !region.Holds(0x60 + cycle)) cycles = false;
    }
    std::printf("[import %s] 8 cycles of read-only then read-write, each followed by a CPU write the GPU reads and a GPU write the CPU reads: %s\n", kind, cycles ? "the import stayed valid" : "the import BROKE");
    if (!cycles) findings.survives = false;
    vulkan.Destroy(imported);
    vulkan.Destroy(staging);
}

void ImportAfterProtect(Vulkan& vulkan, bool resident, Findings& findings) {
    GuestRegion region(resident);
    const auto kind = region.Kind();
    Require(region.Protect(PAGE_READONLY), "cannot write-protect guest memory");
    auto staging = vulkan.Staging(RegionBytes);
    auto imported = vulkan.Import(region.data, RegionBytes);
    if (imported.result != VK_SUCCESS) {
        std::printf("[import %s] importing a range that is already read-only: %s failed with VkResult %d\n", kind, imported.failed, static_cast<int>(imported.result));
        vulkan.Destroy(imported);
        vulkan.Destroy(staging);
        return;
    }
    findings.importsArmed = true;
    std::memset(staging.mapped, 0, RegionBytes);
    const auto read = vulkan.Copy(imported.buffer, staging.buffer, RegionBytes);
    const auto* bytes = static_cast<const unsigned char*>(staging.mapped);
    const bool readMatches = read == VK_SUCCESS && std::all_of(bytes, bytes + RegionBytes, [](unsigned char byte) { return byte == 0x11; });
    const auto before = faults.load();
    std::memset(staging.mapped, 0x77, RegionBytes);
    const auto write = vulkan.Copy(staging.buffer, imported.buffer, RegionBytes);
    const bool landed = region.Holds(0x77);
    std::printf("[import %s] importing a range that is already read-only: succeeded; GPU read %s; GPU copy into it returned %d, bytes %s, %llu CPU faults, protection still %s, the write watch saw %zu pages\n", kind, readMatches ? "matches" : "DIFFERS", static_cast<int>(write), landed ? "landed" : "did NOT land", static_cast<unsigned long long>(faults.load() - before), Named(region.Protection()), region.Watched());
    if (landed) findings.armedImportWrites = true;
    vulkan.Destroy(imported);
    vulkan.Destroy(staging);
}

void ArmedImport(Vulkan& vulkan, bool resident, bool& failed) {
    constexpr std::size_t Blocks = RegionBytes / BlockBytes;
    GuestRegion region(resident);
    watchedBase.store(0, std::memory_order_release);
    const auto kind = region.Kind();
    const auto base = reinterpret_cast<std::uintptr_t>(region.data);
    const auto check = [&](bool condition, const char* what) {
        std::printf("[armed import %s] %s: %s\n", kind, what, condition ? "yes" : "NO");
        if (!condition) failed = true;
    };
    const auto counts = [] {
        std::array<std::uint64_t, 12> values{};
        GuestArena::GuestPageTrackCounts_nid_no_patch(values.data(), values.size());
        return values;
    };
    std::array<std::uint64_t, Blocks> tokens{};
    const auto arm = [&] { return GuestArena::GuestPageTrackArm_nid_no_patch(base, RegionBytes, tokens.data(), nullptr, Blocks); };
    const auto clean = [&] { return GuestArena::GuestPageTrackCleanSince_nid_no_patch(base, RegionBytes, tokens.data(), Blocks); };

    check(arm() && region.Protection() == PAGE_READONLY, "the range is armed before anything imports it");
    auto staging = vulkan.Staging(RegionBytes);
    std::optional<GuestArena::HostImportAccess> access;
    access.emplace(region.data, RegionBytes);
    check(!clean() && region.Protection() == PAGE_READWRITE, "the import access opened the armed range and ended its tokens before the import exists");
    auto imported = vulkan.Import(region.data, RegionBytes);
    access.reset();
    check(imported.result == VK_SUCCESS, "the import of a range that was armed succeeds once the range is open");
    if (imported.result != VK_SUCCESS) {
        vulkan.Destroy(imported);
        vulkan.Destroy(staging);
        return;
    }
    const auto matches = [&] {
        std::memset(staging.mapped, 0, RegionBytes);
        return vulkan.Copy(imported.buffer, staging.buffer, RegionBytes) == VK_SUCCESS && std::memcmp(staging.mapped, region.data, RegionBytes) == 0;
    };

    check(arm() && clean() && region.Protection() == PAGE_READONLY, "the imported range is armed again after the import");
    check(matches() && clean(), "the GPU reads the armed imported range and its tokens still hold");

    const auto before = counts();
    static_cast<volatile unsigned char*>(region.data)[5 * BlockBytes + 7] = 0x5a;
    const auto after = counts();
    check(after[2] == before[2] + 1 && !GuestArena::GuestPageTrackCleanSince_nid_no_patch(base + 5 * BlockBytes, BlockBytes, &tokens[5], 1) && GuestArena::GuestPageTrackCleanSince_nid_no_patch(base + 6 * BlockBytes, BlockBytes, &tokens[6], 1), "a CPU write to an armed imported block faults once and ends the token of that block alone");
    check(matches(), "the GPU reads the byte the CPU wrote through the fault");

    GuestArena::GuestPageTrackGpuWrite_nid_no_patch(base + 2 * BlockBytes + 100, 8);
    check(region.Protection(2 * BlockBytes) == PAGE_READWRITE && region.Protection(3 * BlockBytes) == PAGE_READONLY && !GuestArena::GuestPageTrackCleanSince_nid_no_patch(base + 2 * BlockBytes, BlockBytes, &tokens[2], 1), "an announced GPU write opens its block and ends its token before the write, and leaves the next block armed");

    const auto flight = GuestArena::GuestPageTrackGpuFlightBegin_nid_no_patch();
    GuestArena::GuestPageTrackGpuWrite_nid_no_patch(base, RegionBytes);
    std::memset(staging.mapped, 0x33, RegionBytes);
    const auto copied = vulkan.Copy(staging.buffer, imported.buffer, RegionBytes);
    check(copied == VK_SUCCESS && region.Holds(0x33) && region.Protection() == PAGE_READWRITE, "an announced GPU copy into the import lands in an open range");
    GuestArena::GuestPageTrackFrame_nid_no_patch();
    GuestArena::GuestPageTrackFrame_nid_no_patch();
    const bool rearmed = arm();
    check(!rearmed && std::all_of(tokens.begin(), tokens.end(), [](std::uint64_t token) { return token == 0; }) && region.Protection() == PAGE_READWRITE, "blocks the GPU writes are not armed while the batch that carries the write is in flight");
    GuestArena::GuestPageTrackGpuFlightEnd_nid_no_patch(flight);
    check(arm() && clean() && region.Protection() == PAGE_READONLY && region.Holds(0x33) && matches(), "blocks the GPU wrote are armed again once the batch that carried the write finished, with the bytes it wrote (unless APS5_NO_GPU_WRITE_MARK_EXPIRY=1)");
    vulkan.Destroy(imported);
    vulkan.Destroy(staging);
}

}
#endif

int main() {
#ifdef _WIN32
    try {
        Vulkan vulkan;
        if (!vulkan.ready) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw std::runtime_error("no Vulkan device imports host memory");
            std::puts("skipped, no Vulkan device imports host memory");
            return VulkanTestSkipped;
        }
        std::printf("[import] device %s, driver version 0x%x, host pointer alignment %llu\n", vulkan.properties.deviceName, vulkan.properties.driverVersion, static_cast<unsigned long long>(vulkan.alignment));
        const auto handler = AddVectoredExceptionHandler(1, HandleFault);
        Findings findings;
        for (const bool resident : {false, true}) {
            ProtectAfterImport(vulkan, resident, findings);
            ImportAfterProtect(vulkan, resident, findings);
        }
        bool failed = false;
        if (GuestArena::GuestPageTrackArmsImports_nid_no_patch()) {
            for (const bool resident : {false, true}) ArmedImport(vulkan, resident, failed);
        } else {
            std::puts("[armed import] skipped, set APS5_PAGE_TRACK=1 and APS5_ARM_READ_IMPORTS=1 to arm imported blocks through the page tracking table");
        }
        RemoveVectoredExceptionHandler(handler);
        std::printf("[import] GPU reads of an armed import: %s\n", findings.readsArmed ? "work" : "fail");
        std::printf("[import] GPU writes into an armed import: %s, %s, %s\n", findings.writesLandArmed ? "land" : "do not land", findings.writesFault ? "fault on the CPU" : "raise no CPU fault", findings.writesWatched ? "reach the write watch" : "never reach the write watch");
        std::printf("[import] the import across VirtualProtect in both directions: %s\n", findings.survives ? "survives" : "breaks");
        std::printf("[import] importing an already read-only range: %s\n", !findings.importsArmed ? "refused" : findings.armedImportWrites ? "accepted, and the GPU still writes through it" : "accepted, GPU writes do not land");
        if (!findings.survives || !findings.readsArmed) std::puts("verdict import: impossible: write-protecting an imported range breaks the import");
        else if (findings.writesLandArmed && !findings.writesFault) std::puts("verdict import: safe under a condition: the import and GPU reads survive the protection, but GPU writes bypass the page tables, so a range the GPU can write must never be armed and keeps relying on the driver's own marks");
        else if (!findings.writesLandArmed) std::puts("verdict import: impossible for ranges the GPU writes: the protection makes GPU writes fail");
        else std::puts("verdict import: safe: GPU writes into an armed range fault like CPU writes");
        if (failed) std::puts("verdict armed import: FAILED, an imported block armed through the page tracking table did not behave as the driver relies on");
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
#else
    std::puts("skipped, page protection of host imports is measured on Windows");
    return VulkanTestSkipped;
#endif
}
