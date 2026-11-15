#include "../PortabilitySubset.hpp"
#include "prx/libSceAgcDriver/Execution/include/BdaFeatures.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BufferPool.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureDetiler.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "FixtureDeviceSelection.hpp"
#include <SDL_loadso.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace AgcDriver::Graphics;

constexpr std::uint32_t Side = 256;
constexpr std::size_t DepthOffset = 0;
constexpr std::size_t StencilOffset = 4u * 65536u;
constexpr std::size_t HtileOffset = 6u * 65536u;
constexpr std::size_t OtherOffset = 8u * 65536u;
constexpr std::size_t BlockBytes = 16u * 65536u;
constexpr int Skipped = 77;

enum class Mode { Default, Verify, Off };

class Device {
public:
    Device() {
#ifdef _WIN32
        library = SDL_LoadObject("vulkan-1.dll");
#elif defined(__APPLE__)
        library = SDL_LoadObject("libvulkan.1.dylib");
#else
        library = SDL_LoadObject("libvulkan.so.1");
#endif
        Require(library != nullptr, "cannot load Vulkan");
        try {
            instanceProc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_LoadFunction(library, "vkGetInstanceProcAddr"));
            Require(instanceProc != nullptr, "missing Vulkan instance resolver");
            VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
            application.apiVersion = VK_API_VERSION_1_1;
            VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
            info.pApplicationInfo = &application;
#ifdef __APPLE__
            const char* portability = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
            info.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
            info.enabledExtensionCount = 1;
            info.ppEnabledExtensionNames = &portability;
#endif
            Check(function<PFN_vkCreateInstance>("vkCreateInstance")(&info, nullptr, &instance), "vkCreateInstance");
            std::uint32_t count = 0;
            const auto enumerate = function<PFN_vkEnumeratePhysicalDevices>("vkEnumeratePhysicalDevices");
            Check(enumerate(instance, &count, nullptr), "vkEnumeratePhysicalDevices");
            Require(count != 0, "no Vulkan device");
            std::vector<VkPhysicalDevice> devices(count);
            Check(enumerate(instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
            context.physical = SelectFixturePhysicalDevice(devices, function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties"));
            const auto extensions = function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties");
            Check(extensions(context.physical, nullptr, &count, nullptr), "vkEnumerateDeviceExtensionProperties");
            std::vector<VkExtensionProperties> available(count);
            Check(extensions(context.physical, nullptr, &count, available.data()), "vkEnumerateDeviceExtensionProperties");
            auto bytes = AgcDriver::QueryBdaByteFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            auto address = AgcDriver::QueryBdaFeatures(context.physical, function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), available);
            const auto queues = function<PFN_vkGetPhysicalDeviceQueueFamilyProperties>("vkGetPhysicalDeviceQueueFamilyProperties");
            queues(context.physical, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            queues(context.physical, &count, families.data());
            constexpr VkQueueFlags needed = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
            std::uint32_t family = 0;
            while (family < count && (families[family].queueFlags & needed) != needed) ++family;
            Require(family < count, "no Vulkan graphics and compute queue");
            const float priority = 1;
            VkDeviceQueueCreateInfo queue{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queue.queueFamilyIndex = family;
            queue.queueCount = 1;
            queue.pQueuePriorities = &priority;
            VkPhysicalDeviceFeatures enabled{};
            enabled.shaderInt64 = VK_TRUE;
            address.pNext = &bytes;
            std::vector<const char*> extensionsEnabled{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_EXTENSION_NAME};
            VkDeviceCreateInfo device{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &address};
            device.queueCreateInfoCount = 1;
            device.pQueueCreateInfos = &queue;
            AgcDriver::Tests::PortabilitySubset subset;
            subset.Enable(context.physical, function<PFN_vkEnumerateDeviceExtensionProperties>("vkEnumerateDeviceExtensionProperties"),
                function<PFN_vkGetPhysicalDeviceFeatures2>("vkGetPhysicalDeviceFeatures2"), extensionsEnabled, device);
            device.pEnabledFeatures = &enabled;
            Check(function<PFN_vkCreateDevice>("vkCreateDevice")(context.physical, &device, nullptr, &context.device), "vkCreateDevice");
            context.deviceProc = function<PFN_vkGetDeviceProcAddr>("vkGetDeviceProcAddr");
            function<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(context.physical, &context.memory);
            VkPhysicalDeviceProperties properties{};
            function<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(context.physical, &properties);
            context.limits = properties.limits;
            context.bufferDeviceAddress = true;
            context.formatProperties = function<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties");
            context.imageFormatProperties = function<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
            context.Function<PFN_vkGetDeviceQueue>("vkGetDeviceQueue")(context.device, family, 0, &context.queue);
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pool.queueFamilyIndex = family;
            Check(context.Function<PFN_vkCreateCommandPool>("vkCreateCommandPool")(context.device, &pool, nullptr, &context.pool), "vkCreateCommandPool");
        } catch (...) {
            release();
            throw;
        }
    }

    ~Device() { release(); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    const Context& GetContext() const { return context; }

private:
    template<typename TFunction>
    TFunction function(const char* name) const {
        const auto result = reinterpret_cast<TFunction>(instanceProc(instance, name));
        Require(result != nullptr, name);
        return result;
    }

    void release() noexcept {
        if (context.device != VK_NULL_HANDLE) ClearDepthSurfaces(context.device);
        if (context.pool != VK_NULL_HANDLE) context.Function<PFN_vkDestroyCommandPool>("vkDestroyCommandPool")(context.device, context.pool, nullptr);
        context.bufferPool.reset();
        if (context.device != VK_NULL_HANDLE) function<PFN_vkDestroyDevice>("vkDestroyDevice")(context.device, nullptr);
        if (instance != VK_NULL_HANDLE) function<PFN_vkDestroyInstance>("vkDestroyInstance")(instance, nullptr);
        if (library != nullptr) SDL_UnloadObject(library);
    }

    void* library = nullptr;
    PFN_vkGetInstanceProcAddr instanceProc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    Context context{};
};

class WatchedBlock {
public:
    WatchedBlock() {
#ifdef _WIN32
        block = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, 65536));
        if (block != nullptr) GuestArena::GuestArenaCommit_nid_postfix(block, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        constexpr std::uintptr_t alignment = 65536;
        void* mapped = mmap(nullptr, BlockBytes + alignment, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapped != MAP_FAILED) {
            const auto begin = reinterpret_cast<std::uintptr_t>(mapped);
            const auto aligned = (begin + alignment - 1) & ~(alignment - 1);
            if (aligned != begin) munmap(mapped, aligned - begin);
            if (aligned + BlockBytes != begin + BlockBytes + alignment) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + alignment - aligned);
            block = reinterpret_cast<std::uint8_t*>(aligned);
            GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, BlockBytes);
        }
#endif
        Require(block != nullptr, "depth view proofs: cannot allocate the guest block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
    }

    ~WatchedBlock() {
        GuestAllocations::Mutation().Remove(block);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(block, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(block, BlockBytes);
#else
        munmap(block, BlockBytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(block, BlockBytes);
#endif
    }

    WatchedBlock(const WatchedBlock&) = delete;
    WatchedBlock& operator=(const WatchedBlock&) = delete;

    std::uint8_t* Data() const { return block; }
    std::uint64_t Address(std::size_t offset = 0) const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)) + offset; }

private:
    std::uint8_t* block = nullptr;
};

DrawProofKindCounts DepthCounts() {
    return DrawProofCounted()[static_cast<std::size_t>(DrawProofKind::Depth)];
}

DrawProofKindCounts Delta(const DrawProofKindCounts& before, const DrawProofKindCounts& after) {
    DrawProofKindCounts delta;
    delta.reused = after.reused - before.reused;
    delta.proved = after.proved - before.proved;
    delta.verified = after.verified - before.verified;
    delta.differing = after.differing - before.differing;
    for (std::size_t miss = 0; miss < delta.missed.size(); ++miss) delta.missed[miss] = after.missed[miss] - before.missed[miss];
    return delta;
}

std::string Text(const DrawProofKindCounts& counts) {
    std::string text = "reused " + std::to_string(counts.reused) + ", proved " + std::to_string(counts.proved) + ", verified " + std::to_string(counts.verified) + ", differing " + std::to_string(counts.differing) + ", not reused:";
    for (std::size_t miss = 0; miss < counts.missed.size(); ++miss) {
        if (counts.missed[miss] != 0) text += std::string(" ") + DrawProofMissName(static_cast<DrawProofMiss>(miss)) + " " + std::to_string(counts.missed[miss]);
    }
    return text;
}

std::uint64_t Missed(const DrawProofKindCounts& counts, DrawProofMiss miss) {
    return counts.missed[static_cast<std::size_t>(miss)];
}

std::uint64_t Refused(const DrawProofKindCounts& counts) {
    std::uint64_t total = 0;
    for (const auto count : counts.missed) total += count;
    return total;
}

void Run(const Context& context, const WatchedBlock& block, Mode mode) {
    const bool proofs = mode != Mode::Off;
    const auto skipped = [mode](const DrawProofKindCounts& counts) { return mode == Mode::Verify ? counts.verified : counts.reused; };
    DepthTarget target{block.Address(DepthOffset), block.Address(StencilOffset), {Side, Side}, VK_FORMAT_D32_SFLOAT_S8_UINT, 1.0f, 0};
    target.htileAddress = block.Address(HtileOffset);
    target.depthSwizzle = XorSwizzleMode(TextureTileMode::kZ64KBX);
    struct Intent {
        bool depthWritten = true;
        bool stencilTested = true;
        bool stencilWritten = true;
    };
    const auto view = [&](const DepthTarget& drawn, Intent intent) {
        Require(DepthSurfaceViewWithStencilWriteIntent(context, drawn, intent.depthWritten, intent.stencilTested, intent.stencilWritten) != VK_NULL_HANDLE, "the depth target has no view");
    };
    const auto settled = [&](const char* what) {
        AgcDriver::GuestMemory::BumpCollectEpoch();
        for (int index = 0; index < 3; ++index) view(target, {});
        const auto before = DepthCounts();
        view(target, {});
        const auto counts = Delta(before, DepthCounts());
        std::cout << what << ": " << Text(counts) << '\n';
        if (proofs) Require(skipped(counts) == 1u && Refused(counts) == 0u, std::string(what) + ": repeated views of an unchanged depth target did not reuse their proof");
        else Require(counts.reused == 0u && counts.verified == 0u && Missed(counts, DrawProofMiss::None) == 1u, std::string(what) + ": APS5_NO_DEPTH_VIEW_PROOFS still reused a depth proof");
    };
    const auto refusedAfter = [&](const char* what, const std::function<void()>& change) {
        settled(what);
        change();
        const auto before = DepthCounts();
        view(target, {});
        const auto counts = Delta(before, DepthCounts());
        std::cout << what << " (next view): " << Text(counts) << '\n';
        Require(skipped(counts) == 0u && Refused(counts) == 1u, std::string(what) + " did not refuse the depth proof");
        if (proofs) Require(Missed(counts, DrawProofMiss::Versions) == 1u, std::string(what) + " refused the depth proof for another reason than its versions");
    };

    AgcDriver::GuestMemory::BumpCollectEpoch();
    settled("a depth target viewed again");
    refusedAfter("a CPU store to the depth plane", [&] {
        block.Data()[DepthOffset + 4096] ^= 1u;
        AgcDriver::GuestMemory::BumpCollectEpoch();
    });
    refusedAfter("a CPU store to the stencil plane", [&] {
        block.Data()[StencilOffset + 128] ^= 1u;
        AgcDriver::GuestMemory::BumpCollectEpoch();
    });
    refusedAfter("a CPU store to the HTILE", [&] {
        auto* word = static_cast<volatile std::uint8_t*>(block.Data() + HtileOffset + 64);
        const std::uint8_t value = *word;
        *word = value;
        AgcDriver::GuestMemory::BumpCollectEpoch();
    });
    refusedAfter("a driver store to the depth plane", [&] {
        const std::vector<std::byte> bytes(64, std::byte{0});
        AgcDriver::GuestMemory::Write(block.Address(DepthOffset + 8192), bytes);
    });
    if (proofs) {
        settled("before a CPU store to memory beside the planes");
        block.Data()[OtherOffset + 64] ^= 1u;
        AgcDriver::GuestMemory::BumpCollectEpoch();
        const auto before = DepthCounts();
        view(target, {});
        const auto counts = Delta(before, DepthCounts());
        Require(skipped(counts) == 1u && Refused(counts) == 0u, "a CPU store beside the depth planes refused the depth proof: " + Text(counts));
    }

    settled("before a change of intention");
    const auto planeBytes = DepthSliceBytes(target.extent, 4u);
    static_cast<void>(FlushDepthPlanes(target.address, static_cast<std::size_t>(planeBytes), DepthPlaneScope::Whole));
    AgcDriver::GuestMemory::BumpCollectEpoch();
    for (int index = 0; index < 3; ++index) view(target, {false, false, false});
    static_cast<void>(FlushDepthPlanes(target.address, static_cast<std::size_t>(planeBytes), DepthPlaneScope::Whole));
    AgcDriver::GuestMemory::BumpCollectEpoch();
    for (int index = 0; index < 3; ++index) view(target, {false, false, false});
    {
        const auto before = DepthCounts();
        view(target, {false, false, false});
        const auto counts = Delta(before, DepthCounts());
        if (proofs) Require(skipped(counts) == 1u, "a view without depth or stencil writes did not reuse the depth proof: " + Text(counts));
        Require(!DepthPlanesOverlap(target.address, static_cast<std::size_t>(planeBytes)), "a view without depth writes marked the depth plane stale");
    }
    {
        const auto before = DepthCounts();
        view(target, {true, false, false});
        const auto counts = Delta(before, DepthCounts());
        if (proofs) Require(skipped(counts) == 1u, "a view that writes depth did not reuse the depth proof of the same target: " + Text(counts));
        Require(DepthPlanesOverlap(target.address, static_cast<std::size_t>(planeBytes)), "a reused depth proof skipped marking the written depth plane stale");
    }

    settled("before the depth surface is retired");
    RetireDepthSurfaces(context.device, target.address, planeBytes);
    {
        const auto before = DepthCounts();
        view(target, {});
        const auto counts = Delta(before, DepthCounts());
        std::cout << "a view after the depth surface was retired: " << Text(counts) << '\n';
        Require(skipped(counts) <= 1u, "a view after the depth surface was retired counted twice");
    }
    settled("after the depth surface was retired");

    ClearDepthSurfaces(context.device);
    {
        const auto before = DepthCounts();
        view(target, {});
        const auto counts = Delta(before, DepthCounts());
        Require(skipped(counts) == 0u && Missed(counts, DrawProofMiss::None) == 1u, "a view of a freshly created depth surface reused a proof: " + Text(counts));
    }
    settled("a freshly created depth surface viewed again");

    DepthTarget multisampled = target;
    multisampled.address = block.Address(OtherOffset);
    multisampled.stencilAddress = 0;
    multisampled.htileAddress = 0;
    multisampled.format = VK_FORMAT_D32_SFLOAT;
    multisampled.samples = 4;
    const auto before = DepthCounts();
    bool viewed = true;
    for (int index = 0; index < 3; ++index) {
        try {
            DepthSurfaceViewWithStencilWriteIntent(context, multisampled, true, false, false);
        } catch (const std::exception& error) {
            std::cout << "the multisampled depth target was refused: " << error.what() << '\n';
            viewed = false;
            break;
        }
    }
    const auto counts = Delta(before, DepthCounts());
    std::cout << "a multisampled depth target: " << Text(counts) << '\n';
    Require(counts.reused == 0u && counts.verified == 0u && counts.proved == 0u, "a multisampled depth target used a depth proof");
    if (viewed) Require(Missed(counts, DrawProofMiss::None) == 3u, "a multisampled depth target was not counted as unproved at every view");
    if (mode == Mode::Verify) Require(DepthCounts().differing == 0u, "the full depth view beside a reused depth proof found a difference");
}

}

int main(int argc, char** argv) {
    try {
        const std::string_view argument = argc == 2 ? std::string_view(argv[1]) : std::string_view();
        const auto mode = argument == "--verify" ? Mode::Verify : argument == "--off" ? Mode::Off : Mode::Default;
        Require(mode != Mode::Verify || std::getenv("APS5_VERIFY_DRAW_PROOFS") != nullptr, "--verify needs APS5_VERIFY_DRAW_PROOFS");
        Require(mode != Mode::Off || std::getenv("APS5_NO_DEPTH_VIEW_PROOFS") != nullptr, "--off needs APS5_NO_DEPTH_VIEW_PROOFS");
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::puts("skipped, guest memory has no write watch");
            return Skipped;
        }
        if (mode != Mode::Off && (!DrawProofsEnabled() || !AgcDriver::GuestMemory::ResourceVersions())) {
            std::puts("skipped, draw proofs or resource versions are disabled in this environment");
            return Skipped;
        }
        std::optional<Device> device;
        try {
            device.emplace();
        } catch (const std::exception& error) {
            if (std::getenv("ANYPS5_REQUIRE_VULKAN") != nullptr) throw;
            std::printf("skipped, no usable Vulkan device: %s\n", error.what());
            return Skipped;
        }
        {
            TextureDetiler detiler(device->GetContext());
            auto context = device->GetContext();
            context.detiler = &detiler;
            WatchedBlock block;
            Require(AgcDriver::GuestMemory::Watched(block.Address(), BlockBytes), "depth view proofs: the guest block is not write-watched");
            Run(context, block, mode);
            ClearDepthSurfaces(context.device);
        }
        device.reset();
        std::puts("depth view proof tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
