// Probe for DESIGN.md phase 4 step 2 (GPU-owned blocks): does a GPU store through a host import
// still land once the imported pages are no-access (or read-only) for the process? Reports what
// the device does; it fails only when the fixture itself cannot be set up.
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>

#ifdef _WIN32
namespace {

using AgcDriver::Graphics::Require;

constexpr std::size_t BlockBytes = 65536;

bool Holds(const std::uint8_t* block, std::uint32_t value) {
    for (std::size_t at = 0; at < BlockBytes; at += 4) {
        std::uint32_t word = 0;
        std::memcpy(&word, block + at, 4);
        if (word != value) return false;
    }
    return true;
}

// Fills the block on the GPU while its pages have `protection`, then reads it back as read-write.
const char* StoreUnder(AgcDriver::VulkanDevice& device, std::uint8_t* block, DWORD protection, std::uint32_t value) {
    DWORD previous = 0;
    Require(VirtualProtect(block, BlockBytes, protection, &previous) != 0, "no-access import probe: cannot protect the block");
    const char* outcome = nullptr;
    try {
        const std::array<std::uint32_t, 4> pattern{value, value, value, value};
        if (!device.FillBuffer(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)), BlockBytes, pattern)) outcome = "the fill was refused";
        device.SubmitRecorded(true);
        device.WaitIdle();
    } catch (const std::exception& error) {
        std::printf("  exception: %s\n", error.what());
        outcome = "the fill threw";
    }
    Require(VirtualProtect(block, BlockBytes, PAGE_READWRITE, &previous) != 0, "no-access import probe: cannot open the block again");
    if (outcome != nullptr) return outcome;
    return Holds(block, value) ? "the store landed" : "the store was lost";
}

}
#endif

int main() {
#ifdef _WIN32
    try {
        auto* block = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, BlockBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        Require(block != nullptr, "no-access import probe: cannot allocate the block");
        std::memset(block, 0, BlockBytes);
        GuestAllocations::Mutation().Add(block, BlockBytes, true, true, true);
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        {
            std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
            const std::array<std::uint32_t, 4> first{0x11111111u, 0x11111111u, 0x11111111u, 0x11111111u};
            if (!device->FillBuffer(static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block)), BlockBytes, first)) {
                std::printf("skipped, the device does not import guest memory\n");
                return VulkanTestSkipped;
            }
            device->SubmitRecorded(true);
            device->WaitIdle();
            std::printf("first use, read-write: %s\n", Holds(block, 0x11111111u) ? "the store landed" : "the store was lost");
            std::printf("read-only pages: %s\n", StoreUnder(*device, block, PAGE_READONLY, 0x22222222u));
            std::printf("no-access pages: %s\n", StoreUnder(*device, block, PAGE_NOACCESS, 0x33333333u));
            std::printf("read-write again: %s\n", StoreUnder(*device, block, PAGE_READWRITE, 0x44444444u));
        }
        device.reset();
        GuestAllocations::Mutation().Remove(block);
        VirtualFree(block, 0, MEM_RELEASE);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#else
    return 0;
#endif
}
