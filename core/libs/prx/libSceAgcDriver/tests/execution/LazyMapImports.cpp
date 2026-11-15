#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>

namespace {

using AgcDriver::Graphics::Require;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t AllocationBytes = BlockBytes * 4;
constexpr std::size_t ReservationBytes = AllocationBytes + BlockBytes * 2;

class GuestAllocation {
public:
    GuestAllocation() {
#ifdef _WIN32
        allocation = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, ReservationBytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
        allocation = static_cast<std::uint8_t*>(std::aligned_alloc(BlockBytes, ReservationBytes));
#endif
        Require(allocation != nullptr, "lazy map imports: cannot allocate the guest memory");
        memory = allocation + BlockBytes;
        std::memset(memory, 0, AllocationBytes);
        GuestAllocations::Mutation().Add(memory, AllocationBytes, true, true, true);
    }

    ~GuestAllocation() {
        GuestAllocations::Mutation().Remove(memory);
#ifdef _WIN32
        VirtualFree(allocation, 0, MEM_RELEASE);
#else
        std::free(allocation);
#endif
    }

    GuestAllocation(const GuestAllocation&) = delete;
    GuestAllocation& operator=(const GuestAllocation&) = delete;

    std::uint64_t Address() const { return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(memory)); }

    GuestAllocations::Mapped Ranges() const {
        GuestAllocations::Mapped ranges;
        for (const auto& range : GuestAllocations::GuestAllocationsAcquire_nid_postfix()) {
            if (range->address == Address()) ranges.push_back(range);
        }
        Require(ranges.size() == 1, "lazy map imports: the allocation is not registered");
        return ranges;
    }

private:
    std::uint8_t* allocation = nullptr;
    std::uint8_t* memory = nullptr;
};

void Import(AgcDriver::VulkanDevice& device, const GuestAllocations::Mapped& ranges) {
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    if (device.ImportGuestMemory(ranges, generation, false)) return;
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    Require(device.ImportGuestMemory(ranges, generation, true), "lazy map imports: the import under the GPU lock refused");
}

bool Imported(VkDevice device, std::uint64_t address, std::size_t bytes) {
    AgcDriver::Graphics::Context probe{};
    probe.device = device;
    probe.hostImportAlignment = 1;
    return AgcDriver::Graphics::HostImportCovers(probe, address, bytes);
}

bool Fill(AgcDriver::VulkanDevice& device, std::uint64_t address) {
    std::lock_guard lock(AgcDriver::GuestMemory::GpuMutex());
    const std::array<std::uint32_t, 4> pattern{1u, 2u, 3u, 4u};
    const bool filled = device.FillBuffer(address, BlockBytes, pattern);
    device.WaitIdle();
    return filled;
}

std::uint64_t MadeOnDemand(const AgcDriver::Graphics::HostImportCounts& counts) {
    std::uint64_t made = 0;
    for (std::size_t requester = 0; requester < counts.requesters.size(); ++requester) {
        if (requester != static_cast<std::size_t>(AgcDriver::Graphics::ImportRequester::Map)) made += counts.requesters[requester].made;
    }
    return made;
}

}

int main() {
    try {
#ifdef _WIN32
        _putenv_s("APS5_PIN_WAIT_MS", "2000");
        _putenv_s("APS5_LAZY_MAP_IMPORTS", "1");
#else
        setenv("APS5_PIN_WAIT_MS", "2000", 1);
        setenv("APS5_LAZY_MAP_IMPORTS", "1", 1);
#endif
        Require(AgcDriver::Graphics::LazyMapImports(), "lazy map imports: the switch was read before the test set it");
        auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        GuestAllocation probe;
        if (!Fill(*device, probe.Address())) {
            std::printf("skipped, the device does not import guest memory\n");
            return VulkanTestSkipped;
        }
        const auto handle = device->Device();
        const auto map = static_cast<std::size_t>(AgcDriver::Graphics::ImportRequester::Map);
        GuestAllocation guest;
        const auto before = AgcDriver::Graphics::HostImportCounters();
        Import(*device, guest.Ranges());
        const auto mapped = AgcDriver::Graphics::HostImportCounters();
        Require(!Imported(handle, guest.Address(), AllocationBytes), "lazy map imports: a mapping was imported before anything used it");
        Require(mapped.requesters[map].made == before.requesters[map].made && mapped.mapDeferred == before.mapDeferred + 1 && mapped.mapDeferredBytes == before.mapDeferredBytes + AllocationBytes, "lazy map imports: a mapping left to its first use was not counted");
        Require(Fill(*device, guest.Address()), "lazy map imports: the first use of a mapping was not imported");
        const auto used = AgcDriver::Graphics::HostImportCounters();
        Require(Imported(handle, guest.Address(), BlockBytes), "lazy map imports: the first use of a mapping left no import");
        Require(used.requesters[map].made == mapped.requesters[map].made && MadeOnDemand(used) > MadeOnDemand(mapped), "lazy map imports: the import at first use was not counted for the requester that asked");
        GuestAllocation idle;
        Import(*device, idle.Ranges());
        Require(Fill(*device, guest.Address()), "lazy map imports: a second use of an imported mapping was refused");
        Require(!Imported(handle, idle.Address(), AllocationBytes), "lazy map imports: a mapping nothing used was imported");
        std::puts("lazy map imports tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
