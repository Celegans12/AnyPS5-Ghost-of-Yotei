#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using AgcDriver::Graphics::ResidentStagingCounts;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t PageBytes = 4096;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t BufferBytes = Lanes * 4u;
constexpr std::uint32_t WideBytes = 4u * PageBytes;

alignas(256) constexpr std::array<std::uint32_t, 8> AddOne{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a040481u, 0xe0701000u, 0x80000201u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> AddTwo{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a040482u, 0xe0701000u, 0x80000201u, 0xbf810000u};

struct Block {
    Block() {
#ifdef _WIN32
        data = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes * 2u, BlockBytes));
        GuestArena::GuestArenaCommit_nid_postfix(data, BlockBytes, PAGE_READWRITE, BlockBytes);
#else
        void* raw = mmap(nullptr, BlockBytes * 2u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Require(raw != MAP_FAILED, "resident staging: cannot map a block");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + BlockBytes - 1) & ~(static_cast<std::uintptr_t>(BlockBytes) - 1);
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + BlockBytes != begin + BlockBytes * 2u) munmap(reinterpret_cast<void*>(aligned + BlockBytes), begin + BlockBytes - aligned);
        data = reinterpret_cast<std::uint8_t*>(aligned);
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(data, BlockBytes);
#endif
        Require(data != nullptr, "resident staging: cannot allocate a block");
        std::memset(data, 0, BlockBytes);
        GuestAllocations::Mutation mutation;
        mutation.Add(data, BlockBytes, true, true, true);
    }

    ~Block() {
        AgcDriver::Graphics::StorageTexture::FlushPending(Address(), BlockBytes, nullptr, "test release");
        {
            GuestAllocations::Mutation mutation;
            mutation.Remove(data);
        }
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(data, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, BlockBytes * 2u);
#else
        munmap(data, BlockBytes);
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(data, BlockBytes);
#endif
    }

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    std::uint32_t* Words() const { return reinterpret_cast<std::uint32_t*>(data); }

    std::uint8_t* data = nullptr;
};

std::vector<std::unique_ptr<Block>> blocks;

const Block& NewBlock() {
    blocks.push_back(std::make_unique<Block>());
    return *blocks.back();
}

ShaderRecompiler::RecompileResult Compile(AgcDriver::VulkanDevice& device, std::span<const std::uint32_t> code, std::uint64_t address, std::uint32_t bytes) {
    std::vector<std::uint32_t> userData(8, 0u);
    const std::array<std::uint32_t, 4> descriptor{static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x31016facu};
    std::copy(descriptor.begin(), descriptor.end(), userData.begin());
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Lanes, 1, 1}, 0, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {64, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    return ShaderRecompiler::Recompile(request);
}

void Run(AgcDriver::VulkanDevice& device, const ShaderRecompiler::RecompileResult& shader, std::span<const std::uint32_t> code) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    device.Dispatch(shader, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()));
}

void Land(AgcDriver::VulkanDevice& device) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    device.SubmitRecorded(false);
}

void Expect(const Block& block, std::uint32_t expected, const std::string& what, std::uint32_t skip = Lanes) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        if (lane == skip) continue;
        const auto found = block.Words()[lane];
        Require(found == expected, "resident staging, " + what + ": dword " + std::to_string(lane) + " is " + std::to_string(found) + ", expected " + std::to_string(expected));
    }
}

struct Counts {
    ResidentStagingCounts before = AgcDriver::Graphics::ResidentStagingCounters();

    ResidentStagingCounts Since() const {
        auto now = AgcDriver::Graphics::ResidentStagingCounters();
        now.made -= before.made;
        now.shared -= before.shared;
        now.skipped -= before.skipped;
        now.skippedBytes -= before.skippedBytes;
        now.partial -= before.partial;
        now.partialBytes -= before.partialBytes;
        now.partialSavedBytes -= before.partialSavedBytes;
        now.full -= before.full;
        now.fullBytes -= before.fullBytes;
        now.untracked -= before.untracked;
        now.blocked -= before.blocked;
        now.sealed -= before.sealed;
        now.outdated -= before.outdated;
        now.verified -= before.verified;
        now.differing -= before.differing;
        return now;
    }
};

std::string Text(const ResidentStagingCounts& counts) {
    return std::to_string(counts.skipped) + " skipped, " + std::to_string(counts.partial) + " partial, " + std::to_string(counts.full) + " whole, " + std::to_string(counts.sealed) + " left current, " + std::to_string(counts.outdated) + " outdated, " + std::to_string(counts.untracked) + " untracked";
}

enum class Probe { Unstaged, Untracked, Resident };

Probe ProbeResident(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), BufferBytes);
    const Counts counts;
    Run(device, addOne, AddOne);
    Land(device);
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Expect(block, 2, "two uses in two batches");
    const auto since = counts.Since();
    if (since.made + since.shared == 0) return Probe::Unstaged;
    Require(since.skipped + since.partial + since.full == 2, "resident staging: two uses of one shadow counted " + Text(since));
    return since.skipped == 1 && since.full == 1 ? Probe::Resident : Probe::Untracked;
}

void SteadyState(AgcDriver::VulkanDevice& device, bool strict) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), BufferBytes);
    const auto addTwo = Compile(device, AddTwo, block.Address(), BufferBytes);
    const Counts counts;
    for (std::uint32_t round = 0; round < 4; ++round) {
        Run(device, addOne, AddOne);
        Land(device);
        Run(device, addTwo, AddTwo);
        Land(device);
        AgcDriver::Graphics::NoteHostImportFrame();
    }
    device.WaitIdle();
    Expect(block, 12, "two programs over one region for four frames");
    const auto since = counts.Since();
    Require(since.skipped + since.partial + since.full == 8, "resident staging: eight uses of one shadow counted " + Text(since));
    if (strict) Require(since.full == 1 && since.skipped == 7 && since.skippedBytes == 7u * BufferBytes && since.fullBytes == BufferBytes, "resident staging: eight uses of an untouched region counted " + Text(since) + ", expected one whole copy-in and seven skipped");
}

void CpuStoreBetween(AgcDriver::VulkanDevice& device, bool strict) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), WideBytes);
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Expect(block, 2, "two uses before a CPU store");
    const Counts counts;
    block.Words()[5] = 100;
    block.Words()[PageBytes / 4u] = 7;
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Require(block.Words()[5] == 101, "resident staging: a CPU store between two uses reached the shader as " + std::to_string(block.Words()[5] - 1u) + ", expected 100");
    Require(block.Words()[PageBytes / 4u] == 7, "resident staging: a CPU store the shader leaves alone came back as " + std::to_string(block.Words()[PageBytes / 4u]));
    Expect(block, 3, "a use after a CPU store", 5);
    const auto since = counts.Since();
    Require(since.skipped == 0 && since.partial + since.full == 1, "resident staging: the use after a CPU store counted " + Text(since) + ", expected a copy-in");
    if (strict) Require(since.partial == 1 && since.partialBytes == 2u * PageBytes && since.partialSavedBytes == WideBytes - 2u * PageBytes, "resident staging: the use after CPU stores to two pages counted " + Text(since) + " and copied " + std::to_string(since.partialBytes) + " bytes, expected those two pages");
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Require(block.Words()[5] == 102 && block.Words()[PageBytes / 4u] == 7, "resident staging: the use after the refreshed one lost the CPU stores");
    if (strict) Require(counts.Since().skipped == 1, "resident staging: the use after the refreshed one counted " + Text(counts.Since()) + ", expected a skipped copy-in");
}

void FillBetween(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), BufferBytes);
    const auto addTwo = Compile(device, AddTwo, block.Address(), BufferBytes);
    Run(device, addOne, AddOne);
    Land(device);
    Run(device, addOne, AddOne);
    const Counts counts;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const std::array<std::uint32_t, 4> pattern{7u, 7u, 7u, 7u};
        Require(device.FillBuffer(block.Address(), BufferBytes, pattern), "resident staging: the fill over the staged region was not recorded on the GPU");
    }
    Run(device, addTwo, AddTwo);
    device.WaitIdle();
    Expect(block, 9, "a fill between two uses");
    const auto since = counts.Since();
    Require(since.skipped == 0 && since.partial + since.full == 1, "resident staging: the use after a fill over its region counted " + Text(since) + ", expected a copy-in");
}

void DriverStoreBetween(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), BufferBytes);
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Run(device, addOne, AddOne);
    device.WaitIdle();
    const Counts counts;
    std::array<std::uint32_t, Lanes> stored;
    stored.fill(40);
    AgcDriver::GuestMemory::Write(block.Address(), std::as_bytes(std::span(stored)));
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Expect(block, 41, "a driver store between two uses");
    const auto since = counts.Since();
    Require(since.skipped == 0 && since.partial + since.full == 1, "resident staging: the use after a driver store over its region counted " + Text(since) + ", expected a copy-in");
}

void HalfOverlap(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto low = Compile(device, AddOne, block.Address(), BufferBytes);
    const auto high = Compile(device, AddOne, block.Address() + BufferBytes / 2u, BufferBytes);
    for (std::uint32_t round = 0; round < 3; ++round) {
        Run(device, low, AddOne);
        Land(device);
        Run(device, high, AddOne);
        Land(device);
    }
    device.WaitIdle();
    for (std::uint32_t lane = 0; lane < Lanes + Lanes / 2u; ++lane) {
        const std::uint32_t expected = lane < Lanes / 2u || lane >= Lanes ? 3u : 6u;
        Require(block.Words()[lane] == expected, "resident staging, two half-overlapping regions: dword " + std::to_string(lane) + " is " + std::to_string(block.Words()[lane]) + ", expected " + std::to_string(expected));
    }
}

void Verified(AgcDriver::VulkanDevice& device, bool strict) {
    AgcDriver::Graphics::SetResidentStagingForTests(true, true);
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), BufferBytes);
    const Counts counts;
    for (std::uint32_t round = 0; round < 4; ++round) {
        Run(device, addOne, AddOne);
        device.WaitIdle();
    }
    AgcDriver::Graphics::SetResidentStagingForTests(true, false);
    Expect(block, 4, "four verified uses");
    const auto since = counts.Since();
    Require(since.differing == 0, "resident staging: " + std::to_string(since.differing) + " of " + std::to_string(since.verified) + " verified shadows differed from their import");
    if (strict) Require(since.verified == 3, "resident staging: four uses verified " + std::to_string(since.verified) + " shadows, expected the three kept ones");
}

void SwitchedOff(AgcDriver::VulkanDevice& device) {
    AgcDriver::Graphics::SetResidentStagingForTests(false, false);
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), BufferBytes);
    const Counts counts;
    Run(device, addOne, AddOne);
    Land(device);
    Run(device, addOne, AddOne);
    device.WaitIdle();
    AgcDriver::Graphics::SetResidentStagingForTests(true, false);
    Expect(block, 2, "two uses with resident staging off");
    const auto since = counts.Since();
    Require(since.made + since.shared + since.skipped + since.partial + since.full == 0, "resident staging: a build made with the switch off counted " + Text(since));
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    Require(_putenv_s(name, value) == 0, std::string("resident staging: cannot set ") + name);
#else
    Require(setenv(name, value, 1) == 0, std::string("resident staging: cannot set ") + name);
#endif
}

}

int main() {
    try {
        SetEnvironment("APS5_WRITTEN_SHADOW_MIN_KIB", "0");
        SetEnvironment("APS5_NO_CHANGED_COPY_BACK", "1");
        SetEnvironment("APS5_NO_ADAPTIVE_STAGING", "1");
        AgcDriver::Graphics::SetResidentStagingForTests(true, false);
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        struct ReleaseBlocks {
            AgcDriver::VulkanDevice& device;
            ~ReleaseBlocks() {
                try {
                    device.WaitIdle();
                } catch (const std::exception& error) {
                    std::cerr << error.what() << '\n';
                }
                blocks.clear();
            }
        } releaseBlocks{*device};
        if (!AgcDriver::GuestMemory::WriteWatched()) {
            std::puts("skipped, guest memory is not write-watched, so no shadow can be proved current");
            return VulkanTestSkipped;
        }
        const auto probe = ProbeResident(*device);
        if (probe == Probe::Unstaged) {
            device->WaitIdle();
            std::puts("skipped, the device stages no written buffer in a shared device-local shadow");
            return VulkanTestSkipped;
        }
        const bool strict = probe == Probe::Resident;
        if (!strict) std::puts("resident staging: the write tracking sees this device's copy-backs as CPU stores, so only the results are checked");
        SteadyState(*device, strict);
        std::puts("resident staging: an untouched region was copied in once over eight uses");
        CpuStoreBetween(*device, strict);
        std::puts("resident staging: CPU stores between two uses reached the next one");
        FillBetween(*device);
        std::puts("resident staging: a fill between two uses reached the next one");
        DriverStoreBetween(*device);
        std::puts("resident staging: a driver store between two uses reached the next one");
        HalfOverlap(*device);
        std::puts("resident staging: two half-overlapping regions kept every increment");
        Verified(*device, strict);
        std::puts("resident staging: verified shadows matched their imports");
        SwitchedOff(*device);
        std::puts("resident staging: a build made with the switch off kept the per-use copy-in");
        device->WaitIdle();
        std::printf("resident staging: %s\n", Text(AgcDriver::Graphics::ResidentStagingCounters()).c_str());
        std::puts("resident staging tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
