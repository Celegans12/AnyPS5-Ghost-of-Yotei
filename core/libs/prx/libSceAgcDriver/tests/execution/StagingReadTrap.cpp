#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
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

#ifdef _WIN32

namespace {

using AgcDriver::Graphics::Recorder;
using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::size_t BlockBytes = 65536;
constexpr std::size_t PageBytes = 4096;
constexpr std::uint32_t Lanes = 64;
constexpr std::uint32_t RegionBytes = 4u * PageBytes;

alignas(256) constexpr std::array<std::uint32_t, 8> AddOne{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a040481u, 0xe0701000u, 0x80000201u, 0xbf810000u};
alignas(256) constexpr std::array<std::uint32_t, 8> AddTwo{0x34020082u, 0xe0301000u, 0x80000201u, 0xbf8c3f70u, 0x4a040482u, 0xe0701000u, 0x80000201u, 0xbf810000u};

LONG CALLBACK ServeFault(EXCEPTION_POINTERS* info) {
    return GuestArena::GuestArenaRecoverRead_nid_no_patch(info) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

struct Block {
    Block() {
        data = static_cast<std::uint8_t*>(GuestArena::GuestArenaAllocate_nid_postfix(BlockBytes, BlockBytes));
        Require(data != nullptr, "staging read trap: cannot reserve a block");
        const auto section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE, 0, static_cast<DWORD>(BlockBytes), nullptr);
        Require(section != nullptr, "staging read trap: cannot create a section");
        GuestArena::GuestArenaMap_nid_postfix(data, BlockBytes, section, 0, PAGE_READWRITE);
        CloseHandle(section);
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
        GuestArena::GuestArenaReset_nid_postfix(data, BlockBytes);
        GuestArena::GuestArenaRelease_nid_postfix(data, BlockBytes);
    }

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;

    std::uint64_t Address() const { return reinterpret_cast<std::uintptr_t>(data); }
    volatile std::uint32_t* Words() const { return reinterpret_cast<volatile std::uint32_t*>(data); }

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

void Submit(AgcDriver::VulkanDevice& device) {
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    device.SubmitRecorded(false);
}

void Expect(const Block& block, std::uint32_t expected, const std::string& what, std::uint32_t skip = Lanes) {
    for (std::uint32_t lane = 0; lane < Lanes; ++lane) {
        if (lane == skip) continue;
        const std::uint32_t found = block.Words()[lane];
        Require(found == expected, "staging read trap, " + what + ": dword " + std::to_string(lane) + " is " + std::to_string(found) + ", expected " + std::to_string(expected));
    }
}

struct Counts {
    Recorder::ReadTrapStatistics before = Recorder::ReadTrapCounts();

    Recorder::ReadTrapStatistics Since() const {
        auto now = Recorder::ReadTrapCounts();
        now.guarded -= before.guarded;
        now.unguarded -= before.unguarded;
        now.replaced -= before.replaced;
        now.faults -= before.faults;
        now.nestedFaults -= before.nestedFaults;
        now.forcedReleases -= before.forcedReleases;
        now.faultLanded -= before.faultLanded;
        now.readerLanded -= before.readerLanded;
        now.verified -= before.verified;
        now.differing -= before.differing;
        return now;
    }
};

std::string Text(const Recorder::ReadTrapStatistics& counts) {
    return std::to_string(counts.guarded) + " guarded, " + std::to_string(counts.unguarded) + " unguarded, " + std::to_string(counts.replaced) + " replaced, " + std::to_string(counts.faults) + " faults (" + std::to_string(counts.nestedFaults) + " under the GPU lock, " + std::to_string(counts.forcedReleases) + " released by hand), " + std::to_string(counts.faultLanded) + " landed for a CPU access, " + std::to_string(counts.readerLanded) + " ahead of recorded GPU work, " + std::to_string(counts.held) + " ranges trapped";
}

bool Trapped(const Block& block) {
    return GuestArena::GuestArenaReadsTrapped_nid_no_patch(static_cast<std::uintptr_t>(block.Address()), RegionBytes);
}

enum class Probe { Unstaged, Unguarded, Guarded };

Probe ProbeGuard(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    const Counts counts;
    Run(device, addOne, AddOne);
    Submit(device);
    const auto since = counts.Since();
    const bool trapped = Trapped(block);
    device.WaitIdle();
    Expect(block, 1, "one use");
    Require(!Trapped(block), "staging read trap: the region stayed trapped after the device went idle");
    if (since.guarded + since.unguarded == 0) return Probe::Unstaged;
    if (since.guarded == 0) return Probe::Unguarded;
    Require(trapped, "staging read trap: a guarded copy-back left its region readable after the submit");
    return Probe::Guarded;
}

void NeverRead(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    const Counts counts;
    for (std::uint32_t round = 0; round < 4; ++round) {
        Run(device, addOne, AddOne);
        Submit(device);
    }
    const auto kept = counts.Since();
    Require(Trapped(block), "staging read trap: four uses left their region readable");
    Require(kept.guarded == 4 && kept.faults == 0 && kept.faultLanded == 0 && kept.readerLanded == 0 && kept.replaced == 3, "staging read trap: four unread uses counted " + Text(kept) + ", expected four guarded copy-backs, three of them replaced and none landed");
    Expect(block, 4, "the first CPU read after four uses");
    const auto read = counts.Since();
    Require(read.faults >= 1 && read.faultLanded == 1 && read.nestedFaults == 0 && read.forcedReleases == 0, "staging read trap: the first CPU read counted " + Text(read) + ", expected one copy-back landed for it");
    Require(!Trapped(block), "staging read trap: the region stayed trapped after its read was served");
    device.WaitIdle();
    Expect(block, 4, "four uses read once");
}

void CpuStoreWhileKept(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    Run(device, addOne, AddOne);
    Submit(device);
    Run(device, addOne, AddOne);
    Submit(device);
    const Counts counts;
    block.Words()[5] = 100;
    const auto stored = counts.Since();
    Require(stored.faults >= 1 && stored.faultLanded == 1, "staging read trap: a CPU store over a kept result counted " + Text(stored) + ", expected its copy-back landed first");
    Require(block.Words()[5] == 100 && block.Words()[6] == 2, "staging read trap: a CPU store over a kept result left dword 5 at " + std::to_string(block.Words()[5]) + " and dword 6 at " + std::to_string(block.Words()[6]));
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Require(block.Words()[5] == 101, "staging read trap: the use after a CPU store read " + std::to_string(block.Words()[5] - 1u) + ", expected 100");
    Expect(block, 3, "a use after a CPU store over a kept result", 5);
}

void DriverStoreWhileKept(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    Run(device, addOne, AddOne);
    Submit(device);
    const Counts counts;
    std::array<std::uint32_t, Lanes> stored;
    stored.fill(40);
    AgcDriver::GuestMemory::Write(block.Address(), std::as_bytes(std::span(stored)));
    const auto since = counts.Since();
    Require(since.nestedFaults == 0 && since.forcedReleases == 0, "staging read trap: a driver store over a kept result counted " + Text(since));
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Expect(block, 41, "a driver store over a kept result");
}

void FillWhileKept(AgcDriver::VulkanDevice& device) {
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    const auto addTwo = Compile(device, AddTwo, block.Address(), RegionBytes);
    Run(device, addOne, AddOne);
    Submit(device);
    Run(device, addOne, AddOne);
    Submit(device);
    const Counts counts;
    {
        std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
        const std::array<std::uint32_t, 4> pattern{7u, 7u, 7u, 7u};
        Require(device.FillBuffer(block.Address(), Lanes * 4u, pattern), "staging read trap: the fill over the kept region was not recorded on the GPU");
    }
    const auto since = counts.Since();
    Require(since.readerLanded + since.faultLanded == 1 && since.faults == 0, "staging read trap: a fill over a kept result counted " + Text(since) + ", expected its copy-back landed ahead of the fill");
    Run(device, addTwo, AddTwo);
    device.WaitIdle();
    Expect(block, 9, "a fill over a kept result");
}

void Verified(AgcDriver::VulkanDevice& device) {
    Recorder::SetReadTrapForTests(true, true);
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    const Counts counts;
    for (std::uint32_t round = 0; round < 3; ++round) {
        Run(device, addOne, AddOne);
        Submit(device);
    }
    Expect(block, 3, "the first CPU read in verified mode");
    device.WaitIdle();
    Recorder::SetReadTrapForTests(true, false);
    const auto since = counts.Since();
    Require(since.verified >= 1 && since.differing == 0, "staging read trap: verified mode compared " + std::to_string(since.verified) + " kept results, " + std::to_string(since.differing) + " of them differing from guest memory");
    Require(since.nestedFaults == 0 && since.forcedReleases == 0, "staging read trap: verified mode counted " + Text(since));
}

void SwitchedOff(AgcDriver::VulkanDevice& device) {
    Recorder::SetReadTrapForTests(false, false);
    const auto& block = NewBlock();
    const auto addOne = Compile(device, AddOne, block.Address(), RegionBytes);
    const Counts counts;
    Run(device, addOne, AddOne);
    Submit(device);
    const bool trapped = Trapped(block);
    Run(device, addOne, AddOne);
    device.WaitIdle();
    Recorder::SetReadTrapForTests(true, false);
    Expect(block, 2, "two uses with the read trap off");
    const auto since = counts.Since();
    Require(!trapped && since.guarded + since.unguarded + since.faults == 0, "staging read trap: a use recorded with the switch off counted " + Text(since));
}

void SetEnvironment(const char* name, const char* value) {
    Require(_putenv_s(name, value) == 0, std::string("staging read trap: cannot set ") + name);
}

}

int main() {
    try {
        SetEnvironment("APS5_WRITTEN_SHADOW_MIN_KIB", "0");
        SetEnvironment("APS5_NO_CHANGED_COPY_BACK", "1");
        SetEnvironment("APS5_NO_ADAPTIVE_STAGING", "1");
        Require(AddVectoredExceptionHandler(1, ServeFault) != nullptr, "staging read trap: cannot install the fault handler");
        AgcDriver::Graphics::SetResidentStagingForTests(true, false);
        Recorder::SetReadTrapForTests(true, false);
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
        const auto probe = ProbeGuard(*device);
        if (probe == Probe::Unstaged) {
            std::puts("skipped, the device stages no written buffer of a shared view in a resident shadow");
            return VulkanTestSkipped;
        }
        if (probe == Probe::Unguarded) {
            std::printf("skipped, the read trap refused the staged region of a shared view: %s\n", Text(Recorder::ReadTrapCounts()).c_str());
            return VulkanTestSkipped;
        }
        NeverRead(*device);
        std::puts("staging read trap: four unread uses copied nothing back until the first CPU read");
        CpuStoreWhileKept(*device);
        std::puts("staging read trap: a CPU store over a kept result landed after it");
        DriverStoreWhileKept(*device);
        std::puts("staging read trap: a driver store over a kept result landed after it");
        FillWhileKept(*device);
        std::puts("staging read trap: a fill over a kept result landed after it");
        Verified(*device);
        std::puts("staging read trap: verified kept results matched guest memory");
        SwitchedOff(*device);
        std::puts("staging read trap: a use recorded with the switch off kept its copy-back");
        device->WaitIdle();
        std::printf("staging read trap: %s\n", Text(Recorder::ReadTrapCounts()).c_str());
        std::puts("staging read trap tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#else

int main() {
    std::puts("skipped, the staging read trap is Windows only");
    return VulkanTestSkipped;
}

#endif
