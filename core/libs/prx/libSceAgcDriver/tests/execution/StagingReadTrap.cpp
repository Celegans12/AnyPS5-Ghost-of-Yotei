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
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Site = AgcDriver::Graphics::Recorder::LandingSite;
using Step = AgcDriver::Graphics::Recorder::CopyBackStep;
using Landing = AgcDriver::Graphics::Recorder::CopyBackLanding;

void Hold(bool value, const std::string& what) {
    if (!value) throw std::runtime_error("staging read trap state machine: " + what);
}

struct FakeRecorder {
    struct Entry {
        std::uint64_t range;
        bool guarded;
        bool queuedStore;
        bool landed;
        std::uint32_t result;
    };

    bool verified = false;
    std::vector<Entry> pending;
    std::map<std::uint64_t, std::uint32_t> guest;
    std::set<std::uint64_t> trapped;
    unsigned copies = 0, forAccess = 0, forReader = 0, compared = 0, differing = 0, replaced = 0;

    void Defer(std::uint64_t range, bool guarded, std::uint32_t result, bool queuedStore = false) {
        if (guarded) trapped.insert(range);
        const auto same = std::find_if(pending.begin(), pending.end(), [&](const Entry& held) { return held.range == range; });
        if (same == pending.end()) {
            pending.push_back({range, guarded, queuedStore, false, result});
            return;
        }
        if (AgcDriver::Graphics::Recorder::ReplacedUncopied(same->guarded, same->landed)) ++replaced;
        *same = {range, guarded, queuedStore, false, result};
    }

    void At(Site site) { reach(site, false, {}); }
    void At(Site site, std::initializer_list<std::uint64_t> touched) { reach(site, true, touched); }
    void Access(std::uint64_t range) { reach(Site::ReadTrap, true, {range}); }
    bool Trapped(std::uint64_t range) const { return trapped.contains(range); }
    bool Pending(std::uint64_t range) const { return std::any_of(pending.begin(), pending.end(), [&](const Entry& held) { return held.range == range; }); }

private:
    void reach(Site site, bool scoped, std::initializer_list<std::uint64_t> touched) {
        for (auto held = pending.begin(); held != pending.end();) {
            const bool reached = std::find(touched.begin(), touched.end(), held->range) != touched.end();
            const auto step = AgcDriver::Graphics::Recorder::StepCopyBack(site, scoped, reached, held->guarded, held->guarded && held->queuedStore, verified, held->landed);
            if (step == Step::Lands) {
                land(*held, site);
                held = pending.erase(held);
                continue;
            }
            if (step == Step::StaysCopied) {
                held->landed = true;
                guest[held->range] = held->result;
                ++copies;
            }
            ++held;
        }
    }

    void land(const Entry& entry, Site site) {
        const auto kind = AgcDriver::Graphics::Recorder::LandingOf(site, entry.guarded, entry.landed);
        if (kind == Landing::Compared) {
            ++compared;
            if (guest[entry.range] != entry.result) ++differing;
        } else {
            guest[entry.range] = entry.result;
            ++copies;
            if (kind == Landing::CopiedForAccess) ++forAccess;
            if (kind == Landing::CopiedForReader) ++forReader;
        }
        if (entry.guarded) trapped.erase(entry.range);
    }
};

void StateMachine() {
    constexpr std::uint64_t zone = 0x200000000ull, other = 0x200600000ull;
    constexpr std::array<Site, 4> keeping{Site::LabelStore, Site::StoreRun, Site::Submit, Site::KeptBytes};
    for (std::size_t index = 0; index < AgcDriver::Graphics::Recorder::LandingSites; ++index) {
        const auto site = static_cast<Site>(index);
        const std::string name = AgcDriver::Graphics::Recorder::LandingSiteName(site);
        const bool keeps = std::find(keeping.begin(), keeping.end(), site) != keeping.end();
        Hold(AgcDriver::Graphics::Recorder::KeepsCopyBacks(site) == keeps, name + " keeps guarded results or not, against the four keeping sites");
        for (unsigned bits = 0; bits < 16; ++bits) {
            const bool guarded = (bits & 1u) != 0, queued = (bits & 2u) != 0, verified = (bits & 4u) != 0, landed = (bits & 8u) != 0;
            Hold(AgcDriver::Graphics::Recorder::StepCopyBack(site, true, true, guarded, queued, verified, landed) == Step::Lands, name + ": a result the recorded work reaches does not land");
            Hold(AgcDriver::Graphics::Recorder::StepCopyBack(site, true, false, guarded, queued, verified, landed) == Step::Stays, name + ": a result the recorded work does not reach does not stay");
            const auto unscoped = AgcDriver::Graphics::Recorder::StepCopyBack(site, false, false, guarded, queued, verified, landed);
            const auto expected = !keeps || !guarded || queued ? Step::Lands : verified && !landed ? Step::StaysCopied : Step::Stays;
            Hold(unscoped == expected, name + ": a result met with no scope takes the wrong step, case " + std::to_string(bits));
        }
        Hold(AgcDriver::Graphics::Recorder::LandingOf(site, false, false) == Landing::Copied && AgcDriver::Graphics::Recorder::LandingOf(site, false, true) == Landing::Copied, name + ": an unguarded result is not plainly copied");
        Hold(AgcDriver::Graphics::Recorder::LandingOf(site, true, true) == Landing::Compared, name + ": a guarded result already copied is not compared");
        Hold(AgcDriver::Graphics::Recorder::LandingOf(site, true, false) == (site == Site::ReadTrap ? Landing::CopiedForAccess : Landing::CopiedForReader), name + ": a guarded result lands for the wrong reader");
    }
    Hold(AgcDriver::Graphics::Recorder::ReplacedUncopied(true, false) && !AgcDriver::Graphics::Recorder::ReplacedUncopied(true, true) && !AgcDriver::Graphics::Recorder::ReplacedUncopied(false, false) && !AgcDriver::Graphics::Recorder::ReplacedUncopied(false, true), "only a guarded result not yet copied counts as replaced without a copy");

    {
        FakeRecorder recorder;
        for (std::uint32_t use = 1; use <= 4; ++use) {
            recorder.Defer(zone, true, use);
            recorder.At(Site::LabelStore);
            recorder.At(Site::StoreRun);
            recorder.At(Site::KeptBytes);
            recorder.At(Site::Submit);
        }
        Hold(recorder.copies == 0 && recorder.replaced == 3 && recorder.Trapped(zone) && recorder.Pending(zone) && !recorder.guest.contains(zone), "four unread uses copied a result back or lost their trap");
        recorder.Access(other);
        Hold(recorder.copies == 0 && recorder.Trapped(zone), "a CPU access to another range landed the guarded result");
        recorder.Access(zone);
        Hold(recorder.copies == 1 && recorder.forAccess == 1 && recorder.forReader == 0 && recorder.guest[zone] == 4 && !recorder.Trapped(zone) && !recorder.Pending(zone), "the first CPU access did not land the last result once and release the trap");
        recorder.Access(zone);
        Hold(recorder.copies == 1, "a second CPU access copied again");
    }
    for (const auto site : {Site::Command, Site::DrawPass, Site::Samples, Site::Sync, Site::StagingUnknown, Site::DispatchUnknown}) {
        FakeRecorder recorder;
        recorder.Defer(zone, true, 7);
        recorder.Defer(other, true, 9);
        recorder.At(Site::LabelStore);
        recorder.At(site);
        Hold(recorder.copies == 2 && recorder.forReader == 2 && recorder.forAccess == 0 && recorder.guest[zone] == 7 && recorder.guest[other] == 9 && recorder.trapped.empty() && recorder.pending.empty(), std::string("recorded GPU work of unknown reach (") + AgcDriver::Graphics::Recorder::LandingSiteName(site) + ") did not land every guarded result ahead of itself");
    }
    for (const auto site : {Site::StagingReach, Site::DispatchReach, Site::Overlap}) {
        FakeRecorder recorder;
        recorder.Defer(zone, true, 7);
        recorder.Defer(other, true, 9);
        recorder.At(site, {other});
        Hold(recorder.copies == 1 && recorder.forReader == 1 && recorder.guest[other] == 9 && recorder.Trapped(zone) && !recorder.Trapped(other) && recorder.Pending(zone), std::string("recorded GPU work of known reach (") + AgcDriver::Graphics::Recorder::LandingSiteName(site) + ") did not land exactly the result it reads");
        recorder.Defer(zone, true, 8);
        Hold(recorder.replaced == 1 && recorder.copies == 1, "a newer result over a kept one was not a replacement without a copy");
    }
    {
        FakeRecorder recorder;
        recorder.Defer(zone, false, 3);
        recorder.Defer(other, true, 5, true);
        recorder.At(Site::LabelStore);
        Hold(recorder.copies == 2 && recorder.guest[zone] == 3 && recorder.guest[other] == 5 && recorder.pending.empty() && recorder.trapped.empty() && recorder.forAccess == 0, "an unguarded result or one under a queued store was kept past a label");
    }
    {
        FakeRecorder recorder;
        recorder.verified = true;
        recorder.Defer(zone, true, 1);
        recorder.At(Site::LabelStore);
        Hold(recorder.copies == 1 && recorder.guest[zone] == 1 && recorder.Trapped(zone) && recorder.Pending(zone), "verified mode did not copy at the label and keep the trap");
        recorder.At(Site::Submit);
        recorder.At(Site::LabelStore);
        Hold(recorder.copies == 1, "verified mode copied a kept result twice");
        recorder.Defer(zone, true, 2);
        Hold(recorder.replaced == 0, "verified mode counted a copied result as replaced without a copy");
        recorder.At(Site::LabelStore);
        recorder.Access(zone);
        Hold(recorder.copies == 2 && recorder.compared == 1 && recorder.differing == 0 && !recorder.Trapped(zone) && recorder.forAccess == 0, "verified mode did not compare the kept result with guest memory at the access");
        recorder.Defer(zone, true, 3);
        recorder.At(Site::LabelStore);
        recorder.guest[zone] = 99;
        recorder.At(Site::DrawPass);
        Hold(recorder.compared == 2 && recorder.differing == 1, "verified mode missed guest memory that differs from the kept result");
    }
}

}

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
        StateMachine();
        std::puts("staging read trap: a guarded result stays past labels and submits, lands once for a CPU access or ahead of GPU work that reaches it, and is replaced without a copy");
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
    try {
        StateMachine();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::puts("staging read trap: a guarded result stays past labels and submits, lands once for a CPU access or ahead of GPU work that reaches it, and is replaced without a copy");
    std::puts("skipped past the state machine, the staging read trap is Windows only");
    return VulkanTestSkipped;
}

#endif
