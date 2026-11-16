#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTimer.hpp"
#include "prx/libSceAgcDriver/Execution/include/DispatchIoTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchLater.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/BufferCopy.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DwordPatternFill.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "ThreadOwned.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

namespace {

bool DispatchItemsEnabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_PIPELINE_DISPATCH");
        return text != nullptr && text[0] != '\0' && text[0] != '0';
    }();
    return enabled;
}

enum class ItemFallback : std::size_t { Miss, Reads, NoRecipe, Writes, Verify, Count };
constexpr std::array<const char*, static_cast<std::size_t>(ItemFallback::Count)> ItemFallbackNames{"miss", "reads", "no recipe", "writes", "verify"};
std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(ItemFallback::Count)> itemFallbacks{};
std::atomic<std::uint64_t> itemCommits{0};
std::atomic<std::uint64_t> itemRebuilds{0};
std::atomic<std::uint64_t> itemMisses{0};
std::atomic<std::uint64_t> itemRecipeLess{0};
std::atomic<std::uint64_t> itemNotPlain{0};

bool DispatchMissItemsEnabled() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::PipelineDispatchMisses);
}

void NoteItemFallback(ItemFallback reason) {
    itemFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void ReportDispatchItems(bool profile) {
    static auto lastReport = std::chrono::steady_clock::now();
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    std::string drained;
    for (std::size_t i = 0; i < ItemFallbackNames.size(); ++i) {
        char item[48];
        std::snprintf(item, sizeof(item), " %s %llu", ItemFallbackNames[i], static_cast<unsigned long long>(itemFallbacks[i].load(std::memory_order_relaxed)));
        drained += item;
    }
    std::fprintf(stderr, "[dispatch-item] %llu commit items (%llu recipes rebuilt on the committer, %llu queued after a cache miss, %llu queued without a recipe, %llu cached ones kept off the queue as not plain); drained instead, cumulative:%s\n", static_cast<unsigned long long>(itemCommits.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemRebuilds.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemMisses.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemRecipeLess.load(std::memory_order_relaxed)), static_cast<unsigned long long>(itemNotPlain.load(std::memory_order_relaxed)), drained.c_str());
}

bool CapturedCurrent(std::span<const ShaderRecompiler::MemoryRegion> captured, const ShaderSnapshot& code) {
    for (const auto& region : captured) {
        const auto address = region.guestAddress;
        const auto bytes = region.bytes.size();
        if (bytes == 0 || (address >= code.codeAddress && address < code.codeAddress + code.code.size() * sizeof(std::uint32_t)) || (!code.header.empty() && address >= code.headerAddress && address < code.headerAddress + code.header.size())) continue;
        if (Driver::QueryPendingSpan(address, bytes, {}) != ShaderMemory::PendingWrite::None || !GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) return false;
        GuestMemory::GuardRead(address, bytes);
        GuestMemory::FlushGpuWrites(address, bytes);
        if (std::memcmp(region.bytes.data(), reinterpret_cast<const void*>(address), bytes) != 0) return false;
    }
    return true;
}

struct LateDispatch {
    Registers shader;
    std::array<std::uint32_t, 5> packet{};
    Submission handed{};
    std::shared_ptr<const ShaderSnapshot> registeredShader;
    std::uint64_t epoch = 0;
    std::shared_ptr<ShaderMemory> shadowMemory;
    std::vector<ShaderRecompiler::MemoryRegion> shadow;
    bool shadowed = false;
    bool compared = false;
};

LateDispatch*& RunningLate() {
    static thread_local LateDispatch* running = nullptr;
    return running;
}

std::atomic<std::uint64_t> laterHanded{0}, laterKept{0}, laterKeptKernels{0}, laterQueued{0}, laterInlined{0}, laterVerified{0}, laterDiffering{0}, laterUnverified{0}, laterNs{0};

bool LaterSwitches() {
    static const bool allowed = DispatchLater() && std::getenv("APS5_PROFILE_DRAW") == nullptr && std::getenv("APS5_NO_DISPATCH_CACHE") == nullptr && std::getenv("APS5_PROBE_DISPATCH") == nullptr && std::getenv("APS5_TRACE_DISPATCH_CACHE") == nullptr && std::getenv("APS5_DUMP_SHADERS") == nullptr && std::getenv("APS5_NO_DISPATCH_KEY_HYGIENE") == nullptr;
    return allowed;
}

void ReportDispatchLater() {
    static const bool stats = std::getenv("APS5_PIPELINE_STATS") != nullptr || std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!stats) return;
    static std::atomic<std::int64_t> reportedAt{0};
    const auto tenSeconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count() / 10;
    if (reportedAt.exchange(tenSeconds, std::memory_order_relaxed) == tenSeconds) return;
    static DispatchLaterCounts reported;
    const auto counted = DispatchLaterCounted();
    const auto since = [](std::uint64_t now, std::uint64_t before) { return static_cast<unsigned long long>(now - before); };
    const auto handed = counted.handed - reported.handed;
    AgcDriver::ProfilePrint_nid_no_patch("[dispatch-later] (10 s): %llu direct dispatches handed to the draw finisher with their lookup and capture, %llu kept on the worker (%llu of them by a kernel the worker handles itself); on the finisher %llu queued as commit items, %llu dispatched there after a drain of the committer, %.2f ms resolving (%.1f us each); %llu captures repeated on the worker (APS5_VERIFY_DISPATCH_LATER), %llu differing, %llu not repeated\n",
                                         static_cast<unsigned long long>(handed), since(counted.kept, reported.kept), since(counted.keptKernels, reported.keptKernels), since(counted.queued, reported.queued), since(counted.inlined, reported.inlined),
                                         static_cast<double>(counted.finisherNs - reported.finisherNs) / 1e6, handed != 0 ? static_cast<double>(counted.finisherNs - reported.finisherNs) / 1e3 / static_cast<double>(handed) : 0.0,
                                         since(counted.verified, reported.verified), since(counted.differing, reported.differing), since(counted.unverified, reported.unverified));
    reported = counted;
}

bool InCodeOf(const ShaderRecompiler::MemoryRegion& region, const ShaderSnapshot& code) {
    const auto address = region.guestAddress;
    return (address >= code.codeAddress && address < code.codeAddress + code.code.size() * sizeof(std::uint32_t)) || (!code.header.empty() && address >= code.headerAddress && address < code.headerAddress + code.header.size());
}

struct ByteRun {
    std::uint64_t begin = 0;
    std::vector<std::byte> bytes;
};

std::vector<ByteRun> ByteRuns(std::span<const ShaderRecompiler::MemoryRegion> regions, const ShaderSnapshot& code) {
    std::vector<const ShaderRecompiler::MemoryRegion*> sorted;
    sorted.reserve(regions.size());
    for (const auto& region : regions) {
        if (!region.bytes.empty() && !InCodeOf(region, code)) sorted.push_back(&region);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto* first, const auto* second) { return first->guestAddress < second->guestAddress; });
    std::vector<ByteRun> runs;
    for (const auto* region : sorted) {
        if (runs.empty() || region->guestAddress > runs.back().begin + runs.back().bytes.size()) {
            runs.push_back({region->guestAddress, std::vector<std::byte>(region->bytes.begin(), region->bytes.end())});
            continue;
        }
        auto& run = runs.back();
        const auto offset = static_cast<std::size_t>(region->guestAddress - run.begin);
        if (offset + region->bytes.size() > run.bytes.size()) run.bytes.insert(run.bytes.end(), region->bytes.begin() + static_cast<std::ptrdiff_t>(run.bytes.size() - offset), region->bytes.end());
    }
    return runs;
}

struct ShadowMismatch {
    std::uint64_t address = 0;
    bool covered = false;
};

std::optional<ShadowMismatch> ShadowDiffers(const std::vector<ByteRun>& shadow, const std::vector<ByteRun>& seen) {
    for (const auto& run : shadow) {
        for (std::size_t at = 0; at < run.bytes.size();) {
            const auto address = run.begin + at;
            auto found = std::upper_bound(seen.begin(), seen.end(), address, [](std::uint64_t value, const ByteRun& candidate) { return value < candidate.begin; });
            if (found == seen.begin()) return ShadowMismatch{address, false};
            --found;
            const auto offset = static_cast<std::size_t>(address - found->begin);
            if (offset >= found->bytes.size()) return ShadowMismatch{address, false};
            const auto length = std::min(run.bytes.size() - at, found->bytes.size() - offset);
            for (std::size_t i = 0; i < length; ++i) {
                if (run.bytes[at + i] != found->bytes[offset + i]) return ShadowMismatch{address + i, true};
            }
            at += length;
        }
    }
    return std::nullopt;
}

void CompareShadow(LateDispatch& late, std::span<const ShaderRecompiler::MemoryRegion> captured, const ShaderSnapshot& code, std::uint64_t address) {
    if (!late.shadowed || std::exchange(late.compared, true)) return;
    laterVerified.fetch_add(1, std::memory_order_relaxed);
    const auto mismatch = ShadowDiffers(ByteRuns(late.shadow, code), ByteRuns(captured, code));
    if (!mismatch) return;
    static std::atomic<std::uint64_t> printed{0};
    laterDiffering.fetch_add(1, std::memory_order_relaxed);
    if (printed.fetch_add(1, std::memory_order_relaxed) < 16) std::fprintf(stderr, "[dispatch-later] verify: the dispatch of compute shader 0x%llx at submission %llu read 0x%llx on the worker, which the draw finisher %s\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(late.handed.serial), static_cast<unsigned long long>(mismatch->address), mismatch->covered ? "read with another value" : "did not read");
}

}

bool DispatchLater() {
    static const bool enabled = std::getenv("APS5_DISPATCH_LATER") != nullptr || VerifyDispatchLater();
    return enabled;
}

bool VerifyDispatchLater() {
    static const bool enabled = std::getenv("APS5_VERIFY_DISPATCH_LATER") != nullptr;
    return enabled;
}

DispatchLaterCounts DispatchLaterCounted() {
    DispatchLaterCounts counts;
    counts.handed = laterHanded.load(std::memory_order_relaxed);
    counts.kept = laterKept.load(std::memory_order_relaxed);
    counts.keptKernels = laterKeptKernels.load(std::memory_order_relaxed);
    counts.queued = laterQueued.load(std::memory_order_relaxed);
    counts.inlined = laterInlined.load(std::memory_order_relaxed);
    counts.verified = laterVerified.load(std::memory_order_relaxed);
    counts.differing = laterDiffering.load(std::memory_order_relaxed);
    counts.unverified = laterUnverified.load(std::memory_order_relaxed);
    counts.finisherNs = laterNs.load(std::memory_order_relaxed);
    return counts;
}

bool Driver::pipelineDispatch() {
    return DispatchItemsEnabled();
}

void Driver::dispatch(QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission, std::uint64_t indirectArguments) {
    PerformanceTimer timing("Driver.Dispatch");
    auto* const late = RunningLate();
    const bool direct = indirectArguments == 0;
    const auto address = (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20c)) << 8u) | (static_cast<std::uint64_t>(readRegister(queue.shader, 0x20d) & 0xffu) << 40u);
    if (SkippedShader(address) || RefusedDispatch(address)) return;
    auto registeredShader = late != nullptr ? late->registeredShader : RegisteredProgram(*submission.shaders, address, {0});
    if (!registeredShader) registeredShader = ReadRawShader(address);
    const auto& snapshot = *registeredShader;
    require(snapshot.type == 0, "compute program refers to a non-compute shader");
    const auto userCount = (readRegister(queue.shader, 0x213) >> 1u) & 0x1fu;
    std::vector<std::uint32_t> userData;
    for (std::uint32_t i = 0; i < userCount; ++i) {
        userData.push_back(readUserData(queue.shader, 0x240 + i));
    }
    auto compute = Graphics::DecodeComputeStageInfo(queue.shader, snapshot.header);
    const bool resolving = resolvingAhead();
    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});

    static const bool unlockedDevice = std::getenv("APS5_NO_UNLOCKED_DEVICE") == nullptr;
    std::shared_ptr<VulkanDevice> localDevice = unlockedDevice ? device.Load() : nullptr;
    if (localDevice == nullptr) {
        GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
        std::lock_guard gpuLock(GuestMemory::GpuMutex());
        if (device == nullptr) device = std::make_shared<VulkanDevice>();
        localDevice = device;
    }
    const auto codeOffset = static_cast<std::size_t>((address - snapshot.codeAddress) / sizeof(std::uint32_t));
    std::array<std::uint32_t, 5> resolved{};
    ResolvedDispatch ahead;
    bool fromAhead = false;
    auto& groupStats = groupCaptureStats();
    if (!resolving && late == nullptr) {
        auto& parked = resolvedAhead();
        if (const auto found = parked.find(currentPacketOffset()); found != parked.end()) {
            ahead = std::move(found->second);
            parked.erase(found);
            fromAhead = ahead.address == address;
            if (!fromAhead) ++groupStats.addressMismatches;
        }
    }
    if (fromAhead) {
        resolved = ahead.packet;
        packet = resolved;
        indirectArguments = ahead.indirectArguments;
    } else if (indirectArguments != 0 && !resolving && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) {
        indirectSource().path = IndirectFillKernel;
        const auto readStart = std::chrono::steady_clock::now();
        resolved = readDispatchArguments(submission.queue, indirectArguments, packet[4]);
        countIndirect(IndirectFillKernel, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count());
        packet = resolved;
        indirectArguments = 0;
    }
    std::array<std::uint32_t, 3> partialGroups{};
    if (indirectArguments == 0 && (packet[4] & 0x2u) != 0) {
        partialGroups = Pm4::PartialGroupThreads({packet[1], packet[2], packet[3]}, {readRegister(queue.shader, 0x207), readRegister(queue.shader, 0x208), readRegister(queue.shader, 0x209)});
    }
    const bool pipelineCandidate = !resolving && !fromAhead && DispatchItemsEnabled() && DrawPipeline::Active() && indirectArguments == 0 && deferredLabels().labels.empty() && !matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesTableCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) && !MatchesIndexedCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute);
    const bool partialLastGroup = partialGroups != std::array<std::uint32_t, 3>{};
    const bool laterCandidate = late == nullptr && direct && submission.queue == 0 && DispatchLater();
    const bool handLater = laterCandidate && pipelineCandidate && queue.predication.operation == 0 && LaterSwitches() && DispatchMissItemsEnabled() && !stampValidate() && !verifyDataHits() && DrawFinisher::Enabled() && [&] {
        const auto program = std::span(snapshot.code).subspan(codeOffset);
        bool kernel = static_cast<bool>(Graphics::MatchMultisampledSingleClear(program, userData, compute.numThreads));
        kernel = kernel || ((packet[4] & 0x20u) == 0 && static_cast<bool>(Graphics::MatchMemoryValueFill(program, userData, compute.numThreads, {packet[1], packet[2], packet[3]})));
        kernel = kernel || (!partialLastGroup && (MatchDwordPatternFill(packet, program, userData, compute) || MatchConstantDwordFill(packet, program, userData, compute) || MatchClampedGlobalFill(packet, program, userData, compute) || MatchMemoryValueBufferFill(packet, program, userData, compute) || matchesFillKernel(program, userData, compute)));
        kernel = kernel || Graphics::HtileDepthClearAddress(program, userData, compute.numThreads) != 0 || static_cast<bool>(Graphics::MatchHtileBufferMaskClear(program, userData, compute, packet, indirectArguments));
        if (!kernel && !partialLastGroup) {
            auto shaped = compute;
            std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
            if ((packet[4] & 0x20u) != 0) {
                for (std::uint32_t axis = 0; axis < 3; ++axis) {
                    if (packet[1 + axis] % compute.numThreads[axis] != 0) shaped.partialThreads = {packet[1], packet[2], packet[3]};
                    groups[axis] = ThreadDimensionGroups(groups[axis], std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u));
                }
            }
            kernel = static_cast<bool>(MatchSgprPatternFill(packet, program, userData, shaped));
            if (!kernel) {
                if (const auto clear = Graphics::MatchStorageImageClear(program, userData, shaped, groups)) {
                    const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
                    kernel = clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress);
                }
            }
        }
        if (kernel) laterKeptKernels.fetch_add(1, std::memory_order_relaxed);
        return !kernel;
    }();
    if (laterCandidate && !handLater) {
        laterKept.fetch_add(1, std::memory_order_relaxed);
        if (DrawFinisher::Busy() && !DrawFinisher::OnFinisherThread()) {
            DrawFinisher::Get().Drain();
            if (DrawFinisher::Overlap()) DrawFinisher::NoteDrain(DrawFinisher::DrainSite::Packet, true);
        }
    }
    std::optional<std::pair<std::uint64_t, std::size_t>> htileClearCompletion;
    Graphics::HtileMaskClear htileMask;
    if (resolving) {
        if (indirectArguments == 0 && Graphics::MatchMultisampledSingleClear(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads)) return;
        if (!partialLastGroup && matchesFillKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute)) return;
        if (!partialLastGroup && indirectArguments == 0 && (MatchDwordPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchConstantDwordFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchClampedGlobalFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchMemoryValueBufferFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute))) return;
        if (!partialLastGroup && indirectArguments == 0 && (matchesCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesStreamCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesIndexedCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute) || MatchesTableCopyKernel(std::span(snapshot.code).subspan(codeOffset), userData, compute))) return;
    } else if (late != nullptr || handLater) {
        if (DrawPipeline::Active() && !pipelineCandidate) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    } else {
        if (indirectArguments == 0 && Graphics::TakeMultisampledSingleClear(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads)) {
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            if (resolvedAhead().empty()) gateOpened() = true;
            return;
        }
        if (indirectArguments == 0 && (packet[4] & 0x20u) == 0) {
            if (const auto fill = Graphics::MatchMemoryValueFill(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads, {packet[1], packet[2], packet[3]})) Graphics::NoteMemoryValueFill(*fill);
        }
        if (!partialLastGroup && fillBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice)) {
            pendingDispatchPhases().outcome = DispatchOutcome::FillHle;
            if (resolvedAhead().empty()) gateOpened() = true;
            return;
        }
        if (DrawPipeline::Active() && !pipelineCandidate) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
        if (const auto htile = Graphics::HtileDepthClearAddress(std::span(snapshot.code).subspan(codeOffset), userData, compute.numThreads); htile != 0) {
            const auto threads = (packet[4] & 0x20u) != 0 ? static_cast<std::uint64_t>(packet[1]) : static_cast<std::uint64_t>(packet[1]) * compute.numThreads[0];
            const auto nativeBytes = Graphics::HtileDepthClearBytes_nid_no_patch(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments);
            const auto bytes = static_cast<std::size_t>(threads * 16u);
            if (Graphics::NoteHtileDepthClear(htile, bytes, nativeBytes)) htileClearCompletion = std::pair{htile, bytes};
            if (nativeBytes != 0u) htileMask = {htile, nativeBytes, userData[5], userData[4]};
        } else if (const auto buffered = Graphics::MatchHtileBufferMaskClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, packet, indirectArguments)) {
            if (Graphics::NoteHtileDepthClear(buffered->address, buffered->bytes, buffered->bytes)) htileClearCompletion = std::pair{buffered->address, buffered->bytes};
            htileMask = *buffered;
        }
        if (!partialLastGroup && indirectArguments == 0 && copyBuffer(queue, submission.queue, packet, std::span(snapshot.code).subspan(codeOffset), userData, compute, localDevice, address)) {
            pendingDispatchPhases().outcome = DispatchOutcome::CopyHle;
            if (resolvedAhead().empty()) gateOpened() = true;
            return;
        }
    }
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {
        const std::array<std::uint32_t, 3> threads{packet[1], packet[2], packet[3]};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (threads[axis] % compute.numThreads[axis] != 0) compute.partialThreads = threads;
        }
    }
    if (partialLastGroup) compute.partialThreads = partialGroups;
    std::array<std::uint32_t, 3> threadGroup{};
    if (indirectArguments != 0 && (packet[4] & 0x20u) != 0) {
        for (std::uint32_t axis = 0; axis < 3; ++axis) threadGroup[axis] = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
        compute.partialThreads = {~0u, ~0u, ~0u};
    }
    const auto makeRequest = [&] {
        return ShaderRecompiler::RecompileRequest{
            {ShaderRecompiler::ShaderStage::Compute, address, std::span(snapshot.code).subspan(codeOffset), snapshot.headerAddress, snapshot.header},
            {(packet[4] & 0x8000u) != 0 ? 32u : 64u, 0, userData, compute, std::nullopt, std::nullopt, memory, DispatchFloatMode(snapshot, queue.shader)},
            localDevice->ComputeTarget((packet[4] & 0x8000u) != 0 ? 32u : 64u),
            {0, 0, 0, threadGroup != std::array<std::uint32_t, 3>{} ? 0u : 128u}
        };
    };
    if (handLater) {
        auto handed = std::make_shared<LateDispatch>();
        handed->shader = queue.shader;
        std::copy_n(packet.begin(), std::min(packet.size(), handed->packet.size()), handed->packet.begin());
        handed->handed.serial = submission.serial;
        handed->handed.queue = submission.queue;
        handed->handed.shaders = submission.shaders;
        handed->registeredShader = registeredShader;
        handed->epoch = DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        if (VerifyDispatchLater()) {
            const auto shadowCapture = [&] {
                auto shadowRequest = makeRequest();
                handed->shadowMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
                auto invocation = InvocationFor(snapshot, codeOffset, shadowRequest);
                const SampledReadScope sampling(evidenceReads);
                static_cast<void>(CaptureInvocation(*handed->shadowMemory, snapshot, codeOffset, shadowRequest, invocation));
                handed->shadow = handed->shadowMemory->Regions();
            };
            try {
                shadowCapture();
                if (std::any_of(handed->shadow.begin(), handed->shadow.end(), [&](const auto& region) { return !region.bytes.empty() && !InCodeOf(region, snapshot) && DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); })) {
                    DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Capture, 0x15);
                    shadowCapture();
                }
                handed->shadowed = true;
            } catch (const std::exception&) {
                laterUnverified.fetch_add(1, std::memory_order_relaxed);
            }
        }
        laterHanded.fetch_add(1, std::memory_order_relaxed);
        DrawFinisher::Get().Enqueue([this, handed] {
            const struct Spent {
                std::chrono::steady_clock::time_point from = std::chrono::steady_clock::now();
                ~Spent() { laterNs.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count()), std::memory_order_relaxed); }
            } spent;
            DrawPipeline::FollowEpoch(handed->epoch);
            GuestMemory::SetCurrentPacket(0x15, handed->handed.queue);
            struct LateScope {
                bool active;
                LateDispatch* outer;
                explicit LateScope(LateDispatch* running) : active(std::exchange(DrawPipeline::Active(), true)), outer(std::exchange(RunningLate(), running)) {}
                ~LateScope() {
                    DrawPipeline::Active() = active;
                    RunningLate() = outer;
                }
                LateScope(const LateScope&) = delete;
                LateScope& operator=(const LateScope&) = delete;
            } scope{handed.get()};
            static thread_local QueueState* scratchSlot = nullptr;
            auto& scratch = ShaderRecompiler::ThreadOwned(scratchSlot);
            scratch.shader = std::move(handed->shader);
            dispatch(scratch, handed->packet, handed->handed);
        });
        ReportDispatchLater();
        timing.Mark("dispatch_handed_off");
        return;
    }
    auto request = makeRequest();
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    static std::atomic<double> captureMs{0}, keyMs{0}, recompileMs{0}, deviceMs{0};
    static std::atomic<std::uint64_t> cacheHits{0};
    static std::atomic<std::uint64_t> dispatches{0};
    auto lap = std::chrono::steady_clock::now();

    std::array<double, DriverPhaseCount> phaseMs{};
    auto phaseLap = lap;
    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
    if (profile && packetStartedAt() != std::chrono::steady_clock::time_point{}) phaseMs[PhasePrologue] = std::chrono::duration<double, std::milli>(lap - packetStartedAt()).count();

    static const bool noDispatchCacheEnv = std::getenv("APS5_NO_DISPATCH_CACHE") != nullptr;

    static const std::pair<std::uint64_t, std::uint64_t> probeDispatch = [] {
        const char* text = std::getenv("APS5_PROBE_DISPATCH");
        if (text == nullptr) return std::pair<std::uint64_t, std::uint64_t>{0, 0};
        char* end = nullptr;
        const auto probeAddress = std::strtoull(text, &end, 16);
        const auto index = end != nullptr && *end == ':' ? std::strtoull(end + 1, nullptr, 10) : 0ull;
        return std::pair<std::uint64_t, std::uint64_t>{probeAddress, index};
    }();
    bool probeThis = false;

    if (probeDispatch.first != 0 && (address & 0xfffffffffull) == (probeDispatch.first & 0xfffffffffull)) {
        static std::atomic<std::uint64_t> dispatchesSeen{0};
        probeThis = dispatchesSeen.fetch_add(1) == probeDispatch.second;
        if (probeThis) std::fprintf(stderr, "[gpu] probing dispatch %llu of 0x%llx\n", static_cast<unsigned long long>(probeDispatch.second), static_cast<unsigned long long>(address));
    }

    struct ProbeScope {
        bool active;
        explicit ProbeScope(bool active) : active(active) { if (active) ShaderRecompiler::SetDebugProbeActive(true); }
        ~ProbeScope() { if (active) ShaderRecompiler::SetDebugProbeActive(false); }
    } probeScope{probeThis};
    const bool noDispatchCache = noDispatchCacheEnv || probeThis;
    std::uint64_t key = 0xcbf29ce484222325ull;
    const auto mix = [&](std::uint64_t value) {
        key ^= value;
        key *= 0x100000001b3ull;
    };
    mix(address);
    mix(packet[4] & 0x8000u);
    for (const auto threads : compute.partialThreads) mix(threads);
    for (const auto word : userData) mix(word);

    static const bool keyHygiene = std::getenv("APS5_NO_DISPATCH_KEY_HYGIENE") == nullptr;
    if (keyHygiene) {
        for (const auto offset : {0x207u, 0x208u, 0x209u, 0x212u, 0x213u, 0x228u}) {
            const auto found = queue.shader.find(offset);

            mix(found == queue.shader.end() ? (1ull << 32u) : found->second);
        }
    } else {
        for (const auto& [offset, value] : queue.shader) {
            mix(offset);
            mix(value);
        }
    }
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiledResult;

    std::shared_ptr<DispatchVariant> keepVariant;

    std::shared_ptr<DispatchVariant> attachVariant;

    std::shared_ptr<ShaderMemory> shaderMemory;
    std::vector<ShaderRecompiler::MemoryRegion> captured;

    std::vector<std::uint32_t> liveWords;
    bool dataHit = false;
    bool cached = false;
    bool validated = false;

    std::shared_ptr<DispatchEntry> missedEntry;
    bool missedDiffering = false;
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;

    static const bool traceCache = std::getenv("APS5_TRACE_DISPATCH_CACHE") != nullptr;
    if (traceCache) {
        std::lock_guard traceLock(dispatchCacheMutex);
        struct Last { std::vector<std::uint32_t> userData; std::map<std::uint32_t, std::uint32_t> shader; std::uint64_t key; };
        static std::map<std::uint64_t, Last> last;
        static int reports = 0;
        auto& previous = last[address];
        if (previous.key != 0 && previous.key != key && reports < 200) {
            std::string what;
            for (std::size_t i = 0; i < userData.size(); ++i) {
                if (i >= previous.userData.size() || previous.userData[i] != userData[i]) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " user[%zu] %08x->%08x", i, i < previous.userData.size() ? previous.userData[i] : 0u, userData[i]);
                    what += text;
                }
            }
            for (const auto& [offset, value] : queue.shader) {
                const auto old = previous.shader.find(offset);
                if (old == previous.shader.end() || old->second != value) {
                    char text[48];
                    std::snprintf(text, sizeof(text), " sh[%x] %08x->%08x", offset, old == previous.shader.end() ? 0u : old->second, value);
                    what += text;
                }
            }
            ++reports;
            std::fprintf(stderr, "[dispatch-cache] 0x%llx key changed:%s\n", static_cast<unsigned long long>(address), what.c_str());
        }
        previous.userData = userData;
        previous.shader = std::map<std::uint32_t, std::uint32_t>(queue.shader.begin(), queue.shader.end());
        previous.key = key;
    }

    mix(reinterpret_cast<std::uintptr_t>(registeredShader.get()));
    if (fromAhead && key != ahead.key) {
        fromAhead = false;
        ++groupStats.keyMismatches;
    }
    if (fromAhead) {
        if (const auto hook = aheadAdoptHook.load(std::memory_order_acquire)) hook();
        const auto* aheadViews = ahead.capture != nullptr ? &ahead.capture->snapshot.registeredImages : ahead.keepVariant != nullptr && ahead.keepVariant->registeredViews != nullptr ? ahead.keepVariant->registeredViews.get() : nullptr;
        if (!CapturedCurrent(ahead.captured, snapshot) || (aheadViews != nullptr && !CaptureRegisteredViewsCurrent(*aheadViews))) {
            fromAhead = false;
            ++dispatchesRejectedAhead;
            ++groupStats.memoryMismatches;
        }
    }
    phaseTiming.Phase(PhaseKey);
    timing.Mark("decode_key_device_setup");
    bool readsClear = false;
    const bool queueMisses = pipelineCandidate && DispatchMissItemsEnabled() && !noDispatchCache && !stampValidate() && !verifyDataHits();
    bool lateItem = false;
    bool recipeLess = false;
    if (pipelineCandidate) {
        std::vector<ShaderRecompiler::MemoryRegion> reads;
        bool entryFound = false;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            if (const auto found = dispatchCache.find(key); found != dispatchCache.end()) {
                entryFound = true;
                for (const auto& variant : found->second->variants) appendEntryRegions(*variant, reads);
            }
        }
        if (!entryFound && queueMisses) {
            lateItem = true;
        } else if (!entryFound || noDispatchCache || stampValidate()) {
            NoteItemFallback(ItemFallback::Miss);
        } else if (verifyDataHits()) {
            NoteItemFallback(ItemFallback::Verify);
        } else {
            readsClear = std::none_of(reads.begin(), reads.end(), [](const auto& region) { return DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); });
            if (!readsClear) NoteItemFallback(ItemFallback::Reads);
        }
        if (!readsClear && !lateItem && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    if (fromAhead) {
        compiledResult = std::move(ahead.compiledResult);
        keepVariant = std::move(ahead.keepVariant);
        attachVariant = std::move(ahead.attachVariant);
        shaderMemory = std::move(ahead.shaderMemory);
        captured = std::move(ahead.captured);
        liveWords = std::move(ahead.liveWords);
        capture = std::move(ahead.capture);
        dataHit = ahead.dataHit;
        cached = ahead.cached;
        validated = ahead.validated;
        ++groupStats.adopted;
        ++dispatchesAdoptedAhead;
        const auto age = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ahead.resolvedAt).count();
        groupStats.ageMs += age;
        groupStats.ageMaxMs = std::max(groupStats.ageMaxMs, age);
        phaseTiming.Phase(PhaseLookup);
    } else {
        lookupDispatch(address, submission, key, noDispatchCache, traceCache, profile, memory, phaseTiming, phaseMs, compiledResult, keepVariant, captured, liveWords, dataHit, cached, validated, missedEntry, missedDiffering);
    }
    timing.Mark("lookup_cache");
    std::vector<DrawPipeline::Range> itemWrites;
    bool pipelined = false;
    if (readsClear) {
        if (cached && keepVariant != nullptr && keepVariant->recipe.load(std::memory_order_acquire) != nullptr) {
            bool bounded = false;
            itemWrites = DispatchWriteRanges(*compiledResult, bounded);
            pipelined = bounded;
            if (!bounded) NoteItemFallback(ItemFallback::Writes);
        } else if (queueMisses && cached && keepVariant != nullptr) {
            recipeLess = true;
        } else if (queueMisses && !cached) {
            lateItem = true;
        } else {
            NoteItemFallback(ItemFallback::NoRecipe);
        }
        if (!pipelined && !recipeLess && !lateItem && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    std::uint64_t forgetAtCapture = 0;
    if (cached) {
        require(keepVariant != nullptr && keepVariant->shader == registeredShader, "dispatch cache belongs to another registered shader");
        if (profile) captureMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
    } else if (!fromAhead) {
        shaderMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);

        static const bool dumpShaders = std::getenv("APS5_DUMP_SHADERS") != nullptr;
        try {
            auto invocation = InvocationFor(snapshot, codeOffset, request);
            timing.Mark("prepared_invocation");

            const auto waitedBefore = traceCapSync() ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
            forgetAtCapture = GuestMemory::ForgetSerial();
            capture = [&] {
                const SampledReadScope sampling(evidenceReads);
                return CaptureInvocation(*shaderMemory, snapshot, codeOffset, request, invocation);
            }();
            captured = shaderMemory->Regions();
            request.context.memory = captured;
            if (traceCapSync()) traceCapture("dispatch-capture", address, submission.queue, captured, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
            if (profile) captureMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
            phaseTiming.Phase(PhaseCapture);
            if (dumpShaders) static_cast<void>(dumpRequest(address, request));
            const auto started = std::chrono::steady_clock::now();


            timing.Mark("capture_resources");
            compiledResult = invocation.Materialize(*capture);
            timing.Mark("materialize");
            if (profile && compiledResult->cacheHit) cacheHits.fetch_add(1, std::memory_order_relaxed);
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            static std::atomic<double> totalMs{0};
            if (profile) {
                const auto total = totalMs.fetch_add(elapsed, std::memory_order_relaxed) + elapsed;
                if (elapsed > 200) std::fprintf(stderr, "[gpu] compute shader 0x%llx recompile took %.0f ms (%zu SPIR-V words, %zu captured regions, total %.1f s)\n", static_cast<unsigned long long>(address), elapsed, compiledResult->spirv.size(), captured.size(), total / 1000);
            }
            if (const auto limit = MaxComputeSpirvWords(); limit != 0 && compiledResult->spirv.size() > limit)
                throw std::runtime_error(std::to_string(compiledResult->spirv.size()) + " SPIR-V words, above APS5_MAX_COMPUTE_SPIRV_WORDS (" + std::to_string(limit) + ")");
        } catch (const std::exception& error) {
            std::string dump;
            if (dumpShaders) {
                const auto regions = shaderMemory->Regions();
                auto failed = request;
                failed.context.memory = regions;
                dump = dumpRequest(address, failed, "_failed");
            }

            std::string reason = error.what();
            if (const auto newline = reason.find('\n'); newline != std::string::npos) reason.resize(newline);
            if (SkipRefusedDispatches()) {
                ReportRefusedDispatch(address, reason);
                return;
            }
            char where[96];
            if (dump.empty()) std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
            else std::snprintf(where, sizeof(where), "compute shader 0x%llx (%s): ", static_cast<unsigned long long>(address), dump.c_str());
            throw std::runtime_error(where + reason);
        }
        if (profile) recompileMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
        phaseTiming.Phase(PhaseRecompile);
        if (!resolving) insertDispatch(address, key, noDispatchCache || !CacheableResult(*compiledResult), profile, registeredShader, forgetAtCapture, memory, shaderMemory, captured, capture, compiledResult, missedEntry, missedDiffering, attachVariant, phaseTiming);
    }
    const auto* registeredViews = capture != nullptr ? &capture->snapshot.registeredImages : keepVariant != nullptr && keepVariant->registeredViews != nullptr ? keepVariant->registeredViews.get() : nullptr;
    if (resolving) {
        auto& parked = resolvedAhead()[currentPacketOffset()];
        parked = ResolvedDispatch{};
        parked.address = address;
        parked.key = key;
        parked.indirectArguments = indirectArguments;
        parked.forgetAtCapture = forgetAtCapture;
        std::copy_n(packet.begin(), std::min(packet.size(), parked.packet.size()), parked.packet.begin());
        parked.registeredShader = registeredShader;
        parked.compiledResult = compiledResult;
        parked.keepVariant = keepVariant;
        parked.attachVariant = attachVariant;
        parked.missedEntry = missedEntry;
        parked.shaderMemory = shaderMemory;
        parked.captured = std::move(captured);
        parked.liveWords = std::move(liveWords);
        parked.capture = capture;
        parked.dataHit = dataHit;
        parked.cached = cached;
        parked.validated = validated;
        parked.missedDiffering = missedDiffering;
        parked.noDispatchCache = noDispatchCache || (compiledResult != nullptr && !CacheableResult(*compiledResult));
        parked.resolvedAt = std::chrono::steady_clock::now();
        ++groupStats.resolved;
        if (cached) ++groupStats.resolvedHits;
        if (!cached && compiledResult != nullptr && !localDevice->HasComputePipeline(*compiledResult)) {
            SubmitComputePrebuild([localDevice, compiledResult, address] {
                try { localDevice->PrebuildComputePipeline(*compiledResult, address); } catch (const std::exception&) {}
            });
        }
        return;
    }
    if (late != nullptr) CompareShadow(*late, captured, snapshot, address);
    if (verifyDataHits() && dataHit) verifyDataHit(snapshot, codeOffset, request, memory, address, *keepVariant, liveWords, *compiledResult);

    if (recordQueuedLabelsAfterCapture(submission.queue, captured)) {
        dispatch(queue, packet, submission, indirectArguments);
        return;
    }
    phaseTiming.Phase(PhaseQueuedLabels);
    const auto& compiled = *compiledResult;
    std::vector<Graphics::GuestMemorySnapshot> snapshots;
    for (const auto& region : captured) snapshots.push_back({region.guestAddress, region.bytes});
    std::array<std::uint32_t, 3> groups{packet[1], packet[2], packet[3]};
    if (indirectArguments == 0 && (packet[4] & 0x20u) != 0) {

        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            const auto threads = std::max(readRegister(queue.shader, 0x207 + axis) & 0xffffu, 1u);
            groups[axis] = ThreadDimensionGroups(groups[axis], threads);
        }
    }
    const std::uint64_t dispatchThreads = indirectArguments != 0 ? 0 : std::uint64_t{groups[0]} * groups[1] * groups[2] * compute.numThreads[0] * compute.numThreads[1] * compute.numThreads[2];
    std::optional<DispatchIoTrace> ioTrace;
    if (DispatchIoTrace::Enabled()) {
        char text[96];
        std::snprintf(text, sizeof(text), "queue=0x%x submission=%llu shader 0x%llx", submission.queue, static_cast<unsigned long long>(submission.serial), static_cast<unsigned long long>(address));
        std::string header = text;
        if (DispatchIoTrace::Words()) {
            for (const auto word : userData) {
                std::snprintf(text, sizeof(text), " %08x", word);
                header += text;
            }
        }
        ioTrace.emplace(std::move(header));
    }

    const auto rethrow = [&](const std::exception& error) {
        if (fromAhead) {
            ++groupStats.deviceRejects;
            dispatch(queue, packet, submission, indirectArguments);
            return true;
        }
        char where[64];
        std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
        throw std::runtime_error(where + std::string(error.what()) + describeIndirectSource());
    };
    phaseTiming.Phase(PhaseSnapshots);

    bool plain = false;
    if ((pipelined && DispatchMissItemsEnabled()) || lateItem || recipeLess) {
        plain = compiledResult != nullptr && !htileClearCompletion.has_value() && htileMask.bytes == 0u;
        if (plain && !partialLastGroup && MatchSgprPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute)) plain = false;
        if (plain && !partialLastGroup) {
            if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
                const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
                if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) plain = false;
            }
        }
    }
    if (pipelined && DispatchMissItemsEnabled() && !plain) {
        pipelined = false;
        itemWrites.clear();
        itemNotPlain.fetch_add(1, std::memory_order_relaxed);
        if (DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    if (lateItem || recipeLess) {
        bool queued = plain;
        if (queued && lateItem) {
            queued = std::none_of(captured.begin(), captured.end(), [](const auto& region) { return DrawPipeline::Queue0().Overlaps(region.guestAddress, region.bytes.size()); });
            if (!queued) NoteItemFallback(ItemFallback::Reads);
        } else if (!plain) {
            NoteItemFallback(lateItem ? ItemFallback::Miss : ItemFallback::NoRecipe);
        }
        if (queued) {
            bool bounded = false;
            itemWrites = DispatchWriteRanges(*compiledResult, bounded);
            pipelined = bounded;
            if (!bounded) NoteItemFallback(ItemFallback::Writes);
        }
        if (pipelined) (lateItem ? itemMisses : itemRecipeLess).fetch_add(1, std::memory_order_relaxed);
        else if (DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, 0x15);
    }
    if (pipelined) {
        runCaptureCheckHook();
        if (capturedMemoryChanged(captured)) {
            const RecaptureScope recapture;
            dispatch(queue, packet, submission, indirectArguments);
            return;
        }
        const auto recipe = keepVariant != nullptr ? keepVariant->recipe.load(std::memory_order_acquire) : nullptr;
        const auto attach = stampValidate() ? nullptr : keepVariant != nullptr ? keepVariant : attachVariant;
        const std::uint64_t prepareThreads = lateItem || recipeLess ? dispatchThreads : 0;
        const auto epoch = late != nullptr ? late->epoch : DrawPipeline::EpochToken().load(std::memory_order_relaxed);
        const auto queueId = submission.queue;
        itemCommits.fetch_add(1, std::memory_order_relaxed);
        if (late != nullptr) laterQueued.fetch_add(1, std::memory_order_relaxed);
        DrawPipeline::Queue0().Enqueue(DrawPipeline::CommitKind::Dispatch, [this, ioHeader = ioTrace ? ioTrace->Detach() : std::string{}, localDevice = std::move(localDevice), result = compiledResult, capture, registeredViews, snapshots = std::move(snapshots), liveWords = std::move(liveWords), groups, address, queueId, recipe, keepVariant, attach, prepareThreads, dataHit, epoch, htileClearCompletion, htileMask] {
            DrawPipeline::FollowEpoch(epoch);
            std::optional<DispatchIoTrace> commitTrace;
            if (!ioHeader.empty()) commitTrace.emplace(ioHeader);
            GuestMemory::SetCurrentPacket(0x15, queueId);
            const auto& compiledCommit = *result;
            const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
            try {
                std::shared_ptr<RecipeHit> recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(recipe, false);
                std::shared_ptr<const Recipe> builtRecipe;
                bool writersNoted = false;
                for (;;) {
                    std::shared_ptr<PreparedDispatch> prepared;
                    if (recipeHit == nullptr || VulkanDevice::VerifyRecipes() || htileClearCompletion.has_value()) prepared = localDevice->PrepareDispatch(compiledCommit, snapshots, prepareThreads, address, htileClearCompletion, registeredViews);
                    GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                    std::lock_guard gpuLock(GuestMemory::GpuMutex());
                    recordLabelsForPacket(localDevice.get(), queueId);
                    if (noteWrites && writerKeyedEvidence() && !writersNoted) {
                        noteWrittenBuffers(address, queueId, compiledCommit);
                        writersNoted = true;
                    }
                    if (recipeHit != nullptr) {
                        VulkanDevice::IndirectOutcome outcome{0, 0};
                        const auto recipeResult = localDevice->DispatchRecipe(compiledCommit, groups[0], groups[1], groups[2], 0, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit);
                        if (recipeResult == RecipeOutcome::Rebuild) {
                            recipeHit = nullptr;
                            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, false);
                            itemRebuilds.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                    } else {
                        localDevice->Dispatch(compiledCommit, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), htileClearCompletion.has_value() || attach == nullptr ? nullptr : &builtRecipe, htileClearCompletion, registeredViews);
                    }
                    if (htileMask.bytes != 0u) localDevice->NoteHtileMask(htileMask);
                    break;
                }
                if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, queueId, compiledCommit);
                if (builtRecipe != nullptr) {
                    if (dataHit) {
                        auto own = std::make_shared<Recipe>(*builtRecipe);
                        own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                        builtRecipe = std::move(own);
                    }
                    attach->recipe.store(std::move(builtRecipe), std::memory_order_release);
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, false);
                }
            } catch (const std::exception& error) {
                char where[64];
                std::snprintf(where, sizeof(where), "compute shader 0x%llx: ", static_cast<unsigned long long>(address));
                throw std::runtime_error(where + std::string(error.what()));
            }
        }, std::move(itemWrites));
        ReportDispatchItems(profile);
        return;
    }

    std::shared_ptr<RecipeHit> recipeHit;
    if (cached && keepVariant != nullptr && !stampValidate()) {
        recipeHit = htileClearCompletion.has_value() ? nullptr : localDevice->PrepareRecipe(keepVariant->recipe.load(std::memory_order_acquire), indirectArguments != 0);
        phaseTiming.Phase(PhaseRecipePrecheck);
    }

    std::shared_ptr<const Recipe> builtRecipe;
    auto* const attachTo = stampValidate() ? nullptr : keepVariant != nullptr ? keepVariant.get() : attachVariant.get();
    bool writersNoted = false;
    const bool noteWrites = writeEvidenceEnabled() || traceCapSync();
    const auto keysFill = indirectArguments == 0 && !partialLastGroup ? MatchSgprPatternFill(packet, std::span(snapshot.code).subspan(codeOffset), userData, compute) : std::nullopt;
    const auto filledKeys = keysFill ? KeysFillMemo(keysFill->pattern, false, true) : std::nullopt;
    if (keysFill) Graphics::RetireMultisampleColorTargetsOnFmaskWrite(keysFill->base, keysFill->bytes);

    std::optional<Graphics::DepthPlaneClear> depthPlaneClear;
    if (indirectArguments == 0 && !partialLastGroup) {
        if (const auto clear = Graphics::MatchStorageImageClear(std::span(snapshot.code).subspan(codeOffset), userData, compute, groups)) {
            const auto clearAddress = Graphics::StorageImageClearAddress(*clear);
            if (clearAddress != 0 && Graphics::DepthSurfaceAt(clearAddress)) {
                GuestMemory::TagGpuLockSite(GuestMemory::GpuLockSite::Dispatch);
                std::lock_guard clearLock(GuestMemory::GpuMutex());
                recordLabelsForPacket(localDevice.get(), submission.queue);
                depthPlaneClear.emplace(*clear);
            }
        }
    }
    bool captureChecked = fromAhead;

    for (;;) {

        std::shared_ptr<PreparedDispatch> prepared;
        if (recipeHit == nullptr || VulkanDevice::VerifyRecipes()) {
            try {
                prepared = localDevice->PrepareDispatch(compiled, snapshots, dispatchThreads, address, htileClearCompletion, registeredViews);
            } catch (const std::exception& error) {
                if (rethrow(error)) return;
            }
            if (profile) {
                const auto now = std::chrono::steady_clock::now();
                const auto prepareMs = std::chrono::duration<double, std::milli>(now - phaseLap).count();
                phaseLap = now;
                double parts = 0;
                if (prepared != nullptr) {
                    const auto phases = VulkanDevice::PreparePhaseMs(*prepared);
                    phaseMs[PhasePrepareKey] += phases[0];
                    phaseMs[PhasePrepareFind] += phases[1];
                    phaseMs[PhasePreparePrecollect] += phases[2];
                    phaseMs[PhasePreparePresync] += phases[3];
                    phaseMs[PhasePrepareStageA] += phases[4];
                    for (const auto part : phases) parts += part;
                }
                phaseMs[PhasePrepareOther] += std::max(0.0, prepareMs - parts);
            }
        }
        GuestMemory::TagGpuLockSite(indirectArguments != 0 ? GuestMemory::GpuLockSite::Indirect : GuestMemory::GpuLockSite::Dispatch);
        std::unique_lock gpuLock(GuestMemory::GpuMutex());
        phaseTiming.Phase(PhaseLockWait);
        if (!captureChecked) {
            captureChecked = true;
            runCaptureCheckHook();
            if (capturedMemoryChanged(captured, snapshots)) {
                gpuLock.unlock();
                const RecaptureScope recapture;
                dispatch(queue, packet, submission, indirectArguments);
                return;
            }
        }

        recordLabelsForPacket(localDevice.get(), submission.queue);
        phaseTiming.Phase(PhaseLabels);
        if (noteWrites && writerKeyedEvidence() && !writersNoted) {
            noteWrittenBuffers(address, submission.queue, compiled);
            writersNoted = true;
        }
        phaseTiming.Phase(PhaseNoteWriters);
        auto* const keysRecorder = filledKeys ? Graphics::Recorder::Active() : nullptr;
        const auto keysNoteBefore = keysRecorder != nullptr ? keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes) : 0;
        try {
            if (recipeHit != nullptr) {
                VulkanDevice::IndirectOutcome outcome{0, 0};
                const auto result = localDevice->DispatchRecipe(compiled, groups[0], groups[1], groups[2], indirectArguments, address, recipeHit, outcome, VulkanDevice::VerifyRecipes() ? prepared : nullptr, dataHit, threadGroup);
                if (result == RecipeOutcome::Rebuild) {

                    recipeHit = nullptr;
                    VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Restart, indirectArguments != 0);
                    continue;
                }
                if (indirectArguments != 0) countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else if (indirectArguments != 0) {
                const auto outcome = localDevice->DispatchIndirect(compiled, indirectArguments, snapshots, address, std::move(prepared), attachTo != nullptr ? &builtRecipe : nullptr, threadGroup, registeredViews);
                countIndirect(outcome.cpuReason, outcome.argumentReadMs);
            } else {
                localDevice->Dispatch(compiled, groups[0], groups[1], groups[2], snapshots, address, std::move(prepared), htileClearCompletion.has_value() ? nullptr : (attachTo != nullptr ? &builtRecipe : nullptr), htileClearCompletion, registeredViews);
            }
        } catch (const std::exception& error) {
            if (rethrow(error)) return;
        }
        if (keysRecorder != nullptr && keysRecorder == Graphics::Recorder::Active()) {
            const auto note = keysRecorder->LastWriteNote(keysFill->base, keysFill->bytes);
            if (note != 0 && note != keysNoteBefore) Graphics::NoteKeysFillOnGpu(keysFill->base, keysFill->bytes, *filledKeys);
        }
        if (htileMask.bytes != 0u) localDevice->NoteHtileMask(htileMask);
        if (builtRecipe != nullptr) {
            if (dataHit) {

                auto own = std::make_shared<Recipe>(*builtRecipe);
                own->dataWordsHash = Graphics::ShaderResources::DataWordsHash({ShaderRecompiler::ShaderStage::Compute, keepVariant->compiled.get(), 0});
                builtRecipe = std::move(own);
            }
            attachTo->recipe.store(std::move(builtRecipe), std::memory_order_release);
            VulkanDevice::NoteRecipe(VulkanDevice::RecipeEvent::Attach, indirectArguments != 0);
        }
        break;
    }
    phaseTiming.Phase(PhaseDevice);
    if (late != nullptr) laterInlined.fetch_add(1, std::memory_order_relaxed);
    if (noteWrites && !writerKeyedEvidence()) noteWrittenBuffers(address, submission.queue, compiled);
    if (profile) deviceMs.fetch_add(phaseTiming.Elapsed(), std::memory_order_relaxed);
    phaseTiming.Phase(PhaseTail);
    if (profile) {
        auto& pending = pendingDispatchPhases();
        pending.outcome = DispatchOutcome::Real;
        pending.phases = true;
        pending.hit = cached;
        pending.validated = validated;
        pending.ms = phaseMs;
        pending.tailAt = phaseLap;
    }
    if (profile && dispatches.fetch_add(1, std::memory_order_relaxed) % 100 == 99) {
        std::uint64_t hits = 0;
        std::uint64_t evictions = 0;
        {
            std::lock_guard cacheLock(dispatchCacheMutex);
            hits = dispatchCacheHits;
            evictions = dispatchCacheEvictions;
        }
        AgcDriver::ProfilePrint_nid_no_patch( "[gpu] %llu dispatches (%llu dispatch cache hits, %llu evictions, %llu recompile cache hits): capture %.1f s, cache key %.1f s, recompile %.1f s, device %.1f s\n", static_cast<unsigned long long>(dispatches.load(std::memory_order_relaxed)), static_cast<unsigned long long>(hits), static_cast<unsigned long long>(evictions), static_cast<unsigned long long>(cacheHits.load(std::memory_order_relaxed)), captureMs.load(std::memory_order_relaxed) / 1000, keyMs.load(std::memory_order_relaxed) / 1000, recompileMs.load(std::memory_order_relaxed) / 1000, deviceMs.load(std::memory_order_relaxed) / 1000);
    }
}

}
