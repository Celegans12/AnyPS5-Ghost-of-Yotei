#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "ThreadOwned.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/DrawWriteRanges.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Memory/WriteEvidence.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/DeferredLabels.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace AgcDriver::DriverDetail {

std::map<std::size_t, Driver::ResolvedDispatch>& Driver::resolvedAhead() {
    static thread_local std::map<std::size_t, ResolvedDispatch>* resolved = nullptr;
    return ShaderRecompiler::ThreadOwned(resolved);
}

Driver::GroupCaptureStats& Driver::groupCaptureStats() {
    static thread_local GroupCaptureStats stats;
    return stats;
}

std::size_t& Driver::currentPacketOffset() {
    static thread_local std::size_t offset = 0;
    return offset;
}

bool& Driver::resolvingAhead() {
    static thread_local bool resolving = false;
    return resolving;
}

bool& Driver::gateOpened() {
    static thread_local bool opened = false;
    return opened;
}

std::uint64_t Driver::DispatchesAdoptedAhead() const {
    return dispatchesAdoptedAhead.load(std::memory_order_relaxed);
}

std::uint64_t Driver::DispatchesRejectedAhead() const {
    return dispatchesRejectedAhead.load(std::memory_order_relaxed);
}

std::pair<std::uint64_t, std::uint64_t> Driver::MissAheadTotals() {
    const auto totals = missAheadLedger().Snapshot();
    return {totals.started, totals.hit + totals.waited};
}

void Driver::SetAheadAdoptHook(void (*hook)()) {
    aheadAdoptHook.store(hook, std::memory_order_release);
}

namespace {

std::size_t& PreviewedUntil() {
    static thread_local std::size_t until = 0;
    return until;
}

}

void Driver::clearResolvedAhead() {
    PreviewedUntil() = 0;
    auto& resolved = resolvedAhead();
    if (resolved.empty()) return;
    groupCaptureStats().stale += resolved.size();
    resolved.clear();
}

namespace {

struct RunLabel {
    std::uint64_t address;
    std::size_t size;
    std::uint64_t value;
};

bool InCode(const ShaderRecompiler::MemoryRegion& region, const std::shared_ptr<const ShaderSnapshot>& code) {
    if (code == nullptr) return false;
    if (region.guestAddress >= code->codeAddress && region.guestAddress < code->codeAddress + code->code.size() * sizeof(std::uint32_t)) return true;
    return !code->header.empty() && region.guestAddress >= code->headerAddress && region.guestAddress < code->headerAddress + code->header.size();
}

bool OverlapsRange(std::span<const ShaderRecompiler::MemoryRegion> regions, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges, const std::shared_ptr<const ShaderSnapshot>& code) {
    for (const auto& region : regions) {
        if (InCode(region, code)) continue;
        for (const auto& [begin, end] : ranges) {
            if (begin < region.guestAddress + region.bytes.size() && region.guestAddress < end) return true;
        }
    }
    return false;
}

bool WrittenByRun(std::span<const std::uint32_t> wait, const std::vector<RunLabel>& labels) {
    const std::uint64_t awaited = wait[2] | (static_cast<std::uint64_t>(wait[3]) << 32u);
    const auto bytes = Pm4::WaitAwaitedBytes(wait);
    for (auto it = labels.rbegin(); it != labels.rend(); ++it) {
        if (it->address != awaited || it->size < bytes) continue;
        return Pm4::WaitComparesValue(wait, bytes == 8 ? it->value : (it->value & 0xffffffffu));
    }
    return false;
}

void NoteLabel(std::vector<RunLabel>& labels, std::uint64_t address, std::span<const std::byte> bytes) {
    if (bytes.size() != 4 && bytes.size() != 8) return;
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data(), bytes.size());
    labels.push_back({address, bytes.size(), value});
}

bool OverlapsLabel(std::span<const ShaderRecompiler::MemoryRegion> regions, const std::vector<RunLabel>& labels) {
    for (const auto& label : labels) {
        for (const auto& region : regions) {
            if (label.address < region.guestAddress + region.bytes.size() && region.guestAddress < label.address + label.size) return true;
        }
    }
    return false;
}

}

void Driver::resolveGroupAhead(const Submission& submission, const QueueState& live, std::size_t from) {
    static const bool allQueues = std::getenv("APS5_GROUP_CAPTURE_ALL_QUEUES") != nullptr;
    if (!allQueues && submission.queue == 0) return;
    auto& resolved = resolvedAhead();
    const auto localDevice = device.Load();
    if (resolved.contains(from) || localDevice == nullptr) return;
    if (submission.queue == 0 && DrawPipeline::Depth() != 0 && DrawPipeline::Queue0().Busy()) DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Packet, (submission.commands[from] >> 8u) & 0xffu);
    auto& stats = groupCaptureStats();
    const auto begin = std::chrono::steady_clock::now();
    ++stats.runs;
    QueueState scratch = live;
    std::vector<RunLabel> labels;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> writes;
    for (const auto& label : deferredLabels().labels) NoteLabel(labels, label.address, std::span<const std::byte>(label.bytes).first(label.size));
    struct Resolving {
        std::size_t offset;
        Resolving() : offset(currentPacketOffset()) { resolvingAhead() = true; }
        ~Resolving() {
            resolvingAhead() = false;
            currentPacketOffset() = offset;
        }
    } resolving;
    const auto& commands = submission.commands;
    if (submission.shaders != nullptr && from >= PreviewedUntil()) {
        bool hasFurtherDispatch = false;
        std::size_t limit = from < commands.size() ? from + Pm4::PacketWords(commands[from]) : commands.size();
        while (limit < commands.size()) {
            const auto header = commands[limit];
            if (Pm4::FillerPacket(header)) {
                ++limit;
                continue;
            }
            const auto count = Pm4::PacketWords(header);
            if (limit + count > commands.size()) break;
            const auto opcode = (header >> 8u) & 0xffu;
            if (header == FlipPacketHeader || header == RenderingWaitPacketHeader || opcode == 0x22 || opcode == 0x3f || opcode == 0x3c || opcode == 0x93) break;
            if (opcode == 0x15 || opcode == 0x16) hasFurtherDispatch = true;
            limit += count;
        }
        PreviewedUntil() = limit;
        if (hasFurtherDispatch) {
            static thread_local QueueState* previewSlot = nullptr;
            auto& preview = ShaderRecompiler::ThreadOwned(previewSlot);
            preview.shader = live.shader;
            preview.predication = live.predication;
            std::vector<PreparedShaders*> pendingShaders;
            std::vector<PreparedShaders*> submittedShaders;
            std::size_t scan = from;
            while (scan < limit) {
                const auto header = commands[scan];
                if (Pm4::FillerPacket(header)) {
                    ++scan;
                    continue;
                }
                const auto count = Pm4::PacketWords(header);
                if (scan + count > limit) break;
                const auto packet = std::span(commands).subspan(scan, count);
                const auto opcode = (header >> 8u) & 0xffu;
                const auto next = scan + count;
                if (opcode == 0x15 || opcode == 0x16) {
                    if (count < (opcode == 0x15 ? 5u : 3u)) break;
                    if (scan > from) {
                        const auto low = preview.shader.find(0x20c);
                        const auto high = preview.shader.find(0x20d);
                        if (low != preview.shader.end() && high != preview.shader.end()) {
                            const auto address = (static_cast<std::uint64_t>(low->second) << 8u) | (static_cast<std::uint64_t>(high->second & 0xffu) << 40u);
                            if (auto registered = RegisteredProgram(*submission.shaders, address, {0}); registered != nullptr && registered->type == 0) {
                                bool isPending = false;
                                bool isEmpty = false;
                                {
                                    std::lock_guard lock(registered->prepared->mutex);
                                    isPending = registered->prepared->pending;
                                    isEmpty = registered->prepared->entries.empty();
                                }
                                if (isPending) {
                                    if (std::ranges::find(pendingShaders, registered->prepared.get()) == pendingShaders.end()) {
                                        pendingShaders.push_back(registered->prepared.get());
                                    }
                                } else if (isEmpty && std::ranges::find(submittedShaders, registered->prepared.get()) == submittedShaders.end()) {
                                    const auto codeOffset = static_cast<std::size_t>((address - registered->codeAddress) / sizeof(std::uint32_t));
                                    if (codeOffset < registered->code.size()) {
                                        try {
                                            auto compute = Graphics::DecodeComputeStageInfo(preview.shader, registered->header);
                                            submittedShaders.push_back(registered->prepared.get());
                                            const auto rsrc2 = preview.shader.find(0x213);
                                            const auto userCount = rsrc2 != preview.shader.end() ? ((rsrc2->second >> 1u) & 0x1fu) : 0u;
                                            std::vector<std::uint32_t> userData(userCount, 0u);
                                            for (std::uint32_t i = 0; i < userCount; ++i) {
                                                if (const auto reg = preview.shader.find(0x240u + i); reg != preview.shader.end()) userData[i] = reg->second;
                                            }
                                            const auto wave = (packet.back() & 0x8000u) != 0 ? 32u : 64u;
                                            SubmitRegistrationPreparation([registered, localDevice, address, codeOffset, wave, compute, userData = std::move(userData), state = preview.shader]() mutable {
                                                std::vector<ShaderRecompiler::MemoryRegion> memory{{registered->codeAddress, std::as_bytes(std::span(registered->code))}};
                                                if (!registered->header.empty()) memory.push_back({registered->headerAddress, registered->header});
                                                ShaderRecompiler::RecompileRequest request{
                                                    {ShaderRecompiler::ShaderStage::Compute, address, std::span(registered->code).subspan(codeOffset), registered->headerAddress, registered->header},
                                                    {wave, 0, userData, compute, std::nullopt, std::nullopt, memory, DispatchFloatMode(*registered, state)},
                                                    localDevice->ComputeTarget(wave),
                                                    {0, 0, 0, 128}
                                                };
                                                try { static_cast<void>(InvocationFor(*registered, codeOffset, request)); } catch (const std::exception&) {}
                                            }, registered->prepared.get());
                                        } catch (const std::exception&) {}
                                    }
                                }
                            }
                        }
                    }
                    Pm4::NoteDispatchRegisters(packet, preview);
                } else if (Pm4::IndirectRegisterOpcode(opcode)) {
                    const auto found = submission.registerLists.find(scan);
                    if (count != 5 || found == submission.registerLists.end() || found->second.size() != static_cast<std::size_t>(packet[4]) * 2 || !Pm4::IndirectRegistersValid(found->second)) break;
                    Pm4::ExecuteIndirectRegisters(packet, found->second, preview);
                } else if (!Pm4::AccessesMemory(header) && opcode != 0x3c && opcode != 0x93 && opcode != 0x37 && opcode != 0x42 && opcode != 0x46 && opcode != 0x49 && opcode != 0x58) {
                    try {
                        Pm4::Execute(packet, preview);
                    } catch (const std::exception&) {
                        break;
                    }
                }
                scan = next;
            }
            for (auto it = pendingShaders.rbegin(); it != pendingShaders.rend(); ++it) {
                PrioritizeRegistrationPreparation(**it);
            }
        }
    }
    for (std::size_t cursor = from; cursor < commands.size();) {
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        if (cursor + count > commands.size()) break;
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        const auto next = cursor + count;
        if (header == FlipPacketHeader || header == RenderingWaitPacketHeader || opcode == 0x22 || (Pm4::Predicated(header) && scratch.predication.operation != 0)) break;
        if (opcode == 0x3c || opcode == 0x93) {
            if (!Pm4::WaitSatisfiedUnchecked(packet) && !WrittenByRun(packet, labels)) break;
        } else if (opcode == 0x15 || opcode == 0x16) {
            if (count < (opcode == 0x15 ? 5u : 3u) || resolved.contains(cursor)) break;
            if (opcode == 0x16) {
                std::uint64_t arguments = 0;
                try {
                    arguments = Pm4::DispatchArgumentAddress(packet, scratch);
                } catch (const std::exception&) {
                    break;
                }
                const auto argumentsEnd = arguments + 3u * sizeof(std::uint32_t);
                const bool written = std::any_of(writes.begin(), writes.end(), [&](const auto& range) { return range.first < argumentsEnd && arguments < range.second; })
                    || std::any_of(labels.begin(), labels.end(), [&](const RunLabel& label) { return label.address < argumentsEnd && arguments < label.address + label.size; });
                if (written && indirectArgumentsPath(packet, arguments) != IndirectGpu) {
                    ++stats.writtenBefore;
                    break;
                }
            }
            currentPacketOffset() = cursor;
            try {
                if (opcode == 0x15) dispatch(scratch, packet, submission);
                else dispatchIndirect(scratch, packet, submission);
                Pm4::NoteDispatchRegisters(packet, scratch);
            } catch (const std::exception&) {
                ++stats.failures;
                break;
            }
            if (const auto found = resolved.find(cursor); found != resolved.end()) {
                if (OverlapsLabel(found->second.captured, labels)) {
                    resolved.erase(found);
                    ++stats.labelOverlaps;
                    break;
                }
                if (OverlapsRange(found->second.captured, writes, found->second.registeredShader)) {
                    resolved.erase(found);
                    ++stats.writtenBefore;
                    break;
                }
                if (found->second.compiledResult == nullptr) {
                    ++stats.unknownWrites;
                    break;
                }
                if (!found->second.cached && found->second.registeredShader != nullptr) {
                    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
                    const auto& snapshot = *found->second.registeredShader;
                    std::vector<ShaderRecompiler::MemoryRegion> memory{{snapshot.codeAddress, std::as_bytes(std::span(snapshot.code))}};
                    if (!snapshot.header.empty()) memory.push_back({snapshot.headerAddress, snapshot.header});
                    auto lap = std::chrono::steady_clock::now();
                    std::array<double, DriverPhaseCount> phaseMs{};
                    auto phaseLap = lap;
                    DispatchPhaseTiming phaseTiming{profile, lap, phaseMs, phaseLap};
                    insertDispatch(found->second.address, found->second.key, found->second.noDispatchCache, profile, found->second.registeredShader, found->second.forgetAtCapture, memory, found->second.shaderMemory, found->second.captured, found->second.capture, found->second.compiledResult, found->second.missedEntry, found->second.missedDiffering, found->second.attachVariant, phaseTiming);
                }
                bool bounded = false;
                const auto itemWrites = DispatchWriteRanges(*found->second.compiledResult, bounded);
                if (!bounded) {
                    ++stats.unknownWrites;
                    break;
                }
                writes.insert(writes.end(), itemWrites.begin(), itemWrites.end());
            } else {
                ++stats.unknownWrites;
                break;
            }
        } else if (opcode == 0x37 || opcode == 0x49) {
            const auto label = Pm4::DecodeLabelWrite(packet);
            if (!label) break;
            NoteLabel(labels, label->address, label->Bytes());
            writes.emplace_back(label->address, label->address + label->Bytes().size());
        } else if (opcode != 0x3f && opcode != 0x42 && opcode != 0x46 && opcode != 0x58) {
            if (Pm4::AccessesMemory(header)) break;
            try {
                Pm4::Execute(packet, scratch);
            } catch (const std::exception&) {
                break;
            }
        }
        cursor = next;
    }
    const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    stats.resolveMs += ms;
    stats.resolveMaxMs = std::max(stats.resolveMaxMs, ms);
    reportGroupCapture(submission.queue);
}

MissAheadPlan*& Driver::missAheadPlan() {
    static thread_local MissAheadPlan* plan = nullptr;
    return plan;
}

MissAheadLedger& Driver::missAheadLedger() {
    static MissAheadLedger ledger;
    return ledger;
}

MissAheadKnown& Driver::missAheadKnown() {
    static MissAheadKnown known;
    return known;
}

namespace {

bool MissAheadEnabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_DISPATCH_MISS_AHEAD");
        return text == nullptr || std::strcmp(text, "0") != 0;
    }();
    // APS5_NO_DISPATCH_MISS_AHEAD (a runtime switch, for an in-run A/B) turns it off as APS5_DISPATCH_MISS_AHEAD=0 does.
    return enabled && !Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::NoDispatchMissAhead);
}

std::size_t MissAheadMax() {
    static const std::size_t cap = [] {
        const char* text = std::getenv("APS5_DISPATCH_MISS_AHEAD_MAX");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{32};
    }();
    return cap;
}

std::size_t MissAheadVisits() {
    static const std::size_t cap = [] {
        const char* text = std::getenv("APS5_DISPATCH_MISS_AHEAD_VISITS");
        return text != nullptr ? static_cast<std::size_t>(std::strtoull(text, nullptr, 10)) : std::size_t{4096};
    }();
    return cap;
}

MissAheadRange& ScannedRange() {
    static thread_local MissAheadRange range;
    return range;
}

}

bool Driver::missAheadEnabled() {
    return MissAheadEnabled();
}

bool Driver::missAheadProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

void Driver::reportMissAhead() {
    if (!missAheadProfiled()) return;
    static std::mutex mutex;
    static auto reported = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard lock(mutex);
        if (now - reported < std::chrono::seconds(10)) return;
        reported = now;
    }
    const auto totals = missAheadLedger().Snapshot();
    AgcDriver::ProfilePrint_nid_no_patch("[miss-ahead] cumulative: %llu builds started, %llu later used by a real dispatch (%llu finished before it, %llu still building), %llu not used\n",
                                         static_cast<unsigned long long>(totals.started), static_cast<unsigned long long>(totals.hit + totals.waited), static_cast<unsigned long long>(totals.hit), static_cast<unsigned long long>(totals.waited), static_cast<unsigned long long>(totals.Unused()));
}

void Driver::missAhead(const Submission& submission, const QueueState& live, std::size_t from) {
    if (!MissAheadEnabled() || submission.shaders == nullptr || resolvingAhead() || missAheadPlan() != nullptr) return;
    const auto cap = MissAheadMax();
    const auto& commands = submission.commands;
    if (cap == 0 || from >= commands.size() || ScannedRange().Covered(submission.serial, submission.queue, from)) return;
    const auto first = commands[from];
    const auto firstOpcode = (first >> 8u) & 0xffu;
    const auto firstCount = Pm4::PacketWords(first);
    if ((firstOpcode != 0x15 && firstOpcode != 0x16) || firstCount < (firstOpcode == 0x15 ? 5u : 3u) || from + firstCount > commands.size()) return;
    QueueState scratch = live;
    Pm4::NoteDispatchRegisters(std::span(commands).subspan(from, firstCount), scratch);
    MissAheadPlan plan(cap, MissAheadVisits());
    const char* stop = "end";
    struct Scanning {
        std::size_t offset;
        explicit Scanning(MissAheadPlan& plan) : offset(currentPacketOffset()) {
            missAheadPlan() = &plan;
            resolvingAhead() = true;
        }
        ~Scanning() {
            missAheadPlan() = nullptr;
            resolvingAhead() = false;
            currentPacketOffset() = offset;
        }
    } scanning(plan);
    std::size_t cursor = from + firstCount;
    while (cursor < commands.size()) {
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        if (cursor + count > commands.size()) {
            stop = "malformed";
            break;
        }
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        if (header == FlipPacketHeader) {
            stop = "flip";
            break;
        }
        if (header == RenderingWaitPacketHeader) {
            stop = "rendering_wait";
            break;
        }
        if (opcode == 0x22 || opcode == 0x3f) {
            stop = opcode == 0x22 ? "opcode_0x22" : "opcode_0x3f";
            break;
        }
        if (Pm4::Predicated(header) && scratch.predication.operation != 0) {
            stop = "predicated";
            break;
        }
        if (opcode == 0x15 || opcode == 0x16) {
            if (count < (opcode == 0x15 ? 5u : 3u)) {
                stop = "malformed";
                break;
            }
            if (plan.BuildsFull()) {
                stop = "build_cap";
                break;
            }
            if (!plan.Visit()) {
                stop = "visit_cap";
                break;
            }
            currentPacketOffset() = cursor;
            try {
                if (opcode == 0x15) {
                    dispatchAttempt(scratch, packet, submission, 0, false);
                } else if (const auto arguments = Pm4::DispatchArgumentAddress(packet, scratch); indirectArgumentsPath(packet, arguments) == IndirectGpu) {
                    const std::array<std::uint32_t, 5> unresolved{0xc0031500u, 0, 0, 0, packet.back()};
                    dispatchAttempt(scratch, unresolved, submission, arguments, false);
                } else {
                    plan.Note(MissAheadPlan::Skip::Indirect);
                }
            } catch (const std::exception&) {
                plan.Note(MissAheadPlan::Skip::Failed);
            }
            Pm4::NoteDispatchRegisters(packet, scratch);
        } else if (Pm4::IndirectRegisterOpcode(opcode)) {
            const auto found = submission.registerLists.find(cursor);
            if (count != 5 || found == submission.registerLists.end() || found->second.size() != static_cast<std::size_t>(packet[4]) * 2 || !Pm4::IndirectRegistersValid(found->second)) {
                stop = "malformed";
                break;
            }
            Pm4::ExecuteIndirectRegisters(packet, found->second, scratch);
        } else if (!Pm4::AccessesMemory(header) && opcode != 0x3c && opcode != 0x93 && opcode != 0x37 && opcode != 0x42 && opcode != 0x46 && opcode != 0x49 && opcode != 0x58) {
            try {
                Pm4::Execute(packet, scratch);
            } catch (const std::exception&) {
                stop = "execute_failed";
                break;
            }
        }
        cursor += count;
    }
    ScannedRange().Record(submission.serial, submission.queue, cursor);
    if (missAheadProfiled() && missAheadLedger().Active()) {
        const auto& totals = plan.Totals();
        std::fprintf(stderr, "[miss-ahead] scan queue %u serial %llu from %zu to %zu of %zu: visited %zu started %zu returned_early %zu skipped cached %zu plan_duplicate %zu unprepared %zu unregistered %zu capture_failed %zu early_listed %zu early_kernel %zu early_failed %zu indirect_cpu_path %zu stop %s\n", submission.queue, static_cast<unsigned long long>(submission.serial), from, cursor, commands.size(),
                     totals.visited, totals.started, totals.visited - totals.reached, totals.cached, totals.duplicate, totals.unprepared, totals.unregistered, totals.captureFailed, totals.listed, totals.kernel, totals.failed, totals.indirect, stop);
    }
    reportMissAhead();
}

void Driver::prepareMissAhead(const std::shared_ptr<const ShaderSnapshot>& snapshot, std::size_t codeOffset, const ShaderRecompiler::RecompileRequest& request, std::span<const ShaderRecompiler::MemoryRegion> memory, std::uint64_t key, const std::shared_ptr<VulkanDevice>& localDevice, std::uint64_t address) {
    auto* plan = missAheadPlan();
    if (plan == nullptr || snapshot == nullptr || snapshot->prepared == nullptr || localDevice == nullptr) return;
    {
        std::lock_guard lock(dispatchCacheMutex);
        if (dispatchCache.contains(key)) {
            plan->Note(MissAheadPlan::Skip::Cached);
            return;
        }
    }
    {
        std::lock_guard lock(snapshot->prepared->mutex);
        if (snapshot->prepared->pending || std::none_of(snapshot->prepared->entries.begin(), snapshot->prepared->entries.end(), [&](const auto& entry) { return entry.codeOffset == codeOffset; })) {
            plan->Note(MissAheadPlan::Skip::Unprepared);
            return;
        }
    }
    if (!plan->Admit(key)) {
        plan->Note(MissAheadPlan::Skip::Duplicate);
        return;
    }
    if (missAheadKnown().Contains(key)) {
        plan->Note(MissAheadPlan::Skip::Cached);
        return;
    }
    auto shaderMemory = std::make_shared<ShaderMemory>(memory, &queryPendingWrite, &observePendingWrite, hookWaitCounter(), &QueryPendingSpan);
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> capture;
    try {
        auto invocation = InvocationFor(*snapshot, codeOffset, request);
        const SampledReadScope sampling(evidenceReads);
        capture = CaptureInvocation(*shaderMemory, *snapshot, codeOffset, request, invocation);
    } catch (const std::exception&) {
        plan->Note(MissAheadPlan::Skip::CaptureFailed);
        return;
    }
    if (capture == nullptr) {
        plan->Note(MissAheadPlan::Skip::CaptureFailed);
        return;
    }
    plan->Started();
    missAheadKnown().Insert(key);
    const bool tracked = missAheadProfiled();
    if (tracked) missAheadLedger().Started(key);
    SubmitRegistrationPreparation([snapshot, codeOffset, localDevice, address, key, tracked, shaderMemory, capture, ownedRequest = request, userData = std::vector<std::uint32_t>(request.context.userData.begin(), request.context.userData.end()), ownedMemory = std::vector<ShaderRecompiler::MemoryRegion>(memory.begin(), memory.end())]() mutable {
        struct Finish {
            std::uint64_t key;
            bool tracked;
            ~Finish() {
                if (tracked) missAheadLedger().Finished(key);
            }
        } finish{key, tracked};
        try {
            ownedRequest.context.userData = userData;
            ownedRequest.context.memory = ownedMemory;
            auto invocation = InvocationFor(*snapshot, codeOffset, ownedRequest);
            const auto result = invocation.Materialize(*capture);
            if (result == nullptr) return;
            localDevice->PrebuildComputePipeline(*result, address);
        } catch (const std::exception&) {}
    });
}

void Driver::reportGroupCapture(std::uint32_t queue) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    auto& stats = groupCaptureStats();
    const auto now = std::chrono::steady_clock::now();
    if (now - stats.reported < std::chrono::seconds(10)) return;
    AgcDriver::ProfilePrint_nid_no_patch("[groupcap] queue 0x%x (10 s): %llu runs resolved %llu dispatches (%llu cache hits) in %.1f ms (max %.2f ms); %llu adopted %.2f ms after the resolve on average, max %.2f ms; mismatches: address %llu key %llu memory %llu; %llu resolve failures, %llu stale, %llu label overlaps, %llu read an earlier dispatch's output, %llu stopped at unknown writes, %llu rejected by the device, %llu dropped because the argument re-read differed\n",
                                         queue, static_cast<unsigned long long>(stats.runs), static_cast<unsigned long long>(stats.resolved), static_cast<unsigned long long>(stats.resolvedHits), stats.resolveMs, stats.resolveMaxMs,
                                         static_cast<unsigned long long>(stats.adopted), stats.adopted != 0 ? stats.ageMs / static_cast<double>(stats.adopted) : 0.0, stats.ageMaxMs, static_cast<unsigned long long>(stats.addressMismatches), static_cast<unsigned long long>(stats.keyMismatches), static_cast<unsigned long long>(stats.memoryMismatches),
                                         static_cast<unsigned long long>(stats.failures), static_cast<unsigned long long>(stats.stale), static_cast<unsigned long long>(stats.labelOverlaps), static_cast<unsigned long long>(stats.writtenBefore), static_cast<unsigned long long>(stats.unknownWrites), static_cast<unsigned long long>(stats.deviceRejects), static_cast<unsigned long long>(stats.argumentMismatches));
    stats = GroupCaptureStats{};
    stats.reported = now;
}

}
