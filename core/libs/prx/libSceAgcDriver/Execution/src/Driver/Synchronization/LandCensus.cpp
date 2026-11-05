#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/LandCensus.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace AgcDriver::DriverDetail::LandCensus {

bool Traced();

namespace {

struct QueueStats {
    std::uint64_t landings = 0, waited = 0, ahead = 0, skippedWaits = 0;
    double ms = 0;
    std::map<std::uint32_t, std::uint64_t> writers;
    std::uint64_t commands = 0, adopted = 0, changed = 0, changedAdopted = 0, regions = 0, changedRegions = 0, changedBytes = 0;
    std::uint64_t conds = 0, condNoPending = 0, condLabelEqual = 0, condLabelDiffer = 0, condOther = 0, condWaits = 0;
    double condMs = 0;
};

struct Census {
    std::mutex mutex;
    std::map<std::uint32_t, QueueStats> queues;
    std::chrono::steady_clock::time_point reported = std::chrono::steady_clock::now();
};

Census& Get() {
    static Census census;
    return census;
}

struct Armed {
    bool armed = false;
    std::uint64_t generation = 0;
};

Armed& ThreadArmed() {
    static thread_local Armed armed;
    return armed;
}

void ReportLocked(Census& census) {
    const auto now = std::chrono::steady_clock::now();
    if (now - census.reported < std::chrono::seconds(10)) return;
    census.reported = now;
    for (auto& [queue, s] : census.queues) {
        std::string writers;
        for (const auto& [writer, count] : s.writers) {
            char text[40];
            std::snprintf(text, sizeof(text), " 0x%x:%llu", writer, static_cast<unsigned long long>(count));
            writers += text;
        }
        AgcDriver::ProfilePrint_nid_no_patch(
            "[v3w-land] queue 0x%x (10 s): landings %llu, waited %llu in %.1f ms, resolved ahead %llu, skipped (ceiling) %llu waits; writers%s; commands after a landing %llu (adopted %llu): CPU store since the pass %llu (adopted %llu), regions %llu/%llu, bytes %llu; COND_EXEC %llu: no pending writer %llu, table label equal %llu, differ %llu, other writer %llu, hook waits %llu in %.1f ms\n",
            queue, static_cast<unsigned long long>(s.landings), static_cast<unsigned long long>(s.waited), s.ms, static_cast<unsigned long long>(s.ahead), static_cast<unsigned long long>(s.skippedWaits), writers.empty() ? " none" : writers.c_str(),
            static_cast<unsigned long long>(s.commands), static_cast<unsigned long long>(s.adopted), static_cast<unsigned long long>(s.changed), static_cast<unsigned long long>(s.changedAdopted), static_cast<unsigned long long>(s.changedRegions), static_cast<unsigned long long>(s.regions), static_cast<unsigned long long>(s.changedBytes),
            static_cast<unsigned long long>(s.conds), static_cast<unsigned long long>(s.condNoPending), static_cast<unsigned long long>(s.condLabelEqual), static_cast<unsigned long long>(s.condLabelDiffer), static_cast<unsigned long long>(s.condOther), static_cast<unsigned long long>(s.condWaits), s.condMs);
        s = QueueStats{};
    }
}

}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_V3W_CENSUS") != nullptr;
    return enabled;
}

bool Traced() {
    static const bool traced = [] {
        const char* value = std::getenv("APS5_V3W_CENSUS");
        return value != nullptr && std::string(value) == "trace";
    }();
    return traced;
}

void NoteLanding(std::uint32_t queue, std::uint32_t writer, bool waited, double ms, bool ahead) {
    if (!Enabled()) return;
    if (Traced()) std::fprintf(stderr, "[v3w-land] queue 0x%x landed a wait on queue 0x%x's label: %s %.2f ms, %s\n", queue, writer, waited ? "waited" : "already in memory", ms, ahead ? "group resolved before landing" : "group resolved after landing");
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    auto& s = census.queues[queue];
    ++s.landings;
    if (waited) ++s.waited;
    s.ms += ms;
    if (ahead) ++s.ahead;
    ++s.writers[writer];
    ReportLocked(census);
}

void NoteSkipped(std::uint32_t queue, std::size_t waits) {
    if (!Enabled()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    census.queues[queue].skippedWaits += waits;
    ReportLocked(census);
}

void Arm(std::uint64_t generation) {
    if (!Enabled()) return;
    auto& armed = ThreadArmed();
    if (!armed.armed || generation < armed.generation) armed.generation = generation;
    armed.armed = true;
}

void Disarm() {
    if (!Enabled()) return;
    ThreadArmed() = Armed{};
}

void CheckCaptured(std::uint32_t queue, std::span<const ShaderRecompiler::MemoryRegion> regions, bool adopted) {
    if (!Enabled()) return;
    const auto armed = ThreadArmed();
    if (!armed.armed || armed.generation == 0) return;
    std::uint64_t changedRegions = 0, changedBytes = 0;
    for (const auto& region : regions) {
        if (region.bytes.empty()) continue;
        if (!GuestMemory::UnchangedSinceCollected(region.guestAddress, region.bytes.size(), armed.generation)) {
            ++changedRegions;
            changedBytes += region.bytes.size();
        }
    }
    if (Traced()) std::fprintf(stderr, "[v3w-land] queue 0x%x command after a landing: %s, %zu regions, %llu changed by a CPU store since the wait passed\n", queue, adopted ? "adopted the ahead result" : "captured again", regions.size(), static_cast<unsigned long long>(changedRegions));
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    auto& s = census.queues[queue];
    ++s.commands;
    if (adopted) ++s.adopted;
    s.regions += regions.size();
    if (changedRegions != 0) {
        ++s.changed;
        if (adopted) ++s.changedAdopted;
        s.changedRegions += changedRegions;
        s.changedBytes += changedBytes;
    }
    ReportLocked(census);
}

CondProbe BeforeCondition(std::uint64_t address) {
    CondProbe probe;
    if (!Enabled()) return probe;
    probe.pending = Graphics::Recorder::SnapshotWriteOverlaps(address, 4);
    probe.label = Graphics::Recorder::LookupLabelValue(address, 4, 0);
    probe.waitedBefore = Graphics::Recorder::ThreadWaitedMs();
    return probe;
}

void AfterCondition(std::uint32_t queue, const CondProbe& probe, std::uint32_t value) {
    if (!Enabled()) return;
    const double waited = Graphics::Recorder::ThreadWaitedMs() - probe.waitedBefore;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    auto& s = census.queues[queue];
    ++s.conds;
    if (!probe.pending && !probe.label.has_value()) ++s.condNoPending;
    else if (probe.label.has_value() && static_cast<std::uint32_t>(*probe.label) == value) ++s.condLabelEqual;
    else if (probe.label.has_value()) ++s.condLabelDiffer;
    else ++s.condOther;
    if (waited > 0) {
        ++s.condWaits;
        s.condMs += waited;
    }
    ReportLocked(census);
}

}
