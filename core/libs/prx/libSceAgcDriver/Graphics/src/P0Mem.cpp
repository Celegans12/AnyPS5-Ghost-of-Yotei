#include "prx/libSceAgcDriver/Graphics/include/P0Mem.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>

namespace AgcDriver::Graphics::P0Mem {

namespace {

constexpr std::uint64_t PageShift = 12;
constexpr std::uint64_t PageBytes = std::uint64_t{1} << PageShift;
constexpr std::uint64_t UnitBytes = 0x10000;
constexpr std::uint64_t MaxRange = std::uint64_t{64} << 30u;
constexpr std::size_t ReasonSlots = 32;

unsigned ceilingMode() {
    static const unsigned mode = [] {
        const char* value = std::getenv("APS5_P0M_CEILING");
        if (value == nullptr) return 0u;
        const std::string_view text(value);
        unsigned result = 0;
        if (text == "staging" || text == "all") result |= 1u;
        if (text == "publish" || text == "all") result |= 2u;
        std::fprintf(stderr, "[p0m-ceiling] APS5_P0M_CEILING=%s: staging copies %s, shadow publishes %s (replay ablation, inexact)\n", value, (result & 1u) != 0 ? "skipped" : "kept", (result & 2u) != 0 ? "skipped" : "kept");
        return result;
    }();
    return mode;
}

struct CeilingCounters {
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CeilingCount::Count)> count{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CeilingCount::Count)> bytes{};
};

CeilingCounters& Ceilings() {
    static auto* counters = new CeilingCounters;
    return *counters;
}

double mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / 1048576.0;
}

struct Published {
    std::uint8_t reason = 0;
    bool reached = false;
    std::uint32_t bytes = 0;
};

struct ReasonStats {
    std::uint64_t published = 0;
    std::uint64_t boundRead = 0;
    std::uint64_t reachedThenGone = 0;
    std::uint64_t unreadRepublished = 0;
    std::uint64_t unreadOverwritten = 0;
};

struct Census {
    std::mutex mutex;
    std::set<std::uint64_t> dirty;
    std::uint64_t cpuWriteBytes = 0;
    std::uint64_t cpuDirtied = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(Read::Count)> uploads{};
    std::uint64_t writtenOver = 0;
    std::map<std::uint64_t, Published> pending;
    std::array<ReasonStats, ReasonSlots> reasons{};
    std::array<const char*, ReasonSlots> reasonNames{};
    std::uint64_t writerUses = 0;
    std::uint64_t leaseBytes = 0;
    std::uint64_t boundWritten = 0;
    std::unordered_set<std::uint64_t> leaseUnits;
    std::uint64_t reports = 0;
    std::uint64_t reportedBytes = 0;
};

Census& Get() {
    static auto* census = new Census;
    return *census;
}

void consumeDirty(Census& census, std::uint64_t begin, std::uint64_t end, std::uint64_t& counter) {
    if (end <= begin || end - begin > MaxRange || census.dirty.empty()) return;
    auto it = census.dirty.lower_bound(begin >> PageShift);
    const auto last = (end + PageBytes - 1u) >> PageShift;
    while (it != census.dirty.end() && *it < last) {
        ++counter;
        it = census.dirty.erase(it);
    }
}

template <typename Body>
void forPending(Census& census, std::uint64_t begin, std::uint64_t end, Body body) {
    if (end <= begin || end - begin > MaxRange || census.pending.empty()) return;
    auto it = census.pending.lower_bound(begin & ~(UnitBytes - 1u));
    while (it != census.pending.end() && it->first < end) it = body(it);
}

void gpuRead(Census& census, Read kind, std::uint64_t begin, std::uint64_t end) {
    consumeDirty(census, begin, end, census.uploads[static_cast<std::size_t>(kind)]);
    forPending(census, begin, end, [&](auto it) {
        if (kind == Read::Bound) {
            census.reasons[it->second.reason].boundRead += it->second.bytes;
            return census.pending.erase(it);
        }
        it->second.reached = true;
        return std::next(it);
    });
}

void retire(Census& census, const Published& entry, bool republished) {
    auto& stats = census.reasons[entry.reason];
    if (entry.reached) stats.reachedThenGone += entry.bytes;
    else if (republished) stats.unreadRepublished += entry.bytes;
    else stats.unreadOverwritten += entry.bytes;
}

}

bool CeilingStaging() {
    return (ceilingMode() & 1u) != 0;
}

bool CeilingPublish() {
    return (ceilingMode() & 2u) != 0;
}

void CountCeiling(CeilingCount which, std::uint64_t bytes) {
    auto& counters = Ceilings();
    counters.count[static_cast<std::size_t>(which)].fetch_add(1, std::memory_order_relaxed);
    counters.bytes[static_cast<std::size_t>(which)].fetch_add(bytes, std::memory_order_relaxed);
}

bool CensusEnabled() {
    static const bool enabled = [] {
        const bool on = std::getenv("APS5_PROFILE_P0MEM") != nullptr;
        if (on) std::fprintf(stderr, "[p0m] census on (APS5_PROFILE_P0MEM): uploads at 4 KiB, published shadow reads, unbounded writer leases\n");
        return on;
    }();
    return enabled;
}

void NoteCpuWrite(std::uint64_t address, std::uint64_t bytes) {
    if (!CensusEnabled() || bytes == 0) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    census.cpuWriteBytes += bytes;
    const auto end = address + bytes;
    for (auto page = address >> PageShift; page < (end + PageBytes - 1u) >> PageShift; ++page) {
        if (census.dirty.insert(page).second) ++census.cpuDirtied;
    }
    forPending(census, address, end, [&](auto it) {
        retire(census, it->second, false);
        return census.pending.erase(it);
    });
}

void NoteGpuRead(Read kind, std::uint64_t begin, std::uint64_t end) {
    if (!CensusEnabled()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    gpuRead(census, kind, begin, end);
}

void NoteGpuReads(Read kind, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    if (!CensusEnabled() || ranges.empty()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    for (const auto& [begin, end] : ranges) gpuRead(census, kind, begin, end);
}

void NoteGpuWrite(std::uint64_t begin, std::uint64_t end) {
    if (!CensusEnabled()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    consumeDirty(census, begin, end, census.writtenOver);
    forPending(census, begin, end, [&](auto it) {
        retire(census, it->second, false);
        return census.pending.erase(it);
    });
}

void NotePublish(std::uint64_t begin, std::uint64_t end, std::uint8_t reason, const char* reasonName) {
    if (!CensusEnabled() || end <= begin || reason >= ReasonSlots) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    census.reasonNames[reason] = reasonName;
    for (auto unit = begin & ~(UnitBytes - 1u); unit < end; unit += UnitBytes) {
        const auto from = std::max(unit, begin);
        const auto to = std::min(unit + UnitBytes, end);
        auto& entry = census.pending[unit];
        if (entry.bytes != 0) retire(census, entry, true);
        entry = Published{reason, false, static_cast<std::uint32_t>(to - from)};
        census.reasons[reason].published += to - from;
    }
}

void NoteUnboundedWriter(std::span<const std::pair<std::uint64_t, std::uint64_t>> writableLease, std::uint64_t boundWrittenBytes) {
    if (!CensusEnabled()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    ++census.writerUses;
    census.boundWritten += boundWrittenBytes;
    for (const auto& [begin, end] : writableLease) {
        if (end <= begin || end - begin > MaxRange) continue;
        census.leaseBytes += end - begin;
        for (auto unit = begin & ~(UnitBytes - 1u); unit < end; unit += UnitBytes) census.leaseUnits.insert(unit);
    }
}

void NoteBdaReport(std::uint64_t bytes) {
    if (!CensusEnabled()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    ++census.reports;
    census.reportedBytes += bytes;
}

void Close(const char* window) {
    if (ceilingMode() != 0) {
        auto& counters = Ceilings();
        const auto take = [&](CeilingCount which, std::uint64_t& bytes) {
            bytes = counters.bytes[static_cast<std::size_t>(which)].exchange(0, std::memory_order_relaxed);
            return static_cast<unsigned long long>(counters.count[static_cast<std::size_t>(which)].exchange(0, std::memory_order_relaxed));
        };
        std::uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
        const auto c0 = take(CeilingCount::CopyInSkipped, b0);
        const auto c1 = take(CeilingCount::CopyInFirst, b1);
        const auto c2 = take(CeilingCount::PristineSkipped, b2);
        const auto c3 = take(CeilingCount::CopyBackSkipped, b3);
        const auto c4 = take(CeilingCount::PublishSkipped, b4);
        ProfilePrint_nid_no_patch("[p0m-ceiling] window=%s copy-ins skipped %llu (%.1f MiB), made on first use %llu (%.1f MiB); pristine copies skipped %llu (%.1f MiB); copy-backs skipped %llu (%.1f MiB); shadow publishes skipped %llu (%.1f MiB)\n", window, c0, mib(b0), c1, mib(b1), c2, mib(b2), c3, mib(b3), c4, mib(b4));
    }
    if (!CensusEnabled()) return;
    auto& census = Get();
    std::lock_guard lock(census.mutex);
    const auto pages = [](std::uint64_t count) { return mib(count * PageBytes); };
    ProfilePrint_nid_no_patch("[p0m-upload] window=%s cpu writes %.1f MiB (%.1f MiB of 4 KiB pages newly dirty); uploads a device copy would need (4 KiB pages, CPU-dirty then GPU-used): bound %.1f MiB, reach only %.1f MiB, written over %.1f MiB; dirty pages not used yet %.1f MiB\n", window, mib(census.cpuWriteBytes), pages(census.cpuDirtied), pages(census.uploads[0]), pages(census.uploads[1]), pages(census.writtenOver), pages(census.dirty.size()));
    census.cpuWriteBytes = census.cpuDirtied = census.writtenOver = 0;
    census.uploads = {};
    std::string reasons;
    std::uint64_t pendingReached = 0, pendingUnread = 0;
    for (const auto& [unit, entry] : census.pending) (entry.reached ? pendingReached : pendingUnread) += entry.bytes;
    for (std::size_t index = 0; index < ReasonSlots; ++index) {
        auto& stats = census.reasons[index];
        if (stats.published == 0 && stats.boundRead == 0 && stats.reachedThenGone == 0 && stats.unreadRepublished == 0 && stats.unreadOverwritten == 0) continue;
        char text[256];
        std::snprintf(text, sizeof(text), " %s published %.1f, bound-read %.1f, reach-only %.1f, unread-republished %.1f, unread-overwritten %.1f;", census.reasonNames[index] != nullptr ? census.reasonNames[index] : "?", mib(stats.published), mib(stats.boundRead), mib(stats.reachedThenGone), mib(stats.unreadRepublished), mib(stats.unreadOverwritten));
        reasons += text;
        stats = ReasonStats{};
    }
    ProfilePrint_nid_no_patch("[p0m-publish] window=%s MiB by reason (a unit's publish ends at its first bound GPU read, its next publish, or a GPU/CPU write over it):%s pending now %.1f MiB (%.1f reached by a BDA table only, %.1f unread)\n", window, reasons.c_str(), mib(pendingReached + pendingUnread), mib(pendingReached), mib(pendingUnread));
    ProfilePrint_nid_no_patch("[p0m-lease] window=%s unbounded writer uses %llu: writable lease %.1f MiB summed over uses, %llu distinct 64 KiB units (%.1f MiB), bound written ranges %.1f MiB; BDA written-page reports %llu with %.1f MiB\n", window, static_cast<unsigned long long>(census.writerUses), mib(census.leaseBytes), static_cast<unsigned long long>(census.leaseUnits.size()), mib(census.leaseUnits.size() * UnitBytes), mib(census.boundWritten), static_cast<unsigned long long>(census.reports), mib(census.reportedBytes));
    census.writerUses = census.leaseBytes = census.boundWritten = census.reports = census.reportedBytes = 0;
    census.leaseUnits.clear();
}

}
