#include "prx/libSceAgcDriver/Execution/include/C3Log.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace AgcDriver::C3Log {

namespace {

constexpr std::size_t ProofCount = static_cast<std::size_t>(Proof::Count);
constexpr std::size_t SkipCount = static_cast<std::size_t>(Skip::Count);
constexpr std::array<const char*, ProofCount> ProofNames{"served", "valid-moved", "imports", "evicted", "pending", "changed", "keys", "other"};
constexpr std::array<const char*, SkipCount> SkipNames{"uncacheable", "no-entry", "in-use", "no-log"};
constexpr std::size_t LogLimit = 16384;

struct KeyHash {
    std::size_t operator()(const std::vector<std::uint32_t>& key) const {
        std::uint64_t hash = 0xcbf29ce484222325ull;
        for (const auto word : key) {
            hash ^= word;
            hash *= 0x100000001b3ull;
        }
        return static_cast<std::size_t>(hash);
    }
};

struct Logged {
    std::vector<std::uint64_t> addresses;
    std::vector<std::uint32_t> sizes;
    std::vector<std::byte> bytes;
};

constexpr std::size_t StepCount = static_cast<std::size_t>(Step::Count);
constexpr std::array<const char*, StepCount> StepNames{"none", "not-reusable", "in-use", "snapshots", "space", "moved", "prove", "imports", "threw"};

struct SplitTotals {
    std::uint64_t hits = 0, served = 0, keyNs = 0, findNs = 0, snapshotsNs = 0, spaceNs = 0, proveNs = 0, importsNs = 0, movedNs = 0, epochReused = 0, keysBuilt = 0;
    std::array<std::uint64_t, StepCount> refused{};
};
SplitTotals splits;

struct Totals {
    std::array<std::array<std::uint64_t, ProofCount>, 2> pairs{};
    std::array<std::uint64_t, SkipCount> skips{};
    std::uint64_t compared = 0, proofNs = 0, memcmpNs = 0, bytes = 0, regions = 0, maxBytes = 0;
};

std::mutex mutex;
std::unordered_map<std::vector<std::uint32_t>, Logged, KeyHash> logs;
Totals totals;
std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();

void reportLocked(const char* label) {
    std::string line = "[c3log] ";
    line += label;
    char text[256];
    const auto perHit = [&](std::uint64_t value) { return totals.compared == 0 ? 0.0 : static_cast<double>(value) / static_cast<double>(totals.compared); };
    std::snprintf(text, sizeof(text), ": %llu compared hits, log %.0f bytes in %.1f regions per hit (max %llu), memcmp %.2f us, proof %.2f us per hit; memcmp same / differs by proof verdict:", static_cast<unsigned long long>(totals.compared), perHit(totals.bytes), perHit(totals.regions), static_cast<unsigned long long>(totals.maxBytes), perHit(totals.memcmpNs) / 1e3, perHit(totals.proofNs) / 1e3);
    line += text;
    for (std::size_t i = 0; i < ProofCount; ++i) {
        if (totals.pairs[0][i] == 0 && totals.pairs[1][i] == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu/%llu", ProofNames[i], static_cast<unsigned long long>(totals.pairs[1][i]), static_cast<unsigned long long>(totals.pairs[0][i]));
        line += text;
    }
    line += "; not compared:";
    for (std::size_t i = 0; i < SkipCount; ++i) {
        std::snprintf(text, sizeof(text), " %s %llu", SkipNames[i], static_cast<unsigned long long>(totals.skips[i]));
        line += text;
    }
    std::snprintf(text, sizeof(text), "; %zu logs", logs.size());
    line += text;
    const auto per = [&](std::uint64_t ns) { return splits.hits == 0 ? 0.0 : static_cast<double>(ns) / 1e3 / static_cast<double>(splits.hits); };
    std::snprintf(text, sizeof(text), "; lookup split per hit (%llu hits, %llu served, %llu keys built here, %llu epoch-reused proofs): key %.2f find %.2f snapshots %.2f space %.2f prove %.2f imports %.2f moved %.2f us; refused at:", static_cast<unsigned long long>(splits.hits), static_cast<unsigned long long>(splits.served), static_cast<unsigned long long>(splits.keysBuilt), static_cast<unsigned long long>(splits.epochReused), per(splits.keyNs), per(splits.findNs), per(splits.snapshotsNs), per(splits.spaceNs), per(splits.proveNs), per(splits.importsNs), per(splits.movedNs));
    line += text;
    for (std::size_t i = 1; i < StepCount; ++i) {
        if (splits.refused[i] == 0) continue;
        std::snprintf(text, sizeof(text), " %s %llu", StepNames[i], static_cast<unsigned long long>(splits.refused[i]));
        line += text;
    }
    line += "\n";
    AgcDriver::ProfilePrint_nid_no_patch("%s", line.c_str());
    totals = {};
    splits = {};
}

}

bool Enabled() {
    static const bool enabled = [] {
        const char* text = std::getenv("APS5_V3E_C3_LOG");
        return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0;
    }();
    return enabled;
}

void Record(std::span<const std::uint32_t> key, std::span<const Region> regions) {
    Logged logged;
    std::size_t total = 0;
    for (const auto& [address, bytes] : regions) total += bytes.size();
    logged.addresses.reserve(regions.size());
    logged.sizes.reserve(regions.size());
    logged.bytes.reserve(total);
    for (const auto& [address, bytes] : regions) {
        logged.addresses.push_back(address);
        logged.sizes.push_back(static_cast<std::uint32_t>(bytes.size()));
        logged.bytes.insert(logged.bytes.end(), bytes.begin(), bytes.end());
    }
    std::lock_guard lock(mutex);
    if (logs.size() >= LogLimit) logs.clear();
    logs[std::vector<std::uint32_t>(key.begin(), key.end())] = std::move(logged);
}

bool Same(std::span<const std::uint32_t> key, std::span<const Region> regions, std::uint64_t& bytes, bool& logged, std::uint64_t& compareNs) {
    std::lock_guard lock(mutex);
    bytes = 0;
    compareNs = 0;
    const auto found = logs.find(std::vector<std::uint32_t>(key.begin(), key.end()));
    logged = found != logs.end();
    if (!logged) return false;
    const auto& log = found->second;
    bytes = log.bytes.size();
    const auto start = std::chrono::steady_clock::now();
    const auto compare = [&] {
        if (log.addresses.size() != regions.size()) return false;
        std::size_t offset = 0;
        for (std::size_t i = 0; i < regions.size(); ++i) {
            const auto& [address, current] = regions[i];
            if (address != log.addresses[i] || current.size() != log.sizes[i]) return false;
            if (!current.empty() && std::memcmp(log.bytes.data() + offset, current.data(), current.size()) != 0) return false;
            offset += current.size();
        }
        return true;
    };
    const bool same = compare();
    compareNs = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    return same;
}

void NoteCompared(bool memcmpSame, Proof proof, std::uint64_t proofNs, std::uint64_t memcmpNs, std::uint64_t bytes, std::size_t regions) {
    std::lock_guard lock(mutex);
    ++totals.pairs[memcmpSame ? 1 : 0][static_cast<std::size_t>(proof)];
    ++totals.compared;
    totals.proofNs += proofNs;
    totals.memcmpNs += memcmpNs;
    totals.bytes += bytes;
    totals.regions += regions;
    if (bytes > totals.maxBytes) totals.maxBytes = bytes;
}

Split& ThreadSplit() {
    thread_local Split split;
    return split;
}

void NoteSplit(const Split& split, bool served) {
    std::lock_guard lock(mutex);
    ++splits.hits;
    if (served) ++splits.served;
    splits.keyNs += split.keyNs;
    splits.findNs += split.findNs;
    splits.snapshotsNs += split.snapshotsNs;
    splits.spaceNs += split.spaceNs;
    splits.proveNs += split.proveNs;
    splits.importsNs += split.importsNs;
    splits.movedNs += split.movedNs;
    if (split.epochReused) ++splits.epochReused;
    if (split.keyBuilt) ++splits.keysBuilt;
    ++splits.refused[static_cast<std::size_t>(split.refused)];
}

void NoteSkipped(Skip reason) {
    std::lock_guard lock(mutex);
    ++totals.skips[static_cast<std::size_t>(reason)];
}

void Close(const char* label) {
    if (!Enabled()) return;
    std::lock_guard lock(mutex);
    reportLocked(label);
}

void MaybeReport() {
    if (!Enabled() || ProfileWindows::FrameWindows()) return;
    std::unique_lock lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) return;
    if (!ProfileWindows::ClockDue(lastReport)) return;
    reportLocked("10s");
}

}
