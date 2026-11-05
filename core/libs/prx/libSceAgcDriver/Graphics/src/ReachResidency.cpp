#include "prx/libSceAgcDriver/Graphics/include/ReachResidency.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/MemoryStats.hpp"
#include "BdaAbi.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>

namespace AgcDriver::Graphics::Reach {

namespace {

constexpr std::size_t HistoryUses = 64;
constexpr std::size_t HistoryUnits = 1024;
constexpr std::uint64_t LearnFirst = 4;
constexpr std::uint64_t LearnEvery = 64;
constexpr std::uint32_t HotLimit = 2;
constexpr std::uint64_t DecayMaps = 1024;

std::uint64_t poolSlots() {
    static const std::uint64_t slots = [] {
        const char* value = std::getenv("APS5_REACH_RESIDENCY_MIB");
        const auto mib = value != nullptr && *value != '\0' ? std::strtoull(value, nullptr, 10) : 64ull;
        return std::clamp<std::uint64_t>(mib, 1, 1024) * (std::uint64_t{1} << 20u) / UnitBytes;
    }();
    return slots;
}

struct SlotState {
    std::uint64_t unit = 0;
    bool filled = false;
    bool cross = false;
    std::uint64_t batch = 0;
    std::uint64_t writerSeq = 0;
    std::uint64_t syncGeneration = 0;
    std::uint64_t forget = 0;
    std::uint64_t lastUse = 0;
    // Fills this slot lost to write notes, halved every DecayMaps maps: a unit written this often is not mapped.
    std::uint32_t hot = 0;
};

struct Lane {
    std::shared_ptr<Buffer> pool;
    std::uint64_t epoch = 0;
    std::vector<SlotState> slots;
    std::unordered_map<std::uint64_t, std::uint32_t> byUnit;
    std::uint64_t clock = 0;
    std::uint64_t lastWriter = 0;
    bool refused = false;
};

struct History {
    std::uint64_t builds = 0;
    std::deque<std::vector<std::uint64_t>> uses;
};

struct State {
    std::mutex mutex;
    std::array<Lane, 2> lanes;
    std::uint64_t writerSeq = 0;
    std::map<std::uint64_t, int> incomplete;
    Counts counts;
    Counts reported;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::mutex historyMutex;
    std::unordered_map<std::uint64_t, History> history;
};

State& state() {
    static auto* value = new State();
    return *value;
}

// APS5_REACH_HISTORY=<file> (diagnostic, placement only): the learned units are loaded from the file at the
// first use and written back at every profile window, so a later process (a replay row) starts predicted.
const char* historyFile() {
    static const char* file = [] {
        const char* value = std::getenv("APS5_REACH_HISTORY");
        return value != nullptr && *value != '\0' ? value : nullptr;
    }();
    return file;
}

void loadHistory(State& s) {
    static bool loaded = false;
    if (loaded) return;
    loaded = true;
    const char* file = historyFile();
    if (file == nullptr) return;
    std::ifstream in(file, std::ios::binary);
    std::uint64_t program = 0, count = 0;
    std::size_t programs = 0;
    while (in.read(reinterpret_cast<char*>(&program), 8) && in.read(reinterpret_cast<char*>(&count), 8)) {
        if (count > HistoryUnits * HistoryUses) break;
        std::vector<std::uint64_t> units(static_cast<std::size_t>(count));
        if (count != 0 && !in.read(reinterpret_cast<char*>(units.data()), static_cast<std::streamsize>(count * 8))) break;
        auto& history = s.history[program];
        history.builds = std::max<std::uint64_t>(history.builds, 1);
        history.uses.push_back(std::move(units));
        ++programs;
    }
    std::fprintf(stderr, "[reach] history: %zu programs loaded from %s\n", programs, file);
}

void saveHistory(State& s) {
    const char* file = historyFile();
    if (file == nullptr) return;
    const std::string temporary = std::string(file) + ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        for (const auto& [program, history] : s.history) {
            std::vector<std::uint64_t> units;
            for (const auto& use : history.uses) units.insert(units.end(), use.begin(), use.end());
            std::sort(units.begin(), units.end());
            units.erase(std::unique(units.begin(), units.end()), units.end());
            if (units.empty()) continue;
            const std::uint64_t count = units.size();
            out.write(reinterpret_cast<const char*>(&program), 8);
            out.write(reinterpret_cast<const char*>(&count), 8);
            out.write(reinterpret_cast<const char*>(units.data()), static_cast<std::streamsize>(count * 8));
        }
    }
    std::remove(file);
    std::rename(temporary.c_str(), file);
}

std::uint64_t newestIncomplete(const State& s) {
    return s.incomplete.empty() ? 0 : s.incomplete.rbegin()->first;
}

bool current(State& s, const LaneView& view, const SlotState& slot, bool counted = true) {
    Counts ignored;
    auto& c = counted ? s.counts : ignored;
    const auto& lane = s.lanes[static_cast<std::size_t>(view.lane)];
    if (!slot.filled) return false;
    if (slot.batch == view.batch) {
        if (lane.lastWriter > slot.writerSeq) {
            ++c.invalidWriter;
            return false;
        }
        return true;
    }
    if (!slot.cross) {
        ++c.invalidBatch;
        return false;
    }
    if (newestIncomplete(s) > slot.writerSeq) {
        ++c.invalidWriter;
        return false;
    }
    if (slot.forget != GuestMemory::ForgetSerial()) {
        ++c.invalidStamp;
        return false;
    }
    static_cast<void>(GuestMemory::CollectWrites(slot.unit, static_cast<std::size_t>(UnitBytes)));
    if (!GuestMemory::UnchangedSince(slot.unit, static_cast<std::size_t>(UnitBytes), slot.syncGeneration)) {
        ++c.invalidStamp;
        return false;
    }
    return true;
}

}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_REACH_RESIDENCY") != nullptr;
    return enabled;
}

bool CopiesActive() {
    return Enabled() && !RuntimeSwitches::Set(RuntimeSwitch::NoReachCopies);
}

bool LearnBuild(std::uint64_t program) {
    if (!Enabled()) return false;
    auto& s = state();
    std::lock_guard lock(s.historyMutex);
    loadHistory(s);
    auto& history = s.history[program];
    const auto build = history.builds++;
    // APS5_REACH_NO_LEARN (diagnostic, with APS5_REACH_HISTORY): a program the loaded history knows learns no more.
    static const bool noLearn = std::getenv("APS5_REACH_NO_LEARN") != nullptr;
    if (noLearn && !history.uses.empty()) return false;
    return build < LearnFirst || build % LearnEvery == 0;
}

void NoteLearned(std::uint64_t program, std::uint32_t* words) {
    namespace Abi = ShaderRecompiler::BdaAbi;
    std::vector<std::uint64_t> units;
    const bool valid = Abi::ForEachReachPage(words, [&](std::uint64_t page) { units.push_back(page & ~(UnitBytes - 1u)); });
    const auto count = std::min<std::uint32_t>(words[Abi::ReachCountWord], Abi::ReachSlots);
    for (std::uint32_t claim = 0; claim < count; ++claim) {
        const auto slot = words[Abi::ReachListWord + claim];
        if (slot >= Abi::ReachSlots) continue;
        words[Abi::ReachSlotsWord + 2 * slot] = 0;
        words[Abi::ReachSlotsWord + 2 * slot + 1] = 0;
    }
    words[Abi::ReachCountWord] = 0;
    words[Abi::ReachOverflowWord] = 0;
    if (!valid) return;
    std::sort(units.begin(), units.end());
    units.erase(std::unique(units.begin(), units.end()), units.end());
    auto& s = state();
    std::lock_guard lock(s.historyMutex);
    auto& history = s.history[program];
    history.uses.push_back(std::move(units));
    while (history.uses.size() > HistoryUses) history.uses.pop_front();
    ++s.counts.learnedUses;
}

std::vector<std::uint64_t> Predicted(std::uint64_t program) {
    auto& s = state();
    std::lock_guard lock(s.historyMutex);
    loadHistory(s);
    std::vector<std::uint64_t> units;
    const auto found = s.history.find(program);
    if (found == s.history.end()) return units;
    for (auto use = found->second.uses.rbegin(); use != found->second.uses.rend(); ++use) {
        units.insert(units.end(), use->begin(), use->end());
        if (units.size() >= HistoryUnits) break;
    }
    std::sort(units.begin(), units.end());
    units.erase(std::unique(units.begin(), units.end()), units.end());
    if (units.size() > HistoryUnits) units.resize(HistoryUnits);
    return units;
}

bool LearningPending(std::uint64_t program) {
    auto& s = state();
    std::lock_guard lock(s.historyMutex);
    loadHistory(s);
    const auto found = s.history.find(program);
    return found != s.history.end() && found->second.builds != 0 && found->second.uses.empty();
}

std::uint64_t WriterRecorded(int lane) {
    if (!Enabled()) return 0;
    auto& s = state();
    std::lock_guard stateLock(state().mutex);
    const auto seq = ++s.writerSeq;
    s.incomplete.emplace(seq, lane);
    s.lanes[static_cast<std::size_t>(lane & 1)].lastWriter = seq;
    return seq;
}

void WriterCompleted(std::uint64_t seq) {
    if (seq == 0) return;
    std::lock_guard stateLock(state().mutex);
    state().incomplete.erase(seq);
}

void NoteWrite(std::uint64_t begin, std::uint64_t end) {
    if (!Enabled() || end <= begin) return;
    auto& s = state();
    std::lock_guard stateLock(state().mutex);
    for (auto& lane : s.lanes) {
        if (lane.byUnit.empty()) continue;
        const auto first = begin & ~(UnitBytes - 1u);
        if ((end - first) / UnitBytes <= lane.byUnit.size()) {
            for (auto unit = first; unit < end; unit += UnitBytes) {
                const auto found = lane.byUnit.find(unit);
                if (found == lane.byUnit.end()) continue;
                auto& slot = lane.slots[found->second];
                if (slot.filled) {
                    ++s.counts.invalidNote;
                    ++slot.hot;
                }
                slot.filled = false;
            }
        } else {
            for (auto& slot : lane.slots) {
                if (!slot.filled || slot.unit + UnitBytes <= begin || slot.unit >= end) continue;
                ++s.counts.invalidNote;
                ++slot.hot;
                slot.filled = false;
            }
        }
    }
}

bool Map(const Context& context, const LaneView& view, std::span<const UnitSource> candidates, Mapping& mapping, std::vector<Fill>& fills) {
    auto& s = state();
    std::lock_guard stateLock(state().mutex);
    auto& lane = s.lanes[static_cast<std::size_t>(view.lane)];
    if (lane.slots.empty()) {
        if (lane.refused) return false;
        try {
            const MemoryStats::OwnerScope owner(MemoryStats::Owner::GuestStaging);
            lane.pool = std::make_shared<Buffer>(context, static_cast<std::size_t>(poolSlots() * UnitBytes), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        } catch (const std::exception& error) {
            lane.refused = true;
            std::fprintf(stderr, "[reach] lane %d pool of %llu MiB refused: %s\n", view.lane, static_cast<unsigned long long>(poolSlots() * UnitBytes >> 20u), error.what());
            return false;
        }
        lane.slots.assign(static_cast<std::size_t>(poolSlots()), SlotState{});
        lane.byUnit.clear();
        ++lane.epoch;
    }
    mapping = Mapping{view.lane, lane.epoch, lane.pool, {}, {}};
    const auto now = ++lane.clock;
    if (now % DecayMaps == 0) {
        for (auto& slot : lane.slots) slot.hot >>= 1u;
    }
    // Slots used by this build are never taken by another of its units.
    std::size_t evictCursor = 0;
    std::vector<std::uint32_t> lru;
    for (const auto& candidate : candidates) {
        std::uint32_t index;
        if (const auto found = lane.byUnit.find(candidate.unit); found != lane.byUnit.end()) {
            index = found->second;
            if (lane.slots[index].hot >= HotLimit) {
                ++s.counts.refusedHot;
                continue;
            }
        } else if (lane.byUnit.size() < lane.slots.size()) {
            index = static_cast<std::uint32_t>(lane.byUnit.size());
            lane.byUnit.emplace(candidate.unit, index);
            lane.slots[index] = SlotState{candidate.unit};
        } else {
            if (lru.empty()) {
                lru.resize(lane.slots.size());
                for (std::uint32_t i = 0; i < lru.size(); ++i) lru[i] = i;
                std::sort(lru.begin(), lru.end(), [&](std::uint32_t a, std::uint32_t b) { return lane.slots[a].lastUse < lane.slots[b].lastUse; });
            }
            while (evictCursor < lru.size() && lane.slots[lru[evictCursor]].lastUse == now) ++evictCursor;
            if (evictCursor == lru.size()) break;
            index = lru[evictCursor++];
            lane.byUnit.erase(lane.slots[index].unit);
            lane.byUnit.emplace(candidate.unit, index);
            lane.slots[index] = SlotState{candidate.unit};
            ++s.counts.evicted;
        }
        auto& slot = lane.slots[index];
        slot.lastUse = now;
        mapping.units.push_back(candidate);
        mapping.slots.push_back(index);
        if (current(s, view, slot)) {
            ++s.counts.reused;
        } else {
            fills.push_back({candidate.import, candidate.offset, VkDeviceSize{index} * UnitBytes, candidate.unit});
        }
    }
    return !mapping.units.empty();
}

bool Refresh(const LaneView& view, const Mapping& mapping, std::vector<Fill>& fills, bool counted) {
    auto& s = state();
    std::lock_guard stateLock(state().mutex);
    if (counted) ++s.counts.refreshes;
    auto& lane = s.lanes[static_cast<std::size_t>(view.lane)];
    if (view.lane != mapping.lane || lane.pool != mapping.pool || lane.epoch != mapping.epoch) {
        ++s.counts.refreshFailed;
        return false;
    }
    const auto now = ++lane.clock;
    for (std::size_t i = 0; i < mapping.units.size(); ++i) {
        auto& slot = lane.slots[mapping.slots[i]];
        if (slot.unit != mapping.units[i].unit) {
            ++s.counts.refreshFailed;
            fills.clear();
            return false;
        }
        slot.lastUse = now;
        if (current(s, view, slot, counted)) {
            if (counted) ++s.counts.reused;
        }
        else fills.push_back({mapping.units[i].import, mapping.units[i].offset, VkDeviceSize{mapping.slots[i]} * UnitBytes, mapping.units[i].unit});
    }
    return true;
}

void Filled(const LaneView& view, std::span<const Fill> fills) {
    auto& s = state();
    std::lock_guard stateLock(state().mutex);
    auto& lane = s.lanes[static_cast<std::size_t>(view.lane)];
    for (const auto& fill : fills) {
        const auto found = lane.byUnit.find(fill.unit);
        if (found == lane.byUnit.end()) continue;
        auto& slot = lane.slots[found->second];
        slot.filled = true;
        slot.batch = view.batch;
        slot.writerSeq = s.writerSeq;
        slot.cross = !view.otherLanePending(fill.unit, fill.unit + UnitBytes);
        slot.forget = GuestMemory::ForgetSerial();
        slot.syncGeneration = GuestMemory::CollectWrites(fill.unit, static_cast<std::size_t>(UnitBytes));
        if (slot.syncGeneration == 0) slot.cross = false;
        ++s.counts.fills;
        s.counts.fillBytes += UnitBytes;
    }
}

void CountBuild(bool learning, std::size_t predicted, std::size_t mapped, std::size_t refused) {
    std::lock_guard stateLock(state().mutex);
    auto& c = state().counts;
    ++c.builds;
    if (learning) ++c.learning;
    c.predictedUnits += predicted;
    c.mappedUnits += mapped;
    c.refusedUnits += refused;
}

namespace {
std::array<std::atomic<std::uint64_t>, 6> candidateCounts{};
std::array<std::uint64_t, 6> candidateReported{};
}

void CountCandidate(int what) {
    if (what >= 0 && what < 6) candidateCounts[static_cast<std::size_t>(what)].fetch_add(1, std::memory_order_relaxed);
}

Counts Totals() {
    std::lock_guard stateLock(state().mutex);
    return state().counts;
}

namespace {

void printCounts(State& s, const Counts& r, const char* what) {
    const auto& c = s.counts;
    const std::size_t resident = s.lanes[0].byUnit.size() + s.lanes[1].byUnit.size();
    std::fprintf(stderr, "[reach] %s builds %llu (learning %llu, learned uses %llu); units predicted %llu, mapped %llu, refused %llu; fills %llu (%.1f MiB), reused %llu; invalid: note %llu, writer %llu, batch %llu, stamp %llu; evicted %llu, hot refused %llu; reuse refreshes %llu (failed %llu); slots held %zu, writers incomplete %zu\n", what,
        static_cast<unsigned long long>(c.builds - r.builds), static_cast<unsigned long long>(c.learning - r.learning), static_cast<unsigned long long>(c.learnedUses - r.learnedUses),
        static_cast<unsigned long long>(c.predictedUnits - r.predictedUnits), static_cast<unsigned long long>(c.mappedUnits - r.mappedUnits), static_cast<unsigned long long>(c.refusedUnits - r.refusedUnits),
        static_cast<unsigned long long>(c.fills - r.fills), static_cast<double>(c.fillBytes - r.fillBytes) / 1048576.0, static_cast<unsigned long long>(c.reused - r.reused),
        static_cast<unsigned long long>(c.invalidNote - r.invalidNote), static_cast<unsigned long long>(c.invalidWriter - r.invalidWriter), static_cast<unsigned long long>(c.invalidBatch - r.invalidBatch), static_cast<unsigned long long>(c.invalidStamp - r.invalidStamp),
        static_cast<unsigned long long>(c.evicted - r.evicted), static_cast<unsigned long long>(c.refusedHot - r.refusedHot), static_cast<unsigned long long>(c.refreshes - r.refreshes), static_cast<unsigned long long>(c.refreshFailed - r.refreshFailed), resident, s.incomplete.size());
}

Counts& windowReported() {
    static Counts counts;
    return counts;
}

}

void Report() {
    static const bool profile = std::getenv("APS5_PROFILE_REACH") != nullptr;
    if (!profile) return;
    auto& s = state();
    std::lock_guard stateLock(s.mutex);
    const auto now = std::chrono::steady_clock::now();
    if (now - s.lastReport < std::chrono::seconds(10)) return;
    s.lastReport = now;
    printCounts(s, s.reported, "(10 s)");
    s.reported = s.counts;
}

void CloseWindow(const char* label) {
    if (!Enabled()) return;
    auto& s = state();
    std::lock_guard stateLock(s.mutex);
    std::string what = std::string("window ") + (label != nullptr ? label : "?") + ":";
    printCounts(s, windowReported(), what.c_str());
    {
        std::lock_guard historyLock(s.historyMutex);
        loadHistory(s);
        saveHistory(s);
    }
    windowReported() = s.counts;
    std::uint64_t now[6];
    for (std::size_t i = 0; i < 6; ++i) now[i] = candidateCounts[i].load(std::memory_order_relaxed);
    std::fprintf(stderr, "[reach] %s compute BDA builds: read-only %llu, writers %llu; read-only skipped: no staging %llu, not addressable %llu, no lease %llu, no recorder %llu\n", what.c_str(),
        static_cast<unsigned long long>(now[0] - candidateReported[0]), static_cast<unsigned long long>(now[1] - candidateReported[1]), static_cast<unsigned long long>(now[2] - candidateReported[2]),
        static_cast<unsigned long long>(now[3] - candidateReported[3]), static_cast<unsigned long long>(now[4] - candidateReported[4]), static_cast<unsigned long long>(now[5] - candidateReported[5]));
    for (std::size_t i = 0; i < 6; ++i) candidateReported[i] = now[i];
}

void UsePoolForTests(int lane, std::size_t slots) {
    auto& s = state();
    std::lock_guard stateLock(s.mutex);
    auto& held = s.lanes[static_cast<std::size_t>(lane & 1)];
    held = Lane{};
    held.slots.assign(slots, SlotState{});
    held.epoch = 1;
}

void ResetForTests() {
    auto& s = state();
    std::lock_guard stateLock(s.mutex);
    for (auto& lane : s.lanes) lane = Lane{};
    s.writerSeq = 0;
    s.incomplete.clear();
    s.counts = {};
    s.reported = {};
    std::lock_guard lock(s.historyMutex);
    s.history.clear();
}

}
