#include "prx/libSceAgcDriver/Graphics/include/HomeCensus.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileOutput.hpp"
#include "prx/libSceAgcDriver/Execution/include/ProfileWindows.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace AgcDriver::Graphics::HomeCensus {

namespace {

constexpr std::uint64_t UnitBytes = 65536;
constexpr std::uint64_t ChunkBytes = 1024;
constexpr std::uint64_t MaxRangeBytes = std::uint64_t{1} << 30u;

enum Reason : std::size_t { BatchEndReason, Stale, LaneChange, Foreign, ReadElement, ReadGpuCopy, ReadIndirect, ReadStorage, ReadCopySource, ReadDraw, ReadOther, Bda, ReasonCount };
constexpr std::array<const char*, ReasonCount> ReasonNames{"batch-end", "stale", "lane", "foreign-write", "read-element", "read-gpucopy", "read-indirect", "read-storage", "read-copysrc", "read-draw", "read-other", "bda"};

struct Unit {
    std::uint64_t sync = 0;
    std::uint64_t epoch = 0;
    int lane = -1;
    std::uint64_t seeded = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> dirty;
};

struct Counts {
    std::uint64_t seedBytes = 0;
    std::uint64_t seedUses = 0;
    std::array<std::uint64_t, ReasonCount> settleBytes{};
    std::array<std::uint64_t, ReasonCount> settles{};
    std::uint64_t batchSettles = 0;
};

std::uint64_t chunkMask(std::uint64_t unitBase, std::uint64_t begin, std::uint64_t end) {
    const auto first = (begin - unitBase) / ChunkBytes;
    const auto last = (end - unitBase - 1u) / ChunkBytes;
    const auto width = last - first + 1u;
    const std::uint64_t bits = width >= 64u ? ~std::uint64_t{0} : ((std::uint64_t{1} << width) - 1u);
    return bits << first;
}

std::uint64_t maskedBytes(std::uint64_t unitBase, std::uint64_t mask, std::uint64_t begin, std::uint64_t end) {
    std::uint64_t bytes = 0;
    while (mask != 0) {
        const auto chunk = static_cast<std::uint64_t>(std::countr_zero(mask));
        mask &= mask - 1u;
        const auto from = std::max(begin, unitBase + chunk * ChunkBytes);
        const auto to = std::min(end, unitBase + (chunk + 1u) * ChunkBytes);
        if (from < to) bytes += to - from;
    }
    return bytes;
}

void addInterval(std::vector<std::pair<std::uint64_t, std::uint64_t>>& list, std::uint64_t begin, std::uint64_t end) {
    list.emplace_back(begin, end);
    std::sort(list.begin(), list.end());
    std::size_t out = 0;
    for (std::size_t index = 1; index < list.size(); ++index) {
        if (list[index].first <= list[out].second) list[out].second = std::max(list[out].second, list[index].second);
        else list[++out] = list[index];
    }
    list.resize(out + 1u);
}

bool intervalsOverlap(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& list, std::uint64_t begin, std::uint64_t end) {
    return std::any_of(list.begin(), list.end(), [&](const auto& range) { return range.first < end && begin < range.second; });
}

struct Model {
    bool pessimistic = false;
    std::uint64_t epoch = 1;
    std::unordered_map<std::uint64_t, Unit> units;
    std::unordered_set<std::uint64_t> dirtyUnits;
    Counts counts;
    std::unordered_set<int> settledLanes;

    void settle(std::uint64_t key, Unit& unit, Reason reason) {
        if (unit.dirty.empty()) return;
        std::uint64_t bytes = 0;
        for (const auto& [begin, end] : unit.dirty) bytes += end - begin;
        counts.settleBytes[reason] += bytes;
        counts.settles[reason] += unit.dirty.size();
        unit.dirty.clear();
        dirtyUnits.erase(key);
    }

    bool valid(const Unit& unit) const { return unit.sync != 0 && unit.epoch == epoch; }

    void use(int lane, std::uint64_t begin, std::uint64_t end, Ranges writes, std::uint64_t generation) {
        bool seeded = false;
        for (auto unitBase = begin & ~(UnitBytes - 1u); unitBase < end; unitBase += UnitBytes) {
            const auto key = unitBase / UnitBytes;
            auto& unit = units[key];
            const auto from = std::max(begin, unitBase);
            const auto to = std::min(end, unitBase + UnitBytes);
            if (unit.lane >= 0 && unit.lane != lane) {
                settle(key, unit, LaneChange);
                unit.sync = 0;
                unit.seeded = 0;
            }
            const bool current = valid(unit) && GuestMemory::UnchangedSince(unitBase, static_cast<std::size_t>(UnitBytes), unit.sync);
            if (!current) {
                settle(key, unit, Stale);
                unit.seeded = 0;
            }
            const auto mask = chunkMask(unitBase, from, to);
            const auto need = mask & ~unit.seeded;
            if (need != 0) {
                counts.seedBytes += maskedBytes(unitBase, need, from, to);
                seeded = true;
            }
            unit.seeded |= mask;
            unit.sync = generation;
            unit.epoch = epoch;
            unit.lane = lane;
            for (const auto& [writeBegin, writeEnd] : writes) {
                const auto wFrom = std::max(writeBegin, from);
                const auto wTo = std::min(writeEnd, to);
                if (wFrom >= wTo) continue;
                addInterval(unit.dirty, wFrom, wTo);
                dirtyUnits.insert(key);
            }
        }
        if (seeded) ++counts.seedUses;
    }

    void foreign(std::uint64_t begin, std::uint64_t end) {
        for (auto unitBase = begin & ~(UnitBytes - 1u); unitBase < end; unitBase += UnitBytes) {
            const auto found = units.find(unitBase / UnitBytes);
            if (found == units.end()) continue;
            if (intervalsOverlap(found->second.dirty, begin, end)) settle(found->first, found->second, Foreign);
            found->second.sync = 0;
        }
    }

    void read(Reason reason, std::uint64_t begin, std::uint64_t end) {
        if (dirtyUnits.empty()) return;
        for (auto unitBase = begin & ~(UnitBytes - 1u); unitBase < end; unitBase += UnitBytes) {
            const auto key = unitBase / UnitBytes;
            if (!dirtyUnits.contains(key)) continue;
            auto& unit = units[key];
            if (intervalsOverlap(unit.dirty, begin, end)) settle(key, unit, reason);
        }
    }

    void bda(bool writer) {
        if (!pessimistic) return;
        std::vector<std::uint64_t> keys(dirtyUnits.begin(), dirtyUnits.end());
        for (const auto key : keys) settle(key, units[key], Bda);
        if (writer) ++epoch;
    }

    void batchEnd(int lane) {
        std::vector<std::uint64_t> keys;
        for (const auto key : dirtyUnits) {
            if (units[key].lane == lane) keys.push_back(key);
        }
        if (!keys.empty()) ++counts.batchSettles;
        for (const auto key : keys) settle(key, units[key], BatchEndReason);
    }

    void marksBefore(Ranges staged, std::vector<std::pair<std::uint64_t, bool>>& out) const {
        for (const auto& [begin, end] : staged) {
            for (auto unitBase = begin & ~(UnitBytes - 1u); unitBase < end; unitBase += UnitBytes) {
                const auto found = units.find(unitBase / UnitBytes);
                if (found == units.end()) continue;
                out.emplace_back(found->first, valid(found->second) && GuestMemory::UnchangedSince(unitBase, static_cast<std::size_t>(UnitBytes), found->second.sync));
            }
        }
    }

    void marksAfter(const std::vector<std::pair<std::uint64_t, bool>>& before, std::uint64_t generation) {
        for (const auto& [key, current] : before) {
            const auto found = units.find(key);
            if (found == units.end()) continue;
            found->second.sync = current ? generation : 0;
        }
    }

    std::pair<std::uint64_t, std::uint64_t> footprint() const {
        std::uint64_t live = 0, bytes = 0;
        for (const auto& [key, unit] : units) {
            if (unit.seeded == 0) continue;
            ++live;
            bytes += static_cast<std::uint64_t>(std::popcount(unit.seeded)) * ChunkBytes;
        }
        return {live, bytes};
    }
};

struct Today {
    std::uint64_t inBytes = 0;
    std::uint64_t inUses = 0;
    std::uint64_t residentBytes = 0;
    std::uint64_t backBytes = 0;
    std::uint64_t backRanges = 0;
    std::uint64_t batches = 0;
    std::uint64_t bdaUses = 0;
    std::uint64_t bdaWriters = 0;
};

struct State {
    std::mutex mutex;
    std::array<Model, 2> models;
    Today today;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    State() { models[1].pessimistic = true; }
};

State& Get() {
    static auto* state = new State;
    return *state;
}

thread_local std::vector<std::pair<std::uint64_t, std::uint64_t>>* noteScope = nullptr;
thread_local int ownCopyDepth = 0;
thread_local int muteDepth = 0;
thread_local std::array<std::vector<std::pair<std::uint64_t, bool>>, 2> marks;

double mib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

std::string formatModel(const char* name, const Model& model) {
    const auto& c = model.counts;
    std::uint64_t settled = 0, commands = 0;
    for (std::size_t index = 0; index < ReasonCount; ++index) {
        settled += c.settleBytes[index];
        commands += c.settles[index];
    }
    char text[1024];
    int length = std::snprintf(text, sizeof(text), " %s: seed %.2f MiB/%llu uses, settle %.2f MiB/%llu copies (batch settles %llu;", name, mib(c.seedBytes), static_cast<unsigned long long>(c.seedUses), mib(settled), static_cast<unsigned long long>(commands), static_cast<unsigned long long>(c.batchSettles));
    for (std::size_t index = 0; index < ReasonCount && length > 0 && static_cast<std::size_t>(length) < sizeof(text); ++index) {
        if (c.settles[index] == 0) continue;
        length += std::snprintf(text + length, sizeof(text) - static_cast<std::size_t>(length), " %s %.2f/%llu", ReasonNames[index], mib(c.settleBytes[index]), static_cast<unsigned long long>(c.settles[index]));
    }
    const auto [live, bytes] = model.footprint();
    if (length > 0 && static_cast<std::size_t>(length) < sizeof(text)) std::snprintf(text + length, sizeof(text) - static_cast<std::size_t>(length), "); units %llu = %.1f MiB seeded;", static_cast<unsigned long long>(live), mib(bytes));
    return text;
}

void print(State& state, const char* window) {
    const auto& t = state.today;
    char head[512];
    std::snprintf(head, sizeof(head), "[v3m-census] window=%s today: copy-in %.2f MiB/%llu uses (resident skip %.2f MiB), copy-back %.2f MiB/%llu ranges; batches %llu, bda uses %llu (writers %llu);", window, mib(t.inBytes), static_cast<unsigned long long>(t.inUses), mib(t.residentBytes), mib(t.backBytes), static_cast<unsigned long long>(t.backRanges), static_cast<unsigned long long>(t.batches), static_cast<unsigned long long>(t.bdaUses), static_cast<unsigned long long>(t.bdaWriters));
    const auto line = std::string(head) + formatModel("O", state.models[0]) + formatModel("P", state.models[1]);
    AgcDriver::ProfilePrint_nid_no_patch("%s\n", line.c_str());
    state.today = Today{};
    for (auto& model : state.models) model.counts = Counts{};
}

Reason readReason(int kind) {
    switch (static_cast<Recorder::ReadKind>(kind)) {
    case Recorder::ReadKind::DispatchElement: return ReadElement;
    case Recorder::ReadKind::GpuCopy: return ReadGpuCopy;
    case Recorder::ReadKind::Indirect: return ReadIndirect;
    case Recorder::ReadKind::StorageUpload: return ReadStorage;
    case Recorder::ReadKind::CopySource: return ReadCopySource;
    case Recorder::ReadKind::DrawInput: return ReadDraw;
    default: return ReadOther;
    }
}

bool bdaKind(int kind) {
    return kind == static_cast<int>(Recorder::ReadKind::AddressBased) || kind == static_cast<int>(Recorder::ReadKind::AddressBasedReadOnly);
}

}

bool Enabled() {
    static const bool enabled = std::getenv("APS5_V3M_CENSUS") != nullptr;
    return enabled;
}

void Use(std::uint64_t begin, std::uint64_t end, Ranges writes, bool residentSkip) {
    if (!Enabled() || end <= begin || end - begin > MaxRangeBytes) return;
    const int lane = Recorder::ThreadLane();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> inside;
    for (const auto& [writeBegin, writeEnd] : writes) {
        const auto from = std::max(writeBegin, begin);
        const auto to = std::min(writeEnd, end);
        if (from < to) inside.emplace_back(from, to);
    }
    const auto first = begin & ~(UnitBytes - 1u);
    const auto last = (end + UnitBytes - 1u) & ~(UnitBytes - 1u);
    const auto generation = GuestMemory::CollectWrites(first, static_cast<std::size_t>(last - first));
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    if (residentSkip) state.today.residentBytes += end - begin;
    else state.today.inBytes += end - begin;
    ++state.today.inUses;
    for (auto& model : state.models) model.use(lane, begin, end, inside, generation);
}

void CopyBack(std::uint64_t bytes) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    state.today.backBytes += bytes;
    ++state.today.backRanges;
}

void BeforeMarks(Ranges staged) {
    if (!Enabled()) return;
    for (auto& list : marks) list.clear();
    if (staged.empty()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    for (std::size_t index = 0; index < state.models.size(); ++index) state.models[index].marksBefore(staged, marks[index]);
}

void AfterMarks() {
    if (!Enabled() || (marks[0].empty() && marks[1].empty())) return;
    const auto generation = GuestMemory::TrackerGeneration();
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    for (std::size_t index = 0; index < state.models.size(); ++index) state.models[index].marksAfter(marks[index], generation);
    for (auto& list : marks) list.clear();
}

void WriteNote(std::uint64_t begin, std::uint64_t end) {
    if (!Enabled() || end <= begin) return;
    if (noteScope != nullptr && std::any_of(noteScope->begin(), noteScope->end(), [&](const auto& range) { return range.first <= begin && end <= range.second; })) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    for (auto& model : state.models) model.foreign(begin, end);
}

void Read(int kind, std::uint64_t begin, std::uint64_t end) {
    if (!Enabled() || end <= begin || muteDepth > 0) return;
    if (kind == static_cast<int>(Recorder::ReadKind::GpuCopy) && ownCopyDepth > 0) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    if (bdaKind(kind)) {
        ++state.today.bdaUses;
        const bool writer = kind == static_cast<int>(Recorder::ReadKind::AddressBased);
        if (writer) ++state.today.bdaWriters;
        for (auto& model : state.models) model.bda(writer);
        return;
    }
    if (end - begin > MaxRangeBytes) return;
    for (auto& model : state.models) model.read(readReason(kind), begin, end);
}

void ReadGroup(int kind, Ranges ranges) {
    if (!Enabled() || ranges.empty() || muteDepth > 0) return;
    if (bdaKind(kind)) {
        Read(kind, ranges.front().first, ranges.front().second);
        return;
    }
    for (const auto& [begin, end] : ranges) Read(kind, begin, end);
}

void BatchEnd(int lane) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    ++state.today.batches;
    for (auto& model : state.models) model.batchEnd(lane);
    if (ProfileWindows::ClockDue(state.lastReport)) print(state, "10 s");
}

void Close(const char* window) {
    if (!Enabled()) return;
    auto& state = Get();
    std::lock_guard lock(state.mutex);
    print(state, window);
}

NoteScope::NoteScope(std::vector<std::pair<std::uint64_t, std::uint64_t>> staged) {
    if (!Enabled()) return;
    noteScope = new std::vector<std::pair<std::uint64_t, std::uint64_t>>(std::move(staged));
}

NoteScope::~NoteScope() {
    delete noteScope;
    noteScope = nullptr;
}

Mute::Mute() {
    ++muteDepth;
}

Mute::~Mute() {
    --muteDepth;
}

OwnCopyScope::OwnCopyScope(bool active) : active(active) {
    if (active) ++ownCopyDepth;
}

OwnCopyScope::~OwnCopyScope() {
    if (active) --ownCopyDepth;
}

}
