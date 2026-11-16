#include "prx/libSceAgcDriver/Core/include/WriteBlocks.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace AgcDriver::Core {

namespace {

constexpr std::uint64_t LastByte = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t LeafBlockShift = WriteBlocks::LeafShift - WriteBlocks::BlockShift;
constexpr std::uint64_t LeafBlocks = std::uint64_t{1} << LeafBlockShift;
constexpr std::uint64_t LeafCount = std::uint64_t{1} << (WriteBlocks::CoveredShift - WriteBlocks::LeafShift);
constexpr std::uint64_t LastCoveredBlock = (WriteBlocks::CoveredBytes >> WriteBlocks::BlockShift) - 1;
constexpr std::uint64_t BlocksPerWord = 8;
constexpr std::uint64_t WordsPerLeaf = LeafBlocks / BlocksPerWord;
constexpr std::uint64_t SummaryWords = 16;
constexpr std::uint64_t SpillMark = 0xff;

struct alignas(64) Leaf {
    std::atomic<std::uint64_t> occupied{0};
    alignas(64) std::array<std::atomic<std::uint64_t>, WordsPerLeaf> words{};
};

struct Entry {
    std::atomic<Leaf*> leaf{nullptr};
    std::atomic<std::uint64_t> covering{0};
};

struct RangeHash {
    std::size_t operator()(const WriteBlocks::Range& range) const noexcept {
        auto hash = range.first * 0x9e3779b97f4a7c15ull + range.second;
        hash ^= hash >> 31u;
        hash *= 0xbf58476d1ce4e5b9ull;
        hash ^= hash >> 29u;
        return static_cast<std::size_t>(hash);
    }
};

std::uint64_t lastByteOf(std::uint64_t address, std::uint64_t bytes) {
    return bytes - 1 > LastByte - address ? LastByte : address + bytes - 1;
}

void step(std::atomic<std::uint64_t>& counter, bool enter) {
    const auto value = counter.load(std::memory_order_relaxed);
    counter.store(enter ? value + 1 : value - 1, std::memory_order_release);
}

}

struct WriteBlocks::State {
    mutable std::mutex mutex;
    std::unordered_map<Range, std::uint64_t, RangeHash> registry;
    std::unordered_map<std::uint64_t, std::uint64_t> spill;
    alignas(64) std::atomic<std::uint64_t> high{0};
    alignas(64) std::array<Entry, LeafCount> directory{};

    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() {
        for (auto& entry : directory) delete entry.leaf.load(std::memory_order_relaxed);
    }

    void allocate(std::uint64_t index) {
        if (directory[index].leaf.load(std::memory_order_relaxed) != nullptr) return;
        directory[index].leaf.store(new Leaf(), std::memory_order_release);
    }

    void apply(const Range& range, bool enter) {
        const auto last = range.second - 1;
        if (range.first < CoveredBytes) {
            const auto firstBlock = range.first >> BlockShift;
            const auto lastBlock = std::min(last >> BlockShift, LastCoveredBlock);
            const auto from = [&](std::uint64_t leaf) { return std::max(firstBlock, leaf << LeafBlockShift) - (leaf << LeafBlockShift); };
            const auto to = [&](std::uint64_t leaf) { return std::min(lastBlock, (leaf << LeafBlockShift) + LeafBlocks - 1) - (leaf << LeafBlockShift); };
            const auto whole = [&](std::uint64_t leaf) { return from(leaf) == 0 && to(leaf) == LeafBlocks - 1; };
            const auto firstLeaf = firstBlock >> LeafBlockShift;
            const auto lastLeaf = lastBlock >> LeafBlockShift;
            if (enter) {
                if (!whole(firstLeaf)) allocate(firstLeaf);
                if (!whole(lastLeaf)) allocate(lastLeaf);
            }
            for (auto leaf = firstLeaf; leaf <= lastLeaf; ++leaf) {
                if (whole(leaf)) step(directory[leaf].covering, enter);
                else bump(leaf, from(leaf), to(leaf), enter);
            }
        }
        if (last >= CoveredBytes) step(high, enter);
    }

    void bump(std::uint64_t index, std::uint64_t from, std::uint64_t to, bool enter) {
        auto& leaf = *directory[index].leaf.load(std::memory_order_relaxed);
        const auto base = index << LeafBlockShift;
        for (auto word = from / BlocksPerWord; word <= to / BlocksPerWord; ++word) {
            const auto first = word == from / BlocksPerWord ? from % BlocksPerWord : 0;
            const auto last = word == to / BlocksPerWord ? to % BlocksPerWord : BlocksPerWord - 1;
            auto value = leaf.words[word].load(std::memory_order_relaxed);
            std::uint64_t changed = 0;
            for (auto slot = first; slot <= last; ++slot) {
                const auto block = base + word * BlocksPerWord + slot;
                const auto shift = slot * 8;
                auto holders = (value >> shift) & SpillMark;
                if (enter) {
                    if (holders == 0) ++changed;
                    if (holders < InlineHolders) {
                        ++holders;
                    } else if (holders == InlineHolders) {
                        spill.insert_or_assign(block, SpillMark);
                        holders = SpillMark;
                    } else {
                        ++spill.find(block)->second;
                    }
                } else if (holders == SpillMark) {
                    const auto found = spill.find(block);
                    if (--found->second == InlineHolders) {
                        spill.erase(found);
                        holders = InlineHolders;
                    }
                } else if (--holders == 0) {
                    ++changed;
                }
                value = (value & ~(SpillMark << shift)) | (holders << shift);
            }
            if (enter && changed != 0) leaf.occupied.store(leaf.occupied.load(std::memory_order_relaxed) + changed, std::memory_order_release);
            leaf.words[word].store(value, std::memory_order_release);
            if (!enter && changed != 0) leaf.occupied.store(leaf.occupied.load(std::memory_order_relaxed) - changed, std::memory_order_release);
        }
    }

    void release(const Range& range) {
        const auto found = registry.find(range);
        if (--found->second != 0) return;
        registry.erase(found);
        apply(range, false);
    }
};

WriteBlocks::WriteBlocks() : state(std::make_unique<State>()) {}

WriteBlocks::~WriteBlocks() = default;

void WriteBlocks::Enter(std::span<const Range> ranges) {
    std::lock_guard lock(state->mutex);
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const auto& range = ranges[index];
        if (range.second <= range.first) continue;
        try {
            const auto found = state->registry.try_emplace(range, 0).first;
            if (found->second++ != 0) continue;
            try {
                state->apply(range, true);
            } catch (...) {
                state->registry.erase(found);
                throw;
            }
        } catch (...) {
            for (std::size_t entered = 0; entered < index; ++entered) {
                if (ranges[entered].second > ranges[entered].first) state->release(ranges[entered]);
            }
            throw;
        }
    }
}

void WriteBlocks::Leave(std::span<const Range> ranges) {
    std::lock_guard lock(state->mutex);
    for (std::size_t index = 0; index < ranges.size(); ++index) {
        const auto& range = ranges[index];
        if (range.second <= range.first) continue;
        const auto found = state->registry.find(range);
        if (found != state->registry.end() && found->second > index) continue;
        const auto repeats = static_cast<std::uint64_t>(std::count(ranges.begin(), ranges.begin() + static_cast<std::ptrdiff_t>(index) + 1, range));
        if (found == state->registry.end() || found->second < repeats) throw std::logic_error("WriteBlocks: a range leaves more often than it entered");
    }
    for (const auto& range : ranges) {
        if (range.second > range.first) state->release(range);
    }
}

bool WriteBlocks::Clear(std::uint64_t address, std::uint64_t bytes) const noexcept {
    if (bytes == 0) return true;
    const auto last = lastByteOf(address, bytes);
    if (last >= CoveredBytes && state->high.load(std::memory_order_acquire) != 0) return false;
    if (address >= CoveredBytes) return true;
    const auto firstBlock = address >> BlockShift;
    const auto lastBlock = std::min(last >> BlockShift, LastCoveredBlock);
    for (auto index = firstBlock >> LeafBlockShift; index <= lastBlock >> LeafBlockShift; ++index) {
        const auto& entry = state->directory[index];
        if (entry.covering.load(std::memory_order_acquire) != 0) return false;
        const auto* leaf = entry.leaf.load(std::memory_order_acquire);
        if (leaf == nullptr) continue;
        const auto base = index << LeafBlockShift;
        const auto from = std::max(firstBlock, base) - base;
        const auto to = std::min(lastBlock, base + LeafBlocks - 1) - base;
        const auto firstWord = from / BlocksPerWord;
        const auto lastWord = to / BlocksPerWord;
        if (lastWord - firstWord >= SummaryWords && leaf->occupied.load(std::memory_order_acquire) == 0) continue;
        for (auto word = firstWord; word <= lastWord; ++word) {
            auto value = leaf->words[word].load(std::memory_order_acquire);
            if (word == firstWord) value &= LastByte << (from % BlocksPerWord * 8);
            if (word == lastWord) value &= LastByte >> ((BlocksPerWord - 1 - to % BlocksPerWord) * 8);
            if (value != 0) return false;
        }
    }
    return true;
}

bool WriteBlocks::Clear(std::span<const Range> ranges) const noexcept {
    return std::all_of(ranges.begin(), ranges.end(), [this](const Range& range) { return range.second <= range.first || Clear(range.first, range.second - range.first); });
}

std::uint64_t WriteBlocks::Holders(std::uint64_t address) const {
    std::lock_guard lock(state->mutex);
    if (address >= CoveredBytes) return state->high.load(std::memory_order_relaxed);
    const auto block = address >> BlockShift;
    const auto& entry = state->directory[block >> LeafBlockShift];
    auto holders = entry.covering.load(std::memory_order_relaxed);
    if (const auto* leaf = entry.leaf.load(std::memory_order_relaxed)) {
        const auto local = block & (LeafBlocks - 1);
        const auto kept = (leaf->words[local / BlocksPerWord].load(std::memory_order_relaxed) >> (local % BlocksPerWord * 8)) & SpillMark;
        holders += kept == SpillMark ? state->spill.at(block) : kept;
    }
    return holders;
}

std::size_t WriteBlocks::Ranges() const {
    std::lock_guard lock(state->mutex);
    return state->registry.size();
}

std::size_t WriteBlocks::Leaves() const {
    std::lock_guard lock(state->mutex);
    return static_cast<std::size_t>(std::count_if(state->directory.begin(), state->directory.end(), [](const Entry& entry) { return entry.leaf.load(std::memory_order_relaxed) != nullptr; }));
}

std::size_t WriteBlocks::Spilled() const {
    std::lock_guard lock(state->mutex);
    return state->spill.size();
}

}
