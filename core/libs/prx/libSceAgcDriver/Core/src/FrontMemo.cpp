#include "prx/libSceAgcDriver/Core/include/FrontMemo.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace AgcDriver::Core {

namespace {

constexpr std::uint64_t Golden = 0x9e3779b97f4a7c15ull;

std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33u;
    return value;
}

void bump(std::atomic<std::uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
}

}

struct FrontMemo::State {
    struct Entry {
        MemoKey key;
        std::uint64_t epoch = 0;
        RingSlice copy;
        bool used = false;
    };

    explicit State(std::size_t slots) : entries(std::bit_ceil(std::max<std::size_t>(slots, 1u))) {}

    std::size_t slotOf(const MemoKey& key) const {
        return static_cast<std::size_t>(mix(key.address + std::uint64_t{key.bytes} * Golden)) & (entries.size() - 1u);
    }

    std::vector<Entry> entries;
    std::atomic<std::uint64_t> finds{0};
    std::atomic<std::uint64_t> hits{0};
    std::atomic<std::uint64_t> stale{0};
    std::atomic<std::uint64_t> dead{0};
};

FrontMemo::FrontMemo(std::size_t slots) : state(std::make_unique<State>(slots)) {}

FrontMemo::~FrontMemo() = default;

bool FrontMemo::Find(const MemoKey& key, std::uint64_t epoch, const BlockRing& ring, RingSlice& copy) const {
    auto& memo = *state;
    bump(memo.finds);
    const auto& entry = memo.entries[memo.slotOf(key)];
    if (!entry.used || entry.key != key) return false;
    if (entry.epoch != epoch) {
        bump(memo.stale);
        return false;
    }
    if (!ring.Live(entry.copy)) {
        bump(memo.dead);
        return false;
    }
    bump(memo.hits);
    copy = entry.copy;
    return true;
}

void FrontMemo::Keep(const MemoKey& key, std::uint64_t epoch, const RingSlice& copy) {
    auto& entry = state->entries[state->slotOf(key)];
    entry.key = key;
    entry.epoch = epoch;
    entry.copy = copy;
    entry.used = true;
}

void FrontMemo::Clear() {
    std::fill(state->entries.begin(), state->entries.end(), State::Entry{});
}

FrontMemo::Counts FrontMemo::Counted() const {
    const auto& memo = *state;
    Counts counts;
    counts.finds = memo.finds.load(std::memory_order_relaxed);
    counts.hits = memo.hits.load(std::memory_order_relaxed);
    counts.stale = memo.stale.load(std::memory_order_relaxed);
    counts.dead = memo.dead.load(std::memory_order_relaxed);
    return counts;
}

}
