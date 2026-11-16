#include "prx/libSceAgcDriver/Core/include/LegacySlots.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace AgcDriver::Core {

namespace {

void bump(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

std::size_t checkedSlots(std::size_t slots) {
    if (slots >= std::size_t{LegacySlots::Full}) throw std::length_error("LegacySlots: too many slots");
    return slots;
}

}

struct LegacySlots::State {
    enum class Phase : std::uint8_t { Free, Held, Finished };

    struct alignas(64) Slot {
        Work work;
        std::atomic<Phase> phase{Phase::Free};
    };

    explicit State(std::size_t count) : slots(checkedSlots(count)), done(count) {
        vacant.reserve(count);
        for (auto slot = count; slot != 0; --slot) vacant.push_back(static_cast<std::uint32_t>(slot - 1));
    }

    Slot& claim(std::uint32_t slot) {
        if (slot >= slots.size()) throw std::out_of_range("LegacySlots: slot out of range");
        auto& entry = slots[slot];
        if (entry.phase.load(std::memory_order_relaxed) != Phase::Held) throw std::logic_error("LegacySlots: slot is not held");
        return entry;
    }

    void finish(std::uint32_t slot, std::atomic<std::uint64_t>& counter) noexcept {
        slots[slot].phase.store(Phase::Finished, std::memory_order_relaxed);
        done[writeAt] = slot;
        writeAt = writeAt + 1 == done.size() ? 0 : writeAt + 1;
        bump(counter);
        finished.store(finished.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    std::vector<Slot> slots;
    std::vector<std::uint32_t> done;
    alignas(64) std::vector<std::uint32_t> vacant;
    std::size_t readAt = 0;
    std::uint64_t drained = 0;
    std::atomic<std::uint64_t> held{0};
    std::atomic<std::uint64_t> reclaimed{0};
    std::atomic<std::uint64_t> full{0};
    alignas(64) std::atomic<std::uint64_t> finished{0};
    std::size_t writeAt = 0;
    std::atomic<std::uint64_t> run{0};
    std::atomic<std::uint64_t> skipped{0};
};

LegacySlots::LegacySlots(std::size_t slots) : state(std::make_unique<State>(slots)) {}

LegacySlots::~LegacySlots() = default;

std::uint32_t LegacySlots::Hold(Work work) {
    auto& table = *state;
    if (table.vacant.empty()) Reclaim();
    if (table.vacant.empty()) {
        bump(table.full);
        return Full;
    }
    const auto slot = table.vacant.back();
    table.vacant.pop_back();
    auto& entry = table.slots[slot];
    entry.work = std::move(work);
    entry.phase.store(State::Phase::Held, std::memory_order_relaxed);
    bump(table.held);
    return slot;
}

void LegacySlots::Run(std::uint32_t slot) {
    auto& table = *state;
    auto& entry = table.claim(slot);
    if (!entry.work) {
        table.finish(slot, table.run);
        throw std::bad_function_call();
    }
    try {
        entry.work();
    } catch (...) {
        table.finish(slot, table.run);
        throw;
    }
    table.finish(slot, table.run);
}

void LegacySlots::Skip(std::uint32_t slot) {
    auto& table = *state;
    table.claim(slot);
    table.finish(slot, table.skipped);
}

std::size_t LegacySlots::Reclaim() {
    auto& table = *state;
    const auto end = table.finished.load(std::memory_order_acquire);
    std::size_t count = 0;
    while (table.drained != end) {
        const auto slot = table.done[table.readAt];
        table.readAt = table.readAt + 1 == table.done.size() ? 0 : table.readAt + 1;
        ++table.drained;
        auto& entry = table.slots[slot];
        entry.work = nullptr;
        entry.phase.store(State::Phase::Free, std::memory_order_relaxed);
        table.vacant.push_back(slot);
        ++count;
    }
    if (count != 0) bump(table.reclaimed, count);
    return count;
}

LegacySlots::Counts LegacySlots::Counted() const {
    const auto& table = *state;
    Counts counts;
    counts.held = table.held.load(std::memory_order_relaxed);
    counts.run = table.run.load(std::memory_order_relaxed);
    counts.skipped = table.skipped.load(std::memory_order_relaxed);
    counts.reclaimed = table.reclaimed.load(std::memory_order_relaxed);
    counts.full = table.full.load(std::memory_order_relaxed);
    return counts;
}

}
