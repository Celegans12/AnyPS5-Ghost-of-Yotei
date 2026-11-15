#include "prx/libSceAgcDriver/Core/include/RunPlan.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace AgcDriver::Core {

namespace {

constexpr std::size_t MaxSlots = std::size_t{1} << 30u;

std::size_t slotCount(std::size_t slots) {
    return std::bit_ceil(std::clamp<std::size_t>(slots, 1, MaxSlots));
}

bool usesProgram(const StateKey& key, std::uint64_t program) {
    return std::find(key.programs.begin(), key.programs.end(), program) != key.programs.end();
}

}

struct RunPlans::State {
    explicit State(std::size_t count) : slots(slotCount(count)), mask(static_cast<std::uint64_t>(slots.size() - 1)) {}

    std::unique_ptr<RunPlan>& slotOf(const StateKey& key) { return slots[static_cast<std::size_t>(key.Hash() & mask)]; }

    void retire(std::unique_ptr<RunPlan>& slot) { retired.push_back(std::move(slot)); }

    std::vector<std::unique_ptr<RunPlan>> slots;
    std::vector<std::unique_ptr<RunPlan>> retired;
    std::uint64_t mask = 0;
    std::uint64_t serial = 0;
    Counts counts;
};

RunPlans::RunPlans(std::size_t slots) : state(std::make_unique<State>(slots)) {}

RunPlans::~RunPlans() = default;

const RunPlan* RunPlans::Find(const StateKey& key) const {
    State& table = *state;
    ++table.counts.finds;
    const RunPlan* plan = table.slotOf(key).get();
    if (plan == nullptr || !(plan->key == key)) return nullptr;
    ++table.counts.hits;
    return plan;
}

const RunPlan* RunPlans::Teach(RunPlan plan) {
    State& table = *state;
    std::unique_ptr<RunPlan>& slot = table.slotOf(plan.key);
    if (slot != nullptr && slot->key == plan.key) return slot.get();
    plan.id = ++table.serial;
    std::unique_ptr<RunPlan> taught = std::make_unique<RunPlan>(std::move(plan));
    if (slot != nullptr) {
        table.retire(slot);
        ++table.counts.evicted;
    }
    slot = std::move(taught);
    ++table.counts.taught;
    return slot.get();
}

void RunPlans::Forget(std::uint64_t program) {
    if (program == 0) return;
    State& table = *state;
    for (std::unique_ptr<RunPlan>& slot : table.slots) {
        if (slot == nullptr || !usesProgram(slot->key, program)) continue;
        table.retire(slot);
        ++table.counts.forgotten;
    }
}

RunPlans::Counts RunPlans::Counted() const {
    return state->counts;
}

}
