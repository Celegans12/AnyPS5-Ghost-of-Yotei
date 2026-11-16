#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Core/include/RunPlan.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool Present(const RunPlan* plan, const char* what) {
    Expect(plan != nullptr, what);
    return plan != nullptr;
}

constexpr std::size_t Slots = 16;
constexpr std::size_t DefaultSlots = std::size_t{1} << 14u;

using Programs = std::array<std::uint64_t, 4>;

StateKey stateKey(std::uint64_t device, Programs programs = {0x1000, 0, 0, 0}) {
    StateKey key;
    key.shader = {0x11, 0x12};
    key.context = {0x21, 0x22};
    key.userConfig = {0x31, 0x32};
    key.programs = programs;
    for (std::size_t stage = 0; stage < programs.size(); ++stage) key.programHashes[stage] = programs[stage] == 0 ? 0 : programs[stage] ^ 0xabcd0000u;
    key.indexType = 1;
    key.topology = 4;
    key.device = device;
    return key;
}

std::shared_ptr<void> token(int value) {
    return std::make_shared<int>(value);
}

RunPlan runPlan(const StateKey& key, std::shared_ptr<void> pipeline) {
    RunPlan plan;
    plan.key = key;
    plan.stageCount = 2;
    plan.stages[0].variantKey = 0x55;
    plan.stages[0].userDataBase = 0x8c;
    plan.stages[1].pushOffsetBytes = 16;
    plan.pipeline = std::move(pipeline);
    plan.resources = token(-1);
    plan.accessSummary = 9;
    plan.vertexOffsetRegister = 2;
    plan.instanceOffsetRegister = 3;
    return plan;
}

std::uint64_t slotOf(const StateKey& key, std::size_t slots) {
    return key.Hash() % slots;
}

StateKey placed(StateKey key, std::size_t slots, std::vector<std::uint64_t>& taken) {
    for (std::uint64_t device = 1; device <= 0x10000; ++device) {
        key.device = device;
        const std::uint64_t slot = slotOf(key, slots);
        if (std::find(taken.begin(), taken.end(), slot) != taken.end()) continue;
        taken.push_back(slot);
        return key;
    }
    return key;
}

std::pair<StateKey, StateKey> sharingSlot(std::size_t slots) {
    std::vector<StateKey> seen;
    for (std::uint64_t device = 1; device <= slots + 1; ++device) {
        const StateKey candidate = stateKey(device);
        for (const StateKey& earlier : seen) {
            if (slotOf(earlier, slots) == slotOf(candidate, slots)) return {earlier, candidate};
        }
        seen.push_back(candidate);
    }
    return {seen.front(), seen.front()};
}

void testTeachThenFind() {
    RunPlans plans;
    std::vector<std::uint64_t> taken;
    const StateKey first = placed(stateKey(0), DefaultSlots, taken);
    const StateKey second = placed(stateKey(0, {0x1000, 0x2000, 0, 0}), DefaultSlots, taken);
    Expect(plans.Find(first) == nullptr, "an empty table finds nothing");
    const std::shared_ptr<void> pipeline = token(1);
    const RunPlan* taught = plans.Teach(runPlan(first, pipeline));
    if (!Present(taught, "a taught plan is returned")) return;
    Expect(taught->key == first, "a taught plan keeps its key");
    Expect(taught->id != 0, "a taught plan gets an identifier");
    Expect(taught->pipeline == pipeline && taught->resources != nullptr, "a taught plan holds its pipeline and its resources");
    Expect(taught->stageCount == 2 && taught->stages[0].variantKey == 0x55 && taught->stages[0].userDataBase == 0x8c && taught->stages[1].pushOffsetBytes == 16, "a taught plan keeps its stages");
    Expect(taught->accessSummary == 9 && taught->vertexOffsetRegister == 2 && taught->instanceOffsetRegister == 3, "a taught plan keeps its access summary and its offset registers");
    Expect(plans.Find(first) == taught, "a taught plan is found by its key");
    Expect(plans.Find(second) == nullptr, "a key never taught is not found");
    const RunPlan* other = plans.Teach(runPlan(second, token(2)));
    if (!Present(other, "a second plan is returned")) return;
    Expect(other != taught && other->id != taught->id, "two plans have two identifiers");
    Expect(plans.Find(second) == other && plans.Find(first) == taught, "two plans in two slots are both found");
    RunPlans::Counts counts = plans.Counted();
    Expect(counts.finds == 5 && counts.hits == 3, "finds and hits are counted");
    Expect(counts.taught == 2 && counts.evicted == 0 && counts.forgotten == 0, "taught plans are counted");
    std::shared_ptr<void> twin = token(3);
    const std::weak_ptr<void> twinAlive = twin;
    RunPlan again = runPlan(first, std::move(twin));
    again.stages[0].variantKey = 0x66;
    const RunPlan* kept = plans.Teach(std::move(again));
    Expect(kept == taught && taught->stages[0].variantKey == 0x55 && taught->pipeline == pipeline, "teaching a known key keeps the plan in place");
    Expect(twinAlive.expired(), "the plan taught for a known key is dropped");
    counts = plans.Counted();
    Expect(counts.taught == 2 && counts.evicted == 0, "a plan taught for a known key is neither taught nor evicted");
}

void testCollision() {
    RunPlans plans(Slots);
    const auto [first, second] = sharingSlot(Slots);
    Expect(!(first == second), "two different keys share a slot");
    const RunPlan* a = plans.Teach(runPlan(first, token(1)));
    if (!Present(a, "the first plan of the slot is returned")) return;
    Expect(plans.Find(second) == nullptr, "a key sharing the slot of a taught plan is not found");
    const RunPlan* b = plans.Teach(runPlan(second, token(2)));
    if (!Present(b, "the second plan of the slot is returned")) return;
    Expect(b != a && b->key == second, "the second key gets its own plan");
    Expect(plans.Find(second) == b, "the later plan takes the slot");
    Expect(plans.Find(first) == nullptr, "the earlier plan is no longer found");
    const RunPlans::Counts counts = plans.Counted();
    Expect(counts.taught == 2 && counts.evicted == 1, "the plan pushed out of its slot is counted as evicted");
}

void testWholeKey() {
    RunPlans plans(1);
    const StateKey base = stateKey(1, {0x1000, 0x2000, 0x3000, 0x4000});
    const RunPlan* taught = plans.Teach(runPlan(base, token(1)));
    if (!Present(taught, "a one slot table returns its plan")) return;
    Expect(plans.Find(base) == taught, "a one slot table finds its plan");
    const auto differs = [&](auto change, const char* what) {
        StateKey other = base;
        change(other);
        Expect(plans.Find(other) == nullptr, what);
    };
    differs([](StateKey& k) { k.shader.low ^= 1; }, "the low shader digest is compared");
    differs([](StateKey& k) { k.shader.high ^= 1; }, "the high shader digest is compared");
    differs([](StateKey& k) { k.context.low ^= 1; }, "the low context digest is compared");
    differs([](StateKey& k) { k.context.high ^= 1; }, "the high context digest is compared");
    differs([](StateKey& k) { k.userConfig.low ^= 1; }, "the low user config digest is compared");
    differs([](StateKey& k) { k.userConfig.high ^= 1; }, "the high user config digest is compared");
    differs([](StateKey& k) { k.programs[0] ^= 0x100; }, "the first program is compared");
    differs([](StateKey& k) { k.programs[3] ^= 0x100; }, "the last program is compared");
    differs([](StateKey& k) { k.programHashes[1] ^= 1; }, "the program hashes are compared");
    differs([](StateKey& k) { k.indexType ^= 1; }, "the index type is compared");
    differs([](StateKey& k) { k.topology ^= 1; }, "the topology is compared");
    differs([](StateKey& k) { k.device ^= 1; }, "the device is compared");
    Expect(plans.Counted().hits == 1, "only the whole key hits");
}

void testEvictedStaysValid() {
    std::weak_ptr<void> held;
    {
        RunPlans plans(1);
        const StateKey first = stateKey(1);
        const StateKey second = stateKey(2);
        std::shared_ptr<void> pipeline = token(1);
        held = pipeline;
        const void* raw = pipeline.get();
        const RunPlan* evicted = plans.Teach(runPlan(first, std::move(pipeline)));
        if (!Present(evicted, "the plan to evict is returned")) return;
        const std::uint64_t id = evicted->id;
        const RunPlan* current = plans.Teach(runPlan(second, token(2)));
        if (!Present(current, "the evicting plan is returned")) return;
        Expect(plans.Counted().evicted == 1, "an eviction is counted");
        Expect(plans.Find(first) == nullptr && plans.Find(second) == current, "the evicting plan takes the slot");
        Expect(evicted->key == first && evicted->id == id, "an evicted plan keeps its key and its identifier");
        Expect(evicted->pipeline.get() == raw && evicted->stageCount == 2 && evicted->stages[0].variantKey == 0x55, "an evicted plan keeps its contents");
        Expect(!held.expired(), "an evicted plan still holds its pipeline");
        const RunPlan* back = plans.Teach(runPlan(first, token(3)));
        if (!Present(back, "a key taught again after its eviction is returned")) return;
        Expect(back != evicted && back->id != id && back->id != current->id, "a key taught again after its eviction gets a new plan");
        Expect(plans.Find(first) == back && plans.Find(second) == nullptr, "the plan taught again takes the slot");
        const RunPlans::Counts counts = plans.Counted();
        Expect(counts.taught == 3 && counts.evicted == 2, "each eviction is counted");
        Expect(evicted->key == first && current->key == second && !held.expired(), "every evicted plan stays readable");
    }
    Expect(held.expired(), "evicted plans are released with the table");
}

void testForget() {
    RunPlans plans(Slots);
    std::vector<std::uint64_t> taken;
    const StateKey both = placed(stateKey(0, {0x1000, 0x2000, 0, 0}), Slots, taken);
    const StateKey last = placed(stateKey(0, {0x3000, 0, 0, 0x2000}), Slots, taken);
    const StateKey shared = placed(stateKey(0, {0x1000, 0x4000, 0, 0}), Slots, taken);
    const StateKey alone = placed(stateKey(0, {0x5000, 0, 0, 0}), Slots, taken);
    const std::shared_ptr<void> pipeline = token(1);
    const std::weak_ptr<void> held = pipeline;
    const RunPlan* bothPlan = plans.Teach(runPlan(both, pipeline));
    const RunPlan* lastPlan = plans.Teach(runPlan(last, token(2)));
    const RunPlan* sharedPlan = plans.Teach(runPlan(shared, token(3)));
    const RunPlan* alonePlan = plans.Teach(runPlan(alone, token(4)));
    if (!Present(bothPlan, "the plan of two programs is returned") || !Present(lastPlan, "the plan of a last stage program is returned") || !Present(sharedPlan, "the plan sharing a program is returned") || !Present(alonePlan, "the plan of one program is returned")) return;
    Expect(plans.Counted().evicted == 0, "the test keys land in four slots");
    plans.Forget(0x9000);
    plans.Forget(0);
    Expect(plans.Find(both) == bothPlan && plans.Find(last) == lastPlan && plans.Find(shared) == sharedPlan && plans.Find(alone) == alonePlan, "forgetting an unknown or a null program keeps every plan");
    Expect(plans.Counted().forgotten == 0, "nothing forgotten yet");
    plans.Forget(0x2000);
    Expect(plans.Find(both) == nullptr, "a plan whose second stage is the program is forgotten");
    Expect(plans.Find(last) == nullptr, "a plan whose last stage is the program is forgotten");
    Expect(plans.Find(shared) == sharedPlan && plans.Find(alone) == alonePlan, "plans of other programs are kept");
    Expect(plans.Counted().forgotten == 2 && plans.Counted().evicted == 0, "forgotten plans are counted apart from evictions");
    Expect(bothPlan->key == both && lastPlan->key == last && bothPlan->pipeline == pipeline, "a forgotten plan stays readable");
    plans.Forget(0x1000);
    Expect(plans.Find(shared) == nullptr && plans.Find(alone) == alonePlan, "forgetting the first stage program removes the plan that still had it");
    Expect(plans.Counted().forgotten == 3, "a plan is forgotten once");
    const RunPlan* relearned = plans.Teach(runPlan(both, token(5)));
    if (!Present(relearned, "a forgotten key taught again is returned")) return;
    Expect(relearned != bothPlan && relearned->id != bothPlan->id && plans.Find(both) == relearned, "a forgotten key can be taught again");
    Expect(!held.expired(), "a forgotten plan still holds its pipeline");
}

void testCollect() {
    std::weak_ptr<void> evictedHeld;
    std::weak_ptr<void> forgottenHeld;
    RunPlans plans(1);
    Expect(plans.Collect().empty(), "a table that retired nothing collects nothing");
    {
        std::shared_ptr<void> pipeline = token(1);
        evictedHeld = pipeline;
        static_cast<void>(plans.Teach(runPlan(stateKey(1), std::move(pipeline))));
    }
    const RunPlan* current = plans.Teach(runPlan(stateKey(2, {0x7000, 0, 0, 0}), token(2)));
    if (!Present(current, "the evicting plan is returned")) return;
    forgottenHeld = current->pipeline;
    const std::uint64_t firstId = current->id;
    Expect(!evictedHeld.expired(), "an evicted plan is held until it is collected");
    {
        const auto gone = plans.Collect();
        Expect(gone.size() == 1 && gone.front() != nullptr && gone.front()->key == stateKey(1), "collecting hands over the evicted plan");
        Expect(!evictedHeld.expired(), "a collected plan lives as long as its collector holds it");
    }
    Expect(evictedHeld.expired(), "a collected plan is released by its collector");
    Expect(plans.Find(stateKey(2, {0x7000, 0, 0, 0})) == current && current->id == firstId, "collecting leaves the plans of the table in place");
    plans.Forget(0x7000);
    Expect(!forgottenHeld.expired(), "a forgotten plan is held until it is collected");
    const std::size_t forgotten = plans.Collect().size();
    Expect(forgotten == 1 && forgottenHeld.expired(), "collecting releases the forgotten plan");
    Expect(plans.Collect().empty(), "plans are collected once");
    const RunPlans::Counts counts = plans.Counted();
    Expect(counts.collected == 2 && counts.evicted == 1 && counts.forgotten == 1, "collected plans are counted");
}

void testSlotCounts() {
    RunPlans none(0);
    const RunPlan* kept = none.Teach(runPlan(stateKey(1), token(1)));
    Expect(kept != nullptr && none.Find(stateKey(1)) == kept, "a table asked for no slot keeps one plan");
    RunPlans odd(3);
    std::vector<std::uint64_t> taken;
    std::array<StateKey, 4> keys{};
    std::array<const RunPlan*, 4> taught{};
    for (std::size_t index = 0; index < keys.size(); ++index) {
        keys[index] = placed(stateKey(0, {0x1000 + index * 0x100, 0, 0, 0}), 4, taken);
        taught[index] = odd.Teach(runPlan(keys[index], token(static_cast<int>(index))));
    }
    bool found = true;
    for (std::size_t index = 0; index < keys.size(); ++index) found = found && taught[index] != nullptr && odd.Find(keys[index]) == taught[index];
    Expect(found && odd.Counted().evicted == 0, "a slot count is rounded up to a power of two");
}

}

int main() {
    testTeachThenFind();
    testCollision();
    testWholeKey();
    testEvictedStaysValid();
    testForget();
    testCollect();
    testSlotCounts();
    if (failures != 0) {
        std::fprintf(stderr, "%d run plan checks failed\n", failures);
        return 1;
    }
    std::printf("run plan tests passed\n");
    return 0;
}
