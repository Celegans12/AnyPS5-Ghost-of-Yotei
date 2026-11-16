#include "prx/libSceAgcDriver/Core/include/SubmitPolicy.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

struct Random {
    std::uint64_t state = 0;

    std::uint64_t Next() {
        auto value = state += 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31u);
    }
    std::uint32_t Below(std::uint32_t limit) { return static_cast<std::uint32_t>(Next() % limit); }
    bool OneIn(std::uint32_t chances) { return Below(chances) == 0; }
};

constexpr std::uint32_t FlipBit = 1u << 0u;
constexpr std::uint32_t InterruptBit = 1u << 1u;
constexpr std::uint32_t WaiterBit = 1u << 2u;
constexpr std::uint32_t EmptyBit = 1u << 3u;
constexpr std::uint32_t CapBit = 1u << 4u;
constexpr std::uint32_t WorkBit = 1u << 5u;
constexpr std::uint32_t DeadlineBit = 1u << 6u;
constexpr std::uint32_t Conditions = 1u << 7u;
constexpr std::array<SubmitCause, 7> Priority{SubmitCause::Flip, SubmitCause::Interrupt, SubmitCause::Waiter, SubmitCause::Empty, SubmitCause::Cap, SubmitCause::Work, SubmitCause::Deadline};
constexpr std::array<const char*, 7> ConditionNames{"a flip", "an interrupt", "a waiter", "an empty queue", "the work cap", "the group work", "the deadline"};
constexpr std::size_t Causes = static_cast<std::size_t>(SubmitCause::Count);
constexpr std::uint64_t Now = 50000000u;
constexpr std::uint64_t Late = 1000u;
constexpr std::uint64_t Most = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint32_t MostWork = std::numeric_limits<std::uint32_t>::max();

struct Case {
    SubmitState state;
    SubmitLimits limits;
};

Case caseFor(std::uint32_t mask, bool holding, bool variant) {
    Case made;
    made.limits.deadline = Late;
    made.limits.groupWork = 64;
    const bool cap = (mask & CapBit) != 0;
    const bool work = (mask & WorkBit) != 0;
    if (cap) {
        made.limits.workCap = work ? 100u : 10u;
        made.state.work = made.limits.workCap;
    } else {
        made.limits.workCap = variant ? 0u : 200u;
        made.state.work = work ? (variant ? MostWork : 64u) : 63u;
    }
    made.state.now = Now;
    made.state.flip = (mask & FlipBit) != 0;
    made.state.interrupt = (mask & InterruptBit) != 0;
    made.state.waiter = (mask & WaiterBit) != 0;
    made.state.queued = (mask & EmptyBit) != 0 ? 0u : 3u;
    const bool late = (mask & DeadlineBit) != 0;
    const std::uint64_t due = Now - Late;
    const std::uint64_t recent = due + 1u;
    if (holding) {
        made.state.heldSince = late && variant ? due : recent;
        made.state.oldestLabel = late && !variant ? due : (variant ? 0u : recent);
    } else {
        made.state.oldestLabel = late ? due : (variant ? 0u : recent);
    }
    return made;
}

SubmitCause firstOf(std::uint32_t mask) {
    for (std::size_t index = 0; index < Priority.size(); ++index) {
        if ((mask & (1u << index)) != 0) return Priority[index];
    }
    return SubmitCause::Held;
}

SubmitCause expectedAfterOp(std::uint32_t mask, bool holding) {
    if (holding) return firstOf(mask);
    if ((mask & CapBit) != 0) return SubmitCause::Cap;
    if ((mask & DeadlineBit) != 0) return SubmitCause::Deadline;
    return SubmitCause::Held;
}

SubmitState base() {
    SubmitState state;
    state.now = Now;
    state.queued = 1;
    return state;
}

SubmitLimits limitsOf(std::uint64_t deadline, std::uint32_t groupWork, std::uint32_t workCap) {
    SubmitLimits limits;
    limits.deadline = deadline;
    limits.groupWork = groupWork;
    limits.workCap = workCap;
    return limits;
}

bool both(const SubmitState& state, const SubmitLimits& limits, SubmitCause cause) {
    return AtSubmissionEnd(state, limits) == cause && AfterOp(state, limits) == cause;
}

void testNames() {
    const std::string_view unknown = SubmitCauseName(SubmitCause::Count);
    std::vector<std::string_view> names;
    bool named = true;
    for (std::size_t index = 0; index < Causes; ++index) {
        const char* name = SubmitCauseName(static_cast<SubmitCause>(index));
        if (name == nullptr || name[0] == '\0') {
            named = false;
            continue;
        }
        names.emplace_back(name);
    }
    Expect(named, "every cause has a name");
    Expect(!unknown.empty() && std::find(names.begin(), names.end(), unknown) == names.end(), "no cause is named like the count");
    std::sort(names.begin(), names.end());
    Expect(names.size() == Causes && std::adjacent_find(names.begin(), names.end()) == names.end(), "the names of the causes are distinct");
    Expect(std::string_view(SubmitCauseName(static_cast<SubmitCause>(200u))) == unknown, "a value beyond the table is named like the count");
}

void testTruthTable() {
    std::size_t atEndWrong = 0;
    std::size_t afterWrong = 0;
    std::size_t combinations = 0;
    std::array<bool, Causes> atEndSeen{};
    std::array<bool, Causes> afterSeen{};
    for (std::uint32_t mask = 0; mask < Conditions; ++mask) {
        for (const bool holding : {false, true}) {
            for (const bool variant : {false, true}) {
                ++combinations;
                const Case made = caseFor(mask, holding, variant);
                const SubmitCause atEnd = AtSubmissionEnd(made.state, made.limits);
                const SubmitCause after = AfterOp(made.state, made.limits);
                const SubmitCause wantAtEnd = firstOf(mask);
                const SubmitCause wantAfter = expectedAfterOp(mask, holding);
                if (atEnd != wantAtEnd) {
                    ++atEndWrong;
                    std::fprintf(stderr, "AtSubmissionEnd conditions 0x%02x hold %d variant %d: got %s, want %s\n", static_cast<unsigned>(mask), holding ? 1 : 0, variant ? 1 : 0, SubmitCauseName(atEnd), SubmitCauseName(wantAtEnd));
                }
                if (after != wantAfter) {
                    ++afterWrong;
                    std::fprintf(stderr, "AfterOp conditions 0x%02x hold %d variant %d: got %s, want %s\n", static_cast<unsigned>(mask), holding ? 1 : 0, variant ? 1 : 0, SubmitCauseName(after), SubmitCauseName(wantAfter));
                }
                if (static_cast<std::size_t>(atEnd) < Causes) atEndSeen[static_cast<std::size_t>(atEnd)] = true;
                if (static_cast<std::size_t>(after) < Causes) afterSeen[static_cast<std::size_t>(after)] = true;
            }
        }
    }
    Expect(combinations == 512, "the truth table covers 128 combinations of causes, with and without a hold, in two forms each");
    Expect(atEndWrong == 0, "AtSubmissionEnd follows flip, interrupt, waiter, empty queue, cap, work, deadline, in that order, over the whole table");
    Expect(afterWrong == 0, "AfterOp follows the same order during a hold, and only cap then deadline without one, over the whole table");
    Expect(std::all_of(atEndSeen.begin(), atEndSeen.end(), [](bool seen) { return seen; }), "AtSubmissionEnd returns every cause of the table");
    Expect(std::all_of(afterSeen.begin(), afterSeen.end(), [](bool seen) { return seen; }), "AfterOp returns every cause of the table");
}

void testEachCause() {
    char text[192];
    const Case quiet = caseFor(0, false, false);
    Expect(both(quiet.state, quiet.limits, SubmitCause::Held), "with no cause the end of a submission and an operation both hold");
    const Case quietHold = caseFor(0, true, false);
    Expect(both(quietHold.state, quietHold.limits, SubmitCause::Held), "with no cause a hold goes on");
    for (std::size_t index = 0; index < Priority.size(); ++index) {
        const std::uint32_t bit = 1u << index;
        const std::uint32_t after = (Conditions - 1u) & ~((bit << 1u) - 1u);
        for (const bool holding : {false, true}) {
            const char* during = holding ? " during a hold" : " without a hold";
            const Case alone = caseFor(bit, holding, false);
            std::snprintf(text, sizeof(text), "%s alone ends a submission with cause %s%s", ConditionNames[index], SubmitCauseName(Priority[index]), during);
            Expect(AtSubmissionEnd(alone.state, alone.limits) == Priority[index], text);
            const bool afterOpSubmits = holding || bit == CapBit || bit == DeadlineBit;
            std::snprintf(text, sizeof(text), "%s alone after an operation%s gives %s", ConditionNames[index], during, afterOpSubmits ? SubmitCauseName(Priority[index]) : "held");
            Expect(AfterOp(alone.state, alone.limits) == (afterOpSubmits ? Priority[index] : SubmitCause::Held), text);
            const Case crowded = caseFor(bit | after, holding, false);
            std::snprintf(text, sizeof(text), "%s wins over every cause after it at a submission end%s", ConditionNames[index], during);
            Expect(AtSubmissionEnd(crowded.state, crowded.limits) == Priority[index], text);
            if (holding) {
                std::snprintf(text, sizeof(text), "%s wins over every cause after it after an operation during a hold", ConditionNames[index]);
                Expect(AfterOp(crowded.state, crowded.limits) == Priority[index], text);
            }
        }
    }
    const Case holdOnly = caseFor(DeadlineBit, true, true);
    Expect(holdOnly.state.oldestLabel == 0 && both(holdOnly.state, holdOnly.limits, SubmitCause::Deadline), "a hold alone reaches the deadline from its own start");
    const Case capWithoutHold = caseFor(CapBit | WorkBit | DeadlineBit, false, false);
    Expect(AfterOp(capWithoutHold.state, capWithoutHold.limits) == SubmitCause::Cap, "without a hold the cap wins over the deadline after an operation");
    const Case workWithoutHold = caseFor(WorkBit | DeadlineBit, false, false);
    Expect(AfterOp(workWithoutHold.state, workWithoutHold.limits) == SubmitCause::Deadline, "without a hold the group work is skipped and the deadline still submits");
}

void testDeadlineEdges() {
    const SubmitLimits limits = limitsOf(Late, 64, 0);
    SubmitState state = base();
    state.oldestLabel = Now - Late + 1u;
    Expect(both(state, limits, SubmitCause::Held), "a label one nanosecond short of the deadline is held");
    state.oldestLabel = Now - Late;
    Expect(both(state, limits, SubmitCause::Deadline), "a label exactly at the deadline is submitted");
    state.oldestLabel = Now;
    Expect(both(state, limits, SubmitCause::Held), "a label posted now is held");
    state.oldestLabel = Now + 5u;
    Expect(both(state, limits, SubmitCause::Held), "a label stamped after the clock reading is not taken as overdue");
    state.oldestLabel = 0;
    state.heldSince = Now - Late + 1u;
    Expect(both(state, limits, SubmitCause::Held), "a hold one nanosecond short of the deadline goes on");
    state.heldSince = Now - Late;
    Expect(both(state, limits, SubmitCause::Deadline), "a hold that reaches the deadline is submitted");
    SubmitState none = base();
    Expect(both(none, limitsOf(0, 64, 0), SubmitCause::Held), "a zero deadline holds when there is neither a label nor a hold");
    none.oldestLabel = Now;
    Expect(both(none, limitsOf(0, 64, 0), SubmitCause::Deadline), "a zero deadline submits a pending label at once");
    SubmitState wide = base();
    wide.now = Most;
    wide.oldestLabel = 1u;
    Expect(both(wide, limitsOf(Most - 1u, 64, 0), SubmitCause::Deadline), "the elapsed time is computed without overflow");
    Expect(both(wide, limitsOf(Most, 64, 0), SubmitCause::Held), "a deadline longer than the elapsed time holds");
    SubmitState behind = base();
    behind.now = 1u;
    behind.oldestLabel = Most;
    behind.heldSince = Most;
    Expect(both(behind, limitsOf(1, 64, 0), SubmitCause::Held), "a clock behind the label and the hold does not wrap into an overdue one");
}

void testWorkEdges() {
    const SubmitLimits limits = limitsOf(Late, 64, 0);
    SubmitState state = base();
    SubmitState holding = base();
    holding.heldSince = Now - 1u;
    state.work = 63;
    holding.work = 63;
    Expect(AtSubmissionEnd(state, limits) == SubmitCause::Held, "one operation short of the group work, a submission end holds");
    Expect(AfterOp(holding, limits) == SubmitCause::Held, "one operation short of the group work, a hold goes on");
    state.work = 64;
    holding.work = 64;
    Expect(AtSubmissionEnd(state, limits) == SubmitCause::Work, "the group work submits at a submission end");
    Expect(AfterOp(state, limits) == SubmitCause::Held, "without a hold the group work does not submit after an operation");
    Expect(AfterOp(holding, limits) == SubmitCause::Work, "during a hold the group work submits after an operation");
    const SubmitLimits ungrouped = limitsOf(Late, 0, 0);
    SubmitState idle = base();
    SubmitState idleHold = base();
    idleHold.heldSince = Now - 1u;
    Expect(AtSubmissionEnd(idle, ungrouped) == SubmitCause::Work, "a group work of zero submits at every submission end");
    Expect(AfterOp(idle, ungrouped) == SubmitCause::Held, "a group work of zero does not submit after an operation without a hold");
    Expect(AfterOp(idleHold, ungrouped) == SubmitCause::Work, "a group work of zero ends a hold at the next operation");
    const SubmitLimits capped = limitsOf(Late, 64, 128);
    state.work = 127;
    holding.work = 127;
    Expect(AtSubmissionEnd(state, capped) == SubmitCause::Work && AfterOp(state, capped) == SubmitCause::Held, "one operation short of the cap, only the group work applies");
    state.work = 128;
    holding.work = 128;
    Expect(both(state, capped, SubmitCause::Cap), "the cap submits at a submission end and after an operation without a hold");
    Expect(AfterOp(holding, capped) == SubmitCause::Cap, "the cap wins over the group work during a hold");
    const SubmitLimits tight = limitsOf(Late, 64, 8);
    state.work = 8;
    Expect(both(state, tight, SubmitCause::Cap), "a cap below the group work submits on its own");
    state.work = MostWork;
    holding.work = MostWork;
    Expect(AtSubmissionEnd(state, limits) == SubmitCause::Work && AfterOp(state, limits) == SubmitCause::Held, "a cap of zero never submits");
    Expect(AfterOp(holding, limits) == SubmitCause::Work, "a cap of zero leaves the group work to end a hold");
}

void testDefaults() {
    const SubmitLimits defaults{};
    SubmitState state = base();
    state.work = 63;
    Expect(AtSubmissionEnd(state, defaults) == SubmitCause::Held, "by default 63 operations are held at a submission end");
    state.work = 64;
    Expect(AtSubmissionEnd(state, defaults) == SubmitCause::Work, "by default 64 operations are submitted at a submission end");
    state.work = MostWork;
    Expect(AfterOp(state, defaults) == SubmitCause::Held, "by default there is no cap");
    state.work = 0;
    state.oldestLabel = Now - 999999u;
    Expect(both(state, defaults, SubmitCause::Held), "by default a label 999999 ns old is held");
    state.oldestLabel = Now - 1000000u;
    Expect(both(state, defaults, SubmitCause::Deadline), "by default a label 1 ms old is submitted");
}

void testClockExample() {
    const SubmitLimits limits = limitsOf(1000, 64, 0);
    SubmitState state;
    state.queued = 5;
    state.oldestLabel = 100;
    std::uint64_t submitted = 0;
    for (std::uint64_t now = 100; now <= 3000 && submitted == 0; now += 300) {
        state.now = now;
        if (AfterOp(state, limits) != SubmitCause::Held) submitted = now;
    }
    Expect(submitted == 1300, "a label posted at 100 ns, 1000 ns deadline, one operation every 300 ns: submitted at 1300 ns, one step past the deadline");
    SubmitState hold;
    hold.now = 1000;
    hold.queued = 2;
    hold.work = 3;
    Expect(AtSubmissionEnd(hold, limits) == SubmitCause::Held, "a submission end with little work and operations queued behind it is held");
    hold.heldSince = 1000;
    hold.now = 1500;
    hold.work = 4;
    Expect(AfterOp(hold, limits) == SubmitCause::Held, "the hold goes on while operations are queued");
    hold.now = 1600;
    hold.queued = 0;
    Expect(AfterOp(hold, limits) == SubmitCause::Empty, "the hold ends when the queue runs empty");
    hold.queued = 2;
    hold.now = 1700;
    hold.work = 64;
    Expect(AfterOp(hold, limits) == SubmitCause::Work, "the hold ends when the group work is reached");
    hold.work = 10;
    hold.now = 2000;
    Expect(AfterOp(hold, limits) == SubmitCause::Deadline, "the hold ends at the deadline");
}

struct Tally {
    std::uint64_t calls = 0;
    std::uint64_t signalled = 0;
    std::uint64_t signalledHeld = 0;
    std::uint64_t labelBatches = 0;
    std::uint64_t overdueHeld = 0;
    std::uint64_t late = 0;
    std::uint64_t deadlines = 0;
    std::uint64_t holds = 0;
    std::uint64_t emptyChecks = 0;
    std::uint64_t heldOnEmpty = 0;
    std::uint64_t foreign = 0;
};

void runScenario(Random& random, Tally& tally) {
    constexpr std::array<std::uint64_t, 6> deadlines{0u, 1u, 50u, 1000u, 100000u, 1000000u};
    constexpr std::array<std::uint32_t, 5> groups{0u, 1u, 8u, 64u, 200u};
    SubmitLimits limits;
    limits.deadline = deadlines[random.Below(static_cast<std::uint32_t>(deadlines.size()))];
    limits.groupWork = groups[random.Below(static_cast<std::uint32_t>(groups.size()))];
    limits.workCap = random.OneIn(2) ? 0u : 1u + random.Below(300);
    const std::uint64_t maxStep = random.OneIn(8) ? limits.deadline * 2u + 1u : limits.deadline / (1u + random.Below(16)) + 1u;
    const std::uint32_t steps = 16u + random.Below(240);
    const std::uint32_t emptyOdds = 2u + random.Below(15);
    const std::uint32_t endOdds = 4u + random.Below(29);
    SubmitState state;
    state.now = 1u + random.Below(1000);
    for (std::uint32_t step = 0; step < steps; ++step) {
        const std::uint64_t advance = random.Next() % (maxStep + 1u);
        state.now += advance;
        state.queued = random.OneIn(emptyOdds) ? 0u : 1u + random.Below(16);
        state.waiter = random.OneIn(64);
        state.flip = false;
        state.interrupt = false;
        const bool ending = random.OneIn(endOdds);
        if (ending) {
            state.flip = random.OneIn(16);
        } else if (random.OneIn(4)) {
            if (state.oldestLabel == 0) state.oldestLabel = state.now;
            state.interrupt = random.OneIn(32);
        } else if (!random.OneIn(3)) {
            ++state.work;
        }
        const SubmitCause cause = ending ? AtSubmissionEnd(state, limits) : AfterOp(state, limits);
        ++tally.calls;
        const bool holding = state.heldSince != 0;
        const bool held = cause == SubmitCause::Held;
        if ((state.flip || state.interrupt || state.waiter) && (ending || holding)) {
            ++tally.signalled;
            if (held) ++tally.signalledHeld;
        }
        if (!ending && !holding && cause != SubmitCause::Held && cause != SubmitCause::Deadline && cause != SubmitCause::Cap) ++tally.foreign;
        if (state.queued == 0 && (ending || holding)) {
            ++tally.emptyChecks;
            if (held) ++tally.heldOnEmpty;
        }
        if (state.oldestLabel != 0) {
            const std::uint64_t elapsed = state.now - state.oldestLabel;
            if (held && elapsed >= limits.deadline) ++tally.overdueHeld;
            if (!held) {
                ++tally.labelBatches;
                if (elapsed > limits.deadline + advance) ++tally.late;
            }
        }
        if (cause == SubmitCause::Deadline) ++tally.deadlines;
        if (!held) {
            state.work = 0;
            state.oldestLabel = 0;
            state.heldSince = 0;
        } else if (ending && !holding) {
            state.heldSince = state.now;
            ++tally.holds;
        }
    }
}

void testRandomScenarios() {
    Random random{0x5ab3175u};
    Tally tally;
    constexpr std::uint32_t Scenarios = 10000;
    for (std::uint32_t scenario = 0; scenario < Scenarios; ++scenario) runScenario(random, tally);
    Expect(tally.signalledHeld == 0, "10000 random scenarios: never held with a waiter, an interrupt or a flip, at a submission end or during a hold");
    Expect(tally.overdueHeld == 0, "10000 random scenarios: a label is never held once its deadline is reached");
    Expect(tally.late == 0, "10000 random scenarios: a pending label is submitted at most one clock step after its deadline");
    Expect(tally.heldOnEmpty == 0, "10000 random scenarios: no hold starts or survives on an empty queue");
    Expect(tally.foreign == 0, "10000 random scenarios: without a hold AfterOp only returns cap, deadline or held");
    const bool reached = tally.calls >= 16u * Scenarios && tally.signalled >= 100 && tally.labelBatches >= 100 && tally.deadlines >= 100 && tally.holds >= 100 && tally.emptyChecks >= 100;
    Expect(reached, "10000 random scenarios reach every property many times");
    if (!reached) {
        std::fprintf(stderr, "calls %llu signalled %llu label batches %llu deadlines %llu holds %llu empty checks %llu\n", static_cast<unsigned long long>(tally.calls), static_cast<unsigned long long>(tally.signalled), static_cast<unsigned long long>(tally.labelBatches), static_cast<unsigned long long>(tally.deadlines), static_cast<unsigned long long>(tally.holds), static_cast<unsigned long long>(tally.emptyChecks));
    }
}

}

int main() {
    testNames();
    testTruthTable();
    testEachCause();
    testDeadlineEdges();
    testWorkEdges();
    testDefaults();
    testClockExample();
    testRandomScenarios();
    if (failures != 0) {
        std::fprintf(stderr, "%d submit policy checks failed\n", failures);
        return 1;
    }
    std::printf("submit policy tests passed\n");
    return 0;
}
