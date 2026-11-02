#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/RegistrationPrebuildQueue.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::RegistrationPrebuildQueue;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("registration prebuild queue: " + message);
}

void SetEnvironment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

class Workers {
public:
    ~Workers() {
        Join();
    }

    void Run(std::function<void()> task) {
        std::lock_guard lock(mutex);
        ++submitted;
        threads.emplace_back(std::move(task));
    }

    void Join() {
        for (;;) {
            std::thread thread;
            {
                std::lock_guard lock(mutex);
                if (threads.empty()) return;
                thread = std::move(threads.back());
                threads.pop_back();
            }
            thread.join();
        }
    }

    std::size_t Submitted() {
        std::lock_guard lock(mutex);
        return submitted;
    }

private:
    std::mutex mutex;
    std::vector<std::thread> threads;
    std::size_t submitted = 0;
};

struct JoinOnExit {
    Workers& workers;

    ~JoinOnExit() {
        workers.Join();
    }
};

struct Counters {
    std::atomic<int> running{0};
    std::atomic<int> peak{0};
    std::atomic<int> done{0};

    void Task() {
        const int now = ++running;
        int seen = peak.load();
        while (now > seen && !peak.compare_exchange_weak(seen, now)) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        --running;
        ++done;
    }
};

bool WaitFor(const std::atomic<int>& value, int expected) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (value.load() < expected && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return value.load() >= expected;
}

void NothingBeforePresentEverythingAfter() {
    Workers workers;
    Counters counters;
    std::size_t released = 0;
    {
        RegistrationPrebuildQueue queue({true, 2, std::chrono::hours(1)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, [&](std::size_t queued, const char*) { released = queued; });
        for (int index = 0; index < 9; ++index) queue.Enqueue(0, 0, [&] { counters.Task(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        Require(workers.Submitted() == 0 && counters.done == 0, "a prebuild was submitted before the present signal");
        Require(queue.Queued() == 9, "queued prebuilds were lost before the present signal");
        queue.Open("present");
        Require(WaitFor(counters.done, 9), "queued prebuilds did not all run after the present signal");
        Require(released == 9, "the release reported a wrong queued count");
        Require(counters.peak <= 2, "more prebuilds ran at once than the cap");
        Require(counters.peak == 2, "the cap was never used");
        queue.Open("present");
        queue.Enqueue(0, 0, [&] { counters.Task(); });
        Require(WaitFor(counters.done, 10), "a prebuild registered after the release did not run");
        workers.Join();
        Require(queue.PeakRunning() == 2, "the queue's own peak differs from the observed one");
    }
}

void ImmediateStartsAtOnceWithinCap() {
    Workers workers;
    Counters counters;
    RegistrationPrebuildQueue queue({false, 3, std::chrono::hours(1)}, [&](std::function<void()> task) { workers.Run(std::move(task)); });
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 7; ++index) queue.Enqueue(0, 0, [&] { counters.Task(); });
    Require(WaitFor(counters.done, 7), "immediate prebuilds did not all run");
    Require(counters.peak <= 3, "immediate prebuilds exceeded the cap");
}

void FallbackReleasesWithoutPresent() {
    Workers workers;
    Counters counters;
    const char* reason = nullptr;
    RegistrationPrebuildQueue queue({true, 2, std::chrono::milliseconds(80)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, [&](std::size_t, const char* why) { reason = why; });
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 4; ++index) queue.Enqueue(0, 0, [&] { counters.Task(); });
    Require(WaitFor(counters.done, 4), "the time fallback did not release the queue");
    Require(reason != nullptr && std::string(reason) == "fallback", "the fallback release was not reported as such");
}

void NoFallbackByDefault() {
    Require(RegistrationPrebuildQueue::Settings{}.fallback.count() == 0, "the time fallback is on by default");
    Workers workers;
    Counters counters;
    RegistrationPrebuildQueue::Settings settings;
    settings.concurrency = 2;
    RegistrationPrebuildQueue queue(settings, [&](std::function<void()> task) { workers.Run(std::move(task)); });
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 3; ++index) queue.Enqueue(0, 0, [&] { counters.Task(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    Require(workers.Submitted() == 0 && queue.Queued() == 3, "the queue opened without a present while the fallback was off");
    queue.Open("present");
    Require(WaitFor(counters.done, 3), "the queue did not run after the present with the fallback off");
}

void StopDropsQueued() {
    Workers workers;
    Counters counters;
    RegistrationPrebuildQueue queue({true, 2, std::chrono::hours(1)}, [&](std::function<void()> task) { workers.Run(std::move(task)); });
    const JoinOnExit joiner{workers};
    queue.Enqueue(0, 0, [&] { counters.Task(); });
    queue.Stop();
    queue.Open("present");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Require(counters.done == 0 && workers.Submitted() == 0, "a stopped queue ran a prebuild");
}

void EnvironmentSettings() {
    SetEnvironment("APS5_COMPUTE_REGISTRATION_START", "immediate");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_CONCURRENCY", "5");
    auto settings = RegistrationPrebuildQueue::SettingsFromEnvironment();
    Require(!settings.waitForPresent && settings.concurrency == 5, "the environment was not read");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_START", "present");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_CONCURRENCY", "0");
    settings = RegistrationPrebuildQueue::SettingsFromEnvironment();
    Require(settings.waitForPresent && settings.concurrency == 1, "a zero cap was not raised to one");
}


struct Gate {
    std::atomic<bool> demand{false};
    std::atomic<std::uint64_t> used{0};
    std::atomic<int> summaries{0};
    std::atomic<std::uint64_t> summaryDemand{0};
    std::atomic<std::uint64_t> summaryMemory{0};

    RegistrationPrebuildQueue::Hooks Hooks() {
        return {[this] { return demand.load(); }, [this] { return used.load(); }, [this](std::size_t, std::uint64_t demandCount, std::uint64_t memoryCount) {
            ++summaries;
            summaryDemand += demandCount;
            summaryMemory += memoryCount;
        }};
    }
};

void DemandHoldsBackNewStartsUntilIdle() {
    Workers workers;
    Counters counters;
    Gate gate;
    gate.demand = true;
    RegistrationPrebuildQueue queue({false, 0, std::chrono::hours(1), 0, std::chrono::milliseconds(0)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, gate.Hooks());
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 4; ++index) queue.Enqueue(index, 0, [&] { counters.Task(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Require(workers.Submitted() == 0 && counters.done == 0, "a speculative prebuild started while a demand creation was active");
    Require(queue.Queued() == 4, "deferred prebuilds were lost");
    Require(gate.summaries.load() >= 1 && gate.summaryDemand.load() >= 1, "the deferral was not summarized");
    gate.demand = false;
    queue.Poke();
    Require(WaitFor(counters.done, 4), "deferred prebuilds did not run once demand went idle");
}

void RunningPrebuildsAreNotPreempted() {
    Workers workers;
    std::atomic<bool> release{false};
    std::atomic<int> done{0};
    Gate gate;
    RegistrationPrebuildQueue queue({false, 0, std::chrono::hours(1)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, gate.Hooks());
    const JoinOnExit joiner{workers};
    queue.Enqueue(1, 0, [&] {
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        ++done;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    gate.demand = true;
    queue.Enqueue(2, 0, [&] { ++done; });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Require(workers.Submitted() == 1 && queue.Queued() == 1, "a prebuild started during demand");
    release = true;
    gate.demand = false;
    queue.Poke();
    Require(WaitFor(done, 2), "the second prebuild did not start after demand ended");
}

void MemoryBudgetLimitsConcurrency() {
    Workers workers;
    Counters counters;
    Gate gate;
    RegistrationPrebuildQueue queue({false, 0, std::chrono::hours(1), 10, std::chrono::milliseconds(0)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, gate.Hooks());
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 6; ++index) queue.Enqueue(index, 4, [&] { counters.Task(); });
    Require(WaitFor(counters.done, 6), "budgeted prebuilds did not all run");
    Require(counters.peak == 2, "the budget did not admit exactly two prebuilds of cost 4 at once");
    Require(queue.PeakRunning() == 2, "the queue's peak differs from the observed one");
}

void OversizedItemRunsAlone() {
    Workers workers;
    Counters counters;
    Gate gate;
    RegistrationPrebuildQueue queue({false, 0, std::chrono::hours(1), 10, std::chrono::milliseconds(0)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, gate.Hooks());
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 3; ++index) queue.Enqueue(index, 50, [&] { counters.Task(); });
    Require(WaitFor(counters.done, 3), "oversized prebuilds did not run");
    Require(counters.peak == 1, "oversized prebuilds ran together");
}

void GateUsageCountsAgainstBudget() {
    Workers workers;
    Counters counters;
    Gate gate;
    gate.used = 8;
    RegistrationPrebuildQueue queue({false, 0, std::chrono::hours(1), 10, std::chrono::milliseconds(0)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, gate.Hooks());
    const JoinOnExit joiner{workers};
    queue.Enqueue(1, 4, [&] { counters.Task(); });
    queue.Enqueue(2, 4, [&] { counters.Task(); });
    Require(WaitFor(counters.done, 2), "the second prebuild never ran");
    Require(counters.peak == 1, "gate usage did not count against the budget");
}

void PromoteRemovesQueuedItem() {
    Workers workers;
    Counters counters;
    RegistrationPrebuildQueue queue({true, 0, std::chrono::hours(1)}, [&](std::function<void()> task) { workers.Run(std::move(task)); });
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 3; ++index) queue.Enqueue(100 + index, 0, [&] { counters.Task(); });
    Require(queue.Promote(101), "a queued item was not found");
    Require(!queue.Promote(101), "a promoted item was found twice");
    Require(!queue.Promote(7), "an unknown item was found");
    Require(queue.Queued() == 2, "promotion did not remove exactly one item");
    queue.Open("present");
    Require(WaitFor(counters.done, 2), "the remaining items did not run");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Require(counters.done == 2, "the promoted item still ran");
}

void SummaryIsRateLimited() {
    Workers workers;
    Gate gate;
    gate.demand = true;
    RegistrationPrebuildQueue queue({false, 0, std::chrono::hours(1), 0, std::chrono::hours(1)}, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, gate.Hooks());
    const JoinOnExit joiner{workers};
    for (int index = 0; index < 20; ++index) queue.Enqueue(index, 0, [] {});
    for (int index = 0; index < 20; ++index) queue.Poke();
    Require(gate.summaries.load() == 0, "the deferral summary was not rate limited");
}

void EnvironmentUnsetMeansNoCountCap() {
#ifdef _WIN32
    _putenv_s("APS5_COMPUTE_REGISTRATION_CONCURRENCY", "");
#else
    unsetenv("APS5_COMPUTE_REGISTRATION_CONCURRENCY");
#endif
    const auto settings = RegistrationPrebuildQueue::SettingsFromEnvironment();
    Require(settings.concurrency == 0, "an unset cap is not memory-only");
}

void DemandQuietDelaysStartsUntilThePeriodEnds() {
    Workers workers;
    Counters counters;
    RegistrationPrebuildQueue::Settings settings;
    settings.waitForPresent = false;
    settings.concurrency = 0;
    settings.demandQuiet = std::chrono::milliseconds(150);
    RegistrationPrebuildQueue::Hooks hooks;
    hooks.demandActive = [] { return false; };
    const auto started = std::chrono::steady_clock::now();
    hooks.demandIdleFor = [&] { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started); };
    RegistrationPrebuildQueue queue(settings, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, hooks);
    const JoinOnExit joiner{workers};
    queue.Enqueue(1, 0, [&] { counters.Task(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    Require(workers.Submitted() == 0 && counters.done == 0, "a prebuild started before demand had been quiet long enough");
    Require(WaitFor(counters.done, 1), "the prebuild did not start once the quiet period ended");
    Require(std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(150), "the prebuild started early");
    workers.Join();
}

void LongFrameStallLiftsTheQuietHold() {
    Workers workers;
    Counters counters;
    RegistrationPrebuildQueue::Settings settings;
    settings.waitForPresent = false;
    settings.concurrency = 0;
    settings.demandQuiet = std::chrono::hours(1);
    settings.stallLift = std::chrono::milliseconds(500);
    RegistrationPrebuildQueue::Hooks hooks;
    hooks.demandIdleFor = [] { return std::chrono::milliseconds(0); };
    RegistrationPrebuildQueue queue(settings, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, hooks);
    const JoinOnExit joiner{workers};
    queue.Enqueue(1, 0, [&] { counters.Task(); });
    const auto start = std::chrono::steady_clock::now();
    Require(!queue.Presented(start), "the first present lifted the hold");
    Require(!queue.Presented(start + std::chrono::milliseconds(100)), "a short frame lifted the hold");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    Require(workers.Submitted() == 0, "a prebuild started during the quiet hold");
    Require(queue.Presented(start + std::chrono::milliseconds(700)), "a long frame stall did not lift the hold");
    Require(!queue.Presented(start + std::chrono::milliseconds(1500)), "the hold was lifted twice");
    Require(WaitFor(counters.done, 1), "the held prebuild did not start once the hold was lifted");
    workers.Join();
}

void LargeItemsWaitForTheLiftWhileSmallOnesStart() {
    Workers workers;
    std::atomic<int> small{0};
    std::atomic<int> large{0};
    RegistrationPrebuildQueue::Settings settings;
    settings.waitForPresent = false;
    settings.concurrency = 0;
    settings.demandQuiet = std::chrono::milliseconds(50);
    settings.stallLift = std::chrono::milliseconds(500);
    settings.liftOnlyWords = 1000;
    RegistrationPrebuildQueue::Hooks hooks;
    hooks.demandIdleFor = [] { return std::chrono::hours(1); };
    RegistrationPrebuildQueue queue(settings, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, hooks);
    const JoinOnExit joiner{workers};
    queue.Enqueue(1, 0, [&] { ++large; }, 1000);
    queue.Enqueue(2, 0, [&] { ++small; }, 999);
    Require(WaitFor(small, 1), "a small prebuild queued behind a large one did not start on the quiet period");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Require(large == 0 && queue.Queued() == 1, "a large prebuild started before the stall lift");
    const auto start = std::chrono::steady_clock::now();
    queue.Presented(start);
    Require(queue.Presented(start + std::chrono::milliseconds(700)), "the long frame stall did not lift the hold");
    Require(WaitFor(large, 1), "the large prebuild did not start after the lift");
    workers.Join();
}

void LiftOnlyWordsWithoutStallLiftIsIgnored() {
    Workers workers;
    std::atomic<int> large{0};
    RegistrationPrebuildQueue::Settings settings;
    settings.waitForPresent = false;
    settings.concurrency = 0;
    settings.demandQuiet = std::chrono::milliseconds(20);
    settings.liftOnlyWords = 1000;
    RegistrationPrebuildQueue::Hooks hooks;
    hooks.demandIdleFor = [] { return std::chrono::hours(1); };
    RegistrationPrebuildQueue queue(settings, [&](std::function<void()> task) { workers.Run(std::move(task)); }, {}, hooks);
    const JoinOnExit joiner{workers};
    queue.Enqueue(1, 0, [&] { ++large; }, 5000);
    Require(WaitFor(large, 1), "a large prebuild stayed queued without a stall lift to release it");
}

void EnvironmentSetsDemandQuiet() {
    SetEnvironment("APS5_COMPUTE_REGISTRATION_DEMAND_QUIET_MS", "30000");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_QUIET_LIFT_STALL_MS", "60000");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_LIFT_ONLY_WORDS", "1400000");
    const auto settings = RegistrationPrebuildQueue::SettingsFromEnvironment();
    SetEnvironment("APS5_COMPUTE_REGISTRATION_DEMAND_QUIET_MS", "0");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_QUIET_LIFT_STALL_MS", "0");
    SetEnvironment("APS5_COMPUTE_REGISTRATION_LIFT_ONLY_WORDS", "0");
    Require(settings.liftOnlyWords == 1400000, "the size held for the lift was not read from the environment");
    Require(settings.demandQuiet == std::chrono::milliseconds(30000), "the quiet period was not read from the environment");
    Require(settings.stallLift == std::chrono::milliseconds(60000), "the stall that lifts the hold was not read from the environment");
}

}

int main() {
    try {
        NothingBeforePresentEverythingAfter();
        ImmediateStartsAtOnceWithinCap();
        FallbackReleasesWithoutPresent();
        NoFallbackByDefault();
        StopDropsQueued();
        EnvironmentSettings();
        DemandHoldsBackNewStartsUntilIdle();
        RunningPrebuildsAreNotPreempted();
        MemoryBudgetLimitsConcurrency();
        OversizedItemRunsAlone();
        GateUsageCountsAgainstBudget();
        PromoteRemovesQueuedItem();
        SummaryIsRateLimited();
        EnvironmentUnsetMeansNoCountCap();
        DemandQuietDelaysStartsUntilThePeriodEnds();
        LongFrameStallLiftsTheQuietHold();
        LargeItemsWaitForTheLiftWhileSmallOnesStart();
        LiftOnlyWordsWithoutStallLiftIsIgnored();
        EnvironmentSetsDemandQuiet();
        std::cout << "registration prebuild queue tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
