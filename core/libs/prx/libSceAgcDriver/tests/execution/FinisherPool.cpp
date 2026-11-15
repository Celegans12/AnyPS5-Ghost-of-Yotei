#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using Kind = DrawPipeline::CommitKind;

constexpr std::chrono::seconds Patience{20};
constexpr std::chrono::milliseconds Held{300};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Gate {
    std::promise<void> opener;
    std::shared_future<void> opened = opener.get_future().share();
    bool open = false;
    void Open() {
        if (!std::exchange(open, true)) opener.set_value();
    }
    ~Gate() { Open(); }
};

struct OpenOnExit {
    Gate& gate;
    ~OpenOnExit() { gate.Open(); }
};

struct Commits {
    std::mutex mutex;
    std::vector<int> order;
    void Note(int index) {
        std::lock_guard lock(mutex);
        order.push_back(index);
    }
    std::vector<int> Taken() {
        std::lock_guard lock(mutex);
        return order;
    }
};

DrawFinisher::Pending Known(std::vector<DrawFinisher::Range> writes = {}) {
    DrawFinisher::Pending held;
    held.unknown = false;
    held.writes = std::move(writes);
    return held;
}

bool Within(std::chrono::milliseconds patience, const std::function<bool()>& reached) {
    const auto deadline = std::chrono::steady_clock::now() + patience;
    while (!reached()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

void Settle() {
    DrawFinisher::Get().Drain();
    DrawPipeline::Queue0().Drain(DrawPipeline::DrainReason::Submission);
}

void RequireInOrder(const std::vector<int>& order, int queued) {
    Require(order.size() == static_cast<std::size_t>(queued), "the pipeline committed " + std::to_string(order.size()) + " of " + std::to_string(queued) + " items");
    for (std::size_t index = 0; index < order.size(); ++index) {
        Require(order[index] == static_cast<int>(index), "commit " + std::to_string(index) + " ran item " + std::to_string(order[index]));
    }
}

void CommitsKeepTheQueuedOrder() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    constexpr int Draws = 7;
    const bool relays = DrawFinisher::Relays();
    const auto counted = DrawFinisher::PoolCounted();
    Gate gate;
    const OpenOnExit opening{gate};
    const auto commits = std::make_shared<Commits>();
    const auto finished = std::make_shared<std::atomic<int>>(0);
    int queued = 0;
    for (int draw = 0; draw < Draws; ++draw) {
        finisher.Enqueue([commits, finished, &pipeline, index = queued++, slow = draw == 0, opened = gate.opened] {
            if (slow) opened.wait();
            pipeline.Enqueue(Kind::Draw, [commits, index] { commits->Note(index); }, {}, 0, {}, {}, true);
            finished->fetch_add(1);
        }, Known());
        if (relays && draw + 1 < Draws) pipeline.Enqueue(Kind::DmaWrite, [commits, index = queued++] { commits->Note(index); }, {});
    }
    Require(Within(Patience, [&] { return finished->load() == Draws - 1; }), "the jobs queued behind a slow job did not run beside it");
    Require(commits->Taken().empty(), "an item reached the committer ahead of the item of a slow job queued before it");
    Require(DrawFinisher::Busy(), "the draw finisher reports no work while finished jobs wait for a slow one");
    gate.Open();
    Settle();
    RequireInOrder(commits->Taken(), queued);
    const auto after = DrawFinisher::PoolCounted();
    Require(after.jobs >= counted.jobs + Draws, "the jobs of the pool were not counted");
    Require(after.beside > counted.beside, "no job was counted as started while another ran");
    Require(after.reordered > counted.reordered, "no finished job was counted as waiting in the reorder buffer");
}

void JobQueuedWithoutItsWritesRunsAlone() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    struct State {
        std::atomic<bool> second{false}, aloneStarted{false}, aloneSawItems{false}, aloneDone{false}, later{false}, laterAfterAlone{false};
    };
    const auto counted = DrawFinisher::PoolCounted();
    Gate ahead;
    Gate during;
    const OpenOnExit openAhead{ahead};
    const OpenOnExit openDuring{during};
    const auto state = std::make_shared<State>();
    const auto commits = std::make_shared<Commits>();
    finisher.Enqueue([commits, &pipeline, opened = ahead.opened] {
        opened.wait();
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(0); }, {}, 0, {}, {}, true);
    }, Known());
    finisher.Enqueue([commits, state, &pipeline] {
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(1); }, {}, 0, {}, {}, true);
        state->second = true;
    }, Known());
    finisher.Enqueue([commits, state, &pipeline, opened = during.opened] {
        state->aloneStarted = true;
        pipeline.Drain(DrawPipeline::DrainReason::Capture);
        state->aloneSawItems = commits->Taken() == std::vector<int>{0, 1};
        opened.wait();
        state->aloneDone = true;
    });
    finisher.Enqueue([state] {
        state->laterAfterAlone = state->aloneDone.load();
        state->later = true;
    }, Known());
    Require(Within(Patience, [&] { return state->second.load(); }), "a job queued behind a held job did not run beside it");
    std::this_thread::sleep_for(Held);
    Require(!state->aloneStarted, "a job queued without its writes started before a job queued ahead of it finished");
    Require(!state->later, "a job started ahead of a job that runs alone and was queued before it");
    ahead.Open();
    Require(Within(Patience, [&] { return state->aloneStarted.load(); }), "the job that runs alone did not start once the jobs ahead of it finished");
    std::this_thread::sleep_for(Held);
    Require(!state->later, "a job started beside a job that runs alone");
    during.Open();
    Settle();
    Require(state->aloneSawItems, "the job that runs alone started before the items of the jobs ahead of it were in the committer's queue");
    Require(state->later && state->laterAfterAlone, "the job queued behind the one that runs alone did not run after it");
    Require(DrawFinisher::PoolCounted().aloneUnlisted == counted.aloneUnlisted + 1, "the job that ran alone was not counted");
}

void RangesOfAFinishedJobStayVisible() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    alignas(8) static std::array<std::uint32_t, 4> written{};
    alignas(8) static std::array<std::uint32_t, 4> fetched{};
    alignas(8) static std::array<std::uint32_t, 4> elsewhere{};
    const auto address = reinterpret_cast<std::uintptr_t>(written.data());
    const auto end = address + sizeof(written);
    const auto source = reinterpret_cast<std::uintptr_t>(fetched.data());
    const auto sourceEnd = source + sizeof(fetched);
    const auto other = reinterpret_cast<std::uintptr_t>(elsewhere.data());
    Gate gate;
    const OpenOnExit opening{gate};
    const auto finished = std::make_shared<std::atomic<bool>>(false);
    finisher.Enqueue([opened = gate.opened] { opened.wait(); }, Known());
    finisher.Enqueue([finished, &pipeline, address, end, source, sourceEnd] {
        pipeline.Enqueue(Kind::Draw, [] {}, {{address, end}}, 0, {}, {{source, sourceEnd}}, true);
        finished->store(true);
    }, Known({{address, end}}));
    Require(Within(Patience, [&] { return finished->load(); }), "a job queued behind a held job did not run beside it");
    Require(pipeline.Busy(), "the pipeline reports no work while a finished job waits for its turn");
    Require(pipeline.Overlaps(address, sizeof(written)), "the write range of a finished job is not reported while its item waits for an older job");
    Require(!pipeline.Overlaps(other, sizeof(elsewhere)), "a finished job is reported over a range it does not write");
    Require(DrawPipeline::HeldReads(source, sizeof(fetched)), "the read range of a finished job's item holds no write-back back while the item waits for an older job");
    Require(!DrawPipeline::HeldReads(other, sizeof(elsewhere)), "a finished job's item holds a write-back back over a range it does not read");
    gate.Open();
    Settle();
    Require(!pipeline.Overlaps(address, sizeof(written)), "a committed item still reports its write range");
    Require(!DrawPipeline::HeldReads(source, sizeof(fetched)), "a committed item still holds write-backs back over its read range");
}

void DrainWaitsForTheJobsThatRun() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    constexpr int Jobs = 6;
    Gate gate;
    std::future<void> drained;
    const OpenOnExit opening{gate};
    const auto commits = std::make_shared<Commits>();
    const auto started = std::make_shared<std::atomic<int>>(0);
    for (int index = 0; index < Jobs; ++index) {
        finisher.Enqueue([commits, started, &pipeline, index, opened = gate.opened] {
            started->fetch_add(1);
            opened.wait();
            pipeline.Enqueue(Kind::Draw, [commits, index] { commits->Note(index); }, {}, 0, {}, {}, true);
        }, Known());
    }
    Require(Within(Patience, [&] { return started->load() >= 2; }), "the pool ran one job at a time");
    drained = std::async(std::launch::async, [&finisher] { finisher.Drain(); });
    Require(drained.wait_for(Held) == std::future_status::timeout, "Drain returned while jobs of the pool were running");
    Require(DrawFinisher::Busy(), "the draw finisher reports no work while its jobs run");
    gate.Open();
    Require(drained.wait_for(Patience) == std::future_status::ready, "Drain did not return once the jobs of the pool finished");
    drained.get();
    Require(!DrawFinisher::Busy(), "Drain returned with jobs outstanding");
    finisher.Settle();
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    RequireInOrder(commits->Taken(), Jobs);
}

void JobThatDrainsThePipelineTakesItsTurn() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    const auto turns = DrawFinisher::PoolCounted().turns;
    Gate gate;
    const OpenOnExit opening{gate};
    const auto commits = std::make_shared<Commits>();
    const auto seen = std::make_shared<std::atomic<int>>(-1);
    finisher.Enqueue([commits, &pipeline, opened = gate.opened] {
        opened.wait();
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(0); }, {}, 0, {}, {}, true);
    }, Known());
    finisher.Enqueue([commits, seen, &pipeline] {
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(1); }, {}, 0, {}, {}, true);
        pipeline.Drain(DrawPipeline::DrainReason::Capture);
        seen->store(commits->Taken() == std::vector<int>{0, 1} ? 1 : 0);
    }, Known());
    std::this_thread::sleep_for(Held);
    Require(seen->load() == -1, "a job drained the pipeline ahead of the item of a job queued before it");
    gate.Open();
    Settle();
    Require(seen->load() == 1, "a job that drained the pipeline did not find the earlier job's item and its own committed in order");
    Require(DrawFinisher::PoolCounted().turns > turns, "the turn the draining job took was not counted");
}

void FailureOfAJobReachesTheQueuingThread() {
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    Gate gate;
    const OpenOnExit opening{gate};
    const auto commits = std::make_shared<Commits>();
    const auto ran = std::make_shared<std::atomic<bool>>(false);
    finisher.Enqueue([opened = gate.opened] { opened.wait(); }, Known());
    finisher.Enqueue([commits, &pipeline] {
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(0); }, {}, 0, {}, {}, true);
        throw std::runtime_error("finisher job failed");
    }, Known());
    finisher.Enqueue([commits, ran, &pipeline] {
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(1); }, {}, 0, {}, {}, true);
        ran->store(true);
    }, Known());
    Require(Within(Patience, [&] { return ran->load(); }), "the job behind the failing one did not run while an older job was held");
    gate.Open();
    bool raised = false;
    try {
        finisher.Drain();
    } catch (const std::runtime_error& error) {
        raised = std::string(error.what()) == "finisher job failed";
    }
    Require(raised, "a failed job of the pool did not reach the thread that drains the finisher");
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    Require(commits->Taken() == std::vector<int>{0}, "the pipeline did not commit exactly the item the failed job queued before it failed");
    finisher.Enqueue([commits] { commits->Note(2); }, Known());
    Settle();
    Require(commits->Taken() == std::vector<int>{0, 2}, "the pool did not run a job queued after its failure was taken");
}

}

int main() {
    try {
        if (DrawPipeline::Depth() == 0 || DrawFinisher::Threads() < 2) return 77;
        CommitsKeepTheQueuedOrder();
        JobQueuedWithoutItsWritesRunsAlone();
        if (DrawFinisher::Overlap()) RangesOfAFinishedJobStayVisible();
        DrainWaitsForTheJobsThatRun();
        JobThatDrainsThePipelineTakesItsTurn();
        FailureOfAJobReachesTheQueuingThread();
        std::cout << "the draw finisher pool of " << DrawFinisher::Threads() << " threads kept the queued order\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
