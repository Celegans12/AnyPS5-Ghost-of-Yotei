#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawFinisher;
using AgcDriver::DriverDetail::DrawPipeline;
using Kind = DrawPipeline::CommitKind;
using Range = DrawPipeline::Range;

constexpr std::chrono::seconds Patience{20};

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

bool Within(std::chrono::milliseconds patience, const std::function<bool()>& reached) {
    const auto deadline = std::chrono::steady_clock::now() + patience;
    while (!reached()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

std::vector<std::byte> Counting(std::size_t count) {
    std::vector<std::byte> bytes(count);
    for (std::size_t index = 0; index < count; ++index) bytes[index] = static_cast<std::byte>(index + 1);
    return bytes;
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

void QueriesFollowTheCommitterQueue() {
    auto& pipeline = DrawPipeline::Queue0();
    Settle();
    const auto counted = DrawPipeline::Transported();
    const auto retired = pipeline.Retired();
    Gate gate;
    pipeline.Enqueue(Kind::DmaWrite, [opened = gate.opened] { opened.wait(); }, {{0x1000, 0x1100}});
    const auto stored = Counting(8);
    pipeline.EnqueueStore([] {}, 0x3000, stored);
    pipeline.Enqueue(Kind::Draw, [] {}, {}, 0, {}, {{0x4000, 0x4010}}, true);
    std::vector<Range> wideWrites;
    for (std::uint64_t index = 0; index < 9; ++index) wideWrites.emplace_back(0x7000 + index * 0x100, 0x7010 + index * 0x100);
    pipeline.Enqueue(Kind::Dispatch, [] {}, wideWrites);
    const auto wideStored = Counting(32);
    pipeline.EnqueueStore([] {}, 0x6000, wideStored);

    Require(pipeline.Outstanding() == 5 && DrawPipeline::QueuedItems() == 5, "the queued items were not all counted");
    Require(pipeline.Overlaps(0x10ff, 4), "a range that crosses the end of a queued write was not reported");
    Require(!pipeline.Overlaps(0x1100, 4), "a range that starts where a queued write ends was reported");
    Require(pipeline.Overlaps(0x7800, 1), "the ninth range of an item was not reported");
    const std::array<Range, 2> crossing{{{0x10, 0x20}, {0x3004, 0x3005}}};
    const std::array<Range, 2> clear{{{0x10, 0x20}, {0x9000, 0x9010}}};
    Require(pipeline.OverlapsAny(crossing) && !pipeline.OverlapsAny(clear), "a list of ranges was not checked against the queued writes");

    std::uint64_t expected = 0;
    std::memcpy(&expected, stored.data() + 4, 4);
    Require(pipeline.PendingLabel(0x3004, 4) == std::optional<std::uint64_t>(expected), "the bytes of a queued store did not give its label a value");
    Require(!pipeline.PendingLabel(0x1000, 4).has_value(), "a queued write without bytes gave a label a value");
    Require(!pipeline.PendingLabel(0x9000, 4).has_value(), "a label no queued item writes was given a value");

    std::array<std::byte, 8> out{};
    Require(pipeline.StoredBytes(0x3000, 8, out) == DrawPipeline::QueuedWrite::Stored && std::equal(out.begin(), out.end(), stored.begin()), "the bytes of a queued store were not returned");
    Require(pipeline.StoredBytes(0x6010, 8, out) == DrawPipeline::QueuedWrite::Stored && std::equal(out.begin(), out.end(), wideStored.begin() + 0x10), "the bytes of a queued store of 32 bytes were not returned");
    Require(pipeline.StoredBytes(0x1000, 4, {}) == DrawPipeline::QueuedWrite::Unknown, "a queued write that is not a store was reported as stored");
    Require(pipeline.StoredBytes(0x9000, 4, {}) == DrawPipeline::QueuedWrite::None, "a range no queued item writes was reported as stored");

    Require(DrawPipeline::HeldReads(0x400f, 4), "a range a queued draw reads was not held");
    Require(!DrawPipeline::HeldReads(0x4010, 4), "a range that starts where a queued read ends was held");

    const auto watch = pipeline.WatchWrites();
    Require(watch.retired == retired && watch.items == 5 && watch.writes.size() == 12, "the writes of the queued items were not all copied");
    Require(pipeline.WrittenSince(watch, {{0x1000, 0x1004}}), "a queued write was not reported as written since the copy");
    Require(!pipeline.WrittenSince(watch, {{0x9000, 0x9004}}), "a range no queued item writes was reported as written since the copy");
    Require(pipeline.WrittenWhileQueued(retired, {{0x1000, 0x1004}}) == std::optional<bool>(true), "a queued write was not reported while nothing was retired");
    Require(pipeline.WrittenWhileQueued(retired, {{0x9000, 0x9004}}) == std::optional<bool>(false), "a clear range was reported as written while nothing was retired");

    gate.Open();
    Settle();
    Require(pipeline.Retired() == retired + 5 && pipeline.Outstanding() == 0, "the committed items were not all retired");
    Require(!pipeline.Overlaps(0x1000, 0x100) && !DrawPipeline::HeldReads(0x4000, 4), "a committed item was still reported as queued");
    Require(!pipeline.WrittenWhileQueued(retired, {{0x9000, 0x9004}}).has_value(), "items retired since the count was taken went unnoticed");
    Require(pipeline.WrittenSince(watch, {{0x1000, 0x1004}}), "a write committed since the copy was not reported");
    if (DrawPipeline::Flat() && DrawFinisher::Timed()) {
        const auto after = DrawPipeline::Transported();
        Require(after.elements == counted.elements + 5, "the items of the committer's queue were not counted");
        Require(after.wide == counted.wide + 2, "the items wider than a flat record were not counted");
        Require(after.queries > counted.queries && after.lockedQueries > counted.lockedQueries, "the queries on the committer's queue were not counted");
    }
}

void FullQueueKeepsTheQueuedOrder() {
    auto& pipeline = DrawPipeline::Queue0();
    Settle();
    const auto counted = DrawPipeline::Transported();
    const auto commits = std::make_shared<Commits>();
    const int filling = static_cast<int>(DrawPipeline::Depth());
    const int queued = filling + 3;
    Gate gate;
    pipeline.Enqueue(Kind::DmaWrite, [commits, opened = gate.opened] {
        opened.wait();
        commits->Note(0);
    }, {});
    for (int index = 1; index < filling; ++index) pipeline.Enqueue(Kind::DmaWrite, [commits, index] { commits->Note(index); }, {});
    Require(pipeline.Outstanding() == DrawPipeline::Depth(), "the committer's queue did not fill");
    std::atomic<int> pushed{filling};
    std::thread producer([&] {
        for (int index = filling; index < queued; ++index) {
            pipeline.Enqueue(Kind::DmaWrite, [commits, index] { commits->Note(index); }, {});
            pushed.store(index + 1);
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool early = pushed.load() != filling;
    gate.Open();
    const bool finished = Within(Patience, [&] { return pushed.load() == queued; });
    producer.join();
    Require(!early, "an item entered a full queue");
    Require(finished, "a producer waiting for room in the committer's queue never woke");
    pipeline.Drain(DrawPipeline::DrainReason::Submission);
    RequireInOrder(commits->Taken(), queued);
    if (DrawPipeline::Flat()) Require(DrawPipeline::Transported().roomWaits > counted.roomWaits, "the wait for room in the committer's queue was not counted");
}

void RelayedItemsStayBehindTheFinisher() {
    if (!DrawFinisher::Relays()) return;
    auto& finisher = DrawFinisher::Get();
    auto& pipeline = DrawPipeline::Queue0();
    Settle();
    const auto commits = std::make_shared<Commits>();
    Gate gate;
    DrawFinisher::Pending held;
    held.unknown = false;
    held.key = 7;
    held.shape = 9;
    held.writes = {{0xa000, 0xa100}};
    finisher.Enqueue([commits, &pipeline, opened = gate.opened] {
        opened.wait();
        pipeline.Enqueue(Kind::Draw, [commits] { commits->Note(0); }, {{0xa000, 0xa100}}, 0, {}, {}, true);
    }, held);
    pipeline.Enqueue(Kind::DmaWrite, [commits] { commits->Note(1); }, {{0xb000, 0xb010}});
    const auto stored = Counting(8);
    pipeline.EnqueueStore([commits] { commits->Note(2); }, 0xc000, stored);

    Require(DrawFinisher::Busy(), "the draw finisher reports no work while a job waits");
    Require(pipeline.Overlaps(0xa010, 4), "the writes of a job handed to the finisher were not reported");
    Require(pipeline.Overlaps(0xb000, 4), "the writes of an item relayed behind the finisher were not reported");
    Require(!pipeline.Overlaps(0xd000, 4), "a range nothing queued writes was reported");
    Require(finisher.PendingKey(7, 0) && finisher.PendingKey(1, 9) && !finisher.PendingKey(1, 8), "the draw key or shape of a job handed to the finisher was not matched");
    std::uint64_t expected = 0;
    std::memcpy(&expected, stored.data(), 8);
    Require(pipeline.PendingLabel(0xc000, 8) == std::optional<std::uint64_t>(expected), "the bytes of a store relayed behind the finisher did not give its label a value");
    Require(!pipeline.PendingLabel(0xa000, 4).has_value(), "a job's write without bytes gave a label a value");
    Require(pipeline.StoredBytes(0xb000, 4, {}) == DrawPipeline::QueuedWrite::Unknown, "a write held by the finisher was not reported as unknown");
    Require(commits->Taken().empty(), "a relayed item reached the committer ahead of the job queued before it");

    gate.Open();
    Settle();
    RequireInOrder(commits->Taken(), 3);
    Require(!pipeline.Overlaps(0xa010, 4) && !pipeline.Overlaps(0xb000, 4), "a committed write was still reported as queued");
    Require(!DrawFinisher::Busy(), "the draw finisher reports work after it was drained");
}

}

int main() {
    try {
        if (DrawPipeline::Depth() < 8) return 77;
        QueriesFollowTheCommitterQueue();
        FullQueueKeepsTheQueuedOrder();
        RelayedItemsStayBehindTheFinisher();
        std::cout << "the pipeline queues answered their queries and kept their order with " << (DrawPipeline::Flat() ? "the flat transport" : "the locked queues") << " and " << DrawFinisher::Threads() << " finisher threads\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
