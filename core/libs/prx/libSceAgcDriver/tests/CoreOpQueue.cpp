#include "prx/libSceAgcDriver/Core/include/OpQueue.hpp"
#include "prx/libSceAgcDriver/Core/include/Op.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::uint64_t Patience = std::uint64_t{1} << 26u;
constexpr std::uint64_t Factor = 0x100000001b3ull;

std::uint64_t Mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

template<typename TReady>
bool Until(TReady&& ready) {
    for (std::uint64_t turn = 0; turn < Patience; ++turn) {
        if (ready()) return true;
        std::this_thread::yield();
    }
    return ready();
}

void Idle() {
    for (int turn = 0; turn < 20000; ++turn) std::this_thread::yield();
}

template<typename TCall>
bool Throws(TCall&& call) {
    try {
        call();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

bool Put(OpQueue& queue, std::uint64_t order) {
    Op* slot = queue.Stage();
    if (slot == nullptr) return false;
    *slot = MakeOp(OpKind::Label);
    slot->order = order;
    slot->label.address = Mix(order);
    slot->label.size = static_cast<std::uint32_t>(order & 15u);
    return queue.Publish() == order;
}

bool Holds(const Op& op, std::uint64_t order) {
    return op.kind == OpKind::Label && op.order == order && op.label.address == Mix(order) && op.label.size == static_cast<std::uint32_t>(order & 15u);
}

void TestOrder() {
    constexpr std::uint64_t Total = 200000;
    OpQueue queue(OpQueue::Settings{256, 32, 256, true});
    std::uint64_t expected = 0;
    std::uint64_t wrong = 0;
    std::uint64_t oversized = 0;
    std::uint64_t received = 0;
    std::thread consumer([&] {
        for (;;) {
            const auto slice = queue.Take(64);
            if (slice.empty()) break;
            if (slice.size() > 64) ++oversized;
            for (const Op& op : slice) {
                if (!Holds(op, expected)) ++wrong;
                received = received * Factor + op.label.address + op.order;
                ++expected;
            }
            queue.Retire(slice.size());
        }
    });
    std::uint64_t sent = 0;
    std::uint64_t misplaced = 0;
    for (std::uint64_t order = 0; order < Total; ++order) {
        Op* slot = queue.Stage();
        while (slot == nullptr) {
            queue.AwaitRoom();
            slot = queue.Stage();
        }
        *slot = MakeOp(OpKind::Label);
        slot->order = order;
        slot->label.address = Mix(order);
        slot->label.size = static_cast<std::uint32_t>(order & 15u);
        if (queue.Publish() != order) ++misplaced;
        sent = sent * Factor + Mix(order) + order;
        if ((order & 1023u) == 1023u) queue.Flush();
    }
    queue.Flush();
    queue.AwaitRetired(Total);
    Expect(queue.Retired() == Total, "order: every operation is retired once AwaitRetired returns");
    Expect(queue.Published() == Total, "order: Published counts every operation");
    queue.Close();
    consumer.join();
    Expect(misplaced == 0, "order: Publish returns positions growing from 0");
    Expect(expected == Total, "order: the consumer receives every operation");
    Expect(wrong == 0, "order: operations arrive in order with their payload");
    Expect(oversized == 0, "order: Take never returns more than asked");
    Expect(received == sent, "order: checksum of the payloads in order");
    const auto counts = queue.Counted();
    Expect(counts.published == Total, "order: published count");
    Expect(counts.taken == Total, "order: taken count");
    std::printf("order: parks=%llu batchWakes=%llu flushWakes=%llu roomWaits=%llu spun=%llu\n", static_cast<unsigned long long>(counts.parks), static_cast<unsigned long long>(counts.batchWakes), static_cast<unsigned long long>(counts.flushWakes), static_cast<unsigned long long>(counts.roomWaits), static_cast<unsigned long long>(counts.spun));
}

void TestFull() {
    {
        OpQueue queue(OpQueue::Settings{6, 2, 0, true});
        bool placed = true;
        for (std::uint64_t order = 0; order < 6; ++order) placed = Put(queue, order) && placed;
        Expect(placed, "full: depth operations are staged when depth is not a power of two");
        Expect(queue.Stage() == nullptr, "full: Stage is null once depth operations are queued, below the ring capacity");
    }
    {
        OpQueue queue(OpQueue::Settings{8, 4, 0, true});
        queue.AwaitRoom();
        Expect(queue.Counted().roomWaits == 0, "full: AwaitRoom returns at once when there is room");
        bool placed = true;
        for (std::uint64_t order = 0; order < 8; ++order) placed = Put(queue, order) && placed;
        Expect(placed, "full: depth operations are staged");
        Expect(queue.Stage() == nullptr, "full: Stage is null on a full queue");
        Expect(queue.Published() == 8 && queue.Retired() == 0, "full: positions of a full queue");
        std::uint64_t wrong = 0;
        std::uint64_t expected = 0;
        bool waited = false;
        std::thread consumer([&] {
            waited = Until([&] { return queue.Counted().roomWaits == 1; });
            for (;;) {
                const auto slice = queue.Take(1);
                if (slice.empty()) break;
                if (slice.size() != 1 || !Holds(slice[0], expected)) ++wrong;
                ++expected;
                queue.Retire(1);
            }
        });
        queue.AwaitRoom();
        Expect(queue.Retired() >= 4, "full: AwaitRoom returns once wakeBatch slots are free");
        Expect(queue.Counted().roomWaits == 1, "full: one room wait counted");
        Expect(Put(queue, 8), "full: Stage succeeds after AwaitRoom");
        queue.AwaitRetired(9);
        Expect(queue.Retired() == 9, "full: the queue drains");
        queue.Close();
        consumer.join();
        Expect(waited, "full: the consumer saw the producer wait for room");
        Expect(wrong == 0 && expected == 9, "full: order kept through a full queue");
    }
}

void TestWakes() {
    OpQueue queue(OpQueue::Settings{256, 32, 4, true});
    std::uint64_t wrong = 0;
    std::uint64_t expected = 0;
    std::thread consumer([&] {
        for (;;) {
            const auto slice = queue.Take(256);
            if (slice.empty()) break;
            for (const Op& op : slice) {
                if (!Holds(op, expected)) ++wrong;
                ++expected;
            }
            queue.Retire(slice.size());
        }
    });
    Expect(Until([&] { return queue.Counted().parks == 1; }), "wakes: the consumer sleeps on an empty queue");
    Expect(queue.Counted().spun == 4, "wakes: the pause turns before the first sleep are counted");
    bool placed = true;
    for (std::uint64_t order = 0; order < 31; ++order) placed = Put(queue, order) && placed;
    Idle();
    auto counts = queue.Counted();
    Expect(placed, "wakes: operations below the batch are staged");
    Expect(counts.batchWakes == 0 && counts.flushWakes == 0, "wakes: no wake below wakeBatch");
    Expect(counts.taken == 0 && counts.parks == 1 && queue.Retired() == 0, "wakes: the consumer stays asleep below wakeBatch");
    Expect(Put(queue, 31), "wakes: the operation that completes the batch is staged");
    Expect(Until([&] { return queue.Retired() == 32; }), "wakes: the consumer is woken at wakeBatch");
    counts = queue.Counted();
    Expect(counts.batchWakes == 1 && counts.flushWakes == 0, "wakes: one batch wake counted");
    Expect(counts.taken == 32, "wakes: the whole batch is taken");
    Expect(Until([&] { return queue.Counted().parks == 2; }), "wakes: the consumer sleeps again once the queue is empty");
    Expect(Put(queue, 32), "wakes: one operation after the batch is staged");
    Idle();
    counts = queue.Counted();
    Expect(counts.taken == 32 && counts.batchWakes == 1 && counts.flushWakes == 0, "wakes: one operation alone does not wake the consumer");
    queue.Flush();
    Expect(Until([&] { return queue.Retired() == 33; }), "wakes: Flush wakes the consumer for one operation");
    counts = queue.Counted();
    Expect(counts.flushWakes == 1 && counts.batchWakes == 1, "wakes: one flush wake counted");
    Expect(Until([&] { return queue.Counted().parks == 3; }), "wakes: the consumer sleeps before the close");
    queue.Flush();
    Expect(queue.Counted().flushWakes == 1, "wakes: Flush on an empty queue wakes nobody");
    queue.Close();
    consumer.join();
    Expect(queue.Closed(), "close: Closed is true after Close");
    Expect(queue.Stage() == nullptr, "close: Stage is null after Close");
    Expect(wrong == 0 && expected == 33, "wakes: order kept across sleeps");
    Expect(queue.Counted().parks == 3, "close: a sleeping consumer is woken by Close and leaves");
}

void TestAwaitRetired() {
    OpQueue queue(OpQueue::Settings{64, 32, 0, true});
    std::thread consumer([&] {
        for (;;) {
            const auto slice = queue.Take(8);
            if (slice.empty()) break;
            queue.Retire(slice.size());
        }
    });
    Expect(Until([&] { return queue.Counted().parks == 1; }), "await retired: the consumer sleeps first");
    bool placed = true;
    for (std::uint64_t order = 0; order < 10; ++order) placed = Put(queue, order) && placed;
    Expect(placed, "await retired: operations are staged");
    queue.AwaitRetired(10);
    Expect(queue.Retired() == 10, "await retired: returns once the position is retired");
    auto counts = queue.Counted();
    Expect(counts.flushWakes == 1 && counts.batchWakes == 0, "await retired: flushes first, below wakeBatch");
    queue.AwaitRetired(3);
    queue.AwaitRetired(0);
    Expect(queue.Retired() == 10, "await retired: a position already retired returns at once");
    queue.Close();
    consumer.join();
}

void TestCloseWakesProducer() {
    OpQueue queue(OpQueue::Settings{4, 2, 0, true});
    std::atomic<bool> returned{false};
    bool placed = true;
    bool full = false;
    std::thread producer([&] {
        for (std::uint64_t order = 0; order < 4; ++order) placed = Put(queue, order) && placed;
        full = queue.Stage() == nullptr;
        queue.AwaitRoom();
        returned.store(true);
    });
    Expect(Until([&] { return queue.Counted().roomWaits == 1; }), "close: the producer waits for room");
    Idle();
    Expect(!returned.load(), "close: the producer sleeps while nothing is retired");
    queue.Close();
    producer.join();
    Expect(returned.load(), "close: a producer asleep in AwaitRoom is woken by Close");
    Expect(placed && full, "close: the queue was full before the wait");
    const auto rest = queue.Take(8);
    Expect(rest.size() == 4 && Holds(rest[0], 0) && Holds(rest[3], 3), "close: operations published before Close are still taken");
    queue.Retire(rest.size());
    Expect(queue.Take(8).empty(), "close: Take is empty once closed and drained");
    Expect(queue.TryTake(8).empty(), "close: TryTake is empty once closed and drained");
}

void TestRingEnd() {
    OpQueue queue(OpQueue::Settings{8, 4, 0, true});
    Expect(queue.TryTake(8).empty(), "ring: TryTake is empty on a new queue");
    bool placed = true;
    for (std::uint64_t order = 0; order < 6; ++order) placed = Put(queue, order) && placed;
    const auto bounded = queue.TryTake(3);
    Expect(bounded.size() == 3 && Holds(bounded[0], 0) && Holds(bounded[2], 2), "ring: TryTake is bounded by most");
    const auto first = queue.TryTake(8);
    Expect(first.size() == 6 && first.data() == bounded.data(), "ring: a second take without Retire starts at the same operation");
    Expect(queue.Counted().taken == 6, "ring: operations handed twice are counted once");
    queue.Retire(1);
    const auto after = queue.TryTake(8);
    Expect(after.size() == 5 && Holds(after[0], 1) && after.data() == first.data() + 1, "ring: a partial Retire moves the head by its count");
    queue.Retire(5);
    Expect(queue.Retired() == 6, "ring: Retired follows Retire");
    for (std::uint64_t order = 6; order < 12; ++order) placed = Put(queue, order) && placed;
    Expect(placed, "ring: operations are staged across the ring end");
    const auto tail = queue.TryTake(8);
    Expect(tail.size() == 2 && Holds(tail[0], 6) && Holds(tail[1], 7), "ring: a take stops at the ring end");
    Expect(tail.data() == first.data() + 6, "ring: the tail slice sits at the end of the ring");
    queue.Retire(tail.size());
    const auto wrapped = queue.Take(8);
    Expect(wrapped.size() == 4, "ring: the next take returns the operations after the ring end");
    bool ordered = wrapped.size() == 4;
    for (std::size_t at = 0; ordered && at < wrapped.size(); ++at) ordered = Holds(wrapped[at], 8 + at);
    Expect(ordered, "ring: order kept across the ring end");
    Expect(wrapped.data() == first.data(), "ring: the slice after the ring end starts at the first slot");
    queue.Retire(wrapped.size());
    Expect(queue.TryTake(8).empty(), "ring: empty after everything is retired");
    Expect(queue.Take(0).empty(), "ring: Take of zero returns at once");
    const auto counts = queue.Counted();
    Expect(counts.published == 12 && counts.taken == 12, "ring: counts");
    Expect(queue.Published() == 12 && queue.Retired() == 12, "ring: positions");
}

void TestVerifies() {
    {
        OpQueue queue(OpQueue::Settings{16, 4, 0, true});
        Expect(Put(queue, 0), "verifies: the first producer stages");
        bool stageThrew = false;
        bool publishThrew = false;
        bool flushThrew = false;
        std::thread other([&] {
            stageThrew = Throws([&] { (void)queue.Stage(); });
            publishThrew = Throws([&] { (void)queue.Publish(); });
            flushThrew = Throws([&] { queue.Flush(); });
        });
        other.join();
        Expect(stageThrew, "verifies: Stage from a second producer throws std::logic_error");
        Expect(publishThrew, "verifies: Publish from a second producer throws std::logic_error");
        Expect(flushThrew, "verifies: Flush from a second producer throws std::logic_error");
        Expect(Put(queue, 1), "verifies: the first producer continues");
        Expect(queue.Published() == 2, "verifies: a refused producer publishes nothing");
        const auto slice = queue.TryTake(1);
        Expect(slice.size() == 1, "verifies: the first consumer takes");
        bool takeThrew = false;
        bool retireThrew = false;
        std::thread third([&] {
            takeThrew = Throws([&] { (void)queue.TryTake(1); });
            retireThrew = Throws([&] { queue.Retire(1); });
        });
        third.join();
        Expect(takeThrew, "verifies: TryTake from a second consumer throws std::logic_error");
        Expect(retireThrew, "verifies: Retire from a second consumer throws std::logic_error");
        Expect(Throws([&] { queue.Retire(2); }), "verifies: Retire past what Take returned throws std::logic_error");
        Expect(queue.Retired() == 0, "verifies: a refused Retire retires nothing");
        queue.Retire(1);
        Expect(queue.Retired() == 1, "verifies: Retire within what Take returned passes");
        Expect(Throws([&] { (void)queue.Publish(); }), "verifies: Publish without Stage throws std::logic_error");
    }
    {
        OpQueue queue(OpQueue::Settings{16, 4, 0, false});
        Expect(Put(queue, 0), "verifies off: the first thread stages");
        bool threw = true;
        std::thread other([&] { threw = Throws([&] { queue.Flush(); }); });
        other.join();
        Expect(!threw, "verifies off: no thread check");
    }
}

std::string SourcePath(int argc, char** argv) {
    if (argc > 1) return argv[1];
    std::string path = __FILE__;
    const auto cut = path.find_last_of("/\\");
    path = cut == std::string::npos ? std::string() : path.substr(0, cut + 1);
    return path + "../Core/src/OpQueue.cpp";
}

void TestSource(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "FAIL: source: cannot open %s (pass the path of Core/src/OpQueue.cpp as the first argument)\n", path.c_str());
        ++failures;
        return;
    }
    const std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    Expect(text.find("OpQueue::Take") != std::string::npos, "source: the file read is the queue source");
    Expect(text.find("chrono") == std::string::npos, "source: no chrono in the queue source");
    Expect(text.find("clock") == std::string::npos, "source: no clock in the queue source");
}

}

int main(int argc, char** argv) {
    TestRingEnd();
    TestFull();
    TestWakes();
    TestAwaitRetired();
    TestCloseWakesProducer();
    TestVerifies();
    TestOrder();
    TestSource(SourcePath(argc, argv));
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("core op queue: ok\n");
    return 0;
}
