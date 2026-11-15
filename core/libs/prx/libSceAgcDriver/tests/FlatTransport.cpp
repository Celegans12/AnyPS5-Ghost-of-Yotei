#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/FlatTransport.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using AgcDriver::DriverDetail::FlatQueue;
using AgcDriver::DriverDetail::PendingLedger;
using Range = PendingLedger::Range;
using Held = PendingLedger::Held;
using Stored = PendingLedger::Stored;

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool Within(std::chrono::milliseconds patience, const std::function<bool()>& reached) {
    const auto deadline = std::chrono::steady_clock::now() + patience;
    while (!reached()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

bool Overlaps(const PendingLedger& ledger, std::uint64_t begin, std::uint64_t end, std::uint64_t before = PendingLedger::Every) {
    const Range range{begin, end};
    return ledger.Overlaps(std::span<const Range>(&range, 1), before);
}

std::vector<std::byte> Counting(std::size_t count) {
    std::vector<std::byte> bytes(count);
    for (std::size_t index = 0; index < count; ++index) bytes[index] = static_cast<std::byte>(index + 1);
    return bytes;
}

void LedgerAnswersLikeTheQueue() {
    PendingLedger ledger(8, true, true);
    Require(!Overlaps(ledger, 0, PendingLedger::Every), "an empty ledger reported a queued write");

    const std::vector<Range> plainWrites{{0x1000, 0x1100}};
    PendingLedger::Entry plain;
    plain.writes = plainWrites;
    ledger.Publish(plain);

    std::vector<Range> wideWrites;
    for (std::uint64_t index = 0; index < 9; ++index) wideWrites.emplace_back(0x7000 + index * 0x100, 0x7010 + index * 0x100);
    PendingLedger::Entry wide;
    wide.writes = wideWrites;
    ledger.Publish(wide);

    const auto stored = Counting(8);
    const std::vector<Range> storeWrites{{0x3000, 0x3008}};
    PendingLedger::Entry store;
    store.writes = storeWrites;
    store.labelAddress = 0x3000;
    store.label = stored;
    store.store = true;
    ledger.Publish(store);

    const std::vector<Range> drawReads{{0x4000, 0x4010}};
    PendingLedger::Entry draw;
    draw.reads = drawReads;
    draw.key = 7;
    draw.shape = 9;
    ledger.Publish(draw);

    Require(Overlaps(ledger, 0x10ff, 0x1200), "a range that crosses the end of a queued write was not reported");
    Require(!Overlaps(ledger, 0x1100, 0x1200), "a range that starts where a queued write ends was reported");
    Require(Overlaps(ledger, 0x7800, 0x7801), "the ninth range of an element wider than a flat record was not reported");
    Require(Overlaps(ledger, 0x1000, 0x1001, 1), "the element below the bound of a query was not reported");
    Require(!Overlaps(ledger, 0x7800, 0x7801, 1), "an element at the bound of a query was reported");

    std::uint64_t expected = 0;
    std::memcpy(&expected, stored.data() + 4, 4);
    std::uint64_t value = 0;
    Require(ledger.Label(0x3004, 4, PendingLedger::Every, value) == Held::Value && value == expected, "the bytes of a queued store did not give its label a value");
    Require(ledger.Label(0x1000, 4, PendingLedger::Every, value) == Held::Unknown, "a queued write without bytes gave a label a value");
    Require(ledger.Label(0x9000, 4, PendingLedger::Every, value) == Held::None, "a label no queued element writes was reported as held");
    Require(ledger.Label(0x3004, 4, 2, value) == Held::None, "a store at the bound of a query gave a label a value");

    std::array<std::byte, 8> out{};
    Require(ledger.StoredBytes(0x3000, 8, out) == Stored::Stored && std::equal(out.begin(), out.end(), stored.begin()), "the bytes of a queued store were not returned");
    Require(ledger.StoredBytes(0x1000, 4, std::span<std::byte>(out.data(), 4)) == Stored::Unknown, "a queued write that is not a store returned bytes");
    Require(ledger.StoredBytes(0x3000, 8, {}) == Stored::Stored, "a queued store was not reported without a buffer");
    Require(ledger.StoredBytes(0x9000, 4, {}) == Stored::None, "a range no queued element writes was reported as stored");

    Require(ledger.Reads(0x400f, 0x4020), "a range a queued element reads was not reported");
    Require(!ledger.Reads(0x4010, 0x4020), "a range that starts where a queued read ends was reported");
    Require(ledger.Keyed(7, 0) && ledger.Keyed(1, 9), "the draw key or shape of a queued element was not found");
    Require(!ledger.Keyed(1, 0) && !ledger.Keyed(1, 8), "a draw key or shape no queued element has was found");

    const auto shot = ledger.Writes();
    Require(shot.retired == 0 && shot.items == 4 && shot.writes.size() == 11, "the writes of the queued elements were not all copied");
    Require(ledger.OverlapsUnretired(0, plainWrites) == std::optional<bool>(true), "a queued write was not reported while nothing was retired");

    ledger.Retire();
    Require(!Overlaps(ledger, 0x1000, 0x1100), "a retired element was still reported");
    Require(!ledger.OverlapsUnretired(0, plainWrites).has_value(), "an element retired since the count was taken went unnoticed");
    Require(ledger.OverlapsUnretired(1, plainWrites) == std::optional<bool>(false), "a retired write was reported while nothing else was retired");
    ledger.Retire();
    Require(!Overlaps(ledger, 0x7800, 0x7801), "a retired element wider than a flat record was still reported");
    ledger.Retire();
    Require(ledger.Reads(0x4000, 0x4001), "the reads of the element at the head were not held");
    ledger.ReleaseReads();
    Require(!ledger.Reads(0x4000, 0x4001), "the reads the running element released were still held");
    ledger.Retire();

    PendingLedger::Entry unknown;
    unknown.unknown = true;
    ledger.Publish(unknown);
    Require(Overlaps(ledger, 0, 1), "an element queued without its writes did not overlap every range");
    Require(ledger.Label(0, 4, PendingLedger::Every, value) == Held::Unknown, "an element queued without its writes left a label known");
    ledger.Retire();

    const auto counted = ledger.Counted();
    Require(counted.published == 5 && counted.wide == 1, "the published or wide elements were not counted");
    Require(counted.queries != 0 && counted.lockedQueries != 0 && counted.scanned != 0, "the queries were not counted");
}

void LedgerSurvivesReuseOfItsRecords() {
    PendingLedger ledger(4, true, false);
    std::mt19937_64 random(7);
    std::size_t queued = 0;
    for (int round = 0; round < 6000; ++round) {
        if (queued < 4 && random() % 4 != 0) {
            std::vector<Range> writes(random() % 9);
            for (auto& write : writes) {
                const auto begin = (random() % 64) * 0x10;
                write = {begin, begin + 0x10 * (random() % 3)};
            }
            std::vector<Range> reads(random() % 15);
            for (auto& read : reads) {
                const auto begin = (random() % 64) * 0x10;
                read = {begin, begin + 0x10};
            }
            const auto label = Counting(random() % 3 == 0 ? random() % 24 : 0);
            PendingLedger::Entry entry;
            entry.writes = writes;
            entry.reads = reads;
            entry.labelAddress = (random() % 64) * 0x10;
            entry.label = label;
            entry.key = random() % 4;
            entry.shape = random() % 4;
            entry.unknown = random() % 16 == 0;
            entry.store = random() % 2 == 0;
            ledger.Publish(entry);
            ++queued;
        } else if (queued != 0) {
            if (random() % 4 == 0) ledger.ReleaseReads();
            ledger.Retire();
            --queued;
        }
        const auto address = (random() % 64) * 0x10;
        const Range range{address, address + 0x20};
        const std::span<const Range> ranges(&range, 1);
        std::uint64_t value = 0;
        std::array<std::byte, 4> out{};
        ledger.Overlaps(ranges, ledger.Retired() + random() % 6);
        ledger.Label(address, 4, ledger.Retired() + random() % 6, value);
        ledger.StoredBytes(address, 4, out);
        ledger.StoredBytes(address, 4, {});
        ledger.Reads(address, address + 0x20);
        ledger.Keyed(random() % 4, random() % 4);
        ledger.OverlapsUnretired(ledger.Retired(), ranges);
        Require(ledger.Writes().items == queued, "the copy of the queued writes lost count of the queued elements");
    }
}

template<typename TPayload>
void Push(FlatQueue<TPayload>& queue, TPayload payload, const PendingLedger::Entry& entry, std::atomic<bool>* waiting = nullptr) {
    typename FlatQueue<TPayload>::Turn turn(queue);
    while (queue.Full()) {
        turn.Release();
        if (waiting != nullptr) waiting->store(true);
        queue.AwaitRoom();
        turn.Retake();
    }
    queue.Stage() = std::move(payload);
    queue.Commit(entry);
}

void RetiredPayloadsLeaveOnTheNextPush() {
    FlatQueue<std::shared_ptr<int>> queue(4, 2, true, false);
    auto held = std::make_shared<int>(1);
    const std::weak_ptr<int> watched = held;
    Push(queue, std::move(held), {});
    Require(queue.Queued() == 1 && queue.Front() != nullptr, "a pushed element was not at the head of the queue");
    queue.Retire();
    Require(queue.Front() == nullptr, "a retired element stayed at the head of the queue");
    Require(!watched.expired(), "the consumer destroyed the payload of the element it retired");
    Push(queue, std::make_shared<int>(2), {});
    Require(watched.expired(), "the payload of a retired element outlived the next push");
    queue.Retire();
    queue.Close();
    Require(queue.Closed(), "a closed queue did not report it");
    queue.Park();
}

void WaitingProducerWakesOncePerBatch() {
    FlatQueue<int> queue(8, 4, true, true);
    for (int value = 0; value < 8; ++value) Push(queue, value, {});
    Require(queue.Full(), "a queue holding as many elements as its depth was not full");
    std::atomic<bool> waiting{false};
    std::atomic<bool> pushed{false};
    std::thread producer([&] {
        Push(queue, 8, {}, &waiting);
        pushed.store(true);
    });
    const bool waited = Within(std::chrono::seconds(10), [&] { return waiting.load(); });
    bool ordered = true;
    const auto take = [&](int wanted) {
        const int* front = queue.Front();
        ordered = ordered && front != nullptr && *front == wanted;
        if (front != nullptr) queue.Retire();
    };
    for (int value = 0; value < 3; ++value) take(value);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const bool early = pushed.load();
    take(3);
    const bool woken = Within(std::chrono::seconds(10), [&] { return pushed.load(); });
    while (!pushed.load()) {
        if (queue.Front() != nullptr) queue.Retire();
        std::this_thread::yield();
    }
    producer.join();
    Require(waited, "a producer did not wait for room in a full queue");
    Require(ordered, "the elements did not leave the queue in the order they were pushed in");
    Require(!early, "a producer waiting for room woke before a batch of slots was free");
    Require(woken, "a producer waiting for room did not wake once a batch of slots was free");
    Require(queue.Counted().progressWakes >= 1, "the wake of the waiting producer was not counted");
}

void ConcurrentQueueKeepsOrderAndEveryQueuedWrite() {
    constexpr std::uint64_t Elements = 60000;
    constexpr std::uint64_t Never = Elements + 1000;
    FlatQueue<std::uint64_t> queue(4, 2, false, true);
    std::atomic<const char*> problem{nullptr};
    std::atomic<bool> done{false};
    std::thread consumer([&] {
        for (std::uint64_t taken = 0; taken < Elements;) {
            const std::uint64_t* front = queue.Front();
            if (front == nullptr) {
                queue.Park();
                continue;
            }
            if (*front != taken) problem.store("an element left the queue out of the order it was pushed in");
            ++taken;
            queue.Retire();
        }
    });
    std::thread reader([&] {
        const auto& ledger = queue.Ledger();
        while (!done.load()) {
            const auto published = ledger.Published();
            if (Overlaps(ledger, Never * 0x10, Never * 0x10 + 8)) problem.store("a write no element makes was reported");
            if (published == 0) continue;
            const auto last = published - 1;
            const bool seen = Overlaps(ledger, last * 0x10, last * 0x10 + 8);
            if (!seen && ledger.Retired() <= last) problem.store("an element published before a query and retired after it was not reported");
        }
    });
    for (std::uint64_t element = 0; element < Elements; ++element) {
        const Range write{element * 0x10, element * 0x10 + 8};
        PendingLedger::Entry entry;
        entry.writes = std::span<const Range>(&write, 1);
        Push(queue, element, entry);
    }
    consumer.join();
    done.store(true);
    reader.join();
    const char* found = problem.load();
    Require(found == nullptr, found != nullptr ? found : "");
    Require(queue.Queued() == 0, "elements stayed in the queue after the consumer took them all");
    Require(queue.Ledger().Counted().published == Elements, "the pushed elements were not all counted");
    Require(queue.Counted().collisions == 0, "a single producer contended for its own turn");
}

}

int main() {
    try {
        LedgerAnswersLikeTheQueue();
        LedgerSurvivesReuseOfItsRecords();
        RetiredPayloadsLeaveOnTheNextPush();
        WaitingProducerWakesOncePerBatch();
        ConcurrentQueueKeepsOrderAndEveryQueuedWrite();
        std::cout << "flat transport tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
