#include "prx/libSceAgcDriver/Core/include/LegacySlots.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
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

void ExpectIn(bool condition, std::size_t slots, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: table of %llu slots: %s\n", static_cast<unsigned long long>(slots), what);
    ++failures;
}

template <typename Error, typename Action>
bool Throws(Action&& action) {
    try {
        action();
    } catch (const Error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

struct Tally {
    std::atomic<std::int64_t> live{0};
    std::atomic<std::uint64_t> foreign{0};
    std::mutex lock;
    std::vector<std::thread::id> where;

    std::size_t Destroyed() {
        const std::lock_guard guard(lock);
        return where.size();
    }

    std::vector<std::thread::id> Since(std::size_t from) {
        const std::lock_guard guard(lock);
        return std::vector<std::thread::id>(where.begin() + static_cast<std::ptrdiff_t>(from), where.end());
    }
};

class Probe {
public:
    explicit Probe(Tally& into) : tally(&into), owner(std::this_thread::get_id()) { tally->live.fetch_add(1); }
    Probe(const Probe& other) : tally(other.tally), owner(other.owner) {
        if (tally != nullptr) tally->live.fetch_add(1);
    }
    Probe(Probe&& other) noexcept : tally(std::exchange(other.tally, nullptr)), owner(other.owner) {}
    Probe& operator=(const Probe&) = delete;
    Probe& operator=(Probe&&) = delete;
    ~Probe() {
        if (tally == nullptr) return;
        const auto here = std::this_thread::get_id();
        if (here != owner) tally->foreign.fetch_add(1);
        {
            const std::lock_guard guard(tally->lock);
            tally->where.push_back(here);
        }
        tally->live.fetch_sub(1);
    }

private:
    Tally* tally;
    std::thread::id owner;
};

bool allOn(const std::vector<std::thread::id>& threads, std::thread::id id) {
    for (const auto thread : threads) {
        if (thread != id) return false;
    }
    return true;
}

void testHoldRunReclaim() {
    LegacySlots table;
    Tally tally;
    int hits = 0;
    const auto slot = table.Hold([probe = Probe(tally), &hits] { ++hits; });
    Expect(slot != LegacySlots::Full && slot < 1024u, "Hold gives a slot of the default table of 1024");
    Expect(tally.live.load() == 1, "the held closure lives in its slot");
    Expect(table.Counted().held == 1, "Hold is counted");
    Expect(table.Reclaim() == 0, "Reclaim before Run reclaims nothing");
    Expect(tally.live.load() == 1, "a slot that is not finished keeps its closure");
    table.Run(slot);
    Expect(hits == 1, "Run executes the closure once");
    Expect(tally.live.load() == 1, "Run does not destroy the closure");
    Expect(table.Counted().run == 1, "Run is counted");
    Expect(table.Counted().reclaimed == 0, "Run reclaims nothing");
    Expect(table.Reclaim() == 1, "Reclaim gives the number of finished slots it reclaimed");
    Expect(tally.live.load() == 0, "Reclaim destroys the closure");
    Expect(table.Reclaim() == 0, "a second Reclaim finds nothing");
    const auto counts = table.Counted();
    Expect(counts.held == 1 && counts.run == 1 && counts.skipped == 0 && counts.reclaimed == 1 && counts.full == 0, "counts after one Hold, Run and Reclaim");
    Expect(Throws<std::logic_error>([&] { table.Run(slot); }), "a reclaimed slot cannot run again");
    Expect(Throws<std::out_of_range>([&] { table.Run(1024u); }), "Run past the table is out of range");
    Expect(Throws<std::out_of_range>([&] { table.Skip(LegacySlots::Full); }), "Skip of Full is out of range");
    Expect(hits == 1, "refused calls run nothing");

    std::array<std::uint32_t, 3> order{};
    std::vector<int> seen;
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = table.Hold([index, &seen] { seen.push_back(static_cast<int>(index)); });
    }
    Expect(order[0] != order[1] && order[1] != order[2] && order[0] != order[2], "slots held together are distinct");
    table.Run(order[2]);
    table.Run(order[0]);
    table.Run(order[1]);
    Expect(seen == std::vector<int>{2, 0, 1}, "each slot runs its own closure, in the order of the Run calls");
    Expect(table.Reclaim() == 3, "slots finished out of order are all reclaimed");
}

void testReclaimThread() {
    {
        LegacySlots table;
        Tally tally;
        std::atomic<int> hits{0};
        const auto slot = table.Hold([probe = Probe(tally), &hits] { hits.fetch_add(1); });
        const auto before = tally.Destroyed();
        std::thread::id runner;
        std::thread consumer([&] {
            runner = std::this_thread::get_id();
            table.Run(slot);
        });
        consumer.join();
        Expect(hits.load() == 1, "the consumer thread runs the closure");
        Expect(tally.Destroyed() == before && tally.live.load() == 1, "Run on the consumer thread destroys nothing");
        Expect(table.Reclaim() == 1, "the producer reclaims the slot the consumer finished");
        const auto after = tally.Since(before);
        Expect(after.size() == 1 && after[0] == std::this_thread::get_id(), "the closure is destroyed on the thread that calls Reclaim");
        Expect(after.size() == 1 && after[0] != runner, "the closure is not destroyed on the thread that ran it");
        Expect(tally.foreign.load() == 0, "no closure is destroyed away from the thread that made it");
    }
    {
        LegacySlots table;
        Tally tally;
        const auto slot = table.Hold([probe = Probe(tally)] {});
        table.Run(slot);
        const auto before = tally.Destroyed();
        std::thread::id reclaimer;
        std::size_t reclaimed = 0;
        std::thread other([&] {
            reclaimer = std::this_thread::get_id();
            reclaimed = table.Reclaim();
        });
        other.join();
        const auto after = tally.Since(before);
        Expect(reclaimed == 1 && tally.live.load() == 0, "Reclaim on another thread reclaims the finished slot");
        Expect(after.size() == 1 && after[0] == reclaimer && reclaimer != std::this_thread::get_id(), "the closure is destroyed on that other thread, the one calling Reclaim");
    }
}

void testFullThenFreed() {
    LegacySlots table(4);
    Tally tally;
    std::array<std::uint32_t, 4> held{};
    for (auto& slot : held) slot = table.Hold([probe = Probe(tally)] {});
    bool distinct = true;
    for (std::size_t first = 0; first < held.size(); ++first) {
        if (held[first] >= 4u) distinct = false;
        for (std::size_t second = first + 1; second < held.size(); ++second) {
            if (held[first] == held[second]) distinct = false;
        }
    }
    Expect(distinct, "four Holds on a table of four give the four slots");
    Expect(tally.live.load() == 4, "four closures held");

    const auto before = tally.Destroyed();
    const auto refused = table.Hold([probe = Probe(tally)] {});
    Expect(refused == LegacySlots::Full, "a fifth Hold on a full table gives Full");
    Expect(table.Counted().full == 1, "the refusal is counted");
    Expect(table.Counted().held == 4, "the refusal is not counted as held");
    Expect(tally.live.load() == 4, "the refused closure is destroyed and the four held ones stay");
    const auto refusedDestroyed = tally.Since(before);
    Expect(!refusedDestroyed.empty() && allOn(refusedDestroyed, std::this_thread::get_id()), "the refused closure is destroyed on the thread that called Hold");

    table.Run(held[1]);
    table.Run(held[3]);
    Expect(tally.live.load() == 4, "finished slots keep their closures until reclaimed");
    const auto again = table.Hold([probe = Probe(tally)] {});
    Expect(again == held[1] || again == held[3], "Hold on a full table reclaims first and reuses a finished slot");
    Expect(table.Counted().reclaimed == 2, "the Reclaim inside Hold reclaims both finished slots");
    Expect(tally.live.load() == 3, "both finished closures destroyed, one new held");
    const auto last = table.Hold([probe = Probe(tally)] {});
    Expect(last != again && (last == held[1] || last == held[3]), "the other reclaimed slot is reused next");
    Expect(tally.live.load() == 4, "the table holds four closures again");
    Expect(table.Hold([probe = Probe(tally)] {}) == LegacySlots::Full, "the table is full again");
    const auto counts = table.Counted();
    Expect(counts.held == 6 && counts.full == 2 && counts.run == 2 && counts.reclaimed == 2, "counts after the table filled, freed and filled again");

    for (const auto slot : {held[0], held[2], again, last}) table.Run(slot);
    Expect(table.Reclaim() == 4 && tally.live.load() == 0, "the whole table is freed by one Reclaim");
    std::array<std::uint32_t, 4> refilled{};
    for (auto& slot : refilled) slot = table.Hold([] {});
    bool refilledAll = true;
    for (const auto slot : refilled) refilledAll = refilledAll && slot != LegacySlots::Full;
    Expect(refilledAll, "a freed table takes four Holds again");
}

void testReuseOnlyAfterReclaim() {
    LegacySlots table(3);
    const auto first = table.Hold([] {});
    table.Run(first);
    const auto second = table.Hold([] {});
    const auto third = table.Hold([] {});
    Expect(second != first && third != first && second != third, "a finished slot is not reused while it has not been reclaimed");
    Expect(table.Counted().reclaimed == 0, "Hold does not reclaim while the table has room");
    const auto fourth = table.Hold([] {});
    Expect(fourth == first, "the finished slot comes back only through the Reclaim of a full Hold");
    Expect(table.Counted().reclaimed == 1, "that Reclaim is counted");

    LegacySlots none(0);
    Expect(none.Hold([] {}) == LegacySlots::Full, "a table of no slot is always full");
    Expect(none.Reclaim() == 0, "a table of no slot reclaims nothing");
    Expect(Throws<std::out_of_range>([&] { none.Run(0); }), "a table of no slot has no slot to run");
    Expect(none.Counted().full == 1, "its refusal is counted");
    Expect(Throws<std::length_error>([] { LegacySlots huge(std::size_t{LegacySlots::Full}); }), "a table whose slots would reach Full is refused");
}

void testSkip() {
    LegacySlots table;
    Tally tally;
    int hits = 0;
    const auto slot = table.Hold([probe = Probe(tally), &hits] { ++hits; });
    table.Skip(slot);
    Expect(hits == 0, "Skip does not run the closure");
    Expect(table.Counted().skipped == 1 && table.Counted().run == 0, "Skip is counted apart from Run");
    Expect(tally.live.load() == 1, "Skip does not destroy the closure");
    Expect(table.Reclaim() == 1 && tally.live.load() == 0, "a skipped slot is finished and reclaimed");
    Expect(hits == 0, "Reclaim does not run a skipped closure");

    const auto other = table.Hold([&hits] { ++hits; });
    table.Skip(other);
    Expect(Throws<std::logic_error>([&] { table.Skip(other); }), "a slot cannot be skipped twice");
    Expect(Throws<std::logic_error>([&] { table.Run(other); }), "a skipped slot cannot run");
    Expect(hits == 0, "nothing ran");
    Expect(table.Reclaim() == 1, "the skipped slot is reclaimed once");
    Expect(table.Counted().skipped == 2, "only the accepted Skips are counted");
}

void testThrowingRun() {
    LegacySlots table;
    Tally tally;
    const auto slot = table.Hold([probe = Probe(tally)] { throw std::runtime_error("legacy work failed"); });
    bool caught = false;
    try {
        table.Run(slot);
    } catch (const std::runtime_error& error) {
        caught = std::string_view(error.what()) == "legacy work failed";
    }
    Expect(caught, "the exception of the closure leaves Run unchanged");
    Expect(table.Counted().run == 1, "the throwing Run is counted");
    Expect(tally.live.load() == 1, "the closure that threw lives until Reclaim");
    Expect(Throws<std::logic_error>([&] { table.Run(slot); }), "the slot is finished: it cannot run again");
    Expect(Throws<std::logic_error>([&] { table.Skip(slot); }), "the slot is finished: it cannot be skipped");
    Expect(table.Reclaim() == 1 && tally.live.load() == 0, "the slot finished by a throwing Run is reclaimed");

    const auto empty = table.Hold(LegacySlots::Work{});
    Expect(empty != LegacySlots::Full, "an empty closure is held");
    Expect(Throws<std::bad_function_call>([&] { table.Run(empty); }), "running an empty closure throws bad_function_call");
    Expect(table.Reclaim() == 1, "the slot of the empty closure is finished too");
    Expect(table.Counted().run == 2, "both throwing Runs are counted");
}

struct Handoff {
    std::uint32_t slot = 0;
    std::uint32_t index = 0;
};

class Channel {
public:
    explicit Channel(std::size_t capacity) : items(capacity) {}

    void Push(Handoff handoff) {
        const auto at = tail.load(std::memory_order_relaxed);
        while (at - head.load(std::memory_order_acquire) >= items.size()) std::this_thread::yield();
        items[at % items.size()] = handoff;
        tail.store(at + 1, std::memory_order_release);
    }

    Handoff Pop() {
        const auto at = head.load(std::memory_order_relaxed);
        while (tail.load(std::memory_order_acquire) == at) std::this_thread::yield();
        const auto handoff = items[at % items.size()];
        head.store(at + 1, std::memory_order_release);
        return handoff;
    }

private:
    std::vector<Handoff> items;
    std::atomic<std::uint64_t> head{0};
    std::atomic<std::uint64_t> tail{0};
};

void testTwoThreads(std::size_t slots) {
    constexpr std::uint32_t Total = 50000;
    LegacySlots table(slots);
    Tally tally;
    Channel channel(256);
    std::vector<std::uint8_t> ran(Total, 0);
    std::vector<std::atomic<std::uint8_t>> busy(slots);
    std::atomic<std::uint64_t> twice{0};
    std::uint64_t refusals = 0;
    std::uint64_t skips = 0;
    std::uint64_t mismatched = 0;
    std::uint32_t lastRun = LegacySlots::Full;

    std::thread producer([&] {
        for (std::uint32_t index = 0; index < Total; ++index) {
            auto slot = LegacySlots::Full;
            while ((slot = table.Hold([probe = Probe(tally), index, &ran, &lastRun] {
                        ++ran[index];
                        lastRun = index;
                    })) == LegacySlots::Full) {
                ++refusals;
                std::this_thread::yield();
            }
            if (busy[slot].exchange(1, std::memory_order_relaxed) != 0) twice.fetch_add(1, std::memory_order_relaxed);
            channel.Push({slot, index});
            if (index % 16u == 15u) static_cast<void>(table.Reclaim());
        }
        while (table.Counted().reclaimed < Total) {
            static_cast<void>(table.Reclaim());
            std::this_thread::yield();
        }
    });
    std::thread consumer([&] {
        for (std::uint32_t taken = 0; taken < Total; ++taken) {
            const auto handoff = channel.Pop();
            busy[handoff.slot].store(0, std::memory_order_relaxed);
            if (handoff.index % 7u == 3u) {
                table.Skip(handoff.slot);
                ++skips;
                continue;
            }
            table.Run(handoff.slot);
            if (lastRun != handoff.index) ++mismatched;
        }
    });
    producer.join();
    consumer.join();

    std::uint64_t wrong = 0;
    std::uint64_t sum = 0;
    std::uint64_t expectedSum = 0;
    std::uint64_t expectedSkips = 0;
    for (std::uint32_t index = 0; index < Total; ++index) {
        const bool skipped = index % 7u == 3u;
        const std::uint8_t expected = skipped ? 0 : 1;
        if (skipped) ++expectedSkips;
        else expectedSum += index;
        if (ran[index] != expected) ++wrong;
        sum += static_cast<std::uint64_t>(ran[index]) * index;
    }
    const auto counts = table.Counted();
    ExpectIn(wrong == 0, slots, "every closure ran exactly once, and skipped ones never");
    ExpectIn(sum == expectedSum, slots, "the sum of the indexes that ran matches");
    ExpectIn(mismatched == 0, slots, "each Run executed the closure held for that handoff");
    ExpectIn(twice.load() == 0, slots, "no slot was handed out again while still held");
    ExpectIn(skips == expectedSkips, slots, "the consumer skipped every seventh handoff");
    ExpectIn(counts.held == Total, slots, "50000 Holds counted");
    ExpectIn(counts.run == Total - expectedSkips && counts.skipped == expectedSkips, slots, "Runs and Skips counted");
    ExpectIn(counts.reclaimed == Total, slots, "every slot reclaimed");
    ExpectIn(counts.full == refusals, slots, "every Full counted");
    ExpectIn(tally.live.load() == 0, slots, "every closure destroyed");
    ExpectIn(tally.foreign.load() == 0, slots, "every closure destroyed on the producer thread, none on the consumer");
    ExpectIn(table.Reclaim() == 0, slots, "nothing left to reclaim");
}

}

int main() {
    testHoldRunReclaim();
    testReclaimThread();
    testFullThenFreed();
    testReuseOnlyAfterReclaim();
    testSkip();
    testThrowingRun();
    testTwoThreads(64);
    testTwoThreads(50000);
    if (failures != 0) {
        std::fprintf(stderr, "%d legacy slot checks failed\n", failures);
        return 1;
    }
    std::printf("legacy slot tests passed\n");
    return 0;
}
