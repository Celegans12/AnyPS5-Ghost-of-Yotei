#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

using namespace AgcDriver;
using LockOrder::CacheLock;
using LockOrder::CacheLockFor;
using LockOrder::OrderError;
using LockOrder::Rank;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

template<typename F>
bool Throws(F&& f) {
    try {
        f();
    } catch (const OrderError&) {
        return true;
    }
    return false;
}

void RankOrder() {
    LockOrder::SetCheckForTest(true);
    auto& texture = CacheLockFor(Rank::Texture);
    auto& storage = CacheLockFor(Rank::Storage);
    auto& pipeline = CacheLockFor(Rank::Pipeline);
    const auto before = LockOrder::Totals().violations;
    Expect(!Throws([&] {
        texture.lock();
        storage.lock();
        pipeline.lock_shared();
        pipeline.unlock_shared();
        storage.unlock();
        texture.unlock();
    }), "texture < storage < pipeline is the right order");
    storage.lock();
    Expect(Throws([&] { texture.lock(); }), "texture after storage throws");
    Expect(Throws([&] { texture.lock_shared(); }), "shared texture after storage throws");
    Expect(!texture.HeldByThisThread(), "a refused acquire holds nothing");
    {
        LockOrder::Annotate tracker(Rank::Tracker);
        Expect(Throws([&] { LockOrder::Annotate arm(Rank::Imports); }), "imports after tracker throws");
        Expect(!Throws([&] { LockOrder::Annotate arm(Rank::DirtyArm); }), "dirty arm after tracker is fine");
    }
    storage.unlock();
    Expect(LockOrder::Totals().violations == before + 3, "three violations counted");
    Expect(LockOrder::HeldText() == "nothing", "nothing held at the end: " + LockOrder::HeldText());
    LockOrder::SetCheckForTest(false);
    storage.lock();
    Expect(!Throws([&] {
        texture.lock();
        texture.unlock();
    }), "with the check off a wrong order does not throw");
    storage.unlock();
}

void RecordRecursion() {
    LockOrder::SetCheckForTest(true);
    auto& record = GuestMemory::Record();
    auto& texture = CacheLockFor(Rank::Texture);
    Expect(!Throws([&] {
        std::lock_guard outer(record);
        std::lock_guard cache(texture);
        std::lock_guard inner(record);
        Expect(record.DepthOnThisThread() == 2, "record lock recursion depth 2");
        Expect(LockOrder::HeldOnThisThread(Rank::Record) == 2, "HeldOnThisThread reads the record depth");
        std::lock_guard nested(texture);
        Expect(LockOrder::HeldOnThisThread(Rank::Texture) == 2, "texture lock re-entered");
    }), "record lock re-entered while holding a cache lock is fine");
    texture.lock();
    Expect(Throws([&] { record.lock(); }), "first record acquire while holding a cache lock throws");
    Expect(!record.HeldByThisThread(), "the refused record acquire holds nothing");
    texture.unlock();
    const auto releases = LockOrder::Totals().releasesHolding;
    record.lock();
    texture.lock();
    record.unlock();
    Expect(LockOrder::Totals().releasesHolding == releases + 1, "record released while holding a cache lock is counted");
    texture.unlock();
    LockOrder::SetCheckForTest(false);
}

void SharedExclusive() {
    LockOrder::SetCheckForTest(true);
    auto& storage = CacheLockFor(Rank::Storage);
    storage.lock_shared();
    storage.lock_shared();
    Expect(LockOrder::HeldOnThisThread(Rank::Storage) == 2, "shared re-entered");
    bool otherShared = false;
    bool otherExclusive = true;
    std::thread([&] {
        otherShared = storage.try_lock_shared();
        if (otherShared) storage.unlock_shared();
        otherExclusive = storage.try_lock();
        if (otherExclusive) storage.unlock();
    }).join();
    Expect(otherShared, "a second reader gets the lock shared");
    Expect(!otherExclusive, "a writer is kept out by a reader");
    Expect(Throws([&] { storage.lock(); }), "exclusive while holding shared throws");
    Expect(!storage.HeldExclusiveByThisThread(), "still shared only");
    storage.unlock_shared();
    storage.unlock_shared();
    Expect(!storage.HeldByThisThread(), "released");
    storage.lock();
    storage.lock_shared();
    Expect(storage.HeldExclusiveByThisThread(), "shared inside exclusive keeps it exclusive");
    std::thread([&] {
        otherShared = storage.try_lock_shared();
        if (otherShared) storage.unlock_shared();
    }).join();
    Expect(!otherShared, "a reader is kept out by a writer");
    storage.unlock();
    Expect(storage.HeldExclusiveByThisThread(), "still exclusive while the nested shared hold lasts");
    storage.unlock_shared();
    Expect(!storage.HeldByThisThread(), "released after both");
    std::thread([&] {
        otherExclusive = storage.try_lock();
        if (otherExclusive) storage.unlock();
    }).join();
    Expect(otherExclusive, "free for others again");
    LockOrder::SetCheckForTest(false);
}

void Split() {
    using Graphics::RuntimeSwitch;
    namespace Switches = Graphics::RuntimeSwitches;
    LockOrder::SplitMutex<AgcDriver::Mutex, Rank::ResourceSet> mutex;
    auto& cache = CacheLockFor(Rank::ResourceSet);
    auto& record = GuestMemory::Record();
    LockOrder::SetLeafTakesForTest(true);
    Switches::Assign(RuntimeSwitch::V3eLockSplit, false);
    {
        std::lock_guard recordHold(record);
        std::lock_guard hold(mutex);
        Expect(!cache.HeldByThisThread(), "switch off: no cache lock");
    }
    Switches::Assign(RuntimeSwitch::V3eLockSplit, true);
    {
        std::lock_guard hold(mutex);
        Expect(!cache.HeldByThisThread(), "switch on without the record lock: no cache lock");
    }
    {
        std::lock_guard recordHold(record);
        {
            std::lock_guard hold(mutex);
            Expect(cache.HeldExclusiveByThisThread(), "switch on under the record lock: cache lock held");
            Expect(LockOrder::MustNotWait(), "MustNotWait under the record lock");
        }
        Expect(!cache.HeldByThisThread(), "cache lock released with the mutex");
        std::unique_lock hold(mutex);
        Switches::Assign(RuntimeSwitch::V3eLockSplit, false);
        hold.unlock();
        Expect(!cache.HeldByThisThread(), "a flip between lock and unlock still releases the cache lock");
        Switches::Assign(RuntimeSwitch::V3eLockSplit, true);
        Expect(mutex.try_lock(), "try_lock");
        Expect(cache.HeldByThisThread(), "try_lock takes the cache lock");
        mutex.unlock();
    }
    cache.lock();
    Expect(LockOrder::MustNotWait(), "switch on: a cache lock alone means must not wait");
    Switches::Assign(RuntimeSwitch::V3eLockSplit, false);
    Expect(!LockOrder::MustNotWait(), "switch off: only the record lock counts");
    cache.unlock();
    LockOrder::SplitMutex<std::recursive_mutex, Rank::DepthDcc> recursive;
    Switches::Assign(RuntimeSwitch::V3eLockSplit, true);
    {
        std::lock_guard recordHold(record);
        std::lock_guard a(recursive);
        std::lock_guard b(recursive);
        Expect(LockOrder::HeldOnThisThread(Rank::DepthDcc) == 2, "recursive inner: cache lock re-entered");
    }
    Expect(!CacheLockFor(Rank::DepthDcc).HeldByThisThread(), "recursive inner released");
    LockOrder::SetLeafTakesForTest(false);
    {
        std::lock_guard recordHold(record);
        std::lock_guard hold(mutex);
        Expect(!cache.HeldByThisThread(), "stage mode: a leaf acquire takes no cache lock");
    }
    LockOrder::SetCheckForTest(true);
    {
        LockOrder::StageHold outside;
        Expect(!outside.Held(), "no stage hold without the record lock");
        record.lock();
        {
            LockOrder::StageHold stage;
            Expect(stage.Held(), "stage hold under the record lock");
            for (auto rank : LockOrder::StageRanks) Expect(CacheLockFor(rank).HeldExclusiveByThisThread(), std::string("stage holds ") + LockOrder::Name(rank));
            const auto suspends = LockOrder::StageSuspends();
            const auto releases = LockOrder::Totals().releasesHolding;
            record.unlock();
            Expect(!CacheLockFor(Rank::Texture).HeldByThisThread(), "a record release suspends the stage");
            Expect(LockOrder::StageSuspends() == suspends + 1, "suspend counted");
            Expect(LockOrder::Totals().releasesHolding == releases, "a suspended stage is no release-while-holding");
            bool otherGot = false;
            std::thread([&] {
                otherGot = CacheLockFor(Rank::Storage).try_lock();
                if (otherGot) CacheLockFor(Rank::Storage).unlock();
            }).join();
            Expect(otherGot, "another thread can take a suspended stage's cache lock");
            record.lock();
            Expect(CacheLockFor(Rank::DepthDcc).HeldExclusiveByThisThread(), "the next record acquire resumes the stage");
        }
        for (auto rank : LockOrder::StageRanks) Expect(!CacheLockFor(rank).HeldByThisThread(), std::string("stage released ") + LockOrder::Name(rank));
        record.unlock();
    }
    LockOrder::SetCheckForTest(false);
    Switches::Assign(RuntimeSwitch::V3eLockSplit, false);
}

// The phase (ii) shape in miniature: preparers read cache state outside the record lock under shared cache locks and
// note the versions they read; the recorder validates each item under the record lock in stream order and
// re-prepares on a mismatch; async workers and the recorder write cache state under the record lock and an exclusive
// cache lock. Every committed result must equal the one computed under the lock.
void Stress() {
    LockOrder::SetCheckForTest(true);
    constexpr std::size_t Items = 20000;
    constexpr std::size_t Keys = 256;
    constexpr std::array<Rank, 3> Caches{Rank::Texture, Rank::Storage, Rank::Pipeline};
    struct CacheState {
        std::array<std::uint64_t, Keys> value{};
        std::array<std::uint64_t, Keys> version{};
    };
    std::array<CacheState, 3> caches{};
    struct Item {
        std::array<std::uint32_t, 3> keys{};
        std::array<std::uint64_t, 3> versions{};
        std::uint64_t result = 0;
        std::atomic<bool> ready{false};
    };
    std::vector<Item> items(Items);
    for (std::size_t i = 0; i < Items; ++i) {
        for (std::size_t c = 0; c < 3; ++c) items[i].keys[c] = static_cast<std::uint32_t>((i * (7 + c * 5) + c) % Keys);
    }
    const auto compute = [&](Item& item, bool record) {
        std::uint64_t result = 0;
        for (std::size_t c = 0; c < 3; ++c) {
            std::shared_lock hold(CacheLockFor(Caches[c]));
            const auto key = item.keys[c];
            if (!record) item.versions[c] = caches[c].version[key];
            result = result * 1000003u + caches[c].value[key];
        }
        return result;
    };
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> committed{0};
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> writes{0};
    std::atomic<int> errors{0};
    const auto write = [&](std::uint64_t seed) {
        std::lock_guard record(GuestMemory::Record());
        const std::size_t first = seed % 3;
        const std::size_t last = first + (seed / 3) % (3 - first);
        std::vector<std::unique_lock<CacheLock>> holds;
        for (std::size_t c = first; c <= last; ++c) holds.emplace_back(CacheLockFor(Caches[c]));
        for (std::size_t c = first; c <= last; ++c) {
            const auto key = (seed >> 8) % Keys;
            caches[c].value[key] += seed | 1;
            ++caches[c].version[key];
        }
        writes.fetch_add(1, std::memory_order_relaxed);
    };
    std::vector<std::thread> threads;
    for (int p = 0; p < 4; ++p) {
        threads.emplace_back([&] {
            try {
                while (true) {
                    const auto i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= Items) return;
                    while (i >= committed.load(std::memory_order_acquire) + 64) std::this_thread::yield();
                    if (GuestMemory::Record().HeldByThisThread()) ++errors;
                    items[i].result = compute(items[i], false);
                    items[i].ready.store(true, std::memory_order_release);
                }
            } catch (const OrderError&) {
                ++errors;
            }
        });
    }
    for (int w = 0; w < 2; ++w) {
        threads.emplace_back([&, w] {
            std::uint64_t seed = 0x9e3779b97f4a7c15ull * (w + 1);
            try {
                while (!stop.load(std::memory_order_relaxed)) {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    write(seed);
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            } catch (const OrderError&) {
                ++errors;
            }
        });
    }
    std::uint64_t reprepared = 0;
    std::uint64_t valid = 0;
    std::uint64_t mismatches = 0;
    std::atomic<bool> done{false};
    std::thread watchdog([&] {
        const auto start = std::chrono::steady_clock::now();
        while (!done.load()) {
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(120)) {
                std::fprintf(stderr, "FAIL: stress deadlocked (no progress in 120 s)\n");
                std::_Exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });
    try {
        for (std::size_t i = 0; i < Items; ++i) {
            auto& item = items[i];
            while (!item.ready.load(std::memory_order_acquire)) std::this_thread::yield();
            std::lock_guard record(GuestMemory::Record());
            bool same = true;
            for (std::size_t c = 0; c < 3; ++c) {
                std::shared_lock hold(CacheLockFor(Caches[c]));
                same = same && caches[c].version[item.keys[c]] == item.versions[c];
            }
            const auto truth = compute(item, true);
            if (same) {
                ++valid;
                if (item.result != truth) ++mismatches;
            } else {
                ++reprepared;
                item.result = truth;
            }
            if (i % 64 == 0) write(i * 0x2545F4914F6CDD1Dull + 1);
            committed.store(i + 1, std::memory_order_release);
        }
    } catch (const OrderError&) {
        ++errors;
    }
    stop.store(true);
    for (auto& thread : threads) thread.join();
    done.store(true);
    watchdog.join();
    std::printf("stress: %zu items, %llu valid, %llu re-prepared, %llu writes, %llu contended\n", Items, static_cast<unsigned long long>(valid), static_cast<unsigned long long>(reprepared), static_cast<unsigned long long>(writes.load()), static_cast<unsigned long long>(LockOrder::Totals().contended));
    Expect(errors.load() == 0, "no lock-order error in the stress run");
    Expect(mismatches == 0, "validation caught every conflicting write");
    Expect(reprepared > 0, "the stream saw conflicting writes");
    Expect(valid > Items / 10, "most of the stream validated without a conflict");
    Expect(valid + reprepared == Items, "every item committed once");
    LockOrder::SetCheckForTest(false);
}


void Bench() {
    using Graphics::RuntimeSwitch;
    namespace Switches = Graphics::RuntimeSwitches;
    LockOrder::SplitMutex<AgcDriver::Mutex, Rank::Texture> mutex;
    std::lock_guard record(GuestMemory::Record());
    constexpr int Rounds = 2000000;
    LockOrder::SetLeafTakesForTest(true);
    for (const bool on : {false, true, false, true}) {
        Switches::Assign(RuntimeSwitch::V3eLockSplit, on);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < Rounds; ++i) {
            mutex.lock();
            mutex.unlock();
        }
        const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / Rounds;
        std::printf("bench: SplitMutex lock+unlock under the record lock, switch %s: %.1f ns\n", on ? "on" : "off", ns);
    }
    LockOrder::SetLeafTakesForTest(false);
    Switches::Assign(RuntimeSwitch::V3eLockSplit, true);
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < Rounds; ++i) {
        const LockOrder::StageHold stage;
    }
    const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / Rounds;
    std::printf("bench: StageHold (5 cache locks) per stage: %.1f ns%c", ns, 10);
    Switches::Assign(RuntimeSwitch::V3eLockSplit, false);
}

}

int main() {
    const auto start = LockOrder::Totals();
    RankOrder();
    RecordRecursion();
    SharedExclusive();
    Split();
    Stress();
    Bench();
    Expect(LockOrder::Totals().releasesHolding == start.releasesHolding + 1, "only the deliberate release-while-holding was counted");
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("lock order: all passed\n");
    return 0;
}
