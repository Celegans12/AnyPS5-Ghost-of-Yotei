#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DrawCacheEpochs;
using AgcDriver::DriverDetail::DrawCacheTable;
using AgcDriver::DriverDetail::DrawShapeRemembered;

constexpr std::uint64_t Alive = 0x616c697665ull;
constexpr std::uint64_t Dead = 0x64656164ull;
constexpr std::size_t Readers = 4;
constexpr std::size_t KeySpace = 8192;
constexpr std::size_t Held = 4096;
constexpr std::size_t Operations = 400000;

std::atomic<std::uint64_t> PayloadsBuilt{0};
std::atomic<std::uint64_t> PayloadsDestroyed{0};

struct Payload {
    std::uint64_t key;
    std::uint64_t generation;
    std::atomic<std::uint64_t> state{Alive};

    Payload(std::uint64_t ownKey, std::uint64_t ownGeneration) : key(ownKey), generation(ownGeneration) { PayloadsBuilt.fetch_add(1, std::memory_order_relaxed); }
    ~Payload() {
        state.store(Dead, std::memory_order_relaxed);
        PayloadsDestroyed.fetch_add(1, std::memory_order_relaxed);
    }
};

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::uint64_t Next(std::uint64_t& state) {
    state ^= state << 13u;
    state ^= state >> 7u;
    state ^= state << 17u;
    return state;
}

std::uint64_t KeyOf(std::uint64_t index) {
    return (index + 1) * 0x9e3779b97f4a7c15ull;
}

DrawShapeRemembered ShapeOf(std::uint64_t key, std::uint64_t generation) {
    DrawShapeRemembered remembered;
    remembered.count = static_cast<std::uint8_t>(1 + generation % remembered.keys.size());
    for (std::size_t rank = 0; rank < remembered.count; ++rank) remembered.keys[rank] = key ^ (generation * 0x100000001b3ull + rank);
    return remembered;
}

void CheckShape(std::uint64_t key, const DrawShapeRemembered& remembered) {
    Require(remembered.count >= 1 && remembered.count <= remembered.keys.size(), "a reader saw a shape with an impossible key count");
    const auto generation = (remembered.keys[0] ^ key) / 0x100000001b3ull;
    Require(remembered == ShapeOf(key, generation), "a reader saw a shape whose keys were not published together");
}

void CheckSingleThread() {
    DrawCacheTable<std::uint64_t> table(4);
    std::uint64_t value = 0;
    Require(!table.Find(1, value) && !table.Contains(1), "an empty table found a key");
    for (std::uint64_t i = 0; i < 1000; ++i) table.Publish(KeyOf(i), i);
    Require(table.Size() == 1000, "the table lost count of its inserts");
    for (std::uint64_t i = 0; i < 1000; ++i) Require(table.Find(KeyOf(i), value) && value == i, "the table lost an inserted key");
    for (std::uint64_t i = 0; i < 1000; i += 2) Require(table.Erase(KeyOf(i)), "the table could not erase a held key");
    Require(!table.Erase(KeyOf(0)), "the table erased a key twice");
    for (std::uint64_t i = 0; i < 1000; ++i) Require(table.Find(KeyOf(i), value) == (i % 2 == 1), "the table answers wrongly after erasures");
    for (std::uint64_t round = 0; round < 64; ++round) {
        for (std::uint64_t i = 0; i < 1000; i += 2) table.Publish(KeyOf(i), i + round);
        for (std::uint64_t i = 0; i < 1000; i += 2) Require(table.Erase(KeyOf(i)), "the table lost a republished key");
    }
    Require(table.Size() == 500, "the table lost count after churn");
    Require(table.Cells() <= 16384, "the table grew without bound under churn at a fixed size");
    std::size_t visited = 0;
    table.Each([&](std::uint64_t key, const std::uint64_t& held) {
        Require(KeyOf(held) == key && held % 2 == 1, "the table walk met a key it should not hold");
        ++visited;
    });
    Require(visited == 500, "the table walk did not meet every held key");
    table.Publish(KeyOf(1), 77);
    Require(table.Find(KeyOf(1), value) && value == 77 && table.Size() == 500, "the table did not replace a value in place");
}

void CheckReadersBesideWriter() {
    const auto builtBefore = PayloadsBuilt.load();
    const auto destroyedBefore = PayloadsDestroyed.load();
    {
        DrawCacheTable<std::shared_ptr<Payload>> entries;
        DrawCacheTable<DrawShapeRemembered> shapes;
        std::unordered_map<std::uint64_t, std::shared_ptr<Payload>> mirror;
        std::deque<std::uint64_t> order;
        std::atomic<bool> stop{false};
        std::atomic<std::uint64_t> found{0}, absent{0};
        std::array<std::exception_ptr, Readers> failures{};
        std::vector<std::thread> readers;
        for (std::size_t reader = 0; reader < Readers; ++reader) {
            readers.emplace_back([&, reader] {
                try {
                    std::uint64_t random = 0x1234567ull + reader * 977;
                    std::uint64_t hits = 0, misses = 0;
                    while (!stop.load(std::memory_order_relaxed)) {
                        const auto key = KeyOf(Next(random) % KeySpace);
                        std::shared_ptr<Payload> entry;
                        if (entries.Find(key, entry)) {
                            Require(entry != nullptr, "a reader found a key without its entry");
                            Require(entry->state.load(std::memory_order_relaxed) == Alive, "a reader holds an entry that was already destroyed");
                            Require(entry->key == key, "a reader found the entry of another key");
                            ++hits;
                        } else {
                            ++misses;
                        }
                        DrawShapeRemembered remembered;
                        if (shapes.Find(key, remembered)) CheckShape(key, remembered);
                        if (shapes.Contains(key) && (hits & 1023u) == 0) {
                            const DrawCacheEpochs::Reading reading;
                            if (entries.Find(key, entry)) Require(entry->key == key, "a nested reader found the entry of another key");
                        }
                    }
                    found.fetch_add(hits);
                    absent.fetch_add(misses);
                } catch (...) {
                    failures[reader] = std::current_exception();
                    stop.store(true);
                }
            });
        }
        std::uint64_t random = 0xfeedfaceull;
        std::uint64_t generation = 0;
        for (std::size_t operation = 0; operation < Operations && !stop.load(std::memory_order_relaxed); ++operation) {
            const auto key = KeyOf(Next(random) % KeySpace);
            const auto action = Next(random) % 8;
            if (action == 0) {
                if (mirror.erase(key) != 0) {
                    Require(entries.Erase(key), "the table did not hold a key its mirror held");
                    shapes.Erase(key);
                }
                continue;
            }
            auto payload = std::make_shared<Payload>(key, ++generation);
            if (!mirror.contains(key)) order.push_back(key);
            mirror[key] = payload;
            entries.Publish(key, std::move(payload));
            shapes.Publish(key, ShapeOf(key, generation));
            while (mirror.size() > Held) {
                const auto oldest = order.front();
                order.pop_front();
                if (mirror.erase(oldest) == 0) continue;
                Require(entries.Erase(oldest), "the table did not hold the oldest key of its mirror");
                shapes.Erase(oldest);
            }
        }
        stop.store(true);
        for (auto& reader : readers) reader.join();
        for (const auto& failure : failures) {
            if (failure) std::rethrow_exception(failure);
        }
        Require(found.load() != 0 && absent.load() != 0, "the readers did not both find and miss keys beside the writer");
        Require(entries.Size() == mirror.size(), "the table and its mirror hold different counts");
        for (std::uint64_t index = 0; index < KeySpace; ++index) {
            const auto key = KeyOf(index);
            std::shared_ptr<Payload> entry;
            const bool held = entries.Find(key, entry);
            const auto mirrored = mirror.find(key);
            Require(held == (mirrored != mirror.end()), "the table and its mirror disagree on holding a key");
            if (held) Require(entry == mirrored->second, "the table holds another entry than its mirror");
        }
        Require(entries.Rebuilds() != 0, "the churn never rebuilt the table: the test does not cover a rebuild beside readers");
        std::cout << "draw cache table: " << found.load() << " reads found, " << absent.load() << " absent beside " << Operations << " writes; " << entries.Rebuilds() << " rebuilds, " << entries.Cells() << " cells, " << DrawCacheEpochs::Released() << " retired objects released\n";
    }
    DrawCacheEpochs::Collect();
    Require(DrawCacheEpochs::Retired() == 0, "retired objects stay unreleased with no reader left");
    Require(PayloadsBuilt.load() - builtBefore == PayloadsDestroyed.load() - destroyedBefore, "an entry was never destroyed after the table and its readers were gone");
}

void CheckReaderHoldsRetired() {
    DrawCacheTable<std::shared_ptr<Payload>> entries;
    entries.Publish(7, std::make_shared<Payload>(7, 1));
    const auto destroyedBefore = PayloadsDestroyed.load();
    std::atomic<int> stage{0};
    std::thread reader([&] {
        const DrawCacheEpochs::Reading reading;
        stage.store(1);
        while (stage.load() != 2) std::this_thread::yield();
    });
    while (stage.load() != 1) std::this_thread::yield();
    Require(entries.Erase(7), "the table lost its only key");
    DrawCacheEpochs::Collect();
    Require(PayloadsDestroyed.load() == destroyedBefore, "an entry was destroyed while a reader that began before its removal was still reading");
    stage.store(2);
    reader.join();
    DrawCacheEpochs::Collect();
    Require(PayloadsDestroyed.load() == destroyedBefore + 1, "an entry stayed alive after its last reader left");
}

}

int main() {
    try {
        CheckSingleThread();
        CheckReaderHoldsRetired();
        CheckReadersBesideWriter();
    } catch (const std::exception& error) {
        std::cerr << "FATAL: draw cache lock-free table test failed: " << error.what() << "\n";
        return 1;
    }
    std::cout << "draw cache lock-free table test passed\n";
    return 0;
}
