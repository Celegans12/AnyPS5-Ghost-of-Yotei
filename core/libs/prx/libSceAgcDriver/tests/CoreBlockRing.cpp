#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
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

constexpr std::uint64_t DeviceBase = 0x100000008ull;
constexpr std::uint64_t DeviceStride = 0x1000000ull;
constexpr std::uint64_t Everything = std::numeric_limits<std::uint64_t>::max();

struct FakeSource {
    std::size_t limit = BlockRing::MaxChunks;
    std::size_t allocations = 0;
    std::vector<std::pair<std::byte*, std::uint64_t>> chunks;
    std::atomic<std::uint64_t> executedThrough{0};
    std::atomic<std::uint64_t> asked{0};
};

RingChunk allocateChunk(void* context, std::size_t bytes) {
    auto& fake = *static_cast<FakeSource*>(context);
    if (fake.allocations >= fake.limit) return {};
    void* memory = std::malloc(bytes);
    if (memory == nullptr) return {};
    RingChunk chunk;
    chunk.bytes = static_cast<std::byte*>(memory);
    chunk.deviceAddress = DeviceBase + DeviceStride * fake.allocations;
    chunk.owner = std::shared_ptr<void>(memory, [](void* pointer) { std::free(pointer); });
    ++fake.allocations;
    fake.chunks.emplace_back(chunk.bytes, chunk.deviceAddress);
    return chunk;
}

bool executedSerial(void* context, std::uint64_t serial) {
    auto& fake = *static_cast<FakeSource*>(context);
    fake.asked.store(serial, std::memory_order_relaxed);
    return serial <= fake.executedThrough.load(std::memory_order_acquire);
}

RingSource sourceOf(FakeSource& fake) {
    RingSource source;
    source.allocate = &allocateChunk;
    source.executed = &executedSerial;
    source.context = &fake;
    return source;
}

bool consistent(const FakeSource& fake, const RingSlice& slice) {
    const auto address = reinterpret_cast<std::uintptr_t>(slice.bytes);
    for (const auto& [host, device] : fake.chunks) {
        const auto base = reinterpret_cast<std::uintptr_t>(host);
        if (address < base || address - base >= BlockRing::ChunkBytes) continue;
        return slice.deviceAddress - device == address - base;
    }
    return false;
}

void testAlignment() {
    FakeSource fake;
    BlockRing ring(sourceOf(fake));
    const auto first = ring.Take(3, 1);
    Expect(first.bytes != nullptr && first.deviceAddress == DeviceBase && consistent(fake, first), "the first slice starts the first chunk");
    const auto aligned = ring.Take(16, 16);
    Expect(aligned.bytes != nullptr && aligned.deviceAddress % 16 == 0, "a section is aligned on its device address");
    Expect(aligned.deviceAddress == DeviceBase + 8 && aligned.chunk == first.chunk, "the alignment skips only the padding");
    Expect(aligned.bytes - first.bytes == 8 && consistent(fake, aligned), "the host pointer moves with the device address");
    const auto odd = ring.Take(4, 12);
    Expect(odd.bytes != nullptr && odd.deviceAddress % 12 == 0 && odd.deviceAddress >= aligned.deviceAddress + 16 && odd.deviceAddress < aligned.deviceAddress + 16 + 12 && consistent(fake, odd), "an alignment that is not a power of two");
    const auto packed = ring.Take(5, 0);
    Expect(packed.bytes != nullptr && packed.deviceAddress == odd.deviceAddress + 4 && consistent(fake, packed), "alignment 0 packs like alignment 1");
    const auto empty = ring.Take(0, 16);
    Expect(empty.bytes != nullptr && empty.deviceAddress % 16 == 0 && empty.deviceAddress >= packed.deviceAddress + 5 && consistent(fake, empty), "a section of no bytes still gets an aligned address");
    const auto before = ring.Counted();
    Expect(before.chunks == 1 && before.laps == 0 && before.full == 0 && before.bytes == 3 + 16 + 4 + 5, "taken bytes are counted without the padding");
    Expect(ring.Take(BlockRing::ChunkBytes + 1, 1).bytes == nullptr, "a slice larger than a chunk is refused");
    Expect(ring.Take(16, BlockRing::ChunkBytes * 2).bytes == nullptr, "an alignment larger than a chunk is refused");
    const auto after = ring.Counted();
    Expect(after.full == 2 && after.chunks == 1 && after.bytes == before.bytes && fake.allocations == 1, "a refused size counts as full and allocates nothing");
    const auto again = ring.Take(8, 8);
    Expect(again.bytes != nullptr && again.chunk == first.chunk && again.deviceAddress % 8 == 0 && again.deviceAddress >= empty.deviceAddress && consistent(fake, again), "a refusal leaves the chunk in use");
}

void testNextChunk() {
    FakeSource fake;
    BlockRing ring(sourceOf(fake));
    const auto head = ring.Take(BlockRing::ChunkBytes - 32, 1);
    ring.Posted(head);
    const auto tail = ring.Take(32, 8);
    Expect(tail.bytes != nullptr && tail.chunk == head.chunk && tail.deviceAddress == DeviceBase + BlockRing::ChunkBytes - 32, "a slice that ends at the chunk end stays in the chunk");
    ring.Posted(tail);
    const auto next = ring.Take(16, 16);
    Expect(next.bytes != nullptr && next.chunk != head.chunk && consistent(fake, next), "a slice that does not fit moves to another chunk");
    Expect(next.deviceAddress == DeviceBase + DeviceStride + 8, "the next chunk starts at its first aligned byte");
    const auto counts = ring.Counted();
    Expect(counts.chunks == 2 && counts.laps == 0 && fake.allocations == 2, "a held chunk makes the ring grow");
    ring.Posted(next);
    const auto follow = ring.Take(8, 1);
    Expect(follow.chunk == next.chunk && follow.deviceAddress == next.deviceAddress + 16, "the following slices continue in the new chunk");
    Expect(ring.Live(head) && ring.Live(tail) && ring.Live(next) && ring.Live(follow), "slices stay live while their chunk is not reused");
}

void testHeldInFlightThenByBatch() {
    FakeSource fake;
    fake.executedThrough.store(Everything);
    BlockRing ring(sourceOf(fake));
    const auto a = ring.Take(BlockRing::ChunkBytes, 1);
    ring.Posted(a);
    const auto b = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(a.bytes != nullptr && b.bytes != nullptr && b.chunk != a.chunk && ring.Counted().chunks == 2, "a chunk with a posted record not yet emitted is held even when every batch has run");
    ring.Posted(b);
    ring.Emitted(b, 1);
    ring.Posted(a);
    ring.Emitted(a, 4);
    ring.Emitted(a, 2);
    fake.executedThrough.store(3);
    const auto c = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(c.bytes != nullptr && c.chunk != a.chunk && c.chunk != b.chunk && ring.Counted().chunks == 3, "a chunk whose last batch has not run is held");
    Expect(fake.asked.load() == 4, "the last batch of a chunk is the largest serial emitted for it");
    Expect(ring.Live(a) && ring.Live(b), "held chunks keep their slices live");
    ring.Posted(c);
    ring.Emitted(c, 5);
    fake.executedThrough.store(4);
    const auto d = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(d.bytes == a.bytes && d.deviceAddress == a.deviceAddress && d.chunk == a.chunk, "a chunk is reused once its records are emitted and its batch has run");
    Expect(d.lap == a.lap + 1 && !ring.Live(a) && ring.Live(d), "a reused chunk has a new lap and its old slices are dead");
    ring.Posted(d);
    ring.Emitted(d, 6);
    const auto e = ring.Take(16, 16);
    Expect(e.chunk == b.chunk && e.lap == b.lap + 1 && !ring.Live(b) && ring.Live(c), "the oldest chunk comes next");
    const auto counts = ring.Counted();
    Expect(counts.chunks == 3 && counts.laps == 2 && counts.full == 0, "reuses are counted as laps");
}

void testOpenRecord() {
    {
        FakeSource fake;
        fake.executedThrough.store(Everything);
        BlockRing ring(sourceOf(fake));
        const auto section = ring.Take(BlockRing::ChunkBytes - 8, 1);
        const auto copy = ring.Take(64, 16);
        Expect(copy.bytes != nullptr && copy.chunk != section.chunk && ring.Live(section), "a chunk holding a slice of the record being written is not reused");
        ring.Posted(section);
        ring.Posted(copy);
        ring.Emitted(section, 1);
        ring.Emitted(copy, 1);
        const auto next = ring.Take(BlockRing::ChunkBytes, 1);
        Expect(next.chunk == section.chunk && next.lap == section.lap + 1 && !ring.Live(section) && ring.Live(copy), "the chunk is reused once the record is posted, emitted and run");
    }
    {
        FakeSource fake;
        fake.limit = 1;
        fake.executedThrough.store(Everything);
        BlockRing ring(sourceOf(fake));
        const auto lost = ring.Take(BlockRing::ChunkBytes - 8, 1);
        const auto refused = ring.Take(16, 16);
        Expect(lost.bytes != nullptr && refused.bytes == nullptr && ring.Live(lost), "the only chunk is not reused under the record being written");
        const auto again = ring.Take(16, 16);
        Expect(again.bytes != nullptr && again.chunk == lost.chunk && again.lap == lost.lap + 1 && !ring.Live(lost), "an empty slice abandons the record and its lost slices hold nothing");
    }
    {
        FakeSource fake;
        fake.limit = 1;
        fake.executedThrough.store(Everything);
        BlockRing ring(sourceOf(fake));
        const auto first = ring.Take(BlockRing::ChunkBytes - 8, 1);
        ring.Posted(first);
        ring.Emitted(first, 1);
        const auto second = ring.Take(16, 16);
        Expect(second.bytes != nullptr && second.chunk == first.chunk && second.lap == first.lap + 1, "a posted record closes its hold on the current chunk");
    }
}

void testFullRing() {
    FakeSource fake;
    BlockRing ring(sourceOf(fake));
    std::vector<RingSlice> held;
    for (std::size_t index = 0; index < BlockRing::MaxChunks; ++index) {
        const auto slice = ring.Take(BlockRing::ChunkBytes, 1);
        if (slice.bytes == nullptr) break;
        ring.Posted(slice);
        held.push_back(slice);
    }
    Expect(held.size() == BlockRing::MaxChunks && ring.Counted().chunks == BlockRing::MaxChunks, "the ring grows to 64 chunks");
    const auto refused = ring.Take(16, 16);
    Expect(refused.bytes == nullptr && refused.deviceAddress == 0 && !ring.Live(refused), "64 chunks held in flight give an empty slice");
    Expect(ring.Counted().full == 1 && ring.Counted().chunks == BlockRing::MaxChunks && fake.allocations == BlockRing::MaxChunks, "a full ring is counted and allocates no 65th chunk");
    if (held.size() != BlockRing::MaxChunks) return;
    for (std::size_t index = 1; index < held.size(); ++index) ring.Emitted(held[index], index + 1);
    Expect(ring.Take(16, 16).bytes == nullptr && ring.Counted().full == 2, "64 chunks held by batches not yet run give an empty slice");
    fake.executedThrough.store(11);
    const auto reused = ring.Take(16, 16);
    Expect(reused.bytes != nullptr && reused.chunk == held[1].chunk && reused.lap == held[1].lap + 1, "with 64 chunks the first free chunk after a held one is reused");
    Expect(!ring.Live(held[1]) && ring.Live(held[0]) && ring.Live(held[2]) && ring.Counted().chunks == BlockRing::MaxChunks, "only the reused chunk loses its slices");
    ring.Posted(reused);
    ring.Emitted(reused, 12);
    ring.Emitted(held[0], 1);
    const auto oldest = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(oldest.chunk == held[0].chunk && oldest.lap == held[0].lap + 1 && ring.Counted().laps == 2, "the oldest chunk comes next once it is free");
}

void testRefusedAllocation() {
    {
        BlockRing ring(RingSource{});
        Expect(ring.Take(16, 16).bytes == nullptr && ring.Counted().full == 1 && ring.Counted().chunks == 0, "a source without allocation gives empty slices");
    }
    FakeSource fake;
    fake.limit = 2;
    BlockRing ring(sourceOf(fake));
    const auto a = ring.Take(BlockRing::ChunkBytes, 1);
    ring.Posted(a);
    const auto b = ring.Take(BlockRing::ChunkBytes, 1);
    ring.Posted(b);
    const auto refused = ring.Take(16, 16);
    Expect(refused.bytes == nullptr && ring.Counted().chunks == 2 && ring.Counted().full == 1 && fake.allocations == 2, "a refused allocation gives an empty slice");
    ring.Emitted(b, 1);
    fake.executedThrough.store(1);
    Expect(!ring.Live(RingSlice{}), "an empty slice is never live");
    ring.Posted(RingSlice{});
    ring.Emitted(RingSlice{}, 99);
    const auto reused = ring.Take(16, 16);
    Expect(reused.chunk == b.chunk && reused.lap == b.lap + 1 && ring.Live(a), "the chunk being left is reused when it is the only free one");
    ring.Posted(reused);
    ring.Emitted(reused, 1);
    ring.Emitted(a, 1);
    const auto oldest = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(oldest.chunk == a.chunk && oldest.lap == a.lap + 1, "posting and emitting an empty slice changed no chunk");
}

struct Record {
    RingSlice slice;
    std::uint64_t id = 0;
    std::size_t bytes = 0;
};

void stamp(const Record& record) {
    std::memset(record.slice.bytes, static_cast<int>(record.id & 0xffu), record.bytes);
    std::memcpy(record.slice.bytes, &record.id, sizeof(record.id));
}

bool intact(const Record& record, std::vector<std::byte>& pattern) {
    pattern.assign(record.bytes, static_cast<std::byte>(record.id & 0xffu));
    std::memcpy(pattern.data(), &record.id, sizeof(record.id));
    return std::memcmp(record.slice.bytes, pattern.data(), record.bytes) == 0;
}

void testTwoThreads() {
    constexpr std::uint64_t records = 20000;
    constexpr std::uint64_t perBatch = 16;
    constexpr std::uint64_t lag = 3;
    constexpr std::size_t largest = 32 * 1024;
    FakeSource fake;
    fake.limit = 8;
    BlockRing ring(sourceOf(fake));
    std::mutex lock;
    std::condition_variable wake;
    std::deque<Record> queue;
    bool done = false;
    std::uint64_t refused = 0;
    std::uint64_t taken = 0;
    std::uint64_t reposted = 0;
    std::uint64_t emitted = 0;
    std::uint64_t damaged = 0;
    std::uint64_t dead = 0;
    std::thread writer([&] {
        std::array<Record, 64> recent{};
        std::uint64_t seed = 0x9E3779B97F4A7C15ull;
        for (std::uint64_t id = 1; id <= records; ++id) {
            Record record;
            const auto& old = recent[(id + 24) % recent.size()];
            if (id % 5 == 0 && ring.Live(old.slice)) {
                record = old;
                ++reposted;
            } else {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                record.id = id;
                record.bytes = 16 + static_cast<std::size_t>((seed >> 33) % largest);
                record.slice = ring.Take(record.bytes, 16);
                while (record.slice.bytes == nullptr) {
                    ++refused;
                    std::this_thread::yield();
                    record.slice = ring.Take(record.bytes, 16);
                }
                taken += record.bytes;
                stamp(record);
                recent[id % recent.size()] = record;
            }
            ring.Posted(record.slice);
            {
                std::lock_guard guard(lock);
                queue.push_back(record);
            }
            wake.notify_one();
        }
        std::lock_guard guard(lock);
        done = true;
        wake.notify_one();
    });
    std::thread emitter([&] {
        std::deque<std::pair<std::uint64_t, Record>> running;
        std::vector<std::byte> pattern;
        std::uint64_t serial = 1;
        std::uint64_t ran = 0;
        std::uint64_t inBatch = 0;
        const auto run = [&](std::uint64_t through) {
            if (through <= ran) return;
            while (!running.empty() && running.front().first <= through) {
                if (!intact(running.front().second, pattern)) ++damaged;
                running.pop_front();
            }
            ran = through;
            fake.executedThrough.store(through, std::memory_order_release);
        };
        std::unique_lock guard(lock);
        for (;;) {
            if (queue.empty()) {
                if (done) break;
                guard.unlock();
                run(serial);
                ++serial;
                inBatch = 0;
                guard.lock();
                wake.wait(guard, [&] { return !queue.empty() || done; });
                continue;
            }
            const auto record = queue.front();
            queue.pop_front();
            guard.unlock();
            if (!ring.Live(record.slice)) ++dead;
            if (!intact(record, pattern)) ++damaged;
            ring.Emitted(record.slice, serial);
            running.emplace_back(serial, record);
            ++emitted;
            if (++inBatch == perBatch) {
                inBatch = 0;
                if (serial > lag) run(serial - lag);
                ++serial;
            }
            guard.lock();
        }
        guard.unlock();
        run(serial);
    });
    writer.join();
    emitter.join();
    Expect(emitted == records, "every posted record reaches the emitter");
    Expect(damaged == 0, "no slice is overwritten before its batch has run");
    Expect(dead == 0, "a posted slice stays live until it is emitted");
    Expect(reposted > 0, "live slices are posted again by later records");
    const auto counts = ring.Counted();
    Expect(counts.laps > 0 && counts.chunks <= 8 && fake.allocations <= 8, "the writer wraps around a bounded ring");
    Expect(counts.full == refused && counts.bytes == taken, "empty slices and taken bytes are counted across threads");
}

}

int main() {
    testAlignment();
    testNextChunk();
    testHeldInFlightThenByBatch();
    testOpenRecord();
    testFullRing();
    testRefusedAllocation();
    testTwoThreads();
    if (failures != 0) {
        std::fprintf(stderr, "%d block ring checks failed\n", failures);
        return 1;
    }
    std::printf("block ring tests passed\n");
    return 0;
}
