#include "prx/libSceAgcDriver/Core/include/FrontMemo.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
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
constexpr std::uint32_t RangeBytes = 64;

using Words = std::array<std::uint32_t, RangeBytes / 4u>;

struct FakeSource {
    std::size_t limit = BlockRing::MaxChunks;
    std::size_t allocations = 0;
    std::uint64_t executedThrough = 0;
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
    return chunk;
}

bool executedSerial(void* context, std::uint64_t serial) {
    return serial <= static_cast<const FakeSource*>(context)->executedThrough;
}

RingSource sourceOf(FakeSource& fake) {
    RingSource source;
    source.allocate = &allocateChunk;
    source.executed = &executedSerial;
    source.context = &fake;
    return source;
}

struct Fixture {
    explicit Fixture(std::size_t limit) : ring(sourceOf(fake)) { fake.limit = limit; }
    FakeSource fake;
    BlockRing ring;
};

Words rangeOf(std::uint32_t seed) {
    Words words{};
    for (std::size_t index = 0; index < words.size(); ++index) words[index] = seed * 0x01000193u + static_cast<std::uint32_t>(index);
    return words;
}

MemoKey keyOf(std::uint64_t address, const Words& words) {
    return MemoKey{.address = address, .bytes = RangeBytes, .first = words[0]};
}

RingSlice copyInto(BlockRing& ring, const Words& words) {
    const auto slice = ring.Take(RangeBytes, 16);
    if (slice.bytes != nullptr) std::memcpy(slice.bytes, words.data(), RangeBytes);
    return slice;
}

bool holds(const RingSlice& slice, const Words& words) {
    return slice.bytes != nullptr && std::memcmp(slice.bytes, words.data(), RangeBytes) == 0;
}

bool same(const RingSlice& left, const RingSlice& right) {
    return left.bytes == right.bytes && left.deviceAddress == right.deviceAddress && left.chunk == right.chunk && left.lap == right.lap;
}

RingSlice sentinel() {
    RingSlice slice;
    slice.deviceAddress = 0xdead0000ull;
    slice.chunk = 77;
    slice.lap = 99;
    return slice;
}

bool sameCounts(const FrontMemo::Counts& counts, std::uint64_t finds, std::uint64_t hits, std::uint64_t stale, std::uint64_t dead) {
    return counts.finds == finds && counts.hits == hits && counts.stale == stale && counts.dead == dead;
}

std::vector<MemoKey> spread(const MemoKey& base, std::size_t count, const BlockRing& ring, const RingSlice& live) {
    std::vector<MemoKey> keys;
    for (std::uint64_t step = 0; keys.size() < count && step < 4096; ++step) {
        auto candidate = base;
        candidate.address = base.address + step * 0x100ull;
        FrontMemo probe;
        for (const auto& key : keys) probe.Keep(key, 1, live);
        probe.Keep(candidate, 1, live);
        bool apart = true;
        RingSlice found;
        for (const auto& key : keys) apart = probe.Find(key, 1, ring, found) && apart;
        if (apart) keys.push_back(candidate);
    }
    return keys;
}

void testHit() {
    Fixture fixture(BlockRing::MaxChunks);
    auto& ring = fixture.ring;
    FrontMemo memo;
    Expect(sameCounts(memo.Counted(), 0, 0, 0, 0), "a new memo counts nothing");
    const auto words = rangeOf(1);
    const auto key = keyOf(0x200010000ull, words);
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    memo.Keep(key, 5, copy);
    auto found = sentinel();
    Expect(memo.Find(key, 5, ring, found), "a continued draw finds its copy: same key, same first word, same epoch");
    Expect(same(found, copy) && holds(found, words), "the hit gives back the kept slice and its copied bytes");
    auto again = sentinel();
    Expect(memo.Find(key, 5, ring, again) && same(again, copy), "a hit leaves the entry in place for the next draw");
    Expect(sameCounts(memo.Counted(), 2, 2, 0, 0), "two finds, two hits");
}

void testMissByKey() {
    Fixture fixture(BlockRing::MaxChunks);
    auto& ring = fixture.ring;
    {
        FrontMemo empty;
        auto found = sentinel();
        Expect(!empty.Find(MemoKey{}, 0, ring, found) && same(found, sentinel()), "an empty slot misses even for the zero key at epoch 0");
        Expect(sameCounts(empty.Counted(), 1, 0, 0, 0), "an empty slot is a plain miss");
    }
    FrontMemo memo;
    const auto words = rangeOf(2);
    const auto key = keyOf(0x200020000ull, words);
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    memo.Keep(key, 1, copy);
    auto moved = key;
    moved.address += 0x40u;
    auto found = sentinel();
    Expect(!memo.Find(moved, 1, ring, found) && same(found, sentinel()), "another address misses and leaves the slice untouched");
    auto shorter = key;
    shorter.bytes = RangeBytes - 4u;
    Expect(!memo.Find(shorter, 1, ring, found) && same(found, sentinel()), "a shorter range at the same address misses");
    auto wider = key;
    wider.bytes = RangeBytes * 2u;
    Expect(!memo.Find(wider, 1, ring, found) && same(found, sentinel()), "a wider range at the same address misses");
    Expect(sameCounts(memo.Counted(), 3, 0, 0, 0), "a key miss is neither stale nor dead");
    Expect(memo.Find(key, 1, ring, found) && same(found, copy), "the kept entry still hits after the misses");
}

void testMissByFirst() {
    Fixture fixture(BlockRing::MaxChunks);
    auto& ring = fixture.ring;
    FrontMemo memo;
    const auto words = rangeOf(3);
    const auto key = keyOf(0x200030000ull, words);
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    memo.Keep(key, 4, copy);
    auto rewritten = words;
    rewritten[0] ^= 0x1u;
    const auto changed = keyOf(key.address, rewritten);
    auto found = sentinel();
    Expect(!memo.Find(changed, 4, ring, found) && same(found, sentinel()), "a range whose first word changed misses");
    Expect(sameCounts(memo.Counted(), 1, 0, 0, 0), "a first word miss is neither stale nor dead");
    Expect(memo.Find(key, 4, ring, found) && same(found, copy), "the old first word still finds its entry");
}

void testMissByEpoch() {
    Fixture fixture(BlockRing::MaxChunks);
    auto& ring = fixture.ring;
    FrontMemo memo;
    const auto words = rangeOf(4);
    const auto key = keyOf(0x200040000ull, words);
    const auto before = copyInto(ring, words);
    ring.Posted(before);
    memo.Keep(key, 10, before);
    auto found = sentinel();
    Expect(!memo.Find(key, 11, ring, found) && same(found, sentinel()), "the same range after the front epoch moved misses");
    Expect(!memo.Find(key, 9, ring, found) && same(found, sentinel()), "any other epoch misses, an earlier one too");
    Expect(sameCounts(memo.Counted(), 2, 0, 2, 0), "an epoch miss is counted stale");
    const auto after = copyInto(ring, words);
    ring.Posted(after);
    memo.Keep(key, 11, after);
    Expect(memo.Find(key, 11, ring, found) && same(found, after) && !same(after, before), "a new copy kept in the new epoch is found");
    found = sentinel();
    Expect(!memo.Find(key, 10, ring, found) && same(found, sentinel()), "the old epoch no longer hits once the entry is renewed");
    Expect(sameCounts(memo.Counted(), 4, 1, 3, 0), "one hit, three stale");
}

void testDeadAfterLap() {
    Fixture fixture(1);
    auto& ring = fixture.ring;
    FrontMemo memo;
    const auto words = rangeOf(5);
    const auto key = keyOf(0x200050000ull, words);
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    memo.Keep(key, 2, copy);
    ring.Emitted(copy, 1);
    fixture.fake.executedThrough = 1;
    auto found = sentinel();
    Expect(memo.Find(key, 2, ring, found) && same(found, copy), "the next draw of the run finds the copy");
    ring.Posted(found);
    Expect(ring.Take(BlockRing::ChunkBytes, 1).bytes == nullptr, "the chunk stays held while the draw that reused the copy is not emitted, though the first batch has run");
    Expect(memo.Find(key, 2, ring, found) && holds(found, words), "the entry lives while its chunk is held");
    ring.Emitted(found, 2);
    Expect(ring.Take(BlockRing::ChunkBytes, 1).bytes == nullptr, "the chunk stays held until the batch of the last draw that read the copy has run");
    Expect(memo.Find(key, 2, ring, found) && holds(found, words), "the entry lives and its bytes are intact until that batch has run");
    fixture.fake.executedThrough = 2;
    const auto reused = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(reused.bytes != nullptr && reused.chunk == copy.chunk && reused.lap == copy.lap + 1, "once every batch that read the chunk has run it makes a lap");
    if (reused.bytes != nullptr) std::memset(reused.bytes, 0xee, 256);
    Expect(!holds(copy, words), "the new owner of the chunk overwrites the old copy");
    found = sentinel();
    Expect(!memo.Find(key, 2, ring, found) && same(found, sentinel()), "an entry whose chunk made a lap is dead");
    Expect(sameCounts(memo.Counted(), 4, 3, 0, 1), "three hits then one dead");
    ring.Posted(reused);
    ring.Emitted(reused, 3);
    fixture.fake.executedThrough = 3;
    const auto renewed = copyInto(ring, words);
    ring.Posted(renewed);
    memo.Keep(key, 2, renewed);
    Expect(renewed.bytes != nullptr && renewed.lap == copy.lap + 2 && memo.Find(key, 2, ring, found) && same(found, renewed) && holds(found, words), "a new copy in the next lap takes the place of the dead entry");
    Expect(sameCounts(memo.Counted(), 5, 4, 0, 1), "the renewed entry hits");
}

void testLapInsideTheDraw() {
    Fixture fixture(1);
    auto& ring = fixture.ring;
    FrontMemo memo;
    const auto words = rangeOf(8);
    const auto key = keyOf(0x200080000ull, words);
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    memo.Keep(key, 1, copy);
    ring.Emitted(copy, 1);
    fixture.fake.executedThrough = 1;
    RingSlice found;
    Expect(memo.Find(key, 1, ring, found) && same(found, copy), "the next draw finds the copy");
    const auto section = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(section.bytes != nullptr && section.chunk == copy.chunk && section.lap == copy.lap + 1, "a later Take of the same draw can lap the chunk of the hit");
    Expect(!ring.Live(found), "the slice given by the hit is then dead: the caller checks it again before posting");
    Expect(!memo.Find(key, 1, ring, found), "the memo sees the entry dead");
    Expect(sameCounts(memo.Counted(), 2, 1, 0, 1), "one hit then one dead");
    ring.Posted(section);
}

void testOnlyTheLappedChunkDies() {
    Fixture fixture(2);
    auto& ring = fixture.ring;
    const auto first = ring.Take(BlockRing::ChunkBytes, 1);
    ring.Posted(first);
    ring.Emitted(first, 1);
    const auto second = ring.Take(BlockRing::ChunkBytes, 1);
    ring.Posted(second);
    ring.Emitted(second, 2);
    Expect(first.bytes != nullptr && second.bytes != nullptr && second.chunk != first.chunk, "two chunks hold two copies");
    const auto keys = spread(MemoKey{.address = 0x2000a0000ull, .bytes = RangeBytes, .first = 0x1234u}, 2, ring, first);
    Expect(keys.size() == 2, "two keys in distinct slots");
    if (keys.size() != 2) return;
    FrontMemo memo;
    memo.Keep(keys[0], 6, first);
    memo.Keep(keys[1], 6, second);
    RingSlice found;
    Expect(memo.Find(keys[0], 6, ring, found) && same(found, first), "the entry in the first chunk hits before any lap");
    Expect(memo.Find(keys[1], 6, ring, found) && same(found, second), "the entry in the second chunk hits before any lap");
    fixture.fake.executedThrough = 1;
    const auto next = ring.Take(16, 16);
    Expect(next.bytes != nullptr && next.chunk == first.chunk && next.lap == first.lap + 1, "the chunk whose batch has run makes a lap");
    Expect(!memo.Find(keys[0], 6, ring, found), "the entry in the lapped chunk is dead");
    Expect(memo.Find(keys[1], 6, ring, found) && same(found, second), "the entry in the held chunk still hits");
    Expect(sameCounts(memo.Counted(), 4, 3, 0, 1), "three hits, one dead");
    ring.Posted(next);
}

void testReplace() {
    Fixture fixture(BlockRing::MaxChunks);
    auto& ring = fixture.ring;
    const auto wordsA = rangeOf(6);
    const auto wordsB = rangeOf(7);
    const auto keyA = keyOf(0x200060000ull, wordsA);
    const auto keyB = keyOf(0x200070000ull, wordsB);
    const auto copyA = copyInto(ring, wordsA);
    const auto copyB = copyInto(ring, wordsB);
    ring.Posted(copyA);
    ring.Posted(copyB);
    {
        FrontMemo memo(1);
        memo.Keep(keyA, 1, copyA);
        memo.Keep(keyB, 1, copyB);
        auto found = sentinel();
        Expect(!memo.Find(keyA, 1, ring, found) && same(found, sentinel()), "with one slot a second key replaces the first");
        Expect(memo.Find(keyB, 1, ring, found) && same(found, copyB) && holds(found, wordsB), "the replacing entry is found");
        memo.Keep(keyA, 1, copyA);
        found = sentinel();
        Expect(!memo.Find(keyB, 1, ring, found) && same(found, sentinel()), "keeping the first key again replaces the second");
        Expect(memo.Find(keyA, 1, ring, found) && same(found, copyA) && holds(found, wordsA), "the first key is found again");
        Expect(sameCounts(memo.Counted(), 4, 2, 0, 0), "a replaced entry is a plain miss");
    }
    {
        FrontMemo memo(0);
        memo.Keep(keyA, 1, copyA);
        RingSlice found;
        Expect(memo.Find(keyA, 1, ring, found) && same(found, copyA), "a memo asked for no slot still keeps one");
        memo.Keep(keyB, 1, copyB);
        Expect(!memo.Find(keyA, 1, ring, found) && memo.Find(keyB, 1, ring, found) && same(found, copyB), "a memo asked for no slot has exactly one");
    }
    {
        FrontMemo memo;
        auto rewritten = wordsA;
        rewritten[0] += 1u;
        const auto changed = keyOf(keyA.address, rewritten);
        const auto copyChanged = copyInto(ring, rewritten);
        ring.Posted(copyChanged);
        memo.Keep(keyA, 1, copyA);
        memo.Keep(changed, 1, copyChanged);
        auto found = sentinel();
        Expect(!memo.Find(keyA, 1, ring, found) && same(found, sentinel()), "a new copy of the same range takes the slot of the old one");
        Expect(memo.Find(changed, 1, ring, found) && same(found, copyChanged) && holds(found, rewritten), "the new copy of the range is found");
    }
    {
        FrontMemo memo;
        memo.Keep(keyA, 1, copyA);
        memo.Keep(keyA, 2, copyB);
        auto found = sentinel();
        Expect(!memo.Find(keyA, 1, ring, found) && same(found, sentinel()), "keeping the same key again replaces its epoch");
        Expect(memo.Find(keyA, 2, ring, found) && same(found, copyB), "keeping the same key again replaces its slice");
        Expect(sameCounts(memo.Counted(), 2, 1, 1, 0), "the replaced epoch is counted stale");
    }
}

void testClear() {
    Fixture fixture(BlockRing::MaxChunks);
    auto& ring = fixture.ring;
    const auto words = rangeOf(10);
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    const auto keys = spread(keyOf(0x2000b0000ull, words), 3, ring, copy);
    Expect(keys.size() == 3, "three keys in distinct slots");
    if (keys.size() != 3) return;
    FrontMemo memo;
    for (const auto& key : keys) memo.Keep(key, 1, copy);
    RingSlice found;
    bool all = true;
    for (const auto& key : keys) all = memo.Find(key, 1, ring, found) && all;
    Expect(all, "every kept entry hits before the clear");
    memo.Clear();
    bool none = true;
    for (const auto& key : keys) none = !memo.Find(key, 1, ring, found) && none;
    Expect(none, "a cleared memo finds nothing");
    Expect(sameCounts(memo.Counted(), 6, 3, 0, 0), "a cleared entry is a plain miss and the counts survive the clear");
    Expect(!memo.Find(MemoKey{}, 0, ring, found), "a cleared slot does not match the zero key");
    memo.Keep(keys[0], 1, copy);
    Expect(memo.Find(keys[0], 1, ring, found) && same(found, copy), "the memo keeps entries again after a clear");
    Expect(sameCounts(memo.Counted(), 8, 4, 0, 0), "finds and hits go on after a clear");
}

void testCounts() {
    Fixture fixture(1);
    auto& ring = fixture.ring;
    fixture.fake.executedThrough = Everything;
    FrontMemo memo;
    const auto words = rangeOf(9);
    const auto key = keyOf(0x200090000ull, words);
    RingSlice found;
    Expect(!memo.Find(key, 3, ring, found), "nothing is found before a keep");
    const auto copy = copyInto(ring, words);
    ring.Posted(copy);
    ring.Emitted(copy, 1);
    memo.Keep(key, 3, copy);
    Expect(memo.Find(key, 3, ring, found), "the first continued draw hits");
    Expect(memo.Find(key, 3, ring, found), "the second continued draw hits");
    auto other = key;
    other.first ^= 0xffu;
    Expect(!memo.Find(other, 3, ring, found), "another first word misses");
    Expect(!memo.Find(key, 4, ring, found), "another epoch misses");
    const auto lapped = ring.Take(BlockRing::ChunkBytes, 1);
    Expect(lapped.bytes != nullptr && lapped.chunk == copy.chunk && lapped.lap == copy.lap + 1, "the only chunk makes a lap once its batch has run");
    Expect(!memo.Find(key, 3, ring, found), "the lapped entry misses");
    Expect(!memo.Find(key, 4, ring, found), "an entry of another epoch is stale whatever its chunk");
    const auto counts = memo.Counted();
    Expect(sameCounts(counts, 7, 2, 2, 1), "each find is counted once, with at most one of hit, stale or dead");
    Expect(counts.finds - counts.hits - counts.stale - counts.dead == 2, "plain misses are the remaining finds");
    ring.Posted(lapped);
}

}

int main() {
    testHit();
    testMissByKey();
    testMissByFirst();
    testMissByEpoch();
    testDeadAfterLap();
    testLapInsideTheDraw();
    testOnlyTheLappedChunkDies();
    testReplace();
    testClear();
    testCounts();
    if (failures != 0) {
        std::fprintf(stderr, "%d front memo checks failed\n", failures);
        return 1;
    }
    std::printf("front memo tests passed\n");
    return 0;
}
