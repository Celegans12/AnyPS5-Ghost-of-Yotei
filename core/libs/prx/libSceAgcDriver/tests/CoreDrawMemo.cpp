#include "prx/libSceAgcDriver/Core/include/DrawMemo.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>
#include <vector>

namespace {

std::uint64_t allocations = 0;

void* allocate(std::size_t bytes) {
    ++allocations;
    if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) return memory;
    throw std::bad_alloc();
}

}

void* operator new(std::size_t bytes) { return allocate(bytes); }
void* operator new[](std::size_t bytes) { return allocate(bytes); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

using namespace AgcDriver::Core;
using AgcDriver::GuestMemory::ContentToken;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::size_t MissIndex(ReplayMiss miss) { return static_cast<std::size_t>(miss); }

struct FakeRange {
    std::uint64_t address = 0;
    std::uint32_t bytes = 0;
    ContentToken token;
    bool busy = false;
    bool pending = false;
};

struct FakeMemory {
    std::vector<FakeRange> ranges;
    std::uint64_t currentCalls = 0;
    std::uint64_t pendingCalls = 0;
    bool strange = false;
};

FakeRange* rangeAt(FakeMemory& memory, std::uint64_t address, std::size_t bytes) {
    for (auto& range : memory.ranges) {
        if (range.address == address && range.bytes == bytes) return &range;
    }
    memory.strange = true;
    return nullptr;
}

bool fakeStillCurrent(void* context, std::uint64_t address, std::size_t bytes, const ContentToken& token) {
    auto& memory = *static_cast<FakeMemory*>(context);
    ++memory.currentCalls;
    const auto* range = rangeAt(memory, address, bytes);
    return range != nullptr && !range->busy && range->token.content == token.content && range->token.context == token.context;
}

bool fakePendingGpuWrite(void* context, std::uint64_t address, std::size_t bytes) {
    auto& memory = *static_cast<FakeMemory*>(context);
    ++memory.pendingCalls;
    const auto* range = rangeAt(memory, address, bytes);
    return range == nullptr || range->pending;
}

struct World {
    explicit World(std::size_t ranges) {
        for (std::size_t index = 0; index < ranges; ++index) {
            FakeRange range;
            range.address = 0x200000000ull + 0x10000ull * index;
            range.bytes = static_cast<std::uint32_t>(256u + 64u * index);
            range.token = ContentToken{.content = 100u + index, .context = 5u};
            memory.ranges.push_back(range);
        }
        for (std::size_t index = 0; index < targets.size(); ++index) targets[index] = 1000u + index;
    }
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    DrawProof Proof() const {
        DrawProof proof;
        for (std::size_t index = 0; index < memory.ranges.size(); ++index) {
            proof.ranges[index] = memory.ranges[index].token;
            proof.rangeAddress[index] = memory.ranges[index].address;
            proof.rangeBytes[index] = memory.ranges[index].bytes;
        }
        proof.targetGeneration = targets;
        proof.depthGeneration = depth;
        proof.imageGeneration = image;
        proof.mappingSerial = mapping;
        proof.shaderGeneration = shader;
        proof.device = device;
        return proof;
    }

    ReplayNow Now() {
        ReplayNow now;
        now.targetGeneration = targets;
        now.depthGeneration = depth;
        now.imageGeneration = image;
        now.mappingSerial = mapping;
        now.shaderGeneration = shader;
        now.device = device;
        now.stillCurrent = &fakeStillCurrent;
        now.pendingGpuWrite = &fakePendingGpuWrite;
        now.context = &memory;
        return now;
    }

    void Write(std::size_t index) { ++memory.ranges[index].token.content; }
    void Unwrite(std::size_t index) { --memory.ranges[index].token.content; }

    FakeMemory memory;
    std::array<std::uint64_t, 9> targets{};
    std::uint64_t depth = 7, image = 11, mapping = 13, shader = 17, device = 0xd00du;
};

ReplayKey keyOf(std::uint64_t seed) {
    return ReplayKey{.variant = 0x1000u + seed, .images = 0x2000u + seed * 3u, .inputs = 0x3000u + seed * 5u, .targets = 0x4000u + seed * 7u};
}

std::array<std::uint64_t, 5> recordOf(std::uint64_t seed) {
    return {0xa000u + seed, 0xb000u + seed, 0xc000u + seed, 0xd000u + seed, 0xe000u + seed};
}

struct Released {
    Released() {
        keys.reserve(4096);
        words.reserve(4096);
    }
    std::vector<ReplayKey> keys;
    std::vector<std::uint64_t> words;
};

void recordRelease(void* context, const ReplayEntry& entry) noexcept {
    auto& released = *static_cast<Released*>(context);
    released.keys.push_back(entry.key);
    released.words.push_back(entry.resolved[0]);
}

ReplayRelease releaseInto(Released& released) {
    return ReplayRelease{.release = &recordRelease, .context = &released};
}

std::size_t timesReleased(const Released& released, const ReplayKey& key) {
    return static_cast<std::size_t>(std::count(released.keys.begin(), released.keys.end(), key));
}

ReplayKey collidingWith(const ReplayKey& key, std::uint64_t from) {
    for (std::uint64_t seed = from; seed < from + (std::uint64_t{1} << 24u); ++seed) {
        const auto candidate = keyOf(seed);
        if (candidate != key && DrawMemo::SlotOf(candidate) == DrawMemo::SlotOf(key)) return candidate;
    }
    Expect(false, "setup: a colliding key exists");
    return key;
}

ReplayKey apartFrom(const ReplayKey& key, std::uint64_t from) {
    for (std::uint64_t seed = from;; ++seed) {
        const auto candidate = keyOf(seed);
        if (DrawMemo::SlotOf(candidate) != DrawMemo::SlotOf(key)) return candidate;
    }
}

void testHit() {
    World world(3);
    DrawMemo memo;
    const auto key = keyOf(1);
    const auto record = recordOf(1);
    auto& stored = *memo.Store(key, world.Proof(), record, 4);
    Expect(stored.key == key, "hit: the stored entry carries its key");
    Expect(std::equal(record.begin(), record.end(), stored.resolved.begin()), "hit: the resolved words are kept in order");
    Expect(std::all_of(stored.resolved.begin() + record.size(), stored.resolved.end(), [](std::uint64_t word) { return word == 0; }), "hit: the words after the record are zero");
    Expect(stored.lastBatch == 4, "hit: the entry remembers its batch");
    auto* found = memo.Find(key);
    Expect(found == &stored, "hit: find returns the stored entry");
    Expect(found != nullptr && memo.Fresh(*found, world.Now()) == ReplayMiss::None, "hit: every proof holds");
    Expect(world.memory.currentCalls == 3 && world.memory.pendingCalls == 3, "hit: each used range is asked once for its token and once for a pending GPU write");
    Expect(!world.memory.strange, "hit: the memo asks about the proven address, size and token");
    const auto counts = memo.Counted();
    Expect(counts.finds == 1 && counts.freshChecks == 1 && counts.stores == 1, "hit: one find, one check, one store");
    Expect(counts.outcomes[MissIndex(ReplayMiss::None)] == 1 && counts.outcomes[MissIndex(ReplayMiss::Absent)] == 0, "hit: counted as a hit");
    Expect(counts.evicted == 0 && counts.replaced == 0 && counts.refreshed == 0 && counts.waiting == 0 && counts.released == 0, "hit: nothing evicted, replaced or released");
}

void testWithoutRanges() {
    World world(0);
    DrawMemo memo;
    const auto key = keyOf(3);
    memo.Store(key, world.Proof(), recordOf(3), 1);
    auto* found = memo.Find(key);
    Expect(found != nullptr && memo.Fresh(*found, world.Now()) == ReplayMiss::None, "no ranges: a draw without guest ranges is proven by its generations");
    Expect(world.memory.currentCalls == 0 && world.memory.pendingCalls == 0, "no ranges: guest memory is not asked");
}

void testAbsent() {
    World world(1);
    DrawMemo memo;
    const auto key = keyOf(5);
    Expect(memo.Find(key) == nullptr, "absent: an empty memo finds nothing");
    memo.Store(key, world.Proof(), recordOf(5), 1);
    auto variant = key;
    ++variant.variant;
    auto images = key;
    ++images.images;
    auto inputs = key;
    ++inputs.inputs;
    auto targets = key;
    ++targets.targets;
    Expect(memo.Find(variant) == nullptr, "absent: another variant");
    Expect(memo.Find(images) == nullptr, "absent: other images");
    Expect(memo.Find(inputs) == nullptr, "absent: other input ranges");
    Expect(memo.Find(targets) == nullptr, "absent: other targets");
    Expect(memo.Find(key) != nullptr, "absent: the stored key is still found");
    const auto counts = memo.Counted();
    Expect(counts.finds == 6 && counts.outcomes[MissIndex(ReplayMiss::Absent)] == 5, "absent: each failed find is counted as absent");
    Expect(counts.freshChecks == 0, "absent: no proof is checked without an entry");
}

struct MissBook {
    DrawMemo& memo;
    ReplayEntry& entry;
    World& world;
    std::array<std::uint64_t, ReplayMissKinds> expected{};

    void Miss(const ReplayNow& now, ReplayMiss miss, const char* what) {
        Expect(memo.Fresh(entry, now) == miss, what);
        ++expected[MissIndex(miss)];
    }

    void Holds(const char* what) {
        Expect(memo.Find(entry.key) == &entry, what);
        Expect(memo.Fresh(entry, world.Now()) == ReplayMiss::None, what);
        ++expected[MissIndex(ReplayMiss::None)];
    }
};

void testEachMiss() {
    World world(6);
    DrawMemo memo;
    const auto key = keyOf(2);
    memo.Store(key, world.Proof(), recordOf(2), 1);
    auto* entry = memo.Find(key);
    Expect(entry != nullptr, "miss: the entry is found");
    if (entry == nullptr) return;
    MissBook book{memo, *entry, world};

    auto now = world.Now();
    now.device ^= 1u;
    book.Miss(now, ReplayMiss::Device, "miss: another device");
    book.Holds("miss: the entry holds again on its device");

    now = world.Now();
    ++now.shaderGeneration;
    book.Miss(now, ReplayMiss::Shader, "miss: a shader registered or replaced");
    book.Holds("miss: the entry holds again with its shaders");

    now = world.Now();
    ++now.mappingSerial;
    book.Miss(now, ReplayMiss::Mapping, "miss: a mapping changed");
    book.Holds("miss: the entry holds again with its mappings");

    now = world.Now();
    ++now.imageGeneration;
    book.Miss(now, ReplayMiss::Image, "miss: an image changed");
    book.Holds("miss: the entry holds again with its images");

    now = world.Now();
    ++now.depthGeneration;
    book.Miss(now, ReplayMiss::Depth, "miss: the depth target was cleared or written outside the draw");
    book.Holds("miss: the entry holds again with its depth");

    for (std::size_t index = 0; index < now.targetGeneration.size(); ++index) {
        now = world.Now();
        ++now.targetGeneration[index];
        book.Miss(now, ReplayMiss::Target, "miss: a color target filled outside the draw, whichever slot");
        book.Holds("miss: the entry holds again with its targets");
    }

    for (std::size_t index = 0; index < world.memory.ranges.size(); ++index) {
        world.Write(index);
        book.Miss(world.Now(), ReplayMiss::Range, "miss: a written input range, whichever slot");
        world.Unwrite(index);
        book.Holds("miss: the entry holds again once the token answers");
        world.memory.ranges[index].pending = true;
        book.Miss(world.Now(), ReplayMiss::PendingGpuWrite, "miss: a GPU write pending on an input range, whichever slot");
        world.memory.ranges[index].pending = false;
        book.Holds("miss: the entry holds again once the GPU write ran");
    }

    now = world.Now();
    now.stillCurrent = nullptr;
    book.Miss(now, ReplayMiss::Range, "miss: no token check proves no range");
    now = world.Now();
    now.pendingGpuWrite = nullptr;
    book.Miss(now, ReplayMiss::PendingGpuWrite, "miss: no pending write check proves no range");
    book.Holds("miss: the entry holds again with both checks");

    const auto asked = world.memory.currentCalls;
    now = world.Now();
    now.device ^= 1u;
    ++now.shaderGeneration;
    ++now.targetGeneration[0];
    book.Miss(now, ReplayMiss::Device, "order: the device is checked first");
    now = world.Now();
    ++now.depthGeneration;
    ++now.targetGeneration[3];
    book.Miss(now, ReplayMiss::Depth, "order: the depth is checked before the color targets");
    now = world.Now();
    ++now.targetGeneration[8];
    world.Write(0);
    book.Miss(now, ReplayMiss::Target, "order: generations are checked before any range");
    world.Unwrite(0);
    Expect(world.memory.currentCalls == asked, "order: a generation miss asks nothing of guest memory");
    world.memory.ranges[0].pending = true;
    world.Write(1);
    book.Miss(world.Now(), ReplayMiss::PendingGpuWrite, "order: ranges are proven one after the other");
    world.memory.ranges[0].pending = false;
    world.Unwrite(1);
    book.Holds("order: the entry holds again");

    Expect(memo.Find(keyOf(99)) == nullptr, "miss: another key is absent");
    ++book.expected[MissIndex(ReplayMiss::Absent)];
    Expect(!world.memory.strange, "miss: the memo asks only about the proven ranges");

    const auto counts = memo.Counted();
    Expect(counts.outcomes == book.expected, "miss: each outcome is counted under its own cause");
    for (std::size_t index = 0; index < ReplayMissKinds; ++index) Expect(book.expected[index] > 0, "miss: every cause of ReplayMiss was produced");
    Expect(counts.stores == 1 && counts.evicted == 0 && counts.replaced == 0, "miss: a failed proof never evicts");
}

void testCollisionEvicts() {
    World world(2);
    Released released;
    const auto first = keyOf(10);
    const auto second = collidingWith(first, 11);
    const auto other = apartFrom(first, 1u << 20u);
    {
        DrawMemo memo(releaseInto(released));
        memo.Store(first, world.Proof(), recordOf(10), 1);
        memo.Store(other, world.Proof(), recordOf(30), 1);
        memo.Retire(1);
        memo.Store(second, world.Proof(), recordOf(11), 2);
        Expect(memo.Find(first) == nullptr, "collision: the evicted key is absent");
        auto* entry = memo.Find(second);
        Expect(entry != nullptr && entry->key == second && entry->resolved[0] == recordOf(11)[0], "collision: the new key holds the slot with its own record");
        Expect(entry != nullptr && memo.Fresh(*entry, world.Now()) == ReplayMiss::None, "collision: the new entry is proven");
        Expect(memo.Find(other) != nullptr, "collision: a key of another slot stays");
        Expect(released.keys.size() == 1 && timesReleased(released, first) == 1 && released.words[0] == recordOf(10)[0], "collision: an evicted entry whose batch finished is released at once");
        auto counts = memo.Counted();
        Expect(counts.evicted == 1 && counts.waiting == 0 && counts.released == 1, "collision: one eviction, nothing waiting");
        memo.Store(first, world.Proof(), recordOf(12), 3);
        Expect(memo.Find(second) == nullptr && memo.Find(first) != nullptr, "collision: the slot goes to the last key stored");
        counts = memo.Counted();
        Expect(timesReleased(released, second) == 0 && counts.waiting == 1 && counts.evicted == 2, "collision: an evicted entry whose batch still runs waits");
    }
    Expect(timesReleased(released, second) == 1, "collision: destroying the memo releases what waits");
    Expect(timesReleased(released, first) == 2 && timesReleased(released, other) == 1, "collision: destroying the memo releases what it holds, once each");
    Expect(released.keys.size() == 4, "collision: nothing is released twice");
}

void testRetireWaitsForBatch() {
    World world(1);
    Released released;
    DrawMemo memo(releaseInto(released));
    const auto key = keyOf(40);
    const auto rival = collidingWith(key, 41);
    memo.Store(key, world.Proof(), recordOf(40), 3);
    auto* entry = memo.Find(key);
    Expect(entry != nullptr && memo.Fresh(*entry, world.Now()) == ReplayMiss::None, "retire: the entry is replayed");
    if (entry != nullptr) entry->lastBatch = 5;
    memo.Store(rival, world.Proof(), recordOf(41), 6);
    Expect(memo.Counted().waiting == 1 && released.keys.empty(), "retire: the evicted entry waits for its last batch");
    memo.Retire(4);
    Expect(released.keys.empty() && memo.Counted().waiting == 1, "retire: not released while batch 5, its last use, runs");
    memo.Retire(5);
    Expect(released.keys.size() == 1 && released.keys[0] == key && released.words[0] == recordOf(40)[0], "retire: released once its last batch finished, with its own record");
    Expect(memo.Counted().waiting == 0, "retire: nothing waits after its batch");
    memo.Retire(3);
    Expect(released.keys.size() == 1, "retire: an older finished batch releases nothing more");

    memo.Store(rival, world.Proof(), recordOf(42), 7);
    auto counts = memo.Counted();
    Expect(counts.replaced == 1 && counts.waiting == 1 && released.keys.size() == 1, "retire: a record replaced under the same key waits for its batch");
    memo.Retire(6);
    Expect(released.keys.size() == 2 && released.keys[1] == rival && released.words[1] == recordOf(41)[0], "retire: the replaced record is released after its batch");

    world.Write(0);
    memo.Store(rival, world.Proof(), recordOf(42), 8);
    counts = memo.Counted();
    Expect(counts.refreshed == 1 && counts.waiting == 0 && released.keys.size() == 2, "retire: storing the same record again releases nothing");
    auto* refreshed = memo.Find(rival);
    Expect(refreshed != nullptr && refreshed->lastBatch == 8, "retire: a refreshed entry carries the new batch");
    Expect(refreshed != nullptr && memo.Fresh(*refreshed, world.Now()) == ReplayMiss::None, "retire: a refreshed entry carries the new proof");
    memo.Store(rival, world.Proof(), recordOf(42), 2);
    refreshed = memo.Find(rival);
    Expect(refreshed != nullptr && refreshed->lastBatch == 8, "retire: a refresh from an older batch keeps the last batch");

    memo.Retire(20);
    memo.Store(key, world.Proof(), recordOf(43), 21);
    Expect(released.keys.size() == 3 && released.keys[2] == rival && released.words[2] == recordOf(42)[0], "retire: an entry whose last batch finished is released when evicted");
    Expect(memo.Counted().released == 3, "retire: the release count follows");
}

void testRetireKeepsTheLiving() {
    World world(1);
    Released released;
    DrawMemo memo(releaseInto(released));
    constexpr std::array<std::uint64_t, 4> lastBatches{10, 12, 11, 13};
    std::array<ReplayKey, 4> keys{};
    std::uint64_t seed = 100;
    for (std::size_t index = 0; index < keys.size(); ++index) {
        auto candidate = keyOf(seed++);
        const auto taken = [&](const ReplayKey& key) { return DrawMemo::SlotOf(key) == DrawMemo::SlotOf(candidate); };
        while (std::any_of(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(index), taken)) candidate = keyOf(seed++);
        keys[index] = candidate;
        memo.Store(keys[index], world.Proof(), recordOf(index), lastBatches[index]);
    }
    std::uint64_t from = 1u << 22u;
    for (const auto& key : keys) {
        const auto rival = collidingWith(key, from);
        from = rival.variant - 0x1000u + 1u;
        memo.Store(rival, world.Proof(), recordOf(50), 14);
    }
    Expect(memo.Counted().waiting == 4 && released.keys.empty(), "living: four evicted entries wait");
    memo.Retire(11);
    Expect(released.keys.size() == 2 && timesReleased(released, keys[0]) == 1 && timesReleased(released, keys[2]) == 1, "living: the entries of batches 10 and 11 are released");
    Expect(memo.Counted().waiting == 2, "living: the entries of batches 12 and 13 still wait");
    memo.Retire(12);
    Expect(released.keys.size() == 3 && timesReleased(released, keys[1]) == 1, "living: batch 12 releases its entry only");
    memo.Retire(13);
    Expect(released.keys.size() == 4 && timesReleased(released, keys[3]) == 1 && memo.Counted().waiting == 0, "living: batch 13 releases the last one");
}

void testTokenFalseThenTrue() {
    World world(2);
    DrawMemo memo;
    const auto key = keyOf(60);
    memo.Store(key, world.Proof(), recordOf(60), 1);
    auto* entry = memo.Find(key);
    Expect(entry != nullptr, "token: the entry is found");
    if (entry == nullptr) return;

    world.memory.ranges[0].busy = true;
    Expect(memo.Fresh(*entry, world.Now()) == ReplayMiss::Range, "token: a token that cannot answer is a miss");
    Expect(memo.Find(key) == entry, "token: a false token keeps the entry");
    world.memory.ranges[0].busy = false;
    Expect(memo.Fresh(*entry, world.Now()) == ReplayMiss::None, "token: the same token answers true later without a store");

    world.Write(1);
    Expect(memo.Fresh(*entry, world.Now()) == ReplayMiss::Range, "token: a written range is a miss");
    Expect(memo.Fresh(*entry, world.Now()) == ReplayMiss::Range, "token: it stays a miss while the proof holds the old token");
    auto& stored = *memo.Store(key, world.Proof(), recordOf(60), 2);
    Expect(&stored == entry, "token: the new proof goes to the same slot");
    Expect(memo.Fresh(stored, world.Now()) == ReplayMiss::None, "token: the new token is true");
    const auto counts = memo.Counted();
    Expect(counts.outcomes[MissIndex(ReplayMiss::Range)] == 3 && counts.outcomes[MissIndex(ReplayMiss::None)] == 2, "token: three false answers, two true");
    Expect(counts.refreshed == 1 && counts.replaced == 0 && counts.evicted == 0, "token: the new proof refreshes the same record");
}

void testStoreLimits() {
    World world(1);
    DrawMemo memo;
    const auto key = keyOf(70);
    std::array<std::uint64_t, DrawMemo::ResolvedWords + 1u> tooLong{};
    Expect(memo.Store(key, world.Proof(), tooLong, 1) == nullptr, "limits: a record longer than 16 words is refused");
    Expect(memo.Find(key) == nullptr && memo.Counted().stores == 0 && memo.Counted().refused == 1, "limits: a refused record stores nothing and is counted");
    std::array<std::uint64_t, DrawMemo::ResolvedWords> full{};
    for (std::size_t index = 0; index < full.size(); ++index) full[index] = 0x5000u + index;
    memo.Store(key, world.Proof(), full, 1);
    auto* entry = memo.Find(key);
    Expect(entry != nullptr && entry->resolved == full, "limits: a record of 16 words is kept whole");
    auto& same = *memo.Store(entry->key, entry->proof, entry->resolved, 2);
    Expect(same.resolved == full && same.key == key && same.lastBatch == 2, "limits: storing an entry over itself keeps its record");
}

void testSlots() {
    std::vector<std::uint8_t> used(DrawMemo::Slots, 0);
    bool inside = true;
    for (std::uint64_t seed = 0; seed < DrawMemo::Slots; ++seed) {
        const auto slot = DrawMemo::SlotOf(keyOf(seed));
        if (slot >= DrawMemo::Slots) {
            inside = false;
            continue;
        }
        used[slot] = 1;
    }
    const auto filled = static_cast<std::size_t>(std::count(used.begin(), used.end(), std::uint8_t{1}));
    Expect(inside, "slots: every key lands inside the table");
    Expect(filled > DrawMemo::Slots * 6u / 10u, "slots: 65536 keys spread over most of the table");
    auto swapped = keyOf(1);
    std::swap(swapped.images, swapped.inputs);
    Expect(DrawMemo::SlotOf(swapped) != DrawMemo::SlotOf(keyOf(1)) || swapped == keyOf(1), "slots: swapping two fields moves the key");
}

struct QuickMemory {
    std::uint64_t context = 9;
    std::uint64_t pendingBegin = 0;
    std::uint64_t pendingEnd = 0;
};

bool quickStillCurrent(void* context, std::uint64_t address, std::size_t bytes, const ContentToken& token) {
    const auto& memory = *static_cast<const QuickMemory*>(context);
    return token.content == (address ^ bytes) && token.context == memory.context;
}

bool quickPendingGpuWrite(void* context, std::uint64_t address, std::size_t bytes) {
    const auto& memory = *static_cast<const QuickMemory*>(context);
    return memory.pendingBegin < address + bytes && address < memory.pendingEnd;
}

DrawProof quickProof(const QuickMemory& memory) {
    DrawProof proof;
    for (std::size_t index = 0; index < proof.ranges.size(); ++index) {
        proof.rangeAddress[index] = 0x300000000ull + 0x40000ull * index;
        proof.rangeBytes[index] = static_cast<std::uint32_t>(4096u + 512u * index);
        proof.ranges[index] = ContentToken{.content = proof.rangeAddress[index] ^ proof.rangeBytes[index], .context = memory.context};
    }
    for (std::size_t index = 0; index < proof.targetGeneration.size(); ++index) proof.targetGeneration[index] = 77u + index;
    proof.depthGeneration = 3;
    proof.imageGeneration = 5;
    proof.mappingSerial = 8;
    proof.shaderGeneration = 13;
    proof.device = 21;
    return proof;
}

ReplayNow quickNow(QuickMemory& memory, const DrawProof& proof) {
    ReplayNow now;
    now.targetGeneration = proof.targetGeneration;
    now.depthGeneration = proof.depthGeneration;
    now.imageGeneration = proof.imageGeneration;
    now.mappingSerial = proof.mappingSerial;
    now.shaderGeneration = proof.shaderGeneration;
    now.device = proof.device;
    now.stillCurrent = &quickStillCurrent;
    now.pendingGpuWrite = &quickPendingGpuWrite;
    now.context = &memory;
    return now;
}

struct Timing {
    double nanoseconds = 0;
    std::uint64_t misses = 0;
    std::uint64_t sum = 0;
};

Timing timeCalls(DrawMemo& memo, const std::vector<ReplayKey>& keys, const ReplayNow& now, std::uint64_t calls) {
    Timing timing;
    std::size_t cursor = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t call = 0; call < calls; ++call) {
        auto* entry = memo.Find(keys[cursor]);
        if (entry == nullptr || memo.Fresh(*entry, now) != ReplayMiss::None) ++timing.misses;
        else timing.sum += entry->resolved[0];
        if (++cursor == keys.size()) cursor = 0;
    }
    const auto stop = std::chrono::steady_clock::now();
    timing.nanoseconds = std::chrono::duration<double, std::nano>(stop - start).count() / static_cast<double>(calls);
    return timing;
}

std::vector<ReplayKey> fill(DrawMemo& memo, std::size_t draws, const DrawProof& proof) {
    memo.Retire(1);
    for (std::uint64_t seed = 0; seed < draws; ++seed) memo.Store(keyOf(seed), proof, recordOf(seed), 1);
    std::vector<ReplayKey> kept;
    kept.reserve(draws);
    for (std::uint64_t seed = 0; seed < draws; ++seed) {
        if (memo.Find(keyOf(seed)) != nullptr) kept.push_back(keyOf(seed));
    }
    return kept;
}

void testTiming() {
    constexpr std::uint64_t Calls = 1000000;
    constexpr std::size_t FrameDraws = 3000;
    QuickMemory quick;
    const auto proof = quickProof(quick);
    const auto now = quickNow(quick, proof);
    {
        DrawMemo memo;
        const auto frame = fill(memo, FrameDraws, proof);
        Expect(frame.size() > FrameDraws * 9u / 10u, "timing: most draws of a frame keep their slot");
        const auto warm = timeCalls(memo, frame, now, Calls / 10u);
        Expect(warm.misses == 0, "timing: the warm-up calls succeed");
        const auto before = allocations;
        std::array<Timing, 3> runs{};
        for (auto& run : runs) run = timeCalls(memo, frame, now, Calls);
        const auto allocated = allocations - before;
        double best = runs[0].nanoseconds;
        std::uint64_t misses = 0;
        std::uint64_t sum = 0;
        for (const auto& run : runs) {
            best = std::min(best, run.nanoseconds);
            misses += run.misses;
            sum += run.sum;
        }
        std::printf("draw memo: Find + Fresh %.1f ns per call on success (best of 3 runs of %llu calls over a frame of %llu draws; runs %.1f, %.1f, %.1f ns; 6 ranges, 9 targets)\n", best, static_cast<unsigned long long>(Calls), static_cast<unsigned long long>(frame.size()), runs[0].nanoseconds, runs[1].nanoseconds, runs[2].nanoseconds);
        Expect(misses == 0, "timing: every timed call succeeds");
        Expect(sum != 0, "timing: the resolved words were read");
        Expect(allocated == 0, "timing: a successful Find and Fresh allocate nothing");
        Expect(best < 100.0, "timing: Find + Fresh under 100 ns per call on success");
    }
    {
        DrawMemo memo;
        const auto table = fill(memo, DrawMemo::Slots, proof);
        Expect(table.size() > DrawMemo::Slots * 6u / 10u, "timing: a full table keeps most keys");
        const auto spread = timeCalls(memo, table, now, Calls);
        std::printf("draw memo: Find + Fresh %.1f ns per call on success over %llu draws spread across the whole table (%llu calls)\n", spread.nanoseconds, static_cast<unsigned long long>(table.size()), static_cast<unsigned long long>(Calls));
        Expect(spread.misses == 0, "timing: every spread call succeeds");
    }
}

}

int main() {
    testHit();
    testWithoutRanges();
    testAbsent();
    testEachMiss();
    testCollisionEvicts();
    testRetireWaitsForBatch();
    testRetireKeepsTheLiving();
    testTokenFalseThenTrue();
    testStoreLimits();
    testSlots();
    testTiming();
    if (failures != 0) {
        std::fprintf(stderr, "%d draw memo checks failed\n", failures);
        return 1;
    }
    std::printf("draw memo tests passed\n");
    return 0;
}
