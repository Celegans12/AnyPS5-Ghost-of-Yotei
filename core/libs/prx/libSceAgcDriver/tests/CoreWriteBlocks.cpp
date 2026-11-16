#include "prx/libSceAgcDriver/Core/include/WriteBlocks.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

using Range = WriteBlocks::Range;
constexpr std::uint64_t Block = WriteBlocks::BlockBytes;
constexpr std::uint64_t Leaf = std::uint64_t{1} << WriteBlocks::LeafShift;
constexpr std::uint64_t Top = WriteBlocks::CoveredBytes;
constexpr std::uint64_t Max = std::numeric_limits<std::uint64_t>::max();

void enter(WriteBlocks& blocks, Range range) {
    blocks.Enter(std::span<const Range>(&range, 1));
}

void leave(WriteBlocks& blocks, Range range) {
    blocks.Leave(std::span<const Range>(&range, 1));
}

bool clearRange(const WriteBlocks& blocks, Range range) {
    return blocks.Clear(range.first, range.second - range.first);
}

void testEmpty() {
    WriteBlocks blocks;
    Expect(blocks.Clear(0, Max), "an empty table is clear over the whole address space");
    Expect(blocks.Clear(Max, 1), "an empty table is clear at the last byte");
    Expect(blocks.Clear(0x123456, 0), "a read of zero bytes is clear");
    Expect(blocks.Ranges() == 0 && blocks.Leaves() == 0 && blocks.Spilled() == 0, "an empty table holds no range, no leaf and no spilled block");
    Expect(blocks.Holders(0) == 0 && blocks.Holders(Top) == 0, "an empty table has no holder");
}

void testOneBlock() {
    WriteBlocks blocks;
    const Range write{5 * Block + 0x100, 5 * Block + 0x140};
    enter(blocks, write);
    Expect(!clearRange(blocks, write), "the written bytes are not clear");
    Expect(!blocks.Clear(5 * Block, 1), "the first byte of the written block is not clear");
    Expect(!blocks.Clear(6 * Block - 1, 1), "the last byte of the written block is not clear");
    Expect(blocks.Clear(4 * Block, Block), "the block before the write is clear");
    Expect(blocks.Clear(6 * Block, Block), "the block after the write is clear");
    Expect(!blocks.Clear(4 * Block, Block + 1), "a read ending on the first byte of the written block is not clear");
    Expect(!blocks.Clear(6 * Block - 1, Block), "a read starting on the last byte of the written block is not clear");
    Expect(!blocks.Clear(0, Max), "a read of everything is not clear");
    Expect(blocks.Holders(5 * Block) == 1 && blocks.Holders(4 * Block) == 0 && blocks.Holders(6 * Block) == 0, "one block holds the write");
    Expect(blocks.Leaves() == 1, "a small write allocates one leaf");
    leave(blocks, write);
    Expect(blocks.Clear(0, Max), "a left write leaves the table clear");
    Expect(blocks.Ranges() == 0 && blocks.Holders(5 * Block) == 0, "a left write is forgotten");
}

void testTwoBlocks() {
    WriteBlocks blocks;
    const Range write{8 * Block - 0x10, 8 * Block + 0x10};
    enter(blocks, write);
    Expect(!blocks.Clear(7 * Block, 1), "the first byte of the first block of a straddling write is not clear");
    Expect(!blocks.Clear(9 * Block - 1, 1), "the last byte of the second block of a straddling write is not clear");
    Expect(blocks.Clear(6 * Block, Block), "the block before a straddling write is clear");
    Expect(blocks.Clear(9 * Block, Block), "the block after a straddling write is clear");
    Expect(blocks.Holders(7 * Block) == 1 && blocks.Holders(8 * Block) == 1, "both blocks hold a straddling write");
    const std::array<Range, 2> reads{Range{6 * Block, 7 * Block}, Range{9 * Block, 9 * Block + 1}};
    Expect(blocks.Clear(reads), "a set of reads beside a straddling write is clear");
    const std::array<Range, 3> touching{Range{6 * Block, 7 * Block}, Range{8 * Block + 0x8000, 8 * Block + 0x8001}, Range{9 * Block, 9 * Block + 1}};
    Expect(!blocks.Clear(touching), "a set of reads with one read in a written block is not clear");
    leave(blocks, write);
    Expect(blocks.Clear(0, Max) && blocks.Clear(touching), "a left straddling write leaves the table clear");

    const Range leafEdge{3 * Leaf - 0x20, 3 * Leaf + 0x20};
    enter(blocks, leafEdge);
    Expect(!blocks.Clear(3 * Leaf - Block, 1) && !blocks.Clear(3 * Leaf + Block - 1, 1), "a write straddling two leaves holds the block on each side");
    Expect(blocks.Clear(3 * Leaf - 2 * Block, Block) && blocks.Clear(3 * Leaf + Block, Block), "a write straddling two leaves holds no other block");
    leave(blocks, leafEdge);
    Expect(blocks.Clear(0, Max), "a left write straddling two leaves leaves the table clear");
}

void testLargeRanges() {
    WriteBlocks blocks;
    const Range touching{Leaf + 0x8000, 2 * Leaf - 0x10};
    enter(blocks, touching);
    Expect(blocks.Leaves() == 0, "a write touching every block of a leaf is held by the leaf without allocating it");
    Expect(!blocks.Clear(Leaf, 1) && !blocks.Clear(2 * Leaf - 1, 1), "a write touching every block of a leaf holds its first and last blocks");
    Expect(blocks.Clear(Leaf - Block, Block) && blocks.Clear(2 * Leaf, Block), "a write touching every block of a leaf holds nothing beside the leaf");
    leave(blocks, touching);
    Expect(blocks.Clear(0, Max), "a left write touching every block of a leaf leaves the table clear");

    const Range write{Leaf + Block + 0x8000, 5 * Leaf + 0x10};
    enter(blocks, write);
    Expect(blocks.Leaves() == 2, "a write over many leaves allocates only its two partial leaves");
    Expect(!blocks.Clear(Leaf + Block, 1), "the first block of a large write is not clear");
    Expect(blocks.Clear(Leaf, Block), "the block before a large write is clear");
    Expect(!blocks.Clear(3 * Leaf + 0x12345678, 8), "a fully covered leaf is not clear");
    Expect(!blocks.Clear(2 * Leaf, Leaf), "a read of a fully covered leaf is not clear");
    Expect(!blocks.Clear(5 * Leaf + Block - 1, 1), "the last block of a large write is not clear");
    Expect(blocks.Clear(5 * Leaf + Block, Block), "the block after a large write is clear");
    Expect(blocks.Clear(6 * Leaf, Top - 6 * Leaf), "a huge read after a large write is clear");
    Expect(blocks.Clear(0, Leaf + Block), "the space before a large write is clear");
    Expect(!blocks.Clear(0, Max), "a read of everything is not clear while a large write is held");
    Expect(blocks.Holders(Leaf + Block) == 1 && blocks.Holders(Leaf) == 0 && blocks.Holders(3 * Leaf) == 1 && blocks.Holders(5 * Leaf) == 1 && blocks.Holders(5 * Leaf + Block) == 0, "every block of a large write holds it once");
    leave(blocks, write);
    Expect(blocks.Clear(0, Max), "a left large write leaves the table clear");
    Expect(blocks.Holders(3 * Leaf) == 0 && blocks.Holders(Leaf + Block) == 0, "a left large write is forgotten");

    const Range everything{0, Max};
    enter(blocks, everything);
    Expect(blocks.Leaves() == 2, "a write of the whole address space allocates no leaf");
    Expect(!blocks.Clear(0, 1) && !blocks.Clear(Top - 1, 1) && !blocks.Clear(Top, 1) && !blocks.Clear(Max - 1, 1), "a write of the whole address space holds every byte");
    leave(blocks, everything);
    Expect(blocks.Clear(0, Max), "a left write of the whole address space leaves the table clear");

    const Range tiny{Leaf + 40 * Block + 0x10, Leaf + 40 * Block + 0x18};
    enter(blocks, tiny);
    Expect(!blocks.Clear(Leaf, 64 * Block), "a long read over a held block is not clear");
    Expect(!blocks.Clear(Leaf, Leaf), "a read of the whole leaf of a held block is not clear");
    Expect(blocks.Clear(Leaf + 41 * Block, 400 * Block), "a long read after a held block is clear");
    Expect(blocks.Clear(Leaf, 40 * Block), "a long read before a held block is clear");
    Expect(blocks.Clear(2 * Leaf, Leaf), "a read of another leaf is clear");
    leave(blocks, tiny);
    Expect(blocks.Clear(Leaf, Leaf) && blocks.Clear(Leaf, 64 * Block), "a long read over an allocated but empty leaf is clear");
    Expect(blocks.Ranges() == 0, "large ranges leave no registered range");
}

void testTop() {
    WriteBlocks blocks;
    const Range write{Top - 0x10, Top + 0x10};
    enter(blocks, write);
    Expect(!blocks.Clear(Top - Block, 1), "the last covered block of a write crossing the top is not clear");
    Expect(!blocks.Clear(Top + 0x1000, 1), "a read above the top is not clear while a write above the top is held");
    Expect(!blocks.Clear(Max - 1, 1), "the last bytes are not clear while a write above the top is held");
    Expect(!blocks.Clear(Max - 4, 100), "a wrapping read is not clear while a write above the top is held");
    Expect(blocks.Clear(Top - 2 * Block, Block), "the block before a write crossing the top is clear");
    Expect(blocks.Holders(Top) == 1 && blocks.Holders(Top - 1) == 1, "both sides of the top hold a write crossing it");
    leave(blocks, write);
    Expect(blocks.Clear(Top, Max - Top) && blocks.Clear(0, Max), "a left write crossing the top leaves the table clear");

    const Range last{Max - 8, Max};
    enter(blocks, last);
    Expect(blocks.Clear(0, Top), "a write above the top leaves the covered space clear");
    Expect(!blocks.Clear(Max - 1, 1) && !blocks.Clear(Top, 1), "a write above the top holds the space above the top");
    Expect(blocks.Leaves() == 1, "a write above the top allocates no leaf");
    leave(blocks, last);
    Expect(blocks.Clear(0, Max), "a left write above the top leaves the table clear");
}

void testMultiplicity() {
    WriteBlocks blocks;
    const Range write{10 * Block, 10 * Block + 64};
    const std::array<Range, 2> twice{write, write};
    blocks.Enter(twice);
    Expect(blocks.Ranges() == 1 && blocks.Holders(10 * Block) == 1, "a range entered twice is held once by its block");
    leave(blocks, write);
    Expect(!blocks.Clear(10 * Block, 1), "a range entered twice stays after one leave");

    const std::array<Range, 2> mixed{write, Range{20 * Block, 20 * Block + 8}};
    bool unknown = false;
    try {
        blocks.Leave(mixed);
    } catch (const std::logic_error&) {
        unknown = true;
    }
    Expect(unknown, "leaving a range that never entered throws");
    Expect(!blocks.Clear(10 * Block, 1) && blocks.Ranges() == 1, "a refused leave changes nothing");

    bool repeated = false;
    try {
        blocks.Leave(twice);
    } catch (const std::logic_error&) {
        repeated = true;
    }
    Expect(repeated, "leaving a range more often than it entered throws");
    Expect(!blocks.Clear(10 * Block, 1) && blocks.Ranges() == 1, "a refused repeated leave changes nothing");
    leave(blocks, write);
    Expect(blocks.Clear(0, Max) && blocks.Ranges() == 0, "a range entered twice and left twice is gone");

    const std::array<Range, 2> empty{Range{30 * Block, 30 * Block}, Range{40 * Block, 31 * Block}};
    blocks.Enter(empty);
    Expect(blocks.Ranges() == 0 && blocks.Clear(0, Max), "empty and inverted ranges are not held");
    blocks.Leave(empty);
    Expect(blocks.Ranges() == 0, "empty and inverted ranges leave without error");

    const Range wide{50 * Block, 52 * Block + 10};
    const Range inner{51 * Block + 5, 51 * Block + 6};
    enter(blocks, wide);
    enter(blocks, inner);
    Expect(blocks.Holders(51 * Block) == 2 && blocks.Holders(50 * Block) == 1, "overlapping ranges each hold their blocks");
    leave(blocks, wide);
    Expect(!blocks.Clear(51 * Block, 1) && blocks.Clear(50 * Block, Block) && blocks.Clear(52 * Block, 1), "after the wide range leaves, only the block of the inner range is held");
    leave(blocks, inner);
    Expect(blocks.Clear(0, Max), "overlapping ranges that left leave the table clear");
}

void testOverflow() {
    WriteBlocks blocks;
    const std::uint64_t base = 3 * Leaf + 77 * Block;
    std::vector<Range> writes;
    for (std::uint64_t i = 0; i < 600; ++i) writes.emplace_back(base + i * 16, base + i * 16 + 8);
    bool counted = true;
    for (std::uint64_t i = 0; i < writes.size(); ++i) {
        enter(blocks, writes[i]);
        if (blocks.Holders(base) != i + 1 || blocks.Clear(base, 1)) counted = false;
    }
    Expect(counted, "the holder count of a block follows every enter past the byte limit");
    Expect(blocks.Spilled() == 1, "a block held by more than 254 ranges spills");
    Expect(blocks.Clear(base - Block, Block) && blocks.Clear(base + Block, Block), "a spilled block does not carry into its neighbours");
    Expect(blocks.Holders(base + Block) == 0 && blocks.Holders(base - Block) == 0, "the neighbours of a spilled block hold nothing");

    for (std::uint64_t i = writes.size(); i > 200; --i) {
        leave(blocks, writes[i - 1]);
        if (blocks.Holders(base) != i - 1 || blocks.Clear(base, 1)) counted = false;
    }
    Expect(counted, "the holder count of a block follows every leave back under the byte limit");
    Expect(blocks.Spilled() == 0, "a block back under the byte limit no longer spills");

    for (std::uint64_t i = 200; i < 400; ++i) enter(blocks, writes[i]);
    Expect(blocks.Holders(base) == 400 && blocks.Spilled() == 1, "a block crossing the byte limit again spills again");
    for (std::uint64_t i = 0; i < 399; ++i) {
        leave(blocks, writes[i]);
        if (blocks.Clear(base, Block)) counted = false;
    }
    Expect(counted, "a block is never clear while one of its ranges is held");
    Expect(blocks.Holders(base) == 1 && blocks.Spilled() == 0, "one holder remains after the others left");
    leave(blocks, writes[399]);
    Expect(blocks.Clear(base, Block) && blocks.Holders(base) == 0 && blocks.Ranges() == 0, "the last leave clears a block that spilled");

    for (std::uint64_t i = 0; i < 300; ++i) enter(blocks, Range{base + i * 16, base + Block + 8});
    Expect(blocks.Holders(base) == 300 && blocks.Holders(base + Block) == 300 && blocks.Spilled() == 2, "two neighbouring blocks spill together");
    for (std::uint64_t i = 0; i < 300; ++i) leave(blocks, Range{base + i * 16, base + Block + 8});
    Expect(blocks.Clear(0, Max) && blocks.Spilled() == 0, "two neighbouring blocks that spilled come back clear");
}

struct alignas(64) Published {
    std::atomic<std::uint64_t> sequence{0};
    std::atomic<std::uint64_t> first{0};
    std::atomic<std::uint64_t> last{0};
};

void testConcurrent() {
    WriteBlocks blocks;
    constexpr unsigned Writers = 4;
    constexpr unsigned Readers = 4;
    constexpr std::uint64_t Rounds = 20000;
    constexpr std::uint64_t HotHeld = 80;
    constexpr std::uint64_t HotSlots = 256;
    const std::uint64_t base = 2 * Leaf + 1000 * Block;
    const std::uint64_t hot = base + 500 * Block;
    const auto hotRange = [hot](unsigned writer, std::uint64_t slot) {
        const auto start = hot + writer * (Block / Writers) + (slot % HotSlots) * 16;
        return Range{start, start + 8};
    };
    std::array<Published, Writers> published;
    std::atomic<unsigned> ready{0};
    std::atomic<bool> done{false};
    std::atomic<std::uint64_t> checks{0};
    std::atomic<std::uint64_t> heldChecks{0};
    std::atomic<std::uint64_t> violations{0};
    std::atomic<std::uint64_t> hotChecks{0};
    std::atomic<std::uint64_t> hotViolations{0};
    std::array<std::uint64_t, Writers> hotEnds{};

    for (unsigned writer = 0; writer < Writers; ++writer) {
        for (std::uint64_t slot = 0; slot < HotHeld; ++slot) enter(blocks, hotRange(writer, slot));
    }

    const auto started = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (unsigned writer = 0; writer < Writers; ++writer) {
        threads.emplace_back([&, writer] {
            std::mt19937_64 random(0x5eed0000u + writer);
            auto& slot = published[writer];
            std::uint64_t hotNext = HotHeld;
            ready.fetch_add(1);
            while (ready.load() < Writers + Readers) std::this_thread::yield();
            for (std::uint64_t round = 0; round < Rounds; ++round) {
                Range range;
                switch (random() % 6) {
                case 0: {
                    const auto start = base + (random() % 64) * Block + random() % (Block - 64);
                    range = {start, start + 1 + random() % 64};
                    break;
                }
                case 1: {
                    const auto edge = base + (1 + random() % 64) * Block;
                    range = {edge - 1 - random() % 100, edge + 1 + random() % 100};
                    break;
                }
                case 2: {
                    const auto start = base + (random() % 64) * Block + random() % Block;
                    range = {start, start + (1 + random() % 300) * Block};
                    break;
                }
                case 3:
                    range = {3 * Leaf - 1 - random() % (4 * Block), 3 * Leaf + 1 + random() % (4 * Block)};
                    break;
                case 4:
                    range = {base + 5 * Block + 8, base + 5 * Block + 16};
                    break;
                default:
                    range = {Top - Block + random() % Block, Top + 1 + random() % Block};
                    break;
                }
                enter(blocks, range);
                slot.first.store(range.first, std::memory_order_relaxed);
                slot.last.store(range.second - 1, std::memory_order_relaxed);
                slot.sequence.store(2 * round + 1, std::memory_order_release);
                leave(blocks, hotRange(writer, hotNext - HotHeld));
                enter(blocks, hotRange(writer, hotNext));
                ++hotNext;
                const auto seen = checks.load(std::memory_order_relaxed);
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
                while (checks.load(std::memory_order_relaxed) < seen + Readers && std::chrono::steady_clock::now() < until) {
                }
                slot.sequence.store(2 * round + 2, std::memory_order_release);
                leave(blocks, range);
            }
            hotEnds[writer] = hotNext;
        });
    }
    for (unsigned reader = 0; reader < Readers; ++reader) {
        threads.emplace_back([&, reader] {
            std::mt19937_64 random(0xbeef0000u + reader);
            ready.fetch_add(1);
            while (ready.load() < Writers + Readers) std::this_thread::yield();
            while (!done.load(std::memory_order_acquire)) {
                for (unsigned writer = 0; writer < Writers; ++writer) {
                    auto& slot = published[writer];
                    const auto before = slot.sequence.load(std::memory_order_acquire);
                    if ((before & 1u) == 0) continue;
                    const auto first = slot.first.load(std::memory_order_relaxed);
                    const auto last = slot.last.load(std::memory_order_relaxed);
                    if (last < first) continue;
                    const auto probe = first + random() % (last - first + 1);
                    bool clear = false;
                    switch (random() % 4) {
                    case 0:
                        clear = blocks.Clear(probe, 1);
                        break;
                    case 1: {
                        const auto back = std::min<std::uint64_t>(probe, random() % 256);
                        clear = blocks.Clear(probe - back, back + 1 + random() % 256);
                        break;
                    }
                    case 2: {
                        const auto back = std::min<std::uint64_t>(probe, random() % (64 * Block));
                        clear = blocks.Clear(probe - back, back + 1 + random() % (64 * Block));
                        break;
                    }
                    default: {
                        const std::array<Range, 2> reads{Range{Leaf, Leaf + 1}, Range{probe, probe + 1}};
                        clear = blocks.Clear(reads);
                        break;
                    }
                    }
                    const auto after = slot.sequence.load(std::memory_order_acquire);
                    checks.fetch_add(1, std::memory_order_relaxed);
                    if (after != before) continue;
                    heldChecks.fetch_add(1, std::memory_order_relaxed);
                    if (clear) violations.fetch_add(1, std::memory_order_relaxed);
                }
                hotChecks.fetch_add(1, std::memory_order_relaxed);
                if (blocks.Clear(hot + random() % Block, 1)) hotViolations.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (unsigned writer = 0; writer < Writers; ++writer) threads[writer].join();
    done.store(true, std::memory_order_release);
    for (unsigned reader = Writers; reader < threads.size(); ++reader) threads[reader].join();
    Expect(blocks.Spilled() == 1 && blocks.Holders(hot) == Writers * HotHeld, "the hot block keeps every hot range after the concurrent writers stopped");
    for (unsigned writer = 0; writer < Writers; ++writer) {
        for (std::uint64_t slot = hotEnds[writer] - HotHeld; slot < hotEnds[writer]; ++slot) leave(blocks, hotRange(writer, slot));
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::fprintf(stderr, "concurrent: %llu checks, %llu while a write was held, %llu hot-block checks, %.0f ms\n", static_cast<unsigned long long>(checks.load()), static_cast<unsigned long long>(heldChecks.load()), static_cast<unsigned long long>(hotChecks.load()), elapsed);
    Expect(violations.load() == 0, "no read is clear while a write over it is held, under concurrent writers and readers");
    Expect(hotViolations.load() == 0, "a spilled block shared by concurrent writers is never clear");
    Expect(heldChecks.load() >= Rounds / 10, "readers checked many held writes");
    Expect(hotChecks.load() > 0, "readers checked the spilled block");
    Expect(blocks.Ranges() == 0 && blocks.Spilled() == 0, "every concurrent range left");
    Expect(blocks.Clear(0, Max), "the table is clear after every concurrent writer left");
    Expect(blocks.Holders(hot) == 0 && blocks.Holders(base + 5 * Block) == 0 && blocks.Holders(Top) == 0, "no holder remains after every concurrent writer left");
}

}

int main() {
    testEmpty();
    testOneBlock();
    testTwoBlocks();
    testLargeRanges();
    testTop();
    testMultiplicity();
    testOverflow();
    testConcurrent();
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("agc_core_write_blocks_tests passed");
    return 0;
}
