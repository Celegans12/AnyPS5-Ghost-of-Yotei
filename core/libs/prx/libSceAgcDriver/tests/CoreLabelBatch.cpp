#include "prx/libSceAgcDriver/Core/include/LabelBatch.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

struct Random {
    std::uint64_t state = 0;

    std::uint64_t Next() {
        auto value = state += 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31u);
    }
    std::uint32_t Below(std::uint32_t limit) { return static_cast<std::uint32_t>(Next() % limit); }
    bool OneIn(std::uint32_t chances) { return Below(chances) == 0; }
};

using BlockList = std::vector<std::uint64_t>;

constexpr std::uint64_t Block = std::uint64_t{1} << 16u;
constexpr std::uint64_t Most = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint64_t LastBlock = Most & ~(Block - 1u);

BlockList listed(const LabelBatch& batch) {
    const std::span<const std::uint64_t> blocks = batch.Blocks();
    BlockList sorted(blocks.begin(), blocks.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

bool distinct(const LabelBatch& batch) {
    const BlockList sorted = listed(batch);
    return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

bool holds(const LabelBatch& batch, std::uint64_t block) {
    const std::span<const std::uint64_t> blocks = batch.Blocks();
    return std::find(blocks.begin(), blocks.end(), block) != blocks.end();
}

bool countsAre(const LabelBatch& batch, std::uint64_t labels, std::uint64_t first, std::uint64_t repeated, std::uint64_t overflowed) {
    const LabelBatch::Counts counts = batch.Counted();
    return counts.labels == labels && counts.first == first && counts.repeated == repeated && counts.overflowed == overflowed;
}

void testEmpty() {
    LabelBatch batch;
    Expect(batch.Blocks().empty(), "a new batch holds no block");
    Expect(countsAre(batch, 0, 0, 0, 0), "a new batch has counted nothing");
    batch.Close();
    Expect(batch.Blocks().empty() && countsAre(batch, 0, 0, 0, 0), "closing an empty batch keeps it empty");
}

void testFirstAndRepeated() {
    LabelBatch batch;
    Expect(batch.First(0x12340010ull, 8), "the first label of a batch in a block is first");
    Expect(!batch.First(0x12340010ull, 8), "the same label again is repeated");
    Expect(!batch.First(0x1234fff0ull, 16), "a label at the end of the same block is repeated");
    Expect(!batch.First(0x12340000ull, 4), "a label at the start of the same block is repeated");
    Expect(listed(batch) == BlockList{0x12340000ull}, "the batch holds the block once, by its base address");
    Expect(batch.First(0x12350000ull, 4), "a label in the next block is first");
    Expect(!batch.First(0x1235abcdull, 2), "a second label in the next block is repeated");
    Expect(listed(batch) == BlockList{0x12340000ull, 0x12350000ull}, "the batch holds the two blocks");
    Expect(countsAre(batch, 6, 2, 4, 0), "labels, first and repeated are counted");
    for (const std::uint64_t block : batch.Blocks()) Expect((block & (Block - 1u)) == 0, "every block is listed by its 64 KiB aligned base");
}

void testStraddling() {
    LabelBatch batch;
    Expect(batch.First(0x2fff8ull, 16), "a label across two blocks is first");
    Expect(listed(batch) == BlockList{0x20000ull, 0x30000ull}, "a label across two blocks adds both blocks");
    Expect(!batch.First(0x30000ull, 8), "a label in the second block is then repeated");
    Expect(!batch.First(0x2fff0ull, 16), "a label at the end of the first block is then repeated");
    Expect(!batch.First(0x2fffcull, 8), "the same two blocks again are repeated");
    Expect(batch.First(0x3fffcull, 8), "a label across a seen block and a new one is first");
    Expect(listed(batch) == BlockList{0x20000ull, 0x30000ull, 0x40000ull}, "only the new block is added");
    Expect(distinct(batch), "a straddling label adds no duplicate");
    Expect(countsAre(batch, 5, 2, 3, 0), "straddling labels are counted once each");

    LabelBatch edges;
    Expect(edges.First(0x50000ull, Block), "a range of one whole block is first");
    Expect(listed(edges) == BlockList{0x50000ull}, "a range that ends on the last byte of its block covers that block only");
    Expect(edges.First(0x5ffffull, 2), "a range that steps one byte into the next block is first");
    Expect(listed(edges) == BlockList{0x50000ull, 0x60000ull}, "the byte past the boundary adds the next block");
    Expect(!edges.First(0x50000ull, 0), "an empty range in a seen block is repeated");
    Expect(edges.First(0x70000ull, 0), "an empty range counts as the block of its address");
    Expect(holds(edges, 0x70000ull), "the block of an empty range is retained");
    Expect(edges.First(Most - 7u, 16), "a range past the end of the address space is first");
    Expect(holds(edges, LastBlock), "a range past the end of the address space covers the last block");
    Expect(!edges.First(Most, 1), "the last byte of the address space is then repeated");
    Expect(edges.Blocks().size() == 4 && distinct(edges), "the edge cases hold four distinct blocks");
}

void testOverflow() {
    constexpr std::uint64_t Origin = 0x100000000ull;
    constexpr std::uint64_t Stride = 3u * Block;
    LabelBatch batch;
    BlockList expected;
    bool all = true;
    for (std::uint64_t index = 0; index < LabelBatch::Blocks64k; ++index) {
        all = batch.First(Origin + index * Stride + 8u, 16) && all;
        expected.push_back(Origin + index * Stride);
    }
    Expect(all, "64 labels in 64 blocks are all first");
    Expect(batch.Blocks().size() == LabelBatch::Blocks64k && listed(batch) == expected, "the batch holds the 64 blocks");
    Expect(countsAre(batch, 64, 64, 0, 0), "64 first labels are counted");
    const std::uint64_t extra = Origin + LabelBatch::Blocks64k * Stride;
    Expect(batch.First(extra, 8), "the 65th block is reported first, on the safe side");
    Expect(countsAre(batch, 65, 64, 0, 1), "the 65th block is counted as overflowed");
    Expect(!holds(batch, extra) && listed(batch) == expected, "Blocks() still lists the 64 retained blocks after the 65th");
    Expect(batch.First(extra + 4u, 4), "a label in the block left out is first again");
    Expect(countsAre(batch, 66, 64, 0, 2), "each label in a block left out is counted as overflowed");
    Expect(!batch.First(Origin + 5u * Stride + 32u, 8), "a label in a retained block is repeated in a full batch");
    Expect(countsAre(batch, 67, 64, 1, 2), "a repeated label in a full batch is counted as repeated");
    Expect(listed(batch) == expected && distinct(batch), "a full batch keeps its blocks without duplicates");
    batch.Close();
    Expect(batch.Blocks().empty(), "Close empties a full batch");
    Expect(batch.First(extra, 8) && listed(batch) == BlockList{extra}, "after Close the block left out is first and retained");

    LabelBatch edge;
    bool filled = true;
    for (std::uint64_t index = 0; index + 1u < LabelBatch::Blocks64k; ++index) filled = edge.First(index * Block, 4) && filled;
    Expect(filled && edge.Blocks().size() == LabelBatch::Blocks64k - 1u, "63 labels fill 63 blocks");
    Expect(edge.First(LabelBatch::Blocks64k * Block - 8u, 16), "a label across two new blocks with room for one is first");
    Expect(edge.Blocks().size() == LabelBatch::Blocks64k && holds(edge, (LabelBatch::Blocks64k - 1u) * Block) && !holds(edge, LabelBatch::Blocks64k * Block), "the last room takes the lower block and the upper one is left out");
    Expect(countsAre(edge, 64, 63, 0, 1), "a label that only partly fits is counted as overflowed");
    Expect(!edge.First((LabelBatch::Blocks64k - 1u) * Block + 100u, 4), "the block that fitted is then repeated");
    Expect(edge.First(LabelBatch::Blocks64k * Block + 100u, 4), "the block left out is still first");
}

void testWideRanges() {
    constexpr std::uint64_t Origin = 0x800000000ull;
    LabelBatch batch;
    Expect(batch.First(Origin, 100u * Block), "a range over 100 blocks is first");
    BlockList expected;
    for (std::uint64_t index = 0; index < LabelBatch::Blocks64k; ++index) expected.push_back(Origin + index * Block);
    Expect(listed(batch) == expected, "a range over 100 blocks retains its first 64 blocks");
    Expect(countsAre(batch, 1, 0, 0, 1), "a range over more blocks than the table is counted as overflowed");

    LabelBatch whole;
    Expect(whole.First(0, Most), "the whole address space is first");
    Expect(whole.Blocks().size() == LabelBatch::Blocks64k && distinct(whole) && countsAre(whole, 1, 0, 0, 1), "the whole address space stops at the table size");

    LabelBatch seen;
    Expect(seen.First(Origin, 10u * Block), "a range over 10 blocks is first");
    Expect(seen.Blocks().size() == 10, "a range over 10 blocks adds 10 blocks");
    Expect(!seen.First(Origin + 2u * Block + 1u, 3u * Block), "a range inside seen blocks is repeated");
    Expect(seen.First(Origin + 9u * Block, 2u * Block), "a range that runs one block past the seen ones is first");
    Expect(seen.Blocks().size() == 11 && distinct(seen) && countsAre(seen, 3, 2, 1, 0), "only the block past the seen ones is added");
}

void testClose() {
    LabelBatch batch;
    Expect(batch.First(0x10000ull, 8), "a first label before Close");
    Expect(batch.First(0x20000ull, 8), "a second first label before Close");
    Expect(!batch.First(0x10008ull, 8), "a repeated label before Close");
    batch.Close();
    Expect(batch.Blocks().empty(), "Close empties the batch");
    Expect(batch.First(0x10000ull, 8), "a block seen before Close is first in the next batch");
    Expect(listed(batch) == BlockList{0x10000ull}, "the next batch holds only its own blocks");
    Expect(countsAre(batch, 4, 3, 1, 0), "the counts run on across Close");
    batch.Close();
    batch.Close();
    Expect(batch.Blocks().empty(), "closing twice is harmless");
    Expect(batch.First(0x20000ull, 8) && !batch.First(0x20000ull, 8), "a batch works again after two Close");
}

void testRandomAgainstModel() {
    Random random{0x1abe1u};
    LabelBatch batch;
    BlockList model;
    std::uint64_t labels = 0;
    std::uint64_t first = 0;
    std::uint64_t repeated = 0;
    std::uint64_t overflowed = 0;
    std::size_t wrongAnswers = 0;
    std::size_t wrongBlocks = 0;
    std::size_t wrongCounts = 0;
    std::size_t straddling = 0;
    std::size_t fullBatches = 0;
    for (std::uint32_t round = 0; round < 400; ++round) {
        const std::uint64_t origin = (random.Next() & 0xffffffffull) << 16u;
        const std::uint64_t window = 4u + random.Below(197);
        const std::uint32_t count = 1u + random.Below(300);
        bool full = false;
        for (std::uint32_t label = 0; label < count; ++label) {
            const std::uint64_t address = origin + random.Next() % (window * Block);
            const std::uint64_t bytes = random.OneIn(16) ? 1u + random.Next() % (3u * Block) : 1u + random.Below(16);
            const std::uint64_t low = address & ~(Block - 1u);
            const std::uint64_t high = (address + bytes - 1u) & ~(Block - 1u);
            if (high != low) ++straddling;
            BlockList missing;
            for (std::uint64_t block = low; block <= high; block += Block) {
                if (std::find(model.begin(), model.end(), block) == model.end()) missing.push_back(block);
            }
            bool expected = true;
            ++labels;
            if (missing.empty()) {
                expected = false;
                ++repeated;
            } else if (model.size() + missing.size() <= LabelBatch::Blocks64k) {
                model.insert(model.end(), missing.begin(), missing.end());
                ++first;
            } else {
                const std::size_t room = LabelBatch::Blocks64k - model.size();
                model.insert(model.end(), missing.begin(), missing.begin() + static_cast<std::ptrdiff_t>(room));
                ++overflowed;
                full = true;
            }
            if (batch.First(address, bytes) != expected) ++wrongAnswers;
            BlockList sortedModel = model;
            std::sort(sortedModel.begin(), sortedModel.end());
            if (listed(batch) != sortedModel || !distinct(batch) || batch.Blocks().size() > LabelBatch::Blocks64k) ++wrongBlocks;
        }
        if (full) ++fullBatches;
        if (!countsAre(batch, labels, first, repeated, overflowed) || labels != first + repeated + overflowed) ++wrongCounts;
        batch.Close();
        model.clear();
        if (!batch.Blocks().empty()) ++wrongBlocks;
    }
    Expect(wrongAnswers == 0, "random labels: First is true exactly when a covered block is new to the batch or left out of a full one");
    Expect(wrongBlocks == 0, "random labels: Blocks() lists the retained blocks, without duplicates, at most 64, and nothing after Close");
    Expect(wrongCounts == 0, "random labels: every label is counted once as first, repeated or overflowed");
    Expect(straddling > 0 && fullBatches > 0 && repeated > 0, "random labels cross blocks, fill batches and repeat blocks");
}

}

int main() {
    testEmpty();
    testFirstAndRepeated();
    testStraddling();
    testOverflow();
    testWideRanges();
    testClose();
    testRandomAgainstModel();
    if (failures != 0) {
        std::fprintf(stderr, "%d label batch checks failed\n", failures);
        return 1;
    }
    std::printf("label batch tests passed\n");
    return 0;
}
