#include "BlockAbi.hpp"
#include <cstdint>
#include <cstdio>
#include <type_traits>

using namespace ShaderRecompiler;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

static_assert(std::is_same_v<decltype(BlockAbi::Version), const std::uint32_t>);
static_assert(std::is_same_v<decltype(BlockAbi::SectionBytes(0, 0)), std::uint32_t>);
static_assert(BlockAbi::Version == 1);
static_assert(BlockAbi::SectionAlignmentBytes == 16);
static_assert((BlockAbi::SectionAlignmentBytes & (BlockAbi::SectionAlignmentBytes - 1u)) == 0);
static_assert(BlockAbi::SectionAlignmentBytes % sizeof(std::uint32_t) == 0);
static_assert(BlockAbi::MaxWords == 1024);
static_assert(BlockAbi::MaxBuffers == 64);
static_assert(BlockAbi::BufferEntryWords == 4);
static_assert(BlockAbi::BufferEntryWords * sizeof(std::uint32_t) == BlockAbi::SectionAlignmentBytes);
static_assert(BlockAbi::EntryAddressLow == 0 && BlockAbi::EntryAddressHigh == 1 && BlockAbi::EntryBytes == 2 && BlockAbi::EntryService == 3);
static_assert(BlockAbi::EntryService + 1u == BlockAbi::BufferEntryWords);
static_assert(BlockAbi::ServiceMisalignmentMask == 3u);
static_assert(BlockAbi::ServiceAbsent == 0x80000000u);
static_assert((BlockAbi::ServiceMisalignmentMask & BlockAbi::ServiceAbsent) == 0);
static_assert(BlockAbi::PushAddressDwords == 2);
static_assert(BlockAbi::SectionBytes(BlockAbi::MaxWords, BlockAbi::MaxBuffers) == 5120);

void testConstants() {
    Expect(BlockAbi::Version == 1, "the block format is version 1");
    Expect(BlockAbi::SectionAlignmentBytes == 16, "a section is aligned on 16 bytes");
    Expect(BlockAbi::MaxWords == 1024 && BlockAbi::MaxBuffers == 64, "a section holds at most 1024 words and 64 buffers");
    Expect(BlockAbi::MaxBuffers <= 64, "one bit per buffer fits a 64-bit deferred mask");
    Expect(BlockAbi::BufferEntryWords == 4, "a buffer entry is four words");
    Expect(BlockAbi::EntryAddressLow == 0 && BlockAbi::EntryAddressHigh == 1 && BlockAbi::EntryBytes == 2 && BlockAbi::EntryService == 3, "entry fields are address low, address high, bytes, service");
    Expect(BlockAbi::ServiceMisalignmentMask == 3u && BlockAbi::ServiceAbsent == 0x80000000u, "service word: misalignment in the low two bits, absence in the top bit");
    Expect(BlockAbi::PushAddressDwords == 2, "the section address takes two pushed words");
}

void testBufferEntryWord() {
    Expect(BlockAbi::BufferEntryWord(0, 0) == 0, "without words the first entry starts the section");
    Expect(BlockAbi::BufferEntryWord(5, 0) == 5, "the first entry follows the words");
    Expect(BlockAbi::BufferEntryWord(5, 1) == 9 && BlockAbi::BufferEntryWord(5, 3) == 17, "entries are four words apart");
    Expect(BlockAbi::BufferEntryWord(BlockAbi::MaxWords, BlockAbi::MaxBuffers - 1u) == 1276, "last entry of the largest section");
    Expect(BlockAbi::BufferEntryWord(7, 2) + BlockAbi::EntryService == 18, "a field is the entry word plus its index");
}

void testSectionWords() {
    Expect(BlockAbi::SectionWords(0, 0) == 0, "an empty section has no word");
    Expect(BlockAbi::SectionWords(5, 0) == 5 && BlockAbi::SectionWords(0, 3) == 12 && BlockAbi::SectionWords(5, 3) == 17, "words then four per buffer");
    Expect(BlockAbi::SectionWords(BlockAbi::MaxWords, BlockAbi::MaxBuffers) == 1280, "largest section in words");
}

void testSectionBytes() {
    Expect(BlockAbi::SectionBytes(0, 0) == 0, "an empty section takes no byte");
    Expect(BlockAbi::SectionBytes(1, 0) == 16 && BlockAbi::SectionBytes(3, 0) == 16 && BlockAbi::SectionBytes(4, 0) == 16, "up to four words round to 16 bytes");
    Expect(BlockAbi::SectionBytes(5, 0) == 32, "a fifth word opens the next 16 bytes");
    Expect(BlockAbi::SectionBytes(0, 1) == 16 && BlockAbi::SectionBytes(0, 2) == 32, "a buffer entry is exactly 16 bytes");
    Expect(BlockAbi::SectionBytes(3, 2) == 48, "44 bytes round to 48");
    Expect(BlockAbi::SectionBytes(BlockAbi::MaxWords, BlockAbi::MaxBuffers) == 5120, "largest section in bytes");
}

void testEveryShape() {
    bool aligned = true;
    bool covers = true;
    bool tight = true;
    bool contiguous = true;
    bool closed = true;
    for (std::uint32_t words = 0; words <= BlockAbi::MaxWords; ++words) {
        for (std::uint32_t buffers = 0; buffers <= BlockAbi::MaxBuffers; ++buffers) {
            const std::uint32_t sectionWords = BlockAbi::SectionWords(words, buffers);
            const std::uint32_t bytes = BlockAbi::SectionBytes(words, buffers);
            aligned = aligned && bytes % BlockAbi::SectionAlignmentBytes == 0;
            covers = covers && bytes >= sectionWords * 4u;
            tight = tight && bytes - sectionWords * 4u < BlockAbi::SectionAlignmentBytes;
            closed = closed && BlockAbi::BufferEntryWord(words, buffers) == sectionWords;
            for (std::uint32_t buffer = 0; buffer < buffers; ++buffer) {
                contiguous = contiguous && BlockAbi::BufferEntryWord(words, buffer) == words + buffer * BlockAbi::BufferEntryWords;
                contiguous = contiguous && BlockAbi::BufferEntryWord(words, buffer + 1u) - BlockAbi::BufferEntryWord(words, buffer) == BlockAbi::BufferEntryWords;
            }
        }
    }
    Expect(aligned, "every section size is a multiple of the alignment");
    Expect(covers, "every section size covers its words and entries");
    Expect(tight, "padding stays below one alignment unit");
    Expect(contiguous, "entries follow the words without a gap");
    Expect(closed, "the entry past the last one ends the section");
}

}

int main() {
    testConstants();
    testBufferEntryWord();
    testSectionWords();
    testSectionBytes();
    testEveryShape();
    if (failures != 0) {
        std::fprintf(stderr, "%d block ABI check(s) failed\n", failures);
        return 1;
    }
    std::printf("block ABI checks passed\n");
    return 0;
}
