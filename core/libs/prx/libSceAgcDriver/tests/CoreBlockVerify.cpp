#include "prx/libSceAgcDriver/Core/include/BlockVerify.hpp"
#include "BlockAbi.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

using namespace AgcDriver::Core;

namespace {

namespace Abi = ShaderRecompiler::BlockAbi;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::uint32_t WordCount = 6;
constexpr std::uint64_t FirstAddress = 0x700000001000ull;
constexpr std::uint64_t SecondAddress = 0x700000002000ull;

struct Shadow {
    std::uint64_t address = 0;
    std::vector<std::byte> bytes;
};

struct Memory {
    std::vector<Shadow> regions;
};

std::span<const std::byte> shadowAt(void* context, std::uint64_t deviceAddress, std::uint32_t bytes) {
    const Memory& memory = *static_cast<const Memory*>(context);
    for (const Shadow& region : memory.regions) {
        if (deviceAddress < region.address || deviceAddress - region.address > region.bytes.size()) continue;
        const std::size_t offset = static_cast<std::size_t>(deviceAddress - region.address);
        const std::size_t available = region.bytes.size() - offset;
        return std::span<const std::byte>(region.bytes.data() + offset, available < bytes ? available : bytes);
    }
    return {};
}

std::vector<std::byte> pattern(std::size_t count, std::uint8_t seed) {
    std::vector<std::byte> bytes(count);
    for (std::size_t i = 0; i < count; ++i) bytes[i] = static_cast<std::byte>(static_cast<std::uint8_t>(seed + i * 7u));
    return bytes;
}

struct Scene {
    ShaderRecompiler::BlockLayout layout;
    std::vector<std::uint32_t> flat;
    std::vector<std::uint32_t> section;
    std::vector<BufferTruth> truth;
    Memory memory;
    std::vector<std::byte> classicFirst;
    std::vector<std::byte> classicSecond;

    Scene() {
        layout.words = WordCount;
        layout.buffers.resize(3);
        flat = {0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u, 0x55555555u, 0x66666666u};
        memory.regions.push_back({FirstAddress, pattern(64, 1)});
        memory.regions.push_back({SecondAddress, pattern(32, 9)});
        truth.resize(3);
        classicFirst = pattern(64, 1);
        classicSecond = pattern(32, 9);
        truth[0] = {FirstAddress, std::span<const std::byte>(classicFirst), true};
        truth[1] = {SecondAddress, std::span<const std::byte>(classicSecond), true};
        truth[2] = {0, {}, false};
        section.assign(Abi::SectionWords(WordCount, 3), 0);
        for (std::uint32_t k = 0; k < WordCount; ++k) section[k] = flat[k];
        setEntry(0, FirstAddress, 64, 0);
        setEntry(1, SecondAddress, 32, 0);
        setEntry(2, 0, 0, Abi::ServiceAbsent);
    }

    void setEntry(std::uint32_t buffer, std::uint64_t address, std::uint32_t bytes, std::uint32_t service) {
        const std::uint32_t first = Abi::BufferEntryWord(WordCount, buffer);
        section[first + Abi::EntryAddressLow] = static_cast<std::uint32_t>(address);
        section[first + Abi::EntryAddressHigh] = static_cast<std::uint32_t>(address >> 32u);
        section[first + Abi::EntryBytes] = bytes;
        section[first + Abi::EntryService] = service;
    }

    std::vector<BlockMismatch> verify() const {
        return VerifyBlock(layout, section, flat, truth, &shadowAt, const_cast<Memory*>(&memory));
    }
};

void testRight() {
    Scene scene;
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.empty(), "a right section has no mismatch");
}

void testWord() {
    Scene scene;
    scene.section[3] ^= 0x100u;
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1, "one wrong word gives one mismatch");
    if (mismatches.size() != 1) return;
    const BlockMismatch& mismatch = mismatches[0];
    Expect(mismatch.kind == BlockMismatch::Kind::Word, "a wrong word is a word mismatch");
    Expect(mismatch.index == 3 && mismatch.expected == 0x44444444u && mismatch.found == (0x44444444u ^ 0x100u), "the word mismatch names the index, the expected and the found word");
    const std::string text = Describe(mismatch);
    Expect(text.find("word 3") != std::string::npos && text.find("0x44444444") != std::string::npos && text.find("0x44444544") != std::string::npos, "the word text carries the index and both values");
}

void testWordsLastAndSeveral() {
    Scene scene;
    scene.section[0] = 0;
    scene.section[WordCount - 1] = 1;
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 2 && mismatches[0].index == 0 && mismatches[1].index == WordCount - 1, "each wrong word is reported, in order");
}

void testPresence() {
    Scene absentFound;
    absentFound.setEntry(1, 0, 0, Abi::ServiceAbsent);
    std::vector<BlockMismatch> mismatches = absentFound.verify();
    Expect(mismatches.size() == 1, "a buffer marked absent that the classic path binds gives one mismatch");
    if (mismatches.size() == 1) {
        Expect(mismatches[0].kind == BlockMismatch::Kind::Presence && mismatches[0].index == 1 && mismatches[0].expected == 1 && mismatches[0].found == 0, "the missing buffer is a presence mismatch");
        const std::string text = Describe(mismatches[0]);
        Expect(text.find("presence") != std::string::npos && text.find("buffer 1") != std::string::npos && text.find("present") != std::string::npos, "the presence text names the buffer");
    }
    Scene presentFound;
    presentFound.setEntry(2, FirstAddress, 64, 0);
    mismatches = presentFound.verify();
    Expect(mismatches.size() == 1 && mismatches[0].kind == BlockMismatch::Kind::Presence && mismatches[0].index == 2 && mismatches[0].expected == 0 && mismatches[0].found == 1, "a buffer the classic path leaves out but the block binds is a presence mismatch");
    Scene nullAddress;
    nullAddress.setEntry(0, 0, 64, 0);
    mismatches = nullAddress.verify();
    Expect(mismatches.size() == 1 && mismatches[0].kind == BlockMismatch::Kind::Presence && mismatches[0].index == 0, "a null address counts as absent");
    Scene bothAbsent;
    bothAbsent.setEntry(2, 0, 0, Abi::ServiceAbsent | 1u);
    Expect(bothAbsent.verify().empty(), "an entry absent on both sides is not compared further");
}

void testBound() {
    Scene scene;
    scene.setEntry(0, FirstAddress, 48, 0);
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1, "a wrong bound gives one mismatch");
    if (mismatches.size() != 1) return;
    Expect(mismatches[0].kind == BlockMismatch::Kind::Bound && mismatches[0].index == 0 && mismatches[0].expected == 64 && mismatches[0].found == 48, "a wrong bound is a bound mismatch with both sizes");
    const std::string text = Describe(mismatches[0]);
    Expect(text.find("bound") != std::string::npos && text.find("64") != std::string::npos && text.find("48") != std::string::npos, "the bound text carries both sizes");
}

void testAddress() {
    Scene scene;
    scene.setEntry(1, SecondAddress + 16, 32, 0);
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1 && mismatches[0].kind == BlockMismatch::Kind::Word && mismatches[0].index == 1 && mismatches[0].expected == SecondAddress && mismatches[0].found == SecondAddress + 16, "a wrong address is one word mismatch and the content is not compared at it");
}

void testContent() {
    Scene scene;
    scene.memory.regions[1].bytes[20] = static_cast<std::byte>(0xeeu);
    const std::vector<std::byte>& classic = scene.classicSecond;
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1, "one wrong byte gives one mismatch");
    if (mismatches.size() != 1) return;
    const BlockMismatch& mismatch = mismatches[0];
    Expect(mismatch.kind == BlockMismatch::Kind::Content && mismatch.index == 20, "a wrong byte is a content mismatch at its offset");
    Expect(mismatch.expected == std::to_integer<std::uint8_t>(classic[20]) && mismatch.found == 0xeeu, "the content mismatch carries both bytes");
    const std::string text = Describe(mismatch);
    Expect(text.find("content") != std::string::npos && text.find("byte 20") != std::string::npos && text.find("0xee") != std::string::npos, "the content text names the offset and the byte");
}

void testBytesUnavailable() {
    Scene scene;
    scene.memory.regions[0].bytes.resize(40);
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1 && mismatches[0].kind == BlockMismatch::Kind::Bytes && mismatches[0].index == 0 && mismatches[0].expected == 64 && mismatches[0].found == 40, "a shadow shorter than the classic buffer is a bytes mismatch");
    Scene noReader;
    const std::vector<BlockMismatch> none = VerifyBlock(noReader.layout, noReader.section, noReader.flat, noReader.truth, nullptr, nullptr);
    Expect(none.size() == 2 && none[0].kind == BlockMismatch::Kind::Bytes && none[1].kind == BlockMismatch::Kind::Bytes, "without a reader every present buffer is a bytes mismatch");
    const std::string text = Describe(mismatches[0]);
    Expect(text.find("bytes") != std::string::npos && text.find("64") != std::string::npos && text.find("40") != std::string::npos, "the bytes text carries both sizes");
}

void testShortSection() {
    Scene scene;
    scene.section.resize(scene.section.size() - 1);
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1 && mismatches[0].kind == BlockMismatch::Kind::Bytes && mismatches[0].expected == Abi::SectionWords(WordCount, 3) && mismatches[0].found == Abi::SectionWords(WordCount, 3) - 1, "a section shorter than its layout is one bytes mismatch");
}

void testShortClassic() {
    Scene scene;
    scene.flat.resize(WordCount - 2);
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 1 && mismatches[0].kind == BlockMismatch::Kind::Word && mismatches[0].index == WordCount && mismatches[0].expected == WordCount - 2 && mismatches[0].found == WordCount, "a classic capture with fewer words than the layout is one word mismatch");
    Scene fewer;
    fewer.truth.resize(1);
    const std::vector<BlockMismatch> missing = fewer.verify();
    Expect(missing.size() == 1 && missing[0].kind == BlockMismatch::Kind::Presence && missing[0].index == 1, "a buffer the classic path did not report is absent for it");
}

void testIndependentFaults() {
    Scene scene;
    scene.section[1] ^= 1u;
    scene.setEntry(0, FirstAddress, 60, 0);
    scene.memory.regions[1].bytes[0] = static_cast<std::byte>(0x00u);
    scene.setEntry(2, SecondAddress, 4, 0);
    const std::vector<BlockMismatch> mismatches = scene.verify();
    Expect(mismatches.size() == 4, "four independent faults give four mismatches");
    if (mismatches.size() != 4) return;
    Expect(mismatches[0].kind == BlockMismatch::Kind::Word && mismatches[1].kind == BlockMismatch::Kind::Bound && mismatches[2].kind == BlockMismatch::Kind::Content && mismatches[3].kind == BlockMismatch::Kind::Presence, "the mismatches come in section order, each of its own kind");
}

}

int main() {
    testRight();
    testWord();
    testWordsLastAndSeveral();
    testPresence();
    testBound();
    testAddress();
    testContent();
    testBytesUnavailable();
    testShortSection();
    testShortClassic();
    testIndependentFaults();
    if (failures != 0) {
        std::fprintf(stderr, "%d block verify checks failed\n", failures);
        return 1;
    }
    std::printf("block verify tests passed\n");
    return 0;
}
