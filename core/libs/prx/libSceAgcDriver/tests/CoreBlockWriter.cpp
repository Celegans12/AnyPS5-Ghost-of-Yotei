#include "prx/libSceAgcDriver/Core/include/BlockWriter.hpp"
#include "BlockAbi.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/BlockLayout.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <new>
#include <span>
#include <vector>

using namespace AgcDriver::Core;
using namespace ShaderRecompiler;

namespace {

int failures = 0;
std::uint64_t allocations = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::uint32_t NoWord = 0xffffffffu;
constexpr std::uint64_t TableAddress = 0x0000000500004000ull;
constexpr std::uint64_t DeviceBase = 0x0000004000000000ull;
constexpr std::uint64_t MirrorBase = 0x0000100000000000ull;
constexpr std::size_t ArenaAlignment = 64;
constexpr std::uint32_t FormatMask = 0x7fu << 12u;

constexpr std::uint32_t SceneWords = 13;
constexpr std::uint32_t SceneBuffers = 3;
constexpr std::uint32_t Direct = 0;
constexpr std::uint32_t Tabled = 1;
constexpr std::uint32_t Missing = 2;
constexpr std::uint64_t DirectBase = 0x00000abc1234567aull;
constexpr std::uint32_t DirectStride = 16;
constexpr std::uint32_t DirectRecords = 10;
constexpr std::uint64_t TabledBase = 0x0000000200001000ull;
constexpr std::uint32_t TabledBytes = 0x300;

struct Host {
    std::array<std::uint32_t, 4> table{};
    bool readable = true;
    std::uint32_t reads = 0;
    std::uint64_t refused = 0;
    std::uint64_t deferred = 0;
    std::uint32_t resolved = 0;
    std::array<BufferRequest, BlockAbi::MaxBuffers> requests{};
};

bool readWords(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count) {
    auto& host = *static_cast<Host*>(context);
    ++host.reads;
    if (!host.readable) return false;
    for (std::uint32_t word = 0; word < count; ++word) {
        const auto at = address + std::uint64_t{word} * sizeof(std::uint32_t);
        if (at < TableAddress || (at - TableAddress) % sizeof(std::uint32_t) != 0u || (at - TableAddress) / sizeof(std::uint32_t) >= host.table.size()) return false;
        out[word] = host.table[static_cast<std::size_t>((at - TableAddress) / sizeof(std::uint32_t))];
    }
    return true;
}

bool resolveBuffer(void* context, const BufferRequest& request, BufferAnswer& answer) {
    auto& host = *static_cast<Host*>(context);
    if (host.resolved < host.requests.size()) host.requests[host.resolved] = request;
    ++host.resolved;
    const auto bit = std::uint64_t{1} << (request.buffer & 63u);
    if ((host.refused & bit) != 0u) return false;
    if ((host.deferred & bit) != 0u) {
        answer.deferred = true;
        return true;
    }
    answer.deviceAddress = MirrorBase + request.address;
    answer.bytes = static_cast<std::uint32_t>(request.bytes);
    return true;
}

struct Arena {
    std::shared_ptr<void> memory;
    std::size_t bytes = 0;
    std::uint64_t chunks = 0;
    bool executed = true;
};

RingChunk allocateChunk(void* context, std::size_t bytes) {
    auto& arena = *static_cast<Arena*>(context);
    if (arena.memory == nullptr) {
        arena.memory = std::shared_ptr<void>(::operator new(bytes, std::align_val_t{ArenaAlignment}), [](void* block) { ::operator delete(block, std::align_val_t{ArenaAlignment}); });
        arena.bytes = bytes;
        std::memset(arena.memory.get(), 0xa5, bytes);
    }
    if (bytes > arena.bytes) return {};
    RingChunk chunk;
    chunk.bytes = static_cast<std::byte*>(arena.memory.get());
    chunk.deviceAddress = DeviceBase + arena.chunks * BlockRing::ChunkBytes;
    chunk.owner = arena.memory;
    ++arena.chunks;
    return chunk;
}

bool chunkExecuted(void* context, std::uint64_t) {
    return static_cast<Arena*>(context)->executed;
}

RingSource sourceOf(Arena& arena) {
    RingSource source;
    source.allocate = allocateChunk;
    source.executed = chunkExecuted;
    source.context = &arena;
    return source;
}

struct Values {
    std::vector<std::unique_ptr<IrValue>> owned;

    IrValue& Make(IrOpcode opcode, IrType type, std::initializer_list<IrValue*> arguments = {}) {
        owned.push_back(std::make_unique<IrValue>(opcode, type, static_cast<std::uint32_t>(owned.size())));
        auto& value = *owned.back();
        for (auto* argument : arguments) value.AddArgument(argument);
        return value;
    }

    IrValue& Constant(std::uint32_t immediate) {
        auto& value = Make(IrOpcode::Void, IrType::U32);
        value.SetImmediateU32(immediate);
        return value;
    }

    IrValue& UserData(std::uint32_t index) {
        auto& reg = Make(IrOpcode::Void, IrType::ScalarReg);
        reg.SetRegister({RegisterBank::Scalar, index});
        return Make(IrOpcode::GetUserData, IrType::U32, {&reg});
    }
};

struct Scene {
    Values values;
    IrResourcePlan plan;
    BlockLayout layout;
    std::array<std::uint32_t, 16> userData{};
    Host host;
    const Detail::SrtTape* tape = nullptr;

    Scene() {
        plan.srtPlanComplete = true;
        plan.resourceTrackingComplete = true;
        MemoryInfo memory;
        memory.kind = ResourceKind::ScalarAddress;
        plan.memoryInfo.push_back(memory);
        for (std::uint32_t word = 0; word < 4u; ++word) plan.srtReads.push_back({&values.UserData(word), word});
        auto& pointer = values.Make(IrOpcode::CompositeConstructU64, IrType::U64, {&values.UserData(4), &values.UserData(5)});
        for (std::uint32_t word = 0; word < 4u; ++word) {
            auto& read = values.Make(IrOpcode::LoadAddressU32, IrType::U32);
            read.SetFlags(MemoryFlags{0, 0});
            read.AddArgument(&pointer);
            read.AddArgument(&values.Constant(word * 4u));
            plan.srtReads.push_back({&read, 4u + word});
        }
        plan.srtReads.push_back({&values.UserData(6), 8u});
        for (std::uint32_t word = 0; word < 4u; ++word) plan.srtReads.push_back({&values.UserData(8u + word), 9u + word});
        tape = Detail::SrtTape::For(plan);

        layout.words = SceneWords;
        layout.buffers.resize(SceneBuffers);
        layout.buffers[Direct].resource = Direct;
        layout.buffers[Direct].descriptorWords = {0u, 1u, 2u, 3u};
        layout.buffers[Direct].formatFixed = true;
        layout.buffers[Tabled].resource = Tabled;
        layout.buffers[Tabled].descriptorWords = {4u, 5u, 6u, 7u};
        layout.buffers[Missing].resource = Missing;
        layout.buffers[Missing].descriptorWords = {9u, 10u, 11u, 12u};
        layout.variantKey.push_back({3u, FormatMask});

        userData[0] = static_cast<std::uint32_t>(DirectBase);
        userData[1] = static_cast<std::uint32_t>(DirectBase >> 32u) | (DirectStride << 16u);
        userData[2] = DirectRecords;
        userData[3] = (5u << 12u) | 0xfacu;
        userData[4] = static_cast<std::uint32_t>(TableAddress);
        userData[5] = static_cast<std::uint32_t>(TableAddress >> 32u);
        userData[6] = 0x00decadeu;
        host.table = {static_cast<std::uint32_t>(TabledBase), static_cast<std::uint32_t>(TabledBase >> 32u), TabledBytes, 0xfacu};
    }

    std::array<std::uint32_t, SceneWords> Words() const {
        return {userData[0], userData[1], userData[2], userData[3], host.table[0], host.table[1], host.table[2], host.table[3], userData[6], userData[8], userData[9], userData[10], userData[11]};
    }

    BlockInputs Inputs() {
        const auto words = Words();
        BlockInputs inputs;
        inputs.plan = &plan;
        inputs.layout = &layout;
        inputs.tape = tape;
        inputs.userData = userData;
        inputs.variantKey = BlockVariantKey(layout, words);
        inputs.read = readWords;
        inputs.resolve = resolveBuffer;
        inputs.context = &host;
        return inputs;
    }
};

std::span<const std::uint32_t> sectionOf(const RingSlice& slice, std::size_t words) {
    return {reinterpret_cast<const std::uint32_t*>(slice.bytes), words};
}

bool entryIs(std::span<const std::uint32_t> section, std::uint32_t words, std::uint32_t buffer, std::uint64_t address, std::uint32_t bytes, std::uint32_t service) {
    const auto first = BlockAbi::BufferEntryWord(words, buffer);
    return section[first + BlockAbi::EntryAddressLow] == static_cast<std::uint32_t>(address) && section[first + BlockAbi::EntryAddressHigh] == static_cast<std::uint32_t>(address >> 32u) && section[first + BlockAbi::EntryBytes] == bytes && section[first + BlockAbi::EntryService] == service;
}

bool faultedWith(const BlockOutput& output, BlockFault fault) {
    return output.fault == fault && output.section.bytes == nullptr && output.deferred == 0u;
}

BlockLayout layoutOf(std::array<std::uint32_t, 4> descriptorWords) {
    BlockLayout layout;
    layout.words = 8;
    layout.buffers.resize(1);
    layout.buffers[0].descriptorWords = descriptorWords;
    return layout;
}

void testDecode() {
    const auto layout = layoutOf({0u, 1u, 2u, 3u});
    std::array<std::uint32_t, 8> words{0x12345679u, 0x00000abcu | (12u << 16u) | 0xc0000000u, 100u, 0xffffffffu, 0u, 0u, 0u, 0u};
    auto request = DecodeBuffer(layout, words, 0);
    Expect(request.present && request.address == 0x00000abc12345679ull, "a 44-bit base is rebuilt from both words, without the stride or the flag bits");
    Expect(request.bytes == 1200u, "bytes are the stride times the record count");
    Expect(request.misalignment == 1u && request.buffer == 0u, "the misalignment is the two low bits of the base");
    words[1] = 0x00000abcu;
    request = DecodeBuffer(layout, words, 0);
    Expect(request.present && request.bytes == 100u && request.address == 0x00000abc12345679ull, "a zero stride counts the records as bytes");
    words[1] = 0x0000fabcu | (0x3fffu << 16u);
    words[2] = 0xffffffffu;
    request = DecodeBuffer(layout, words, 0);
    Expect(request.present && request.address == 0x0000fabc12345679ull, "the base keeps the 48 bits the classic path keeps");
    Expect(request.bytes == std::uint64_t{0x3fffu} * 0xffffffffull, "the largest stride and count do not wrap");
    words = {0x00000008u, 0u, 4u, 0u, 0u, 0u, 0u, 0u};
    request = DecodeBuffer(layout, words, 0);
    Expect(request.present && request.address == 8u && request.bytes == 4u && request.misalignment == 0u, "an aligned low base decodes without misalignment");
    words = {0u, 0x00000001u, 4u, 0u, 0u, 0u, 0u, 0u};
    request = DecodeBuffer(layout, words, 0);
    Expect(request.present && request.address == 0x0000000100000000ull, "a base held by the high word alone is present");
    words = {0x12345679u, 0x00000abcu | (12u << 16u), 0u, 0u, 0u, 0u, 0u, 0u};
    request = DecodeBuffer(layout, words, 0);
    Expect(!request.present && request.address == 0u && request.bytes == 0u && request.misalignment == 0u, "a V# with no record is absent");
    words = {0u, 12u << 16u, 100u, 0xffffffffu, 0u, 0u, 0u, 0u};
    request = DecodeBuffer(layout, words, 0);
    Expect(!request.present && request.address == 0u && request.bytes == 0u, "a V# with a null base is absent");
    words = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    Expect(!DecodeBuffer(layout, words, 0).present, "a null V# is absent");

    const auto scattered = layoutOf({5u, 2u, 7u, 0u});
    words = {0xffffffffu, 0xffffffffu, 0x00000abcu | (4u << 16u), 0xffffffffu, 0xffffffffu, 0x12345678u + 3u, 0xffffffffu, 9u};
    request = DecodeBuffer(scattered, words, 0);
    Expect(request.present && request.address == 0x00000abc1234567bull && request.bytes == 36u && request.misalignment == 3u, "the V# words are taken where the layout says");

    words = {0x12345679u, 0x00000abcu | (12u << 16u), 100u, 0u, 0u, 0u, 0u, 0u};
    Expect(!DecodeBuffer(layout, words, 1).present && DecodeBuffer(layout, words, 1).buffer == 1u, "a buffer the layout does not have is absent");
    Expect(!DecodeBuffer(layoutOf({NoWord, 1u, 2u, 3u}), words, 0).present && !DecodeBuffer(layoutOf({0u, NoWord, 2u, 3u}), words, 0).present && !DecodeBuffer(layoutOf({0u, 1u, NoWord, 3u}), words, 0).present, "a V# whose base, stride or count word the plan does not read is absent");
    Expect(DecodeBuffer(layoutOf({0u, 1u, 2u, NoWord}), words, 0).present, "the format word is not needed to locate a buffer");
    Expect(!DecodeBuffer(layout, std::span<const std::uint32_t>(words).first(2), 0).present, "a V# word outside the given words is absent");
}

void testSection() {
    Scene scene;
    Expect(scene.tape != nullptr, "the scene plan compiles to a tape");
    if (scene.tape == nullptr) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto output = WriteBlock(scene.Inputs(), ring);
    Expect(output.fault == BlockFault::None && output.deferred == 0u, "a readable draw writes its block");
    Expect(output.section.bytes != nullptr && output.section.deviceAddress != 0u, "the section names its ring slice");
    if (output.section.bytes == nullptr) return;
    Expect(reinterpret_cast<std::uintptr_t>(output.section.bytes) % BlockAbi::SectionAlignmentBytes == 0u && output.section.deviceAddress % BlockAbi::SectionAlignmentBytes == 0u, "the section is aligned to 16 bytes");
    Expect(BlockAbi::SectionBytes(SceneWords, SceneBuffers) == 112u, "thirteen words and three buffers take 112 bytes");
    const auto section = sectionOf(output.section, BlockAbi::SectionBytes(SceneWords, SceneBuffers) / sizeof(std::uint32_t));
    const auto expected = scene.Words();
    bool same = true;
    for (std::uint32_t word = 0; word < SceneWords; ++word) same = same && section[word] == expected[word];
    Expect(same, "the words are the plan's flat slots, in plan order");
    Expect(section[4] == static_cast<std::uint32_t>(TabledBase) && section[6] == TabledBytes && scene.host.reads != 0u, "the words behind a pointer come through the reader");
    Expect(entryIs(section, SceneWords, Direct, MirrorBase + DirectBase, DirectStride * DirectRecords, 2u), "a resolved entry holds the device address, the bytes and the misalignment");
    Expect(entryIs(section, SceneWords, Tabled, MirrorBase + TabledBase, TabledBytes, 0u), "an aligned resolved entry has an empty service word");
    Expect(entryIs(section, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent), "an absent buffer is a null entry with the absent bit");
    bool padded = true;
    for (std::size_t word = BlockAbi::SectionWords(SceneWords, SceneBuffers); word < section.size(); ++word) padded = padded && section[word] == 0u;
    Expect(padded, "the padding up to the alignment is zeroed");
    Expect(scene.host.resolved == 2u, "only present buffers are resolved");
    const auto& direct = scene.host.requests[0];
    const auto& tabled = scene.host.requests[1];
    Expect(direct.buffer == Direct && direct.present && direct.address == DirectBase && direct.bytes == DirectStride * DirectRecords && direct.misalignment == 2u, "the first request is the decoded V# of the first buffer");
    Expect(tabled.buffer == Tabled && tabled.present && tabled.address == TabledBase && tabled.bytes == TabledBytes && tabled.misalignment == 0u, "the second request is the decoded V# of the second buffer");

    const auto second = WriteBlock(scene.Inputs(), ring);
    Expect(second.fault == BlockFault::None && second.section.bytes != nullptr && second.section.bytes != output.section.bytes && second.section.deviceAddress != output.section.deviceAddress, "each draw takes its own slice");

    scene.host.resolved = 0;
    const auto before = allocations;
    const auto third = WriteBlock(scene.Inputs(), ring);
    Expect(third.fault == BlockFault::None && scene.host.resolved == 2u, "a third draw still writes its block");
    Expect(allocations == before, "writing a block allocates nothing");
}

void testEmpty() {
    IrResourcePlan plan;
    plan.srtPlanComplete = true;
    plan.resourceTrackingComplete = true;
    const auto* tape = Detail::SrtTape::For(plan);
    Expect(tape != nullptr, "a plan with no read compiles to a tape");
    if (tape == nullptr) return;
    BlockLayout layout;
    Host host;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    BlockInputs inputs;
    inputs.plan = &plan;
    inputs.layout = &layout;
    inputs.tape = tape;
    inputs.variantKey = BlockVariantKey(layout, {});
    inputs.read = readWords;
    inputs.resolve = resolveBuffer;
    inputs.context = &host;
    const auto output = WriteBlock(inputs, ring);
    Expect(output.fault == BlockFault::None && output.section.bytes == nullptr && output.deferred == 0u && arena.chunks == 0u, "a stage with no word and no buffer needs no section and no ring");
    inputs.variantKey ^= 1u;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Variant), "a stage with no word still checks its variant key");
}

void testDeferred() {
    Scene scene;
    if (scene.tape == nullptr) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    scene.host.table[0] += 3u;
    scene.host.deferred = std::uint64_t{1} << Tabled;
    const auto output = WriteBlock(scene.Inputs(), ring);
    Expect(output.fault == BlockFault::None && output.deferred == std::uint64_t{1} << Tabled, "a deferred buffer sets its bit and nothing else");
    if (output.section.bytes == nullptr) {
        Expect(false, "a deferred draw still returns its section");
        return;
    }
    const auto section = sectionOf(output.section, BlockAbi::SectionBytes(SceneWords, SceneBuffers) / sizeof(std::uint32_t));
    Expect(entryIs(section, SceneWords, Tabled, 0u, 0u, 0u), "a deferred entry is left at zero");
    Expect(entryIs(section, SceneWords, Direct, MirrorBase + DirectBase, DirectStride * DirectRecords, 2u) && entryIs(section, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent), "the other entries are written at once");

    BufferAnswer pending;
    pending.deferred = true;
    pending.deviceAddress = 0x1234u;
    pending.bytes = 4u;
    CompleteBuffer(scene.layout, output.section, Tabled, pending);
    Expect(entryIs(section, SceneWords, Tabled, 0u, 0u, 0u), "an answer that is still deferred completes nothing");

    BufferAnswer answer;
    answer.deviceAddress = 0x0000007f00010000ull;
    answer.bytes = 0x280u;
    CompleteBuffer(scene.layout, output.section, SceneBuffers, answer);
    CompleteBuffer(scene.layout, RingSlice{}, Tabled, answer);
    Expect(entryIs(section, SceneWords, Tabled, 0u, 0u, 0u), "a buffer the layout does not have, or an empty section, completes nothing");
    CompleteBuffer(scene.layout, output.section, Tabled, answer);
    Expect(entryIs(section, SceneWords, Tabled, 0x0000007f00010000ull, 0x280u, 3u), "a completed entry holds the answer and the misalignment of its V#");
    Expect(entryIs(section, SceneWords, Direct, MirrorBase + DirectBase, DirectStride * DirectRecords, 2u) && entryIs(section, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent), "completing one entry leaves the others");
    const auto expected = scene.Words();
    bool same = true;
    for (std::uint32_t word = 0; word < SceneWords; ++word) same = same && section[word] == expected[word];
    Expect(same, "completing one entry leaves the words");

    CompleteBuffer(scene.layout, output.section, Tabled, BufferAnswer{});
    Expect(entryIs(section, SceneWords, Tabled, 0u, 0u, BlockAbi::ServiceAbsent), "a completion with no device address makes the entry absent");
}

void testVariant() {
    Scene scene;
    if (scene.tape == nullptr) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    auto inputs = scene.Inputs();
    inputs.variantKey ^= 1u;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Variant) && scene.host.resolved == 0u, "another variant key faults before any buffer is resolved");
    inputs = scene.Inputs();
    scene.userData[3] ^= 1u << 12u;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Variant), "a format bit that changed since the key was taken faults");
    scene.userData[3] ^= 1u << 12u;
    scene.userData[3] ^= 1u;
    scene.userData[6] ^= 0xffu;
    Expect(WriteBlock(inputs, ring).fault == BlockFault::None, "a bit outside the variant mask does not fault");
    scene.layout.variantKey.push_back({SceneWords, 0xffffffffu});
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Variant) && arena.chunks == 1u, "a variant word outside the block faults");
}

void testUnreadable() {
    Scene scene;
    if (scene.tape == nullptr) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    scene.host.readable = false;
    Expect(faultedWith(WriteBlock(scene.Inputs(), ring), BlockFault::Unreadable) && scene.host.reads != 0u && scene.host.resolved == 0u, "a word the reader refuses faults as unreadable");
    scene.host.readable = true;
    scene.host.refused = std::uint64_t{1} << Tabled;
    Expect(faultedWith(WriteBlock(scene.Inputs(), ring), BlockFault::Unreadable) && scene.host.resolved == 2u, "a buffer the host cannot resolve faults as unreadable");
    scene.host.refused = 0;
    Expect(WriteBlock(scene.Inputs(), ring).fault == BlockFault::None, "the same draw writes its block once everything is readable");
}

void testWalk() {
    Scene scene;
    if (scene.tape == nullptr) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    auto inputs = scene.Inputs();
    inputs.tape = nullptr;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a draw with no tape faults as a walk");
    inputs = scene.Inputs();
    inputs.plan = nullptr;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a draw with no plan faults as a walk");
    inputs = scene.Inputs();
    inputs.layout = nullptr;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a draw with no layout faults as a walk");
    inputs = scene.Inputs();
    inputs.read = nullptr;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a draw with no reader faults as a walk");
    inputs = scene.Inputs();
    inputs.resolve = nullptr;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a draw with no resolver faults as a walk");
    inputs = scene.Inputs();

    auto refused = scene.layout;
    refused.refusal = BlockRefusal::WrittenBuffer;
    inputs.layout = &refused;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a refused layout faults as a walk");
    auto versioned = scene.layout;
    versioned.abiVersion = BlockAbi::Version + 1u;
    inputs.layout = &versioned;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a layout of another block version faults as a walk");
    auto shorter = scene.layout;
    shorter.words = SceneWords - 1u;
    shorter.buffers.pop_back();
    inputs.layout = &shorter;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a layout that does not count the plan's slots faults as a walk");
    auto unread = scene.layout;
    unread.buffers[Tabled].descriptorWords[2] = NoWord;
    inputs.layout = &unread;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "a buffer whose count word the plan does not read faults as a walk");
    auto crowded = scene.layout;
    crowded.buffers.resize(BlockAbi::MaxBuffers + 1u, crowded.buffers[Direct]);
    inputs.layout = &crowded;
    Expect(faultedWith(WriteBlock(inputs, ring), BlockFault::Walk), "more buffers than the block format holds faults as a walk");
    Expect(arena.chunks == 0u && scene.host.reads == 0u && scene.host.resolved == 0u, "a draw refused before the walk takes nothing from the ring and reads nothing");

    Values values;
    IrResourcePlan wide;
    wide.srtPlanComplete = true;
    wide.resourceTrackingComplete = true;
    constexpr std::uint32_t WideWords = 600;
    for (std::uint32_t word = 0; word < WideWords; ++word) wide.srtReads.push_back({&values.Constant(word), word});
    const auto* wideTape = Detail::SrtTape::For(wide);
    Expect(wideTape != nullptr && wideTape->Nodes() > 512u, "a plan of 600 constants compiles to a tape of more than 512 nodes");
    if (wideTape != nullptr) {
        BlockLayout wideLayout;
        wideLayout.words = WideWords;
        BlockInputs wideInputs = scene.Inputs();
        wideInputs.plan = &wide;
        wideInputs.layout = &wideLayout;
        wideInputs.tape = wideTape;
        wideInputs.variantKey = BlockVariantKey(wideLayout, {});
        Expect(faultedWith(WriteBlock(wideInputs, ring), BlockFault::Walk), "a tape the flat walker does not support faults as a walk");
    }

    IrResourcePlan broken;
    broken.srtPlanComplete = true;
    broken.resourceTrackingComplete = true;
    broken.srtReads.push_back({&values.Make(IrOpcode::Void, IrType::U32), 0u});
    const auto* brokenTape = Detail::SrtTape::For(broken);
    Expect(brokenTape != nullptr, "a plan with a slot the tape cannot evaluate still compiles");
    if (brokenTape != nullptr) {
        BlockLayout brokenLayout;
        brokenLayout.words = 1;
        BlockInputs brokenInputs = scene.Inputs();
        brokenInputs.plan = &broken;
        brokenInputs.layout = &brokenLayout;
        brokenInputs.tape = brokenTape;
        brokenInputs.variantKey = BlockVariantKey(brokenLayout, {});
        const auto output = WriteBlock(brokenInputs, ring);
        Expect(output.fault != BlockFault::None && output.section.bytes == nullptr && output.deferred == 0u, "a slot the tape cannot evaluate faults and returns no section");
    }
}

void testRingFull() {
    Scene scene;
    if (scene.tape == nullptr) return;
    Arena arena;
    arena.executed = false;
    BlockRing ring(sourceOf(arena));
    std::size_t taken = 0;
    for (; taken <= BlockRing::MaxChunks; ++taken) {
        const auto slice = ring.Take(BlockRing::ChunkBytes, BlockAbi::SectionAlignmentBytes);
        if (slice.bytes == nullptr) break;
        ring.Posted(slice);
    }
    Expect(taken == BlockRing::MaxChunks, "the test ring holds 64 chunks in flight");
    if (taken != BlockRing::MaxChunks) return;
    Expect(faultedWith(WriteBlock(scene.Inputs(), ring), BlockFault::RingFull) && scene.host.reads == 0u && scene.host.resolved == 0u, "a full ring faults before the walk");
}

}

void* operator new(std::size_t bytes) {
    ++allocations;
    if (auto* allocation = std::malloc(bytes == 0 ? 1 : bytes)) return allocation;
    throw std::bad_alloc();
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* allocation) noexcept { std::free(allocation); }
void operator delete[](void* allocation) noexcept { std::free(allocation); }
void operator delete(void* allocation, std::size_t) noexcept { std::free(allocation); }
void operator delete[](void* allocation, std::size_t) noexcept { std::free(allocation); }

int main() {
    testDecode();
    testSection();
    testEmpty();
    testDeferred();
    testVariant();
    testUnreadable();
    testWalk();
    testRingFull();
    if (failures != 0) {
        std::fprintf(stderr, "%d block writer check(s) failed\n", failures);
        return 1;
    }
    std::printf("block writer: ok\n");
    return 0;
}
