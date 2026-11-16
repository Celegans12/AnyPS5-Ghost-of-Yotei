#include "prx/libSceAgcDriver/Core/include/DecodeDraw.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
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
#include <string_view>
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

constexpr std::uint64_t TableAddress = 0x0000000500004000ull;
constexpr std::uint64_t DeviceBase = 0x0000004000000000ull;
constexpr std::uint64_t MirrorBase = 0x0000100000000000ull;
constexpr std::uint32_t FormatMask = 0x7fu << 12u;
constexpr std::uint32_t ImageMask = 0x00ffffffu;
constexpr std::uint32_t UserImageMask = 0x0000ffffu;

constexpr std::uint32_t SceneWords = 13;
constexpr std::uint32_t SceneBuffers = 3;
constexpr std::uint32_t SceneBytes = 112;
constexpr std::uint32_t TableWords = 4;
constexpr std::uint32_t Direct = 0;
constexpr std::uint32_t Tabled = 1;
constexpr std::uint32_t Missing = 2;
constexpr std::uint32_t ImageWord = 8;
constexpr std::uint32_t ImageUserWord = 6;
constexpr std::uint32_t UserImageWord = 7;

constexpr std::uint64_t FirstDirect = 0x00000abc1234567aull;
constexpr std::uint64_t FirstTabled = 0x0000000200001000ull;
constexpr std::uint64_t SecondDirect = 0x00000abd76543211ull;
constexpr std::uint64_t SecondTabled = 0x0000000300002003ull;

struct Host {
    std::array<std::uint32_t, 2u * TableWords> table{};
    std::uint64_t unreadableFrom = ~std::uint64_t{0};
    std::uint32_t reads = 0;
    std::uint64_t refused = 0;
    std::array<std::uint64_t, 2> deferred{};
    std::uint32_t resolved = 0;
    std::array<BufferRequest, 16> requests{};
};

bool readWords(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count) {
    auto& host = *static_cast<Host*>(context);
    ++host.reads;
    for (std::uint32_t word = 0; word < count; ++word) {
        const auto at = address + std::uint64_t{word} * sizeof(std::uint32_t);
        if (at >= host.unreadableFrom) return false;
        if (at < TableAddress || (at - TableAddress) % sizeof(std::uint32_t) != 0u || (at - TableAddress) / sizeof(std::uint32_t) >= host.table.size()) return false;
        out[word] = host.table[static_cast<std::size_t>((at - TableAddress) / sizeof(std::uint32_t))];
    }
    return true;
}

bool resolveBuffer(void* context, const BufferRequest& request, BufferAnswer& answer) {
    auto& host = *static_cast<Host*>(context);
    if (host.resolved < host.requests.size()) host.requests[host.resolved] = request;
    ++host.resolved;
    if (request.address == host.refused) return false;
    for (const auto address : host.deferred) {
        if (request.address == address) {
            answer.deferred = true;
            return true;
        }
    }
    answer.deviceAddress = MirrorBase + request.address;
    answer.bytes = static_cast<std::uint32_t>(request.bytes);
    return true;
}

struct Arena {
    void* memory = nullptr;
    std::size_t bytes = 0;
    std::uint64_t chunks = 0;
    bool executed = true;

    Arena() = default;
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    ~Arena() { std::free(memory); }
};

RingChunk allocateChunk(void* context, std::size_t bytes) {
    auto& arena = *static_cast<Arena*>(context);
    if (arena.memory == nullptr) {
        arena.memory = std::malloc(bytes);
        if (arena.memory == nullptr) return {};
        arena.bytes = bytes;
        std::memset(arena.memory, 0xa5, bytes);
    }
    if (bytes > arena.bytes) return {};
    RingChunk chunk;
    chunk.bytes = static_cast<std::byte*>(arena.memory);
    chunk.deviceAddress = DeviceBase + arena.chunks * BlockRing::ChunkBytes;
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
    const Detail::SrtTape* tape = nullptr;
    std::uint32_t tableWord = 0;
    std::uint64_t directBase = 0;
    std::uint32_t directBytes = 0;
    std::uint64_t tabledBase = 0;
    std::uint32_t tabledBytes = 0;

    Scene(Host& host, std::uint32_t firstTableWord, std::uint64_t direct, std::uint32_t stride, std::uint32_t records, std::uint64_t tabled, std::uint32_t tabledSize) : tableWord(firstTableWord), directBase(direct), directBytes(stride * records), tabledBase(tabled), tabledBytes(tabledSize) {
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
        plan.srtReads.push_back({&values.UserData(ImageUserWord), ImageWord});
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
        layout.imageKey.push_back({ImageWord, ImageMask});
        layout.userImageKey.push_back({UserImageWord, UserImageMask});

        const auto table = TableAddress + std::uint64_t{tableWord} * sizeof(std::uint32_t);
        userData[0] = static_cast<std::uint32_t>(directBase);
        userData[1] = static_cast<std::uint32_t>(directBase >> 32u) | (stride << 16u);
        userData[2] = records;
        userData[3] = (5u << 12u) | 0xfacu;
        userData[4] = static_cast<std::uint32_t>(table);
        userData[5] = static_cast<std::uint32_t>(table >> 32u);
        userData[ImageUserWord] = 0x00decadeu + tableWord;
        userData[UserImageWord] = 0x00001234u + tableWord;
        host.table[tableWord] = static_cast<std::uint32_t>(tabledBase);
        host.table[tableWord + 1u] = static_cast<std::uint32_t>(tabledBase >> 32u);
        host.table[tableWord + 2u] = tabledBytes;
        host.table[tableWord + 3u] = 0xfacu + tableWord;
    }

    Scene(const Scene&) = delete;
    Scene& operator=(const Scene&) = delete;

    std::array<std::uint32_t, SceneWords> Words(const Host& host) const {
        return {userData[0], userData[1], userData[2], userData[3], host.table[tableWord], host.table[tableWord + 1u], host.table[tableWord + 2u], host.table[tableWord + 3u], userData[ImageUserWord], userData[8], userData[9], userData[10], userData[11]};
    }

    std::array<std::uint32_t, SceneWords> Walked(Host& host) const {
        std::array<std::uint32_t, SceneWords> words{};
        const auto walk = tape->RunInto(plan, userData, 0u, readWords, &host, words);
        Expect(walk == BlockWalk::Done, "the scene walks alone");
        return words;
    }

    DrawStage Stage(const Host& host) const {
        const auto words = Words(host);
        DrawStage stage;
        stage.plan = &plan;
        stage.layout = &layout;
        stage.tape = tape;
        stage.userData = userData;
        stage.variantKey = BlockVariantKey(layout, words, userData);
        return stage;
    }
};

struct Draw {
    Host host;
    Scene first;
    Scene second;

    Draw() : first(host, 0u, FirstDirect, 16u, 10u, FirstTabled, 0x300u), second(host, TableWords, SecondDirect, 8u, 6u, SecondTabled, 0x140u) {}

    bool Ready() const { return first.tape != nullptr && second.tape != nullptr; }

    std::array<DrawStage, 2> Stages() const { return {first.Stage(host), second.Stage(host)}; }

    DrawReads Reads() {
        DrawReads reads;
        reads.read = readWords;
        reads.resolve = resolveBuffer;
        reads.context = &host;
        return reads;
    }
};

constexpr std::uint32_t DrivenBuffers = 2;
constexpr std::uint32_t DrivenZoneWords = 21;
constexpr std::uint32_t DrivenSectionBytes = 32;
constexpr std::uint32_t DrivenBytes = 128;
constexpr std::uint32_t FilterWord = 4;
constexpr std::uint32_t PictureUserWord = 8;
constexpr std::uint64_t DrivenDirect = 0x00000abe22446682ull;
constexpr std::uint64_t DrivenTabled = 0x0000000400003001ull;

struct Driven {
    Values values;
    IrResourcePlan plan;
    BlockLayout layout;
    std::array<std::uint32_t, 16> userData{};
    const Detail::SrtTape* tape = nullptr;

    explicit Driven(Host& host) {
        plan.srtPlanComplete = true;
        plan.resourceTrackingComplete = true;
        MemoryInfo memory;
        memory.kind = ResourceKind::ScalarAddress;
        plan.memoryInfo.push_back(memory);
        auto& pointer = values.Make(IrOpcode::CompositeConstructU64, IrType::U64, {&values.UserData(4), &values.UserData(5)});
        const auto load = [&](std::uint32_t word) -> IrValue* {
            auto& read = values.Make(IrOpcode::LoadAddressU32, IrType::U32);
            read.SetFlags(MemoryFlags{0, 0});
            read.AddArgument(&pointer);
            read.AddArgument(&values.Constant(word * 4u));
            return &read;
        };
        DescriptorSource direct;
        DescriptorSource tabled;
        DescriptorSource picture;
        DescriptorSource filter;
        direct.dwordCount = tabled.dwordCount = filter.dwordCount = 4u;
        picture.dwordCount = 8u;
        for (std::uint32_t word = 0; word < 4u; ++word) {
            direct.dwords[word] = &values.UserData(word);
            tabled.dwords[word] = load(word);
            filter.dwords[word] = load(FilterWord + word);
        }
        for (std::uint32_t word = 0; word < 8u; ++word) picture.dwords[word] = &values.UserData(PictureUserWord + word);
        plan.descriptorSources = {direct, tabled, picture, filter};
        plan.materializationSources = {0u, 1u, 2u, 3u};
        plan.info.buffers.resize(DrivenBuffers);
        plan.info.buffers[Direct].source = 0u;
        plan.info.buffers[Tabled].source = 1u;
        plan.info.images.resize(1);
        plan.info.images[0].source = 2u;
        plan.info.runtimeImageModes.resize(1);
        plan.info.samplers.resize(1);
        plan.info.samplers[0].source = 3u;
        tape = Detail::SrtTape::For(plan);
        layout = BuildBlockLayout(plan);

        userData[0] = static_cast<std::uint32_t>(DrivenDirect);
        userData[1] = static_cast<std::uint32_t>(DrivenDirect >> 32u) | (12u << 16u);
        userData[2] = 5u;
        userData[3] = 0xfacu;
        userData[4] = static_cast<std::uint32_t>(TableAddress);
        userData[5] = static_cast<std::uint32_t>(TableAddress >> 32u);
        for (std::uint32_t word = 0; word < 8u; ++word) userData[PictureUserWord + word] = 0x70000000u + word;
        host.table[0] = static_cast<std::uint32_t>(DrivenTabled);
        host.table[1] = static_cast<std::uint32_t>(DrivenTabled >> 32u);
        host.table[2] = 0x240u;
        host.table[3] = 0xbeefu;
        for (std::uint32_t word = 0; word < 4u; ++word) host.table[FilterWord + word] = 0x5a000000u + word;
    }

    Driven(const Driven&) = delete;
    Driven& operator=(const Driven&) = delete;

    DrawStage Stage() const {
        DrawStage stage;
        stage.plan = &plan;
        stage.layout = &layout;
        stage.tape = tape;
        stage.userData = userData;
        stage.variantKey = BlockVariantKey(layout, {}, userData);
        return stage;
    }
};

std::span<const std::uint32_t> sectionOf(const DrawWritten& written, std::uint32_t stage, std::size_t words) {
    return {reinterpret_cast<const std::uint32_t*>(written.slice.bytes + written.section[stage]), words};
}

RingSlice sliceOf(const DrawWritten& written, std::uint32_t stage) {
    RingSlice part = written.slice;
    part.bytes += written.section[stage];
    part.deviceAddress += written.section[stage];
    return part;
}

bool entryIs(std::span<const std::uint32_t> section, std::uint32_t words, std::uint32_t buffer, std::uint64_t address, std::uint32_t bytes, std::uint32_t service) {
    const auto first = BlockAbi::BufferEntryWord(words, buffer);
    return section[first + BlockAbi::EntryAddressLow] == static_cast<std::uint32_t>(address) && section[first + BlockAbi::EntryAddressHigh] == static_cast<std::uint32_t>(address >> 32u) && section[first + BlockAbi::EntryBytes] == bytes && section[first + BlockAbi::EntryService] == service;
}

bool wordsAre(std::span<const std::uint32_t> section, const std::array<std::uint32_t, SceneWords>& expected) {
    for (std::uint32_t word = 0; word < SceneWords; ++word) {
        if (section[word] != expected[word]) return false;
    }
    return true;
}

bool leftWith(const DrawWritten& written, DrawLeft left, std::uint32_t stage) {
    return written.left == left && written.stage == stage && written.slice.bytes == nullptr && written.deferred == 0u && written.images[0] == 0u && written.images[1] == 0u;
}

void testNames() {
    Expect(std::string_view(DrawLeftName(DrawLeft::None)) == "None" && std::string_view(DrawLeftName(DrawLeft::Ring)) == "Ring" && std::string_view(DrawLeftName(DrawLeft::Walk)) == "Walk", "the first causes have their names");
    Expect(std::string_view(DrawLeftName(DrawLeft::Variant)) == "Variant" && std::string_view(DrawLeftName(DrawLeft::Buffers)) == "Buffers", "the last causes have their names");
    Expect(DrawLeftName(DrawLeft::Count) != nullptr && std::string_view(DrawLeftName(DrawLeft::Count)) != "None", "a cause out of range has a name that is not a cause");
}

void testBytes() {
    Draw draw;
    if (!draw.Ready()) {
        Expect(false, "the scene plans compile to tapes");
        return;
    }
    const auto stages = draw.Stages();
    Expect(draw.first.layout.SectionBytes() == SceneBytes && BlockAbi::SectionBytes(SceneWords, SceneBuffers) == SceneBytes, "thirteen words and three buffers take 112 bytes");
    Expect(DrawBytes(stages, 0u) == 2u * SceneBytes, "the bytes of a draw are the sum of its sections");
    Expect(DrawBytes(stages, 1u) == 2u * SceneBytes + 16u && DrawBytes(stages, 4u) == 2u * SceneBytes + 16u && DrawBytes(stages, 5u) == 2u * SceneBytes + 32u, "the free words are rounded up to 16 bytes");
    Expect(DrawBytes(std::span<const DrawStage>(stages).first(1), 0u) == SceneBytes, "one stage takes one section");
    Expect(DrawBytes({}, 0u) == 0u && DrawBytes({}, 3u) == 16u, "a draw with no stage takes its free words alone");
    std::array<DrawStage, 2> holed = stages;
    holed[0].layout = nullptr;
    Expect(DrawBytes(holed, 0u) == SceneBytes, "a stage with no layout counts no byte");
    Expect(DrawBytes(stages, 0xffffffffu) == 0xffffffffu, "bytes that do not fit 32 bits saturate");
}

void testDraw() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto stages = draw.Stages();
    const auto reads = draw.Reads();
    constexpr std::uint32_t ExtraWords = 5;
    const auto written = WriteDraw(stages, reads, ExtraWords, ring);
    Expect(written.left == DrawLeft::None && written.stage == 0u && written.deferred == 0u, "a readable draw of two stages is written");
    Expect(written.slice.bytes != nullptr && written.slice.deviceAddress != 0u, "the draw names its ring slice");
    if (written.slice.bytes == nullptr) return;
    Expect(written.slice.deviceAddress % BlockAbi::SectionAlignmentBytes == 0u && reinterpret_cast<std::uintptr_t>(written.slice.bytes) % BlockAbi::SectionAlignmentBytes == 0u, "the slice is aligned to 16 bytes");
    Expect(arena.chunks == 1u && ring.Counted().bytes == DrawBytes(stages, ExtraWords) && DrawBytes(stages, ExtraWords) == 2u * SceneBytes + 32u, "the whole draw takes DrawBytes from the ring, once");
    Expect(written.section[0] == 0u && written.section[1] == SceneBytes && written.section[2] == 0u && written.section[3] == 0u, "the sections follow each other in stage order");
    Expect(written.extra == 2u * SceneBytes && written.extra + ExtraWords * sizeof(std::uint32_t) <= DrawBytes(stages, ExtraWords), "the free words follow the sections and fit the slice");
    Expect(written.images[0] != 0u || written.images[1] != 0u, "the draw has an image digest");

    const auto sectionWords = SceneBytes / sizeof(std::uint32_t);
    const auto first = sectionOf(written, 0, sectionWords);
    const auto second = sectionOf(written, 1, sectionWords);
    Expect(wordsAre(first, draw.first.Words(draw.host)) && wordsAre(second, draw.second.Words(draw.host)), "each section holds the flat slots of its stage");
    Expect(draw.host.resolved == 4u, "only present buffers are resolved");
    const auto& requests = draw.host.requests;
    Expect(requests[0].buffer == Direct && requests[0].address == FirstDirect && requests[0].bytes == 160u && requests[0].misalignment == 2u && requests[0].present, "the first request is the first buffer of the first stage");
    Expect(requests[1].buffer == Tabled && requests[1].address == FirstTabled && requests[1].bytes == 0x300u && requests[1].misalignment == 0u, "the second request is the second buffer of the first stage");
    Expect(requests[2].buffer == Direct && requests[2].address == SecondDirect && requests[2].bytes == 48u && requests[2].misalignment == 1u, "the third request is the first buffer of the second stage, numbered in its layout");
    Expect(requests[3].buffer == Tabled && requests[3].address == SecondTabled && requests[3].bytes == 0x140u && requests[3].misalignment == 3u, "the fourth request is the second buffer of the second stage");
    Expect(wordsAre(first, draw.first.Walked(draw.host)) && wordsAre(second, draw.second.Walked(draw.host)), "the words are those RunInto writes alone");
    Expect(first[4] == static_cast<std::uint32_t>(FirstTabled) && second[4] == static_cast<std::uint32_t>(SecondTabled) && second[6] == 0x140u, "the words behind a pointer come through the reader");

    Expect(entryIs(first, SceneWords, Direct, MirrorBase + FirstDirect, 160u, 2u) && entryIs(first, SceneWords, Tabled, MirrorBase + FirstTabled, 0x300u, 0u), "the resolved entries of the first stage hold the answers");
    Expect(entryIs(second, SceneWords, Direct, MirrorBase + SecondDirect, 48u, 1u) && entryIs(second, SceneWords, Tabled, MirrorBase + SecondTabled, 0x140u, 3u), "the resolved entries of the second stage hold the answers");
    Expect(entryIs(first, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent) && entryIs(second, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent), "an absent buffer is a null entry with the absent bit");
    bool padded = true;
    for (std::size_t word = BlockAbi::SectionWords(SceneWords, SceneBuffers); word < sectionWords; ++word) padded = padded && first[word] == 0u && second[word] == 0u;
    Expect(padded, "the padding of each section is zeroed");

    std::array<std::uint32_t, 2u * SceneBytes / sizeof(std::uint32_t)> before{};
    std::memcpy(before.data(), written.slice.bytes, 2u * SceneBytes);
    auto* extra = reinterpret_cast<std::uint32_t*>(written.slice.bytes + written.extra);
    for (std::uint32_t word = 0; word < ExtraWords; ++word) extra[word] = 0xe0000000u + word;
    Expect(std::memcmp(before.data(), written.slice.bytes, 2u * SceneBytes) == 0, "writing the free words leaves the sections");

    const auto next = ring.Take(16u, BlockAbi::SectionAlignmentBytes);
    Expect(next.bytes != nullptr && next.deviceAddress == written.slice.deviceAddress + DrawBytes(stages, ExtraWords) && next.chunk == written.slice.chunk, "the next slice starts after the free words");
    bool kept = true;
    for (std::uint32_t word = 0; word < ExtraWords; ++word) kept = kept && extra[word] == 0xe0000000u + word;
    Expect(kept && next.bytes >= written.slice.bytes + written.extra + ExtraWords * sizeof(std::uint32_t), "the free words lie inside the slice of the draw");

    const auto again = WriteDraw(stages, reads, 0u, ring);
    Expect(again.left == DrawLeft::None && again.slice.bytes != nullptr && again.slice.bytes != written.slice.bytes && again.extra == 2u * SceneBytes, "each draw takes its own slice");
    Expect(again.images == written.images, "the same draw has the same image digest");
    const auto after = ring.Take(16u, BlockAbi::SectionAlignmentBytes);
    Expect(after.deviceAddress == again.slice.deviceAddress + 2u * SceneBytes, "a draw with no free word takes its sections alone");
}

void testDeferred() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto stages = draw.Stages();
    const auto reads = draw.Reads();
    const auto sectionWords = SceneBytes / sizeof(std::uint32_t);

    draw.host.deferred = {SecondTabled, 0u};
    const auto written = WriteDraw(stages, reads, 0u, ring);
    Expect(written.left == DrawLeft::None && written.deferred == std::uint64_t{1} << (SceneBuffers + Tabled), "a deferred buffer of the second stage sets the bit of its rank in the draw");
    if (written.slice.bytes == nullptr) {
        Expect(false, "a deferred draw still returns its slice");
        return;
    }
    const auto first = sectionOf(written, 0, sectionWords);
    const auto second = sectionOf(written, 1, sectionWords);
    Expect(entryIs(second, SceneWords, Tabled, 0u, 0u, 0u), "a deferred entry is left at zero");
    Expect(entryIs(first, SceneWords, Direct, MirrorBase + FirstDirect, 160u, 2u) && entryIs(first, SceneWords, Tabled, MirrorBase + FirstTabled, 0x300u, 0u) && entryIs(first, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent), "the entries of the other stage are written at once");
    Expect(entryIs(second, SceneWords, Direct, MirrorBase + SecondDirect, 48u, 1u) && entryIs(second, SceneWords, Missing, 0u, 0u, BlockAbi::ServiceAbsent), "the other entries of the stage are written at once");

    BufferAnswer answer;
    answer.deviceAddress = 0x0000007f00010000ull;
    answer.bytes = 0x100u;
    CompleteBuffer(draw.second.layout, sliceOf(written, 1), Tabled, answer);
    Expect(entryIs(second, SceneWords, Tabled, 0x0000007f00010000ull, 0x100u, 3u), "a deferred entry is completed through the slice of its stage");
    Expect(entryIs(first, SceneWords, Tabled, MirrorBase + FirstTabled, 0x300u, 0u) && wordsAre(second, draw.second.Words(draw.host)), "completing an entry leaves the other stage and the words");

    draw.host.deferred = {FirstDirect, SecondDirect};
    const auto both = WriteDraw(stages, reads, 0u, ring);
    Expect(both.left == DrawLeft::None && both.deferred == ((std::uint64_t{1} << Direct) | (std::uint64_t{1} << (SceneBuffers + Direct))), "a deferred buffer in each stage sets both ranks");
    if (both.slice.bytes != nullptr) Expect(entryIs(sectionOf(both, 0, sectionWords), SceneWords, Direct, 0u, 0u, 0u) && entryIs(sectionOf(both, 1, sectionWords), SceneWords, Direct, 0u, 0u, 0u), "both deferred entries are left at zero");

    draw.host.deferred = {SecondDirect, 0u};
    draw.first.layout.buffers.resize(32u, draw.first.layout.buffers[Direct]);
    draw.second.layout.buffers.resize(32u, draw.second.layout.buffers[Direct]);
    const auto wide = WriteDraw(stages, reads, 0u, ring);
    const auto expected = (std::uint64_t{1} << 32u) | (~std::uint64_t{0} << 35u);
    Expect(wide.left == DrawLeft::None && wide.deferred == expected, "a draw of 64 buffers defers up to the last rank");
    Expect(wide.section[1] == BlockAbi::SectionBytes(SceneWords, 32u) && wide.extra == 2u * BlockAbi::SectionBytes(SceneWords, 32u), "wider sections move the later offsets");
}

void testRing() {
    Draw draw;
    if (!draw.Ready()) return;
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
    const auto stages = draw.Stages();
    Expect(leftWith(WriteDraw(stages, draw.Reads(), 0u, ring), DrawLeft::Ring, 0u) && draw.host.reads == 0u && draw.host.resolved == 0u, "a full ring leaves the draw before the walk");

    Arena roomy;
    BlockRing other(sourceOf(roomy));
    Expect(leftWith(WriteDraw(stages, draw.Reads(), 0xffffffffu, other), DrawLeft::Ring, 0u) && draw.host.reads == 0u, "free words no chunk can hold leave the draw to the ring");
}

void testWalk() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto reads = draw.Reads();
    auto stages = draw.Stages();

    stages[1].tape = nullptr;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u), "a second stage with no tape leaves the draw as a walk of stage 1");
    stages = draw.Stages();
    stages[0].plan = nullptr;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 0u), "a first stage with no plan leaves the draw as a walk of stage 0");
    stages = draw.Stages();
    stages[1].layout = nullptr;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u), "a stage with no layout leaves the draw as a walk");
    stages = draw.Stages();
    auto refused = draw.second.layout;
    refused.refusal = BlockRefusal::WrittenBuffer;
    stages[1].layout = &refused;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u), "a refused layout leaves the draw as a walk");
    auto versioned = draw.second.layout;
    versioned.abiVersion = BlockAbi::Version + 1u;
    stages[1].layout = &versioned;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u), "a layout of another block version leaves the draw as a walk");
    auto shorter = draw.second.layout;
    shorter.words = SceneWords - 1u;
    shorter.buffers.pop_back();
    stages[1].layout = &shorter;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u), "a layout that does not count the plan's slots leaves the draw as a walk");
    auto unread = draw.second.layout;
    unread.buffers[Tabled].descriptorWords[2] = 0xffffffffu;
    stages[1].layout = &unread;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u), "a buffer whose count word the plan does not read leaves the draw as a walk");
    stages = draw.Stages();
    auto unreadable = reads;
    unreadable.read = nullptr;
    Expect(leftWith(WriteDraw(stages, unreadable, 0u, ring), DrawLeft::Walk, 0u), "a draw with no reader is left as a walk");
    auto unresolved = reads;
    unresolved.resolve = nullptr;
    Expect(leftWith(WriteDraw(stages, unresolved, 0u, ring), DrawLeft::Walk, 0u), "a draw with no resolver is left as a walk");
    std::array<DrawStage, 5> many{stages[0], stages[1], stages[0], stages[1], stages[0]};
    Expect(leftWith(WriteDraw(many, reads, 0u, ring), DrawLeft::Walk, 4u), "a fifth stage leaves the draw as a walk of stage 4");
    Expect(arena.chunks == 0u && ring.Counted().bytes == 0u && draw.host.reads == 0u && draw.host.resolved == 0u, "a draw refused before the walk takes nothing from the ring and reads nothing");

    draw.host.unreadableFrom = TableAddress + TableWords * sizeof(std::uint32_t);
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 1u) && draw.host.resolved == 2u, "a word the reader refuses in the second stage leaves the draw as a walk of stage 1");
    draw.host.resolved = 0;
    draw.host.unreadableFrom = TableAddress;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Walk, 0u) && draw.host.resolved == 0u, "a word the reader refuses in the first stage leaves the draw as a walk of stage 0");
    draw.host.unreadableFrom = ~std::uint64_t{0};
    Expect(WriteDraw(stages, reads, 0u, ring).left == DrawLeft::None, "the same draw is written once everything is readable");
}

void testVariant() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto reads = draw.Reads();
    auto stages = draw.Stages();

    const auto keyed = WriteDraw(stages, reads, 0u, ring);
    Expect(keyed.left == DrawLeft::None && keyed.variant[0] == stages[0].variantKey && keyed.variant[1] == stages[1].variantKey, "a written draw reports the variant key of each stage");
    draw.host.resolved = 0;
    stages[1].variantKey ^= 1u;
    auto unkeyed = reads;
    unkeyed.keyed = false;
    const auto free = WriteDraw(stages, unkeyed, 0u, ring);
    Expect(free.left == DrawLeft::None && free.variant[1] == (stages[1].variantKey ^ 1u) && free.images == keyed.images, "a draw written without a key to compare reports the key its words give");
    draw.host.resolved = 0;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 1u) && draw.host.resolved == 2u, "another variant key on the second stage leaves the draw as a variant of stage 1");
    draw.host.resolved = 0;
    stages = draw.Stages();
    stages[0].variantKey ^= 1u;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 0u) && draw.host.resolved == 0u, "another variant key on the first stage leaves the draw before any buffer is resolved");
    stages = draw.Stages();
    draw.second.userData[3] ^= 1u << 12u;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 1u), "a format bit that changed since the key was taken leaves the draw as a variant");
    draw.second.userData[3] ^= 1u << 12u;
    draw.second.userData[3] ^= 1u;
    Expect(WriteDraw(stages, reads, 0u, ring).left == DrawLeft::None, "a bit outside the variant mask leaves nothing");

    const auto taken = ring.Counted().bytes;
    draw.second.layout.variantKey.push_back({SceneWords, 0xffffffffu});
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 1u) && ring.Counted().bytes == taken, "a variant word outside the block leaves the draw before the ring");
    draw.second.layout.variantKey.pop_back();
    draw.first.layout.userVariantKey.push_back({16u, 0xffffffffu});
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 0u) && ring.Counted().bytes == taken, "a variant word outside the user data leaves the draw before the ring");
    draw.first.layout.userVariantKey.back().word = 12u;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 0u), "a user data variant word the key did not count leaves the draw as a variant");
    stages = draw.Stages();
    Expect(WriteDraw(stages, reads, 0u, ring).left == DrawLeft::None, "a key taken with its user data word is accepted");
    draw.first.userData[12] ^= 1u;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Variant, 0u), "a user data variant word that changed leaves the draw as a variant");
}

void testBuffers() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto reads = draw.Reads();
    const auto stages = draw.Stages();

    draw.host.refused = SecondTabled;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Buffers, 1u) && draw.host.resolved == 4u, "a buffer of the second stage the host cannot resolve leaves the draw as buffers of stage 1");
    draw.host.resolved = 0;
    draw.host.refused = FirstDirect;
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Buffers, 0u) && draw.host.resolved == 1u, "a buffer of the first stage the host cannot resolve leaves the draw as buffers of stage 0");
    draw.host.refused = 0;
    Expect(WriteDraw(stages, reads, 0u, ring).left == DrawLeft::None, "the same draw is written once every buffer resolves");

    const auto taken = ring.Counted().bytes;
    draw.host.resolved = 0;
    draw.first.layout.buffers.resize(33u, draw.first.layout.buffers[Direct]);
    draw.second.layout.buffers.resize(33u, draw.second.layout.buffers[Direct]);
    Expect(leftWith(WriteDraw(stages, reads, 0u, ring), DrawLeft::Buffers, 1u) && ring.Counted().bytes == taken && draw.host.resolved == 0u, "a 65th buffer leaves the draw as buffers of its stage, before the ring");
}

void testImages() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto reads = draw.Reads();
    const auto stages = draw.Stages();
    const auto base = WriteDraw(stages, reads, 0u, ring);
    Expect(base.left == DrawLeft::None, "the image scene is written");

    draw.second.userData[ImageUserWord] ^= 0x80000000u;
    const auto masked = WriteDraw(stages, reads, 0u, ring);
    Expect(masked.left == DrawLeft::None && masked.images == base.images, "a masked bit of an image word keeps the image digest");
    draw.second.userData[ImageUserWord] ^= 0x80000000u;

    draw.second.userData[ImageUserWord] ^= 1u;
    const auto keyed = WriteDraw(stages, reads, 0u, ring);
    Expect(keyed.left == DrawLeft::None && keyed.images != base.images, "a key bit of an image word of the second stage changes the image digest");
    draw.second.userData[ImageUserWord] ^= 1u;

    draw.first.userData[ImageUserWord] ^= 1u;
    const auto other = WriteDraw(stages, reads, 0u, ring);
    Expect(other.left == DrawLeft::None && other.images != base.images && other.images != keyed.images, "the same bit in the first stage gives another image digest");
    draw.first.userData[ImageUserWord] ^= 1u;

    draw.second.userData[UserImageWord] ^= 0x00010000u;
    const auto userMasked = WriteDraw(stages, reads, 0u, ring);
    Expect(userMasked.left == DrawLeft::None && userMasked.images == base.images, "a masked bit of a user data image word keeps the image digest");
    draw.second.userData[UserImageWord] ^= 0x00010000u;

    draw.second.userData[UserImageWord] ^= 1u;
    const auto userKeyed = WriteDraw(stages, reads, 0u, ring);
    Expect(userKeyed.left == DrawLeft::None && userKeyed.images != base.images && userKeyed.images != keyed.images, "a key bit of a user data image word changes the image digest");
    draw.second.userData[UserImageWord] ^= 1u;

    draw.second.userData[2] ^= 1u;
    draw.host.table[TableWords + 2u] ^= 4u;
    const auto buffers = WriteDraw(stages, reads, 0u, ring);
    Expect(buffers.left == DrawLeft::None && buffers.images == base.images, "words outside the image key keep the image digest");
    draw.second.userData[2] ^= 1u;
    draw.host.table[TableWords + 2u] ^= 4u;

    const auto last = WriteDraw(stages, reads, 0u, ring);
    Expect(last.left == DrawLeft::None && last.images == base.images, "the restored draw has its first image digest");
    const std::array<DrawStage, 2> swapped{stages[1], stages[0]};
    const auto reordered = WriteDraw(swapped, reads, 0u, ring);
    Expect(reordered.left == DrawLeft::None && reordered.images != base.images, "the image digest depends on the order of the stages");
}

void testEmpty() {
    Draw draw;
    if (!draw.Ready()) return;
    IrResourcePlan plan;
    plan.srtPlanComplete = true;
    plan.resourceTrackingComplete = true;
    const auto* tape = Detail::SrtTape::For(plan);
    Expect(tape != nullptr, "a plan with no read compiles to a tape");
    if (tape == nullptr) return;
    BlockLayout layout;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto reads = draw.Reads();
    DrawStage empty;
    empty.plan = &plan;
    empty.layout = &layout;
    empty.tape = tape;
    empty.variantKey = BlockVariantKey(layout, {});
    const std::array<DrawStage, 2> stages{empty, draw.second.Stage(draw.host)};
    Expect(DrawBytes(stages, 0u) == SceneBytes, "a stage with no word and no buffer takes no byte");
    const auto written = WriteDraw(stages, reads, 2u, ring);
    Expect(written.left == DrawLeft::None && written.section[0] == 0u && written.section[1] == 0u && written.extra == SceneBytes, "the section after an empty stage starts where the empty one does");
    if (written.slice.bytes != nullptr) Expect(wordsAre(sectionOf(written, 1, SceneWords), draw.second.Words(draw.host)) && written.deferred == 0u, "the stage after an empty one is written");
    draw.host.deferred = {SecondDirect, 0u};
    Expect(WriteDraw(stages, reads, 0u, ring).deferred == 1u, "an empty stage does not move the rank of the buffers that follow");
    auto wrong = stages;
    wrong[0].variantKey ^= 1u;
    Expect(leftWith(WriteDraw(wrong, reads, 0u, ring), DrawLeft::Variant, 0u), "a stage with no word still checks its variant key");

    const auto none = WriteDraw({}, reads, 0u, ring);
    Expect(none.left == DrawLeft::None && none.deferred == 0u && none.extra == 0u, "a draw with no stage leaves nothing");
    const auto loose = WriteDraw({}, reads, 3u, ring);
    Expect(loose.left == DrawLeft::None && loose.slice.bytes != nullptr && loose.extra == 0u, "a draw with no stage still takes its free words");
}

void testDriver() {
    Host host;
    Driven driven(host);
    Expect(driven.tape != nullptr, "the driven plan compiles to a tape");
    if (driven.tape == nullptr) return;
    Expect(driven.layout.Eligible() && driven.layout.words == 0u && driven.layout.buffers.size() == DrivenBuffers && driven.layout.buffers[Direct].descriptorWords[0] == 0xffffffffu && driven.layout.buffers[Tabled].descriptorWords[2] == 0xffffffffu, "a shader that reads no descriptor word has a block with no word and buffers the block does not locate");
    Expect(driven.tape->Driver().complete && driven.tape->Driver().words == DrivenZoneWords, "the tape counts the V#, T# and S# words and the active word for the driver");
    Arena arena;
    BlockRing ring(sourceOf(arena));
    DrawReads reads;
    reads.read = readWords;
    reads.resolve = resolveBuffer;
    reads.context = &host;
    const std::array<DrawStage, 1> stages{driven.Stage()};
    Expect(DrawDriverWords(stages[0]) == DrivenZoneWords && DrawBytes(stages, 0u) == DrivenBytes && DrawBytes(stages, 1u) == DrivenBytes + 16u, "the bytes of a driven draw count its driver zone, rounded up to 16 bytes");

    const auto written = WriteDraw(stages, reads, 3u, ring);
    Expect(written.left == DrawLeft::None && written.deferred == 0u && written.slice.bytes != nullptr, "a draw whose buffers only the driver reads is written");
    if (written.slice.bytes == nullptr) return;
    Expect(written.section[0] == 0u && written.driver[0] == DrivenSectionBytes && written.extra == DrivenBytes && ring.Counted().bytes == DrivenBytes + 16u, "the driver zone follows the sections and the free words follow the zone");
    const auto section = sectionOf(written, 0, DrivenSectionBytes / sizeof(std::uint32_t));
    Expect(host.resolved == 2u && host.requests[0].buffer == Direct && host.requests[0].address == DrivenDirect && host.requests[0].bytes == 60u && host.requests[0].misalignment == 2u && host.requests[0].present, "a V# in user data is decoded from the driver zone");
    Expect(host.requests[1].buffer == Tabled && host.requests[1].address == DrivenTabled && host.requests[1].bytes == 0x240u && host.requests[1].misalignment == 1u && host.requests[1].present, "a V# behind a pointer is decoded from the driver zone");
    Expect(entryIs(section, 0u, Direct, MirrorBase + DrivenDirect, 60u, 2u) && entryIs(section, 0u, Tabled, MirrorBase + DrivenTabled, 0x240u, 1u), "the entries of driven buffers hold the answers");
    const auto zone = DrawDriverZone(stages[0], written, 0u);
    const std::array<std::uint32_t, DrivenZoneWords> expected{driven.userData[0], driven.userData[1], driven.userData[2], driven.userData[3], host.table[0], host.table[1], host.table[2], host.table[3], 0x70000000u, 0x70000001u, 0x70000002u, 0x70000003u, 0x70000004u, 0x70000005u, 0x70000006u, 0x70000007u, 0x5a000000u, 0x5a000001u, 0x5a000002u, 0x5a000003u, 0x1fu};
    bool held = zone.size() == DrivenZoneWords;
    for (std::uint32_t word = 0; held && word < DrivenZoneWords; ++word) held = zone[word] == expected[word];
    Expect(held, "the driver zone holds the words of every buffer, image and sampler, then the active sources");
    Expect(driven.tape->Driver().Active(zone, driven.tape->Driver().imageBase), "the image of the driven draw is active");

    const auto again = WriteDraw(stages, reads, 0u, ring);
    Expect(again.left == DrawLeft::None && again.images == written.images && again.variant[0] == stages[0].variantKey, "the same driven draw has the same image digest and its variant key");
    driven.userData[PictureUserWord + 5u] ^= 1u;
    const auto pictured = WriteDraw(stages, reads, 0u, ring);
    Expect(pictured.left == DrawLeft::None && pictured.images != written.images, "an image word only the driver reads changes the image digest");
    driven.userData[PictureUserWord + 5u] ^= 1u;
    host.table[FilterWord + 3u] ^= 1u;
    const auto filtered = WriteDraw(stages, reads, 0u, ring);
    Expect(filtered.left == DrawLeft::None && filtered.images != written.images && filtered.images != pictured.images, "a sampler word only the driver reads changes the image digest");
    host.table[FilterWord + 3u] ^= 1u;
    driven.userData[2] ^= 1u;
    host.table[2] ^= 4u;
    const auto resized = WriteDraw(stages, reads, 0u, ring);
    Expect(resized.left == DrawLeft::None && resized.images == written.images, "the buffer words of the driver zone keep the image digest");
    driven.userData[2] ^= 1u;
    host.table[2] ^= 4u;

    host.deferred = {DrivenTabled, 0u};
    const auto deferred = WriteDraw(stages, reads, 0u, ring);
    Expect(deferred.left == DrawLeft::None && deferred.deferred == std::uint64_t{1} << Tabled, "a deferred driven buffer sets the bit of its rank");
    if (deferred.slice.bytes != nullptr) {
        const auto late = sectionOf(deferred, 0, DrivenSectionBytes / sizeof(std::uint32_t));
        Expect(entryIs(late, 0u, Tabled, 0u, 0u, 0u) && entryIs(late, 0u, Direct, MirrorBase + DrivenDirect, 60u, 2u), "a deferred driven entry is left at zero beside the written one");
        BufferAnswer answer;
        answer.deviceAddress = 0x0000007f00020000ull;
        answer.bytes = 0x200u;
        CompleteDrawBuffer(stages[0], deferred, 0u, Tabled, answer);
        Expect(entryIs(late, 0u, Tabled, 0x0000007f00020000ull, 0x200u, 1u) && entryIs(late, 0u, Direct, MirrorBase + DrivenDirect, 60u, 2u), "a deferred driven entry is completed from the driver zone of its draw");
    }
    host.deferred = {};

    driven.userData[0] = 0u;
    driven.userData[1] &= 0xffff0000u;
    const auto absent = WriteDraw(stages, reads, 0u, ring);
    Expect(absent.left == DrawLeft::None && absent.slice.bytes != nullptr, "a driven draw with a null V# is written");
    if (absent.slice.bytes != nullptr) Expect(entryIs(sectionOf(absent, 0, DrivenSectionBytes / sizeof(std::uint32_t)), 0u, Direct, 0u, 0u, BlockAbi::ServiceAbsent), "a null V# in the driver zone is an absent entry");
    driven.userData[0] = static_cast<std::uint32_t>(DrivenDirect);
    driven.userData[1] |= static_cast<std::uint32_t>(DrivenDirect >> 32u);

    Draw draw;
    if (!draw.Ready()) return;
    Driven second(draw.host);
    const std::array<DrawStage, 2> mixed{draw.second.Stage(draw.host), second.Stage()};
    const auto both = WriteDraw(mixed, draw.Reads(), 0u, ring);
    Expect(both.left == DrawLeft::None && both.section[1] == SceneBytes && both.driver[0] == SceneBytes + DrivenSectionBytes && both.driver[1] == SceneBytes + DrivenSectionBytes && both.extra == SceneBytes + DrivenBytes, "a stage with no driver zone takes none before the zone of the next stage");
    Expect(DrawDriverZone(mixed[0], both, 0u).empty() && DrawDriverZone(mixed[1], both, 1u).size() == DrivenZoneWords && DrawBytes(mixed, 0u) == SceneBytes + DrivenBytes, "each stage names its own driver zone");

    const auto before = allocations;
    const auto plain = WriteDraw(stages, reads, 0u, ring);
    host.unreadableFrom = TableAddress;
    const auto unreadable = WriteDraw(stages, reads, 0u, ring);
    host.unreadableFrom = ~std::uint64_t{0};
    Expect(allocations == before, "writing a driven draw, or leaving one, allocates nothing");
    Expect(plain.left == DrawLeft::None && leftWith(unreadable, DrawLeft::Walk, 0u), "a driver word the reader refuses leaves the draw as a walk");

    Driven loose(host);
    loose.plan.materializationSources = {1u, 2u, 3u};
    loose.plan.tapeSlot = std::make_shared<SrtTapeSlot>();
    loose.tape = Detail::SrtTape::For(loose.plan);
    const auto stage = loose.Stage();
    Expect(loose.tape != nullptr && !loose.tape->Driver().complete && leftWith(WriteDraw(std::span<const DrawStage>(&stage, 1), reads, 0u, ring), DrawLeft::Walk, 0u), "a buffer neither the block nor the driver zone locates leaves the draw as a walk");
}

void testAllocations() {
    Draw draw;
    if (!draw.Ready()) return;
    Arena arena;
    BlockRing ring(sourceOf(arena));
    const auto reads = draw.Reads();
    const auto stages = draw.Stages();
    auto wrong = stages;
    wrong[1].variantKey ^= 1u;
    Expect(WriteDraw(stages, reads, 4u, ring).left == DrawLeft::None, "the first draw warms the ring");
    draw.host.deferred = {SecondTabled, 0u};
    const auto before = allocations;
    const auto plain = WriteDraw(stages, reads, 4u, ring);
    const auto bytes = DrawBytes(stages, 4u);
    const auto variant = WriteDraw(wrong, reads, 0u, ring);
    draw.host.refused = FirstTabled;
    const auto refused = WriteDraw(stages, reads, 0u, ring);
    draw.host.unreadableFrom = TableAddress;
    const auto unreadable = WriteDraw(stages, reads, 0u, ring);
    const auto after = allocations;
    Expect(after == before, "writing a draw, or leaving one, allocates nothing");
    Expect(plain.left == DrawLeft::None && plain.deferred == std::uint64_t{1} << (SceneBuffers + Tabled) && bytes == 2u * SceneBytes + 16u, "the measured draw is written with its deferred buffer");
    Expect(variant.left == DrawLeft::Variant && refused.left == DrawLeft::Buffers && refused.stage == 0u && unreadable.left == DrawLeft::Walk, "the measured faults are the expected ones");
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
    testNames();
    testBytes();
    testDraw();
    testDeferred();
    testRing();
    testWalk();
    testVariant();
    testBuffers();
    testImages();
    testEmpty();
    testDriver();
    testAllocations();
    if (failures != 0) {
        std::fprintf(stderr, "%d decode draw check(s) failed\n", failures);
        return 1;
    }
    std::printf("decode draw: ok\n");
    return 0;
}
