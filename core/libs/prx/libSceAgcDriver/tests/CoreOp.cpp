#include "prx/libSceAgcDriver/Core/include/Op.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string_view>
#include <type_traits>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::array<OpKind, 7> Kinds{OpKind::None, OpKind::Draw, OpKind::Label, OpKind::SubmitEnd, OpKind::Seal, OpKind::Legacy, OpKind::Finished};
constexpr std::size_t HeaderBytes = 16;
constexpr std::size_t UsedBytes = HeaderBytes + sizeof(DrawOp);
constexpr OpRange Untouched{0x1111111111111111ull, 0x2222222222222222ull};

bool same(const OpRange& left, const OpRange& right) { return left.address == right.address && left.bytes == right.bytes; }

std::array<OpRange, 6> untouched() {
    std::array<OpRange, 6> ranges;
    ranges.fill(Untouched);
    return ranges;
}

bool untouchedFrom(std::span<const OpRange> ranges, std::size_t first) {
    for (std::size_t index = first; index < ranges.size(); ++index) {
        if (!same(ranges[index], Untouched)) return false;
    }
    return true;
}

Op legacyOp(std::uint32_t writeCount) {
    auto op = MakeOp(OpKind::Legacy);
    op.legacy.slot = 7;
    op.legacy.writeCount = writeCount;
    for (std::size_t index = 0; index < op.legacy.writes.size(); ++index) op.legacy.writes[index] = OpRange{0x1000u * (index + 1u), 0x10u + index};
    return op;
}

void testLayout() {
    Expect(sizeof(Op) == 128, "an operation is 128 bytes");
    Expect(alignof(Op) == 64, "an operation is aligned on 64 bytes");
    Expect(std::is_trivially_copyable_v<Op>, "an operation is trivially copyable");
    Expect(std::is_standard_layout_v<Op>, "an operation has a standard layout");
    Expect(std::is_trivially_copyable_v<DrawOp> && std::is_trivially_copyable_v<LabelOp> && std::is_trivially_copyable_v<SubmitOp> && std::is_trivially_copyable_v<LegacyOp> && std::is_trivially_copyable_v<FinishedOp>, "every payload is trivially copyable");
    Expect(sizeof(RingSlice) == 24, "a ring slice is 24 bytes");
    Expect(sizeof(OpRange) == 16, "a range is 16 bytes");
    Expect(sizeof(DrawOp) == 104, "a draw payload is 104 bytes");
    Expect(sizeof(LabelOp) == 40, "a label payload is 40 bytes");
    Expect(sizeof(LegacyOp) == 80, "a legacy payload is 80 bytes");
    Expect(sizeof(DrawOp) >= sizeof(LabelOp) && sizeof(DrawOp) >= sizeof(SubmitOp) && sizeof(DrawOp) >= sizeof(LegacyOp) && sizeof(DrawOp) >= sizeof(FinishedOp), "the draw payload is the largest");
    Expect(offsetof(Op, kind) == 0 && offsetof(Op, stages) == 1 && offsetof(Op, flags) == 2 && offsetof(Op, epoch) == 4 && offsetof(Op, order) == 8, "the header fields are packed in 16 bytes");
    Expect(offsetof(Op, draw) == HeaderBytes && offsetof(Op, label) == HeaderBytes && offsetof(Op, submit) == HeaderBytes && offsetof(Op, legacy) == HeaderBytes && offsetof(Op, finished) == HeaderBytes, "every payload starts after the header");
    Expect(UsedBytes == 120 && UsedBytes <= sizeof(Op), "header and payload fit in the operation");
    Expect(static_cast<std::size_t>(OpKind::Count) == Kinds.size(), "the kind list of the test is complete");
}

void testMakeOp() {
    for (const auto kind : Kinds) {
        const auto op = MakeOp(kind);
        Expect(op.kind == kind, "a made operation has its kind");
        Expect(op.stages == 0 && op.flags == 0 && op.epoch == 0 && op.order == 0, "a made operation has a zero header");
        std::array<unsigned char, sizeof(Op)> bytes{};
        std::memcpy(bytes.data(), &op, sizeof(Op));
        Expect(bytes[0] == static_cast<unsigned char>(kind), "the kind is the first byte");
        bool zero = true;
        for (std::size_t index = 1; index < UsedBytes; ++index) zero = zero && bytes[index] == 0;
        Expect(zero, "a made operation is zero after its kind");
    }
    const auto draw = MakeOp(OpKind::Draw);
    Expect(draw.draw.variant == nullptr && draw.draw.slice.bytes == nullptr && draw.draw.slice.deviceAddress == 0 && draw.draw.slice.chunk == 0 && draw.draw.slice.lap == 0, "a made draw has no variant and no slice");
    Expect(draw.draw.section[0] == 0 && draw.draw.section[3] == 0 && draw.draw.deferred == 0 && draw.draw.indexAddress == 0 && draw.draw.indexBytes == 0 && draw.draw.indexCount == 0, "a made draw has zero sections and indices");
    Expect(draw.draw.firstIndex == 0 && draw.draw.baseVertex == 0 && draw.draw.instances == 0 && draw.draw.firstInstance == 0 && draw.draw.extra == 0 && draw.draw.extraWords == 0 && draw.draw.opcode == 0 && draw.draw.drawFlags == 0, "a made draw has zero parameters");
    const auto label = MakeOp(OpKind::Label);
    bool labelZero = label.label.address == 0 && label.label.size == 0 && label.label.interrupt == 0 && label.label.opcode == 0 && label.label.labelFlags == 0;
    for (const auto byte : label.label.bytes) labelZero = labelZero && byte == std::byte{0};
    Expect(labelZero, "a made label is zero");
    const auto submit = MakeOp(OpKind::SubmitEnd);
    Expect(submit.submit.submission == 0 && submit.submit.submitFlags == 0, "a made submission end is zero");
    const auto legacy = MakeOp(OpKind::Legacy);
    bool legacyZero = legacy.legacy.slot == 0 && legacy.legacy.commitKind == 0 && legacy.legacy.writeCount == 0 && legacy.legacy.legacyFlags == 0;
    for (const auto& range : legacy.legacy.writes) legacyZero = legacyZero && range.address == 0 && range.bytes == 0;
    Expect(legacyZero, "a made legacy operation is zero");
    Expect(MakeOp(OpKind::Finished).finished.job == 0, "a made finished operation is zero");
}

void testCopies() {
    auto op = MakeOp(OpKind::Draw);
    op.stages = 2;
    op.flags = 0x1234u;
    op.epoch = 0x89abcdefu;
    op.order = 0x0123456789abcdefull;
    op.draw.variant = &op;
    op.draw.section = {0u, 64u, 128u, 192u};
    op.draw.deferred = 0x5u;
    op.draw.indexAddress = 0x7000u;
    op.draw.indexCount = 36;
    op.draw.baseVertex = -3;
    op.draw.drawFlags = DrawIndexed | DrawIndex32;
    Op raw{};
    std::memcpy(static_cast<void*>(&raw), &op, sizeof(Op));
    const Op copy = op;
    Expect(std::memcmp(&raw, &op, UsedBytes) == 0 && std::memcmp(&copy, &op, UsedBytes) == 0, "an operation copies byte for byte");
    Expect(copy.kind == OpKind::Draw && copy.stages == 2 && copy.flags == 0x1234u && copy.epoch == 0x89abcdefu && copy.order == 0x0123456789abcdefull, "a copy keeps the header");
    Expect(copy.draw.variant == &op && copy.draw.section[2] == 128u && copy.draw.deferred == 0x5u && copy.draw.indexAddress == 0x7000u && copy.draw.indexCount == 36 && copy.draw.baseVertex == -3 && copy.draw.drawFlags == 3u, "a copy keeps the draw payload");
}

void testFlags() {
    Expect(DrawIndexed == 1u && DrawIndex32 == 2u, "the draw flags");
    Expect(LabelInterrupt == 1u, "the label flag");
    Expect(SubmitHoldsFlip == 1u, "the submission flag");
    Expect(LegacyUnknownWrites == 1u && LegacyStore == 2u, "the legacy flags");
}

void testWritesOfOtherKinds() {
    for (const auto kind : {OpKind::None, OpKind::Draw, OpKind::SubmitEnd, OpKind::Seal, OpKind::Finished}) {
        auto op = MakeOp(kind);
        auto ranges = untouched();
        Expect(OpWrites(op, ranges) == 0 && untouchedFrom(ranges, 0), "an operation that is neither a label nor legacy writes nothing");
        std::memset(static_cast<void*>(&op.draw), 0xff, sizeof(DrawOp));
        Expect(OpWrites(op, ranges) == 0 && untouchedFrom(ranges, 0), "its payload does not make it write");
        Expect(OpWrites(op, std::span<OpRange>{}) == 0, "it writes nothing into an empty output");
    }
    auto count = MakeOp(OpKind::Count);
    auto ranges = untouched();
    Expect(OpWrites(count, ranges) == 0 && untouchedFrom(ranges, 0), "a kind beyond the list writes nothing");
}

void testWritesOfLabel() {
    auto op = MakeOp(OpKind::Label);
    op.label.address = 0x200040000ull;
    op.label.size = 8;
    op.label.interrupt = 3;
    op.label.labelFlags = LabelInterrupt;
    auto ranges = untouched();
    Expect(OpWrites(op, ranges) == 1, "a label writes one range");
    Expect(same(ranges[0], OpRange{0x200040000ull, 8u}) && untouchedFrom(ranges, 1), "the range of a label is its address and its size");
    op.label.address = 0xfffffffffffffff0ull;
    op.label.size = 16;
    ranges = untouched();
    Expect(OpWrites(op, ranges) == 1 && same(ranges[0], OpRange{0xfffffffffffffff0ull, 16u}), "the range of a label keeps the 64 bits of its address");
    std::array<OpRange, 1> one{Untouched};
    Expect(OpWrites(op, one) == 1 && same(one[0], OpRange{0xfffffffffffffff0ull, 16u}), "a label fits an output of one range");
    Expect(OpWrites(op, std::span<OpRange>{}) == 0, "a label writes nothing into an empty output");
}

void testWritesOfLegacy() {
    for (std::uint32_t writeCount = 0; writeCount <= 4; ++writeCount) {
        const auto op = legacyOp(writeCount);
        auto ranges = untouched();
        const auto written = OpWrites(op, ranges);
        bool matches = written == writeCount;
        for (std::uint32_t index = 0; index < writeCount && matches; ++index) matches = same(ranges[index], op.legacy.writes[index]);
        Expect(matches && untouchedFrom(ranges, writeCount), "a legacy operation writes its first writeCount ranges, in order");
    }
    const auto full = legacyOp(4);
    std::array<OpRange, 2> two{Untouched, Untouched};
    Expect(OpWrites(full, two) == 2 && same(two[0], full.legacy.writes[0]) && same(two[1], full.legacy.writes[1]), "a short output takes the first ranges only");
    Expect(OpWrites(full, std::span<OpRange>{}) == 0, "a legacy operation writes nothing into an empty output");
    for (const auto writeCount : {5u, 64u, 0xffffffffu}) {
        const auto op = legacyOp(writeCount);
        auto ranges = untouched();
        Expect(OpWrites(op, ranges) == 4 && same(ranges[3], op.legacy.writes[3]) && untouchedFrom(ranges, 4), "a write count beyond the payload is bounded to its four ranges");
    }
    auto unknown = legacyOp(1);
    unknown.legacy.legacyFlags = LegacyUnknownWrites | LegacyStore;
    auto ranges = untouched();
    Expect(OpWrites(unknown, ranges) == 1 && same(ranges[0], unknown.legacy.writes[0]), "the legacy flags do not change the listed ranges");
}

void testNames() {
    constexpr std::array<std::string_view, 7> names{"None", "Draw", "Label", "SubmitEnd", "Seal", "Legacy", "Finished"};
    for (std::size_t index = 0; index < Kinds.size(); ++index) {
        const auto* name = OpKindName(Kinds[index]);
        Expect(name != nullptr && names[index] == name, "every kind has its name");
    }
    for (std::uint32_t value = 0; value < 256; ++value) {
        const auto* name = OpKindName(static_cast<OpKind>(value));
        Expect(name != nullptr && name[0] != 0, "every value of the kind byte has a name");
        if (value < Kinds.size() || name == nullptr) continue;
        for (const auto listed : names) Expect(listed != name, "a value beyond the list does not take the name of a kind");
    }
}

}

int main() {
    testLayout();
    testMakeOp();
    testCopies();
    testFlags();
    testWritesOfOtherKinds();
    testWritesOfLabel();
    testWritesOfLegacy();
    testNames();
    if (failures != 0) {
        std::fprintf(stderr, "%d op checks failed\n", failures);
        return 1;
    }
    std::printf("op tests passed\n");
    return 0;
}
