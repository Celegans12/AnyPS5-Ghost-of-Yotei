#include "CoreDecoderStreams.hpp"
#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace AgcDriver;
using namespace AgcDriver::Core;
using namespace DecoderTests;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

void ExpectAt(bool condition, std::uint64_t seed, std::size_t draw, const char* what) {
    if (condition) return;
    if (failures < 40) std::fprintf(stderr, "FAIL: seed %llu draw %zu: %s\n", static_cast<unsigned long long>(seed), draw, what);
    ++failures;
}

bool sameQueue(const QueueState& first, const QueueState& second) {
    return first.shader == second.shader && first.context == second.context && first.userConfig == second.userConfig && first.indexBase == second.indexBase && first.indexBufferSize == second.indexBufferSize && first.indexType == second.indexType && first.instanceCount == second.instanceCount && first.staleDigests == second.staleDigests && first.shaderDigest == second.shaderDigest && first.contextDigest == second.contextDigest && first.userConfigDigest == second.userConfigDigest;
}

QueueState freshQueue() {
    QueueState queue;
    queue.staleDigests = 0;
    queue.shaderDigest = {};
    queue.contextDigest = {};
    queue.userConfigDigest = {};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> registers;
    queue.context.ForEach([&](std::uint32_t offset, std::uint32_t value) { registers.emplace_back(offset, value); });
    queue.contextDigest = DigestOf(registers, Bank::Context);
    registers.clear();
    queue.userConfig.ForEach([&](std::uint32_t offset, std::uint32_t value) { registers.emplace_back(offset, value); });
    queue.userConfigDigest = DigestOf(registers, Bank::UserConfig);
    return queue;
}

struct Harness {
    std::uint64_t seed = 0;
    const std::vector<std::pair<QueueState, std::uint32_t>>* expected = nullptr;
    std::vector<DecodedOp> ops;
    DecodedView view;
    std::uint64_t retired = 0;
    bool lagging = false;
    std::uint64_t heavy = 0;
    std::size_t stopAt = std::numeric_limits<std::size_t>::max();
    std::size_t jumpFrom = std::numeric_limits<std::size_t>::max();
    std::size_t jumpTo = 0;
    bool stuck = false;
    std::deque<std::pair<DecodedOp, std::size_t>> flying;

    void check(const DecodedOp& op, std::size_t index, const char* when) {
        if (expected == nullptr || index >= expected->size()) {
            ExpectAt(expected == nullptr, seed, index, "the decoder posts more draws than the reference executes");
            return;
        }
        const auto& [queue, bits] = (*expected)[index];
        ExpectAt(SameDecoded(op.state, queue), seed, index, when);
    }

    void retire(std::uint64_t count) {
        while (retired < count && !flying.empty()) {
            check(flying.front().first, flying.front().second, "a register snapshot changed before the operation reading it retired");
            flying.pop_front();
            ++retired;
        }
    }
};

std::uint64_t onDraw(void* context, const DecodedOp& op) {
    auto& harness = *static_cast<Harness*>(context);
    const auto index = harness.ops.size();
    harness.ops.push_back(op);
    harness.ops.back().order = index;
    harness.check(op, index, "the state an operation carries differs from the reference state at its draw");
    if (harness.expected != nullptr && index < harness.expected->size()) {
        const auto& [queue, bits] = (*harness.expected)[index];
        ExpectAt(op.state.runBits == bits, harness.seed, index, "the run mask an operation carries differs from the reference");
        ApplyDecoded(op.state, harness.view);
        ExpectAt(sameQueue(harness.view.queue, queue), harness.seed, index, "the state applied to a view differs from the reference state");
        ExpectAt(harness.view.queue.runMask.bits == bits, harness.seed, index, "the run mask applied to a view differs from the reference");
    }
    if (harness.lagging) harness.flying.emplace_back(op, index);
    else harness.retired = index + 1;
    return index;
}

DecoderHeavy onHeavy(void* context, std::size_t cursor, std::span<const std::uint32_t> packet, QueueState& live) {
    auto& harness = *static_cast<Harness*>(context);
    ++harness.heavy;
    if (harness.stuck) return {cursor, 0, false};
    if (cursor == harness.stopAt) return {cursor, 0, true};
    if (cursor == harness.jumpFrom) return {harness.jumpTo, 0, false};
    const auto lane = LaneOf(packet[0]);
    if (lane != Lane::Heavy) return {cursor + packet.size(), DecoderEvery, false};
    return {cursor + packet.size(), ReferenceHeavy(live, packet), false};
}

std::uint64_t onRetired(void* context) {
    return static_cast<Harness*>(context)->retired;
}

void onAwait(void* context, std::uint64_t count) {
    static_cast<Harness*>(context)->retire(count);
}

DecoderHost hostOf(Harness& harness) {
    return {&harness, &onDraw, &onHeavy, &onRetired, &onAwait};
}

Decoder::Input inputOf(const Stream& stream) {
    Decoder::Input input;
    input.commands = stream.words;
    input.limit = stream.words.size();
    input.lists = &stream.lists;
    return input;
}

void testRandomStreams() {
    for (std::uint64_t seed = 1; seed <= 24; ++seed) {
        const auto stream = RandomStream(seed, 3000);
        const auto expected = ReferenceDraws(stream, freshQueue());
        Harness harness;
        harness.seed = seed;
        harness.expected = &expected;
        Decoder decoder({512}, hostOf(harness));
        auto live = freshQueue();
        const auto result = decoder.Decode(live, inputOf(stream));
        Expect(!result.stopped && result.cursor == stream.words.size(), "a random stream decodes to its end");
        Expect(harness.ops.size() == expected.size() && expected.size() == stream.draws, "the decoder posts one operation per draw of the stream");
        const auto counts = decoder.Counted();
        Expect(counts.draws == stream.draws, "the decoder counts every draw");
        Expect(counts.listsLeft == 0, "every register list of a random stream is taken from its copy");
        Expect(counts.shaderSnapshots <= counts.draws && counts.contextSnapshots <= counts.draws && counts.userConfigSnapshots <= counts.draws, "a bank is copied at most once per draw");
        Expect(counts.shaderSnapshots + counts.contextSnapshots + counts.userConfigSnapshots < 3 * counts.draws, "a bank that did not change is not copied again");
        Expect(counts.unchanged != 0, "writes of the value a register already holds are counted");
        for (std::size_t index = 0; index < harness.ops.size(); ++index) {
            const auto& op = harness.ops[index];
            const auto carried = op.Words();
            if (carried.data() != op.inlineWords.data() || !std::equal(carried.begin(), carried.end(), stream.words.begin() + op.offset) || op.kind != DecodedKind::Draw || op.packet != stream.words.data() + op.offset || op.words != LaneWords(op.packet[0]) || op.opcode != ((op.packet[0] >> 8u) & 0xffu) || LaneOf(op.packet[0]) != Lane::Draw) {
                ExpectAt(false, seed, index, "an operation does not name its draw packet");
                break;
            }
        }
    }
}

void testSplitDecode() {
    const auto stream = RandomStream(77, 2000);
    const auto expected = ReferenceDraws(stream, freshQueue());
    Harness harness;
    harness.seed = 77;
    harness.expected = &expected;
    Decoder decoder({512}, hostOf(harness));
    auto live = freshQueue();
    Random random{5};
    std::size_t cursor = 0;
    std::size_t calls = 0;
    while (cursor < stream.words.size()) {
        auto input = inputOf(stream);
        input.cursor = cursor;
        input.limit = std::min(stream.words.size(), cursor + 1 + random.Below(40));
        const auto result = decoder.Decode(live, input);
        if (result.stopped) {
            Expect(result.cursor < input.limit && result.cursor + LaneWords(stream.words[result.cursor]) > input.limit, "a decode stops inside its limit only at a packet that crosses it");
            input.cursor = result.cursor;
            input.limit = result.cursor + LaneWords(stream.words[result.cursor]);
            const auto whole = decoder.Decode(live, input);
            Expect(!whole.stopped && whole.cursor == input.limit, "the packet that crossed a limit decodes once the limit covers it");
            cursor = whole.cursor;
        } else {
            cursor = result.cursor;
        }
        ++calls;
    }
    Expect(calls > 50, "the stream was decoded in many calls");
    Expect(harness.ops.size() == expected.size(), "a stream decoded in pieces posts the same draws");
}

void testSnapshotsFollowChanges() {
    Stream stream;
    const std::uint32_t one[] = {1u};
    const std::uint32_t two[] = {2u};
    stream.Set(0x69, 0x318, one);
    stream.Set(0x76, 0x0c, one);
    stream.DrawAuto(3);
    stream.Set(0x76, 0x0c, two);
    stream.DrawAuto(3);
    stream.Set(0x76, 0x0c, two);
    stream.Set(0x69, 0x318, one);
    stream.DrawAuto(3);
    stream.Packet(0x2f, {7u});
    stream.DrawAuto(3);
    stream.Set(0x79, 0x250, one);
    stream.DrawAuto(3);
    const auto expected = ReferenceDraws(stream, freshQueue());
    Harness harness;
    harness.expected = &expected;
    Decoder decoder({8}, hostOf(harness));
    auto live = freshQueue();
    const auto result = decoder.Decode(live, inputOf(stream));
    Expect(!result.stopped && harness.ops.size() == 5, "five draws are posted");
    if (harness.ops.size() != 5) return;
    const auto& ops = harness.ops;
    Expect(ops[1].state.shaderVersion != ops[0].state.shaderVersion && ops[1].state.shader != ops[0].state.shader, "a changed shader bank gets a new snapshot");
    Expect(ops[1].state.contextVersion == ops[0].state.contextVersion && ops[1].state.context == ops[0].state.context, "an unchanged context bank keeps its snapshot");
    Expect(ops[2].state.shaderVersion == ops[1].state.shaderVersion && ops[2].state.contextVersion == ops[1].state.contextVersion, "rewriting the values the registers hold takes no snapshot");
    Expect(ops[3].state.shaderVersion == ops[2].state.shaderVersion && ops[3].state.instanceCount == 7 && ops[2].state.instanceCount == 1, "a parameter travels in the operation without a snapshot");
    Expect(ops[4].state.userConfigVersion != ops[3].state.userConfigVersion && ops[4].state.shaderVersion == ops[3].state.shaderVersion, "a changed user configuration bank alone gets a new snapshot");
    const auto counts = decoder.Counted();
    Expect(counts.shaderSnapshots == 2 && counts.contextSnapshots == 1 && counts.userConfigSnapshots == 2, "snapshots are counted per bank");
    Expect(counts.unchanged == 2, "two writes found their value already there");
    Expect((ops[0].state.runBits & RunMask::State) != 0, "the first draw follows a state change");
    Expect(ops[1].state.runBits == RunMask::Words, "a draw after user words alone continues the run");
    Expect(ops[2].state.runBits == 0, "a draw after rewrites of the same values continues the run untouched");
    Expect(ops[3].state.runBits == RunMask::Parameters, "a draw after a parameter carries the parameter bit");
}

void testSnapshotLifetime() {
    for (std::uint64_t seed = 100; seed < 106; ++seed) {
        const auto stream = RandomStream(seed, 4000);
        const auto expected = ReferenceDraws(stream, freshQueue());
        Harness harness;
        harness.seed = seed;
        harness.expected = &expected;
        harness.lagging = true;
        Decoder decoder({4}, hostOf(harness));
        auto live = freshQueue();
        const auto result = decoder.Decode(live, inputOf(stream));
        Expect(!result.stopped && harness.ops.size() == expected.size(), "a stream decodes with a ring of four snapshots");
        Expect(decoder.Counted().snapshotWaits != 0, "the decoder waits for readers before it reuses a snapshot");
        Expect(!harness.flying.empty(), "operations are still unretired at the end of the stream");
        harness.retire(std::numeric_limits<std::uint64_t>::max());
    }
}

void testSnapshotWithoutWait() {
    Stream stream;
    const std::uint32_t one[] = {1u};
    for (std::uint32_t index = 0; index < 8; ++index) {
        const std::uint32_t value[] = {index + 10};
        stream.Set(0x76, 0x0c, value);
        stream.DrawAuto(3);
    }
    stream.Set(0x76, 0x0d, one);
    Harness harness;
    harness.lagging = true;
    DecoderHost host = hostOf(harness);
    host.awaitRetired = nullptr;
    Decoder decoder({4}, host);
    auto live = freshQueue();
    bool threw = false;
    try {
        static_cast<void>(decoder.Decode(live, inputOf(stream)));
    } catch (const std::logic_error&) {
        threw = true;
    }
    Expect(threw, "a snapshot still read cannot be reused when the host cannot wait");
    Expect(harness.ops.size() == 4, "four draws fit four snapshots before the decoder must wait");
}

void testHeavyContract() {
    Stream stream;
    const std::uint32_t one[] = {1u};
    stream.Set(0x69, 0x318, one);
    stream.DrawAuto(3);
    const auto clear = stream.Packet(HeavyClearState, {0u});
    stream.DrawAuto(3);
    const auto jump = stream.Packet(HeavyNop, {0u});
    stream.Set(0x69, 0x318, one);
    stream.DrawAuto(3);
    const auto landing = stream.words.size();
    stream.DrawAuto(3);
    const auto stopAt = stream.Packet(HeavyNop, {0u});
    stream.DrawAuto(3);
    {
        Harness harness;
        Decoder decoder({8}, hostOf(harness));
        auto live = freshQueue();
        harness.jumpFrom = jump;
        harness.jumpTo = landing;
        harness.stopAt = stopAt;
        const auto result = decoder.Decode(live, inputOf(stream));
        Expect(result.stopped && result.cursor == stopAt, "the decode stops where the host stops it");
        Expect(harness.ops.size() == 3, "the packets the host jumps over are not decoded");
        if (harness.ops.size() == 3) {
            Expect(harness.ops[1].state.contextVersion != harness.ops[0].state.contextVersion, "a bank the host changed gets a new snapshot");
            Expect(std::as_const(*harness.ops[1].state.context).at(0x318) == 0u && std::as_const(*harness.ops[0].state.context).at(0x318) == 1u, "the snapshot after a host packet holds what the host wrote");
            Expect(harness.ops[2].state.contextVersion == harness.ops[1].state.contextVersion, "a host packet that changes no bank takes no snapshot");
            Expect(harness.ops[2].offset == landing, "decoding resumes where the host says");
        }
        Expect(harness.heavy == 3 && decoder.Counted().heavy == 3, "three packets went to the host");
        static_cast<void>(clear);
    }
    {
        Harness harness;
        harness.stuck = true;
        Decoder decoder({8}, hostOf(harness));
        auto live = freshQueue();
        bool threw = false;
        try {
            static_cast<void>(decoder.Decode(live, inputOf(stream)));
        } catch (const std::logic_error&) {
            threw = true;
        }
        Expect(threw, "a host that does not advance past its packet is refused");
    }
    {
        Harness harness;
        DecoderHost host = hostOf(harness);
        host.heavy = nullptr;
        Decoder decoder({8}, host);
        auto live = freshQueue();
        const auto result = decoder.Decode(live, inputOf(stream));
        Expect(result.stopped && result.cursor == clear && harness.ops.size() == 1, "without a host for heavy packets the decode stops at the first one");
    }
}

void testListsAndEdges() {
    Stream stream;
    const std::uint32_t one[] = {1u};
    const auto valid = stream.List(0x63, {0x0cu, 5u, 0x20000243u, 9u});
    const auto wrongBank = stream.List(0x64, {0x20000243u, 2u});
    const auto missing = stream.Packet(0x9f, {0x1000u, 0u, 0x80000000u, 1u});
    const auto invalid = stream.List(0x9f, {0xffffffffu, 1u});
    const auto shortList = stream.List(0x9f, {0x318u, 1u});
    stream.lists[shortList] = {0x318u, 1u, 0x319u, 2u};
    const auto tag = stream.Set(0x79, 0x342, one);
    const auto extended = stream.Set(0x69, 0x80000318u, one);
    const auto indexed = stream.Set(0x7a, 0x20000243u, one);
    stream.DrawAuto(3);
    Harness harness;
    Decoder decoder({8}, hostOf(harness));
    auto live = freshQueue();
    const auto result = decoder.Decode(live, inputOf(stream));
    Expect(!result.stopped && harness.ops.size() == 1, "the stream of edge cases decodes");
    const auto counts = decoder.Counted();
    Expect(counts.lists == 2 && counts.listsLeft == 3, "two register lists are taken, three are left to the host");
    Expect(harness.heavy == 4, "the missing, invalid and mismatched lists and the extended offset go to the host");
    Expect(live.shader.contains(0x0c) && std::as_const(live.shader).at(0x0c) == 5u && live.shader.contains(0x243) && std::as_const(live.shader).at(0x243) == 9u, "a register list writes its bank with the offset flags masked");
    Expect(live.userConfig.contains(0x243) && std::as_const(live.userConfig).at(0x243) == 1u && live.indexType == 1u, "the index type follows the user configuration register 0x243");
    Expect(!live.userConfig.contains(0x342), "a tag marker writes nothing");
    Expect(std::as_const(live.context).at(0x318) == 0u, "a register packet with an extended offset is not decoded here");
    static_cast<void>(valid);
    static_cast<void>(wrongBank);
    static_cast<void>(missing);
    static_cast<void>(invalid);
    static_cast<void>(tag);
    static_cast<void>(extended);
    static_cast<void>(indexed);
}

void testAdoptAndCapture() {
    Stream stream;
    const std::uint32_t one[] = {1u};
    stream.Set(0x76, 0x0c, one);
    stream.DrawAuto(3);
    Harness harness;
    Decoder decoder({8}, hostOf(harness));
    auto first = freshQueue();
    auto second = freshQueue();
    static_cast<void>(decoder.Decode(first, inputOf(stream)));
    static_cast<void>(decoder.Decode(first, inputOf(stream)));
    static_cast<void>(decoder.Decode(second, inputOf(stream)));
    Expect(harness.ops.size() == 3, "three draws are posted over two queue states");
    if (harness.ops.size() != 3) return;
    Expect(harness.ops[1].state.shaderVersion == harness.ops[0].state.shaderVersion, "the same queue state decoded again keeps its snapshots");
    Expect(harness.ops[2].state.shaderVersion != harness.ops[1].state.shaderVersion && harness.ops[2].state.contextVersion != harness.ops[1].state.contextVersion && harness.ops[2].state.userConfigVersion != harness.ops[1].state.userConfigVersion, "another queue state gets snapshots of its own");
    second.context.insert_or_assign(0x318, 44u);
    const auto stale = decoder.Capture(second);
    Expect(stale.contextVersion == harness.ops[2].state.contextVersion, "a change the decoder was not told of takes no snapshot");
    decoder.Touch(DecoderContext);
    const auto captured = decoder.Capture(second);
    Expect(captured.contextVersion != stale.contextVersion && std::as_const(*captured.context).at(0x318) == 44u, "a touched bank is copied at the next capture");
    decoder.Bind(captured, 9);
    DecodedState foreign = captured;
    Registers elsewhere;
    foreign.shader = &elsewhere;
    bool threw = false;
    try {
        decoder.Bind(foreign, 10);
    } catch (const std::logic_error&) {
        threw = true;
    }
    Expect(threw, "binding an operation to a snapshot of another decoder is refused");
}

void testViewCopiesOnlyChanges() {
    const auto stream = RandomStream(9, 2500);
    Harness harness;
    Decoder decoder({512}, hostOf(harness));
    auto live = freshQueue();
    static_cast<void>(decoder.Decode(live, inputOf(stream)));
    std::uint64_t changes = 0;
    for (std::size_t index = 0; index < harness.ops.size(); ++index) {
        const auto& state = harness.ops[index].state;
        if (index == 0) changes += 3;
        else {
            const auto& before = harness.ops[index - 1].state;
            changes += (state.shaderVersion != before.shaderVersion ? 1 : 0) + (state.contextVersion != before.contextVersion ? 1 : 0) + (state.userConfigVersion != before.userConfigVersion ? 1 : 0);
        }
    }
    DecodedView fresh;
    for (const auto& op : harness.ops) ApplyDecoded(op.state, fresh);
    Expect(fresh.copies == changes && changes < 3 * harness.ops.size(), "a view copies a bank only when its version changes");
}

}

int main() {
    testRandomStreams();
    testSplitDecode();
    testSnapshotsFollowChanges();
    testSnapshotLifetime();
    testSnapshotWithoutWait();
    testHeavyContract();
    testListsAndEdges();
    testAdoptAndCapture();
    testViewCopiesOnlyChanges();
    if (failures != 0) {
        std::fprintf(stderr, "%d decoder checks failed\n", failures);
        return 1;
    }
    std::printf("decoder tests passed\n");
    return 0;
}
