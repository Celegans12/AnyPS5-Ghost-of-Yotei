#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4Opcodes.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

void ExpectHeader(bool condition, std::uint32_t header, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: header 0x%08x: %s\n", static_cast<unsigned>(header), what);
    ++failures;
}

void ExpectOpcode(bool condition, std::uint32_t opcode, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: opcode 0x%x: %s\n", static_cast<unsigned>(opcode), what);
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

struct LightOpcode {
    std::string_view name;
    Lane lane;
};

constexpr std::array<LightOpcode, 13> LightOpcodes{{
    {"SET_SH_REG", Lane::Register},
    {"SET_CONTEXT_REG", Lane::Register},
    {"SET_UCONFIG_REG", Lane::Register},
    {"SET_UCONFIG_REG_INDEX", Lane::Register},
    {"SET_SH_REG_INDIRECT", Lane::RegisterList},
    {"SET_CONTEXT_REG_INDIRECT", Lane::RegisterList},
    {"SET_UCONFIG_REG_INDIRECT", Lane::RegisterList},
    {"INDEX_BUFFER_SIZE", Lane::Parameter},
    {"INDEX_BASE", Lane::Parameter},
    {"NUM_INSTANCES", Lane::Parameter},
    {"DRAW_INDEX_2", Lane::Draw},
    {"DRAW_INDEX_AUTO", Lane::Draw},
    {"DRAW_INDEX_OFFSET_2", Lane::Draw},
}};

constexpr std::uint32_t Type0 = 0x00000000u;
constexpr std::uint32_t Type1 = 0x40000000u;
constexpr std::uint32_t Type2 = 0x80000000u;
constexpr std::uint32_t Type3 = 0xc0000000u;
constexpr std::uint32_t FillerWord = 0xffff1000u;
constexpr std::uint32_t FlipHeader = 0xc004105cu;
constexpr std::uint32_t RenderingWaitHeader = 0xc0021018u;
constexpr std::uint32_t MostPackets = 64;

std::uint32_t opcodeOf(std::string_view name) {
    for (const auto& entry : AgcDriver::Pm4::Opcodes) {
        if (entry.name == name) return entry.value;
    }
    return 0xffffffffu;
}

std::string_view nameOf(std::uint32_t opcode) {
    for (const auto& entry : AgcDriver::Pm4::Opcodes) {
        if (entry.value == opcode) return entry.name;
    }
    return {};
}

Lane laneByName(std::uint32_t opcode) {
    const auto name = nameOf(opcode);
    if (name.empty()) return Lane::Heavy;
    for (const auto& entry : LightOpcodes) {
        if (entry.name == name) return entry.lane;
    }
    return Lane::Heavy;
}

const std::array<Lane, 256>& expectedLanes() {
    static const std::array<Lane, 256> lanes = [] {
        std::array<Lane, 256> table{};
        for (std::uint32_t opcode = 0; opcode < 0x100u; ++opcode) table[opcode] = laneByName(opcode);
        return table;
    }();
    return lanes;
}

std::uint32_t headerOf(std::uint32_t type, std::uint32_t opcode, std::uint32_t words) {
    return type | ((words - 2u) << 16u) | (opcode << 8u);
}

bool lightLane(Lane lane) {
    return lane != Lane::Skip && lane != Lane::Heavy;
}

struct Packet {
    std::size_t start = 0;
    std::size_t words = 0;
    Lane lane = Lane::Heavy;
};

struct Stream {
    std::vector<std::uint32_t> words;
    std::vector<Packet> packets;

    std::size_t Add(std::uint32_t header, std::size_t count, Lane lane) {
        const auto start = words.size();
        words.push_back(header);
        for (std::size_t word = 1; word < count; ++word) words.push_back((word & 1u) != 0 ? 0xc0004600u : Type2);
        packets.push_back({start, count, lane});
        return start;
    }
    std::size_t Typed(std::uint32_t opcode, std::uint32_t payload, std::uint32_t flags = 0u) {
        const auto lane = (flags & 1u) != 0 ? Lane::Heavy : expectedLanes()[opcode & 0xffu];
        return Add(headerOf(Type3, opcode, payload + 1u) | flags, payload + 1u, lane);
    }
    std::size_t Filler(std::uint32_t header) { return Add(header, 1u, Lane::Skip); }
};

const Packet* packetAt(const Stream& stream, std::size_t start) {
    const auto found = std::lower_bound(stream.packets.begin(), stream.packets.end(), start, [](const Packet& packet, std::size_t at) { return packet.start < at; });
    if (found == stream.packets.end() || found->start != start) return nullptr;
    return &*found;
}

LaneRun referenceRun(const Stream& stream, std::size_t size, std::size_t cursor, std::size_t limit, std::uint32_t most) {
    const auto stop = std::min(limit, size);
    LaneRun run;
    run.end = cursor;
    for (const auto& packet : stream.packets) {
        if (packet.start < cursor) continue;
        if (packet.start != run.end || run.packets == most || packet.lane == Lane::Heavy || packet.start + packet.words > stop) break;
        run.end = packet.start + packet.words;
        if (!lightLane(packet.lane)) continue;
        ++run.packets;
        if (packet.lane == Lane::Draw) ++run.draws;
    }
    return run;
}

bool sameRun(const LaneRun& left, const LaneRun& right) {
    return left.end == right.end && left.packets == right.packets && left.draws == right.draws;
}

void testOpcodes() {
    for (const auto& entry : LightOpcodes) {
        const auto opcode = opcodeOf(entry.name);
        ExpectOpcode(opcode < 0x100u && LaneOf(headerOf(Type3, opcode, 2u)) == entry.lane, opcode, "a listed light opcode is in the decoder table and has its lane");
    }
    std::size_t light = 0;
    for (std::uint32_t opcode = 0; opcode < 0x100u; ++opcode) {
        const auto expected = expectedLanes()[opcode];
        if (expected != Lane::Heavy) ++light;
        for (const auto words : {2u, 3u, 18u, 0x4001u}) {
            for (const auto flags : {0x00u, 0x02u, 0xfeu}) {
                const auto header = headerOf(Type3, opcode, words) | flags;
                const bool filler = header == FillerWord;
                ExpectHeader(LaneOf(header) == (filler ? Lane::Skip : expected), header, filler ? "the filler word is skipped" : expected == Lane::Heavy ? "an opcode outside the light list is heavy" : "a light opcode keeps its lane whatever its size and its flags other than predication");
                ExpectHeader(LaneWords(header) == (filler ? 1u : words), header, "a type 3 packet spans its count plus two words");
            }
        }
    }
    Expect(light == LightOpcodes.size(), "exactly the thirteen listed opcodes are light");
    for (const auto name : {"NOP", "SET_BASE", "CLEAR_STATE", "DISPATCH_DIRECT", "DISPATCH_INDIRECT", "SET_PREDICATION", "COND_EXEC", "DRAW_INDIRECT", "DRAW_INDEX_INDIRECT", "INDEX_TYPE", "DRAW_INDIRECT_MULTI", "DRAW_INDEX_INDIRECT_MULTI", "DRAW_INDEX_MULTI_INST", "INDIRECT_BUFFER_CNST", "INDIRECT_BUFFER", "WRITE_DATA", "WAIT_REG_MEM", "WAIT_REG_MEM_64", "COPY_DATA", "EVENT_WRITE", "EVENT_WRITE_EOP", "RELEASE_MEM", "DMA_DATA", "CONTEXT_REG_RMW", "ACQUIRE_MEM", "REWIND", "SET_CONFIG_REG", "SET_QUEUE_REG", "WRITE_CONST_RAM", "DUMP_CONST_RAM", "DISPATCH_DRAW", "INDEX_ATTRIBUTES_INDIRECT"}) {
        const auto opcode = opcodeOf(name);
        ExpectOpcode(opcode < 0x100u && LaneOf(headerOf(Type3, opcode, 4u)) == Lane::Heavy, opcode, name);
    }
    ExpectHeader(LaneOf(FlipHeader) == Lane::Heavy && LaneWords(FlipHeader) == 6u, FlipHeader, "a flip is heavy");
    ExpectHeader(LaneOf(RenderingWaitHeader) == Lane::Heavy && LaneWords(RenderingWaitHeader) == 4u, RenderingWaitHeader, "a rendering wait is heavy");
}

void testPredication() {
    for (std::uint32_t opcode = 0; opcode < 0x100u; ++opcode) {
        for (const auto words : {2u, 5u, 0x4001u}) {
            for (const auto flags : {0x01u, 0x03u, 0xffu}) {
                const auto header = headerOf(Type3, opcode, words) | flags;
                ExpectHeader(LaneOf(header) == Lane::Heavy, header, "a predicated packet is heavy whatever its opcode");
                ExpectHeader(LaneWords(header) == words, header, "predication does not change the size of a packet");
            }
        }
    }
    ExpectHeader(LaneOf(FillerWord | 1u) == Lane::Heavy && LaneWords(FillerWord | 1u) == 0x4001u, FillerWord | 1u, "the filler word with the predication bit is a predicated NOP");
    for (const auto header : {Type2 | 1u, Type2 | 0x3fffffffu, Type2 | 0x00007601u, Type2 | 0x00002701u}) {
        ExpectHeader(LaneOf(header) == Lane::Skip && LaneWords(header) == 1u, header, "a type 2 filler with bit 0 set is still a filler");
    }
}

void testFiller() {
    Random random{0xf111e5u};
    std::size_t wrong = 0;
    for (std::uint32_t round = 0; round < 4096; ++round) {
        const auto header = Type2 | (static_cast<std::uint32_t>(random.Next()) & 0x3fffffffu);
        if (LaneOf(header) != Lane::Skip || LaneWords(header) != 1u) ++wrong;
    }
    Expect(wrong == 0, "every type 2 header is a one word filler");
    for (const auto header : {Type2, Type2 | 0x3fffffffu, FillerWord}) {
        ExpectHeader(LaneOf(header) == Lane::Skip && LaneWords(header) == 1u, header, "a filler is skipped and spans one word");
    }
    ExpectHeader(LaneOf(0xfffe1000u) == Lane::Heavy && LaneWords(0xfffe1000u) == 0x4000u, 0xfffe1000u, "a NOP one count short of the filler word is a heavy packet");
    ExpectHeader(LaneOf(0xffff1100u) == Lane::Heavy && LaneWords(0xffff1100u) == 0x4001u, 0xffff1100u, "the filler count on another heavy opcode is a heavy packet");
    ExpectHeader(LaneOf(0xffff7600u) == Lane::Register && LaneWords(0xffff7600u) == 0x4001u, 0xffff7600u, "the filler count on a register write is a long register packet");

    Stream fillers;
    for (std::uint32_t word = 0; word < 300; ++word) fillers.Filler(word % 3u == 0 ? FillerWord : Type2 | word);
    const auto all = ScanLane(fillers.words, 0, fillers.words.size(), MostPackets);
    Expect(all.end == fillers.words.size() && all.packets == 0u && all.draws == 0u, "fillers are taken without counting, beyond the packet bound");

    Stream padded;
    padded.Filler(Type2);
    padded.Filler(FillerWord);
    const auto draw = padded.Typed(opcodeOf("DRAW_INDEX_AUTO"), 2u);
    padded.Filler(Type2 | 0x1234u);
    const auto nop = padded.Typed(opcodeOf("NOP"), 1u);
    padded.Filler(Type2);
    const auto run = ScanLane(padded.words, 0, padded.words.size(), MostPackets);
    Expect(run.end == nop && run.packets == 1u && run.draws == 1u, "fillers around a draw are taken up to the next heavy packet");
    const auto tail = ScanLane(padded.words, draw + 3u, padded.words.size(), MostPackets);
    Expect(tail.end == nop && tail.packets == 0u && tail.draws == 0u, "a run of fillers alone ends on the heavy packet");
}

void testOtherTypes() {
    for (std::uint32_t opcode = 0; opcode < 0x100u; ++opcode) {
        for (const auto type : {Type0, Type1}) {
            for (const auto words : {2u, 4u, 0x4001u}) {
                const auto header = headerOf(type, opcode, words);
                ExpectHeader(LaneOf(header) == Lane::Heavy, header, "a type 0 or type 1 packet is heavy whatever its opcode byte");
                ExpectHeader(LaneWords(header) == words, header, "a type 0 or type 1 packet spans its count plus two words");
            }
        }
    }
    ExpectHeader(LaneOf(0u) == Lane::Heavy && LaneWords(0u) == 2u, 0u, "a zero header is a heavy two word packet");
}

void testLaneWords() {
    std::size_t wrong = 0;
    for (std::uint32_t count = 0; count <= 0x3fffu; ++count) {
        for (const auto type : {Type0, Type1, Type3}) {
            for (const auto opcode : {0x00u, 0x10u, 0x27u, 0x76u, 0xffu}) {
                for (const auto flags : {0x00u, 0x01u}) {
                    const auto header = type | (count << 16u) | (opcode << 8u) | flags;
                    const auto expected = header == FillerWord ? 1u : count + 2u;
                    if (LaneWords(header) != expected) ++wrong;
                }
            }
        }
        const auto filler = Type2 | (count << 16u) | 0x7601u;
        if (LaneWords(filler) != 1u || LaneOf(filler) != Lane::Skip) ++wrong;
    }
    Expect(wrong == 0, "LaneWords follows the PM4 size rule for every count and type");
}

void testEmptyRuns() {
    const LaneRun fresh;
    Expect(fresh.end == 0 && fresh.packets == 0u && fresh.draws == 0u, "a default run is empty");
    const std::span<const std::uint32_t> none;
    Expect(sameRun(ScanLane(none, 0, 0, MostPackets), LaneRun{}), "no commands give an empty run");
    Expect(sameRun(ScanLane(none, 0, 100, MostPackets), LaneRun{}), "no commands and a large limit give an empty run");
    const auto beyond = ScanLane(none, 7, 100, MostPackets);
    Expect(beyond.end == 7 && beyond.packets == 0u && beyond.draws == 0u, "a cursor past the commands takes nothing");
    Stream stream;
    stream.Typed(opcodeOf("SET_SH_REG"), 2u);
    stream.Typed(opcodeOf("DRAW_INDEX_AUTO"), 2u);
    const auto size = stream.words.size();
    const auto end = ScanLane(stream.words, size, size, MostPackets);
    Expect(end.end == size && end.packets == 0u && end.draws == 0u, "a run from the end of the commands is empty");
    const auto past = ScanLane(stream.words, size + 3u, size + 10u, MostPackets);
    Expect(past.end == size + 3u && past.packets == 0u, "a cursor past the end keeps its place");
}

void testRunCutByHeavy() {
    const auto setShReg = opcodeOf("SET_SH_REG");
    const auto indexBase = opcodeOf("INDEX_BASE");
    const auto numInstances = opcodeOf("NUM_INSTANCES");
    const auto drawIndex2 = opcodeOf("DRAW_INDEX_2");
    const auto drawIndexAuto = opcodeOf("DRAW_INDEX_AUTO");
    const auto setContextRegIndirect = opcodeOf("SET_CONTEXT_REG_INDIRECT");
    const auto eventWrite = opcodeOf("EVENT_WRITE");

    Stream stream;
    stream.Typed(setShReg, 4u);
    stream.Typed(indexBase, 2u);
    stream.Filler(Type2);
    stream.Typed(numInstances, 1u);
    stream.Filler(FillerWord);
    stream.Typed(drawIndexAuto, 2u);
    stream.Typed(setContextRegIndirect, 2u);
    stream.Typed(drawIndex2, 4u);
    const auto heavy = stream.Typed(eventWrite, 1u);
    stream.Typed(setShReg, 2u);
    stream.Typed(drawIndexAuto, 2u);
    const auto size = stream.words.size();
    const auto run = ScanLane(stream.words, 0, size, MostPackets);
    Expect(run.end == heavy && run.packets == 6u && run.draws == 2u, "a run stops on the first heavy packet and counts its light packets and draws, fillers aside");
    const auto stuck = ScanLane(stream.words, heavy, size, MostPackets);
    Expect(stuck.end == heavy && stuck.packets == 0u && stuck.draws == 0u, "a run that starts on a heavy packet is empty");
    const auto rest = ScanLane(stream.words, heavy + 2u, size, MostPackets);
    Expect(rest.end == size && rest.packets == 2u && rest.draws == 1u, "the run after the heavy packet reaches the end of the commands");

    Stream predicated;
    predicated.Typed(setShReg, 3u);
    const auto drawn = predicated.Typed(drawIndexAuto, 2u, 1u);
    predicated.Typed(setShReg, 3u);
    const auto cut = ScanLane(predicated.words, 0, predicated.words.size(), MostPackets);
    Expect(cut.end == drawn && cut.packets == 1u && cut.draws == 0u, "a predicated draw ends the run");

    Stream typed;
    typed.Typed(indexBase, 2u);
    const auto other = typed.Add(headerOf(Type0, setShReg, 3u), 3u, Lane::Heavy);
    typed.Typed(drawIndex2, 4u);
    const auto typeCut = ScanLane(typed.words, 0, typed.words.size(), MostPackets);
    Expect(typeCut.end == other && typeCut.packets == 1u && typeCut.draws == 0u, "a type 0 packet ends the run");

    Stream flipped;
    flipped.Typed(drawIndexAuto, 2u);
    const auto flip = flipped.Add(FlipHeader, 6u, Lane::Heavy);
    flipped.Typed(drawIndexAuto, 2u);
    const auto flipCut = ScanLane(flipped.words, 0, flipped.words.size(), MostPackets);
    Expect(flipCut.end == flip && flipCut.packets == 1u && flipCut.draws == 1u, "a flip ends the run");
}

void testRunCutByLimit() {
    Stream stream;
    stream.Typed(opcodeOf("SET_SH_REG"), 4u);
    const auto second = stream.Typed(opcodeOf("INDEX_BASE"), 2u);
    const auto filler = stream.Filler(Type2);
    const auto draw = stream.Typed(opcodeOf("DRAW_INDEX_OFFSET_2"), 4u);
    const auto last = stream.Typed(opcodeOf("SET_UCONFIG_REG"), 2u);
    const auto size = stream.words.size();

    const auto middle = ScanLane(stream.words, 0, draw + 2u, MostPackets);
    Expect(middle.end == draw && middle.packets == 2u && middle.draws == 0u, "a limit in the middle of a draw ends the run before the draw");
    const auto exact = ScanLane(stream.words, 0, draw + 5u, MostPackets);
    Expect(exact.end == last && exact.packets == 3u && exact.draws == 1u, "a limit at the end of a draw takes the draw");
    const auto afterHeader = ScanLane(stream.words, 0, second + 1u, MostPackets);
    Expect(afterHeader.end == second && afterHeader.packets == 1u, "a limit just after a header leaves that packet");
    const auto beforeFiller = ScanLane(stream.words, 0, filler, MostPackets);
    Expect(beforeFiller.end == filler && beforeFiller.packets == 2u, "a limit on a filler leaves the filler");
    const auto overFiller = ScanLane(stream.words, 0, filler + 1u, MostPackets);
    Expect(overFiller.end == draw && overFiller.packets == 2u, "a filler under the limit is taken");
    const auto zero = ScanLane(stream.words, 0, 0, MostPackets);
    Expect(zero.end == 0 && zero.packets == 0u, "a zero limit takes nothing");
    const auto below = ScanLane(stream.words, draw, second, MostPackets);
    Expect(below.end == draw && below.packets == 0u && below.draws == 0u, "a limit below the cursor takes nothing");
    const auto beyond = ScanLane(stream.words, 0, size + 100u, MostPackets);
    Expect(beyond.end == size && beyond.packets == 4u && beyond.draws == 1u, "a limit beyond the commands stops at their end");
    const auto widest = ScanLane(stream.words, 0, std::numeric_limits<std::size_t>::max(), MostPackets);
    Expect(sameRun(widest, beyond), "the largest limit stops at the end of the commands");
    const std::span<const std::uint32_t> truncated(stream.words.data(), draw + 3u);
    const auto shortened = ScanLane(truncated, 0, size, MostPackets);
    Expect(shortened.end == draw && shortened.packets == 2u && shortened.draws == 0u, "a packet cut by the end of the commands is not taken");

    std::size_t mismatches = 0;
    for (std::size_t cursor = 0; cursor <= size; ++cursor) {
        if (cursor != size && packetAt(stream, cursor) == nullptr) continue;
        for (std::size_t limit = 0; limit <= size + 2u; ++limit) {
            if (!sameRun(ScanLane(stream.words, cursor, limit, MostPackets), referenceRun(stream, size, cursor, limit, MostPackets))) ++mismatches;
        }
    }
    Expect(mismatches == 0, "every limit from every packet start ends the run on the last packet that fits");
}

void testRunCutByMost() {
    Stream stream;
    stream.Filler(Type2);
    stream.Typed(opcodeOf("SET_SH_REG"), 2u);
    stream.Filler(FillerWord);
    stream.Typed(opcodeOf("DRAW_INDEX_2"), 4u);
    stream.Typed(opcodeOf("SET_CONTEXT_REG"), 2u);
    const auto padding = stream.Filler(Type2 | 0x1234u);
    stream.Typed(opcodeOf("DRAW_INDEX_AUTO"), 2u);
    stream.Typed(opcodeOf("NUM_INSTANCES"), 1u);
    const auto size = stream.words.size();

    std::vector<std::size_t> ends;
    std::vector<std::uint32_t> draws;
    std::uint32_t drawn = 0;
    for (const auto& packet : stream.packets) {
        if (!lightLane(packet.lane)) continue;
        if (packet.lane == Lane::Draw) ++drawn;
        ends.push_back(packet.start + packet.words);
        draws.push_back(drawn);
    }
    Expect(ends.size() == 5u && drawn == 2u, "the most stream holds five light packets and two draws");
    for (std::uint32_t most = 1; most <= ends.size(); ++most) {
        const auto run = ScanLane(stream.words, 0, size, most);
        Expect(run.end == ends[most - 1u] && run.packets == most && run.draws == draws[most - 1u], "a run holds at most most packets and ends right after the last of them");
    }
    const auto three = ScanLane(stream.words, 0, size, 3u);
    Expect(three.end == padding, "a full run leaves the filler that follows it");
    const auto none = ScanLane(stream.words, 0, size, 0u);
    Expect(none.end == 0 && none.packets == 0u && none.draws == 0u, "most zero takes nothing, not even a filler");
    for (const auto most : {6u, MostPackets, std::numeric_limits<std::uint32_t>::max()}) {
        const auto run = ScanLane(stream.words, 0, size, most);
        Expect(run.end == size && run.packets == 5u && run.draws == 2u, "a bound above the packet count takes the whole stream");
    }

    Stream stretch;
    for (std::uint32_t packet = 0; packet < 200; ++packet) {
        if (packet % 10u == 9u) stretch.Typed(opcodeOf("DRAW_INDEX_AUTO"), 2u);
        else stretch.Typed(opcodeOf("SET_SH_REG"), 2u);
        if (packet % 7u == 0u) stretch.Filler(Type2);
    }
    const std::array<std::uint32_t, 4> lengths{64u, 64u, 64u, 8u};
    std::size_t cursor = 0;
    std::uint32_t total = 0;
    std::uint32_t totalDraws = 0;
    for (const auto length : lengths) {
        const auto run = ScanLane(stretch.words, cursor, stretch.words.size(), MostPackets);
        Expect(run.packets == length, "a long light stream is cut into runs of 64 packets");
        Expect(run.end > cursor && (packetAt(stretch, run.end) != nullptr || run.end == stretch.words.size()), "each run ends on a packet start");
        total += run.packets;
        totalDraws += run.draws;
        cursor = run.end;
    }
    Expect(cursor == stretch.words.size() && total == 200u && totalDraws == 20u, "the runs of a long light stream cover it with all its draws");
}

std::uint32_t randomLightOpcode(Random& random) {
    return opcodeOf(LightOpcodes[random.Below(static_cast<std::uint32_t>(LightOpcodes.size()))].name);
}

void addRandomPacket(Stream& stream, Random& random) {
    switch (random.Below(12)) {
        case 0:
            stream.Filler(random.OneIn(3) ? FillerWord : Type2 | (static_cast<std::uint32_t>(random.Next()) & 0x3fffffffu));
            return;
        case 1: {
            std::uint32_t opcode = 0;
            do {
                opcode = random.Below(0x100u);
            } while (expectedLanes()[opcode] != Lane::Heavy);
            stream.Typed(opcode, 1u + random.Below(8), random.OneIn(2) ? 0u : 2u);
            return;
        }
        case 2:
            stream.Typed(randomLightOpcode(random), 1u + random.Below(8), random.OneIn(2) ? 1u : 3u);
            return;
        case 3: {
            const auto words = 2u + random.Below(8);
            stream.Add(headerOf(random.OneIn(2) ? Type0 : Type1, randomLightOpcode(random), words), words, Lane::Heavy);
            return;
        }
        default:
            stream.Typed(randomLightOpcode(random), 1u + random.Below(random.OneIn(16) ? 600u : 8u), random.OneIn(4) ? 2u : 0u);
            return;
    }
}

void testRandomStreams() {
    Random random{0xd1a7e5u};
    std::size_t mismatches = 0;
    std::size_t probes = 0;
    std::size_t misplaced = 0;
    std::size_t unexplained = 0;
    std::size_t wrongTotals = 0;
    std::size_t overfull = 0;
    std::size_t cutByHeavy = 0;
    std::size_t cutByLimit = 0;
    std::size_t cutByMost = 0;
    for (std::uint32_t round = 0; round < 3000; ++round) {
        Stream stream;
        const auto count = random.Below(120);
        for (std::uint32_t packet = 0; packet < count; ++packet) addRandomPacket(stream, random);
        auto size = stream.words.size();
        if (size != 0 && random.OneIn(4)) size -= 1u + random.Below(static_cast<std::uint32_t>(std::min<std::size_t>(size, 8u)));
        const std::span<const std::uint32_t> commands(stream.words.data(), size);

        for (std::uint32_t probe = 0; probe < 8; ++probe) {
            std::size_t cursor = size;
            if (!stream.packets.empty() && !random.OneIn(16)) cursor = stream.packets[random.Below(static_cast<std::uint32_t>(stream.packets.size()))].start;
            std::size_t limit = size;
            switch (random.Below(4)) {
                case 0: limit = size + random.Below(4); break;
                case 1: limit = cursor - std::min<std::size_t>(cursor, random.Below(3)); break;
                default: limit = cursor + random.Below(static_cast<std::uint32_t>(std::max(size, cursor) - cursor + 4u)); break;
            }
            const auto most = random.OneIn(8) ? 0u : random.OneIn(2) ? MostPackets : 1u + random.Below(16);
            ++probes;
            if (!sameRun(ScanLane(commands, cursor, limit, most), referenceRun(stream, size, cursor, limit, most))) ++mismatches;
        }

        std::uint32_t expectedPackets = 0;
        std::uint32_t expectedDraws = 0;
        for (const auto& packet : stream.packets) {
            if (!lightLane(packet.lane) || packet.start + packet.words > size) continue;
            ++expectedPackets;
            if (packet.lane == Lane::Draw) ++expectedDraws;
        }
        const auto most = 1u + random.Below(MostPackets);
        std::uint32_t packets = 0;
        std::uint32_t draws = 0;
        std::size_t cursor = 0;
        std::size_t runs = 0;
        while (cursor < size && runs <= stream.words.size()) {
            ++runs;
            const auto run = ScanLane(commands, cursor, size, most);
            const auto* next = packetAt(stream, run.end);
            if (next == nullptr && run.end != stream.words.size()) ++misplaced;
            if (run.packets > most) ++overfull;
            const bool atEnd = run.end >= size;
            const bool heavy = next != nullptr && next->lane == Lane::Heavy;
            const bool cut = next != nullptr && next->start + next->words > size;
            const bool full = run.packets == most;
            if (!atEnd && !heavy && !cut && !full) ++unexplained;
            if (!atEnd && heavy) ++cutByHeavy;
            if (!atEnd && cut) ++cutByLimit;
            if (!atEnd && full && !heavy && !cut) ++cutByMost;
            packets += run.packets;
            draws += run.draws;
            if (run.end != cursor) {
                cursor = run.end;
                continue;
            }
            if (next == nullptr || !heavy) break;
            cursor += next->words;
        }
        if (packets != expectedPackets || draws != expectedDraws) ++wrongTotals;
    }
    Expect(probes >= 20000 && mismatches == 0, "ScanLane matches the reference on random streams, cursors, limits and bounds");
    Expect(misplaced == 0, "a run always ends on a packet start");
    Expect(overfull == 0, "a run never holds more than most packets");
    Expect(unexplained == 0, "a run ends only at the end, on a heavy packet, on a packet that does not fit, or when full");
    Expect(wrongTotals == 0, "the runs between heavy packets count every light packet and every draw once");
    Expect(cutByHeavy > 1000 && cutByLimit > 100 && cutByMost > 100, "random streams cut runs by heavy packets, by the end of the commands and by the bound");
}

void testRunMaskAgreement() {
    std::size_t light = 0;
    for (std::uint32_t opcode = 0; opcode < 0x100u; ++opcode) {
        const auto lane = LaneOf(headerOf(Type3, opcode, 2u));
        RunMask mask;
        mask.Packet(opcode);
        const bool other = (mask.bits & RunMask::Other) != 0;
        if (lane != Lane::Heavy) {
            ++light;
            ExpectOpcode(!other, opcode, "a light opcode never marks other in the run mask");
            ExpectOpcode(mask.Continues(), opcode, "a light opcode keeps the run of the run mask open");
            ExpectOpcode(((mask.bits & RunMask::Parameters) != 0) == (lane == Lane::Parameter), opcode, "the parameter lane is the set of opcodes that mark draw parameters");
        } else {
            ExpectOpcode(other, opcode, "an opcode the run mask leaves unmarked is light");
        }
    }
    Expect(light == LightOpcodes.size(), "the light lane and the run mask agree on thirteen opcodes");
}

}

int main() {
    testOpcodes();
    testPredication();
    testFiller();
    testOtherTypes();
    testLaneWords();
    testEmptyRuns();
    testRunCutByHeavy();
    testRunCutByLimit();
    testRunCutByMost();
    testRandomStreams();
    testRunMaskAgreement();
    if (failures != 0) {
        std::fprintf(stderr, "%d decode lane checks failed\n", failures);
        return 1;
    }
    std::printf("decode lane tests passed\n");
    return 0;
}
