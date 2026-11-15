#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4Opcodes.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

using namespace AgcDriver::Core;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
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

using Model = std::map<std::uint32_t, std::uint32_t>;
using Pairs = std::vector<std::pair<std::uint32_t, std::uint32_t>>;

constexpr std::array<std::uint32_t, 3> UserDataBases{0x00cu, 0x08cu, 0x10cu};
constexpr std::uint32_t UserDataWords = 32;
constexpr std::array<Bank, 3> Banks{Bank::Shader, Bank::Context, Bank::UserConfig};
constexpr std::uint32_t PlanSlots = 1u << 14u;

struct Tracked {
    Model model;
    BankDigest digest;
};

BankDigest digestOf(const Model& model, Bank bank) {
    const Pairs pairs(model.begin(), model.end());
    return DigestOf(pairs, bank);
}

void writeRegister(Tracked& tracked, RunMask& mask, Bank bank, std::uint32_t offset, std::uint32_t value) {
    const auto found = tracked.model.find(offset);
    const bool had = found != tracked.model.end();
    const std::uint32_t before = had ? found->second : 0u;
    tracked.model.insert_or_assign(offset, value);
    mask.Register(bank, offset);
    if (!UserDataRegister(bank, offset)) DigestWrite(tracked.digest, offset, had, before, value);
}

std::uint32_t userDataOffset(Random& random) {
    const auto base = UserDataBases[random.Below(static_cast<std::uint32_t>(UserDataBases.size()))];
    return base + random.Below(UserDataWords);
}

std::uint32_t anyOffset(Random& random) {
    switch (random.Below(5)) {
        case 0: return userDataOffset(random);
        case 1: {
            const auto base = UserDataBases[random.Below(static_cast<std::uint32_t>(UserDataBases.size()))];
            return random.OneIn(2) ? base - 1u : base + UserDataWords;
        }
        case 2: return random.Below(16);
        case 3: return random.Below(0x400);
        default: return random.Below(0x10000);
    }
}

std::uint32_t anyValue(Random& random, const Model& model, std::uint32_t offset) {
    const auto found = model.find(offset);
    if (found != model.end() && random.OneIn(4)) return found->second;
    if (random.OneIn(8)) return 0;
    return static_cast<std::uint32_t>(random.Next());
}

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

bool listed(std::span<const std::string_view> names, std::string_view name) {
    return !name.empty() && std::find(names.begin(), names.end(), name) != names.end();
}

bool distinct(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    return std::adjacent_find(values.begin(), values.end()) == values.end();
}

StateKey randomKey(Random& random) {
    StateKey key;
    key.shader = {random.Next(), random.Next()};
    key.context = {random.Next(), random.Next()};
    key.userConfig = {random.Next(), random.Next()};
    for (auto& program : key.programs) program = random.Next();
    for (auto& programHash : key.programHashes) programHash = random.Next();
    key.indexType = random.Below(4);
    key.topology = random.Below(32);
    key.device = random.Next();
    return key;
}

constexpr std::uint32_t KeyFields = 17;

void change(StateKey& key, std::uint32_t field, std::uint64_t delta) {
    const auto narrow = static_cast<std::uint32_t>(delta) | 1u;
    switch (field) {
        case 0: key.shader.low ^= delta; return;
        case 1: key.shader.high ^= delta; return;
        case 2: key.context.low ^= delta; return;
        case 3: key.context.high ^= delta; return;
        case 4: key.userConfig.low ^= delta; return;
        case 5: key.userConfig.high ^= delta; return;
        case 6: case 7: case 8: case 9: key.programs[field - 6] ^= delta; return;
        case 10: case 11: case 12: case 13: key.programHashes[field - 10] ^= delta; return;
        case 14: key.indexType ^= narrow; return;
        case 15: key.topology ^= narrow; return;
        default: key.device ^= delta; return;
    }
}

void testUserDataRegisters() {
    for (const auto base : UserDataBases) {
        Expect(!UserDataRegister(Bank::Shader, base - 1u), "the register below a user data range is state");
        Expect(UserDataRegister(Bank::Shader, base), "the first register of a user data range");
        Expect(UserDataRegister(Bank::Shader, base + UserDataWords - 1u), "the last register of a user data range");
        Expect(!UserDataRegister(Bank::Shader, base + UserDataWords), "the register above a user data range is state");
        for (const auto bank : {Bank::Context, Bank::UserConfig}) {
            for (std::uint32_t word = 0; word < UserDataWords; ++word) Expect(!UserDataRegister(bank, base + word), "user data registers belong to the shader bank only");
        }
    }
    std::uint32_t count = 0;
    for (std::uint32_t offset = 0; offset < 0x10000u; ++offset) count += UserDataRegister(Bank::Shader, offset) ? 1u : 0u;
    Expect(count == 3u * UserDataWords, "the shader bank has three ranges of user data registers");
    Expect(!UserDataRegister(Bank::Shader, 0xffffffffu), "a wrapped offset is state");
}

void testDigestProperties() {
    Expect(DigestOf(Pairs{}, Bank::Context) == BankDigest{}, "an empty bank has the empty digest");
    BankDigest digest;
    DigestWrite(digest, 0x100u, false, 0u, 0u);
    Expect(digest != BankDigest{}, "a register written with zero differs from an absent register");
    Expect(digest == DigestOf(Pairs{{0x100u, 0u}}, Bank::Context), "one write matches the digest of its bank");
    const auto once = digest;
    DigestWrite(digest, 0x100u, true, 0u, 0u);
    Expect(digest == once, "rewriting a register with its value changes nothing");
    DigestWrite(digest, 0x100u, true, 0u, 5u);
    Expect(digest != once && digest == DigestOf(Pairs{{0x100u, 5u}}, Bank::Context), "a new value replaces the old one");
    DigestWrite(digest, 0x100u, true, 5u, 0u);
    Expect(digest == once, "restoring a value restores the digest");
    Expect(DigestOf(Pairs{{1u, 2u}, {3u, 4u}}, Bank::Context) != DigestOf(Pairs{{1u, 4u}, {3u, 2u}}, Bank::Context), "values are bound to their offsets");

    std::vector<std::uint64_t> lows;
    std::vector<std::uint64_t> highs;
    std::size_t equalHalves = 0;
    for (std::uint32_t offset = 0; offset < 64; ++offset) {
        for (std::uint32_t index = 0; index < 64; ++index) {
            const auto value = index < 32 ? index : 0xffffffffu - (index - 32u);
            const auto one = DigestOf(Pairs{{offset, value}}, Bank::UserConfig);
            lows.push_back(one.low);
            highs.push_back(one.high);
            if (one.low == one.high) ++equalHalves;
        }
    }
    Expect(distinct(lows) && distinct(highs) && equalHalves == 0, "both halves of a register digest separate registers and values");

    Random random{0xd16e57u};
    Pairs pairs;
    for (std::uint32_t i = 0; i < 256; ++i) pairs.emplace_back(i * 3u, static_cast<std::uint32_t>(random.Next()));
    const auto ordered = DigestOf(pairs, Bank::Context);
    for (std::size_t i = pairs.size() - 1; i > 0; --i) std::swap(pairs[i], pairs[random.Below(static_cast<std::uint32_t>(i + 1))]);
    Expect(DigestOf(pairs, Bank::Context) == ordered, "the digest does not depend on the order of the registers");
}

void testIncrementalDigest() {
    Random random{0x5eedu};
    const auto nop = opcodeOf("NOP");
    const auto clearState = opcodeOf("CLEAR_STATE");
    std::size_t checks = 0;
    std::size_t mismatches = 0;
    std::size_t wrongMasks = 0;
    for (std::uint32_t sequence = 0; sequence < 10000; ++sequence) {
        const auto bank = Banks[sequence % Banks.size()];
        Tracked cleared;
        cleared.model = {{0x00cu, 7u}, {0x08eu, 0xfu}, {0x10cu, 0u}, {0x1b6u, 2u}};
        cleared.digest = digestOf(cleared.model, bank);
        Tracked tracked;
        RunMask ignored;
        const auto prefill = random.Below(48);
        for (std::uint32_t write = 0; write < prefill; ++write) {
            const auto offset = anyOffset(random);
            writeRegister(tracked, ignored, bank, offset, anyValue(random, tracked.model, offset));
        }
        std::optional<Tracked> saved;
        RunMask mask;
        bool words = false;
        bool state = false;
        const auto length = 1u + random.Below(64);
        for (std::uint32_t step = 0; step < length; ++step) {
            const auto action = random.Below(32);
            if (action == 0 && !saved.has_value()) {
                saved = tracked;
                mask.Packet(nop);
                state = true;
            } else if (action == 1 && saved.has_value()) {
                tracked = std::move(*saved);
                saved.reset();
                mask.Packet(nop);
                state = true;
            } else if (action == 2) {
                tracked = cleared;
                mask.Packet(clearState);
                state = true;
            } else {
                const auto offset = anyOffset(random);
                const bool userData = UserDataRegister(bank, offset);
                words = words || userData;
                state = state || !userData;
                writeRegister(tracked, mask, bank, offset, anyValue(random, tracked.model, offset));
            }
            if (random.OneIn(8)) {
                ++checks;
                if (tracked.digest != digestOf(tracked.model, bank)) ++mismatches;
            }
        }
        ++checks;
        if (tracked.digest != digestOf(tracked.model, bank)) ++mismatches;
        if (saved.has_value()) {
            ++checks;
            if (saved->digest != digestOf(saved->model, bank)) ++mismatches;
        }
        const bool wordsMarked = (mask.bits & RunMask::Words) != 0;
        const bool stateMarked = (mask.bits & RunMask::State) != 0;
        if (wordsMarked != words || stateMarked != state || (mask.bits & RunMask::Parameters) != 0 || mask.Continues() == state) ++wrongMasks;
    }
    Expect(checks >= 10000 && mismatches == 0, "the incremental digest equals DigestOf after 10000 random write sequences");
    Expect(wrongMasks == 0, "the run mask marks words for user data writes and state for every other register");
}

void testUserDataWithoutEffect() {
    Random random{0x7a11u};
    std::size_t moved = 0;
    std::size_t wrongMasks = 0;
    for (std::uint32_t round = 0; round < 1000; ++round) {
        Tracked tracked;
        RunMask ignored;
        const auto prefill = random.Below(64);
        for (std::uint32_t write = 0; write < prefill; ++write) {
            const auto offset = anyOffset(random);
            writeRegister(tracked, ignored, Bank::Shader, offset, anyValue(random, tracked.model, offset));
        }
        const auto before = tracked.digest;
        Model withoutWords;
        for (const auto& [offset, value] : tracked.model) {
            if (!UserDataRegister(Bank::Shader, offset)) withoutWords.emplace(offset, value);
        }
        RunMask mask;
        const auto writes = 1u + random.Below(64);
        for (std::uint32_t write = 0; write < writes; ++write) writeRegister(tracked, mask, Bank::Shader, userDataOffset(random), static_cast<std::uint32_t>(random.Next()));
        if (tracked.digest != before || digestOf(tracked.model, Bank::Shader) != before || digestOf(withoutWords, Bank::Shader) != before) ++moved;
        if (mask.bits != RunMask::Words || !mask.Continues()) ++wrongMasks;
    }
    Expect(moved == 0, "user data writes leave the shader digest unchanged");
    Expect(wrongMasks == 0, "user data writes only mark the words and keep the run open");
    for (const auto base : UserDataBases) {
        for (std::uint32_t word = 0; word < UserDataWords; ++word) {
            const auto offset = base + word;
            Expect(DigestOf(Pairs{{offset, 1u}}, Bank::Shader) == BankDigest{}, "a user data register is outside the shader digest");
            for (const auto bank : {Bank::Context, Bank::UserConfig}) {
                Expect(DigestOf(Pairs{{offset, 1u}}, bank) != BankDigest{}, "the same offset in another bank is in its digest");
                RunMask mask;
                mask.Register(bank, offset);
                Expect(mask.bits == RunMask::State && !mask.Continues(), "the same offset in another bank is state");
            }
        }
    }
}

void testOpcodes() {
    const std::array<std::string_view, 3> parameters{"INDEX_BASE", "INDEX_BUFFER_SIZE", "NUM_INSTANCES"};
    const std::array<std::string_view, 10> neutral{"DRAW_INDEX_2", "DRAW_INDEX_AUTO", "DRAW_INDEX_OFFSET_2", "SET_SH_REG", "SET_SH_REG_INDIRECT", "SET_CONTEXT_REG", "SET_CONTEXT_REG_INDIRECT", "SET_UCONFIG_REG", "SET_UCONFIG_REG_INDEX", "SET_UCONFIG_REG_INDIRECT"};
    const std::array<std::string_view, 4> bulkState{"NOP", "CLEAR_STATE", "INDEX_TYPE", "INDEX_ATTRIBUTES_INDIRECT"};
    std::size_t named = 0;
    for (std::uint32_t opcode = 0; opcode < 0x100u; ++opcode) {
        const auto name = nameOf(opcode);
        RunMask mask;
        mask.Packet(opcode);
        if (listed(parameters, name)) {
            ++named;
            ExpectOpcode(mask.bits == RunMask::Parameters && mask.Continues(), opcode, "draw parameters keep the run open");
        } else if (listed(neutral, name)) {
            ++named;
            ExpectOpcode(mask.bits == 0 && mask.Continues(), opcode, "a direct draw or a register write packet marks nothing by itself");
        } else {
            ExpectOpcode((mask.bits & RunMask::Other) != 0 && !mask.Continues(), opcode, "an opcode outside the whitelist marks other");
            if (listed(bulkState, name)) {
                ++named;
                ExpectOpcode((mask.bits & RunMask::State) != 0, opcode, "a packet that changes registers in bulk also marks state");
            }
        }
    }
    Expect(named == parameters.size() + neutral.size() + bulkState.size(), "every listed opcode is in the decoder table");
    for (const auto opcode : {0x100u, 0x163u, 0x10026u, 0xffffffffu}) {
        RunMask mask;
        mask.Packet(opcode);
        ExpectOpcode(mask.bits == RunMask::Other && !mask.Continues(), opcode, "a value beyond the opcode byte marks other");
    }
    for (const auto name : {"DRAW_INDIRECT", "DRAW_INDEX_INDIRECT", "DRAW_INDIRECT_MULTI", "DRAW_INDEX_INDIRECT_MULTI", "DISPATCH_DIRECT", "EVENT_WRITE", "RELEASE_MEM", "WRITE_DATA", "COPY_DATA", "CONTEXT_REG_RMW", "SET_BASE", "ACQUIRE_MEM", "WAIT_REG_MEM"}) {
        const auto opcode = opcodeOf(name);
        RunMask mask;
        mask.Packet(opcode);
        ExpectOpcode(opcode < 0x100u && (mask.bits & RunMask::Other) != 0, opcode, name);
    }
}

void testRunMask() {
    RunMask mask;
    Expect(mask.bits == 0 && mask.Continues(), "a fresh mask keeps the run open");
    mask.Register(Bank::Shader, 0x08cu);
    mask.Register(Bank::Shader, 0x10cu + 31u);
    mask.Packet(opcodeOf("INDEX_BASE"));
    mask.Packet(opcodeOf("INDEX_BUFFER_SIZE"));
    mask.Packet(opcodeOf("NUM_INSTANCES"));
    mask.Packet(opcodeOf("SET_SH_REG"));
    mask.Packet(opcodeOf("DRAW_INDEX_2"));
    Expect(mask.bits == (RunMask::Words | RunMask::Parameters) && mask.Continues(), "user data words and draw parameters continue the run");
    mask.Register(Bank::Shader, 0x088u);
    Expect(mask.bits == (RunMask::Words | RunMask::Parameters | RunMask::State) && !mask.Continues(), "a program address register ends the run");
    mask.Drawn();
    Expect(mask.bits == 0 && mask.Continues(), "a draw clears the mask");
    mask.Register(Bank::Context, 0x08eu);
    Expect(mask.bits == RunMask::State && !mask.Continues(), "a context register ends the run");
    mask.Drawn();
    mask.Register(Bank::UserConfig, 0x243u);
    Expect(mask.bits == RunMask::State && !mask.Continues(), "a user config register ends the run");
    mask.Drawn();
    mask.Packet(opcodeOf("EVENT_WRITE"));
    Expect(mask.bits == RunMask::Other && !mask.Continues(), "an event ends the run");
    mask.Drawn();
    mask.Packet(opcodeOf("CLEAR_STATE"));
    Expect(mask.bits == (RunMask::Other | RunMask::State) && !mask.Continues(), "clearing the context state ends the run and marks state");
    mask.Drawn();
    Expect(mask.bits == 0 && mask.Continues(), "the next draw starts again from an empty mask");
}

void testStateKeyHash() {
    const StateKey empty{};
    Expect(empty.Hash() == StateKey{}.Hash() && empty.Hash() == empty.Hash(), "the default key hashes the same every time");
    Random random{0x4a5bu};
    std::size_t unstable = 0;
    std::size_t insensitive = 0;
    std::vector<std::uint64_t> hashes;
    for (std::uint32_t round = 0; round < 1000; ++round) {
        const auto seed = random.Next();
        Random first{seed};
        Random second{seed};
        const auto key = randomKey(first);
        const auto same = randomKey(second);
        const std::vector<StateKey> stored(3, key);
        const auto hash = key.Hash();
        if (key != same || same.Hash() != hash || key.Hash() != hash || stored[2].Hash() != hash) ++unstable;
        hashes.push_back(hash);
        for (std::uint32_t field = 0; field < KeyFields; ++field) {
            auto changed = key;
            change(changed, field, random.Next() | 1u);
            if (changed == key || changed.Hash() == hash) ++insensitive;
        }
        auto banksSwapped = key;
        std::swap(banksSwapped.shader, banksSwapped.context);
        auto programsSwapped = key;
        std::swap(programsSwapped.programs[0], programsSwapped.programs[1]);
        auto crossed = key;
        std::swap(crossed.programs[3], crossed.programHashes[3]);
        if (banksSwapped.Hash() == hash || programsSwapped.Hash() == hash || crossed.Hash() == hash) ++insensitive;
    }
    Expect(unstable == 0, "equal keys hash the same, in any copy and on every call");
    Expect(insensitive == 0, "every field of the key, and its position, changes the hash");
    Expect(distinct(hashes), "random keys hash apart");

    const auto base = randomKey(random);
    std::vector<bool> lowSlots(PlanSlots);
    std::vector<bool> highSlots(PlanSlots);
    std::uint32_t lowUsed = 0;
    std::uint32_t highUsed = 0;
    for (std::uint32_t indexType = 0; indexType < 4; ++indexType) {
        for (std::uint32_t topology = 0; topology < 1024; ++topology) {
            auto key = base;
            key.indexType = indexType;
            key.topology = topology;
            const auto hash = key.Hash();
            const auto low = static_cast<std::size_t>(hash & (PlanSlots - 1u));
            const auto high = static_cast<std::size_t>(hash >> 50u);
            if (!lowSlots[low]) ++lowUsed;
            if (!highSlots[high]) ++highUsed;
            lowSlots[low] = true;
            highSlots[high] = true;
        }
    }
    Expect(lowUsed > 3200 && highUsed > 3200, "4096 keys that differ in small fields spread over a table of 16384 slots");
}

}

int main() {
    testUserDataRegisters();
    testDigestProperties();
    testIncrementalDigest();
    testUserDataWithoutEffect();
    testOpcodes();
    testRunMask();
    testStateKeyHash();
    if (failures != 0) {
        std::fprintf(stderr, "%d run key checks failed\n", failures);
        return 1;
    }
    std::printf("run key tests passed\n");
    return 0;
}
