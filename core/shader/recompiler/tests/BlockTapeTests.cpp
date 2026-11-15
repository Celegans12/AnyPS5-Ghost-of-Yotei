#include "IntermediateRepresentation/GuestRegister.hpp"
#include "IntermediateRepresentation/IrOpcode.hpp"
#include "IntermediateRepresentation/IrProgram.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <new>
#include <span>
#include <vector>

using namespace ShaderRecompiler;

namespace {

std::size_t allocations = 0;
bool countAllocations = false;

}

void* operator new(std::size_t size) {
    if (countAllocations) ++allocations;
    if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
    throw std::bad_alloc();
}
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete[](void* memory) noexcept { ::operator delete(memory); }
void operator delete[](void* memory, std::size_t) noexcept { ::operator delete(memory); }

namespace {

constexpr std::uint64_t MemoryBase = 0x0000004000010000ull;
constexpr std::uint32_t InnerWord = 256;
constexpr std::uint32_t BufferWord = 512;
constexpr std::uint32_t MemoryWords = 1024;
constexpr std::uint64_t ShaderBase = 0x100;
constexpr std::uint32_t WideSlots = 520;

bool failed = false;

void expect(bool condition, const char* name, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "block tape: %s: %s\n", name, what);
    failed = true;
}

std::uint64_t addressOf(std::uint32_t word) { return MemoryBase + std::uint64_t {word} * 4u; }
std::uint32_t low(std::uint64_t address) { return static_cast<std::uint32_t>(address); }
std::uint32_t high(std::uint64_t address) { return static_cast<std::uint32_t>(address >> 32u); }

struct Memory {
    std::vector<std::uint32_t> words;
    std::uint64_t refused = 0;
    std::size_t calls = 0;
};

Memory makeMemory() {
    Memory memory;
    memory.words.resize(MemoryWords);
    for (std::uint32_t index = 0; index < MemoryWords; ++index) memory.words[index] = 0xa5000000u | (index * 0x10u + 3u);
    const auto inner = addressOf(InnerWord);
    const auto buffer = addressOf(BufferWord);
    memory.words[0] = low(inner);
    memory.words[1] = high(inner) | 0xabcd0000u;
    memory.words[4] = low(buffer);
    memory.words[5] = high(buffer);
    memory.words[6] = 64u;
    memory.words[7] = 0u;
    return memory;
}

bool fetch(const Memory& memory, std::uint64_t address, std::uint32_t& word) {
    if (address == memory.refused || address < MemoryBase || (address & 3u) != 0u) return false;
    const auto index = (address - MemoryBase) / 4u;
    if (index >= memory.words.size()) return false;
    word = memory.words[static_cast<std::size_t>(index)];
    return true;
}

bool readWords(void* context, std::uint64_t address, std::uint32_t* out, std::uint32_t count) {
    auto& memory = *static_cast<Memory*>(context);
    ++memory.calls;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (!fetch(memory, address + std::uint64_t {index} * 4u, out[index])) return false;
    }
    return true;
}

bool readWord(void* context, std::uint64_t address, std::uint32_t* value) {
    return fetch(*static_cast<const Memory*>(context), address, *value);
}

class TestPlan {
public:
    TestPlan() { plan.srtPlanComplete = true; }

    IrValue& Constant(std::uint32_t number) {
        auto& value = program.CreateValue(IrOpcode::Void, IrType::U32);
        value.SetImmediateU32(number);
        return value;
    }

    IrValue& Operation(IrOpcode opcode, IrType type, std::initializer_list<IrValue*> arguments) {
        auto& value = program.CreateValue(opcode, type);
        for (auto* argument : arguments) value.AddArgument(argument);
        return value;
    }

    IrValue& UserData(std::uint32_t index) {
        auto& reg = program.CreateValue(IrOpcode::Void, IrType::ScalarReg);
        reg.SetRegister(GuestRegister{RegisterBank::UserData, index});
        return Operation(IrOpcode::GetUserData, IrType::U32, {&reg});
    }

    IrValue& Load(IrValue& base, IrValue& top, IrValue& offset, std::uint32_t immediate) {
        auto& handle = Operation(IrOpcode::GetAddressResource, IrType::AddressResource, {&base, &top});
        auto& load = Operation(IrOpcode::LoadAddressU32, IrType::U32, {&handle, &offset});
        load.SetFlags(MemoryFlags{memory(ResourceKind::ScalarAddress, immediate), 0u});
        return load;
    }

    IrValue& LoadBuffer(const std::array<IrValue*, 4>& descriptor, IrValue& offset, std::uint32_t immediate) {
        auto& handle = Operation(IrOpcode::GetBufferResource, IrType::BufferResource, {descriptor[0], descriptor[1], descriptor[2], descriptor[3]});
        auto& load = Operation(IrOpcode::ReadConstBuffer, IrType::U32, {&handle, &offset});
        load.SetFlags(MemoryFlags{memory(ResourceKind::ScalarBuffer, immediate), 0u});
        return load;
    }

    IrValue& Slot(std::uint32_t slot) {
        auto& srt = program.CreateValue(IrOpcode::Void, IrType::SrtResource);
        return Operation(IrOpcode::ReadConst, IrType::U32, {&srt, &Constant(slot)});
    }

    std::uint32_t Flat(IrValue& value, bool clean = false) {
        const auto slot = static_cast<std::uint32_t>(plan.srtReads.size());
        plan.srtReads.push_back({&value, slot});
        plan.cleanFlatSlots.push_back(clean ? 1u : 0u);
        return slot;
    }

    IrProgram program;
    IrResourcePlan plan;

private:
    std::uint32_t memory(ResourceKind kind, std::uint32_t offset) {
        MemoryInfo info;
        info.kind = kind;
        info.offset = offset;
        plan.memoryInfo.push_back(info);
        return static_cast<std::uint32_t>(plan.memoryInfo.size() - 1u);
    }
};

struct Walk {
    bool ok = false;
    std::vector<std::uint32_t> flat;
};

Walk runTape(const Detail::SrtTape& tape, const IrResourcePlan& plan, std::span<const std::uint32_t> userData, Memory& memory) {
    SrtRuntime runtime{userData, ShaderBase};
    runtime.readMemory = readWord;
    runtime.userContext = &memory;
    runtime.readSpecializationMemory = readWord;
    std::vector<DescriptorValue> results;
    std::vector<std::uint8_t> active;
    Walk walk;
    walk.ok = tape.Run(plan, runtime, results, walk.flat, active);
    return walk;
}

BlockWalk walkInto(const char* name, const Detail::SrtTape& tape, const IrResourcePlan& plan, std::span<const std::uint32_t> userData, Memory& memory, std::span<std::uint32_t> out, BlockWordReader read = readWords) {
    allocations = 0;
    countAllocations = true;
    const auto walk = tape.RunInto(plan, userData, ShaderBase, read, &memory, out);
    countAllocations = false;
    expect(allocations == 0u, name, "RunInto allocated");
    return walk;
}

void buildFlat(TestPlan& test) {
    auto& table = test.UserData(0);
    auto& tableHigh = test.UserData(1);
    auto& index = test.UserData(2);
    auto& zero = test.Constant(0);
    test.Flat(test.Load(table, tableHigh, zero, 0u));
    test.Flat(test.Load(table, tableHigh, zero, 4u), true);
    test.Flat(test.Load(table, tableHigh, test.Operation(IrOpcode::ShiftLeftLogical32, IrType::U32, {&index, &test.Constant(2)}), 8u));
    auto& innerLow = test.Slot(0);
    auto& innerHigh = test.Operation(IrOpcode::BitFieldUExtract, IrType::U32, {&test.Slot(1), &test.Constant(0), &test.Constant(16)});
    test.Flat(test.Load(innerLow, innerHigh, zero, 0u));
    test.Flat(test.Load(innerLow, innerHigh, zero, 12u));
    std::array<IrValue*, 4> descriptor{};
    for (std::uint32_t word = 0; word < 4u; ++word) {
        auto& value = test.Load(table, tableHigh, zero, 16u + word * 4u);
        test.Flat(value);
        descriptor[word] = &value;
    }
    test.Flat(test.LoadBuffer(descriptor, test.Operation(IrOpcode::ShiftLeftLogical32, IrType::U32, {&index, &test.Constant(2)}), 8u));
    test.Flat(test.LoadBuffer(descriptor, zero, 64u));
    auto& first = test.Operation(IrOpcode::IEqual32, IrType::U1, {&index, &test.Constant(1)});
    test.Flat(test.Load(table, tableHigh, test.Operation(IrOpcode::SelectU32, IrType::U32, {&first, &test.Constant(4), &test.Constant(8)}), 32u));
    auto& shader = test.Operation(IrOpcode::GetShaderBase, IrType::U64, {});
    test.Flat(test.Load(test.Operation(IrOpcode::IAdd32, IrType::U32, {&table, &shader}), tableHigh, zero, 0u));
}

std::vector<std::uint32_t> flatWords(const Memory& memory, std::uint32_t index) {
    const auto& words = memory.words;
    return {words[0], words[1], words[2 + index], words[InnerWord], words[InnerWord + 3u], words[4], words[5], words[6], words[7], words[BufferWord + 2u + index], 0u, words[index == 1u ? 9u : 10u], words[ShaderBase / 4u]};
}

void checkFlat() {
    TestPlan test;
    buildFlat(test);
    const auto* tape = Detail::SrtTape::For(test.plan);
    expect(tape != nullptr, "flat", "a complete plan has no tape");
    if (tape == nullptr) return;
    auto memory = makeMemory();
    const auto table = addressOf(0);
    const auto slots = test.plan.srtReads.size();
    for (const std::uint32_t index : {1u, 2u}) {
        const std::array<std::uint32_t, 3> userData{low(table), high(table), index};
        allocations = 0;
        countAllocations = true;
        const auto run = runTape(*tape, test.plan, userData, memory);
        countAllocations = false;
        expect(allocations != 0u, "flat", "the allocation counter did not see the vectors of Run");
        std::vector<std::uint32_t> out(slots, 0xdeadbeefu);
        memory.calls = 0;
        const auto walk = walkInto("flat", *tape, test.plan, userData, memory, out);
        expect(run.ok, "flat", "Run refused the plan");
        expect(walk == BlockWalk::Done, "flat", "RunInto did not finish the plan");
        expect(memory.calls != 0u, "flat", "RunInto did not read through the given reader");
        expect(out == run.flat, "flat", "RunInto wrote other words than Run");
        expect(out == flatWords(memory, index), "flat", "RunInto wrote other words than the memory holds");
    }
    const std::array<std::uint32_t, 3> userData{low(table), high(table), 1u};
    std::vector<std::uint32_t> out(slots, 0u);
    memory.refused = addressOf(InnerWord + 3u);
    expect(!runTape(*tape, test.plan, userData, memory).ok, "unreadable", "Run accepted a refused read");
    expect(walkInto("unreadable", *tape, test.plan, userData, memory, out) == BlockWalk::Unreadable, "unreadable", "a refused read did not end the walk as Unreadable");
    memory.refused = 0;
    const std::array<std::uint32_t, 3> nullTable{0u, 0u, 1u};
    expect(!runTape(*tape, test.plan, nullTable, memory).ok, "poisoned", "Run accepted a read under a null root");
    expect(walkInto("poisoned", *tape, test.plan, nullTable, memory, out) == BlockWalk::Poisoned, "poisoned", "a read under a null root did not end the walk as Poisoned");
    std::vector<std::uint32_t> longer(slots + 1u, 0u);
    expect(walkInto("size", *tape, test.plan, userData, memory, longer) == BlockWalk::Unsupported, "size", "an output of the wrong size was accepted");
    expect(walkInto("reader", *tape, test.plan, userData, memory, out, nullptr) == BlockWalk::Unsupported, "reader", "a walk without a reader was accepted");
    test.plan.srtPlanComplete = false;
    expect(walkInto("incomplete", *tape, test.plan, userData, memory, out) == BlockWalk::Unsupported, "incomplete", "an incomplete plan was accepted");
    test.plan.srtPlanComplete = true;
}

void checkGuarded() {
    TestPlan test;
    auto& table = test.UserData(0);
    auto& tableHigh = test.UserData(1);
    auto& zero = test.Constant(0);
    test.Flat(test.Load(table, tableHigh, zero, 13u * 4u));
    const auto taken = test.Flat(test.Load(table, tableHigh, zero, 9u * 4u));
    const auto other = test.Flat(test.Load(table, tableHigh, zero, 10u * 4u));
    const auto unreachable = test.Flat(test.Load(table, tableHigh, zero, 11u * 4u));
    auto& selector = test.Load(table, tableHigh, zero, 12u * 4u);
    auto& condition = test.Operation(IrOpcode::IEqual32, IrType::U1, {&selector, &test.Constant(7)});
    test.plan.controlFlow.resize(4);
    test.plan.controlFlow[0].condition = &condition;
    test.plan.controlFlow[0].successors = {1u, 2u};
    test.plan.controlFlow[1].flatSlots = {taken};
    test.plan.controlFlow[2].flatSlots = {other};
    test.plan.controlFlow[3].flatSlots = {unreachable};
    const auto* tape = Detail::SrtTape::For(test.plan);
    expect(tape != nullptr, "guarded", "a complete plan has no tape");
    if (tape == nullptr) return;
    auto memory = makeMemory();
    const auto& words = memory.words;
    const auto table0 = addressOf(0);
    const std::array<std::uint32_t, 2> userData{low(table0), high(table0)};
    struct Case {
        const char* name;
        std::uint32_t selector;
        std::uint32_t refused;
        BlockWalk walk;
        std::array<std::uint32_t, 4> words;
    };
    const std::array<Case, 5> cases{{
        {"taken branch", 7u, 0u, BlockWalk::Done, {words[13], words[9], 0u, 0u}},
        {"other branch", 3u, 0u, BlockWalk::Done, {words[13], 0u, words[10], 0u}},
        {"unreadable condition", 7u, 12u, BlockWalk::Done, {words[13], words[9], words[10], 0u}},
        {"refused word behind the branch not taken", 3u, 9u, BlockWalk::Done, {words[13], 0u, words[10], 0u}},
        {"refused word behind the branch taken", 7u, 9u, BlockWalk::Unreadable, {}},
    }};
    for (const auto& entry : cases) {
        memory.words[12] = entry.selector;
        memory.refused = entry.refused != 0u ? addressOf(entry.refused) : 0u;
        const auto run = runTape(*tape, test.plan, userData, memory);
        std::vector<std::uint32_t> out(test.plan.srtReads.size(), 0xdeadbeefu);
        const auto walk = walkInto(entry.name, *tape, test.plan, userData, memory, out);
        expect(walk == entry.walk, entry.name, "RunInto ended with another result");
        expect(run.ok == (entry.walk == BlockWalk::Done), entry.name, "Run and RunInto disagree on the outcome");
        if (walk != BlockWalk::Done || !run.ok) continue;
        expect(out == run.flat, entry.name, "RunInto wrote other words than Run");
        expect(std::equal(out.begin(), out.end(), entry.words.begin(), entry.words.end()), entry.name, "RunInto wrote other words than the active branches hold");
    }
}

void checkWide() {
    TestPlan test;
    auto& table = test.UserData(0);
    auto& tableHigh = test.UserData(1);
    auto& zero = test.Constant(0);
    for (std::uint32_t slot = 0; slot < WideSlots; ++slot) test.Flat(test.Load(table, tableHigh, zero, slot * 4u));
    const auto* tape = Detail::SrtTape::For(test.plan);
    expect(tape != nullptr, "wide", "a complete plan has no tape");
    if (tape == nullptr) return;
    expect(tape->Nodes() > 512u, "wide", "the tape is not wider than the inline node arrays");
    auto memory = makeMemory();
    const auto table0 = addressOf(0);
    const std::array<std::uint32_t, 2> userData{low(table0), high(table0)};
    const auto run = runTape(*tape, test.plan, userData, memory);
    expect(run.ok && std::equal(run.flat.begin(), run.flat.end(), memory.words.begin(), memory.words.begin() + WideSlots), "wide", "Run did not walk the wide plan");
    std::vector<std::uint32_t> out(test.plan.srtReads.size(), 0u);
    expect(walkInto("wide", *tape, test.plan, userData, memory, out) == BlockWalk::Unsupported, "wide", "a tape wider than the inline node arrays was walked");
}

void checkFailing() {
    TestPlan test;
    auto& table = test.UserData(0);
    auto& tableHigh = test.UserData(1);
    test.Flat(test.Load(table, tableHigh, test.Operation(IrOpcode::LaneId, IrType::U32, {}), 0u));
    const auto* tape = Detail::SrtTape::For(test.plan);
    expect(tape != nullptr, "failing", "a plan with an unsupported operation has no tape");
    if (tape == nullptr) return;
    auto memory = makeMemory();
    const auto table0 = addressOf(0);
    const std::array<std::uint32_t, 2> userData{low(table0), high(table0)};
    expect(!runTape(*tape, test.plan, userData, memory).ok, "failing", "Run accepted an unsupported operation");
    std::vector<std::uint32_t> out(test.plan.srtReads.size(), 0u);
    expect(walkInto("failing", *tape, test.plan, userData, memory, out) == BlockWalk::Unsupported, "failing", "an unsupported operation did not end the walk as Unsupported");
}

}

int main() {
    try {
        checkFlat();
        checkGuarded();
        checkWide();
        checkFailing();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "block tape: %s\n", error.what());
        return EXIT_FAILURE;
    }
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
