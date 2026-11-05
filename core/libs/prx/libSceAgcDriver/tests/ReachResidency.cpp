#include "prx/libSceAgcDriver/Graphics/include/ReachResidency.hpp"
#include "BdaAbi.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace AgcDriver::Graphics;
namespace Abi = ShaderRecompiler::BdaAbi;

namespace {

int failures = 0;

void Expect(bool condition, const std::string& what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++failures;
}

constexpr std::uint64_t Base = 0x200000000ull;
constexpr std::uint64_t Unit = Reach::UnitBytes;

std::vector<Reach::UnitSource> Units(std::initializer_list<std::uint64_t> indices) {
    std::vector<Reach::UnitSource> units;
    for (const auto index : indices) units.push_back({Base + index * Unit, VK_NULL_HANDLE, index * Unit});
    return units;
}

Reach::LaneView View(int lane, std::uint64_t batch, bool otherPending = false) {
    return {lane, batch, [otherPending](std::uint64_t, std::uint64_t) { return otherPending; }};
}

std::size_t MapFills(const Reach::LaneView& view, std::initializer_list<std::uint64_t> indices, Reach::Mapping& mapping) {
    std::vector<Reach::Fill> fills;
    const auto units = Units(indices);
    Expect(Reach::Map(Context{}, view, units, mapping, fills), "Map maps the units");
    Reach::Filled(view, fills);
    return fills.size();
}

void SameBatchReuse() {
    Reach::ResetForTests();
    Reach::UsePoolForTests(0, 8);
    Reach::Mapping mapping;
    Expect(MapFills(View(0, 10), {0, 1, 2}, mapping) == 3, "the first use fills every unit");
    Expect(MapFills(View(0, 10), {0, 1, 2}, mapping) == 0, "a later reader of the same batch reuses every copy");
    Expect(mapping.slots.size() == 3 && mapping.slots[0] != mapping.slots[1], "each unit has its own slot");
}

void WriteNoteInvalidates() {
    Reach::ResetForTests();
    Reach::UsePoolForTests(0, 8);
    Reach::Mapping mapping;
    MapFills(View(0, 10), {0, 1, 2}, mapping);
    Reach::NoteWrite(Base + Unit + 100, Base + Unit + 104);
    std::vector<Reach::Fill> fills;
    Expect(Reach::Refresh(View(0, 10), mapping, fills), "the mapping still holds its slots");
    Expect(fills.size() == 1 && fills[0].unit == Base + Unit, "a write note over one unit refills that unit only");
    Reach::Filled(View(0, 10), fills);
    Reach::NoteWrite(Base, Base + 3 * Unit);
    fills.clear();
    Reach::Refresh(View(0, 10), mapping, fills);
    Expect(fills.size() == 3, "a write note over all units refills all");
}

void WritersInvalidate() {
    Reach::ResetForTests();
    Reach::UsePoolForTests(0, 8);
    Reach::UsePoolForTests(1, 8);
    Reach::Mapping mapping;
    MapFills(View(0, 10), {0, 1}, mapping);
    const auto other = Reach::WriterRecorded(1);
    std::vector<Reach::Fill> fills;
    Reach::Refresh(View(0, 10), mapping, fills);
    Expect(fills.empty(), "a writer of the other lane leaves this lane's batch copies current");
    const auto own = Reach::WriterRecorded(0);
    Reach::Refresh(View(0, 10), mapping, fills);
    Expect(fills.size() == 2, "a writer recorded on the lane after the fill invalidates its copies");
    Reach::Filled(View(0, 10), fills);
    fills.clear();
    Reach::Refresh(View(0, 10), mapping, fills);
    Expect(fills.empty(), "copies filled after the writer are current again in the batch");
    Reach::WriterCompleted(other);
    Reach::WriterCompleted(own);
}

void BatchBoundary() {
    Reach::ResetForTests();
    Reach::UsePoolForTests(0, 8);
    Reach::Mapping mapping;
    MapFills(View(0, 10), {0, 1}, mapping);
    std::vector<Reach::Fill> fills;
    Reach::Refresh(View(0, 11), mapping, fills);
    Expect(fills.size() == 2, "without write tracking a later batch refills every copy");
    fills.clear();
    Reach::ResetForTests();
    Reach::UsePoolForTests(0, 8);
    MapFills(View(0, 10, true), {0}, mapping);
    Reach::Refresh(View(0, 12), mapping, fills);
    Expect(fills.size() == 1, "a fill made while the other lane had a pending writer is batch-only");
}

void RefreshRefusals() {
    Reach::ResetForTests();
    Reach::UsePoolForTests(0, 2);
    Reach::Mapping first;
    MapFills(View(0, 10), {0, 1}, first);
    std::vector<Reach::Fill> fills;
    Expect(!Reach::Refresh(View(1, 10), first, fills), "a mapping of another lane cannot be refreshed");
    Reach::Mapping second;
    MapFills(View(0, 11), {2, 3}, second);
    Expect(second.units.size() == 2, "a full pool evicts the least recently used slots");
    fills.clear();
    Expect(!Reach::Refresh(View(0, 11), first, fills), "a mapping whose slots were taken by other units cannot be refreshed");
    Reach::Mapping third;
    std::vector<Reach::Fill> thirdFills;
    const auto many = Units({4, 5, 6});
    Reach::Map(Context{}, View(0, 12), many, third, thirdFills);
    Expect(third.units.size() == 2, "a build maps at most the pool's slots, never evicting its own");
}

void Learning() {
    Reach::ResetForTests();
    const std::uint64_t program = 0x1234;
    for (int build = 0; build < 4; ++build) Expect(Reach::LearnBuild(program), "the first four builds learn");
    Expect(!Reach::LearnBuild(program), "the fifth does not");
    std::vector<std::uint32_t> words(Abi::ReachFaultBufferBytes / 4, 0);
    const auto note = [&](std::uint64_t address) {
        const auto key = Abi::WrittenChunkKey(address);
        const auto slot = words[Abi::ReachCountWord];
        words[Abi::ReachSlotsWord + 2 * slot] = key;
        words[Abi::ReachSlotsWord + 2 * slot + 1] |= 1u << ((address >> Abi::WrittenPageShift) & 31u);
        words[Abi::ReachListWord + slot] = slot;
        ++words[Abi::ReachCountWord];
    };
    note(Base + 5 * Unit + 0x3000);
    note(Base + 9 * Unit);
    Reach::NoteLearned(program, words.data());
    const auto predicted = Reach::Predicted(program);
    Expect(predicted.size() == 2 && predicted[0] == Base + 5 * Unit && predicted[1] == Base + 9 * Unit, "the learned pages give their units");
    Expect(words[Abi::ReachCountWord] == 0 && words[Abi::ReachSlotsWord] == 0, "the notes are cleared for the next use");
    Expect(Reach::Predicted(0x9999).empty(), "an unknown program has no prediction");
}

}

int main() {
#ifdef _WIN32
    _putenv("APS5_REACH_RESIDENCY=1");
#else
    setenv("APS5_REACH_RESIDENCY", "1", 1);
#endif
    SameBatchReuse();
    WriteNoteInvalidates();
    WritersInvalidate();
    BatchBoundary();
    RefreshRefusals();
    Learning();
    if (failures != 0) return 1;
    std::puts("agc_reach_residency: all passed");
    return 0;
}
