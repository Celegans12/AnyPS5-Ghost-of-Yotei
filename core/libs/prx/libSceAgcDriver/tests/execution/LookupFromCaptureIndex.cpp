#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using AgcDriver::DriverDetail::DispatchVariant;
using AgcDriver::DriverDetail::ShapeStageFound;
using AgcDriver::DriverDetail::ShapeStageIndex;
using AgcDriver::DriverDetail::ShapeStagePlan;
using AgcDriver::DriverDetail::ShapeStageReader;

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Memory {
    std::map<std::uint64_t, std::uint32_t> words;

    void Store(std::uint64_t address, std::span<const std::uint32_t> values) {
        for (std::size_t word = 0; word < values.size(); ++word) words[address + word * sizeof(std::uint32_t)] = values[word];
    }

    [[nodiscard]] ShapeStageReader Reader() const {
        return [this](std::uint64_t address, std::span<std::uint32_t> out) {
            for (std::size_t word = 0; word < out.size(); ++word) {
                const auto found = words.find(address + word * sizeof(std::uint32_t));
                if (found == words.end()) return false;
                out[word] = found->second;
            }
            return true;
        };
    }
};

constexpr std::uint64_t TableBytes = 8 * sizeof(std::uint32_t);
constexpr std::uint64_t FixedTable = 0x9000;

std::shared_ptr<DispatchVariant> Stage(std::uint32_t table, std::vector<std::uint32_t> moved, std::vector<std::uint32_t> fixed) {
    auto variant = std::make_shared<DispatchVariant>();
    const auto base = static_cast<std::uint64_t>(table & ~3u);
    variant->runs = {{base + 0x10, base + 0x10 + TableBytes}, {FixedTable, FixedTable + fixed.size() * sizeof(std::uint32_t)}};
    variant->runBases = {2, -1};
    variant->words = std::move(moved);
    variant->words.insert(variant->words.end(), fixed.begin(), fixed.end());
    variant->userDataMask = 0b1110;
    variant->relocatableWords = 0b0100;
    variant->forwardedWords = 0b1000;
    variant->userWordsWalked = 0b0100;
    variant->userWordsAsBase = 0b0100;
    return variant;
}

void StoreStage(Memory& memory, std::uint32_t table, std::span<const std::uint32_t> moved, std::span<const std::uint32_t> fixed) {
    memory.Store(static_cast<std::uint64_t>(table & ~3u) + 0x10, moved);
    memory.Store(FixedTable, fixed);
}

const std::vector<std::uint32_t> Rock{1, 2, 3, 4, 5, 6, 7, 8};
const std::vector<std::uint32_t> Sand{1, 2, 3, 4, 50, 6, 7, 8};
const std::vector<std::uint32_t> Fixed{0xa, 0xb, 0xc, 0xd};

void PlanFollowsTheMovedTable() {
    const std::vector<std::uint32_t> was{7, 0x11, 0x5000, 0x99};
    const auto kept = Stage(0x5000, Rock, Fixed);
    const auto plan = ShapeStagePlan::Of(*kept, was);
    Check(plan.has_value(), "a stage with one moved table and one fixed table has no plan");
    Check(plan->runs.size() == 2 && plan->runs[0].base == 2 && plan->runs[0].at == 0x10 && plan->runs[0].words == 8, "the moved table is not planned relative to its user word");
    Check(plan->runs[1].base == -1 && plan->runs[1].at == FixedTable && plan->runs[1].words == 4, "the fixed table is not planned at its address");
    Check(plan->fixedWords == 0b0010 && plan->alignedWords == 0b0100, "the user words are not split into fixed and moved ones");

    Memory memory;
    StoreStage(memory, 0x7000, Rock, Fixed);
    const auto read = memory.Reader();
    const auto kept_hash = plan->HashKept(*kept, was);
    const std::vector<std::uint32_t> moved{123, 0x11, 0x7000, 0x1234};
    Check(plan->HashRead(moved, read) == kept_hash, "the same words read behind a moved table hash unlike the kept stage");
    const std::vector<std::uint32_t> otherFixed{123, 0x12, 0x7000, 0x1234};
    Check(plan->HashRead(otherFixed, read) != kept_hash, "a fixed user word that differs hashes like the kept stage");
    const std::vector<std::uint32_t> misaligned{123, 0x11, 0x7001, 0x1234};
    Check(plan->HashRead(misaligned, read) != kept_hash, "a moved table at another alignment hashes like the kept stage");
    const std::vector<std::uint32_t> unmapped{123, 0x11, 0x8000, 0x1234};
    Check(!plan->HashRead(unmapped, read).has_value(), "a table that cannot be read still gives a hash");
    const std::vector<std::uint32_t> null{123, 0x11, 0, 0x1234};
    Check(!plan->HashRead(null, read).has_value(), "a null table pointer still gives a hash");
    const std::vector<std::uint32_t> shorter{123, 0x11, 0x7000};
    Check(!plan->HashRead(shorter, read).has_value(), "another count of user words still gives a hash");

    StoreStage(memory, 0x7000, Sand, Fixed);
    Check(plan->HashRead(moved, read) != kept_hash, "another table content hashes like the kept stage");

    const auto masked = Stage(0x5000, Rock, Fixed);
    masked->dataPositions = {4};
    const auto maskedPlan = ShapeStagePlan::Of(*masked, was);
    Check(maskedPlan.has_value() && !(*maskedPlan == *plan), "a stage with a volatile position shares the layout of one without");
    Check(maskedPlan->HashRead(moved, read) == maskedPlan->HashKept(*masked, was), "a word at a volatile position changes the hash");
}

void PlanRefusesInconsistentStages() {
    const std::vector<std::uint32_t> was{7, 0x11, 0x5000, 0x99};
    const auto tailed = Stage(0x5000, Rock, Fixed);
    tailed->decodeRuns = 3;
    Check(!ShapeStagePlan::Of(*tailed, was).has_value(), "more decode runs than runs still gives a plan");
    const auto based = Stage(0x5000, Rock, Fixed);
    based->runBases = {9, -1};
    Check(!ShapeStagePlan::Of(*based, was).has_value(), "a run based on a user word the draw lacks still gives a plan");
    const auto truncated = Stage(0x5000, Rock, Fixed);
    truncated->words.resize(5);
    Check(!ShapeStagePlan::Of(*truncated, was).has_value(), "runs longer than the kept words still give a plan");
    const auto decoded = Stage(0x5000, Rock, Fixed);
    decoded->decodeRuns = 1;
    const auto plan = ShapeStagePlan::Of(*decoded, was);
    Check(plan.has_value() && plan->runs.size() == 1, "the decode runs of a stage are planned with its tables");
}

void IndexFindsTheStageByItsWords() {
    ShapeStageIndex index(64);
    Memory memory;
    const std::vector<std::uint32_t> rockWas{7, 0x11, 0x5000, 0x99};
    const std::vector<std::uint32_t> sandWas{7, 0x11, 0x6000, 0x98};
    const auto rock = Stage(0x5000, Rock, Fixed);
    const auto sand = Stage(0x6000, Sand, Fixed);
    Check(index.Insert(1, 0, 0x100, rock, rockWas), "the first stage was not kept");
    const auto afterRock = index.Serial();
    Check(index.Insert(1, 0, 0x200, sand, sandWas), "a second stage with other words was not kept");
    Check(index.Counted().held == 2 && index.Counted().nodes == 1, "two stages of one shape are not held under one node");

    StoreStage(memory, 0x7000, Sand, Fixed);
    const auto read = memory.Reader();
    const std::vector<std::uint32_t> now{1, 0x11, 0x7000, 0x55};
    std::vector<ShapeStageFound> found;
    index.Find(1, 0, now, ShapeStageIndex::NoBound, read, found);
    Check(found.size() == 1 && found.front().variant == sand && found.front().key == 0x200 && *found.front().was == sandWas, "the stage whose words are in memory was not the one found");
    index.Find(1, 0, now, afterRock, read, found);
    Check(found.empty(), "a stage kept after the bound was found");
    index.Find(2, 0, now, ShapeStageIndex::NoBound, read, found);
    Check(found.empty(), "a stage was found under another shape");
    index.Find(1, 1, now, ShapeStageIndex::NoBound, read, found);
    Check(found.empty(), "a stage was found under another stage index");
    const std::vector<std::uint32_t> gone{1, 0x11, 0x8000, 0x55};
    index.Find(1, 0, gone, ShapeStageIndex::NoBound, read, found);
    Check(found.empty() && index.Counted().unreadable == 1, "an unreadable table found a stage or was not counted");

    const auto again = Stage(0x7000, Sand, Fixed);
    const std::vector<std::uint32_t> againWas{1, 0x11, 0x7000, 0x55};
    Check(!index.Insert(1, 0, 0x300, again, againWas) && index.Counted().held == 2 && index.Counted().present == 1, "the same words at another address were kept twice");
    index.Find(1, 0, now, ShapeStageIndex::NoBound, read, found);
    Check(found.size() == 1 && found.front().variant == again && found.front().key == 0x300, "the newer capture of the same words did not replace the older one");
}

void IndexBoundsItsMemory() {
    ShapeStageIndex index(2);
    Memory memory;
    std::vector<std::shared_ptr<DispatchVariant>> stages;
    for (std::uint32_t k = 0; k < 3; ++k) {
        auto words = Rock;
        words[0] = 100 + k;
        stages.push_back(Stage(0x5000, words, Fixed));
    }
    const std::vector<std::uint32_t> was{7, 0x11, 0x5000, 0x99};
    Check(index.Insert(1, 0, 1, stages[0], was) && index.Insert(1, 0, 2, stages[1], was), "two stages were not kept");
    StoreStage(memory, 0x5000, std::span<const std::uint32_t>(stages[0]->words).first(8), Fixed);
    std::vector<ShapeStageFound> found;
    index.Find(1, 0, was, ShapeStageIndex::NoBound, memory.Reader(), found);
    Check(found.size() == 1 && found.front().variant == stages[0], "the first stage was not found before the index filled");
    Check(index.Insert(1, 0, 3, stages[2], was), "a third stage was not kept");
    Check(index.Counted().held == 2 && index.Counted().evicted == 1, "the index holds more stages than its capacity");
    index.Find(1, 0, was, ShapeStageIndex::NoBound, memory.Reader(), found);
    Check(found.size() == 1 && found.front().variant == stages[0], "the stage found last was evicted before the one never found");
    StoreStage(memory, 0x5000, std::span<const std::uint32_t>(stages[1]->words).first(8), Fixed);
    index.Find(1, 0, was, ShapeStageIndex::NoBound, memory.Reader(), found);
    Check(found.empty(), "the evicted stage was still found");

    ShapeStageIndex layouts(1024);
    for (std::uint32_t k = 0; k <= ShapeStageIndex::MaxLayouts; ++k) {
        const auto stage = Stage(0x5000, Rock, Fixed);
        stage->pushOffset = k * 16;
        Check(layouts.Insert(1, 0, k, stage, was), "a stage with another layout was not kept");
    }
    Check(layouts.Counted().held == ShapeStageIndex::MaxLayouts && layouts.Counted().layoutsDropped == 1, "a stage keeps more layouts than its limit");
    Check(!layouts.Insert(1, 0, 9, nullptr, was), "an absent stage was kept");
}

}

int main() {
    try {
        PlanFollowsTheMovedTable();
        PlanRefusesInconsistentStages();
        IndexFindsTheStageByItsWords();
        IndexBoundsItsMemory();
        std::cout << "lookup from capture index tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
