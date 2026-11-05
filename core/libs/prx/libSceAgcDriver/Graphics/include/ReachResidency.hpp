#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_REACHRESIDENCY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_REACHRESIDENCY_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {
class Buffer;
}

// Reach residency (APS5_REACH_RESIDENCY=1; research/speed/newpath/A2-design.md). The BDA heap reads of
// compute programs that never store through BDA go to device-local copies of the 64 KiB units their
// program read on earlier uses (learned from the reach notes of learning uses, BdaAbi::NoteReadsTable).
// Each lane has one pool; a unit gets a fixed slot until it is evicted. A copy is current for a reader
// when it was filled (a vkCmdCopyBuffer from the import, recorded on the reader's lane) and nothing that
// could change the unit's bytes in the reader's GPU order happened since:
// - same open batch: no write note over the unit and no BDA writer recorded on the lane after the fill;
// - a later batch: the fill saw no pending write of the other lane, no BDA writer recorded after the
//   fill is still incomplete (their written pages are stamped at completion), no write note over the
//   unit since, and the unit's blocks are unchanged since the fill (CPU stores collected).
// Everything here runs under GuestMemory::GpuMutex except NoteLearned (own mutex).
namespace AgcDriver::Graphics::Reach {

inline constexpr std::uint64_t UnitBytes = 65536;

bool Enabled();
// Enabled and APS5_NO_REACH_COPIES unset (taken once per build).
bool CopiesActive();

// Learning: whether the next fresh build of `program` is a learning build (its table notes reads).
bool LearnBuild(std::uint64_t program);
// A learning build's fault words after its use: its read pages go into the program's history and the
// note slots are cleared for the next use.
void NoteLearned(std::uint64_t program, std::uint32_t* faultWords);
// The units (base addresses, ascending) the program read in its recent learning uses.
std::vector<std::uint64_t> Predicted(std::uint64_t program);
// Whether the program learned nothing yet though a learning build of it exists (its use has not completed).
bool LearningPending(std::uint64_t program);

// BDA writer uses (shaders that may store through BDA): recorded with their lane, completed once their
// written pages are reported.
std::uint64_t WriterRecorded(int lane);
void WriterCompleted(std::uint64_t seq);

// Every pending-write note of the recorder (any lane, open or in-flight batch).
void NoteWrite(std::uint64_t begin, std::uint64_t end);

struct UnitSource {
    std::uint64_t unit;
    VkBuffer import;
    VkDeviceSize offset;
};
struct Fill {
    VkBuffer import;
    VkDeviceSize sourceOffset;
    VkDeviceSize slotOffset;
    std::uint64_t unit;
};
// A build's mapping: (unit, slot) pairs of one lane's pool generation.
struct Mapping {
    int lane = 0;
    std::uint64_t epoch = 0;
    std::shared_ptr<Buffer> pool;
    std::vector<UnitSource> units;
    std::vector<std::uint32_t> slots;
};
struct LaneView {
    int lane;
    std::uint64_t batch;
    // Whether a pending write of the other lane overlaps [begin, end) (or a BDA writer of the other lane
    // is pending at all): such a fill is current only within its batch.
    std::function<bool(std::uint64_t, std::uint64_t)> otherLanePending;
};
// Assigns slots to the candidates (at most the pool), and lists the fills needed for them to be current
// for a reader recorded now. Nothing when the pool cannot be had.
bool Map(const Context& context, const LaneView& view, std::span<const UnitSource> candidates, Mapping& mapping, std::vector<Fill>& fills);
// For a reused build: the fills needed for its mapping to be current now; false when a slot was given to
// another unit, the pool replaced, or the lane differs (the build cannot be reused).
bool Refresh(const LaneView& view, const Mapping& mapping, std::vector<Fill>& fills, bool counted = true);
// After the fills were recorded (before the reader): they are current from now.
void Filled(const LaneView& view, std::span<const Fill> fills);

// Counters: [reach] line (APS5_PROFILE_REACH=1) every 10 s.
void CountBuild(bool learning, std::size_t predicted, std::size_t mapped, std::size_t refused);
// Address-based compute builds seen (0 read-only, 1 BDA writer) and why a read-only one took no reach path (2 no staging, 3 not addressable, 4 no lease, 5 no recorder).
void CountCandidate(int what);
void Report();
// The same counters per profile window (ProfileWindows::Close), with APS5_REACH_RESIDENCY set.
void CloseWindow(const char* label);
struct Counts {
    std::uint64_t builds = 0, learning = 0, learnedUses = 0, predictedUnits = 0, mappedUnits = 0, refusedUnits = 0;
    std::uint64_t fills = 0, fillBytes = 0, reused = 0, invalidNote = 0, invalidWriter = 0, invalidBatch = 0, invalidStamp = 0, evicted = 0, refreshes = 0, refreshFailed = 0, refusedHot = 0;
};
Counts Totals();
// Tests: forget every lane, history and writer; give a lane `slots` slots without a device pool.
void ResetForTests();
void UsePoolForTests(int lane, std::size_t slots);

}

#endif
