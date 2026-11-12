#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDARESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BDARESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "Recompiler.hpp"
#include <array>
#include <limits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace AgcDriver::Graphics {

bool LoopGuardTripped();

struct BdaFaultChunk;

// rewrite/direct (DESIGN section B, second step): a device-local copy of a window of guest memory
// that read-only draws reach through the address table. The table of such a draw names the copy
// instead of the host import, and holds it: the recorder brings every copy a draw's table names up
// to date before the draw (Recorder::UseResidentRanges). The fields after `buffer` belong to that
// recorder.
struct ResidentRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::shared_ptr<Buffer> buffer;
    // Tracker generation of the last fill, and the guest mapping generation it was made under.
    std::uint64_t generation = 0;
    std::uint64_t registryGeneration = 0;
    // The whole range is copied at the next use (new, remapped, or its fill was never submitted).
    bool refill = true;
    // No table may name the copy any more (its import is gone): it left the published set.
    bool dead = false;
    std::uint64_t lastUse = 0;
    // The batch that last kept `buffer` alive for its draws.
    std::uint64_t keptSerial = 0;
    // The last check and the tracker state (generation, collect epoch, registry) it was made in.
    std::uint64_t checkedFrame = std::numeric_limits<std::uint64_t>::max();
    std::array<std::uint64_t, 3> checkedAt{};
    // Frames with a fill, for the "keeps changing" rule.
    std::uint64_t filledFrame = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t filledFrames = 0;
    std::uint64_t countedFrom = 0;
    // Bytes copied in since countedFrom: a window is dropped for its refill cost, not for how often
    // a few blocks of it change.
    std::uint64_t filledBytes = 0;
};

class BdaResources {
public:
    explicit BdaResources(const Context& context, bool reportOnly = false);
    // `residentReads`: the table is for a read-only draw and names the published resident ranges.
    BdaResources(const Context& context, const GuestBufferMemory& memory, bool reportOnly = false, bool residentReads = false);
    // The set tables of read-only draws are built with from now on; a table built with another set
    // fails TableGuardCurrent, so its draw builds a new one. A table in use is never rewritten.
    static void PublishResidentRanges(VkDevice device, std::vector<std::shared_ptr<ResidentRange>> ranges);
    std::span<const std::shared_ptr<ResidentRange>> ResidentRanges() const { return residentRanges; }
    ~BdaResources();
    BdaResources(const BdaResources&) = delete;
    BdaResources& operator=(const BdaResources&) = delete;
    VkDescriptorBufferInfo Table() const;
    std::span<const std::byte> TableBytes() const;
    bool TableGuardCurrent(VkDevice device, std::span<const ShaderRecompiler::BdaAbi::Range> ranges);
    VkDescriptorBufferInfo Fault() const;
    void CheckFault() const;
    void CheckUse(const BdaResources* record) const;
    void NoteUse(std::uint64_t serial);
    bool TakeFaultSlot(const Context& context);
    bool HoldsFaultSlot() const { return faultChunk != nullptr; }
    static void NoteFinished(VkDevice device, std::uint64_t serial);
    struct FaultSlotCounts {
        std::uint64_t chunks = 0;
        std::uint64_t slots = 0;
        std::uint64_t held = 0;
        std::uint64_t reclaimed = 0;
        std::uint64_t retaken = 0;
    };
    static FaultSlotCounts FaultSlots(VkDevice device);
    struct ImageTableFaults {
        std::uint32_t stage = 0;
        std::uint64_t shader = 0;
        std::vector<ShaderRecompiler::ImageTableEntryPoison> poison;
        std::vector<std::shared_ptr<const std::string>> messages;
    };
    void SetImageTables(std::vector<ImageTableFaults> tables, std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges);
    struct SrtGuardFaults {
        std::uint32_t stage = 0;
        std::vector<std::uint32_t> flattenedSrt;
    };
    void SetSrtGuards(std::vector<SrtGuardFaults> guards);
    // APS5_PROFILE_DRAW: page tables served from the per-device cache and built anew (cumulative),
    // and how many recent tables the cache holds right now.
    struct TableCacheStats {
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::size_t held = 0;
        std::uint64_t spaceTables = 0;
        // Why the most recently used table did not serve a build ([bda-table] first-differing entry
        // class): its owners were gone (expired weak reference), or its word hash differed, split by
        // whether the first differing range begins below 0x10000000 (a low mapping: the exe image,
        // the system heaps) or in the guest heap above; `sameHash` is a hash match whose ranges
        // still differed, `empty` a lookup with no entry held.
        std::uint64_t firstExpired = 0;
        std::uint64_t firstDiffersLow = 0;
        std::uint64_t firstDiffersHeap = 0;
        std::uint64_t firstSameHash = 0;
        std::uint64_t firstEmpty = 0;
    };
    static TableCacheStats TableCacheCounters();

private:
    // The page table is read-only to the shader, so consecutive builds mapping the same ranges to the
    // same device addresses share one buffer while one of them is alive (see the table cache in
    // BdaResources.cpp, which refers to it weakly).
    void checkFault(const BdaResources& record) const;
    void markWrittenPages(const BdaResources& record) const;
    void checkSrtGuard(const BdaResources& faults) const;
    std::span<std::byte> faultBytes() const;
    std::shared_ptr<Buffer> table;
    VkDevice faultDevice = VK_NULL_HANDLE;
    std::shared_ptr<BdaFaultChunk> faultChunk;
    VkDeviceSize faultOffset = 0;
    bool reportOnly = false;
    std::uint64_t lastUse = 0;
    std::size_t tableBytes = 0;
    std::uint64_t tableGuardSerial = 0;
    bool residentReads = false;
    std::uint64_t residentSerial = 0;
    std::vector<std::shared_ptr<ResidentRange>> residentRanges;
    std::vector<ImageTableFaults> imageTables;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> imageTableRanges;
    std::vector<SrtGuardFaults> srtGuards;
};

}

#endif
