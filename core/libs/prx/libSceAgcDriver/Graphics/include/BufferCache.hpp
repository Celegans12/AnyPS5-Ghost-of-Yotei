#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BUFFERCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_BUFFERCACHE_HPP

#include "prx/libSceAgcDriver/Execution/include/Mutex.hpp"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

class BufferCache {
public:
    static constexpr std::uint64_t PageBytes = 4096;
    static constexpr std::uint64_t GranuleBytes = 65536;
    static constexpr std::uint64_t ZoneBytes = std::uint64_t{2} << 20u;
    static constexpr std::uint64_t MirrorLimit = std::uint64_t{64} << 20u;
    static constexpr std::uint64_t Budget = std::uint64_t{1} << 30u;
    static constexpr std::uint64_t AddressLimit = std::uint64_t{1} << 47u;
    static constexpr std::size_t DerivedEntries = 8192;
    static constexpr std::uint64_t StreamRest = 128;
    static constexpr std::uint64_t StreamSpan = 4 * GranuleBytes;
    static constexpr std::size_t RingChunkBytes = std::size_t{4} << 20u;
    static constexpr std::size_t RingChunks = 64;
    static constexpr std::size_t RingEntries = 1024;
    static constexpr std::size_t RingReach = RingChunkBytes / 4;
    static constexpr std::size_t LooseEntries = 16;
    static constexpr std::uint64_t ArmRest = 128;
    static constexpr std::uint64_t CopyIds = std::uint64_t{1} << 62u;

    class Source {
    public:
        virtual ~Source() = default;
        virtual std::uint64_t Layout() = 0;
        virtual bool Poll(std::uint64_t address, std::size_t bytes, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written) = 0;
        virtual bool Quiet(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> tokens) = 0;
        virtual bool Volatile(std::uint64_t) { return false; }
        virtual bool Holds(std::uint64_t, std::size_t, std::span<const std::uint64_t>) { return false; }
        virtual void Arm(std::uint64_t, std::size_t, std::span<std::uint64_t> tokens, std::span<std::uint16_t> written) {
            std::fill(tokens.begin(), tokens.end(), std::uint64_t{0});
            std::fill(written.begin(), written.end(), std::uint16_t{0xffffu});
        }
    };

    enum class Unarmed : std::uint8_t { Armed, Held, Imported, Barred, Resting, Edge, Other, GpuWritten, Count };
    static constexpr std::size_t UnarmedReasons = static_cast<std::size_t>(Unarmed::Count);

    struct ArmedBlocks {
        bool (*tokens)(std::uint64_t address, std::size_t bytes, std::span<std::uint64_t> words) = nullptr;
        bool (*cleanSince)(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> words) = nullptr;
        bool (*arm)(std::uint64_t address, std::size_t bytes, std::span<std::uint64_t> words) = nullptr;
        bool (*rearm)(std::uint64_t address, std::size_t bytes, std::span<const std::uint64_t> since, std::span<std::uint64_t> words, std::span<std::uint16_t> written) = nullptr;
        std::uint8_t (*reason)(std::uint64_t address) = nullptr;
    };
    static constexpr std::uint64_t ArmedToken = std::uint64_t{1} << 63u;
    static constexpr std::uint16_t EveryPage = 0xffffu;

    struct Backing {
        std::shared_ptr<void> owner;
        std::byte* bytes = nullptr;
        VkBuffer handle = VK_NULL_HANDLE;
    };
    using Allocate = std::function<Backing(std::size_t bytes)>;
    using Executed = std::function<bool(std::uint64_t serial)>;

    enum class Refusal : std::uint8_t { Range, Alignment, GpuWrite, Untracked, Unmapped, Busy, Span, Budget, Allocation, Raced, Streamed, Label, Count };
    static constexpr std::size_t Refusals = static_cast<std::size_t>(Refusal::Count);

    enum class Served : std::uint8_t { Tracked, Copied, Repeated, Kept, Clean };

    struct Input {
        std::size_t alignment = 1;
        bool apart = false;
        bool fresh = false;
        bool partialHead = false;
        bool partialTail = false;
    };

    struct View {
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        const std::byte* bytes = nullptr;
        std::shared_ptr<void> owner;
        std::uint64_t mirror = 0;
        std::uint64_t sequence = 0;
        bool streamed = false;
        Served served = Served::Tracked;
        explicit operator bool() const { return handle != VK_NULL_HANDLE; }
    };

    struct Derived {
        bool some = false;
        std::uint32_t highest = 0;
        std::uint32_t lowest = 0;
    };

    struct Counts {
        std::uint64_t lookups = 0;
        std::uint64_t clean = 0;
        std::uint64_t uploads = 0;
        std::uint64_t pages = 0;
        std::uint64_t bytes = 0;
        std::uint64_t equalPages = 0;
        std::uint64_t mirrors = 0;
        std::uint64_t merges = 0;
        std::uint64_t evictions = 0;
        std::uint64_t invalidations = 0;
        std::uint64_t derivedHits = 0;
        std::uint64_t verified = 0;
        std::uint64_t verifyRaces = 0;
        std::uint64_t armedPolls = 0;
        std::uint64_t armsRequested = 0;
        std::uint64_t armsGranted = 0;
        std::uint64_t streamed = 0;
        std::uint64_t streamedBytes = 0;
        std::uint64_t streamSwitches = 0;
        std::uint64_t streamReturns = 0;
        std::uint64_t edgePages = 0;
        std::uint64_t stageViews = 0;
        std::uint64_t stageLeft = 0;
        std::uint64_t movedViews = 0;
        std::array<std::uint64_t, Refusals> stageRefused{};
        std::uint64_t heldBytes = 0;
        std::array<std::uint64_t, Refusals> refused{};
        std::array<std::uint64_t, UnarmedReasons> unarmed{};
        std::uint64_t inputs = 0;
        std::uint64_t inputClean = 0;
        std::uint64_t inputKept = 0;
        std::uint64_t inputUploads = 0;
        std::uint64_t inputUploadBytes = 0;
        std::uint64_t inputRing = 0;
        std::uint64_t inputRingBytes = 0;
        std::uint64_t inputLoose = 0;
        std::uint64_t inputLooseBytes = 0;
        std::uint64_t inputRepeats = 0;
        std::uint64_t inputRefused = 0;
        std::uint64_t inputVerified = 0;
        std::uint64_t inputRaces = 0;
        std::uint64_t inputRunDiffers = 0;
        std::uint64_t ringChunks = 0;
        std::uint64_t ringLaps = 0;
        std::uint64_t ringFull = 0;
        std::uint64_t drawRuns = 0;
    };

    BufferCache(Allocate allocate, Executed executed, Source& source);
    ~BufferCache();
    BufferCache(const BufferCache&) = delete;
    BufferCache& operator=(const BufferCache&) = delete;

    static bool Enabled();
    static bool Verifying();
    static bool DirtyFirst();
    static bool VerifyingDirtyFirst();
    static std::size_t RingLimit();
    static void BreakDrawRun();
    static void EndDrawRun();
    static std::uint64_t DrawRun();
    static bool Arming();
    static bool Streaming();
    static void NoteStreamed(std::size_t bytes);
    static bool StageViews();
    static void NoteStage(bool served);
    static bool MovedViews();
    static void NoteMoved();
    static Source& TrackerSource();
    static void SetArmedBlocks(const ArmedBlocks& table);
    static Counts Counted();
    static const char* RefusalName(Refusal reason);
    static std::string Report();

    View Acquire(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial, bool partialHead = false, bool partialTail = false);
    View AcquireInput(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial, const Input& input);
    void VerifyInput(const View& view, std::uint64_t address, std::size_t bytes, std::size_t from);
    View Refuse(Refusal reason);
    void Invalidate(std::uint64_t address, std::size_t bytes);
    void Verify(const View& view, std::uint64_t address, std::size_t bytes, std::size_t from);
    std::optional<Derived> Known(const View& view, std::uint64_t address, std::size_t bytes, std::uint32_t kind) const;
    void Keep(const View& view, std::uint64_t address, std::size_t bytes, std::uint32_t kind, const Derived& value);
    std::size_t Mirrors() const { return mirrors.size(); }
    void Stage(bool serving) { stage = serving; }

private:
    struct Granule {
        std::atomic<std::uint64_t> token{0};
        std::uint64_t sequence = 0;
        std::uint64_t streamUntil = 0;
        std::atomic<std::uint64_t> run{0};
        std::uint64_t armAfter = 0;
        std::uint16_t valid = 0;
        std::uint16_t hot = 0;
    };
    struct Mirror {
        std::uint64_t id = 0;
        std::uint64_t begin = 0;
        std::uint64_t end = 0;
        Backing backing;
        std::unique_ptr<Granule[]> granules;
        std::unique_ptr<std::uint64_t[]> bound;
        std::uint64_t layout = 0;
        std::uint64_t used = 0;
    };
    struct DerivedEntry {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t mirror = 0;
        std::uint64_t sequence = 0;
        std::uint32_t kind = 0;
        Derived value;
    };
    struct RingChunk {
        Backing backing;
        std::uint64_t serial = 0;
        std::uint64_t lap = 0;
    };
    struct RingEntry {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t from = 0;
        std::uint64_t run = 0;
        std::uint64_t lap = 0;
        std::uint64_t id = 0;
        RingChunk* chunk = nullptr;
        VkDeviceSize offset = 0;
    };
    struct LooseEntry {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t from = 0;
        View view;
    };
    using Slot = std::atomic<Mirror*>;
    static constexpr unsigned ChunkShift = 30;
    static constexpr unsigned ZoneShift = 21;
    static constexpr std::size_t ChunkZones = std::size_t{1} << (ChunkShift - ZoneShift);
    static constexpr std::size_t Chunks = static_cast<std::size_t>(AddressLimit >> ChunkShift);

    Mirror* find(std::uint64_t address) const;
    Slot& slot(std::uint64_t address);
    Mirror* cover(std::uint64_t address, std::uint64_t end);
    bool evict(std::span<Mirror* const> spared);
    void drop(Mirror* mirror);
    bool pending(std::uint64_t serial);
    View acquire(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial, bool partialHead, bool partialTail, bool dirtyFirst);
    View cleanInput(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial);
    View ringInput(std::uint64_t address, std::size_t bytes, std::size_t from, std::uint64_t serial, std::uint64_t run, const Input& input);
    View looseInput(std::uint64_t address, std::size_t bytes, std::size_t from, const Input& input);
    bool nextRingChunk();
    static std::size_t ringIndex(std::uint64_t address, std::size_t bytes, std::size_t from);
    static std::size_t derivedIndex(std::uint64_t address, std::size_t bytes, std::uint32_t kind);

    Allocate allocate;
    Executed executed;
    Source& source;
    AgcDriver::Mutex mutex;
    std::unique_ptr<std::atomic<Slot*>[]> chunks;
    std::vector<std::unique_ptr<Mirror>> mirrors;
    std::vector<DerivedEntry> derived;
    std::vector<std::uint64_t> held;
    std::vector<std::uint64_t> tokens;
    std::vector<std::uint16_t> written;
    std::vector<Mirror*> absorbed;
    std::vector<std::unique_ptr<RingChunk>> ring;
    std::vector<RingEntry> ringEntries;
    std::vector<LooseEntry> loose;
    std::size_t ringAt = 0;
    std::size_t ringUsed = 0;
    std::uint64_t looseRun = 0;
    std::uint64_t copyIds = 0;
    std::atomic<std::uint64_t> invalidations{0};
    std::uint64_t total = 0;
    std::uint64_t tick = 0;
    std::uint64_t mirrorIds = 0;
    std::uint64_t sequence = 0;
    std::uint64_t executedFloor = 0;
    std::uint64_t pendingFloor = 0;
    bool stage = false;
};

class Recorder;

BufferCache::View CachedRange(Recorder& recorder, std::uint64_t address, std::size_t bytes, std::size_t from, std::size_t alignment);
BufferCache::View CachedStageRange(Recorder& recorder, std::uint64_t address, std::size_t bytes, std::size_t alignment);
BufferCache::View DirtyFirstRange(Recorder& recorder, std::uint64_t address, std::size_t bytes, std::size_t from, std::size_t alignment);

}

#endif
