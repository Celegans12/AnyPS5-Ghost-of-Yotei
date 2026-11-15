#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWPIPELINE_HPP

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/FlatTransport.hpp"

namespace AgcDriver::DriverDetail {

class DrawFinisher;

class DrawPipeline {
public:
    using Commit = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    struct Watch {
        std::uint64_t retired = 0;
        std::size_t items = 0;
        std::vector<Range> writes;
    };
    enum class DrainReason : std::uint8_t { Packet, Flush, Labels, Capture, Indirect, Submission, Count };
    enum class QueuedWrite : std::uint8_t { None, Stored, Unknown };
    enum class CommitKind : std::uint8_t { Other, Draw, IndirectDraw, ColorMetadata, Dispatch, SampleDump, DmaFill, DmaWrite, DmaCopy, Store, EndOfPipe, Seal, SubmitEnd, Completion, BufferFill, Count };
    struct Order {
        std::uint64_t producer = 0;
        std::uint64_t sequence = 0;
    };
    class CommitOrder {
    public:
        bool Accept(Order next);
    private:
        std::vector<Order> last;
    };

    static DrawPipeline& Queue0();
    static std::size_t Depth();
    static std::size_t QueuedItems();
    static bool& Active();
    static std::atomic<std::uint64_t>& EpochToken();
    static void FollowEpoch(std::uint64_t token);
    static bool SharedCollectEpoch();
    static bool VerifiesOrder();
    static unsigned Committers();
    static bool Flat();
    static std::size_t WakeBatch(std::size_t depth);
    struct TransportCounts {
        std::uint64_t elements = 0;
        std::uint64_t wide = 0;
        std::uint64_t roomWaits = 0;
        std::uint64_t consumerWakes = 0;
        std::uint64_t consumerSleeps = 0;
        std::uint64_t waiterWakes = 0;
        std::uint64_t collisions = 0;
        std::uint64_t queries = 0;
        std::uint64_t scanned = 0;
        std::uint64_t lockedQueries = 0;
    };
    static TransportCounts Transported();

    void Enqueue(Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, bool draw = false);
    void Enqueue(CommitKind kind, Commit commit, std::vector<Range> writes, std::uint64_t labelAddress = 0, std::vector<std::byte> labelBytes = {}, std::vector<Range> reads = {}, bool draw = false);
    void EnqueueStore(Commit commit, std::uint64_t address, std::vector<std::byte> bytes);
    static void NoteDrawKey(std::uint64_t key, std::uint64_t shape);
    void Drain(DrainReason reason, std::uint32_t opcode = 0x100);
    void Stop();
    void ReleaseRetired();
    static std::uint64_t DrainRequestsByOpcode(std::uint32_t opcode);
    static std::uint64_t DrainRequestsByReason(DrainReason reason);
    bool Busy() const;
    std::size_t Outstanding() const;
    bool Overlaps(std::uint64_t address, std::size_t bytes);
    bool OverlapsAny(std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
    Watch WatchWrites();
    bool WrittenSince(const Watch& watch, const std::vector<Range>& reads);
    std::optional<bool> WrittenWhileQueued(std::uint64_t retiredBefore, const std::vector<Range>& reads);
    std::uint64_t Retired() const;
    void NoteDecodeDrain() { decodeDrains.fetch_add(1, std::memory_order_relaxed); }
    std::uint64_t DecodeDrains() const { return decodeDrains.load(std::memory_order_relaxed); }
    std::optional<std::uint64_t> PendingLabel(std::uint64_t address, std::size_t bytes);
    static bool HeldReads(std::uint64_t address, std::size_t bytes);
    static std::uint64_t HeldReadQueries();
    void ReleaseRunningReads();
    QueuedWrite StoredBytes(std::uint64_t address, std::size_t bytes, std::span<std::byte> out);

    static bool MetadataPassesPipelined();
    static void CountPipelinedMetadataPass();
    static std::uint64_t PipelinedMetadataPasses();
private:
    friend class DrawFinisher;
    struct Item {
        Commit commit;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
        std::vector<Range> reads;
        bool draw = false;
        bool store = false;
        CommitKind kind = CommitKind::Other;
        std::uint64_t collectEpoch = 0;
        Order order;
    };
    DrawPipeline() = default;
    void enqueue(Item item);
    void relay(Item item);
    void push(Item item);
    void pushFlat(Item item);
    void run();
    void runFlat();
    void rethrowFailure();
    void rethrowFlatFailure();
    void report(std::chrono::steady_clock::time_point now);
    void countWrites(const std::vector<Range>& writes, bool added);
    static void breakProofsAfter(CommitKind kind, const std::vector<Range>& writes);

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Item> items;
    std::vector<std::pair<Range, std::uint32_t>> queuedWrites;
    std::vector<Item> retired;
    std::exception_ptr failure;
    std::atomic<std::size_t> outstanding{0};
    std::atomic<std::uint64_t> retiredItems{0};
    std::atomic<std::uint64_t> decodeDrains{0};
    std::uint64_t reportedDecodeDrains = 0;
    std::uint32_t idleWaiters = 0;
    std::size_t resumeAt = 0;
    bool committerWaiting = false;
    bool stopped = false;
    bool closing = false;
    std::uint64_t wakes = 0;
    std::uint64_t idleWaits = 0;
    std::uint64_t idleNs = 0;
    std::uint64_t lockWaitNs = 0;
    std::thread thread;
    std::uint64_t commits = 0;
    std::uint64_t commitNs = 0;
    std::uint64_t commitErrors = 0;
    std::uint64_t enqueued = 0;
    std::uint64_t fullWaitNs = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drains{};
    std::array<std::uint64_t, static_cast<std::size_t>(DrainReason::Count)> drainWaitNs{};
    std::array<std::uint64_t, 257> drainOpcodes{};
    std::array<std::atomic<std::uint64_t>, 257> drainRequests{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(DrainReason::Count)> reasonRequests{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CommitKind::Count)> kindCommits{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CommitKind::Count)> kindNs{};
    std::uint64_t depthSum = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::unique_ptr<FlatQueue<Item>> flat;
    std::atomic<bool> flatFailed{false};
    std::atomic<bool> flatStarted{false};
    std::atomic<std::uint64_t> flatCommits{0};
    std::atomic<std::uint64_t> flatCommitNs{0};
    std::atomic<std::uint64_t> flatCommitErrors{0};
    std::atomic<std::uint64_t> flatIdleWaits{0};
    std::atomic<std::uint64_t> flatIdleNs{0};
    std::atomic<std::uint64_t> flatRoomWaits{0};
    std::uint64_t flatReportedWakes = 0;
};

class DrawFinisher {
public:
    using Job = std::function<void()>;
    using Range = std::pair<std::uint64_t, std::uint64_t>;
    struct Pending {
        std::uint64_t key = 0;
        std::uint64_t shape = 0;
        bool unknown = true;
        std::vector<Range> writes;
        std::uint64_t labelAddress = 0;
        std::vector<std::byte> labelBytes;
    };
    enum class LabelLookup : std::uint8_t { NotHeld, Value, Unknown };
    static bool Overlap();
    bool PendingOverlaps(std::uint64_t begin, std::uint64_t end);
    LabelLookup PendingLabel(std::uint64_t address, std::size_t bytes, std::uint64_t& value);
    bool PendingOverlapsAny(std::span<const Range> ranges);
    bool PendingKey(std::uint64_t key, std::uint64_t shape);
    enum class DrainSite : std::uint32_t { Lookup, Reads, Packet, Kept, SubmissionEnd, Item, Count };
    static void NoteDrain(DrainSite site, bool drained);
    static void NoteItemDrain(DrawPipeline::CommitKind kind);
    static void NoteItemRelay(DrawPipeline::CommitKind kind);
    static std::uint64_t DrainCount(DrainSite site, bool drained);
    static bool Relays();
    static bool RelaysSubmissionEnd();
    static std::string DrainText();
    static void NoteDrainedOpcode(std::uint32_t opcode, bool besideCommits);
    static DrawFinisher& Get();
    static bool Enabled();
    static bool OnFinisherThread();
    static bool Busy() { return outstanding.load(std::memory_order_acquire) != 0; }
    static constexpr unsigned MaxThreads = 16;
    static constexpr std::size_t QueueDepth = 16;
    static unsigned Threads();
    static bool Defers();
    static void Defer(Job output, std::vector<Range> reads);
    bool DeferredReads(std::uint64_t address, std::uint64_t end);
    static bool Behind();
    void TakeTurn();
    struct PoolCounts {
        std::uint64_t jobs = 0;
        std::uint64_t beside = 0;
        std::uint64_t aloneDraws = 0;
        std::uint64_t aloneUnlisted = 0;
        std::uint64_t skipped = 0;
        std::uint64_t turns = 0;
        std::uint64_t reordered = 0;
        std::uint64_t reorderNs = 0;
    };
    static PoolCounts PoolCounted();
    struct ThreadCounts {
        std::uint64_t jobs = 0;
        std::uint64_t busyNs = 0;
        std::uint64_t idleNs = 0;
        std::uint64_t turns = 0;
        std::uint64_t turnNs = 0;
        std::uint64_t outputNs = 0;
        std::uint64_t lookups = 0;
        std::uint64_t lookupNs = 0;
        std::uint64_t cacheWaits = 0;
        std::uint64_t cacheWaitNs = 0;
        std::uint64_t unlockedHits = 0;
    };
    static constexpr unsigned OtherThreads = MaxThreads;
    static ThreadCounts ThreadCounted(unsigned index);
    static bool Timed();
    static void NoteLookup(std::uint64_t nanoseconds);
    static void NoteCacheWait(std::uint64_t nanoseconds);
    static void NoteUnlockedHit();
    static std::string LookupText();
    static DrawPipeline::TransportCounts Transported();
    void Enqueue(Job job);
    void Enqueue(Job job, Pending held);
    void Drain();
    void DrainAt(DrainSite site);
    void CheckFailure();
    void Settle();
    void ReleaseRan();

private:
    friend class DrawPipeline;
    struct Work {
        Job job;
        DrawPipeline::Item relayed;
        DrawPipeline* target = nullptr;
        std::uint64_t epoch = 0;
        DrawPipeline::Order order;
    };
    struct Slot {
        Job job;
        Pending held;
        std::vector<Job> outputs;
        std::vector<Range> reads;
        std::exception_ptr error;
        DrawPipeline::Order order;
        std::chrono::steady_clock::time_point finished{};
        std::uint64_t epoch = 0;
        std::uint64_t sequence = 0;
        bool alone = false;
        bool done = false;
        bool shutdown = false;
    };
    DrawFinisher() = default;
    void run();
    void runFlat();
    void enqueueFlat(Job job, DrawPipeline::Item* relayed, DrawPipeline* target, const Pending& held);
    void rethrowFlatFailure();
    void enqueuePooled(Job job, Pending held);
    void work(unsigned index);
    void retire(std::unique_lock<std::mutex>& lock);
    bool startable() const;
    inline static std::atomic<std::size_t> outstanding{0};
    inline static std::atomic<std::uint64_t> retiredSequence{0};
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Job> jobs;
    std::deque<std::uint64_t> epochs;
    std::deque<DrawPipeline::Order> orders;
    std::deque<Pending> pending;
    std::deque<std::unique_ptr<Slot>> slots;
    std::vector<Job> ran;
    std::exception_ptr failure;
    std::vector<std::thread> workers;
    std::atomic<std::size_t> unstarted{0};
    std::uint64_t queuedSequence = 0;
    std::size_t started = 0;
    unsigned active = 0;
    unsigned sleepers = 0;
    bool barrier = false;
    bool flushing = false;
    bool running = false;
    bool sleeping = false;
    std::thread thread;
    std::unique_ptr<FlatQueue<Work>> flat;
    std::unique_ptr<PendingLedger> poolLedger;
    std::atomic<bool> flatFailed{false};
    std::atomic<bool> flatStarted{false};
};

template<typename TMutex>
class ContendedLock {
public:
    explicit ContendedLock(TMutex& held) : mutex(held) {
        if (mutex.try_lock()) return;
        const auto from = std::chrono::steady_clock::now();
        mutex.lock();
        DrawFinisher::NoteCacheWait(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - from).count()));
    }
    ~ContendedLock() { mutex.unlock(); }
    ContendedLock(const ContendedLock&) = delete;
    ContendedLock& operator=(const ContendedLock&) = delete;

private:
    TMutex& mutex;
};

class CollectWalker {
public:
    struct Counts {
        std::uint64_t handed = 0;
        std::uint64_t collected = 0;
        std::uint64_t dropped = 0;
        std::uint64_t collectNs = 0;
        std::uint64_t sleeps = 0;
    };
    static constexpr std::size_t Capacity = 1024;
    static CollectWalker& Get();
    static bool Enabled();
    bool Offer(std::uint64_t epoch, std::uint64_t address, std::uint64_t bytes);
    bool Idle() const;
    Counts Counted() const;
    void Stop();

private:
    struct Entry {
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t epoch = 0;
    };
    CollectWalker() = default;
    void start();
    void run();
    void rethrowFailure();

    std::array<Entry, Capacity> ring{};
    std::mutex producers;
    alignas(64) std::atomic<std::uint64_t> tail{0};
    std::uint64_t knownHead = 0;
    std::atomic<std::uint64_t> handed{0};
    std::atomic<std::uint64_t> dropped{0};
    alignas(64) std::atomic<std::uint64_t> head{0};
    std::atomic<std::uint64_t> collected{0};
    std::atomic<std::uint64_t> collectNs{0};
    std::atomic<std::uint64_t> sleeps{0};
    alignas(64) std::atomic<std::uint32_t> parked{0};
    std::atomic<bool> closed{false};
    std::atomic<bool> started{false};
    std::atomic<bool> failed{false};
    std::mutex lifecycle;
    std::exception_ptr failure;
    std::thread thread;
};

}

#endif
