#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCKORDER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCKORDER_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#ifndef _WIN32
#include <shared_mutex>
#endif
#include <stdexcept>
#include <string>

// v3 lock split (PLAN-v3-draft §5b, exec-design.md "Lock split"). The record lock is GuestMemory::GpuMutex
// (GuestMemory::RecordLock); below it sit ranked cache locks, then mem's tracker and dirty-tracker locks. The
// agreed order, lowest rank first: record lock -> texture < storage < resource-set < pipeline < depth/DCC < imports
// (R6) < residency (R5) -> tracker -> GuestDirty armMutex -> GuestDirty block lock -> WindowsMappings mutex. A thread
// may take a lock only while every lock it holds ranks lower (re-entering a lock it holds is allowed).
// APS5_V3E_LOCK_ORDER_CHECK (start-time) keeps a per-thread record of held ranks and throws OrderError on a
// wrong-order acquire. APS5_V3E_LOCK_SPLIT (RuntimeSwitch) makes the caches' own mutexes (SplitMutex) also take
// their cache lock while the record lock is held; phase (i) moves no work out of the record lock.
namespace AgcDriver::LockOrder {

enum class Rank : std::uint8_t {
    Record,
    Texture,
    Storage,
    ResourceSet,
    Pipeline,
    DepthDcc,
    Imports,
    Residency,
    Tracker,
    DirtyArm,
    DirtyBlock,
    WindowsMappings,
    Count,
};
constexpr std::size_t RankCount = static_cast<std::size_t>(Rank::Count);
constexpr bool IsCacheRank(Rank rank) { return rank >= Rank::Texture && rank <= Rank::Residency; }

const char* Name(Rank rank);

class OrderError : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

bool CheckEnabled();
void SetCheckForTest(bool enabled);

// Before a first (not re-entering) acquire of a lock of `rank` on this thread: with the check on, throws
// OrderError when the thread holds a lock of a higher rank, or a cache lock when `rank` is the record lock.
void BeforeAcquire(Rank rank);
// The record lock's outermost release: with the check on, a cache lock still held is a violation (another thread
// can then take the record lock and wait on that cache lock while this one waits for the record lock). It is
// printed and counted, not thrown (it runs in unlock()).
void RecordReleased();

// Marks a lock of `rank` held by this thread without locking anything: for mem's locks (tracker, GuestDirty,
// WindowsMappings) and the tests.
void NoteHeld(Rank rank);
void NoteReleased(Rank rank);
class Annotate {
public:
    explicit Annotate(Rank rank) : rank(rank) {
        BeforeAcquire(rank);
        NoteHeld(rank);
    }
    ~Annotate() { NoteReleased(rank); }
    Annotate(const Annotate&) = delete;
    Annotate& operator=(const Annotate&) = delete;

private:
    Rank rank;
};

std::uint32_t HeldOnThisThread(Rank rank);
bool HoldsCacheLock();
// Whether this thread must not block on another thread's progress (the DrawPipeline/finisher guards, the pipeline
// creation gate, the residency fault service): it holds the record lock, or, with APS5_V3E_LOCK_SPLIT, any cache lock.
bool MustNotWait();
std::string HeldText();

struct Counters {
    std::uint64_t violations = 0;
    std::uint64_t releasesHolding = 0;
    std::uint64_t contended = 0;
    std::uint64_t splitTakes = 0;
};
Counters Totals();

// A ranked reader/writer lock. Re-entrant on one thread: exclusive inside exclusive, shared inside shared or
// exclusive. Asking for exclusive while holding it shared throws OrderError (it would deadlock).
class CacheLock {
public:
    explicit CacheLock(Rank rank) : rank(rank) {}
    CacheLock(const CacheLock&) = delete;
    CacheLock& operator=(const CacheLock&) = delete;
    void lock();
    bool try_lock();
    void unlock();
    void lock_shared();
    bool try_lock_shared();
    void unlock_shared();
    bool HeldByThisThread() const;
    bool HeldExclusiveByThisThread() const;
    Rank LockRank() const { return rank; }

private:
    void rawLock();
    bool rawTryLock();
    void rawUnlock();
    void rawLockShared();
    bool rawTryLockShared();
    void rawUnlockShared();
    Rank rank;
#ifdef _WIN32
    void* srw = nullptr;
#else
    std::shared_mutex mutex;
#endif
};

CacheLock& CacheLockFor(Rank rank);

// Whether a SplitMutex acquire takes its cache lock now: APS5_V3E_LOCK_SPLIT is set and the thread holds the record lock.
bool SplitTakes();
// APS5_V3E_LOCK_SPLIT_LEAF (start-time): the first phase (i) build's per-acquisition takes in SplitMutex (v3e-ab6/ab7:
// ~110k takes per frame, +29 ms/frame of mutex). Unset, the cache locks are held per stage instead (StageHold).
void SetLeafTakesForTest(bool enabled);
// A draw commit's stage hold: with APS5_V3E_LOCK_SPLIT set (and not the leaf mode), while the record lock is held, the
// draw caches' locks (texture, storage, resource-set, pipeline, depth/DCC) are taken exclusive once, in rank order,
// for the whole stage.
constexpr std::array<Rank, 5> StageRanks{Rank::Texture, Rank::Storage, Rank::ResourceSet, Rank::Pipeline, Rank::DepthDcc};
class StageHold {
public:
    StageHold();
    ~StageHold();
    StageHold(const StageHold&) = delete;
    StageHold& operator=(const StageHold&) = delete;
    bool Held() const { return held; }

private:
    friend void RecordAcquired();
    friend void RecordReleased();
    void suspend();
    void resume();
    bool held = false;
    bool suspended = false;
    StageHold* previous = nullptr;
};
// A record-lock release at depth 1 (SyncThrough around a GPU wait) releases the thread's stage holds first, and
// the next outermost acquire takes them again, so no cache lock is held while the record lock is not.
void RecordAcquired();
std::uint64_t StageSuspends();
void CountSplitTake();
[[noreturn]] void NestedTooDeep(Rank rank);

// A cache's own mutex (Inner), which with APS5_V3E_LOCK_SPLIT also takes the cache's ranked CacheLock (exclusive)
// first while the record lock is held. Unlocked readers (stage A, PrepareDispatch probes, the preparer) take only
// Inner, as today.
template<typename Inner, Rank R>
class SplitMutex {
public:
    SplitMutex() = default;
    SplitMutex(const SplitMutex&) = delete;
    SplitMutex& operator=(const SplitMutex&) = delete;
    void lock() {
        const bool take = SplitTakes();
        if (take) CacheLockFor(R).lock();
        inner.lock();
        push(take);
    }
    bool try_lock() {
        const bool take = SplitTakes();
        if (take && !CacheLockFor(R).try_lock()) return false;
        if (!inner.try_lock()) {
            if (take) CacheLockFor(R).unlock();
            return false;
        }
        push(take);
        return true;
    }
    void unlock() {
        const std::uint32_t level = depth != 0 ? --depth : 64;
        const bool took = level < 64 && (takenBits >> level & 1u) != 0;
        inner.unlock();
        if (took) CacheLockFor(R).unlock();
    }

private:
    // Written and read only by the thread holding `inner` (a recursive Inner nests its holds on that one thread).
    void push(bool take) {
        if (depth >= 64) NestedTooDeep(R);
        if (take) takenBits |= std::uint64_t{1} << depth;
        else takenBits &= ~(std::uint64_t{1} << depth);
        ++depth;
        if (take) CountSplitTake();
    }
    Inner inner;
    std::uint64_t takenBits = 0;
    std::uint32_t depth = 0;
};

}

#endif
