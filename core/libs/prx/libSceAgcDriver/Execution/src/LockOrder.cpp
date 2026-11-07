#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif

namespace AgcDriver::LockOrder {

namespace {

constexpr std::array<const char*, RankCount> Names{
    "record", "texture", "storage", "resource-set", "pipeline", "depth/DCC", "imports", "residency",
    "tracker", "dirty-arm", "dirty-block", "windows-mappings",
};

std::atomic<int> checkState{-1};
std::atomic<std::uint64_t> violations{0};
std::atomic<std::uint64_t> releasesHolding{0};
std::atomic<std::uint64_t> contended{0};
std::atomic<std::uint64_t> splitTakes{0};

// Per thread and rank: held depth (annotations and cache locks; the record lock is GpuMutex's own owner token),
// a cache lock's exclusive and shared depths and the mode it really holds (SplitMutex keeps its taken bits itself).
struct ThreadLocks {
    std::array<std::uint32_t, RankCount> held{};
    std::array<std::uint32_t, RankCount> exclusive{};
    std::array<std::uint32_t, RankCount> shared{};
    std::array<std::uint8_t, RankCount> real{};
};

ThreadLocks& Thread() {
    thread_local ThreadLocks locks;
    return locks;
}

constexpr std::uint8_t RealNone = 0;
constexpr std::uint8_t RealShared = 1;
constexpr std::uint8_t RealExclusive = 2;

std::size_t Index(Rank rank) {
    return static_cast<std::size_t>(rank);
}

[[noreturn]] void Violate(const std::string& text) {
    violations.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "[lock-order] VIOLATION: %s\n", text.c_str());
    throw OrderError(text);
}

}

const char* Name(Rank rank) {
    return Index(rank) < RankCount ? Names[Index(rank)] : "?";
}

bool CheckEnabled() {
    int state = checkState.load(std::memory_order_relaxed);
    if (state < 0) {
        const char* value = std::getenv("APS5_V3E_LOCK_ORDER_CHECK");
        state = value != nullptr && value[0] != '\0' && value[0] != '0' ? 1 : 0;
        checkState.store(state, std::memory_order_relaxed);
    }
    return state == 1;
}

void SetCheckForTest(bool enabled) {
    checkState.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

std::string HeldText() {
    const auto& locks = Thread();
    std::string text;
    if (GuestMemory::GpuMutex().HeldByThisThread()) text = "record";
    for (std::size_t i = 1; i < RankCount; ++i) {
        if (locks.held[i] == 0) continue;
        if (!text.empty()) text += ", ";
        text += Names[i];
        if (locks.real[i] == RealShared) text += " (shared)";
    }
    return text.empty() ? "nothing" : text;
}

void BeforeAcquire(Rank rank) {
    if (!CheckEnabled()) return;
    const auto& locks = Thread();
    for (std::size_t i = Index(rank) + 1; i < RankCount; ++i) {
        if (locks.held[i] == 0) continue;
        Violate(std::string("acquiring the ") + Name(rank) + " lock while holding the " + Names[i] + " lock (held: " + HeldText() + ")");
    }
}

namespace {
thread_local StageHold* activeStage = nullptr;
std::atomic<std::uint64_t> stageSuspends{0};
}

void RecordAcquired() {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3eLockSplit)) return;
    for (auto* stage = activeStage; stage != nullptr; stage = stage->previous) stage->resume();
}

void RecordReleased() {
    if (Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3eLockSplit)) {
        for (auto* stage = activeStage; stage != nullptr; stage = stage->previous) stage->suspend();
    }
    if (!CheckEnabled()) return;
    const auto& locks = Thread();
    for (std::size_t i = 1; i < RankCount; ++i) {
        if (locks.held[i] == 0 || !IsCacheRank(static_cast<Rank>(i))) continue;
        releasesHolding.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "[lock-order] VIOLATION: record lock released while holding the %s lock\n", Names[i]);
        return;
    }
}

void NoteHeld(Rank rank) {
    ++Thread().held[Index(rank)];
}

void NoteReleased(Rank rank) {
    auto& held = Thread().held[Index(rank)];
    if (held != 0) --held;
}

std::uint32_t HeldOnThisThread(Rank rank) {
    if (rank == Rank::Record) return GuestMemory::GpuMutex().DepthOnThisThread();
    return Thread().held[Index(rank)];
}

bool HoldsCacheLock() {
    const auto& locks = Thread();
    for (std::size_t i = Index(Rank::Texture); i <= Index(Rank::Residency); ++i) {
        if (locks.held[i] != 0) return true;
    }
    return false;
}

bool MustNotWait() {
    if (GuestMemory::GpuMutex().HeldByThisThread()) return true;
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3eLockSplit) && HoldsCacheLock();
}

Counters Totals() {
    return {violations.load(std::memory_order_relaxed), releasesHolding.load(std::memory_order_relaxed), contended.load(std::memory_order_relaxed), splitTakes.load(std::memory_order_relaxed)};
}

void CacheLock::lock() {
    auto& locks = Thread();
    const auto i = Index(rank);
    if (locks.real[i] == RealExclusive) {
        ++locks.exclusive[i];
        ++locks.held[i];
        return;
    }
    if (locks.real[i] == RealShared) {
        violations.fetch_add(1, std::memory_order_relaxed);
        std::fprintf(stderr, "[lock-order] VIOLATION: exclusive %s lock asked for while holding it shared\n", Name(rank));
        throw OrderError(std::string("exclusive ") + Name(rank) + " lock asked for while holding it shared");
    }
    BeforeAcquire(rank);
    if (!rawTryLock()) {
        contended.fetch_add(1, std::memory_order_relaxed);
        rawLock();
    }
    locks.real[i] = RealExclusive;
    ++locks.exclusive[i];
    ++locks.held[i];
}

bool CacheLock::try_lock() {
    auto& locks = Thread();
    const auto i = Index(rank);
    if (locks.real[i] == RealExclusive) {
        ++locks.exclusive[i];
        ++locks.held[i];
        return true;
    }
    if (locks.real[i] == RealShared) return false;
    if (!rawTryLock()) return false;
    locks.real[i] = RealExclusive;
    ++locks.exclusive[i];
    ++locks.held[i];
    return true;
}

void CacheLock::unlock() {
    auto& locks = Thread();
    const auto i = Index(rank);
    --locks.exclusive[i];
    --locks.held[i];
    if (locks.held[i] != 0) return;
    locks.real[i] = RealNone;
    rawUnlock();
}

void CacheLock::lock_shared() {
    auto& locks = Thread();
    const auto i = Index(rank);
    if (locks.real[i] != RealNone) {
        ++locks.shared[i];
        ++locks.held[i];
        return;
    }
    BeforeAcquire(rank);
    if (!rawTryLockShared()) {
        contended.fetch_add(1, std::memory_order_relaxed);
        rawLockShared();
    }
    locks.real[i] = RealShared;
    ++locks.shared[i];
    ++locks.held[i];
}

bool CacheLock::try_lock_shared() {
    auto& locks = Thread();
    const auto i = Index(rank);
    if (locks.real[i] == RealNone && !rawTryLockShared()) return false;
    if (locks.real[i] == RealNone) locks.real[i] = RealShared;
    ++locks.shared[i];
    ++locks.held[i];
    return true;
}

void CacheLock::unlock_shared() {
    auto& locks = Thread();
    const auto i = Index(rank);
    --locks.shared[i];
    --locks.held[i];
    if (locks.held[i] != 0) return;
    const auto real = locks.real[i];
    locks.real[i] = RealNone;
    if (real == RealExclusive) rawUnlock();
    else rawUnlockShared();
}

bool CacheLock::HeldByThisThread() const {
    return Thread().real[Index(rank)] != RealNone;
}

bool CacheLock::HeldExclusiveByThisThread() const {
    return Thread().real[Index(rank)] == RealExclusive;
}

CacheLock& CacheLockFor(Rank rank) {
    static CacheLock texture{Rank::Texture}, storage{Rank::Storage}, resourceSet{Rank::ResourceSet}, pipeline{Rank::Pipeline}, depthDcc{Rank::DepthDcc}, imports{Rank::Imports}, residency{Rank::Residency};
    switch (rank) {
    case Rank::Texture: return texture;
    case Rank::Storage: return storage;
    case Rank::ResourceSet: return resourceSet;
    case Rank::Pipeline: return pipeline;
    case Rank::DepthDcc: return depthDcc;
    case Rank::Imports: return imports;
    case Rank::Residency: return residency;
    default: break;
    }
    std::fprintf(stderr, "[lock-order] no cache lock of rank %s\n", Name(rank));
    std::abort();
}

namespace {
std::atomic<int> leafState{-1};
bool leafTakes() {
    int state = leafState.load(std::memory_order_relaxed);
    if (state < 0) {
        state = std::getenv("APS5_V3E_LOCK_SPLIT_LEAF") != nullptr ? 1 : 0;
        leafState.store(state, std::memory_order_relaxed);
    }
    return state == 1;
}
}

void SetLeafTakesForTest(bool enabled) {
    leafState.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

bool SplitTakes() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3eLockSplit) && leafTakes() && GuestMemory::GpuMutex().HeldByThisThread();
}

StageHold::StageHold() {
    if (!Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3eLockSplit) || leafTakes() || !GuestMemory::GpuMutex().HeldByThisThread()) return;
    for (auto rank : StageRanks) CacheLockFor(rank).lock();
    held = true;
    previous = activeStage;
    activeStage = this;
    CountSplitTake();
}

StageHold::~StageHold() {
    if (!held) return;
    activeStage = previous;
    if (!suspended) {
        for (auto it = StageRanks.rbegin(); it != StageRanks.rend(); ++it) CacheLockFor(*it).unlock();
    }
}

void StageHold::suspend() {
    if (suspended) return;
    for (auto it = StageRanks.rbegin(); it != StageRanks.rend(); ++it) CacheLockFor(*it).unlock();
    suspended = true;
    stageSuspends.fetch_add(1, std::memory_order_relaxed);
}

void StageHold::resume() {
    if (!suspended) return;
    for (auto rank : StageRanks) CacheLockFor(rank).lock();
    suspended = false;
}

std::uint64_t StageSuspends() {
    return stageSuspends.load(std::memory_order_relaxed);
}

void CountSplitTake() {
    thread_local std::uint32_t batch = 0;
    if (++batch < 1024) return;
    batch = 0;
    const auto takes = splitTakes.fetch_add(1024, std::memory_order_relaxed) + 1024;
    if ((takes & 0xffffu) == 0) std::fprintf(stderr, "[lock-split] %llu cache-lock takes (stage holds, or leaf takes in APS5_V3E_LOCK_SPLIT_LEAF), %llu stage suspends at a record release, %llu contended, %llu violations, %llu record releases holding a cache lock\n", static_cast<unsigned long long>(takes), static_cast<unsigned long long>(stageSuspends.load(std::memory_order_relaxed)),static_cast<unsigned long long>(contended.load(std::memory_order_relaxed)), static_cast<unsigned long long>(violations.load(std::memory_order_relaxed)), static_cast<unsigned long long>(releasesHolding.load(std::memory_order_relaxed)));
}

void NestedTooDeep(Rank rank) {
    std::fprintf(stderr, "[lock-order] SplitMutex %s nested deeper than 64\n", Name(rank));
    std::abort();
}

#ifdef _WIN32
void CacheLock::rawLock() { AcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw)); }
bool CacheLock::rawTryLock() { return TryAcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw)) != 0; }
void CacheLock::rawUnlock() { ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(&srw)); }
void CacheLock::rawLockShared() { AcquireSRWLockShared(reinterpret_cast<PSRWLOCK>(&srw)); }
bool CacheLock::rawTryLockShared() { return TryAcquireSRWLockShared(reinterpret_cast<PSRWLOCK>(&srw)) != 0; }
void CacheLock::rawUnlockShared() { ReleaseSRWLockShared(reinterpret_cast<PSRWLOCK>(&srw)); }
#else
void CacheLock::rawLock() { mutex.lock(); }
bool CacheLock::rawTryLock() { return mutex.try_lock(); }
void CacheLock::rawUnlock() { mutex.unlock(); }
void CacheLock::rawLockShared() { mutex.lock_shared(); }
bool CacheLock::rawTryLockShared() { return mutex.try_lock_shared(); }
void CacheLock::rawUnlockShared() { mutex.unlock_shared(); }
#endif

}
