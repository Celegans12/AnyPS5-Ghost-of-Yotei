#include "prx/libSceAgcDriver/Execution/include/LockOrder.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/RuntimeSwitches.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>

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
// a cache lock's exclusive and shared depths and the mode it really holds, and the SplitMutex taken-bit stack.
struct ThreadLocks {
    std::array<std::uint32_t, RankCount> held{};
    std::array<std::uint32_t, RankCount> exclusive{};
    std::array<std::uint32_t, RankCount> shared{};
    std::array<std::uint8_t, RankCount> real{};
    std::array<std::uint64_t, RankCount> takenBits{};
    std::array<std::uint8_t, RankCount> takenDepth{};
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

void RecordReleased() {
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
    if (!mutex.try_lock()) {
        contended.fetch_add(1, std::memory_order_relaxed);
        mutex.lock();
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
    if (!mutex.try_lock()) return false;
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
    mutex.unlock();
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
    if (!mutex.try_lock_shared()) {
        contended.fetch_add(1, std::memory_order_relaxed);
        mutex.lock_shared();
    }
    locks.real[i] = RealShared;
    ++locks.shared[i];
    ++locks.held[i];
}

bool CacheLock::try_lock_shared() {
    auto& locks = Thread();
    const auto i = Index(rank);
    if (locks.real[i] == RealNone && !mutex.try_lock_shared()) return false;
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
    if (real == RealExclusive) mutex.unlock();
    else mutex.unlock_shared();
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

bool SplitTakes() {
    return Graphics::RuntimeSwitches::Set(Graphics::RuntimeSwitch::V3eLockSplit) && GuestMemory::GpuMutex().HeldByThisThread();
}

void PushSplitTaken(Rank rank, bool taken) {
    auto& locks = Thread();
    const auto i = Index(rank);
    const auto depth = locks.takenDepth[i];
    if (depth >= 64) {
        std::fprintf(stderr, "[lock-order] SplitMutex %s nested deeper than 64\n", Name(rank));
        std::abort();
    }
    if (taken) {
        locks.takenBits[i] |= 1ull << depth;
        splitTakes.fetch_add(1, std::memory_order_relaxed);
    } else {
        locks.takenBits[i] &= ~(1ull << depth);
    }
    locks.takenDepth[i] = static_cast<std::uint8_t>(depth + 1);
}

bool PopSplitTaken(Rank rank) {
    auto& locks = Thread();
    const auto i = Index(rank);
    if (locks.takenDepth[i] == 0) return false;
    const auto depth = --locks.takenDepth[i];
    return (locks.takenBits[i] >> depth & 1u) != 0;
}

}
