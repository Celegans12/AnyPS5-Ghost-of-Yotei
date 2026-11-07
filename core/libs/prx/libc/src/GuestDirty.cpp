#include "prx/libc/include/GuestDirty.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#include "prx/libc/include/WindowsMappings.hpp"
#include <windows.h>
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

namespace GuestArena {

namespace {

constexpr std::uintptr_t PageBytes = 4096;
constexpr std::uintptr_t BlockBytes = 65536;
constexpr std::uintptr_t PagesPerBlock = BlockBytes / PageBytes;
constexpr unsigned LeafShift = 30;
constexpr std::uintptr_t BlocksPerLeaf = (std::uintptr_t{1} << LeafShift) / BlockBytes;
constexpr std::uintptr_t LeafCount = std::uintptr_t{1} << (47 - LeafShift);

struct alignas(32) Block {
    std::atomic<std::uint64_t> dirty{0};
    std::atomic<std::uint64_t> refusedEpoch{0};
    std::atomic<std::uint64_t> foreign{0};
    std::atomic<std::uint64_t> opened{0};
    std::atomic<std::uint32_t> armed{0};
    std::atomic<std::uint32_t> viewed{0};
    std::atomic<std::uint32_t> lock{0};
};

std::uint32_t tracked(const Block& block) {
    return block.armed.load(std::memory_order_seq_cst) | block.viewed.load(std::memory_order_seq_cst);
}

struct Leaf {
    Block blocks[BlocksPerLeaf];
};

std::atomic<Leaf*> leaves[LeafCount];
std::atomic<std::uint64_t> clock{1};
std::atomic<std::uint64_t> mappingEpoch{1};
std::atomic<int> forced{-1};

struct Counters {
    std::atomic<std::uint64_t> arms{0}, armedPages{0}, protectCalls{0}, armNs{0}, refused{0}, faults{0}, faultNs{0}, marks{0}, markedBlocks{0}, bumps{0}, viewLosses{0}, viewArms{0}, cleanQueries{0}, cleanAnswers{0}, importArmedPages{0}, importMarks{0}, renewRefused{0}, lateFaults{0}, declined{0}, retriedFaults{0}, armWaits{0}, armWaitNs{0};
    std::atomic<std::uint64_t> refusedBy[4][8]{};
    std::atomic<std::uint64_t> armsBy[4]{};
};
thread_local unsigned armKind = 0;
thread_local unsigned lastRefusal = 0;
std::atomic<std::uintptr_t> lastDeclined{0};
std::atomic<std::uint64_t> lastDeclinedState{0};
struct OwnWindow {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    std::uint32_t depth = 0;
};
thread_local OwnWindow ownWindow;

bool ownedHere(std::uintptr_t blockBase) {
    return ownWindow.depth != 0 && blockBase < ownWindow.end && blockBase + BlockBytes > ownWindow.begin;
}

void noteWrite(Block& block, std::uintptr_t blockBase, std::uint64_t generation, bool forceForeign) {
    if (forceForeign || !ownedHere(blockBase)) block.foreign.store(generation, std::memory_order_seq_cst);
}

thread_local bool foreignMark = false;
Counters counters;

std::mutex armMutex;
std::mutex flightMutex;
std::vector<std::pair<std::uintptr_t, std::uintptr_t>> inFlight;

std::uint64_t nowNs() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool enabled() {
    const auto force = forced.load(std::memory_order_acquire);
    if (force >= 0) return force != 0;
    static const bool fromEnvironment = std::getenv("APS5_V3M_DIRTY") != nullptr;
    return fromEnvironment;
}

Block* findBlock(std::uintptr_t address) {
    if ((address >> 47) != 0) return nullptr;
    auto* leaf = leaves[address >> LeafShift].load(std::memory_order_acquire);
    if (leaf == nullptr) return nullptr;
    return &leaf->blocks[(address & ((std::uintptr_t{1} << LeafShift) - 1)) / BlockBytes];
}

Block* makeBlock(std::uintptr_t address) {
    if ((address >> 47) != 0) return nullptr;
    auto& slot = leaves[address >> LeafShift];
    auto* leaf = slot.load(std::memory_order_acquire);
    if (leaf == nullptr) {
        auto* made = new Leaf;
        if (slot.compare_exchange_strong(leaf, made, std::memory_order_acq_rel)) leaf = made;
        else delete made;
    }
    return &leaf->blocks[(address & ((std::uintptr_t{1} << LeafShift) - 1)) / BlockBytes];
}

std::uint32_t threadToken() {
#ifdef _WIN32
    return static_cast<std::uint32_t>(GetCurrentThreadId()) | 0x80000000u;
#else
    return 1;
#endif
}

thread_local unsigned blockLocksHeld = 0;
thread_local bool inFaultHandler = false;

void lockBlock(Block& block) {
    GuestArena::GuestArenaNoteLock_nid_postfix(GuestArena::GuestLockDirtyBlock, blockLocksHeld == 0 && !inFaultHandler ? GuestArena::GuestLockBefore : GuestArena::GuestLockHeldUnchecked);
    std::uint32_t expected = 0;
    const auto self = threadToken();
    while (!block.lock.compare_exchange_weak(expected, self, std::memory_order_acquire)) {
        expected = 0;
#ifdef _WIN32
        YieldProcessor();
#endif
    }
    if (blockLocksHeld++ == 0 && !inFaultHandler) GuestArena::GuestArenaNoteLock_nid_postfix(GuestArena::GuestLockDirtyBlock, GuestArena::GuestLockHeld);
}

void unlockBlock(Block& block) {
    block.lock.store(0, std::memory_order_release);
    if (blockLocksHeld != 0) --blockLocksHeld;
    GuestArena::GuestArenaNoteLock_nid_postfix(GuestArena::GuestLockDirtyBlock, GuestArena::GuestLockReleased);
}

std::uint32_t pageMask(std::uintptr_t blockBase, std::uintptr_t begin, std::uintptr_t end) {
    const auto from = std::max(begin, blockBase);
    const auto to = std::min(end, blockBase + BlockBytes);
    if (from >= to) return 0;
    const auto first = (from - blockBase) / PageBytes;
    const auto last = (to - blockBase - 1) / PageBytes;
    return static_cast<std::uint32_t>(((std::uint64_t{1} << (last + 1)) - 1) & ~((std::uint64_t{1} << first) - 1));
}

#ifdef _WIN32
void protectMask(std::uintptr_t blockBase, std::uint32_t mask, DWORD protection) {
    while (mask != 0) {
        const auto first = static_cast<unsigned>(__builtin_ctz(mask));
        auto last = first;
        while (last + 1 < PagesPerBlock && (mask & (1u << (last + 1))) != 0) ++last;
        DWORD previous = 0;
        VirtualProtect(reinterpret_cast<void*>(blockBase + first * PageBytes), (last - first + 1) * PageBytes, protection, &previous);
        counters.protectCalls.fetch_add(1, std::memory_order_relaxed);
        mask &= ~static_cast<std::uint32_t>(((std::uint64_t{1} << (last + 1)) - 1) & ~((std::uint64_t{1} << first) - 1));
    }
}

void disarmLocked(std::uintptr_t blockBase, Block& block) {
    const auto generation = clock.fetch_add(1, std::memory_order_seq_cst) + 1;
    block.dirty.store(generation, std::memory_order_seq_cst);
    block.opened.store(generation, std::memory_order_seq_cst);
    noteWrite(block, blockBase, generation, foreignMark);
    block.viewed.store(0, std::memory_order_seq_cst);
    const auto mask = block.armed.load(std::memory_order_seq_cst);
    if (mask == 0) return;
    protectMask(blockBase, mask, PAGE_READWRITE);
    block.armed.fetch_and(~mask, std::memory_order_seq_cst);
}

void viewLost(std::uintptr_t address, std::size_t bytes) {
    const auto end = address + bytes;
    for (auto blockBase = address & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        auto* block = findBlock(blockBase);
        if (block == nullptr) continue;
        const auto mask = pageMask(blockBase, address & ~(PageBytes - 1), (end + PageBytes - 1) & ~(PageBytes - 1));
        const auto generation = clock.fetch_add(1, std::memory_order_seq_cst) + 1;
        noteWrite(*block, blockBase, generation, false);
        if ((tracked(*block) & mask) == 0) continue;
        block->dirty.store(generation, std::memory_order_seq_cst);
        block->opened.store(generation, std::memory_order_seq_cst);
        block->viewed.fetch_and(~mask, std::memory_order_seq_cst);
        block->armed.fetch_and(~mask, std::memory_order_seq_cst);
        counters.viewLosses.fetch_add(1, std::memory_order_relaxed);
    }
}

unsigned classify(std::uintptr_t begin, std::uintptr_t end, std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& privatePieces, std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& viewPieces) {
    for (auto cursor = begin; cursor < end;) {
        MEMORY_BASIC_INFORMATION memory{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &memory, sizeof(memory)) != sizeof(memory)) return GuestArena::RefusedUncommitted + 1;
        const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
        if (memory.State != MEM_COMMIT) return GuestArena::RefusedUncommitted + 1;
        if (memory.Type == MEM_PRIVATE && memory.Protect == PAGE_READWRITE) privatePieces.emplace_back(cursor, stop);
        else if (memory.Type == MEM_MAPPED && (memory.Protect == PAGE_READWRITE || memory.Protect == PAGE_READONLY || memory.Protect == PAGE_EXECUTE_READWRITE || memory.Protect == PAGE_EXECUTE_READ)) viewPieces.emplace_back(cursor, stop);
        else return (memory.Type == MEM_PRIVATE ? GuestArena::RefusedPrivateNotWritable : GuestArena::RefusedOtherMemory) + 1;
        cursor = stop;
    }
    return 0;
}


#endif

bool overlapsFlight(std::uintptr_t begin, std::uintptr_t end) {
    std::lock_guard lock(flightMutex);
    return std::any_of(inFlight.begin(), inFlight.end(), [&](const auto& range) { return range.first < end && begin < range.second; });
}

}

extern "C" {

bool GuestDirtyEnabled_nid_postfix() {
#ifdef _WIN32
    return enabled();
#else
    return false;
#endif
}

void GuestDirtyForceEnable_nid_no_patch(bool on) {
    forced.store(on ? 1 : 0, std::memory_order_release);
}

std::uint64_t GuestDirtyArm_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#ifdef _WIN32
    if (!enabled() || bytes == 0 || address + bytes < address) {
        lastRefusal = GuestArena::RefusedDisabled + 1;
        return 0;
    }
    lastRefusal = 0;
    const auto start = nowNs();
    const auto begin = address & ~(PageBytes - 1);
    const auto end = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    counters.arms.fetch_add(1, std::memory_order_relaxed);
    counters.armsBy[armKind].fetch_add(1, std::memory_order_relaxed);
    const auto epoch = mappingEpoch.load(std::memory_order_acquire);
    bool allArmed = true;
    for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        const auto* block = findBlock(blockBase);
        if (block != nullptr && block->refusedEpoch.load(std::memory_order_acquire) == epoch) {
            counters.refused.fetch_add(1, std::memory_order_relaxed);
            counters.refusedBy[armKind][GuestArena::RefusedEpoch].fetch_add(1, std::memory_order_relaxed);
            lastRefusal = GuestArena::RefusedEpoch + 1;
            counters.armNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
            return 0;
        }
        const auto want = pageMask(blockBase, begin, end);
        if (block == nullptr || (tracked(*block) & want) != want) {
            allArmed = false;
            break;
        }
    }
    if (allArmed && !overlapsFlight(begin, end)) {
        const auto token = clock.fetch_add(1, std::memory_order_seq_cst) + 1;
        counters.armNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
        return token;
    }
    GuestArena::GuestArenaNoteLock_nid_postfix(GuestArena::GuestLockDirtyArm, GuestArena::GuestLockBefore);
    std::unique_lock lock(armMutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        const auto waitStart = nowNs();
        lock.lock();
        counters.armWaits.fetch_add(1, std::memory_order_relaxed);
        counters.armWaitNs.fetch_add(nowNs() - waitStart, std::memory_order_relaxed);
    }
    GuestArena::GuestArenaNoteLock_nid_postfix(GuestArena::GuestLockDirtyArm, GuestArena::GuestLockHeld);
    struct ArmReleased {
        ~ArmReleased() { GuestArena::GuestArenaNoteLock_nid_postfix(GuestArena::GuestLockDirtyArm, GuestArena::GuestLockReleased); }
    } armReleased;
    const auto refuse = [&](unsigned reason) {
        counters.refused.fetch_add(1, std::memory_order_relaxed);
        counters.refusedBy[armKind][reason & 7u].fetch_add(1, std::memory_order_relaxed);
        lastRefusal = (reason & 7u) + 1;
        counters.armNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
        return std::uint64_t{0};
    };
    if (overlapsFlight(begin, end)) return refuse(GuestArena::RefusedInFlight);
    struct Held {
        Block* block;
        std::uintptr_t base;
        std::uint32_t add;
    };
    thread_local std::vector<Held> held;
    thread_local std::vector<std::pair<std::uintptr_t, std::uintptr_t>> runs;
    held.clear();
    runs.clear();
    const auto release = [&] {
        for (auto& entry : held) unlockBlock(*entry.block);
        held.clear();
    };
    if (WindowsMappings::Get().AccessGuardsUsed()) return refuse(GuestArena::RefusedGuards);
    for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        const auto want = pageMask(blockBase, begin, end);
        auto* existing = findBlock(blockBase);
        if (existing != nullptr && (tracked(*existing) & want) == want) continue;
        auto* block = makeBlock(blockBase);
        if (block == nullptr) {
            release();
            return refuse(GuestArena::RefusedBlock);
        }
        lockBlock(*block);
        const auto add = want & ~tracked(*block);
        if (add == 0) {
            unlockBlock(*block);
            continue;
        }
        held.push_back({block, blockBase, add});
        auto mask = add;
        while (mask != 0) {
            const auto first = static_cast<unsigned>(__builtin_ctz(mask));
            auto last = first;
            while (last + 1 < PagesPerBlock && (mask & (1u << (last + 1))) != 0) ++last;
            const auto from = blockBase + first * PageBytes;
            const auto to = blockBase + (last + 1) * PageBytes;
            if (!runs.empty() && runs.back().second == from) runs.back().second = to;
            else runs.emplace_back(from, to);
            mask &= ~static_cast<std::uint32_t>(((std::uint64_t{1} << (last + 1)) - 1) & ~((std::uint64_t{1} << first) - 1));
        }
    }
    thread_local std::vector<std::pair<std::uintptr_t, std::uintptr_t>> privatePieces;
    thread_local std::vector<std::pair<std::uintptr_t, std::uintptr_t>> viewPieces;
    privatePieces.clear();
    viewPieces.clear();
    const auto refuseRange = [&](std::uintptr_t from, std::uintptr_t to, unsigned reason) {
        for (auto& entry : held) {
            if (entry.base + BlockBytes > from && entry.base < to) entry.block->refusedEpoch.store(epoch, std::memory_order_release);
        }
        release();
        return refuse(reason);
    };
    for (const auto& [from, to] : runs) {
        if (const auto reason = classify(from, to, privatePieces, viewPieces); reason != 0) return refuseRange(from, to, reason - 1);
    }
    if (overlapsFlight(begin, end)) return refuseRange(begin, end, GuestArena::RefusedInFlight);
    if (!viewPieces.empty()) {
        WindowsMappings::trackLostHook.store(&viewLost, std::memory_order_release);
        for (const auto& [from, to] : viewPieces) {
            if (!WindowsMappings::Get().ArmTracked(from, to - from)) return refuseRange(from, to, GuestArena::RefusedView);
        }
    }
    std::uint64_t pages = 0;
    const auto inPieces = [](const std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& pieces, std::uintptr_t blockBase, std::uint32_t add) {
        std::uint32_t mask = 0;
        for (const auto& [from, to] : pieces) mask |= pageMask(blockBase, from, to);
        return mask & add;
    };
    for (auto& entry : held) {
        const auto viewMask = inPieces(viewPieces, entry.base, entry.add);
        const auto privateMask = entry.add & ~viewMask;
        if (viewMask != 0) entry.block->viewed.fetch_or(viewMask, std::memory_order_seq_cst);
        if (privateMask != 0) entry.block->armed.fetch_or(privateMask, std::memory_order_seq_cst);
        pages += static_cast<std::uint64_t>(__builtin_popcount(entry.add));
    }
    for (const auto& [from, to] : privatePieces) {
        DWORD previous = 0;
        VirtualProtect(reinterpret_cast<void*>(from), to - from, PAGE_READONLY, &previous);
        counters.protectCalls.fetch_add(1, std::memory_order_relaxed);
    }
    counters.viewArms.fetch_add(viewPieces.size(), std::memory_order_relaxed);
    release();
    counters.armedPages.fetch_add(pages, std::memory_order_relaxed);
    const auto token = clock.fetch_add(1, std::memory_order_seq_cst) + 1;
    counters.armNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
    return token;
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return 0;
#endif
}

unsigned GuestDirtyTakeLastRefusal_nid_postfix() {
    return std::exchange(lastRefusal, 0u);
}

bool GuestDirtyCleanSince_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t token) {
    if (token == 0 || bytes == 0 || !enabled()) return false;
    counters.cleanQueries.fetch_add(1, std::memory_order_relaxed);
    const auto begin = address & ~(PageBytes - 1);
    const auto end = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        const auto* block = findBlock(blockBase);
        if (block == nullptr) return false;
        const auto want = pageMask(blockBase, begin, end);
        if ((tracked(*block) & want) != want) return false;
        if (block->dirty.load(std::memory_order_seq_cst) >= token) return false;
    }
    counters.cleanAnswers.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool GuestDirtyCleanSinceAll_nid_postfix(const GuestDirtyQuery* queries, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        if (!GuestDirtyCleanSince_nid_postfix(queries[index].address, queries[index].bytes, queries[index].token)) return false;
    }
    return count != 0;
}

void GuestDirtyMark_nid_postfix(std::uintptr_t address, std::size_t bytes) {
#ifdef _WIN32
    if (bytes == 0 || !enabled()) return;
    const auto begin = address & ~(PageBytes - 1);
    const auto end = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    counters.marks.fetch_add(1, std::memory_order_relaxed);
    for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        auto* block = findBlock(blockBase);
        if (block == nullptr) {
            blockBase = ((blockBase >> LeafShift) + 1) << LeafShift;
            blockBase -= BlockBytes;
            continue;
        }
        if (tracked(*block) == 0) {
            noteWrite(*block, blockBase, clock.fetch_add(1, std::memory_order_seq_cst) + 1, foreignMark);
            continue;
        }
        lockBlock(*block);
        disarmLocked(blockBase, *block);
        unlockBlock(*block);
        counters.markedBlocks.fetch_add(1, std::memory_order_relaxed);
    }
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
#endif
}

void GuestDirtyBump_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (bytes == 0 || !enabled()) return;
    const auto begin = address & ~(PageBytes - 1);
    const auto end = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        auto* block = findBlock(blockBase);
        if (block == nullptr) {
            blockBase = (((blockBase >> LeafShift) + 1) << LeafShift) - BlockBytes;
            continue;
        }
        const auto generation = clock.fetch_add(1, std::memory_order_seq_cst) + 1;
        noteWrite(*block, blockBase, generation, false);
        if (tracked(*block) == 0) continue;
        block->dirty.store(generation, std::memory_order_seq_cst);
        counters.bumps.fetch_add(1, std::memory_order_relaxed);
    }
}

bool GuestDirtyHandleWrite_nid_postfix(std::uintptr_t address) {
#ifdef _WIN32
    if (!enabled()) return false;
    struct FaultScope {
        bool outer = std::exchange(inFaultHandler, true);
        ~FaultScope() { inFaultHandler = outer; }
    } faultScope;
    const auto blockBase = address & ~(BlockBytes - 1);
    auto* block = findBlock(blockBase);
    if (block == nullptr) return false;
    const auto bit = 1u << ((address - blockBase) / PageBytes);
    const auto armed = [&] { return (block->armed.load(std::memory_order_seq_cst) & bit) != 0; };
    DWORD type = 0;
    const auto writable = [&](DWORD* protection) {
        MEMORY_BASIC_INFORMATION memory{};
        const bool queried = VirtualQuery(reinterpret_cast<const void*>(address), &memory, sizeof(memory)) == sizeof(memory);
        *protection = queried ? memory.Protect : 0;
        type = queried ? memory.Type : 0;
        return queried && memory.State == MEM_COMMIT && (memory.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE)) != 0 && (memory.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0;
    };
    const auto start = nowNs();
    DWORD protection = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        if (armed()) {
            const bool own = block->lock.load(std::memory_order_acquire) == threadToken();
            if (!own) lockBlock(*block);
            const bool still = armed();
            if (still) disarmLocked(blockBase, *block);
            if (!own) unlockBlock(*block);
            if (!still) continue;
            counters.faults.fetch_add(1, std::memory_order_relaxed);
            counters.faultNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
            return true;
        }
        if (writable(&protection)) {
            if (block->opened.load(std::memory_order_relaxed) != 0) counters.lateFaults.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        if (!armed()) {
            if (writable(&protection)) {
                counters.lateFaults.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            if (!armed()) break;
        }
    }
    thread_local std::uintptr_t retriedAddress = 0;
    thread_local unsigned retries = 0;
    if (tracked(*block) != 0 || block->opened.load(std::memory_order_relaxed) != 0) {
        if (retriedAddress != address) {
            retriedAddress = address;
            retries = 0;
        }
        if (type == MEM_PRIVATE && ++retries <= 64) {
            counters.retriedFaults.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        retries = 0;
        counters.declined.fetch_add(1, std::memory_order_relaxed);
        lastDeclined.store(address, std::memory_order_relaxed);
        lastDeclinedState.store((static_cast<std::uint64_t>(block->armed.load()) << 32) | (static_cast<std::uint64_t>(protection & 0xffff) << 16) | (block->viewed.load() & 0xffff), std::memory_order_relaxed);
    }
    return false;
#else
    static_cast<void>(address);
    return false;
#endif
}

bool GuestDirtyArmed_nid_postfix(std::uintptr_t address) {
    if (!enabled()) return false;
    const auto blockBase = address & ~(BlockBytes - 1);
    const auto* block = findBlock(blockBase);
    return block != nullptr && (block->armed.load(std::memory_order_acquire) & (1u << ((address - blockBase) / PageBytes))) != 0;
}

static void markForeign(std::uintptr_t address, std::size_t bytes) {
    const bool outer = std::exchange(foreignMark, true);
    GuestDirtyMark_nid_postfix(address, bytes);
    foreignMark = outer;
}

std::uint64_t GuestDirtyOpenWindow_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return 0;
    if (ownWindow.depth++ == 0) {
        ownWindow.begin = address;
        ownWindow.end = address + bytes;
    }
    return clock.load(std::memory_order_seq_cst);
}

void GuestDirtyCloseWindow_nid_postfix() {
    if (ownWindow.depth != 0) --ownWindow.depth;
}

std::uint64_t GuestDirtyRenew_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t window) {
    if (!enabled() || bytes == 0 || window == 0) return 0;
    const auto begin = address & ~(PageBytes - 1);
    const auto end = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    const auto quiet = [&] {
        for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
            const auto* block = findBlock(blockBase);
            if (block == nullptr || block->foreign.load(std::memory_order_seq_cst) > window || block->opened.load(std::memory_order_seq_cst) > window) return false;
        }
        return true;
    };
    if (!quiet()) {
        counters.renewRefused.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const auto token = GuestDirtyArm_nid_postfix(address, bytes);
    if (token == 0) return 0;
    if (!quiet()) {
        counters.renewRefused.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    return token;
}

void GuestDirtyMappingChange_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    mappingEpoch.fetch_add(1, std::memory_order_acq_rel);
    markForeign(address, bytes);
}

void GuestDirtyBeginHostWrite_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    {
        std::lock_guard lock(flightMutex);
        inFlight.emplace_back(address, address + bytes);
    }
    markForeign(address, bytes);
}

void GuestDirtyBeginProtection_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    mappingEpoch.fetch_add(1, std::memory_order_acq_rel);
    {
        std::lock_guard lock(flightMutex);
        inFlight.emplace_back(address, address + bytes);
    }
    markForeign(address, bytes);
}

void GuestDirtyEndProtection_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    std::lock_guard lock(flightMutex);
    const auto found = std::find(inFlight.begin(), inFlight.end(), std::make_pair(address, address + bytes));
    if (found != inFlight.end()) inFlight.erase(found);
}

void GuestDirtyBeginHostImport_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    counters.importMarks.fetch_add(1, std::memory_order_relaxed);
    GuestDirtyBeginHostWrite_nid_postfix(address, bytes);
}

unsigned GuestDirtySetArmKind_nid_postfix(unsigned kind) {
    return std::exchange(armKind, kind & 3u);
}

void GuestDirtyNoteImportArmed_nid_postfix(std::uintptr_t) {
    counters.importArmedPages.fetch_add(1, std::memory_order_relaxed);
}

void GuestDirtyRunLocked_nid_no_patch(std::uintptr_t address, void (*body)(void*), void* argument) {
    auto* block = makeBlock(address & ~(BlockBytes - 1));
    if (block == nullptr) return;
    lockBlock(*block);
    body(argument);
    unlockBlock(*block);
}

void GuestDirtyTakeCounts_nid_postfix(GuestDirtyCounts* out) {
    if (out == nullptr) return;
    const auto take = [](std::atomic<std::uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
    out->arms = take(counters.arms);
    out->armedPages = take(counters.armedPages);
    out->protectCalls = take(counters.protectCalls);
    out->armNs = take(counters.armNs);
    out->refused = take(counters.refused);
    out->faults = take(counters.faults);
    out->faultNs = take(counters.faultNs);
    out->marks = take(counters.marks);
    out->markedBlocks = take(counters.markedBlocks);
    out->bumps = take(counters.bumps);
    out->viewLosses = take(counters.viewLosses);
    out->viewArms = take(counters.viewArms);
    out->cleanQueries = take(counters.cleanQueries);
    out->cleanAnswers = take(counters.cleanAnswers);
    out->importArmedPages = take(counters.importArmedPages);
    out->importMarks = take(counters.importMarks);
    out->renewRefused = take(counters.renewRefused);
    out->lateFaults = take(counters.lateFaults);
    out->declined = take(counters.declined);
    out->retriedFaults = take(counters.retriedFaults);
    out->armWaits = take(counters.armWaits);
    out->armWaitNs = take(counters.armWaitNs);
    out->lastDeclined = lastDeclined.load(std::memory_order_relaxed);
    out->lastDeclinedState = lastDeclinedState.load(std::memory_order_relaxed);
    for (std::size_t k = 0; k < 4; ++k) {
        out->armsBy[k] = take(counters.armsBy[k]);
        for (std::size_t i = 0; i < 8; ++i) out->refusedBy[k][i] = take(counters.refusedBy[k][i]);
    }
}

}

}
