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

struct alignas(16) Block {
    std::atomic<std::uint64_t> dirty{0};
    std::atomic<std::uint32_t> armed{0};
    std::atomic<std::uint32_t> lock{0};
};

struct Leaf {
    Block blocks[BlocksPerLeaf];
};

std::atomic<Leaf*> leaves[LeafCount];
std::atomic<std::uint64_t> clock{1};
std::atomic<int> forced{-1};

struct Counters {
    std::atomic<std::uint64_t> arms{0}, armedPages{0}, protectCalls{0}, armNs{0}, refused{0}, faults{0}, faultNs{0}, marks{0}, markedBlocks{0}, cleanQueries{0}, cleanAnswers{0};
};
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

void lockBlock(Block& block) {
    std::uint32_t expected = 0;
    const auto self = threadToken();
    while (!block.lock.compare_exchange_weak(expected, self, std::memory_order_acquire)) {
        expected = 0;
#ifdef _WIN32
        YieldProcessor();
#endif
    }
}

void unlockBlock(Block& block) {
    block.lock.store(0, std::memory_order_release);
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
    block.dirty.store(clock.fetch_add(1, std::memory_order_seq_cst) + 1, std::memory_order_seq_cst);
    const auto mask = block.armed.exchange(0, std::memory_order_seq_cst);
    if (mask != 0) protectMask(blockBase, mask, PAGE_READWRITE);
}

bool eligibleRun(std::uintptr_t begin, std::uintptr_t end) {
    for (auto cursor = begin; cursor < end;) {
        MEMORY_BASIC_INFORMATION memory{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &memory, sizeof(memory)) != sizeof(memory)) return false;
        if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE || memory.Protect != PAGE_READWRITE) return false;
        cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
    }
    return true;
}

bool eligibleMask(std::uintptr_t blockBase, std::uint32_t mask) {
    std::uint32_t protection = 0;
    if (WindowsMappings::Get().Protection(blockBase, &protection)) return false;
    std::uintptr_t start = 0, stop = 0;
    if (WindowsMappings::Get().ResidentSpan(blockBase, BlockBytes, &start, &stop)) return false;
    while (mask != 0) {
        const auto first = static_cast<unsigned>(__builtin_ctz(mask));
        auto last = first;
        while (last + 1 < PagesPerBlock && (mask & (1u << (last + 1))) != 0) ++last;
        if (!eligibleRun(blockBase + first * PageBytes, blockBase + (last + 1) * PageBytes)) return false;
        mask &= ~static_cast<std::uint32_t>(((std::uint64_t{1} << (last + 1)) - 1) & ~((std::uint64_t{1} << first) - 1));
    }
    return true;
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
    if (!enabled() || bytes == 0 || address + bytes < address) return 0;
    const auto start = nowNs();
    const auto begin = address & ~(PageBytes - 1);
    const auto end = (address + bytes + PageBytes - 1) & ~(PageBytes - 1);
    counters.arms.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(armMutex);
    const auto refuse = [&] {
        counters.refused.fetch_add(1, std::memory_order_relaxed);
        counters.armNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
        return std::uint64_t{0};
    };
    if (overlapsFlight(begin, end)) return refuse();
    for (auto blockBase = begin & ~(BlockBytes - 1); blockBase < end; blockBase += BlockBytes) {
        const auto want = pageMask(blockBase, begin, end);
        auto* existing = findBlock(blockBase);
        if (existing != nullptr && (existing->armed.load(std::memory_order_acquire) & want) == want) continue;
        auto* block = makeBlock(blockBase);
        if (block == nullptr) return refuse();
        if (!eligibleMask(blockBase, want & ~block->armed.load(std::memory_order_acquire))) return refuse();
        lockBlock(*block);
        const auto add = want & ~block->armed.load(std::memory_order_acquire);
        block->armed.fetch_or(add, std::memory_order_seq_cst);
        protectMask(blockBase, add, PAGE_READONLY);
        unlockBlock(*block);
        counters.armedPages.fetch_add(static_cast<std::uint64_t>(__builtin_popcount(add)), std::memory_order_relaxed);
    }
    const auto token = clock.fetch_add(1, std::memory_order_seq_cst) + 1;
    counters.armNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
    return token;
#else
    static_cast<void>(address);
    static_cast<void>(bytes);
    return 0;
#endif
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
        if ((block->armed.load(std::memory_order_seq_cst) & want) != want) return false;
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
        if (block->armed.load(std::memory_order_acquire) == 0) continue;
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

bool GuestDirtyHandleWrite_nid_postfix(std::uintptr_t address) {
#ifdef _WIN32
    if (!enabled()) return false;
    const auto blockBase = address & ~(BlockBytes - 1);
    auto* block = findBlock(blockBase);
    if (block == nullptr) return false;
    const auto bit = 1u << ((address - blockBase) / PageBytes);
    if ((block->armed.load(std::memory_order_seq_cst) & bit) == 0) return false;
    const auto start = nowNs();
    if (block->lock.load(std::memory_order_acquire) == threadToken()) {
        disarmLocked(blockBase, *block);
    } else {
        lockBlock(*block);
        disarmLocked(blockBase, *block);
        unlockBlock(*block);
    }
    counters.faults.fetch_add(1, std::memory_order_relaxed);
    counters.faultNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
    return true;
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

void GuestDirtyBeginProtection_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    {
        std::lock_guard lock(flightMutex);
        inFlight.emplace_back(address, address + bytes);
    }
    GuestDirtyMark_nid_postfix(address, bytes);
}

void GuestDirtyEndProtection_nid_postfix(std::uintptr_t address, std::size_t bytes) {
    if (!enabled() || bytes == 0) return;
    std::lock_guard lock(flightMutex);
    const auto found = std::find(inFlight.begin(), inFlight.end(), std::make_pair(address, address + bytes));
    if (found != inFlight.end()) inFlight.erase(found);
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
    out->cleanQueries = take(counters.cleanQueries);
    out->cleanAnswers = take(counters.cleanAnswers);
}

}

}
