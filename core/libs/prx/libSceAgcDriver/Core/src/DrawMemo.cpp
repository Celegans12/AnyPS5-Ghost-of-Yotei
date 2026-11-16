#include "prx/libSceAgcDriver/Core/include/DrawMemo.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace AgcDriver::Core {

namespace {

std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 33u;
    value *= 0xff51afd7ed558ccdull;
    value ^= value >> 33u;
    value *= 0xc4ceb9fe1a85ec53ull;
    value ^= value >> 33u;
    return value;
}

void bump(std::atomic<std::uint64_t>& counter) {
    counter.store(counter.load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
}

struct alignas(64) Slot {
    std::uint64_t used = 0;
    ReplayEntry entry;
};

ReplayMiss rangesMiss(const DrawProof& proof, const ReplayNow& now) {
    for (std::size_t index = 0; index < proof.ranges.size(); ++index) {
        const std::size_t bytes = proof.rangeBytes[index];
        if (bytes == 0) continue;
        const auto address = proof.rangeAddress[index];
        if (now.stillCurrent == nullptr || !now.stillCurrent(now.context, address, bytes, proof.ranges[index])) return ReplayMiss::Range;
        if (now.pendingGpuWrite == nullptr || now.pendingGpuWrite(now.context, address, bytes)) return ReplayMiss::PendingGpuWrite;
    }
    return ReplayMiss::None;
}

ReplayMiss proofMiss(const DrawProof& proof, const ReplayNow& now) {
    if (proof.device != now.device) return ReplayMiss::Device;
    if (proof.shaderGeneration != now.shaderGeneration) return ReplayMiss::Shader;
    if (proof.mappingSerial != now.mappingSerial) return ReplayMiss::Mapping;
    if (proof.imageGeneration != now.imageGeneration) return ReplayMiss::Image;
    if (proof.depthGeneration != now.depthGeneration) return ReplayMiss::Depth;
    if (proof.targetGeneration != now.targetGeneration) return ReplayMiss::Target;
    return rangesMiss(proof, now);
}

}

struct DrawMemo::State {
    explicit State(ReplayRelease release) : release(release), slots(std::make_unique<Slot[]>(DrawMemo::Slots)) {}

    void hand(const ReplayEntry& entry) {
        if (release.release != nullptr) release.release(release.context, entry);
        bump(released);
    }

    void forget(const ReplayEntry& entry) {
        if (entry.lastBatch <= finished) {
            hand(entry);
            return;
        }
        waiting.push_back(entry);
        waitingCount.store(waiting.size(), std::memory_order_relaxed);
    }

    ReplayRelease release;
    std::unique_ptr<Slot[]> slots;
    std::vector<ReplayEntry> waiting;
    std::uint64_t finished = 0;
    std::atomic<std::uint64_t> finds{0};
    std::atomic<std::uint64_t> freshChecks{0};
    std::atomic<std::uint64_t> stores{0};
    std::atomic<std::uint64_t> evicted{0};
    std::atomic<std::uint64_t> replaced{0};
    std::atomic<std::uint64_t> refreshed{0};
    std::atomic<std::uint64_t> waitingCount{0};
    std::atomic<std::uint64_t> released{0};
    std::atomic<std::uint64_t> refused{0};
    std::array<std::atomic<std::uint64_t>, ReplayMissKinds> outcomes{};
};

DrawMemo::DrawMemo(ReplayRelease release) : state(std::make_unique<State>(release)) {}

DrawMemo::~DrawMemo() {
    auto& memo = *state;
    for (const auto& entry : memo.waiting) memo.hand(entry);
    for (std::size_t index = 0; index < Slots; ++index) {
        if (memo.slots[index].used != 0) memo.hand(memo.slots[index].entry);
    }
}

std::size_t DrawMemo::SlotOf(const ReplayKey& key) {
    const auto folded = (key.variant * 0x9e3779b97f4a7c15ull) ^ (key.images * 0xc2b2ae3d27d4eb4full) ^ (key.inputs * 0x165667b19e3779f9ull) ^ (key.targets * 0xd6e8feb86659fd93ull);
    return static_cast<std::size_t>(mix(folded)) & (Slots - 1u);
}

ReplayEntry* DrawMemo::Find(const ReplayKey& key) {
    auto& memo = *state;
    bump(memo.finds);
    auto& slot = memo.slots[SlotOf(key)];
    if (slot.used != 0 && slot.entry.key == key) return &slot.entry;
    bump(memo.outcomes[static_cast<std::size_t>(ReplayMiss::Absent)]);
    return nullptr;
}

ReplayMiss DrawMemo::Fresh(const ReplayEntry& entry, const ReplayNow& now) {
    auto& memo = *state;
    bump(memo.freshChecks);
    const auto miss = proofMiss(entry.proof, now);
    bump(memo.outcomes[static_cast<std::size_t>(miss)]);
    return miss;
}

ReplayEntry* DrawMemo::Store(const ReplayKey& key, const DrawProof& proof, std::span<const std::uint64_t> resolved, std::uint64_t batch) {
    auto& memo = *state;
    if (resolved.size() > ResolvedWords) {
        bump(memo.refused);
        return nullptr;
    }
    std::array<std::uint64_t, ResolvedWords> words{};
    std::copy(resolved.begin(), resolved.end(), words.begin());
    const ReplayKey storedKey = key;
    const DrawProof storedProof = proof;
    auto& slot = memo.slots[SlotOf(storedKey)];
    auto lastBatch = batch;
    if (slot.used != 0) {
        if (slot.entry.key != storedKey) {
            memo.forget(slot.entry);
            bump(memo.evicted);
        } else if (slot.entry.resolved != words) {
            memo.forget(slot.entry);
            bump(memo.replaced);
        } else {
            lastBatch = std::max(lastBatch, slot.entry.lastBatch);
            bump(memo.refreshed);
        }
    }
    slot.entry.key = storedKey;
    slot.entry.proof = storedProof;
    slot.entry.resolved = words;
    slot.entry.lastBatch = lastBatch;
    slot.used = 1;
    bump(memo.stores);
    return &slot.entry;
}

void DrawMemo::Retire(std::uint64_t finishedBatch) {
    auto& memo = *state;
    if (finishedBatch <= memo.finished) return;
    memo.finished = finishedBatch;
    std::size_t kept = 0;
    for (std::size_t index = 0; index < memo.waiting.size(); ++index) {
        if (memo.waiting[index].lastBatch <= finishedBatch) {
            memo.hand(memo.waiting[index]);
            continue;
        }
        if (kept != index) memo.waiting[kept] = memo.waiting[index];
        ++kept;
    }
    memo.waiting.resize(kept);
    memo.waitingCount.store(memo.waiting.size(), std::memory_order_relaxed);
}

DrawMemo::Counts DrawMemo::Counted() const {
    const auto& memo = *state;
    Counts counts;
    counts.finds = memo.finds.load(std::memory_order_relaxed);
    counts.freshChecks = memo.freshChecks.load(std::memory_order_relaxed);
    counts.stores = memo.stores.load(std::memory_order_relaxed);
    counts.evicted = memo.evicted.load(std::memory_order_relaxed);
    counts.replaced = memo.replaced.load(std::memory_order_relaxed);
    counts.refreshed = memo.refreshed.load(std::memory_order_relaxed);
    counts.waiting = memo.waitingCount.load(std::memory_order_relaxed);
    counts.released = memo.released.load(std::memory_order_relaxed);
    counts.refused = memo.refused.load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < ReplayMissKinds; ++index) counts.outcomes[index] = memo.outcomes[index].load(std::memory_order_relaxed);
    return counts;
}

}
