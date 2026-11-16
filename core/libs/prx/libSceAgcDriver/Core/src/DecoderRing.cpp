#include "prx/libSceAgcDriver/Core/include/DecoderRing.hpp"
#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace AgcDriver::Core {

namespace {

void relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

}

struct DecoderRing::State {
    explicit State(Settings settings)
        : depth(std::max<std::uint64_t>(settings.depth, 2)),
          capacity(std::bit_ceil(depth)),
          mask(capacity - 1),
          wakeBatch(std::max<std::uint32_t>(settings.wakeBatch, 1)),
          spins(settings.spins),
          verifies(settings.verifies),
          slots(static_cast<std::size_t>(capacity)),
          done(static_cast<std::size_t>(capacity), 0) {}

    const std::uint64_t depth;
    const std::uint64_t capacity;
    const std::uint64_t mask;
    const std::uint32_t wakeBatch;
    const std::uint32_t spins;
    const bool verifies;
    std::vector<DecodedOp> slots;
    std::vector<std::uint8_t> done;

    alignas(64) std::atomic<std::uint64_t> posted{0};
    std::uint32_t unwoken = 0;
    std::atomic<std::uint64_t> roomWaits{0};
    std::atomic<std::uint64_t> wakes{0};
    std::atomic<std::uint64_t> flushes{0};

    alignas(64) std::atomic<std::uint64_t> retired{0};
    std::atomic<std::uint32_t> sleepers{0};
    std::atomic<bool> closed{false};
    std::atomic<std::uint64_t> reach{0};
    std::atomic<std::uint64_t> spun{0};
    std::atomic<std::uint64_t> full{0};

    alignas(64) std::mutex mutex;
    std::condition_variable work;
    std::condition_variable progress;
    std::uint64_t claimed = 0;
    std::uint64_t barrier = 0;
    std::uint64_t flying = 0;
    std::uint64_t emitted = 0;
    bool retiring = false;
    bool resting = false;
    std::uint64_t rests = 0;
    std::uint64_t claims = 0;
    std::uint64_t alone = 0;
    std::uint64_t parks = 0;
    std::uint64_t retireRuns = 0;

    void wake() {
        {
            std::lock_guard lock(mutex);
        }
        work.notify_all();
        unwoken = 0;
        wakes.store(wakes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    Batch claimable(std::uint32_t most) {
        const auto head = retired.load(std::memory_order_seq_cst);
        if (resting || head < barrier || most == 0) return {};
        const auto published = posted.load(std::memory_order_seq_cst);
        if (claimed >= published) return {};
        const auto first = claimed;
        if ((slots[first & mask].flags & DecodedAlone) != 0) {
            if (head != first) return {};
            if (verifies && flying != 0) throw std::logic_error("decoder ring: an operation that runs alone starts beside another (APS5_VERIFY_DECODER)");
            claimed = first + 1;
            reach.store(claimed, std::memory_order_relaxed);
            barrier = first + 1;
            flying = 1;
            ++claims;
            ++alone;
            return {first, 1};
        }
        std::uint32_t count = 0;
        while (count < most && first + count < published && (slots[(first + count) & mask].flags & DecodedAlone) == 0) ++count;
        claimed += count;
        reach.store(claimed, std::memory_order_relaxed);
        if (count == most) full.store(full.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        flying += count;
        ++claims;
        return {first, count};
    }

    void shut() {
        closed.store(true, std::memory_order_seq_cst);
        work.notify_all();
        progress.notify_all();
    }
};

DecoderRing::DecoderRing(Settings settings) : state(std::make_unique<State>(settings)) {}

DecoderRing::~DecoderRing() = default;

std::uint64_t DecoderRing::Post(const DecodedOp& op) {
    auto& ring = *state;
    const auto position = ring.posted.load(std::memory_order_relaxed);
    if (position - ring.retired.load(std::memory_order_acquire) >= ring.depth) {
        ring.roomWaits.store(ring.roomWaits.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        AwaitRetired(position - ring.depth + 1);
    }
    if (ring.closed.load(std::memory_order_acquire)) return Shut;
    auto& slot = ring.slots[position & ring.mask];
    slot = op;
    slot.order = position;
    ring.posted.store(position + 1, std::memory_order_seq_cst);
    if ((op.flags & DecodedAlone) != 0) ring.flushes.store(ring.flushes.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    if (ring.sleepers.load(std::memory_order_seq_cst) != 0 && ((op.flags & DecodedAlone) != 0 || ++ring.unwoken >= ring.wakeBatch)) ring.wake();
    return position;
}

void DecoderRing::Flush() {
    auto& ring = *state;
    ring.flushes.store(ring.flushes.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    if (ring.sleepers.load(std::memory_order_seq_cst) != 0) ring.wake();
}

void DecoderRing::Join() {
    AwaitRetired(state->posted.load(std::memory_order_relaxed));
}

void DecoderRing::AwaitRetired(std::uint64_t count) {
    auto& ring = *state;
    if (ring.retired.load(std::memory_order_acquire) >= count) return;
    Flush();
    std::unique_lock lock(ring.mutex);
    ring.progress.wait(lock, [&] { return ring.retired.load(std::memory_order_seq_cst) >= count || ring.closed.load(std::memory_order_seq_cst); });
}

DecoderRing::Batch DecoderRing::TryClaim(std::uint32_t most) {
    auto& ring = *state;
    std::lock_guard lock(ring.mutex);
    return ring.claimable(most);
}

DecoderRing::Batch DecoderRing::Claim(std::uint32_t most) {
    auto& ring = *state;
    if (most == 0) return {};
    const auto flushed = ring.flushes.load(std::memory_order_acquire);
    for (std::uint32_t spin = 0; spin < ring.spins; ++spin) {
        if (ring.closed.load(std::memory_order_relaxed) || ring.flushes.load(std::memory_order_acquire) != flushed) break;
        if (ring.posted.load(std::memory_order_acquire) - ring.reach.load(std::memory_order_relaxed) >= most) {
            std::lock_guard lock(ring.mutex);
            const auto batch = ring.claimable(most);
            if (batch.count != 0) {
                ring.spun.store(ring.spun.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
                return batch;
            }
        }
        relax();
    }
    std::unique_lock lock(ring.mutex);
    for (;;) {
        auto batch = ring.claimable(most);
        if (batch.count != 0) return batch;
        if (ring.closed.load(std::memory_order_seq_cst)) return {};
        ring.sleepers.fetch_add(1, std::memory_order_seq_cst);
        batch = ring.claimable(most);
        if (batch.count != 0 || ring.closed.load(std::memory_order_seq_cst)) {
            ring.sleepers.fetch_sub(1, std::memory_order_seq_cst);
            return batch;
        }
        ++ring.parks;
        ring.work.wait(lock);
        ring.sleepers.fetch_sub(1, std::memory_order_seq_cst);
    }
}

const DecodedOp& DecoderRing::At(std::uint64_t order) const {
    return state->slots[order & state->mask];
}

void DecoderRing::Complete(Batch batch, void* context, Emit emit) {
    auto& ring = *state;
    if (batch.count == 0) return;
    std::unique_lock lock(ring.mutex);
    if (ring.verifies && (batch.first < ring.retired.load(std::memory_order_seq_cst) || batch.first + batch.count > ring.claimed)) throw std::logic_error("decoder ring: a batch that was not claimed is completed (APS5_VERIFY_DECODER)");
    for (std::uint32_t index = 0; index < batch.count; ++index) {
        auto& flag = ring.done[(batch.first + index) & ring.mask];
        if (ring.verifies && flag != 0) throw std::logic_error("decoder ring: an operation is completed twice (APS5_VERIFY_DECODER)");
        flag = 1;
    }
    ring.flying -= std::min<std::uint64_t>(ring.flying, batch.count);
    if (ring.flying == 0) ring.progress.notify_all();
    if (ring.retiring) return;
    ring.retiring = true;
    for (;;) {
        const auto head = ring.retired.load(std::memory_order_relaxed);
        auto end = head;
        while (end < ring.claimed && ring.done[end & ring.mask] != 0) ++end;
        if (end == head) break;
        lock.unlock();
        try {
            for (auto order = head; order < end; ++order) {
                const auto& op = ring.slots[order & ring.mask];
                if (ring.verifies && (op.order != order || order != ring.emitted)) throw std::logic_error("decoder ring: operations leave out of order (APS5_VERIFY_DECODER)");
                ring.emitted = order + 1;
                if (emit != nullptr) emit(context, op);
            }
        } catch (...) {
            lock.lock();
            ring.retiring = false;
            ring.shut();
            throw;
        }
        lock.lock();
        for (auto order = head; order < end; ++order) ring.done[order & ring.mask] = 0;
        ring.retired.store(end, std::memory_order_seq_cst);
        ++ring.retireRuns;
        ring.progress.notify_all();
        if (ring.sleepers.load(std::memory_order_seq_cst) != 0) ring.work.notify_all();
    }
    ring.retiring = false;
    ring.progress.notify_all();
}

void DecoderRing::Rest(void* context, Work work) {
    auto& ring = *state;
    std::unique_lock lock(ring.mutex);
    ring.progress.wait(lock, [&] { return !ring.resting; });
    ring.resting = true;
    ring.progress.wait(lock, [&] { return ring.flying == 0 && !ring.retiring; });
    ++ring.rests;
    lock.unlock();
    try {
        if (work != nullptr) work(context);
    } catch (...) {
        lock.lock();
        ring.resting = false;
        ring.progress.notify_all();
        ring.work.notify_all();
        throw;
    }
    lock.lock();
    ring.resting = false;
    ring.progress.notify_all();
    ring.work.notify_all();
}

std::uint64_t DecoderRing::Posted() const {
    return state->posted.load(std::memory_order_acquire);
}

std::uint64_t DecoderRing::Retired() const {
    return state->retired.load(std::memory_order_acquire);
}

void DecoderRing::Close() {
    auto& ring = *state;
    std::lock_guard lock(ring.mutex);
    ring.shut();
}

bool DecoderRing::Closed() const {
    return state->closed.load(std::memory_order_acquire);
}

DecoderRing::Counts DecoderRing::Counted() const {
    auto& ring = *state;
    std::lock_guard lock(ring.mutex);
    Counts counts;
    counts.posted = ring.posted.load(std::memory_order_acquire);
    counts.retired = ring.retired.load(std::memory_order_acquire);
    counts.claims = ring.claims;
    counts.claimed = ring.claimed;
    counts.alone = ring.alone;
    counts.roomWaits = ring.roomWaits.load(std::memory_order_relaxed);
    counts.wakes = ring.wakes.load(std::memory_order_relaxed);
    counts.parks = ring.parks;
    counts.retireRuns = ring.retireRuns;
    counts.rests = ring.rests;
    counts.spun = ring.spun.load(std::memory_order_relaxed);
    counts.full = ring.full.load(std::memory_order_relaxed);
    return counts;
}

}
