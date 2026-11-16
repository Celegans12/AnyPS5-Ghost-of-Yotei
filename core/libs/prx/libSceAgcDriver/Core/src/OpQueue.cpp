#include "prx/libSceAgcDriver/Core/include/OpQueue.hpp"
#include "prx/libSceAgcDriver/Core/include/Op.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace AgcDriver::Core {

namespace {

constexpr std::uint64_t Never = std::numeric_limits<std::uint64_t>::max();

void bump(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

void relax() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

std::uintptr_t self() {
    static thread_local char mark = 0;
    return reinterpret_cast<std::uintptr_t>(&mark);
}

}

struct OpQueue::State {
    explicit State(Settings settings)
        : depth(std::max<std::uint64_t>(settings.depth, 1)),
          capacity(std::bit_ceil(depth)),
          mask(capacity - 1),
          batch(std::clamp<std::uint64_t>(settings.wakeBatch, 1, depth)),
          spins(settings.spins),
          verifies(settings.verifies),
          slots(static_cast<std::size_t>(capacity), MakeOp(OpKind::None)) {}

    const std::uint64_t depth;
    const std::uint64_t capacity;
    const std::uint64_t mask;
    const std::uint64_t batch;
    const std::uint32_t spins;
    const bool verifies;
    std::vector<Op> slots;
    std::atomic<bool> closed{false};

    alignas(64) std::atomic<std::uint64_t> published{0};
    std::uint64_t floor = 0;
    bool staged = false;
    std::atomic<std::uint64_t> batchWakes{0};
    std::atomic<std::uint64_t> roomWaits{0};
    std::atomic<std::uintptr_t> producer{0};

    alignas(64) std::atomic<std::uint64_t> retired{0};
    std::uint64_t handed = 0;
    std::atomic<std::uint64_t> taken{0};
    std::atomic<std::uint64_t> parks{0};
    std::atomic<std::uint64_t> spun{0};
    std::atomic<std::uintptr_t> consumer{0};

    alignas(64) std::atomic<std::uint32_t> parked{0};
    std::atomic<std::uint32_t> progress{0};
    std::atomic<std::uint64_t> wakeAt{Never};
    std::atomic<std::uint64_t> flushWakes{0};

    void own(std::atomic<std::uintptr_t>& owner, const char* what) {
        if (!verifies) return;
        const auto me = self();
        auto seen = owner.load(std::memory_order_relaxed);
        if (seen == me) return;
        if (seen == 0 && owner.compare_exchange_strong(seen, me, std::memory_order_relaxed)) return;
        throw std::logic_error(what);
    }

    void producing() { own(producer, "op queue: a second thread stages, publishes or flushes (APS5_VERIFY_CORE_OPS)"); }

    void consuming() { own(consumer, "op queue: a second thread takes or retires (APS5_VERIFY_CORE_OPS)"); }

    bool wake() {
        if (parked.load(std::memory_order_seq_cst) == 0) return false;
        parked.store(0, std::memory_order_seq_cst);
        parked.notify_one();
        return true;
    }

    void flush() {
        if (parked.load(std::memory_order_seq_cst) == 0) return;
        if (published.load(std::memory_order_seq_cst) == retired.load(std::memory_order_seq_cst)) return;
        if (wake()) flushWakes.fetch_add(1, std::memory_order_relaxed);
    }

    void await(std::uint64_t target) {
        if (retired.load(std::memory_order_seq_cst) >= target) return;
        for (;;) {
            const auto seen = progress.load(std::memory_order_seq_cst);
            auto armed = wakeAt.load(std::memory_order_seq_cst);
            while (target < armed && !wakeAt.compare_exchange_weak(armed, target, std::memory_order_seq_cst)) {
            }
            if (retired.load(std::memory_order_seq_cst) >= target || closed.load(std::memory_order_seq_cst)) return;
            progress.wait(seen, std::memory_order_seq_cst);
        }
    }

    std::span<const Op> ready(std::size_t most) {
        const auto head = retired.load(std::memory_order_relaxed);
        const auto available = published.load(std::memory_order_acquire) - head;
        if (available == 0 || most == 0) return {};
        const auto index = head & mask;
        const auto count = std::min<std::uint64_t>(std::min<std::uint64_t>(available, capacity - index), most);
        const auto end = head + count;
        if (end > handed) {
            bump(taken, end - handed);
            handed = end;
        }
        return std::span<const Op>(slots.data() + index, static_cast<std::size_t>(count));
    }
};

OpQueue::OpQueue(Settings settings) : state(std::make_unique<State>(settings)) {}

OpQueue::~OpQueue() = default;

Op* OpQueue::Stage() {
    auto& queue = *state;
    queue.producing();
    if (queue.closed.load(std::memory_order_acquire)) return nullptr;
    const auto position = queue.published.load(std::memory_order_relaxed);
    if (position - queue.floor >= queue.depth) {
        queue.floor = queue.retired.load(std::memory_order_acquire);
        if (position - queue.floor >= queue.depth) return nullptr;
    }
    queue.staged = true;
    return queue.slots.data() + (position & queue.mask);
}

std::uint64_t OpQueue::Publish() {
    auto& queue = *state;
    queue.producing();
    if (queue.verifies && !queue.staged) throw std::logic_error("op queue: publish without a staged operation (APS5_VERIFY_CORE_OPS)");
    queue.staged = false;
    const auto position = queue.published.load(std::memory_order_relaxed);
    queue.published.store(position + 1, std::memory_order_seq_cst);
    if (queue.parked.load(std::memory_order_seq_cst) == 0) return position;
    if (position + 1 - queue.retired.load(std::memory_order_seq_cst) < queue.batch) return position;
    if (queue.wake()) bump(queue.batchWakes);
    return position;
}

void OpQueue::Flush() {
    auto& queue = *state;
    queue.producing();
    queue.flush();
}

void OpQueue::AwaitRoom() {
    auto& queue = *state;
    queue.producing();
    queue.flush();
    const auto last = queue.published.load(std::memory_order_relaxed);
    const auto target = last + queue.batch > queue.depth ? last + queue.batch - queue.depth : 0;
    if (queue.retired.load(std::memory_order_seq_cst) >= target || queue.closed.load(std::memory_order_seq_cst)) return;
    bump(queue.roomWaits);
    queue.await(target);
}

void OpQueue::AwaitRetired(std::uint64_t position) {
    auto& queue = *state;
    queue.flush();
    queue.await(position);
}

std::span<const Op> OpQueue::Take(std::size_t most) {
    auto& queue = *state;
    queue.consuming();
    if (most == 0) return {};
    for (;;) {
        const bool shut = queue.closed.load(std::memory_order_seq_cst);
        if (const auto slice = queue.ready(most); !slice.empty()) return slice;
        if (shut) return {};
        const auto head = queue.retired.load(std::memory_order_relaxed);
        std::uint32_t turns = 0;
        bool found = false;
        while (turns < queue.spins && !found) {
            relax();
            ++turns;
            found = queue.published.load(std::memory_order_relaxed) != head;
        }
        if (turns != 0) bump(queue.spun, turns);
        if (found) continue;
        queue.parked.store(1, std::memory_order_seq_cst);
        if (queue.published.load(std::memory_order_seq_cst) == head && !queue.closed.load(std::memory_order_seq_cst)) {
            bump(queue.parks);
            queue.parked.wait(1, std::memory_order_seq_cst);
        }
        queue.parked.store(0, std::memory_order_seq_cst);
    }
}

std::span<const Op> OpQueue::TryTake(std::size_t most) {
    auto& queue = *state;
    queue.consuming();
    return queue.ready(most);
}

void OpQueue::Retire(std::size_t count) {
    auto& queue = *state;
    queue.consuming();
    if (count == 0) return;
    const auto next = queue.retired.load(std::memory_order_relaxed) + count;
    if (queue.verifies && next > queue.handed) throw std::logic_error("op queue: retire past what take returned (APS5_VERIFY_CORE_OPS)");
    queue.retired.store(next, std::memory_order_seq_cst);
    if (next < queue.wakeAt.load(std::memory_order_seq_cst)) return;
    queue.wakeAt.store(Never, std::memory_order_seq_cst);
    queue.progress.fetch_add(1, std::memory_order_seq_cst);
    queue.progress.notify_all();
}

void OpQueue::Close() {
    auto& queue = *state;
    queue.closed.store(true, std::memory_order_seq_cst);
    queue.parked.store(0, std::memory_order_seq_cst);
    queue.parked.notify_all();
    queue.progress.fetch_add(1, std::memory_order_seq_cst);
    queue.progress.notify_all();
}

bool OpQueue::Closed() const {
    return state->closed.load(std::memory_order_acquire);
}

std::uint64_t OpQueue::Published() const {
    return state->published.load(std::memory_order_acquire);
}

std::uint64_t OpQueue::Retired() const {
    return state->retired.load(std::memory_order_acquire);
}

OpQueue::Counts OpQueue::Counted() const {
    const auto& queue = *state;
    Counts counts;
    counts.published = queue.published.load(std::memory_order_acquire);
    counts.taken = queue.taken.load(std::memory_order_relaxed);
    counts.parks = queue.parks.load(std::memory_order_relaxed);
    counts.batchWakes = queue.batchWakes.load(std::memory_order_relaxed);
    counts.flushWakes = queue.flushWakes.load(std::memory_order_relaxed);
    counts.roomWaits = queue.roomWaits.load(std::memory_order_relaxed);
    counts.spun = queue.spun.load(std::memory_order_relaxed);
    return counts;
}

}
