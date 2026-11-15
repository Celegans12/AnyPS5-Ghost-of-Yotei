#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace AgcDriver::Core {

namespace {

void bump(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) {
    counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

}

struct BlockRing::State {
    struct Slot {
        RingChunk chunk;
        std::uint64_t taken = 0;
        std::atomic<std::uint32_t> lap{0};
        std::atomic<std::uint32_t> inFlight{0};
        std::atomic<std::uint64_t> lastSerial{0};
    };

    RingSource source;
    std::array<Slot, MaxChunks> slots;
    std::array<std::uint32_t, MaxChunks> order{};
    std::size_t count = 0;
    std::size_t at = 0;
    std::size_t used = 0;
    std::uint64_t group = 0;
    bool open = false;
    std::atomic<std::uint64_t> chunks{0};
    std::atomic<std::uint64_t> laps{0};
    std::atomic<std::uint64_t> full{0};
    std::atomic<std::uint64_t> bytes{0};

    Slot& current() { return slots[order[at]]; }

    bool fits(std::size_t from, std::size_t wanted, std::size_t alignment, std::size_t& position) {
        const auto past = static_cast<std::size_t>((current().chunk.deviceAddress + from) % alignment);
        position = past == 0 ? from : from + (alignment - past);
        return position <= ChunkBytes && wanted <= ChunkBytes - position;
    }

    bool reusable(const Slot& slot) const {
        if (open && slot.taken == group) return false;
        if (slot.inFlight.load(std::memory_order_acquire) != 0) return false;
        const auto last = slot.lastSerial.load(std::memory_order_relaxed);
        return last == 0 || (source.executed != nullptr && source.executed(source.context, last));
    }

    std::size_t follow(std::size_t position) {
        if (position == at || position == (at + 1) % count) return position;
        if (position > at) {
            std::rotate(order.data() + at + 1, order.data() + position, order.data() + position + 1);
            return at + 1;
        }
        std::rotate(order.data() + position, order.data() + position + 1, order.data() + at + 1);
        return at;
    }

    bool reuse(std::size_t position) {
        at = position;
        used = 0;
        auto& slot = current();
        slot.lastSerial.store(0, std::memory_order_relaxed);
        slot.lap.store(slot.lap.load(std::memory_order_relaxed) + 1u, std::memory_order_release);
        bump(laps);
        return true;
    }

    bool grow() {
        if (count >= MaxChunks || source.allocate == nullptr) return false;
        auto created = source.allocate(source.context, ChunkBytes);
        if (created.bytes == nullptr) return false;
        const auto id = static_cast<std::uint32_t>(count);
        slots[id].chunk = std::move(created);
        const auto position = count == 0 ? std::size_t{0} : at + 1;
        for (auto index = count; index > position; --index) order[index] = order[index - 1];
        order[position] = id;
        at = position;
        used = 0;
        ++count;
        bump(chunks);
        return true;
    }

    bool advance() {
        if (count != 0) {
            const auto next = (at + 1) % count;
            if (reusable(slots[order[next]])) return reuse(next);
        }
        if (grow()) return true;
        for (std::size_t step = 2; step <= count; ++step) {
            const auto position = (at + step) % count;
            if (reusable(slots[order[position]])) return reuse(follow(position));
        }
        return false;
    }

    RingSlice refuse() {
        open = false;
        bump(full);
        return {};
    }
};

BlockRing::BlockRing(RingSource source) : state(std::make_unique<State>()) {
    state->source = source;
}

BlockRing::~BlockRing() = default;

RingSlice BlockRing::Take(std::size_t bytes, std::size_t alignment) {
    auto& ring = *state;
    if (!ring.open) {
        ++ring.group;
        ring.open = true;
    }
    const auto align = std::max<std::size_t>(alignment, 1);
    if (bytes > ChunkBytes || align > ChunkBytes) return ring.refuse();
    std::size_t position = 0;
    if (ring.count == 0 || !ring.fits(ring.used, bytes, align, position)) {
        if (!ring.advance() || !ring.fits(0, bytes, align, position)) return ring.refuse();
    }
    const auto id = ring.order[ring.at];
    auto& slot = ring.slots[id];
    slot.taken = ring.group;
    ring.used = position + bytes;
    bump(ring.bytes, bytes);
    RingSlice slice;
    slice.bytes = slot.chunk.bytes + position;
    slice.deviceAddress = slot.chunk.deviceAddress + position;
    slice.chunk = id;
    slice.lap = slot.lap.load(std::memory_order_relaxed);
    return slice;
}

void BlockRing::Posted(const RingSlice& slice) {
    state->open = false;
    if (slice.bytes == nullptr || slice.chunk >= MaxChunks) return;
    state->slots[slice.chunk].inFlight.fetch_add(1, std::memory_order_relaxed);
}

void BlockRing::Emitted(const RingSlice& slice, std::uint64_t serial) {
    if (slice.bytes == nullptr || slice.chunk >= MaxChunks) return;
    auto& slot = state->slots[slice.chunk];
    auto last = slot.lastSerial.load(std::memory_order_relaxed);
    while (last < serial && !slot.lastSerial.compare_exchange_weak(last, serial, std::memory_order_relaxed)) {
    }
    slot.inFlight.fetch_sub(1, std::memory_order_release);
}

bool BlockRing::Live(const RingSlice& slice) const {
    return slice.bytes != nullptr && slice.chunk < MaxChunks && state->slots[slice.chunk].lap.load(std::memory_order_acquire) == slice.lap;
}

BlockRing::Counts BlockRing::Counted() const {
    const auto& ring = *state;
    Counts counts;
    counts.chunks = ring.chunks.load(std::memory_order_relaxed);
    counts.laps = ring.laps.load(std::memory_order_relaxed);
    counts.full = ring.full.load(std::memory_order_relaxed);
    counts.bytes = ring.bytes.load(std::memory_order_relaxed);
    return counts;
}

}
