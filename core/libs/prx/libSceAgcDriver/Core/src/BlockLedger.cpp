#include "prx/libSceAgcDriver/Core/include/BlockLedger.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>

namespace AgcDriver::Core {

namespace {

constexpr std::uint64_t BlockShift = 16;
constexpr std::uint64_t RegionShift = 22;
constexpr std::uint64_t PageShift = 30;
constexpr std::uint64_t AddressShift = 47;
constexpr std::uint64_t BlocksPerRegion = std::uint64_t{1} << (RegionShift - BlockShift);
constexpr std::uint64_t BlocksPerPage = std::uint64_t{1} << (PageShift - BlockShift);
constexpr std::uint64_t RegionsPerPage = std::uint64_t{1} << (PageShift - RegionShift);
constexpr std::uint64_t PageCount = std::uint64_t{1} << (AddressShift - PageShift);
constexpr std::uint64_t AddressLimit = std::uint64_t{1} << AddressShift;
constexpr std::uint64_t PendingMask = 0xffu;
constexpr std::uint64_t PendingFull = 0xffu;
constexpr std::uint64_t OwnerMask = std::uint64_t{3} << 8u;
constexpr std::uint64_t OwnerGpu = std::uint64_t{1} << 8u;
constexpr std::uint64_t GenerationStep = std::uint64_t{1} << 32u;

static_assert(BlockLedger::BlockBytes == std::uint64_t{1} << BlockShift);
static_assert(BlockLedger::RegionBytes == std::uint64_t{1} << RegionShift);

using Word = std::atomic<std::uint64_t>;
using RegionCounters = std::array<Word, RegionsPerPage>;

struct Span {
    std::uint64_t firstBlock = 0;
    std::uint64_t endBlock = 0;
    bool far = false;
};

Span spanOf(std::uint64_t address, std::uint64_t bytes) {
    Span span;
    if (bytes == 0) return span;
    const auto last = bytes - 1 > ~address ? ~std::uint64_t{0} : address + (bytes - 1);
    span.far = last >= AddressLimit;
    if (address < AddressLimit) {
        span.firstBlock = address >> BlockShift;
        span.endBlock = (std::min(last, AddressLimit - 1) >> BlockShift) + 1;
    }
    return span;
}

template <typename TStep>
void update(Word& word, TStep step) {
    auto value = word.load(std::memory_order_relaxed);
    while (!word.compare_exchange_weak(value, step(value), std::memory_order_acq_rel, std::memory_order_relaxed)) {}
}

void decrementToZero(Word& word) {
    update(word, [](std::uint64_t value) { return value == 0 ? value : value - 1; });
}

std::uint64_t withPending(std::uint64_t entry, std::uint64_t pending) {
    return (entry & ~PendingMask) | pending;
}

bool readable(std::uint64_t entry, bool idle) {
    if ((entry & OwnerMask) == OwnerGpu) return false;
    const auto pending = entry & PendingMask;
    return pending == 0 || (pending == PendingFull && idle);
}

}

struct BlockLedger::State {
    struct Page {
        std::array<Word, BlocksPerPage> entries{};
        RegionCounters content{};
        RegionCounters registration{};
    };

    std::unique_ptr<std::atomic<Page*>[]> pages = std::make_unique<std::atomic<Page*>[]>(PageCount);
    Word farEntry{0};
    Word farContent{0};
    Word farRegistration{0};
    Word pendingRanges{0};
    Word unknown{0};
    Word mapping{0};
    Word images{0};

    ~State() {
        for (std::uint64_t index = 0; index < PageCount; ++index) delete pages[index].load(std::memory_order_relaxed);
    }

    Page* acquire(std::uint64_t index) {
        auto& slot = pages[index];
        if (auto* page = slot.load(std::memory_order_acquire)) return page;
        auto fresh = std::make_unique<Page>();
        Page* expected = nullptr;
        if (slot.compare_exchange_strong(expected, fresh.get(), std::memory_order_acq_rel, std::memory_order_acquire)) return fresh.release();
        return expected;
    }

    const Page* find(std::uint64_t index) const {
        return pages[index].load(std::memory_order_acquire);
    }

    template <typename TStep>
    void write(std::uint64_t address, std::uint64_t bytes, TStep step, RegionCounters Page::*counters, Word* farCounter) {
        const auto span = spanOf(address, bytes);
        auto block = span.firstBlock;
        while (block < span.endBlock) {
            auto* page = acquire(block / BlocksPerPage);
            const auto region = block / BlocksPerRegion;
            const auto regionEnd = std::min(span.endBlock, (region + 1) * BlocksPerRegion);
            for (; block < regionEnd; ++block) step(page->entries[block % BlocksPerPage]);
            if (counters != nullptr) (page->*counters)[region % RegionsPerPage].fetch_add(1, std::memory_order_acq_rel);
        }
        if (!span.far) return;
        step(farEntry);
        if (farCounter != nullptr) farCounter->fetch_add(1, std::memory_order_acq_rel);
    }

    bool readableSpan(std::uint64_t address, std::uint64_t bytes, bool idle) const {
        const auto span = spanOf(address, bytes);
        auto block = span.firstBlock;
        while (block < span.endBlock) {
            const auto index = block / BlocksPerPage;
            const auto pageEnd = std::min(span.endBlock, (index + 1) * BlocksPerPage);
            if (const auto* page = find(index)) {
                for (; block < pageEnd; ++block) {
                    if (!readable(page->entries[block % BlocksPerPage].load(std::memory_order_acquire), idle)) return false;
                }
            }
            block = pageEnd;
        }
        return !span.far || readable(farEntry.load(std::memory_order_acquire), idle);
    }

    std::uint64_t sum(std::uint64_t address, std::uint64_t bytes, RegionCounters Page::*counters, const Word& farCounter) const {
        const auto span = spanOf(address, bytes);
        std::uint64_t total = span.far ? farCounter.load(std::memory_order_acquire) : 0;
        if (span.firstBlock == span.endBlock) return total;
        auto region = span.firstBlock / BlocksPerRegion;
        const auto endRegion = (span.endBlock - 1) / BlocksPerRegion + 1;
        while (region < endRegion) {
            const auto index = region / RegionsPerPage;
            const auto pageEnd = std::min(endRegion, (index + 1) * RegionsPerPage);
            if (const auto* page = find(index)) {
                for (; region < pageEnd; ++region) total += (page->*counters)[region % RegionsPerPage].load(std::memory_order_acquire);
            }
            region = pageEnd;
        }
        return total;
    }
};

BlockLedger::BlockLedger() : state(std::make_unique<State>()) {}

BlockLedger::~BlockLedger() = default;

void BlockLedger::Queue(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    const bool idle = state->pendingRanges.fetch_add(1, std::memory_order_acq_rel) == 0;
    state->write(address, bytes, [idle](Word& entry) {
        update(entry, [idle](std::uint64_t value) {
            const auto pending = value & PendingMask;
            if (pending == PendingFull) return withPending(value, idle ? 1 : PendingFull);
            return withPending(value, pending + 1);
        });
    }, nullptr, nullptr);
}

void BlockLedger::Applied(std::uint64_t address, std::uint64_t bytes) {
    if (bytes == 0) return;
    state->write(address, bytes, [](Word& entry) {
        update(entry, [](std::uint64_t value) {
            const auto pending = value & PendingMask;
            return (pending == 0 || pending == PendingFull) ? value : withPending(value, pending - 1);
        });
    }, nullptr, nullptr);
    decrementToZero(state->pendingRanges);
}

void BlockLedger::QueueUnknown() {
    state->unknown.fetch_add(1, std::memory_order_acq_rel);
}

void BlockLedger::AppliedUnknown() {
    decrementToZero(state->unknown);
}

void BlockLedger::GpuWrote(std::uint64_t address, std::uint64_t bytes) {
    state->write(address, bytes, [](Word& entry) { update(entry, [](std::uint64_t value) { return ((value & ~OwnerMask) | OwnerGpu) + GenerationStep; }); }, &State::Page::content, &state->farContent);
}

void BlockLedger::Stored(std::uint64_t address, std::uint64_t bytes) {
    state->write(address, bytes, [](Word& entry) { update(entry, [](std::uint64_t value) { return (value & ~OwnerMask) + GenerationStep; }); }, &State::Page::content, &state->farContent);
}

void BlockLedger::CpuDirtied(std::uint64_t address, std::uint64_t bytes) {
    state->write(address, bytes, [](Word& entry) { entry.fetch_add(GenerationStep, std::memory_order_acq_rel); }, &State::Page::content, &state->farContent);
}

void BlockLedger::Registered(std::uint64_t address, std::uint64_t bytes) {
    state->write(address, bytes, [](Word&) {}, &State::Page::registration, &state->farRegistration);
}

void BlockLedger::Remapped() {
    state->mapping.fetch_add(1, std::memory_order_acq_rel);
}

void BlockLedger::ImagesChanged() {
    state->images.fetch_add(1, std::memory_order_acq_rel);
}

bool BlockLedger::CpuReadable(std::uint64_t address, std::uint64_t bytes) const {
    if (state->unknown.load(std::memory_order_acquire) != 0) return false;
    return state->readableSpan(address, bytes, state->pendingRanges.load(std::memory_order_acquire) == 0);
}

Owner BlockLedger::OwnerOf(std::uint64_t address) const {
    std::uint64_t entry = 0;
    if (address >= AddressLimit) {
        entry = state->farEntry.load(std::memory_order_acquire);
    } else {
        const auto block = address >> BlockShift;
        if (const auto* page = state->find(block / BlocksPerPage)) entry = page->entries[block % BlocksPerPage].load(std::memory_order_acquire);
    }
    return (entry & OwnerMask) == OwnerGpu ? Owner::Gpu : Owner::Cpu;
}

std::uint64_t BlockLedger::ContentEpoch(std::uint64_t address, std::uint64_t bytes) const {
    return state->sum(address, bytes, &State::Page::content, state->farContent);
}

std::uint64_t BlockLedger::RegistrationEpoch(std::uint64_t address, std::uint64_t bytes) const {
    return state->sum(address, bytes, &State::Page::registration, state->farRegistration);
}

std::uint64_t BlockLedger::MappingEpoch() const {
    return state->mapping.load(std::memory_order_acquire);
}

std::uint64_t BlockLedger::ImageEpoch() const {
    return state->images.load(std::memory_order_acquire);
}

}
