#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

namespace {

constexpr std::array<std::uint64_t EpochBumps::*, 10> EpochBumpReasons{&EpochBumps::submissions, &EpochBumps::waits, &EpochBumps::drains, &EpochBumps::reaps, &EpochBumps::packets, &EpochBumps::waitsOrdered, &EpochBumps::interrupts, &EpochBumps::queuedInterrupts, &EpochBumps::waitsBeforeEpoch, &EpochBumps::awaitedPageCollects};
std::array<std::atomic<std::uint64_t>, EpochBumpReasons.size()> epochBumpTotals{};

void countEpochTotal(std::uint64_t EpochBumps::*reason) {
    for (std::size_t index = 0; index < EpochBumpReasons.size(); ++index) {
        if (EpochBumpReasons[index] != reason) continue;
        epochBumpTotals[index].fetch_add(1, std::memory_order_relaxed);
        return;
    }
}

constexpr std::uint64_t AwaitedPageBytes = 4096;
constexpr std::uint32_t AwaitedPageBumps = 3;
constexpr std::uint32_t AwaitedPageRetries = 1;

struct AwaitedPage {
    std::uint64_t address = 0;
    std::uint32_t bumps = 0;
};

struct AwaitedPages {
    std::array<AwaitedPage, 4> pages{};
    std::size_t next = 0;
};

AwaitedPages& awaitedPages() {
    static thread_local AwaitedPages pages;
    return pages;
}

bool epochOrdersWaits() {
    static const bool enabled = std::getenv("APS5_NO_EPOCH_ORDERED_WAITS") == nullptr;
    return enabled;
}

AwaitedPage& awaitedPage(std::uint64_t address) {
    auto& awaited = awaitedPages();
    const auto page = address & ~(AwaitedPageBytes - 1);
    for (auto& entry : awaited.pages) {
        if (entry.address == page) return entry;
    }
    auto& entry = awaited.pages[awaited.next++ % awaited.pages.size()];
    entry = {page, 0};
    return entry;
}

void collectAwaitedPages() {
    for (auto& page : awaitedPages().pages) {
        if (page.bumps == 0) continue;
        --page.bumps;
        if (GuestMemory::Accessible(reinterpret_cast<const void*>(page.address), AwaitedPageBytes) && GuestMemory::CollectWritesUncached(page.address, AwaitedPageBytes) != 0) {
            countEpochTotal(&EpochBumps::awaitedPageCollects);
            continue;
        }
        page = {};
    }
}

}

EpochBumps EpochBumpsCounted() {
    EpochBumps counted;
    for (std::size_t index = 0; index < EpochBumpReasons.size(); ++index) counted.*EpochBumpReasons[index] = epochBumpTotals[index].load(std::memory_order_relaxed);
    return counted;
}

PollStats& Driver::pollStats() {
    static thread_local PollStats stats;
    return stats;
}

WaitOutcomes& Driver::waitOutcomes() {
    static thread_local WaitOutcomes outcomes;
    return outcomes;
}

Graphics::Recorder::LateStatistics& Driver::lateCountsSeen() {
    static thread_local Graphics::Recorder::LateStatistics seen{};
    return seen;
}

EpochBumps& Driver::epochBumps() {
    static thread_local EpochBumps bumps;
    return bumps;
}

std::chrono::steady_clock::time_point& Driver::lastEpochBump() {
    static thread_local std::chrono::steady_clock::time_point at{};
    return at;
}

void Driver::bumpEpoch(std::uint64_t EpochBumps::*counter) {
    lastEpochBump() = std::chrono::steady_clock::now();
    collectAwaitedPages();
    GuestMemory::BumpCollectEpoch();
    if (DrawPipeline::Active()) DrawPipeline::EpochToken().fetch_add(1, std::memory_order_relaxed);
    ++(epochBumps().*counter);
    countEpochTotal(counter);
}

void Driver::bumpWaitEpoch(std::uint64_t address, std::size_t bytes, bool memory) {
    struct Ordered {
        std::uint64_t value = 0;
        std::uint64_t generation = 0;
    };
    struct OrderedStorage {};
    auto& ordered = HostThreadLocal<std::unordered_map<std::uint64_t, Ordered>, OrderedStorage>();
    static const bool memo = std::getenv("APS5_NO_ORDERED_WAITS") == nullptr;
    if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), bytes)) {
        bumpEpoch(&EpochBumps::waits);
        return;
    }
    const bool proves = memory && epochOrdersWaits();
    const auto generation = memo || proves ? GuestMemory::CollectWritesUncached(address, bytes) : 0;
    auto* const awaited = proves && generation != 0 ? &awaitedPage(address) : nullptr;
    if (awaited != nullptr) {
        if (!Graphics::Recorder::SnapshotWriteOverlaps(address, bytes) && GuestMemory::UnchangedSinceEpochStart(address, bytes)) {
            awaited->bumps = AwaitedPageBumps;
            ++epochBumps().waitsBeforeEpoch;
            countEpochTotal(&EpochBumps::waitsBeforeEpoch);
            return;
        }
        awaited->bumps = 0;
    }
    const std::uint64_t value = bytes == 8 ? *reinterpret_cast<const volatile std::uint64_t*>(address) : *reinterpret_cast<const volatile std::uint32_t*>(address);
    if (memo && generation != 0) {
        const auto found = ordered.find(address);
        if (found != ordered.end() && found->second.value == value && GuestMemory::UnchangedSince(address, bytes, found->second.generation)) {
            if (awaited != nullptr) awaited->bumps = AwaitedPageBumps;
            ++epochBumps().waitsOrdered;
            countEpochTotal(&EpochBumps::waitsOrdered);
            return;
        }
    }
    bumpEpoch(&EpochBumps::waits);
    if (awaited != nullptr) awaited->bumps = AwaitedPageRetries;
    if (!memo || generation == 0) return;
    if (ordered.size() >= 4096) ordered.clear();
    ordered[address] = {value, generation};
}

bool Driver::packetEpoch() {
    static const bool packet = std::getenv("APS5_PACKET_EPOCH") != nullptr;
    return packet;
}

}
