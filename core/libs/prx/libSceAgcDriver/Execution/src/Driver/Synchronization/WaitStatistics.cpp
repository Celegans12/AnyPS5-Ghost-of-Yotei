#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <cstdlib>
#include <unordered_map>

namespace AgcDriver::DriverDetail {

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

std::uint64_t& Driver::lastBumpGeneration() {
    static thread_local std::uint64_t generation = 0;
    return generation;
}

bool Driver::unwrittenSinceLastBump(std::uint64_t address, std::size_t bytes) {
    const auto since = lastBumpGeneration();
    if (since == 0 || !GuestMemory::Watched(address, bytes)) return false;
    // Walk the awaited bytes now (not through the epoch memo) so a CPU store not yet collected
    // stamps its block; block stamps are 64 KiB, so a neighbouring store also counts as written.
    GuestMemory::CollectWritesUncached(address, bytes);
    // A collect racing the bump may stamp the very generation recorded there: count it as written.
    return !GuestMemory::WrittenSince(address, bytes, since - 1u);
}

void Driver::bumpEpoch(std::uint64_t EpochBumps::*counter) {
    lastEpochBump() = std::chrono::steady_clock::now();
    lastBumpGeneration() = GuestMemory::WriteGeneration();
    GuestMemory::BumpCollectEpoch();
    if (DrawPipeline::Active()) DrawPipeline::EpochToken().fetch_add(1, std::memory_order_relaxed);
    ++(epochBumps().*counter);
}

void Driver::bumpWaitEpoch(std::uint64_t address, std::size_t bytes) {
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
    const auto generation = memo ? GuestMemory::CollectWritesUncached(address, bytes) : 0;
    const std::uint64_t value = bytes == 8 ? *reinterpret_cast<const volatile std::uint64_t*>(address) : *reinterpret_cast<const volatile std::uint32_t*>(address);
    if (generation != 0) {
        const auto found = ordered.find(address);
        if (found != ordered.end() && found->second.value == value && GuestMemory::UnchangedSince(address, bytes, found->second.generation)) {
            ++epochBumps().waitsOrdered;
            return;
        }
    }
    bumpEpoch(&EpochBumps::waits);
    if (generation == 0) return;
    if (ordered.size() >= 4096) ordered.clear();
    ordered[address] = {value, generation};
}

bool Driver::packetEpoch() {
    static const bool packet = std::getenv("APS5_PACKET_EPOCH") != nullptr;
    return packet;
}

}
