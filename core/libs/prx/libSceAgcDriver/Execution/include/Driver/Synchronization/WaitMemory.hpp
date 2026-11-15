#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_WAITMEMORY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_WAITMEMORY_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Synchronization/LabelStores.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Execution/include/WaitEvents.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::DriverDetail {

inline constexpr const char* LabelRefusalNames[6] = {"none", "trust-off", "queued", "overwritten", "unclosed", "behind-completion"};

inline constexpr std::chrono::microseconds PollTryInterval{1000};

// A blocked wait (section C). EventFallback bounds a sleep whose producer the driver knows of, so
// the stall watch, the timeout report and a missed event cost at most this; CpuStoreInterval is the
// timed wait for a value no driver event announces (a guest CPU store).
inline constexpr std::chrono::microseconds EventFallback{2000};
inline constexpr std::chrono::microseconds CpuStoreInterval{200};
// How often a blocked wait asks the flush hook for results held behind queued draws, and a landing
// wait asks whether its recorded writes settled.
inline constexpr std::chrono::microseconds HeldFlushInterval{6400};
inline constexpr std::chrono::microseconds SettleInterval{3200};

struct LabelStore {
    std::uint64_t stamp = 0;
    std::uint32_t value = 0;
    std::uint32_t queue = 0xffffffffu;
};

struct WriteRecord {
    std::uint64_t target;
    std::uint64_t length;
    std::uint32_t queue;
    std::uint32_t opcode;
};

struct PollStats {
    std::uint64_t ticks = 0, tableHits = 0, tries = 0, triesFailed = 0, submits = 0, reaps = 0, reapsWithWork = 0;

    std::uint64_t reapsBehindCompletion = 0, reapsOther = 0;
    std::array<std::uint64_t, 6> refusals{};
    double longestTryHoldUs = 0;
    double tryHoldUs = 0;
    // Blocked rounds of a wait: how each ended (an event, the fallback with a known producer, the
    // timed wait for a CPU store), the time blocked, and the waits that ended right after a timer
    // wake (a value that came without an event).
    std::uint64_t blocks = 0, eventWakes = 0, timerWakesKnown = 0, timerWakesCpu = 0, endedByTimer = 0;
    double blockedUs = 0;
    // Event wakes by WaitEvents::Kind bit: range, progress, driver state, mutex free, labels.
    std::array<std::uint64_t, 5> wakeKinds{};
    // The queue worker's own sleeps (order waits, idle with completions pending): rounds, those the
    // fallback ended, and order waits released right after a fallback wake (a missed event).
    std::uint64_t workerSleeps = 0, workerFallbacks = 0, orderEndedByTimer = 0;
    // Waits that left the entry checks, by what ended them: the value in memory, the driver's label
    // history, the recorder's label table, settled recorded writes.
    enum End : std::size_t { EndMemory, EndHistory, EndTable, EndSettled, EndCount };
    std::array<std::uint64_t, EndCount> ended{};
    std::array<double, EndCount> endedMs{};
};

struct WaitOutcomes {
    std::uint64_t atEntry = 0, fromRecorder = 0, fromRecorderSameQueue = 0, fromRecorderPolling = 0, polled = 0, timedOut = 0, pollSubmits = 0, pollReaps = 0;
    std::uint64_t fromRecorderUnlocked = 0, entriesUnlocked = 0, entryTriesFailed = 0, fromRecorderLate = 0, lateRefusedCpuStore = 0;
};

struct EpochBumps {
    std::uint64_t submissions = 0, waits = 0, drains = 0, reaps = 0, packets = 0, waitsOrdered = 0;
};

void PollSleep();

bool OwnStoreRejects(std::span<const std::uint32_t> packet, std::uint64_t awaited, std::size_t bytes, std::uint32_t queue);

}

#endif
