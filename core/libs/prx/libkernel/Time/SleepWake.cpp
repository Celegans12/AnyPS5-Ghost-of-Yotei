#include "prx/libkernel/Time/include/SleepWake.hpp"
#include "prx/libkernel/Time/include/TimedWait.hpp"
#include "prx/libkernel/Pthread/include/WindowsThreadLocal.hpp"

#include <atomic>
#include <cstdlib>
#include <limits>

namespace SleepWake {

namespace {

struct ThreadState {
    unsigned earlyWakes = 0;
    std::uint64_t lastSleepEndNanos = 0;
    bool sleeping = false;
};

ThreadState& State() {
#ifdef _WIN32
    return WindowsThreadLocal<ThreadState>::Get();
#else
    static thread_local ThreadState state;
    return state;
#endif
}

std::atomic<std::uint64_t> completions{0};

bool Enabled() {
    static const bool enabled = std::getenv("APS5_NO_SLEEP_WAKE") == nullptr;
    return enabled;
}

bool SleepOnce(std::uint64_t nanos) {
    ThreadState& state = State();
    bool woken = false;
    bool early = false;
    if (nanos != 0 && nanos <= MAX_WAKEABLE_NANOS && !state.sleeping && Enabled()) {
        if (TimedWait::NowNanos() - state.lastSleepEndNanos >= nanos) state.earlyWakes = 0;
        if (state.earlyWakes < EARLY_WAKES_PER_RUN) {
            state.sleeping = true;
            early = TimedWait::SleepNanosUnlessChanged(nanos, completions, completions.load(std::memory_order_seq_cst));
            state.sleeping = false;
            if (early) ++state.earlyWakes;
            woken = true;
        }
    }
    if (!woken) TimedWait::SleepNanos(nanos);
    state.lastSleepEndNanos = TimedWait::NowNanos();
    return early;
}

}

void Sleep(std::uint64_t nanos) {
    const bool guest = TimedWait::FollowsGuestClock();
    const std::uint64_t start = guest ? TimedWait::ClockNanos() : 0;
    if (SleepOnce(nanos) || !guest) return;
    const std::uint64_t deadline = nanos > std::numeric_limits<std::uint64_t>::max() - start ? std::numeric_limits<std::uint64_t>::max() : start + nanos;
    for (std::uint64_t wait = TimedWait::WaitNanos(deadline); wait != 0; wait = TimedWait::WaitNanos(deadline)) {
        if (SleepOnce(wait)) return;
    }
}

}

extern "C" {

void KernelWakeSleepers_nid_no_patch() {
    SleepWake::completions.fetch_add(1, std::memory_order_seq_cst);
    TimedWait::WakeChangeSleepers();
}

void KernelNoteGpuSubmit_nid_no_patch() {
    SleepWake::State().earlyWakes = 0;
}

}
