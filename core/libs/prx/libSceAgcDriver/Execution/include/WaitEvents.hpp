#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WAITEVENTS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_WAITEVENTS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#include <windows.h>
#else
#include <condition_variable>
#include <mutex>
#endif

// What a blocked queue worker sleeps on instead of a timed poll. A worker registers a Listener for
// the events that can end its wait, reads Token(), evaluates its condition and then calls Wait with
// that token: an event signaled after the token was read ends the wait at once, so none is lost.
// The timeout is the deadline of time-based work (a label flush) or the fallback for values that
// arrive without an event (a guest CPU store).
namespace AgcDriver::WaitEvents {

enum Kind : std::uint32_t {
    // A label store was noted, or a GPU write was recorded, over the listener's awaited range.
    Range = 1u << 0,
    // The GPU finished a batch, or completion actions of finished batches ran.
    Progress = 1u << 1,
    // Driver queue state changed (what Driver::changed announces, and the order-wait inputs).
    DriverState = 1u << 2,
    // The outermost hold of the GPU mutex ended; only for a listener whose try just failed.
    MutexFree = 1u << 3,
    // The label table changed in a way no range names (a label group closed).
    Labels = 1u << 4,
};

namespace Detail {

struct Slot {
    std::atomic<std::uint32_t> kinds{0};
    std::atomic<std::uint64_t> begin{0}, end{0};
    std::atomic<std::uint32_t> signals{0};
    // The kinds signaled since the listener last asked (TakeArrived): for the profile only.
    std::atomic<std::uint32_t> arrived{0};
    std::atomic<bool> sleeping{false};
    std::atomic<bool> taken{false};
#ifdef _WIN32
    HANDLE event = nullptr;
#else
    std::mutex mutex;
    std::condition_variable wake;
#endif
};

inline constexpr std::size_t SlotCount = 32;
inline std::array<Slot, SlotCount> slots;
inline std::atomic<std::uint32_t> listeners{0};
inline std::atomic<std::uint32_t> mutexWanted{0};

inline void Wake(Slot& slot, std::uint32_t kind) {
    slot.arrived.fetch_or(kind, std::memory_order_relaxed);
    slot.signals.fetch_add(1, std::memory_order_seq_cst);
    if (!slot.sleeping.load(std::memory_order_seq_cst)) return;
#ifdef _WIN32
    SetEvent(slot.event);
#else
    {
        std::lock_guard lock(slot.mutex);
    }
    slot.wake.notify_one();
#endif
}

}

// Wakes the listeners of `kind`. Cheap when nobody listens (one load).
inline void Signal(Kind kind) {
    if (Detail::listeners.load(std::memory_order_seq_cst) == 0) return;
    for (auto& slot : Detail::slots) {
        if ((slot.kinds.load(std::memory_order_seq_cst) & kind) != 0) Detail::Wake(slot, kind);
    }
}

// Wakes the Range listeners whose awaited range overlaps [address, address + bytes).
inline void SignalRange(std::uint64_t address, std::uint64_t bytes) {
    if (Detail::listeners.load(std::memory_order_seq_cst) == 0) return;
    for (auto& slot : Detail::slots) {
        if ((slot.kinds.load(std::memory_order_seq_cst) & Range) == 0) continue;
        if (address < slot.end.load(std::memory_order_seq_cst) && slot.begin.load(std::memory_order_seq_cst) < address + bytes) Detail::Wake(slot, Range);
    }
}

// Announces a driver state change to both kinds of sleeper: those on the driver's condition
// variable and the event listeners.
template <class Condition>
inline void NotifyState(Condition& changed) {
    changed.notify_all();
    Signal(DriverState);
}

// Called by GpuMutexType::unlock after an outermost release.
inline void SignalMutexFree() {
    if (Detail::mutexWanted.load(std::memory_order_seq_cst) != 0) Signal(MutexFree);
}

class Listener {
public:
    Listener(std::uint32_t kinds, std::uint64_t address = 0, std::uint64_t bytes = 0) {
        for (auto& candidate : Detail::slots) {
            bool expected = false;
            if (!candidate.taken.compare_exchange_strong(expected, true, std::memory_order_seq_cst)) continue;
            slot = &candidate;
            break;
        }
        if (slot == nullptr) return;
#ifdef _WIN32
        if (slot->event == nullptr) slot->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        // One timer per thread, kept: a listener is made for every wait that outlasts its spin.
        thread_local const HANDLE threadTimer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        timer = threadTimer;
        if (slot->event == nullptr || timer == nullptr) {
            timer = nullptr;
            slot->taken.store(false, std::memory_order_seq_cst);
            slot = nullptr;
            return;
        }
#endif
        slot->begin.store(address, std::memory_order_seq_cst);
        slot->end.store(address + bytes, std::memory_order_seq_cst);
        Detail::listeners.fetch_add(1, std::memory_order_seq_cst);
        slot->kinds.store(kinds, std::memory_order_seq_cst);
    }

    ~Listener() {
        if (slot == nullptr) return;
        if (wantsMutex) Detail::mutexWanted.fetch_sub(1, std::memory_order_seq_cst);
        slot->kinds.store(0, std::memory_order_seq_cst);
        Detail::listeners.fetch_sub(1, std::memory_order_seq_cst);
        slot->taken.store(false, std::memory_order_seq_cst);
    }

    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;

    std::uint32_t Token() const { return slot != nullptr ? slot->signals.load(std::memory_order_seq_cst) : 0; }

    // The kinds signaled since the last call (profile output).
    std::uint32_t TakeArrived() { return slot != nullptr ? slot->arrived.exchange(0, std::memory_order_relaxed) : 0; }

    // Turns one kind on or off. Returns true when it turned the kind on: an event of that kind
    // before this call was not counted, so the caller checks its condition again before it waits.
    bool Listen(Kind kind, bool on) {
        if (slot == nullptr) return false;
        const bool was = (slot->kinds.load(std::memory_order_seq_cst) & kind) != 0;
        if (was == on) return false;
        if (on) slot->kinds.fetch_or(kind, std::memory_order_seq_cst);
        else slot->kinds.fetch_and(~static_cast<std::uint32_t>(kind), std::memory_order_seq_cst);
        return on;
    }

    // Registers (or drops) this listener's interest in the GPU mutex becoming free. Call it before
    // the try that decides whether to wait, so a release after that try is signaled.
    void WantMutex(bool wanted) {
        if (slot == nullptr || wanted == wantsMutex) return;
        wantsMutex = wanted;
        if (wanted) {
            Detail::mutexWanted.fetch_add(1, std::memory_order_seq_cst);
            slot->kinds.fetch_or(MutexFree, std::memory_order_seq_cst);
        } else {
            slot->kinds.fetch_and(~static_cast<std::uint32_t>(MutexFree), std::memory_order_seq_cst);
            Detail::mutexWanted.fetch_sub(1, std::memory_order_seq_cst);
        }
    }

    // Blocks until an event was signaled after `token` was read, or `timeout` passed. Returns true
    // for an event, false for the timeout.
    bool Wait(std::uint32_t token, std::chrono::microseconds timeout) {
        if (slot == nullptr) {
            // No slot (more listeners than slots): the timed sleep this replaces.
            SleepFor(std::min(timeout, std::chrono::microseconds(200)));
            return false;
        }
#ifdef _WIN32
        ResetEvent(slot->event);
        slot->sleeping.store(true, std::memory_order_seq_cst);
        bool signaled = slot->signals.load(std::memory_order_seq_cst) != token;
        if (!signaled) {
            LARGE_INTEGER due{};
            due.QuadPart = -static_cast<LONGLONG>(std::max<std::int64_t>(timeout.count(), 1) * 10);
            if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
                const HANDLE handles[2] = {slot->event, timer};
                signaled = WaitForMultipleObjects(2, handles, FALSE, INFINITE) == WAIT_OBJECT_0;
                if (signaled) CancelWaitableTimer(timer);
            } else {
                signaled = WaitForSingleObject(slot->event, static_cast<DWORD>(std::max<std::int64_t>(timeout.count() / 1000, 1))) == WAIT_OBJECT_0;
            }
        }
        slot->sleeping.store(false, std::memory_order_seq_cst);
        return signaled || slot->signals.load(std::memory_order_seq_cst) != token;
#else
        std::unique_lock lock(slot->mutex);
        slot->sleeping.store(true, std::memory_order_seq_cst);
        const bool signaled = slot->wake.wait_for(lock, timeout, [&] { return slot->signals.load(std::memory_order_seq_cst) != token; });
        slot->sleeping.store(false, std::memory_order_seq_cst);
        return signaled;
#endif
    }

private:
    static void SleepFor(std::chrono::microseconds time) {
#ifdef _WIN32
        thread_local HANDLE sleepTimer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        LARGE_INTEGER due{};
        due.QuadPart = -static_cast<LONGLONG>(std::max<std::int64_t>(time.count(), 1) * 10);
        if (sleepTimer != nullptr && SetWaitableTimer(sleepTimer, &due, 0, nullptr, nullptr, FALSE)) {
            WaitForSingleObject(sleepTimer, INFINITE);
            return;
        }
        Sleep(1);
#else
        std::mutex none;
        std::condition_variable never;
        std::unique_lock lock(none);
        never.wait_for(lock, time);
#endif
    }

    Detail::Slot* slot = nullptr;
    bool wantsMutex = false;
#ifdef _WIN32
    HANDLE timer = nullptr;
#endif
};

}

#endif
