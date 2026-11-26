#ifndef CORE_LIBS_PRX_LIBKERNEL_TIME_INCLUDE_TIMEDWAIT_HPP
#define CORE_LIBS_PRX_LIBKERNEL_TIME_INCLUDE_TIMEDWAIT_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#elif !defined(_WIN32)
#include <semaphore.h>
#endif

namespace TimedWait {

struct GuestClockSource {
    std::uint64_t (*now)();
    bool (*paused)();
    bool (*initialThread)();
};

bool Coarse();
std::uint64_t TimerLateNanos();
void BindThreadWaitState(std::atomic<int>* state);
void SetGuestClockSource(const GuestClockSource* source);
void FollowGuestClock();
bool GuestClockPaused();
bool FollowsGuestClock();
std::uint64_t NowNanos();
std::uint64_t GuestNanos();
std::uint64_t ClockNanos();
std::uint64_t WaitNanos(std::uint64_t deadlineNanos);
std::uint64_t WaitMicros(std::uint64_t deadlineNanos);
std::uint64_t GuestSpanWaitNanos(std::uint64_t remainingNanos);
std::uint64_t DeadlineNanos(std::uint64_t microseconds);
std::uint64_t RemainingMicros(std::uint64_t deadlineNanos);

class CivilClockWait {
public:
    CivilClockWait() = default;
    explicit CivilClockWait(bool civil);
    ~CivilClockWait();
    CivilClockWait(const CivilClockWait&) = delete;
    CivilClockWait& operator=(const CivilClockWait&) = delete;
    void Engage();

private:
    int saved = 0;
    bool engaged = false;
};
void SleepNanos(std::uint64_t nanos);
void SleepUntil(std::uint64_t deadlineNanos);
bool SleepNanosUnlessChanged(std::uint64_t nanos, const std::atomic<std::uint64_t>& counter, std::uint64_t observed);
void WakeChangeSleepers();

struct Waiter;

class DeliveryDeferral {
public:
    DeliveryDeferral();
    ~DeliveryDeferral();
    DeliveryDeferral(const DeliveryDeferral&) = delete;
    DeliveryDeferral& operator=(const DeliveryDeferral&) = delete;
};

void DeliverDeferred();

#ifndef _WIN32
void HoldDelivery();
void ReleaseDelivery();

template <class Mutex>
class DeferredDeliveryLock {
public:
    explicit DeferredDeliveryLock(Mutex& mutex) : lock(mutex) {}
    DeferredDeliveryLock(const DeferredDeliveryLock&) = delete;
    DeferredDeliveryLock& operator=(const DeferredDeliveryLock&) = delete;

private:
    struct Hold {
        Hold() { HoldDelivery(); }
        ~Hold() {
            ReleaseDelivery();
            DeliverDeferred();
        }
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;
    };

    Hold hold;
    std::lock_guard<Mutex> lock;
};

bool DeliveryAllowed();
bool DeliveryDeferred();

class DeliverableWait {
public:
    DeliverableWait();
    ~DeliverableWait();
    DeliverableWait(const DeliverableWait&) = delete;
    DeliverableWait& operator=(const DeliverableWait&) = delete;
};
#else
class DeliverableWait {
public:
    DeliverableWait() = default;
    DeliverableWait(const DeliverableWait&) = delete;
    DeliverableWait& operator=(const DeliverableWait&) = delete;
};
#endif

#ifdef _WIN32
void QueueForDeliveryDrain(unsigned long thread, const std::atomic<bool>* targetFinished, void (*run)(void*), void* argument);
void ReapDeliveryDrain();
std::size_t DeliveryDrainPending();
void SetDeliveryDrainPostHook(void (*hook)());

constexpr unsigned PARK_SPINS = 1000;

void SpinPause();
Waiter* BeginPark(const void* key);
bool FinishPark(const void* key, Waiter* waiter, const std::uint64_t* deadlineNanos);
void CancelPark(const void* key, Waiter* waiter);
void Unpark(const void* key);

template <class TTryLock>
bool ParkAcquire(const void* key, TTryLock tryLock, const std::uint64_t* deadlineNanos) {
    for (unsigned spin = 0; spin < PARK_SPINS; ++spin) {
        if (tryLock()) return true;
        SpinPause();
    }
    for (;;) {
        Waiter* waiter = BeginPark(key);
        if (tryLock()) {
            CancelPark(key, waiter);
            return true;
        }
        if (!FinishPark(key, waiter, deadlineNanos)) return tryLock();
        if (tryLock()) return true;
    }
}
#endif

template <class TTryLock, class TTryLockFor>
bool AcquireUntil(const void* key, std::uint64_t deadlineNanos, TTryLock tryLock, TTryLockFor tryLockFor) {
#ifdef _WIN32
    if (!Coarse()) return ParkAcquire(key, tryLock, &deadlineNanos);
#endif
    static_cast<void>(key);
    static_cast<void>(tryLock);
    for (;;) {
        if (tryLockFor(WaitMicros(deadlineNanos))) return true;
        if (ClockNanos() >= deadlineNanos) return false;
    }
}

class Condition {
public:
    Condition() = default;
    ~Condition() = default;
    Condition(const Condition&) = delete;
    Condition& operator=(const Condition&) = delete;

    void NotifyOne();
    void NotifyAll();

    template <class TLock>
    void Wait(TLock& lock) {
#ifdef _WIN32
        if (!Coarse()) {
            Waiter* waiter = enqueue();
            lock.unlock();
            waitSignal(waiter);
            lock.lock();
            return;
        }
#endif
        CoarseWaiter registered{coarseWaiters};
        coarse.wait(lock);
    }

    template <class TLock>
    bool WaitUntil(TLock& lock, std::uint64_t deadlineNanos) {
#ifdef _WIN32
        if (!Coarse()) {
            Waiter* waiter = enqueue();
            lock.unlock();
            const bool signaled = waitSignalUntil(waiter, deadlineNanos);
            lock.lock();
            return signaled;
        }
#endif
        CoarseWaiter registered{coarseWaiters};
        for (;;) {
            if (coarse.wait_for(lock, std::chrono::microseconds(WaitMicros(deadlineNanos))) == std::cv_status::no_timeout) return true;
            if (ClockNanos() >= deadlineNanos) return false;
        }
    }

    template <class TLock, class TPredicate>
    void Wait(TLock& lock, TPredicate predicate) {
        while (!predicate()) Wait(lock);
    }

    template <class TLock, class TPredicate>
    bool WaitUntil(TLock& lock, std::uint64_t deadlineNanos, TPredicate predicate) {
        while (!predicate()) {
            if (!WaitUntil(lock, deadlineNanos)) return predicate();
        }
        return true;
    }

private:
    struct CoarseWaiter {
        std::atomic<std::uint32_t>& count;
        explicit CoarseWaiter(std::atomic<std::uint32_t>& counter) : count(counter) { count.fetch_add(1, std::memory_order_seq_cst); }
        ~CoarseWaiter() { count.fetch_sub(1, std::memory_order_release); }
    };

#ifdef _WIN32
    Waiter* enqueue();
    void unlink(Waiter* waiter);
    void waitSignal(Waiter* waiter);
    bool waitSignalUntil(Waiter* waiter, std::uint64_t deadlineNanos);

    std::mutex queueLock;
    Waiter* head = nullptr;
    Waiter* tail = nullptr;
#endif
    std::atomic<std::uint32_t> coarseWaiters{0};
    std::condition_variable_any coarse;
};

#ifdef _WIN32
using SingleWaiterCondition = Condition;
#else
bool DeferDelivery(int signum);

class SingleWaiterCondition {
public:
    SingleWaiterCondition();
    ~SingleWaiterCondition();
    SingleWaiterCondition(const SingleWaiterCondition&) = delete;
    SingleWaiterCondition& operator=(const SingleWaiterCondition&) = delete;

    void NotifyOne();

    template <class TLock, class TPredicate>
    void Wait(TLock& lock, TPredicate predicate) {
        while (!predicate()) {
            lock.unlock();
            waitSignal(nullptr);
            lock.lock();
        }
    }

    template <class TLock, class TPredicate>
    bool WaitUntil(TLock& lock, std::uint64_t deadlineNanos, TPredicate predicate) {
        while (!predicate()) {
            lock.unlock();
            const bool signaled = waitSignal(&deadlineNanos);
            lock.lock();
            if (!signaled) return predicate();
        }
        return true;
    }

private:
    bool waitSignal(const std::uint64_t* deadlineNanos);

#ifdef __APPLE__
    dispatch_semaphore_t semaphore;
#else
    sem_t semaphore;
#endif
};
#endif

}  // namespace TimedWait

#endif
