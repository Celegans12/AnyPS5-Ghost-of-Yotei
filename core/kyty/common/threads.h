#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"
#include "common/threadSafety.h"

#include <span>

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <condition_variable>
#include <mutex>
#endif

namespace Common {

void InitializeThreads();

using wait_poll_func_t = void (*)();

namespace Thread {

void SleepMicro(uint32_t micros);
void SleepNano(uint64_t nanos);
bool IsMainThread();
int GetThreadIdUnique();

void RaiseCurrentPriority();
void LowerCurrentPriority();
void PinCurrent(int logical_cpu);

void ReserveCores(std::span<const int> logical_cpus);
bool HasReservedCore();

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool LogicalCpuExists(int logical_cpu);
#endif

#if defined(ANYPS5_KYTY)
void KeepCurrentOnLargestCache();
#endif

#if defined(ANYPS5_KYTY) && KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool PinCurrentToOwnCore(uint32_t index, uint32_t count);
#endif

}

class KYTY_CAPABILITY("mutex") Mutex {
public:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Mutex();
	~Mutex();

	void Lock() KYTY_ACQUIRE();
	void Unlock() KYTY_RELEASE();

	KYTY_CLASS_NO_COPY(Mutex);

private:
	alignas(8) unsigned char m_cs[40];
#else
	void Lock() KYTY_ACQUIRE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_mutex.lock(); }
	void Unlock() KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_mutex.unlock(); }

private:
	std::recursive_mutex m_mutex;
#endif

	friend class CondVar;
};

class CondVar {
public:
	CondVar() = default;

	void Wait(Mutex* mutex) KYTY_NO_THREAD_SAFETY_ANALYSIS;
	bool WaitFor(Mutex* mutex, uint32_t micros) KYTY_NO_THREAD_SAFETY_ANALYSIS;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void Signal();
	void SignalAll();
#else
	void Signal() { m_cv.notify_one(); }
	void SignalAll() { m_cv.notify_all(); }
#endif

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void* m_cv = nullptr;
#else
	std::condition_variable_any m_cv;
#endif
};

class KYTY_SCOPED_CAPABILITY LockGuard {
public:
	using mutex_type = Mutex;

	explicit LockGuard(mutex_type& m) KYTY_ACQUIRE(m) KYTY_NO_THREAD_SAFETY_ANALYSIS: m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

}

#endif
