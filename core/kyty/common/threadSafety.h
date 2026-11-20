#ifndef KYTY_COMMON_THREADSAFETY_H_
#define KYTY_COMMON_THREADSAFETY_H_

#include <mutex>

#if defined(__clang__)
#define KYTY_THREAD_ANNOTATION(x) __attribute__((x))
#else
#define KYTY_THREAD_ANNOTATION(x)
#endif

#define KYTY_CAPABILITY(name)          KYTY_THREAD_ANNOTATION(capability(name))
#define KYTY_SCOPED_CAPABILITY         KYTY_THREAD_ANNOTATION(scoped_lockable)
#define KYTY_GUARDED_BY(x)             KYTY_THREAD_ANNOTATION(guarded_by(x))
#define KYTY_PT_GUARDED_BY(x)          KYTY_THREAD_ANNOTATION(pt_guarded_by(x))
#define KYTY_ACQUIRED_BEFORE(...)      KYTY_THREAD_ANNOTATION(acquired_before(__VA_ARGS__))
#define KYTY_ACQUIRED_AFTER(...)       KYTY_THREAD_ANNOTATION(acquired_after(__VA_ARGS__))
#define KYTY_REQUIRES(...)             KYTY_THREAD_ANNOTATION(requires_capability(__VA_ARGS__))
#define KYTY_REQUIRES_SHARED(...)      KYTY_THREAD_ANNOTATION(requires_shared_capability(__VA_ARGS__))
#define KYTY_ACQUIRE(...)              KYTY_THREAD_ANNOTATION(acquire_capability(__VA_ARGS__))
#define KYTY_ACQUIRE_SHARED(...)       KYTY_THREAD_ANNOTATION(acquire_shared_capability(__VA_ARGS__))
#define KYTY_RELEASE(...)              KYTY_THREAD_ANNOTATION(release_capability(__VA_ARGS__))
#define KYTY_RELEASE_SHARED(...)       KYTY_THREAD_ANNOTATION(release_shared_capability(__VA_ARGS__))
#define KYTY_TRY_ACQUIRE(...)          KYTY_THREAD_ANNOTATION(try_acquire_capability(__VA_ARGS__))
#define KYTY_EXCLUDES(...)             KYTY_THREAD_ANNOTATION(locks_excluded(__VA_ARGS__))
#define KYTY_ASSERT_CAPABILITY(x)      KYTY_THREAD_ANNOTATION(assert_capability(x))
#define KYTY_RETURN_CAPABILITY(x)      KYTY_THREAD_ANNOTATION(lock_returned(x))
#define KYTY_NO_THREAD_SAFETY_ANALYSIS KYTY_THREAD_ANNOTATION(no_thread_safety_analysis)

namespace Common {

class KYTY_CAPABILITY("mutex") AnnotatedMutex {
public:
	AnnotatedMutex() = default;

	AnnotatedMutex(const AnnotatedMutex&)            = delete;
	AnnotatedMutex& operator=(const AnnotatedMutex&) = delete;

	void lock() KYTY_ACQUIRE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_mutex.lock(); }
	void unlock() KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_mutex.unlock(); }
	[[nodiscard]] bool try_lock() KYTY_TRY_ACQUIRE(true) KYTY_NO_THREAD_SAFETY_ANALYSIS { return m_mutex.try_lock(); }

	[[nodiscard]] std::mutex& Native() noexcept { return m_mutex; }

private:
	std::mutex m_mutex;
};

template <typename Lockable>
class KYTY_SCOPED_CAPABILITY ScopedLock {
public:
	explicit ScopedLock(Lockable& lockable) KYTY_ACQUIRE(lockable) KYTY_NO_THREAD_SAFETY_ANALYSIS: m_lockable(lockable) {
		m_lockable.lock();
	}
	~ScopedLock() KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_lockable.unlock(); }

	ScopedLock(const ScopedLock&)            = delete;
	ScopedLock& operator=(const ScopedLock&) = delete;

private:
	Lockable& m_lockable;
};

template <typename Lockable>
ScopedLock(Lockable&) -> ScopedLock<Lockable>;

class KYTY_SCOPED_CAPABILITY UniqueLock {
public:
	explicit UniqueLock(AnnotatedMutex& mutex) KYTY_ACQUIRE(mutex) KYTY_NO_THREAD_SAFETY_ANALYSIS
	    : m_lock(mutex.Native()) {}
	~UniqueLock() KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS = default;

	UniqueLock(const UniqueLock&)            = delete;
	UniqueLock& operator=(const UniqueLock&) = delete;

	void lock() KYTY_ACQUIRE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_lock.lock(); }
	void unlock() KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS { m_lock.unlock(); }
	[[nodiscard]] std::unique_lock<std::mutex>& Native() noexcept { return m_lock; }

private:
	std::unique_lock<std::mutex> m_lock;
};

}

#endif
