#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_QUEUELOCKSITE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_QUEUELOCKSITE_H_

#include "common/common.h"
#include "common/threads.h"

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

// Who holds GraphicContext::queue_mutex, and since when (diagnostic): the recording thread's stall warning prints it
// (commandRecorder.cpp, "RECORD_THREAD warning"), so a submit that cannot take the queue lock names the holder.
struct QueueLockSite {
	static inline std::atomic<const char*> site {nullptr};
	static inline std::atomic<int64_t>     since_ns {0};

	static int64_t NowNs() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
		    .count();
	}
};

// Common::LockGuard on the queue lock that also records the site while it is held.
class QueueLockGuard {
public:
	QueueLockGuard(Common::Mutex& mutex, const char* site): m_guard(mutex) {
		QueueLockSite::since_ns.store(QueueLockSite::NowNs(), std::memory_order_relaxed);
		QueueLockSite::site.store(site, std::memory_order_release);
	}
	~QueueLockGuard() { QueueLockSite::site.store(nullptr, std::memory_order_release); }

	KYTY_CLASS_NO_COPY(QueueLockGuard);

private:
	Common::LockGuard m_guard;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_QUEUELOCKSITE_H_ */
