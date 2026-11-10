#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	explicit MasterSemaphore(GraphicContext& graphics);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		return m_gpu_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }

	// Refresh queries the timeline semaphore (a kernel call) at most once per 100 microseconds; in between it keeps the
	// last known GPU tick, which only makes IsFree conservative for that long. RefreshNow and Wait always query.
	void Refresh();
	void RefreshNow();
	void Wait(uint64_t tick);

	// KYTY_TRACE_PERF (PERF semaphore): semaphore queries (calls/ns) by caller: 0 Refresh, 1 Wait before blocking, 2 Wait
	// after blocking, 3 other; blocking waits (calls/ns). GPU thread only; reset by TakeStats.
	struct Stats {
		std::atomic<uint64_t> queries[4] {}, query_ns[4] {}, blocks {0}, block_ns {0};
	};
	static Stats& GetStats();

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::atomic<int64_t>  m_last_refresh_ns {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
