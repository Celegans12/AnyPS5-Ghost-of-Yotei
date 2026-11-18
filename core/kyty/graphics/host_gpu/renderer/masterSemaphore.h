#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#if defined(ANYPS5_KYTY)
#include <array>
#endif

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

	void Refresh();
	void Wait(uint64_t tick);
#if defined(ANYPS5_KYTY)
	void NoteWork(uint64_t tick, uint64_t shader) noexcept {
		const auto next = m_work_next.load(std::memory_order_relaxed);
		const auto at   = next % m_work.size();
		m_work_next.store(next + 1, std::memory_order_relaxed);
		m_work[at].tick.store(tick, std::memory_order_relaxed);
		m_work[at].shader.store(shader, std::memory_order_relaxed);
	}
#endif

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
#if defined(ANYPS5_KYTY)
	struct Work {
		std::atomic<uint64_t> tick {0};
		std::atomic<uint64_t> shader {0};
	};
	std::array<Work, 512> m_work {};
	std::atomic<uint32_t> m_work_next {0};
	void ReportStall(uint64_t tick);
#endif
};

}

#endif
