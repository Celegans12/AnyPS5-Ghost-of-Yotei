#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EXECHELPERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EXECHELPERS_H_

#include "common/common.h"
#include "common/spscQueue.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class ExecHelpers {
public:
	static constexpr uint32_t Helpers = 3;

	ExecHelpers() {
		m_threads.reserve(Helpers);
		for (uint32_t i = 0; i < Helpers; i++) {
			m_threads.emplace_back([this] { Run(); });
		}
	}
	~ExecHelpers() {
		m_stop.store(true, std::memory_order_relaxed);
		m_generation.fetch_add(1, std::memory_order_release);
		m_generation.notify_all();
		for (auto& thread: m_threads) {
			thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(ExecHelpers);

	template <typename Func>
	void ParallelFor(uint32_t count, uint32_t grain, Func&& func) {
		if (count <= grain) {
			if (count != 0) {
				func(0u, count);
			}
			return;
		}
		m_call = [](void* context, uint32_t begin, uint32_t end) {
			(*static_cast<Func*>(context))(begin, end);
		};
		m_context = &func;
		m_count   = count;
		m_grain   = grain;
		m_next.store(0, std::memory_order_relaxed);
		m_finished.store(0, std::memory_order_relaxed);
		m_generation.fetch_add(1, std::memory_order_seq_cst);
		if (m_sleepers.load(std::memory_order_seq_cst) != 0) {
			m_generation.notify_all();
		}
		Work();
		for (Common::SpinWait spin; m_finished.load(std::memory_order_acquire) != Helpers;) {
			if (!spin.Spin()) {
				std::this_thread::yield();
			}
		}
	}

private:
	void Work() {
		for (;;) {
			const auto chunk = m_next.fetch_add(1, std::memory_order_relaxed);
			const auto begin = static_cast<uint64_t>(chunk) * m_grain;
			if (begin >= m_count) {
				return;
			}
			const auto end = std::min<uint64_t>(begin + m_grain, m_count);
			m_call(m_context, static_cast<uint32_t>(begin), static_cast<uint32_t>(end));
		}
	}

	void Run() {
		uint64_t seen = 0;
		for (;;) {
			uint64_t generation = m_generation.load(std::memory_order_acquire);
			if (generation == seen) {
				Common::SpinWait spin;
				while ((generation = m_generation.load(std::memory_order_acquire)) == seen) {
					if (!spin.Spin()) {
						m_sleepers.fetch_add(1, std::memory_order_seq_cst);
						generation = m_generation.load(std::memory_order_seq_cst);
						if (generation == seen) {
							m_generation.wait(seen, std::memory_order_seq_cst);
						}
						m_sleepers.fetch_sub(1, std::memory_order_relaxed);
						spin = Common::SpinWait {};
					}
				}
			}
			seen = generation;
			if (m_stop.load(std::memory_order_relaxed)) {
				return;
			}
			Work();
			m_finished.fetch_add(1, std::memory_order_release);
		}
	}

	void (*m_call)(void*, uint32_t, uint32_t) = nullptr;
	void*    m_context                         = nullptr;
	uint32_t m_count                           = 0;
	uint32_t m_grain                           = 1;
	alignas(64) std::atomic_uint32_t m_next {0};
	alignas(64) std::atomic_uint32_t m_finished {0};
	alignas(64) std::atomic_uint64_t m_generation {0};
	alignas(64) std::atomic_uint32_t m_sleepers {0};
	std::atomic_bool         m_stop {false};
	std::vector<std::thread> m_threads;
};

inline ExecHelpers& GetExecHelpers() {
	static ExecHelpers helpers;
	return helpers;
}

inline bool ParallelEqual(const void* a, const void* b, uint64_t size) {
	constexpr uint64_t Chunk = uint64_t {64} * 1024;
	if (size < 4 * Chunk) {
		return std::memcmp(a, b, size) == 0;
	}
	std::atomic_bool differ {false};
	GetExecHelpers().ParallelFor(static_cast<uint32_t>((size + Chunk - 1) / Chunk), 1, [&](uint32_t begin, uint32_t end) {
		for (auto chunk = begin; chunk < end && !differ.load(std::memory_order_relaxed); chunk++) {
			const auto offset = uint64_t {chunk} * Chunk;
			if (std::memcmp(static_cast<const uint8_t*>(a) + offset, static_cast<const uint8_t*>(b) + offset,
			                std::min(Chunk, size - offset)) != 0) {
				differ.store(true, std::memory_order_relaxed);
			}
		}
	});
	return !differ.load(std::memory_order_relaxed);
}

}

#endif
