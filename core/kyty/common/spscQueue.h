#ifndef KYTY_COMMON_SPSCQUEUE_H_
#define KYTY_COMMON_SPSCQUEUE_H_

#include "common/assert.h"
#include "common/common.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

inline void SpinPause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#endif
}

inline std::chrono::nanoseconds SpinBudget() noexcept {
	static const auto budget = [] {
		uint64_t micros = 2000;
		if (const char* value = std::getenv("KYTY_GPU_SPIN_US"); value != nullptr) {
			micros = std::strtoull(value, nullptr, 10);
		}
		return std::chrono::nanoseconds(micros * 1000u);
	}();
	return budget;
}

class SpinWait {
public:
	SpinWait() noexcept: m_budget(SpinBudget()) {}

	bool Spin() noexcept {
		for (uint32_t i = 0; i < m_pauses; i++) {
			SpinPause();
		}
		m_iterations += m_pauses;
		if (m_pauses < MaxPauses) {
			m_pauses <<= 1u;
		}
		if ((++m_rounds & ClockMask) != 0) {
			return true;
		}
		const auto now = std::chrono::steady_clock::now();
		if (m_rounds == ClockMask + 1) {
			m_start = now;
			return true;
		}
		return now - m_start < m_budget;
	}

	[[nodiscard]] uint64_t Iterations() const noexcept { return m_iterations; }

private:
	static constexpr uint32_t MaxPauses = 64;
	static constexpr uint32_t ClockMask = 15;

	std::chrono::nanoseconds              m_budget;
	std::chrono::steady_clock::time_point m_start {};
	uint32_t                              m_pauses     = 1;
	uint32_t                              m_rounds     = 0;
	uint64_t                              m_iterations = 0;
};

class ProgressCounter {
public:
	ProgressCounter() = default;
	KYTY_CLASS_NO_COPY(ProgressCounter);

	[[nodiscard]] uint64_t Load() const noexcept { return m_value.load(std::memory_order_acquire); }

	void Publish(uint64_t value) noexcept { m_value.store(value, std::memory_order_release); }

	void Notify() noexcept {
		std::atomic_thread_fence(std::memory_order_seq_cst);
		auto registered = m_wait_target.load(std::memory_order_relaxed);
		if (registered <= m_value.load(std::memory_order_relaxed)) {
			(void)m_wait_target.compare_exchange_strong(registered, NoTarget,
			                                            std::memory_order_seq_cst);
			m_wakeups.fetch_add(1, std::memory_order_seq_cst);
			m_wakeups.notify_all();
		}
	}

	void Advance(uint64_t value) noexcept {
		Publish(value);
		Notify();
	}

	void WaitFor(uint64_t target) noexcept {
		if (m_value.load(std::memory_order_acquire) >= target) {
			return;
		}
		SpinWait spin;
		while (spin.Spin()) {
			if (m_value.load(std::memory_order_acquire) >= target) {
				return;
			}
		}
		for (;;) {
			const auto wakeups = m_wakeups.load(std::memory_order_seq_cst);
			auto registered = m_wait_target.load(std::memory_order_seq_cst);
			while (registered > target &&
			       !m_wait_target.compare_exchange_weak(registered, target,
			                                            std::memory_order_seq_cst)) {
			}
			if (m_value.load(std::memory_order_seq_cst) >= target) {
				break;
			}
			m_wakeups.wait(wakeups, std::memory_order_seq_cst);
		}
	}

private:
	static constexpr uint64_t NoTarget = ~uint64_t {0};

	alignas(64) std::atomic_uint64_t m_value {0};
	alignas(64) std::atomic_uint64_t m_wait_target {NoTarget};
	std::atomic_uint32_t             m_wakeups {0};
};

template <typename T>
class SpscQueue {
public:
	explicit SpscQueue(size_t capacity)
	    : m_capacity(std::bit_ceil(capacity)), m_mask(m_capacity - 1),
	      m_slots(std::make_unique<Slot[]>(m_capacity)) {
		EXIT_IF(capacity == 0);
	}
	~SpscQueue() {
		const auto tail = m_tail.load(std::memory_order_acquire);
		for (auto head = m_head.load(std::memory_order_acquire); head != tail; head++) {
			Element(head)->~T();
		}
	}
	KYTY_CLASS_NO_COPY(SpscQueue);

	void Push(T&& value) {
		const auto tail = m_tail.load(std::memory_order_relaxed);
		if (tail - m_producer_head >= m_capacity) {
			m_producer_head = m_head.load(std::memory_order_acquire);
			if (tail - m_producer_head >= m_capacity) {
				WaitForSpace(tail);
			}
		}
		Construct(tail, std::move(value));
	}

	bool TryPush(T& value) {
		const auto tail = m_tail.load(std::memory_order_relaxed);
		if (tail - m_producer_head >= m_capacity) {
			m_producer_head = m_head.load(std::memory_order_acquire);
			if (tail - m_producer_head >= m_capacity) {
				return false;
			}
		}
		Construct(tail, std::move(value));
		return true;
	}

	[[nodiscard]] std::optional<T> Pop() {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (m_consumer_tail == head) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail == head && !WaitForElement(head)) {
				return std::nullopt;
			}
		}
		return Take(head);
	}

	[[nodiscard]] std::optional<T> TryPop() {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (m_consumer_tail == head) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail == head) {
				return std::nullopt;
			}
		}
		return Take(head);
	}

	[[nodiscard]] const T* TryPeek() {
		const auto head = m_head.load(std::memory_order_relaxed);
		if (m_consumer_tail == head) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail == head) {
				return nullptr;
			}
		}
		return Element(head);
	}

	void Stop() {
		m_stopped.store(true, std::memory_order_seq_cst);
		m_consumer_signal.fetch_add(1, std::memory_order_seq_cst);
		m_consumer_signal.notify_all();
	}

	void Interrupt() {
		m_interrupted.store(true, std::memory_order_seq_cst);
		m_consumer_signal.fetch_add(1, std::memory_order_seq_cst);
		m_consumer_signal.notify_all();
	}

	[[nodiscard]] bool IsStopped() const noexcept {
		return m_stopped.load(std::memory_order_acquire);
	}

	[[nodiscard]] size_t Capacity() const noexcept { return m_capacity; }

private:
	static constexpr uint32_t CheckInterval = 16;

	struct alignas(alignof(T) > 64 ? alignof(T) : 64) Slot {
		std::byte storage[sizeof(T)];
	};

	T* Element(uint64_t index) {
		return std::launder(reinterpret_cast<T*>(m_slots[index & m_mask].storage));
	}

	void Construct(uint64_t tail, T&& value) {
		::new (static_cast<void*>(m_slots[tail & m_mask].storage)) T(std::move(value));
		m_tail.store(tail + 1, std::memory_order_release);
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (m_consumer_sleeping.load(std::memory_order_relaxed)) {
			m_consumer_signal.fetch_add(1, std::memory_order_seq_cst);
			m_consumer_signal.notify_one();
		}
	}

	std::optional<T> Take(uint64_t head) {
		auto*            element = Element(head);
		std::optional<T> value(std::move(*element));
		element->~T();
		m_head.store(head + 1, std::memory_order_release);
		if (++m_unchecked_pops >= CheckInterval) {
			WakeProducer();
		}
		return value;
	}

	void WakeProducer() {
		m_unchecked_pops = 0;
		std::atomic_thread_fence(std::memory_order_seq_cst);
		if (m_producer_sleeping.load(std::memory_order_relaxed)) {
			m_producer_signal.fetch_add(1, std::memory_order_seq_cst);
			m_producer_signal.notify_one();
		}
	}

	void WaitForSpace(uint64_t tail) {
		SpinWait spin;
		while (spin.Spin()) {
			m_producer_head = m_head.load(std::memory_order_acquire);
			if (tail - m_producer_head < m_capacity) {
				return;
			}
		}
		for (;;) {
			m_producer_sleeping.store(true, std::memory_order_seq_cst);
			const auto signal = m_producer_signal.load(std::memory_order_seq_cst);
			m_producer_head   = m_head.load(std::memory_order_seq_cst);
			if (tail - m_producer_head < m_capacity) {
				break;
			}
			m_producer_signal.wait(signal, std::memory_order_seq_cst);
		}
		m_producer_sleeping.store(false, std::memory_order_relaxed);
	}

	bool WaitForElement(uint64_t head) {
		WakeProducer();
		SpinWait spin;
		while (spin.Spin()) {
			m_consumer_tail = m_tail.load(std::memory_order_acquire);
			if (m_consumer_tail != head) {
				return true;
			}
			if (m_stopped.load(std::memory_order_relaxed) ||
			    m_interrupted.load(std::memory_order_relaxed)) {
				break;
			}
		}
		bool available = false;
		for (;;) {
			m_consumer_sleeping.store(true, std::memory_order_seq_cst);
			const auto signal = m_consumer_signal.load(std::memory_order_seq_cst);
			m_consumer_tail   = m_tail.load(std::memory_order_seq_cst);
			if (m_consumer_tail != head) {
				available = true;
				break;
			}
			if (m_stopped.load(std::memory_order_seq_cst) ||
			    m_interrupted.load(std::memory_order_seq_cst)) {
				break;
			}
			m_consumer_signal.wait(signal, std::memory_order_seq_cst);
		}
		m_consumer_sleeping.store(false, std::memory_order_relaxed);
		if (!available) {
			m_interrupted.store(false, std::memory_order_relaxed);
		}
		return available;
	}

	const uint64_t          m_capacity;
	const uint64_t          m_mask;
	std::unique_ptr<Slot[]> m_slots;

	alignas(64) std::atomic_uint64_t m_head {0};
	uint64_t             m_consumer_tail  = 0;
	uint32_t             m_unchecked_pops = 0;
	std::atomic_bool     m_consumer_sleeping {false};
	std::atomic_uint32_t m_consumer_signal {0};
	std::atomic_bool     m_stopped {false};
	std::atomic_bool     m_interrupted {false};

	alignas(64) std::atomic_uint64_t m_tail {0};
	uint64_t             m_producer_head = 0;
	std::atomic_bool     m_producer_sleeping {false};
	std::atomic_uint32_t m_producer_signal {0};
};

}

#endif
