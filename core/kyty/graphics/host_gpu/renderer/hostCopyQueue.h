#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HOSTCOPYQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_HOSTCOPYQUEUE_H_

#include "common/alignment.h"
#include "common/common.h"
#include "common/spscQueue.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <memory>
#include <thread>
#include <utility>

namespace Libs::Graphics {

class HostCopyQueue {
public:
	static constexpr size_t BatchSize = 8;
	static constexpr size_t Batches   = 2048;

	explicit HostCopyQueue(std::function<void()> configure_thread = {})
	    : m_batches(Batches), m_thread([this, configure = std::move(configure_thread)] {
		      if (configure) {
			      configure();
		      }
		      Run();
	      }) {}
	~HostCopyQueue() {
		(void)Flush();
		m_batches.Stop();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	KYTY_CLASS_NO_COPY(HostCopyQueue);

	void Copy(void* destination, const void* source, size_t size) {
		Queue(destination, source, size);
		m_source_copies = m_queued + m_pending.count;
	}
	void CopyInline(void* destination, const void* source, size_t size) {
		if (size > RingChunkSize) {
			std::memcpy(destination, source, size);
			return;
		}
		if (m_ring_offset + size > RingChunkSize) {
			NextRingChunk();
		}
		auto* staged = m_ring.get() + m_ring_chunk * RingChunkSize + m_ring_offset;
		std::memcpy(staged, source, size);
		m_ring_offset += Common::AlignUp(size, size_t {16});
		m_ring_users[m_ring_chunk] = m_queued + m_pending.count + 1;
		Queue(destination, staged, size);
	}
	void Call(void (*call)(void*, uint64_t, uint64_t), void* context, uint64_t a, uint64_t b) {
		m_pending.jobs[m_pending.count++] = {reinterpret_cast<void*>(a), context, static_cast<size_t>(b), call};
		if (m_pending.count == BatchSize) {
			(void)Flush();
		}
	}
	uint64_t Flush() {
		if (m_pending.count != 0) {
			m_queued += m_pending.count;
			m_batches.Push(std::move(m_pending));
			m_pending.count = 0;
		}
		return m_queued;
	}
	void WaitFor(uint64_t count) noexcept { m_completed.WaitFor(count); }
	void Drain() {
		if (m_completed.Load() >= m_source_copies) {
			return;
		}
		(void)Flush();
		m_completed.WaitFor(m_source_copies);
	}

private:
	static constexpr size_t RingChunkSize = size_t {128} << 10u;
	static constexpr size_t RingChunks    = 64;

	void NextRingChunk() {
		m_ring_chunk  = (m_ring_chunk + 1) % RingChunks;
		m_ring_offset = 0;
		if (const auto user = m_ring_users[m_ring_chunk]; user != 0 && m_completed.Load() < user) {
			(void)Flush();
			m_completed.WaitFor(user);
		}
	}

	void Queue(void* destination, const void* source, size_t size) {
		m_pending.jobs[m_pending.count++] = {destination, source, size};
		if (m_pending.count == BatchSize) {
			(void)Flush();
		}
	}

	struct Job {
		void*       destination = nullptr;
		const void* source      = nullptr;
		size_t      size        = 0;
		void (*call)(void*, uint64_t, uint64_t) = nullptr;
	};
	struct Batch {
		std::array<Job, BatchSize> jobs {};
		size_t                     count = 0;
	};

	void Run() {
		uint64_t done = 0;
		for (;;) {
			auto batch = m_batches.TryPop();
			if (!batch) {
				m_completed.Notify();
				batch = m_batches.Pop();
				if (!batch) {
					break;
				}
			}
			for (size_t i = 0; i < batch->count; i++) {
				const auto& job = batch->jobs[i];
				if (job.call != nullptr) [[unlikely]] {
					job.call(const_cast<void*>(job.source), reinterpret_cast<uint64_t>(job.destination), job.size);
				} else {
					std::memcpy(job.destination, job.source, job.size);
				}
			}
			done += batch->count;
			_mm_sfence();
			m_completed.Publish(done);
		}
		m_completed.Notify();
	}

	Common::SpscQueue<Batch> m_batches;
	Common::ProgressCounter  m_completed;
	alignas(64) Batch m_pending;
	uint64_t    m_queued = 0;
	uint64_t    m_source_copies = 0;
	std::unique_ptr<std::byte[]>     m_ring = std::make_unique<std::byte[]>(RingChunkSize * RingChunks);
	std::array<uint64_t, RingChunks> m_ring_users {};
	size_t                           m_ring_chunk  = 0;
	size_t                           m_ring_offset = 0;
	std::thread m_thread;
};

}

#endif
