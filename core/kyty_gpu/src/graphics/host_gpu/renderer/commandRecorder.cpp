#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/queueLockSite.h"

#include "common/threads.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "graphics/host_gpu/renderer/recordPackets.h"

#include <mutex>
#include <optional>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <new>
#include <thread>
#include <type_traits>
#include <vector>

// The render scheduler's Vulkan commands are copied into a ring and recorded
// by a worker thread. See commandRecorder.h for the overview; this file holds the details.
//
// Which calls go where (the dispatcher members are replaced once, in Install, and never again):
// - Serialized (KYTY_RECORD_CUSTOM_HOOKS and KYTY_RECORD_POD_HOOKS): every command the renderer
//   records, into the routed command buffer, is copied into the ring and later made by the worker with
//   the real function pointer. Arguments are copied by value; arrays and structs they point to are
//   copied into the packet; a pNext chain, a descriptor type or a size nothing here knows is not
//   copied: that one call drains the ring and goes straight to the driver ("unsupported").
// - Into any other command buffer (the presentation scheduler's, the readback and upload helpers'),
//   every hook calls the driver directly: one relaxed load and a compare.
// - All other vkCmd* functions (KYTY_RECORD_FALLBACK_HOOKS, one per remaining dispatcher member) into
//   the routed buffer: drain the ring, then call the driver directly.
// - vkBeginCommandBuffer/vkResetCommandBuffer on the routed buffer drain first. vkEndCommandBuffer on it
//   drains, ends the buffer (the caller sees the real result) and stops routing it.
// - Pool and destroy calls (KYTY_RECORD_POOL_HOOKS, KYTY_RECORD_DESTROY_HOOKS, and the VMA destroy
//   sites through DrainBeforeDestroy) drain what is published: an appended command may still name the
//   object. Vulkan only allows that after the commands finished, but nothing here relies on it.
//
// The worker never submits, presents, waits for a fence or takes a lock of the emulator: it only calls
// vkCmd* functions with arguments out of the ring. A thread waits for it only before ending a routed
// buffer, before direct commands, before destroying and when the ring is full, and the worker never
// waits for such a thread, so the waits end whenever the driver returns from the recorded calls.
// A wait that lasts more than a few seconds prints a RECORD_THREAD warning with the ring positions.

namespace Libs::Graphics::CommandRecorder {

namespace detail {
bool g_active = false;
} // namespace detail

namespace {

using Clock = std::chrono::steady_clock;

// Why a thread waited for the ring (counted in the RECORD_THREAD line).
enum class Reason : uint8_t {
	End,         // vkEndCommandBuffer on the routed buffer
	Fallback,    // a command nothing serializes, or begin/reset of the routed buffer
	Destroy,     // a resource is destroyed
	Pool,        // a command pool call
	Space,       // the ring was full
	Unsupported, // a serialized command with a shape that is not copied
	Shutdown,
	Other,
	Queue,       // another queue call (present, overlay or readback submits, idle waits) waits for the ring
	Count,
};

constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> ReasonNames {
    "end", "fallback", "destroy", "pool", "space", "unsupported", "shutdown", "other", "queue"};

constexpr size_t   RingBytes    = size_t {32} << 20u;
constexpr size_t   MaxPacket    = RingBytes / 8u; // Larger calls are made directly.
constexpr size_t   Align        = 16;
constexpr size_t   ClosureMax   = 96;   // The largest closure (static_assert in Packet::Commit).
constexpr uint64_t PublishBytes = 4096; // Appended bytes that are published without a draw or dispatch.
constexpr auto     SpinTime     = std::chrono::microseconds(1500); // How long the recording worker busy-waits before it sleeps.
constexpr auto     ReportEvery  = std::chrono::seconds(5);
constexpr auto     StallWarning = std::chrono::seconds(3);

// A packet: this header, copies of the arrays and structs the call points to, then the closure that
// makes the call. All sizes are multiples of Align.
struct Header {
	void (*run)(const void* closure) = nullptr; // Null: the rest of the ring is skipped.
	uint32_t size                    = 0;       // The whole packet.
	uint32_t closure_offset          = 0;       // From the header.
};
static_assert(sizeof(Header) == 16);
constexpr size_t HeaderBytes = sizeof(Header);

template <typename T>
[[nodiscard]] constexpr size_t ArrayBytes(uint32_t count) noexcept {
	return (sizeof(T) * static_cast<size_t>(count) + Align - 1u) & ~(Align - 1u);
}

[[nodiscard]] constexpr bool Fits(size_t array_bytes) noexcept {
	return HeaderBytes + array_bytes + ClosureMax <= MaxPacket;
}

// Statistics only: a single writer in practice (the recording thread), so no atomic read-modify-write.
void Bump(std::atomic<uint64_t>& counter, uint64_t amount = 1) noexcept {
	counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

[[nodiscard]] const void* ThreadToken() noexcept {
	static thread_local char token;
	return &token;
}

// The worker's thread token (null before it starts): queue calls made by the worker itself do not wait for it.
std::atomic<const void*> g_worker_token {nullptr};

// Set while the worker replays packets: commands it makes are not routed again.
thread_local bool t_replaying = false;
class ReplayScope {
public:
	ReplayScope() noexcept: m_outer(t_replaying) { t_replaying = true; }
	~ReplayScope() { t_replaying = m_outer; }
	KYTY_CLASS_NO_COPY(ReplayScope);

private:
	bool m_outer;
};

// Based on chenxiao07/KytyPS5 v20261020b (vulkan-recording.h), GPL-2.0 (commit 030c2a412).
// The record packets (recordPackets.h): descriptor writes of the bindless set, queued by any thread in call order
// (QueueDescriptorWrites) and moved into the ring by the recording side before its next packet (FlushQueuedWritesIfAny),
// so that only the recording thread updates the set. A write keeps its struct with the info pointers cleared; the infos
// its descriptor type reads are kept in `infos` at `info`.
struct QueuedWrite {
	VkWriteDescriptorSet write {};
	size_t               info  = 0;
	uint32_t             bytes = 0;
	uint8_t              kind  = 0; // 0: pImageInfo, 1: pBufferInfo, 2: pTexelBufferView
};
struct WriteQueue {
	std::mutex               mutex;
	VkDevice                 device = nullptr;
	std::vector<QueuedWrite> writes;
	std::vector<uint8_t>     infos;
	std::atomic<uint32_t>    count {0}; // writes queued and not moved yet (the recording side's cheap check)
	std::atomic<uint64_t>    queued {0};
	std::atomic<uint64_t>    flushes {0};
	uint64_t                 reported_queued  = 0; // the report's last values (the recording side only)
	uint64_t                 reported_flushes = 0;
};
WriteQueue g_write_queue;
// Set while the recording side moves the queue into the ring (the packets it makes do not move it again).
thread_local bool t_flushing_writes = false;
// The recording side, before it appends a packet (and before it drains): the queued writes go first. Defined after Packet.
void FlushQueuedWritesSlow();
inline void FlushQueuedWritesIfAny() {
	if (g_write_queue.count.load(std::memory_order_acquire) != 0 && !t_flushing_writes) [[unlikely]] {
		FlushQueuedWritesSlow();
	}
}

class Recorder {
public:
	Recorder(): m_ring(static_cast<uint8_t*>(::operator new(RingBytes))), m_last_report(Clock::now()) {}
	KYTY_CLASS_NO_COPY(Recorder);

	[[nodiscard]] bool Stopped() const noexcept { return m_stopped.load(std::memory_order_acquire); }

	void Start() {
		m_thread = std::thread([this] { ConsumerMain(); });
	}

	// ---- The recording side (the thread recording into the routed buffer; Vulkan requires that
	// recording into a command buffer is externally synchronized, which also orders these plain
	// variables between threads that take turns).

	// Room for a packet of up to `bytes`, waiting for the worker to free ring space if needed.
	[[nodiscard]] uint8_t* Reserve(size_t bytes) {
		size_t offset = static_cast<size_t>(m_pending % RingBytes);
		if (RingBytes - offset < bytes) {
			// The packet would cross the end of the ring: a skip header takes what is left of it.
			const size_t rest = RingBytes - offset;
			WaitForSpace(rest);
			::new (static_cast<void*>(m_ring + offset)) Header {};
			m_pending += rest;
			offset = 0;
		}
		WaitForSpace(bytes);
		return m_ring + offset;
	}

	// The packet at the last Reserve is complete and `used` bytes long.
	void Commit(size_t used) {
		m_pending += used;
		const uint64_t count = m_cmds.load(std::memory_order_relaxed) + 1u;
		m_cmds.store(count, std::memory_order_relaxed);
		Bump(m_bytes, used);
		if (m_pending - m_published >= PublishBytes) {
			PublishPending();
		}
		if ((count & 63u) == 0) {
			NoteBacklog(m_pending - m_read.load(std::memory_order_relaxed));
		}
	}

	// Makes what was appended visible to the worker (a draw or dispatch ends a batch).
	void PublishPending() {
		if (m_pending == m_published) {
			return;
		}
		m_published = m_pending;
		// Sequentially consistent with the worker's "sleeping, then look at write", so a worker about
		// to sleep either sees this or is seen asleep.
		m_write.store(m_published, std::memory_order_seq_cst);
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			Wake();
		}
	}

	// Waits until everything appended is recorded.
	void Drain(Reason reason) {
		m_drains[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
		FlushQueuedWritesIfAny(); // the record packets: the queued descriptor writes are part of "everything"
		PublishPending();
		NoteBacklog(m_published - m_read.load(std::memory_order_relaxed));
		WaitForRead(m_published, reason);
	}

	// Pool and destroy calls. The recording thread publishes what it appended and waits for all of it.
	// Another thread waits only for what is published, since it cannot publish for the recording
	// thread; an object it destroys (the renderer does that once the tick of the last use completed)
	// is not named by the commands that are still being appended.
	void DrainForObjects(Reason reason) {
		// A packet's own calls (the record packets: the recording thread's descriptor heap resets its pools): the
		// thread replaying is the one that would have to read further; what it replays was appended before the object
		// stopped being used.
		if (t_replaying) {
			return;
		}
		const bool recording_thread = m_owner.load(std::memory_order_relaxed) == ThreadToken();
		if (recording_thread) {
			// The record packets: queued descriptor writes may name the object too. Another thread cannot move them;
			// it needs not: the objects they name are freed only after the tick they were queued in completed, and the
			// queue is moved into the ring before that tick's submission.
			FlushQueuedWritesIfAny();
			PublishPending();
		}
		const uint64_t target = m_write.load(std::memory_order_acquire);
		if (m_read.load(std::memory_order_acquire) >= target) {
			return;
		}
		m_drains[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
		WaitForRead(target, reason);
	}

	void SetOwner() noexcept { m_owner.store(ThreadToken(), std::memory_order_relaxed); }

	void NoteFallback(const char* name, std::atomic<uint64_t>* hits) { m_fallbacks.push_back({name, hits, 0}); }

	void Report(bool is_final);
	void Stop();

private:
	// ---- The worker.

	void ConsumerMain() {
		KYTY_PROFILER_THREAD("Thread_VkRecord");
		// The stage's core, normal priority: it spins while the ring is empty.
		Config::ConfigureGpuStageThread(Config::GpuStageThread::Recording);
		g_worker_token.store(ThreadToken(), std::memory_order_release);
		uint64_t position = 0;
		for (;;) {
			const uint64_t available = m_write.load(std::memory_order_acquire);
			if (position != available) {
				position = Replay(position, available);
				continue;
			}
			if (m_stop.load(std::memory_order_acquire)) {
				return;
			}
			if (SpinForWork(position)) {
				continue;
			}
			const uint32_t seen = m_signal.load(std::memory_order_acquire);
			m_sleeping.store(true, std::memory_order_seq_cst);
			if (m_write.load(std::memory_order_seq_cst) == position &&
			    !m_stop.load(std::memory_order_seq_cst)) {
				m_signal.wait(seen, std::memory_order_acquire);
			}
			m_sleeping.store(false, std::memory_order_seq_cst);
		}
	}

	// Busy-waits for a while: the recording thread appends a few commands per draw, so the gap to the
	// next batch is shorter than a wake-up.
	[[nodiscard]] bool SpinForWork(uint64_t position) {
		const auto deadline = Clock::now() + SpinTime;
		for (uint32_t i = 0;; ++i) {
			_mm_pause();
			if (m_write.load(std::memory_order_relaxed) != position) {
				return true;
			}
			if ((i & 127u) == 127u) {
				if (m_stop.load(std::memory_order_relaxed) || Clock::now() >= deadline) {
					return m_stop.load(std::memory_order_relaxed);
				}
			}
		}
	}

	// Makes the calls of the packets in [position, available) and publishes the read position (every
	// 64 KiB while it goes, and when it is done). Returns the new position.
	uint64_t Replay(uint64_t position, uint64_t available) {
		// A packet's own Vulkan calls (CommitCall packets call the dispatcher) go straight to the driver.
		const ReplayScope replaying;
		uint64_t          stored = position;
		while (position != available) {
			const uint8_t* start  = m_ring + position % RingBytes;
			const Header   header = *reinterpret_cast<const Header*>(start);
			if (header.run == nullptr) {
				position += RingBytes - position % RingBytes;
			} else {
				EXIT_IF(header.size < HeaderBytes || header.size > MaxPacket ||
				        position + header.size > available);
				header.run(start + header.closure_offset);
				position += header.size;
			}
			if (position - stored >= 65536u) {
				m_read.store(position, std::memory_order_release);
				stored = position;
			}
		}
		m_read.store(position, std::memory_order_release);
		return position;
	}

	void Wake() {
		m_wakes.fetch_add(1, std::memory_order_relaxed);
		m_signal.fetch_add(1, std::memory_order_release);
		m_signal.notify_one();
	}

	// ---- Waiting.

	void WaitForSpace(size_t bytes) {
		if (m_pending + bytes - m_cached_read <= RingBytes) {
			return;
		}
		m_cached_read = m_read.load(std::memory_order_acquire);
		if (m_pending + bytes - m_cached_read <= RingBytes) {
			return;
		}
		WaitForSpaceSlow(bytes);
	}

	void WaitForSpaceSlow(size_t bytes) {
		KYTY_PROFILER_BLOCK("RecordThread::RingFull");
		m_drains[static_cast<size_t>(Reason::Space)].fetch_add(1, std::memory_order_relaxed);
		// The worker frees only what it can see.
		PublishPending();
		NoteBacklog(m_pending - m_cached_read);
		const auto start = Clock::now();
		WaitUntil(
		    [&] {
			    m_cached_read = m_read.load(std::memory_order_acquire);
			    return m_pending + bytes - m_cached_read <= RingBytes;
		    },
		    Reason::Space);
		AddWait(start);
	}

	void WaitForRead(uint64_t target, Reason reason) {
		if (m_read.load(std::memory_order_acquire) >= target) {
			return;
		}
		KYTY_PROFILER_BLOCK("RecordThread::Wait");
		const auto start = Clock::now();
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			Wake();
		}
		WaitUntil([&] { return m_read.load(std::memory_order_acquire) >= target; }, reason);
		AddWait(start);
		if (GpuTimer::Enabled() && GpuTimer::TimelineCpuActive()) GpuTimer::NoteCpuWait('r', start); // KYTY_TRACE_GPU_TIMELINE
	}

	// Spins, then yields. A worker that went to sleep is woken again now and then; a wait that goes on
	// is reported (the worker waits for no one, so it means the driver does not return).
	template <typename Done>
	void WaitUntil(Done&& done, Reason reason) {
		auto     next_warning = Clock::now() + StallWarning;
		uint32_t spins        = 0;
		while (!done()) {
			if (++spins < 256u) {
				_mm_pause();
				continue;
			}
			std::this_thread::yield();
			if ((spins & 0x3ffu) == 0) {
				if (m_sleeping.load(std::memory_order_seq_cst)) {
					Wake();
				}
				const auto now = Clock::now();
				if (now >= next_warning) {
					next_warning = now + StallWarning;
					// The holder of the Vulkan queue lock (QueueLockGuard), when the worker may be blocked taking it.
					const char*   lock_site = QueueLockSite::site.load(std::memory_order_acquire);
					const int64_t lock_ms   = lock_site != nullptr
					                              ? (QueueLockSite::NowNs() -
					                                 QueueLockSite::since_ns.load(std::memory_order_relaxed)) /
					                                    1000000
					                              : 0;
					std::fprintf(stderr,
					             "RECORD_THREAD warning: still waiting for the recording thread (%s): "
					             "read=%" PRIu64 " published=%" PRIu64 " pending=%" PRIu64 " sleeping=%d queue_lock=%s held_ms=%lld\n",
					             ReasonNames[static_cast<size_t>(reason)],
					             m_read.load(std::memory_order_relaxed),
					             m_write.load(std::memory_order_relaxed), m_pending,
					             m_sleeping.load(std::memory_order_relaxed) ? 1 : 0, lock_site != nullptr ? lock_site : "-",
					             static_cast<long long>(lock_ms));
				}
			}
		}
	}

	void AddWait(Clock::time_point start) {
		m_wait_us.fetch_add(
		    static_cast<uint64_t>(
		        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count()),
		    std::memory_order_relaxed);
	}

	void NoteBacklog(uint64_t bytes) {
		if (bytes > m_max_ring.load(std::memory_order_relaxed)) {
			m_max_ring.store(bytes, std::memory_order_relaxed);
		}
	}

	struct FallbackEntry {
		const char*            name;
		std::atomic<uint64_t>* hits;
		uint64_t               reported;
	};

	uint8_t* const m_ring;

	// Three groups, at least a cache line apart (the padding), by who writes them and how often:
	// 1. what the recording side changes for every packet and nobody else reads,
	uint64_t                 m_pending     = 0; // Bytes appended; the ring position of the next packet.
	uint64_t                 m_published   = 0; // The value last stored into m_write.
	uint64_t                 m_cached_read = 0; // The last m_read seen.
	std::atomic<const void*> m_owner {nullptr}; // The thread that last called Route.
	std::atomic<uint64_t>    m_cmds {0};
	std::atomic<uint64_t>    m_bytes {0};
	uint8_t                  m_pad0[64] {};

	// 2. what the recording side changes once a batch and the worker polls while it waits,
	std::atomic<uint64_t> m_write {0};         // Bytes published.
	std::atomic<bool>     m_sleeping {false};  // The worker is about to wait on m_signal, or does.
	std::atomic<uint32_t> m_signal {0};
	std::atomic<bool>     m_stop {false};
	std::atomic<bool>     m_stopped {false};
	uint8_t               m_pad1[64] {};

	// 3. what the worker changes as it goes.
	std::atomic<uint64_t> m_read {0}; // Bytes recorded.
	uint8_t               m_pad2[64] {};

	std::thread m_thread;

	// Statistics.
	std::atomic<uint64_t> m_wakes {0};
	std::atomic<uint64_t> m_wait_us {0};
	std::atomic<uint64_t> m_max_ring {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Reason::Count)> m_drains {};
	std::vector<FallbackEntry>                                            m_fallbacks;
	Clock::time_point m_last_report;
	uint64_t          m_last_cmds  = 0;
	uint64_t          m_last_bytes = 0;
	uint64_t          m_last_wakes = 0;
	uint64_t          m_last_wait  = 0;
	std::array<uint64_t, static_cast<size_t>(Reason::Count)> m_last_drains {};
};

Recorder* g_recorder = nullptr;

// The command buffer the recorder serializes commands for (null: none).
std::atomic<VkCommandBuffer> g_routed {nullptr};

[[nodiscard]] inline bool IsRouted(VkCommandBuffer buffer) noexcept {
	const bool routed = buffer != nullptr && buffer == g_routed.load(std::memory_order_relaxed) && !t_replaying;
	return routed;
}

void Recorder::Report(bool is_final) {
	const auto now = Clock::now();
	if (!is_final && (now - m_last_report < ReportEvery || !Common::EnvRead::PerfLines())) {
		return;
	}
	const double seconds = std::chrono::duration<double>(now - m_last_report).count();
	m_last_report        = now;

	const uint64_t cmds  = m_cmds.load(std::memory_order_relaxed);
	const uint64_t bytes = m_bytes.load(std::memory_order_relaxed);
	const uint64_t wakes = m_wakes.load(std::memory_order_relaxed);
	const uint64_t wait  = m_wait_us.load(std::memory_order_relaxed);
	std::array<uint64_t, static_cast<size_t>(Reason::Count)> drains {};
	for (size_t i = 0; i < drains.size(); ++i) {
		drains[i] = m_drains[i].load(std::memory_order_relaxed);
	}
	// "cmds" and "bytes" are packets and ring bytes appended in the last interval; max_ring is the
	// most the worker was behind (bytes appended and not yet recorded) at the points where it was
	// looked at, in that interval.
	std::fprintf(stderr,
	             "RECORD_THREAD%s s=%.1f cmds=%" PRIu64 " bytes=%" PRIu64 " drains(end=%" PRIu64
	             " fallback=%" PRIu64 " destroy=%" PRIu64 " pool=%" PRIu64 " space=%" PRIu64
	             " unsupported=%" PRIu64 " shutdown=%" PRIu64 " other=%" PRIu64 ") wait_ms=%.1f wakes=%" PRIu64
	             " max_ring=%" PRIu64 " total_cmds=%" PRIu64 "\n",
	             is_final ? " final" : "", seconds, cmds - m_last_cmds, bytes - m_last_bytes,
	             drains[0] - m_last_drains[0], drains[1] - m_last_drains[1],
	             drains[2] - m_last_drains[2], drains[3] - m_last_drains[3],
	             drains[4] - m_last_drains[4], drains[5] - m_last_drains[5],
	             drains[6] - m_last_drains[6], drains[7] - m_last_drains[7],
	             static_cast<double>(wait - m_last_wait) / 1000.0, wakes - m_last_wakes,
	             m_max_ring.exchange(0, std::memory_order_relaxed), cmds);
	m_last_cmds   = cmds;
	m_last_bytes  = bytes;
	m_last_wakes  = wakes;
	m_last_wait   = wait;
	m_last_drains = drains;
	{ // the record packets: the RECORD_PACKETS line, with the descriptor-write queue's counters of the interval.
		auto&          q       = g_write_queue;
		const uint64_t queued  = q.queued.load(std::memory_order_relaxed);
		const uint64_t flushes = q.flushes.load(std::memory_order_relaxed);
		RecordPackets::Report(seconds, queued - q.reported_queued, flushes - q.reported_flushes);
		q.reported_queued  = queued;
		q.reported_flushes = flushes;
	}

	// The unserialized commands that were recorded most since the last line: candidates for a hook.
	const FallbackEntry* top[3] = {};
	uint64_t             top_n[3] = {};
	for (auto& entry: m_fallbacks) {
		const uint64_t n = entry.hits->load(std::memory_order_relaxed) - entry.reported;
		for (size_t slot = 0; n != 0 && slot < 3; ++slot) {
			if (n > top_n[slot]) {
				for (size_t move = 2; move > slot; --move) {
					top[move]   = top[move - 1];
					top_n[move] = top_n[move - 1];
				}
				top[slot]   = &entry;
				top_n[slot] = n;
				break;
			}
		}
	}
	if (top[0] != nullptr) {
		std::fprintf(stderr, "RECORD_THREAD fallback_top");
		for (size_t slot = 0; slot < 3 && top[slot] != nullptr; ++slot) {
			std::fprintf(stderr, " %s=%" PRIu64, top[slot]->name, top_n[slot]);
		}
		std::fprintf(stderr, "\n");
	}
	for (auto& entry: m_fallbacks) {
		entry.reported = entry.hits->load(std::memory_order_relaxed);
	}
}

void Recorder::Stop() {
	if (m_stopped.load(std::memory_order_acquire)) {
		return;
	}
	FlushQueuedWritesIfAny(); // the record packets: the last queued descriptor writes are made too
	// Everything appended is recorded first; nothing is appended any more (the last buffer was
	// ended and submitted), but a caller that appended after all gets the same wait.
	m_drains[static_cast<size_t>(Reason::Shutdown)].fetch_add(1, std::memory_order_relaxed);
	PublishPending();
	WaitForRead(m_published, Reason::Shutdown);
	m_stop.store(true, std::memory_order_seq_cst);
	Wake();
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_stopped.store(true, std::memory_order_release);
	Report(true);
}

// ---- Packets.

// The record packets: the queued descriptor writes go into the ring before the packet that is reserved now.
[[nodiscard]] uint8_t* ReserveAfterQueuedWrites(size_t bytes) {
	FlushQueuedWritesIfAny();
	return g_recorder->Reserve(bytes);
}

class Packet {
public:
	// Room for `array_bytes` of copied arrays and structs (a sum of ArrayBytes) and the closure.
	explicit Packet(size_t array_bytes)
	    : m_start(ReserveAfterQueuedWrites(HeaderBytes + array_bytes + ClosureMax)),
	      m_next(m_start + HeaderBytes), m_end(m_start + HeaderBytes + array_bytes) {}
	KYTY_CLASS_NO_COPY(Packet);

	// Copies `count` items; null for an empty or null array.
	template <typename T>
	[[nodiscard]] T* Copy(const T* source, uint32_t count) {
		if (source == nullptr || count == 0) {
			return nullptr;
		}
		auto* target = reinterpret_cast<T*>(m_next);
		std::memcpy(static_cast<void*>(target), static_cast<const void*>(source),
		            sizeof(T) * static_cast<size_t>(count));
		m_next += ArrayBytes<T>(count);
		EXIT_IF(m_next > m_end);
		return target;
	}

	// Room for `count` bytes the caller fills (ReserveCall).
	[[nodiscard]] uint8_t* Take(uint32_t count) {
		auto* target = m_next;
		m_next += ArrayBytes<uint8_t>(count);
		EXIT_IF(m_next > m_end);
		return target;
	}

	// The call to make: a lambda that captures plain values and pointers returned by Copy.
	template <typename Call>
	void Commit(const Call& call) {
		static_assert(std::is_trivially_destructible_v<Call>);
		static_assert(sizeof(Call) <= ClosureMax && alignof(Call) <= Align);
		// The closure follows the arrays that were copied; an array that was null or empty left its
		// share of the reservation unused.
		EXIT_IF(m_next > m_end);
		::new (static_cast<void*>(m_next)) Call(call);
		const auto offset = static_cast<uint32_t>(m_next - m_start);
		const auto size   = static_cast<uint32_t>((offset + sizeof(Call) + Align - 1u) & ~(Align - 1u));
		::new (static_cast<void*>(m_start)) Header {
		    +[](const void* closure) { (*static_cast<const Call*>(closure))(); }, size, offset};
		g_recorder->Commit(size);
	}

private:
	uint8_t* m_start;
	uint8_t* m_next;
	uint8_t* m_end;
};

// The record packets: the queued descriptor writes, in their order, as packets of the recording thread's
// vkUpdateDescriptorSets (as many writes per packet as fit). The recording side only.
void FlushQueuedWritesSlow() {
	if (g_recorder == nullptr || g_recorder->Stopped()) {
		return; // Shutdown: Stop moved the queue before it stopped; nothing records any more.
	}
	auto&                                        q = g_write_queue;
	static thread_local std::vector<QueuedWrite> writes;
	static thread_local std::vector<uint8_t>     infos;
	VkDevice                                     device = nullptr;
	{
		std::lock_guard lock(q.mutex);
		writes.swap(q.writes);
		infos.swap(q.infos);
		q.writes.clear();
		q.infos.clear();
		device = q.device;
		q.count.store(0, std::memory_order_relaxed);
	}
	if (writes.empty()) {
		return;
	}
	t_flushing_writes = true;
	const PFN_vkUpdateDescriptorSets update = VULKAN_HPP_DEFAULT_DISPATCHER.vkUpdateDescriptorSets;
	EXIT_IF(update == nullptr || device == nullptr);
	size_t first = 0;
	while (first < writes.size()) {
		// As many writes as fit one packet (at least one).
		size_t last  = first;
		size_t bytes = 0;
		for (; last < writes.size(); ++last) {
			const size_t add = ArrayBytes<VkWriteDescriptorSet>(1) + ArrayBytes<uint8_t>(writes[last].bytes);
			if (last > first && !Fits(bytes + add)) break;
			bytes += add;
		}
		EXIT_IF(!Fits(bytes));
		const auto n = static_cast<uint32_t>(last - first);
		Packet     packet(bytes);
		auto*      out =
		    reinterpret_cast<VkWriteDescriptorSet*>(packet.Take(static_cast<uint32_t>(sizeof(VkWriteDescriptorSet) * n)));
		for (size_t k = first; k < last; ++k) {
			const auto& entry = writes[k];
			auto&       copy  = out[k - first];
			copy              = entry.write;
			uint8_t* info     = packet.Take(entry.bytes);
			std::memcpy(info, infos.data() + entry.info, entry.bytes);
			switch (entry.kind) {
				case 0: copy.pImageInfo = reinterpret_cast<const VkDescriptorImageInfo*>(info); break;
				case 1: copy.pBufferInfo = reinterpret_cast<const VkDescriptorBufferInfo*>(info); break;
				default: copy.pTexelBufferView = reinterpret_cast<const VkBufferView*>(info); break;
			}
		}
		const VkWriteDescriptorSet* copies = out;
		packet.Commit([=] { update(device, n, copies, 0, nullptr); });
		first = last;
	}
	q.flushes.fetch_add(1, std::memory_order_relaxed);
	writes.clear();
	infos.clear();
	t_flushing_writes = false;
}

// A draw or dispatch ends a batch the worker may start on.
void PublishBatch() {
	g_recorder->PublishPending();
}

// A serialized command whose shape is not copied: everything before it is recorded, then it is made
// directly by the caller.
void Unsupported() {
	g_recorder->Drain(Reason::Unsupported);
}

template <typename T>
[[nodiscard]] bool NoNext(const T* items, uint32_t count) noexcept {
	if (items == nullptr) {
		return true;
	}
	for (uint32_t i = 0; i < count; ++i) {
		if (items[i].pNext != nullptr) {
			return false;
		}
	}
	return true;
}

// ---- The functions that are replaced.

// Serialized with arrays or structs copied by a function of its own.
#define KYTY_RECORD_CUSTOM_HOOKS(X)                                                                \
	X(vkBeginCommandBuffer)                                                                        \
	X(vkEndCommandBuffer)                                                                          \
	X(vkResetCommandBuffer)                                                                        \
	X(vkCmdPipelineBarrier)                                                                        \
	X(vkCmdPipelineBarrier2)                                                                       \
	X(vkCmdBindDescriptorSets)                                                                     \
	X(vkCmdPushDescriptorSetKHR)                                                                   \
	X(vkCmdPushConstants)                                                                          \
	X(vkCmdUpdateBuffer)                                                                           \
	X(vkCmdBindVertexBuffers2)                                                                     \
	X(vkCmdBeginRendering)                                                                         \
	X(vkCmdCopyBuffer)                                                                             \
	X(vkCmdCopyImage)                                                                              \
	X(vkCmdCopyBufferToImage)                                                                      \
	X(vkCmdCopyImageToBuffer)                                                                      \
	X(vkCmdResolveImage)                                                                           \
	X(vkCmdBlitImage)                                                                              \
	X(vkCmdClearColorImage)                                                                        \
	X(vkCmdClearDepthStencilImage)                                                                 \
	X(vkCmdClearAttachments)                                                                       \
	X(vkCmdSetViewport)                                                                            \
	X(vkCmdSetViewportWithCount)                                                                   \
	X(vkCmdSetScissor)                                                                             \
	X(vkCmdSetScissorWithCount)                                                                    \
	X(vkCmdSetBlendConstants)                                                                      \
	X(vkCmdSetColorWriteEnableEXT)                                                                 \
	X(vkCmdBeginConditionalRenderingEXT)

// Serialized by value only (the second argument: whether the command ends a batch). The checkpoint
// marker of vkCmdSetCheckpointNV is a value the driver never dereferences.
#define KYTY_RECORD_POD_HOOKS(X)                                                                   \
	X(vkCmdBindPipeline, false)                                                                    \
	X(vkCmdBindIndexBuffer, false)                                                                 \
	X(vkCmdEndRendering, true)                                                                     \
	X(vkCmdDraw, true)                                                                             \
	X(vkCmdDrawIndexed, true)                                                                      \
	X(vkCmdDrawIndirect, true)                                                                     \
	X(vkCmdDrawIndexedIndirect, true)                                                              \
	X(vkCmdDrawIndirectCount, true)                                                                \
	X(vkCmdDrawIndexedIndirectCount, true)                                                         \
	X(vkCmdDrawMeshTasksEXT, true)                                                                 \
	X(vkCmdDrawMeshTasksIndirectEXT, true)                                                         \
	X(vkCmdDispatch, true)                                                                         \
	X(vkCmdDispatchIndirect, true)                                                                 \
	X(vkCmdFillBuffer, false)                                                                      \
	X(vkCmdWriteTimestamp, false)                                                                  \
	X(vkCmdWriteTimestamp2, false)                                                                 \
	X(vkCmdResetQueryPool, false)                                                                  \
	X(vkCmdBeginQuery, false)                                                                      \
	X(vkCmdEndQuery, false)                                                                        \
	X(vkCmdEndConditionalRenderingEXT, false)                                                      \
	X(vkCmdSetLineWidth, false)                                                                    \
	X(vkCmdSetDepthBias, false)                                                                    \
	X(vkCmdSetDepthBiasEnable, false)                                                              \
	X(vkCmdSetDepthBounds, false)                                                                  \
	X(vkCmdSetDepthBoundsTestEnable, false)                                                        \
	X(vkCmdSetDepthTestEnable, false)                                                              \
	X(vkCmdSetDepthWriteEnable, false)                                                             \
	X(vkCmdSetDepthCompareOp, false)                                                               \
	X(vkCmdSetStencilTestEnable, false)                                                            \
	X(vkCmdSetStencilOp, false)                                                                    \
	X(vkCmdSetStencilCompareMask, false)                                                           \
	X(vkCmdSetStencilWriteMask, false)                                                             \
	X(vkCmdSetStencilReference, false)                                                             \
	X(vkCmdSetAttachmentFeedbackLoopEnableEXT, false)                                              \
	X(vkCmdSetCheckpointNV, false)

// Pool calls: the worker may be recording into a buffer of the pool.
#define KYTY_RECORD_POOL_HOOKS(X)                                                                  \
	X(vkAllocateCommandBuffers)                                                                    \
	X(vkFreeCommandBuffers)                                                                        \
	X(vkResetCommandPool)                                                                          \
	X(vkDestroyCommandPool)

// Objects a recorded command may still name.
#define KYTY_RECORD_DESTROY_HOOKS(X)                                                               \
	X(vkDestroyBuffer)                                                                             \
	X(vkDestroyImage)                                                                              \
	X(vkDestroyImageView)                                                                          \
	X(vkDestroyBufferView)                                                                         \
	X(vkDestroySampler)                                                                            \
	X(vkDestroyPipeline)                                                                           \
	X(vkDestroyPipelineLayout)                                                                     \
	X(vkDestroyDescriptorSetLayout)                                                                \
	X(vkDestroyDescriptorPool)                                                                     \
	X(vkResetDescriptorPool)                                                                       \
	X(vkFreeDescriptorSets)                                                                        \
	X(vkFreeMemory)                                                                                \
	X(vkDestroyQueryPool)

#include "graphics/host_gpu/renderer/commandRecorderFallbacks.inc"

// The driver's functions, as the dispatcher held them before Install.
struct RealFunctions {
#define KYTY_DECLARE_REAL(name) PFN_##name name = nullptr;
	KYTY_RECORD_CUSTOM_HOOKS(KYTY_DECLARE_REAL)
#undef KYTY_DECLARE_REAL
};
RealFunctions g_real;

// --- Commands of one call shape.

template <typename Tag, bool EndsBatch, typename Fn>
struct PodHook;

template <typename Tag, bool EndsBatch, typename... A>
struct PodHook<Tag, EndsBatch, void(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	static inline void(VKAPI_PTR* real)(VkCommandBuffer, A...) = nullptr;

	static void VKAPI_PTR Call(VkCommandBuffer buffer, A... args) {
		if (IsRouted(buffer)) {
			Packet packet(0);
			packet.Commit([=] { real(buffer, args...); });
			if constexpr (EndsBatch) {
				PublishBatch();
			}
			return;
		}
		real(buffer, args...);
	}
};

// Any other vkCmd*: after what the ring holds.
template <typename Tag, typename Fn>
struct FallbackHook;

template <typename Tag, typename R, typename... A>
struct FallbackHook<Tag, R(VKAPI_PTR*)(VkCommandBuffer, A...)> {
	static inline R(VKAPI_PTR* real)(VkCommandBuffer, A...) = nullptr;
	static inline std::atomic<uint64_t> hits {0};

	static R VKAPI_PTR Call(VkCommandBuffer buffer, A... args) {
		if (IsRouted(buffer)) {
			hits.fetch_add(1, std::memory_order_relaxed);
			g_recorder->Drain(Reason::Fallback);
		}
		return real(buffer, args...);
	}
};

// Pool and destroy calls: after what the ring holds.
template <typename Tag, Reason Why, typename Fn>
struct DrainHook;

template <typename Tag, Reason Why, typename R, typename... A>
struct DrainHook<Tag, Why, R(VKAPI_PTR*)(A...)> {
	static inline R(VKAPI_PTR* real)(A...) = nullptr;

	static R VKAPI_PTR Call(A... args) {
		g_recorder->DrainForObjects(Why);
		return real(args...);
	}
};

// --- Command buffer life cycle.

VkResult VKAPI_PTR Hook_vkBeginCommandBuffer(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
	if (IsRouted(buffer)) {
		// The buffer was routed before it was begun (the worker owns the pool while it records the previous buffers): the
		// begin is a packet. A primary buffer has no inheritance info.
		if (info != nullptr && info->pNext == nullptr && info->pInheritanceInfo == nullptr) {
			const VkCommandBufferBeginInfo copy = *info;
			Packet                         packet(0);
			packet.Commit([=] {
				const VkResult result = g_real.vkBeginCommandBuffer(buffer, &copy);
				EXIT_IF(result != VK_SUCCESS);
			});
			return VK_SUCCESS;
		}
		g_recorder->Drain(Reason::Fallback);
	}
	return g_real.vkBeginCommandBuffer(buffer, info);
}

// Commands recorded after the end (invalid for Vulkan anyway) are not routed. The end is a packet (the worker ends the
// buffer and then submits it, see EnqueueCall); the caller does not wait, and a failure ends the emulator on the worker.
VkResult VKAPI_PTR Hook_vkEndCommandBuffer(VkCommandBuffer buffer) {
	if (IsRouted(buffer)) {
		Packet packet(0);
		packet.Commit([=] {
			const VkResult result = g_real.vkEndCommandBuffer(buffer);
			EXIT_IF(result != VK_SUCCESS);
		});
		g_routed.store(nullptr, std::memory_order_release);
		return VK_SUCCESS;
	}
	return g_real.vkEndCommandBuffer(buffer);
}

VkResult VKAPI_PTR Hook_vkResetCommandBuffer(VkCommandBuffer buffer, VkCommandBufferResetFlags flags) {
	if (IsRouted(buffer)) {
		g_recorder->Drain(Reason::Fallback);
	}
	return g_real.vkResetCommandBuffer(buffer, flags);
}

// --- Barriers.

void VKAPI_PTR Hook_vkCmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags src,
                                         VkPipelineStageFlags dst, VkDependencyFlags flags,
                                         uint32_t memory_count, const VkMemoryBarrier* memory,
                                         uint32_t buffer_count, const VkBufferMemoryBarrier* buffers,
                                         uint32_t image_count, const VkImageMemoryBarrier* images) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkMemoryBarrier>(memory_count) +
		                     ArrayBytes<VkBufferMemoryBarrier>(buffer_count) +
		                     ArrayBytes<VkImageMemoryBarrier>(image_count);
		if (Fits(bytes) && NoNext(memory, memory_count) && NoNext(buffers, buffer_count) &&
		    NoNext(images, image_count)) {
			Packet      packet(bytes);
			const auto* m = packet.Copy(memory, memory_count);
			const auto* b = packet.Copy(buffers, buffer_count);
			const auto* i = packet.Copy(images, image_count);
			packet.Commit([=] {
				g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, m, buffer_count, b,
				                            image_count, i);
			});
			return;
		}
		Unsupported();
	}
	g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, memory, buffer_count, buffers,
	                            image_count, images);
}

void VKAPI_PTR Hook_vkCmdPipelineBarrier2(VkCommandBuffer buffer, const VkDependencyInfo* info) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkDependencyInfo>(1) +
		                     ArrayBytes<VkMemoryBarrier2>(info->memoryBarrierCount) +
		                     ArrayBytes<VkBufferMemoryBarrier2>(info->bufferMemoryBarrierCount) +
		                     ArrayBytes<VkImageMemoryBarrier2>(info->imageMemoryBarrierCount);
		if (info->pNext == nullptr && Fits(bytes) &&
		    NoNext(info->pMemoryBarriers, info->memoryBarrierCount) &&
		    NoNext(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount) &&
		    NoNext(info->pImageMemoryBarriers, info->imageMemoryBarrierCount)) {
			Packet packet(bytes);
			auto*  copy             = packet.Copy(info, 1);
			copy->pMemoryBarriers   = packet.Copy(info->pMemoryBarriers, info->memoryBarrierCount);
			copy->pBufferMemoryBarriers =
			    packet.Copy(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
			copy->pImageMemoryBarriers =
			    packet.Copy(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
			packet.Commit([=] { g_real.vkCmdPipelineBarrier2(buffer, copy); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdPipelineBarrier2(buffer, info);
}

// --- Bindings.

void VKAPI_PTR Hook_vkCmdBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                            VkPipelineLayout layout, uint32_t first, uint32_t count,
                                            const VkDescriptorSet* sets, uint32_t dynamic_count,
                                            const uint32_t* dynamic_offsets) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkDescriptorSet>(count) + ArrayBytes<uint32_t>(dynamic_count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* s = packet.Copy(sets, count);
			const auto* d = packet.Copy(dynamic_offsets, dynamic_count);
			packet.Commit([=] {
				g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, s, dynamic_count, d);
			});
			return;
		}
		Unsupported();
	}
	g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, sets, dynamic_count,
	                               dynamic_offsets);
}

// The array of a descriptor write that the descriptor type reads (Vulkan ignores the other two
// pointers, and they may hold anything). False: a type or a pNext chain that is not copied.
[[nodiscard]] bool WritesBytes(uint32_t count, const VkWriteDescriptorSet* writes, size_t& bytes) {
	bytes = ArrayBytes<VkWriteDescriptorSet>(count);
	for (uint32_t i = 0; i < count; ++i) {
		const auto& write = writes[i];
		if (write.pNext != nullptr) {
			return false;
		}
		switch (write.descriptorType) {
			case VK_DESCRIPTOR_TYPE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
				bytes += ArrayBytes<VkDescriptorImageInfo>(write.descriptorCount);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
				bytes += ArrayBytes<VkBufferView>(write.descriptorCount);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
				bytes += ArrayBytes<VkDescriptorBufferInfo>(write.descriptorCount);
				break;
			default: return false;
		}
	}
	return true;
}

// Copies the writes and the arrays they read (sized by WritesBytes).
[[nodiscard]] VkWriteDescriptorSet* CopyWrites(Packet& packet, uint32_t count,
                                               const VkWriteDescriptorSet* writes) {
	auto* copies = packet.Copy(writes, count);
	for (uint32_t i = 0; i < count; ++i) {
		const auto& write        = writes[i];
		auto&       copy         = copies[i];
		copy.pImageInfo          = nullptr;
		copy.pBufferInfo         = nullptr;
		copy.pTexelBufferView    = nullptr;
		switch (write.descriptorType) {
			case VK_DESCRIPTOR_TYPE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
				copy.pImageInfo = packet.Copy(write.pImageInfo, write.descriptorCount);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
				copy.pTexelBufferView = packet.Copy(write.pTexelBufferView, write.descriptorCount);
				break;
			default: // WritesBytes accepted nothing else: the buffer types.
				copy.pBufferInfo = packet.Copy(write.pBufferInfo, write.descriptorCount);
				break;
		}
	}
	return copies;
}

void VKAPI_PTR Hook_vkCmdPushDescriptorSetKHR(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                              VkPipelineLayout layout, uint32_t set, uint32_t count,
                                              const VkWriteDescriptorSet* writes) {
	if (IsRouted(buffer)) {
		size_t bytes = 0;
		if (WritesBytes(count, writes, bytes) && Fits(bytes)) {
			Packet packet(bytes);
			const auto* copies = CopyWrites(packet, count, writes);
			packet.Commit(
			    [=] { g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, copies); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, writes);
}

void VKAPI_PTR Hook_vkCmdPushConstants(VkCommandBuffer buffer, VkPipelineLayout layout,
                                       VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                                       const void* values) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<uint8_t>(size);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* data = packet.Copy(static_cast<const uint8_t*>(values), size);
			packet.Commit([=] { g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, data); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, values);
}

// vkCmdUpdateBuffer (at most 64 KiB of data, Vulkan) is copied into a packet like vkCmdPushConstants, instead of draining
// the ring and being made directly. The recording thread makes the same call with the same bytes in the same place of
// the buffer.
void VKAPI_PTR Hook_vkCmdUpdateBuffer(VkCommandBuffer buffer, VkBuffer destination, VkDeviceSize offset,
                                      VkDeviceSize size, const void* data) {
	if (IsRouted(buffer)) {
		if (size <= 65536u) {
			const size_t bytes = ArrayBytes<uint8_t>(static_cast<uint32_t>(size));
			if (Fits(bytes)) {
				Packet      packet(bytes);
				const auto* copy = packet.Copy(static_cast<const uint8_t*>(data), static_cast<uint32_t>(size));
				packet.Commit([=] { g_real.vkCmdUpdateBuffer(buffer, destination, offset, size, copy); });
				return;
			}
		}
		g_recorder->Drain(Reason::Fallback); // as the fallback hook did
	}
	g_real.vkCmdUpdateBuffer(buffer, destination, offset, size, data);
}

void VKAPI_PTR Hook_vkCmdBindVertexBuffers2(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                            const VkBuffer* buffers, const VkDeviceSize* offsets,
                                            const VkDeviceSize* sizes, const VkDeviceSize* strides) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkBuffer>(count) + 3u * ArrayBytes<VkDeviceSize>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* b  = packet.Copy(buffers, count);
			const auto* o  = packet.Copy(offsets, count);
			const auto* sz = packet.Copy(sizes, count); // Null: whole buffers.
			const auto* st = packet.Copy(strides, count); // Null: the pipeline's strides.
			packet.Commit([=] { g_real.vkCmdBindVertexBuffers2(buffer, first, count, b, o, sz, st); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdBindVertexBuffers2(buffer, first, count, buffers, offsets, sizes, strides);
}

// --- Rendering.

void VKAPI_PTR Hook_vkCmdBeginRendering(VkCommandBuffer buffer, const VkRenderingInfo* info) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkRenderingInfo>(1) +
		                     ArrayBytes<VkRenderingAttachmentInfo>(info->colorAttachmentCount) +
		                     2u * ArrayBytes<VkRenderingAttachmentInfo>(1);
		if (info->pNext == nullptr && Fits(bytes) &&
		    NoNext(info->pColorAttachments, info->colorAttachmentCount) &&
		    NoNext(info->pDepthAttachment, 1) && NoNext(info->pStencilAttachment, 1)) {
			Packet packet(bytes);
			auto*  copy              = packet.Copy(info, 1);
			copy->pColorAttachments  = packet.Copy(info->pColorAttachments, info->colorAttachmentCount);
			copy->pDepthAttachment   = packet.Copy(info->pDepthAttachment, 1);
			copy->pStencilAttachment = packet.Copy(info->pStencilAttachment, 1);
			packet.Commit([=] { g_real.vkCmdBeginRendering(buffer, copy); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdBeginRendering(buffer, info);
}

void VKAPI_PTR Hook_vkCmdBeginConditionalRenderingEXT(
    VkCommandBuffer buffer, const VkConditionalRenderingBeginInfoEXT* info) {
	if (IsRouted(buffer)) {
		if (info->pNext == nullptr) {
			Packet      packet(ArrayBytes<VkConditionalRenderingBeginInfoEXT>(1));
			const auto* copy = packet.Copy(info, 1);
			packet.Commit([=] { g_real.vkCmdBeginConditionalRenderingEXT(buffer, copy); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdBeginConditionalRenderingEXT(buffer, info);
}

// --- Copies, clears and blits.

void VKAPI_PTR Hook_vkCmdCopyBuffer(VkCommandBuffer buffer, VkBuffer source, VkBuffer target,
                                    uint32_t count, const VkBufferCopy* regions) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkBufferCopy>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* r = packet.Copy(regions, count);
			packet.Commit([=] { g_real.vkCmdCopyBuffer(buffer, source, target, count, r); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdCopyBuffer(buffer, source, target, count, regions);
}

void VKAPI_PTR Hook_vkCmdCopyImage(VkCommandBuffer buffer, VkImage source, VkImageLayout source_layout,
                                   VkImage target, VkImageLayout target_layout, uint32_t count,
                                   const VkImageCopy* regions) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkImageCopy>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* r = packet.Copy(regions, count);
			packet.Commit([=] {
				g_real.vkCmdCopyImage(buffer, source, source_layout, target, target_layout, count, r);
			});
			return;
		}
		Unsupported();
	}
	g_real.vkCmdCopyImage(buffer, source, source_layout, target, target_layout, count, regions);
}

void VKAPI_PTR Hook_vkCmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer source, VkImage target,
                                           VkImageLayout layout, uint32_t count,
                                           const VkBufferImageCopy* regions) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkBufferImageCopy>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* r = packet.Copy(regions, count);
			packet.Commit([=] { g_real.vkCmdCopyBufferToImage(buffer, source, target, layout, count, r); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdCopyBufferToImage(buffer, source, target, layout, count, regions);
}

void VKAPI_PTR Hook_vkCmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage source, VkImageLayout layout,
                                           VkBuffer target, uint32_t count,
                                           const VkBufferImageCopy* regions) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkBufferImageCopy>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* r = packet.Copy(regions, count);
			packet.Commit([=] { g_real.vkCmdCopyImageToBuffer(buffer, source, layout, target, count, r); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdCopyImageToBuffer(buffer, source, layout, target, count, regions);
}

void VKAPI_PTR Hook_vkCmdResolveImage(VkCommandBuffer buffer, VkImage source,
                                      VkImageLayout source_layout, VkImage target,
                                      VkImageLayout target_layout, uint32_t count,
                                      const VkImageResolve* regions) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkImageResolve>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* r = packet.Copy(regions, count);
			packet.Commit([=] {
				g_real.vkCmdResolveImage(buffer, source, source_layout, target, target_layout, count, r);
			});
			return;
		}
		Unsupported();
	}
	g_real.vkCmdResolveImage(buffer, source, source_layout, target, target_layout, count, regions);
}

void VKAPI_PTR Hook_vkCmdBlitImage(VkCommandBuffer buffer, VkImage source, VkImageLayout source_layout,
                                   VkImage target, VkImageLayout target_layout, uint32_t count,
                                   const VkImageBlit* regions, VkFilter filter) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkImageBlit>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* r = packet.Copy(regions, count);
			packet.Commit([=] {
				g_real.vkCmdBlitImage(buffer, source, source_layout, target, target_layout, count, r,
				                      filter);
			});
			return;
		}
		Unsupported();
	}
	g_real.vkCmdBlitImage(buffer, source, source_layout, target, target_layout, count, regions, filter);
}

void VKAPI_PTR Hook_vkCmdClearColorImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                         const VkClearColorValue* color, uint32_t count,
                                         const VkImageSubresourceRange* ranges) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkClearColorValue>(1) + ArrayBytes<VkImageSubresourceRange>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* c = packet.Copy(color, 1);
			const auto* r = packet.Copy(ranges, count);
			packet.Commit([=] { g_real.vkCmdClearColorImage(buffer, image, layout, c, count, r); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdClearColorImage(buffer, image, layout, color, count, ranges);
}

void VKAPI_PTR Hook_vkCmdClearDepthStencilImage(VkCommandBuffer buffer, VkImage image,
                                                VkImageLayout layout,
                                                const VkClearDepthStencilValue* value,
                                                uint32_t count, const VkImageSubresourceRange* ranges) {
	if (IsRouted(buffer)) {
		const size_t bytes =
		    ArrayBytes<VkClearDepthStencilValue>(1) + ArrayBytes<VkImageSubresourceRange>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* v = packet.Copy(value, 1);
			const auto* r = packet.Copy(ranges, count);
			packet.Commit([=] { g_real.vkCmdClearDepthStencilImage(buffer, image, layout, v, count, r); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdClearDepthStencilImage(buffer, image, layout, value, count, ranges);
}

void VKAPI_PTR Hook_vkCmdClearAttachments(VkCommandBuffer buffer, uint32_t attachment_count,
                                          const VkClearAttachment* attachments, uint32_t rect_count,
                                          const VkClearRect* rects) {
	if (IsRouted(buffer)) {
		const size_t bytes =
		    ArrayBytes<VkClearAttachment>(attachment_count) + ArrayBytes<VkClearRect>(rect_count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* a = packet.Copy(attachments, attachment_count);
			const auto* r = packet.Copy(rects, rect_count);
			packet.Commit([=] { g_real.vkCmdClearAttachments(buffer, attachment_count, a, rect_count, r); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdClearAttachments(buffer, attachment_count, attachments, rect_count, rects);
}

// --- Dynamic state with arrays.

void VKAPI_PTR Hook_vkCmdSetViewport(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                     const VkViewport* viewports) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkViewport>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* v = packet.Copy(viewports, count);
			packet.Commit([=] { g_real.vkCmdSetViewport(buffer, first, count, v); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdSetViewport(buffer, first, count, viewports);
}

void VKAPI_PTR Hook_vkCmdSetViewportWithCount(VkCommandBuffer buffer, uint32_t count,
                                              const VkViewport* viewports) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkViewport>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* v = packet.Copy(viewports, count);
			packet.Commit([=] { g_real.vkCmdSetViewportWithCount(buffer, count, v); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdSetViewportWithCount(buffer, count, viewports);
}

void VKAPI_PTR Hook_vkCmdSetScissor(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                    const VkRect2D* scissors) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkRect2D>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* s = packet.Copy(scissors, count);
			packet.Commit([=] { g_real.vkCmdSetScissor(buffer, first, count, s); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdSetScissor(buffer, first, count, scissors);
}

void VKAPI_PTR Hook_vkCmdSetScissorWithCount(VkCommandBuffer buffer, uint32_t count,
                                             const VkRect2D* scissors) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkRect2D>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* s = packet.Copy(scissors, count);
			packet.Commit([=] { g_real.vkCmdSetScissorWithCount(buffer, count, s); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdSetScissorWithCount(buffer, count, scissors);
}

void VKAPI_PTR Hook_vkCmdSetBlendConstants(VkCommandBuffer buffer, const float constants[4]) {
	if (IsRouted(buffer)) {
		const float c0 = constants[0];
		const float c1 = constants[1];
		const float c2 = constants[2];
		const float c3 = constants[3];
		Packet      packet(0);
		packet.Commit([=] {
			const float values[4] = {c0, c1, c2, c3};
			g_real.vkCmdSetBlendConstants(buffer, values);
		});
		return;
	}
	g_real.vkCmdSetBlendConstants(buffer, constants);
}

void VKAPI_PTR Hook_vkCmdSetColorWriteEnableEXT(VkCommandBuffer buffer, uint32_t count,
                                                const VkBool32* enables) {
	if (IsRouted(buffer)) {
		const size_t bytes = ArrayBytes<VkBool32>(count);
		if (Fits(bytes)) {
			Packet      packet(bytes);
			const auto* e = packet.Copy(enables, count);
			packet.Commit([=] { g_real.vkCmdSetColorWriteEnableEXT(buffer, count, e); });
			return;
		}
		Unsupported();
	}
	g_real.vkCmdSetColorWriteEnableEXT(buffer, count, enables);
}

// ---- Install.

struct HookedNames {
	std::vector<const char*> names;
};
HookedNames g_hooked;

[[nodiscard]] bool IsHooked(const char* name) {
	for (const char* hooked: g_hooked.names) {
		if (std::strcmp(hooked, name) == 0) {
			return true;
		}
	}
	return false;
}

} // namespace

namespace detail {

void DrainBeforeDestroySlow() {
	if (g_recorder != nullptr) {
		g_recorder->DrainForObjects(Reason::Destroy);
	}
}

} // namespace detail

void Install() {
	if (g_recorder != nullptr) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkBeginCommandBuffer == nullptr || d.vkEndCommandBuffer == nullptr ||
	    d.vkResetCommandBuffer == nullptr || d.vkQueueSubmit == nullptr) {
		std::fprintf(stderr, "RECORD_THREAD: the device functions are not loaded yet; recording stays on the calling thread\n");
		return;
	}
	g_recorder = new Recorder(); // Kept for the process: the worker may outlive any owner.

#define KYTY_INSTALL_CUSTOM_HOOK(name)                                                             \
	g_real.name = d.name;                                                                          \
	if (d.name != nullptr) {                                                                       \
		d.name = Hook_##name;                                                                      \
		g_hooked.names.push_back(#name);                                                           \
	}
	KYTY_RECORD_CUSTOM_HOOKS(KYTY_INSTALL_CUSTOM_HOOK)
#undef KYTY_INSTALL_CUSTOM_HOOK

#define KYTY_INSTALL_POD_HOOK(name, ends_batch)                                                    \
	if (d.name != nullptr) {                                                                       \
		struct Tag {};                                                                             \
		using Hook = PodHook<Tag, ends_batch, decltype(d.name)>;                                   \
		Hook::real = d.name;                                                                       \
		d.name     = Hook::Call;                                                                   \
		g_hooked.names.push_back(#name);                                                           \
	}
	KYTY_RECORD_POD_HOOKS(KYTY_INSTALL_POD_HOOK)
#undef KYTY_INSTALL_POD_HOOK

#define KYTY_INSTALL_POOL_HOOK(name)                                                               \
	if (d.name != nullptr) {                                                                       \
		struct Tag {};                                                                             \
		using Hook = DrainHook<Tag, Reason::Pool, decltype(d.name)>;                               \
		Hook::real = d.name;                                                                       \
		d.name     = Hook::Call;                                                                   \
		g_hooked.names.push_back(#name);                                                           \
	}
	KYTY_RECORD_POOL_HOOKS(KYTY_INSTALL_POOL_HOOK)
#undef KYTY_INSTALL_POOL_HOOK

#define KYTY_INSTALL_DESTROY_HOOK(name)                                                            \
	if (d.name != nullptr) {                                                                       \
		struct Tag {};                                                                             \
		using Hook = DrainHook<Tag, Reason::Destroy, decltype(d.name)>;                            \
		Hook::real = d.name;                                                                       \
		d.name     = Hook::Call;                                                                   \
		g_hooked.names.push_back(#name);                                                           \
	}
	KYTY_RECORD_DESTROY_HOOKS(KYTY_INSTALL_DESTROY_HOOK)
#undef KYTY_INSTALL_DESTROY_HOOK



#define KYTY_INSTALL_FALLBACK_HOOK(name)                                                           \
	if (d.name != nullptr && !IsHooked(#name)) {                                                   \
		struct Tag {};                                                                             \
		using Hook = FallbackHook<Tag, decltype(d.name)>;                                          \
		Hook::real = d.name;                                                                       \
		d.name     = Hook::Call;                                                                   \
		g_recorder->NoteFallback(#name, &Hook::hits);                                              \
	}
	KYTY_RECORD_FALLBACK_HOOKS(KYTY_INSTALL_FALLBACK_HOOK)
#undef KYTY_INSTALL_FALLBACK_HOOK

	g_recorder->Start();
	detail::g_active = true;
}

void Route(vk::CommandBuffer buffer) {
	if (g_recorder == nullptr || g_recorder->Stopped()) {
		return;
	}
	static bool warned = false;
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEndCommandBuffer != Hook_vkEndCommandBuffer) {
		// The dispatcher was initialized again after Install: its functions are the driver's.
		if (!warned) {
			warned = true;
			std::fprintf(stderr, "RECORD_THREAD: the dispatcher was reloaded; commands are recorded directly\n");
		}
		return;
	}
	if (g_routed.load(std::memory_order_relaxed) != nullptr) {
		// The previous buffer was not ended through vkEndCommandBuffer.
		g_recorder->Drain(Reason::Other);
	}
	g_recorder->SetOwner();
	g_routed.store(static_cast<VkCommandBuffer>(buffer), std::memory_order_release);
	g_recorder->Report(false);
}

bool AsyncSubmit() noexcept {
	return detail::g_active && g_recorder != nullptr && !g_recorder->Stopped();
}

void SyncQueue() {
	if (detail::g_active && g_recorder != nullptr) {
		g_recorder->DrainForObjects(Reason::Queue);
	}
}

void EnqueueCall(void (*call)(const void* blob), const void* blob, size_t bytes) {
	EXIT_IF(g_recorder == nullptr || call == nullptr || bytes == 0 || bytes > UINT32_MAX);
	Packet      packet(ArrayBytes<uint8_t>(static_cast<uint32_t>(bytes)));
	const auto* copy = packet.Copy(static_cast<const uint8_t*>(blob), static_cast<uint32_t>(bytes));
	packet.Commit([=] { call(copy); });
	g_recorder->PublishPending();
}

namespace {
// The packet ReserveCall opened (the recording thread only; at most one at a time).
std::optional<Packet> g_open_call;
uint8_t*              g_open_blob = nullptr;
} // namespace

void* ReserveCall(vk::CommandBuffer buffer, size_t bytes) {
	if (!detail::g_active || g_recorder == nullptr || !IsRouted(static_cast<VkCommandBuffer>(buffer))) {
		return nullptr;
	}
	EXIT_IF(g_open_call.has_value() || bytes == 0 || bytes > UINT32_MAX ||
	        !Fits(ArrayBytes<uint8_t>(static_cast<uint32_t>(bytes))));
	g_open_call.emplace(ArrayBytes<uint8_t>(static_cast<uint32_t>(bytes)));
	g_open_blob = g_open_call->Take(static_cast<uint32_t>(bytes));
	return g_open_blob;
}

void CommitCall(void (*call)(const void* blob)) {
	EXIT_IF(!g_open_call.has_value() || call == nullptr);
	const void* blob = g_open_blob;
	g_open_call->Commit([=] { call(blob); });
	g_open_call.reset();
	g_open_blob = nullptr;
}

bool RecordedByWorker(vk::CommandBuffer buffer) noexcept {
	const auto handle = static_cast<VkCommandBuffer>(buffer);
	return detail::g_active && g_recorder != nullptr && !g_recorder->Stopped() && !t_replaying && handle != nullptr && handle == g_routed.load(std::memory_order_relaxed);
}

bool QueueDescriptorWrites(VkDevice device, uint32_t count, const VkWriteDescriptorSet* writes) {
	if (count == 0 || writes == nullptr || !AsyncSubmit()) {
		return false;
	}
	auto&           q = g_write_queue;
	std::lock_guard lock(q.mutex);
	EXIT_IF(q.device != nullptr && q.device != device);
	q.device = device;
	for (uint32_t i = 0; i < count; ++i) {
		const auto& write = writes[i];
		// Every caller writes images, samplers or buffers without extensions; anything else would have to wait for the
		// writes queued before it, which a thread that does not record cannot do.
		if (write.pNext != nullptr) {
			EXIT("QueueDescriptorWrites: a descriptor write with a pNext chain\n");
		}
		QueuedWrite entry;
		entry.write                  = write;
		entry.write.pImageInfo       = nullptr;
		entry.write.pBufferInfo      = nullptr;
		entry.write.pTexelBufferView = nullptr;
		const void* source           = nullptr;
		size_t      item             = 0;
		switch (write.descriptorType) {
			case VK_DESCRIPTOR_TYPE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
				entry.kind = 0;
				source     = write.pImageInfo;
				item       = sizeof(VkDescriptorImageInfo);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
			case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
			case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
				entry.kind = 1;
				source     = write.pBufferInfo;
				item       = sizeof(VkDescriptorBufferInfo);
				break;
			case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
				entry.kind = 2;
				source     = write.pTexelBufferView;
				item       = sizeof(VkBufferView);
				break;
			default: EXIT("QueueDescriptorWrites: descriptor type %d is not copied\n", static_cast<int>(write.descriptorType));
		}
		EXIT_IF(source == nullptr || write.descriptorCount == 0);
		entry.bytes = static_cast<uint32_t>(item * write.descriptorCount);
		entry.info  = q.infos.size();
		q.infos.insert(q.infos.end(), static_cast<const uint8_t*>(source), static_cast<const uint8_t*>(source) + entry.bytes);
		q.writes.push_back(entry);
	}
	q.queued.fetch_add(count, std::memory_order_relaxed);
	q.count.store(static_cast<uint32_t>(q.writes.size()), std::memory_order_release);
	return true;
}

void Shutdown() {
	if (g_recorder == nullptr) {
		return;
	}
	g_routed.store(nullptr, std::memory_order_release);
	g_recorder->Stop();
}

} // namespace Libs::Graphics::CommandRecorder
