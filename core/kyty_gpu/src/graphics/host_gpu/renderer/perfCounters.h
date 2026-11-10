#ifndef KYTY_GRAPHICS_PERF_COUNTERS_H_
#define KYTY_GRAPHICS_PERF_COUNTERS_H_

#include "common/envRead.h"
#include "common/liveAb.h"
#include "common/onceValue.h"
#include "common/startupTrace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

// KYTY_TRACE_LONG_PREP (diagnostic): the preparing thread's ns per part of the draw being prepared (reset per draw).
// 1-7 PrepareBindings' parts (head, set, images, samplers, store, bindless, data; all stages summed),
// 8 FindBuffers, 9 PrepareBda, 10 RebindImages, 11 render targets, 12 RebindBuffers,
// 13-20 PrepareBindlessHeaps' parts (head, requests, tables frame, table uses, heap, refresh, collect, tail),
// 21, 24-25 PrepareBda's (fallback maps, region pass, epoch report),
// 26-29 counts: tracker regions taken, buffers visited, uploads, bytes uploaded,
// 30-32 PrepareBindlessTable's (read and hash, translate, refresh), 33-35 counts: tables read, translated, slots refreshed.
inline thread_local std::array<uint64_t, 43> t_long_prep_parts {};
// The calling thread holds the texture cache's lock (TrackingSpinLock of PerfLockSiteTextureCache): a preparation job must
// not wait for the previous draw's commit then (the commit takes that lock). PREP_WORKER_COMMIT_WAIT_UNDER_TC_LOCK.
inline thread_local uint32_t t_tc_lock_held = 0;
bool LongPrepPartsOn();
// The texture cache's write sequence (project A, A1). Odd while a holder of its lock is inside a
// section that may write image state (every section but the read-only ones, TrackingSpinLock::lock(false)); a reader
// that sees the same even value before and after its reads saw no such section in between (a seqlock; only the lock's
// holder writes it).
inline std::atomic<uint64_t> g_tc_write_seq {0};

// KYTY_TRACE_PERF=1: coarse time accounting (nanoseconds) reported by the GPU thread every ~5 s.
enum class PerfKind : uint32_t { Process, FenceWait, ReadMemory, EagerHit, EagerWait, SyncRead, Count };

inline bool PerfEnabled() {
	static constinit Common::OnceValue<bool> enabled_once;
	const bool enabled = enabled_once.Get([] { return Common::EnvRead::Perf(); });
	return enabled;
}

inline std::atomic<uint64_t>* PerfNanos() {
	static std::atomic<uint64_t> values[static_cast<uint32_t>(PerfKind::Count)] {};
	return values;
}

inline std::atomic<uint64_t>* PerfCalls() {
	static std::atomic<uint64_t> values[static_cast<uint32_t>(PerfKind::Count)] {};
	return values;
}

// Per-64KiB-page counts of BufferCache::ReadMemory calls (KYTY_TRACE_PERF).
inline std::mutex& PerfPageMutex() {
	static std::mutex mutex;
	return mutex;
}
inline std::unordered_map<uint64_t, uint32_t>& PerfReadPages() {
	static std::unordered_map<uint64_t, uint32_t> pages;
	return pages;
}
inline void PerfRecordRead(uint64_t vaddr, uint64_t size) {
	if (!PerfEnabled()) return;
	std::scoped_lock lock(PerfPageMutex());
	PerfReadPages()[(vaddr >> 16u) | (std::min<uint64_t>(size, 0xffffu) << 48u)]++;
}

// KYTY_TRACE_PERF_SAMPLE=1: starts a sampling profiler for the calling thread (perfSampler.cpp).
void PerfSamplerStartForCurrentThread();
// Diagnostics (a stuck wait, perfSampler.cpp): STACK_DUMP lines of another thread (a HANDLE) and of the calling thread.
void PerfDumpThreadStack(void* thread_handle, const char* label);
void PerfDumpCurrentStack(const char* label);
// KYTY_TRACE_PERF: a fixed CPU loop's time, the CPU and page faults (perfSampler.cpp; GPU thread, per report).
void CpuProbeReport();

// KYTY_TRACE_PERF: who asked for each synchronous readback (BufferCache::ReadMemory), with the time
// spent in the readbacks that really copied GPU-written bytes and waited for the GPU ("drains").
enum class PerfReason : uint32_t { Other, FaultGame, FaultGpu, SrtWalker, DccMeta, IndirectArgs, Invalidate, Count };

inline PerfReason& PerfCurrentReason() {
	static thread_local PerfReason reason = PerfReason::Other;
	return reason;
}

class PerfReasonScope {
public:
	explicit PerfReasonScope(PerfReason reason): m_previous(PerfCurrentReason()) { PerfCurrentReason() = reason; }
	~PerfReasonScope() { PerfCurrentReason() = m_previous; }
	PerfReasonScope(const PerfReasonScope&)            = delete;
	PerfReasonScope& operator=(const PerfReasonScope&) = delete;

private:
	PerfReason m_previous;
};

struct PerfReasonStats {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> drains {0};
	std::atomic<uint64_t> drain_nanos {0};
	// PERF sync_time: calls made on the GPU thread itself (run inline), time the caller waited for
	// the GPU thread to pick the command up (queue) and time the GPU thread spent running it (exec).
	std::atomic<uint64_t> gpu_thread_calls {0};
	std::atomic<uint64_t> queue_nanos {0};
	std::atomic<uint64_t> exec_nanos {0};
};

// True on the guest GPU thread (set by GuestGpu::ThreadRun; headers below cannot include graphicsRun.h).
inline bool& PerfIsGpuThread() {
	static thread_local bool flag = false;
	return flag;
}

// PERF locks (KYTY_TRACE_PERF): contended tracker region locks (RegionLock::Wait) and tracker page
// protections (PageManager -> VirtualProtect/mprotect), split by GPU thread [0] and other threads [1].
// The TrackingSpinLock a contended wait was for.
enum PerfLockSite : uint8_t {
	PerfLockSiteRegion = 0,     // a tracker region
	PerfLockSiteBufferLookup,   // BufferCache::m_lookup_lock
	PerfLockSiteTextureCache,   // TextureCache::m_lock
	PerfLockSiteCount
};
struct PerfLockStats {
	std::atomic<uint64_t> region_waits[2] {};
	std::atomic<uint64_t> region_wait_nanos[2] {};
	std::atomic<uint64_t> site_waits[PerfLockSiteCount][2] {};      // every TrackingSpinLock, by PerfLockSite
	std::atomic<uint64_t> site_wait_nanos[PerfLockSiteCount][2] {};
	std::atomic<uint64_t> protects[2] {};
	std::atomic<uint64_t> protect_nanos[2] {};
};

inline PerfLockStats& PerfLocks() {
	static PerfLockStats stats;
	return stats;
}

// KYTY_TRACE_PERF: contended waits for the texture cache's lock by code address (the caller of TrackingSpinLock::lock),
// of the holder (stored at every acquisition) and of the waiter, by the waiter's kind: 0 the GPU thread, 1 the prep
// worker, 2 other threads (PERF tc_lock_holders / tc_lock_waiters, perfSampler.cpp).
inline std::atomic<uintptr_t>& PerfTcOwnerPc() {
	static std::atomic<uintptr_t> pc {0};
	return pc;
}
struct PerfPcSlot {
	std::atomic<uintptr_t> pc {0};
	std::atomic<uint64_t>  waits[3] {};
	std::atomic<uint64_t>  nanos[3] {};
};
constexpr uint32_t PerfPcSlotCount = 512;
inline PerfPcSlot* PerfTcHolders() {
	static PerfPcSlot slots[PerfPcSlotCount];
	return slots;
}
inline PerfPcSlot* PerfTcWaiters() {
	static PerfPcSlot slots[PerfPcSlotCount];
	return slots;
}
inline void PerfNotePcWait(PerfPcSlot* slots, uintptr_t pc, uint32_t kind, uint64_t nanos) {
	for (uint32_t i = 0, h = static_cast<uint32_t>((pc * 0x9E3779B97F4A7C15ull) >> 55u); i < PerfPcSlotCount; i++) {
		auto&     slot  = slots[(h + i) & (PerfPcSlotCount - 1u)];
		uintptr_t found = slot.pc.load(std::memory_order_relaxed);
		if (found == 0 && slot.pc.compare_exchange_strong(found, pc, std::memory_order_relaxed)) found = pc;
		if (found != pc) continue;
		slot.waits[kind].fetch_add(1, std::memory_order_relaxed);
		slot.nanos[kind].fetch_add(nanos, std::memory_order_relaxed);
		return;
	}
}
// Prints and resets the two tables (graphicsRun.cpp's PERF report).
void PerfReportTcLockPcs();

inline PerfReasonStats* PerfReasons() {
	static PerfReasonStats values[static_cast<uint32_t>(PerfReason::Count)];
	return values;
}

// The shader whose resources are being materialized on this thread (KYTY_TRACE_SRT_CLASSES).
inline uint64_t& PerfMaterializeHash() {
	static thread_local uint64_t hash = 0;
	return hash;
}

inline void PerfCount(PerfKind kind) {
	if (PerfEnabled()) PerfCalls()[static_cast<uint32_t>(kind)].fetch_add(1, std::memory_order_relaxed);
}

// KYTY_TRACE_STARTUP: the PerfKind kinds its per-second lines show (busy, fence, rb).
inline Common::StartupTrace::Counter PerfStartupCounter(PerfKind kind) {
	switch (kind) {
		case PerfKind::Process: return Common::StartupTrace::Counter::GpuBusy;
		case PerfKind::FenceWait: return Common::StartupTrace::Counter::GpuFence;
		case PerfKind::ReadMemory: return Common::StartupTrace::Counter::GpuReadback;
		default: return Common::StartupTrace::Counter::Count;
	}
}

class PerfScope {
public:
	// KYTY_LIVE_AB: the GPU thread's fence waits also count for the running live A/B phase (fence_wait_ms).
	// KYTY_TRACE_STARTUP: fence waits count only on the GPU thread (the priority-operation thread waits
	// for the host GPU all the time; its wait is nobody's stall).
	explicit PerfScope(PerfKind kind)
	    : m_kind(kind), m_on(PerfEnabled()),
	      m_live(kind == PerfKind::FenceWait && PerfIsGpuThread() && Common::LiveAb::Enabled()),
	      m_startup(Common::StartupTrace::Enabled() && PerfStartupCounter(kind) != Common::StartupTrace::Counter::Count &&
	                (kind != PerfKind::FenceWait || PerfIsGpuThread())) {
		if (m_on || m_live || m_startup) m_start = std::chrono::steady_clock::now();
	}
	~PerfScope() {
		if (!m_on && !m_live && !m_startup) return;
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start)
		                    .count();
		if (m_live) Common::LiveAb::AddFenceWait(static_cast<uint64_t>(ns));
		if (m_startup) Common::StartupTrace::Add(PerfStartupCounter(m_kind), static_cast<uint64_t>(ns));
		if (!m_on) return;
		PerfNanos()[static_cast<uint32_t>(m_kind)].fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
		PerfCalls()[static_cast<uint32_t>(m_kind)].fetch_add(1, std::memory_order_relaxed);
	}
	PerfScope(const PerfScope&)            = delete;
	PerfScope& operator=(const PerfScope&) = delete;

private:
	PerfKind                              m_kind;
	bool                                  m_on;
	bool                                  m_live;
	bool                                  m_startup; // KYTY_TRACE_STARTUP
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace Libs::Graphics

#endif
