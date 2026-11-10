#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/perfCounters.h"

#include <array>
#include <atomic>
#include <chrono>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Statistics: the code address the current function returns to (TrackingSpinLock::lock's caller; lock() is not inlined
// into the texture cache's functions).
#if defined(_MSC_VER)
#define KYTY_LOCK_CALLER_PC() reinterpret_cast<uintptr_t>(_ReturnAddress())
#else
#define KYTY_LOCK_CALLER_PC() reinterpret_cast<uintptr_t>(__builtin_return_address(0))
#endif

namespace Libs::Graphics {

inline void SpinPause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

class TrackingSpinLock final {
public:
	// KYTY_TRACE_PERF: the site its contended waits are counted under (PerfLockSite).
	constexpr TrackingSpinLock() noexcept = default;
	constexpr explicit TrackingSpinLock(uint8_t perf_site) noexcept: m_perf_site(perf_site) {}

	// write = false (the texture cache's read-only sections): the section writes none of the
	// image state lock-free readers check, so it leaves g_tc_write_seq as it is. Every other section opens it (odd).
	void lock(bool write = true) noexcept {
		const auto thread = CurrentThread();
		if (m_lock.test_and_set(std::memory_order_acquire)) {
			// Contended: visible in Tracy (another thread holds the region, e.g. across a protect).
			KYTY_PROFILER_BLOCK("RegionLock::Wait");
			// KYTY_TRACE_PERF: contended waits by thread kind (PERF locks); only this branch pays for it.
			struct WaitTimer {
				bool                                  on = PerfEnabled();
				uint8_t                               site;
				uintptr_t                             holder_pc = 0; // the texture cache's lock: who holds it
				uintptr_t                             waiter_pc = 0;
				std::chrono::steady_clock::time_point start {};
				explicit WaitTimer(uint8_t s, uintptr_t pc): site(s) {
					if (!on) return;
					start = std::chrono::steady_clock::now();
					if (site == PerfLockSiteTextureCache) {
						holder_pc = PerfTcOwnerPc().load(std::memory_order_relaxed);
						waiter_pc = pc;
					}
				}
				~WaitTimer() {
					if (!on) return;
					const auto kind  = PerfIsGpuThread() ? 0u : 1u;
					const auto nanos = static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
					        .count());
					PerfLocks().region_waits[kind].fetch_add(1, std::memory_order_relaxed);
					PerfLocks().region_wait_nanos[kind].fetch_add(nanos, std::memory_order_relaxed);
					PerfLocks().site_waits[site][kind].fetch_add(1, std::memory_order_relaxed);
					PerfLocks().site_wait_nanos[site][kind].fetch_add(nanos, std::memory_order_relaxed);
					if (waiter_pc != 0) {
						const uint32_t who = PerfIsGpuThread() ? 0u : 2u;
						PerfNotePcWait(PerfTcHolders(), holder_pc, who, nanos);
						PerfNotePcWait(PerfTcWaiters(), waiter_pc, who, nanos);
					}
				}
			} wait_timer(m_perf_site, KYTY_LOCK_CALLER_PC());
			// Other threads' lookup batches let a waiting GPU thread in (GpuWaiting).
			const bool gpu_thread = PerfIsGpuThread();
			if (gpu_thread) m_gpu_waiting.store(true, std::memory_order_relaxed);
			uint32_t spins = 0;
			do {
				EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
				// Test before test-and-set: waiting reads do not pull the line away from the holder.
				while (m_lock.test(std::memory_order_relaxed)) {
					SpinPause();
					// After ~25 us of spinning: gives up the time slice, so a holder preempted inside a protect can run.
					if (++spins >= 2048u) {
						spins = 0;
						std::this_thread::yield();
					}
				}
			} while (m_lock.test_and_set(std::memory_order_acquire));
			if (gpu_thread) m_gpu_waiting.store(false, std::memory_order_relaxed);
		}
		m_owner.store(thread, std::memory_order_relaxed);
		if (m_perf_site == PerfLockSiteTextureCache) {
			t_tc_lock_held++;
			if (write) OpenWrite();
		}
		// KYTY_TRACE_PERF: the texture cache lock's holder (PerfTcOwnerPc).
		if (m_perf_site == PerfLockSiteTextureCache && PerfEnabled()) {
			PerfTcOwnerPc().store(KYTY_LOCK_CALLER_PC(), std::memory_order_relaxed);
		}
	}
	// The GPU thread is spinning for this lock now (a hint; relaxed).
	[[nodiscard]] bool GpuWaiting() const noexcept { return m_gpu_waiting.load(std::memory_order_relaxed); }
	// The resolve thread's lookups: takes the lock only if it is free, never waits. A failed
	// attempt only reads the lock word (it does not pull the line away from the holder).
	[[nodiscard]] bool try_lock() noexcept {
		if (m_lock.test(std::memory_order_relaxed) || m_lock.test_and_set(std::memory_order_acquire)) {
			return false;
		}
		m_owner.store(CurrentThread(), std::memory_order_relaxed);
		if (m_perf_site == PerfLockSiteTextureCache) {
			t_tc_lock_held++;
			OpenWrite();
		}
		return true;
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		if (m_perf_site == PerfLockSiteTextureCache) {
			t_tc_lock_held--;
			if (m_write_open) { // the section's writes are done (even again)
				m_write_open = false;
				g_tc_write_seq.store(g_tc_write_seq.load(std::memory_order_relaxed) + 1u, std::memory_order_release);
			}
		}
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	// The holder may write image state from here on (odd); the fence orders that before its writes.
	void OpenWrite() noexcept {
		m_write_open = true;
		g_tc_write_seq.store(g_tc_write_seq.load(std::memory_order_relaxed) + 1u, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
	}

	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
	std::atomic_bool     m_gpu_waiting {false};
	uint8_t              m_perf_site = PerfLockSiteRegion;
	bool                 m_write_open = false; // the holder's section opened g_tc_write_seq (texture cache only)
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

inline std::atomic<uint64_t> g_cpu_dirty_epoch {1};

// The obtain and write memos (renderer/cache/bufferCache.cpp): one epoch per tracker region, set to the next value of
// g_cpu_dirty_region_clock whenever MarkCpuDirtyRegion (or a deferred mark) marks the region: a region's epoch changes at
// every mark, and a region whose epoch is newer than a clock value read earlier was marked since. Null (nothing counted)
// before the BufferCache exists.
inline std::atomic<std::atomic<uint32_t>*> g_cpu_dirty_region_epochs {nullptr};
inline std::atomic<uint32_t>               g_cpu_dirty_region_clock {0};

inline void NoteCpuDirtyRegionEpoch(std::atomic<uint32_t>* epochs, uint64_t region) {
	epochs[region].store(g_cpu_dirty_region_clock.fetch_add(1, std::memory_order_relaxed) + 1u, std::memory_order_release);
}

// The write memo (renderer/cache/bufferCache.cpp): one epoch per tracker region, set to the next value of
// g_cpu_dirty_region_clock whenever GPU-modified pages of the region stop being GPU-modified (a download, a readback, an
// unmark, an untrack) or the buffer cache's GPU-dirty set loses bytes there. Null (nothing counted) before the BufferCache
// exists.
inline std::atomic<std::atomic<uint32_t>*> g_gpu_clean_region_epochs {nullptr};
inline void NoteGpuCleanRegion(uint64_t region) {
	if (auto* epochs = g_gpu_clean_region_epochs.load(std::memory_order_acquire); epochs != nullptr) {
		NoteCpuDirtyRegionEpoch(epochs, region);
	}
}
inline void NoteGpuCleanRange(uint64_t vaddr, uint64_t size) {
	if (size == 0 || g_gpu_clean_region_epochs.load(std::memory_order_acquire) == nullptr) return;
	for (uint64_t region = vaddr / TRACKER_REGION_SIZE; region <= (vaddr + size - 1u) / TRACKER_REGION_SIZE; region++) {
		NoteGpuCleanRegion(region);
	}
}

// Which 4 MB tracker regions became CPU-dirty (or got a buffer) since the GPU thread last looked.
// Two levels so the consumer scans little: bottom = 1 bit per region, top = 1 bit per bottom word.
// It lives outside RegionManager so it exists for regions that have no manager yet. Writers set the
// bits (release) BEFORE bumping g_cpu_dirty_epoch; the single consumer (GPU thread, PrepareBda)
// reads the epoch first, then takes the bits, so a cleared bit is set again by any later dirtying.
constexpr uint64_t CPU_DIRTY_REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
static_assert(CPU_DIRTY_REGION_COUNT % 4096 == 0);

struct CpuDirtyRegionMap {
	std::atomic<uint64_t> bottom[CPU_DIRTY_REGION_COUNT / 64] {};
	std::atomic<uint64_t> top[CPU_DIRTY_REGION_COUNT / 4096] {};
};
inline CpuDirtyRegionMap g_cpu_dirty_regions;

inline void MarkCpuDirtyRegion(uint64_t region) {
	if (region >= CPU_DIRTY_REGION_COUNT) {
		EXIT("CPU dirty region lies outside the tracked address space\n");
	}
	g_cpu_dirty_regions.bottom[region / 64].fetch_or(1ull << (region % 64), std::memory_order_release);
	g_cpu_dirty_regions.top[region / 4096].fetch_or(1ull << ((region / 64) % 64),
	                                                std::memory_order_release);
	if (auto* epochs = g_cpu_dirty_region_epochs.load(std::memory_order_acquire); epochs != nullptr) {
		NoteCpuDirtyRegionEpoch(epochs, region);
	}
}

// Every region touched by [vaddr, vaddr + size).
inline void MarkCpuDirtyRange(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	const auto last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
	for (auto region = vaddr / TRACKER_REGION_SIZE; region <= last; ++region) {
		MarkCpuDirtyRegion(region);
	}
}

// Takes (clears) all pending regions and calls func(region_index) for each. One consumer only.
template <typename Func>
void TakeCpuDirtyRegions(Func&& func) {
	for (uint64_t top_index = 0; top_index < CPU_DIRTY_REGION_COUNT / 4096; ++top_index) {
		auto& top_word = g_cpu_dirty_regions.top[top_index];
		if (top_word.load(std::memory_order_relaxed) == 0) {
			continue;
		}
		auto top = top_word.exchange(0, std::memory_order_acq_rel);
		while (top != 0) {
			const auto bottom_index = top_index * 64 + static_cast<uint64_t>(std::countr_zero(top));
			top &= top - 1;
			auto bottom =
			    g_cpu_dirty_regions.bottom[bottom_index].exchange(0, std::memory_order_acq_rel);
			while (bottom != 0) {
				func(bottom_index * 64 + static_cast<uint64_t>(std::countr_zero(bottom)));
				bottom &= bottom - 1;
			}
		}
	}
}

// BDA sync epoch (RenderContext::HandleFault, PrepareBda): a guest thread's own write can only be
// visible to GPU work the guest orders after it: a later submission or a label the GPU waits for (on the console
// a write after the submission without one races the GPU). While g_cpu_dirty_deferred is set (a guest thread's
// write fault), the regions it dirties wait in g_cpu_deferred_regions and g_cpu_dirty_epoch does not move. The
// GPU thread moves them into g_cpu_dirty_regions (and moves the epoch) where its command stream orders guest
// writes before the commands that follow (PublishCpuDirtyRegions), and moves the deferred regions of a guest
// range it writes itself (PublishCpuDirtyRange: a page a guest write made writable takes that write without a
// fault, and the commands after it must see it). Writers set bottom, top, then group; the GPU thread clears the
// group bits, then takes top and bottom (all read-modify-writes, so every writer's bits are seen by the
// exchange that takes its group bit).
inline CpuDirtyRegionMap     g_cpu_deferred_regions;
inline std::atomic<uint64_t> g_cpu_deferred_groups {0}; // 1 bit per 64 top words
inline thread_local bool     g_cpu_dirty_deferred = false;
static_assert(CPU_DIRTY_REGION_COUNT / 4096 / 64 <= 64 && (CPU_DIRTY_REGION_COUNT / 4096) % 64 == 0);

// BDA sync epoch counters (relaxed; the PERF line in PrepareBda).
struct CpuDeferredCounters {
	std::atomic<uint64_t> deferred_marks {0};    // guest-thread marks that waited
	std::atomic<uint64_t> publishes {0};         // PublishCpuDirtyRegions calls that moved a region
	std::atomic<uint64_t> published_regions {0}; // regions they moved
	std::atomic<uint64_t> write_regions {0};     // regions moved by a GPU-thread write (PublishCpuDirtyRange)
};
inline CpuDeferredCounters g_cpu_deferred_counters;

// Sets g_cpu_dirty_deferred for the current thread while it lives (restores the previous value).
class CpuDirtyDeferScope final {
public:
	explicit CpuDirtyDeferScope(bool defer) noexcept: m_previous(g_cpu_dirty_deferred) {
		g_cpu_dirty_deferred = defer;
	}
	~CpuDirtyDeferScope() { g_cpu_dirty_deferred = m_previous; }
	KYTY_CLASS_NO_COPY(CpuDirtyDeferScope);

private:
	bool m_previous;
};

inline void MarkCpuDeferredRegion(uint64_t region) {
	if (region >= CPU_DIRTY_REGION_COUNT) {
		EXIT("CPU dirty region lies outside the tracked address space\n");
	}
	const auto top_index = region / 4096;
	g_cpu_deferred_regions.bottom[region / 64].fetch_or(1ull << (region % 64), std::memory_order_release);
	g_cpu_deferred_regions.top[top_index].fetch_or(1ull << ((region / 64) % 64), std::memory_order_release);
	g_cpu_deferred_groups.fetch_or(1ull << (top_index / 64), std::memory_order_acq_rel);
	g_cpu_deferred_counters.deferred_marks.fetch_add(1, std::memory_order_relaxed);
	// The obtain and write memos: the bytes changed now, whenever the GPU work may see them (as
	// MarkCpuDirtyRegion).
	if (auto* epochs = g_cpu_dirty_region_epochs.load(std::memory_order_acquire); epochs != nullptr) {
		NoteCpuDirtyRegionEpoch(epochs, region);
	}
}

// RegionManager::ChangeState: a region got CPU-dirty pages. Visible to the next PrepareBda pass at once, or
// deferred while the current thread is a guest thread's write (g_cpu_dirty_deferred).
inline void NoteCpuDirtyRegion(uint64_t region) {
	if (g_cpu_dirty_deferred) {
		MarkCpuDeferredRegion(region);
		return;
	}
	MarkCpuDirtyRegion(region);
	g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
}

// GPU thread: every deferred region becomes visible to the next PrepareBda pass. Called after the command stream
// read the guest memory that orders the guest's earlier writes (a submission slice starts, a WAIT_REG_MEM passes,
// a packet decides on guest memory). True when a region moved.
inline bool HasDeferredCpuDirtyRegions() {
	return g_cpu_deferred_groups.load(std::memory_order_acquire) != 0;
}

inline bool PublishCpuDirtyRegions() {
	// Keeps the caller's read of the guest memory (label, predicate) before the reads below.
	std::atomic_thread_fence(std::memory_order_acquire);
	// A writer whose group bit is not visible yet is concurrent with this point: the next one takes it.
	if (g_cpu_deferred_groups.load(std::memory_order_relaxed) == 0) {
		return false;
	}
	uint64_t moved  = 0;
	auto     groups = g_cpu_deferred_groups.exchange(0, std::memory_order_acq_rel);
	while (groups != 0) {
		const auto group = static_cast<uint64_t>(std::countr_zero(groups));
		groups &= groups - 1;
		for (uint64_t top_index = group * 64; top_index < group * 64 + 64; ++top_index) {
			auto& top_word = g_cpu_deferred_regions.top[top_index];
			if (top_word.load(std::memory_order_relaxed) == 0) {
				continue;
			}
			auto top = top_word.exchange(0, std::memory_order_acq_rel);
			while (top != 0) {
				const auto bottom_bit   = static_cast<uint64_t>(std::countr_zero(top));
				const auto bottom_index = top_index * 64 + bottom_bit;
				top &= top - 1;
				const auto bits =
				    g_cpu_deferred_regions.bottom[bottom_index].exchange(0, std::memory_order_acq_rel);
				if (bits == 0) {
					continue;
				}
				g_cpu_dirty_regions.bottom[bottom_index].fetch_or(bits, std::memory_order_release);
				g_cpu_dirty_regions.top[top_index].fetch_or(1ull << bottom_bit, std::memory_order_release);
				moved += static_cast<uint64_t>(std::popcount(bits));
			}
		}
	}
	if (moved == 0) {
		return false;
	}
	g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
	g_cpu_deferred_counters.publishes.fetch_add(1, std::memory_order_relaxed);
	g_cpu_deferred_counters.published_regions.fetch_add(moved, std::memory_order_relaxed);
	return true;
}

// GPU thread: it wrote [vaddr, vaddr + size) of guest memory itself (CP packets, labels, CPU fills and copies).
// The deferred regions of the range become visible now (see above). Ranges outside the tracked space are ignored.
inline void PublishCpuDirtyRange(uint64_t vaddr, uint64_t size) {
	if (size == 0 || vaddr >= TRACKER_ADDRESS_SIZE) {
		return;
	}
	const uint64_t end   = size > TRACKER_ADDRESS_SIZE - vaddr ? TRACKER_ADDRESS_SIZE : vaddr + size;
	const auto     last  = (end - 1) / TRACKER_REGION_SIZE;
	bool           moved = false;
	for (auto region = vaddr / TRACKER_REGION_SIZE; region <= last; ++region) {
		auto&      word = g_cpu_deferred_regions.bottom[region / 64];
		const auto bit  = 1ull << (region % 64);
		if ((word.load(std::memory_order_acquire) & bit) == 0 ||
		    (word.fetch_and(~bit, std::memory_order_acq_rel) & bit) == 0) {
			continue;
		}
		MarkCpuDirtyRegion(region);
		g_cpu_deferred_counters.write_regions.fetch_add(1, std::memory_order_relaxed);
		moved = true;
	}
	if (moved) {
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
	}
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		MarkCpuDirtyRegion(m_cpu_addr / TRACKER_REGION_SIZE);
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }

	template <DirtySource source>
	[[nodiscard]] bool MaybeModified() const noexcept {
		return GetSummary<source>().load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	// apply_protection = false (ReleaseGpuToCpu): the bookkeeping only; the caller applies the protection update itself.
	template <DirtySource source, bool enable, bool apply_protection = true>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);

		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		if constexpr (enable) {
			bits.SetRange(start, end);
			GetSummary<source>().store(true, std::memory_order_release);
			if constexpr (source == DirtySource::Cpu) {
				NoteCpuDirtyRegion(m_cpu_addr / TRACKER_REGION_SIZE);
			}
		} else {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				GetSummary<source>().store(false, std::memory_order_release);
			}
			if constexpr (source == DirtySource::Gpu) {
				NoteGpuCleanRegion(m_cpu_addr / TRACKER_REGION_SIZE); // the write memo
			}
		}
		if constexpr (!apply_protection) {
			return;
		} else if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	// MemoryTracker::ReleaseGpuToCpu, with the lock held: ChangeState<Gpu, false>
	// followed by ChangeState<Cpu, true> on [vaddr, vaddr + size) (BufferCache::ReadMemory's write request after the
	// page's GPU bytes were published or dropped), with the same bookkeeping in the same order, but the opening halves of
	// their two protection updates (access side, then write side) are applied in one page-manager pass: a page that goes
	// from GPU-modified to CPU-dirty becomes read-write in one protection call instead of read-only, then read-write.
	// The applied state afterwards is the one the two updates leave. A pending closing change of the write side (a claim
	// of the upload batch waiting for its commit) is applied after the pass, as UpdateProtection<false, false>
	// applies it after its opening half; a closing change of the access side (never pending: GPU marks apply at once)
	// makes it run today's two updates instead. True when the merged pass was taken.
	bool ReleaseGpuToCpu(uint64_t vaddr, uint64_t size) {
		ChangeState<DirtySource::Gpu, false, false>(vaddr, size);
		ChangeState<DirtySource::Cpu, true, false>(vaddr, size);
		const RegionBits readable     = ~m_gpu_dirty;
		const RegionBits read_mask    = readable ^ m_readable;
		const RegionBits read_closing = read_mask & m_readable;
		if (read_closing.Any()) {
			UpdateProtection<false, true>();
			UpdateProtection<false, false>();
			return false;
		}
		const RegionBits read_opening  = read_mask & readable;
		const RegionBits write_mask    = m_cpu_dirty ^ m_writable;
		const RegionBits write_closing = write_mask & m_writable;
		const RegionBits write_opening = write_mask & m_cpu_dirty;
		m_readable                     = readable;
		m_writable                     = m_cpu_dirty;
		if (read_opening.Any() || write_opening.Any()) {
			m_page_manager.OpenPageWatchersForRegion(m_cpu_addr, read_opening, write_opening);
		}
		if (write_closing.Any()) {
			auto bits = write_closing;
			m_page_manager.UpdatePageWatchersForRegion<true, false>(m_cpu_addr, bits);
		}
		return true;
	}

	// defer_protection (the upload batch, CPU-dirty walks that clear): the cleared pages stay writable; true is
	// returned when such pages are left for CommitDeferredProtection (MemoryTracker::ClaimUploadRanges). Otherwise false.
	template <DirtySource source, bool clear, typename Func>
	bool ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func, [[maybe_unused]] bool defer_protection = false) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return false;
		}
		RegionBits mask(bits, start, end);
		bool       deferred = false;

		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if (bits.None()) {
				GetSummary<source>().store(false, std::memory_order_release);
			}
			if constexpr (source == DirtySource::Gpu) {
				NoteGpuCleanRegion(m_cpu_addr / TRACKER_REGION_SIZE); // the write memo
			}
			if constexpr (source == DirtySource::Cpu) {
				if (defer_protection) {
					deferred = true;
				} else {
					UpdateProtection<true, false>();
				}
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
		return deferred;
	}

	// The upload batch (MemoryTracker::ClaimUploadRanges / CommitDeferredProtections), all with the lock held:
	// a claim cleared CPU-dirty pages of this region and left them writable (recorded once per region until the
	// commit); the commit write-protects every CPU-clean page that is still writable (UpdateProtection compares the
	// CPU-dirty bits with the applied protection, so pages another thread's update protected meanwhile, or that became
	// CPU-dirty again, need nothing). True when the commit changed a protection.
	[[nodiscard]] bool ProtectionDeferred() const { return m_protect_deferred; }
	void               SetProtectionDeferred() { m_protect_deferred = true; }
	bool               CommitDeferredProtection() {
		m_protect_deferred = false;
		return UpdateProtection<true, false>();
	}

	TrackingSpinLock lock;

private:

	// Applies the page manager's watchers for every page whose state differs from what was applied last. True when
	// anything changed. `track` is the direction of the caller's own change (true: pages lose access, a watcher is
	// added). Pages changed the other way since the last update are applied with the other direction: the caller's
	// update applies every pending page, and an upload batch's claim leaves CPU-clean pages writable
	// until its commit, so a CPU-dirty mark in the same region meanwhile (another thread's write fault, an
	// invalidation) finds pages of both kinds. Each page's watcher changes exactly once per state change either way.
	template <bool track, bool is_read>
	bool UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		const auto mask       = protection ^ previous;
		if (mask.None()) {
			return false;
		}
		// A set bit of `protection` / `previous` means accessible (writable for the CPU side, readable for the GPU side).
		const RegionBits closing = mask & previous;   // accessible before, not now: add a watcher
		const RegionBits opening = mask & protection; // not accessible before, accessible now: remove a watcher
		previous                 = protection;
		auto apply_closing = [&] {
			if (closing.Any()) {
				auto bits = closing;
				m_page_manager.UpdatePageWatchersForRegion<true, is_read>(m_cpu_addr, bits);
			}
		};
		auto apply_opening = [&] {
			if (opening.Any()) {
				auto bits = opening;
				m_page_manager.UpdatePageWatchersForRegion<false, is_read>(m_cpu_addr, bits);
			}
		};
		if constexpr (track) {
			apply_closing();
			apply_opening();
		} else {
			apply_opening();
			apply_closing();
		}
		return true;
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	std::atomic<bool>& GetSummary() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	template <DirtySource source>
	const std::atomic<bool>& GetSummary() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_maybe_dirty;
		} else {
			return m_gpu_maybe_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	std::atomic<bool> m_cpu_maybe_dirty {true};
	std::atomic<bool> m_gpu_maybe_dirty {false};
	RegionBits   m_writable;
	RegionBits   m_readable;
	// The upload batch: a claim left CPU-clean pages of this region writable (ProtectionDeferred).
	bool m_protect_deferred = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
