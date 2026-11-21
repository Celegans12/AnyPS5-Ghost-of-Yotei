#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/threads.h"
#include "common/threadSafety.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#pragma comment(lib, "Synchronization.lib")
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

enum class SpinLockClass : uint8_t { Other, Tracker, Pages, Textures };

struct SpinLockCounters {
	std::atomic<uint64_t> waits {0};
	std::atomic<uint64_t> ns {0};
};

class KYTY_CAPABILITY("spin lock") TrackingSpinLock final {
public:
	static constexpr size_t CLASSES = 4;

	struct Contention {
		uint64_t waits = 0;
		uint64_t ns    = 0;
	};

	TrackingSpinLock() noexcept = default;
	explicit TrackingSpinLock(SpinLockClass kind) noexcept: m_class(kind) {}
	TrackingSpinLock(const TrackingSpinLock&)            = delete;
	TrackingSpinLock& operator=(const TrackingSpinLock&) = delete;

	[[nodiscard]] static Contention TakeContention(SpinLockClass kind) noexcept {
		auto& counters = s_contention[static_cast<size_t>(kind)];
		return {.waits = counters.waits.exchange(0, std::memory_order_relaxed),
		        .ns    = counters.ns.exchange(0, std::memory_order_relaxed)};
	}

	void lock() noexcept KYTY_ACQUIRE() KYTY_NO_THREAD_SAFETY_ANALYSIS {
		const auto thread = CurrentThread();
		if (m_state.exchange(1, std::memory_order_acquire) != 0) [[unlikely]] {
			LockSlow(thread);
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept KYTY_RELEASE() KYTY_NO_THREAD_SAFETY_ANALYSIS {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_state.store(0, std::memory_order_seq_cst);
		if (m_sleepers.load(std::memory_order_seq_cst) != 0) [[unlikely]] {
			WakeSleeper();
		}
	}
	[[nodiscard]] bool TryLockSpinning() noexcept KYTY_TRY_ACQUIRE(true) KYTY_NO_THREAD_SAFETY_ANALYSIS {
		const auto thread = CurrentThread();
		for (uint32_t round = 0;; round++) {
			if (m_state.load(std::memory_order_relaxed) == 0 &&
			    m_state.exchange(1, std::memory_order_acquire) == 0) {
				m_owner.store(thread, std::memory_order_relaxed);
				return true;
			}
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			if (round == SpinRounds) {
				return false;
			}
			Relax();
		}
	}

private:
	static constexpr uint32_t SpinRounds = 4096;

	static void Relax() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
		_mm_pause();
#elif defined(_M_ARM64)
		__yield();
#elif defined(__aarch64__)
		asm volatile("yield");
#endif
	}

#if defined(_MSC_VER) && !defined(__clang__)
	__declspec(noinline)
#else
	[[gnu::noinline]]
#endif
	void LockSlow(uint32_t thread) noexcept {
		const bool own_core = Common::Thread::HasReservedCore();
		const auto started  = std::chrono::steady_clock::now();
		for (uint32_t round = 0;; round++) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			if (round < SpinRounds || own_core) {
				Relax();
			} else {
				m_sleepers.fetch_add(1, std::memory_order_seq_cst);
				SleepWhileHeld();
				m_sleepers.fetch_sub(1, std::memory_order_relaxed);
			}
			if (m_state.load(std::memory_order_relaxed) == 0 &&
			    m_state.exchange(1, std::memory_order_acquire) == 0) {
				const auto waited = std::chrono::steady_clock::now() - started;
				auto&      counters = s_contention[static_cast<size_t>(m_class)];
				counters.waits.fetch_add(1, std::memory_order_relaxed);
				counters.ns.fetch_add(
				    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(waited).count()),
				    std::memory_order_relaxed);
				return;
			}
		}
	}

	void SleepWhileHeld() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		uint32_t held = 1;
		(void)::WaitOnAddress(&m_state, &held, sizeof(held), 1);
#else
		std::this_thread::yield();
#endif
	}

	void WakeSleeper() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		::WakeByAddressSingle(&m_state);
#endif
	}

	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		static thread_local const uint32_t tid = GetCurrentThreadId();
		return tid;
#elif defined(__APPLE__)
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

	inline static std::array<SpinLockCounters, CLASSES> s_contention {};

	std::atomic_uint32_t m_state {0};
	std::atomic_uint32_t m_owner {0};
	std::atomic_uint32_t m_sleepers {0};
	SpinLockClass        m_class = SpinLockClass::Other;
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	using DeferredRanges = std::vector<PageManager::DeferredRange>;

	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		m_cpu_dirty.Fill();
		for (auto& written: m_cpu_written) {
			written.Fill();
		}
		s_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
	}

	[[nodiscard]] static uint64_t CpuDirtyEpoch() {
		return s_cpu_dirty_epoch.load(std::memory_order_acquire);
	}

	[[nodiscard]] uint64_t RegionCpuDirtyEpoch() const {
		return m_cpu_dirty_epoch.load(std::memory_order_acquire);
	}

	[[nodiscard]] uint64_t CpuModificationEpoch() const {
		if (m_deferred_protects.load(std::memory_order_acquire) != 0) return 0;
		return m_cpu_dirty_epoch.load(std::memory_order_acquire) + m_gpu_clean_epoch.load(std::memory_order_acquire) +
		       m_host_write_epoch.load(std::memory_order_acquire);
	}
	void BeginDeferredProtection() noexcept { m_deferred_protects.fetch_add(1, std::memory_order_acq_rel); }
	[[nodiscard]] bool DeferredProtectionPending() const noexcept {
		return m_deferred_protects.load(std::memory_order_acquire) != 0;
	}
	void ApplyDeferredProtection(uint64_t vaddr, uint64_t size) {
		m_page_manager.ReapplyProtection(vaddr, size);
		m_host_write_epoch.fetch_add(1, std::memory_order_acq_rel);
		m_deferred_protects.fetch_sub(1, std::memory_order_acq_rel);
	}
	void NoteHostWrite() noexcept { m_host_write_epoch.fetch_add(1, std::memory_order_release); }

	[[nodiscard]] static uint64_t GpuCleanEpoch() {
		return s_gpu_clean_epoch.load(std::memory_order_acquire);
	}

	[[nodiscard]] uint64_t RegionGpuCleanEpoch() const {
		return m_gpu_clean_epoch.load(std::memory_order_acquire);
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] bool EverGpuWritten(uint64_t offset, uint64_t size) const KYTY_NO_THREAD_SAFETY_ANALYSIS {
		const auto [start, end] = GetPageRange(offset, size);
		return m_gpu_written.Any(start, end);
	}

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		return GetBits<source>().Any(start, end);
	}

	template <DirtySource source>
	[[nodiscard]] bool IsModifiedUnlocked(uint64_t offset, uint64_t size) const KYTY_NO_THREAD_SAFETY_ANALYSIS {
		const bool any = source == DirtySource::Cpu ? m_has_cpu_dirty.load(std::memory_order_acquire)
		                                            : m_has_gpu_dirty.load(std::memory_order_acquire);
		return any && IsModified<source>(offset, size);
	}

	struct WriteBlockProbe {
		uint32_t opened  = 0;
		bool     gpu     = false;
		bool     tracked = false;
	};

	[[nodiscard]] WriteBlockProbe OpenWriteBlock(uint64_t offset, uint64_t block_size, bool open_block) KYTY_REQUIRES(lock) {
		const auto page  = static_cast<size_t>(offset / TRACKER_PAGE_SIZE);
		const auto first = static_cast<size_t>(offset / block_size * block_size / TRACKER_PAGE_SIZE);
		const auto last  = first + static_cast<size_t>(block_size / TRACKER_PAGE_SIZE);
		WriteBlockProbe probe {.gpu = m_gpu_written.Any(first, last), .tracked = !m_cpu_dirty.Get(page)};
		NoteGuestWriteFault(offset);
		if (!open_block || m_gpu_dirty.Get(page)) {
			return probe;
		}
		probe.opened = OpenBlockPages(page, first, last);
		return probe;
	}

	struct WriteFaultProbe {
		uint32_t opened  = 0;
		bool     gpu     = false;
		bool     tracked = false;
		bool     done    = false;
	};

	[[nodiscard]] WriteFaultProbe FaultWrite(uint64_t offset, uint64_t block_size, bool open_block, uint64_t mark_size)
	    KYTY_REQUIRES(lock) {
		const auto page  = static_cast<size_t>(offset / TRACKER_PAGE_SIZE);
		const auto first = static_cast<size_t>(offset / block_size * block_size / TRACKER_PAGE_SIZE);
		const auto last  = first + static_cast<size_t>(block_size / TRACKER_PAGE_SIZE);
		WriteFaultProbe probe {.gpu = m_gpu_written.Any(first, last), .tracked = !m_cpu_dirty.Get(page)};
		NoteGuestWriteFault(offset);
		if (open_block) {
			if (!m_gpu_dirty.Get(page)) {
				probe.opened = OpenBlockPages(page, first, last);
				probe.done   = true;
			}
			return probe;
		}
		const auto [start, end] = GetPageRange(offset, mark_size);
		if (!m_gpu_dirty.Any(start, end)) {
			ChangeState<DirtySource::Cpu, true>(offset, mark_size);
			probe.done = true;
		}
		return probe;
	}

	[[nodiscard]] bool WriteFaultFindsOpen(uint64_t offset, uint64_t block_size, bool open_block, uint64_t mark_size,
	                                       bool* gpu) const KYTY_NO_THREAD_SAFETY_ANALYSIS {
		const auto page  = static_cast<size_t>(offset / TRACKER_PAGE_SIZE);
		const auto first = static_cast<size_t>(offset / block_size * block_size / TRACKER_PAGE_SIZE);
		const auto last  = first + static_cast<size_t>(block_size / TRACKER_PAGE_SIZE);
		if ((!m_cpu_dirty.Get(page) && !m_in_place.Get(page)) || m_gpu_dirty.Get(page)) {
			return false;
		}
		*gpu = m_gpu_written.Any(first, last);
		if (!open_block) {
			const auto [start, end] = GetPageRange(offset, mark_size);
			RegionBits closed(~m_cpu_dirty, start, end);
			closed &= RegionBits(~m_in_place, start, end);
			return closed.None() && !m_gpu_dirty.Any(start, end);
		}
		RegionBits closed(~m_cpu_dirty, first, last);
		closed &= RegionBits(~m_gpu_dirty, first, last);
		closed &= RegionBits(~m_gpu_noted, first, last);
		closed &= RegionBits(~m_in_place, first, last);
		return closed.None();
	}

	void NoteCpuWriteOnOpenPage(uint64_t offset, uint64_t block_size, bool open_block, uint64_t mark_size) noexcept
	    KYTY_NO_THREAD_SAFETY_ANALYSIS {
		const auto page         = static_cast<size_t>(offset / TRACKER_PAGE_SIZE);
		const auto first        = static_cast<size_t>(offset / block_size * block_size / TRACKER_PAGE_SIZE);
		const auto last         = first + static_cast<size_t>(block_size / TRACKER_PAGE_SIZE);
		const auto [start, end] = open_block ? std::pair {first, last} : GetPageRange(offset, mark_size);
		RegionBits written(m_cpu_dirty, start, end);
		if (open_block) {
			RegionBits noted(m_gpu_noted, first, last);
			noted.Unset(page);
			written &= ~noted;
		}
		for (auto& consumer: m_cpu_written) {
			consumer.SetAtomic(written);
		}
		NoteGuestWriteFault(offset);
		m_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
		s_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
	}

	[[nodiscard]] static uint64_t TakeOpenedUploadBytes() noexcept {
		return s_opened_upload_bytes.exchange(0, std::memory_order_relaxed);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t offset, uint64_t size) KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		if constexpr (enable) {
			if (m_any_in_place && m_in_place.Any(start, end)) {
				for (const auto [first, last]: RegionBits(~m_in_place, start, end)) {
					ChangeState<source, enable>(first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
				}
				return;
			}
		}
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.Any(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		auto&      bits = GetBits<source>();
		RegionBits changed = enable ? RegionBits(~bits, start, end) : RegionBits(bits, start, end);
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			if constexpr (enable) {
				for (auto& written: m_cpu_written) {
					written.SetRangeAtomic(start, end);
				}
				m_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
				s_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
				BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
			}
			if constexpr (!enable) {
				NoteOpenedUpload(nullptr, changed);
			}
			m_has_cpu_dirty.store(!m_cpu_dirty.None(), std::memory_order_release);
			UpdateProtection<!enable, false>(changed);
		} else {
			if constexpr (!enable) {
				m_gpu_clean_epoch.fetch_add(1, std::memory_order_acq_rel);
				s_gpu_clean_epoch.fetch_add(1, std::memory_order_acq_rel);
			} else {
				m_gpu_written.SetRange(start, end);
			}
			UpdateProtection<enable, true>(changed);
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t offset, uint64_t size, Func&& func, DeferredRanges* deferred = nullptr)
	    KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		auto&      bits         = GetBits<source>();
		if (!bits.Any(start, end)) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				NoteOpenedUpload(&mask, mask);
				m_has_cpu_dirty.store(!m_cpu_dirty.None(), std::memory_order_release);
				UpdateProtection<true, false>(mask, deferred);
			} else {
				m_gpu_clean_epoch.fetch_add(1, std::memory_order_acq_rel);
				s_gpu_clean_epoch.fetch_add(1, std::memory_order_acq_rel);
				UpdateProtection<false, true>(mask);
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	template <typename Func>
	void ConsumeCpuDirtyRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges, Func&& func,
	                           DeferredRanges* deferred = nullptr) KYTY_REQUIRES(lock) {
		RegionBits mask;
		for (const auto& [offset, size]: ranges) {
			const auto [start, end] = GetPageRange(offset, size);
			if (m_cpu_dirty.FirstRangeFrom(start).first < end) {
				mask |= RegionBits(m_cpu_dirty, start, end);
			}
		}
		if (mask.None()) {
			return;
		}
		for (const auto [first, last]: mask) {
			m_cpu_dirty.UnsetRange(first, last);
		}
		NoteOpenedUpload(&mask, mask);
		m_has_cpu_dirty.store(!m_cpu_dirty.None(), std::memory_order_release);
		UpdateProtection<true, false>(mask, deferred);
		for (size_t index = 0; index < ranges.size(); index++) {
			const auto [start, end] = GetPageRange(ranges[index].first, ranges[index].second);
			for (const auto [first, last]: RegionBits(mask, start, end)) {
				func(index, m_cpu_addr + first * TRACKER_PAGE_SIZE,
				     (last - first) * TRACKER_PAGE_SIZE);
			}
		}
	}

	template <typename Func>
	bool ConsumeCpuDirtyRangesKeepingBusy(std::span<const std::pair<uint64_t, uint64_t>> ranges, uint32_t sync,
	                                      Func&& func, DeferredRanges* deferred = nullptr) KYTY_REQUIRES(lock) {
		constexpr uint32_t MaxGap           = 3;
		constexpr uint8_t  MinWrites        = 3;
		constexpr uint32_t RevalidatePeriod = 12;
		RegionBits mask;
		for (const auto& [offset, size]: ranges) {
			const auto [start, end] = GetPageRange(offset, size);
			if (m_cpu_dirty.FirstRangeFrom(start).first < end) {
				mask |= RegionBits(m_cpu_dirty, start, end);
			}
		}
		if (mask.None()) {
			return false;
		}
		if (m_busy == nullptr) {
			m_busy = std::make_unique<BusyPages>();
			m_busy_published.store(m_busy.get(), std::memory_order_release);
		}
		auto&      busy = *m_busy;
		RegionBits clean;
		bool       kept = false;
		for (const auto [first, last]: mask) {
			for (auto page = first; page < last; page++) {
				if (busy.faulted.Get(page)) {
					busy.faulted.UnsetAtomic(page);
					const bool soon   = busy.seen[page] != 0 && sync - busy.seen[page] <= MaxGap;
					busy.writes[page] = soon ? static_cast<uint8_t>(std::min(busy.writes[page] + 1, 255)) : 1;
				}
				busy.seen[page] = sync;
				if (busy.writes[page] >= MinWrites && sync - busy.cleaned[page] < RevalidatePeriod &&
				    !m_gpu_written.Get(page)) {
					kept = true;
				} else {
					busy.writes[page]  = std::min(busy.writes[page], static_cast<uint8_t>(MinWrites - 1));
					busy.cleaned[page] = sync;
					clean.Set(page);
					m_cpu_dirty.Unset(page);
				}
			}
		}
		NoteOpenedUpload(&mask, clean);
		m_has_cpu_dirty.store(!m_cpu_dirty.None(), std::memory_order_release);
		UpdateProtection<true, false>(clean, deferred);
		if (kept) {
			m_kept_sync.store(sync, std::memory_order_relaxed);
		}
		for (size_t index = 0; index < ranges.size(); index++) {
			const auto [start, end] = GetPageRange(ranges[index].first, ranges[index].second);
			for (const auto [first, last]: RegionBits(mask, start, end)) {
				func(index, m_cpu_addr + first * TRACKER_PAGE_SIZE,
				     (last - first) * TRACKER_PAGE_SIZE);
			}
		}
		return kept;
	}

	void NoteGuestWriteFault(uint64_t offset) noexcept {
		if (auto* busy = m_busy_published.load(std::memory_order_acquire); busy != nullptr) {
			busy->faulted.SetAtomic(static_cast<size_t>(offset / TRACKER_PAGE_SIZE));
		}
	}

	[[nodiscard]] uint32_t KeptSync() const { return m_kept_sync.load(std::memory_order_relaxed); }

	template <typename Func>
	void ConsumeCpuWritten(CpuWriteConsumer consumer, uint64_t offset, uint64_t size, Func&& func) KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits mask = m_cpu_written[static_cast<size_t>(consumer)].TakeRange(start, end);
		mask |= RegionBits(m_cpu_dirty, start, end);
		if (m_any_in_place) {
			mask |= RegionBits(m_in_place, start, end);
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	template <typename Func>
	void MarkGpuDirtyUnlessCpuDirty(uint64_t offset, uint64_t size, Func&& func) KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits tracked(~m_in_place, start, end);
		m_gpu_written |= tracked;
		m_gpu_noted |= tracked;
		RegionBits mask(~m_cpu_dirty, start, end);
		mask &= tracked;
		if (mask.None()) {
			return;
		}
		auto either = m_cpu_dirty;
		either |= m_gpu_dirty;
		either |= m_in_place;
		RegionBits changed(~either, start, end);
		for (const auto [first, last]: mask) {
			m_gpu_dirty.SetRange(first, last);
		}
		UpdateProtection<true, true>(changed);
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	void MarkCpuDirtyUnlessGpuDirty(uint64_t offset, uint64_t size) KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		for (const auto [first, last]: RegionBits(~m_gpu_dirty, start, end)) {
			ChangeState<DirtySource::Cpu, true>(first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	uint64_t SetInPlace(uint64_t offset, uint64_t size) KYTY_REQUIRES(lock) {
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits newly(~m_in_place, start, end);
		if (newly.None()) {
			return 0;
		}
		RegionBits clean(~m_cpu_dirty, start, end);
		clean &= newly;
		RegionBits gpu(m_gpu_dirty, start, end);
		gpu &= newly;
		RegionBits keep(~newly);
		m_cpu_dirty &= keep;
		m_gpu_dirty &= keep;
		m_in_place |= newly;
		m_any_in_place = true;
		m_has_cpu_dirty.store(!m_cpu_dirty.None(), std::memory_order_release);
		UpdateProtection<false, false>(clean);
		UpdateProtection<false, true>(gpu);
		m_has_gpu_dirty.store(!m_gpu_dirty.None(), std::memory_order_release);
		return CountPages(newly);
	}

	uint64_t ClearInPlace(uint64_t offset, uint64_t size) KYTY_REQUIRES(lock) {
		if (!m_any_in_place) {
			return 0;
		}
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits leaving(m_in_place, start, end);
		if (leaving.None()) {
			return 0;
		}
		m_in_place &= RegionBits(~leaving);
		m_any_in_place = !m_in_place.None();
		m_cpu_dirty |= leaving;
		for (auto& written: m_cpu_written) {
			written.SetAtomic(leaving);
		}
		m_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
		s_cpu_dirty_epoch.fetch_add(1, std::memory_order_acq_rel);
		BdaDirtyRegions::Mark(m_cpu_addr / TRACKER_REGION_SIZE);
		m_has_cpu_dirty.store(true, std::memory_order_release);
		return CountPages(leaving);
	}

	bool CopyCpuDirty(RegionBits& pages) const KYTY_REQUIRES(lock) {
		pages.Clear();
		pages |= m_cpu_dirty;
		return !pages.None();
	}

	[[nodiscard]] bool HasCpuDirty() const {
		return m_has_cpu_dirty.load(std::memory_order_acquire);
	}

	[[nodiscard]] bool HasGpuDirty() const {
		return m_has_gpu_dirty.load(std::memory_order_acquire);
	}

	TrackingSpinLock lock {SpinLockClass::Tracker};

private:
	static uint64_t CountPages(const RegionBits& pages) {
		uint64_t count = 0;
		for (const auto [first, last]: pages) {
			count += last - first;
		}
		return count;
	}

	uint32_t OpenBlockPages(size_t page, size_t first, size_t last) KYTY_REQUIRES(lock) {
		RegionBits open(~m_gpu_dirty, first, last);
		RegionBits noted(m_gpu_noted, first, last);
		noted.Unset(page);
		open &= ~noted;
		if (m_any_in_place) {
			open &= RegionBits(~m_in_place, first, last);
		}
		RegionBits clean(~m_cpu_dirty, first, last);
		clean &= open;
		uint32_t opened = 0;
		for (const auto [begin, end]: clean) {
			opened += static_cast<uint32_t>(end - begin);
		}
		for (const auto [begin, end]: open) {
			ChangeState<DirtySource::Cpu, true>(begin * TRACKER_PAGE_SIZE, (end - begin) * TRACKER_PAGE_SIZE);
		}
		clean.Unset(page);
		if (!clean.None()) {
			m_block_opened |= clean;
			m_any_block_opened = true;
		}
		return opened;
	}

	void NoteOpenedUpload(const RegionBits* uploaded, const RegionBits& cleaned) KYTY_REQUIRES(lock) {
		if (!m_any_block_opened) {
			return;
		}
		if (uploaded != nullptr) {
			RegionBits hit(*uploaded);
			hit &= m_block_opened;
			uint64_t pages = 0;
			for (const auto [first, last]: hit) {
				pages += last - first;
			}
			if (pages != 0) {
				s_opened_upload_bytes.fetch_add(pages * TRACKER_PAGE_SIZE, std::memory_order_relaxed);
			}
		}
		m_block_opened &= ~cleaned;
		m_any_block_opened = !m_block_opened.None();
	}

	template <bool track, bool is_read>
	void UpdateProtection(RegionBits& changed, DeferredRanges* deferred = nullptr) KYTY_REQUIRES(lock) {
		if (changed.None()) {
			return;
		}
		if constexpr (is_read) {
			m_has_gpu_dirty.store(!m_gpu_dirty.None(), std::memory_order_release);
		}
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, changed, deferred);
	}

	template <DirtySource source>
	RegionBits& GetBits() KYTY_REQUIRES(lock) {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const KYTY_REQUIRES(lock) {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] static std::pair<size_t, size_t> GetPageRange(uint64_t offset, uint64_t size) {
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty KYTY_GUARDED_BY(lock);
	std::array<RegionBits, CPU_WRITE_CONSUMERS> m_cpu_written KYTY_GUARDED_BY(lock);
	RegionBits   m_gpu_dirty KYTY_GUARDED_BY(lock);
	std::atomic_bool      m_has_cpu_dirty {true};
	std::atomic_bool      m_has_gpu_dirty {false};
	std::atomic<uint64_t> m_cpu_dirty_epoch {1};
	std::atomic<uint64_t> m_gpu_clean_epoch {1};
	struct BusyPages {
		std::array<uint32_t, TRACKER_REGION_PAGES> seen {};
		std::array<uint32_t, TRACKER_REGION_PAGES> cleaned {};
		std::array<uint8_t, TRACKER_REGION_PAGES>  writes {};
		RegionBits                                 faulted;
	};
	RegionBits                 m_gpu_written KYTY_GUARDED_BY(lock);
	RegionBits                 m_gpu_noted KYTY_GUARDED_BY(lock);
	RegionBits                 m_block_opened KYTY_GUARDED_BY(lock);
	bool                       m_any_block_opened KYTY_GUARDED_BY(lock) = false;
	RegionBits                 m_in_place KYTY_GUARDED_BY(lock);
	bool                       m_any_in_place KYTY_GUARDED_BY(lock) = false;
	std::unique_ptr<BusyPages> m_busy KYTY_GUARDED_BY(lock);
	std::atomic<BusyPages*>    m_busy_published {nullptr};
	std::atomic<uint32_t>      m_kept_sync {0};
	std::atomic<uint64_t> m_host_write_epoch {0};
	std::atomic<uint32_t> m_deferred_protects {0};

	inline static std::atomic<uint64_t> s_cpu_dirty_epoch {0};
	inline static std::atomic<uint64_t> s_gpu_clean_epoch {0};
	inline static std::atomic<uint64_t> s_opened_upload_bytes {0};
};

}

#endif
