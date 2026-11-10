#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class MemoryTracker final {
public:
	explicit MemoryTracker(PageManager& page_manager);
	~MemoryTracker();

	KYTY_CLASS_NO_COPY(MemoryTracker);

	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	// UnmarkRegionAsGpuModified(vaddr, size) followed by MarkRegionAsCpuModified(vaddr, size), each region's two
	// protection updates applied together under one hold of its lock (RegionManager::ReleaseGpuToCpu). True when every
	// region took the merged pass.
	bool               ReleaseGpuToCpu(uint64_t vaddr, uint64_t size);
	void               UntrackMemory(uint64_t vaddr, uint64_t size);
	// D4 step K0 (KYTY_D6_C0=2/3, diagnostic; null otherwise): told of every change of the GPU-modified state, kind
	// CensusMark (MarkRegionAsGpuModified, a written ForEachUploadRange), CensusCleanMark (MarkCleanPagesAsGpuModified),
	// CensusUnmark (UnmarkRegionAsGpuModified, a clearing ForEachDownloadRange) or CensusRelease (ReleaseGpuToCpu).
	// Set once by BufferCache before any tracker use. site: the return address of the tracker call (0 when inlined).
	enum : uint32_t { CensusMark = 0, CensusCleanMark = 1, CensusUnmark = 2, CensusRelease = 3 };
	static inline void (*s_census_hook)(uint64_t vaddr, uint64_t size, uint32_t kind, uint64_t site) = nullptr;
	// D6 step C0 (KYTY_D6_C0): the 4 KiB pages of [vaddr, vaddr + size) that are not CPU-dirty become GPU-modified;
	// every CPU-dirty page keeps its state and is passed to on_conflict(page address) instead. A page's check and mark
	// are one step under its region's lock, so a CPU write fault cannot make it CPU-dirty in between (which
	// MarkRegionAsGpuModified would stop on: "GPU dirty state conflicts with CPU dirty state"). on_conflict runs under
	// that lock: it only records.
	template <typename Conflict>
	void MarkCleanPagesAsGpuModified(uint64_t vaddr, uint64_t size, Conflict&& on_conflict) {
		CheckNotInUploadCallback();
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const uint64_t end = offset + bytes;
			uint64_t       run = offset; // first byte of the current run of pages that are not CPU-dirty
			for (uint64_t page = offset; page < end;) {
				const uint64_t next = std::min((page & ~(TRACKER_PAGE_SIZE - 1)) + TRACKER_PAGE_SIZE, end);
				if (manager->IsModified<DirtySource::Cpu>(page, next - page)) {
					if (page > run) {
						manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + run, page - run);
					}
					on_conflict(manager->GetCpuAddr() + (page & ~(TRACKER_PAGE_SIZE - 1)));
					run = next;
				}
				page = next;
			}
			if (end > run) manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + run, end - run);
		});
		if (s_census_hook != nullptr) s_census_hook(vaddr, size, CensusCleanMark, 0);
	}
	// Queue after changing ownership; a concurrent drain must not consume an event
	// before the dirty pages it describes become visible.
	// Upstream's BDA sync queue (d58aefde4). Our RenderContext::PrepareBda takes the CPU-dirty tracker regions instead
	// (the BDA region pass, the BDA sync epoch) and never drains this queue, so the write-fault path
	// (InvalidateRegion) does not feed it.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	// Removes protection from a range and flushes GPU-owned data when required.
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		CheckNotInUploadCallback();

		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				// Perform both the GPU modification check and CPU state change with the lock in
				// case the GPU thread is racing to mark the page modified. If a flush is needed,
				// on_flush performs the CPU state change.
				std::scoped_lock lock(manager->lock);
				KYTY_PROFILER_BLOCK("RegionLock::HeldInvalidate");
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
				return false;
			}();
			if (should_flush) {
				on_flush();
			}
		});
	}
#if KYTY_BUILD == KYTY_BUILD_DEBUG
	void ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                           const char* operation) const noexcept;
	void ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
	                               const char* operation);
#else
	void ValidateGpuDirtyPages(const RangeSet&, uint64_t, uint64_t, const char*) const noexcept {}
	void ValidateGpuDirtyOwnership(const RangeSet&, uint64_t, uint64_t, const char*) {}
#endif

	template <bool clear, typename Func>
	void ForEachDownloadRange(uint64_t vaddr, uint64_t size, Func&& func) {
		static_assert(std::is_nothrow_invocable_v<Func&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			std::scoped_lock lock(manager->lock);
			const auto       address = manager->GetCpuAddr() + offset;
			manager->template ForEachModifiedRange<DirtySource::Gpu, false>(address, bytes, func);
			if constexpr (clear) {
				manager->template ChangeState<DirtySource::Gpu, false>(address, bytes);
			}
		});
		if constexpr (clear) {
			if (s_census_hook != nullptr) s_census_hook(vaddr, size, CensusUnmark, 0);
		}
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		CheckNotInUploadCallback();
		if (!is_written && AllRegionsCpuClean(vaddr, size)) {
			// The per-region early-out below would skip every region: nothing to upload.
			return;
		}
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool summary_clean = !manager->MaybeModified<DirtySource::Cpu>();
			if (!is_written && summary_clean) {
				return;
			}
			manager->lock.lock();
			manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset,
			                                                      bytes, range_func);
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
				               manager->template ChangeState<DirtySource::Gpu, true>(
				                   manager->GetCpuAddr() + offset, bytes);
				               manager->lock.unlock();
			               });
			if (s_census_hook != nullptr) s_census_hook(vaddr, size, CensusMark, 0);
		}
		s_upload_owner = previous_upload_owner;
	}

	// BufferCache's upload batch (GPU thread): the first phase of a read-only upload. Like
	// ForEachUploadRange(vaddr, size, false, ...), it clears the CPU-dirty bits of the range and calls
	// range_func(address, bytes) for every run of pages to upload, but it leaves the cleared pages writable; every region
	// that needs its protection updated is appended to `pending` once (until the commit). The bytes of a claimed range
	// must not be read before CommitDeferredProtections(pending) returned: the order stays "clear the dirty bits,
	// write-protect, copy", so a guest write before the protection lands in guest memory without a fault and is copied,
	// and a write after it faults and makes the page CPU-dirty again. Returns the number of regions in which this claim
	// cleared pages (each one protection update of the unbatched path).
	template <typename RangeFunc>
	uint32_t ClaimUploadRanges(uint64_t vaddr, uint64_t size, RangeFunc&& range_func, std::vector<RegionManager*>& pending) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		CheckNotInUploadCallback();
		if (AllRegionsCpuClean(vaddr, size)) {
			return 0;
		}
		uint32_t   claimed               = 0;
		const auto* previous_upload_owner = std::exchange(s_upload_owner, this);
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			if (!manager->MaybeModified<DirtySource::Cpu>()) {
				return;
			}
			std::scoped_lock lock(manager->lock);
			if (manager->ForEachModifiedRange<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes, range_func,
			                                                          true)) {
				claimed++;
				if (!manager->ProtectionDeferred()) {
					manager->SetProtectionDeferred();
					pending.push_back(manager);
				}
			}
		});
		s_upload_owner = previous_upload_owner;
		return claimed;
	}
	// The upload batch, the second phase: write-protects every page the claims since the last commit left
	// writable, one protection update per region, regions in ascending address order (each under its own lock only, as
	// everywhere else), and empties `pending`. Returns the number of regions whose protection changed.
	uint32_t CommitDeferredProtections(std::vector<RegionManager*>& pending);


private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	inline static thread_local const MemoryTracker* s_upload_owner = nullptr;

	void CheckNotInUploadCallback() const noexcept {
		if (s_upload_owner == this) {
			EXIT("memory tracker re-entered from upload callback\n");
		}
	}

	// True when every region of the range exists and its CPU-dirty summary is clear (a missing
	// region is not proven clean: ForEachUploadRange creates it and may upload).
	bool AllRegionsCpuClean(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		uint64_t remaining = size;
		uint64_t index     = vaddr / TRACKER_REGION_SIZE;
		uint64_t offset    = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr || manager->MaybeModified<DirtySource::Cpu>()) {
				return false;
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return true;
	}

	template <bool create, typename Func>
	bool Iterate(uint64_t vaddr, uint64_t size, Func&& func) {
		ValidateRange(vaddr, size);
		using Result = std::invoke_result_t<Func, RegionManager*, uint64_t, uint64_t>;
		constexpr bool returns_bool = std::is_same_v<Result, bool>;
		uint64_t       remaining    = size;
		uint64_t       index        = vaddr / TRACKER_REGION_SIZE;
		uint64_t       offset       = vaddr % TRACKER_REGION_SIZE;
		while (remaining != 0) {
			const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr && create) {
				manager = GetOrCreateRegion(index);
			}
			if (manager != nullptr) {
				if constexpr (returns_bool) {
					if (func(manager, offset, bytes)) {
						return true;
					}
				} else {
					func(manager, offset, bytes);
				}
			}
			remaining -= bytes;
			offset = 0;
			index++;
		}
		return false;
	}

	static void    ValidateRange(uint64_t vaddr, uint64_t size);
	RegionManager* GetOrCreateRegion(uint64_t index);

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	std::mutex                                     m_bda_sync_mutex;
	RangeSet                                       m_bda_sync;
	PageManager&                                   m_page_manager;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
