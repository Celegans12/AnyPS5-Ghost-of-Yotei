#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORYTRACKER_H_

#include "common/assert.h"
#include "graphics/host_gpu/commandBlocks.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <span>
#include <cstdlib>
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
	[[nodiscard]] bool MayBeGpuModified(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || size > TRACKER_ADDRESS_SIZE || vaddr > TRACKER_ADDRESS_SIZE - size) {
			return true;
		}
		const auto last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		for (auto index = vaddr / TRACKER_REGION_SIZE; index <= last && index < REGION_COUNT;
		     index++) {
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager != nullptr && manager->HasGpuDirty()) {
				return true;
			}
		}
		return false;
	}
	void               MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size);
	void               UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size);
	template <typename Flush>
	void InvalidateRegion(uint64_t vaddr, uint64_t size, Flush&& on_flush) noexcept {
		static_assert(std::is_invocable_v<Flush&>);
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			const bool should_flush = [&] {
				Common::ScopedLock lock(manager->lock);
				if (manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
					return true;
				}
				manager->ChangeState<DirtySource::Cpu, true>(offset, bytes);
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
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			Common::ScopedLock lock(manager->lock);
			manager->ForEachModifiedRange<DirtySource::Gpu, false>(offset, bytes, func);
			if constexpr (clear) {
				manager->ChangeState<DirtySource::Gpu, false>(offset, bytes);
			}
		});
	}

	[[nodiscard]] uint32_t RegionKeptSync(uint64_t index) const {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		return manager != nullptr ? manager->KeptSync() : 0;
	}

	void NoteGuestWriteFault(uint64_t vaddr) {
		const auto index = vaddr / TRACKER_REGION_SIZE;
		if (index >= REGION_COUNT) {
			return;
		}
		if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
			Common::ScopedLock lock(manager->lock);
			manager->NoteGuestWriteFault(vaddr % TRACKER_REGION_SIZE);
		}
	}

	[[nodiscard]] RegionManager::WriteBlockProbe OpenWriteBlock(uint64_t vaddr, uint64_t block_size, bool open_block) {
		const auto index = vaddr / TRACKER_REGION_SIZE;
		if (index >= REGION_COUNT) {
			return {};
		}
		auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return {};
		}
		Common::ScopedLock lock(manager->lock);
		return manager->OpenWriteBlock(vaddr % TRACKER_REGION_SIZE, block_size, open_block);
	}

	[[nodiscard]] RegionManager::WriteFaultProbe FaultWrite(uint64_t vaddr, uint64_t block_size, bool open_block,
	                                                        uint64_t mark_size) {
		const auto index = vaddr / TRACKER_REGION_SIZE;
		if (index >= REGION_COUNT) {
			return {};
		}
		auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return {.done = true};
		}
		Common::ScopedLock lock(manager->lock);
		return manager->FaultWrite(vaddr % TRACKER_REGION_SIZE, block_size, open_block, mark_size);
	}

	[[nodiscard]] bool WriteFaultFindsOpen(uint64_t vaddr, uint64_t block_size, bool open_block, uint64_t mark_size,
	                                       bool* gpu) {
		*gpu             = false;
		const auto index = vaddr / TRACKER_REGION_SIZE;
		if (index >= REGION_COUNT) {
			return false;
		}
		auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return true;
		}
		if (!manager->WriteFaultFindsOpen(vaddr % TRACKER_REGION_SIZE, block_size, open_block, mark_size, gpu)) {
			return false;
		}
		manager->NoteCpuWriteOnOpenPage();
		return true;
	}

	uint64_t SetInPlace(uint64_t vaddr, uint64_t size) {
		uint64_t pages = 0;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			Common::ScopedLock lock(manager->lock);
			pages += manager->SetInPlace(offset, bytes);
		});
		return pages;
	}

	uint64_t ClearInPlace(uint64_t vaddr, uint64_t size) {
		uint64_t pages = 0;
		Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			Common::ScopedLock lock(manager->lock);
			pages += manager->ClearInPlace(offset, bytes);
		});
		return pages;
	}

	[[nodiscard]] bool EverGpuWritten(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
			return false;
		}
		for (auto index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < vaddr + size; ++index) {
			const auto* manager = m_regions[index].load(std::memory_order_acquire);
			if (manager == nullptr) {
				continue;
			}
			const auto begin = std::max(vaddr, index * TRACKER_REGION_SIZE);
			const auto end   = std::min(vaddr + size, (index + 1) * TRACKER_REGION_SIZE);
			if (manager->EverGpuWritten(begin - index * TRACKER_REGION_SIZE, end - begin)) {
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] bool TouchesCommandMemory(uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0) {
			return false;
		}
		for (auto block = vaddr & ~(CommandBlocks::BlockSize - 1); block < vaddr + size; block += CommandBlocks::BlockSize) {
			if (m_command_blocks.Contains(block) || m_segment_blocks.Contains(block)) {
				return true;
			}
		}
		return false;
	}

	void MarkCommandBlocks(uint64_t vaddr, uint64_t size) { m_command_blocks.Mark(vaddr, size); }
	void MarkSegmentBlocks(uint64_t vaddr, uint64_t size) { m_segment_blocks.Mark(vaddr, size); }
	void ForgetCommandBlocks(uint64_t vaddr, uint64_t size) {
		m_command_blocks.Forget(vaddr, size);
		m_segment_blocks.Forget(vaddr, size);
	}
	[[nodiscard]] bool IsCommandBlock(uint64_t vaddr) const noexcept { return m_command_blocks.Contains(vaddr); }
	[[nodiscard]] bool IsSegmentBlock(uint64_t vaddr) const noexcept { return m_segment_blocks.Contains(vaddr); }

	[[nodiscard]] bool TakeKeptPages() {
		return m_kept_pages.exchange(false, std::memory_order_relaxed);
	}

	[[nodiscard]] uint64_t RegionCpuDirtyEpoch(uint64_t index) const {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		return manager != nullptr ? manager->RegionCpuDirtyEpoch() : 0;
	}

	[[nodiscard]] uint64_t CpuModificationEpoch(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || !GuestRange {vaddr, size}.Valid() ||
		    vaddr / TRACKER_REGION_SIZE != (vaddr + size - 1) / TRACKER_REGION_SIZE)
			return 0;
		const auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		return manager == nullptr ? 0 : manager->CpuModificationEpoch();
	}
	[[nodiscard]] bool DeferredProtectionPending(uint64_t vaddr) const {
		if (!GuestRange {vaddr, 1}.Valid()) return false;
		const auto* manager = m_regions[vaddr / TRACKER_REGION_SIZE].load(std::memory_order_acquire);
		return manager != nullptr && manager->DeferredProtectionPending();
	}
	void NoteHostWrite(uint64_t vaddr, uint64_t size) {
		if (size == 0 || !GuestRange {vaddr, size}.Valid()) return;
		for (auto index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < vaddr + size; ++index)
			if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) manager->NoteHostWrite();
	}

	[[nodiscard]] uint64_t RegionGpuCleanEpoch(uint64_t index) const {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		return manager != nullptr ? manager->RegionGpuCleanEpoch() : 0;
	}

	template <typename Func>
	void ForEachCpuDirtySpan(uint64_t vaddr, uint64_t size, uint64_t max_gap, Func&& func) {
		ValidateRange(vaddr, size);
		const auto                                  end = vaddr + size;
		std::vector<std::pair<uint64_t, uint64_t>> spans;
		const auto add = [&](uint64_t begin, uint64_t finish) {
			begin  = std::max(begin, vaddr);
			finish = std::min(finish, end);
			if (begin >= finish) {
				return;
			}
			if (!spans.empty() && begin <= spans.back().second + max_gap) {
				spans.back().second = finish;
			} else {
				spans.emplace_back(begin, finish);
			}
		};
		for (uint64_t index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < end;
		     index++) {
			auto*      manager = m_regions[index].load(std::memory_order_acquire);
			const auto start   = std::max(index * TRACKER_REGION_SIZE, vaddr);
			const auto finish  = std::min((index + 1) * TRACKER_REGION_SIZE, end);
			if (manager == nullptr) {
				add(start, finish);
				continue;
			}
			if (!manager->HasCpuDirty()) {
				continue;
			}
			Common::ScopedLock lock(manager->lock);
			manager->ForEachModifiedRange<DirtySource::Cpu, false>(
			    start - index * TRACKER_REGION_SIZE, finish - start,
			    [&](uint64_t address, uint64_t bytes) { add(address, address + bytes); });
		}
		for (const auto& [begin, finish]: spans) {
			func(begin, finish - begin);
		}
	}

	bool CopyRegionCpuDirty(uint64_t index, RegionBits& pages) {
		if (index >= REGION_COUNT) {
			EXIT("invalid memory tracker region\n");
		}
		auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			pages.Fill();
			return true;
		}
		if (!manager->HasCpuDirty()) {
			return false;
		}
		Common::ScopedLock lock(manager->lock);
		return manager->CopyCpuDirty(pages);
	}

	template <typename Func>
	void ConsumeCpuWrites(CpuWriteConsumer consumer, uint64_t vaddr, uint64_t size, Func&& func) {
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			Common::ScopedLock lock(manager->lock);
			manager->ConsumeCpuWritten(consumer, offset, bytes,
			                           [&](uint64_t address, uint64_t length) {
				                           if (!runs.empty() &&
				                               runs.back().first + runs.back().second == address) {
					                           runs.back().second += length;
				                           } else {
					                           runs.emplace_back(address, length);
				                           }
			                           });
		});
		const auto end = vaddr + size;
		for (const auto& [address, length]: runs) {
			const auto begin  = std::max(address, vaddr);
			const auto finish = std::min(address + length, end);
			if (begin < finish) {
				func(begin, finish - begin);
			}
		}
	}

	template <typename Func>
	void MarkRegionAsGpuModifiedUnlessCpuDirty(uint64_t vaddr, uint64_t size, Func&& func) {
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			Common::ScopedLock lock(manager->lock);
			manager->MarkGpuDirtyUnlessCpuDirty(
			    offset, bytes,
			    [&](uint64_t address, uint64_t length) { runs.emplace_back(address, length); });
		});
		for (const auto& [address, length]: runs) {
			func(address, length);
		}
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRange(uint64_t vaddr, uint64_t size, bool is_written, RangeFunc&& range_func,
	                        UploadFunc&& upload_func, uint32_t device_sync = 0) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		const bool defer = !is_written && DeferProtections();
		Iterate<true>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) KYTY_NO_THREAD_SAFETY_ANALYSIS {
			if (!is_written && !manager->IsModifiedUnlocked<DirtySource::Cpu>(offset, bytes)) {
				return;
			}
			if (!is_written && device_sync != 0) {
				const std::pair<uint64_t, uint64_t> range {offset, bytes};
				Common::ScopedLock                    lock(manager->lock);
				DeferScope deferred(defer, manager);
				if (manager->ConsumeCpuDirtyRangesKeepingBusy(
				        std::span(&range, 1), device_sync,
				        [&](size_t, uint64_t address, uint64_t run) noexcept { range_func(address, run); })) {
					m_kept_pages.store(true, std::memory_order_relaxed);
				}
				return;
			}
			manager->lock.lock();
			{
				DeferScope deferred(defer, manager);
				manager->ForEachModifiedRange<DirtySource::Cpu, true>(offset, bytes, range_func);
			}
			if (!is_written) {
				manager->lock.unlock();
			}
		});
		upload_func();
		ApplyDeferredProtections();
		if (is_written) {
			Iterate<false>(vaddr, size,
			               [](RegionManager* manager, uint64_t offset, uint64_t bytes) KYTY_NO_THREAD_SAFETY_ANALYSIS {
				               manager->ChangeState<DirtySource::Gpu, true>(offset, bytes);
				               manager->lock.unlock();
			               });
		}
	}

	template <typename RangeFunc, typename UploadFunc>
	void ForEachUploadRanges(std::span<const std::pair<uint64_t, uint64_t>> ranges,
	                         RangeFunc&& range_func, UploadFunc&& upload_func,
	                         uint32_t device_sync = 0) {
		static_assert(std::is_nothrow_invocable_v<RangeFunc&, size_t, uint64_t, uint64_t>);
		static_assert(std::is_nothrow_invocable_v<UploadFunc&>);
		const bool defer = DeferProtections();
		struct Piece {
			uint64_t region;
			uint64_t offset;
			uint64_t size;
			size_t   range;
		};
		static thread_local std::vector<Piece>                         pieces;
		static thread_local std::vector<std::pair<uint64_t, uint64_t>> offsets;
		pieces.clear();
		for (size_t index = 0; index < ranges.size(); index++) {
			const auto [address, size] = ranges[index];
			ValidateRange(address, size);
			for (auto at = address; at < address + size;) {
				const auto region = at / TRACKER_REGION_SIZE;
				const auto end    = std::min(address + size, (region + 1) * TRACKER_REGION_SIZE);
				pieces.push_back({region, at - region * TRACKER_REGION_SIZE, end - at, index});
				at = end;
			}
		}
		for (size_t first = 0; first < pieces.size();) {
			auto last = first + 1;
			while (last < pieces.size() && pieces[last].region == pieces[first].region) {
				last++;
			}
			auto* manager = GetOrCreateRegion(pieces[first].region);
			if (manager->HasCpuDirty()) {
				offsets.clear();
				for (auto piece = first; piece < last; piece++) {
					offsets.emplace_back(pieces[piece].offset, pieces[piece].size);
				}
				Common::ScopedLock lock(manager->lock);
				DeferScope       deferred(defer, manager);
				const auto piece_func = [&](size_t piece, uint64_t address, uint64_t bytes) noexcept {
					range_func(pieces[first + piece].range, address, bytes);
				};
				const std::span<const std::pair<uint64_t, uint64_t>> region_ranges(offsets);
				if (device_sync == 0) {
					manager->ConsumeCpuDirtyRanges(region_ranges, piece_func);
				} else if (manager->ConsumeCpuDirtyRangesKeepingBusy(region_ranges, device_sync,
				                                                    piece_func)) {
					m_kept_pages.store(true, std::memory_order_relaxed);
				}
			}
			first = last;
		}
		upload_func();
		ApplyDeferredProtections();
	}

	struct DeferredProtect {
		RegionManager* manager = nullptr;
		uint64_t       address = 0, size = 0;
	};
	struct UploadScope {
		bool previous;
		UploadScope() noexcept: previous(std::exchange(s_upload_takes_protections, true)) {}
		~UploadScope() { s_upload_takes_protections = previous; }
		UploadScope(const UploadScope&)            = delete;
		UploadScope& operator=(const UploadScope&) = delete;
	};
	[[nodiscard]] static std::vector<DeferredProtect>& DeferredProtects() noexcept { return s_deferred; }
	[[nodiscard]] static bool AsyncReprotect() noexcept {
		static const bool on = [] {
			const char* value = std::getenv("KYTY_ASYNC_REPROTECT");
			return value == nullptr || value[0] != '0';
		}();
		return on;
	}

private:
	static constexpr size_t REGION_COUNT = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
	static_assert(CommandBlocks::AddressSize == TRACKER_ADDRESS_SIZE);
	static_assert(TRACKER_REGION_SIZE % CommandBlocks::BlockSize == 0);

	inline static thread_local bool                         s_upload_takes_protections = false;
#if defined(ANYPS5_KYTY) && defined(_WIN32) && (!defined(__clang__) || defined(ANYPS5_KYTY_DLL))
	static thread_local std::vector<DeferredProtect> s_deferred;
#else
	inline static thread_local std::vector<DeferredProtect> s_deferred;
#endif
	[[nodiscard]] static bool DeferProtections() noexcept { return s_upload_takes_protections && AsyncReprotect(); }
	struct DeferScope {
		RegionManager*                          manager = nullptr;
		std::vector<PageManager::DeferredRange> ranges;
		DeferScope(bool defer, RegionManager* region) noexcept: manager(defer ? region : nullptr) {
			if (manager != nullptr) PageManager::SetDeferredWriteProtectSink(&ranges);
		}
		~DeferScope() {
			if (manager == nullptr) return;
			PageManager::SetDeferredWriteProtectSink(nullptr);
			for (const auto& range: ranges) {
				manager->BeginDeferredProtection();
				s_deferred.push_back({manager, range.address, range.size});
			}
		}
		DeferScope(const DeferScope&)            = delete;
		DeferScope& operator=(const DeferScope&) = delete;
	};
	static void ApplyDeferredProtections() {
		for (const auto& deferred: s_deferred) deferred.manager->ApplyDeferredProtection(deferred.address, deferred.size);
		s_deferred.clear();
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

	std::atomic<bool> m_kept_pages {false};
	CommandBlocks     m_command_blocks;
	CommandBlocks     m_segment_blocks;

	std::unique_ptr<std::atomic<RegionManager*>[]> m_regions;
	std::vector<std::unique_ptr<RegionManager>>    m_region_storage;
	std::mutex                                     m_region_mutex;
	PageManager&                                   m_page_manager;
};

}

#endif
