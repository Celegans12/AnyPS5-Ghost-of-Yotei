#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/runtimeSwitches.h"

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

#if defined(ANYPS5_KYTY_DLL)
thread_local std::vector<MemoryTracker::DeferredProtect> MemoryTracker::s_deferred;
#endif

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
}

MemoryTracker::~MemoryTracker() = default;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	if (!Common::RuntimeSwitches::On(Common::RuntimeSwitches::Switch::TrackerLazyRegions)) {
		return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			return manager->IsModifiedUnlocked<DirtySource::Cpu>(offset, bytes);
		});
	}
	ValidateRange(vaddr, size);
	const auto last = (vaddr + size - 1) / TRACKER_REGION_SIZE;
	for (auto index = vaddr / TRACKER_REGION_SIZE; index <= last; index++) {
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return true;
		}
		const auto begin = std::max(vaddr, index * TRACKER_REGION_SIZE);
		const auto end   = std::min(vaddr + size, (index + 1) * TRACKER_REGION_SIZE);
		if (manager->IsModifiedUnlocked<DirtySource::Cpu>(begin - index * TRACKER_REGION_SIZE, end - begin)) {
			return true;
		}
	}
	return false;
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	if (!MayBeGpuModified(vaddr, size)) {
		ValidateRange(vaddr, size);
		return false;
	}
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return manager->IsModifiedUnlocked<DirtySource::Gpu>(offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		Common::ScopedLock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		Common::ScopedLock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(offset, bytes);
	});
}

}
