#include "common/runtimeSwitches.h"
#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <utility>
#include <vector>

namespace {

std::atomic<uint64_t> g_allocations {0};
thread_local bool     g_counting = false;

void* Allocate(std::size_t size, std::size_t alignment) {
	if (g_counting) g_allocations.fetch_add(1, std::memory_order_relaxed);
	const auto bytes = size == 0 ? alignment : (size + alignment - 1) / alignment * alignment;
	if (void* memory = std::aligned_alloc(alignment, bytes); memory != nullptr) return memory;
	throw std::bad_alloc();
}

}

void* operator new(std::size_t size) {
	return Allocate(size, alignof(std::max_align_t));
}
void* operator new[](std::size_t size) {
	return Allocate(size, alignof(std::max_align_t));
}
void* operator new(std::size_t size, std::align_val_t alignment) {
	return Allocate(size, std::max(static_cast<std::size_t>(alignment), alignof(std::max_align_t)));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
	return Allocate(size, std::max(static_cast<std::size_t>(alignment), alignof(std::max_align_t)));
}
void operator delete(void* memory) noexcept {
	std::free(memory);
}
void operator delete[](void* memory) noexcept {
	std::free(memory);
}
void operator delete(void* memory, std::size_t) noexcept {
	std::free(memory);
}
void operator delete[](void* memory, std::size_t) noexcept {
	std::free(memory);
}
void operator delete(void* memory, std::align_val_t) noexcept {
	std::free(memory);
}
void operator delete[](void* memory, std::align_val_t) noexcept {
	std::free(memory);
}
void operator delete(void* memory, std::size_t, std::align_val_t) noexcept {
	std::free(memory);
}
void operator delete[](void* memory, std::size_t, std::align_val_t) noexcept {
	std::free(memory);
}

namespace {

namespace Bda      = Libs::Graphics::BdaDirtyRegions;
namespace Switches = Common::RuntimeSwitches;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RegionManager;
using Libs::Graphics::TRACKER_PAGE_SIZE;
using Libs::Graphics::TRACKER_REGION_SIZE;

constexpr uint64_t RegionBase = uint64_t {1} << 32u;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_tracker_transitions: FAILED %s\n", what);
	g_failures++;
}

void Check(bool condition, const char* where, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_tracker_transitions: FAILED %s: %s\n", where, what);
	g_failures++;
}

struct Counting {
	uint64_t before;
	Counting(): before(g_allocations.load(std::memory_order_relaxed)) { g_counting = true; }
	~Counting() { g_counting = false; }
	Counting(const Counting&)            = delete;
	Counting& operator=(const Counting&) = delete;
	[[nodiscard]] uint64_t Allocations() const { return g_allocations.load(std::memory_order_relaxed) - before; }
};

bool Pending(uint64_t index) {
	return (Bda::g_pending[index / 64].load() & (uint64_t {1} << (index % 64))) != 0;
}

void Unmark(uint64_t index) {
	Bda::g_pending[index / 64].fetch_and(~(uint64_t {1} << (index % 64)));
}

void QuestionsCreateNoRegion(PageManager& pages) {
	MemoryTracker  tracker(pages);
	const uint64_t region = RegionBase / TRACKER_REGION_SIZE + 8;
	const uint64_t start  = region * TRACKER_REGION_SIZE;
	const uint64_t next   = start + TRACKER_REGION_SIZE;
	Unmark(region);
	Unmark(region + 1);
	const auto global = RegionManager::CpuDirtyEpoch();
	bool       modified = false;
	{
		const Counting counting;
		modified = tracker.IsRegionCpuModified(next - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
		Check(counting.Allocations() == 0, "questions: a question on a region never tracked allocated");
	}
	Check(modified, "questions: memory never tracked does not read as written by the CPU, an upload would skip it");
	Check(tracker.RegionCpuDirtyEpoch(region) == 0, "questions: a question created the region");
	Check(RegionManager::CpuDirtyEpoch() == global, "questions: a question advanced the global CPU write epoch");
	Check(!Pending(region), "questions: a question marked the region for device address reads");
	Check(tracker.CpuModificationEpoch(start, TRACKER_PAGE_SIZE) == 0, "questions: a question gave the region a CPU modification epoch");

	tracker.ForEachUploadRange(next, TRACKER_PAGE_SIZE, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
	Check(tracker.RegionCpuDirtyEpoch(region + 1) != 0, "questions: an upload did not create its region");
	Check(RegionManager::CpuDirtyEpoch() == global + 1, "questions: the region an upload created did not count once");
	Check(!tracker.IsRegionCpuModified(next, TRACKER_PAGE_SIZE), "questions: the upload left its page written");
	Check(tracker.IsRegionCpuModified(next + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE),
	      "questions: a page the upload left alone is not written");
	{
		const Counting counting;
		modified = tracker.IsRegionCpuModified(next - TRACKER_PAGE_SIZE, 2 * TRACKER_PAGE_SIZE);
		Check(counting.Allocations() == 0, "questions: a question across a missing region allocated");
	}
	Check(modified, "questions: a range that starts in a region never tracked is not written");
	Check(tracker.RegionCpuDirtyEpoch(region) == 0, "questions: a question across two regions created the first");
	Check(RegionManager::CpuDirtyEpoch() == global + 1, "questions: a question across two regions advanced the epoch");

	Switches::Assign(Switches::Switch::TrackerLazyRegions, false);
	Check(tracker.IsRegionCpuModified(next - TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE),
	      "questions: the switch turned off does not answer as before");
	Check(tracker.RegionCpuDirtyEpoch(region) != 0, "questions: the switch turned off does not create the region as before");
	Switches::Assign(Switches::Switch::TrackerLazyRegions, true);
	tracker.MarkRegionAsCpuModified(next, TRACKER_PAGE_SIZE);
}


void CallbacksKeepTheirOwnProtections(PageManager& pages) {
	MemoryTracker  tracker(pages);
	const uint64_t start   = RegionBase + 16 * TRACKER_REGION_SIZE;
	const uint64_t watched = start + TRACKER_REGION_SIZE + 5 * TRACKER_PAGE_SIZE;
	std::vector<MemoryTracker::DeferredProtect> taken;
	bool                                        watching = false;
	const auto                                  watch    = [&]() noexcept {
		if (!watching) {
			pages.UpdatePageWatchers<true>(watched, TRACKER_PAGE_SIZE);
			watching = true;
		}
	};
	const auto take  = [&]() noexcept { taken = MemoryTracker::DeferredProtects(); };
	const auto check = [&](const char* what) {
		bool own     = false;
		bool foreign = false;
		for (const auto& item: taken) {
			own     = own || (item.address == start && item.size == 2 * TRACKER_PAGE_SIZE);
			foreign = foreign || (item.address < watched + TRACKER_PAGE_SIZE && watched < item.address + item.size);
		}
		Check(watching, what, "the upload called no callback");
		Check(own, what, "the upload did not defer the protection of its own pages");
		Check(!foreign, what, "a protection a callback asked for was deferred with the upload's");
		Check(!tracker.DeferredProtectionPending(start), what, "a deferred protection was left pending");
		pages.UpdatePageWatchers<false>(watched, TRACKER_PAGE_SIZE);
		watching = false;
		taken.clear();
		tracker.MarkRegionAsCpuModified(start, 2 * TRACKER_PAGE_SIZE);
	};
	const std::pair<uint64_t, uint64_t> ranges[] = {{start, TRACKER_PAGE_SIZE}, {start + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE}};
	for (const uint32_t device_sync: {0u, 1u}) {
		{
			const MemoryTracker::UploadScope scope;
			tracker.ForEachUploadRange(start, 2 * TRACKER_PAGE_SIZE, false, [&](uint64_t, uint64_t) noexcept { watch(); },
			                           take, device_sync);
		}
		check(device_sync == 0 ? "callbacks (one range)" : "callbacks (one range kept busy)");
		{
			const MemoryTracker::UploadScope scope;
			tracker.ForEachUploadRanges(ranges, [&](size_t, uint64_t, uint64_t) noexcept { watch(); }, take, device_sync);
		}
		check(device_sync == 0 ? "callbacks (several ranges)" : "callbacks (several ranges kept busy)");
	}
}
}

int main() {
	Check(Switches::On(Switches::Switch::TrackerLazyRegions), "KYTY_TRACKER_LAZY_REGIONS is on without the variable");
	PageManager pages;
	QuestionsCreateNoRegion(pages);
	CallbacksKeepTheirOwnProtections(pages);
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_tracker_transitions: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_tracker_transitions: ok\n");
	return 0;
}
