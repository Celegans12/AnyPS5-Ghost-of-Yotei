#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>
#include <thread>
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

void QuestionsCreateNoRegion(PageManager& pages) {
	MemoryTracker  tracker(pages);
	const uint64_t region = RegionBase / TRACKER_REGION_SIZE + 8;
	const uint64_t start  = region * TRACKER_REGION_SIZE;
	const uint64_t next   = start + TRACKER_REGION_SIZE;
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

struct Seen {
	size_t   range;
	uint64_t address;
	uint64_t size;
	bool     operator==(const Seen&) const = default;
};

struct NoUpload {
	void operator()() const noexcept {}
};

struct NestedRanges {
	MemoryTracker*                                 tracker;
	std::span<const std::pair<uint64_t, uint64_t>> inner;
	std::vector<Seen>                              outer_seen;
	std::vector<Seen>                              inner_seen;
	bool                                           inside = false;
	bool                                           nested = false;

	void operator()(size_t range, uint64_t address, uint64_t size) noexcept {
		(inside ? inner_seen : outer_seen).push_back({range, address, size});
		if (!inside && !nested) {
			nested = true;
			inside = true;
			tracker->ForEachUploadRanges(inner, *this, NoUpload {});
			inside = false;
		}
	}
};

void NestedUploadsKeepTheirRanges(PageManager& pages) {
	MemoryTracker  tracker(pages);
	const uint64_t outer = RegionBase + 24 * TRACKER_REGION_SIZE;
	const uint64_t inner = outer + TRACKER_REGION_SIZE;
	std::vector<std::pair<uint64_t, uint64_t>> warm;
	for (uint64_t page = 0; page < 8; page++) {
		warm.emplace_back(inner + (16 + 2 * page) * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	}
	const std::pair<uint64_t, uint64_t> outer_ranges[] = {{outer + 2 * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE},
	                                                      {outer + 3 * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE}};
	const std::pair<uint64_t, uint64_t> inner_ranges[] = {{inner + 7 * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE},
	                                                      {inner + 9 * TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE}};
	NestedRanges warm_up {.tracker = &tracker, .inner = {}, .nested = true};
	tracker.ForEachUploadRanges(warm, warm_up, NoUpload {});
	NestedRanges nesting {.tracker = &tracker, .inner = inner_ranges};
	tracker.ForEachUploadRanges(outer_ranges, nesting, NoUpload {});
	const std::vector<Seen> outer_expected {{0, outer_ranges[0].first, TRACKER_PAGE_SIZE},
	                                        {1, outer_ranges[1].first, TRACKER_PAGE_SIZE}};
	const std::vector<Seen> inner_expected {{0, inner_ranges[0].first, TRACKER_PAGE_SIZE},
	                                        {1, inner_ranges[1].first, TRACKER_PAGE_SIZE}};
	Check(nesting.nested, "nested uploads: the outer upload called no callback");
	Check(nesting.inner_seen == inner_expected, "nested uploads: the inner upload reported other ranges");
	Check(nesting.outer_seen == outer_expected,
	      "nested uploads: an upload made by a callback changed the ranges of its caller");
	Check(!tracker.IsRegionCpuModified(outer + 2 * TRACKER_PAGE_SIZE, 2 * TRACKER_PAGE_SIZE),
	      "nested uploads: the outer upload left a page written");
	tracker.MarkRegionAsCpuModified(outer + 2 * TRACKER_PAGE_SIZE, 2 * TRACKER_PAGE_SIZE);
	for (const auto& [address, size]: inner_ranges) {
		tracker.MarkRegionAsCpuModified(address, size);
	}
	for (const auto& [address, size]: warm) {
		tracker.MarkRegionAsCpuModified(address, size);
	}
}

constexpr uint64_t UploadPages = 4;

struct GuestPages {
	MemoryTracker*                                 tracker;
	uint64_t                                       start;
	std::array<std::atomic<uint32_t>, UploadPages> guest {};
	std::array<std::atomic<uint32_t>, UploadPages> device {};
	std::vector<std::pair<uint64_t, uint64_t>>    copies;

	void Take(uint64_t address, uint64_t size) noexcept { copies.emplace_back(address, size); }

	void Copy() noexcept {
		for (const auto& [address, size]: copies) {
			for (auto page = (address - start) / TRACKER_PAGE_SIZE; page < (address + size - start) / TRACKER_PAGE_SIZE; page++) {
				device[page].store(guest[page].load(std::memory_order_acquire), std::memory_order_release);
			}
		}
		copies.clear();
	}

	void Write(uint64_t page, uint32_t value) {
		const auto address = start + page * TRACKER_PAGE_SIZE;
		bool       gpu     = false;
		tracker->InvalidateRegion(address, 1, [&] { gpu = true; });
		if (gpu) {
			guest[page].store(device[page].load(std::memory_order_acquire), std::memory_order_release);
			tracker->UnmarkRegionAsGpuModified(address, TRACKER_PAGE_SIZE);
			tracker->MarkRegionAsCpuModified(address, TRACKER_PAGE_SIZE);
		}
		guest[page].store(value, std::memory_order_release);
	}

	void CheckNoWriteLost(const char* what) {
		for (uint64_t page = 0; page < UploadPages; page++) {
			const auto address = start + page * TRACKER_PAGE_SIZE;
			const bool cpu     = tracker->IsRegionCpuModified(address, TRACKER_PAGE_SIZE);
			const bool gpu     = tracker->IsRegionGpuModified(address, TRACKER_PAGE_SIZE);
			Check(!(cpu && gpu), what, "a page is written by both the CPU and the GPU");
			Check(cpu || device[page].load() == guest[page].load(), what,
			      "a page the tracking no longer sees written by the CPU lost a guest write");
			Check(gpu, what, "the upload did not hand every page of a buffer the GPU writes to the GPU");
		}
	}

	void Reset() {
		for (uint64_t page = 0; page < UploadPages; page++) {
			const auto address = start + page * TRACKER_PAGE_SIZE;
			if (tracker->IsRegionGpuModified(address, TRACKER_PAGE_SIZE)) {
				tracker->UnmarkRegionAsGpuModified(address, TRACKER_PAGE_SIZE);
			}
			tracker->MarkRegionAsCpuModified(address, TRACKER_PAGE_SIZE);
		}
	}
};

void WrittenUploadLosesNoGuestWrite(PageManager& pages) {
	using namespace std::chrono_literals;
	MemoryTracker    tracker(pages);
	GuestPages       memory {.tracker = &tracker, .start = RegionBase + 32 * TRACKER_REGION_SIZE};
	constexpr auto   Written = 2;
	std::atomic<int> stage {0};
	for (auto& value: memory.guest) value.store(1);
	std::thread game([&] {
		while (stage.load(std::memory_order_acquire) == 0) std::this_thread::yield();
		memory.Write(Written, 2);
		stage.store(2, std::memory_order_release);
	});
	uint32_t uploads = 0;
	bool     waited  = false;
	tracker.ForEachUploadRange(
	    memory.start, UploadPages * TRACKER_PAGE_SIZE, true,
	    [&](uint64_t address, uint64_t size) noexcept { memory.Take(address, size); },
	    [&]() noexcept {
		    memory.Copy();
		    if (uploads++ != 0) return;
		    stage.store(1, std::memory_order_release);
		    const auto deadline = std::chrono::steady_clock::now() + 2s;
		    while (stage.load(std::memory_order_acquire) != 2) {
			    if (std::chrono::steady_clock::now() > deadline) {
				    waited = true;
				    break;
			    }
			    std::this_thread::yield();
		    }
	    });
	game.join();
	Check(!waited, "written upload: a guest write fault waited for the whole copy");
	Check(uploads >= 2, "written upload: a page the guest wrote during the copy was not copied again");
	Check(memory.guest[Written].load() == 2, "written upload: the guest write did not land");
	memory.CheckNoWriteLost("written upload");
	memory.Reset();
}

void FaultOnTheUploadingThread(PageManager& pages) {
	MemoryTracker tracker(pages);
	GuestPages    memory {.tracker = &tracker, .start = RegionBase + 40 * TRACKER_REGION_SIZE};
	for (auto& value: memory.guest) value.store(1);
	uint32_t uploads = 0;
	tracker.ForEachUploadRange(
	    memory.start, UploadPages * TRACKER_PAGE_SIZE, true,
	    [&](uint64_t address, uint64_t size) noexcept { memory.Take(address, size); },
	    [&]() noexcept {
		    memory.Copy();
		    if (uploads++ == 0) memory.Write(1, 3);
	    });
	Check(memory.guest[1].load() == 3, "uploading thread: its own write did not land");
	memory.CheckNoWriteLost("uploading thread");
	memory.Reset();
}
}

int main() {
	PageManager pages;
	QuestionsCreateNoRegion(pages);
	CallbacksKeepTheirOwnProtections(pages);
	NestedUploadsKeepTheirRanges(pages);
	WrittenUploadLosesNoGuestWrite(pages);
	FaultOnTheUploadingThread(pages);
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_tracker_transitions: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_tracker_transitions: ok\n");
	return 0;
}
