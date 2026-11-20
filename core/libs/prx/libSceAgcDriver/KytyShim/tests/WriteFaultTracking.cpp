#include "graphics/host_gpu/bdaDirtyRegions.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::CpuWriteConsumer;
using Libs::Graphics::DirtySource;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RegionBits;
using Libs::Graphics::RegionManager;
using Libs::Graphics::SpinLockClass;
using Libs::Graphics::TRACKER_PAGE_SIZE;
using Libs::Graphics::TRACKER_REGION_SIZE;
using Libs::Graphics::TrackingSpinLock;
using namespace std::chrono_literals;

constexpr uint64_t RegionBase = uint64_t {1} << 32u;
constexpr uint64_t Block      = 64 * 1024;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_write_fault_tracking: FAILED %s\n", what);
	g_failures++;
}

void Check(bool condition, const char* where, const char* what) {
	if (condition) return;
	std::fprintf(stderr, "kyty_shim_write_fault_tracking: FAILED %s: %s\n", where, what);
	g_failures++;
}

void UncontendedLockCountsNothing() {
	(void)TrackingSpinLock::TakeContention(SpinLockClass::Tracker);
	TrackingSpinLock lock {SpinLockClass::Tracker};
	for (int i = 0; i < 1000; i++) {
		std::scoped_lock guard(lock);
	}
	Check(TrackingSpinLock::TakeContention(SpinLockClass::Tracker).waits == 0, "uncontended: a wait was counted");
}

void ContentionIsCountedPerFamily() {
	for (const auto kind: {SpinLockClass::Other, SpinLockClass::Tracker, SpinLockClass::Pages, SpinLockClass::Textures}) {
		(void)TrackingSpinLock::TakeContention(kind);
	}
	TrackingSpinLock  lock {SpinLockClass::Pages};
	std::atomic<bool> waiting {false};
	lock.lock();
	std::thread other([&] {
		waiting.store(true, std::memory_order_release);
		std::scoped_lock guard(lock);
	});
	while (!waiting.load(std::memory_order_acquire)) {
		std::this_thread::yield();
	}
	std::this_thread::sleep_for(20ms);
	lock.unlock();
	other.join();
	const auto pages = TrackingSpinLock::TakeContention(SpinLockClass::Pages);
	Check(pages.waits == 1, "contended: the wait was not counted once");
	Check(pages.ns >= 10'000'000, "contended: the waiting time is too short");
	Check(TrackingSpinLock::TakeContention(SpinLockClass::Tracker).waits == 0, "contended: counted on another family");
	Check(TrackingSpinLock::TakeContention(SpinLockClass::Textures).waits == 0, "contended: counted on the texture cache");
	Check(TrackingSpinLock::TakeContention(SpinLockClass::Pages).waits == 0, "contended: not reset when taken");
}

void DirtyPagesNeedNoLockedWork(RegionManager& region) {
	bool gpu = true;
	Check(region.WriteFaultFindsOpen(0, Block, false, TRACKER_PAGE_SIZE, &gpu), "fresh region: a written page is not open");
	Check(!gpu, "fresh region: a block reports a GPU page");
	Check(region.WriteFaultFindsOpen(5 * TRACKER_PAGE_SIZE, Block, true, TRACKER_PAGE_SIZE, &gpu), "fresh region: a written block is not open");
	Check(region.IsModifiedUnlocked<DirtySource::Cpu>(0, Block), "fresh region: the unlocked probe misses CPU writes");
	Check(!region.IsModifiedUnlocked<DirtySource::Gpu>(0, Block), "fresh region: the unlocked probe sees GPU writes");
	std::scoped_lock guard(region.lock);
	const auto       page = region.FaultWrite(0, Block, false, TRACKER_PAGE_SIZE);
	Check(page.done && !page.tracked && page.opened == 0, "fresh region: a page fault did not finish on the page");
	const auto block = region.FaultWrite(Block, Block, true, TRACKER_PAGE_SIZE);
	Check(block.done && block.opened == 0, "fresh region: a block fault opened written pages again");
}

void PagesInPlaceAreNeverTracked(RegionManager& region) {
	std::scoped_lock guard(region.lock);
	const auto       begin = 2 * Block;
	Check(region.SetInPlace(begin, Block) == Block / TRACKER_PAGE_SIZE, "in place: wrong page count");
	Check(region.SetInPlace(begin, Block) == 0, "in place: pages counted twice");
	Check(!region.IsModified<DirtySource::Cpu>(begin, Block), "in place: pages still written by the CPU");
	bool gpu = true;
	Check(region.WriteFaultFindsOpen(begin + TRACKER_PAGE_SIZE, Block, true, TRACKER_PAGE_SIZE, &gpu), "in place: a fault needs the lock");
	region.ChangeState<DirtySource::Cpu, true>(begin, Block);
	Check(!region.IsModified<DirtySource::Cpu>(begin, Block), "in place: marked written by the CPU");
	int noted = 0;
	region.MarkGpuDirtyUnlessCpuDirty(begin, Block, [&](uint64_t, uint64_t) { noted++; });
	Check(noted == 0 && !region.IsModified<DirtySource::Gpu>(begin, Block), "in place: marked written by the GPU");
	region.ChangeState<DirtySource::Gpu, true>(begin, Block);
	Check(!region.IsModified<DirtySource::Gpu>(begin, Block), "in place: a GPU write marked the pages");
	uint64_t reported = 0;
	region.ConsumeCpuWritten(CpuWriteConsumer::BufferTables, begin, Block,
	                         [&](uint64_t, uint64_t bytes) { reported += bytes; });
	Check(reported == Block, "in place: a consumer did not see the pages as written");
	reported = 0;
	region.ConsumeCpuWritten(CpuWriteConsumer::BufferTables, begin, Block,
	                         [&](uint64_t, uint64_t bytes) { reported += bytes; });
	Check(reported == Block, "in place: a consumer saw the pages unchanged");
	const auto mixed = region.FaultWrite(begin - TRACKER_PAGE_SIZE, Block, false, 2 * TRACKER_PAGE_SIZE);
	Check(mixed.done, "in place: a fault across the edge did not finish");
	Check(!region.IsModified<DirtySource::Cpu>(begin, TRACKER_PAGE_SIZE), "in place: a neighbour fault wrote the page");
	Check(region.ClearInPlace(begin, Block) == Block / TRACKER_PAGE_SIZE, "in place: wrong page count handed back");
	Check(region.IsModified<DirtySource::Cpu>(begin, Block), "handed back: pages not written by the CPU");
	Check(region.ClearInPlace(begin, Block) == 0, "handed back: pages handed back twice");
}

void ConcurrentFaultsOnOpenPagesTakeNoLock(RegionManager& region) {
	constexpr int      Threads = 10;
	constexpr int      Faults  = 20000;
	std::atomic<int>   locked {0};
	std::atomic<bool>  wrong {false};
	std::vector<std::thread> threads;
	(void)TrackingSpinLock::TakeContention(SpinLockClass::Tracker);
	for (int t = 0; t < Threads; t++) {
		threads.emplace_back([&, t] {
			for (int i = 0; i < Faults; i++) {
				const auto offset = (static_cast<uint64_t>(t) * 16 + static_cast<uint64_t>(i % 16)) * TRACKER_PAGE_SIZE;
				bool       gpu    = false;
				if (region.WriteFaultFindsOpen(offset, Block, (i & 1) != 0, TRACKER_PAGE_SIZE, &gpu)) {
					region.NoteCpuWriteOnOpenPage(offset, Block, (i & 1) != 0, TRACKER_PAGE_SIZE);
					continue;
				}
				locked.fetch_add(1, std::memory_order_relaxed);
				std::scoped_lock guard(region.lock);
				if (!region.FaultWrite(offset, Block, (i & 1) != 0, TRACKER_PAGE_SIZE).done) {
					wrong.store(true, std::memory_order_relaxed);
				}
			}
		});
	}
	for (auto& thread: threads) {
		thread.join();
	}
	Check(locked.load() == 0, "concurrent: a fault on an open page took the lock");
	Check(!wrong.load(), "concurrent: a fault did not finish");
	Check(TrackingSpinLock::TakeContention(SpinLockClass::Tracker).waits == 0, "concurrent: the region lock was contended");
}

void OpenPageFaultStillSignalsTheWrite(PageManager& pages) {
	namespace Bda = Libs::Graphics::BdaDirtyRegions;
	MemoryTracker  tracker(pages);
	const uint64_t vaddr = RegionBase + 64 * TRACKER_REGION_SIZE + 3 * TRACKER_PAGE_SIZE;
	const uint64_t index = vaddr / TRACKER_REGION_SIZE;
	const uint64_t bit   = uint64_t {1} << (index % 64);
	Check(tracker.IsRegionCpuModified(vaddr, TRACKER_PAGE_SIZE), "signal: a fresh region is not written by the CPU");
	Bda::g_pending[index / 64].fetch_and(~bit);
	const auto region = tracker.RegionCpuDirtyEpoch(index);
	const auto global = RegionManager::CpuDirtyEpoch();
	bool       gpu    = true;
	Check(tracker.WriteFaultFindsOpen(vaddr, Block, false, TRACKER_PAGE_SIZE, &gpu), "signal: the open page took the lock");
	Check(tracker.RegionCpuDirtyEpoch(index) > region, "signal: the region CPU write epoch did not move");
	Check(RegionManager::CpuDirtyEpoch() > global, "signal: the global CPU write epoch did not move");
	Check((Bda::g_pending[index / 64].load() & bit) != 0, "signal: the region was not marked for device address reads");

	std::vector<PageManager::DeferredRange> deferred;
	const auto consumed = [&](CpuWriteConsumer consumer, uint64_t address, uint64_t size) {
		uint64_t bytes = 0;
		tracker.ConsumeCpuWrites(consumer, address, size, [&](uint64_t, uint64_t run) { bytes += run; });
		return bytes;
	};
	const auto upload = [&](uint64_t address, uint64_t size) {
		PageManager::SetDeferredWriteProtectSink(&deferred);
		tracker.ForEachUploadRange(address, size, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
		PageManager::SetDeferredWriteProtectSink(nullptr);
	};
	const struct {
		uint64_t    address;
		uint64_t    size;
		bool        open_block;
		const char* what;
	} cases[] = {
	    {vaddr, TRACKER_PAGE_SIZE, false, "signal (page)"},
	    {(vaddr / Block + 1) * Block, Block, true, "signal (block)"},
	};
	for (const auto& test: cases) {
		const auto fault = test.address + (test.open_block ? 5 * TRACKER_PAGE_SIZE : 0);
		for (const auto consumer: {CpuWriteConsumer::BufferTables, CpuWriteConsumer::IndirectWriteTables}) {
			Check(consumed(consumer, test.address, test.size) == test.size, test.what, "a table did not see its open pages");
		}
		Check(tracker.WriteFaultFindsOpen(fault, Block, test.open_block, TRACKER_PAGE_SIZE, &gpu), test.what,
		      "the pages read by the tables took the lock");
		upload(test.address, test.size);
		Check(!tracker.IsRegionCpuModified(test.address, test.size), test.what, "the upload left the pages written");
		Check(!deferred.empty(), test.what, "the upload did not protect the pages");
		deferred.clear();
		Check(consumed(CpuWriteConsumer::BufferTables, test.address, test.size) == test.size, test.what,
		      "the descriptor tables missed a write served without the lock");
		Check(consumed(CpuWriteConsumer::IndirectWriteTables, test.address, test.size) == test.size, test.what,
		      "the indirect write tables missed a write served without the lock");
		Check(consumed(CpuWriteConsumer::BufferTables, test.address, test.size) == 0, test.what,
		      "the descriptor tables saw one write twice");
		tracker.MarkRegionAsCpuModified(test.address, test.size);
	}
}

void OpenPageFaultCountsForBusyPages(PageManager& pages) {
	MemoryTracker  tracker(pages);
	const uint64_t vaddr = RegionBase + 65 * TRACKER_REGION_SIZE + 7 * TRACKER_PAGE_SIZE;
	std::vector<PageManager::DeferredRange> deferred;
	const auto sync = [&](uint32_t device_sync) {
		PageManager::SetDeferredWriteProtectSink(&deferred);
		tracker.ForEachUploadRange(vaddr, TRACKER_PAGE_SIZE, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {},
		                           device_sync);
		PageManager::SetDeferredWriteProtectSink(nullptr);
	};
	sync(1);
	Check(!tracker.IsRegionCpuModified(vaddr, TRACKER_PAGE_SIZE), "busy: the first upload kept the page");
	for (uint32_t device_sync = 2; device_sync <= 3; device_sync++) {
		Check(tracker.FaultWrite(vaddr, Block, false, TRACKER_PAGE_SIZE).done, "busy: a fault on a protected page did not finish");
		sync(device_sync);
		Check(!tracker.IsRegionCpuModified(vaddr, TRACKER_PAGE_SIZE), "busy: a page written twice was kept");
	}
	tracker.MarkRegionAsCpuModified(vaddr, TRACKER_PAGE_SIZE);
	(void)tracker.TakeKeptPages();
	bool gpu = true;
	Check(tracker.WriteFaultFindsOpen(vaddr, Block, false, TRACKER_PAGE_SIZE, &gpu), "busy: the open page took the lock");
	sync(4);
	Check(tracker.IsRegionCpuModified(vaddr, TRACKER_PAGE_SIZE), "busy: a fault served without the lock was not counted");
	Check(tracker.TakeKeptPages(), "busy: the page kept written was not reported");
	tracker.MarkRegionAsCpuModified(vaddr, TRACKER_PAGE_SIZE);
}

void AtomicBitsLoseNoConcurrentWrite() {
	constexpr size_t   Setters = 4;
	constexpr uint32_t Rounds  = 5000;
	RegionBits         bits;
	std::atomic<bool>  stop {false};
	std::atomic<bool>  stuck {false};
	std::array<std::atomic<uint32_t>, Setters> taken {};
	std::thread consumer([&] {
		while (!stop.load(std::memory_order_acquire)) {
			const RegionBits batch = bits.TakeRange(0, 2 * Setters);
			for (size_t setter = 0; setter < Setters; setter++) {
				if (batch.Get(2 * setter)) taken[setter].fetch_add(1, std::memory_order_release);
			}
		}
	});
	std::vector<std::thread> setters;
	for (size_t setter = 0; setter < Setters; setter++) {
		setters.emplace_back([&, setter] {
			for (uint32_t round = 0; round < Rounds && !stuck.load(std::memory_order_relaxed); round++) {
				if ((round & 1) != 0) {
					bits.SetAtomic(2 * setter);
				} else {
					RegionBits own;
					own.Set(2 * setter);
					bits.SetAtomic(own);
				}
				bits.SetRangeAtomic(2 * setter + 1, 2 * setter + 2);
				const auto deadline = std::chrono::steady_clock::now() + 5s;
				while (taken[setter].load(std::memory_order_acquire) <= round) {
					if (std::chrono::steady_clock::now() > deadline) {
						stuck.store(true, std::memory_order_relaxed);
						return;
					}
					std::this_thread::yield();
				}
			}
		});
	}
	for (auto& thread: setters) {
		thread.join();
	}
	stop.store(true, std::memory_order_release);
	consumer.join();
	Check(!stuck.load(), "atomic bits: a write set during a take was lost");
	for (size_t setter = 0; setter < Setters; setter++) {
		Check(taken[setter].load() == Rounds, "atomic bits: a write was taken twice or never");
	}
}

}

int main() {
	UncontendedLockCountsNothing();
	ContentionIsCountedPerFamily();
	PageManager   pages;
	RegionManager region(pages, RegionBase);
	DirtyPagesNeedNoLockedWork(region);
	PagesInPlaceAreNeverTracked(region);
	ConcurrentFaultsOnOpenPagesTakeNoLock(region);
	OpenPageFaultStillSignalsTheWrite(pages);
	OpenPageFaultCountsForBusyPages(pages);
	AtomicBitsLoseNoConcurrentWrite();
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_write_fault_tracking: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_write_fault_tracking: ok\n");
	return 0;
}
