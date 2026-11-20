#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/logging/log.h"
#include "common/timer.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/hotWriteWatch.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/regionManager.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cinttypes>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;
constexpr uint64_t READ_GRANULE_SIZE  = 64 * 1024;
constexpr uint64_t READ_GRANULE_PAGES = READ_GRANULE_SIZE / PAGE_SIZE;

constexpr uint8_t  HOT_COLD           = 0;
constexpr uint8_t  HOT_ON             = 1;
constexpr uint8_t  HOT_REFUSED        = 2;
constexpr uint8_t  HOT_PROMOTING      = 3;
constexpr uint8_t  HOT_FAULTS         = 2;
constexpr uint64_t HOT_WINDOW_NS      = 1'000'000'000;
constexpr uint32_t HOT_IDLE_SECONDS   = 4;
constexpr uint32_t HOT_PAGE_CAP       = 16384;
constexpr size_t   HOT_IDLE_PER_SWEEP = 1024;
constexpr uint64_t HOT_GRANULE_SIZE   = 2 * 1024 * 1024;
constexpr uint64_t HOT_CLUSTER_GAP    = 1024 * 1024;

[[noreturn]] void Fatal(const char* format, ...) {
	char    text[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	std::fprintf(stderr, "PageManager fatal: %s\n", text);
	std::fflush(stderr);
	Log::WriteToSessionLog(std::string("PageManager fatal: ") + text + "\n");
	Log::Flush();
	std::_Exit(322);
}

using SpinGuard = Common::ScopedLock<TrackingSpinLock>;

thread_local std::vector<PageManager::DeferredRange>* g_deferred_write_protect = nullptr;

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

}

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  = 0;
		uint8_t access_watchers = 0;
		uint8_t hot             = HOT_COLD;
		uint8_t write_faults    = 0;
		uint8_t fault_window    = 0;

		[[nodiscard]] bool WriteProtected() const noexcept { return write_watchers != 0 && hot != HOT_ON; }

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (WriteProtected()) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers == UINT8_MAX) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == UINT8_MAX) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 5);

	struct Region {
		TrackingSpinLock                    lock {SpinLockClass::Pages};
		std::array<PageState, REGION_PAGES> pages KYTY_GUARDED_BY(lock);
		std::array<std::atomic<uint8_t>, REGION_PAGES> read_hint {};
	};

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					Fatal("destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return regions[vaddr / REGION_SIZE].load(std::memory_order_acquire);
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		const auto kind = mode == Common::VirtualMemory::Mode::Read        ? 0
		                  : mode == Common::VirtualMemory::Mode::ReadWrite ? 1
		                  : mode == Common::VirtualMemory::Mode::NoAccess  ? 2
		                                                                   : 3;
		protects[kind].fetch_add(1, std::memory_order_relaxed);
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard lock(region.lock);
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				if (track && !is_read && g_deferred_write_protect != nullptr) {
					g_deferred_write_protect->push_back({base_addr + range_begin * PAGE_SIZE, range_bytes});
				} else {
					Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				}
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		constexpr int delta = track ? 1 : -1;
		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<delta, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();
			if constexpr (is_read) {
				const bool faults = page.access_watchers != 0;
				if (faults) MarkReadGranule(address);
				region.read_hint[page_index].store(faults, std::memory_order_release);
			}

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
		if constexpr (is_read) SyncReadGranules(region, base_addr, first, last);
	}

	std::unique_ptr<std::atomic<uint64_t>[]> read_granules =
	    std::make_unique<std::atomic<uint64_t>[]>(ADDRESS_SIZE / READ_GRANULE_SIZE / 64);

	void MarkReadGranule(uint64_t address) noexcept {
		const auto granule = address / READ_GRANULE_SIZE;
		read_granules[granule / 64].fetch_or(uint64_t {1} << (granule % 64), std::memory_order_release);
	}

	void SyncReadGranules(const Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		if (first >= last) return;
		for (size_t g = first / READ_GRANULE_PAGES; g <= (last - 1) / READ_GRANULE_PAGES; ++g) {
			bool any = false;
			for (size_t p = g * READ_GRANULE_PAGES; p < (g + 1) * READ_GRANULE_PAGES; ++p)
				any |= region.read_hint[p].load(std::memory_order_relaxed) != 0;
			const auto granule = base_addr / READ_GRANULE_SIZE + g;
			const auto bit     = uint64_t {1} << (granule % 64);
			if (any)
				read_granules[granule / 64].fetch_or(bit, std::memory_order_release);
			else if ((read_granules[granule / 64].load(std::memory_order_relaxed) & bit) != 0)
				read_granules[granule / 64].fetch_and(~bit, std::memory_order_release);
		}
	}

	[[nodiscard]] bool MayHaveReadWatchers(uint64_t vaddr, uint64_t last_byte) const noexcept {
		if (last_byte >= ADDRESS_SIZE) return true;
		for (auto g = vaddr / READ_GRANULE_SIZE; g <= last_byte / READ_GRANULE_SIZE; ++g)
			if (((read_granules[g / 64].load(std::memory_order_acquire) >> (g % 64)) & 1u) != 0) return true;
		return false;
	}

	void ReapplyProtection(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			if (auto* region = FindRegion(chunk_begin); region != nullptr) {
				SpinGuard  lock(region->lock);
				const auto first     = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
				const auto last      = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				size_t     run_begin = first;
				auto       run_perms = Common::VirtualMemory::Mode::ReadWrite;
				bool       in_run    = false;
				for (size_t page_index = first; page_index <= last; page_index++) {
					const bool watched = page_index < last && (region->pages[page_index].WriteProtected() ||
					                                           region->pages[page_index].access_watchers != 0);
					const auto perms   = watched ? region->pages[page_index].Perms() : run_perms;
					if (in_run && (!watched || perms != run_perms)) {
						Protect(region_base + run_begin * PAGE_SIZE, (page_index - run_begin) * PAGE_SIZE, run_perms);
						in_run = false;
					}
					if (watched && !in_run) {
						run_begin = page_index;
						run_perms = perms;
						in_run    = true;
					}
				}
			}
			chunk_begin = chunk_end;
		}
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	void MarkHotGranule(uint64_t page) noexcept {
		const auto granule = page / HOT_GRANULE_SIZE;
		hot_granules[granule / 64].fetch_or(uint64_t {1} << (granule % 64), std::memory_order_release);
	}

	[[nodiscard]] bool MayHaveHotPages(uint64_t vaddr, uint64_t size) const noexcept {
		if (size == 0 || vaddr >= ADDRESS_SIZE) return false;
		const auto last = std::min(vaddr + size - 1, ADDRESS_SIZE - 1);
		for (auto g = vaddr / HOT_GRANULE_SIZE; g <= last / HOT_GRANULE_SIZE; ++g) {
			if (((hot_granules[g / 64].load(std::memory_order_acquire) >> (g % 64)) & 1u) != 0) return true;
		}
		return false;
	}

	PageState* FindPage(uint64_t page, Region** region) const noexcept KYTY_NO_THREAD_SAFETY_ANALYSIS {
		*region = FindRegion(page);
		return *region == nullptr ? nullptr : &(*region)->pages[(page % REGION_SIZE) / PAGE_SIZE];
	}

	void MergePendingHotPages(uint32_t now) {
		std::vector<uint64_t> pending;
		{
			std::lock_guard lock(hot_pending_mutex);
			pending.swap(hot_pending);
		}
		for (const auto page: pending) {
			Region* region = nullptr;
			bool    on     = false;
			if (auto* state = FindPage(page, &region); state != nullptr) {
				SpinGuard lock(region->lock);
				on = state->hot == HOT_ON || state->hot == HOT_PROMOTING;
			}
			if (on && hot_pages.emplace(page, now).second) {
				hot_runs_dirty = true;
			} else {
				hot_count.fetch_sub(1, std::memory_order_relaxed);
			}
		}
	}

	void RebuildHotRuns() {
		hot_runs.clear();
		hot_clusters.clear();
		for (const auto& [page, last]: hot_pages) {
			if (!hot_runs.empty() && hot_runs.back().second == page) {
				hot_runs.back().second = page + PAGE_SIZE;
			} else {
				hot_runs.emplace_back(page, page + PAGE_SIZE);
			}
		}
		for (const auto& run: hot_runs) {
			if (!hot_clusters.empty() && run.first - hot_clusters.back().second <= HOT_CLUSTER_GAP) {
				hot_clusters.back().second = run.second;
			} else {
				hot_clusters.push_back(run);
			}
		}
		hot_runs_dirty = false;
	}

	void Demote(uint64_t page, bool keep_protection) {
		Region* region = nullptr;
		auto*   state  = FindPage(page, &region);
		if (state == nullptr) return;
		SpinGuard lock(region->lock);
		if (state->hot != HOT_ON && state->hot != HOT_PROMOTING) return;
		state->hot          = HOT_COLD;
		state->write_faults = 0;
		if (keep_protection && state->WriteProtected() && state->access_watchers == 0) {
			Protect(page, PAGE_SIZE, Common::VirtualMemory::Mode::Read);
		}
	}

	void LogHotSummary(uint64_t now_ns) {
		auto last = hot_last_line.load(std::memory_order_relaxed);
		if (last == 0) {
			hot_last_line.compare_exchange_strong(last, now_ns, std::memory_order_relaxed);
			return;
		}
		if (now_ns - last < 10'000'000'000 ||
		    !hot_last_line.compare_exchange_strong(last, now_ns, std::memory_order_relaxed)) {
			return;
		}
		const auto take = [](std::atomic<uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
		const auto faults    = take(guest_write_faults);
		const auto read_only = take(protects[0]);
		const auto writable  = take(protects[1]);
		const auto no_access = take(protects[2]);
		const auto other     = take(protects[3]);
		const auto scans     = take(hot_scans);
		const auto scoped    = take(hot_scoped_scans);
		const auto scan_ns   = take(hot_scan_ns);
		const auto collected = take(hot_collected);
		const auto promoted  = take(hot_promoted);
		const auto idle      = take(hot_idle);
		const auto lost      = take(hot_lost);
		const auto refused   = take(hot_refused);
		char       line[896];
		std::snprintf(line, sizeof(line),
		              "[kyty-hot-pages] summary of the last %.0f s: %" PRIu64 " guest write faults, %" PRIu64
		              " page protections (%" PRIu64 " read only, %" PRIu64 " writable, %" PRIu64 " no access, %" PRIu64
		              " other); %" PRIu64 " pages watched without a fault in %" PRIu64 " clusters, %" PRIu64
		              " collections and %" PRIu64 " before GPU writes in %.3f s, %" PRIu64 " page writes collected, %" PRIu64 " pages promoted, %" PRIu64
		              " demoted when idle, %" PRIu64 " that lost their watch, %" PRIu64 " refused\n",
		              static_cast<double>(now_ns - last) / 1e9, faults, read_only + writable + no_access + other, read_only,
		              writable, no_access, other, hot_page_total.load(std::memory_order_relaxed),
		              hot_run_total.load(std::memory_order_relaxed), scans, scoped, static_cast<double>(scan_ns) / 1e9, collected,
		              promoted, idle, lost, refused);
		Log::WriteToConsoleAndLog(line);
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;

	std::mutex                                  hot_mutex;
	std::map<uint64_t, uint32_t>                hot_pages;
	std::vector<std::pair<uint64_t, uint64_t>>  hot_runs;
	std::vector<std::pair<uint64_t, uint64_t>>  hot_clusters;
	bool                                        hot_runs_dirty = false;
	uint32_t                                    hot_last_sweep = 0;
	std::atomic<uint64_t>                       hot_last_line {0};
	std::atomic<uint64_t>                       hot_scans {0};
	std::atomic<uint64_t>                       hot_scan_ns {0};
	std::atomic<uint64_t>                       hot_collected {0};
	std::atomic<uint64_t>                       hot_idle {0};
	std::atomic<uint64_t>                       hot_lost {0};
	std::atomic<uint64_t>                       hot_page_total {0};
	std::atomic<uint64_t>                       hot_run_total {0};
	std::atomic<uint64_t>                       guest_write_faults {0};
	std::array<std::atomic<uint64_t>, 4>        protects {};
	std::mutex                                  hot_pending_mutex;
	std::vector<uint64_t>                       hot_pending;
	std::atomic<bool>                           hot_any {false};
	std::unique_ptr<std::atomic<uint64_t>[]>    hot_granules =
	    std::make_unique<std::atomic<uint64_t>[]>(ADDRESS_SIZE / HOT_GRANULE_SIZE / 64);
	std::atomic<uint64_t>                       hot_scoped_scans {0};
	std::atomic<uint32_t>                       hot_count {0};
	std::atomic<uint64_t>                       hot_promoted {0};
	std::atomic<uint64_t>                       hot_refused {0};
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {
	(void)HotPagesEnabled();
}

PageManager::~PageManager() = default;

const std::atomic<uint64_t>* PageManager::ReadGranules() const noexcept {
	static_assert(READ_GRANULE_SIZE == uint64_t {1} << READ_GRANULE_BITS);
	return m_impl->read_granules.get();
}

bool PageManager::HasReadWatchers(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) return false;
	if (!m_impl->MayHaveReadWatchers(vaddr, vaddr + size - 1)) return false;
	const auto end = Common::AlignDown(vaddr + size - 1, PAGE_SIZE);
	for (auto page = Common::AlignDown(vaddr, PAGE_SIZE);; page += PAGE_SIZE) {
		if (const auto* region = m_impl->FindRegion(page);
		    region != nullptr &&
		    region->read_hint[(page % REGION_SIZE) / PAGE_SIZE].load(std::memory_order_acquire))
			return true;
		if (page == end) return false;
	}
}

bool PageManager::IsWatched(uint64_t vaddr) const {
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto page = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	SpinGuard  lock(region->lock);
	return region->pages[page].write_watchers != 0 || region->pages[page].access_watchers != 0;
}

bool PageManager::WatchedSpan(uint64_t vaddr, uint64_t size, uint64_t* first, uint64_t* end) const noexcept {
	bool           any  = false;
	const uint64_t stop = UINT64_MAX - vaddr < size ? UINT64_MAX : vaddr + size;
	for (uint64_t at = Common::AlignDown(vaddr, PAGE_SIZE); at < stop && at < ADDRESS_SIZE;) {
		const uint64_t until = std::min(stop, (at / REGION_SIZE + 1) * REGION_SIZE);
		if (auto* region = m_impl->FindRegion(at); region != nullptr) {
			SpinGuard lock(region->lock);
			for (uint64_t page = at; page < until; page += PAGE_SIZE) {
				const auto& state = region->pages[(page % REGION_SIZE) / PAGE_SIZE];
				if (state.write_watchers == 0 && state.access_watchers == 0) continue;
				if (!any) *first = page;
				any  = true;
				*end = page + PAGE_SIZE;
			}
		}
		at = until;
	}
	return any;
}

bool PageManager::AllowExecute(uint64_t vaddr) {
	using Mode   = Common::VirtualMemory::Mode;
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto page = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	SpinGuard  lock(region->lock);
	const auto perms = region->pages[page].Perms();
	if (perms == Mode::NoAccess) {
		return false;
	}
	return Libs::LibKernel::Memory::ProtectGuestHostMemory(
	    Common::AlignDown(vaddr, PAGE_SIZE), PAGE_SIZE,
	    static_cast<Mode>(static_cast<uint32_t>(perms) | static_cast<uint32_t>(Mode::Execute)));
}

void PageManager::SetDeferredWriteProtectSink(std::vector<DeferredRange>* sink) noexcept {
	g_deferred_write_protect = sink;
}

void PageManager::ReapplyProtection(uint64_t vaddr, uint64_t size) {
	m_impl->ReapplyProtection(vaddr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, is_read>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true, true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<true, false>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false, true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false, false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	const auto first       = start_range.first;
	const auto last        = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

bool PageManager::HotPagesEnabled() noexcept {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOT_PAGE_WRITE_WATCH");
		if (value == nullptr || value[0] != '1') return false;
		return HotWriteWatch::Get().Available();
	}();
	return enabled;
}

void PageManager::NoteWriteFault(uint64_t vaddr) {
	auto& impl = *m_impl;
	impl.guest_write_faults.fetch_add(1, std::memory_order_relaxed);
	if (!HotPagesEnabled() || !GuestRange {vaddr, 1}.Valid()) return;
	const auto page   = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto window = static_cast<uint8_t>(Common::SteadyNs() / HOT_WINDOW_NS);
	Impl::Region* region = nullptr;
	auto*         state  = impl.FindPage(page, &region);
	if (state == nullptr) return;
	{
		SpinGuard lock(region->lock);
		if (state->hot != HOT_COLD || state->access_watchers != 0) return;
		if (state->fault_window != window) {
			state->fault_window = window;
			state->write_faults = 0;
		}
		if (++state->write_faults < HOT_FAULTS) return;
		state->write_faults = 0;
		if (impl.hot_count.load(std::memory_order_relaxed) >= HOT_PAGE_CAP) return;
		state->hot = HOT_PROMOTING;
	}
	if (!HotWriteWatch::Get().Watch(page, page + PAGE_SIZE)) {
		SpinGuard lock(region->lock);
		if (state->hot == HOT_PROMOTING) state->hot = HOT_REFUSED;
		impl.hot_refused.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	impl.hot_count.fetch_add(1, std::memory_order_relaxed);
	impl.MarkHotGranule(page);
	{
		std::lock_guard lock(impl.hot_pending_mutex);
		impl.hot_pending.push_back(page);
	}
	impl.hot_promoted.fetch_add(1, std::memory_order_relaxed);
	impl.hot_any.store(true, std::memory_order_release);
	SpinGuard lock(region->lock);
	if (state->hot != HOT_PROMOTING) return;
	state->hot = HOT_ON;
	if (state->write_watchers != 0 && state->access_watchers == 0) {
		impl.Protect(page, PAGE_SIZE, Common::VirtualMemory::Mode::ReadWrite);
	}
}

void PageManager::CollectHotPageWrites(HotPageWritten written, void* context, uint64_t vaddr, uint64_t size) {
	auto&      impl   = *m_impl;
	const bool scoped = size != 0;
	if (!scoped) impl.LogHotSummary(Common::SteadyNs());
	if (!impl.hot_any.load(std::memory_order_acquire)) return;
	if (scoped && (vaddr > UINT64_MAX - size || !impl.MayHaveHotPages(vaddr, size))) return;
	std::vector<uint64_t> reported;
	std::vector<uint64_t> lost;
	std::vector<uint64_t> idle;
	const auto  started = Common::SteadyNs();
	const auto  now     = static_cast<uint32_t>(started / 1'000'000'000);
	const auto& watch   = HotWriteWatch::Get();
	const auto  push    = [](void* pages, uint64_t begin, uint64_t end) {
		for (auto page = begin; page < end; page += PAGE_SIZE) static_cast<std::vector<uint64_t>*>(pages)->push_back(page);
	};
	const auto collect = [&](uint64_t begin, uint64_t end, bool strict) {
		const auto before = reported.size();
		if (watch.Collect(begin, end, push, &reported, strict)) return true;
		reported.resize(before);
		return false;
	};
	const auto lose = [&](uint64_t begin, uint64_t end) {
		for (auto it = impl.hot_pages.lower_bound(begin); it != impl.hot_pages.end() && it->first < end; ++it) {
			lost.push_back(it->first);
		}
	};
	{
		std::lock_guard lock(impl.hot_mutex);
		impl.MergePendingHotPages(now);
		if (scoped) {
			const auto end   = vaddr + size;
			auto       it    = impl.hot_pages.lower_bound(Common::AlignDown(vaddr, PAGE_SIZE));
			uint64_t   first = 0;
			uint64_t   last  = 0;
			for (; it != impl.hot_pages.end() && it->first < end; ++it) {
				if (last == 0) first = it->first;
				last = it->first + PAGE_SIZE;
			}
			if (last != 0 && !collect(first, last, false)) lose(first, last);
		} else {
			if (impl.hot_runs_dirty) impl.RebuildHotRuns();
			if (now != impl.hot_last_sweep) {
				for (const auto& [begin, end]: impl.hot_runs) {
					if (!collect(begin, end, true)) push(&lost, begin, end);
				}
			} else {
				for (const auto& [begin, end]: impl.hot_clusters) {
					if (!collect(begin, end, false)) lose(begin, end);
				}
			}
		}
		for (const auto page: reported) {
			if (auto it = impl.hot_pages.find(page); it != impl.hot_pages.end()) it->second = now;
		}
		if (!scoped && now != impl.hot_last_sweep) {
			impl.hot_last_sweep = now;
			for (const auto& [page, last]: impl.hot_pages) {
				if (now - last >= HOT_IDLE_SECONDS && idle.size() < HOT_IDLE_PER_SWEEP) idle.push_back(page);
			}
		}
		for (const auto* pages: {&lost, &idle}) {
			size_t removed = 0;
			for (const auto page: *pages) removed += impl.hot_pages.erase(page);
			impl.hot_runs_dirty |= removed != 0;
			impl.hot_count.fetch_sub(static_cast<uint32_t>(removed), std::memory_order_relaxed);
		}
		impl.hot_any.store(!impl.hot_pages.empty() || impl.hot_count.load(std::memory_order_relaxed) != 0,
		                   std::memory_order_release);
		impl.hot_page_total.store(impl.hot_pages.size(), std::memory_order_relaxed);
		if (!scoped) impl.hot_run_total.store(impl.hot_clusters.size(), std::memory_order_relaxed);
		impl.hot_lost.fetch_add(lost.size(), std::memory_order_relaxed);
		impl.hot_idle.fetch_add(idle.size(), std::memory_order_relaxed);
		impl.hot_collected.fetch_add(reported.size() + lost.size(), std::memory_order_relaxed);
		(scoped ? impl.hot_scoped_scans : impl.hot_scans).fetch_add(1, std::memory_order_relaxed);
		impl.hot_scan_ns.fetch_add(Common::SteadyNs() - started, std::memory_order_relaxed);
	}
	for (const auto page: lost) {
		impl.Demote(page, true);
		watch.Unwatch(page, page + PAGE_SIZE);
		reported.push_back(page);
	}
	for (const auto page: idle) {
		impl.Demote(page, true);
		if (!watch.Collect(page, page + PAGE_SIZE, push, &reported)) reported.push_back(page);
		watch.Unwatch(page, page + PAGE_SIZE);
	}
	if (reported.empty()) return;
	std::sort(reported.begin(), reported.end());
	reported.erase(std::unique(reported.begin(), reported.end()), reported.end());
	auto begin = reported.front();
	auto end   = begin + PAGE_SIZE;
	for (size_t i = 1; i < reported.size(); i++) {
		if (reported[i] != end) {
			written(context, begin, end);
			begin = reported[i];
		}
		end = reported[i] + PAGE_SIZE;
	}
	written(context, begin, end);
}

bool PageManager::HasHotPages() const noexcept {
	return m_impl->hot_any.load(std::memory_order_acquire);
}

void PageManager::ForgetHotPages(uint64_t vaddr, uint64_t size) {
	auto& impl = *m_impl;
	if (!impl.hot_any.load(std::memory_order_acquire) || size == 0 || vaddr > UINT64_MAX - size) return;
	std::vector<uint64_t> pages;
	{
		std::lock_guard lock(impl.hot_mutex);
		impl.MergePendingHotPages(static_cast<uint32_t>(Common::SteadyNs() / 1'000'000'000));
		for (auto it = impl.hot_pages.lower_bound(Common::AlignDown(vaddr, PAGE_SIZE));
		     it != impl.hot_pages.end() && it->first < vaddr + size;) {
			pages.push_back(it->first);
			it = impl.hot_pages.erase(it);
		}
		if (pages.empty()) return;
		impl.hot_runs_dirty = true;
		impl.hot_count.fetch_sub(static_cast<uint32_t>(pages.size()), std::memory_order_relaxed);
		impl.hot_any.store(!impl.hot_pages.empty() || impl.hot_count.load(std::memory_order_relaxed) != 0,
		                   std::memory_order_release);
	}
	const auto& watch = HotWriteWatch::Get();
	for (const auto page: pages) {
		impl.Demote(page, false);
		watch.Unwatch(page, page + PAGE_SIZE);
	}
}

}
