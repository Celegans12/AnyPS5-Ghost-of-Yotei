#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "kernel/memory.h"

#include <thread>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

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

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

static inline void SpinPause() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

class SpinGuard final {
public:
	explicit SpinGuard(std::atomic_flag& lock): m_lock(lock) {
		if (m_lock.test_and_set(std::memory_order_acquire)) {
			KYTY_PROFILER_BLOCK("PageLock::Wait");
			// After ~25 us of spinning the waiter yields, after ~1 ms it sleeps 1 ms per round. The holder may be inside
			// VirtualProtect (UpdateRegionWatchers / UpdateImageRead protect under this lock); with ~10 game threads spinning
			// here at once in the fault handler it got no core until Windows boosted it: ~2.5 s hitches while walking
			// (found with a stall watchdog). Live A/B on replays of a recorded town walk: 0 stalls instead of 1-8, +13-18% median fps.
			uint32_t spins  = 0;
			uint32_t yields = 0;
			while (m_lock.test_and_set(std::memory_order_acquire)) {
				while (m_lock.test(std::memory_order_relaxed)) {
					SpinPause();
					if (++spins >= 2048u) {
						spins = 0;
						if (++yields >= 40u) {
							std::this_thread::sleep_for(std::chrono::milliseconds(1));
						} else {
							std::this_thread::yield();
						}
					}
				}
			}
		}
	}
	~SpinGuard() { m_lock.clear(std::memory_order_release); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	std::atomic_flag& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

} // namespace

struct PageManager::Impl {
	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
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
					if (write_watchers == 0x7f) {
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
	static_assert(sizeof(PageState) == 1);

	struct Region {
		std::atomic_flag                    lock = ATOMIC_FLAG_INIT;
		std::array<PageState, REGION_PAGES> pages;
		// The image read watch, one bit per page (ArmImageRead). Written under the lock; lock-free readers may be stale.
		std::array<std::atomic<uint64_t>, REGION_PAGES / 64u> image_read {};
	};
	static_assert(REGION_PAGES % 64u == 0);

	[[nodiscard]] static bool ImageBit(const std::atomic<uint64_t>* image_read, size_t index) noexcept {
		return ((image_read[index / 64u].load(std::memory_order_relaxed) >> (index % 64u)) & 1u) != 0;
	}

	// The page's protection: no access under the image read watch, otherwise what its watchers need.
	[[nodiscard]] static Common::VirtualMemory::Mode ModeOf(const PageState& page, bool image_read) noexcept {
		return image_read ? Common::VirtualMemory::Mode::NoAccess : page.Perms();
	}
	[[nodiscard]] static Common::VirtualMemory::Mode PermsOf(const Region& region, size_t index) noexcept {
		return ModeOf(region.pages[index], ImageBit(region.image_read.data(), index));
	}

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
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
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
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
		// KYTY_TRACE_PERF: tracker protections by thread kind (PERF locks).
		struct ProtectTimer {
			bool                                  on = PerfEnabled();
			std::chrono::steady_clock::time_point start {};
			ProtectTimer() {
				if (on) start = std::chrono::steady_clock::now();
			}
			~ProtectTimer() {
				if (!on) return;
				const auto kind = PerfIsGpuThread() ? 0u : 1u;
				PerfLocks().protects[kind].fetch_add(1, std::memory_order_relaxed);
				PerfLocks().protect_nanos[kind].fetch_add(
				    static_cast<uint64_t>(
				        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()),
				    std::memory_order_relaxed);
			}
		} protect_timer;
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		SpinGuard lock(region.lock);
		auto      perms                 = PermsOf(region, first);
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				Protect(base_addr + range_begin * PAGE_SIZE, range_bytes, perms);
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = PermsOf(region, page_index);
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = PermsOf(region, page_index);

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
	}

	// OpenRegionWatchers: removes the access watcher of every page of read_open and the write watcher of every page of
	// write_open (either may be null) of states[first, last), with the run rule of UpdateRegionWatchers<false, is_read,
	// masked>: a run starts at a page whose mode changed at a watcher edge and takes in the following pages of the same
	// final mode, up to the last changed one; sink(first_page, pages, mode) gets one call per run. A page under the image
	// read watch (image_read: the region's bits) stays no access whatever its counts do.
	template <typename Sink>
	static void OpenRuns(PageState* states, const std::atomic<uint64_t>* image_read, uint64_t base_addr, size_t first,
	                     size_t last, const RegionBits* read_open, const RegionBits* write_open, Sink&& sink) {
		const auto mode_of = [&](size_t index) { return ModeOf(states[index], ImageBit(image_read, index)); };
		auto   perms                 = mode_of(first);
		size_t range_begin           = 0;
		size_t range_pages           = 0;
		size_t potential_range_pages = 0;

		const auto release_pending = [&] {
			if (range_pages != 0) {
				sink(range_begin, range_pages, perms);
				range_pages           = 0;
				potential_range_pages = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page       = states[page_index];
			const auto address    = base_addr + page_index * PAGE_SIZE;
			const bool open_read  = read_open != nullptr && read_open->Get(page_index);
			const bool open_write = write_open != nullptr && write_open->Get(page_index);

			const auto     old_perms   = mode_of(page_index);
			const uint32_t read_count  = open_read ? page.AddDelta<-1, true>(address) : 1u;
			const uint32_t write_count = open_write ? page.AddDelta<-1, false>(address) : 1u;
			const auto     new_perms   = mode_of(page_index);

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_pages != 0) {
				potential_range_pages++;
			}

			if (!open_read && !open_write) {
				continue;
			}

			const bool watcher_edge = (open_read && read_count == 0) || (open_write && write_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				if (range_pages == 0) {
					range_begin           = page_index;
					potential_range_pages = 1;
				}
				range_pages = potential_range_pages;
			}
		}

		release_pending();
	}

	// PageManager::OpenPageWatchersForRegion (masks not both empty).
	void OpenRegionWatchers(Region& region, uint64_t base_addr, const RegionBits& read_open, const RegionBits& write_open) {
		const bool   read_any    = read_open.Any();
		const bool   write_any   = write_open.Any();
		const size_t read_first  = read_any ? read_open.FirstRange().first : REGION_PAGES;
		const size_t read_last   = read_any ? read_open.LastRange().second : 0;
		const size_t write_first = write_any ? write_open.FirstRange().first : REGION_PAGES;
		const size_t write_last  = write_any ? write_open.LastRange().second : 0;
		const size_t first       = std::min(read_first, write_first);
		const size_t last        = std::max(read_last, write_last);
		const auto*  read_mask   = read_any ? &read_open : nullptr;
		const auto*  write_mask  = write_any ? &write_open : nullptr;

		SpinGuard  lock(region.lock);
		const auto* image_read = region.image_read.data();
		OpenRuns(region.pages.data(), image_read, base_addr, first, last, read_mask, write_mask,
		         [&](size_t begin, size_t pages, Common::VirtualMemory::Mode mode) {
			         Protect(base_addr + begin * PAGE_SIZE, pages * PAGE_SIZE, mode);
		         });
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

	// Sets (arm) or clears the image read watch of the pages of the range and reprotects the runs of pages whose protection
	// changes. Disarming skips addresses without a tracker region (nothing set).
	template <bool arm>
	void UpdateImageRead(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region      = arm ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region != nullptr) {
				const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
				const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
				SpinGuard  lock(region->lock);
				uint64_t   run_begin = 0;
				uint64_t   run_bytes = 0;
				auto       run_perms = Common::VirtualMemory::Mode::NoAccess;
				const auto flush     = [&] {
					if (run_bytes != 0) {
						Protect(run_begin, run_bytes, run_perms);
						run_bytes = 0;
					}
				};
				for (size_t index = first; index < last; index++) {
					auto&          word      = region->image_read[index / 64u];
					const uint64_t bit       = uint64_t {1} << (index % 64u);
					const auto     old_perms = PermsOf(*region, index);
					if constexpr (arm) {
						word.fetch_or(bit, std::memory_order_relaxed);
					} else {
						word.fetch_and(~bit, std::memory_order_relaxed);
					}
					const auto new_perms = PermsOf(*region, index);
					const auto address   = region_base + index * PAGE_SIZE;
					if (old_perms == new_perms) {
						flush();
						continue;
					}
					if (run_bytes != 0 && run_perms == new_perms && run_begin + run_bytes == address) {
						run_bytes += PAGE_SIZE;
						continue;
					}
					flush();
					run_begin = address;
					run_bytes = PAGE_SIZE;
					run_perms = new_perms;
				}
				flush();
			}
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
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

void PageManager::OpenPageWatchersForRegion(uint64_t base_addr, const RegionBits& read_open, const RegionBits& write_open) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE || REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}
	if (read_open.None() && write_open.None()) {
		return;
	}
	auto* region = m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->OpenRegionWatchers(*region, base_addr, read_open, write_open);
}

void PageManager::ArmImageRead(uint64_t vaddr, uint64_t size) {
	m_impl->UpdateImageRead<true>(vaddr, size);
}

void PageManager::DisarmImageRead(uint64_t vaddr, uint64_t size) {
	m_impl->UpdateImageRead<false>(vaddr, size);
}

bool PageManager::ImageReadArmed(uint64_t vaddr) const noexcept {
	const auto* region = m_impl->FindRegion(vaddr);
	return region != nullptr &&
	       Impl::ImageBit(region->image_read.data(), static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE));
}

bool PageManager::ImageReadArmedInRange(uint64_t vaddr, uint64_t size) const noexcept {
	if (size == 0) {
		return false;
	}
	const uint64_t last = Common::AlignDown(vaddr + size - 1, PAGE_SIZE);
	for (uint64_t page = Common::AlignDown(vaddr, PAGE_SIZE);; page += PAGE_SIZE) {
		if (ImageReadArmed(page)) {
			return true;
		}
		if (page == last) {
			return false;
		}
	}
}

bool PageManager::ImageReadBypass(uint64_t vaddr) const noexcept {
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	const auto index = static_cast<size_t>((vaddr % REGION_SIZE) / PAGE_SIZE);
	if (!Impl::ImageBit(region->image_read.data(), index)) {
		return false;
	}
	SpinGuard lock(region->lock);
	return region->pages[index].access_watchers == 0;
}

bool PageManager::WatchersAllow(uint64_t vaddr, bool is_write) const {
	auto* region = m_impl->FindRegion(vaddr);
	if (region == nullptr) {
		return false;
	}
	SpinGuard  lock(region->lock);
	const auto perms = region->pages[(vaddr % REGION_SIZE) / PAGE_SIZE].Perms();
	return is_write ? perms == Common::VirtualMemory::Mode::ReadWrite : perms != Common::VirtualMemory::Mode::NoAccess;
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
