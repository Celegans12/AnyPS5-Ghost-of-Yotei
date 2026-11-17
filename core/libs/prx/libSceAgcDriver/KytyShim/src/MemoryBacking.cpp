#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>

#include <sys/mman.h>

namespace Libs::LibKernel::Memory {

namespace {

constexpr uint64_t PageBits = 14;
constexpr uint64_t PageSize = uint64_t {1} << PageBits;
constexpr uint64_t PageMask = PageSize - 1u;
constexpr uint64_t BucketPages = uint64_t {1} << 16u;
constexpr uint64_t BucketCount = ((kExtendedMemoryBase + kExtendedMemorySize) >> PageBits) / BucketPages;
constexpr uint32_t PrivateEntry = ~uint32_t {0};

class PageTable {
public:
	[[nodiscard]] uint32_t Load(uint64_t page) const {
		if (page / BucketPages >= BucketCount) return 0;
		const auto* entries = m_buckets[page / BucketPages].load(std::memory_order_acquire);
		return entries != nullptr ? entries[page % BucketPages].load(std::memory_order_relaxed) : 0;
	}

	[[nodiscard]] uint32_t LoadBacking(uint64_t page) const {
		const auto entry = Load(page);
		return entry == PrivateEntry ? 0 : entry;
	}

	void Store(uint64_t vaddr, uint64_t size, uint32_t first, bool step) {
		const auto end = vaddr + size;
		auto entry = first;
		for (auto page = vaddr >> PageBits; page < end >> PageBits; ++page) {
			auto& bucket = m_buckets[page / BucketPages];
			auto* entries = bucket.load(std::memory_order_acquire);
			if (entries == nullptr) {
				if (first == 0) {
					page = (page / BucketPages + 1u) * BucketPages - 1u;
					continue;
				}
				auto* created = new std::atomic<uint32_t>[BucketPages] {};
				if (bucket.compare_exchange_strong(entries, created, std::memory_order_acq_rel)) {
					entries = created;
				} else {
					delete[] created;
				}
			}
			entries[page % BucketPages].store(entry, std::memory_order_relaxed);
			if (step) ++entry;
		}
	}

	[[nodiscard]] bool AnyMapped(uint64_t vaddr, uint64_t size) const {
		for (auto page = vaddr >> PageBits; page < (vaddr + size + PageMask) >> PageBits; ++page) {
			if (page / BucketPages >= BucketCount) return false;
			if (m_buckets[page / BucketPages].load(std::memory_order_acquire) == nullptr) {
				page = (page / BucketPages + 1u) * BucketPages - 1u;
				continue;
			}
			if (Load(page) != 0) return true;
		}
		return false;
	}

	template <typename Run>
	void ForEachMappedRun(Run&& run) const {
		uint64_t start = 0;
		bool open = false;
		for (uint64_t bucket = 0; bucket < BucketCount; ++bucket) {
			const auto* entries = m_buckets[bucket].load(std::memory_order_acquire);
			for (uint64_t index = 0; index < BucketPages; ++index) {
				const bool mapped = entries != nullptr && entries[index].load(std::memory_order_relaxed) != 0;
				const auto address = (bucket * BucketPages + index) << PageBits;
				if (mapped && !open) {
					start = address;
					open = true;
				} else if (!mapped && open) {
					run(start, address - start);
					open = false;
				}
				if (entries == nullptr && !open) break;
			}
		}
		if (open) run(start, (BucketCount * BucketPages << PageBits) - start);
	}

private:
	std::array<std::atomic<std::atomic<uint32_t>*>, BucketCount> m_buckets {};
};

struct ViewState {
	uint8_t* base = nullptr;
	uint64_t guest_bytes = 0;
	uint64_t view_bytes = 0;
	std::atomic<bool> enabled {false};
};

PageTable& Pages() {
	static auto* pages = new PageTable;
	return *pages;
}

ViewState& State() {
	static auto* state = new ViewState;
	return *state;
}

std::atomic<uint64_t> g_backing_epoch {0};
constexpr uint64_t BackingChangeHistory = 4096;
std::mutex g_backing_changes_mutex;
std::array<std::pair<uint64_t, uint64_t>, BackingChangeHistory> g_backing_changes {};

void BumpBackingEpoch(uint64_t start, uint64_t size) noexcept {
	std::lock_guard lock(g_backing_changes_mutex);
	const auto epoch = g_backing_epoch.load(std::memory_order_relaxed) + 1;
	g_backing_changes[epoch % BackingChangeHistory] = {start, size};
	g_backing_epoch.store(epoch, std::memory_order_release);
}

std::recursive_mutex g_hook_mutex;
Shim::RangeHook g_map_hook = nullptr;
Shim::RangeHook g_unmap_hook = nullptr;

bool ValidRange(uint64_t vaddr, uint64_t size) {
	return size != 0 && UINT64_MAX - vaddr >= size;
}

template <typename Region, typename Gap>
void ForEachRun(uint64_t vaddr, uint64_t size, Region&& region, Gap&& gap) {
	const auto& pages = Pages();
	const auto end = vaddr + size;
	for (auto start = vaddr; start < end;) {
		const auto entry = pages.LoadBacking(start >> PageBits);
		const uint32_t step = entry != 0 ? 1 : 0;
		auto expected = entry + step;
		auto stop = (start | PageMask) + 1u;
		while (stop < end && pages.LoadBacking(stop >> PageBits) == expected) {
			stop += PageSize;
			expected += step;
		}
		stop = std::min(stop, end);
		if (entry != 0) {
			region((uint64_t {entry - 1u} << PageBits) + (start & PageMask), start, stop - start);
		} else {
			gap(start, stop - start);
		}
		start = stop;
	}
}

bool PageGuestMapped(uint64_t page_address) {
	const auto entry = Pages().Load(page_address >> PageBits);
	if (entry != 0) return true;
	try {
		return GuestRangeMapped(page_address, page_address + PageSize);
	} catch (const std::exception&) {
		return false;
	}
}

void CopyGuestPages(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto end = start + bytes;
	for (auto cursor = start; cursor < end;) {
		const auto page = cursor & ~PageMask;
		const auto stop = std::min(end, page + PageSize);
		if (PageGuestMapped(page)) {
			std::memcpy(destination + (cursor - start), reinterpret_cast<const void*>(cursor), stop - cursor);
		} else {
			std::memset(destination + (cursor - start), 0, stop - cursor);
		}
		cursor = stop;
	}
}

int HostProtection(Common::VirtualMemory::Mode mode) {
	switch (mode) {
		case Common::VirtualMemory::Mode::Read: return PROT_READ;
		case Common::VirtualMemory::Mode::Write:
		case Common::VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
		case Common::VirtualMemory::Mode::Execute: return PROT_EXEC;
		case Common::VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
		case Common::VirtualMemory::Mode::ExecuteWrite:
		case Common::VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
		case Common::VirtualMemory::Mode::NoAccess:
		default: return PROT_NONE;
	}
}

int GuestProtectionBits(Common::VirtualMemory::Mode mode) {
	const auto host = HostProtection(mode);
	return ((host & PROT_READ) != 0 ? 0x01 : 0) | ((host & PROT_WRITE) != 0 ? 0x02 : 0) | ((host & PROT_EXEC) != 0 ? 0x04 : 0);
}

void OnViewChange(std::uintptr_t address, std::size_t bytes, std::uint64_t offset) {
	if (((address | bytes) & PageMask) != 0) {
		Shim::Fail("KytyShim memory: libkernel reported a view change 0x%016" PRIx64 "+0x%zx that is not 16 KiB aligned\n", static_cast<uint64_t>(address), bytes);
	}
	std::lock_guard lock(g_hook_mutex);
	auto& pages = Pages();
	if (offset == DirectMemoryViewReplacing) {
		if (!pages.AnyMapped(address, bytes)) return;
		if (g_unmap_hook != nullptr) g_unmap_hook(address, bytes);
		pages.Store(address, bytes, 0, false);
		BumpBackingEpoch(address, bytes);
		return;
	}
	if (offset == DirectMemoryViewUnmapped) {
		const bool mapped = pages.AnyMapped(address, bytes);
		if (mapped && g_unmap_hook != nullptr) g_unmap_hook(address, bytes);
		pages.Store(address, bytes, 0, false);
		BumpBackingEpoch(address, bytes);
		return;
	}
	uint32_t first = PrivateEntry;
	bool step = false;
	if (offset != DirectMemoryViewPrivate) {
		if ((offset & PageMask) != 0 || (offset >> PageBits) + (bytes >> PageBits) >= PrivateEntry) {
			Shim::Fail("KytyShim memory: libkernel reported view offset 0x%016" PRIx64 " for 0x%016" PRIx64 "+0x%zx\n", offset, static_cast<uint64_t>(address), bytes);
		}
		first = static_cast<uint32_t>(offset >> PageBits) + 1u;
		step = true;
	}
	pages.Store(address, bytes, first, step);
	BumpBackingEpoch(address, bytes);
	if (g_map_hook != nullptr) g_map_hook(address, bytes);
}

void EnableView() {
	auto& state = State();
	uint8_t* view = nullptr;
	std::size_t guest_bytes = 0;
	std::size_t view_bytes = 0;
	try {
		if (!DirectMemoryEnableView_nid_no_patch(&OnViewChange, &view, &guest_bytes, &view_bytes)) {
			Shim::Fail("KytyShim memory: the guest arena is not available, so guest memory has no second view\n");
		}
	} catch (const std::exception& error) {
		Shim::Fail("KytyShim memory: enabling the second view of guest memory failed: %s\n", error.what());
	}
	state.base = view;
	state.guest_bytes = guest_bytes;
	state.view_bytes = view_bytes;
	state.enabled.store(true, std::memory_order_release);
}

[[maybe_unused]] const bool g_view_enabled = (EnableView(), true);

}

namespace Shim {

void Fail(const char* format, ...) {
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fflush(stderr);
	std::abort();
}

bool ViewEnabled() {
	return State().enabled.load(std::memory_order_acquire);
}

bool IsBacked(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) return false;
	const auto& pages = Pages();
	for (auto page = vaddr >> PageBits; page <= (vaddr + size - 1u) >> PageBits; ++page) {
		if (pages.LoadBacking(page) == 0) return false;
	}
	return true;
}

bool InGuestSpace(uint64_t vaddr, uint64_t size) {
	return ValidRange(vaddr, size) && ViewEnabled() && vaddr < State().guest_bytes && size <= State().guest_bytes - vaddr;
}

bool OverlapsPrivate(uint64_t vaddr, uint64_t size) {
	const auto& pages = Pages();
	const auto end = vaddr + size;
	for (auto page = vaddr & ~PageMask; page < end; page += PageSize) {
		const auto entry = pages.Load(page >> PageBits);
		if (entry == PrivateEntry) return true;
		if (entry == 0 && PageGuestMapped(page)) return true;
	}
	return false;
}

bool TryWriteBackedRuns(uint64_t vaddr, const void* data, uint64_t size) {
	if (data == nullptr || !InGuestSpace(vaddr, size) || OverlapsPrivate(vaddr, size)) return false;
	const auto* source = static_cast<const uint8_t*>(data);
	auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { std::memcpy(base + offset, source + (start - vaddr), bytes); },
	    [](uint64_t, uint64_t) {});
	return true;
}

bool TryWriteBackedQwords(uint64_t vaddr, const uint64_t* values, uint32_t count, uint64_t stride) {
	if (count == 0 || values == nullptr) return false;
	const auto size = uint64_t {count - 1u} * stride + sizeof(uint64_t);
	if (!InGuestSpace(vaddr, size) || OverlapsPrivate(vaddr, size)) return false;
	if (const auto* backing = static_cast<const uint8_t*>(FindBackingPointer(vaddr, size)); backing != nullptr) {
		auto* target = const_cast<uint8_t*>(backing);
		for (uint32_t i = count; i-- > 0;) {
			*reinterpret_cast<volatile uint64_t*>(target + i * stride) = values[i];
		}
		return true;
	}
	for (uint32_t i = count; i-- > 0;) {
		(void)TryWriteBackedRuns(vaddr + i * stride, &values[i], sizeof(uint64_t));
	}
	return true;
}

void SetGpuRangeHooks(RangeHook map, RangeHook unmap) {
	std::lock_guard lock(g_hook_mutex);
	g_map_hook = map;
	g_unmap_hook = unmap;
	if (map != nullptr) Pages().ForEachMappedRun([&](uint64_t start, uint64_t size) { map(start, size); });
}

}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (data == nullptr || !Shim::IsBacked(vaddr, size)) return false;
	auto* destination = static_cast<uint8_t*>(data);
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { std::memcpy(destination + (start - vaddr), base + offset, bytes); },
	    [&](uint64_t start, uint64_t bytes) { std::memset(destination + (start - vaddr), 0, bytes); });
	return true;
}

const void* FindBackingPointer(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size) || !Shim::ViewEnabled()) return nullptr;
	const uint8_t* result = nullptr;
	bool split = false;
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t) {
		    if (start == vaddr) {
			    result = base + offset;
		    } else {
			    split = true;
		    }
	    },
	    [&](uint64_t, uint64_t) { split = true; });
	return split ? nullptr : result;
}

bool TryGetBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
	pieces.clear();
	if (!ValidRange(vaddr, size) || !Shim::ViewEnabled()) return false;
	bool complete = true;
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t, uint64_t bytes) { pieces.emplace_back(base + offset, bytes); },
	    [&](uint64_t, uint64_t) { complete = false; });
	return complete && !pieces.empty();
}

bool TryGetSparseBackingPieces(uint64_t vaddr, uint64_t size, std::vector<std::pair<const uint8_t*, uint64_t>>& pieces) {
	if (TryGetBackingPieces(vaddr, size, pieces)) return true;
	pieces.clear();
	if (!Shim::InGuestSpace(vaddr, size) || Shim::OverlapsPrivate(vaddr, size)) return false;
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t, uint64_t bytes) { pieces.emplace_back(base + offset, bytes); },
	    [&](uint64_t, uint64_t bytes) { pieces.emplace_back(nullptr, bytes); });
	return true;
}

bool IsBackingReadable(uint64_t vaddr, uint64_t size) {
	return Shim::IsBacked(vaddr, size);
}

uint64_t BackingEpoch() noexcept {
	return g_backing_epoch.load(std::memory_order_acquire);
}

bool BackingChangesSince(uint64_t epoch, std::vector<std::pair<uint64_t, uint64_t>>& changes) {
	std::lock_guard lock(g_backing_changes_mutex);
	const auto current = g_backing_epoch.load(std::memory_order_relaxed);
	if (current - epoch > BackingChangeHistory) return false;
	for (auto change = epoch + 1; change <= current; ++change) {
		changes.push_back(g_backing_changes[change % BackingChangeHistory]);
	}
	return true;
}

static uint64_t MappedPrefix(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto cursor = vaddr;
	while (cursor < end) {
		const auto page = cursor & ~PageMask;
		if (!PageGuestMapped(page)) break;
		cursor = std::min(end, page + PageSize);
	}
	return cursor - vaddr;
}

static bool GuestMapped(uint64_t vaddr, uint64_t size) {
	try {
		return GuestRangeMapped(vaddr, vaddr + size);
	} catch (const std::exception&) {
		return false;
	}
}

bool IsFullyMapped(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) return size == 0;
	if (Shim::IsBacked(vaddr, size)) return true;
	if (!Pages().AnyMapped(vaddr, size) && GuestMapped(vaddr, size)) return true;
	return MappedPrefix(vaddr, size) == size;
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) {
		Shim::Fail("KytyShim memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	if (IsFullyMapped(vaddr, size)) return size;
	const auto clamped = MappedPrefix(vaddr, size);
	if (clamped == 0) {
		Shim::Fail("KytyShim memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	std::fprintf(stderr, "KytyShim memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64 " to 0x%016" PRIx64 "\n", vaddr, size, clamped);
	return clamped;
}

template <bool CopyPrivate>
static bool CopySparse(uint64_t vaddr, void* data, uint64_t size) {
	if (TryReadBacking(vaddr, data, size)) return true;
	if (data == nullptr || !Shim::InGuestSpace(vaddr, size)) return false;
	if constexpr (!CopyPrivate) {
		if (Shim::OverlapsPrivate(vaddr, size)) return false;
	}
	auto* destination = static_cast<uint8_t*>(data);
	const auto* base = State().base;
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { std::memcpy(destination + (start - vaddr), base + offset, bytes); },
	    [&](uint64_t start, uint64_t bytes) {
		    if constexpr (CopyPrivate) {
			    CopyGuestPages(start, destination + (start - vaddr), bytes);
		    } else {
			    std::memset(destination + (start - vaddr), 0, bytes);
		    }
	    });
	return true;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	return CopySparse<false>(vaddr, data, size);
}

bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
	return CopySparse<true>(vaddr, data, size);
}

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) {
	if (!ValidRange(vaddr, size)) return false;
	const auto protection = HostProtection(mode);
	const auto& pages = Pages();
	const auto first = vaddr & ~uint64_t {0xfff};
	const auto end = (vaddr + size + 0xfff) & ~uint64_t {0xfff};
	for (auto start = first; start < end;) {
		const bool mapped = pages.Load(start >> PageBits) != 0;
		auto stop = std::min(end, (start | PageMask) + 1u);
		while (stop < end && (pages.Load(stop >> PageBits) != 0) == mapped) {
			stop = std::min(end, stop + PageSize);
		}
		if (mapped && ::mprotect(reinterpret_cast<void*>(start), stop - start, protection) != 0) return false;
		start = stop;
	}
	return true;
}

bool IsGuestExecutable(uint64_t vaddr) {
	int protection = 0;
	return GuestProtection(static_cast<std::uintptr_t>(vaddr), &protection) && (protection & 0x04) != 0;
}

uint64_t WaitForTransientUnmap(uint64_t) noexcept {
	return 0;
}

bool RetryTransientUnmapFault(uint64_t, uint64_t) noexcept {
	return false;
}

uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size, Common::VirtualMemory::Mode mode, const char* name, bool fixed) {
	constexpr int GuestMapFixed = 0x10;
	if (size == 0 || size > UINT64_MAX - PageMask || name == nullptr || (fixed && (search_addr == 0 || (search_addr & PageMask) != 0))) return 0;
	void* address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(search_addr));
	try {
		if (DoMapAnon(&address, (size + PageMask) & ~PageMask, GuestProtectionBits(mode), fixed ? GuestMapFixed : 0, PageSize, fixed) != 0) return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "KytyShim memory: runtime memory \"%s\" of 0x%" PRIx64 " bytes not mapped: %s\n", name, size, error.what());
		return 0;
	}
	return reinterpret_cast<std::uintptr_t>(address);
}

bool FreeGuestMemory(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - PageMask) return false;
	try {
		return DoMunmap(reinterpret_cast<void*>(static_cast<std::uintptr_t>(vaddr)), (size + PageMask) & ~PageMask) == 0;
	} catch (const std::exception& error) {
		Shim::Fail("KytyShim memory: freeing runtime memory 0x%016" PRIx64 "+0x%" PRIx64 " failed: %s\n", vaddr, size, error.what());
	}
}

DebugStats GetDebugStats() {
	DebugStats stats {};
	stats.direct_total = DirectMemorySize();
	int64_t start = 0;
	int64_t end = 0;
	int type = 0;
	for (int64_t offset = 0; DirectMemoryFind(offset, true, &start, &end, &type) && end > offset; offset = end) {
		stats.direct_allocated += static_cast<uint64_t>(end - start);
	}
	const auto guest_bytes = State().guest_bytes;
	Pages().ForEachMappedRun([&](uint64_t run_start, uint64_t run_size) {
		for (auto page = run_start; page < run_start + run_size; page += PageSize) {
			const auto entry = Pages().LoadBacking(page >> PageBits);
			if (entry != 0 && (uint64_t {entry - 1u} << PageBits) >= guest_bytes) stats.direct_mapped += PageSize;
		}
	});
	size_t configured = 0;
	size_t available = 0;
	if (sceKernelConfiguredFlexibleMemorySize(&configured) == 0 && sceKernelAvailableFlexibleMemorySize(&available) == 0) {
		stats.flexible_total = configured;
		stats.flexible_used = configured - std::min(configured, available);
	}
	constexpr uint64_t TableGranularity = 2ull * 1024ull * 1024ull;
	GuestPageTableUsage(TableGranularity, &stats.cpu_page_entries, &stats.gpu_page_entries);
	return stats;
}

}
