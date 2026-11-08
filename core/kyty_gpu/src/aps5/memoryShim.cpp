// kernel/memory.h for kyty_gpu.dll: the guest-memory services Kyty's graphics uses, over AnyPS5's direct memory (the host
// table's backing view and direct-mapping lookups) instead of Kyty's own kernel. The GPU glue below (the resource walker's
// read hooks, the strict reader, ClampRangeSize, the plain reader, the page map) is Kyty's code from src/kernel/memory.cpp
// and src/kernel/memoryAddressSpace.inc of KytyPS5-Yotei v0.1.5 (GPL-2.0-or-later), kept as it is; what is new is the
// mapping table, filled from AnyPS5's mapping notifications (Aps5::MemoryMapped/MemoryUnmapped) instead of Kyty's kernel.

#include "aps5/aps5Memory.h"

#include "aps5/aps5Host.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"
#include "kernel/memoryTrace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

namespace Libs::LibKernel::Memory {

namespace VirtualMemory = Common::VirtualMemory;

static Graphics::RenderContext* g_gpu_resources = nullptr;

static Graphics::RenderContext& GetGpuResources() {
	EXIT_IF(g_gpu_resources == nullptr);
	return *g_gpu_resources;
}

static bool IsGpuAddressRange(uint64_t vaddr, uint64_t size) {
	return Graphics::GuestRange {vaddr, size}.Valid();
}

// Based on TheCruZ/KytyPS5-GTA daaf45201 (memory.cpp), GPL-2.0, as in Kyty's memory.cpp: one bit per committed 4 KiB page
// of the GPU-visible windows, so that the GPU threads clamp the buffers they bind without a lock.
class CommittedPageMap {
public:
	static constexpr uint64_t Unknown = UINT64_MAX;

	CommittedPageMap() = default;
	~CommittedPageMap() {
		for (auto& leaf: m_leaves) {
			delete[] leaf.load(std::memory_order_relaxed);
		}
	}
	KYTY_CLASS_NO_COPY(CommittedPageMap);

	void Set(uint64_t start, uint64_t size, bool committed) {
		if (size == 0) {
			return;
		}
		uint64_t first = 0;
		uint64_t end   = 0;
		if (((start | size) & PageMask) != 0 || !PageSpan(start, size, &first, &end)) {
			if (((start | size) & PageMask) != 0) {
				m_imprecise.store(true, std::memory_order_relaxed);
			}
			return;
		}
		for (auto page = first; page < end;) {
			const auto leaf_end = std::min(end, (page / LeafPages + 1) * LeafPages);
			auto*      words    = Leaf(page / LeafPages, committed);
			for (; words != nullptr && page < leaf_end;) {
				const auto bit   = page % 64u;
				const auto count = std::min<uint64_t>(64u - bit, leaf_end - page);
				const auto mask  = (count == 64u ? ~uint64_t {0} : ((uint64_t {1} << count) - 1u)) << bit;
				auto&      word  = words[(page % LeafPages) / 64u];
				if (committed) {
					word.fetch_or(mask, std::memory_order_release);
				} else {
					word.fetch_and(~mask, std::memory_order_release);
				}
				page += count;
			}
			page = leaf_end;
		}
	}

	[[nodiscard]] uint64_t CommittedRun(uint64_t address, uint64_t size) const {
		uint64_t first = 0;
		uint64_t end   = 0;
		if (m_imprecise.load(std::memory_order_relaxed) || size == 0 || size > LowerSize ||
		    !PageSpan(address & ~PageMask, ((address & PageMask) + size + PageMask) & ~PageMask, &first, &end)) {
			return Unknown;
		}
		for (auto page = first; page < end;) {
			const auto* words = m_leaves[page / LeafPages].load(std::memory_order_acquire);
			const auto  bit   = page % 64u;
			const auto  count = std::min<uint64_t>(64u - bit, end - page);
			const auto  word  = words == nullptr ? 0 : words[(page % LeafPages) / 64u].load(std::memory_order_acquire);
			const auto  missing = ~(word >> bit) & (count == 64u ? ~uint64_t {0} : ((uint64_t {1} << count) - 1u));
			if (missing != 0) {
				const auto run_end = (page + static_cast<uint64_t>(std::countr_zero(missing))) << PageBits;
				const auto base    = WindowBase(address);
				return run_end + base <= address ? 0 : run_end + base - address;
			}
			page += count;
		}
		return size;
	}

private:
	static constexpr uint64_t PageBits       = 12;
	static constexpr uint64_t PageMask       = (uint64_t {1} << PageBits) - 1u;
	static constexpr uint64_t LowerSize      = uint64_t {1} << 40u;
	static constexpr uint64_t LeafPages      = uint64_t {1} << 19u;
	static constexpr uint64_t LowerLeaves    = (LowerSize >> PageBits) / LeafPages;
	static constexpr uint64_t ExtendedLeaves = (kExtendedMemorySize >> PageBits) / LeafPages;

	static uint64_t WindowBase(uint64_t address) { return address < LowerSize ? 0 : kExtendedMemoryBase - LowerSize; }
	static bool     PageSpan(uint64_t start, uint64_t size, uint64_t* first, uint64_t* end) {
        if (size > UINT64_MAX - start) {
            return false;
        }
        const auto finish = start + size;
        if (finish <= LowerSize) {
            *first = start >> PageBits;
            *end   = finish >> PageBits;
            return true;
        }
        if (start >= kExtendedMemoryBase && finish <= kExtendedMemoryBase + kExtendedMemorySize) {
            *first = (start - kExtendedMemoryBase + LowerSize) >> PageBits;
            *end   = (finish - kExtendedMemoryBase + LowerSize) >> PageBits;
            return true;
        }
        return false;
	}

	std::atomic<uint64_t>* Leaf(uint64_t index, bool create) {
		auto* leaf = m_leaves[index].load(std::memory_order_relaxed);
		if (leaf == nullptr && create) {
			leaf = new std::atomic<uint64_t>[LeafPages / 64u] {};
			m_leaves[index].store(leaf, std::memory_order_release);
		}
		return leaf;
	}

	std::array<std::atomic<std::atomic<uint64_t>*>, LowerLeaves + ExtendedLeaves> m_leaves {};
	std::atomic_bool                                                             m_imprecise {false};
};

// The guest's committed ranges as AnyPS5 reported them (Aps5::MemoryMapped/MemoryUnmapped), and for the ones that are
// direct memory their offset in AnyPS5's backing view (one section for all direct memory, the host table's
// backing_view). This takes the place of Kyty's VirtualRanges (the committed part) and GuestBackingStore (the backing
// part). A range AnyPS5 mapped before the backend existed is found by a lookup on first use (Discover).
class GuestMap {
public:
	struct Entry {
		uint64_t vaddr          = 0;
		uint64_t size           = 0;
		uint64_t backing_offset = 0;
		bool     direct         = false;
	};

	struct CompareSpan {
		uint64_t       vaddr    = 0;
		const uint8_t* expected = nullptr;
		uint64_t       size     = 0;
	};
	enum class CompareResult { Equal, Different, Unknown };

	void SetBacking(uint8_t* base, uint64_t size) {
		m_backing_base = base;
		m_backing_size = size;
	}
	[[nodiscard]] uint8_t* BackingBase() const noexcept { return m_backing_base; }
	[[nodiscard]] uint64_t BackingSize() const noexcept { return m_backing_size; }
	[[nodiscard]] bool     IsAvailable() const noexcept { return m_backing_base != nullptr; }

	// AnyPS5 may hold its direct-memory table lock here, so the range comes with its offset in the backing view
	// (direct memory) or without one (private memory); nothing is looked up.
	void Map(uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return;
		}
		const bool direct = backing_offset != APS5_GPU_NOT_DIRECT && RangeOk(backing_offset, size);
		if (backing_offset != APS5_GPU_NOT_DIRECT && !direct) {
			Aps5::Log("memory: direct mapping 0x%llx+0x%llx at backing offset 0x%llx lies outside the backing view",
			          static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			          static_cast<unsigned long long>(backing_offset));
		}
		std::unique_lock lock(m_mutex);
		RemoveUnlocked(vaddr, size);
		m_entries.emplace(vaddr, Entry {vaddr, size, direct ? backing_offset : 0, direct});
		m_committed.Set(vaddr, size, true);
		m_generation.fetch_add(1, std::memory_order_release);
	}

	void Unmap(uint64_t vaddr, uint64_t size) {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return;
		}
		std::unique_lock lock(m_mutex);
		RemoveUnlocked(vaddr, size);
		m_committed.Set(vaddr, size, false);
		m_generation.fetch_add(1, std::memory_order_release);
	}

	[[nodiscard]] uint64_t Generation() const noexcept { return m_generation.load(std::memory_order_acquire); }

	// The part of [vaddr, vaddr + size) covered by contiguous committed ranges, 0 when vaddr is not committed.
	uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
		if (vaddr == 0 || size == 0 || size > UINT64_MAX - vaddr) {
			return 0;
		}
		const auto run = m_committed.CommittedRun(vaddr, size);
		if (run != CommittedPageMap::Unknown && run == size) {
			return run;
		}
		uint64_t generation = 0;
		return ClampRangeSizeAt(vaddr, size, generation);
	}

	uint64_t ClampRangeSizeAt(uint64_t vaddr, uint64_t size, uint64_t& generation) {
		if (vaddr == 0 || size == 0 || size > UINT64_MAX - vaddr) {
			return 0;
		}
		for (int attempt = 0; attempt < 2; attempt++) {
			{
				std::shared_lock lock(m_mutex);
				generation = m_generation.load(std::memory_order_relaxed);
				auto it    = m_entries.upper_bound(vaddr);
				if (it != m_entries.begin()) {
					--it;
					if (vaddr < it->second.vaddr + it->second.size) {
						uint64_t clamped  = std::min(size, it->second.vaddr + it->second.size - vaddr);
						uint64_t expected = vaddr + clamped;
						for (++it; it != m_entries.end() && it->second.vaddr == expected && clamped < size; ++it) {
							const auto chunk = std::min(size - clamped, it->second.size);
							clamped += chunk;
							expected += chunk;
						}
						return clamped;
					}
				}
			}
			if (attempt != 0 || !Discover(vaddr)) {
				break;
			}
		}
		return 0;
	}

	// The backing offset of [vaddr, vaddr + size) when one direct mapping holds all of it.
	bool BackingOffset(uint64_t vaddr, uint64_t size, uint64_t* offset) {
		if (!IsAvailable() || offset == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		Entry entry {};
		if (!FindDirect(vaddr, size, &entry)) {
			return false;
		}
		*offset = entry.backing_offset + (vaddr - entry.vaddr);
		return RangeOk(*offset, size);
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		uint64_t offset = 0;
		if (TryCachedOffset(vaddr, size, &offset)) {
			std::memcpy(data, m_backing_base + offset, size);
			return true;
		}
		for (int attempt = 0; attempt < 2; attempt++) {
			{
				std::shared_lock lock(m_mutex);
				if (SlotOffsetUnlocked(vaddr, size, &offset)) {
					std::memcpy(data, m_backing_base + offset, size);
					return true;
				}
				if (CoveredByDirectUnlocked(vaddr, size)) {
					ForEachDirectUnlocked(vaddr, size, [&](const Entry& entry, uint64_t start, uint64_t bytes) {
						std::memcpy(static_cast<uint8_t*>(data) + (start - vaddr),
						            m_backing_base + entry.backing_offset + (start - entry.vaddr), bytes);
					});
					return true;
				}
			}
			if (attempt != 0 || !Discover(vaddr)) {
				break;
			}
		}
		return false;
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		uint64_t offset = 0;
		if (TryCachedOffset(vaddr, size, &offset)) {
			std::memcpy(m_backing_base + offset, data, size);
			return true;
		}
		for (int attempt = 0; attempt < 2; attempt++) {
			{
				std::shared_lock lock(m_mutex);
				if (SlotOffsetUnlocked(vaddr, size, &offset)) {
					std::memcpy(m_backing_base + offset, data, size);
					return true;
				}
				if (CoveredByDirectUnlocked(vaddr, size)) {
					ForEachDirectUnlocked(vaddr, size, [&](const Entry& entry, uint64_t start, uint64_t bytes) {
						std::memcpy(m_backing_base + entry.backing_offset + (start - entry.vaddr),
						            static_cast<const uint8_t*>(data) + (start - vaddr), bytes);
					});
					return true;
				}
			}
			if (attempt != 0 || !Discover(vaddr)) {
				break;
			}
		}
		return false;
	}

	// As Kyty's GuestBackingStore::CompareBackingBatch: one shared lock for the whole batch.
	CompareResult CompareBackingBatch(const CompareSpan* spans, size_t count, size_t* first_bad) {
		*first_bad = 0;
		if (count == 0) {
			return CompareResult::Equal;
		}
		if (!IsAvailable()) {
			return CompareResult::Unknown;
		}
		std::shared_lock lock(m_mutex);
		Entry            mapping {};
		bool             have_mapping = false;
		for (size_t i = 0; i < count; ++i) {
			const auto& span = spans[i];
			*first_bad       = i;
			if (span.expected == nullptr || span.size == 0 || UINT64_MAX - span.vaddr < span.size) {
				return CompareResult::Unknown;
			}
			const auto end = span.vaddr + span.size;
			if (!have_mapping || span.vaddr < mapping.vaddr || span.vaddr - mapping.vaddr >= mapping.size) {
				have_mapping = FindSlotUnlocked(span.vaddr, &mapping) && mapping.direct;
				if (!have_mapping) {
					return CompareResult::Unknown;
				}
			}
			if (end > mapping.vaddr + mapping.size) {
				return CompareResult::Unknown;
			}
			const auto offset = mapping.backing_offset + span.vaddr - mapping.vaddr;
			if (!RangeOk(offset, span.size)) {
				return CompareResult::Unknown;
			}
			if (std::memcmp(m_backing_base + offset, span.expected, static_cast<size_t>(span.size)) != 0) {
				return CompareResult::Different;
			}
		}
		return CompareResult::Equal;
	}

	// Kyty's CopySparseMemory: the backed bytes of the range, gaps zero; false when a committed private (not direct) range
	// overlaps it (copy_private false) or the range has no committed byte at all.
	template <bool copy_private>
	bool CopySparse(uint64_t vaddr, void* data, uint64_t size) {
		if (!IsAvailable() || data == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		std::shared_lock lock(m_mutex);
		const auto       end    = vaddr + size;
		auto*            out    = static_cast<uint8_t*>(data);
		auto             cursor = vaddr;
		auto             it     = m_entries.upper_bound(vaddr);
		if (it != m_entries.begin() && std::prev(it)->second.vaddr + std::prev(it)->second.size > vaddr) {
			--it;
		}
		for (; it != m_entries.end() && it->second.vaddr < end; ++it) {
			const auto& entry = it->second;
			const auto  start = std::max(vaddr, entry.vaddr);
			const auto  bytes = std::min(end, entry.vaddr + entry.size) - start;
			if (cursor < start) {
				std::memset(out + (cursor - vaddr), 0, start - cursor);
			}
			if (entry.direct) {
				std::memcpy(out + (start - vaddr), m_backing_base + entry.backing_offset + (start - entry.vaddr), bytes);
			} else if (copy_private) {
				std::memcpy(out + (start - vaddr), reinterpret_cast<const void*>(start), bytes);
			} else {
				return false;
			}
			cursor = start + bytes;
		}
		if (cursor < end) {
			std::memset(out + (cursor - vaddr), 0, end - cursor);
		}
		return true;
	}

	// The committed pieces of a range, for the tracker's protections (Kyty protects mapped regions only).
	template <typename Func>
	void ForEachCommitted(uint64_t vaddr, uint64_t size, Func&& func) {
		std::shared_lock lock(m_mutex);
		const auto       end = vaddr + size;
		auto             it  = m_entries.upper_bound(vaddr);
		if (it != m_entries.begin() && std::prev(it)->second.vaddr + std::prev(it)->second.size > vaddr) {
			--it;
		}
		for (; it != m_entries.end() && it->second.vaddr < end; ++it) {
			const auto start = std::max(vaddr, it->second.vaddr);
			const auto stop  = std::min(end, it->second.vaddr + it->second.size);
			if (start < stop) {
				func(start, stop - start);
			}
		}
	}

	// The committed ranges in address order, adjacent ones joined.
	void ForEachRun(void (*callback)(void* context, uint64_t address, uint64_t bytes), void* context) {
		std::vector<std::pair<uint64_t, uint64_t>> runs;
		{
			std::shared_lock lock(m_mutex);
			for (const auto& [start, entry]: m_entries) {
				(void)start;
				if (!runs.empty() && runs.back().first + runs.back().second == entry.vaddr) {
					runs.back().second += entry.size;
				} else {
					runs.emplace_back(entry.vaddr, entry.size);
				}
			}
		}
		for (const auto& [address, bytes]: runs) {
			callback(context, address, bytes);
		}
	}

	struct Stats {
		uint64_t entries   = 0;
		uint64_t direct    = 0;
		uint64_t private_  = 0;
		uint64_t discovered = 0;
	};
	Stats GetStats() {
		std::shared_lock lock(m_mutex);
		Stats            stats;
		stats.entries    = m_entries.size();
		stats.discovered = m_discovered.load(std::memory_order_relaxed);
		for (const auto& [start, entry]: m_entries) {
			(void)start;
			(entry.direct ? stats.direct : stats.private_) += entry.size;
		}
		return stats;
	}

private:
	[[nodiscard]] bool RangeOk(uint64_t backing_offset, uint64_t size) const {
		return size != 0 && backing_offset < m_backing_size && size <= m_backing_size - backing_offset;
	}

	// A mapping AnyPS5 made before the backend existed (or one it never reported): a direct mapping the host knows is
	// added on first use. Private memory cannot be found this way (no lookup for it). The host's lookup only tries its
	// direct-memory lock (a mapping change may hold it while it waits for the GPU thread): a busy lookup is tried again a
	// few times, then the access fails as for unmapped memory.
	bool Discover(uint64_t vaddr) {
		if (!Aps5::HasHost()) {
			return false;
		}
		const auto& host   = Aps5::Host();
		uint64_t    start  = 0;
		uint64_t    stop   = 0;
		uint64_t    offset = 0;
		int32_t     type   = 0;
		bool        found  = false;
		for (int attempt = 0; attempt < 8 && !found; attempt++) {
			if (attempt != 0) {
				SwitchToThread();
			}
			found = host.direct_mapping(host.context, vaddr, &start, &stop, &offset, &type) == APS5_GPU_OK;
		}
		if (!found || stop <= start || vaddr < start || vaddr >= stop) {
			return false;
		}
		{
			std::unique_lock lock(m_mutex);
			auto             it = m_entries.upper_bound(vaddr);
			if (it != m_entries.begin() && vaddr < std::prev(it)->second.vaddr + std::prev(it)->second.size) {
				return false;
			}
			RemoveUnlocked(start, stop - start);
			m_entries.emplace(start, Entry {start, stop - start, offset, true});
			m_committed.Set(start, stop - start, true);
			m_generation.fetch_add(1, std::memory_order_release);
		}
		if (m_discovered.fetch_add(1, std::memory_order_relaxed) < 16) {
			Aps5::Log("memory: direct mapping 0x%llx-0x%llx (physical 0x%llx, type %d) found by lookup, not by a mapping "
			          "notification",
			          static_cast<unsigned long long>(start), static_cast<unsigned long long>(stop),
			          static_cast<unsigned long long>(offset), type);
		}
		return true;
	}

	void RemoveUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end = vaddr + size;
		auto       it  = m_entries.upper_bound(vaddr);
		if (it != m_entries.begin()) {
			--it;
		}
		while (it != m_entries.end() && it->first < end) {
			const auto old     = it->second;
			const auto old_end = old.vaddr + old.size;
			if (old_end <= vaddr) {
				++it;
				continue;
			}
			it = m_entries.erase(it);
			if (old.vaddr < vaddr) {
				m_entries.emplace(old.vaddr, Entry {old.vaddr, vaddr - old.vaddr, old.backing_offset, old.direct});
			}
			if (end < old_end) {
				it = m_entries.emplace(end, Entry {end, old_end - end, old.backing_offset + (end - old.vaddr), old.direct}).first;
				++it;
			}
		}
	}

	bool FindContainingUnlocked(uint64_t vaddr, Entry* out) const {
		auto next = m_entries.upper_bound(vaddr);
		if (next == m_entries.begin()) {
			return false;
		}
		const auto& entry = std::prev(next)->second;
		if (vaddr < entry.vaddr || vaddr - entry.vaddr >= entry.size) {
			return false;
		}
		*out = entry;
		return true;
	}

	bool FindDirect(uint64_t vaddr, uint64_t size, Entry* out) {
		for (int attempt = 0; attempt < 2; attempt++) {
			{
				std::shared_lock lock(m_mutex);
				if (FindSlotUnlocked(vaddr, out) && out->direct && vaddr + size <= out->vaddr + out->size) {
					return true;
				}
			}
			if (attempt != 0 || !Discover(vaddr)) {
				break;
			}
		}
		return false;
	}

	[[nodiscard]] bool CoveredByDirectUnlocked(uint64_t vaddr, uint64_t size) const {
		const auto end    = vaddr + size;
		auto       cursor = vaddr;
		while (cursor < end) {
			Entry entry {};
			if (!FindContainingUnlocked(cursor, &entry) || !entry.direct ||
			    !RangeOk(entry.backing_offset + (cursor - entry.vaddr), std::min(end, entry.vaddr + entry.size) - cursor)) {
				return false;
			}
			cursor = std::min(end, entry.vaddr + entry.size);
		}
		return true;
	}

	template <typename Func>
	void ForEachDirectUnlocked(uint64_t vaddr, uint64_t size, Func&& func) const {
		const auto end    = vaddr + size;
		auto       cursor = vaddr;
		while (cursor < end) {
			Entry entry {};
			EXIT_IF(!FindContainingUnlocked(cursor, &entry));
			const auto bytes = std::min(end, entry.vaddr + entry.size) - cursor;
			func(entry, cursor, bytes);
			cursor += bytes;
		}
	}

	// Kyty's per-thread memo of the mapping found last per 64 KiB of guest addresses (TheCruZ's fork 2a1dc4486 and
	// c88eadcad, GPL-2.0, as in memoryAddressSpace.inc), tagged with the generation every change moves.
	struct BackingSlot {
		const GuestMap* owner      = nullptr;
		uint64_t        generation = 0;
		Entry           entry {};
	};
	static constexpr uint32_t BackingSlotCount = 1024;

	static BackingSlot& SlotFor(uint64_t vaddr) {
		thread_local std::unique_ptr<std::array<BackingSlot, BackingSlotCount>> slots;
		if (slots == nullptr) {
			slots = std::make_unique<std::array<BackingSlot, BackingSlotCount>>();
		}
		return (*slots)[(vaddr >> 16u) & (BackingSlotCount - 1u)];
	}

	bool FindSlotUnlocked(uint64_t vaddr, Entry* out) {
		BackingSlot& slot       = SlotFor(vaddr);
		const auto   generation = m_generation.load(std::memory_order_relaxed);
		if (slot.owner == this && slot.generation == generation && vaddr >= slot.entry.vaddr &&
		    vaddr - slot.entry.vaddr < slot.entry.size) {
			*out = slot.entry;
			return true;
		}
		const bool found = FindContainingUnlocked(vaddr, out);
		if (found) {
			slot = BackingSlot {this, generation, *out};
		}
		return found;
	}

	bool TryCachedOffset(uint64_t vaddr, uint64_t size, uint64_t* offset) {
		const BackingSlot& slot = SlotFor(vaddr);
		if (slot.owner != this || !slot.entry.direct || slot.generation != m_generation.load(std::memory_order_acquire) ||
		    vaddr < slot.entry.vaddr || vaddr - slot.entry.vaddr >= slot.entry.size ||
		    vaddr + size > slot.entry.vaddr + slot.entry.size) {
			return false;
		}
		*offset = slot.entry.backing_offset + (vaddr - slot.entry.vaddr);
		return RangeOk(*offset, size);
	}

	bool SlotOffsetUnlocked(uint64_t vaddr, uint64_t size, uint64_t* offset) {
		Entry entry {};
		if (!FindSlotUnlocked(vaddr, &entry) || !entry.direct || vaddr + size > entry.vaddr + entry.size) {
			return false;
		}
		*offset = entry.backing_offset + (vaddr - entry.vaddr);
		return RangeOk(*offset, size);
	}

	uint8_t*                  m_backing_base = nullptr;
	uint64_t                  m_backing_size = 0;
	std::shared_mutex         m_mutex;
	std::map<uint64_t, Entry> m_entries;
	std::atomic<uint64_t>     m_generation {uint64_t {1} << 40u};
	std::atomic<uint64_t>     m_discovered {0};
	CommittedPageMap          m_committed;
};

static GuestMap& Map() {
	static auto* map = new GuestMap;
	return *map;
}

using GuestAddressSpace = GuestMap;

// M109 crash report (Kyty's memory.cpp): the last guest unmaps, so an invalid-address exit can tell a range the game
// unmapped moments ago from one that was never mapped.
struct RecentUnmap {
	uint64_t vaddr   = 0;
	uint64_t size    = 0;
	uint64_t time_us = 0;
	uint32_t frame   = 0;
};
static std::mutex                    g_recent_unmaps_mutex;
static std::array<RecentUnmap, 8192> g_recent_unmaps;
static uint64_t                      g_recent_unmaps_count = 0;

static uint64_t SteadyMicroseconds() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

static void NoteRecentUnmap(uint64_t vaddr, uint64_t size) {
	std::scoped_lock lock(g_recent_unmaps_mutex);
	g_recent_unmaps[g_recent_unmaps_count++ % g_recent_unmaps.size()] = {vaddr, size, SteadyMicroseconds(),
	                                                                      Graphics::DebugPresentCount().load()};
}

static void ReportRecentUnmaps(uint64_t vaddr, uint64_t size) {
	std::scoped_lock lock(g_recent_unmaps_mutex);
	const auto       now   = SteadyMicroseconds();
	const auto       first = g_recent_unmaps_count > g_recent_unmaps.size() ? g_recent_unmaps_count - g_recent_unmaps.size() : 0;
	uint32_t         found = 0;
	for (auto i = g_recent_unmaps_count; i > first && found < 8u; i--) {
		const auto& unmap = g_recent_unmaps[(i - 1) % g_recent_unmaps.size()];
		if (unmap.vaddr >= vaddr + size || vaddr >= unmap.vaddr + unmap.size) continue;
		found++;
		std::fprintf(stderr, "INVALID_ADDRESS_RECENT_UNMAP range=0x%016llx+0x%llx %.1f ms ago at frame %u (now %u)\n",
		             static_cast<unsigned long long>(unmap.vaddr), static_cast<unsigned long long>(unmap.size),
		             static_cast<double>(now - unmap.time_us) / 1000.0, unmap.frame, Graphics::DebugPresentCount().load());
	}
	if (found == 0)
		std::fprintf(stderr, "INVALID_ADDRESS_RECENT_UNMAP none of the last %llu unmaps overlaps 0x%016llx+0x%llx\n",
		             static_cast<unsigned long long>(g_recent_unmaps_count - first), static_cast<unsigned long long>(vaddr),
		             static_cast<unsigned long long>(size));
}

static thread_local const ClampContext* t_clamp_context = nullptr;

ClampContextScope::ClampContextScope(const ClampContext& context): m_previous(t_clamp_context) {
	t_clamp_context = &context;
}

ClampContextScope::~ClampContextScope() {
	t_clamp_context = m_previous;
}

bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
	return Map().TryWriteBacking(vaddr, data, size);
}

std::vector<std::pair<uint64_t, uint64_t>> FindDirectMemoryAliases(uint64_t /*vaddr*/, uint64_t /*size*/) {
	return {};
}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	return Map().TryReadBacking(vaddr, data, size);
}

int CompareBacking(uint64_t vaddr, const void* expected, uint64_t size) {
	const GuestMap::CompareSpan span {vaddr, static_cast<const uint8_t*>(expected), size};
	size_t                      bad     = 0;
	const auto                  outcome = Map().CompareBackingBatch(&span, 1, &bad);
	return outcome == GuestMap::CompareResult::Equal ? 0 : outcome == GuestMap::CompareResult::Different ? 1 : 2;
}

uint64_t GuestBackingBase() {
	return reinterpret_cast<uint64_t>(Map().BackingBase());
}

uint64_t GuestBackingSize() {
	return Map().BackingSize();
}

bool GuestBackingOffset(uint64_t vaddr, uint64_t size, uint64_t* offset) {
	return Map().BackingOffset(vaddr, size, offset);
}

// From here to InstallGpuResources: Kyty's memory.cpp (v0.1.5), unchanged except that GuestAddressSpace is the map above.

struct OrdinaryReadPageCache {
	static constexpr size_t kSlots = 8;
	uint32_t                depth  = 0;
	bool                    active = false;
	uint64_t                pages[kSlots] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
};

static thread_local OrdinaryReadPageCache t_ordinary_read_pages;
static_assert(Graphics::TRACKER_PAGE_SIZE == 4096u, "OrdinaryReadBufferBacking caches per 4 KiB tracker page");

static bool SrtPageCacheVerify() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FIX_SRT_TAPE_VERIFY");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

static void SrtReadSession(bool begin) {
	auto& cache = t_ordinary_read_pages;
	if (!begin) {
		if (cache.depth != 0u && --cache.depth == 0u) cache.active = false;
		return;
	}
	++cache.depth;
	std::fill(std::begin(cache.pages), std::end(cache.pages), ~0ull);
	cache.active = g_gpu_resources != nullptr && Graphics::GuestGpu::IsGpuThread();
}

static thread_local bool t_srt_classes_nested = false;

static bool OrdinaryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size) {
	auto&          cache    = t_ordinary_read_pages;
	const uint64_t page     = vaddr >> 12u;
	const bool     one_page = size != 0u && ((vaddr + size - 1u) >> 12u) == page;
	const bool     classes  = Graphics::ShaderRecompiler::IR::SrtClassTrace() && Graphics::GuestGpu::IsGpuThread();
	if (classes) SrtClassesNoteRead(vaddr, size);
	if (cache.active && one_page && cache.pages[page % OrdinaryReadPageCache::kSlots] == page) {
		if (SrtPageCacheVerify()) {
			static std::atomic_uint hits {0};
			static std::atomic_uint stale {0};
			const auto              count = hits.fetch_add(1) + 1u;
			if (g_gpu_resources != nullptr && GetGpuResources().GetBufferCache().IsRegionGpuModified(vaddr, size) &&
			    stale.fetch_add(1) < 16u) {
				std::fprintf(stderr, "SRT_READ_PAGE_CACHE_STALE addr=0x%llx size=%llu\n",
				             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
			}
			if ((count & 0xfffffu) == 0u) {
				std::fprintf(stderr, "SRT_READ_PAGE_CACHE hits=%u stale=%u\n", count, stale.load());
			}
		}
		return false;
	}
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) && Graphics::GuestGpu::IsGpuThread()) {
		auto&          buffers    = GetGpuResources().GetBufferCache();
		const uint64_t wait_start = classes ? SrtClassesNowNs() : 0;
		const bool     overlay    = buffers.WaitPendingReadbacks(vaddr, size);
		if (classes) {
			SrtClassesWait("ordinary", Graphics::ShaderRecompiler::IR::t_srt_read_class, vaddr, SrtClassesNowNs() - wait_start);
		}
		if (!buffers.IsRegionGpuModified(vaddr, size)) {
			if (overlay) {
				ReadGuestPlain(data, vaddr, size);
				buffers.ApplyDirectCpWrites(vaddr, data, size);
				return true;
			}
			if (cache.active && one_page && IsGpuAddressRange(page << 12u, 4096u)) {
				cache.pages[page % OrdinaryReadPageCache::kSlots] = page;
			}
			return false;
		}
	}
	t_srt_classes_nested = classes;
	const bool ok        = TryReadBufferBacking(vaddr, data, size);
	t_srt_classes_nested = false;
	return ok;
}

bool TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size) {
	bool overlay = false;
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		if (!Graphics::GuestGpu::IsGpuThread()) {
			return false;
		}
		auto& buffers = GetGpuResources().GetBufferCache();
		namespace Ir         = Graphics::ShaderRecompiler::IR;
		const bool classes   = Ir::SrtClassTrace();
		if (classes && !t_srt_classes_nested) SrtClassesNoteRead(vaddr, size);
		uint64_t wait_start  = classes ? SrtClassesNowNs() : 0;
		overlay = buffers.WaitPendingReadbacks(vaddr, size);
		if (classes) {
			const uint64_t now = SrtClassesNowNs();
			SrtClassesWait("strict", Ir::t_srt_read_class, vaddr, now - wait_start);
			wait_start = now;
		}
		const bool gpu_modified = buffers.IsRegionGpuModified(vaddr, size);
		const bool gpu_dirty    = gpu_modified && buffers.HasGpuDirtyBytes(vaddr, size);
		if (gpu_dirty) {
			KYTY_PROFILER_BLOCK("SrtWalker::Read");
			Graphics::PerfReasonScope reason(Graphics::PerfReason::SrtWalker);
			buffers.ReadMemory(vaddr, size);
			if (classes) SrtClassesWait("download", Ir::t_srt_read_class, vaddr, SrtClassesNowNs() - wait_start);
		}
	}
	const bool ok = TryReadBacking(vaddr, data, size);
	if (ok && overlay) GetGpuResources().GetBufferCache().ApplyDirectCpWrites(vaddr, data, size);
	return ok;
}

static Graphics::ShaderRecompiler::IR::AheadReadStatus AheadReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	using Graphics::ShaderRecompiler::IR::AheadReadStatus;
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		auto& buffers = GetGpuResources().GetBufferCache();
		if (Graphics::GuestGpu::IsGpuThread()) {
			if (buffers.HasPendingReadback(vaddr, size) ||
			    (buffers.IsRegionGpuModified(vaddr, size) && buffers.HasGpuDirtyBytes(vaddr, size))) {
				return AheadReadStatus::GpuDirty;
			}
		} else if (buffers.IsRegionGpuModified(vaddr, size) && buffers.HasGpuDirtyBytesShared(vaddr, size)) {
			return AheadReadStatus::GpuDirty;
		}
	}
	return TryReadBacking(vaddr, data, size) ? AheadReadStatus::Ok : AheadReadStatus::Backing;
}

static Graphics::ShaderRecompiler::IR::FastVerifyResult FastVerifyLoggedReads(
    const Graphics::ShaderRecompiler::IR::ReadLog& log, size_t begin, size_t& stop,
    Graphics::ShaderRecompiler::IR::FastAdoptStats& stats) {
	namespace Ir = Graphics::ShaderRecompiler::IR;
	using Result = Ir::FastVerifyResult;
	const size_t count = log.entries.size();
	auto&        cache = t_ordinary_read_pages;
	stop               = begin;
	if (begin >= count) {
		stop = count;
		return Result::Unchanged;
	}
	if (g_gpu_resources == nullptr || !Map().IsAvailable() || !Graphics::GuestGpu::IsGpuThread()) {
		return Result::NeedsReader;
	}
	uint64_t  local_pages[OrdinaryReadPageCache::kSlots] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
	uint64_t* pages = cache.active ? cache.pages : local_pages;
	auto&          buffers    = GetGpuResources().GetBufferCache();
	const uint64_t byte_count = log.bytes.size();
	constexpr size_t kMaxPending = 32;
	GuestMap::CompareSpan spans[kMaxPending];
	size_t                span_entry[kMaxPending];
	size_t                span_entries[kMaxPending];
	size_t                pending = 0;

	const auto flush = [&]() -> Result {
		if (pending == 0) {
			return Result::Unchanged;
		}
		size_t     bad     = 0;
		const auto outcome = Map().CompareBackingBatch(spans, pending, &bad);
		size_t     proven  = pending;
		if (outcome != GuestMap::CompareResult::Equal) {
			proven = bad;
		}
		for (size_t i = 0; i < proven; ++i) {
			stats.entries_fast += span_entries[i];
			++stats.runs;
		}
		pending = 0;
		if (outcome == GuestMap::CompareResult::Equal) {
			return Result::Unchanged;
		}
		stop = span_entry[bad];
		return outcome == GuestMap::CompareResult::Different ? Result::Changed : Result::NeedsReader;
	};

	const auto plain = [&](const Ir::ReadLog::Entry& e) {
		return e.ok && e.size != 0 && e.size <= Ir::ReadLog::MaxCallBytes &&
		       (e.kind == Ir::ReadLog::Kind::Ordinary || e.size % sizeof(uint32_t) == 0u) && e.first <= byte_count &&
		       e.size <= byte_count - e.first;
	};

	uint64_t last_page = ~0ull;
	size_t   index     = begin;
	while (index < count) {
		const auto& head = log.entries[index];
		if (!plain(head)) {
			const auto flushed = flush();
			if (flushed != Result::Unchanged) {
				return flushed;
			}
			stop = index;
			return Result::NeedsReader;
		}
		uint64_t run_size = head.size;
		size_t   next     = index + 1;
		while (next < count) {
			const auto& entry = log.entries[next];
			if (entry.kind != head.kind || !plain(entry) || entry.address != head.address + run_size ||
			    entry.first != head.first + run_size) {
				break;
			}
			run_size += entry.size;
			++next;
		}
		uint8_t run_class = 0;
		if (Ir::SrtClassTrace()) {
			for (size_t i = index; i < next; ++i) run_class |= log.entries[i].cls;
		}
		bool clean = IsGpuAddressRange(head.address, run_size);
		if (clean) {
			const uint64_t last = (head.address + run_size - 1u) >> 12u;
			for (uint64_t page = head.address >> 12u; page <= last; ++page) {
				if (page == last_page) {
					continue;
				}
				auto& slot = pages[page % OrdinaryReadPageCache::kSlots];
				if (slot != page) {
					const uint64_t base = page << 12u;
					if (!IsGpuAddressRange(base, 4096u)) {
						clean = false;
						break;
					}
					const uint64_t wait_start = Ir::SrtClassTrace() ? SrtClassesNowNs() : 0;
					const bool     page_busy  = buffers.WaitPendingReadbacks(base, 4096u);
					if (Ir::SrtClassTrace()) SrtClassesWait("verify", run_class, base, SrtClassesNowNs() - wait_start);
					if (page_busy || buffers.IsRegionGpuModified(base, 4096u)) {
						clean = false;
						break;
					}
					slot = page;
				}
				last_page = page;
			}
		}
		if (!clean) {
			const auto flushed = flush();
			if (flushed != Result::Unchanged) {
				return flushed;
			}
			stop = index;
			return Result::NeedsReader;
		}
		const uint8_t* const expected = log.bytes.data() + head.first;
		if (head.kind == Ir::ReadLog::Kind::Ordinary) {
			bool same = true;
			if (GuestImageReadArmed(head.address, run_size)) {
				thread_local std::vector<uint8_t> watched;
				watched.resize(run_size);
				ReadGuestPlain(watched.data(), head.address, run_size);
				same = std::memcmp(watched.data(), expected, static_cast<size_t>(run_size)) == 0;
			} else {
				same = std::memcmp(reinterpret_cast<const void*>(head.address), expected, static_cast<size_t>(run_size)) == 0;
			}
			if (!same) {
				stop = index;
				return Result::Changed;
			}
			stats.entries_fast += next - index;
			++stats.runs;
		} else {
			spans[pending]        = {head.address, expected, run_size};
			span_entry[pending]   = index;
			span_entries[pending] = next - index;
			if (++pending == kMaxPending) {
				const auto flushed = flush();
				if (flushed != Result::Unchanged) {
					return flushed;
				}
			}
		}
		index = next;
	}
	const auto flushed = flush();
	if (flushed != Result::Unchanged) {
		return flushed;
	}
	stop = count;
	return Result::Unchanged;
}

uint64_t TryClampReservedRangeSize(uint64_t vaddr, uint64_t size) {
	return Map().ClampRangeSize(vaddr, size);
}

uint64_t ClampRangeSizeAhead(uint64_t vaddr, uint64_t size, uint64_t* generation) {
	return Map().ClampRangeSizeAt(vaddr, size, *generation);
}

uint64_t RangesGeneration() {
	return Map().Generation();
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	const auto clamped_size = Map().ClampRangeSize(vaddr, size);
	if (clamped_size == 0) {
		if (const auto* context = t_clamp_context; context != nullptr) {
			const auto* v = context->vsharp;
			std::fprintf(stderr,
			             "INVALID_ADDRESS_BINDING shader=0x%016llx resource=%u written=%d vsharp=%08x %08x %08x %08x frame=%u\n",
			             static_cast<unsigned long long>(context->shader_hash), context->resource, context->written ? 1 : 0,
			             v != nullptr ? v[0] : 0u, v != nullptr ? v[1] : 0u, v != nullptr ? v[2] : 0u,
			             v != nullptr ? v[3] : 0u, Graphics::DebugPresentCount().load());
		}
		ReportRecentUnmaps(vaddr, size);
		EXIT("Memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	if (clamped_size != size) {
		LOGF("Memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64 " to 0x%016" PRIx64 "\n", vaddr, size,
		     clamped_size);
	}
	return clamped_size;
}

void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept {
	if (!TryWriteBacking(vaddr, data, size)) {
		EXIT("Memory: required direct-backing write failed, addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n", vaddr, size);
	}
}

void InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	(void)GetGpuResources().InvalidateMemory(vaddr, size);
}

void InstallGpuResources(Graphics::RenderContext* resources) noexcept {
	EXIT_IF(resources != nullptr && g_gpu_resources != nullptr);
	g_gpu_resources = resources;
	Graphics::ShaderRecompiler::IR::SetAheadReadHook(resources != nullptr ? &AheadReadBacking : nullptr);
	Graphics::ShaderRecompiler::IR::SetPlainReadHook(resources != nullptr ? &ReadGuestPlain : nullptr);
	Graphics::ShaderRecompiler::IR::SetFastVerifyHook(resources != nullptr ? &FastVerifyLoggedReads : nullptr);
	if (resources != nullptr) {
		Graphics::ShaderRecompiler::IR::SetOrdinaryReadHook(&OrdinaryReadBufferBacking);
		Graphics::ShaderRecompiler::IR::SetOrdinaryReadSessionHook(&SrtReadSession);
	}
}

bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->HandleFault(access, fault_vaddr);
}

void ReadGuestPlain(void* dst, uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr) {
		std::memcpy(dst, reinterpret_cast<const void*>(vaddr), size);
		return;
	}
	constexpr uint64_t page_size = 0x1000;
	const auto&        pages     = g_gpu_resources->GetPageManager();
	const uint64_t     last      = vaddr + size;
	auto*              out       = static_cast<uint8_t*>(dst);
	for (uint64_t address = vaddr; address < last;) {
		const bool watched = pages.ImageReadBypass(address);
		uint64_t   end     = (address & ~(page_size - 1u)) + page_size;
		while (end < last && pages.ImageReadBypass(end) == watched) {
			end += page_size;
		}
		end = std::min(end, last);
		if (!watched || !TryReadBacking(address, out + (address - vaddr), end - address)) {
			std::memcpy(out + (address - vaddr), reinterpret_cast<const void*>(address), end - address);
		} else {
			Graphics::TextureCache::NoteImageReadPlain(end - address);
		}
		address = end;
	}
}

bool GuestImageReadArmed(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && g_gpu_resources->GetPageManager().ImageReadArmedInRange(vaddr, size);
}

const void* GuestBytesForKernelRead(const void* guest, size_t size, std::vector<uint8_t>& scratch) {
	if (!GuestImageReadArmed(reinterpret_cast<uint64_t>(guest), size)) {
		return guest;
	}
	scratch.resize(size);
	std::memcpy(scratch.data(), guest, size);
	return scratch.data();
}

bool GuardedGuestCopy(void* dst, const void* src, size_t size) {
	const auto s = reinterpret_cast<uint64_t>(src);
	const auto d = reinterpret_cast<uint64_t>(dst);
	if (size == 0 || (d < s + size && s < d + size)) {
		return false;
	}
	SIZE_T copied = 0;
	return ReadProcessMemory(GetCurrentProcess(), src, dst, size, &copied) != 0 && copied == size;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	return Map().CopySparse<false>(vaddr, data, size);
}

bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
	return Map().CopySparse<true>(vaddr, data, size);
}

static DWORD WindowsProtection(VirtualMemory::Mode mode) {
	switch (mode) {
		case VirtualMemory::Mode::Read: return PAGE_READONLY;
		case VirtualMemory::Mode::Write:
		case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
		case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
		case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
		case VirtualMemory::Mode::ExecuteWrite:
		case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
		case VirtualMemory::Mode::NoAccess:
		default: return PAGE_NOACCESS;
	}
}

static std::atomic<uint64_t> g_protect_splits {0};

// One VirtualProtect cannot cross from one allocation to the next. Kyty maps one view per guest mapping; AnyPS5 maps a
// guest range as several views or commits it in granules (128 KiB views in the world), so a range the host refuses
// whole is protected again allocation by allocation, over its committed parts only (as Kyty skips what is not mapped).
static bool ProtectPiece(uint64_t vaddr, uint64_t size, DWORD protection) {
	if (Aps5::ProtectGuest(vaddr, size, protection)) {
		return true;
	}
	g_protect_splits.fetch_add(1, std::memory_order_relaxed);
	const auto end    = vaddr + size;
	auto       cursor = vaddr;
	bool       ok     = true;
	while (cursor < end) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(cursor), &info, sizeof(info)) == 0) {
			return false;
		}
		auto stop = std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
		if (info.State != MEM_COMMIT) {
			cursor = stop;
			continue;
		}
		// The next regions of the same allocation join the piece (VirtualQuery splits one by protection).
		while (stop < end) {
			MEMORY_BASIC_INFORMATION next {};
			if (VirtualQuery(reinterpret_cast<const void*>(stop), &next, sizeof(next)) == 0 ||
			    next.AllocationBase != info.AllocationBase || next.State != MEM_COMMIT) {
				break;
			}
			stop = std::min(end, reinterpret_cast<uint64_t>(next.BaseAddress) + next.RegionSize);
		}
		ok     = Aps5::ProtectGuest(cursor, stop - cursor, protection) && ok;
		cursor = stop;
	}
	return ok;
}

// Kyty's ProtectTransient: the tracker's protection, over the committed pieces of the range only (Kyty protects mapped
// regions and skips gaps), through the host, which orders it against AnyPS5's own mapping and protection changes.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}
	const auto protection = WindowsProtection(mode);
	std::array<std::pair<uint64_t, uint64_t>, 16> pieces {};
	size_t                                        count    = 0;
	bool                                          overflow = false;
	Map().ForEachCommitted(vaddr, size, [&](uint64_t start, uint64_t bytes) {
		if (count == pieces.size()) {
			overflow = true;
			return;
		}
		pieces[count++] = {start, bytes};
	});
	if (count == 0 || overflow) {
		return ProtectPiece(vaddr, size, protection);
	}
	bool ok = true;
	for (size_t i = 0; i < count; i++) {
		ok = ProtectPiece(pieces[i].first, pieces[i].second, protection) && ok;
	}
	return ok;
}

DebugStats GetDebugStats() {
	return {};
}

} // namespace Libs::LibKernel::Memory

namespace Aps5 {

// Kyty commits its whole direct-memory backing up front (one SEC_COMMIT section); AnyPS5's kyty mode reserves the section
// and commits on map. The BDA fallback imports 2 MiB granules of the backing as Vulkan host memory, and a granule with an
// uncommitted page would be refused (its reads would stay zero): so the whole view is committed here, as on Kyty.
void MemorySetBacking(uint8_t* backing_base, uint64_t backing_size) {
	Libs::LibKernel::Memory::Map().SetBacking(backing_base, backing_size);
}

void MemoryCommitBacking() {
	auto*      backing_base = Libs::LibKernel::Memory::Map().BackingBase();
	const auto backing_size = Libs::LibKernel::Memory::Map().BackingSize();
	if (backing_base == nullptr) {
		return;
	}
	if (VirtualAlloc(backing_base, backing_size, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
		Log("memory: committing the direct memory backing view failed (error %lu); BDA fallback imports of granules with "
		    "uncommitted pages will be refused",
		    GetLastError());
	} else {
		Log("memory: direct memory backing view committed (%llu MiB), as Kyty's backing store",
		    static_cast<unsigned long long>(backing_size >> 20u));
	}
}

void MemoryMapped(uint64_t address, uint64_t bytes, uint64_t backing_offset) {
	Libs::LibKernel::Memory::Map().Map(address, bytes, backing_offset);
}

void MemoryUnmapped(uint64_t address, uint64_t bytes) {
	Libs::LibKernel::Memory::NoteRecentUnmap(address, bytes);
	Libs::LibKernel::Memory::Map().Unmap(address, bytes);
}

MemoryStats GetMemoryStats() {
	const auto stats = Libs::LibKernel::Memory::Map().GetStats();
	return {stats.entries, stats.direct, stats.private_, stats.discovered,
	        Libs::LibKernel::Memory::g_protect_splits.load(std::memory_order_relaxed)};
}

void MemoryForEachMapping(void (*callback)(void* context, uint64_t address, uint64_t bytes), void* context) {
	Libs::LibKernel::Memory::Map().ForEachRun(callback, context);
}

} // namespace Aps5
