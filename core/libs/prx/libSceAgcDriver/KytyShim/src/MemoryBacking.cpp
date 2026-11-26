#include "kernel/memory.h"

#include "common/assert.h"
#include "common/hackNotice.h"

#include "MemoryShim.hpp"
#include "prx/libc/include/GuestArena.hpp"
#ifdef _WIN32
#include "prx/libc/include/RedZoneGate.hpp"
#endif
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/Pthread/include/Pthread.hpp"

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
#include <optional>
#include <set>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel::Memory {

namespace {

#ifdef _WIN32
constexpr int PROT_NONE = 0;
constexpr int PROT_READ = 1;
constexpr int PROT_WRITE = 2;
constexpr int PROT_EXEC = 4;
#endif

constexpr uint64_t PageBits = 14;
constexpr uint64_t PageSize = uint64_t {1} << PageBits;
constexpr uint64_t PageMask = PageSize - 1u;
constexpr uint64_t BucketPages = uint64_t {1} << 16u;
constexpr uint64_t BucketCount = ((kExtendedMemoryBase + kExtendedMemorySize) >> PageBits) / BucketPages;
constexpr uint32_t PrivateEntry = ~uint32_t {0};
constexpr uint64_t HostPageSize = 0x1000;
constexpr uint64_t HostPageMask = HostPageSize - 1u;
constexpr size_t ProbeBatch = 512;
#ifndef _WIN32
constexpr uint64_t SeekMinBytes = uint64_t {64} << 10u;
constexpr uint64_t SeekFirstChunk = uint64_t {64} << 10u;
constexpr uint64_t SeekLastChunk = uint64_t {1} << 20u;
#endif

#if defined(__linux__)
template <class Mutex>
class GuestDeliveryLock {
public:
	explicit GuestDeliveryLock(Mutex& mutex) : m_lock(mutex) {}
	GuestDeliveryLock(const GuestDeliveryLock&) = delete;
	GuestDeliveryLock& operator=(const GuestDeliveryLock&) = delete;

private:
	struct Hold {
		Hold() { GuestDeliveryHold_nid_no_patch(); }
		~Hold() { GuestDeliveryRelease_nid_no_patch(); }
		Hold(const Hold&) = delete;
		Hold& operator=(const Hold&) = delete;
	};

	Hold m_hold;
	std::lock_guard<Mutex> m_lock;
};
#else
template <class Mutex>
class GuestDeliveryLock {
public:
	explicit GuestDeliveryLock(Mutex& mutex) : m_lock(mutex) {}
	GuestDeliveryLock(const GuestDeliveryLock&) = delete;
	GuestDeliveryLock& operator=(const GuestDeliveryLock&) = delete;

private:
	std::lock_guard<Mutex> m_lock;
};
#endif

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

	template <typename Visit>
	void ForEachBacking(uint64_t vaddr, uint64_t size, Visit&& visit) const {
		for (auto page = vaddr >> PageBits; page < (vaddr + size) >> PageBits; ++page) {
			if (page / BucketPages >= BucketCount) return;
			if (m_buckets[page / BucketPages].load(std::memory_order_acquire) == nullptr) {
				page = (page / BucketPages + 1u) * BucketPages - 1u;
				continue;
			}
			if (const auto entry = LoadBacking(page); entry != 0) visit(entry);
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

	void StoreExecutable(uint64_t start, uint64_t end, bool executable) {
		for (auto host = start & ~HostPageMask; host < end; host += HostPageSize) {
			const auto page = host >> PageBits;
			if (page / BucketPages >= BucketCount) return;
			auto& bucket = m_exec[page / BucketPages];
			auto* bits = bucket.load(std::memory_order_acquire);
			if (bits == nullptr) {
				if (!executable) {
					host = (((page / BucketPages + 1u) * BucketPages) << PageBits) - HostPageSize;
					continue;
				}
				auto* created = new std::atomic<uint8_t>[BucketPages] {};
				if (bucket.compare_exchange_strong(bits, created, std::memory_order_acq_rel)) {
					bits = created;
				} else {
					delete[] created;
				}
			}
			const auto bit = static_cast<uint8_t>(1u << ((host >> 12u) & 3u));
			if (executable) {
				bits[page % BucketPages].fetch_or(bit, std::memory_order_relaxed);
			} else {
				bits[page % BucketPages].fetch_and(static_cast<uint8_t>(~bit), std::memory_order_relaxed);
			}
		}
	}

	[[nodiscard]] bool Executable(uint64_t vaddr) const {
		const auto page = vaddr >> PageBits;
		if (page / BucketPages >= BucketCount) return false;
		const auto* bits = m_exec[page / BucketPages].load(std::memory_order_acquire);
		return bits != nullptr && (bits[page % BucketPages].load(std::memory_order_relaxed) & (1u << ((vaddr >> 12u) & 3u))) != 0;
	}

	[[nodiscard]] uint8_t Probed(uint64_t host) const {
		const auto page = host >> PageBits;
		if (page / BucketPages >= BucketCount) return ProbeUnknown;
		const auto* bits = m_probe[page / BucketPages].load(std::memory_order_acquire);
		if (bits == nullptr) return ProbeUnknown;
		return static_cast<uint8_t>((bits[page % BucketPages].load(std::memory_order_acquire) >> ProbeShift(host)) & ProbeMask);
	}

	void StoreProbe(uint64_t host, bool readable) {
		const auto page = host >> PageBits;
		if (page / BucketPages >= BucketCount) return;
		auto& bucket = m_probe[page / BucketPages];
		auto* bits = bucket.load(std::memory_order_acquire);
		if (bits == nullptr) {
			auto* created = new std::atomic<uint8_t>[BucketPages] {};
			if (bucket.compare_exchange_strong(bits, created, std::memory_order_acq_rel)) {
				bits = created;
			} else {
				delete[] created;
			}
		}
		const auto shift = ProbeShift(host);
		const auto value = static_cast<uint8_t>((readable ? ProbeReadable : ProbeUnreadable) << shift);
		auto& slot = bits[page % BucketPages];
		auto current = slot.load(std::memory_order_relaxed);
		while (!slot.compare_exchange_weak(current, static_cast<uint8_t>((current & ~(ProbeMask << shift)) | value), std::memory_order_acq_rel)) {}
	}

	void ClearProbes(uint64_t start, uint64_t end) {
		for (auto host = start & ~HostPageMask; host < end; host += HostPageSize) {
			const auto page = host >> PageBits;
			if (page / BucketPages >= BucketCount) return;
			auto* bits = m_probe[page / BucketPages].load(std::memory_order_acquire);
			if (bits == nullptr) {
				host = (((page / BucketPages + 1u) * BucketPages) << PageBits) - HostPageSize;
				continue;
			}
			if ((host & PageMask) == 0 && end - host >= PageSize) {
				bits[page % BucketPages].store(0, std::memory_order_release);
				host += PageSize - HostPageSize;
				continue;
			}
			bits[page % BucketPages].fetch_and(static_cast<uint8_t>(~(ProbeMask << ProbeShift(host))), std::memory_order_acq_rel);
		}
	}

	static constexpr uint8_t ProbeUnknown = 0;
	static constexpr uint8_t ProbeUnreadable = 1;
	static constexpr uint8_t ProbeReadable = 3;

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
	static constexpr uint8_t ProbeMask = 3;

	[[nodiscard]] static unsigned ProbeShift(uint64_t host) {
		return static_cast<unsigned>(((host >> 12u) & 3u) * 2u);
	}

	std::array<std::atomic<std::atomic<uint32_t>*>, BucketCount> m_buckets {};
	std::array<std::atomic<std::atomic<uint8_t>*>, BucketCount> m_exec {};
	std::array<std::atomic<std::atomic<uint8_t>*>, BucketCount> m_probe {};
};

struct ViewState {
	uint8_t* base = nullptr;
	uint64_t guest_bytes = 0;
	uint64_t view_bytes = 0;
	uint64_t arena_start = 0;
	uint64_t arena_end = 0;
#ifndef _WIN32
	pid_t pid = 0;
	int guest_file = -1;
	int direct_file = -1;
#endif
	std::atomic<bool> enabled {false};
};

PageTable& Pages() {
	static auto* pages = new PageTable;
	return *pages;
}

constexpr uint64_t ShareBucketCount = (uint64_t {PrivateEntry} + BucketPages - 1u) / BucketPages;

class ShareTable {
public:
	static constexpr uint32_t CountMask = 0xffu;
	static constexpr uint32_t Generation = CountMask + 1u;

	[[nodiscard]] uint32_t Load(uint32_t entry) const {
		const auto page = uint64_t {entry} - 1u;
		if (page / BucketPages >= ShareBucketCount) return CountMask;
		const auto* words = m_buckets[page / BucketPages].load(std::memory_order_acquire);
		return words != nullptr ? words[page % BucketPages].load(std::memory_order_acquire) : 0;
	}

	void Change(uint32_t entry, bool mapped) {
		const auto page = uint64_t {entry} - 1u;
		if (page / BucketPages >= ShareBucketCount) return;
		auto& bucket = m_buckets[page / BucketPages];
		auto* words = bucket.load(std::memory_order_acquire);
		if (words == nullptr) {
			words = new std::atomic<uint32_t>[BucketPages] {};
			bucket.store(words, std::memory_order_release);
		}
		auto& word = words[page % BucketPages];
		const auto current = word.load(std::memory_order_relaxed);
		auto count = current & CountMask;
		if (!mapped && count == 0) {
			Shim::Fail("KytyShim memory: view page 0x%" PRIx64 " was unmapped more often than it was mapped\n", page);
		}
		if (count != CountMask) {
			count = mapped ? count + 1u : count - 1u;
		}
		word.store(((current & ~CountMask) + Generation) | count, std::memory_order_release);
	}

private:
	std::array<std::atomic<std::atomic<uint32_t>*>, ShareBucketCount> m_buckets {};
};

ShareTable& Shares() {
	static auto* shares = new ShareTable;
	return *shares;
}

void ReleaseShares(uint64_t address, uint64_t bytes) {
	Pages().ForEachBacking(address, bytes, [](uint32_t entry) { Shares().Change(entry, false); });
}

ViewState& State() {
	static auto* state = new ViewState;
	return *state;
}

std::atomic<uint64_t> g_probe_epoch {0};
std::atomic<uint64_t> g_probe_calls {0};
std::atomic<bool> g_any_probe {false};

void ForgetProbes(uint64_t start, uint64_t size) {
	g_probe_epoch.fetch_add(1, std::memory_order_seq_cst);
	if (g_any_probe.load(std::memory_order_seq_cst)) Pages().ClearProbes(start, start + size);
}

std::atomic<uint64_t> g_backing_epoch {0};
constexpr uint64_t BackingChangeHistory = 4096;
std::mutex g_backing_changes_mutex;
std::array<std::pair<uint64_t, uint64_t>, BackingChangeHistory> g_backing_changes {};

void BumpBackingEpoch(uint64_t start, uint64_t size) noexcept {
	GuestDeliveryLock lock(g_backing_changes_mutex);
	const auto epoch = g_backing_epoch.load(std::memory_order_relaxed) + 1;
	g_backing_changes[epoch % BackingChangeHistory] = {start, size};
	g_backing_epoch.store(epoch, std::memory_order_release);
}

bool FirstClamp(uint64_t vaddr, uint64_t size, uint64_t clamped) {
	static std::mutex mutex;
	static std::set<std::tuple<uint64_t, uint64_t, uint64_t>> reported;
	GuestDeliveryLock lock(mutex);
	return reported.emplace(vaddr, size, clamped).second;
}

#ifndef _WIN32
constexpr uint64_t HostPage = 0x1000;

struct HostRange {
	uint64_t first = 0;
	uint64_t end = 0;
};

HostRange HostPages(uint64_t address, uint64_t bytes) {
	return {address & ~(HostPage - 1u), (address + bytes + HostPage - 1u) & ~(HostPage - 1u)};
}

struct HostWriteHold {
	HostRange pages;
	bool overridden = false;
};

std::mutex g_host_write_holds_mutex;
std::vector<HostWriteHold> g_host_write_holds;

enum class PageKeep : uint8_t { Writable, Untouched };

struct KeptRange {
	HostRange pages;
	PageKeep keep = PageKeep::Writable;
};

#if defined(__linux__)
void KeepThreadStack(void* context, std::uintptr_t first, std::uintptr_t end) {
	static_cast<std::vector<KeptRange>*>(context)->push_back({HostPages(first, end - first), PageKeep::Untouched});
}
#endif

bool ProtectHostPages(uint64_t start, uint64_t stop, int protection, const std::vector<KeptRange>& kept) {
	if (kept.empty()) return ::mprotect(reinterpret_cast<void*>(start), stop - start, protection) == 0;
	std::vector<uint64_t> bounds {start, stop};
	for (const auto& range : kept) {
		if (range.pages.first > start && range.pages.first < stop) bounds.push_back(range.pages.first);
		if (range.pages.end > start && range.pages.end < stop) bounds.push_back(range.pages.end);
	}
	std::sort(bounds.begin(), bounds.end());
	bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
	const auto protection_at = [&](uint64_t page) {
		auto applied = protection;
		for (const auto& range : kept) {
			if (page < range.pages.first || page >= range.pages.end) continue;
			if (range.keep == PageKeep::Untouched) return -1;
			applied = protection | PROT_READ | PROT_WRITE;
		}
		return applied;
	};
	for (std::size_t index = 0; index + 1 < bounds.size();) {
		const auto applied = protection_at(bounds[index]);
		auto next = index + 1;
		while (next + 1 < bounds.size() && protection_at(bounds[next]) == applied) ++next;
		if (applied >= 0 && ::mprotect(reinterpret_cast<void*>(bounds[index]), bounds[next] - bounds[index], applied) != 0) return false;
		index = next;
	}
	return true;
}
#endif

std::recursive_mutex& HookMutex() {
	static auto* mutex = new std::recursive_mutex;
	return *mutex;
}

Shim::RangeHook g_map_hook = nullptr;
Shim::RangeHook g_unmap_hook = nullptr;

bool ValidRange(uint64_t vaddr, uint64_t size) {
	return size != 0 && UINT64_MAX - vaddr >= size;
}

#ifndef _WIN32
bool SparseReadsEnabled() {
	const char* value = std::getenv("ANYPS5_KYTY_SPARSE_READS");
	if (value == nullptr || *value == '\0') return true;
	const std::string_view text(value);
	if (text != "0" && text != "1") Shim::Fail("KytyShim memory: ANYPS5_KYTY_SPARSE_READS=%s is neither 0 nor 1\n", value);
	return text == "1";
}
#endif

void CopyFromView(uint8_t* destination, uint64_t offset, uint64_t bytes) {
	const auto& state = State();
#ifdef _WIN32
	std::memcpy(destination, state.base + offset, bytes);
#else
	if (bytes < SeekMinBytes || state.guest_file < 0) {
		std::memcpy(destination, state.base + offset, bytes);
		return;
	}
	const auto end = offset + bytes;
	auto chunk = SeekFirstChunk;
	for (auto cursor = offset; cursor < end;) {
		const bool guest = cursor < state.guest_bytes;
		const auto file = guest ? state.guest_file : state.direct_file;
		const auto file_start = guest ? 0 : state.guest_bytes;
		const auto file_stop = guest ? std::min(end, state.guest_bytes) : end;
		auto* out = destination + (cursor - offset);
		const auto data = ::lseek(file, static_cast<off_t>(cursor - file_start), SEEK_DATA);
		if (data < 0 && errno != ENXIO) {
			std::memcpy(out, state.base + cursor, file_stop - cursor);
			cursor = file_stop;
			continue;
		}
		const auto hole_end = data < 0 ? file_stop : std::min(file_stop, file_start + static_cast<uint64_t>(data));
		if (hole_end > cursor) {
			std::memset(out, 0, hole_end - cursor);
			cursor = hole_end;
			chunk = SeekFirstChunk;
			continue;
		}
		const auto copy_end = std::min(file_stop, (cursor & ~(chunk - 1u)) + chunk);
		std::memcpy(out, state.base + cursor, copy_end - cursor);
		cursor = copy_end;
		chunk = std::min(chunk * 2u, SeekLastChunk);
	}
#endif
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

#ifdef _WIN32
bool HostReadable(const MEMORY_BASIC_INFORMATION& memory) {
	constexpr DWORD Readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
	return memory.State == MEM_COMMIT && (memory.Protect & PAGE_GUARD) == 0 && (memory.Protect & Readable) != 0;
}

uint64_t ReadablePrefix(uint64_t start, uint64_t end) {
	for (auto cursor = start; cursor < end;) {
		MEMORY_BASIC_INFORMATION memory {};
		if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), &memory, sizeof(memory)) != sizeof(memory) || !HostReadable(memory)) {
			return cursor - start;
		}
		cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
	}
	return end - start;
}

void CopyReadable(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto end = start + bytes;
	const HANDLE process = GetCurrentProcess();
	for (auto cursor = start; cursor < end;) {
		SIZE_T read = 0;
		const bool complete = ReadProcessMemory(process, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), destination + (cursor - start), end - cursor, &read) != 0;
		if (complete) return;
		cursor += read;
		if (cursor < end) {
			const auto page_end = std::min(end, (cursor | HostPageMask) + 1u);
			SIZE_T page_read = 0;
			if (ReadProcessMemory(process, reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), destination + (cursor - start), page_end - cursor, &page_read) == 0) {
				std::memset(destination + (cursor - start), 0, page_end - cursor);
			}
			cursor = page_end;
		}
	}
}
#else
size_t ReadOwnMemory(const iovec& local, const iovec* remote, size_t count) {
	const auto read = ::process_vm_readv(State().pid, &local, 1, remote, count, 0);
	if (read >= 0) return static_cast<size_t>(read);
	if (errno != EFAULT) Shim::Fail("KytyShim memory: process_vm_readv on guest memory failed: %s\n", std::strerror(errno));
	return 0;
}

uint64_t ReadablePrefix(uint64_t start, uint64_t end) {
	auto& pages = Pages();
	for (auto cursor = start; cursor < end;) {
		const auto known = pages.Probed(cursor);
		if (known == PageTable::ProbeUnreadable) return cursor - start;
		if (known == PageTable::ProbeReadable) {
			cursor = (cursor | HostPageMask) + 1u;
			continue;
		}
		std::array<iovec, ProbeBatch> remote {};
		std::array<char, ProbeBatch> sink {};
		size_t count = 0;
		for (auto page = cursor; page < end && count < ProbeBatch && (count == 0 || pages.Probed(page) == PageTable::ProbeUnknown); page = (page | HostPageMask) + 1u) {
			remote[count++] = {reinterpret_cast<void*>(static_cast<std::uintptr_t>(page)), 1};
		}
		const auto epoch = g_probe_epoch.load(std::memory_order_seq_cst);
		g_any_probe.store(true, std::memory_order_seq_cst);
		g_probe_calls.fetch_add(1, std::memory_order_relaxed);
		const iovec local {sink.data(), count};
		const auto read = ReadOwnMemory(local, remote.data(), count);
		const auto known_end = std::min(read + 1u, count);
		for (size_t i = 0; i < known_end; ++i) pages.StoreProbe(reinterpret_cast<std::uintptr_t>(remote[i].iov_base), i < read);
		const auto probed_end = (reinterpret_cast<std::uintptr_t>(remote[known_end - 1u].iov_base) | HostPageMask) + 1u;
		if (g_probe_epoch.load(std::memory_order_seq_cst) != epoch) pages.ClearProbes(cursor, probed_end);
		if (read < count) return reinterpret_cast<std::uintptr_t>(remote[read].iov_base) - start;
		cursor = probed_end;
	}
	return end - start;
}
#endif

uint64_t UnreportedMappedPrefix(uint64_t start, uint64_t end) {
	const auto& state = State();
	if (start < state.arena_start || start >= state.arena_end) return 0;
	return ReadablePrefix(start, std::min(end, state.arena_end));
}

bool AnyUnreportedMapped(uint64_t start, uint64_t end) {
	const auto& state = State();
	end = std::min(end, state.arena_end);
	for (auto cursor = std::max(start, state.arena_start); cursor < end; cursor = (cursor | HostPageMask) + 1u) {
		if (ReadablePrefix(cursor, end) != 0) return true;
	}
	return false;
}

#ifndef _WIN32
void CopyReadable(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto end = start + bytes;
	for (auto cursor = start; cursor < end;) {
		std::array<iovec, ProbeBatch> remote {};
		size_t count = 0;
		auto stop = cursor;
		while (stop < end && count < ProbeBatch) {
			const auto next = std::min(end, (stop | HostPageMask) + 1u);
			remote[count++] = {reinterpret_cast<void*>(static_cast<std::uintptr_t>(stop)), next - stop};
			stop = next;
		}
		const iovec local {destination + (cursor - start), stop - cursor};
		cursor += ReadOwnMemory(local, remote.data(), count);
		if (cursor < stop) {
			const auto skipped = std::min(end, (cursor | HostPageMask) + 1u);
			std::memset(destination + (cursor - start), 0, skipped - cursor);
			cursor = skipped;
		}
	}
}
#endif

void CopyUnreported(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto& state = State();
	const auto end = start + bytes;
	const auto low = std::max(start, state.arena_start);
	const auto high = std::min(end, state.arena_end);
	if (low >= high) {
		std::memset(destination, 0, bytes);
		return;
	}
	std::memset(destination, 0, low - start);
	CopyReadable(low, destination + (low - start), high - low);
	std::memset(destination + (high - start), 0, end - high);
}

void CopyGuestPages(uint64_t start, uint8_t* destination, uint64_t bytes) {
	const auto& pages = Pages();
	const auto end = start + bytes;
	for (auto cursor = start; cursor < end;) {
		const bool reported = pages.Load(cursor >> PageBits) == PrivateEntry;
		auto stop = std::min(end, (cursor | PageMask) + 1u);
		while (stop < end && (pages.Load(stop >> PageBits) == PrivateEntry) == reported) stop = std::min(end, stop + PageSize);
		if (reported) {
			std::memcpy(destination + (cursor - start), reinterpret_cast<const void*>(cursor), stop - cursor);
		} else {
			CopyUnreported(cursor, destination + (cursor - start), stop - cursor);
		}
		cursor = stop;
	}
}

uint64_t MappedPrefix(uint64_t vaddr, uint64_t size) {
	const auto& pages = Pages();
	const auto end = vaddr + size;
	auto cursor = vaddr;
	while (cursor < end) {
		if (pages.Load(cursor >> PageBits) != 0) {
			cursor = std::min(end, (cursor | PageMask) + 1u);
			continue;
		}
		auto stop = std::min(end, (cursor | PageMask) + 1u);
		while (stop < end && pages.Load(stop >> PageBits) == 0) stop = std::min(end, stop + PageSize);
		cursor += UnreportedMappedPrefix(cursor, stop);
		if (cursor < stop) break;
	}
	return cursor - vaddr;
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
	GuestDeliveryLock lock(HookMutex());
	auto& pages = Pages();
	ForgetProbes(address, bytes);
	if (offset == DirectMemoryViewReplacing) {
		if (pages.AnyMapped(address, bytes)) {
			if (g_unmap_hook != nullptr) g_unmap_hook(address, bytes);
			ReleaseShares(address, bytes);
			pages.Store(address, bytes, 0, false);
			BumpBackingEpoch(address, bytes);
		}
		GuestArena::GuestArenaNoteHostProtection_nid_no_patch(address, bytes, true);
		return;
	}
	if (offset == DirectMemoryViewUnmapped) {
		const bool mapped = pages.AnyMapped(address, bytes);
		if (mapped && g_unmap_hook != nullptr) g_unmap_hook(address, bytes);
		ReleaseShares(address, bytes);
		pages.Store(address, bytes, 0, false);
		BumpBackingEpoch(address, bytes);
		GuestArena::GuestArenaNoteHostProtection_nid_no_patch(address, bytes, true);
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
	ReleaseShares(address, bytes);
	pages.Store(address, bytes, first, step);
	if (step) {
		for (uint64_t page = 0; page < bytes >> PageBits; ++page) {
			Shares().Change(first + static_cast<uint32_t>(page), true);
		}
	}
	BumpBackingEpoch(address, bytes);
	if (g_map_hook != nullptr) g_map_hook(address, bytes);
}

void OnProtectionChange(std::uintptr_t address, std::size_t bytes, int protection) {
	if (UINT64_MAX - address < bytes) {
		Shim::Fail("KytyShim memory: libkernel reported a protection change 0x%016" PRIx64 "+0x%zx past the address space\n", static_cast<uint64_t>(address), bytes);
	}
	Pages().StoreExecutable(address, address + bytes, protection >= 0 && (protection & 0x04) != 0);
	ForgetProbes(address, bytes);
}

void EnableViewOnce() {
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
	std::uintptr_t arena_base = 0;
	std::size_t arena_bytes = 0;
	GuestArena::GuestArenaRange_nid_postfix(&arena_base, &arena_bytes);
	state.arena_start = arena_base;
	state.arena_end = arena_base + arena_bytes;
#ifndef _WIN32
	state.pid = ::getpid();
#endif
	state.base = view;
	state.guest_bytes = guest_bytes;
	state.view_bytes = view_bytes;
	std::fprintf(stderr, "[kyty] second view of guest memory at %p: guest range 0x%zx bytes, direct memory after it, 0x%zx bytes in all\n", static_cast<void*>(view), guest_bytes, view_bytes);
#ifndef _WIN32
	if (SparseReadsEnabled()) {
		int guest_file = -1;
		int direct_file = -1;
		if (!DirectMemoryViewFiles_nid_no_patch(&guest_file, &direct_file) || guest_file < 0 || direct_file < 0) {
			Shim::Fail("KytyShim memory: libkernel did not give the files behind the second view of guest memory\n");
		}
		state.guest_file = guest_file;
		state.direct_file = direct_file;
	}
#endif
	DirectMemoryWatchProtections_nid_no_patch(&OnProtectionChange);
	state.enabled.store(true, std::memory_order_release);
}

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

#ifndef _WIN32
void HoldHostWrite(std::uintptr_t address, std::size_t bytes) {
	GuestDeliveryLock lock(g_host_write_holds_mutex);
	g_host_write_holds.push_back({HostPages(address, bytes)});
}

bool ReleaseHostWrite(std::uintptr_t address, std::size_t bytes) {
	const auto pages = HostPages(address, bytes);
	GuestDeliveryLock lock(g_host_write_holds_mutex);
	const auto hold = std::find_if(g_host_write_holds.begin(), g_host_write_holds.end(), [&pages](const auto& held) { return held.pages.first == pages.first && held.pages.end == pages.end; });
	if (hold == g_host_write_holds.end()) return false;
	const bool overridden = hold->overridden;
	*hold = g_host_write_holds.back();
	g_host_write_holds.pop_back();
	return overridden;
}
#endif

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
		if (entry == 0 && AnyUnreportedMapped(std::max(page, vaddr & ~HostPageMask), std::min(page + PageSize, (end + HostPageMask) & ~HostPageMask))) return true;
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
	std::lock_guard lock(HookMutex());
	g_map_hook = map;
	g_unmap_hook = unmap;
	if (map != nullptr) Pages().ForEachMappedRun([&](uint64_t start, uint64_t size) { map(start, size); });
}

uint64_t ProbeCalls() {
	return g_probe_calls.load(std::memory_order_relaxed);
}

void EnableView() {
	EnableViewOnce();
}

uint64_t DirectMappedBytes() {
	uint64_t bytes = 0;
	const auto guest_bytes = State().guest_bytes;
	Pages().ForEachMappedRun([&](uint64_t run_start, uint64_t run_size) {
		for (auto page = run_start; page < run_start + run_size; page += PageSize) {
			const auto entry = Pages().LoadBacking(page >> PageBits);
			if (entry != 0 && (uint64_t {entry - 1u} << PageBits) >= guest_bytes) bytes += PageSize;
		}
	});
	return bytes;
}

}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (data == nullptr || !Shim::IsBacked(vaddr, size)) return false;
	auto* destination = static_cast<uint8_t*>(data);
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { CopyFromView(destination + (start - vaddr), offset, bytes); },
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

bool GuestViewRange(const uint8_t** base, uint64_t* guest_bytes, uint64_t* view_bytes) {
	if (!Shim::ViewEnabled()) return false;
	const auto& state = State();
	*base = state.base;
	*guest_bytes = state.guest_bytes;
	*view_bytes = state.view_bytes;
	return true;
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

BackingShare SoleBackingStamp(uint64_t vaddr, uint64_t size, uint64_t* stamp) {
	if (!ValidRange(vaddr, size) || !Shim::ViewEnabled()) return BackingShare::Unmapped;
	const auto& pages = Pages();
	const auto& shares = Shares();
	uint64_t mixed = 0xcbf29ce484222325ull;
	for (auto page = vaddr >> PageBits; page <= (vaddr + size - 1u) >> PageBits; ++page) {
		const auto entry = pages.Load(page);
		if (entry == 0) return BackingShare::Unmapped;
		if (entry == PrivateEntry) return BackingShare::Private;
		const auto word = shares.Load(entry);
		if ((word & ShareTable::CountMask) != 1u) return BackingShare::Aliased;
		mixed = (mixed ^ ((uint64_t {entry} << 32u) | word)) * 0x100000001b3ull;
	}
	*stamp = mixed;
	return BackingShare::Sole;
}

bool BackingChangesSince(uint64_t epoch, std::vector<std::pair<uint64_t, uint64_t>>& changes) {
	GuestDeliveryLock lock(g_backing_changes_mutex);
	const auto current = g_backing_epoch.load(std::memory_order_relaxed);
	if (current - epoch > BackingChangeHistory) return false;
	for (auto change = epoch + 1; change <= current; ++change) {
		changes.push_back(g_backing_changes[change % BackingChangeHistory]);
	}
	return true;
}

bool IsFullyMapped(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) return size == 0;
	return Shim::IsBacked(vaddr, size) || MappedPrefix(vaddr, size) == size;
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) {
		Shim::Fail("KytyShim memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	if (Shim::IsBacked(vaddr, size)) return size;
	const auto clamped = MappedPrefix(vaddr, size);
	if (clamped == size) return size;
	if (clamped == 0) {
		Shim::Fail("KytyShim memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64 "\n", vaddr, size);
	}
	if (FirstClamp(vaddr, size, clamped)) std::fprintf(stderr, "KytyShim memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64 " to 0x%016" PRIx64 "\n", vaddr, size, clamped);
	return clamped;
}

uint64_t ClampRangeSizeOrZero(uint64_t vaddr, uint64_t size) {
	if (!ValidRange(vaddr, size)) return 0;
	if (IsFullyMapped(vaddr, size)) return size;
	const auto clamped = MappedPrefix(vaddr, size);
	if (clamped == 0) {
		Common::HackNotice::Report("V# outside guest memory bound as a null buffer", "ClampRangeSizeOrZero", 0, 0);
	}
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
	ForEachRun(
	    vaddr, size,
	    [&](uint64_t offset, uint64_t start, uint64_t bytes) { CopyFromView(destination + (start - vaddr), offset, bytes); },
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

#ifdef _WIN32
namespace {

void RequireFramedGuest() {
	if (RedZoneGate::Open()) return;
	EXIT("KytyShim: Kyty tracks guest writes by page protection and cannot run without it. On Windows a fault on a guest thread overwrites the values a leaf function keeps below RSP, which corrupts guest memory without a trace. Convert the title again with --frame-red-zone, or set APS5_ALLOW_UNFRAMED_FAULTS=1 to take the faults anyway\n");
}

}
#endif

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) {
	if (!ValidRange(vaddr, size)) return false;
	const auto protection = HostProtection(mode);
	const bool writable = (protection & PROT_WRITE) != 0;
#ifdef _WIN32
	const DWORD host_protection = protection == PROT_NONE                               ? PAGE_NOACCESS
	                              : (protection & PROT_EXEC) != 0 && writable             ? PAGE_EXECUTE_READWRITE
	                              : (protection & PROT_EXEC) != 0 && (protection & PROT_READ) != 0 ? PAGE_EXECUTE_READ
	                              : (protection & PROT_EXEC) != 0                         ? PAGE_EXECUTE
	                              : writable                                              ? PAGE_READWRITE
	                                                                                      : PAGE_READONLY;
	if (!writable) RequireFramedGuest();
	if (!writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, false);
	const auto& mapped_pages = Pages();
	const auto host_first = vaddr & ~uint64_t {0xfff};
	const auto host_end = (vaddr + size + 0xfff) & ~uint64_t {0xfff};
	for (auto start = host_first; start < host_end;) {
		auto stop = std::min(host_end, (start | PageMask) + 1u);
		if (mapped_pages.Load(start >> PageBits) != 0) {
			DWORD previous = 0;
			if (VirtualProtect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(start)), stop - start, host_protection, &previous) == 0) {
				for (auto cursor = start; cursor < stop;) {
					MEMORY_BASIC_INFORMATION memory {};
					if (VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(cursor)), &memory, sizeof(memory)) != sizeof(memory)) return false;
					const auto region_end = std::min<uint64_t>(stop, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
					const bool read_only_view = memory.Type == MEM_MAPPED && (memory.AllocationProtect == PAGE_READONLY || memory.AllocationProtect == PAGE_EXECUTE_READ);
					const DWORD view_protection = !read_only_view ? host_protection
					                              : host_protection == PAGE_READWRITE ? PAGE_READONLY
					                              : host_protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ
					                                                                         : host_protection;
					if (memory.State == MEM_COMMIT && VirtualProtect(reinterpret_cast<void*>(static_cast<std::uintptr_t>(cursor)), region_end - cursor, view_protection, &previous) == 0) return false;
					cursor = region_end;
				}
			}
		}
		start = stop;
	}
	if (writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, true);
	return true;
#else
	if (!writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, false);
	const auto first = vaddr & ~uint64_t {0xfff};
	const auto end = (vaddr + size + 0xfff) & ~uint64_t {0xfff};
	std::vector<KeptRange> kept;
#if defined(__linux__)
	GuestThreadStackMappings_nid_no_patch(first, end, &KeepThreadStack, &kept);
#endif
	std::optional<GuestDeliveryLock<std::mutex>> holds;
	if (!writable) {
		holds.emplace(g_host_write_holds_mutex);
		for (auto& hold : g_host_write_holds) {
			if (hold.pages.first >= end || hold.pages.end <= first) continue;
			hold.overridden = true;
			kept.push_back({hold.pages, PageKeep::Writable});
		}
	}
	const auto& pages = Pages();
	for (auto start = first; start < end;) {
		const bool mapped = pages.Load(start >> PageBits) != 0;
		auto stop = std::min(end, (start | PageMask) + 1u);
		while (stop < end && (pages.Load(stop >> PageBits) != 0) == mapped) {
			stop = std::min(end, stop + PageSize);
		}
		if (mapped && !ProtectHostPages(start, stop, protection, kept)) return false;
		start = stop;
	}
	if (writable) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(vaddr, size, true);
	return true;
#endif
}

bool IsGuestExecutable(uint64_t vaddr) {
	return Pages().Executable(vaddr);
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
		if (DoMapAnon_nid_no_patch(&address, (size + PageMask) & ~PageMask, GuestProtectionBits(mode), fixed ? GuestMapFixed : 0, PageSize, fixed) != 0) return 0;
	} catch (const std::exception& error) {
		std::fprintf(stderr, "KytyShim memory: runtime memory \"%s\" of 0x%" PRIx64 " bytes not mapped: %s\n", name, size, error.what());
		return 0;
	}
	return reinterpret_cast<std::uintptr_t>(address);
}

bool FreeGuestMemory(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - PageMask) return false;
	try {
		return DoMunmap_nid_no_patch(reinterpret_cast<void*>(static_cast<std::uintptr_t>(vaddr)), (size + PageMask) & ~PageMask) == 0;
	} catch (const std::exception& error) {
		Shim::Fail("KytyShim memory: freeing runtime memory 0x%016" PRIx64 "+0x%" PRIx64 " failed: %s\n", vaddr, size, error.what());
	}
}


}
