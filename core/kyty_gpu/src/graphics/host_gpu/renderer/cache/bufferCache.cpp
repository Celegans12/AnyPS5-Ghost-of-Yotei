#include "graphics/host_gpu/renderer/dispatchCensus.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "common/envRead.h"
#include "common/shaderEnv.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/bufferCacheTrace.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveAb.h"
#include "common/threads.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/stringUtils.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/cache/textureCacheShared.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <optional>
#include <atomic>
#include <thread>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>
#include <cctype>
#include <cstring>
#include <memory>
#include <array>
#include <mutex>
#include <semaphore>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <xxhash.h>


#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"

#include "graphics/host_gpu/queueLockSite.h"
namespace Libs::Graphics {
// Two-phase uploads. A read-only synchronization of a cached buffer normally clears the CPU-dirty bits of its pages,
// write-protects them (one page-manager update, i.e.
// one or more protection syscalls, per tracker region it touches) and copies the bytes, all in one call. A PrepareBda
// region pass and the buffer bindings of one draw or dispatch (RenderExecutor::RebindBuffers) make many such calls
// over the same 4 MiB regions (the data rings). Inside such a scope (BufferCache::UploadBatchScope) each call only
// claims its pages (clears the bits, notes the page runs, leaves them writable); the batch commit then write-protects
// every claimed page with one update per region, in ascending region order, and only after that stages the bytes and
// records the uploads, in claim order. The order "clear the dirty bits, write-protect, copy" holds for every page: a
// guest write before the protection lands in guest memory without a fault and is copied, a write after it faults and
// makes the page CPU-dirty again. Everything that must see the uploaded bytes in a cached buffer, or that changes the
// buffers, commits first: a written or texel-buffer synchronization, an image source (ObtainBufferForImage), buffer
// creation and deletion (joins, the collector), an image copy into a buffer, a BDA store report, a CP write-through.
// The bindings of a draw are only used by the draw itself, recorded after the scope.

// An upload batch commit copies first, then write-protects, then re-checks. Each staged range is read once into scratch,
// hashed and copied to the staging buffer;
// after CommitDeferredProtections the guest bytes are hashed again, and a range whose bytes changed (a guest write
// between the copy and the protection, which did not fault) is marked CPU-dirty, so the next synchronization uploads it
// again. A write after the protection faults as before. The bytes this commit's copy carries are those of the copy
// instead of those at the protection: both are points in the same race with the writer. This order lets the protection
// and the re-check leave the GPU thread.

// KYTY_TRACE_PERF: commits, re-checked ranges and their bytes, ranges that changed (one line every 4096 commits).
static std::atomic<uint64_t> g_pac_commits {0};
static std::atomic<uint64_t> g_pac_ranges {0};
static std::atomic<uint64_t> g_pac_bytes {0};
static std::atomic<uint64_t> g_pac_changed {0};

} // namespace Libs::Graphics

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

// KYTY_STAGING_BUFFER_MB / KYTY_STREAM_BUFFER_MB: sizes of the upload and stream rings (MiB). A ring
// the work of a frame does not fit in waits for the GPU to release space.
uint64_t RingSize(const char* name, uint64_t fallback) {
	const char* value = std::getenv(name);
	const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
	return mib != 0 ? mib : fallback;
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, Staging().Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = Staging().Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(Staging().Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

// ---- the BDA fallback (M06) ----------------------------------------------------------------------------------
namespace {
// The import granule of the direct-memory backing. A granule is imported (pinned, counted as the card's shared memory)
// only once the GPU uses it: a shader read the fault pass reports as 0, a direct download into it, or the game's
// granule list from earlier sessions (map §3m, 3.11: one Snake cutscene used ~160 granules, ~330 MB, of the 12 GB the
// whole backing in 256 MiB chunks pinned).
constexpr uint64_t FallbackChunkBytes = uint64_t {2} << 20u;
constexpr uint64_t FallbackMaxChunks  = 1536; // 3 GB pinned at most (and Vulkan's allocation count)
} // namespace

int BufferCache::BdaEntryShareMode() {
	static const int mode = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_PERF_BDA_ENTRY_SHARE");
		// Default 1 for the shaders of KYTY_PERF_BDA_ENTRY_SHARE_HASHES (default: cloth fc6f8c56; =2
		// verify 24.5M checks, 0 mismatches; 0.612 -> 0.586 ms per dispatch); =0 off.
		const int   m     = value == nullptr ? 1 : value[0] == '1' ? 1 : value[0] == '2' ? 2 : 0;
		if (m != 0) std::fprintf(stderr, "PERF_BDA_ENTRY_SHARE enabled mode=%d\n", m);
		return m;
	}();
	return mode;
}

void BufferCache::QueueFallbackMap(uint64_t vaddr, uint64_t size) {
	if (size == 0) return;
	std::scoped_lock lock(m_fallback_pending_mutex);
	m_fallback_pending.emplace_back(vaddr, size);
}

// The fallback entry of a 16 KiB guest page: the device address of its bytes in the imported backing, bit 0 set.
bool BufferCache::FallbackEntry(uint64_t page, vk::DeviceAddress* entry) {
	uint64_t offset = 0;
	if (!LibKernel::Memory::GuestBackingOffset(page, CACHING_PAGESIZE, &offset)) return false;
	const uint64_t total = LibKernel::Memory::GuestBackingSize();
	if (LibKernel::Memory::GuestBackingBase() == 0 || offset >= total) return false;
	const size_t chunk  = static_cast<size_t>(offset / FallbackChunkBytes);
	Buffer*      import = FallbackChunk(chunk, false);
	if (import == nullptr) return false;
	const uint64_t in_chunk = offset - static_cast<uint64_t>(chunk) * FallbackChunkBytes;
	if (in_chunk + CACHING_PAGESIZE > import->Size()) return false;
	*entry = (import->BufferDeviceAddress() + in_chunk) | vk::DeviceAddress {1};
	return true;
}

// The game's granule list: _PipelineCache/<title id>_<version>.bda_granules, one granule index per line (the granules
// earlier sessions used; imported when mapped, so their first reads are served too). Read once, on the GPU thread.
void BufferCache::LoadFallbackList() {
	m_fallback_list_loaded = true;
	const uint64_t total   = LibKernel::Memory::GuestBackingSize();
	const size_t   count   = static_cast<size_t>((total + FallbackChunkBytes - 1) / FallbackChunkBytes);
	m_fallback_chunks.resize(count);
	m_fallback_chunk_failed.resize(count, 0);
	m_fallback_wanted.resize(count, 0);
	std::string title, version;
	if (!::Loader::SystemContentParamSfoGetString("TITLE_ID", &title) || title.empty() ||
	    !std::ranges::all_of(title, [](unsigned char c) { return std::isalnum(c) != 0; }))
		return;
	(void)::Loader::SystemContentParamSfoGetString("APP_VER", &version);
	std::ranges::replace_if(version, [](unsigned char c) { return std::isalnum(c) == 0; }, '_');
	m_fallback_list_path = std::filesystem::path("_PipelineCache") / (title + "_" + version + ".bda_granules");
	if (std::FILE* f = std::fopen(Common::PathToString(m_fallback_list_path).c_str(), "r"); f != nullptr) {
		unsigned long long g = 0;
		while (std::fscanf(f, "%llu", &g) == 1) {
			if (g < count && m_fallback_wanted[g] == 0) {
				m_fallback_wanted[g] = 1;
				m_fallback_stats.listed++;
			}
		}
		std::fclose(f);
	}
	std::fprintf(stderr, "FIX_BDA_FALLBACK granule list %s: %llu granules of %zu\n",
	             Common::PathToString(m_fallback_list_path).c_str(), static_cast<unsigned long long>(m_fallback_stats.listed),
	             count);
}

// The imported backing granule `chunk` (FallbackChunkBytes each). Imported when `want` (the GPU uses it now: a fault,
// a direct download) or the game's list names it, and then added to the list; null when it is neither, or the driver
// refuses it (remembered and counted: the granule's pages then miss as without the fallback). The direct downloads into
// the backing take their imports from here (one import of each host range).
Buffer* BufferCache::FallbackChunk(size_t chunk, bool want) {
	const uint64_t base  = LibKernel::Memory::GuestBackingBase();
	const uint64_t total = LibKernel::Memory::GuestBackingSize();
	if (base == 0 || static_cast<uint64_t>(chunk) * FallbackChunkBytes >= total) return nullptr;
	if (!m_fallback_list_loaded) LoadFallbackList();
	if (!m_fallback_chunks[chunk]) {
		if (m_fallback_chunk_failed[chunk] != 0 || (!want && m_fallback_wanted[chunk] == 0)) return nullptr;
		if (m_fallback_stats.chunks >= FallbackMaxChunks) {
			m_fallback_chunk_failed[chunk] = 1;
			m_fallback_stats.failed++;
			return nullptr;
		}
		if (m_fallback_wanted[chunk] == 0) {
			m_fallback_wanted[chunk] = 1;
			m_fallback_stats.learned++;
			if (!m_fallback_list_path.empty()) {
				std::error_code error;
				std::filesystem::create_directories(m_fallback_list_path.parent_path(), error);
				if (std::FILE* f = std::fopen(Common::PathToString(m_fallback_list_path).c_str(), "a"); f != nullptr) {
					std::fprintf(f, "%zu\n", chunk);
					std::fclose(f);
				}
			}
		}
		const uint64_t start = static_cast<uint64_t>(chunk) * FallbackChunkBytes;
		const uint64_t bytes = std::min(FallbackChunkBytes, total - start) & ~uint64_t {0xfff};
		if (bytes == 0 || !Buffer::CanImportGuestMemory(m_graphics, base + start, bytes)) {
			m_fallback_chunk_failed[chunk] = 1;
			m_fallback_stats.failed++;
			std::fprintf(stderr, "FIX_BDA_FALLBACK chunk %zu (backing 0x%llx+0x%llx) cannot be imported\n", chunk,
			             static_cast<unsigned long long>(start), static_cast<unsigned long long>(bytes));
			return nullptr;
		}
		const auto t0     = std::chrono::steady_clock::now();
		bool       failed = false;
		auto       import = std::make_unique<Buffer>(m_graphics, m_scheduler, GuestDirectTag {&failed}, base + start, bytes);
		m_fallback_stats.import_us += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
		if (failed) { // D6 step K4: the driver refused the import (e.g. no more host memory it can lock)
			m_fallback_chunk_failed[chunk] = 1;
			m_fallback_stats.failed++;
			std::fprintf(stderr, "FIX_BDA_FALLBACK chunk %zu (backing 0x%llx+0x%llx) import refused by the driver\n", chunk,
			             static_cast<unsigned long long>(start), static_cast<unsigned long long>(bytes));
			return nullptr;
		}
		m_fallback_chunks[chunk] = std::move(import);
		m_fallback_stats.chunks++;
		m_fallback_stats.imported_bytes += bytes;
	}
	return m_fallback_chunks[chunk].get();
}

void BufferCache::WantFallback(uint64_t page) {
	page            = page & ~(CACHING_PAGESIZE - 1);
	uint64_t offset = 0;
	if (!LibKernel::Memory::GuestBackingOffset(page, CACHING_PAGESIZE, &offset)) return;
	if (FallbackChunk(static_cast<size_t>(offset / FallbackChunkBytes), true) == nullptr) return;
	// The guest range that maps the granule where `page` maps it (WriteFallbackRange checks each page's own granule;
	// another view of the granule gets its entries when it is mapped again or faults itself).
	const uint64_t in_granule = offset % FallbackChunkBytes;
	if (page < in_granule) return;
	WriteFallbackRange(page - in_granule, FallbackChunkBytes);
}

// Fallback entries for the pages of [vaddr, vaddr + size) that hold no buffer (one table write per run of pages).
void BufferCache::WriteFallbackRange(uint64_t vaddr, uint64_t size) {
	const uint64_t                 end = vaddr + size;
	std::vector<vk::DeviceAddress> run;
	uint64_t                       run_start = 0;
	const auto                     flush     = [&]() {
		if (run.empty()) return;
		WriteDataBuffer(m_bda_pagetable_buffer, PageIndex(run_start) * sizeof(vk::DeviceAddress), run.data(),
		                run.size() * sizeof(vk::DeviceAddress));
		run.clear();
	};
	for (uint64_t page = vaddr & ~(CACHING_PAGESIZE - 1); page < end; page += CACHING_PAGESIZE) {
		vk::DeviceAddress entry = 0;
		if (!GuestRange {page, CACHING_PAGESIZE}.Valid() || IsRegionRegistered(page, CACHING_PAGESIZE) ||
		    !FallbackEntry(page, &entry)) {
			flush();
			continue;
		}
		if (!run.empty() && PageIndex(page) != PageIndex(run_start) + run.size()) flush();
		if (run.empty()) run_start = page;
		run.push_back(entry);
		if (m_fallback_pages.insert(page).second) m_fallback_stats.pages++;
	}
	flush();
}

bool BufferCache::WasFallbackPage(uint64_t address, uint32_t frames) const {
	// Entries older than 16 frames may or may not be pruned yet (ChangeRegister): the answer is the same either way only
	// while no caller asks about more than 16 frames.
	EXIT_IF(frames > 16u);
	const auto found = m_fallback_replaced.find(address & ~(CACHING_PAGESIZE - 1));
	return found != m_fallback_replaced.end() && DebugPresentCount().load() - found->second <= frames;
}

void BufferCache::ApplyFallbackMaps() {
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	{
		std::scoped_lock lock(m_fallback_pending_mutex);
		pending.swap(m_fallback_pending);
	}
	for (const auto& [vaddr, size]: pending) {
		m_fallback_stats.maps++;
		WriteFallbackRange(vaddr, size);
	}
	// Checked every 1024 calls; printed at most every 5 s and only when a value changed (D6b: every 1024 calls it was half
	// of a long session's log).
	if ((++m_fallback_stats.applies & 0x3ffu) == 0u) {
		const auto now = std::chrono::steady_clock::now();
		auto       f   = m_fallback_stats;
		f.applies      = m_fallback_printed.applies;
		if (now - m_fallback_print_time >= std::chrono::seconds(5) && !(f == m_fallback_printed)) {
			m_fallback_print_time = now;
			m_fallback_printed    = m_fallback_stats;
			std::fprintf(stderr,
			             "FIX_BDA_FALLBACK stats maps=%llu pages=%llu live=%zu replaced=%llu restored=%llu cleared=%llu chunks=%llu "
			             "failed=%llu listed=%llu learned=%llu imported_MiB=%llu import_ms=%llu clear_fills=%llu clear_ms=%llu\n",
			             static_cast<unsigned long long>(f.maps), static_cast<unsigned long long>(f.pages),
			             m_fallback_pages.size(), static_cast<unsigned long long>(f.replaced),
			             static_cast<unsigned long long>(f.restored), static_cast<unsigned long long>(f.cleared),
			             static_cast<unsigned long long>(f.chunks), static_cast<unsigned long long>(f.failed),
			             static_cast<unsigned long long>(f.listed), static_cast<unsigned long long>(f.learned),
			             static_cast<unsigned long long>(f.imported_bytes >> 20u),
			             static_cast<unsigned long long>(f.import_us / 1000u), static_cast<unsigned long long>(f.clear_fills),
			             static_cast<unsigned long long>(f.clear_us / 1000u));
		}
	}
}

// ClearFallback zeroes the fallback entries of an unmapped range with one Fill per run of consecutive page-table
// entries (D6b): until then one Fill, each with its two barriers, per 16 KiB page - up to 94,000 Fills per
// 5 s at scene changes in a 34-minute session. The same entries are zeroed in the same place of the command stream
// (nothing was recorded between the per-page Fills either); checked: 0 mismatches at the gate, open world (also walking),
// night, night RT, the Atsu house and a run from the start.
void BufferCache::ClearFallback(uint64_t vaddr, uint64_t size) {
	if (m_fallback_pages.empty()) return;
	const auto     t0   = std::chrono::steady_clock::now();
	uint64_t   run_first = 0, run_count = 0;
	const auto flush     = [&]() {
		if (run_count == 0) return;
		m_bda_pagetable_buffer.Fill(run_first * sizeof(vk::DeviceAddress), run_count * sizeof(vk::DeviceAddress), 0);
		m_fallback_stats.clear_fills++;
		run_count = 0;
	};
	for (uint64_t page = vaddr & ~(CACHING_PAGESIZE - 1); page < vaddr + size; page += CACHING_PAGESIZE) {
		if (m_fallback_pages.erase(page) == 0) continue;
		m_fallback_stats.cleared++;
		if (IsRegionRegistered(page, CACHING_PAGESIZE)) continue; // a buffer's entry is already there
		const uint64_t entry = PageIndex(page);
		if (run_count != 0 && entry != run_first + run_count) flush();
		if (run_count == 0) run_first = entry;
		run_count++;
	}
	flush();

	m_fallback_stats.clear_us += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	{
		const LookupWriteGuard lookup_guard(*this); // the resolve thread's lookups: the resolve thread's LookupBuffer reads these
		for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
			if constexpr (insert) {
				m_page_table[page] = id;
			} else {
				m_page_table[page] = {};
			}
		}
		// The resolve thread's lookups (under the guard: the resolve thread reads it there)
		DrawRepeat::NoteBufferRegistryChange();
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		m_registered_bytes += buffer.Size();
		if (!buffer.IsDirect()) MarkCpuDirtyRange(buffer.CpuAddress(), buffer.Size()); // KYTY_PERF_DIRECT_RANGES
		g_cpu_dirty_epoch.fetch_add(1, std::memory_order_release);
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		// Insert gave the item the current tick: a touch at it is a no-op (TouchBuffer).
		buffer.lru_touch_tick = m_gc_tick;
		if (!m_fallback_pages.empty()) { // The BDA fallback: the buffer's entries replace them
			for (uint64_t page = buffer.CpuAddress() & ~(CACHING_PAGESIZE - 1); page < buffer.CpuAddress() + buffer.Size();
			     page += CACHING_PAGESIZE) {
				if (m_fallback_pages.erase(page) != 0) {
					m_fallback_stats.replaced++;
					// D6 step K4: remembered for the fault pass of the work recorded before (WasFallbackPage).
					const uint32_t frame = DebugPresentCount().load();
					// Pruned at most once per presented frame: a second pass in the same frame finds nothing older than
					// 16 frames. Pruning per page (the earlier way) rescanned the >= 65,536 entries for every page
					// while a load replaced more pages than that within 16 frames and no frame was presented: the
					// minutes-long stalls after loading the Snake checkpoint (BufferCache::ChangeRegister at 100%).
					if (m_fallback_replaced.size() >= 65536u && frame != m_fallback_replaced_pruned) {
						m_fallback_replaced_pruned = frame;
						std::erase_if(m_fallback_replaced, [frame](const auto& entry) { return frame - entry.second > 16u; });
					}
					m_fallback_replaced[page] = frame;
				}
			}
		}
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		// (Upstream queues the buffer for its BDA sync here, d58aefde4; RenderContext::PrepareBda's region pass takes
		// the regions that got a buffer instead.)
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_registered_bytes -= std::min(m_registered_bytes, buffer.Size());
		{
			m_lru_cache.Free(buffer.lru_id);
			// Set with Free under the LRU lock: a thread past TouchBuffer's first check re-checks it under this
			// lock, so the freed item is never linked back into the list (the GC then met a deleted buffer, 21.10).
			buffer.is_deleted = true;
		}
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
// The BDA fallback: still-mapped pages read guest memory again, not zeros
		const auto before = m_fallback_stats.pages;
		WriteFallbackRange(buffer.CpuAddress(), buffer.Size());
		m_fallback_stats.restored += m_fallback_stats.pages - before;
	}
}

void BufferCache::TouchBuffer(Buffer& buffer) {
	if (!buffer.is_deleted) {
		// m_gc_tick only rises (RunGarbageCollector) and an LRU item's tick never drops, so a buffer
		// touched (or registered) at this tick already holds it: Touch would return at its first check,
		// after a cache-missing deque lookup.
		if (buffer.lru_touch_tick == m_gc_tick) return;
		if (buffer.is_deleted) return;  // unregistered meanwhile on the GPU thread (set under this lock with Free)
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
		buffer.lru_touch_tick = m_gc_tick;
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	CommitUploadBatch(BatchCommit::DeleteBuffer); // claimed uploads may target this buffer
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] {
			const LookupWriteGuard lookup_guard(*this); // the resolve thread's lookups
			m_slot_buffers.erase(id);
		});
	} else {
		const LookupWriteGuard lookup_guard(*this); // the resolve thread's lookups
		m_slot_buffers.erase(id);
	}
}


// KYTY_DIAG_GPU_BURN_AT=readback (measurement only, with KYTY_DIAG_GPU_BURN_NS or its live switch; renderDraw.cpp): one
// GPU-thread burn per guest read of GPU-written data the GPU thread serves (a download that copied GPU bytes, a wait for
// a pending eager readback, an exact walker read that publishes GPU bytes). Other sites: one comparison per event.
static void ReadbackBurn() {
	if (RenderExecutor::GpuBurnSelected() == RenderExecutor::GpuBurnSite::Readback && GuestGpu::IsGpuThread()) {
		RenderExecutor::GpuBurnAt(RenderExecutor::GpuBurnSite::Readback);
	}
}

// ---- The download into the backing ----------------------------------------------------------------------------------
// A synchronous download that takes the copy path (TryBarDownload declined: the read window holds more than BarMaxBytes
// of GPU bytes, or they are in the open command buffer) copies its GPU-dirty intervals with the GPU straight into the
// guest backing - the host view of direct memory, imported in the BDA fallback's chunks (VK_EXT_external_memory_host) -
// instead of into the download ring followed by a CPU copy
// (WriteBacking) after the wait. The request still waits for the same tick before it returns and unmarks its pages, and
// the pages are no-access to the guest until then (GPU-modified), so no guest thread can see the bytes earlier. Only the
// emulator's own readers of the backing view could (none reads GPU-dirty bytes there). An interval outside one guest
// mapping, a chunk the driver does not import, or more than DownloadToBackingMaxChunks imported chunks (512 MiB pinned)
// sends the whole download down the download ring (counted). Imported chunks stay for the process (no eviction, so no
// copy can outlive its target). KYTY_TRACE_PERF: the PERF download_to_backing line every ~5 s.
namespace {


struct DownloadToBackingStats {
	std::atomic<uint64_t> downloads {0};      // copy-path downloads
	std::atomic<uint64_t> direct {0};         // ... copied straight into the backing
	std::atomic<uint64_t> direct_kb {0};      // ... their bytes (KiB)
	std::atomic<uint64_t> intervals {0};      // ... their GPU-dirty intervals
	std::atomic<uint64_t> chunk_copies {0};   // ... copyBuffer calls (one per imported chunk touched)
	std::atomic<uint64_t> fallback_map {0};   // an interval outside one guest mapping (or no backing view)
	std::atomic<uint64_t> fallback_import {0}; // a chunk the driver does not import
	std::atomic<int64_t>  window_start {0};
};
DownloadToBackingStats g_dtb;

void ReportDownloadToBacking() {
	if (!PerfEnabled()) return;
	auto&         s   = g_dtb;
	const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	int64_t start = s.window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		s.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 5000 || !s.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) return;
	const auto take = [](std::atomic<uint64_t>& counter) {
		return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed));
	};
	std::fprintf(stderr,
	             "PERF download_to_backing downloads=%llu direct=%llu direct_kb=%llu intervals=%llu chunk_copies=%llu "
	             "fallback_map=%llu fallback_import=%llu (per ~5 s)\n",
	             take(s.downloads), take(s.direct), take(s.direct_kb), take(s.intervals), take(s.chunk_copies),
	             take(s.fallback_map), take(s.fallback_import));
}
} // namespace

bool BufferCache::PlanDownloadToBacking(uint64_t buffer_address, const std::vector<vk::BufferCopy>& copies,
                                        DownloadToBackingPlan& plan) {
	auto&          s     = g_dtb;
	const uint64_t total = LibKernel::Memory::GuestBackingSize();
	if (LibKernel::Memory::GuestBackingBase() == 0 || total == 0) {
		s.fallback_map.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	plan.items.reserve(copies.size());
	for (const auto& copy: copies) {
		const uint64_t vaddr   = buffer_address + copy.srcOffset;
		uint64_t       backing = 0;
		// One guest mapping holds the whole interval, so its backing bytes are contiguous (as TryWriteBacking writes them).
		if (!LibKernel::Memory::GuestBackingOffset(vaddr, copy.size, &backing) || backing > total ||
		    copy.size > total - backing) {
			s.fallback_map.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		plan.items.push_back({vaddr, backing, copy.size});
		// D6 step K4: the BDA fallback's import table serves the copies too, so no host range is imported twice; a granule
		// a download needs is imported now (and so serves first reads afterwards).
		const uint64_t granule = FallbackChunkBytes;
		for (uint64_t done = 0; done < copy.size;) {
			const uint64_t at         = backing + done;
			const uint64_t chunk      = at / granule;
			const uint64_t chunk_base = chunk * granule;
			const uint64_t bytes      = std::min(copy.size - done, chunk_base + granule - at);
			Buffer*        target     = FallbackChunk(static_cast<size_t>(chunk), true);
			// A tail chunk is shorter (whole pages only).
			if (target == nullptr || at + bytes > chunk_base + target->Size()) {
				s.fallback_import.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			auto it = std::ranges::find_if(plan.chunks, [&](const auto& entry) { return entry.first == target; });
			if (it == plan.chunks.end()) {
				plan.chunks.emplace_back(target, std::vector<vk::BufferCopy> {});
				it = plan.chunks.end() - 1;
			}
			it->second.emplace_back(copy.srcOffset + done, at - chunk_base, bytes);
			done += bytes;
		}
	}
	return true;
}

// KYTY_DIAG_SKIP_READBACK=1 (ceiling test only, map 5.11 "ruch przez miasto": the read-back of GPU-written bytes for the
// guest was ~20% of the GPU thread in the town): a synchronous download changes the states as after the copy but copies
// nothing and waits for nothing - the game reads stale guest memory (the image and what it draws may change; compare the
// draws per frame). Read once.
static bool SkipReadbackCeiling() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_SKIP_READBACK");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "DIAG_SKIP_READBACK ceiling: synchronous downloads copy nothing\n");
		return set;
	}();
	return on;
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	KYTY_PROFILER_BLOCK("BufferCache::DownloadBufferMemory");
	if constexpr (!async) {
		if (SkipReadbackCeiling()) {
			bool any = false;
			m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
				m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t, uint64_t) { any = true; });
				GpuDirtySubtract(address, bytes);
			});
			return any;
		}
		if (!buffer.Mapped().empty()) {
			DownloadWaitTrace::g_last_bar_reason = -1;
			if (TryBarDownload(buffer, vaddr, size) == 1) return true;
		}
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	KYTY_PROFILER_BLOCK("Download::Record");
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    GpuDirtySubtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}
	if (UploadClobberCheck()) { // D4 step K0
		for (const auto& copy: copies) C0Note(buffer_address + copy.srcOffset, copy.size, C0Event::CopyRead, m_scheduler.CurrentTick());
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	// A synchronous download copies straight into the imported backing and skips the download ring (above); one it cannot
	// take, and an asynchronous one, goes down the ring.
	bool                  direct = false;
	DownloadToBackingPlan direct_plan;
	if constexpr (!async) {
		g_dtb.downloads.fetch_add(1, std::memory_order_relaxed);
		direct = PlanDownloadToBacking(buffer_address, copies, direct_plan);
	}
	const bool ring = !direct;

	uint8_t*                mapped = nullptr;
	uint64_t                offset = 0;
	std::unique_ptr<Buffer> temporary;
	if (ring) {
		const auto [ring_mapped, ring_offset] = m_download_buffer.Map(total_size, 64);
		mapped = ring_mapped;
		offset = ring_offset;
		if (mapped == nullptr) {
			temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
			                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
			mapped = temporary->Mapped().data();
		} else {
			m_download_buffer.Commit();
		}
		for (auto& copy: copies) {
			copy.dstOffset += offset;
		}
	}
	Buffer& download = temporary ? *temporary : static_cast<Buffer&>(m_download_buffer);

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	const bool              label  = GpuTimer::LabelsEnabled(); // KYTY_TRACE_VK_LABELS
	if (label) GpuTimer::LabelEmu(native, "buffer download");
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	GpuCheckpoints::Mark(native, 0xFEED000000000000ull | (1ull << 16u) | 161u);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	if (ring) {
		native.copyBuffer(buffer.Handle(), download.Handle(),
		                  static_cast<uint32_t>(copies.size()), copies.data());
		after.buffer = download.Handle();
		after.offset = offset;
		after.size   = total_size;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands |
		                           vk::PipelineStageFlagBits::eHost,
		                       {}, 0, nullptr, 1, &after, 0, nullptr);
	}
	// The intervals into the imported backing chunks, made visible to the host the same way.
	if (direct) {
		uint64_t bytes = 0;
		for (const auto& item: direct_plan.items) bytes += item.size;
		for (auto& [target, regions]: direct_plan.chunks) {
			native.copyBuffer(buffer.Handle(), target->Handle(), static_cast<uint32_t>(regions.size()), regions.data());
			after.buffer = target->Handle();
			after.offset = 0;
			after.size   = target->Size();
			native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0,
			                       nullptr, 1, &after, 0, nullptr);
		}
		g_dtb.direct.fetch_add(1, std::memory_order_relaxed);
		g_dtb.direct_kb.fetch_add(bytes >> 10u, std::memory_order_relaxed);
		g_dtb.intervals.fetch_add(direct_plan.items.size(), std::memory_order_relaxed);
		g_dtb.chunk_copies.fetch_add(direct_plan.chunks.size(), std::memory_order_relaxed);
	}
	if (label) GpuTimer::LabelEnd(native);
	auto publish = [this, mapped, offset, total_size, buffer_address, ring,
	                copies = std::move(copies), owner = std::move(temporary)] {
		KYTY_PROFILER_BLOCK("Download::Publish");
		if (!ring) return; // the GPU wrote the backing (see the sync branch)
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			SyncTimeline::NoteDownload(buffer_address + copy.srcOffset, mapped + (copy.dstOffset - offset), copy.size);
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto              tick       = m_scheduler.CurrentTick();
		const auto              wait_start = std::chrono::steady_clock::now();
		SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Copy); // KYTY_TRACE_SYNC_TIMELINE
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		if (DownloadWaitTrace::Enabled()) {
			using namespace DownloadWaitTrace;
			const int r    = g_last_bar_reason;
			const Kind kind = r == 1 ? BarNoTick : r == 2 ? BarOpen : r == 3 ? BarPending
			                : r == 4 ? (g_last_bar_old ? BarLargeOld : BarLargeOpen) : r == 5 ? BarOutside
			                : r == 6 ? BarEmpty : NoBar;
			Add(kind, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                    std::chrono::steady_clock::now() - wait_start).count()), total_size);
		}
		if (direct) {
			// The bytes are in the backing; only the observers of a publish see them.
			// (KYTY_TRACE_SYNC_TIMELINE's changed-byte counts compare with guest memory, which already holds them: 0.)
			const uint64_t base = LibKernel::Memory::GuestBackingBase();
			for (const auto& item: direct_plan.items) {
				const auto* data = reinterpret_cast<const uint8_t*>(base + item.backing);
				SyncTimeline::NoteDownload(item.vaddr, data, item.size);
			}
		}
		publish();
		ReportDownloadToBacking();
	}
	return true;
}

namespace {
BufferCache* g_c0_census = nullptr; // D4 step K0: the cache the tracker hook reports to
[[gnu::noinline]] void C0SiteAnchor() {
	std::fprintf(stderr, "anchor\n");
}
} // namespace

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, RingSize("KYTY_STAGING_BUFFER_MB", 512) * MiB),
      // The ring is only written by the CPU (sequential memcpy), so it lives in host memory like the upload ring. The GDS
      // and bindless buffers keep their placement.
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, RingSize("KYTY_STREAM_BUFFER_MB", 64) * MiB, true),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	// D4b step 1 (append windows): open append windows get their closing counter copy at every submit.
	m_scheduler.SetPreSubmitHook([this] { AppendCloseWindows(); });
	// D4 step K0 (KYTY_D6_C0=2/3): the tracker's GPU-modified changes go into the page event log.
	if (UploadClobberCheck()) {
		g_c0_census                  = this;
		MemoryTracker::s_census_hook = [](uint64_t vaddr, uint64_t size, uint32_t kind, uint64_t site) {
			constexpr std::array<C0Event, 4> map {C0Event::Mark, C0Event::CleanMark, C0Event::Unmark, C0Event::Release};
			g_c0_census->C0Note(vaddr, size, map[kind & 3u], site);
		};
		// The tag of a tracker event is a return address: this anchor's address lets the link map name the caller.
		std::fprintf(stderr, "D6_C0 census site anchor C0SiteAnchor=0x%llx\n",
		             static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&C0SiteAnchor)));
	}
	// The region epochs of the obtain and write memos count from here on (entries are kept only after this).
	if (g_cpu_dirty_region_epochs.load(std::memory_order_acquire) == nullptr) {
		g_cpu_dirty_region_epochs.store(new std::atomic<uint32_t>[CPU_DIRTY_REGION_COUNT] {}, std::memory_order_release);
	}
	g_gpu_clean_region_epochs.store(new std::atomic<uint32_t>[CPU_DIRTY_REGION_COUNT] {}, std::memory_order_release);
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	// The BDA page table and the fault buffer are zeroed at the first CommandScheduler::Begin (ClearDeviceTablesOnce;
	// upstream 4644c5292 opens a command buffer here instead).
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	m_memory_tracker.InvalidateRegion(vaddr, size, [this, vaddr, size] {
		// KYTY_DIAG_GPU_BURN_AT=invalidate_caller (measurement only): extra wait of the faulting guest thread.
		if (!GuestGpu::IsGpuThread()) RenderExecutor::GpuBurnAt(RenderExecutor::GpuBurnSite::InvalidateCaller);
		ReadMemory(vaddr, size, true);
	});
}

namespace {
// ReadWindowBytes: the half-width of the fault read window, the GPU-modified bytes of the
// read's buffer around a CPU read that the read downloads at once (ReadMemory, its host-mapped and caller-wait paths).
// Only how much is published ahead changes: bytes outside the window stay GPU-modified and synchronize when accessed,
// so every read still gets the newest GPU bytes. The open-world profile (18.10) shows the BAR copy at ~10% of the GPU
// thread for ~16 MB/5 s of bytes the game needs.
constexpr uint64_t ReadWindowBytes = uint64_t {4096} << 10u;

// A CPU write into a GPU-modified page whose GPU bytes the write request (ReadMemory with is_write) published (download)
// or dropped (DropDeadRingBytes) unmarks the page and marks it CPU-dirty under one hold of the region's lock, and the
// two protection updates, which only open the page, run as one page-manager pass (MemoryTracker::ReleaseGpuToCpu): one
// protection call (no access -> read-write) instead of two (no access -> read-only -> read-write). Watcher counts, page
// modes, bits, epochs and the deferred CPU-dirty marks (the BDA sync epoch) are those of the two updates; only the ~1 us
// between the two calls in which the page was read-only is gone (a read in it now faults and is served as any read
// fault). KYTY_TRACE_PERF: the PERF invalidate_protect_merge line every ~5 s.
struct InvalidateProtectMergeStats {
	std::atomic<uint64_t> pairs {0};    // write requests with a pair to apply
	std::atomic<uint64_t> merged {0};   // ... applied as one pass
	std::atomic<uint64_t> fallback {0}; // ... applied as two updates (a pending access-side closing change)
	std::atomic<int64_t>  window_start {0};
};
InvalidateProtectMergeStats g_invalidate_protect_merge;

void NoteInvalidateProtectMerge(bool merged) {
	auto& stats = g_invalidate_protect_merge;
	stats.pairs.fetch_add(1, std::memory_order_relaxed);
	(merged ? stats.merged : stats.fallback).fetch_add(1, std::memory_order_relaxed);
	if (!PerfEnabled()) return;
	const int64_t now =
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	int64_t start = stats.window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		stats.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 5000 || !stats.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) return;
	const auto take = [](std::atomic<uint64_t>& counter) {
		return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed));
	};
	std::fprintf(stderr, "PERF invalidate_protect_merge pairs=%llu merged=%llu fallback=%llu (per ~5 s)\n",
	             take(stats.pairs), take(stats.merged), take(stats.fallback));
}
} // namespace

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	KYTY_PROFILER_BLOCK("BufferCache::ReadMemory");
	WaitDirectGpuWrites(vaddr, size); // KYTY_PERF_DIRECT_RANGES
	PerfScope perf_read(PerfKind::ReadMemory);
	PerfRecordRead(vaddr, size);
	const auto reason = is_write ? PerfReason::Invalidate : PerfCurrentReason();
	if (PerfEnabled()) PerfReasons()[static_cast<uint32_t>(reason)].calls.fetch_add(1, std::memory_order_relaxed);
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// KYTY_TRACE_PERF (PERF sync_time): how long the caller waited for the GPU thread to start the
	// command, and how long the GPU thread ran it.
	const bool perf_time = PerfEnabled();
	if (perf_time && GuestGpu::IsGpuThread()) {
		PerfReasons()[static_cast<uint32_t>(reason)].gpu_thread_calls.fetch_add(1, std::memory_order_relaxed);
	}
	// The caller wait (see PlanCallerWait): the request of a guest fault that the exception
	// handler retries, from another thread than the GPU thread, can be handed back by the GPU thread, before it
	// changes anything, when its bytes still wait for the GPU: the faulting thread then waits for that tick itself
	// and sends the request again, which is decided again from scratch. At most CallerMaxDeferrals hand-backs;
	// after them the request is served as before (the GPU thread waits).
	constexpr uint32_t CallerMaxDeferrals = 2;
	const bool         caller_eligible    = InRetriedFault() && !GuestGpu::IsGpuThread();
	if (caller_eligible) m_caller_stats.eligible.fetch_add(1, std::memory_order_relaxed);
	// BDA sync epoch: a guest thread's write fault (RenderContext::HandleFault) defers its CPU-dirty
	// mark; the marks below run on the GPU thread, so they are deferred explicitly. A handed-back attempt
	// (the caller wait) returns before any mark, so the completed attempt marks once.
	const bool deferred  = is_write && g_cpu_dirty_deferred;
	uint64_t   last_tick = 0;
	for (uint32_t attempt = 0;; ++attempt) {
		uint64_t   defer_tick  = 0;
		const bool allow_defer = caller_eligible && attempt < CallerMaxDeferrals;
		const auto call_start  = perf_time ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		// KYTY_TRACE_SYNC_TIMELINE: when, and from which thread, the command was sent.
		const auto timeline_call =
		    SyncTimeline::Enabled() ? SyncTimeline::Call(GuestGpu::IsGpuThread()) : SyncTimeline::CallInfo {};
		// Today's request: runs on the GPU thread (SendCommandSync below).
		auto request = [this, vaddr, size, is_write, reason, perf_time, call_start, timeline_call, caller_eligible,
		                allow_defer, attempt, last_tick, defer = &defer_tick, deferred] {
			const auto mark_cpu_written = [&] {
				CpuDirtyDeferScope defer_scope(deferred);
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			};
			SyncTimeline::RequestScope timeline(reason, vaddr, size, is_write, timeline_call);
			if (timeline.Active()) NoteSyncTimelineNeed(vaddr, size);
			const PerfReasonScope perf_reason(reason); // KYTY_TRACE_DOWNLOAD_WAITS: the request's reason on the GPU thread
			// The caller wait: GPU-thread time of the eligible attempts.
			struct CallerTimer {
				CallerWaitStats*                      stats    = nullptr;
				bool                                  deferred = false; // handed back
				bool                                  retry    = false;
				std::chrono::steady_clock::time_point start {};
				~CallerTimer() {
					if (stats == nullptr) return;
					const auto ns = static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
					stats->gpu_ns.fetch_add(ns, std::memory_order_relaxed);
					if (deferred) stats->gpu_defer_ns.fetch_add(ns, std::memory_order_relaxed);
					if (retry) {
						stats->gpu_retry_ns.fetch_add(ns, std::memory_order_relaxed);
						if (ns >= 1000000u) stats->retry_slow.fetch_add(1, std::memory_order_relaxed);
					}
				}
			} caller_timer;
			if (caller_eligible) {
				caller_timer.stats = &m_caller_stats;
				caller_timer.retry = attempt != 0;
				caller_timer.start = std::chrono::steady_clock::now();
			}
			struct ExecTimer {
				PerfReasonStats*                      stats = nullptr;
				std::chrono::steady_clock::time_point start {};
				~ExecTimer() {
					if (stats == nullptr) return;
					stats->exec_nanos.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                                                      std::chrono::steady_clock::now() - start)
					                                                      .count()),
					                            std::memory_order_relaxed);
				}
			} exec_timer;
			if (perf_time) {
				exec_timer.stats = &PerfReasons()[static_cast<uint32_t>(reason)];
				exec_timer.start = std::chrono::steady_clock::now();
				exec_timer.stats->queue_nanos.fetch_add(
				    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(exec_timer.start - call_start).count()),
				    std::memory_order_relaxed);
			}
			// A download that copied GPU-written bytes (and so waited for the GPU), timed per reason.
			const auto timed_download = [&](Buffer& target, uint64_t address, uint64_t bytes) {
				const auto download = [&] { return DownloadBufferMemory<false>(target, address, bytes); };
				if (!PerfEnabled()) {
					const bool copied = download();
					if (copied) ReadbackBurn(); // KYTY_DIAG_GPU_BURN_AT=readback (measurement only)
					return copied;
				}
				const auto start  = std::chrono::steady_clock::now();
				const bool copied = download();
				// KYTY_TRACE_GPU_READ_WAITS=<us> (diagnostic, with KYTY_TRACE_PERF; read once): the GPU thread's downloads
				// of GPU-written bytes that took at least that long - who read (reason, the draw's marker), what (the
				// requested range, the downloaded piece), how long - from frame 1100, first 2000 lines.
				static const uint32_t trace_us = [] {
					const char* value = std::getenv("KYTY_TRACE_GPU_READ_WAITS");
					return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
				}();
				if (trace_us != 0 && GuestGpu::IsGpuThread()) {
					const auto us = static_cast<uint64_t>(
					    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
					static std::atomic<uint32_t> lines {0};
					if (us >= trace_us && DebugPresentCount().load() >= 1100u && lines.fetch_add(1) < 2000u) {
						std::fprintf(stderr,
						             "GPU_READ_WAIT us=%llu copied=%d reason=%u addr=0x%llx size=0x%llx piece=0x%llx+0x%llx frame=%u "
						             "marker=[%s]\n",
						             static_cast<unsigned long long>(us), copied ? 1 : 0, static_cast<uint32_t>(reason),
						             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
						             static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes),
						             DebugPresentCount().load(), DebugMarkerPath().c_str());
					}
				}
				if (copied) {
					ReadbackBurn(); // KYTY_DIAG_GPU_BURN_AT=readback (measurement only)
					auto& stats = PerfReasons()[static_cast<uint32_t>(reason)];
					stats.drains.fetch_add(1, std::memory_order_relaxed);
					stats.drain_nanos.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
					                                                      std::chrono::steady_clock::now() - start)
					                                                      .count()),
					                            std::memory_order_relaxed);
				}
				return copied;
			};
			if (is_write && !IsRegionRegistered(vaddr, size)) {
				return;
			}
			// The caller wait: before this request changes anything, what it would wait for. Bytes that
			// still wait for the GPU hand it back: the open command buffer is submitted when that tick is its own (no
			// wait), and the faulting thread waits for the tick and for the priority completions up to it (asynchronous
			// publishes) itself. Serving the request later is what the GPU thread would do if the faulting thread had
			// been descheduled until then: the same code moves the same bytes and changes the same states, only after
			// the GPU finished them, while the GPU thread kept translating.
			if (caller_eligible) {
				const auto plan = PlanCallerWait(vaddr, size, is_write);
				if (plan.fallback != CallerFallback::None) {
					m_caller_stats.fallback[static_cast<uint32_t>(plan.fallback)].fetch_add(1, std::memory_order_relaxed);
				}
				if (plan.tick != 0) {
					if (attempt != 0 && plan.tick > last_tick) m_caller_stats.new_writer.fetch_add(1, std::memory_order_relaxed);
					if (allow_defer) {
						if (plan.tick >= m_scheduler.CurrentTick()) m_scheduler.Flush();
						*defer                = plan.tick;
						caller_timer.deferred = true;
						return;
					}
					m_caller_stats.exhausted.fetch_add(1, std::memory_order_relaxed);
				}
			}
			// A write whose page's GPU bytes were just published or dropped: in place of UnmarkRegionAsGpuModified +
			// mark_cpu_written.
			const auto release_gpu_to_cpu = [&] {
				CpuDirtyDeferScope defer_scope(deferred);
				NoteInvalidateProtectMerge(m_memory_tracker.ReleaseGpuToCpu(vaddr, size));
			};
			// A pending asynchronous copy of this window must publish before the CPU reads or
			// writes it; it may also already make the access fault-free.
			const bool pending = std::ranges::any_of(m_eager_readbacks, [&](const EagerReadback& entry) {
				return vaddr < entry.end && entry.start < vaddr + size;
			});
			ApplyEagerReadbacks(vaddr, size, true);
			if (!is_write && size == 1u) RecordHotRead(vaddr);
			if (!m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
				PerfCount(pending ? PerfKind::EagerWait : PerfKind::EagerHit);
				if (pending) ReadbackBurn(); // KYTY_DIAG_GPU_BURN_AT=readback (measurement only)
				if (is_write) mark_cpu_written();
				return;
			}
			// D4b step 1 (append windows): the append pool's bytes no window wrote are released (once per cycle); a
			// range left without GPU bytes needs nothing more.
			if (!m_append_pools.empty() && AppendRelease(vaddr, size)) {
				if (is_write) {
					release_gpu_to_cpu();
					return;
				}
				m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
				return;
			}
			// Dead GPU bytes of a data-ring page are dropped here, right after the pending
			// eager copies (which must publish first), before the read window and download planning; a page left
			// without GPU bytes needs nothing more. Newer GPU bytes go on to the download below.
			if (is_write && DropDeadRingBytes(vaddr)) {
				release_gpu_to_cpu();
				return;
			}
			auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

			// A CPU access of a GPU-modified page downloads every GPU-modified range of the same buffer within +-4 MiB in one
			// synchronization, so neighbouring accesses do not each wait for the GPU again (an approximation of timing only).
			// A write publishes the pages around its own and releases its page to the CPU (map 3c, 4.11: since 9919560e7 the
			// game's 523 MiB buffer is device-local, so each page download is a flush, a wait and a copy, ~85 us; the cutscene
			// after the Snake fight writes ~2000 consecutive pages a frame: 5.5 -> 8.4 fps with the window, live A/B; checked
			// against the page download: 0 different bytes in 4.1 GB). A published neighbour holds the GPU bytes a later
			// fault would download (a GPU write recorded later marks it again), and a CPU-dirty page is never GPU-modified.
			const auto [lo, hi] = ReadRange(buffer, vaddr, size);
			if (lo < hi && lo <= vaddr && vaddr + size <= hi) {
				PerfCount(PerfKind::SyncRead);

				if (timed_download(buffer, lo, hi - lo)) {
					if (!is_write) {
						UnmarkWindow(lo, hi);
						return;
					}
					const uint64_t page_lo = std::max(lo, vaddr & ~(TRACKER_PAGE_SIZE - 1));
					const uint64_t page_hi = std::min(hi, (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1));
					UnmarkWindow(lo, page_lo);
					UnmarkWindow(page_hi, hi);
					release_gpu_to_cpu();
					return;
				}
				if (is_write) mark_cpu_written();
				return;
			}
			PerfCount(PerfKind::SyncRead);

			if (timed_download(buffer, vaddr, size)) {
				if (is_write) {
					release_gpu_to_cpu();
					return;
				}
				m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			}
			if (is_write) mark_cpu_written();
		};
		m_scheduler.Context().GetGpu().SendCommandSync(std::move(request));
		if (defer_tick == 0) break;
		// Handed back; the tick is submitted. Wait here, not on the GPU thread.
		if (attempt == 0) m_caller_stats.deferred.fetch_add(1, std::memory_order_relaxed);
		m_caller_stats.deferrals.fetch_add(1, std::memory_order_relaxed);
		const auto wait_start = std::chrono::steady_clock::now();
		{
			SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Caller); // KYTY_TRACE_SYNC_TIMELINE
			m_scheduler.GetMasterSemaphore().Wait(defer_tick);
			m_scheduler.WaitPriorityOperations(defer_tick);
		}
		m_caller_stats.caller_wait_ns.fetch_add(
		    static_cast<uint64_t>(
		        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_start).count()),
		    std::memory_order_relaxed);
		m_caller_stats.retries.fetch_add(1, std::memory_order_relaxed);
		last_tick = defer_tick;
	}
}

// The caller wait: a guest fault (read or write) on a GPU-written page, taken by the exception handler on another thread
// than the GPU thread, does not make the GPU thread wait for the GPU. When the bytes the request would download (the
// faulting page, or the read window) still wait for a GPU write, or a pending eager copy of them must publish first, the
// GPU thread submits the open command buffer if that write is in it and hands the request back unchanged; the faulting
// thread waits for that tick and the priority completions up to it, then sends the request again. The second request is
// decided from scratch (a GPU write recorded meanwhile is waited for again, at most twice; then the GPU thread waits as
// before). Requests the host-mapped read cannot serve (unknown write tick, more than 1 MiB, unmapped buffer, write not
// recorded yet) are served as before.

BufferCache::CallerPlan BufferCache::PlanCallerWait(uint64_t vaddr, uint64_t size, bool is_write) {
	CallerPlan plan;
	if (!m_scheduler.Active() || m_scheduler.Current().IsInvalid()) return plan;
	// 1. Pending eager copies of the bytes, which ReadMemory waits for first (ApplyEagerReadbacks). The download is
	// planned by the next request, once they published.
	for (const auto& entry: m_eager_readbacks) {
		if (vaddr < entry.end && entry.start < vaddr + size && !entry.done->load(std::memory_order_acquire)) {
			plan.tick = std::max(plan.tick, entry.tick);
		}
	}
	if (plan.tick != 0) return plan;
	// ReadMemory returns before downloading when the bytes are not GPU-modified (after the eager copies).
	if (!m_memory_tracker.IsRegionGpuModified(vaddr, size)) return plan;
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];
	// 2. The range the request downloads (ReadMemory): the read window; else the
	// pages of the access.
	uint64_t lo = vaddr;
	uint64_t hi = vaddr + size;
	{
		const auto [w_lo, w_hi] = ReadRange(buffer, vaddr, size);
		if (w_lo < w_hi && w_lo <= vaddr && vaddr + size <= w_hi) {
			lo = w_lo;
			hi = w_hi;
		}
	}
	// 3. What its host-mapped read waits for (TryBarDownload): the newest 64 KiB granule write tick of the GPU-dirty
	// bytes of those pages only (not of the rest of the buffer), then the priority completions up to it. What
	// TryBarDownload declines takes the copy path, which waits on the GPU thread as before.
	if (buffer.Mapped().empty()) {
		plan.fallback = CallerFallback::NoMapping;
		return plan;
	}
	std::vector<std::pair<uint64_t, uint64_t>> intervals;
	uint64_t                                   newest = 0;
	uint64_t                                   total  = 0;
	if (const int scan = ScanBarDownload(buffer, lo, hi - lo, intervals, newest, total); scan >= 0) {
		if (scan != 6) plan.fallback = scan == 5 ? CallerFallback::Outside : CallerFallback::UnknownTick;
		return plan;
	}
	const auto ready = m_buffer_ready_tick.find(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.Handle())));
	if (ready == m_buffer_ready_tick.end()) {
		plan.fallback = CallerFallback::UnknownTick;
		return plan;
	}
	newest = std::max(newest, ready->second);
	if (total > BarMaxBytes) {
		plan.fallback = CallerFallback::Large;
		return plan;
	}
	for (const auto& [start, end]: intervals) {
		if (std::ranges::any_of(m_bar_pending_writes,
		                        [&](const auto& w) { return w.first < end && start < w.first + w.second; })) {
			plan.fallback = CallerFallback::Pending;
			return plan;
		}
	}
	if (newest < m_scheduler.CurrentTick() && m_scheduler.IsFree(newest) && m_scheduler.PriorityOperationsDone(newest)) {
		return plan; // the read waits for nothing
	}
	plan.tick = newest;
	return plan;
}

// The write-through of the GPU thread's own writes of guest memory (end-of-pipe labels and
// timestamps, WRITE_DATA, COPY_DATA of the reference clock) into a GPU-modified page no longer fault when none of the
// written bytes is GPU-written (the GPU-dirty set is byte-exact). At the torii-gate scene the game's frame timestamps
// (EOP, 8 bytes at +0x40 per frame) share a 4 KiB page with 512 bytes the GPU writes: every frame the first label
// write faulted, and the GPU thread waited ~30-35 ms (KYTY_TRACE_SYNC_TIMELINE: reason=invalidate from=gpu site=bar,
// need_dirty=0, verdict=page) for the GPU to finish those other bytes, only to make the page CPU-owned. The fault
// happens on the GPU thread itself, so the caller wait cannot move that wait anywhere.
// Instead the bytes are written to guest memory through the backing store (the page stays protected and GPU-owned)
// and to the GPU copy of the buffer with vkCmdUpdateBuffer at this point of the command stream. Exact: guest memory
// holds the written bytes at once, as before; the GPU copy holds them from this point of the stream on (after every
// GPU write recorded before, before every one recorded after), as the upload after the CPU-owned transition gave it;
// the page's GPU-written bytes stay tracked and are synchronized byte-exactly when the CPU reads or writes the page,
// and they never cover the written bytes (a later GPU write of them marks them GPU-written again, as before). Declined
// (the write faults as before): written bytes the GPU owns, a write obtained but not yet recorded or an eager copy
// pending in the page, a registered image over the bytes (it would reload the page from guest memory
// before the GPU bytes are there), no single cached buffer, a write crossing a page or not in whole aligned dwords.
namespace {
constexpr uint64_t CpWriteMaxBytes = 256; // the largest write taken
} // namespace

namespace {
// KYTY_PERF_DIRECT_RANGES statistics (one "PERF direct_ranges" line per ~5 s while ranges are configured).
struct DirectStats {
	std::atomic<uint64_t> writes {0}, write_bytes {0}, checks {0}, waits {0}, flushes {0}, wait_us {0}, off_thread {0};
	std::atomic<uint64_t> cp_writes {0}; // of the writes: command-processor writes at translation time
	// KYTY_TRACE_PERF (early-submit ceiling): per wait, the time from recording the newest awaited write to the wait,
	// and the waits that found every submitted tick complete (the GPU idle: it only had the open buffer left to run).
	std::atomic<uint64_t> gap_us {0}, gap_max_us {0}, idle_waits {0}, idle_wait_us {0};
};
DirectStats g_direct_stats;
// Reads that took pending command-processor writes from the overlay.
std::atomic<uint64_t> g_direct_overlay_reads {0};
void ReportDirectStats(size_t pending) {
	// Without the lines cp_writes is not reset either, so PERF_DIRECT_CP_WRITE logs its first 32 writes of the run.
	if (!Common::EnvRead::PerfLines()) return;
	static int64_t window_start = 0;
	const int64_t  now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	if (window_start == 0) window_start = now;
	if (now - window_start < 5000) return;
	window_start = now;
	auto& s      = g_direct_stats;
	std::fprintf(stderr,
	             "PERF direct_ranges gpu_writes=%llu write_KiB=%llu read_checks=%llu waits=%llu flushes=%llu wait_ms=%llu "
	             "off_thread=%llu cp_writes=%llu pending=%zu overlay_reads=%llu "
	             "gap_ms=%llu gap_max_ms=%llu idle_waits=%llu idle_wait_ms=%llu\n",
	             static_cast<unsigned long long>(s.writes.exchange(0)), static_cast<unsigned long long>(s.write_bytes.exchange(0) / 1024u),
	             static_cast<unsigned long long>(s.checks.exchange(0)), static_cast<unsigned long long>(s.waits.exchange(0)),
	             static_cast<unsigned long long>(s.flushes.exchange(0)), static_cast<unsigned long long>(s.wait_us.exchange(0) / 1000u),
	             static_cast<unsigned long long>(s.off_thread.exchange(0)),
	             static_cast<unsigned long long>(s.cp_writes.exchange(0)), pending,
	             static_cast<unsigned long long>(g_direct_overlay_reads.exchange(0)),
	             static_cast<unsigned long long>(s.gap_us.exchange(0) / 1000u),
	             static_cast<unsigned long long>(s.gap_max_us.exchange(0) / 1000u),
	             static_cast<unsigned long long>(s.idle_waits.exchange(0)),
	             static_cast<unsigned long long>(s.idle_wait_us.exchange(0) / 1000u));
}
} // namespace

namespace {
struct RingDeadStats {
	std::atomic<uint64_t> faults {0}, freed {0}, dropped_bytes {0}, partial {0};
};
RingDeadStats g_ring_dead;
std::array<uint64_t, 8>  g_ring_frame_ticks {}; // tick at the end of frame n (collector pass), slot n % 8
uint64_t                 g_ring_frames = 0;
} // namespace

bool BufferCache::TryCpWriteThrough(uint64_t address, const void* data, uint64_t size) {
	// KYTY_PERF_DIRECT_RANGES: a command-processor write into a direct range goes into the stream (the GPU work
	// recorded before it must not see it), never into guest memory at translation time.
	if (!m_direct_ranges.empty() && data != nullptr && size != 0 && InDirectRange(address, size) &&
	    GuestGpu::IsGpuThread() && m_scheduler.Active() && !m_scheduler.Current().IsInvalid() && size % 4u == 0 &&
	    address % 4u == 0 && size <= 65536u) {
		if (const BufferId id = DirectBufferFor(address, size); id) {
			auto& buffer  = m_slot_buffers[id];
			auto& command = m_scheduler.Current();
			command.EndRendering();
			const auto              native = command.Handle();
			vk::BufferMemoryBarrier before {};
			before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
			before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			before.buffer              = buffer.Handle();
			before.offset              = buffer.Offset(address);
			before.size                = size;
			native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
			                       nullptr, 1, &before, 0, nullptr);
			native.updateBuffer(buffer.Handle(), buffer.Offset(address), size, data);
			auto after          = before;
			after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 0,
			                       nullptr, 1, &after, 0, nullptr);
			NoteDirectGpuWrite(address, size, data);
			if (g_direct_stats.cp_writes.fetch_add(1, std::memory_order_relaxed) < 32u) {
				uint32_t first = 0;
				std::memcpy(&first, data, std::min<uint64_t>(size, 4));
				std::fprintf(stderr, "PERF_DIRECT_CP_WRITE addr=0x%llx size=%llu value=0x%08x rva=0x%llx tick=%llu marker=[%s]\n",
				             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size), first,
				             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(__builtin_return_address(0)) -
				                                             0x140000000ull),
				             static_cast<unsigned long long>(m_scheduler.CurrentTick()), DebugMarkerPath().c_str());
			}
			m_texture_cache.InvalidateMemory(address, size);
			return true;
		}
	}
	if (size == 0 || data == nullptr || !GuestRange {address, size}.Valid()) return false;
	// Only a write that would fault on a GPU-modified page; any other write is a plain store as before.
	if (!m_memory_tracker.IsRegionGpuModified(address, size)) return false;
	const auto reason = CpWriteThroughOwnedPage(address, data, size, true);
	if (reason == CpWriteDecline::Count) return true;
	m_cp_write_stats.declined[static_cast<uint32_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	return false;
}

BufferCache::CompletedWrite BufferCache::WriteCompletedValue(uint64_t address, const void* data, uint64_t size) {
	if (size == 0 || data == nullptr || !GuestRange {address, size}.Valid()) return CompletedWrite::Drop;
	if (!m_memory_tracker.IsRegionGpuModified(address, size)) return CompletedWrite::Plain;
	// Never over GPU-owned bytes: the value belongs to an earlier point of the stream than the update would.
	switch (CpWriteThroughOwnedPage(address, data, size, false)) {
		case CpWriteDecline::Count: return CompletedWrite::Stored;
		case CpWriteDecline::Shape:   // no open command buffer to put the GPU copy's update into (or a bad shape:
		case CpWriteDecline::Pending: // the caller gives up after a while), or a recorded write / pending copy
			return CompletedWrite::Retry;
		default: return CompletedWrite::Drop;
	}
}

BufferCache::CpWriteDecline BufferCache::CpWriteThroughOwnedPage(uint64_t address, const void* data, uint64_t size,
                                                                  bool replace) {
	const uint64_t page = address & ~(TRACKER_PAGE_SIZE - 1);
	if (!GuestGpu::IsGpuThread() || !m_scheduler.Active() || m_scheduler.Current().IsInvalid() || size > CpWriteMaxBytes ||
	    size % sizeof(uint32_t) != 0 || address % sizeof(uint32_t) != 0 || address + size > page + TRACKER_PAGE_SIZE) {
		return CpWriteDecline::Shape;
	}
	// Bytes the GPU owns are ordered after its writes only by the synchronizing path, unless this write replaces them
	// (`replace`). It does: its bytes reach the GPU copy through
	// the update below, recorded after every GPU write recorded before (the console's CP writes them when it gets there),
	// and guest memory at once, so the GPU's older bytes under it are dead and leave the GPU-dirty set. GPU-owned bytes
	// next to it in the page stay GPU-owned. (A flare occlusion query clears its whole 256-byte record, which holds the
	// previous frame's GPU-written dumps, before every query.)
	const bool replaces_gpu_bytes = m_gpu_modified_ranges.Intersects(address, size);
	if (replaces_gpu_bytes && !replace) return CpWriteDecline::Overlap;
	// A binding obtained for writing whose tick is not final, or an eager copy that will publish the page's older bytes.
	// (An eager copy in flight that holds none of the written bytes publishes only its own bytes: no conflict.)
	if (std::ranges::any_of(m_bar_pending_writes, [&](const auto& w) { return w.first < address + size && address < w.first + w.second; }) ||
	    PendingReadbackCovers(address, size)) {
		return CpWriteDecline::Pending;
	}
	if (m_texture_cache.IsRegionRegistered(address, size)) return CpWriteDecline::Image;
	const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) return CpWriteDecline::NoBuffer;
	auto& buffer = m_slot_buffers[*owner];
	if (buffer.is_deleted || !buffer.IsInBounds(address, size)) return CpWriteDecline::NoBuffer;
	// The upload batch: the GPU copy's update below is recorded after every claimed upload.
	CommitUploadBatch(BatchCommit::CpWrite);

	// Guest memory: through the backing store (the page stays protected and GPU-owned).
	Libs::LibKernel::Memory::WriteBacking(address, data, size);
	// The GPU copy: at this point of the command stream.
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	const uint64_t          offset = buffer.Offset(address);
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = offset;
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1,
	                       &before, 0, nullptr);
	native.updateBuffer(buffer.Handle(), offset, size, data);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1,
	                       &after, 0, nullptr);
	// What the write fault does for images and checked DCC keys (no registered image covers the bytes).
	m_texture_cache.InvalidateMemory(address, size);
	if (replaces_gpu_bytes) {
		GpuDirtySubtract(address, size);
		// A page left without GPU bytes holds current data in guest memory ("GPU-modified page <=> bytes to download").
		if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) && m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
			m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
		}
		m_cp_write_stats.replaced.fetch_add(1, std::memory_order_relaxed);
	}
	m_cp_write_stats.writes.fetch_add(1, std::memory_order_relaxed);
	m_cp_write_stats.bytes.fetch_add(size, std::memory_order_relaxed);
	return CpWriteDecline::Count;
}

void BufferCache::NoteSyncTimelineNeed(uint64_t vaddr, uint64_t size) {
	uint64_t lo = 0;
	uint64_t hi = 0;
	SyncTimeline::NeedRange(vaddr, size, lo, hi);
	const uint64_t page_lo    = lo & ~(TRACKER_PAGE_SIZE - 1);
	const uint64_t page_hi    = Common::AlignUp(hi, TRACKER_PAGE_SIZE);
	uint64_t       need_dirty = 0;
	uint64_t       page_dirty = 0;
	uint64_t       need_tick  = 0;
	uint64_t       exact_tick = 0;
	bool           unknown    = false;
	bool           inexact    = false;
	// The newest write tick of the 64 KiB granules holding the needed GPU-dirty bytes (the BAR reads keep
	// them; the BAR read waits for it; a granule without one makes the tick unknown), and of the written ranges
	// themselves (NoteWriteTick).
	m_gpu_modified_ranges.ForEachInRange(lo, hi - lo, [&](uint64_t start, uint64_t end) {
		need_dirty += end - start;
		for (uint64_t key = WriteTickKey(start); key <= WriteTickKey(end - 1); ++key) {
			const auto it = GranuleWriteTick(key);
			if (!it) {
				unknown = true;
			} else {
				need_tick = std::max(need_tick, *it);
			}
		}
		const auto tick = QueryWriteTick(start, end);
		inexact |= tick == 0;
		exact_tick = std::max(exact_tick, tick);
	});
	m_gpu_modified_ranges.ForEachInRange(page_lo, page_hi - page_lo,
	                                     [&](uint64_t start, uint64_t end) { page_dirty += end - start; });
	SyncTimeline::SetNeed(unknown ? 0 : need_tick, inexact ? 0 : exact_tick, need_dirty, page_dirty);
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	// KYTY_PERF_DIRECT_RANGES: a direct range's buffer (never joined into another one).
	if (const BufferId direct = DirectBufferFor(vaddr, size); direct) {
		return direct;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

// ---- KYTY_PERF_DIRECT_RANGES (A1 stage 2) ---------------------------------------------
// Default on (=0 off; live A/B is not a fair test of it: each switch to off re-uploads and re-protects
// the whole range). 3+2 alternating runs from the start (20261014-2038..2049): 6.34 fps on against 6.28 off (+1%),
// GPU-thread protection time halved, game-thread lock waits ~10x shorter, images correct; Vulkan validation: no new
// errors (the two extra ones of the "on" run were a zero-sized window while the monitor was off). The time saved mostly
// turns into waiting for the host GPU at the frame's sync points. Guest memory of the direct ranges is the memory of one cache buffer each (imported host
// pointer, Buffer(GuestDirectTag)); the host GPU reads the guest's bytes in place, its writes land there, and none of
// the range's pages is protected, uploaded or downloaded. Correct because guest labels reach the guest only once the
// host GPU did the work before them (labels at completion) and because the GPU thread's own reads of guest
// memory wait for the GPU writes recorded before them (WaitDirectGpuWrites). Ranges: "1" = the Yotei SRT ring
// (0x2010000000-0x2015E00000, 72% of the write faults, 2a probe 20261014-184732), or "start-end[,start-end]" in hex.
// A request reaching past a range grows it (re-import); a range that cannot be imported or holds images stays on the
// tracked path.
uint32_t BufferCache::DirectMode() {
	static constinit Common::OnceValue<uint32_t> once;
	const uint32_t mode = Common::LiveAb::Gate(Common::LiveAb::Switch::DirectRanges, once.Get([] {
		const uint32_t m = Common::EnvRead::DirectRanges() ? 1u : 0u;
		if (m == 0) std::fprintf(stderr, "PERF_DIRECT_RANGES off\n");
		return m;
	}));
	if (!m_direct_parsed) {
		m_direct_parsed   = true;
		m_direct_ranges.reserve(64); // the thread that makes the operations reads them (InDirectRange): no reallocation
		const char* value = nullptr;
		if (Common::EnvRead::DirectRanges(&value)) {
			if (value == nullptr || (value[0] == '1' && value[1] == '\0')) {
				m_direct_ranges.push_back({0x2010000000ull, 0x2015E00000ull});
			} else {
				const char* p = value;
				while (*p != '\0') {
					char*          next  = nullptr;
					const uint64_t start = std::strtoull(p, &next, 16);
					if (next == p || *next != '-') break;
					p                  = next + 1;
					const uint64_t end = std::strtoull(p, &next, 16);
					if (next == p) break;
					p = *next == ',' ? next + 1 : next;
					const uint64_t s = Common::AlignDown(start, CACHING_PAGESIZE);
					const uint64_t e = Common::AlignUp(end, CACHING_PAGESIZE);
					if (s < e) m_direct_ranges.push_back({s, e});
				}
			}
		}
	}
	return mode;
}

BufferId BufferCache::DirectBufferFor(uint64_t vaddr, uint64_t size) {
	const uint32_t mode = DirectMode();
	if (mode != m_direct_mode_seen) {
		// Live A/B: off ends the active ranges (their memory is tracked again from guest memory).
		if (mode == 0) {
			for (auto& range: m_direct_ranges) DeactivateDirect(range);
		}
		m_direct_mode_seen = mode;
	}
	if (mode == 0 || m_direct_ranges.empty() || size == 0) return {};
	const uint64_t end = vaddr + size;
	for (auto& range: m_direct_ranges) {
		if (range.refused || end <= range.start || vaddr >= range.end) continue;
		if (vaddr < range.start || end > range.end) {
			// A request reaching past the range: the range grows to hold it (a direct buffer is never joined).
			DeactivateDirect(range, false);
			range.start = Common::AlignDown(std::min(vaddr, range.start), CACHING_PAGESIZE);
			range.end   = Common::AlignUp(std::max(end, range.end), CACHING_PAGESIZE);
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 16u) {
				std::fprintf(stderr, "PERF_DIRECT_RANGES grow to 0x%llx-0x%llx (request 0x%llx+0x%llx)\n",
				             static_cast<unsigned long long>(range.start), static_cast<unsigned long long>(range.end),
				             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
			}
		}
		if (!range.active) ActivateDirect(range);
		return range.active ? range.id : BufferId {};
	}
	return {};
}

void BufferCache::ActivateDirect(DirectRange& range) {
	if (range.active || range.refused) return;
	const uint64_t size = range.end - range.start;
	if (m_texture_cache.IsRegionRegistered(range.start, size)) {
		range.refused = true;
		std::fprintf(stderr, "PERF_DIRECT_RANGES 0x%llx-0x%llx refused: images in the range\n",
		             static_cast<unsigned long long>(range.start), static_cast<unsigned long long>(range.end));
		m_texture_cache.DescribeImagesInRegion(range.start, size, 16);
		return;
	}
	CommitUploadBatch(BatchCommit::CreateBuffer);
	// Every recorded GPU access to the range done; the GPU-written bytes of the buffers over it reach guest memory.
	m_scheduler.Finish();
	std::vector<BufferId>                      old_ids;
	std::vector<std::pair<uint64_t, uint64_t>> outside; // parts of the replaced buffers outside the range
	for (uint64_t page = range.start >> PageTable::kPageBits; page < (range.end >> PageTable::kPageBits); page++) {
		const auto* owner = m_page_table.Find(page);
		if (owner != nullptr && *owner && std::ranges::find(old_ids, *owner) == old_ids.end()) old_ids.push_back(*owner);
	}
	for (const auto id: old_ids) {
		auto&          buffer  = m_slot_buffers[id];
		const uint64_t address = buffer.CpuAddress();
		const uint64_t bytes   = buffer.Size();
		if (address < range.start) outside.emplace_back(address, range.start - address);
		if (address + bytes > range.end) outside.emplace_back(range.end, address + bytes - range.end);
		ApplyEagerReadbacks(address, bytes, true);
		if (m_memory_tracker.IsRegionGpuModified(address, bytes)) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<false>(buffer, address, bytes));
			m_memory_tracker.UnmarkRegionAsGpuModified(address, bytes);
		}
		m_memory_tracker.UntrackMemory(address, bytes);
		DeleteBuffer(id);
	}
	if (m_memory_tracker.IsRegionGpuModified(range.start, size)) {
		m_memory_tracker.UnmarkRegionAsGpuModified(range.start, size);
	}
	m_memory_tracker.UntrackMemory(range.start, size); // CPU-dirty and writable: never protected while direct
	if (!Buffer::CanImportGuestMemory(m_graphics, range.start, size)) {
		range.refused = true;
		std::fprintf(stderr, "PERF_DIRECT_RANGES 0x%llx-0x%llx refused: the guest memory cannot be imported\n",
		             static_cast<unsigned long long>(range.start), static_cast<unsigned long long>(range.end));
		return;
	}
	const auto id = [&] {
		const LookupWriteGuard lookup_guard(*this); // the resolve thread's lookups: the slot storage may grow
		return m_slot_buffers.insert(m_graphics, m_scheduler, GuestDirectTag {}, range.start, size);
	}();
	SetVulkanObjectNameF(m_graphics.device, m_slot_buffers[id].Handle(), "Kyty.DirectBuffer[guest=0x{:016x} size=0x{:x}]",
	                     range.start, size);
	Register(id);
	range.id     = id;
	range.active = true;
	for (const auto& [address, bytes]: outside) (void)CreateBuffer(address, bytes); // stop at the direct buffer
	std::fprintf(stderr, "PERF_DIRECT_RANGES active 0x%llx-0x%llx (%llu old buffers replaced)\n",
	             static_cast<unsigned long long>(range.start), static_cast<unsigned long long>(range.end),
	             static_cast<unsigned long long>(old_ids.size()));
}

void BufferCache::DeactivateDirect(DirectRange& range, bool replace) {
	if (!range.active) return;
	// The GPU's writes to the range are in guest memory once it finished; the pages stay CPU-dirty, so the next
	// ordinary buffer over them uploads guest memory.
	m_scheduler.Finish();
	DeleteBuffer(range.id);
	m_memory_tracker.UntrackMemory(range.start, range.end - range.start);
	std::erase_if(m_direct_gpu_writes, [&](const DirectGpuWrite& w) { return w.start < range.end && range.start < w.end; });
	range.id     = {};
	range.active = false;
	// An ordinary buffer over the whole range at once: without one, shaders reading the range through pointers (the BDA
	// table) would read nothing until a binding created one (black squares at every live A/B switch to off).
	if (replace) (void)CreateBuffer(range.start, range.end - range.start);
}

bool BufferCache::InDirectRange(uint64_t vaddr, uint64_t size) const {
	for (const auto& range: m_direct_ranges) {
		if (range.active && vaddr < range.end && range.start < vaddr + size) return true;
	}
	return false;
}


// A CPU write fault into a GPU-modified page of the data ring (two
// halves, each reused every second frame; the game overwrites what it allocates there and never reads an older
// frame's data) drops the page's GPU bytes written before the previous frame began instead of downloading them (and
// waiting for their writer). Newer GPU bytes are downloaded as before. Returns true when no GPU byte is left in the page.
// Measured (run 20261014-212758): ~290 such faults a frame, 74% with only such old bytes. The
// frame boundary is the GPU thread's (it lags the game), which only makes the rule stricter.
bool BufferCache::DropDeadRingBytes(uint64_t vaddr) {
	if (vaddr < 0x2002600000ull || vaddr >= 0x2010000000ull || g_ring_frames < 3) return false;
	const uint64_t page       = vaddr & ~uint64_t {0xfff};
	const uint64_t prev_start = g_ring_frame_ticks[(g_ring_frames - 2) % g_ring_frame_ticks.size()];
	std::vector<std::pair<uint64_t, uint64_t>> dead;
	bool                                       live = false;
	m_gpu_modified_ranges.ForEachInRange(page, 4096, [&](uint64_t start, uint64_t end) {
		const uint64_t tick = QueryWriteTick(start, end);
		if (tick != 0 && tick < prev_start) {
			dead.emplace_back(start, end);
		} else {
			live = true;
		}
	});
	uint64_t bytes = 0;
	for (const auto& [start, end]: dead) {
		GpuDirtySubtract(start, end - start);
		bytes += end - start;
	}
	if (bytes != 0 && UploadClobberCheck()) { // KYTY_D6_C0=2/3: no clobber there
		m_c0_dead_pages.insert(page);
		C0Note(page, TRACKER_PAGE_SIZE, C0Event::RingDead);
	}
	g_ring_dead.faults.fetch_add(1, std::memory_order_relaxed);
	g_ring_dead.dropped_bytes.fetch_add(bytes, std::memory_order_relaxed);
	if (!live) g_ring_dead.freed.fetch_add(1, std::memory_order_relaxed);
	if (live && bytes != 0) g_ring_dead.partial.fetch_add(1, std::memory_order_relaxed);
	static int64_t window_start = 0;
	const int64_t  now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	if (window_start == 0) window_start = now;
	if (now - window_start >= 5000 && Common::EnvRead::PerfLines()) {
		window_start = now;
		std::fprintf(stderr, "PERF ring_dead faults=%llu freed=%llu partial=%llu dropped_KiB=%llu\n",
		             static_cast<unsigned long long>(g_ring_dead.faults.exchange(0)),
		             static_cast<unsigned long long>(g_ring_dead.freed.exchange(0)),
		             static_cast<unsigned long long>(g_ring_dead.partial.exchange(0)),
		             static_cast<unsigned long long>(g_ring_dead.dropped_bytes.exchange(0) / 1024u));
	}
	return !live;
}

void BufferCache::NoteDirectGpuWrite(uint64_t vaddr, uint64_t size, const void* cp_data) {
	if (!m_append_pools.empty() || !m_append_windows.empty()) AppendForeignWrite(vaddr, size); // D4b step 1 (append windows)
	if (m_direct_gpu_writes.size() >= 64u) {
		auto& master = m_scheduler.GetMasterSemaphore();
		master.Refresh(); // once, not per listed write
		std::erase_if(m_direct_gpu_writes, [&](const DirectGpuWrite& w) { return master.IsFree(w.tick); });
	}
	auto& write = m_direct_gpu_writes.emplace_back(DirectGpuWrite {vaddr, vaddr + size, m_scheduler.CurrentTick()});
	if (PerfEnabled()) {
		write.noted_us = std::chrono::duration_cast<std::chrono::microseconds>(
		                     std::chrono::steady_clock::now().time_since_epoch())
		                     .count();
	}
	// A command-processor write keeps its bytes for the reader's overlay (larger ones are waited for).
	if (cp_data != nullptr && size <= write.cp_bytes.size()) {
		write.cp_size = static_cast<uint32_t>(size);
		std::memcpy(write.cp_bytes.data(), cp_data, size);
	}
	g_direct_stats.writes.fetch_add(1, std::memory_order_relaxed);
	g_direct_stats.write_bytes.fetch_add(size, std::memory_order_relaxed);
}

// The GPU thread's reads of a direct range that WaitPendingReadbacks guards (the resource walker's ordinary and strict
// reads, the resolve thread's ReadsUnchanged check) do not wait for command-processor writes recorded into it. Ghost of
// Yotei writes ~100 single dwords per frame into SRT-ring tables with WRITE_DATA (PERF direct_ranges cp_writes), and the
// walker reads those pages shortly after. The GPU thread knows those bytes, so the read takes guest memory with the
// writes still pending on the host GPU put over it in stream order: what guest memory holds once the host GPU ran them,
// which is the point of the stream the reader stands at. Writes of shaders (ranges obtained for writing, bytes unknown)
// are still waited for, and every write recorded before them with them.
void BufferCache::ApplyDirectCpWrites(uint64_t vaddr, void* data, uint64_t size) const {
	auto* out = static_cast<uint8_t*>(data);
	for (const auto& w: m_direct_cp_overlay) {
		const uint64_t lo = std::max(w.start, vaddr);
		const uint64_t hi = std::min(w.end, vaddr + size);
		if (lo < hi) std::memcpy(out + (lo - vaddr), w.cp_bytes.data() + (lo - w.start), hi - lo);
	}
}

bool BufferCache::WaitDirectGpuWrites(uint64_t vaddr, uint64_t size, bool cp_overlay) {
	m_direct_cp_overlay.clear();
	if (m_direct_ranges.empty()) return false;
	// GPU thread only: the list and the scheduler belong to it. Other threads read guest memory as the console's CPU
	// does (the guest orders those reads after the GPU's writes through its labels, published at completion).
	if (!GuestGpu::IsGpuThread()) {
		g_direct_stats.off_thread.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	// Writes the host GPU already ran are dropped from the front (recorded in tick order); the known GPU tick is read
	// without asking the driver (a write still listed after it finished is only a wasted check, never a wrong answer).
	// ~1.2 million calls per second at the torii gate (the walker and the page checks of the read logs).
	auto& master = m_scheduler.GetMasterSemaphore();
	if (!m_direct_gpu_writes.empty() && master.IsFree(m_direct_gpu_writes.front().tick)) {
		std::erase_if(m_direct_gpu_writes, [&](const DirectGpuWrite& w) { return master.IsFree(w.tick); });
	}
	if (m_direct_gpu_writes.empty()) return false;
	g_direct_stats.checks.fetch_add(1, std::memory_order_relaxed);
	// With the overlay, only writes of unknown bytes are waited for here.
	uint64_t tick    = 0; // newest overlapping write
	uint64_t unknown = 0; // newest overlapping write whose bytes the GPU thread does not know
	int64_t  unknown_noted_us = 0; // KYTY_TRACE_PERF: when the newest unknown-bytes write was recorded
	for (const auto& w: m_direct_gpu_writes) {
		if (w.start < vaddr + size && vaddr < w.end) {
			tick = std::max(tick, w.tick);
			if (w.cp_size == 0u) {
				unknown          = std::max(unknown, w.tick);
				unknown_noted_us = std::max(unknown_noted_us, w.noted_us);
			}
		}
	}
	const auto caller   = reinterpret_cast<uint64_t>(__builtin_return_address(0)) - 0x140000000ull; // the reader
	const auto wait_for = [&](uint64_t target) {
		if (target == 0 || m_scheduler.IsFree(target)) return false;
		static std::atomic_uint wait_logged {0};
		if (wait_logged.fetch_add(1) < 32u) {
			std::fprintf(stderr, "PERF_DIRECT_WAIT addr=0x%llx size=%llu tick=%llu current=%llu rva=0x%llx marker=[%s]\n",
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			             static_cast<unsigned long long>(target), static_cast<unsigned long long>(m_scheduler.CurrentTick()),
			             static_cast<unsigned long long>(caller), DebugMarkerPath().c_str());
		}
		const auto start = std::chrono::steady_clock::now();
		// KYTY_TRACE_PERF: was the GPU idle (every submitted tick complete) when the wait began?
		const bool gpu_idle = PerfEnabled() && m_scheduler.CurrentTick() > 1u && [&] {
			auto& master_semaphore = m_scheduler.GetMasterSemaphore();
			master_semaphore.Refresh();
			return master_semaphore.IsFree(m_scheduler.CurrentTick() - 1u);
		}();
		if (PerfEnabled() && target == unknown && unknown_noted_us != 0) {
			const auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count();
			const auto gap    = static_cast<uint64_t>(std::max<int64_t>(now_us - unknown_noted_us, 0));
			g_direct_stats.gap_us.fetch_add(gap, std::memory_order_relaxed);
			if (gap > g_direct_stats.gap_max_us.load(std::memory_order_relaxed))
				g_direct_stats.gap_max_us.store(gap, std::memory_order_relaxed);
		}
		if (target >= m_scheduler.CurrentTick()) {
			m_scheduler.Flush();
			g_direct_stats.flushes.fetch_add(1, std::memory_order_relaxed);
		}
		m_scheduler.Wait(target);
		RenderExecutor::GpuBurnAt(RenderExecutor::GpuBurnSite::DirectWait); // KYTY_DIAG_GPU_BURN_AT=direct_wait (measurement only)
		g_direct_stats.waits.fetch_add(1, std::memory_order_relaxed);
		const auto us = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
		g_direct_stats.wait_us.fetch_add(us, std::memory_order_relaxed);
		if (gpu_idle) {
			g_direct_stats.idle_waits.fetch_add(1, std::memory_order_relaxed);
			g_direct_stats.idle_wait_us.fetch_add(us, std::memory_order_relaxed);
		}
		// Waits of 2 ms or more, with the reader that asked (WaitPendingReadbacks, ReadMemory, WAIT_REG_MEM).
		static std::atomic_uint long_logged {0};
		if (us >= 2000u && long_logged.fetch_add(1) < 48u) {
			std::fprintf(stderr, "PERF_DIRECT_LONG_WAIT frame=%u us=%llu addr=0x%llx size=%llu tick=%llu current=%llu overlay=%u "
			                     "caller=0x%llx marker=[%s]\n",
			             DebugPresentCount().load(), static_cast<unsigned long long>(us), static_cast<unsigned long long>(vaddr),
			             static_cast<unsigned long long>(size), static_cast<unsigned long long>(target),
			             static_cast<unsigned long long>(m_scheduler.CurrentTick()), cp_overlay ? 1u : 0u,
			             static_cast<unsigned long long>(caller),
			             DebugMarkerPath().c_str());
		}
		return true;
	};
	if (!cp_overlay) {
		if (wait_for(tick)) {
			std::erase_if(m_direct_gpu_writes, [&](const DirectGpuWrite& w) { return m_scheduler.IsFree(w.tick); });
		}
		return false;
	}
	// The writes of unknown bytes first (with them every write recorded before), then the command-processor writes still
	// pending after that, in recording order: the reader puts them over the guest memory it reads next. One that the
	// host GPU finishes between now and that read leaves guest memory holding the same bytes (no unknown write after it
	// overlaps them: those were all waited for).
	wait_for(unknown);
	for (const auto& w: m_direct_gpu_writes) {
		// Not asking the driver: a write that finished meanwhile puts the bytes guest memory already holds (no unknown
		// write after it overlaps them: those were all waited for above).
		if (w.start < vaddr + size && vaddr < w.end && w.cp_size != 0u && !master.IsFree(w.tick)) {
			m_direct_cp_overlay.push_back(w);
		}
	}
	if (!m_direct_cp_overlay.empty()) g_direct_overlay_reads.fetch_add(1, std::memory_order_relaxed);
	return !m_direct_cp_overlay.empty();
}

void BufferCache::UnmapDirect(uint64_t vaddr, uint64_t size) {
	for (auto& range: m_direct_ranges) {
		if (range.active && vaddr < range.end && range.start < vaddr + size) DeactivateDirect(range);
	}
}

// The resolve thread's lookups (stageFLookup.h), resolve thread between TryLockLookup and UnlockLookup: FindBuffer's answer
// when it finds a buffer; a null id where FindBuffer would return NULL_BUFFER_ID, exit or create.
BufferId BufferCache::LookupBuffer(uint64_t vaddr, uint64_t size) const {
	if (vaddr == 0 || !GuestRange {vaddr, size}.Valid()) {
		return {};
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		const auto* buffer = m_slot_buffers.try_get(*owner);
		if (buffer != nullptr && buffer->IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return {};
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	// KYTY_PERF_DIRECT_RANGES: a direct buffer is never joined. Requests over a direct range never get here (FindBuffer
	// hands out the direct buffer); a new buffer next to one stops at it (no stream leap into it).
	if (!m_direct_ranges.empty() &&
	    std::any_of(first, last, [&](const auto& entry) { return m_slot_buffers[entry.second].IsDirect(); })) {
		begin           = vaddr;
		end             = vaddr + size;
		has_stream_leap = false;
		first           = find_first(begin);
		last            = first;
		for (; last != m_buffers.end() && last->first < end; ++last) {
			const auto& buffer = m_slot_buffers[last->second];
			if (buffer.IsDirect()) break;
			begin = std::min(begin, buffer.CpuAddress());
			end   = std::max(end, buffer.CpuAddress() + buffer.Size());
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	if (UploadClobberCheck()) C0Note(overlap.CpuAddress(), overlap.Size(), C0Event::Join, overlap.CpuAddress()); // D4 K0
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	// The upload batch: claimed uploads hold buffer pointers (the slot storage may move) and a joined buffer copies
	// its overlaps' contents, which must include them.
	CommitUploadBatch(BatchCommit::CreateBuffer);
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const Common::StartupTrace::Scope startup_scope(Common::StartupTrace::Counter::BufferCreate, size); // KYTY_TRACE_STARTUP
	if (VramBudgetEnabled()) {
		MakeRoomForBuffer(vaddr, size);
	}
	const auto overlap = ResolveOverlaps(vaddr, size);
	// Diagnostic (black textures at full VRAM: one 2064 MiB buffer appeared): a buffer of 256 MiB or more,
	// with the request that made it and the buffers it joins (first 64 lines).
	if (overlap.end - overlap.begin >= (uint64_t {256} << 20u)) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
			uint64_t joined = 0, joined_bytes = 0, largest = 0;
			for (auto it = overlap.first; it != overlap.last; ++it) {
				const auto bytes = m_slot_buffers[it->second].Size();
				joined++;
				joined_bytes += bytes;
				largest = std::max(largest, bytes);
			}
			std::fprintf(stderr,
			             "BIG_BUFFER_CREATE request=0x%016llx+0x%llx span=0x%016llx+0x%llx (%llu MiB) joined=%llu "
			             "joined_MiB=%llu largest_joined_MiB=%llu stream_leap=%d\n",
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			             static_cast<unsigned long long>(overlap.begin),
			             static_cast<unsigned long long>(overlap.end - overlap.begin),
			             static_cast<unsigned long long>((overlap.end - overlap.begin) >> 20u),
			             static_cast<unsigned long long>(joined), static_cast<unsigned long long>(joined_bytes >> 20u),
			             static_cast<unsigned long long>(largest >> 20u), overlap.has_stream_leap ? 1 : 0);
			PrintBigBufferBacktrace();
		}
	}

	const auto id = [&] {
		const LookupWriteGuard lookup_guard(*this); // the resolve thread's lookups: the slot storage may grow
		return m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
		                             AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                             overlap.end - overlap.begin);
	}();
	const auto& buffer = m_slot_buffers[id];
	if (UploadClobberCheck()) C0Note(overlap.begin, overlap.end - overlap.begin, C0Event::Create, overlap.end - overlap.begin); // D4 K0
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	// Its contents come from copies recorded now (joined overlaps).
	m_buffer_ready_tick[reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.Handle()))] = m_scheduler.CurrentTick();
	return id;
}

// KYTY_VRAM_BUDGET: before a buffer is created over the target, buffers the collector would retire anyway (unused
// for 160+ collector ticks, the age of its non-aggressive pass) that hold nothing guest memory lacks (no GPU-written
// bytes, no pending readback, not BDA-pinned) are retired, oldest first. Buffers near the new range are left to
// ResolveOverlaps (it may join them). The texture cache is not touched here: its lock may be held by the caller.
void BufferCache::MakeRoomForBuffer(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t MinAge  = 160;
	constexpr uint64_t Margin  = CACHING_PAGESIZE * 128; // ResolveOverlaps' stream leap
	constexpr uint64_t Slack   = 64ull << 20u;           // retires a little more than needed: fewer scans
	constexpr size_t   MaxScan = 4096;
	if (!m_graphics.CanReportMemoryUsage() || m_gc_tick <= MinAge) {
		return;
	}
	uint64_t target = 0, effective = 0;
	m_graphics.GetVramBudgetState(target, effective);
	if (target == 0) {
		return;
	}
	const uint64_t pending = RecentlyRetiredBytes();
	const uint64_t used    = effective > pending ? effective - pending : 0;
	// Over the target: a scan that found nothing to retire at this collector tick is not repeated for every buffer.
	if (used + size <= target || m_make_room_dry_tick == m_gc_tick) {
		return;
	}
	uint64_t              need     = used + size - target + Slack;
	size_t                examined = 0;
	const uint64_t        low      = vaddr > Margin ? vaddr - Margin : 0;
	const uint64_t        high     = vaddr + size + Margin;
	std::vector<BufferId> victims;
	const auto            consider = [&](BufferId id) {
		if (need == 0 || examined == MaxScan) {
			return true;
		}
		examined++;
		const auto& buffer  = m_slot_buffers[id];
		const auto  address = buffer.CpuAddress();
		const auto  bytes   = buffer.Size();
		if (buffer.is_deleted || buffer.IsDirect() || (address < high && low < address + bytes) ||
		    m_gpu_modified_ranges.Intersects(address, bytes) || m_memory_tracker.IsRegionGpuModified(address, bytes) ||
		    HasPendingReadback(address, bytes)) {
			return false;
		}
		victims.push_back(id);
		need -= std::min(need, bytes);
		return false;
	};
	m_lru_cache.ForEachItemBelow(m_gc_tick - MinAge, consider);
	if (victims.empty()) {
		m_make_room_dry_tick = m_gc_tick;
		return;
	}
	for (const auto id: victims) {
		if (IsBufferInvalid(id)) {
			continue;
		}
		auto& buffer = m_slot_buffers[id];
		m_alloc_retired++;
		m_alloc_retired_bytes += buffer.Size();
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		NoteRetiredBytes(buffer.Size());
		DeleteBuffer(id);
	}
}

// KYTY_VRAM_BUDGET: a retired buffer is destroyed once the GPU retired the command buffer recording at this tick
// (deferred deletion); until then the measured usage still holds it. Bytes are kept per tick until the tick is free.
void BufferCache::NoteRetiredBytes(uint64_t bytes) {
	if (!VramBudgetEnabled()) {
		return;
	}
	const uint64_t tick = m_scheduler.CurrentTick();
	if (!m_releasing.empty() && m_releasing.back().tick == tick) {
		m_releasing.back().bytes += bytes;
	} else {
		m_releasing.push_back({tick, bytes});
	}
}

uint64_t BufferCache::RecentlyRetiredBytes() {
	uint64_t total = 0;
	size_t   kept  = 0;
	for (size_t i = 0; i < m_releasing.size(); i++) {
		const auto entry = m_releasing[i];
		if (m_scheduler.IsFree(entry.tick)) {
			continue;
		}
		m_releasing[kept++] = entry;
		total += entry.bytes;
	}
	m_releasing.resize(kept);
	return total;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer, uint64_t* uploaded_bytes, UploadCaller caller) {
	if (buffer.IsDirect()) return false; // KYTY_PERF_DIRECT_RANGES: its memory is guest memory
	if (Batch().active) {
		// The upload batch: a plain read-only synchronization only claims its pages. A written one (its GPU marks
		// must follow the claimed copies), a texel buffer (the image copies below must follow its upload) and an image
		// source (copied from at once) commit the batch first. So does a caller that needs the uploaded byte count of
		// this very call (ObtainBuffer's memo path), except the region pass, whose count only feeds statistics.
		const bool deferrable = !is_written && !is_texel_buffer && caller != UploadCaller::Image &&
		                        (uploaded_bytes == nullptr || caller == UploadCaller::Region);
		if (deferrable) {
			ClaimBatchedUpload(buffer, vaddr, size, uploaded_bytes);
			return false;
		}
		const DispatchSubTimer commit_timer(is_written ? DispatchSub::OwBatchCommit : DispatchSub::Count); // KYTY_TRACE_DISPATCH_CENSUS
		CommitUploadBatch(BatchCommit::Sync);
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	std::optional<DispatchSubTimer> walk_timer(std::in_place, is_written ? DispatchSub::OwUploadWalk : DispatchSub::Count);
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	walk_timer.reset();
	if (source) {
		if (uploaded_bytes != nullptr) {
			*uploaded_bytes = total_size;
		}
		RecordBufferUpload(buffer, source, copies);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}
	// The upload batch: the image copy below must land after the claimed uploads of the buffer's bytes (committed
	// before the texture cache's lock is taken).
	CommitUploadBatch(BatchCommit::ImageCopy);

	std::unique_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain.
		return false;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	if (!m_texture_cache.CanDownload(image) ||
	    (TextureCache::UploadBinding(image) == TextureCache::BindingType::DepthTarget &&
	     copy_size != image.info.data.size)) {
		return false;
	}
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, levels);
	if (UploadClobberCheck()) C0Note(image.info.data.address, copy_size, C0Event::ImageCopy, image.info.data.address); // D4 K0
	// The copy rewrites the cached buffer's bytes in the open command buffer; host-mapped reads of these granules wait for
	// it too.
	NoteGpuWriteTick(image.info.data.address, copy_size);
	return true;
}

// A read-only buffer covering an image the GPU wrote since the last such copy takes
// the image contents first. Compute shaders that copy images through raw buffers (Ghost of Yotei: CS 4337635433d63234
// copies the sky LUTs written as images by 95c013c75f8e77ad) otherwise read stale guest bytes.
bool BufferCache::SyncBufferFromWrittenImage(Buffer& buffer, uint64_t vaddr, uint64_t size, bool* found_image) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (found_image != nullptr) *found_image = static_cast<bool>(selected);
	if (!selected) {
		return false;
	}
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		auto& image = m_texture_cache.m_slot_images[selected];
		if (!image.IsGpuModified() || image.buffer_sync_serial == image.content_serial) {
			return false;
		}
		image.buffer_sync_serial = image.content_serial;
	}
	return SynchronizeBufferFromImage(buffer, vaddr, size);
}

void BufferCache::RecordBufferUpload(Buffer& buffer, vk::Buffer source, std::span<const vk::BufferCopy> copies) {
	if (UploadClobberCheck()) {
		m_scheduler.Current().EndRendering();
		CheckUploadClobber(buffer, copies); // KYTY_D6_C0=2/3: the GPU copy, right before
		for (const auto& copy: copies) C0Note(buffer.CpuAddress() + copy.dstOffset, copy.size, C0Event::Upload);
	}
	RecordBufferUploadCommands(buffer.Handle(), buffer.Size(), source, copies);
}

void BufferCache::RecordBufferUploadCommands(vk::Buffer target, uint64_t target_size, vk::Buffer source,
                                             std::span<const vk::BufferCopy> copies) {
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();
	const bool label  = GpuTimer::LabelsEnabled(); // KYTY_TRACE_VK_LABELS
	if (label) GpuTimer::LabelEmu(native, "buffer upload");
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                       vk::AccessFlagBits::eTransferRead |
	                       vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = target;
	before.offset              = 0;
	before.size                = target_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
	GpuCheckpoints::Mark(native, 0xFEED000000000000ull | (1ull << 16u) | 402u);
	native.copyBuffer(source, target, static_cast<uint32_t>(copies.size()), copies.data());
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	if (label) GpuTimer::LabelEnd(native);
}

// ---- The upload batch (two-phase uploads, see the top of this file) ------------------------------------------------

BufferCache::UploadBatchScope::UploadBatchScope(BufferCache& cache) {
	// GPU thread only (the batch, the staging buffer and the command buffer are its own); read once per scope.
	if (cache.Batch().active || !GuestGpu::IsGpuThread()) {
		return;
	}
	cache.Batch().active = true;
	m_cache                     = &cache;
	cache.m_protect_batch_stats.scopes.fetch_add(1, std::memory_order_relaxed);
}

BufferCache::UploadBatchScope::~UploadBatchScope() {
	if (m_cache == nullptr) {
		return;
	}
	// Closed before the final commit: nothing the commit runs (staging waits, deferred work) can claim into a batch
	// that would then outlive its scope.
	auto& batch  = m_cache->Batch();
	batch.active = false;
	m_cache->CommitUploadBatch(BatchCommit::ScopeEnd);
	EXIT_IF(!batch.entries.empty() || !batch.ranges.empty() || !batch.regions.empty());
}

BufferCache::UploadBatchPause::UploadBatchPause(BufferCache& cache, bool pause) {
	if (!pause || !cache.Batch().active) {
		return;
	}
	cache.CommitUploadBatch(BatchCommit::Sync);
	cache.Batch().active = false;
	m_cache                     = &cache;
}

BufferCache::UploadBatchPause::~UploadBatchPause() {
	if (m_cache == nullptr) {
		return;
	}
	m_cache->CommitUploadBatch(BatchCommit::Sync); // (nothing: no claims while paused)
	m_cache->Batch().active = true;
}

void BufferCache::ClaimBatchedUpload(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t* uploaded_bytes) {
	auto&      batch       = Batch();
	const auto first_range = batch.ranges.size();
	uint64_t   claimed     = 0;
	const auto note_range  = [&](uint64_t address, uint64_t bytes) noexcept {
		batch.ranges.emplace_back(address, bytes);
		claimed += bytes;
	};
	const auto regions = m_memory_tracker.ClaimUploadRanges(vaddr, size, note_range, batch.regions);
	if (uploaded_bytes != nullptr) {
		*uploaded_bytes = claimed;
	}
	if (batch.ranges.size() == first_range) {
		return;
	}
	batch.entries.push_back({&buffer, static_cast<uint32_t>(first_range),
	                         static_cast<uint32_t>(batch.ranges.size() - first_range)});
	auto& stats = m_protect_batch_stats;
	stats.claims.fetch_add(1, std::memory_order_relaxed);
	stats.claim_regions.fetch_add(regions, std::memory_order_relaxed);
	stats.claimed_bytes.fetch_add(claimed, std::memory_order_relaxed);
}

void BufferCache::CommitUploadBatch(BatchCommit reason) {
	auto& batch = Batch();
	if (batch.entries.empty() && batch.regions.empty()) {
		return;
	}
	const bool timed = PerfEnabled();
	const auto began = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	auto&      stats = m_protect_batch_stats;
	struct Recheck {
		uint64_t address;
		uint64_t size;
		uint64_t hash;
	};
	thread_local std::vector<Recheck>  rechecks;
	thread_local std::vector<uint8_t>  scratch;
	const size_t                       recheck_base = rechecks.size(); // a nested commit appends after ours
	bool                               protected_now = false;
	uint32_t updated = 0;
	// Phase 3: the copies, in claim order. The lists are taken out first: staging can wait for the GPU, and deferred
	// work that runs meanwhile may commit too; it must find an empty batch.
	std::vector<UploadBatchEntry>              entries;
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	entries.swap(batch.entries);
	ranges.swap(batch.ranges);
	std::vector<vk::BufferCopy> copies;
	for (const auto& entry: entries) {
		auto&    buffer     = *entry.buffer;
		uint64_t total_size = 0;
		copies.clear();
		for (uint32_t index = 0; index < entry.range_count; ++index) {
			const auto [address, bytes] = ranges[entry.first_range + index];
			// SynchronizeBuffer's range function, now that the pages are write-protected.
			copies.emplace_back(total_size, buffer.Offset(address), bytes);
			total_size += bytes;
		}
		vk::Buffer source;
		if (!protected_now && !copies.empty()) {
			// Stage through scratch, hashing exactly the staged bytes.
			auto [mapped, base_offset] = Staging().Map(total_size, 4);
			if (mapped == nullptr) {
				// No staging room: from here on the usual order (protect, then copy) for this commit.
				updated += m_memory_tracker.CommitDeferredProtections(batch.regions);
				protected_now = true;
			} else {
				if (UploadClobberCheck()) m_c0_staged.clear(); // KYTY_D6_C0=2/3: the staged bytes are kept
				for (auto& copy: copies) {
					const auto address = buffer.CpuAddress() + copy.dstOffset;
					scratch.resize(copy.size);
					LibKernel::Memory::ReadGuestPlain(scratch.data(), address, copy.size);
					if (UploadClobberCheck()) NoteC0Staged(address, scratch.data(), copy.size);
					rechecks.push_back({address, copy.size, XXH3_64bits(scratch.data(), copy.size)});
					std::memcpy(mapped + copy.srcOffset, scratch.data(), copy.size);
					copy.srcOffset += base_offset;
				}
				Staging().Commit();
				source = Staging().Handle();
				// (UploadCopies counts the uploads it stages itself.)
				Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
			}
		}
		if (!source) {
			source = UploadCopies(buffer, copies, total_size);
		}
		if (!source) {
			continue;
		}
		RecordBufferUpload(buffer, source, copies);
	}
	// Protect what is still writable, then re-check this commit's ranges.
	std::vector<Recheck>                       checks(rechecks.begin() + static_cast<ptrdiff_t>(recheck_base),
	                                                  rechecks.end());
	rechecks.resize(recheck_base);
	updated += m_memory_tracker.CommitDeferredProtections(batch.regions);
	uint64_t changed = 0;
	uint64_t bytes   = 0;
	for (size_t k = 0; k < checks.size(); k++) {
		const auto& check = checks[k];
		bytes += check.size;
		// A page under the image read watch is hashed as it was copied (ReadGuestPlain).
		uint64_t hash = 0;
		if (LibKernel::Memory::GuestImageReadArmed(check.address, check.size)) {
			std::vector<uint8_t> armed(check.size);
			LibKernel::Memory::ReadGuestPlain(armed.data(), check.address, check.size);
			hash = XXH3_64bits(armed.data(), armed.size());
		} else {
			hash = XXH3_64bits(reinterpret_cast<const void*>(check.address), check.size);
		}
		const bool differs = hash != check.hash;
		if (differs) {
			m_memory_tracker.MarkRegionAsCpuModified(check.address, check.size);
			changed++;
		}
	}
	const uint64_t done = g_pac_commits.fetch_add(1, std::memory_order_relaxed) + 1;
	g_pac_ranges.fetch_add(checks.size(), std::memory_order_relaxed);
	g_pac_bytes.fetch_add(bytes, std::memory_order_relaxed);
	g_pac_changed.fetch_add(changed, std::memory_order_relaxed);
	if (PerfEnabled() && (done & 4095u) == 0) {
		std::fprintf(stderr, "PERF protect_after_copy commits=%llu ranges=%llu kb=%llu changed=%llu\n",
		             static_cast<unsigned long long>(done),
		             static_cast<unsigned long long>(g_pac_ranges.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(g_pac_bytes.load(std::memory_order_relaxed) / 1024u),
		             static_cast<unsigned long long>(g_pac_changed.load(std::memory_order_relaxed)));
	}
	// Keep the lists' storage for the next batch (unless a nested claim refilled them meanwhile).
	entries.clear();
	ranges.clear();
	if (batch.entries.empty()) batch.entries.swap(entries);
	if (batch.ranges.empty()) batch.ranges.swap(ranges);
	stats.commits.fetch_add(1, std::memory_order_relaxed);
	stats.region_updates.fetch_add(updated, std::memory_order_relaxed);
	stats.reasons[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	if (timed) {
		stats.commit_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                    std::chrono::steady_clock::now() - began)
		                                                    .count()),
		                          std::memory_order_relaxed);
	}
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (!m_append_windows.empty()) { // D4b step 1 (append windows)
		for (const auto& copy: copies) AppendNoteUpload(buffer.CpuAddress() + copy.dstOffset, copy.size);
	}
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);
	// KYTY_TRACE_STARTUP: guest bytes staged for the GPU (tmp: the staging buffer had no room).
	const Common::StartupTrace::Scope startup_scope(Common::StartupTrace::Counter::BufferUpload, total_size);

	const bool c0_check = UploadClobberCheck(); // KYTY_D6_C0=2/3: the staged bytes are kept (m_c0_staged)
	if (c0_check) m_c0_staged.clear();
	auto [mapped, base_offset] = Staging().Map(total_size, 4);
	if (mapped != nullptr) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (c0_check) {
				NoteC0Staged(address, reinterpret_cast<const uint8_t*>(address), copy.size);
				std::memcpy(mapped + copy.srcOffset, m_c0_staged.back().second.data(), copy.size);
			} else {
				LibKernel::Memory::ReadGuestPlain(mapped + copy.srcOffset, address, copy.size);
			}
			copy.srcOffset += base_offset;
		}
		Staging().Commit();
		return Staging().Handle();
	}

	if (Common::StartupTrace::Enabled()) Common::StartupTrace::Add(Common::StartupTrace::Counter::BufferUploadTemp, 0);
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (const auto& copy: copies) {
		const auto address = buffer.CpuAddress() + copy.dstOffset;
		if (c0_check) {
			NoteC0Staged(address, reinterpret_cast<const uint8_t*>(address), copy.size);
			std::memcpy(temporary->Mapped().data() + copy.srcOffset, m_c0_staged.back().second.data(), copy.size);
		} else {
			LibKernel::Memory::ReadGuestPlain(temporary->Mapped().data() + copy.srcOffset, address, copy.size);
		}
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

// The clean-range early return (the rest of BryanKAdams 151d7916 part 3, whose RegionManager::ForEachModifiedRange early
// return is already here as 0bb2bdad): ObtainBuffer checks a small read-only range for CPU-dirty pages under the region
// locks (IsRegionCpuModified) before it streams a dirty one; a range found clean then went through SynchronizeBuffer,
// which walked the same pages under the same locks again only to find nothing to upload. Such a range now skips that walk
// (a texel buffer still synchronizes from images). A guest write landing between the check and the walk is a race with
// this draw either way: its page stays CPU-dirty (unprotected) and the next synchronization of the range uploads it.
// The ranges found clean: a PERF line about every 10 s (checked every 4096 ranges).
static void NoteCleanRange() {
	static std::atomic<uint64_t> skipped {0};
	static std::atomic<int64_t>  last_log {0};
	const auto ranges = skipped.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((ranges & 4095u) != 0 || !Common::EnvRead::PerfLines()) return;
	const auto now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto last = last_log.load(std::memory_order_relaxed);
	if (now - last >= 10 && last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		std::fprintf(stderr, "PERF clean_range clean_ranges=%llu\n",
		             static_cast<unsigned long long>(skipped.load(std::memory_order_relaxed)));
	}
}

// image_source: the range is the guest data of an image the texture cache is about to upload (ObtainBufferForImage,
// called with the texture cache locked); it is never synced from a GPU-written image (that would copy the image into
// its own upload source, and re-enter the texture cache's lock).
// The obtain memo: a read-only ObtainBuffer of a range the GPU thread obtained before from the cached path takes the same
// buffer without the tracker queries, the buffer search, the synchronization and the written-image search, while nothing
// that could change their answer happened since: no buffer was registered or unregistered (g_buffer_registry_epoch), no
// page of the range's 4 MiB tracker regions became CPU-dirty (g_cpu_dirty_region_epochs, bumped by every CPU-dirty mark,
// immediate or deferred), no image was registered or unregistered over its 1 MiB pages (image page epochs), and when it
// was kept the range was CPU-clean after its synchronization and no image lay over it. A later CPU write therefore always
// misses; a GPU write into the buffer leaves the cached path's answer as it was (its bytes stay in the buffer).

namespace {
struct ObtainMemoEntry {
	uint64_t vaddr        = 0;
	uint64_t size         = 0;
	uint64_t buffer_epoch = 0;
	uint64_t image_sum    = 0;
	uint64_t region       = 0; // the first 4 MiB tracker region of the range
	uint32_t epochs[2] {};
	uint32_t regions = 0;      // 1 or 2; 0: empty
	BufferId id {};
};
constexpr size_t kObtainMemoSlots = 8192;
ObtainMemoEntry  g_obtain_memo[kObtainMemoSlots]; // GPU thread
struct ObtainMemoStats {
	uint64_t hits = 0, misses = 0, stores = 0;
};
ObtainMemoStats              g_obtain_memo_stats_shared;
} // namespace

static ObtainMemoStats& ObtainMemoStatsRef() {
	return g_obtain_memo_stats_shared;
}

namespace {

size_t ObtainMemoIndex(uint64_t vaddr, uint64_t size) {
	uint64_t h = vaddr * 0x9E3779B97F4A7C15ull ^ (size + (size << 17u));
	h ^= h >> 29u;
	return static_cast<size_t>(h & (kObtainMemoSlots - 1u));
}

// The tracker regions of the range (at most 2 are kept) and their epochs now; false when the range spans more.
bool ObtainMemoRegions(uint64_t vaddr, uint64_t size, uint64_t& first, uint32_t& count, uint32_t (&epochs)[2]) {
	auto* table = g_cpu_dirty_region_epochs.load(std::memory_order_acquire);
	if (size == 0) return false;
	first             = vaddr / TRACKER_REGION_SIZE;
	const uint64_t last = (vaddr + size - 1u) / TRACKER_REGION_SIZE;
	if (last - first > 1u || last >= CPU_DIRTY_REGION_COUNT) return false;
	count = static_cast<uint32_t>(last - first + 1u);
	for (uint32_t i = 0; i < count; i++) epochs[i] = table[first + i].load(std::memory_order_acquire);
	return true;
}

// The entry of the range when it still holds; null when it does not.
const ObtainMemoEntry* ObtainMemoFind(uint64_t vaddr, uint64_t size) {
	const ObtainMemoEntry* entry = &g_obtain_memo[ObtainMemoIndex(vaddr, size)];
	const auto&            e     = *entry;
	if (e.regions == 0 || e.vaddr != vaddr || e.size != size ||
	    e.buffer_epoch != DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_acquire)) {
		return nullptr;
	}
	auto* table = g_cpu_dirty_region_epochs.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < e.regions; i++) {
		if (table[e.region + i].load(std::memory_order_acquire) != e.epochs[i]) return nullptr;
	}
	return DrawRepeat::ImagePageEpochSum(vaddr, size) == e.image_sum ? entry : nullptr;
}

void ObtainMemoStore(uint64_t vaddr, uint64_t size, const ObtainMemoEntry& fresh) {
	g_obtain_memo[ObtainMemoIndex(vaddr, size)] = fresh;
}

void ObtainMemoReport() {
	auto& s = ObtainMemoStatsRef();
	if (((s.hits + s.misses) & 0x3ffffu) != 0 || !Common::EnvRead::PerfLines()) return;
	std::fprintf(stderr, "PERF obtain_memo hits=%llu misses=%llu stores=%llu\n", static_cast<unsigned long long>(s.hits),
	             static_cast<unsigned long long>(s.misses), static_cast<unsigned long long>(s.stores));
}
} // namespace

// The write memo: a written ObtainBuffer (GPU thread, not an image source) of a range the GPU thread obtained for writing
// before, through the full path, in the same command buffer (the same scheduler tick), returns the same buffer and keeps
// only the write's notifications, while nothing that the full path would redo happened since: no buffer was registered or
// unregistered (g_buffer_registry_epoch), no page of the range's tracker regions became CPU-dirty
// (g_cpu_dirty_region_epochs) and none stopped being GPU-modified or left the GPU-dirty set (g_gpu_clean_region_epochs:
// downloads, readbacks, unmarks, untracks, GPU-dirty subtractions). Then the full path would upload nothing (no CPU-dirty
// page), mark pages that are GPU-modified already, add a range the GPU-dirty set holds, and note write ticks and granule
// ticks the same tick already holds: it would leave the same state. The notifications still run (GPU-side write
// generation, the GPU-write observer, hot-read generations, the BAR pending write for the recording
// tick, traces), and the upload batch commits at its scope's end as always.

namespace {
struct WriteMemoEntry {
	uint64_t vaddr        = 0;
	uint64_t size         = 0;
	uint64_t tick         = 0;
	uint64_t buffer_epoch = 0;
	uint64_t region       = 0; // the first 4 MiB tracker region of the range
	uint32_t cpu_epochs[2] {};
	uint32_t gpu_epochs[2] {};
	uint32_t regions = 0; // 1 or 2; 0: empty
	BufferId id {};
};
constexpr size_t kWriteMemoSlots = 8192;
WriteMemoEntry   g_write_memo[kWriteMemoSlots]; // GPU thread
struct {
	uint64_t hits = 0, retick_hits = 0, misses = 0, stores = 0;
	// Misses by the first reason: 0 empty or another range, 1 another tick, 2 buffer registry, 3 CPU-dirty epoch,
	// 4 GPU-clean epoch, 5 another buffer id / out of bounds.
	uint64_t miss_reason[6] {};
} g_write_memo_stats;

WriteMemoEntry& WriteMemoSlot(uint64_t vaddr, uint64_t size) {
	uint64_t h = vaddr * 0x9E3779B97F4A7C15ull ^ (size + (size << 21u));
	h ^= h >> 31u;
	return g_write_memo[h & (kWriteMemoSlots - 1u)];
}

// The tracker regions of the range (at most 2) and both epochs of each now; false when the range spans more.
bool WriteMemoEpochs(uint64_t vaddr, uint64_t size, WriteMemoEntry& e) {
	auto* cpu = g_cpu_dirty_region_epochs.load(std::memory_order_acquire);
	auto* gpu = g_gpu_clean_region_epochs.load(std::memory_order_acquire);
	if (size == 0) return false;
	const uint64_t first = vaddr / TRACKER_REGION_SIZE;
	const uint64_t last  = (vaddr + size - 1u) / TRACKER_REGION_SIZE;
	if (last - first > 1u || last >= CPU_DIRTY_REGION_COUNT) return false;
	e.region  = first;
	e.regions = static_cast<uint32_t>(last - first + 1u);
	for (uint32_t i = 0; i < e.regions; i++) {
		e.cpu_epochs[i] = cpu[first + i].load(std::memory_order_acquire);
		e.gpu_epochs[i] = gpu[first + i].load(std::memory_order_acquire);
	}
	return true;
}

WriteMemoEntry* WriteMemoFind(uint64_t vaddr, uint64_t size, uint32_t& reason) {
	auto& e = WriteMemoSlot(vaddr, size);
	reason  = 0;
	if (e.regions == 0 || e.vaddr != vaddr || e.size != size) return nullptr;
	reason = 2;
	if (e.buffer_epoch != DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_acquire)) return nullptr;
	auto* cpu = g_cpu_dirty_region_epochs.load(std::memory_order_acquire);
	auto* gpu = g_gpu_clean_region_epochs.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < e.regions; i++) {
		reason = 3;
		if (cpu[e.region + i].load(std::memory_order_acquire) != e.cpu_epochs[i]) return nullptr;
		reason = 4;
		if (gpu[e.region + i].load(std::memory_order_acquire) != e.gpu_epochs[i]) return nullptr;
	}
	reason = 5;
	return &e;
}

void WriteMemoReport() {
	auto& s = g_write_memo_stats;
	if (((s.hits + s.misses) & 0x3ffffu) != 0 || !Common::EnvRead::PerfLines()) return;
	std::fprintf(stderr,
	             "PERF write_memo hits=%llu (later tick %llu) misses=%llu stores=%llu "
	             "miss(range/-/registry/cpu/gpu/buffer)=%llu/%llu/%llu/%llu/%llu/%llu\n",
	             static_cast<unsigned long long>(s.hits), static_cast<unsigned long long>(s.retick_hits),
	             static_cast<unsigned long long>(s.misses),
	             static_cast<unsigned long long>(s.stores), static_cast<unsigned long long>(s.miss_reason[0]),
	             static_cast<unsigned long long>(s.miss_reason[1]), static_cast<unsigned long long>(s.miss_reason[2]),
	             static_cast<unsigned long long>(s.miss_reason[3]), static_cast<unsigned long long>(s.miss_reason[4]),
	             static_cast<unsigned long long>(s.miss_reason[5]));
}

void PrefetchEntry(const void* entry, size_t bytes) {
	const auto* first = static_cast<const char*>(entry);
	_mm_prefetch(first, _MM_HINT_T0);
	_mm_prefetch(first + bytes - 1u, _MM_HINT_T0); // (an entry may cross a cache line)
}
} // namespace

void BufferCache::PrefetchObtainMemo(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer) {
	if (is_written) {
		// (The write memo serves the GPU thread only.)
		if (!GuestGpu::IsGpuThread()) return;
		PrefetchEntry(&WriteMemoSlot(vaddr, size), sizeof(WriteMemoEntry));
	} else {
		if (is_texel_buffer) return;
		PrefetchEntry(&g_obtain_memo[ObtainMemoIndex(vaddr, size)], sizeof(ObtainMemoEntry));
	}
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id, bool image_source) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	// KYTY_PERF_DIRECT_RANGES: a direct range's bytes are the buffer's memory; nothing to upload, protect or snapshot.
	if (const BufferId direct = DirectBufferFor(vaddr, size); direct) {
		auto& buffer = m_slot_buffers[direct];
		if (is_written) NoteObtainedWrite(vaddr, size);
		return {&buffer, buffer.Offset(vaddr)};
	}
	// The write memo: the diagnostic and approximation paths below never use it.
	const bool write_memo = is_written && !image_source && GuestGpu::IsGpuThread() && !InDirectRange(vaddr, size);
	WriteMemoEntry write_memo_fresh; // the epochs before the full path (a change during it makes the entry stale)
	bool           write_memo_keep = false;
	if (write_memo) {
		const uint64_t tick = m_scheduler.CurrentTick();
		uint32_t       reason = 0;
		auto*          hit    = WriteMemoFind(vaddr, size, reason);
		auto&          s      = g_write_memo_stats;
		if (hit != nullptr && (IsBufferInvalid(id) || id == hit->id) && m_slot_buffers[hit->id].IsInBounds(vaddr, size)) {
			// A later tick: the tracker and the GPU-dirty set are as the full path would leave them, the write ticks are not.
			const bool same_tick = hit->tick == tick;
			s.hits++;
			if (!same_tick) s.retick_hits++;
			WriteMemoReport();
			auto& buffer = m_slot_buffers[hit->id];
			TouchBuffer(buffer);
			NoteObtainedWriteRepeat(vaddr, size, same_tick);
			hit->tick = tick; // the write ticks hold this tick now
			return {&buffer, buffer.Offset(vaddr)};
		}
		s.misses++;
		s.miss_reason[reason]++;
		WriteMemoReport();
		write_memo_fresh.vaddr        = vaddr;
		write_memo_fresh.size         = size;
		write_memo_fresh.tick         = tick;
		write_memo_fresh.buffer_epoch = DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_acquire);
		write_memo_keep               = WriteMemoEpochs(vaddr, size, write_memo_fresh);
	}
	// The obtain memo: a read of a kept range (the diagnostic and approximation paths below never use it).
	const bool memo_on = !is_written && !is_texel_buffer && !image_source;
	const ObtainMemoEntry* memo_hit = memo_on ? ObtainMemoFind(vaddr, size) : nullptr;
	if (memo_on) {
		auto& s = ObtainMemoStatsRef();
		(memo_hit != nullptr ? s.hits : s.misses)++;
		ObtainMemoReport();
		if (memo_hit != nullptr) {
			auto& buffer = m_slot_buffers[memo_hit->id];
			TouchBuffer(buffer);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	// The upload batch: an image source is copied from as soon as this returns; its upload is not deferred.
	const UploadBatchPause batch_pause(*this, image_source);

	// The range was found CPU-clean below.
	bool found_clean = false;
	if (!is_written && size <= CACHING_PAGESIZE && !m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		const bool cpu_modified = m_memory_tracker.IsRegionCpuModified(vaddr, size);
		if (cpu_modified) {
			const auto alignment = std::max<uint64_t>(
			    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
			auto [mapped, offset] = Stream().Map(size, alignment, false);
			if (mapped != nullptr) {
				LibKernel::Memory::ReadGuestPlain(mapped, vaddr, size);
				Stream().Commit();
				return {&Stream(), offset};
			}
		}
		found_clean = !cpu_modified;
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	uint64_t   memo_uploaded  = 0; // the obtain memo
	if (found_clean) {
		// Nothing to upload; SynchronizeBuffer's remaining work for a read.
		if (is_texel_buffer) {
			(void)SynchronizeBufferFromImage(buffer, vaddr, size);
		}
		NoteCleanRange();
	} else {
		(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer, memo_on ? &memo_uploaded : nullptr,
		                        UploadCaller::Obtain);
	}
	bool image_over = false; // the obtain memo
	bool image_sync = false;
	if (!is_written && !is_texel_buffer && !image_source &&
	    (image_sync = SyncBufferFromWrittenImage(buffer, vaddr, size, &image_over))) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 32u) {
			std::fprintf(stderr, "BUFFER_SYNC_FROM_IMAGE frame=%u addr=0x%llx size=0x%llx\n", DebugPresentCount().load(),
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
		}
	}
	if (is_written) {
		NoteObtainedWrite(vaddr, size);
	}
	// The write memo: kept with the epochs read before the full path, in the command buffer it was noted for.
	if (write_memo_keep && m_scheduler.CurrentTick() == write_memo_fresh.tick) {
		write_memo_fresh.id = id;
		WriteMemoSlot(vaddr, size) = write_memo_fresh;
		g_write_memo_stats.stores++;
	}
	if (memo_on) {
		// Kept: a range the cached path left CPU-clean, with no image over it. The epochs are read after the work above, so
		// a CPU-dirty mark during it makes the entry stale at once.
		ObtainMemoEntry fresh;
		if (!image_over && !image_sync && ObtainMemoRegions(vaddr, size, fresh.region, fresh.regions, fresh.epochs) &&
		    !m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
			fresh.vaddr        = vaddr;
			fresh.size         = size;
			fresh.buffer_epoch = DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_acquire);
			fresh.image_sum    = DrawRepeat::ImagePageEpochSum(vaddr, size);
			fresh.id           = id;
			ObtainMemoStore(vaddr, size, fresh);
			ObtainMemoStatsRef().stores++;
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

void BufferCache::NoteObtainedWrite(uint64_t vaddr, uint64_t size) {
	if (!m_append_pools.empty() || !m_append_windows.empty()) AppendForeignWrite(vaddr, size); // D4b step 1 (append windows)
	if (UploadClobberCheck()) { // D4 step K0
		C0Note(vaddr, size, InDirectRange(vaddr, size) ? C0Event::DirectWrite : C0Event::ShaderWrite,
		       DebugWriterHash().load(std::memory_order_relaxed));
	}
	if (InDirectRange(vaddr, size)) {
		// KYTY_PERF_DIRECT_RANGES: the GPU writes guest memory itself; the GPU thread's own reads wait for it.
		NoteDirectGpuWrite(vaddr, size);
		if (m_gpu_write_observer) m_gpu_write_observer(vaddr, size);
		return;
	}
	{
		const DispatchSubTimer dirty_timer(DispatchSub::OwDirtyAdd); // KYTY_TRACE_DISPATCH_CENSUS
		GpuDirtyAdd(vaddr, size);
	}
	std::optional<DispatchSubTimer> observer_timer(std::in_place, DispatchSub::OwObserver); // KYTY_TRACE_DISPATCH_CENSUS
	{ // The per-range write ticks (DropDeadRingBytes, the readback queue, KYTY_TRACE_SYNC_TIMELINE).
		NoteWriteTick(vaddr, vaddr + size, m_scheduler.CurrentTick());
	}
	if (m_gpu_write_observer) m_gpu_write_observer(vaddr, size);
	BumpHotGeneration(vaddr, size);
	observer_timer.reset();
	{
		const DispatchSubTimer tick_timer(DispatchSub::OwWriteTick); // KYTY_TRACE_DISPATCH_CENSUS
		NoteGpuWriteTick(vaddr, size);
	}
	m_bar_pending_writes.emplace_back(vaddr, size);
}

void BufferCache::NoteObtainedWriteRepeat(uint64_t vaddr, uint64_t size, bool same_tick) {
	if (!m_append_pools.empty() || !m_append_windows.empty()) AppendForeignWrite(vaddr, size); // D4b step 1 (append windows)
	// As NoteObtainedWrite, without what the noted range already holds: the GPU-dirty set's range (the generation still
	// moves, as for every GPU write) and, at the same tick, the per-range write ticks and the granule write ticks.
	if (UploadClobberCheck()) C0Note(vaddr, size, C0Event::ShaderWrite, DebugWriterHash().load(std::memory_order_relaxed));
	if (!same_tick) NoteWriteTick(vaddr, vaddr + size, m_scheduler.CurrentTick());
	if (m_gpu_write_observer) m_gpu_write_observer(vaddr, size);
	BumpHotGeneration(vaddr, size);
	if (!same_tick) NoteGpuWriteTick(vaddr, size);
	m_bar_pending_writes.emplace_back(vaddr, size); // the recording command buffer's tick is final
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBdaRange(uint64_t vaddr, uint64_t size, bool is_written,
                                                         BufferId id) {
	if (m_scheduler.Current().IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: BDA range request requires a recording command buffer\n");
	}
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, false, nullptr, UploadCaller::BdaRange);
	if (is_written) {
		NoteObtainedWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::CachedBufferFor(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) return {nullptr, 0};
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) return {nullptr, 0};
	auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) return {nullptr, 0};
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false, nullptr, UploadCaller::Image);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false, {}, true);
	}

	const auto ring          = Staging().Map(size, 16);
	uint8_t*   staging       = ring.first;
	uint64_t   stage_offset  = ring.second;
	// Upstream 9bcf80bf6: an image larger than the staging ring's free room is staged through a one-time upload
	// buffer (as UploadCopies does) instead of stopping the emulator.
	std::unique_ptr<Buffer> temporary;
	if (staging == nullptr) {
		temporary    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
		                                        vk::BufferUsageFlagBits::eTransferSrc |
		                                            vk::BufferUsageFlagBits::eStorageBuffer,
		                                        size);
		staging      = temporary->Mapped().data();
		stage_offset = 0;
	}
	if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size)) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	if (temporary != nullptr) {
		temporary->Flush(0, size);
		auto* buffer = temporary.get();
		m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
		return {buffer, 0};
	}
	Staging().Commit();
	return {&Staging(), stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMetaWrite(vaddr, size); // a larger fill is an ordinary write
	if (!IsRegionGpuModified(vaddr, size) && !InDirectRange(vaddr, size)) { // KYTY_PERF_DIRECT_RANGES: in-stream fill
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	if (UploadClobberCheck()) C0Note(vaddr, size, C0Event::Fill, value); // D4 step K0
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size) &&
	    !InDirectRange(dst_vaddr, size) && !InDirectRange(src_vaddr, size)) { // KYTY_PERF_DIRECT_RANGES: in-stream copy
		LibKernel::Memory::ReadGuestPlain(reinterpret_cast<void*>(dst_vaddr), src_vaddr, size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	if (UploadClobberCheck() && dst_memory) C0Note(dst_vaddr, size, C0Event::Copy, src_vaddr); // D4 step K0
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size) {
	std::shared_lock lock(m_gpu_dirty_mutex);
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

void BufferCache::GpuDirtyAdd(uint64_t vaddr, uint64_t size) {
	{
		std::unique_lock lock(m_gpu_dirty_mutex);
		m_gpu_modified_ranges.Add(vaddr, size);
	}
}

void BufferCache::GpuDirtySubtract(uint64_t vaddr, uint64_t size) {
	NoteGpuCleanRange(vaddr, size); // the write memo
	std::unique_lock lock(m_gpu_dirty_mutex);
	m_gpu_modified_ranges.Subtract(vaddr, size);
}

bool BufferCache::HasPendingReadback(uint64_t vaddr, uint64_t size) const {
	return std::ranges::any_of(m_eager_readbacks, [&](const EagerReadback& entry) {
		return vaddr < entry.end && entry.start < vaddr + size;
	});
}


bool BufferCache::WaitPendingReadbacks(uint64_t vaddr, uint64_t size) {
	// KYTY_PERF_DIRECT_RANGES: resource-table reads of guest memory (last: nothing below may reset the overlay).
	const auto direct = [&] { return WaitDirectGpuWrites(vaddr, size, true); };
	if (m_eager_readbacks.empty()) {
		return direct();
	}
	const bool pending = std::ranges::any_of(m_eager_readbacks, [&](const EagerReadback& entry) {
		return vaddr < entry.end && entry.start < vaddr + size;
	});
	if (!pending) {
		return direct();
	}
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 16u) {
		std::fprintf(stderr, "EAGER_PENDING_READ addr=0x%llx size=0x%llx (shader-data read waits for the readback)\n",
		             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
	}
	ApplyEagerReadbacks(vaddr, size, true);
	ReadbackBurn(); // KYTY_DIAG_GPU_BURN_AT=readback (measurement only)
	return direct();
}

uint32_t BufferCache::D6C0Mode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	return mode_once.Get([] {
		const char*    value = std::getenv("KYTY_D6_C0");
		// Default =1 (user session: walls, fences, barrels no longer vanish; LOD clobbers 0); =0 off.
		const uint32_t mode  = value == nullptr ? 1u
		                      : value[0] >= '1' && value[0] <= '3' && value[1] == '\0' ? static_cast<uint32_t>(value[0] - '0')
		                                                                                : 0u;
		if (mode != 0u) {
			std::fprintf(stderr, "D6_C0 mode=%u (%s%s)\n", mode, mode == 3u ? "path without C0" : "C0",
			             mode >= 2u ? ", upload clobber check" : "");
		}
		return mode;
	});
}

bool BufferCache::RecordKeyedVerifyEnabled() {
	static const bool enabled = [] {
		// Read by the SPIR-V emitter through BdaStoreReportCompiled: Common::ShaderEnv names (the cache fingerprint).
		const bool on = Common::EnvRead::RecordKeyedVerify();
		if (on) std::fprintf(stderr, "RECORD_KEYED_VERIFY enabled\n");
		return on;
	}();
	return enabled;
}

void BufferCache::NoteRecordKeyedWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0u) return;
	auto& pages = m_rk_listed_pages[m_rk_frame % m_rk_listed_pages.size()];
	for (uint64_t page = vaddr >> CACHING_PAGEBITS; page <= (vaddr + size - 1u) >> CACHING_PAGEBITS; ++page) {
		pages.insert(page);
	}
}

bool BufferCache::BdaStoreListed(uint64_t page_address, uint64_t first_frame, uint64_t last_frame) const {
	// Only the frames still kept (as VerifyBdaWrittenPage), and none with an incomplete listing: a call whose listing is
	// incomplete stores through ranges nobody listed, so every report of those frames is taken as unlisted.
	if (last_frame < first_frame || m_rk_frame - first_frame >= m_rk_listed_pages.size()) return false;
	if (m_rk_incomplete_frame != 0u && m_rk_incomplete_frame - 1u >= first_frame) return false;
	const auto page = page_address >> CACHING_PAGEBITS;
	for (uint64_t frame = first_frame; frame <= last_frame; ++frame) {
		if (m_rk_listed_pages[frame % m_rk_listed_pages.size()].contains(page)) return true;
	}
	return false;
}

void BufferCache::RecordHotRead(uint64_t vaddr) {
	constexpr uint64_t Window = 64u * 1024u;
	const uint64_t     start  = vaddr & ~(Window - 1u);
	for (auto& hot: m_hot_reads) {
		if (hot.start == start) {
			hot.hits++;
			hot.last_use = m_hot_clock;
			return;
		}
	}
	constexpr size_t MaxHot = 64;
	if (m_hot_reads.size() >= MaxHot) {
		auto oldest = std::ranges::min_element(m_hot_reads, {}, &HotRead::last_use);
		*oldest = {};
		oldest->start = start;
		oldest->end = start + Window;
		oldest->hits = 1;
		oldest->last_use = m_hot_clock;
		return;
	}
	m_hot_reads.push_back({.start = start, .end = start + Window, .hits = 1, .last_use = m_hot_clock});
}

void BufferCache::BumpHotGeneration(uint64_t vaddr, uint64_t size) {
	for (auto& hot: m_hot_reads) {
		if (vaddr < hot.end && hot.start < vaddr + size) hot.generation++;
	}
}

void BufferCache::ApplyEagerReadbacks(uint64_t vaddr, uint64_t size, bool wait) {
	for (size_t i = 0; i < m_eager_readbacks.size();) {
		auto& entry = m_eager_readbacks[i];
		if (size != 0 && !(vaddr < entry.end && entry.start < vaddr + size)) {
			++i;
			continue;
		}
		if (!entry.done->load(std::memory_order_acquire)) {
			if (!wait) {
				++i;
				continue;
			}
			const bool              open       = entry.tick >= m_scheduler.CurrentTick();
			const auto              wait_start = std::chrono::steady_clock::now();
			SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Eager); // KYTY_TRACE_SYNC_TIMELINE
			m_scheduler.Wait(entry.tick);
			m_scheduler.WaitPriorityOperations(entry.tick);
			EXIT_IF(!entry.done->load(std::memory_order_acquire));
			if (DownloadWaitTrace::Enabled()) {
				DownloadWaitTrace::Add(open ? DownloadWaitTrace::EagerOpen : DownloadWaitTrace::EagerOld,
				                       static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
				                                                 std::chrono::steady_clock::now() - wait_start).count()),
				                       entry.end - entry.start);
			}
		}
		// The copy published every byte it removed from m_gpu_modified_ranges. A tracker page
		// with no remaining (newer) GPU bytes now holds current data in guest memory; a page that
		// still has GPU bytes stays modified, keeping "GPU-modified page <=> bytes to download".
		for (uint64_t page = entry.start; page < entry.end; page += TRACKER_PAGE_SIZE) {
			if (!m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE) &&
			    m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
				m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
			}
		}
		entry = std::move(m_eager_readbacks.back());
		m_eager_readbacks.pop_back();
	}
}

void BufferCache::IssueEagerReadbacks() {
	ApplyEagerReadbacks(0, 0, false);
	++m_hot_clock;
	for (const auto& hot: m_hot_reads) {
		if (hot.hits < 2u || m_hot_clock - hot.last_use > 256u) continue;
		if (!m_gpu_modified_ranges.Intersects(hot.start, hot.end - hot.start)) continue;
		if (std::ranges::any_of(m_eager_readbacks, [&](const EagerReadback& entry) {
			    return entry.start == hot.start;
		    })) continue;
		bool any = false;
		std::vector<std::pair<uint64_t, uint64_t>> bytes;
		m_gpu_modified_ranges.ForEachInRange(hot.start, hot.end - hot.start,
		                                     [&](uint64_t start, uint64_t end) { bytes.emplace_back(start, end); });
		auto it  = m_buffers.upper_bound(hot.start);
		if (it != m_buffers.begin()) --it;
		for (; it != m_buffers.end() && it->first < hot.end; ++it) {
			auto&      buffer = m_slot_buffers[it->second];
			const auto start  = std::max(buffer.CpuAddress(), hot.start);
			const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), hot.end);
			if (start < finish && DownloadBufferMemory<true>(buffer, start, finish - start)) any = true;
		}
		if (!any) continue;
		auto done = std::make_shared<std::atomic_bool>(false);
		m_scheduler.DeferPriorityOperation([done] { done->store(true, std::memory_order_release); });
		m_eager_readbacks.push_back({.start = hot.start, .end = hot.end, .generation = hot.generation,
		                             .tick = m_scheduler.CurrentTick(), .done = std::move(done),
		                             .bytes = std::move(bytes)});
	}
}

// Three skips in the bookkeeping of obtained buffers, each leaving every answer as without it:
//  - write ticks (m_write_ticks, NoteWriteTick): an insert that an entry already holds at the same tick is skipped and
//    equal-tick neighbours are joined. QueryWriteTick reads coverage and the newest tick, which joined entries keep;
//    every insert uses CurrentTick(), so the per-16 KiB BDA marks of one pass collapse into one entry and a later large
//    write erases one entry instead of many.
//  - granule write ticks (NoteGpuWriteTick): the values only rise and are never erased, so a range inside one noted
//    at the same tick (the obtain, then NoteCommandEmitted for the same draw; consecutive dispatches writing one UAV)
//    needs no loop.
//  - image search of a buffer obtain (TextureCache::FindImageFromRange): only images starting at the address are kept,
//    and an image is listed in the page of its first byte, so the first page alone finds them in the same order (as
//    the first-page lookup does for InvalidateMemoryFromGPU).
namespace {
struct ObtainExactStats {
	std::atomic<uint64_t> granule_checks {0};
	std::atomic<uint64_t> granule_skips {0};
};
ObtainExactStats g_obtain_exact;

void ReportObtainExact(uint64_t count) {
	if ((count & 0xfffffu) != 0 || !Common::EnvRead::PerfLines()) return;
	std::fprintf(stderr, "PERF obtain_exact granule_checks=%llu granule_skips=%llu\n",
	             static_cast<unsigned long long>(g_obtain_exact.granule_checks.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(g_obtain_exact.granule_skips.load(std::memory_order_relaxed)));
}
} // namespace

// Only DropDeadRingBytes reads the per-range write ticks by default, and it asks only
// about GPU-dirty bytes of pages inside the data ring 0x2002600000-0x2010000000. QueryTicks looks only at entries
// overlapping the queried range, so ranges that do not touch the ring never change its answers; they are left out while
// no other reader of the ticks is on (sync timeline, SRT exact reads).
void BufferCache::NoteWriteTick(uint64_t start, uint64_t end, uint64_t tick) {
	if (!SyncTimeline::Enabled() && (end <= 0x2002600000ull || start >= 0x2010000000ull)) return;
	InsertTickRangeJoined(m_write_ticks, start, end, tick);
}

void BufferCache::InsertTickRangeJoined(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end, uint64_t tick) {
	if (start >= end) return;
	// An entry already holding [start, end) at this tick: the insert would leave the same answers.
	auto at = ticks.upper_bound(start);
	if (at != ticks.begin()) {
		const auto holder = std::prev(at);
		if (holder->second.end >= end && holder->second.tick == tick) return;
	}
	InsertTickRange(ticks, start, end, tick);
	auto self = ticks.find(start);
	if (self != ticks.begin()) {
		const auto left = std::prev(self);
		if (left->second.end == start && left->second.tick == tick) {
			left->second.end = self->second.end;
			ticks.erase(self);
			self = left;
		}
	}
	const auto right = std::next(self);
	if (right != ticks.end() && right->first == self->second.end && right->second.tick == tick) {
		self->second.end = right->second.end;
		ticks.erase(right);
	}
}

// [start, end) gets `tick`; the parts of older ranges outside it keep theirs.
void BufferCache::InsertTickRange(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end, uint64_t tick) {
	auto& m_write_ticks = ticks;
	if (start >= end) return;
	auto it = m_write_ticks.lower_bound(start);
	if (it != m_write_ticks.begin()) {
		auto previous = std::prev(it);
		if (previous->second.end > start) {
			const auto old = previous->second;
			previous->second.end = start;
			if (old.end > end) m_write_ticks[end] = old;
		}
	}
	it = m_write_ticks.lower_bound(start);
	while (it != m_write_ticks.end() && it->first < end) {
		if (it->second.end > end) {
			const auto old = it->second;
			m_write_ticks.erase(it);
			m_write_ticks[end] = old;
			break;
		}
		it = m_write_ticks.erase(it);
	}
	m_write_ticks[start] = {end, tick};
}

uint64_t BufferCache::QueryTicks(const std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end) {
	uint64_t covered = start;
	uint64_t newest  = 0;
	auto     it      = ticks.upper_bound(start);
	if (it != ticks.begin() && std::prev(it)->second.end > start) --it;
	for (; it != ticks.end() && it->first < end && covered < end; ++it) {
		if (it->first > covered) return 0;
		newest  = std::max(newest, it->second.tick);
		covered = std::max(covered, it->second.end);
	}
	return covered >= end ? newest : 0;
}

uint64_t BufferCache::QueryWriteTick(uint64_t start, uint64_t end) const {
	return QueryTicks(m_write_ticks, start, end);
}

// Write ticks are kept per 64 KiB granule (WriteTickKey): a granule's tick is the newest write to
// any of its bytes, a safe upper bound for each page in it. Per 4 KiB page the map updates cost
// ~9% of the GPU thread (the BAR reads note every written binding at each draw/dispatch).

void BufferCache::NoteGpuWriteTick(uint64_t vaddr, uint64_t size) {
	const auto tick = m_scheduler.CurrentTick();
	// A range inside one already raised to this tick needs no new loop.
	bool noted = false;
	for (const auto& note: m_granule_notes) {
		if (note.tick == tick && note.start <= vaddr && vaddr + size <= note.end) {
			noted = true;
			break;
		}
	}
	ReportObtainExact(g_obtain_exact.granule_checks.fetch_add(1, std::memory_order_relaxed) + 1);
	if (noted) {
		g_obtain_exact.granule_skips.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	m_granule_ticks.RaiseRange(WriteTickKey(vaddr), WriteTickKey(vaddr + size - 1), tick);
	m_granule_notes[m_granule_note_next] = {vaddr, vaddr + size, tick};
	m_granule_note_next = (m_granule_note_next + 1u) % static_cast<uint32_t>(m_granule_notes.size());
}

std::optional<uint64_t> BufferCache::GranuleWriteTick(uint64_t key) const {
	if (const uint64_t tick = m_granule_ticks.Get(key); tick != 0u) return tick;
	return std::nullopt;
}

void BufferCache::NoteCommandEmitted() {
	// A buffer obtained for writing can be recorded into a later command buffer than the one open
	// when it was obtained (a flush in between); the recording command buffer's tick is final.
	for (const auto& [vaddr, size]: m_bar_pending_writes) {
		NoteGpuWriteTick(vaddr, size);
		// KYTY_TRACE_SYNC_TIMELINE: the final tick for its per-range write ticks too.
		if (SyncTimeline::Enabled()) {
			NoteWriteTick(vaddr, vaddr + size, m_scheduler.CurrentTick());
		}
	}
	m_bar_pending_writes.clear();
}

namespace {

// KYTY_TRACE_PERF: bytes and time of the BAR copies (all threads / the GPU thread), PERF bar_copy.
std::atomic<uint64_t> g_bar_copy_calls {0};
std::atomic<uint64_t> g_bar_copy_bytes {0};
std::atomic<uint64_t> g_bar_copy_ns {0};
std::atomic<uint64_t> g_bar_copy_gpu_bytes {0};
std::atomic<uint64_t> g_bar_copy_gpu_ns {0};

// KYTY_TRACE_PERF: the BAR copies' bytes and time by size (<4 KiB, <64 KiB, <1 MiB, >=1 MiB), PERF bar_copy_sizes.
std::array<std::atomic<uint64_t>, 4> g_bar_size_bytes {};
std::array<std::atomic<uint64_t>, 4> g_bar_size_ns {};
std::array<std::atomic<uint64_t>, 4> g_bar_size_calls {};

void CopyFromWriteCombinedImpl(uint8_t* dst, const uint8_t* src, uint64_t size);

// A read of GPU memory of 256 KiB or more is split into 64 KiB-aligned parts copied by the caller and BarCopyThreads
// helper threads at once (one core keeps only a few PCIe reads in flight). The same bytes go to the same places; the
// caller returns when every part is copied. The helpers spin ~50 us after a copy, then sleep.
constexpr uint32_t BarCopyThreads = 2;

struct BarCopyPool {
	struct Part {
		uint8_t*       dst  = nullptr;
		const uint8_t* src  = nullptr;
		uint64_t       size = 0;
	};
	std::array<Part, BarCopyThreads + 1u> parts {};
	std::atomic<uint32_t> generation {0};
	std::atomic<uint32_t> remaining {0};
	uint32_t              helpers = 0; // the helpers of the current copy (parts 1..helpers)
	std::vector<std::thread> threads;
	std::mutex            mutex; // one copy at a time (the GPU thread and others may both read)

	BarCopyPool() {
		for (uint32_t i = 0; i < BarCopyThreads; i++) threads.emplace_back([this, i] { Main(i + 1u); });
	}
	void Main(uint32_t index) {
		Common::Thread::SetCurrentName(("BarCopy" + std::to_string(index)).c_str());
		uint32_t seen = 0;
		for (;;) {
			uint32_t g    = generation.load(std::memory_order_acquire);
			const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(50);
			while (g == seen) {
				_mm_pause();
				if (std::chrono::steady_clock::now() >= until) {
					generation.wait(seen, std::memory_order_acquire);
				}
				g = generation.load(std::memory_order_acquire);
			}
			seen = g;
			if (index <= helpers) {
				const auto& p = parts[index];
				CopyFromWriteCombinedImpl(p.dst, p.src, p.size);
				remaining.fetch_sub(1, std::memory_order_acq_rel);
			}
		}
	}
	void Copy(uint8_t* dst, const uint8_t* src, uint64_t size) {
		std::scoped_lock lock(mutex);
		const uint32_t   count = BarCopyThreads;
		// Parts of whole 64 KiB blocks; the caller's part takes the rest.
		const uint64_t block = 65536;
		const uint64_t per   = (size / (count + 1u)) / block * block;
		uint64_t       offset = 0;
		for (uint32_t i = 1; i <= count; i++) {
			parts[i] = {dst + offset, src + offset, per};
			offset += per;
		}
		parts[0] = {dst + offset, src + offset, size - offset};
		helpers  = count;
		remaining.store(count, std::memory_order_release);
		generation.fetch_add(1, std::memory_order_acq_rel);
		generation.notify_all();
		CopyFromWriteCombinedImpl(parts[0].dst, parts[0].src, parts[0].size);
		while (remaining.load(std::memory_order_acquire) != 0) _mm_pause();
	}
};

BarCopyPool& GetBarCopyPool() {
	static auto* pool = new BarCopyPool; // lives for the process
	return *pool;
}

void CopyFromWriteCombined(uint8_t* dst, const uint8_t* src, uint64_t size) {
	const auto start = std::chrono::steady_clock::now();
	if (size >= 262144u) {
		GetBarCopyPool().Copy(dst, src, size);
	} else {
		CopyFromWriteCombinedImpl(dst, src, size);
	}
	const auto ns = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
	g_bar_copy_calls.fetch_add(1, std::memory_order_relaxed);
	g_bar_copy_bytes.fetch_add(size, std::memory_order_relaxed);
	g_bar_copy_ns.fetch_add(ns, std::memory_order_relaxed);
	const size_t bucket = size < 4096u ? 0u : size < 65536u ? 1u : size < 1048576u ? 2u : 3u;
	g_bar_size_calls[bucket].fetch_add(1, std::memory_order_relaxed);
	g_bar_size_bytes[bucket].fetch_add(size, std::memory_order_relaxed);
	g_bar_size_ns[bucket].fetch_add(ns, std::memory_order_relaxed);
	if (GuestGpu::IsGpuThread()) {
		g_bar_copy_gpu_bytes.fetch_add(size, std::memory_order_relaxed);
		g_bar_copy_gpu_ns.fetch_add(ns, std::memory_order_relaxed);
	}
}

void CopyFromWriteCombinedImpl(uint8_t* dst, const uint8_t* src, uint64_t size) {
	// Streaming-load buffers are not coherent: drop any line filled by an earlier copy.
	_mm_mfence();
	const auto head = std::min<uint64_t>(size, (32u - (reinterpret_cast<uintptr_t>(src) & 31u)) & 31u);
	if (head != 0) {
		std::memcpy(dst, src, head);
		dst += head;
		src += head;
		size -= head;
	}
	// Four lines per step keep several PCIe reads in flight.
	while (size >= 256) {
		auto* in = reinterpret_cast<__m256i*>(const_cast<uint8_t*>(src));
		const __m256i v0 = _mm256_stream_load_si256(in + 0);
		const __m256i v1 = _mm256_stream_load_si256(in + 1);
		const __m256i v2 = _mm256_stream_load_si256(in + 2);
		const __m256i v3 = _mm256_stream_load_si256(in + 3);
		const __m256i v4 = _mm256_stream_load_si256(in + 4);
		const __m256i v5 = _mm256_stream_load_si256(in + 5);
		const __m256i v6 = _mm256_stream_load_si256(in + 6);
		const __m256i v7 = _mm256_stream_load_si256(in + 7);
		auto* out = reinterpret_cast<__m256i*>(dst);
		_mm256_storeu_si256(out + 0, v0);
		_mm256_storeu_si256(out + 1, v1);
		_mm256_storeu_si256(out + 2, v2);
		_mm256_storeu_si256(out + 3, v3);
		_mm256_storeu_si256(out + 4, v4);
		_mm256_storeu_si256(out + 5, v5);
		_mm256_storeu_si256(out + 6, v6);
		_mm256_storeu_si256(out + 7, v7);
		src += 256;
		dst += 256;
		size -= 256;
	}
	while (size >= 32) {
		_mm256_storeu_si256(reinterpret_cast<__m256i*>(dst),
		                    _mm256_stream_load_si256(reinterpret_cast<__m256i*>(const_cast<uint8_t*>(src))));
		src += 32;
		dst += 32;
		size -= 32;
	}
	if (size != 0) std::memcpy(dst, src, size);
}

} // namespace

void BufferCache::ReportBarCopyPerf() {
	const auto calls   = g_bar_copy_calls.exchange(0, std::memory_order_relaxed);
	const auto bytes   = g_bar_copy_bytes.exchange(0, std::memory_order_relaxed);
	const auto ns      = g_bar_copy_ns.exchange(0, std::memory_order_relaxed);
	const auto g_bytes = g_bar_copy_gpu_bytes.exchange(0, std::memory_order_relaxed);
	const auto g_ns    = g_bar_copy_gpu_ns.exchange(0, std::memory_order_relaxed);
	std::fprintf(stderr, "PERF bar_copy calls=%llu MiB=%llu ms=%llu gpu_thread_MiB=%llu gpu_thread_ms=%llu window_kb=%llu\n",
	             static_cast<unsigned long long>(calls), static_cast<unsigned long long>(bytes >> 20u),
	             static_cast<unsigned long long>(ns / 1000000u), static_cast<unsigned long long>(g_bytes >> 20u),
	             static_cast<unsigned long long>(g_ns / 1000000u), static_cast<unsigned long long>(ReadWindowBytes >> 10u));
	std::fprintf(stderr, "PERF bar_copy_sizes (calls/KiB/ms <4K <64K <1M >=1M)=");
	for (size_t i = 0; i < 4; i++) {
		std::fprintf(stderr, "%llu/%llu/%llu ", static_cast<unsigned long long>(g_bar_size_calls[i].exchange(0)),
		             static_cast<unsigned long long>(g_bar_size_bytes[i].exchange(0) >> 10u),
		             static_cast<unsigned long long>(g_bar_size_ns[i].exchange(0) / 1000000u));
	}
	std::fprintf(stderr, "threads=%u\n", BarCopyThreads);
}

int BufferCache::ScanBarDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                 std::vector<std::pair<uint64_t, uint64_t>>& intervals, uint64_t& newest,
                                 uint64_t& total) {
	const auto base   = buffer.CpuAddress();
	int        reason = -1;
	intervals.clear();
	newest = 0;
	total  = 0;
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			if (reason >= 0) return;
			if (start < base || end > base + buffer.Size()) {
				reason = 5;
				return;
			}
			for (uint64_t page = start & ~uint64_t {TRACKER_PAGE_SIZE - 1}; page < end; page += TRACKER_PAGE_SIZE) {
				const auto it = GranuleWriteTick(WriteTickKey(page));
				if (!it) {
					reason = 1;
					return;
				}
				newest = std::max(newest, *it);
			}
			intervals.emplace_back(start, end);
			total += end - start;
		});
	});
	if (reason >= 0) return reason;
	return intervals.empty() ? 6 : -1;
}

// A host-mapped download (TryBarDownload) reads its GPU-dirty intervals from the BAR mapping in
// spans: the next interval joins the span when the gap to it is at most BarBatchGap bytes. Every streamed read of the
// uncached mapping costs ~1.7 us before its first byte (fence, PCIe round trips) and ~0.54 us per KiB; a page with three
// dirty intervals paid the start three times. Only the intervals are published, from the same finished bytes (every
// interval's write tick was waited for; the gap bytes, read from the same buffer, are dropped), in the same order.
// KYTY_TRACE_PERF: the PERF bar_batch line every ~5 s.
namespace {
constexpr uint64_t BarBatchGap = 2048;

struct BarBatchStats {
	std::atomic<uint64_t> downloads {0};  // host-mapped downloads read in spans
	std::atomic<uint64_t> intervals {0};  // ... their GPU-dirty intervals (the reads before)
	std::atomic<uint64_t> spans {0};      // ... their span reads
	std::atomic<uint64_t> bytes {0};      // ... interval bytes
	std::atomic<uint64_t> extra {0};      // ... gap bytes read and dropped
	std::atomic<int64_t>  window_start {0};
};
BarBatchStats g_bar_batch;

void ReportBarBatch() {
	if (!PerfEnabled()) return;
	auto&         s   = g_bar_batch;
	const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	int64_t start = s.window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		s.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed);
		return;
	}
	if (now - start < 5000 || !s.window_start.compare_exchange_strong(start, now, std::memory_order_relaxed)) return;
	const auto take = [](std::atomic<uint64_t>& counter) {
		return static_cast<unsigned long long>(counter.exchange(0, std::memory_order_relaxed));
	};
	const auto intervals = take(s.intervals);
	const auto spans     = take(s.spans);
	std::fprintf(stderr,
	             "PERF bar_batch gap=%llu downloads=%llu intervals=%llu spans=%llu calls_saved=%llu bytes_kb=%llu "
	             "extra_kb=%llu (per ~5 s)\n",
	             static_cast<unsigned long long>(BarBatchGap), take(s.downloads), intervals, spans,
	             intervals > spans ? intervals - spans : 0ull, take(s.bytes) >> 10u, take(s.extra) >> 10u);
}
} // namespace

// The publish of TryBarDownload's intervals (finished, already taken out of the GPU-dirty set) from span reads of the
// host mapping `mapped` of the buffer at guest address `base`.
void BufferCache::BarBatchRead(const uint8_t* mapped, uint64_t base,
                               const std::vector<std::pair<uint64_t, uint64_t>>& intervals) {
	auto& s = g_bar_batch;
	// Spans over the intervals in their order (ScanBarDownload walks them by address; an interval below the open span
	// starts a new one). staging: the span's offset in the staging copy.
	struct Span {
		uint64_t start = 0, end = 0, staging = 0;
	};
	static thread_local std::vector<Span>     spans;
	static thread_local std::vector<uint32_t> owner; // the span of each interval
	static thread_local std::vector<uint8_t>  staging;
	spans.clear();
	owner.clear();
	const uint64_t gap   = BarBatchGap;
	uint64_t       total = 0, bytes = 0;
	for (const auto& [start, end]: intervals) {
		bytes += end - start;
		if (!spans.empty() && start >= spans.back().end && start - spans.back().end <= gap) {
			total += end - spans.back().end;
			spans.back().end = end;
		} else {
			spans.push_back({start, end, total});
			total += end - start;
		}
		owner.push_back(static_cast<uint32_t>(spans.size() - 1u));
	}
	if (staging.size() < total) staging.resize(total);
	{
		KYTY_PROFILER_BLOCK("Download::BarRead");
		for (const auto& span: spans) {
			CopyFromWriteCombined(staging.data() + span.staging, mapped + (span.start - base), span.end - span.start);
		}
	}
	s.downloads.fetch_add(1, std::memory_order_relaxed);
	s.intervals.fetch_add(intervals.size(), std::memory_order_relaxed);
	s.spans.fetch_add(spans.size(), std::memory_order_relaxed);
	s.bytes.fetch_add(bytes, std::memory_order_relaxed);
	s.extra.fetch_add(total - bytes, std::memory_order_relaxed);
	for (size_t i = 0; i < intervals.size(); i++) {
		const auto [start, end] = intervals[i];
		const auto&    span     = spans[owner[i]];
		const uint8_t* data     = staging.data() + span.staging + (start - span.start);
		SyncTimeline::NoteDownload(start, data, end - start);
		Libs::LibKernel::Memory::WriteBacking(start, data, end - start);
	}
	ReportBarBatch();
}

// The read window, unless the host mapping cannot serve it at once (more than BarMaxBytes GPU-dirty, or written by work
// not finished) while the pages of the access can: then those pages, so the read waits for their writes only and not
// for the whole queue (frame timeline 2.11: a 4-byte SRT read at the start of a frame took a 1.36 MB window and waited
// 6-9.5 ms for 5 command buffers, its own bytes long written; live A/B +1.8% standing, +7.7% walking, draws equal). The
// bytes left outside stay GPU-modified and synchronize when accessed.
std::pair<uint64_t, uint64_t> BufferCache::ReadRange(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const uint64_t start = buffer.CpuAddress();
	const uint64_t end   = start + buffer.Size();
	const uint64_t lo    = std::max(start, vaddr > ReadWindowBytes ? (vaddr - ReadWindowBytes) & ~0xfffull : 0ull);
	const uint64_t hi    = std::min(end, (vaddr + size + ReadWindowBytes + 0xfffull) & ~0xfffull);
	if (!(lo < hi && lo <= vaddr && vaddr + size <= hi)) return {vaddr, vaddr + size};
	if (buffer.Mapped().empty()) return {lo, hi};
	const auto ready = m_buffer_ready_tick.find(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.Handle())));
	if (ready == m_buffer_ready_tick.end()) return {lo, hi};
	const auto servable = [&](uint64_t a, uint64_t b) {
		std::vector<std::pair<uint64_t, uint64_t>> intervals;
		uint64_t                                   newest = 0, total = 0;
		if (ScanBarDownload(buffer, a, b - a, intervals, newest, total) >= 0) return false;
		return total <= BarMaxBytes && std::max(newest, ready->second) < m_scheduler.CurrentTick();
	};
	if (servable(lo, hi)) return {lo, hi};
	const uint64_t p_lo = std::max(start, vaddr & ~0xfffull);
	const uint64_t p_hi = std::min(end, (vaddr + size + 0xfffull) & ~0xfffull);
	return servable(p_lo, p_hi) ? std::pair {p_lo, p_hi} : std::pair {lo, hi};
}

int BufferCache::TryBarDownload(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// Larger reads copy faster through the GPU into cached host memory than from the BAR mapping.
	const uint64_t MaxBytes = BarMaxBytes;
	// Outcomes: 0 read from the mapping, 1 unknown write tick, 2 written by the open command buffer,
	// 3 write not recorded yet, 4 too large, 5 outside the buffer, 6 nothing GPU-written.
	static std::array<uint64_t, 7> outcomes {};
	static uint64_t                calls = 0;
	const auto                     outcome = [&](size_t index) {
        outcomes[index]++;
        DownloadWaitTrace::g_last_bar_reason = static_cast<int>(index);
        if ((++calls & 0xfffu) == 0 && Common::EnvRead::PerfLines()) {
            std::fprintf(stderr,
                         "BAR_READS calls=%llu mapped=%llu no_tick=%llu open=%llu pending=%llu large=%llu outside=%llu "
                         "empty=%llu\n",
                         static_cast<unsigned long long>(calls), static_cast<unsigned long long>(outcomes[0]),
                         static_cast<unsigned long long>(outcomes[1]), static_cast<unsigned long long>(outcomes[2]),
                         static_cast<unsigned long long>(outcomes[3]), static_cast<unsigned long long>(outcomes[4]),
                         static_cast<unsigned long long>(outcomes[5]), static_cast<unsigned long long>(outcomes[6]));
        }
        return index == 0 ? 1 : -1;
	};
	const auto base   = buffer.CpuAddress();
	uint64_t   newest = 0;
	uint64_t   total  = 0;
	std::vector<std::pair<uint64_t, uint64_t>> intervals;
	KYTY_PROFILER_BLOCK("Download::BarScan");
	if (const int reason = ScanBarDownload(buffer, vaddr, size, intervals, newest, total); reason >= 0) {
		return outcome(static_cast<size_t>(reason));
	}
	// A buffer created by joining older ones (CreateBuffer ->
	// JoinOverlap) receives their bytes, GPU-written ones included, through copies recorded at its creation
	// tick, while the granules keep the older write ticks. Its mapping holds those bytes only once that tick
	// finished; until then the read takes the GPU copy path, recorded after the join copies.
	const auto ready = m_buffer_ready_tick.find(reinterpret_cast<uint64_t>(static_cast<VkBuffer>(buffer.Handle())));
	if (ready == m_buffer_ready_tick.end()) return outcome(1);
	newest = std::max(newest, ready->second);
	DownloadWaitTrace::g_last_bar_old = newest < m_scheduler.CurrentTick();
	if (total > MaxBytes) return outcome(4);
	if (newest >= m_scheduler.CurrentTick()) return outcome(2);
	for (const auto& [start, end]: intervals) {
		if (std::ranges::any_of(m_bar_pending_writes,
		                        [&](const auto& w) { return w.first < end && start < w.first + w.second; })) {
			return outcome(3);
		}
	}
	// Every command buffer ends with a device-to-host memory barrier (CommandScheduler::Submit), so
	// the writes are visible to the host once their command buffer finished.
	SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Bar); // KYTY_TRACE_SYNC_TIMELINE
	m_scheduler.Wait(newest);
	m_scheduler.WaitPriorityOperations(newest);
	KYTY_PROFILER_BLOCK("Download::BarCopy");
	if (!buffer.IsCoherent()) buffer.Invalidate(0, buffer.Size());
	const auto* mapped = buffer.Mapped().data();
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes, "BAR download");
		GpuDirtySubtract(address, bytes);
	});
	if (UploadClobberCheck()) { // D4 step K0: the ticks this read relied on
		const uint64_t tag = (newest & 0xffffffffu) | (m_scheduler.GetMasterSemaphore().KnownGpuTick() << 32u);
		for (const auto& [start, end]: intervals) C0Note(start, end - start, C0Event::BarRead, tag);
	}
	// The BAR mapping is uncached (write-combined) PCIe memory, so an ordinary memcpy issues one uncached read per load.
	// Streaming loads (VMOVNTDQA) fetch a whole 64-byte line per transaction and several lines can be in flight. The bytes
	// go to a host staging copy first, so the slow PCIe reads do not run under the address-space mutex (WriteBacking) and
	// the merge copies do not read the BAR a second time.
	BarBatchRead(mapped, base, intervals);
	return outcome(0);
}

void BufferCache::UnmarkPublished(uint64_t start, uint64_t end) {
	// The pages hold no GPU-written bytes any more: "GPU-modified page <=> bytes to download" is kept, and one
	// protection change covers the whole run.
	if (start < end) m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
}

// The fault read window with eager readback: the window download skips bytes an eager copy took out
// of the GPU-dirty set, but the old code then unmarked every page of the window, so the CPU could read such a page
// before the copy published it (older bytes), and a CPU write there was later overwritten by that publish.
// Those pages stay GPU-modified; ApplyEagerReadbacks unmarks them once published.
void BufferCache::UnmarkWindow(uint64_t lo, uint64_t hi) {
	if (lo >= hi) return;
	if (m_eager_readbacks.empty()) {
		m_memory_tracker.UnmarkRegionAsGpuModified(lo, hi - lo);
		return;
	}
	std::vector<std::pair<uint64_t, uint64_t>> keep;
	for (const auto& entry: m_eager_readbacks) {
		if (entry.done->load(std::memory_order_acquire) || entry.end <= lo || hi <= entry.start) continue;
		for (const auto& [start, end]: entry.bytes) {
			const uint64_t a = std::max(lo, start & ~(TRACKER_PAGE_SIZE - 1));
			const uint64_t b = std::min(hi, (end + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1));
			if (a < b) keep.emplace_back(a, b);
		}
	}
	std::ranges::sort(keep);
	uint64_t cursor = lo;
	for (const auto& [a, b]: keep) {
		if (a > cursor) m_memory_tracker.UnmarkRegionAsGpuModified(cursor, a - cursor);
		cursor = std::max(cursor, b);
	}
	if (cursor < hi) m_memory_tracker.UnmarkRegionAsGpuModified(cursor, hi - cursor);
}

bool BufferCache::PendingReadbackCovers(uint64_t vaddr, uint64_t size) const {
	for (const auto& entry: m_eager_readbacks) {
		if (!(vaddr < entry.end && entry.start < vaddr + size) || entry.done->load(std::memory_order_acquire)) continue;
		for (const auto& [start, end]: entry.bytes) {
			if (vaddr < end && start < vaddr + size) return true;
		}
	}
	return false;
}

bool BufferCache::ReadGpuCopy(uint64_t vaddr, uint64_t size, void* out) {
	if (vaddr == 0 || size == 0 || size > 1024u * 1024u) return false;
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) return false;
	auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size) || m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		return false;
	}
	auto [mapped, offset] = m_download_buffer.Map(size, 64);
	if (mapped == nullptr) return false;
	m_download_buffer.Commit();
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {buffer.Offset(vaddr), offset, size};
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0,
	                       nullptr, 1, &after, 0, nullptr);
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	m_download_buffer.Invalidate(offset, size);
	std::memcpy(out, mapped, size);
	return true;
}

bool BufferCache::PeekGpuCopyAsync(uint64_t vaddr, uint64_t size, std::function<void(const uint8_t*)> done) {
	if (vaddr == 0 || size == 0 || size > 256u || !done) return false;
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner) return false;
	auto& buffer = m_slot_buffers[*owner];
	if (buffer.is_deleted || !buffer.IsInBounds(vaddr, size) || m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		return false;
	}
	auto [mapped, offset] = m_download_buffer.Map(size, 64);
	if (mapped == nullptr) return false;
	m_download_buffer.Commit();
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	const vk::BufferCopy copy {buffer.Offset(vaddr), offset, size};
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0,
	                       nullptr, 1, &after, 0, nullptr);
	// As the asynchronous publish of DownloadBufferMemory: the ring slot is read once its tick finished.
	m_scheduler.DeferPriorityOperation([this, mapped, offset, size, done = std::move(done)] {
		m_download_buffer.Invalidate(offset, size);
		done(mapped);
	});
	return true;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	CommitUploadBatch(BatchCommit::Collector); // (no scope is open here; defensive)
	if (!m_direct_ranges.empty()) ReportDirectStats(m_direct_gpu_writes.size()); // KYTY_PERF_DIRECT_RANGES, per frame
	if (D6C0Mode() != 0u && !RecordKeyedVerifyEnabled()) {
		// KYTY_D6_C0: a new frame of listed pages (as below); a summary every 64 frames.
		++m_rk_frame;
		m_rk_listed_pages[m_rk_frame % m_rk_listed_pages.size()].clear();
	}
	if (D6C0Mode() != 0u && m_rk_frame % 64u == 0u && Common::EnvRead::PerfLines()) {
		std::fprintf(stderr,
		             "PERF d6_c0 frame=%llu listed_skipped=%llu unlisted_pages=%llu conflicts=%llu incomplete_calls=%llu "
		             "would_mark=%llu " // D6 step K2
		             "checked=%llu clobber_pages=%llu clobber_bytes=%llu dead_bytes=%llu stale_bytes=%llu no_base_bytes=%llu "
		             "unchecked=%llu snap_dropped=%llu snapshots=%zu\n",
		             static_cast<unsigned long long>(m_rk_frame), static_cast<unsigned long long>(m_c0_listed_skipped),
		             static_cast<unsigned long long>(m_c0_unlisted_pages), static_cast<unsigned long long>(m_c0_conflicts),
		             static_cast<unsigned long long>(m_rk_incomplete), static_cast<unsigned long long>(m_c0_would_mark),
		             static_cast<unsigned long long>(m_c0_checked.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_clobber_pages.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_clobber_bytes.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_dead_bytes.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_stale_bytes.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_no_base_bytes.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_unchecked.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(m_c0_snap_dropped.load(std::memory_order_relaxed)),
		             [&] {
			             std::scoped_lock lock(m_c0_snap_mutex);
			             return m_c0_snapshots.size();
		             }());
		if (UploadClobberCheck()) { // D4 step K0: clobbered pages by the newest event that changed the GPU copy
			const auto n = [&](C0Event kind) {
				return static_cast<unsigned long long>(m_c0_writer_pages[static_cast<size_t>(kind)].load(std::memory_order_relaxed));
			};
			std::fprintf(stderr, "PERF d6_c0_writer shader=%llu direct=%llu imgcopy=%llu fill=%llu copy=%llu none=%llu snap_raced=%llu\n",
			             n(C0Event::ShaderWrite), n(C0Event::DirectWrite), n(C0Event::ImageCopy), n(C0Event::Fill),
			             n(C0Event::Copy), n(C0Event::Count),
			             static_cast<unsigned long long>(m_c0_snap_raced.load(std::memory_order_relaxed)));
		}
	}
	if (RecordKeyedVerifyEnabled()) {
		// KYTY_RECORD_KEYED_VERIFY: a new frame of listed pages (the oldest is dropped); a summary every 64 frames.
		++m_rk_frame;
		m_rk_listed_pages[m_rk_frame % m_rk_listed_pages.size()].clear();
		if (m_rk_frame % 64u == 0u) {
			std::fprintf(stderr, "RECORD_KEYED_VERIFY checks=%llu uncovered=%llu unverified=%llu incomplete_calls=%llu\n",
			             static_cast<unsigned long long>(m_rk_checks), static_cast<unsigned long long>(m_rk_uncovered),
			             static_cast<unsigned long long>(m_rk_unverified),
			             static_cast<unsigned long long>(m_rk_incomplete));
		}
	}
	// Frame boundaries in ticks (DropDeadRingBytes).
	g_ring_frame_ticks[g_ring_frames % g_ring_frame_ticks.size()] = m_scheduler.CurrentTick();
	g_ring_frames++;
	for (auto& range: m_direct_ranges) { // KYTY_PERF_DIRECT_RANGES: an image in a direct range ends it
		if (range.active && m_texture_cache.IsRegionRegistered(range.start, range.end - range.start)) {
			std::fprintf(stderr, "PERF_DIRECT_RANGES 0x%llx-0x%llx ended: an image was registered in it\n",
			             static_cast<unsigned long long>(range.start), static_cast<unsigned long long>(range.end));
			m_texture_cache.DescribeImagesInRegion(range.start, range.end - range.start, 16);
			DeactivateDirect(range);
			range.refused = true;
		}
	}
	IssueEagerReadbacks();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	bool collect    = m_total_used_memory >= m_trigger_gc_memory;
	bool aggressive = m_total_used_memory >= m_critical_gc_memory;
	// KYTY_VRAM_BUDGET: against the live target, like the texture collector, instead of the thresholds computed at
	// start; the memory both collectors freed in the last ticks is still in the measured usage.
	if (VramBudgetEnabled() && m_graphics.CanReportMemoryUsage()) {
		uint64_t target = 0, effective = 0;
		m_graphics.GetVramBudgetState(target, effective);
		if (target != 0) {
			constexpr uint64_t MiB     = 1024ull * 1024;
			const uint64_t     pending = m_texture_cache.PendingReleaseBytes() + RecentlyRetiredBytes();
			const uint64_t     used    = effective > pending ? effective - pending : 0;
			collect                    = used + 512 * MiB >= target;
			aggressive                 = used >= target;
		}
	}
	if (!collect) {
		return;
	}
	m_gc_runs++;

	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	std::vector<BufferId> pinned_buffers;
	size_t                retire_count = 0;
	const auto            consider     = [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (buffer.IsDirect()) { // KYTY_PERF_DIRECT_RANGES: kept
			pinned_buffers.push_back(id);
			return false;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		ApplyEagerReadbacks(buffer.CpuAddress(), buffer.Size(), true);
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			m_gc_dirty_skipped++;
			return false;
		}
		m_gc_retired++;
		m_gc_retired_bytes += buffer.Size();
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			NoteRetiredBytes(buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	};
	// The candidates are listed first, then considered (as MakeRoom does). Retiring one can run deferred operations: with
	// the BDA fallback its Unregister writes the fallback entries through the staging ring, whose wait for ring
	// space runs them (D6 step K4, 23.10: the fault pass created and joined buffers mid-walk, so a walk over the live
	// list met a buffer deleted meanwhile and exited). Buffers deleted meanwhile are skipped; a buffer touched after the
	// listing is no longer below the age and is skipped,
	// as the walk itself would have.
	{
		std::vector<BufferId> scan;
		{
			m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) { scan.push_back(id); });
		}
		for (const auto id: scan) {
			if (IsBufferInvalid(id) || m_slot_buffers[id].lru_touch_tick > tick - age) continue;
			if (consider(id)) break;
		}
	}
	m_gc_kept_bda += pinned_buffers.size();
	for (const auto id: pinned_buffers) {
		TouchBuffer(m_slot_buffers[id]);
	}
	{
		static uint64_t retired_total = 0;
		static uint64_t kept_total    = 0;
		static uint32_t logged        = 0;
		retired_total += retire_count;
		kept_total += pinned_buffers.size();
		if ((retire_count != 0 || !pinned_buffers.empty()) && logged < 40u) {
			++logged;
			std::fprintf(stderr, "BUFFER_GC tick=%llu retired=%zu kept_bda=%zu totals retired=%llu kept=%llu used=%llu MiB\n",
			             static_cast<unsigned long long>(tick), retire_count, pinned_buffers.size(),
			             static_cast<unsigned long long>(retired_total), static_cast<unsigned long long>(kept_total),
			             static_cast<unsigned long long>(m_total_used_memory >> 20u));
		}
	}
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto              completion_tick = m_scheduler.CurrentTick();
	SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Gc); // KYTY_TRACE_SYNC_TIMELINE
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		const LookupWriteGuard lookup_guard(*this); // the resolve thread's lookups
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::ClearDeviceTablesOnce() {
	// Upstream 4644c5292: device memory starts undefined, so a shader must not find stale page-table entries (or
	// fault bits) for memory no buffer was registered for yet.
	if (m_device_tables_cleared) return;
	m_device_tables_cleared = true;
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size, BufferSyncStats* stats) {
	const auto end = vaddr + size;
	// The buffer before the first one starting after vaddr can still overlap the range.
	auto it = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	const auto sync_piece = [&](Buffer& buffer, uint64_t start, uint64_t finish) {
		uint64_t uploaded = 0;
		(void)SynchronizeBuffer(buffer, start, finish - start, false, false,
		                        stats != nullptr ? &uploaded : nullptr, UploadCaller::Region);
		if (stats != nullptr) {
			stats->buffers++;
			if (uploaded != 0) {
				stats->uploads++;
				stats->bytes += uploaded;
			}
		}
	};
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start >= finish) {
			continue;
		}
		sync_piece(buffer, start, finish);
	}
}

// ---- D4b step 1: append windows (bufferCache.h) -------------------------------------------------------------------
// The bytes of an append pool that the GPU did not write are released at the CPU's first request (check
// mode 0 mismatches over ~166k pages; Snake cutscene +7.9%, the fight after it +10.8%, open world +5.3%, gate even).
void BufferCache::AppendBeforePoolMark(uint64_t vaddr, uint64_t size) {
	if (size == 0) return;
	auto pool = std::ranges::find_if(m_append_pools, [&](const AppendPool& p) { return p.base == vaddr && p.size == size; });
	if (pool == m_append_pools.end()) {
		if (m_append_pools.size() < 256u) {
			pool = m_append_pools.emplace(m_append_pools.end());
		} else {
			// A slot is reused in place (windows hold pool indices): an ended cycle first, else the least recently
			// marked pool without a running window, whose cycle ends unreleased (its bytes stay GPU-dirty, as today).
			for (auto it = m_append_pools.begin(); it != m_append_pools.end(); ++it) {
				if (it->windows != 0u) continue;
				if (pool == m_append_pools.end() || (!it->active && pool->active) ||
				    (it->active == pool->active && it->last_mark < pool->last_mark)) {
					pool = it;
				}
			}
			if (pool == m_append_pools.end()) {
				m_append_stats.no_slot++;
				return; // no slot: the pool stays as today (whole binding)
			}
			*pool = AppendPool {};
		}
		pool->base = vaddr;
		pool->size = size;
	}
	pool->last_mark = ++m_append_mark_clock;
	m_append_stats.marks++;
	if (pool->active) return;
	pool->active   = true;
	pool->poisoned = false;
	pool->windows  = 0;
	pool->stride   = 0;
	pool->keep.Clear();
	// The bytes GPU-dirty now hold GPU writes from before this cycle: never released by it.
	{
		std::shared_lock lock(m_gpu_dirty_mutex);
		m_gpu_modified_ranges.ForEachInRange(vaddr, size, [&](uint64_t a, uint64_t b) { pool->keep.Add(a, b - a); });
	}
	// Another pool over the same bytes (other base or size): neither releases anything this cycle.
	for (auto& other: m_append_pools) {
		if (&other != &*pool && other.active && other.base < vaddr + size && vaddr < other.base + other.size) {
			other.poisoned = true;
			pool->poisoned = true;
		}
	}
}

void BufferCache::AppendDispatch(const std::vector<AppendBinding>&                    bindings,
                                 const std::vector<std::pair<uint64_t, uint64_t>>& unqualified) {
	m_append_self_counters.clear();
	AppendResolveWindows(); // finished windows of earlier command buffers (also without CPU requests)
	AppendReport();
	for (const auto& [base, size]: unqualified) {
		m_append_stats.unqualified++;
		for (auto& pool: m_append_pools) {
			if (pool.active && pool.base < base + size && base < pool.base + pool.size) pool.poisoned = true;
		}
	}
	for (const auto& b: bindings) {
		auto pool = std::ranges::find_if(m_append_pools,
		                                 [&](const AppendPool& p) { return p.base == b.pool && p.size == b.pool_size; });
		if (pool == m_append_pools.end() || !pool->active) continue; // no cycle (no slot): as today
		const auto pool_index = static_cast<uint32_t>(pool - m_append_pools.begin());
		if (pool->stride == 0u) {
			pool->stride  = b.stride;
			pool->min_off = b.min_off;
			pool->max_end = b.max_end;
		}
		if (pool->stride != b.stride || pool->min_off != b.min_off || pool->max_end != b.max_end) {
			pool->poisoned = true; // another record layout within one cycle
		}
		m_append_stats.dispatches++;
		auto window = std::ranges::find_if(m_append_windows,
		                                   [&](const AppendWindow& w) { return !w.closed && w.counter == b.counter; });
		if (window == m_append_windows.end()) {
			window          = m_append_windows.emplace(m_append_windows.end());
			window->counter = b.counter;
			window->state   = std::make_shared<AppendWindowState>();
			m_append_stats.windows++;
			// The counter before this dispatch (recorded ahead of it in the same command buffer).
			const bool copied = PeekGpuCopyAsync(b.counter, 4, [state = window->state](const uint8_t* bytes) {
				uint32_t value = 0;
				std::memcpy(&value, bytes, sizeof(value));
				state->before.store(value, std::memory_order_relaxed);
				state->done.fetch_or(1u, std::memory_order_release);
			});
			if (!copied) {
				window->poisoned = true;
				m_append_stats.copy_failed++;
			}
		}
		window->max_records += b.threads;
		if (std::ranges::find(window->pools, pool_index) == window->pools.end()) {
			window->pools.push_back(pool_index);
			pool->windows++;
		}
	}
}

void BufferCache::AppendCloseWindows() {
	if (m_append_windows.empty()) return;
	const bool gpu_thread = GuestGpu::IsGpuThread();
	for (auto& window: m_append_windows) {
		if (window.closed) continue;
		window.closed = true;
		if (window.poisoned) continue;
		// The counter after the last dispatch of this command buffer (only the GPU thread records here).
		const bool copied = gpu_thread && PeekGpuCopyAsync(window.counter, 4, [state = window.state](const uint8_t* bytes) {
			uint32_t value = 0;
			std::memcpy(&value, bytes, sizeof(value));
			state->after.store(value, std::memory_order_relaxed);
			state->done.fetch_or(2u, std::memory_order_release);
		});
		if (!copied) {
			window.poisoned = true;
			m_append_stats.copy_failed++;
		}
	}
}

void BufferCache::AppendResolveWindows() {
	for (auto it = m_append_windows.begin(); it != m_append_windows.end();) {
		auto& window = *it;
		if (!window.closed) {
			++it;
			continue;
		}
		const uint32_t done = window.state->done.load(std::memory_order_acquire);
		// A poisoned window resolves at once; a copied one once both copies arrived.
		if (!window.poisoned && done != 3u) {
			++it;
			continue;
		}
		const uint64_t before = window.state->before.load(std::memory_order_relaxed);
		const uint64_t after  = window.state->after.load(std::memory_order_relaxed);
		bool           ok     = !window.poisoned;
		if (ok && (after < before || after - before > window.max_records)) {
			ok = false; // another writer moved the counter
			m_append_stats.poison_bounds++;
		}
		for (const auto index: window.pools) {
			if (index >= m_append_pools.size()) continue;
			auto& pool = m_append_pools[index];
			if (pool.windows != 0u) pool.windows--;
			if (!ok) {
				pool.poisoned = true;
				continue;
			}
			if (after == before || pool.stride == 0u) continue;
			const uint64_t lo = pool.base + before * pool.stride + pool.min_off;
			const uint64_t hi = pool.base + (after - 1u) * pool.stride + pool.max_end;
			const uint64_t a  = std::max(lo, pool.base);
			const uint64_t b  = std::min(hi, pool.base + pool.size);
			if (a < b) pool.keep.Add(a, b - a);
		}
		if (ok) m_append_stats.records += after - before;
		m_append_stats.resolved++;
		it = m_append_windows.erase(it);
	}
}

void BufferCache::AppendPoisonWindowsOf(uint64_t vaddr, uint64_t size, uint64_t AppendStats::*cause) {
	for (auto& window: m_append_windows) {
		if (window.closed || window.poisoned || window.counter >= vaddr + size || vaddr >= window.counter + 4u) continue;
		window.poisoned = true;
		m_append_stats.*cause += 1u;
	}
}

void BufferCache::AppendForeignWrite(uint64_t vaddr, uint64_t size) {
	if (size == 0) return;
	// A counter written by anything but the dispatches of its window (their own binding is not foreign).
	if (!m_append_windows.empty()) {
		const bool own = std::ranges::any_of(m_append_self_counters,
		                                     [&](uint64_t c) { return c < vaddr + size && vaddr < c + 4u; });
		if (!own) AppendPoisonWindowsOf(vaddr, size, &AppendStats::poison_foreign);
	}
	if (m_append_tagged_obtain) return; // the pool's own whole-binding mark
	for (auto& pool: m_append_pools) {
		if (!pool.active || pool.base >= vaddr + size || vaddr >= pool.base + pool.size) continue;
		const uint64_t a = std::max(vaddr, pool.base);
		const uint64_t b = std::min(vaddr + size, pool.base + pool.size);
		pool.keep.Add(a, b - a);
	}
}

void BufferCache::AppendNoteUpload(uint64_t vaddr, uint64_t size) {
	AppendPoisonWindowsOf(vaddr, size, &AppendStats::poison_upload);
}

bool BufferCache::AppendRelease(uint64_t vaddr, uint64_t size) {
	if (m_append_pools.empty()) return false;
	auto pool = std::ranges::find_if(m_append_pools, [&](const AppendPool& p) {
		return p.active && p.base <= vaddr && vaddr + size <= p.base + p.size;
	});
	if (pool == m_append_pools.end()) return false;
	AppendResolveWindows();
	if (pool->windows != 0u) {
		m_append_stats.pending++; // a window still runs on the GPU: today's path
		AppendReport();
		return false;
	}
	const auto reset = [&] {
		pool->active   = false;
		pool->poisoned = false;
		pool->keep.Clear();
	};
	if (pool->poisoned) {
		reset();
		AppendReport();
		return false;
	}
	// Released: the pool's bytes outside `keep` that are GPU-dirty now.
	std::vector<std::pair<uint64_t, uint64_t>> release;
	uint64_t                                   cursor = pool->base;
	const auto                                 gap    = [&](uint64_t a, uint64_t b) {
        if (a >= b) return;
        m_gpu_modified_ranges.ForEachInRange(a, b - a, [&](uint64_t x, uint64_t y) { release.emplace_back(x, y); });
	};
	pool->keep.ForEachInRange(pool->base, pool->size, [&](uint64_t a, uint64_t b) {
		gap(cursor, a);
		cursor = b;
	});
	gap(cursor, pool->base + pool->size);
	uint64_t kept = pool->size;
	for (const auto& [a, b]: release) kept -= b - a;
	m_append_stats.kept_kb += kept >> 10u;
	uint64_t released = 0;
	for (const auto& [a, b]: release) {
		GpuDirtySubtract(a, b - a);
		released += b - a;
	}
	// Pages left without GPU bytes are no longer GPU-modified (pending eager copies and BAR reads keep theirs).
	for (const auto& [a, b]: release) {
		for (uint64_t page = a & ~(TRACKER_PAGE_SIZE - 1); page < b; page += TRACKER_PAGE_SIZE) {
			if (m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) continue;
			const bool bar = std::ranges::any_of(m_bar_pending_writes, [&](const std::pair<uint64_t, uint64_t>& w) {
				return w.first < page + TRACKER_PAGE_SIZE && page < w.second;
			});
			if (!bar) UnmarkWindow(page, page + TRACKER_PAGE_SIZE);
		}
	}
	m_append_stats.releases++;
	m_append_stats.released_kb += released >> 10u;
	reset();
	AppendReport();
	return !m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

} // namespace Libs::Graphics
