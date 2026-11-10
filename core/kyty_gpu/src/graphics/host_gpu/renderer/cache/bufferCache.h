#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <memory>
#include <span>
#include <utility>
#include <set>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

// Counters of one SynchronizeBuffersInRange call (the BDA region pass's statistics).
struct BufferSyncStats {
	uint64_t buffers = 0;  // buffers visited
	uint64_t uploads = 0;  // buffers that issued a copy
	uint64_t bytes   = 0;  // bytes copied
};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// The fault buffer: the read-fault page bitmap, the BDA store bitmap (KYTY_RECORD_KEYED_VERIFY), then a diagnostic
	// tail (KYTY_DIAG_RT_TRIPS, FaultDiagTripsEnabled). Indices are 32-bit words of the buffer.
	static constexpr uint64_t FAULT_BITMAP_BYTES = CACHING_NUMPAGES / 8;
	static constexpr uint32_t FAULT_DIAG_BASE    = static_cast<uint32_t>(FAULT_BITMAP_BYTES * 2 / 4);
	static constexpr uint32_t FAULT_DIAG_WORDS   = 32768;
	// Tail layout: [0] invocations that hit a loop watchdog cap, [1] watchdog reports taken, [2] BDA stores to a page
	// without a host buffer (dropped), [3] store-miss reports taken; watchdog reports of FAULT_DIAG_TRIP_WORDS words
	// from FAULT_DIAG_TRIP_FIRST, store-miss reports of 4 words from FAULT_DIAG_MISS_FIRST.
	static constexpr uint32_t FAULT_DIAG_TRIP_FIRST = 64;
	static constexpr uint32_t FAULT_DIAG_TRIP_WORDS = 128;
	static constexpr uint32_t FAULT_DIAG_TRIP_SLOTS = 128;
	static constexpr uint32_t FAULT_DIAG_TRIP_LAST  = 8; // reports per invocation: its last iterations before the cap
	static constexpr uint32_t FAULT_DIAG_TRIP_PHIS  = 56; // header phi values per lane half (words 16.., 72..)
	static constexpr uint32_t FAULT_DIAG_MISS_FIRST = FAULT_DIAG_TRIP_FIRST + FAULT_DIAG_TRIP_WORDS * FAULT_DIAG_TRIP_SLOTS;
	static constexpr uint32_t FAULT_DIAG_MISS_SLOTS = 1024;
	static_assert(FAULT_DIAG_MISS_FIRST + 4 * FAULT_DIAG_MISS_SLOTS <= FAULT_DIAG_WORDS);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// The resolve thread's lookups (stageFLookup.h). Resolve thread: the registry's lookup lock if it is free (never waits), and its
	// release. Between them LookupBuffer answers FindBuffer(vaddr, size) when FindBuffer would find a buffer (a null id
	// when it would create one), writing nothing; the buffer registry epoch it saw is DrawRepeat::g_buffer_registry_epoch,
	// read under the same lock. GPU thread: the registry's page table and slots change only under this lock.
	[[nodiscard]] bool     TryLockLookup() noexcept { return m_lookup_lock.try_lock(); }
	void                   UnlockLookup() noexcept { m_lookup_lock.unlock(); }
	[[nodiscard]] BufferId LookupBuffer(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {},
	                                                        bool     image_source    = false);
	// Asks the CPU to fetch the memo entry a later ObtainBuffer of this binding reads (the write memo's for a written one on
	// the GPU thread, the obtain memo's for a read that is not texel), so their cache misses overlap instead of following
	// each other. Changes no data.
	static void PrefetchObtainMemo(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer);
	// FindBuffer's and ObtainBuffer's first question, asked as they ask it (KYTY_PERF_DIRECT_RANGES: the direct range's
	// buffer holding the range, activating or growing it): a null id when none does.
	[[nodiscard]] BufferId DirectBufferAt(uint64_t vaddr, uint64_t size) { return DirectBufferFor(vaddr, size); }
	// Record-keyed V#s: [vaddr, vaddr + size) as a shader reaches it through buffer device addresses: held by
	// a cached buffer (found or created; never the stream shortcut of ObtainBuffer, whose copies the BDA
	// page table does not map), its CPU-modified bytes uploaded, and when written GPU-modified from now on, like a
	// written storage buffer.
	std::pair<Buffer*, uint64_t> ObtainBdaRange(uint64_t vaddr, uint64_t size, bool is_written, BufferId id = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return Staging();
			case MemoryUsage::Stream: return Stream();
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// KYTY_FIX_IMAGE_MEMORY_OWNER: the cached buffer that holds all of [vaddr, vaddr + size) and its offset there, or
	// none; nothing is created, synchronized or touched.
	[[nodiscard]] std::pair<Buffer*, uint64_t> CachedBufferFor(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// HasGpuDirtyBytes for another thread (the resolve thread): the dirty-byte set changes only
	// through GpuDirtyAdd/GpuDirtySubtract, which take m_gpu_dirty_mutex exclusively; this always takes it shared.
	[[nodiscard]] bool HasGpuDirtyBytesShared(uint64_t vaddr, uint64_t size);
	// Publishes pending asynchronous (eager) readbacks overlapping the range: their bytes already left
	// the GPU-dirty set but reach guest memory only when the copy completes. True: the caller puts ApplyDirectCpWrites
	// over what it reads (WaitDirectGpuWrites with `cp_overlay`).
	bool WaitPendingReadbacks(uint64_t vaddr, uint64_t size);
	// True when an eager readback overlapping the range is pending (WaitPendingReadbacks would publish it):
	// guest memory there still holds the older bytes. Changes nothing; GPU thread only, like HasGpuDirtyBytes.
	[[nodiscard]] bool HasPendingReadback(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// BAR reads: a draw or dispatch was recorded into the current command buffer; the
	// writable buffers obtained for it get that command buffer's tick as their write tick.
	void NoteCommandEmitted();
	// Upstream 4644c5292: zeroes the BDA page table and the fault buffer once, in the render scheduler's first command
	// buffer (CommandScheduler::Begin), before the first buffer registration writes into the table.
	void               ClearDeviceTablesOnce();
	void               ProcessFaultBuffer();
	// stats (optional) counts the visited buffers and the uploads.
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size, BufferSyncStats* stats = nullptr);
	// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS: copies the cached GPU buffer contents of a range
	// (written by shaders, possibly through untracked BDA stores) after waiting for queued work.
	[[nodiscard]] bool ReadGpuCopy(uint64_t vaddr, uint64_t size, void* out);
	// The same copy (at most 256 bytes) recorded at this point of the command stream without
	// waiting; `done` gets the bytes on the priority-completion thread once the GPU finished the copy. False (nothing
	// recorded): no single cached buffer holds the range, or guest memory holds newer bytes (CPU-modified).
	[[nodiscard]] bool PeekGpuCopyAsync(uint64_t vaddr, uint64_t size, std::function<void(const uint8_t*)> done);

	// ---- D4b step 1: append windows (bufferCache.cpp) ----------------------------------------------------------------
	// An append pool (ShaderIR.h BufferResource::append_counter) is marked GPU-dirty whole, as every written binding, but
	// a dispatch writes only the records [counter before, counter after). The counter is copied (no CPU wait) before the
	// first tagged dispatch of a command buffer and when it is submitted; the pool bytes no window wrote, that were not
	// GPU-dirty when the pool's cycle began and that no other writer touched are released at the CPU's first request.
	struct AppendBinding {
		uint64_t pool = 0, pool_size = 0, counter = 0, threads = 0;
		uint32_t stride = 0, min_off = 0, max_end = 0;
	};
	// Only pools this large are tracked: whole-binding marks of small pools cost little, and the pool table and the
	// per-write checks stay short.
	static constexpr uint64_t AppendMinPoolBytes = 1024u * 1024u;
	// RebindBuffers of a dispatch with tagged pools: their counters (own marks, not foreign writes), cleared after.
	void AppendSetDispatchCounters(std::vector<uint64_t> counters) { m_append_self_counters = std::move(counters); }
	// NativeStorageBuffer, before ObtainBuffer marks a tagged pool: opens the pool's cycle (bytes GPU-dirty now are kept).
	void AppendBeforePoolMark(uint64_t vaddr, uint64_t size);
	void AppendSetTaggedObtain(bool on) { m_append_tagged_obtain = on; }
	// After RebindBuffers, before the dispatch is recorded: qualified bindings (empty: the tagged pools did not qualify).
	void AppendDispatch(const std::vector<AppendBinding>& bindings, const std::vector<std::pair<uint64_t, uint64_t>>& unqualified);
	void AppendCloseWindows(); // CommandScheduler::Submit (pre-submit hook)
	// The store report (D6 verification, KYTY_RECORD_KEYED_VERIFY): shaders set a bit per 16 KiB page they store to through
	// BDA, and the fault pass checks each reported page against the record-time listing.
	[[nodiscard]] static bool BdaStoreReportCompiled() { return RecordKeyedVerifyEnabled(); }
	// KYTY_RECORD_KEYED_VERIFY: a reported page no listing covers (direct ranges and unregistered memory aside); counted
	// (PERF d6_c0 would_mark) and logged (BDA_STORE_WOULD_MARK).
	void NoteBdaStoreWouldMark(uint64_t vaddr, uint64_t size);
	// KYTY_RECORD_KEYED_VERIFY=1 (D6 verification mode; compiles the store report by itself): every page the GPU reports
	// written (or targeted, when no buffer backs it) through a BDA pointer must overlap a written record-keyed range
	// listed in the frames of the work that fault-buffer pass covers (frames first..last, NoteRecordKeyedWrite); other
	// pages are counted and logged (RECORD_KEYED_VERIFY ... uncovered), passes spanning more frames than kept count as
	// unverified, and so do pages beyond the report's capacity.
	[[nodiscard]] static bool RecordKeyedVerifyEnabled();
	[[nodiscard]] uint64_t    RecordKeyedFrame() const noexcept { return m_rk_frame; }
	void                      NoteRecordKeyedWrite(uint64_t vaddr, uint64_t size);
	void                      NoteRecordKeyedIncomplete() noexcept {
		++m_rk_incomplete;
		m_rk_incomplete_frame = m_rk_frame + 1u; // + 1: 0 means "none yet"
	}
	// D6 step C0 (Stage C0), KYTY_D6_C0 (read once, logged):
	//   0 - the path without C0: every store report marks its pages GPU-modified;
	//   1 (default) - C0. The written record-keyed ranges own their bytes exactly (ObtainBdaRange
	//       makes them GPU-modified when the call is recorded); a store page the fault buffer reports for the work of a
	//       pass is skipped when a written record-keyed range of that work's frames lies on it and no call of those
	//       frames had an incomplete listing (the report repeats bytes the GPU already owns). Every other report is an
	//       unlisted store (the safety net): its pages that are not CPU-dirty become GPU-modified as on path 0; a
	//       CPU-dirty page has no exact answer (neither side's bytes are known) and stays CPU-owned, counted and logged
	//       as UNLISTED_BDA_STORE_CONFLICT (expected 0: a listing gap of step B);
	//   2 - C0 with the upload clobber check (verify mode, see UploadClobberCheck);
	//   3 - the path without C0 with the upload clobber check (the control of the check).
	[[nodiscard]] static uint32_t D6C0Mode();
	[[nodiscard]] static bool     D6C0Enabled() { return D6C0Mode() == 1u || D6C0Mode() == 2u; }
	// KYTY_D6_C0=2/3 (verify only, GPU thread + fault handlers): a guest page that a guest write fault makes CPU-dirty
	// from clean (no CPU-dirty, no GPU-modified mark: guest memory equals the GPU copy) keeps a copy of its bytes from
	// before that write. When an upload of the page is recorded, the GPU copy is copied out right before it (GPU order)
	// and compared once that work completed: a byte where the GPU copy differs from the copy (the GPU wrote it while the
	// tracker did not know) and the uploaded byte differs from the GPU copy is a byte the upload overwrote = a clobber.
	// PERF d6_c0 counts pages checked, pages and bytes clobbered, pages without a copy (unchecked).
	[[nodiscard]] static bool UploadClobberCheck() { return D6C0Mode() >= 2u; }
	void                      NoteGuestWriteFault(uint64_t vaddr);
	// D6 step C0: a written record-keyed range of the frames first..last lies on the 16 KiB store page and no call of
	// those frames (or later) had an incomplete listing; false when the frames are no longer kept.
	[[nodiscard]] bool BdaStoreListed(uint64_t page_address, uint64_t first_frame, uint64_t last_frame) const;
	void               NoteBdaStoreListedSkip() noexcept { ++m_c0_listed_skipped; }
	// D4 step K0 (KYTY_D6_C0=2/3, diagnostic): the last 8 GPU-side events of each 4 KiB page (any thread), printed with
	// each D6_C0_CLOBBER line; PERF d6_c0_writer counts clobbered pages by the newest event that changed the GPU copy
	// (a shader write binding, an image copy, a fill, a buffer copy, a direct-range write) before the check.
	enum class C0Event : uint8_t {
		Snapshot,    // the CPU's first write fault on a clean page (the clobber check's "before")
		Upload,      // CPU-dirty bytes copied into the GPU copy
		Mark,        // pages marked GPU-modified (tracker)
		CleanMark,   // clean pages marked GPU-modified (an unlisted BDA store report)
		Unmark,      // GPU-modified mark removed (tracker)
		Release,     // GPU-modified -> CPU-dirty in one step (tracker)
		ShaderWrite, // a written buffer binding (tag: the writer's shader hash)
		DirectWrite, // a written binding in a direct range (no tracker)
		ImageCopy,   // an image's contents copied into the cached buffer (tag: the image address)
		Fill,        // a GPU fill (tag: the value)
		Copy,        // a GPU buffer copy (destination; tag: the source address)
		RingDead,    // GPU-dirty bytes of the data ring dropped
		ListedSkip,  // a BDA store report skipped as listed (16 KiB page)
		BarRead,     // GPU bytes taken from the host mapping (tag: newest write tick waited for | finished GPU tick << 32)
		CopyRead,    // GPU bytes downloaded through a GPU copy (tag: the recording tick)
		BdaStore,    // a store through a pointer reported by the fault buffer (16 KiB page; tag: listed 1 / not 0)
		Join,        // an older buffer's contents copied into a new one (tag: the older buffer's first address)
		Create,      // a buffer created over the page (tag: its size)
		Count
	};
	void C0Note(uint64_t vaddr, uint64_t size, C0Event kind, uint64_t tag = 0);
	void                      NoteRecordKeyedUnverified(uint64_t pages) noexcept { m_rk_unverified += pages; }
	void                      VerifyBdaWrittenPage(uint64_t page_address, uint64_t first_frame, uint64_t last_frame);
	void               RunGarbageCollector();
	// Bytes of cached guest-memory buffers (KYTY_TRACE_PERF memory split).
	[[nodiscard]] uint64_t UsedMemory() const noexcept { return m_total_used_memory; }
	// Bytes of cached guest-memory buffers (KYTY_TRACE_PERF buffers=): unlike UsedMemory(), which the collector
	// replaces with the device usage of the whole process at every run, only these buffers count.
	[[nodiscard]] uint64_t RegisteredBytes() const noexcept { return m_registered_bytes; }
	// KYTY_TRACE_PERF / KYTY_TRACE_MEM_CENSUS (GPU thread): cached buffers by class ({count, bytes}), the fixed
	// buffers, host-side copies, and the collector's work since the previous census.
	struct Census {
		uint64_t count = 0, bytes = 0, gpu_dirty = 0, gpu_dirty_bytes = 0;
		uint64_t bar = 0, bar_bytes = 0, largest = 0;
		uint64_t bda_table_bytes = 0, staging_bytes = 0, stream_bytes = 0, download_bytes = 0, device_bytes = 0;
		uint64_t other_fixed_bytes = 0; // GDS, fault buffer, readback staging
		uint64_t gc_runs = 0, gc_retired = 0, gc_retired_bytes = 0, gc_kept_bda = 0, gc_dirty_skipped = 0;
		// KYTY_VRAM_BUDGET: buffers retired to make room for a new one over the target, and the bytes retired in
		// the last collector ticks (their deletion waits for the GPU).
		uint64_t alloc_retired = 0, alloc_retired_bytes = 0, recently_retired = 0;
	};
	void TakeCensus(Census& census);
	// KYTY_VRAM_BUDGET (GPU thread): bytes of buffers retired in command buffers the GPU has not retired yet
	// (deferred deletion): the measured usage still holds them.
	[[nodiscard]] uint64_t PendingReleaseBytes() { return RecentlyRetiredBytes(); }


	// Called (GPU thread) for every range a draw, dispatch or tracked BDA store writes; the texture
	// cache uses it to forget metadata keys it knows (KYTY_FIX_DCC_KNOWN_KEYS).
	void SetGpuWriteObserver(std::function<void(uint64_t, uint64_t)> observer) { m_gpu_write_observer = std::move(observer); }
	// The BDA fallback (M06): a mapped direct-memory page without a buffer gets a BDA table entry into the imported
	// direct-memory backing (bit 0 set = fallback), so a shader reads and writes the guest's real bytes instead of zeros /
	// a dropped store; the shader still reports the page (BDA_FAULT), and the fault pass gives it a device-local buffer as
	// before. Only the 2 MiB granules of the backing the GPU used are imported (on the first read of one, which reads 0,
	// or a direct download into it), and the game's granule list keeps them for the next session.
	// KYTY_PERF_BDA_ENTRY_SHARE (0 off, 1 on, 2 verify; spirvEmitterMemory.cpp LoadBdaDwordShared): a scalar load through
	// a GPU-built V# reuses the page-table entry of the invocation's previous such load when the page is the same.
	// Verify counts checks / mismatches in fault-buffer diag words FAULT_DIAG_ENTRY_SHARE (+0, +1).
	[[nodiscard]] static int BdaEntryShareMode();
	static constexpr uint32_t FAULT_DIAG_ENTRY_SHARE = 4;
	// (Words 8, 9, 12 and 13 are free.)
	void QueueFallbackMap(uint64_t vaddr, uint64_t size); // any thread (RenderContext::MapMemory)
	void ApplyFallbackMaps();                             // GPU thread (RenderContext::PrepareBda)
	void ClearFallback(uint64_t vaddr, uint64_t size);    // GPU thread (unmap, after InvalidateMemory)
	[[nodiscard]] bool IsFallbackPage(uint64_t address) const {
		return m_fallback_pages.contains(address & ~(CACHING_PAGESIZE - 1));
	}
	// D6 step K4: the page had a fallback entry until a buffer replaced it at most `frames` frames ago (a miss the fault
	// pass reports later was then served by the entry, not read as 0).
	[[nodiscard]] bool WasFallbackPage(uint64_t address, uint32_t frames) const;
	// The fault pass: `page` (backing memory) was read as 0 because its granule is not imported yet: import it, add it to
	// the game's granule list and give the granule's pages around `page` their fallback entries.
	void WantFallback(uint64_t page);

	// KYTY_TRACE_PERF: the PERF fault_wait_on_caller line (the caller wait's counters).
	void ReportCallerWaitPerf();
	static void ReportBarCopyPerf(); // KYTY_TRACE_PERF: PERF bar_copy (BAR copy bytes/time, read window size)
	// The write-through (see the .cpp), GPU thread: a command-processor write (end-of-pipe label, WRITE_DATA,
	// reference clock) of [address, address + size) into a GPU-modified page, none of whose written bytes the GPU owns.
	// true: written to guest memory and to the GPU copy at this point of the command stream, the page left GPU-owned;
	// false: the caller writes it as before (the write faults and synchronizes the page).
	[[nodiscard]] bool TryCpWriteThrough(uint64_t address, const void* data, uint64_t size);
	// KYTY_PERF_DIRECT_RANGES (A1 stage 2), GPU thread. A direct range's guest memory is the
	// memory of one cache buffer: the host GPU reads and writes the guest's bytes in place, no page of it is tracked.
	//   - InDirectRange: [vaddr, vaddr + size) overlaps an active direct range;
	//   - WaitDirectGpuWrites: before the GPU thread reads guest memory itself (indirect arguments, waits, predicates,
	//     resource tables, DCC keys): waits for GPU writes recorded earlier into those bytes;
	//   - UnmapDirect: an unmap of guest memory (after Finish) ends the direct ranges it overlaps.
	// With `cp_overlay`, command-processor writes (whose bytes the GPU
	// thread knows) are not waited for; WaitDirectGpuWrites then returns true when such writes are still pending over
	// the range, and the caller puts ApplyDirectCpWrites over the guest memory it reads right after.
	[[nodiscard]] bool InDirectRange(uint64_t vaddr, uint64_t size) const;
	bool               WaitDirectGpuWrites(uint64_t vaddr, uint64_t size, bool cp_overlay = false);
	void               ApplyDirectCpWrites(uint64_t vaddr, void* data, uint64_t size) const;
	void               UnmapDirect(uint64_t vaddr, uint64_t size);
	// KYTY_FIX_VIRTUAL_GPU_CLOCK (virtualGpuClock.cpp), GPU thread: a value the GPU produced at an earlier, finished point
	// of the command stream (an end-of-pipe clock value resolved once its command buffer completed) for [address,
	// address + size).
	// Plain: the bytes are in no GPU-modified page; the caller stores them (at most the page becomes CPU-modified).
	// Stored: written to guest memory and to the GPU copy at the current point of the stream, the page left GPU-owned
	// (the TryCpWriteThrough path).
	// Retry: not now (no open command buffer, or a recorded write or pending copy over the bytes).
	// Drop: the GPU owns the bytes, or an image covers them; only a synchronizing store could write them.
	enum class CompletedWrite : uint32_t { Stored, Plain, Retry, Drop };
	[[nodiscard]] CompletedWrite WriteCompletedValue(uint64_t address, const void* data, uint64_t size);
	// KYTY_TRACE_PERF: the PERF cp_write_through line.
	void ReportCpWriteThroughPerf();

	// The upload batch (see the .cpp), GPU thread: while the outermost scope lives, read-only synchronizations of
	// cached buffers only claim their CPU-modified pages (MemoryTracker::ClaimUploadRanges). The scope's end, or the
	// first operation that must see the bytes in a cached buffer before that, commits the batch: one protection update
	// per tracker region for all claims, then the staging copies and upload commands in claim order. Inner scopes do
	// nothing.
	class UploadBatchScope {
	public:
		explicit UploadBatchScope(BufferCache& cache);
		~UploadBatchScope();
		UploadBatchScope(const UploadBatchScope&)            = delete;
		UploadBatchScope& operator=(const UploadBatchScope&) = delete;

	private:
		BufferCache* m_cache = nullptr;
	};
	// KYTY_TRACE_PERF: the PERF protect_batch line.
	void ReportProtectPerf();

private:
	// The join tick: a host-mapped read waits for the buffer's creation tick too (see TryBarDownload).
	enum class CpWriteDecline : uint32_t { Overlap, Pending, Merge, Image, NoBuffer, Shape, Count };
	// The write-through itself, once the page is known GPU-modified: Count when written, else the reason it is not.
	// replace: also over GPU-owned bytes.
	CpWriteDecline CpWriteThroughOwnedPage(uint64_t address, const void* data, uint64_t size, bool replace);
	struct CpWriteStats {
		std::atomic<uint64_t> writes {0}, bytes {0}, replaced {0};
		std::array<std::atomic<uint64_t>, static_cast<uint32_t>(CpWriteDecline::Count)> declined {};
	};
	CpWriteStats m_cp_write_stats;
	enum class CallerFallback : uint32_t { None, UnknownTick, Outside, Large, NoMapping, Pending, Count };
	struct CallerPlan {
		uint64_t       tick     = 0; // the request would wait for the GPU up to this tick (0: it would not wait)
		CallerFallback fallback = CallerFallback::None; // it would wait, for a reason only the drain path serves
	};
	// GPU thread, before the request changes anything: what a guest fault's ReadMemory request would wait for.
	[[nodiscard]] CallerPlan PlanCallerWait(uint64_t vaddr, uint64_t size, bool is_write);
	struct CallerWaitStats {
		std::atomic<uint64_t> eligible {0};       // requests of retried guest faults from other threads than the GPU thread
		std::atomic<uint64_t> deferred {0};       // requests handed back to the faulting thread at least once
		std::atomic<uint64_t> deferrals {0};      // hand-backs (at most CallerMaxDeferrals per request)
		std::atomic<uint64_t> caller_wait_ns {0}; // time the faulting threads waited for the GPU themselves
		std::atomic<uint64_t> retries {0};        // attempts sent again after a wait
		std::atomic<uint64_t> retry_slow {0};     // of those, attempts that still ran >= 1 ms on the GPU thread
		std::atomic<uint64_t> new_writer {0};     // retries that found a newer unfinished writer of the bytes
		std::atomic<uint64_t> exhausted {0};      // requests served by the drain path after the last hand-back
		std::atomic<uint64_t> gpu_ns {0};         // GPU-thread time of all eligible attempts
		std::atomic<uint64_t> gpu_defer_ns {0};   // of it, attempts that were (1) / would have been (2) handed back
		std::atomic<uint64_t> gpu_retry_ns {0};   // of it, attempts after a wait
		std::atomic<uint64_t> fallback[static_cast<uint32_t>(CallerFallback::Count)] {};
	};
	CallerWaitStats m_caller_stats;
	// KYTY_RECORD_KEYED_VERIFY: pages of the written record-keyed ranges listed per frame (a ring of frames, rotated by
	// RunGarbageCollector), and the counts of checked and uncovered BDA-written pages.
	std::array<std::unordered_set<uint64_t>, 16> m_rk_listed_pages;
	uint64_t                                     m_rk_frame      = 0;
	uint64_t                                     m_rk_checks     = 0;
	uint64_t                                     m_rk_uncovered  = 0;
	uint64_t                                     m_rk_unverified = 0;
	uint64_t                                     m_rk_incomplete = 0;
	uint64_t                                     m_rk_incomplete_frame = 0; // last frame with one, + 1 (0: none)
	// D6 step C0 (KYTY_D6_C0), GPU thread: reported store pages skipped as listed, unlisted store pages marked, and
	// CPU-dirty pages an unlisted store reported (UNLISTED_BDA_STORE_CONFLICT).
	uint64_t m_c0_listed_skipped = 0, m_c0_unlisted_pages = 0, m_c0_conflicts = 0;
	// D6 step K2 (KYTY_RECORD_KEYED_VERIFY), GPU thread: reported store pages no listing covers (NoteBdaStoreWouldMark).
	uint64_t m_c0_would_mark = 0;
	// KYTY_D6_C0=2/3 (UploadClobberCheck): the guest bytes of pages from before the write fault that made them CPU-dirty
	// (m_c0_snap_mutex: fault handlers add, the GPU thread takes them at the upload), and the pages whose GPU-owned bytes
	// DropDeadRingBytes dropped without a copy since (their old GPU bytes are no clobber; GPU thread only).
	std::mutex                                                               m_c0_snap_mutex; // leaf
	std::unordered_map<uint64_t, std::unique_ptr<std::array<uint8_t, 4096>>> m_c0_snapshots;
	std::unordered_set<uint64_t>                                              m_c0_dead_pages;
	std::unique_ptr<Buffer>                                                   m_c0_ring;      // download ring
	uint64_t                                                                  m_c0_ring_head = 0;
	std::atomic<uint64_t>                                                     m_c0_ring_tail {0};
	std::atomic<uint64_t> m_c0_checked {0}, m_c0_clobber_pages {0}, m_c0_clobber_bytes {0}, m_c0_dead_bytes {0};
	std::atomic<uint64_t> m_c0_unchecked {0}, m_c0_snap_dropped {0}, m_c0_stale_bytes {0}, m_c0_no_base_bytes {0};
	std::atomic<uint64_t> m_c0_snap_raced {0}; // D4 step K0: later faults of a clean period (their copies not kept)
	// The bytes UploadCopies staged (guest address, bytes), taken by the RecordBufferUpload that follows, and per page
	// a hash of each 32-byte chunk as last uploaded: a chunk the GPU copy still holds as uploaded was not written by
	// the GPU since, so a difference to the snapshot there is guest memory that changed without the tracker (a stale GPU
	// copy the upload refreshes), not a clobber. GPU thread only.
	struct C0Baseline {
		std::array<uint32_t, 128> hash {};
		std::array<uint64_t, 2>   valid {};
	};
	std::vector<std::pair<uint64_t, std::vector<uint8_t>>> m_c0_staged;
	std::unordered_map<uint64_t, C0Baseline>               m_c0_baseline;
	void NoteC0Staged(uint64_t address, const uint8_t* bytes, uint64_t size);
	// D4 step K0: the page event log (C0Note) and the clobbered pages per newest changing event.
	struct C0PageEvent {
		uint64_t tag   = 0;
		uint64_t tick  = 0;
		uint32_t frame = 0;
		uint16_t lo    = 0; // the event's bytes in the page: [lo, hi]
		uint16_t hi    = 0;
		C0Event  kind  = C0Event::Count;
	};
	struct C0PageLog {
		std::array<C0PageEvent, 8> events {};
		uint32_t                   head = 0;
	};
	std::mutex                                                             m_c0_log_mutex; // leaf
	std::unordered_map<uint64_t, C0PageLog>                                m_c0_log;
	std::array<std::atomic<uint64_t>, static_cast<size_t>(C0Event::Count) + 1u> m_c0_writer_pages {}; // last: none
	[[nodiscard]] C0Event C0PageHistory(uint64_t page, std::string& text);
	void                  CheckUploadClobber(Buffer& buffer, std::span<const vk::BufferCopy> copies);
	void UnmarkPublished(uint64_t start, uint64_t end);
	// The read window unmarks only pages no unfinished eager copy still has to publish.
	void        UnmarkWindow(uint64_t lo, uint64_t hi);

	std::function<void(uint64_t, uint64_t)> m_gpu_write_observer;

	// D4b step 1, append windows (GPU thread)
	struct AppendWindowState {
		std::atomic<uint32_t> before {0}, after {0};
		std::atomic<uint32_t> done {0}; // bit 0: before copied, bit 1: after copied
	};
	struct AppendWindow {
		uint64_t                           counter = 0;
		std::shared_ptr<AppendWindowState> state;
		uint64_t                           max_records = 0; // threads of its dispatches: the counter moves at most this
		std::vector<uint32_t>              pools;           // indices into m_append_pools
		bool                               closed = false, poisoned = false;
	};
	struct AppendPool {
		uint64_t base = 0, size = 0;
		uint32_t stride = 0, min_off = 0, max_end = 0;
		bool     active = false, poisoned = false;
		uint32_t windows = 0; // windows of this cycle not resolved yet
		uint64_t last_mark = 0; // m_append_mark_clock at its last mark
		RangeSet keep;        // never released this cycle: GPU-dirty at its start, written by windows or other writers
	};
	struct AppendStats {
		uint64_t dispatches = 0, unqualified = 0, windows = 0, resolved = 0, copy_failed = 0, poison_foreign = 0,
		         poison_upload = 0, poison_bounds = 0, releases = 0, released_kb = 0, kept_kb = 0, pending = 0,
		         records = 0, marks = 0, no_slot = 0;
	};
	std::vector<AppendPool>   m_append_pools;
	uint64_t                  m_append_mark_clock = 0;
	std::vector<AppendWindow> m_append_windows;
	std::vector<uint64_t>     m_append_self_counters;
	bool                      m_append_tagged_obtain = false;
	AppendStats               m_append_stats;
	void AppendResolveWindows();
	void AppendPoisonWindowsOf(uint64_t vaddr, uint64_t size, uint64_t AppendStats::*cause);
	void AppendForeignWrite(uint64_t vaddr, uint64_t size); // a GPU write of pool or counter bytes not by the windows
	void AppendNoteUpload(uint64_t vaddr, uint64_t size);   // CPU bytes uploaded over a counter
	bool AppendRelease(uint64_t vaddr, uint64_t size);      // ReadMemory: true when the range holds no GPU byte now
	void AppendReport();

	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Who asked for a synchronization (an image source and a sized request outside the region pass are not batched).
	enum class UploadCaller : uint8_t { Other, Obtain, BdaRange, Image, Region };
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer,
	                                     uint64_t* uploaded_bytes = nullptr,
	                                     UploadCaller caller = UploadCaller::Other);
	// Records the copies of `copies` from `source` into `buffer` at this point of the command stream, between barriers
	// (SynchronizeBuffer, CommitUploadBatch).
	void RecordBufferUpload(Buffer& buffer, vk::Buffer source, std::span<const vk::BufferCopy> copies);
	// The barrier, copy, barrier of an upload into `target` (deferredCommands.h may record them later).
	void RecordBufferUploadCommands(vk::Buffer target, uint64_t target_size, vk::Buffer source,
	                                std::span<const vk::BufferCopy> copies);

	// The upload batch (UploadBatchScope), GPU thread. Why a batch was committed (PERF protect_batch commits[]).
	enum class BatchCommit : uint8_t { ScopeEnd, Sync, CreateBuffer, DeleteBuffer, ImageCopy, BdaStore, CpWrite, Collector, Count };
	struct UploadBatchEntry {
		Buffer*      buffer      = nullptr;
		uint32_t     first_range = 0; // into UploadBatch::ranges
		uint32_t     range_count = 0;
	};
	struct UploadBatch {
		bool                                       active = false; // an UploadBatchScope is open (and not paused)
		std::vector<UploadBatchEntry>              entries;        // claimed synchronizations, in claim order
		std::vector<std::pair<uint64_t, uint64_t>> ranges;         // their claimed page runs (address, bytes)
		std::vector<RegionManager*>                regions;        // tracker regions whose protection waits
	};
	UploadBatch m_upload_batch;
	[[nodiscard]] UploadBatch& Batch() noexcept { return m_upload_batch; }
	// SynchronizeBuffer's read-only path inside an open batch: claims the CPU-modified pages of the range.
	void ClaimBatchedUpload(Buffer& buffer, uint64_t vaddr, uint64_t size, uint64_t* uploaded_bytes);
	// Write-protects every claimed page (one update per tracker region), then stages and records the claimed uploads.
	// Nothing when the batch is empty; safe to call again from inside (a staging wait that runs deferred work).
	void CommitUploadBatch(BatchCommit reason);
	// Commits the open batch and keeps it closed while it lives (an obtained range that is copied from right away).
	class UploadBatchPause {
	public:
		UploadBatchPause(BufferCache& cache, bool pause);
		~UploadBatchPause();
		UploadBatchPause(const UploadBatchPause&)            = delete;
		UploadBatchPause& operator=(const UploadBatchPause&) = delete;

	private:
		BufferCache* m_cache = nullptr;
	};
	struct ProtectBatchStats {
		std::atomic<uint64_t> scopes {0};         // outermost scopes opened
		std::atomic<uint64_t> claims {0};         // synchronizations that claimed pages
		std::atomic<uint64_t> claim_regions {0};  // tracker regions they cleared pages in (protection updates unbatched)
		std::atomic<uint64_t> claimed_bytes {0};  // bytes claimed
		std::atomic<uint64_t> commits {0};        // commits that had claims
		std::atomic<uint64_t> region_updates {0}; // protection updates the commits made
		std::atomic<uint64_t> commit_ns {0};      // time in commits (KYTY_TRACE_PERF only)
		std::array<std::atomic<uint64_t>, static_cast<size_t>(BatchCommit::Count)> reasons {};
	};
	ProtectBatchStats m_protect_batch_stats;
	// The bookkeeping of a range a draw or dispatch is about to write (ObtainBuffer, ObtainBdaRange).
	void NoteObtainedWrite(uint64_t vaddr, uint64_t size);
	// The write memo: NoteObtainedWrite of a range noted at this tick with nothing undone since (the GPU-dirty set,
	// the write ticks and the granule ticks already hold it): only its notifications.
	void NoteObtainedWriteRepeat(uint64_t vaddr, uint64_t size, bool same_tick);
	// KYTY_PERF_DIRECT_RANGES (see InDirectRange).
	struct DirectRange {
		uint64_t start   = 0;
		uint64_t end     = 0;
		BufferId id      = {};
		bool     active  = false;
		bool     refused = false; // cannot be imported / holds images: the range stays on the tracked path
	};
	struct DirectGpuWrite {
		uint64_t start = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
		// A command-processor write's bytes (cp_size = end - start), 0 for others.
		uint32_t                cp_size = 0;
		std::array<uint8_t, 64> cp_bytes {};
		int64_t                 noted_us = 0; // KYTY_TRACE_PERF: steady-clock microseconds when the write was recorded
	};
	std::vector<DirectRange>    m_direct_ranges;
	std::vector<DirectGpuWrite> m_direct_gpu_writes;
	// The pending command-processor writes the last WaitDirectGpuWrites left to the reader.
	std::vector<DirectGpuWrite> m_direct_cp_overlay;
	uint32_t                    m_direct_mode_seen = 0;
	bool                        m_direct_parsed    = false;
	[[nodiscard]] uint32_t DirectMode();
	// The active direct buffer holding [vaddr, vaddr + size) (activating or growing its range), else a null id.
	[[nodiscard]] BufferId DirectBufferFor(uint64_t vaddr, uint64_t size);
	void                   ActivateDirect(DirectRange& range);
	void                   DeactivateDirect(DirectRange& range, bool replace = true);
	void                   NoteDirectGpuWrite(uint64_t vaddr, uint64_t size, const void* cp_data = nullptr);
	[[nodiscard]] bool     DropDeadRingBytes(uint64_t vaddr);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// found_image (optional): whether an image lies over the range (the obtain memo keeps none of those).
	[[nodiscard]] bool SyncBufferFromWrittenImage(Buffer& buffer, uint64_t vaddr, uint64_t size, bool* found_image = nullptr);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

	// Eager readback: 64 KiB windows the CPU keeps reading after GPU writes ("hot")
	// are copied back asynchronously after each processed submission. When the copy has been
	// published and no newer GPU write was recorded for the window, its pages stop being
	// GPU-modified, so the CPU reads them without a fault or a GPU wait.
	struct HotRead {
		uint64_t start      = 0;
		uint64_t end        = 0;
		uint32_t hits       = 0;
		uint64_t last_use   = 0;
		uint64_t generation = 0;
	};
	struct EagerReadback {
		uint64_t                          start      = 0;
		uint64_t                          end        = 0;
		uint64_t                          generation = 0;
		uint64_t                          tick       = 0;
		std::shared_ptr<std::atomic_bool> done;
		// GPU-dirty byte intervals the copy took out of m_gpu_modified_ranges (a superset).
		std::vector<std::pair<uint64_t, uint64_t>> bytes;
	};
	// True when an unfinished eager readback is copying one of the bytes of the range.
	[[nodiscard]] bool PendingReadbackCovers(uint64_t vaddr, uint64_t size) const;
	void        RecordHotRead(uint64_t vaddr);
	void        BumpHotGeneration(uint64_t vaddr, uint64_t size);
	void        ApplyEagerReadbacks(uint64_t vaddr, uint64_t size, bool wait);
	void        IssueEagerReadbacks();
	// KYTY_TRACE_SYNC_TIMELINE: write tick and GPU-dirty bytes of what the running reader request needs.
	void        NoteSyncTimelineNeed(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	// the BDA fallback (M06)
	bool    FallbackEntry(uint64_t page, vk::DeviceAddress* entry);
	// The imported backing granule (null: not imported, refused, or not wanted and `want` false); also the direct downloads'
	Buffer* FallbackChunk(size_t chunk, bool want);
	void    LoadFallbackList();
	void    WriteFallbackRange(uint64_t vaddr, uint64_t size);
	std::vector<std::unique_ptr<Buffer>>              m_fallback_chunks; // imported backing per granule, null: not yet
	std::vector<uint8_t>                              m_fallback_chunk_failed;
	std::vector<uint8_t>                              m_fallback_wanted; // granules the GPU used (now or in the list)
	std::filesystem::path                             m_fallback_list_path; // the game's granule list (empty: none)
	bool                                              m_fallback_list_loaded = false;
	std::mutex                                        m_fallback_pending_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_fallback_pending;
	std::unordered_set<uint64_t>                      m_fallback_pages; // 16 KiB guest pages with a fallback entry
	std::unordered_map<uint64_t, uint32_t>            m_fallback_replaced; // page -> frame a buffer replaced its entry
	uint32_t m_fallback_replaced_pruned = ~0u; // present count of the last pruning of m_fallback_replaced
	struct FallbackStats {
		uint64_t maps = 0, pages = 0, replaced = 0, restored = 0, cleared = 0, chunks = 0, failed = 0, applies = 0;
		uint64_t listed = 0, learned = 0; // granules named by the game's list at start; granules added to it now
		uint64_t import_us = 0, imported_bytes = 0; // D6 step K4 (PERF bda_fallback)
		// D6b: ClearFallback's Fill calls, its GPU-thread time.
		uint64_t clear_fills = 0, clear_us = 0;
		bool operator==(const FallbackStats&) const = default;
	} m_fallback_stats;
	// D6b: the stats line is printed at most every 5 s and only when a value changed since the last one (applies aside).
	FallbackStats                         m_fallback_printed;
	std::chrono::steady_clock::time_point m_fallback_print_time;
	// DownloadBufferMemory<false>: the copy path's GPU copies go straight into the guest backing (imported per 4 MiB of
	// the backing view) instead of the download ring. Per GPU-dirty interval: its guest address, its offset in the
	// backing and its size.
	struct DownloadToBackingItem {
		uint64_t vaddr = 0, backing = 0, size = 0;
	};
	struct DownloadToBackingPlan {
		std::vector<DownloadToBackingItem>                             items;
		std::vector<std::pair<Buffer*, std::vector<vk::BufferCopy>>>   chunks; // copies per imported chunk
	};
	// False: some interval cannot go straight to the backing (no single mapping, import failed); the whole
	// download then takes the download ring as before.
	[[nodiscard]] bool    PlanDownloadToBacking(uint64_t buffer_address, const std::vector<vk::BufferCopy>& copies,
	                                            DownloadToBackingPlan& plan);
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// Every mutation of m_gpu_modified_ranges goes through these two, under the lock (the resolve thread
	// read the set while the GPU thread changes it).
	void                                              GpuDirtyAdd(uint64_t vaddr, uint64_t size);
	void                                              GpuDirtySubtract(uint64_t vaddr, uint64_t size);
	std::shared_mutex                                 m_gpu_dirty_mutex;
	// Guards m_page_table and m_slot_buffers' insertions/erasures against the resolve thread's LookupBuffer; taken by the GPU thread
	// only for those changes.
	TrackingSpinLock                                  m_lookup_lock {PerfLockSiteBufferLookup};
	// The GPU thread's guard for one registry change.
	class LookupWriteGuard {
	public:
		explicit LookupWriteGuard(BufferCache& cache): m_cache(cache) { m_cache.m_lookup_lock.lock(); }
		~LookupWriteGuard() { m_cache.m_lookup_lock.unlock(); }
		LookupWriteGuard(const LookupWriteGuard&)            = delete;
		LookupWriteGuard& operator=(const LookupWriteGuard&) = delete;

	private:
		BufferCache& m_cache;
	};
	MemoryTracker                                     m_memory_tracker;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	bool                                              m_device_tables_cleared = false; // ClearDeviceTablesOnce
	[[nodiscard]] StreamBuffer& Staging() noexcept { return m_staging_buffer; }
	[[nodiscard]] StreamBuffer& Stream() noexcept { return m_stream_buffer; }
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint64_t m_registered_bytes   = 0; // RegisteredBytes()
	// TakeCensus: the collector's work since the previous census.
	uint64_t m_gc_runs = 0, m_gc_retired = 0, m_gc_retired_bytes = 0, m_gc_kept_bda = 0, m_gc_dirty_skipped = 0;
	// KYTY_VRAM_BUDGET: bytes retired per command-buffer tick, kept until the GPU retired the tick (the buffers are
	// destroyed then); buffers retired before an allocation; the collector tick at which a make-room scan found
	// nothing to retire.
	struct ReleasingBytes {
		uint64_t tick  = 0;
		uint64_t bytes = 0;
	};
	std::vector<ReleasingBytes> m_releasing;
	uint64_t                    m_alloc_retired = 0, m_alloc_retired_bytes = 0;
	uint64_t                    m_make_room_dry_tick = ~uint64_t {0};
	void                        MakeRoomForBuffer(uint64_t vaddr, uint64_t size);
	void                        NoteRetiredBytes(uint64_t bytes);
	[[nodiscard]] uint64_t      RecentlyRetiredBytes();
	std::vector<HotRead>       m_hot_reads;
	std::vector<EagerReadback> m_eager_readbacks;

	// Write ticks are tracked per 64 KiB granule; buffers by creation tick.
	void NoteGpuWriteTick(uint64_t vaddr, uint64_t size);
	// The submission tick of the newest GPU write of each byte range (interval map).
	void                   NoteWriteTick(uint64_t start, uint64_t end, uint64_t tick);
	[[nodiscard]] uint64_t QueryWriteTick(uint64_t start, uint64_t end) const; // 0: part of it unknown
	struct WriteTick {
		uint64_t end;
		uint64_t tick;
	};
	std::map<uint64_t, WriteTick> m_write_ticks;
	static void InsertTickRange(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end, uint64_t tick);
	// The same answers with fewer entries (an insert already held is skipped, equal-tick neighbours are joined).
	static void InsertTickRangeJoined(std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end, uint64_t tick);
	[[nodiscard]] static uint64_t QueryTicks(const std::map<uint64_t, WriteTick>& ticks, uint64_t start, uint64_t end);
	// Ranges whose granule write ticks were raised to `tick` (NoteGpuWriteTick); the granule
	// values only rise, so a range inside one of them at the same tick needs no new loop.
	struct GranuleNote {
		uint64_t start = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::array<GranuleNote, 8> m_granule_notes {};
	uint32_t                   m_granule_note_next = 0;
	[[nodiscard]] std::optional<uint64_t> GranuleWriteTick(uint64_t key) const; // by WriteTickKey; empty: never written
	// BAR reads: GPU-dirty bytes of a mapped buffer whose writes all have a known, submitted
	// write tick are copied from the mapping after waiting for that tick only. 1: copied and
	// published, -1: not possible (use the GPU copy path).
	int TryBarDownload(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// The range a CPU read of [vaddr, vaddr + size) downloads (ReadMemory, PlanCallerWait).
	[[nodiscard]] std::pair<uint64_t, uint64_t> ReadRange(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Its publish of the intervals from span reads of the mapping.
	void BarBatchRead(const uint8_t* mapped, uint64_t base, const std::vector<std::pair<uint64_t, uint64_t>>& intervals);
	// Its scan: the GPU-dirty intervals of the GPU-modified pages of [vaddr, vaddr + size), the newest 64 KiB granule
	// write tick of their pages and their size. -1: usable; 1 a page without a write tick, 5 outside the buffer,
	// 6 nothing GPU-written (TryBarDownload's outcomes).
	int ScanBarDownload(Buffer& buffer, uint64_t vaddr, uint64_t size, std::vector<std::pair<uint64_t, uint64_t>>& intervals,
	                    uint64_t& newest, uint64_t& total);
	// The GPU-written bytes a read takes from the host mapping (BAR, uncached PCIe reads, ~0.66 GB/s measured 18.10 with PERF
	// bar_copy) at most; more go through the GPU copy into cached host memory (the copy path), which is faster for them.
	// Only how the same bytes are copied changes.
	static constexpr uint64_t BarMaxBytes = uint64_t {1024} << 10u;
	// Writable ranges obtained since the last recorded draw/dispatch (write tick not final yet).
	std::vector<std::pair<uint64_t, uint64_t>> m_bar_pending_writes;
	static constexpr uint64_t WriteTickKey(uint64_t address) noexcept { return address >> 16u; }
	// The granule write ticks in two levels - 2^16 blocks
	// of 4096 granules (256 MiB of guest addresses each, 32 KiB), a block allocated at its first write, consecutive
	// granules side by side; 0 = never written (ticks start at 1). Keys beyond the tracked address space go to `high`.
	class GranuleTickTable {
	public:
		static constexpr uint32_t kBlockBits = 12;
		static constexpr uint32_t kTopBits   = 16;
		[[nodiscard]] uint64_t Get(uint64_t key) const {
			if ((key >> (kBlockBits + kTopBits)) != 0u) {
				const auto it = m_high.find(key);
				return it != m_high.end() ? it->second : 0u;
			}
			if (m_top == nullptr) return 0u;
			const auto& block = m_top[key >> kBlockBits];
			return block != nullptr ? block[key & ((1u << kBlockBits) - 1u)] : 0u;
		}
		// The granule raised to `tick` (inserted when it had none).
		void Raise(uint64_t key, uint64_t tick) {
			uint64_t* value = Slot(key);
			if (*value < tick) *value = tick;
		}
		// Raise for every key of [first, last].
		void RaiseRange(uint64_t first, uint64_t last, uint64_t tick) {
			constexpr uint64_t mask = (uint64_t {1} << kBlockBits) - 1u;
			for (uint64_t key = first; key <= last;) {
				if ((key >> (kBlockBits + kTopBits)) != 0u) {
					Raise(key, tick);
					if (key == UINT64_MAX) break;
					key++;
					continue;
				}
				uint64_t* const block = Slot(key) - (key & mask);
				const uint64_t  end   = std::min(last, key | mask);
				for (; key <= end; key++) {
					uint64_t& value = block[key & mask];
					if (value < tick) value = tick;
				}
			}
		}
	private:
		uint64_t* Slot(uint64_t key) {
			if ((key >> (kBlockBits + kTopBits)) != 0u) return &m_high[key];
			if (m_top == nullptr) m_top = std::make_unique<std::unique_ptr<uint64_t[]>[]>(size_t {1} << kTopBits);
			auto& block = m_top[key >> kBlockBits];
			if (block == nullptr) block = std::make_unique<uint64_t[]>(size_t {1} << kBlockBits); // (zeros)
			return &block[key & ((1u << kBlockBits) - 1u)];
		}
		std::unique_ptr<std::unique_ptr<uint64_t[]>[]> m_top;
		std::unordered_map<uint64_t, uint64_t>         m_high;
	};
	GranuleTickTable m_granule_ticks;
	std::unordered_map<uint64_t, uint64_t> m_buffer_ready_tick;
	uint64_t                   m_hot_clock = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
