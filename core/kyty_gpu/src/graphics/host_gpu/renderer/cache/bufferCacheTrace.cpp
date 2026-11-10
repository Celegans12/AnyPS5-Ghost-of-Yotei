#include "graphics/host_gpu/renderer/cache/bufferCacheTrace.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// BufferCache's verify tools and the logs that stay (bufferCacheTrace.h): nothing here runs unless its name is set.
namespace Libs::Graphics {

namespace DownloadWaitTrace {
static const char* const Names[Count] = {"bar_no_tick", "bar_open", "bar_pending", "large_submitted", "large_open",
                                        "bar_outside", "bar_empty", "no_bar", "eager_open", "eager_submitted"};
static uint64_t g_n[Count]          = {};
static uint64_t g_us[Count]         = {};
static uint64_t g_bytes[Count]      = {};
// By the request's reason too (PerfReason: other, fault_game, fault_gpu, srt_walker, dcc_meta, indirect_args, invalidate).
static uint64_t g_reason_n[static_cast<uint32_t>(PerfReason::Count)]  = {};
static uint64_t g_reason_us[static_cast<uint32_t>(PerfReason::Count)] = {};
void Add(Kind kind, uint64_t us, uint64_t bytes) {
	g_n[kind]++;
	g_us[kind] += us;
	g_bytes[kind] += bytes;
	const auto reason = static_cast<uint32_t>(PerfCurrentReason());
	g_reason_n[reason]++;
	g_reason_us[reason] += us;
	static auto last = std::chrono::steady_clock::now();
	const auto  now  = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(10)) return;
	last = now;
	std::fprintf(stderr, "DOWNLOAD_WAITS frame=%u", DebugPresentCount().load());
	for (uint32_t i = 0; i < Count; i++) {
		if (g_n[i] == 0) continue;
		std::fprintf(stderr, " [%s n=%llu ms=%llu kb=%llu]", Names[i], static_cast<unsigned long long>(g_n[i]),
		             static_cast<unsigned long long>(g_us[i] / 1000u), static_cast<unsigned long long>(g_bytes[i] >> 10u));
	}
	static const char* const reasons[] = {"other", "fault_game", "fault_gpu", "srt_walker", "dcc_meta", "indirect_args",
	                                      "invalidate"};
	std::fprintf(stderr, " by_reason");
	for (uint32_t i = 0; i < static_cast<uint32_t>(PerfReason::Count); i++) {
		if (g_reason_n[i] == 0) continue;
		std::fprintf(stderr, " [%s n=%llu ms=%llu]", reasons[i], static_cast<unsigned long long>(g_reason_n[i]),
		             static_cast<unsigned long long>(g_reason_us[i] / 1000u));
	}
	std::fprintf(stderr, "\n");
}
} // namespace DownloadWaitTrace

void BufferCache::NoteBdaStoreWouldMark(uint64_t vaddr, uint64_t size) {
	// Stores into a direct range landed in guest memory; unregistered memory has nothing to own.
	for (const auto& range: m_direct_ranges) {
		if (!range.active || vaddr >= range.end || range.start >= vaddr + size) continue;
		if (vaddr < range.start) NoteBdaStoreWouldMark(vaddr, range.start - vaddr);
		if (vaddr + size > range.end) NoteBdaStoreWouldMark(range.end, vaddr + size - range.end);
		return;
	}
	if (!IsRegionRegistered(vaddr, size)) return;
	if (m_c0_would_mark++ < 64u) {
		std::fprintf(stderr, "BDA_STORE_WOULD_MARK frame=%u page=0x%llx+0x%llx\n", DebugPresentCount().load(),
		             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size));
	}
}

void BufferCache::VerifyBdaWrittenPage(uint64_t page_address, uint64_t first_frame, uint64_t last_frame) {
	const auto page = page_address >> CACHING_PAGEBITS;
	// Only the frames still kept (the ring drops the oldest at every frame boundary).
	if (last_frame < first_frame || m_rk_frame - first_frame >= m_rk_listed_pages.size()) {
		++m_rk_unverified;
		return;
	}
	++m_rk_checks;
	for (uint64_t frame = first_frame; frame <= last_frame; ++frame) {
		if (m_rk_listed_pages[frame % m_rk_listed_pages.size()].contains(page)) return;
	}
	if (m_rk_uncovered++ < 64u) {
		std::fprintf(stderr, "RECORD_KEYED_VERIFY uncovered page=0x%llx (checks=%llu)\n",
		             static_cast<unsigned long long>(page_address), static_cast<unsigned long long>(m_rk_checks));
	}
}

// ---- KYTY_D6_C0=2/3: the upload clobber check (UploadClobberCheck) ------------------------------------------------

namespace {
constexpr uint64_t C0RingSlots    = 4096;   // pages of GPU copies in flight (16 MiB of download memory)
constexpr size_t   C0MaxSnapshots = 32768;  // 128 MiB of page copies at most
constexpr size_t   C0MaxBaselines = 262144; // 1 GiB of guest pages (~140 MiB of chunk hashes)
constexpr uint32_t C0Chunk        = 32;     // bytes per baseline hash

uint32_t C0ChunkHash(const uint8_t* bytes) {
	uint32_t hash = 2166136261u; // FNV-1a
	for (uint32_t i = 0; i < C0Chunk; ++i) hash = (hash ^ bytes[i]) * 16777619u;
	return hash;
}
} // namespace

void BufferCache::NoteGuestWriteFault(uint64_t vaddr) {
	const uint64_t page = vaddr & ~(TRACKER_PAGE_SIZE - 1u);
	// Clean pages only: guest memory equals the GPU copy there. A GPU-modified page is downloaded first (its upload is
	// then counted as unchecked).
	if (m_memory_tracker.IsRegionCpuModified(page, TRACKER_PAGE_SIZE) ||
	    m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE))
		return;
	auto copy = std::make_unique<std::array<uint8_t, 4096>>();
	if (!Libs::LibKernel::Memory::TryReadBacking(page, copy->data(), TRACKER_PAGE_SIZE)) return;
	std::scoped_lock lock(m_c0_snap_mutex);
	if (m_c0_snapshots.size() >= C0MaxSnapshots && !m_c0_snapshots.contains(page)) {
		// Pages written but never uploaded (e.g. memory only images use) fill the table: start over.
		m_c0_snap_dropped.fetch_add(m_c0_snapshots.size(), std::memory_order_relaxed);
		m_c0_snapshots.clear();
	}
	// D4 step K0: several threads can fault on the same clean page at once; the later faults run after the first
	// thread's handler opened the page, so their copy may already hold that thread's write. Only the first copy of a
	// clean period (it ends with the page's upload, which takes or drops the copy) is the bytes from before the CPU.
	if (m_c0_snapshots.contains(page)) {
		m_c0_snap_raced.fetch_add(1, std::memory_order_relaxed);
		C0Note(page, TRACKER_PAGE_SIZE, C0Event::Snapshot, 1u);
		return;
	}
	m_c0_snapshots[page] = std::move(copy);
	C0Note(page, TRACKER_PAGE_SIZE, C0Event::Snapshot); // D4 step K0 (tag 0: the copy kept)
}

// D4 step K0 (KYTY_D6_C0=2/3): the page event log. A range of more than 4096 pages that also exceeds the number of
// logged pages only updates the logged pages inside it (a page the clobber check looks at got its Snapshot event).
void BufferCache::C0Note(uint64_t vaddr, uint64_t size, C0Event kind, uint64_t tag) {
	if (size == 0) return;
	constexpr size_t  MaxLogPages = size_t {1} << 20u;
	const uint64_t    first       = vaddr & ~(TRACKER_PAGE_SIZE - 1u);
	const uint64_t    end         = (vaddr + size + TRACKER_PAGE_SIZE - 1u) & ~(TRACKER_PAGE_SIZE - 1u);
	const uint64_t    pages       = (end - first) / TRACKER_PAGE_SIZE;
	const C0PageEvent event {tag, m_scheduler.CurrentTick(), DebugPresentCount().load(), 0, 0, kind};
	const uint64_t    last_byte = vaddr + size - 1u;
	const auto        put_at    = [&](C0PageLog& log, uint64_t page) {
        auto& slot = log.events[log.head % log.events.size()];
        slot       = event;
        slot.lo    = static_cast<uint16_t>(std::max(vaddr, page) - page);
        slot.hi    = static_cast<uint16_t>(std::min(last_byte, page + TRACKER_PAGE_SIZE - 1u) - page);
        log.head++;
	};
	std::scoped_lock lock(m_c0_log_mutex);
	if (pages > 4096u && pages > m_c0_log.size()) {
		for (auto& [page, log]: m_c0_log) {
			if (page >= first && page < end) put_at(log, page);
		}
		return;
	}
	if (m_c0_log.size() >= MaxLogPages) m_c0_log.clear();
	for (uint64_t page = first; page < end; page += TRACKER_PAGE_SIZE) put_at(m_c0_log[page], page);
}

// The page's events, oldest first, as text; returns the newest event that changed the GPU copy (Count: none).
BufferCache::C0Event BufferCache::C0PageHistory(uint64_t page, std::string& text) {
	static constexpr std::array<const char*, static_cast<size_t>(C0Event::Count)> names {
	    "snap", "upload", "mark", "cleanmark", "unmark", "release", "shader", "direct", "imgcopy", "fill", "copy", "ringdead",
	    "listed", "barread", "copyread", "bdastore", "join", "create"};
	std::scoped_lock lock(m_c0_log_mutex);
	const auto       it = m_c0_log.find(page);
	if (it == m_c0_log.end()) {
		text = " (no events)";
		return C0Event::Count;
	}
	const auto&    log     = it->second;
	C0Event        changer = C0Event::Count;
	const uint32_t n       = std::min<uint32_t>(log.head, static_cast<uint32_t>(log.events.size()));
	for (uint32_t i = 0; i < n; ++i) {
		const auto& e = log.events[(log.head - n + i) % log.events.size()];
		text += fmt::format(" {}@f{}/t{}", names[static_cast<size_t>(e.kind)], e.frame, e.tick);
		if (e.lo != 0 || e.hi != TRACKER_PAGE_SIZE - 1u) text += fmt::format("[{:x}-{:x}]", e.lo, e.hi);
		if (e.tag != 0) text += fmt::format(":{:x}", e.tag);
		switch (e.kind) {
			case C0Event::ShaderWrite:
			case C0Event::DirectWrite:
			case C0Event::ImageCopy:
			case C0Event::Fill:
			case C0Event::Copy: changer = e.kind; break;
			default: break;
		}
	}
	return changer;
}

void BufferCache::NoteC0Staged(uint64_t address, const uint8_t* bytes, uint64_t size) {
	auto& staged = m_c0_staged.emplace_back(address, std::vector<uint8_t>(size));
	std::memcpy(staged.second.data(), bytes, size); // read once: staged and kept bytes are the same
}

void BufferCache::CheckUploadClobber(Buffer& buffer, std::span<const vk::BufferCopy> copies) {
	struct Part {
		uint64_t                                  address  = 0; // guest address of the first byte
		uint32_t                                  bytes    = 0;
		uint64_t                                  ring_off = 0;
		bool                                      dead     = false; // DropDeadRingBytes dropped GPU bytes here
		std::unique_ptr<std::array<uint8_t, 4096>> before;           // guest bytes before the CPU's first write
		std::array<uint8_t, 4096>                 upload {};         // the uploaded bytes (page offsets)
		C0Baseline                                base;              // the page's chunks as last uploaded
	};
	std::vector<Part> parts;
	for (size_t copy_index = 0; copy_index < copies.size(); ++copy_index) {
		const auto&    copy  = copies[copy_index];
		const uint64_t first = buffer.CpuAddress() + copy.dstOffset;
		const uint64_t last  = first + copy.size;
		// The staged bytes of this copy: UploadCopies kept them in the order of the copies (same thread, just before).
		const uint8_t* staged = copy_index < m_c0_staged.size() && m_c0_staged[copy_index].first == first &&
		                                m_c0_staged[copy_index].second.size() == copy.size
		                            ? m_c0_staged[copy_index].second.data()
		                            : nullptr;
		for (uint64_t page = first & ~(TRACKER_PAGE_SIZE - 1u); page < last; page += TRACKER_PAGE_SIZE) {
			const uint64_t lo     = std::max(page, first);
			const uint64_t hi     = std::min(page + TRACKER_PAGE_SIZE, last);
			const auto*    upload = staged != nullptr ? staged + (lo - first) : nullptr;
			if (upload == nullptr) {
				m_c0_unchecked.fetch_add(1, std::memory_order_relaxed);
				std::scoped_lock lock(m_c0_snap_mutex); // the clean period ends here too
				m_c0_snapshots.erase(page);
				continue;
			}
			std::unique_ptr<std::array<uint8_t, 4096>> before;
			{
				std::scoped_lock lock(m_c0_snap_mutex);
				if (auto it = m_c0_snapshots.find(page); it != m_c0_snapshots.end()) {
					before = std::move(it->second);
					m_c0_snapshots.erase(it);
				}
			}
			if (m_c0_baseline.size() >= C0MaxBaselines && !m_c0_baseline.contains(page)) m_c0_baseline.clear();
			auto& base = m_c0_baseline[page];
			if (before != nullptr) {
				auto& part   = parts.emplace_back();
				part.address = lo;
				part.bytes   = static_cast<uint32_t>(hi - lo);
				part.before  = std::move(before);
				part.base    = base;
				std::memcpy(part.upload.data() + (lo - page), upload, hi - lo);
			} else {
				m_c0_unchecked.fetch_add(1, std::memory_order_relaxed);
			}
			// The page's chunks as uploaded now (whole chunks inside the upload only).
			for (uint64_t chunk = (lo - page + C0Chunk - 1u) / C0Chunk; (chunk + 1u) * C0Chunk <= hi - page; ++chunk) {
				base.hash[chunk] = C0ChunkHash(upload + (page + chunk * C0Chunk - lo));
				base.valid[chunk / 64u] |= uint64_t {1} << (chunk % 64u);
			}
		}
	}
	m_c0_staged.clear();
	if (parts.empty()) return;
	const uint64_t sequence = m_c0_ring_head;
	if (sequence + parts.size() - m_c0_ring_tail.load(std::memory_order_acquire) > C0RingSlots) {
		m_c0_unchecked.fetch_add(parts.size(), std::memory_order_relaxed);
		return;
	}
	if (m_c0_ring == nullptr) {
		m_c0_ring = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, C0RingSlots * TRACKER_PAGE_SIZE);
	}
	m_c0_ring_head = sequence + parts.size();
	std::vector<vk::BufferCopy> readback;
	readback.reserve(parts.size());
	for (size_t index = 0; index < parts.size(); ++index) {
		auto& part    = parts[index];
		part.ring_off = ((sequence + index) % C0RingSlots) * TRACKER_PAGE_SIZE;
		part.dead     = m_c0_dead_pages.erase(part.address & ~(TRACKER_PAGE_SIZE - 1u)) != 0u;
		readback.emplace_back(buffer.Offset(part.address), part.ring_off, part.bytes);
	}
	auto&             command = m_scheduler.Current();
	const auto        native  = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0,
	                       nullptr, 0, nullptr);
	native.copyBuffer(buffer.Handle(), m_c0_ring->Handle(), static_cast<uint32_t>(readback.size()), readback.data());
	vk::MemoryBarrier host {};
	host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0, nullptr,
	                       0, nullptr);
	m_scheduler.DeferPriorityOperation([this, sequence, parts = std::move(parts)] {
		for (const auto& part: parts) {
			m_c0_ring->Invalidate(part.ring_off, part.bytes);
			const uint64_t page_offset = part.address & (TRACKER_PAGE_SIZE - 1u);
			const auto*    gpu         = m_c0_ring->Mapped().data() + part.ring_off; // [0, bytes)
			const auto*    before      = part.before->data() + page_offset;
			const auto*    upload      = part.upload.data() + page_offset;
			uint32_t       lost = 0, stale = 0, no_base = 0, first = 0;
			// Whole chunks inside the part only (a partial chunk has no hash of its own).
			for (uint64_t chunk = (page_offset + C0Chunk - 1u) / C0Chunk;
			     (chunk + 1u) * C0Chunk <= page_offset + part.bytes; ++chunk) {
				const uint64_t start = chunk * C0Chunk - page_offset; // part-relative
				uint32_t       differ = 0;
				for (uint32_t i = 0; i < C0Chunk; ++i) {
					if (gpu[start + i] != before[start + i] && upload[start + i] != gpu[start + i]) ++differ;
				}
				if (differ == 0u) continue;
				if ((part.base.valid[chunk / 64u] & (uint64_t {1} << (chunk % 64u))) == 0u) {
					no_base += differ; // never uploaded as a whole chunk: who changed it is unknown
				} else if (C0ChunkHash(gpu + start) == part.base.hash[chunk]) {
					stale += differ; // the GPU copy is as last uploaded: guest memory changed without the tracker
				} else {
					if (lost == 0u) {
						for (uint32_t i = 0; i < C0Chunk; ++i) {
							if (gpu[start + i] != before[start + i] && upload[start + i] != gpu[start + i]) {
								first = static_cast<uint32_t>(start + i);
								break;
							}
						}
					}
					lost += differ;
				}
			}
			m_c0_checked.fetch_add(1, std::memory_order_relaxed);
			if (stale != 0u) m_c0_stale_bytes.fetch_add(stale, std::memory_order_relaxed);
			if (no_base != 0u) m_c0_no_base_bytes.fetch_add(no_base, std::memory_order_relaxed);
			if (lost == 0u) continue;
			if (part.dead) {
				m_c0_dead_bytes.fetch_add(lost, std::memory_order_relaxed);
				continue;
			}
			m_c0_clobber_bytes.fetch_add(lost, std::memory_order_relaxed);
			std::string   history; // D4 step K0: the page's events
			const C0Event changer = C0PageHistory(part.address & ~(TRACKER_PAGE_SIZE - 1u), history);
			m_c0_writer_pages[static_cast<size_t>(changer)].fetch_add(1, std::memory_order_relaxed);
			if (m_c0_clobber_pages.fetch_add(1, std::memory_order_relaxed) < 256u) {
				std::fprintf(stderr,
				             "D6_C0_CLOBBER frame=%u page=0x%llx bytes=%u first=0x%llx gpu=0x%02x before=0x%02x upload=0x%02x "
				             "events:%s\n",
				             DebugPresentCount().load(),
				             static_cast<unsigned long long>(part.address & ~(TRACKER_PAGE_SIZE - 1u)), lost,
				             static_cast<unsigned long long>(part.address + first), gpu[first], before[first],
				             upload[first], history.c_str());
			}
		}
		m_c0_ring_tail.store(sequence + parts.size(), std::memory_order_release);
	});
}

void BufferCache::ReportCpWriteThroughPerf() {
	auto&      s    = m_cp_write_stats;
	const auto take = [](std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0)); };
	const auto d    = [&](CpWriteDecline reason) { return take(s.declined[static_cast<uint32_t>(reason)]); };
	// writes / bytes taken without a fault (replaced: of them, writes over GPU-owned bytes);
	// declined (they fault as before) by reason.
	std::fprintf(stderr,
	             "PERF cp_write_through writes=%llu bytes=%llu replaced=%llu declined[overlap=%llu pending=%llu "
	             "merge=%llu image=%llu no_buffer=%llu shape=%llu]\n",
	             take(s.writes), take(s.bytes), take(s.replaced), d(CpWriteDecline::Overlap), d(CpWriteDecline::Pending),
	             d(CpWriteDecline::Merge), d(CpWriteDecline::Image), d(CpWriteDecline::NoBuffer), d(CpWriteDecline::Shape));
}

void BufferCache::ReportProtectPerf() {
	const auto take = [](std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0)); };
	auto&          s       = m_protect_batch_stats;
	const auto     reason  = [&](BatchCommit r) { return take(s.reasons[static_cast<size_t>(r)]); };
	const uint64_t regions = s.claim_regions.exchange(0);
	const uint64_t updates = s.region_updates.exchange(0);
	// scopes: outermost batch scopes opened (PrepareBda region passes, draw/dispatch buffer bindings); claims:
	// read-only synchronizations that cleared pages; claim_regions: the tracker regions they cleared pages in (one
	// protection update each without the batch); region_updates: the protection updates the commits made instead
	// (updates_saved = the difference; other threads' updates that protected claimed pages first are not counted);
	// commits[]: why the batches were committed (scope_end is the normal case; the others are operations that had
	// to see the claimed uploads first).
	std::fprintf(stderr,
	             "PERF protect_batch scopes=%llu claims=%llu claim_regions=%llu claimed_kb=%llu commits=%llu "
	             "region_updates=%llu updates_saved=%lld commit_ms=%llu commits[scope_end=%llu sync=%llu create=%llu "
	             "delete=%llu image=%llu bda_store=%llu cp_write=%llu collector=%llu]\n",
	             take(s.scopes), take(s.claims), static_cast<unsigned long long>(regions), take(s.claimed_bytes) >> 10u,
	             take(s.commits), static_cast<unsigned long long>(updates),
	             static_cast<long long>(regions) - static_cast<long long>(updates), take(s.commit_ns) / 1000000u,
	             reason(BatchCommit::ScopeEnd), reason(BatchCommit::Sync), reason(BatchCommit::CreateBuffer),
	             reason(BatchCommit::DeleteBuffer), reason(BatchCommit::ImageCopy), reason(BatchCommit::BdaStore),
	             reason(BatchCommit::CpWrite), reason(BatchCommit::Collector));
}

void BufferCache::ReportCallerWaitPerf() {
	auto&      s    = m_caller_stats;
	const auto take = [](std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0)); };
	const auto ms   = [](std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0) / 1000000u); };
	auto&      f    = s.fallback;
	const auto fb   = [&](CallerFallback reason) { return take(f[static_cast<uint32_t>(reason)]); };
	// eligible: requests of retried guest faults from other threads; deferred / deferrals: handed back to the
	// faulting thread; caller_wait_ms: those threads waiting for the GPU; retries (retry_slow: still >= 1 ms on the
	// GPU thread); new_writer: a retry found a newer unfinished write (64 KiB granule); exhausted: served as before
	// after the last hand-back; gpu_ms: GPU-thread time of all eligible attempts (defer: handed back; retry: after a
	// wait); fallback: requests only the drain path serves.
	std::fprintf(stderr,
	             "PERF fault_wait_on_caller eligible=%llu deferred=%llu deferrals=%llu caller_wait_ms=%llu "
	             "retries=%llu retry_slow=%llu new_writer=%llu exhausted=%llu gpu_ms=%llu "
	             "gpu_defer_ms=%llu gpu_retry_ms=%llu fallback[unknown_tick=%llu outside=%llu large=%llu no_mapping=%llu "
	             "pending=%llu]\n",
	             take(s.eligible), take(s.deferred), take(s.deferrals), ms(s.caller_wait_ns), take(s.retries),
	             take(s.retry_slow), take(s.new_writer), take(s.exhausted), ms(s.gpu_ns),
	             ms(s.gpu_defer_ns), ms(s.gpu_retry_ns), fb(CallerFallback::UnknownTick), fb(CallerFallback::Outside),
	             fb(CallerFallback::Large), fb(CallerFallback::NoMapping), fb(CallerFallback::Pending));
}

void BufferCache::TakeCensus(Census& census) {
	census = {};
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer = m_slot_buffers[id];
		const auto  size   = buffer.Size();
		census.count++;
		census.bytes += size;
		census.largest = std::max(census.largest, size);
		if (m_gpu_modified_ranges.Intersects(address, size)) {
			census.gpu_dirty++;
			census.gpu_dirty_bytes += size;
		}
		if (!buffer.Mapped().empty()) {
			census.bar++;
			census.bar_bytes += size;
		}
	}
	census.bda_table_bytes   = m_bda_pagetable_buffer.Size();
	census.staging_bytes     = m_staging_buffer.Size();
	census.stream_bytes      = m_stream_buffer.Size();
	census.download_bytes    = m_download_buffer.Size();
	census.device_bytes      = m_device_buffer.Size();
	census.other_fixed_bytes = m_gds_buffer.Size() + m_fault_manager.GetFaultBuffer()->Size();
	census.gc_runs          = m_gc_runs;
	census.gc_retired       = m_gc_retired;
	census.gc_retired_bytes = m_gc_retired_bytes;
	census.gc_kept_bda      = m_gc_kept_bda;
	census.gc_dirty_skipped = m_gc_dirty_skipped;
	m_gc_runs = m_gc_retired = m_gc_retired_bytes = m_gc_kept_bda = m_gc_dirty_skipped = 0;
	census.alloc_retired       = m_alloc_retired;
	census.alloc_retired_bytes = m_alloc_retired_bytes;
	census.recently_retired    = RecentlyRetiredBytes();
	m_alloc_retired = m_alloc_retired_bytes = 0;
}

void BufferCache::AppendReport() {
	static int64_t window_start = 0;
	const int64_t  now =
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	if (window_start == 0) window_start = now;
	if (now - window_start < 5000) return;
	window_start = now;
	auto& s      = m_append_stats;
	std::fprintf(stderr,
	             "PERF append_range marks=%llu no_slot=%llu dispatches=%llu unqualified=%llu windows=%llu resolved=%llu records=%llu "
	             "copy_failed=%llu poison(foreign/upload/bounds)=%llu/%llu/%llu pending=%llu releases=%llu "
	             "released_KiB=%llu kept_KiB=%llu pools=%zu "
	             "windows_open=%zu\n",
	             static_cast<unsigned long long>(s.marks), static_cast<unsigned long long>(s.no_slot), static_cast<unsigned long long>(s.dispatches),
	             static_cast<unsigned long long>(s.unqualified), static_cast<unsigned long long>(s.windows),
	             static_cast<unsigned long long>(s.resolved), static_cast<unsigned long long>(s.records),
	             static_cast<unsigned long long>(s.copy_failed), static_cast<unsigned long long>(s.poison_foreign),
	             static_cast<unsigned long long>(s.poison_upload), static_cast<unsigned long long>(s.poison_bounds),
	             static_cast<unsigned long long>(s.pending), static_cast<unsigned long long>(s.releases),
	             static_cast<unsigned long long>(s.released_kb), static_cast<unsigned long long>(s.kept_kb),
	             m_append_pools.size(), m_append_windows.size());
	s = {};
}

} // namespace Libs::Graphics

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep

namespace Libs::Graphics {
void PrintBigBufferBacktrace() {
	void*      frames[24] {};
	const auto count = RtlCaptureStackBackTrace(1, 24, frames, nullptr);
	const auto base  = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	for (USHORT i = 0; i < count; i++) {
		std::fprintf(stderr, "  BIG_BUFFER_CREATE frame %u: exe+0x%llx\n", static_cast<unsigned>(i),
		             static_cast<unsigned long long>(reinterpret_cast<uint64_t>(frames[i]) - base));
	}
}
} // namespace Libs::Graphics
#else
namespace Libs::Graphics {
void PrintBigBufferBacktrace() {}
} // namespace Libs::Graphics
#endif
