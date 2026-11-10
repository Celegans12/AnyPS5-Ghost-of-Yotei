#include "graphics/host_gpu/renderer/syncTimeline.h"

#include "common/envRead.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>



namespace Libs::Graphics::SyncTimeline {

namespace {

constexpr uint32_t kSites   = static_cast<uint32_t>(Site::Count);
constexpr uint32_t kThreads = 3; // gpu, prio, other
// Readers of PerfReason, plus waits of the GPU thread outside a reader request.
constexpr uint32_t kReasons = static_cast<uint32_t>(PerfReason::Count) + 1;
enum Verdict : uint32_t { Real, Page, Window, Same, NoDownload, kVerdicts };

const char* const kSiteNames[kSites]       = {"none",   "copy",  "bar",         "eager", "prio_ops",
                                              "finish", "unmap", "stream_wrap", "gc",    "fault_buffer", "flip_done", "suspend",
                                              "caller", "srt_exact"};
const char* const kThreadNames[kThreads]   = {"gpu", "prio", "other"};
const char* const kReasonNames[kReasons]   = {"other",         "fault_game", "fault_gpu", "srt_walker", "dcc_meta",
                                              "indirect_args", "invalidate", "none"};
const char* const kVerdictNames[kVerdicts] = {"real", "page", "window", "same", "nodl"};

// Cumulative counters in one flat array: window and frame totals are differences of snapshots.
enum : uint32_t {
	kSiteN     = 0,                            // [thread][site] waits
	kSiteNs    = kSiteN + kThreads * kSites,   // [thread][site] ns
	kReasonN   = kSiteNs + kThreads * kSites,  // GPU-thread reader requests that waited, by reader
	kReasonNs  = kReasonN + kReasons,          // their wait ns
	kVerdictN  = kReasonNs + kReasons,         // the same requests by verdict
	kVerdictNs = kVerdictN + kVerdicts,
	kNeedDoneN = kVerdictNs + kVerdicts, // requests whose needed bytes were finished (or not GPU-written) when they waited
	kNeedDoneNs,
	kGranuleN, // of those, requests whose exact write tick was finished but not their 64 KiB granule tick
	kGranuleNs,
	kBeyondN, // requests whose needed write was unfinished but that waited for a later tick
	kBeyondNs,
	kOpenN, // GPU-thread waits that first submitted the open command buffer
	kOpenNs,
	kPendingSamples, // GPU-thread timeline waits: pending submissions and age of the oldest one when the wait began
	kPendingTicks,
	kBacklogNs,
	kGameN, // reader requests sent by game threads: time until the GPU thread ran them, the part of it the GPU
	kGameQueueNs, // thread spent waiting for the GPU, requests queued >= 1 ms
	kGameBehindNs,
	kGameSlowN,
	kDlBytes, // synchronously downloaded bytes compared, and those that differed from guest memory
	kDlChanged,
	kLabels, // end-of-pipe label writes, WAIT_REG_MEM packets, blocked ones and their blocked time
	kWrm,
	kWrmBlocks,
	kWrmBlockedNs,
	kCount
};

using Values = std::array<uint64_t, kCount>;

std::array<std::atomic<uint64_t>, kCount>& Counters() {
	static std::array<std::atomic<uint64_t>, kCount> counters {};
	return counters;
}

void Add(uint32_t index, uint64_t value) {
	if (value != 0) Counters()[index].fetch_add(value, std::memory_order_relaxed);
}

Values Load() {
	Values values {};
	for (uint32_t i = 0; i < kCount; i++) {
		values[i] = Counters()[i].load(std::memory_order_relaxed);
	}
	return values;
}

Values Delta(const Values& now, const Values& before) {
	Values values {};
	for (uint32_t i = 0; i < kCount; i++) {
		values[i] = now[i] - before[i];
	}
	return values;
}

struct DetailFrames {
	uint32_t first = 0;
	uint32_t count = 0;
};

const DetailFrames& Detail() {
	static const DetailFrames frames = [] {
		DetailFrames result;
		uint32_t     count = 3;
		if (const char* value = std::getenv("KYTY_TRACE_SYNC_TIMELINE_FRAMES"); value != nullptr) {
			count = std::max<uint32_t>(1u, static_cast<uint32_t>(std::strtoul(value, nullptr, 10)));
		}
		if (const char* value = std::getenv("KYTY_TRACE_SYNC_TIMELINE_FRAME"); value != nullptr) {
			result = {static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) + 1u, count};
		} else if (const auto [first, last] = Common::EnvRead::FrameRange(); first <= last) {
			result = {first, std::min<uint32_t>(count, last - first + 1u)};
		}
		if (result.count != 0) {
			std::fprintf(stderr, "TRACE_SYNC_TIMELINE detail frames %u-%u\n", result.first, result.first + result.count - 1u);
		}
		return result;
	}();
	return frames;
}

bool DetailFrame(uint32_t frame) {
	const auto& detail = Detail();
	return detail.count != 0 && frame >= detail.first && frame < detail.first + detail.count;
}

std::atomic<int64_t>  g_frame_start_ns {0};
std::atomic<uint64_t> g_gpu_wait_ns {0};
std::atomic<uint32_t> g_threads {0};

constexpr uint64_t                         kRing = 4096;
std::array<std::atomic<int64_t>, kRing>    g_submit_ns {};

// Submission time of a submitted tick (0: unknown, not submitted, or too old for the ring).
int64_t SubmitNs(uint64_t tick, uint64_t current_tick) {
	if (tick == 0 || tick >= current_tick || current_tick - tick >= kRing) return 0;
	return g_submit_ns[tick & (kRing - 1u)].load(std::memory_order_relaxed);
}

double Ms(int64_t ns) {
	return static_cast<double>(ns) / 1e6;
}

double FrameMs(int64_t now) {
	const auto start = g_frame_start_ns.load(std::memory_order_relaxed);
	return start != 0 ? Ms(now - start) : 0.0;
}

struct Request {
	PerfReason reason        = PerfReason::Other;
	uint64_t   vaddr         = 0;
	uint64_t   size          = 0;
	bool       is_write      = false;
	bool       game_thread   = false;
	uint32_t   thread_number = 0;
	int64_t    start_ns      = 0;
	int64_t    queue_ns      = 0;
	int64_t    behind_ns     = 0;
	uint64_t   need_lo       = 0;
	uint64_t   need_hi       = 0;
	uint64_t   page_lo       = 0;
	uint64_t   page_hi       = 0;
	bool       need_known    = false;
	uint64_t   need_tick     = 0; // by 64 KiB granule
	uint64_t   exact_tick    = 0; // by written range
	uint64_t   need_dirty    = 0;
	uint64_t   page_dirty    = 0;
	// The waits it made (the longest one describes the pending work).
	uint32_t   waits         = 0;
	int64_t    wait_ns       = 0;
	int64_t    longest_ns    = -1;
	int64_t    need_done_ns  = 0;
	int64_t    granule_ns    = 0;
	int64_t    beyond_ns     = 0;
	Site       site          = Site::None;
	uint64_t   waited_tick   = 0;
	uint64_t   current_tick  = 0;
	uint64_t   pending_ticks = 0;
	int64_t    backlog_ns    = 0;
	int64_t    age_ns        = 0;
	bool       open          = false;
	// Its synchronous downloads, compared with guest memory before their publish.
	uint64_t   dl_bytes      = 0;
	uint64_t   dl_changed    = 0;
	uint64_t   page_changed  = 0;
	uint64_t   need_changed  = 0;
	uint64_t   compare_fails = 0;
};

thread_local std::vector<Request> t_requests;
thread_local Site                 t_site     = Site::None;
thread_local bool                 t_open     = false;
thread_local uint64_t             t_caller   = 0;
thread_local bool                 t_priority = false;

Request* CurrentRequest() {
	return t_requests.empty() ? nullptr : &t_requests.back();
}

uint32_t ThreadKind() {
	return PerfIsGpuThread() ? 0u : t_priority ? 1u : 2u;
}

uint64_t Rva(uint64_t address) {
	// The executable's preferred base.
	return address >= 0x140000000ull ? address - 0x140000000ull : address;
}

Verdict Classify(const Request& r) {
	if (r.dl_bytes == 0 || r.compare_fails >= r.dl_bytes) return NoDownload;
	if (r.need_changed != 0) return Real;
	if (r.page_changed != 0) return Page;
	if (r.dl_changed != 0) return Window;
	return Same;
}

void PrintTotals(const char* prefix, const Values& d) {
	char       line[4096];
	size_t     used = 0;
	const auto put  = [&](int written) {
        if (written > 0) used = std::min(sizeof(line) - 1u, used + static_cast<size_t>(written));
	};
	const auto ull = [](uint64_t value) { return static_cast<unsigned long long>(value); };
	const auto ms  = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	put(std::snprintf(line + used, sizeof(line) - used, "%s", prefix));
	for (uint32_t thread = 0; thread < kThreads; thread++) {
		uint64_t n  = 0;
		uint64_t ns = 0;
		for (uint32_t site = 0; site < kSites; site++) {
			n += d[kSiteN + thread * kSites + site];
			ns += d[kSiteNs + thread * kSites + site];
		}
		put(std::snprintf(line + used, sizeof(line) - used, " %s_wait=%llu/%.1fms[", kThreadNames[thread], ull(n), ms(ns)));
		bool first = true;
		for (uint32_t site = 0; site < kSites; site++) {
			const auto sn = d[kSiteN + thread * kSites + site];
			if (sn == 0) continue;
			put(std::snprintf(line + used, sizeof(line) - used, "%s%s=%llu/%.1f", first ? "" : " ", kSiteNames[site], ull(sn),
			                  ms(d[kSiteNs + thread * kSites + site])));
			first = false;
		}
		put(std::snprintf(line + used, sizeof(line) - used, "]"));
	}
	put(std::snprintf(line + used, sizeof(line) - used, " readers["));
	for (uint32_t reason = 0; reason < kReasons; reason++) {
		put(std::snprintf(line + used, sizeof(line) - used, "%s%s=%llu/%.1f", reason == 0 ? "" : " ", kReasonNames[reason],
		                  ull(d[kReasonN + reason]), ms(d[kReasonNs + reason])));
	}
	put(std::snprintf(line + used, sizeof(line) - used, "] verdict["));
	for (uint32_t verdict = 0; verdict < kVerdicts; verdict++) {
		put(std::snprintf(line + used, sizeof(line) - used, "%s%s=%llu/%.1f", verdict == 0 ? "" : " ", kVerdictNames[verdict],
		                  ull(d[kVerdictN + verdict]), ms(d[kVerdictNs + verdict])));
	}
	const auto samples = d[kPendingSamples];
	put(std::snprintf(line + used, sizeof(line) - used,
	                  "] need_done=%llu/%.1fms granule=%llu/%.1fms beyond=%llu/%.1fms open_flush=%llu/%.1fms pending_avg=%.1f "
	                  "backlog_avg_ms=%.2f",
	                  ull(d[kNeedDoneN]), ms(d[kNeedDoneNs]), ull(d[kGranuleN]), ms(d[kGranuleNs]), ull(d[kBeyondN]),
	                  ms(d[kBeyondNs]), ull(d[kOpenN]), ms(d[kOpenNs]),
	                  samples != 0 ? static_cast<double>(d[kPendingTicks]) / static_cast<double>(samples) : 0.0,
	                  samples != 0 ? ms(d[kBacklogNs]) / static_cast<double>(samples) : 0.0));
	put(std::snprintf(line + used, sizeof(line) - used, " game_queue=%llu/%.1fms behind_gpu_wait=%.1fms slow=%llu",
	                  ull(d[kGameN]), ms(d[kGameQueueNs]), ms(d[kGameBehindNs]), ull(d[kGameSlowN])));
	put(std::snprintf(line + used, sizeof(line) - used,
	                  " dl_kb=%llu changed_kb=%llu guest[labels=%llu wrm=%llu wrm_block=%llu/%.1fms]", ull(d[kDlBytes] >> 10u),
	                  ull(d[kDlChanged] >> 10u), ull(d[kLabels]), ull(d[kWrm]), ull(d[kWrmBlocks]), ms(d[kWrmBlockedNs])));
	std::fprintf(stderr, "%s\n", line);
}

} // namespace

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

uint64_t GpuThreadWaitNs() {
	return g_gpu_wait_ns.load(std::memory_order_relaxed);
}

uint32_t ThreadNumber() {
	static thread_local const uint32_t number = g_threads.fetch_add(1, std::memory_order_relaxed) + 1u;
	return number;
}

CallInfo Call(bool gpu_thread) {
	CallInfo info;
	info.call_ns       = NowNs();
	info.wait_snapshot = GpuThreadWaitNs();
	info.thread_number = ThreadNumber();
	info.game_thread   = !gpu_thread;
	return info;
}

void NoteSubmit(uint64_t tick) {
	if (!Enabled()) return;
	g_submit_ns[tick & (kRing - 1u)].store(NowNs(), std::memory_order_relaxed);
}

void NoteFrame(uint32_t frame) {
	if (!Enabled()) return;
	// Presenter thread only.
	static Values frame_start {};
	const auto    now      = NowNs();
	const auto    previous = g_frame_start_ns.exchange(now, std::memory_order_relaxed);
	const auto    values   = Load();
	if (frame > 0 && DetailFrame(frame - 1u)) {
		char prefix[96];
		std::snprintf(prefix, sizeof(prefix), "SYNC_TL_FRAME frame=%u frame_ms=%.1f", frame - 1u,
		              previous != 0 ? Ms(now - previous) : 0.0);
		PrintTotals(prefix, Delta(values, frame_start));
	}
	frame_start = values;
}

void MarkPriorityThread() {
	t_priority = true;
}

void NoteOpenFlush() {
	t_open = true;
}

void NoteCaller(uint64_t address) {
	t_caller = address;
	t_open   = false;
}

SiteScope::SiteScope(Site site) {
	if (!Enabled() || t_site != Site::None) return;
	t_site = site;
	m_set  = true;
}

SiteScope::~SiteScope() {
	if (m_set) t_site = Site::None;
}

WaitScope::WaitScope(Site site, uint64_t tick, uint64_t current_tick, uint64_t known_tick) {
	if (!Enabled()) return;
	m_on       = true;
	m_site     = site != Site::None ? site : t_site;
	m_tick     = tick;
	m_current  = current_tick;
	m_known    = known_tick;
	m_start_ns = NowNs();
}

WaitScope::~WaitScope() {
	if (!m_on) return;
	const auto     end    = NowNs();
	const auto     ns     = end - m_start_ns;
	const uint32_t kind   = ThreadKind();
	const auto     site   = static_cast<uint32_t>(m_site);
	const bool     open   = t_open;
	t_open                = false;
	Add(kSiteN + kind * kSites + site, 1);
	Add(kSiteNs + kind * kSites + site, static_cast<uint64_t>(ns));
	uint64_t pending = 0;
	int64_t  backlog = 0;
	int64_t  age     = 0;
	if (m_tick != 0) {
		pending            = m_tick > m_known ? m_tick - m_known : 0;
		const auto oldest  = SubmitNs(m_known + 1u, m_current);
		const auto waited  = SubmitNs(m_tick, m_current);
		backlog            = oldest != 0 ? m_start_ns - oldest : 0;
		age                = waited != 0 ? m_start_ns - waited : 0;
	}
	if (kind == 0) {
		g_gpu_wait_ns.fetch_add(static_cast<uint64_t>(ns), std::memory_order_relaxed);
		if (m_tick != 0) {
			Add(kPendingSamples, 1);
			Add(kPendingTicks, pending);
			Add(kBacklogNs, static_cast<uint64_t>(std::max<int64_t>(backlog, 0)));
		}
		if (open) {
			Add(kOpenN, 1);
			Add(kOpenNs, static_cast<uint64_t>(ns));
		}
		if (auto* r = CurrentRequest(); r != nullptr) {
			r->waits++;
			r->wait_ns += ns;
			// The needed bytes were already finished (or are not GPU-written): the wait was for other work. The exact
			// (per written range) tick decides when known, else the 64 KiB granule tick.
			if (m_tick != 0 && r->need_known) {
				const uint64_t need = r->exact_tick != 0 ? r->exact_tick : r->need_tick;
				if (r->need_dirty == 0 || (need != 0 && need <= m_known)) {
					r->need_done_ns += ns;
					if (r->need_dirty != 0 && r->exact_tick != 0 && r->need_tick > m_known) r->granule_ns += ns;
				} else if (need != 0 && need < m_tick) {
					r->beyond_ns += ns;
				}
			}
			if (ns > r->longest_ns) {
				r->longest_ns    = ns;
				r->site          = m_site;
				r->waited_tick   = m_tick;
				r->current_tick  = m_current;
				r->pending_ticks = pending;
				r->backlog_ns    = backlog;
				r->age_ns        = age;
			}
			r->open |= open;
			return;
		}
		Add(kReasonN + kReasons - 1u, 1);
		Add(kReasonNs + kReasons - 1u, static_cast<uint64_t>(ns));
	}
	const auto frame = DebugPresentCount().load(std::memory_order_relaxed);
	if (DetailFrame(frame)) {
		std::fprintf(stderr,
		             "SYNC_TL frame=%u t=%.2f thr=%s#%u site=%s caller=0x%llx wait_us=%lld tick=%llu cur=%llu "
		             "gpu_tick=%llu pending=%llu backlog_ms=%.2f age_ms=%.2f open=%d\n",
		             frame, FrameMs(m_start_ns), kThreadNames[kind], ThreadNumber(), kSiteNames[site],
		             static_cast<unsigned long long>(Rva(t_caller)), static_cast<long long>(ns / 1000),
		             static_cast<unsigned long long>(m_tick), static_cast<unsigned long long>(m_current),
		             static_cast<unsigned long long>(m_known), static_cast<unsigned long long>(pending), Ms(backlog),
		             Ms(age), open ? 1 : 0);
	}
}

RequestScope::RequestScope(PerfReason reason, uint64_t vaddr, uint64_t size, bool is_write, const CallInfo& call) {
	if (!Enabled()) return;
	m_on = true;
	Request r;
	r.reason        = reason;
	r.vaddr         = vaddr;
	r.size          = size;
	r.is_write      = is_write;
	r.game_thread   = call.game_thread;
	r.thread_number = call.thread_number;
	r.start_ns      = NowNs();
	if (call.game_thread && call.call_ns != 0) {
		r.queue_ns           = std::max<int64_t>(r.start_ns - call.call_ns, 0);
		const auto waited    = GpuThreadWaitNs();
		r.behind_ns          = std::min<int64_t>(r.queue_ns, waited >= call.wait_snapshot
		                                                        ? static_cast<int64_t>(waited - call.wait_snapshot)
		                                                        : 0);
	}
	NeedRange(vaddr, size, r.need_lo, r.need_hi);
	r.page_lo = r.need_lo & ~uint64_t {0xfff};
	r.page_hi = (r.need_hi + 0xfffu) & ~uint64_t {0xfff};
	t_requests.push_back(r);
}

RequestScope::~RequestScope() {
	if (!m_on) return;
	const Request r = t_requests.back();
	t_requests.pop_back();
	const auto reason  = static_cast<uint32_t>(r.reason);
	const auto verdict = static_cast<uint32_t>(Classify(r));
	if (r.game_thread) {
		Add(kGameN, 1);
		Add(kGameQueueNs, static_cast<uint64_t>(r.queue_ns));
		Add(kGameBehindNs, static_cast<uint64_t>(r.behind_ns));
		if (r.queue_ns >= 1000000) Add(kGameSlowN, 1);
	}
	Add(kDlBytes, r.dl_bytes);
	Add(kDlChanged, r.dl_changed);
	if (r.waits != 0) {
		Add(kReasonN + reason, 1);
		Add(kReasonNs + reason, static_cast<uint64_t>(r.wait_ns));
		Add(kVerdictN + verdict, 1);
		Add(kVerdictNs + verdict, static_cast<uint64_t>(r.wait_ns));
		if (r.need_done_ns != 0) {
			Add(kNeedDoneN, 1);
			Add(kNeedDoneNs, static_cast<uint64_t>(r.need_done_ns));
		}
		if (r.granule_ns != 0) {
			Add(kGranuleN, 1);
			Add(kGranuleNs, static_cast<uint64_t>(r.granule_ns));
		}
		if (r.beyond_ns != 0) {
			Add(kBeyondN, 1);
			Add(kBeyondNs, static_cast<uint64_t>(r.beyond_ns));
		}
	}
	if (r.waits == 0 && r.queue_ns < 1000000) return;
	const auto frame = DebugPresentCount().load(std::memory_order_relaxed);
	if (!DetailFrame(frame)) return;
	const auto ull = [](uint64_t value) { return static_cast<unsigned long long>(value); };
	std::fprintf(stderr,
	             "SYNC_TL frame=%u t=%.2f thr=gpu reason=%s rw=%c addr=0x%llx size=0x%llx page=0x%llx from=%s#%u "
	             "queue_us=%lld behind_wait_us=%lld waits=%u wait_us=%lld site=%s open=%d tick=%llu cur=%llu "
	             "pending=%llu backlog_ms=%.2f age_ms=%.2f need=0x%llx+0x%llx need_tick=%llu exact_tick=%llu "
	             "need_done_us=%lld granule_us=%lld beyond_us=%lld "
	             "need_dirty=%llu page_dirty=%llu dl=%llu dl_changed=%llu page_changed=%llu need_changed=%llu "
	             "compare_fails=%llu verdict=%s\n",
	             frame, FrameMs(r.start_ns), kReasonNames[reason], r.is_write ? 'w' : 'r', ull(r.vaddr), ull(r.size),
	             ull(r.vaddr & ~uint64_t {0xfff}), r.game_thread ? "game" : "gpu", r.thread_number,
	             static_cast<long long>(r.queue_ns / 1000), static_cast<long long>(r.behind_ns / 1000), r.waits,
	             static_cast<long long>(r.wait_ns / 1000), kSiteNames[static_cast<uint32_t>(r.site)], r.open ? 1 : 0,
	             ull(r.waited_tick), ull(r.current_tick), ull(r.pending_ticks), Ms(r.backlog_ns), Ms(r.age_ns),
	             ull(r.need_lo), ull(r.need_hi - r.need_lo), ull(r.need_tick), ull(r.exact_tick),
	             static_cast<long long>(r.need_done_ns / 1000), static_cast<long long>(r.granule_ns / 1000),
	             static_cast<long long>(r.beyond_ns / 1000),
	             ull(r.need_dirty), ull(r.page_dirty), ull(r.dl_bytes), ull(r.dl_changed), ull(r.page_changed),
	             ull(r.need_changed), ull(r.compare_fails), kVerdictNames[verdict]);
}

void SetNeed(uint64_t granule_tick, uint64_t exact_tick, uint64_t need_dirty, uint64_t page_dirty) {
	if (!Enabled()) return;
	auto* r = CurrentRequest();
	if (r == nullptr) return;
	r->need_known = true;
	r->need_tick  = granule_tick;
	r->exact_tick = exact_tick;
	r->need_dirty = need_dirty;
	r->page_dirty = page_dirty;
}

void NoteDownload(uint64_t vaddr, const void* data, uint64_t size) {
	if (!Enabled()) return;
	auto* r = CurrentRequest();
	if (r == nullptr || size == 0 || data == nullptr) return;
	r->dl_bytes += size;
	thread_local std::vector<uint8_t> guest;
	constexpr uint64_t                Chunk = 256u * 1024u;
	const auto*                       bytes = static_cast<const uint8_t*>(data);
	for (uint64_t done = 0; done < size; done += Chunk) {
		const uint64_t n = std::min(size - done, Chunk);
		if (guest.size() < n) guest.resize(n);
		if (!Libs::LibKernel::Memory::TryReadBacking(vaddr + done, guest.data(), n)) {
			r->compare_fails += n;
			continue;
		}
		if (std::memcmp(guest.data(), bytes + done, n) == 0) continue;
		for (uint64_t i = 0; i < n; i += 8u) {
			const uint64_t w = std::min<uint64_t>(8u, n - i);
			if (std::memcmp(guest.data() + i, bytes + done + i, w) == 0) continue;
			const uint64_t address = vaddr + done + i;
			r->dl_changed += w;
			if (address < r->page_hi && r->page_lo < address + w) r->page_changed += w;
			if (address < r->need_hi && r->need_lo < address + w) r->need_changed += w;
		}
	}
}

void GuestLabel(uint64_t addr, uint64_t value, bool compute_queue, uint64_t current_tick, uint64_t known_tick) {
	if (!Enabled()) return;
	Add(kLabels, 1);
	const auto frame = DebugPresentCount().load(std::memory_order_relaxed);
	if (!DetailFrame(frame)) return;
	std::fprintf(stderr, "SYNC_TL_GUEST frame=%u t=%.2f kind=label queue=%s addr=0x%llx value=0x%llx cur=%llu gpu_tick=%llu\n",
	             frame, FrameMs(NowNs()), compute_queue ? "compute" : "gfx", static_cast<unsigned long long>(addr),
	             static_cast<unsigned long long>(value), static_cast<unsigned long long>(current_tick),
	             static_cast<unsigned long long>(known_tick));
}

void GuestWaitRegMem(uint64_t addr, uint64_t value, uint64_t ref, bool passed, uint32_t queue) {
	if (!Enabled()) return;
	// GPU thread only: the first block of each address until it passes.
	static std::unordered_map<uint64_t, int64_t> blocked;
	const auto                                   it = blocked.find(addr);
	if (it == blocked.end()) Add(kWrm, 1);
	// The waiting queue as "gfx" or "cqNN" (guest queue id), for the detail lines.
	const auto queue_name = [queue] {
		std::array<char, 8> name {};
		if (queue == 0) {
			std::snprintf(name.data(), name.size(), "gfx");
		} else {
			std::snprintf(name.data(), name.size(), "cq%02x", queue);
		}
		return name;
	};
	if (!passed) {
		if (it != blocked.end()) return;
		const auto now = NowNs();
		blocked.emplace(addr, now);
		Add(kWrmBlocks, 1);
		const auto frame = DebugPresentCount().load(std::memory_order_relaxed);
		if (DetailFrame(frame)) {
			std::fprintf(stderr,
			             "SYNC_TL_GUEST frame=%u t=%.2f kind=wrm_block addr=0x%llx value=0x%llx ref=0x%llx q=%s "
			             "marker=[%s]\n",
			             frame, FrameMs(now), static_cast<unsigned long long>(addr), static_cast<unsigned long long>(value),
			             static_cast<unsigned long long>(ref), queue_name().data(), DebugMarkerPath().c_str());
		}
		return;
	}
	if (it == blocked.end()) return;
	const auto now = NowNs();
	const auto ns  = now - it->second;
	blocked.erase(it);
	Add(kWrmBlockedNs, static_cast<uint64_t>(ns));
	const auto frame = DebugPresentCount().load(std::memory_order_relaxed);
	if (DetailFrame(frame)) {
		std::fprintf(stderr,
		             "SYNC_TL_GUEST frame=%u t=%.2f kind=wrm_pass addr=0x%llx value=0x%llx ref=0x%llx blocked_ms=%.2f q=%s "
		             "marker=[%s]\n",
		             frame, FrameMs(now), static_cast<unsigned long long>(addr), static_cast<unsigned long long>(value),
		             static_cast<unsigned long long>(ref), Ms(ns), queue_name().data(), DebugMarkerPath().c_str());
	}
}

void Report() {
	if (!Enabled()) return;
	// GPU thread only.
	static int64_t window_start = NowNs();
	static Values  window {};
	const auto     now = NowNs();
	if (now - window_start < 5'000'000'000ll) return;
	const auto values = Load();
	char       prefix[64];
	std::snprintf(prefix, sizeof(prefix), "PERF sync_timeline window_ms=%lld",
	              static_cast<long long>((now - window_start) / 1000000));
	PrintTotals(prefix, Delta(values, window));
	window       = values;
	window_start = now;
}

} // namespace Libs::Graphics::SyncTimeline
