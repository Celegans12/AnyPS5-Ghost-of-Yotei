#ifndef KYTY_GRAPHICS_SYNC_TIMELINE_H_
#define KYTY_GRAPHICS_SYNC_TIMELINE_H_

#include "graphics/host_gpu/renderer/perfCounters.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {
class MasterSemaphore;
} // namespace Libs::Graphics

namespace Libs::Graphics::SyncTimeline {

// KYTY_TRACE_SYNC_TIMELINE=1 (diagnostic, default off; one static load per hook when off): every wait for the host
// GPU is attributed. Waits: MasterSemaphore::Wait calls that block, readback-queue and OOB fences, the GPU thread
// waiting for priority completions, WAIT_FLIP_DONE, game threads at a SuspendPoint. For each wait it records:
//   - the thread (gpu / prio / other);
//   - the site (copy / bar / eager / finish / unmap / ...);
//   - for BufferCache::ReadMemory, the reader (fault_game, invalidate, srt_walker, dcc_meta, ...) and the guest
//     address;
//   - the GPU work pending when the wait began (submissions not finished, age of the oldest one);
//   - whether the newest write of the needed bytes had already finished when the wait began (need_done: the wait
//     was for other work; granule: only the 64 KiB write-tick granule was unfinished) or the wait went beyond it
//     (beyond: it also waited for later submissions).
// Whether the GPU really changed the bytes the waiter needed is found by comparing each synchronous download with
// guest memory before the publish:
//   real   = the needed bytes changed;
//   page   = only other bytes of their 4 KiB pages changed (false sharing in a page);
//   window = only other downloaded pages changed (the read window);
//   same   = nothing downloaded differs (the bytes were only bound for writing, or rewritten unchanged);
//   nodl   = no download was compared (eager copy, priority completions, no GPU bytes).
// A fault reports only the faulting byte, so for faults the needed bytes are its 64-byte line. Changed bytes are
// counted in 8-byte words.
// Every ~5 s one "PERF sync_timeline" line. For 3 frames, detail lines: SYNC_TL (one per reader request or wait),
// SYNC_TL_GUEST (label writes, WAIT_REG_MEM blocks) and SYNC_TL_FRAME (per-frame totals). The detail frames are
// n+1..n+3 with KYTY_TRACE_SYNC_TIMELINE_FRAME=<n>, else the first 3 frames of KYTY_TRACE_FRAME_RANGE.
// KYTY_TRACE_SYNC_TIMELINE_FRAMES=<k> changes the 3.
inline bool Enabled() {
	static const bool on = [] {
		const char* value   = std::getenv("KYTY_TRACE_SYNC_TIMELINE");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) std::fprintf(stderr, "TRACE_SYNC_TIMELINE enabled\n");
		return enabled;
	}();
	return on;
}

enum class Site : uint32_t {
	None,          // no site tag: CommandScheduler::Wait from elsewhere (the caller is logged)
	Copy,          // DownloadBufferMemory: copy recorded in the open command buffer, submit + wait for all recorded work
	Bar,           // TryBarDownload: wait for the newest write tick of the bytes, then read the host mapping
	Eager,         // ApplyEagerReadbacks: a pending asynchronous copy of the window must publish first
	PriorityOps,   // the GPU thread waiting for priority completions (asynchronous publishes) up to a tick
	Finish,        // CommandScheduler::Finish (GDS end-of-pipe read, shutdown, ...)
	Unmap,         // RenderContext::UnmapMemory: finish all GPU work before cached memory is unmapped
	StreamWrap,    // a ring buffer waiting for the GPU to release space
	Gc,            // BufferCache garbage collector publishing GPU-dirty buffers
	FaultBuffer,   // the BDA fault buffer area being reused
	FlipDone,      // WAIT_FLIP_DONE: the GPU thread waiting for the presenter
	Suspend,       // a game thread at a SuspendPoint waiting for the previous frame's GPU work
	Caller,        // a faulting thread waiting for the GPU itself
	SrtExact,      // (the exact walker read is gone; the site's number stays)
	Count
};

// Submission ticks and present times.
void NoteSubmit(uint64_t tick);
void NoteFrame(uint32_t frame);

// The current thread is the priority-completion thread (CommandScheduler::PriorityOperationsThread).
void MarkPriorityThread();

// Total time the GPU thread spent in attributed waits so far (ns).
[[nodiscard]] uint64_t GpuThreadWaitNs();
// A small number for the calling thread (first use order).
[[nodiscard]] uint32_t ThreadNumber();
[[nodiscard]] int64_t  NowNs();

// CommandScheduler::Wait: its caller's return address (0 when it returns; also clears the open flag), and that the
// waited tick was the open command buffer (submitted by the wait).
void NoteOpenFlush();
void NoteCaller(uint64_t return_address);

// The site of the waits in this scope (an outer tag wins).
class SiteScope {
public:
	explicit SiteScope(Site site);
	~SiteScope();
	SiteScope(const SiteScope&)            = delete;
	SiteScope& operator=(const SiteScope&) = delete;

private:
	bool m_set = false;
};

// One wait. site None: the innermost SiteScope tag. tick 0: not a timeline wait.
class WaitScope {
public:
	WaitScope(Site site, uint64_t tick, uint64_t current_tick, uint64_t known_tick);
	~WaitScope();
	WaitScope(const WaitScope&)            = delete;
	WaitScope& operator=(const WaitScope&) = delete;

private:
	bool     m_on       = false;
	Site     m_site     = Site::None;
	uint64_t m_tick     = 0;
	uint64_t m_current  = 0;
	uint64_t m_known    = 0;
	int64_t  m_start_ns = 0;
};

// A reader request as sent by the requesting thread (BufferCache::ReadMemory); zero when the trace is off.
struct CallInfo {
	int64_t  call_ns       = 0;     // NowNs() when the command was sent
	uint64_t wait_snapshot = 0;     // GpuThreadWaitNs() then
	uint32_t thread_number = 0;
	bool     game_thread   = false; // sent by another thread than the GPU thread
};
// Only called when Enabled().
[[nodiscard]] CallInfo Call(bool gpu_thread);

// The bytes a request needs: its range, or for a fault (size <= 1, the faulting byte) its 64-byte line.
inline void NeedRange(uint64_t vaddr, uint64_t size, uint64_t& lo, uint64_t& hi) {
	if (size <= 1u) {
		lo = vaddr & ~uint64_t {63};
		hi = lo + 64u;
	} else {
		lo = vaddr;
		hi = vaddr + size;
	}
}

// A BufferCache::ReadMemory command running on the GPU thread.
class RequestScope {
public:
	RequestScope(PerfReason reason, uint64_t vaddr, uint64_t size, bool is_write, const CallInfo& call);
	~RequestScope();
	RequestScope(const RequestScope&)            = delete;
	RequestScope& operator=(const RequestScope&) = delete;

	[[nodiscard]] bool Active() const noexcept { return m_on; }

private:
	bool m_on = false;
};

// For the active request: the newest write tick of the GPU-dirty bytes of its NeedRange by 64 KiB granule (what a
// host-mapped read waits for) and by written range (exact), 0 when unknown, and the GPU-dirty bytes (bound for
// writing) in the range and in its 4 KiB pages.
void SetNeed(uint64_t granule_tick, uint64_t exact_tick, uint64_t need_dirty, uint64_t page_dirty);

// A synchronous download about to be published to guest memory: compared with guest memory (active request only).
void NoteDownload(uint64_t vaddr, const void* data, uint64_t size);

// Guest synchronization seen by the command processor (GPU thread): end-of-pipe label writes (written when the
// packet is translated, with the open and the finished tick) and WAIT_REG_MEM tests (the first block and the pass
// after it; the waiting queue, 0 = graphics, and its innermost debug markers).
void GuestLabel(uint64_t addr, uint64_t value, bool compute_queue, uint64_t current_tick, uint64_t known_tick);
void GuestWaitRegMem(uint64_t addr, uint64_t value, uint64_t ref, bool passed, uint32_t queue);

// The GPU thread's ~5 s PERF sync_timeline line.
void Report();

} // namespace Libs::Graphics::SyncTimeline

#endif
