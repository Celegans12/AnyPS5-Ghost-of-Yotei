#ifndef GRAPHICS_HOST_GPU_RENDERER_DISPATCH_CENSUS_H
#define GRAPHICS_HOST_GPU_RENDERER_DISPATCH_CENSUS_H

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

// KYTY_TRACE_DISPATCH_CENSUS: nanoseconds of the parts of one dispatch (DispatchCensusSub), summed by the census.
enum class DispatchSub : uint32_t {
	PrepareProgram, KeyFind, Adopt, Permutation, FindBuffers, PrepareBda, RebindImages, SharedMemory, RebindBuffers,
	Pipeline, PrepareBindings, CommitBindings, Record, Reset, AdoptReads, BdaRanges, ObtainWritten, ObtainRead,
	TcInvalidate, SrtUpload, DataUpload, OwBatchCommit, OwUploadWalk, OwDirtyAdd, OwObserver, OwWriteTick,
	TailPre, TailPost, TailEmitted, BindlessHeaps,
	// KYTY_TRACE_DRAW_CENSUS (draws on the execution thread, renderDraw.cpp).
	DrawPre, DrawRenderState, DrawPrograms, DrawTargets, DrawPipeline, DrawPlanSetup, DrawWait, DrawPost, DrawCommit,
	CReplay, CAcquire, CVtx, CBind, CIndex, CDyn,
	// CommitBindings parts (the record split) and the graphics program lookup (ProgramCache::Get of non-compute
	// stages: key, resolved result taken, materialization, permutation; counts of each path).
	BPre, BScan, BDesc, BTransit, BEmit, GLookup, GTake, GMr, GPerm, GPathResolved, GPathMr, Count
};
inline thread_local bool                                                     t_dispatch_sub_on = false;
inline thread_local std::array<uint64_t, static_cast<size_t>(DispatchSub::Count) + 1u> t_dispatch_sub_ns {}; // [Count]: unused
class DispatchSubTimer {
public:
	explicit DispatchSubTimer(DispatchSub sub): m_sub(sub), m_on(t_dispatch_sub_on) {
		if (m_on) m_start = std::chrono::steady_clock::now();
	}
	~DispatchSubTimer() {
		if (m_on) {
			t_dispatch_sub_ns[static_cast<size_t>(m_sub)] += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count());
		}
	}
	DispatchSubTimer(const DispatchSubTimer&)            = delete;
	DispatchSubTimer& operator=(const DispatchSubTimer&) = delete;

private:
	DispatchSub                           m_sub;
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace Libs::Graphics

#endif
