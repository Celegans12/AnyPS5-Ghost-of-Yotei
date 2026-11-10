#ifndef KYTY_COMMON_LIVEAB_H_
#define KYTY_COMMON_LIVEAB_H_

#include <atomic>
#include <cstdint>

// KYTY_LIVE_AB=<SWITCH>[=<value>],<SWITCH>,... (measurement, default off): same-process A/B of CPU-side performance
// switches. From presented frame KYTY_LIVE_AB_START_FRAME (default 1100) the GPU thread rotates phases at frame
// boundaries (after the flip's submission): phase "off" (every listed switch off), then one phase per listed switch
// (that switch at its value, default 1, the others off), and with KYTY_LIVE_AB_ALL=1 a phase "all" (every listed
// switch on). Each phase lasts until the first frame boundary KYTY_LIVE_AB_PERIOD_MS (default 5000) after its start;
// its first KYTY_LIVE_AB_WARMUP_FRAMES (default 1) frames are run but not measured. KYTY_LIVE_AB_ORDER=rotate keeps
// the order of every cycle; the default (alternate) runs every second cycle backwards, so a slow drift of the scene
// cancels in the per-cycle comparisons. Each phase prints
//   PERF live_ab phase=<name> cycle=<n> frames=<n> ms=<wall ms> fps=<..> gpu_thread_busy_ms=<..> fence_wait_ms=<..>
//        ft_mean_ms=<..> ft_median_ms=<..> frame=<last presented frame>
// and every KYTY_LIVE_AB_SUMMARY_CYCLES (default 2) complete cycles a summary of all cycles so far, per phase: mean fps
// over cycles and its standard deviation, pooled fps, frame time mean and median, GPU-thread busy and fence-wait time
// per frame, and the per-cycle fps difference to the "off" phase of the same cycle (mean and standard error).
//
// Only switches whose effect is a per-call choice with no state that outlives the call (or a pure memo whose entries
// stay valid whether it is used or not) toggle live; they are listed in liveAb.cpp with the reason each switch is
// exact at any frame boundary. Every other name is refused with a warning and keeps its start value. Before the start
// frame, and for every switch not listed, the gates return their usual once-read values. Unset: a gate costs one
// relaxed load, and the GPU thread checks a once-read flag per work slice, fence wait and flip; nothing else runs.

namespace Common::LiveAb {

enum class Switch : uint32_t {
	DirectRanges,          // KYTY_PERF_DIRECT_RANGES
	SkipCsLive,            // KYTY_DIAG_SKIP_CS_LIVE (measurement only)
	GpuBurn,               // KYTY_DIAG_GPU_BURN_NS (measurement only)
	IndirectLoadFast,      // KYTY_PERF_INDIRECT_LOAD_FAST (=3: shader-side, SpirvLive flags)
	SharedImageMemory,     // KYTY_FIX_SHARED_IMAGE_MEMORY (0/1 per bind; =2, the check, only as the start value; D4 K1)
	DiagCeilingFreePasses,     // KYTY_DIAG_CEILING_FREE_PASSES (measurement only, APPROXIMATION)
	Count
};

inline constexpr uint32_t SwitchCount = static_cast<uint32_t>(Switch::Count);
static_assert(SwitchCount <= 192, "the listed-switch mask is 192 bits");

// The live value of each switch plus one; 0 while it is not toggled live. Only the GPU thread writes them, at frame
// boundaries; any thread reads them (relaxed: no other data is published with a value).
inline constinit std::atomic<uint32_t> g_live_values[SwitchCount] {};

// A switch gate's value: the live value while KYTY_LIVE_AB toggles the switch, else `fixed` (the gate's once-read
// environment value). Callers read it once per call (per request, per draw) and use that value for the whole call.
template <typename T>
[[nodiscard]] inline T Gate(Switch id, T fixed) noexcept {
	const uint32_t stored = g_live_values[static_cast<uint32_t>(id)].load(std::memory_order_relaxed);
	if (stored == 0) [[likely]] {
		return fixed;
	}
	return static_cast<T>(stored - 1u);
}

// KYTY_LIVE_AB is set and names at least one switch it accepted (parsed once, on the first call; warnings then).
[[nodiscard]] bool Enabled();
// KYTY_LIVE_AB accepted this switch: it will be toggled live from the start frame on.
[[nodiscard]] bool Listed(Switch id);
// Whether a switch can be on at any time of this run: its once-read value is non-zero or KYTY_LIVE_AB lists it. Code a
// switch adds to a shared path (extra copies, lists, shared writes) runs only then; otherwise the path is the one before
// the switch (callers keep the answer once per run).
[[nodiscard]] inline bool Possible(Switch id, uint32_t fixed) {
	return fixed != 0u || Listed(id);
}

// GPU thread, after each flip's submission: `frame` is the presented-frame counter (window title numbering). Phases
// start, end and switch only here.
void OnFrameBoundary(uint32_t frame);

// GPU thread: time spent fence-waiting for the GPU (counted inside the busy time too).
void AddFenceWait(uint64_t nanoseconds);

// GPU thread: one slice of work (a submission or a command), counted as busy time of the phase it runs in. A frame
// boundary inside the slice splits it between the two phases.
class BusyScope {
public:
	BusyScope();
	~BusyScope();
	BusyScope(const BusyScope&)            = delete;
	BusyScope& operator=(const BusyScope&) = delete;

private:
	bool m_on = false;
};

} // namespace Common::LiveAb

#endif // KYTY_COMMON_LIVEAB_H_
