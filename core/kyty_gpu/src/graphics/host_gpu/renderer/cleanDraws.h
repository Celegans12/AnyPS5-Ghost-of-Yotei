#ifndef GRAPHICS_HOST_GPU_RENDERER_CLEAN_DRAWS_H
#define GRAPHICS_HOST_GPU_RENDERER_CLEAN_DRAWS_H

#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::CleanDraws {

// KYTY_TRACE_CLEAN_DRAWS=1 (project C, stage C0; diagnostic, default off, measurement only): for each draw, whether it
// is "clean" against the previous one (the same programs, pipeline, render targets, dynamic state and image / sampler
// set; only the user data, index and geometry addresses may differ), the reasons of the others, the lengths of clean
// series and the GPU thread's / worker's time on each kind. One PERF clean_draws line every ~5 s (renderDrawTrace.cpp).
inline bool On() {
	static const bool on = std::getenv("KYTY_TRACE_CLEAN_DRAWS") != nullptr;
	return on;
}

// What came between two draws and ends a series whatever their state (set by the dispatch and draw paths, taken by
// the next draw's plan).
inline constexpr uint32_t BreakDispatch = 1u; // a dispatch on the graphics queue
inline constexpr uint32_t BreakMetaOp   = 2u; // a metadata / copy / resolve operation dropped as a draw
inline constexpr uint32_t BreakAsync    = 4u; // a dispatch of an asynchronous compute queue
inline std::atomic<uint32_t> g_breaks {0};
// Metadata / copy / resolve operations so far (a draw's target memo, renderDraw.cpp).
inline std::atomic<uint64_t> g_meta_ops {0};
inline void NoteBreak(uint32_t bit) {
	if (On()) g_breaks.fetch_or(bit, std::memory_order_relaxed);
}

// Parts of the GPU thread's draw front (active project step K1; C0 in renderDrawTrace.cpp takes them per kind at the draw's
// plan): its program lookups (RefreshShaders), its render target search and its pipeline lookup (without the program
// lookups inside it).
struct FrontParts {
	uint64_t programs_ns = 0;
	uint64_t targets_ns  = 0;
	uint64_t pipeline_ns = 0;
	bool     target_hit  = false; // the target search took the previous draw's targets
};
inline thread_local FrontParts t_front_parts;

} // namespace Libs::Graphics::CleanDraws

#endif
