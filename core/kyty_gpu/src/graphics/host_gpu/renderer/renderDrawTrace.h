#pragma once

#include "graphics/host_gpu/renderer/dispatchCensus.h"
#include "graphics/host_gpu/renderer/render.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>

// The draw paths' logs and censuses (renderDrawTrace.cpp; renderDraw.cpp calls them). Every one is off unless its name
// or the debug dump is set; each is reached through its gate or on a draw that is dropped anyway.
namespace Libs::Graphics {

// A draw the renderer drops without recording anything (KYTY_TRACE_CLEAN_DRAWS: a metadata / copy / resolve operation
// ends a series of clean draws).
void TraceDrawDrop(const char* reason, const CommandBuffer& buffer);

// --graphics-debug-dump: the first draws' targets, inputs and skipped framebuffers.
void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color, const RenderDepthInfo& depth,
                        const CommandBuffer& buffer, uint32_t index_count, uint32_t flags);
void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color, const RenderDepthInfo& depth,
                        const CommandBuffer& buffer, const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                        uint32_t flags);
void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color, const ShaderVertexInputInfo& vs_input_info,
                       uint32_t index_type_and_size, uint32_t index_count, const void* index_addr);

// KYTY_TRACE_DRAW_STATE_PS=<ps hash>[,...][:<frame>] (diagnostic): the full output state of the listed pixel shaders' draws
// (first 12 from the presented frame on): targets and masks, blend, depth/stencil, cull, viewport, scissor, counts.
void TraceDrawStatePs(const CommandBuffer& buffer, const DrawRenderState& state, const DrawCallInfo& draw);

// KYTY_TRACE_DRAW_CENSUS=1 (diagnostic, default off): the execution thread's time per draw (DrawIndex, DrawAuto,
// GPU indirect draws called by FlushRunQueue) and its parts: the front (entry checks, programs, targets, pipeline, plan
// setup), the wait for the worker's previous job, the post of this one, and the previous draw's commit with its parts
// (deferred commands and plan flags, targets, vertex buffers, bindings, index buffer, dynamic state; the rest is the
// recording). One PERF draw_census line every ~5 s, us per draw.
inline bool DrawCensusOn() {
	static const bool on = std::getenv("KYTY_TRACE_DRAW_CENSUS") != nullptr;
	return on;
}
struct DrawCensusScope {
	bool                                  on = DrawCensusOn();
	std::chrono::steady_clock::time_point start {};
	DrawCensusScope() {
		if (!on) return;
		t_dispatch_sub_ns = {};
		t_dispatch_sub_on = true;
		start             = std::chrono::steady_clock::now();
	}
	~DrawCensusScope() {
		if (on) Finish();
	}

private:
	void Finish();
};

// KYTY_TRACE_CLEAN_DRAWS=1 (project C, stage C0; diagnostic, default off, measurement only; cleanDraws.h): a draw is
// clean when, against the previous draw, nothing came between them (graphics dispatch, metadata operation, another
// command buffer) and its programs, pipeline, render targets, dynamic state, images and samplers are the same; the user
// data, index and geometry addresses and buffers may differ. Its plan compares the registers (by group) and the
// programs and pipeline, its commit the prepared images, samplers and buffers. "Strict": no register group changed
// either (what a check of the registers alone would see). Every ~5 s one PERF clean_draws line.
namespace C0 {
// Kinds for the times: clean, clean but for the images / samplers, the rest.
enum Kind : uint8_t { KClean, KImagesOnly, KRest, KindCount };

// The GPU thread's draw inside FlushRunQueue: its plan (null once committed inside its front), its kind if so, and the
// time of the commits and worker waits inside its front (taken out of the front's time).
struct Current {
	DrawPlan* plan      = nullptr;
	bool      committed = false;
	uint8_t   kind      = KRest;
	uint64_t  commit_ns = 0;
	uint64_t  wait_ns   = 0;
};
inline thread_local Current t_current;

inline uint64_t Ns(std::chrono::steady_clock::duration d) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count());
}

void    Plan(DrawPlan& plan, const CommandBuffer& buffer, const DrawRenderState& state,
             std::span<ShaderVertexInputInfo> vertex_stages, const PipelineCache::Pipeline* pipeline);
uint8_t Commit(DrawPlan& plan, std::span<PreparedBindings* const> stages, vk::CommandBuffer handle);
void    FrontEnd(std::chrono::steady_clock::time_point start);

// The commit's time, to its kind (and out of the front it ran in).
struct CommitScope {
	uint8_t                               kind;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	explicit CommitScope(uint8_t k): kind(k) {}
	~CommitScope();
	CommitScope(const CommitScope&)            = delete;
	CommitScope& operator=(const CommitScope&) = delete;
};
} // namespace C0

} // namespace Libs::Graphics
