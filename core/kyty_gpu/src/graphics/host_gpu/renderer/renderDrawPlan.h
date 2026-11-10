#pragma once

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

// One draw on its way through RenderExecutor: the types of renderDraw.cpp that renderDrawTrace.cpp reads too.
namespace Libs::Graphics {

// S2 step B.6b: the registers a draw's commit reads, captured where the draw is prepared (the commit may follow the next
// draw's register writes). Captured at the commit itself when the draw is not pipelined.
struct DrawDynamicRegs {
	HW::ScreenViewport  viewport;
	HW::ClipControl     clip;
	HW::ScanModeControl scan;
	HW::BlendColor      blend;
	HW::ModeControl     mode;
	HW::PolyOffset      poly;
	float               line_width   = 1.0f;
	float               bounds_min   = 0.0f;
	float               bounds_max   = 0.0f;
	uint32_t            rt_mask      = 0;
	uint32_t            object_id    = 0;
	decltype(std::declval<const HW::UserConfig&>().GetPrimType()) prim_type {};
	// Upstream feed075a9: PA_CL_UCP_0..5 (copied only while PA_CL_CLIP_CNTL enables a user plane).
	std::array<uint32_t, 24> user_clip_planes {};

	void Capture(const CommandBuffer& buffer) {
		const auto& ctx = buffer.GetRegisters();
		viewport        = ctx.GetScreenViewport();
		clip            = ctx.GetClipControl();
		if (clip.user_clip_planes != 0) user_clip_planes = ctx.GetUserClipPlanes();
		scan            = ctx.GetScanModeControl();
		blend           = ctx.GetBlendColor();
		mode            = ctx.GetModeControl();
		poly            = ctx.GetPolyOffset();
		line_width      = ctx.GetLineWidth();
		bounds_min      = ctx.GetDepthBoundsMin();
		bounds_max      = ctx.GetDepthBoundsMax();
		rt_mask         = ctx.GetRenderTargetMask();
		object_id       = buffer.GetUserConfig().GetObjectId();
		prim_type       = buffer.GetUserConfig().GetPrimType();
	}
};

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
	ShaderIndirectDrawInfo indirect_draw; // GPU indirect vertex variant
	// PrepareDrawRenderState: why a draw without any output is left out (null: the generic reason).
	const char*           drop_reason = nullptr;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
	bool     cpu_indirect   = false; // the CPU expansion of an indirect draw
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

// GPU indirect draws: the guest ranges a GPU-executed indirect draw reads its argument
// records and draw count from.
struct DrawIndirectSource {
	uint64_t args_addr  = 0;
	uint64_t args_size  = 0; // bytes of max_count records
	uint64_t count_addr = 0; // 0: exactly max_count draws
	uint32_t max_count  = 0;
	uint32_t stride     = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	// AcquireVertexBuffers sets the first `count` entries; CommitVertexBuffers reads only those.
	std::array<vk::Buffer, MaxBuffers>     buffers;
	std::array<vk::DeviceSize, MaxBuffers> offsets;
	std::array<vk::DeviceSize, MaxBuffers> sizes;
	uint32_t                               count = 0;
};

// S1: what the preparation of a draw (ExecutePreparedDraw up to its
// "cannot touch guest memory" point) hands to its commit (CommitPreparedDraw). Step 1 only moves the locals here; the
// preparation still runs right before the commit, on the same thread.
struct DrawPlan {
	// The commit's inputs (copies: the commit may run after the draw's entry function returned).
	uint64_t                             submit_id = 0;
	CommandBuffer*                       buffer    = nullptr;
	DrawCallInfo                         draw;
	DrawRenderState*                     state     = nullptr;
	DrawEmitInfo                         emit;
	DrawIndexBufferSource                index_source;
	std::optional<DrawIndirectSource>    indirect;
	DrawDynamicRegs                      regs;           // S2 step B.6b: the registers its commit reads
	std::span<ShaderVertexInputInfo>     vertex_stages;
	bool                                 mesh_active = false;
	uint32_t                             mesh_groups = 0;
	std::array<PreparedBindings*, 4>     descriptor_stages {};
	uint32_t                             stage_count = 0;
	PreparedVertexBuffers                vertex_bindings;
	PreparedIndexBuffer                  index_binding;
	std::pair<Buffer*, uint64_t>         indirect_args {};
	std::pair<Buffer*, uint64_t>         indirect_count {};
	const PipelineCache::Pipeline*       pipeline = nullptr;
	RenderExecutor::GraphicsBindings*    bindings = nullptr; // the stage bindings the preparation filled
	size_t                               bindless_heaps_end  = SIZE_MAX; // S2 step B.4f: its bindless in-draw marks
	size_t                               bindless_tables_end = SIZE_MAX;
	// KYTY_TRACE_CLEAN_DRAWS (stage C0): the changes its plan found against the previous draw, its GPU thread times once
	// its front ended (the front without the worker wait, the wait) and its preparation job's time on the worker.
	uint64_t                             c0_mask     = 0;
	bool                                 c0_front    = false;
	uint64_t                             c0_front_ns = 0;
	uint64_t                             c0_wait_ns  = 0;
	uint64_t                             c0_prep_ns  = 0;
	uint32_t                             c0_cause    = 0; // the observer's cause of dirt (t_draw_observer_cause)
	uint64_t                             c0_programs_ns = 0, c0_targets_ns = 0, c0_pipeline_ns = 0; // CleanDraws::t_front_parts
	bool                                 c0_target_hit  = false;
};

} // namespace Libs::Graphics
