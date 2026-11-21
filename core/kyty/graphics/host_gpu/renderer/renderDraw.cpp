#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/host_gpu/renderer/cxCompat.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/spscQueue.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawTemplates.h"
#include "graphics/host_gpu/renderer/gpuTime.h"
#include "graphics/host_gpu/renderer/meshDrawSplit.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "loader/systemContent.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <immintrin.h>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <xxhash.h>
#include <new>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <string_view>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {
uint32_t TableXprMode();
}

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderDepthInfo& depth,
                               const CommandBuffer& buffer, uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF("DrawFramebufferSkip[%u]: %s depth_format=%s depth_image=%s depth_vaddr_num=%d "
	     "target_mask=0x%08" PRIx32 " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	     log_id, draw_name, vk::to_string(depth.desc.view_info.format).c_str(),
	     depth.image_id ? "yes" : "no",
	     static_cast<int>(!depth.desc.info.data.Empty()) +
	         static_cast<int>(depth.desc.info.HasStencil()),
	     ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.module->info.info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}
}

static void PrepareDrawRayTracing(RenderContext& context, CommandBuffer& buffer, const PreparedBindings& bindings) {
	auto* native_rt = context.GetRayTracing();
	if (native_rt == nullptr || !native_rt->Enabled() || bindings.runtime == nullptr || bindings.runtime->module == nullptr) {
		return;
	}
	const auto& module = *bindings.runtime->module;
	const auto& info   = module.info.info;
	RayTracing::TlasSource source;
	if (!info.bvh || !RayTracing::FindTlasSource(info, source) || source.user_sgpr + 1 >= HW::UserSgprInfo::SGPRS_MAX) {
		return;
	}
	const auto& user    = buffer.GetShaders().GetPs().ps_user_sgpr.value;
	const auto  pointer = RayTracing::UserPointer(
	    source, uint64_t {user[source.user_sgpr]} | (uint64_t {user[source.user_sgpr + 1]} << 32u),
	    [&](uint64_t address, uint64_t size) { return context.IsMapped(address, size); });
	const auto frame = static_cast<uint64_t>(context.GetGpu().GetFrameNum());
	RayTracing::Preparation preparation {.tick          = context.GetCommandScheduler().CurrentTick(),
	                                     .frame         = frame,
	                                     .user_pointer  = pointer,
	                                     .generation    = native_rt->Generation(),
	                                     .header_offset = source.header_offset,
	                                     .root          = source.root,
	                                     .layout        = source.layout};
	bool known = true;
	if ((source.layout[3] & RayTracing::PsrHeaderLayout) != 0 || source.header != 0 ||
	    RayTracing::HeaderInUserData(source)) {
		preparation.header = RayTracing::HeaderInUserData(source) ? RayTracing::UserDataHeader(pointer) : source.header;
	} else if (const auto& words = bindings.runtime->resources->flattened_srt;
	           bindings.flattened_srt.buffer && (uint64_t {source.header_slots[0]} + 1) * 4 <= bindings.flattened_srt.range &&
	           (uint64_t {source.header_slots[1]} + 1) * 4 <= bindings.flattened_srt.range) {
		known = bindings.flattened_srt_known && source.header_slots[0] < words.size() &&
		        source.header_slots[1] < words.size();
		if (known) {
			preparation.header = uint64_t {words[source.header_slots[0]]} | uint64_t {words[source.header_slots[1]]} << 32u;
		}
	}
	if (known && native_rt->Prepared(preparation)) {
		return;
	}
	buffer.EndRendering();
	native_rt->Prepare(buffer.Handle(), pointer, source, bindings.flattened_srt, frame, module.info.shader_hash);
	if (known) {
		native_rt->KeepPrepared(preparation);
	}
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

struct RenderExecutor::DrawRenderStorage {
	DrawRenderState state {};
	bool tessellation_stages_written = false;
};
static_assert(std::is_trivially_destructible_v<DrawRenderState>);

template <typename T>
static void ValueInitialize(T& object) {
	::new (static_cast<void*>(std::addressof(object))) T {};
}

struct RenderExecutor::ResolvedDraw {
	DrawRenderStorage storage;
	ProgramResolution programs;
	ShaderComputeInputInfo compute;
	const ShaderModule*    compute_module = nullptr;
	uint64_t               resource_key = 0;
	std::atomic_bool       busy {false};
	enum : uint8_t { Ready, Pending, Failed };
	std::atomic<uint8_t> state {Ready};
};

struct RenderExecutor::ResolvedDrawRing {
	static constexpr size_t Size = 320;

	std::array<ResolvedDraw, Size> draws;
	size_t                         next = 0;
};

#if defined(ANYPS5_KYTY)
namespace {
std::atomic<uint64_t> g_flat_defer_resolutions {0};
std::atomic<uint64_t> g_flat_defer_words {0};
std::atomic<uint64_t> g_flat_defer_gpu_words {0};
std::atomic<uint64_t> g_flat_defer_line {0};

void NoteFlatDefer(uint32_t words, uint32_t gpu_words) {
	const auto now = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	        .count());
	auto last = g_flat_defer_line.load(std::memory_order_relaxed);
	if (last == 0) {
		g_flat_defer_line.compare_exchange_strong(last, now, std::memory_order_relaxed);
	} else if (now - last >= 10'000'000'000ull &&
	           g_flat_defer_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-flat-defer] summary of the last {:.1f} s: {} deferred resolutions, {} deferred words; {} of the words "
		    "left to a GPU copy\n",
		    static_cast<double>(now - last) / 1e9, g_flat_defer_resolutions.exchange(0, std::memory_order_relaxed),
		    g_flat_defer_words.exchange(0, std::memory_order_relaxed),
		    g_flat_defer_gpu_words.exchange(0, std::memory_order_relaxed)));
	}
	g_flat_defer_resolutions.fetch_add(1, std::memory_order_relaxed);
	g_flat_defer_words.fetch_add(words, std::memory_order_relaxed);
	g_flat_defer_gpu_words.fetch_add(gpu_words, std::memory_order_relaxed);
}
}
#endif

void RenderExecutor::ReleaseResolvedDraw(ResolvedDraw* resolved) noexcept {
	(void)WaitResolvedDraw(resolved);
	resolved->busy.store(false, std::memory_order_release);
}

RenderExecutor::ResolvedDraw* RenderExecutor::WaitResolvedDraw(ResolvedDraw* resolved) noexcept {
	auto state = resolved->state.load(std::memory_order_acquire);
	for (Common::SpinWait wait; state == ResolvedDraw::Pending;
	     state = resolved->state.load(std::memory_order_acquire)) {
		if (!wait.Spin()) {
			std::this_thread::yield();
		}
	}
	return state == ResolvedDraw::Ready ? resolved : nullptr;
}

namespace {
constexpr size_t   RefusedCauses        = static_cast<size_t>(StaleRead::Count);
constexpr size_t   ExecuteCauses        = static_cast<size_t>(RenderExecutor::ExecuteCause::Count);
constexpr size_t   ShapeRefusal         = static_cast<size_t>(StaleRead::None);
constexpr size_t   AheadRefusals        = static_cast<size_t>(AheadRefusal::Count);
constexpr uint32_t RevalidationSampling = 16;

struct alignas(64) ExecuteResolveCounters {
	std::array<std::atomic<uint64_t>, 2>                                 used {};
	std::array<std::atomic<uint64_t>, 2>                                 used_around {};
	std::array<std::atomic<uint64_t>, 2>                                 table {};
	std::atomic<uint64_t>                                                table_bare {0};
	std::array<std::atomic<uint64_t>, 2>                                 unused {};
	std::array<std::array<std::atomic<uint64_t>, RefusedCauses>, 2>      refused {};
	std::array<std::array<std::atomic<uint64_t>, ExecuteCauses>, 2>      here {};
	std::array<std::atomic<uint64_t>, 2>                                 here_ns {};
	std::atomic<uint64_t>                                                revalidation_ns {0};
};

struct alignas(64) AheadResolveCounters {
	std::array<std::atomic<uint64_t>, 2>                                 prepared {};
	std::array<std::array<std::atomic<uint64_t>, AheadRefusals>, 2>      refused {};
	std::atomic<uint64_t>                                                table_skips {0};
	std::array<std::atomic<uint64_t>, 2>                                 ns {};
};

ExecuteResolveCounters g_execute_resolve;
AheadResolveCounters   g_ahead_resolve;
std::atomic<uint64_t>  g_resolve_line {0};

void Bump(std::atomic<uint64_t>& counter, uint64_t value = 1) {
	counter.fetch_add(value, std::memory_order_relaxed);
}

uint64_t Take(std::atomic<uint64_t>& counter) {
	return counter.exchange(0, std::memory_order_relaxed);
}

void NoteAheadRefusal(bool dispatch, AheadRefusal refusal) {
	Bump(g_ahead_resolve.refused[dispatch ? 1 : 0][static_cast<size_t>(refusal)]);
}
}

void RenderExecutor::NoteAhead(bool dispatch, AheadNote note, uint64_t ns) noexcept {
	const size_t kind = dispatch ? 1 : 0;
	if (note == AheadNote::Prepared) {
		Bump(g_ahead_resolve.prepared[kind]);
	} else if (note == AheadNote::TableSkip) {
		Bump(g_ahead_resolve.table_skips);
	}
	Bump(g_ahead_resolve.ns[kind], ns);
}

void RenderExecutor::BeginAheadOperation(ResolvedDraw* resolved, AheadNote note) noexcept {
	UseResolvedDraw(resolved);
	m_ahead_note     = note;
	m_ahead_present  = m_resolved_draw != nullptr;
	m_ahead_dispatch = m_ahead_present && m_resolved_draw->compute_module != nullptr;
	m_ahead_taken    = false;
	m_ahead_table    = false;
}

void RenderExecutor::EndAheadOperation() noexcept {
	if (m_ahead_present && !m_ahead_taken) {
		const size_t kind = m_ahead_dispatch ? 1 : 0;
		Bump(m_ahead_table ? g_execute_resolve.table[kind] : g_execute_resolve.unused[kind]);
	} else if (!m_ahead_present && m_ahead_table) {
		Bump(g_execute_resolve.table_bare);
	}
	m_resolved_draw = nullptr;
	m_ahead_present = false;
	m_ahead_taken   = false;
	m_ahead_table   = false;
}

RenderExecutor::ExecuteCause RenderExecutor::MissingAheadCause() const noexcept {
	if (m_ahead_taken) return ExecuteCause::Again;
	if (m_ahead_present) return ExecuteCause::Revalidation;
	switch (m_ahead_note) {
		case AheadNote::TableSkip: return ExecuteCause::TableSkip;
		case AheadNote::NotPrepared: return ExecuteCause::NotPrepared;
		default: return ExecuteCause::Other;
	}
}

bool RenderExecutor::RevalidateAhead(ResolvedDraw& resolved, bool dispatch) {
	const bool sampled = m_revalidations++ % RevalidationSampling == 0;
	const auto begin   = sampled ? Common::SteadyNs() : 0;
#if defined(ANYPS5_KYTY)
	const bool valid = dispatch ? resolved.programs.Revalidate(true, &resolved.compute_module->info)
	                            : resolved.programs.Revalidate();
#else
	const bool valid = dispatch ? resolved.programs.Revalidate(true) : resolved.programs.Revalidate();
#endif
	if (sampled) Bump(g_execute_resolve.revalidation_ns, (Common::SteadyNs() - begin) * RevalidationSampling);
	const size_t kind = dispatch ? 1 : 0;
	if (valid) {
		Bump(g_execute_resolve.used[kind]);
		if (resolved.programs.reads.ReadAroundGpuWrites()) Bump(g_execute_resolve.used_around[kind]);
	} else {
		Bump(g_execute_resolve.refused[kind][static_cast<size_t>(resolved.programs.reads.Stale())]);
	}
	return valid;
}

const ShaderModule* RenderExecutor::ResolveComputeHere(const HW::ComputeShaderInfo& regs,
                                                       ShaderComputeInputInfo&      input_info) {
	const auto begin  = Common::SteadyNs();
	const auto* module = m_context.GetPipelineCache().GetComputeProgram(regs, input_info);
	Bump(g_execute_resolve.here_ns[1], Common::SteadyNs() - begin);
	Bump(g_execute_resolve.here[1][static_cast<size_t>(m_execute_cause)]);
	return module;
}

void RenderExecutor::LogResolveSummary() {
	const auto now  = Common::SteadyNs();
	auto       last = g_resolve_line.load(std::memory_order_relaxed);
	if (last == 0) {
		g_resolve_line.store(now, std::memory_order_relaxed);
		m_resolve_frames = m_context.FrameNumber();
		return;
	}
	if (now - last < 10'000'000'000ull) return;
	g_resolve_line.store(now, std::memory_order_relaxed);
	const auto frame_now = m_context.FrameNumber();
	const auto frames    = frame_now - m_resolve_frames;
	m_resolve_frames     = frame_now;
	const auto per_frame = [&](uint64_t ns) {
		return frames != 0 ? static_cast<double>(ns) / 1e6 / static_cast<double>(frames) : 0.0;
	};
	const auto execute_side = [&](size_t kind) {
		std::array<uint64_t, RefusedCauses> refused {};
		uint64_t                            refused_total = 0;
		for (size_t i = 0; i < RefusedCauses; i++) refused_total += refused[i] = Take(g_execute_resolve.refused[kind][i]);
		std::array<uint64_t, ExecuteCauses> here {};
		uint64_t                            here_total = 0;
		for (size_t i = 0; i < ExecuteCauses; i++) here_total += here[i] = Take(g_execute_resolve.here[kind][i]);
		return fmt::format(
		    "ahead used {} (read around GPU writes {}), dropped for the table path {}, dropped unused {}, refused by "
		    "revalidation {} (shape {}, failed {}, zero test {}, CPU write {}, GPU write {}, strict read {}, GPU write since "
		    "the read around it {}); resolved by execute {} (table hint skip {}, revalidation {}, not prepared {}, again in "
		    "the operation {}, other {}){}",
		    Take(g_execute_resolve.used[kind]), Take(g_execute_resolve.used_around[kind]),
		    Take(g_execute_resolve.table[kind]), Take(g_execute_resolve.unused[kind]), refused_total, refused[0],
		    refused[1], refused[2], refused[3], refused[4], refused[5], refused[6], here_total, here[0], here[1], here[2],
		    here[3], here[4],
		    kind == 0 ? fmt::format(", table path with nothing prepared {}", Take(g_execute_resolve.table_bare))
		              : std::string {});
	};
	const auto resolve_side = [&](size_t kind) {
		std::array<uint64_t, AheadRefusals> refused {};
		uint64_t                            total = 0;
		for (size_t i = 0; i < AheadRefusals; i++) total += refused[i] = Take(g_ahead_resolve.refused[kind][i]);
		return fmt::format("prepared {}, refused {} (read {}, stage {}, compiling {}, shape {}, calls {}, image heap {}, "
		                   "materialize {}, variant {}, GPU read variant {}, other {})",
		                   Take(g_ahead_resolve.prepared[kind]), total, refused[1], refused[2], refused[3], refused[4],
		                   refused[5], refused[6], refused[7], refused[8], refused[9], refused[0]);
	};
	const auto draws_here      = Take(g_execute_resolve.here_ns[0]);
	const auto dispatches_here = Take(g_execute_resolve.here_ns[1]);
	const auto draws_ahead     = Take(g_ahead_resolve.ns[0]);
	const auto dispatches_ahead = Take(g_ahead_resolve.ns[1]);
	const auto execute_draws      = execute_side(0);
	const auto execute_dispatches = execute_side(1);
	const auto ahead_draws        = resolve_side(0);
	const auto ahead_dispatches   = resolve_side(1);
	const auto around             = Libs::LibKernel::Memory::TakeAheadAroundReads();
	Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-resolve] summary of the last {:.1f} s, {} frames: draws: {}; dispatches: {}; resolve thread: draws {}, "
	    "skipped by the table hint {}; dispatches {}; walker loads on GPU written pages: read around GPU writes {}, "
	    "refused on GPU written lines {}, pages refused for images {}; ms per frame: execute resolution {:.2f} (draws "
	    "{:.2f}, dispatches {:.2f}), execute revalidation {:.2f}, resolve thread resolution {:.2f} (draws {:.2f}, "
	    "dispatches {:.2f})\n",
	    static_cast<double>(now - last) / 1e9, frames, execute_draws, execute_dispatches, ahead_draws,
	    Take(g_ahead_resolve.table_skips), ahead_dispatches, around.read, around.gpu_bytes, around.image_page,
	    per_frame(draws_here + dispatches_here),
	    per_frame(draws_here), per_frame(dispatches_here), per_frame(Take(g_execute_resolve.revalidation_ns)),
	    per_frame(draws_ahead + dispatches_ahead), per_frame(draws_ahead), per_frame(dispatches_ahead)));
}

struct RenderExecutor::RenderTargetMemo {
	struct MetadataEpoch {
		GuestRange range;
		uint64_t   epoch = 0;
	};

	uint64_t                                                   context_id   = 0;
	uint32_t                                                   mrt_mask     = 0;
	uint32_t                                                   target_mask  = 0;
	std::array<HW::RenderTarget, RENDER_COLOR_ATTACHMENTS_MAX> targets {};
	HW::DepthRenderTarget                                      depth {};
	HW::RenderControl                                          render_control {};
	std::array<float, 3>                                       depth_values {};
	uint8_t                                                    stencil_clear = 0;
	bool                                                       depth_wanted  = false;

	bool pending = false;
	bool                                                           valid                = false;
	uint64_t                                                       tick                 = 0;
	uint64_t                                                       image_set_epoch      = 0;
	uint64_t                                                       gpu_write_generation = 0;
	std::array<MetadataEpoch, RENDER_COLOR_ATTACHMENTS_MAX + 1>    metadata {};
	uint32_t                                                       metadata_count = 0;
	std::array<ImageLookupState, RENDER_COLOR_ATTACHMENTS_MAX + 2> images {};
	uint32_t                                                       image_count = 0;
	DrawTargets                                                    draw_targets;
	std::array<vk::ImageView, RENDER_COLOR_ATTACHMENTS_MAX>        color_views {};
	vk::ImageView                                                  depth_view = nullptr;
};

template <typename T>
static bool SameBytes(const T& a, const T& b) {
	static_assert(std::is_trivially_copyable_v<T>);
	return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
static void CopyBytes(T& destination, const T& source) {
	static_assert(std::is_trivially_copyable_v<T>);
	std::memcpy(&destination, &source, sizeof(T));
}

void RenderExecutor::InvalidateRenderTargetMemo() noexcept {
	m_draw_window++;
	if (m_render_target_memo != nullptr) {
		m_render_target_memo->valid   = false;
		m_render_target_memo->pending = false;
	}
	m_render_targets_reused = false;
}

void RenderExecutor::CaptureRenderTargetKey(const HW::Context& hw, uint32_t mrt_mask) {
	if (m_render_target_memo == nullptr) {
		m_render_target_memo = {new RenderTargetMemo {},
		                        [](RenderTargetMemo* memo) { delete memo; }};
	}
	auto& memo   = *m_render_target_memo;
	memo.valid   = false;
	memo.pending = !graphics_debug_dump_enabled();
	if (!memo.pending) {
		return;
	}
	memo.context_id   = m_context_id;
	memo.mrt_mask     = mrt_mask;
	memo.target_mask  = hw.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			CopyBytes(memo.targets[slot], hw.GetRenderTarget(slot));
		}
	}
	CopyBytes(memo.depth, hw.GetDepthRenderTarget());
	CopyBytes(memo.render_control, hw.GetRenderControl());
	memo.depth_wanted  = DepthStencilTargetWanted(hw);
	memo.depth_values  = {hw.GetDepthClearValue(), hw.GetDepthBoundsMin(), hw.GetDepthBoundsMax()};
	memo.stencil_clear = hw.GetStencilClearValue();
}

bool RenderExecutor::RenderTargetMemoCurrent() {
	const auto& memo  = *m_render_target_memo;
	auto&       cache = m_context.GetTextureCache();
	if (memo.tick != m_context.GetCommandScheduler().CurrentTick() ||
	    memo.image_set_epoch != cache.ImageSetEpoch() ||
	    memo.gpu_write_generation != m_context.GetBufferCache().GpuWriteGeneration()) {
		return false;
	}
	for (uint32_t i = 0; i < memo.image_count; i++) {
		const auto& state = memo.images[i];
		const auto* image = cache.m_slot_images.try_get(state.id);
		if (image == nullptr || !image->registered || image->binding.needs_rebind ||
		    ImageLookupState::Of(state.id, *image) != state) {
			return false;
		}
	}
	auto& buffers = m_context.GetBufferCache();
	for (uint32_t i = 0; i < memo.metadata_count; i++) {
		const auto& metadata = memo.metadata[i];
		if (buffers.CpuDirtyEpoch(metadata.range.address, metadata.range.size) != metadata.epoch) {
			return false;
		}
	}
	return true;
}

bool RenderExecutor::TryReuseRenderTargets(const HW::Context& hw, uint32_t mrt_mask) {
	if (m_render_target_memo == nullptr || !m_render_target_memo->valid) {
		return false;
	}
	auto& memo = *m_render_target_memo;
	if (memo.mrt_mask != mrt_mask) {
		return false;
	}
	if (m_context_id == 0 || memo.context_id != m_context_id) {
		const float depth_clear = hw.GetDepthClearValue();
		if (memo.target_mask != hw.GetRenderTargetMask() ||
		    !SameBytes(memo.depth, hw.GetDepthRenderTarget()) ||
		    !SameBytes(memo.render_control, hw.GetRenderControl()) ||
		    memo.depth_wanted != DepthStencilTargetWanted(hw) ||
		    !SameBytes(memo.depth_values[0], depth_clear) ||
		    memo.stencil_clear != hw.GetStencilClearValue()) {
			return false;
		}
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) != 0 &&
			    !SameBytes(memo.targets[slot], hw.GetRenderTarget(slot))) {
				return false;
			}
		}
		memo.context_id = m_context_id;
	}
	if (!RenderTargetMemoCurrent()) {
		return false;
	}
	auto& targets = memo.draw_targets;
	if (targets.depth.image_id) {
		ApplyDepthStencilTestState(hw, targets.depth);
	}
	m_draw_targets = &targets;
	for (uint32_t i = 0; i < targets.color_count; i++) {
		BindRenderTarget(targets.colors[i].image_id);
	}
	if (targets.depth.image_id) {
		BindRenderTarget(targets.depth.image_id);
	}
	return true;
}

void RenderExecutor::StoreRenderTargetMemo(const RenderColorInfo* colors, uint32_t color_count,
                                           const RenderDepthInfo& depth, const vk::ImageView* views,
                                           vk::ImageView depth_view, ImageId stencil) {
	auto& memo   = *m_render_target_memo;
	memo.pending = false;
	if (depth.depth_clear_enable || depth.load_clear_aspects || depth.stencil_clear_enable) {
		return;
	}
	auto& cache             = m_context.GetTextureCache();
	auto& buffers           = m_context.GetBufferCache();
	memo.metadata_count     = 0;
	memo.image_count        = 0;
	const auto add_metadata = [&](const ImageInfo& info) {
		if (info.metadata.kind == ImageMetadataKind::None || info.metadata.range.Empty()) {
			return true;
		}
		const auto range                     = info.metadata.range;
		memo.metadata[memo.metadata_count++] = {range,
		                                        buffers.CpuDirtyEpoch(range.address, range.size)};
		return true;
	};
	const auto add_image = [&](ImageId id) {
		memo.images[memo.image_count++] = ImageLookupState::Of(id, cache.m_slot_images[id]);
	};
	for (uint32_t i = 0; i < color_count; i++) {
		if (!add_metadata(colors[i].desc.info)) {
			return;
		}
		add_image(colors[i].image_id);
		memo.draw_targets.colors[i] = colors[i];
		memo.color_views[i]         = views[i];
	}
	if (depth.image_id) {
		if (!add_metadata(depth.desc.info)) {
			return;
		}
		add_image(depth.image_id);
		if (stencil) {
			add_image(stencil);
		}
	}
	memo.draw_targets.color_count = color_count;
	memo.draw_targets.depth       = depth;
	memo.depth_view               = depth_view;
	memo.tick                 = m_context.GetCommandScheduler().CurrentTick();
	memo.image_set_epoch      = cache.ImageSetEpoch();
	memo.gpu_write_generation = buffers.GpuWriteGeneration();
	memo.valid                = true;
}

static void ResetDrawRenderState(DrawRenderState& state, bool tessellation_stages) {
	state.mrt_mask                 = 0;
	state.topology                 = vk::PrimitiveTopology::ePointList;
	state.primitive_restart_enable = false;
	state.ps_active                = false;
	ValueInitialize(state.vertex_info[0]);
	if (tessellation_stages) {
		ValueInitialize(state.vertex_info[1]);
		ValueInitialize(state.vertex_info[2]);
	}
	ValueInitialize(state.ps_input_info);
	state.indirect_offsets = {};
	state.indirect_draw    = false;
}

DrawRenderState& RenderExecutor::AcquireDrawRenderState(bool tessellation, bool& shaders_resolved,
                                                        const ShaderIndirectOffsets& indirect_offsets,
                                                        bool indirect_draw) {
	auto* resolved   = std::exchange(m_resolved_draw, nullptr);
	shaders_resolved = false;
	if (resolved == nullptr) {
		m_execute_cause = MissingAheadCause();
		if (m_execute_cause == ExecuteCause::Revalidation) Bump(g_execute_resolve.refused[0][ShapeRefusal]);
	} else if (resolved->compute_module != nullptr ||
	           !(resolved->storage.state.indirect_offsets == indirect_offsets) ||
	           resolved->storage.state.indirect_draw != indirect_draw) {
		Bump(g_execute_resolve.refused[0][ShapeRefusal]);
		m_execute_cause = ExecuteCause::Revalidation;
	} else if (RevalidateAhead(*resolved, false)) {
		shaders_resolved = true;
	} else {
		m_execute_cause = ExecuteCause::Revalidation;
	}
	m_ahead_taken = m_ahead_taken || m_ahead_present;
#if defined(ANYPS5_KYTY)
	if (shaders_resolved) {
		if (const auto words = resolved->programs.reads.DeferredWords(); words != 0) NoteFlatDefer(words, 0);
	}
#endif
	m_draw_resource_key =
	    shaders_resolved && !resolved->programs.refreshed_words ? resolved->resource_key : 0;
	if (shaders_resolved) {
		return resolved->storage.state;
	}
	if (m_draw_state == nullptr) {
		m_draw_state = {new DrawRenderStorage {},
		                [](DrawRenderStorage* storage) { delete storage; }};
	} else {
		ResetDrawRenderState(m_draw_state->state, m_draw_state->tessellation_stages_written);
	}
	m_draw_state->tessellation_stages_written = tessellation;
	return m_draw_state->state;
}

struct DrawCallInfo {
	bool     indexed        = true;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;

	[[nodiscard]] const char* Name() const { return indexed ? "DrawIndex" : "DrawIndexAuto"; }
};

#if defined(ANYPS5_KYTY)
void RenderExecutor::PreparePackedTargets(CommandBuffer& buffer, PreparedBindings& pixel,
                                          const ShaderPixelInputInfo& info) {
	auto& cache = m_context.GetTextureCache();
	pixel.packed_targets.clear();
	for (uint32_t export_index = 0; export_index < 8u; export_index++) {
		if ((info.target_packed_blend_mask & (1u << export_index)) == 0u) continue;
		const auto slot   = ShaderPixelExportTarget(info.target_shader_mask, export_index);
		const auto colors = std::span {m_draw_targets->colors, m_draw_targets->color_count};
		const auto found  = std::ranges::find(colors, slot, &RenderColorInfo::target_slot);
		if (found == colors.end()) {
			EXIT("packed UNorm render target %u of export %u is not bound\n", slot, export_index);
		}
		auto& image = cache.GetImage(found->image_id);
		if (image.binding.is_bound) {
			EXIT("packed UNorm render target %u is also a shader resource of its draw\n", slot);
		}
		auto view_info = found->desc.view_info;
		if (found->desc.info.samples != 1u || found->desc.info.IsVolume() ||
		    view_info.level_count != 1u || view_info.layer_count != 1u ||
		    view_info.format != vk::Format::eR32Uint) {
			EXIT("unsupported packed UNorm render target %u: samples=%u volume=%u levels=%u layers=%u "
			     "format=%d\n",
			     slot, found->desc.info.samples, static_cast<uint32_t>(found->desc.info.IsVolume()),
			     view_info.level_count, view_info.layer_count, static_cast<int>(view_info.format));
		}
		(void)cache.FindRenderTarget(found->image_id, found->desc);
		view_info.type    = vk::ImageViewType::e2D;
		view_info.mapping = {};
		view_info.min_lod = 0;
		view_info.usage   = vk::ImageUsageFlagBits::eStorage;
		const auto view   = image.FindView(view_info);
		EXIT_IF(view == nullptr);
		BindImage(found->image_id, true);
		image.Transit(vk::ImageLayout::eGeneral,
		              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
		              ImageSubresourceRange {view_info.base_level, 1u, view_info.base_layer, 1u},
		              buffer.Handle());
		pixel.packed_targets.push_back({nullptr, view, vk::ImageLayout::eGeneral});
		m_packed_target_slots |= 1u << slot;
	}
}
#endif

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	const RenderTargetMemo* memo = nullptr;
	if (m_render_targets_reused && m_render_target_memo->valid &&
	    m_render_target_memo->draw_targets.color_count == color_count &&
	    m_render_target_memo->draw_targets.depth.image_id == depth.image_id &&
	    RenderTargetMemoCurrent()) {
		memo = m_render_target_memo.get();
		for (uint32_t i = 0; i < color_count && memo != nullptr; i++) {
			if (memo->draw_targets.colors[i].image_id != colors[i].image_id) {
				memo = nullptr;
			}
		}
	}
	if (m_render_targets_reused && memo == nullptr) {
		m_render_target_memo->valid = false;
	}
	const bool store =
	    memo == nullptr && m_render_target_memo != nullptr && m_render_target_memo->pending;
	std::array<vk::ImageView, RENDER_COLOR_ATTACHMENTS_MAX> color_views {};
	vk::ImageView                                           depth_view = nullptr;
	ImageId                                                 stencil_id {};
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto image_view = memo != nullptr
		                            ? memo->color_views[i]
		                            : cache.FindRenderTarget(target.image_id, target.desc);
		color_views[i]        = image_view;
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
#if defined(ANYPS5_KYTY)
		if ((m_packed_target_slots & (1u << target.target_slot)) != 0u) {
			const auto extent           = target.Extent();
			state.width                 = std::min(state.width, extent.width);
			state.height                = std::min(state.height, extent.height);
			state.num_layers            = std::min(state.num_layers, target.desc.view_info.layer_count);
			state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
			continue;
		}
#endif
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto image_view =
		    memo != nullptr ? memo->depth_view
		                    : cache.FindDepthTarget(depth.image_id, depth.desc, &stencil_id);
		depth_view = image_view;
		if (memo == nullptr) {
			const auto& metadata = depth.desc.info.metadata;
			const bool meta_clear =
			    metadata.kind == ImageMetadataKind::Htile &&
			    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
			depth.load_clear_aspects = {};
			if (depth.depth_clear_enable || meta_clear)
				depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
			if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
				depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
			if (meta_clear &&
			    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
				EXIT("failed to consume HTile clear state\n");
			}
		}
		EXIT_IF(image_view == nullptr ||
		        cache.GetImage(depth.image_id).backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		const bool feedback_loops = m_context.GetGraphics().attachment_feedback_loop_enabled;
		vk::ImageAspectFlags sampled_aspects;
		for (auto* stage: stages) {
			for (auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto& views = cache.GetImage(depth.image_id).views;
				const auto  native =
				    std::ranges::find(views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == views.end());
				const auto info     = native->info;
				const auto feedback = DepthFeedbackAspects(draw_writes, depth.desc.view_info, info);
				if (feedback && !feedback_loops) {
					static std::atomic_bool logged = false;
					if (!logged.exchange(true, std::memory_order_relaxed)) {
						std::printf("depth attachment feedback loop is not supported by the host: "
						            "draws sample a pre-draw copy of the depth target\n");
					}
					std::vector<ImageViewInfo> mip_infos;
					for (const auto mip_view: binding.mip_views) {
						const auto found =
						    std::ranges::find(views, mip_view, &CachedImageView::view);
						EXIT_IF(found == views.end());
						mip_infos.push_back(found->info);
					}
					const auto copy_id = cache.CopyDepthForFeedback(
					    depth.image_id,
					    {info.base_level, info.level_count, info.base_layer, info.layer_count},
					    info.aspect);
					auto& copy         = cache.GetImage(copy_id);
					binding.image_id   = copy_id;
					binding.image_view = copy.FindView(info);
					for (size_t i = 0; i < mip_infos.size(); i++) {
						binding.mip_views[i] = copy.FindView(mip_infos[i]);
					}
					continue;
				}
				sampled_aspects |= info.aspect;
				feedback_aspects |= feedback;
			}
			const auto table_element = [&](const TextureBinding& binding) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) return;
				const auto& views = cache.GetImage(depth.image_id).views;
				const auto  native =
				    std::ranges::find(views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == views.end());
				const auto feedback =
				    DepthFeedbackAspects(draw_writes, depth.desc.view_info, native->info);
				if (feedback && !feedback_loops) {
					static std::atomic_bool logged = false;
					if (!logged.exchange(true, std::memory_order_relaxed)) {
						std::printf("depth attachment feedback loop is not supported by the host: a "
						            "bindless table element samples the depth target its draw writes\n");
					}
				}
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= feedback;
			};
			if (stage->tables_reused) {
				const auto& table_state = *static_cast<const TableBindingState*>(stage->table_state);
				for (const auto index: stage->table_visits) {
					table_element(table_state.elements[index].resolution->texture);
				}
			} else {
				for (const auto* binding: stage->table_images) {
					table_element(*binding);
				}
			}
		}
		auto& image = cache.GetImage(depth.image_id);
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	if (store) {
		StoreRenderTargetMemo(colors, color_count, depth, color_views.data(), depth_view,
		                      stencil_id);
	}
	m_targets_from_memo = memo != nullptr;
	return state;
}

bool RenderExecutor::TakeTemplateTargets(const DrawTemplate& found) {
	if (!m_render_targets_reused || m_render_target_memo == nullptr || !m_render_target_memo->valid) {
		return false;
	}
	const auto& memo  = *m_render_target_memo;
	auto&       cache = m_context.GetTextureCache();
	if (memo.tick != m_context.GetCommandScheduler().CurrentTick() ||
	    memo.image_set_epoch != cache.ImageSetEpoch() ||
	    memo.gpu_write_generation != m_context.GetBufferCache().GpuWriteGeneration()) {
		return false;
	}
	const auto& targets = memo.draw_targets;
	const auto  count   = targets.color_count + (targets.depth.image_id ? 1u : 0u);
	if (found.target_count != count) {
		return false;
	}
	for (uint32_t i = 0; i < targets.color_count; i++) {
		const auto& target = targets.colors[i];
		if (found.target_images[i] != target.image_id.index ||
		    cache.ImageVersion(target.image_id) != found.target_versions[i] ||
		    found.rendering.color_attachments[target.target_slot].image_view != memo.color_views[i]) {
			return false;
		}
	}
	if (targets.depth.image_id) {
		const auto i = targets.color_count;
		if (found.target_images[i] != targets.depth.image_id.index ||
		    cache.ImageVersion(targets.depth.image_id) != found.target_versions[i] ||
		    found.rendering.depth_stencil_attachment.image_view != memo.depth_view) {
			return false;
		}
	}
	const auto take = [&](ImageId id, uint32_t i) {
		auto& image                     = cache.GetImage(id);
		image.binding.attachment_layout = found.target_layouts[i];
		image.binding.attachment_access = found.target_access[i];
		image.write_count++;
	};
	for (uint32_t i = 0; i < targets.color_count; i++) {
		take(targets.colors[i].image_id, i);
	}
	if (targets.depth.image_id) {
		take(targets.depth.image_id, targets.color_count);
	}
	return true;
}

void RenderExecutor::StoreTemplateTargets(DrawTemplate& stored, const RenderState& rendering,
                                          vk::ImageAspectFlags feedback_aspects) {
	auto&       cache   = m_context.GetTextureCache();
	const auto& targets = *m_draw_targets;
	uint32_t    count   = 0;

	const auto add = [&](ImageId id) {
		const auto& image             = cache.GetImage(id);
		stored.target_images[count]   = id.index;
		stored.target_versions[count] = cache.ImageVersion(id);
		stored.target_layouts[count]  = image.binding.attachment_layout;
		stored.target_access[count]   = image.binding.attachment_access;
		count++;
	};
	for (uint32_t i = 0; i < targets.color_count; i++) {
		add(targets.colors[i].image_id);
	}
	if (targets.depth.image_id) {
		add(targets.depth.image_id);
	}
	stored.target_count     = count;
	stored.rendering        = rendering;
	stored.feedback_aspects = feedback_aspects;
	stored.targets_valid    = true;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, 0, true);
		return true;
	}
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
};

struct DrawIndexBufferSource {
	uint64_t    address            = 0;
	const void* host_data          = nullptr;
	uint64_t    size               = 0;
	uint32_t    guest_element_size = 0;
};

struct MeshTaskParams {
	vk::DeviceAddress records        = 0;
	vk::DeviceAddress index_address  = 0;
	uint32_t          stride         = 0;
	uint32_t          index_size     = 0;
	uint32_t          index_limit    = 0;
	uint32_t          primitive_size = 0;
	uint32_t          group_step     = 0;
	uint32_t          max_groups     = 0;
	uint32_t          max_instances  = 0;
	uint32_t          max_total      = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t base_address;
	uint64_t requested_end;
	uint64_t acquired_end;
	Buffer*  buffer;
	uint64_t offset;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers;
	std::array<vk::DeviceSize, MaxBuffers> offsets;
	std::array<vk::DeviceSize, MaxBuffers> sizes;
	uint32_t                               count = 0;
};

static void AcquireVertexBuffers(CommandBuffer& buffer, const ShaderVertexInputInfo& vs_input_info,
                                 PreparedVertexBuffers& prepared) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes;
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges;
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {.base_address = vertex.addr, .requested_end = vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	uint32_t merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 && ranges[merged_count - 1].requested_end >= range.base_address) {
			auto& merged         = ranges[merged_count - 1];
			merged.requested_end = std::max(merged.requested_end, range.requested_end);
			continue;
		}
		ranges[merged_count++] = range;
	}
	const auto merged_ranges = std::span(ranges).first(merged_count);

	auto& cache = buffer.GetContext().GetBufferCache();
	for (auto& range: merged_ranges) {
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		std::tie(range.buffer, range.offset) = cache.ObtainBuffer(range.base_address, size, false);
	}

	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			prepared.sizes[i]   = 0;
			continue;
		}

		const auto range = std::ranges::find_if(merged_ranges, [&](const VertexBufferRange& value) {
			return vertex.addr >= value.base_address && vertex.addr < value.acquired_end;
		});
		if (range == merged_ranges.end()) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->buffer->Handle();
		prepared.offsets[i] = range->offset + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		return true;
	}

	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

RenderExecutor::ResolvedDraw* RenderExecutor::ResolveDrawAhead(
    const HW::Context& context, const HW::UserConfig& user_config, const HW::Shader& shaders,
    const ShaderIndirectOffsets& indirect_offsets, bool indirect_draw, uint64_t context_id,
    uint64_t user_config_id) {
	if (!DrawHasValidVertexShader(shaders)) {
		return nullptr;
	}
	auto& draw = NextResolvedDraw();
	draw.state.store(ResolvedDraw::Ready, std::memory_order_relaxed);
	if (!ResolveDrawPrograms(draw, context, user_config, shaders, indirect_offsets, indirect_draw, context_id,
	                         user_config_id)) {
		NoteAheadRefusal(false, draw.programs.reads.Refusal());
		return nullptr;
	}
	draw.busy.store(true, std::memory_order_relaxed);
	return &draw;
}

RenderExecutor::ResolvedDraw& RenderExecutor::NextResolvedDraw() {
	if (m_resolved_ring == nullptr) {
		m_resolved_ring = {new ResolvedDrawRing {}, [](ResolvedDrawRing* ring) { delete ring; }};
	}
	auto& ring = *m_resolved_ring;
	auto& draw = ring.draws[ring.next];
	for (Common::SpinWait wait; draw.busy.load(std::memory_order_acquire);) {
		if (!wait.Spin()) {
			std::this_thread::yield();
		}
	}
	ring.next = (ring.next + 1) % ResolvedDrawRing::Size;
	return draw;
}

bool RenderExecutor::ResolveDrawPrograms(ResolvedDraw& draw, const HW::Context& context,
                                         const HW::UserConfig& user_config, const HW::Shader& shaders,
                                         const ShaderIndirectOffsets& indirect_offsets, bool indirect_draw,
                                         uint64_t context_id, uint64_t user_config_id) {
	draw.compute_module = nullptr;
	auto& state         = draw.storage.state;
	ResetDrawRenderState(state, draw.storage.tessellation_stages_written);
	draw.storage.tessellation_stages_written =
	    user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	state.indirect_offsets = indirect_offsets;
	state.indirect_draw    = indirect_draw;
	draw.programs.Reset();
	if (!m_context.GetPipelineCache().GetGraphicsProgramsAhead(context, user_config, shaders, state,
	                                                          draw.programs) ||
	    draw.programs.reads.Failed() || state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
#if defined(ANYPS5_KYTY)
	if (m_context.GetPipelineCache().PreparesGraphicsLibraries()) {
		m_context.GetPipelineCache().PrepareGraphicsLibraries(context, user_config, state);
	}
#endif
	draw.resource_key = 0;
	if (DrawTemplatesEnabled()) {
		std::array<const ShaderStageRuntime*, 4> stages {};
		uint32_t                                 stage_count = 0;
		for (const auto& stage: state.VertexStages()) {
			stages[stage_count++] = &stage.stage;
		}
		if (state.ps_active) {
			stages[stage_count++] = &state.ps_input_info.stage;
		}
		if (std::ranges::all_of(std::span {stages.data(), stage_count},
		                        [](const ShaderStageRuntime* stage) { return static_cast<bool>(*stage); })) {
			draw.resource_key = DrawResourceKey({stages.data(), stage_count});
		}
		if (draw.resource_key != 0 && DrawStateTemplatesEnabled()) {
			thread_local uint64_t context_key_id = 0, context_key = 0, user_config_key_id = 0, user_config_key = 0;
			if (context_id == 0 || context_id != context_key_id) {
				context_key    = DrawContextKey(context);
				context_key_id = context_id;
			}
			if (user_config_id == 0 || user_config_id != user_config_key_id) {
				user_config_key    = DrawUserConfigKey(user_config);
				user_config_key_id = user_config_id;
			}
			draw.resource_key =
			    DrawStateKey(draw.resource_key, context_key, user_config_key, shaders, state.vertex_info[0]);
		}
	}
	return true;
}

RenderExecutor::ResolvedDraw* RenderExecutor::ResolveDispatchAhead(const HW::Context& context,
                                                                   const HW::Shader&  shaders,
                                                                   const uint32_t (&groups)[3],
                                                                   uint32_t mode,
                                                                   bool async_compute) {
	(void)context;
	const auto& cs_regs = shaders.GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return nullptr;
	}
	auto& slot = NextResolvedDraw();
	slot.state.store(ResolvedDraw::Ready, std::memory_order_relaxed);
	slot.programs.Reset();
	ValueInitialize(slot.compute);
	slot.compute.async_compute = async_compute;
	slot.compute.dispatch_thread_dimensions = (mode & (1u << 5u)) != 0;
	const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
	                                cs_regs.cs_regs.num_thread_z};
	for (uint32_t axis = 0; axis < 3u; ++axis) {
		const auto size = slot.compute.dispatch_thread_dimensions ? std::max(group_sizes[axis], 1u) : 1u;
		slot.compute.workgroup_counts[axis] = groups[axis] / size + (groups[axis] % size != 0u);
	}
	slot.compute_module =
	    m_context.GetPipelineCache().GetComputeProgramAhead(cs_regs, slot.compute, slot.programs);
	if (slot.programs.reads.Failed() || slot.compute_module == nullptr) {
		slot.compute_module = nullptr;
		NoteAheadRefusal(true, slot.programs.reads.Refusal());
		return nullptr;
	}
	slot.busy.store(true, std::memory_order_relaxed);
	return &slot;
}

RenderExecutor::ResolvedDraw* RenderExecutor::ResolveIndirectDispatchAhead(
    const HW::Context& context, const HW::Shader& shaders, uint32_t mode, uint64_t args_addr,
    bool async_compute) {
	(void)context;
	const auto& cs_regs = shaders.GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return nullptr;
	}
	auto& slot = NextResolvedDraw();
	slot.state.store(ResolvedDraw::Ready, std::memory_order_relaxed);
	slot.programs.Reset();
	ValueInitialize(slot.compute);
	slot.compute.async_compute = async_compute;
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		slot.compute.dispatch_thread_dimensions = true;
		slot.compute.dispatch_indirect_threads  = true;
	} else {
		slot.compute.indirect_args_address = args_addr;
	}
	slot.compute_module =
	    m_context.GetPipelineCache().GetComputeProgramAhead(cs_regs, slot.compute, slot.programs);
	if (slot.programs.reads.Failed() || slot.compute_module == nullptr) {
		slot.compute_module = nullptr;
		NoteAheadRefusal(true, slot.programs.reads.Refusal());
		return nullptr;
	}
	slot.busy.store(true, std::memory_order_relaxed);
	return &slot;
}

bool RenderExecutor::TakeResolvedDispatch(ShaderComputeInputInfo& input_info,
                                          const ShaderModule*&    module) {
	auto* resolved = std::exchange(m_resolved_draw, nullptr);
	if (resolved == nullptr) {
		m_execute_cause = MissingAheadCause();
		return false;
	}
	m_ahead_taken = true;
	if (resolved->compute_module == nullptr) {
		Bump(g_execute_resolve.refused[1][ShapeRefusal]);
		m_execute_cause = ExecuteCause::Revalidation;
		return false;
	}
	if (!RevalidateAhead(*resolved, true)) {
		m_execute_cause = ExecuteCause::Revalidation;
		return false;
	}
#if defined(ANYPS5_KYTY)
	if (const auto words = resolved->programs.reads.DeferredWords(); words != 0) {
		NoteFlatDefer(words, resolved->programs.reads.DeferredGpuWords());
	}
#endif
	input_info = resolved->compute;
	module     = resolved->compute_module;
	return true;
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, bool shaders_resolved,
                                            const DrawCallInfo& draw,
                                            DrawRenderState&    state) {
	if (!shaders_resolved) {
		m_context.GetImageHeap().BeginDraw();
		const auto begin = Common::SteadyNs();
		m_context.GetPipelineCache().GetGraphicsPrograms(buffer, state);
		Bump(g_execute_resolve.here_ns[0], Common::SteadyNs() - begin);
		Bump(g_execute_resolve.here[0][static_cast<size_t>(m_execute_cause)]);
	}
	if (state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
	const auto& vp = buffer.GetRegisters().GetScreenViewport();
	if ((vp.screen_scissor_right <= vp.screen_scissor_left ||
	     vp.screen_scissor_bottom <= vp.screen_scissor_top) &&
	    std::ranges::none_of(state.VertexStages(), HasShaderBufferWrites,
	                         &ShaderVertexInputInfo::stage)) {
		return false;
	}
	const auto mrt_mask = state.mrt_mask;
	m_render_targets_reused = TryReuseRenderTargets(buffer.GetRegisters(), mrt_mask);
	if (!m_render_targets_reused) {
		m_draw_targets      = &m_targets;
		auto& targets       = m_targets;
		targets.color_count = 0;
		CaptureRenderTargetKey(buffer.GetRegisters(), mrt_mask);
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((mrt_mask & (1u << slot)) != 0) {
				ResolveRenderColorTarget(buffer, targets.colors[targets.color_count], slot);
				if (targets.colors[targets.color_count].image_id) {
					targets.color_count++;
				}
			}
		}
		ResolveRenderDepthTarget(buffer, targets.depth);
		const auto colors       = std::span {targets.colors, targets.color_count};
		const auto pass_samples = targets.depth.image_id ? targets.depth.desc.info.samples
		                                                 : targets.colors[0].desc.info.samples;
		if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
			    return color.desc.info.samples != pass_samples;
		    })) {
			const auto raster =
			    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
			const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
				return color.desc.info.samples != raster;
			});
			targets.color_count -= static_cast<uint32_t>(dropped.size());
			if (targets.depth.image_id && targets.depth.desc.info.samples != raster) {
				targets.depth = {};
			}
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping attachments whose sample count differs from the raster.\n");
			}
		}
	}

	if (m_draw_targets->color_count == 0 && !m_draw_targets->depth.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), m_draw_targets->depth, buffer, draw.index_count, 0);
		return false;
	}

	state.ps_plain = state.ps_active && RecordsNoMipStats(state.ps_input_info.stage);
	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type =
	    source.guest_element_size == 4 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
	if (source.host_data != nullptr) {
		auto& cache     = buffer.GetContext().GetBufferCache();
		auto& stream    = cache.GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = cache.StreamUpload(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(CommandRecorder              vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	if (prepared.count != 0) {
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(CommandRecorder vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
                                 const DrawRenderState& state, const DrawTargets& targets,
                                 uint32_t index_type_and_size, const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.indexed && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	static const RenderColorInfo none {};

	const auto& color = targets.color_count != 0 ? targets.colors[0] : none;
	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), color, targets.depth, buffer, state.ps_input_info,
		                   draw.index_count, 0);
	}
	LogDrawInputState(buffer, color, state.vertex_info[0], index_type_and_size, draw.index_count,
	                  index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, CommandRecorder vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                      const DrawEmitInfo& emit, const DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto&       depth = m_draw_targets->depth;
	auto&       color = m_draw_targets->colors[0];
	EXIT_NOT_IMPLEMENTED(
	    !m_context.GetGraphics().maintenance8_enabled ||
	    !Prospero::IsRectList(ucfg.GetPrimType()) || draw.indexed || draw.index_count != 3 ||
	    draw.instance_count != 1 || emit.vertex_offset != 0 || emit.first_instance != 0 ||
	    m_draw_targets->color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	    color.desc.info.samples != 1 || depth.desc.info.extent.width != color.Extent().width ||
	    depth.desc.info.extent.height != color.Extent().height ||
	    depth.desc.view_info.layer_count != 1 || color.desc.view_info.layer_count != 1 ||
	    hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	    state.ps_input_info.ps_depth_export_enable ||
	    state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		buffer.BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		buffer.EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

bool RecordsNoMipStats(const ShaderStageRuntime& stage) {
	if (stage.module == nullptr || stage.module->plain == nullptr || stage.resources == nullptr) {
		return false;
	}
	const auto& images = stage.module->info.info.images;
	const auto& values = stage.resources->images;
	for (uint32_t i = 0; i < images.size() && i < values.size(); i++) {
		if (!images[i].heap && images[i].table_capacity == 0u && values[i].dword_count >= 8u &&
		    LodStatsReport::HasCounter(values[i].dwords.data())) {
			return false;
		}
	}
	return true;
}

static bool DrawOnlyChangesPicture(RenderContext& context, const DrawRenderState& state,
                                   const DrawTargets&                     targets,
                                   std::span<const ShaderVertexInputInfo> vertex_stages) {
	const auto writes_memory = [](const ShaderStageRuntime& stage) {
		if (stage.module == nullptr) {
			return true;
		}
		const auto& program = stage.module->info;
		return program.has_address_writes ||
		       std::ranges::any_of(program.info.buffers,
		                           [](const auto& buffer) { return buffer.written; }) ||
		       std::ranges::any_of(program.info.images,
		                           [](const auto& image) { return image.written; });
	};
	if (context.GetOcclusionQueries().Counting()) {
		return false;
	}
	for (const auto& stage: vertex_stages) {
		if (writes_memory(stage.stage)) {
			return false;
		}
	}
	if (state.ps_active && writes_memory(state.ps_input_info.stage)) {
		return false;
	}
	for (uint32_t i = 0; i < targets.color_count; i++) {
		if (!targets.colors[i].desc.info.IsTiled()) {
			return false;
		}
	}
	return !static_cast<bool>(targets.depth.image_id) || targets.depth.desc.info.IsTiled();
}

static uint64_t DrawTemplateKey(uint64_t resource_key, const void* pipeline, const DrawTargets& targets) {
	std::array<uint64_t, 2 + RENDER_COLOR_ATTACHMENTS_MAX + 1> words {};
	size_t                                                     count = 0;
	words[count++] = resource_key;
	words[count++] = reinterpret_cast<uint64_t>(pipeline);
	for (uint32_t i = 0; i < targets.color_count; i++) {
		words[count++] = targets.colors[i].image_id.index;
	}
	words[count++] = targets.depth.image_id ? targets.depth.image_id.index + 1u : 0u;
	const auto key = XXH3_64bits(words.data(), count * sizeof(uint64_t));
	return key != 0 ? key : 1;
}

static uint64_t DrawStateTemplateKey(uint64_t state_key, const DrawTargets& targets, const DrawRenderState& state) {
	std::array<uint64_t, 2 + RENDER_COLOR_ATTACHMENTS_MAX + 1> words {};
	size_t                                                     count = 0;
	words[count++] = state_key;
	words[count++] = static_cast<uint64_t>(state.topology) | uint64_t {state.primitive_restart_enable} << 32u |
	                 uint64_t {state.ps_plain} << 33u | uint64_t {targets.color_count} << 40u;
	for (uint32_t i = 0; i < targets.color_count; i++) {
		words[count++] = targets.colors[i].image_id.index | uint64_t {targets.colors[i].target_slot} << 32u;
	}
	words[count++] = targets.depth.image_id ? targets.depth.image_id.index + 1u : 0u;
	const auto key = XXH3_64bits(words.data(), count * sizeof(uint64_t));
	return key != 0 ? key : 1;
}

void RenderExecutor::ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
                                         const DrawRenderState& state, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source) {
	auto&      ucfg                     = buffer.GetUserConfig();
	const bool primitive_restart_enable = state.primitive_restart_enable;
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto  vertex_stages = state.VertexStages();
	const bool  mesh_active   = state.vertex_info[0].stage.module->info.stage == ShaderType::Mesh;
	const bool  mesh_task     = state.vertex_info[0].stage.module->mesh_task;
	const auto& mesh          = state.vertex_info[0].mesh;
	const auto& limits        = m_context.GetGraphics().mesh_shader_properties;
	uint32_t    mesh_groups   = 0;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.indexed || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed);
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed, primitive_restart_enable);
		}
	}
	if (mesh_task) {
		EXIT_NOT_IMPLEMENTED(emit.indirect.count > std::min(limits.maxTaskWorkGroupCount[0],
		                                                    limits.maxTaskWorkGroupTotalCount));
	} else if (mesh_active) {
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    mesh_groups > limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

#if defined(ANYPS5_KYTY)
	if (state.ps_active && state.ps_input_info.target_packed_blend_mask != 0u) {
		m_draw_resource_key = 0;
	}
#endif
	std::array<const ShaderStageRuntime*, 4> stage_runtimes {};
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		stage_runtimes[i] = &state.vertex_info[i].stage;
	}
	const uint32_t runtime_count = static_cast<uint32_t>(vertex_stages.size()) + (state.ps_active ? 1u : 0u);
	if (state.ps_active) {
		stage_runtimes[runtime_count - 1] = &state.ps_input_info.stage;
	}
	const auto    runtimes        = std::span {stage_runtimes.data(), runtime_count};
	const bool    state_templates = DrawStateTemplatesEnabled();
	uint64_t      template_key    = 0;
	DrawTemplate* draw_template   = nullptr;
	DrawTemplate* state_template  = nullptr;
	if (state_templates && m_draw_resource_key != 0 && !mesh_active && DrawTemplateEligible(runtimes)) {
		template_key = DrawStateTemplateKey(m_draw_resource_key, *m_draw_targets, state);
		if (m_draw_templates != nullptr) {
			state_template = m_draw_templates->Find(template_key);
			if (state_template != nullptr && DrawTemplateStagesMatch(*state_template, runtimes)) {
				draw_template = state_template;
			}
		}
	}

	auto* const pipeline_ptr =
	    state_template != nullptr && state_template->pipeline != nullptr
	        ? static_cast<PipelineCache::Pipeline*>(state_template->pipeline)
	        : m_context.GetPipelineCache().GetGraphicsPipeline(
	              buffer, state, *m_draw_targets,
	              Config::AsyncPipelinesEnabled() &&
	                  DrawOnlyChangesPicture(m_context, state, *m_draw_targets, vertex_stages) &&
	                  !SmallColorTargetDraw(buffer.GetRegisters()));
	if (pipeline_ptr == nullptr) {
		return;
	}
	auto& pipeline = *pipeline_ptr;
	if (pipeline.uses_image_heap) {
		template_key   = 0;
		draw_template  = nullptr;
		state_template = nullptr;
	}

	uint64_t index_begin = 0;
	uint64_t index_end   = 0;
	if (mesh_active && draw.indexed) {
		const auto size = static_cast<uint64_t>(draw.index_count) * index_source.guest_element_size;
		index_begin     = index_source.address & ~uint64_t {3};
		index_end       = (index_source.address + size + 3u) & ~uint64_t {3};
		(void)m_context.GetBufferCache().FindBuffer(index_begin, index_end - index_begin);
	}
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	if (!state_templates && m_draw_resource_key != 0 && !pipeline.uses_image_heap &&
	    DrawTemplateEligible(runtimes)) {
		template_key  = DrawTemplateKey(m_draw_resource_key, pipeline_ptr, *m_draw_targets);
		draw_template = FindDrawTemplate(template_key, runtimes);
	}
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		if (draw_template != nullptr) {
			PrepareBindingsFromTemplate(state.vertex_info[i].stage, draw_template->stages[i],
			                            bindings.vertex[i]);
		} else {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		}
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		if (draw_template != nullptr) {
			PrepareBindingsFromTemplate(state.ps_input_info.stage, draw_template->stages[stage_count],
			                            bindings.pixel);
		} else {
			PrepareBindings(state.ps_input_info.stage, bindings.pixel);
		}
		descriptor_stages[stage_count++] = &bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {m_draw_targets->colors, m_draw_targets->color_count});
#if defined(ANYPS5_KYTY)
	m_packed_target_slots = 0;
	if (state.ps_active && state.ps_input_info.target_packed_blend_mask != 0u) {
		PreparePackedTargets(buffer, bindings.pixel, state.ps_input_info);
	}
#endif
	const bool template_held =
	    draw_template != nullptr && std::ranges::none_of(stages, [](const PreparedBindings* stage) {
		    return stage->template_stage == nullptr;
	    });
	const bool store_template =
	    template_key != 0 && (draw_template != nullptr ? !template_held : !DrawTemplateRefused(template_key));
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	vk::DeviceAddress     mesh_indices = 0;
	if (!mesh_active) {
		AcquireVertexBuffers(buffer, state.vertex_info[0], vertex_bindings);
		index_binding = PrepareIndexBuffer(buffer, index_source);
	} else if (draw.indexed) {
		const auto [indices, offset] =
		    m_context.GetBufferCache().ObtainBuffer(index_begin, index_end - index_begin, false);
		mesh_indices = indices->BufferDeviceAddress() + offset + (index_source.address & 3u);
	}
	std::pair<Buffer*, uint64_t> args_buffer {};
	std::pair<Buffer*, uint64_t> count_buffer {};
	if (emit.indirect.address != 0) {
		const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
		                  (draw.indexed ? sizeof(vk::DrawIndexedIndirectCommand)
		                                : sizeof(vk::DrawIndirectCommand));
		args_buffer = m_context.GetBufferCache().ObtainBuffer(emit.indirect.address, size, false);
		EXIT_IF((args_buffer.second & 3u) != 0);
		if (emit.indirect.count_address != 0) {
			count_buffer = m_context.GetBufferCache().ObtainBuffer(emit.indirect.count_address,
			                                                       sizeof(uint32_t), false);
			EXIT_IF(count_buffer.first == nullptr || (count_buffer.second & 3u) != 0);
		}
	}
	vk::ImageAspectFlags feedback_aspects;
	RenderState rendering_state;
	const bool targets_from_template = template_held && state_template != nullptr &&
	                                   state_template->targets_valid && TakeTemplateTargets(*state_template);
	if (targets_from_template) {
		rendering_state  = state_template->rendering;
		feedback_aspects = state_template->feedback_aspects;
	} else {
		rendering_state = AcquireRenderTargets(buffer, m_draw_targets->colors, m_draw_targets->color_count,
		                                       m_draw_targets->depth, feedback_aspects, stages);
	}
#if defined(ANYPS5_KYTY)
	m_packed_target_slots = 0;
#endif
	const auto& rendering = rendering_state;
	vk::DeviceAddress task_params = 0;
	if (mesh_task) {
		const MeshTaskParams params {
		    .records        = args_buffer.first->BufferDeviceAddress() + args_buffer.second,
		    .index_address  = mesh_indices,
		    .stride         = emit.indirect.stride,
		    .index_size     = index_source.guest_element_size,
		    .index_limit    = draw.index_count,
		    .primitive_size = mesh.InputPrimitiveSize(),
		    .group_step     = mesh.primitives_per_group * mesh.InputPrimitiveStep(),
		    .max_groups     = limits.maxMeshWorkGroupCount[0],
		    .max_instances  = limits.maxMeshWorkGroupCount[1],
		    .max_total      = limits.maxMeshWorkGroupTotalCount,
		};
		auto& stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		task_params  = stream.BufferDeviceAddress() + stream.Copy(&params, sizeof(params), 8);
	}

	m_context.GetOcclusionQueries().WriteForReaders();
	if (state.ps_active) {
		PrepareDrawRayTracing(m_context, buffer, bindings.pixel);
	}
	auto vk_buffer = buffer.Handle();
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (template_held && state_template != nullptr && state_template->commit_valid &&
	    state_template->commit_layout == pipeline.pipeline_layout) {
		CommitTemplateBindings(buffer, pipeline, stages, *state_template);
	} else {
		m_commit_capture =
		    template_held && state_template != nullptr && TemplateCommitsSupported() ? state_template : nullptr;
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
		m_commit_capture = nullptr;
	}
	if (store_template) {
		StoreDrawTemplate(template_key, stages);
	}
	if (state_templates && template_key != 0) {
		auto* stored = state_template != nullptr && !store_template ? state_template
		               : m_draw_templates != nullptr                ? m_draw_templates->Find(template_key)
		                                                            : nullptr;
		if (stored != nullptr) {
			stored->pipeline = pipeline_ptr;
			const bool images_held = template_held || (store_template && stored->stage_count == stages.size());
			if (!targets_from_template) {
				if (m_targets_from_memo && images_held) {
					StoreTemplateTargets(*stored, rendering_state, feedback_aspects);
				} else {
					stored->targets_valid = false;
				}
			}
		}
	}
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	using ShaderRecompiler::IR::PushData;
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_task) {
			draw_data[0] = static_cast<uint32_t>(task_params);
			draw_data[1] = static_cast<uint32_t>(task_params >> 32u);
		} else if (mesh_active) {
			draw_data = {draw.index_count,
			             static_cast<uint32_t>(emit.vertex_offset),
			             index_source.guest_element_size,
			             static_cast<uint32_t>(mesh_indices),
			             static_cast<uint32_t>(mesh_indices >> 32u),
			             0u};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), m_draw_targets->depth,
	                           rendering);
	const bool query_bias = m_context.GetOcclusionQueries().Counting() &&
	                        dynamic_state.BiasTowardViewer(OcclusionCounters::ProxyBiasUnits,
	                                                       OcclusionCounters::ProxyBiasSlope);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);
	dynamic_state.MoveGeneration();

	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages && !buffer.HasPendingShaderWrites()) {
		buffer.EndRendering();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
	}
	buffer.BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
#if defined(ANYPS5_KYTY)
	const bool gpu_time = GpuTime::Enabled();
	if (gpu_time) {
		GpuTime::BeforeWork(buffer);
	}
#endif
	if (mesh_task) {
		vk_buffer.drawMeshTasksEXT(emit.indirect.count, 1, 1);
	} else if (mesh_active) {
		const auto  rows   = MeshDrawSplit::RowsPerDraw(limits.maxMeshWorkGroupCount[1],
		                                                limits.maxMeshWorkGroupTotalCount, mesh_groups);
		MeshDrawSplit::ForEachDraw(draw.instance_count, rows, [&](uint32_t first, uint32_t count) {
			if (first != 0) {
				vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages,
				                        (PushData::MeshDrawDwordCount - 1u) * sizeof(uint32_t),
				                        sizeof(uint32_t), &first);
			}
			vk_buffer.drawMeshTasksEXT(mesh_groups, count, 1);
		});
	} else if (args_buffer.first == nullptr) {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	} else if (count_buffer.first != nullptr) {
		if (draw.indexed) {
			vk_buffer.drawIndexedIndirectCount(args_buffer.first->Handle(), args_buffer.second,
			                                   count_buffer.first->Handle(), count_buffer.second,
			                                   emit.indirect.count, emit.indirect.stride);
		} else {
			vk_buffer.drawIndirectCount(args_buffer.first->Handle(), args_buffer.second,
			                            count_buffer.first->Handle(), count_buffer.second,
			                            emit.indirect.count, emit.indirect.stride);
		}
	} else if (draw.indexed) {
		vk_buffer.drawIndexedIndirect(args_buffer.first->Handle(), args_buffer.second,
		                              emit.indirect.count, emit.indirect.stride);
	} else {
		vk_buffer.drawIndirect(args_buffer.first->Handle(), args_buffer.second,
		                       emit.indirect.count, emit.indirect.stride);
	}
#if defined(ANYPS5_KYTY)
	if (gpu_time) {
		GpuTime::AfterDraw(buffer, state.vertex_info[0].stage.module->info.shader_hash,
		                   state.ps_active ? state.ps_input_info.stage.module->info.shader_hash : 0);
	}
#endif
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (shader_write_stages) {
		buffer.DeferShaderWriteBarrier(shader_write_stages);
		m_shader_write_queue    = m_guest_queue;
		m_shader_write_deferred = true;
	}
	if (query_bias) {
		dynamic_state.SetDepthBiasEnabled(false);
	}
	if (std::exchange(m_table_store, false)) {
		TableStore(buffer, state, state.topology, primitive_restart_enable, rendering, dynamic_state,
		           mesh_active                                                          ? "unstorable: mesh shader"
		           : !draw.indexed                                                      ? "unstorable: not indexed"
		           : emit.indirect.address == 0 && index_source.host_data != nullptr ? "unstorable: host indices"
		           : draw_dwords != 0                                                   ? "unstorable: draw data"
		           : vertex_bindings.count != 0                                         ? "unstorable: vertex buffers"
		                                                                                : nullptr);
	}
}

static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource             source,
                                                   int32_t                      first_vertex,
                                                   uint32_t                     first_instance,
                                                   const DrawIndirectArgs&      indirect) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance} : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

namespace {
struct TableNormalDraw {
	RenderExecutor&         executor;
	const DrawIndirectArgs& indirect;
	const bool              indexed;
	const uint64_t          started;
	~TableNormalDraw() { executor.TableNoteNormalDraw(indirect, indexed, started); }
};
}

bool RenderExecutor::DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	const TableNormalDraw table_normal {*this, args.indirect, true, TableCount::TableSampleStart()};

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		InvalidateRenderTargetMemo();
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	if (m_context_id == 0 || m_context_id != m_checked_context_id ||
	    graphics_debug_dump_enabled()) {
		hw_check(buffer);
		m_checked_context_id = m_context_id;
	}

	bool  shaders_resolved = false;
	auto& state = AcquireDrawRenderState(ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch,
	                                     shaders_resolved, args.indirect.offsets,
	                                     args.indirect.address != 0);
	state.indirect_offsets = args.indirect.offsets;
	state.indirect_draw    = args.indirect.address != 0;
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16: index_source.guest_element_size = 2; break;
		case Prospero::IndexType::kIndex32: index_source.guest_element_size = 4; break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	state.primitive_restart_enable = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {true, args.index_count, args.instance_count};
	if (!PrepareDrawRenderState(buffer, shaders_resolved, draw, state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, *m_draw_targets, args.index_type_and_size, args.index_addr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  args.base_vertex, args.first_instance, args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, index_source);
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	const TableNormalDraw table_normal {*this, args.indirect, false, TableCount::TableSampleStart()};

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.indirect.address == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		InvalidateRenderTargetMemo();
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	if (m_context_id == 0 || m_context_id != m_checked_context_id ||
	    graphics_debug_dump_enabled()) {
		hw_check(buffer);
		m_checked_context_id = m_context_id;
	}

	const DrawCallInfo draw {false, args.vertex_count, args.instance_count};

	bool  shaders_resolved = false;
	auto& state = AcquireDrawRenderState(ucfg.GetPrimType() == Prospero::PrimitiveType::kPatch,
	                                     shaders_resolved, args.indirect.offsets,
	                                     args.indirect.address != 0);
	state.indirect_offsets = args.indirect.offsets;
	state.indirect_draw    = args.indirect.address != 0;
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, shaders_resolved, draw, state)) {
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, *m_draw_targets, 0, nullptr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  static_cast<int32_t>(args.first_vertex), args.first_instance,
	                                  args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, {});
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, 1, true, true);
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

#include "graphics/host_gpu/renderer/nativeXpr.inc"
#include "graphics/host_gpu/renderer/tableXpr.inc"

}
