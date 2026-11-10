#include "graphics/host_gpu/renderer/renderDrawTrace.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cleanDraws.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderDrawPlan.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

void TraceDrawDrop(const char* reason, [[maybe_unused]] const CommandBuffer& buffer) {
	// KYTY_TRACE_CLEAN_DRAWS: a metadata / copy / resolve operation ends a series of clean draws.
	if (std::strncmp(reason, "metadata", 8) == 0) {
		CleanDraws::g_meta_ops.fetch_add(1, std::memory_order_relaxed);
		CleanDraws::NoteBreak(CleanDraws::BreakMetaOp);
	}
}

void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
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

void TraceDrawStatePs(const CommandBuffer& buffer, const DrawRenderState& state, const DrawCallInfo& draw) {
	struct Spec {
		std::string list;
		uint32_t    from = 0;
	};
	static const Spec spec = [] {
		Spec        result;
		const char* value = std::getenv("KYTY_TRACE_DRAW_STATE_PS");
		if (value == nullptr) return result;
		result.list = value;
		if (const auto colon = result.list.find(':'); colon != std::string::npos) {
			result.from = static_cast<uint32_t>(std::strtoul(result.list.c_str() + colon + 1, nullptr, 10));
			result.list.resize(colon);
		}
		return result;
	}();
	if (spec.list.empty() || !state.ps_active || DebugPresentCount().load() < spec.from) return;
	const auto ps_hash = state.ps_input_info.stage.program->shader_hash;
	if (spec.list.find(fmt::format("{:016x}", ps_hash)) == std::string::npos) return;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1) >= 12u) return;
	const auto& ctx     = buffer.GetRegisters();
	const auto& sh_regs = ctx.GetShaderRegisters();
	const auto& dc      = ctx.GetDepthControl();
	const auto& mc      = ctx.GetModeControl();
	const auto& cc      = ctx.GetColorControl();
	const auto& vp0     = ctx.GetScreenViewport().viewports[0];
	std::fprintf(stderr,
	             "DRAW_STATE frame=%u ps=%016llx %s count=%u inst=%u rt_mask=0x%08x cb_shader_mask=0x%08x cc_mode=%u cc_op=0x%02x "
	             "ps_kill=%d out_mode=%u,%u colors=%u depth=0x%llx z=%d zw=%d zfunc=%u stencil=%d sfunc=%u bounds=%d "
	             "cull_front=%d cull_back=%d face=%d viewport=(%.1f,%.1f %.1fx%.1f z %.3f..%.3f) prim=%u\n",
	             DebugPresentCount().load(), static_cast<unsigned long long>(ps_hash), draw.Name(), draw.index_count,
	             draw.instance_count, ctx.GetRenderTargetMask(), sh_regs.m_cbShaderMask, cc.mode, cc.op,
	             state.ps_input_info.ps_pixel_kill_enable ? 1 : 0, state.ps_input_info.target_output_mode[0],
	             state.ps_input_info.target_output_mode[1], state.color_count,
	             static_cast<unsigned long long>(state.depth_info.image_id ? state.depth_info.desc.info.data.address : 0ull),
	             dc.z_enable ? 1 : 0, dc.z_write_enable ? 1 : 0, dc.zfunc, dc.stencil_enable ? 1 : 0, dc.stencilfunc,
	             dc.depth_bounds_enable ? 1 : 0, mc.cull_front ? 1 : 0, mc.cull_back ? 1 : 0, mc.face ? 1 : 0,
	             vp0.xoffset - vp0.xscale, vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, vp0.zoffset,
	             vp0.zoffset + vp0.zscale, static_cast<uint32_t>(buffer.GetUserConfig().GetPrimType()));
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto& color = state.color_info[i];
		const auto& bc    = ctx.GetBlendControl(color.target_slot);
		std::fprintf(stderr,
		             "DRAW_STATE   color%u slot=%u addr=0x%llx %ux%u fmt=%s blend=%d src=%u dst=%u comb=%u asrc=%u adst=%u "
		             "acomb=%u sep=%d target_mask=0x%x shader_mask=0x%x\n",
		             i, color.target_slot, static_cast<unsigned long long>(color.desc.info.data.address),
		             color.desc.info.extent.width, color.desc.info.extent.height,
		             vk::to_string(color.desc.view_info.format).c_str(), bc.enable ? 1 : 0, bc.color_srcblend,
		             bc.color_destblend, bc.color_comb_fcn, bc.alpha_srcblend, bc.alpha_destblend, bc.alpha_comb_fcn,
		             bc.separate_alpha_blend ? 1 : 0, (ctx.GetRenderTargetMask() >> (color.target_slot * 4u)) & 0xfu,
		             (sh_regs.m_cbShaderMask >> (color.target_slot * 4u)) & 0xfu);
	}
}

void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
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
	const auto& ps_resources   = ps_input_info.stage.program->info;
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

void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
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

void DrawCensusScope::Finish() {
	const auto now = std::chrono::steady_clock::now();
	t_dispatch_sub_on = false;
	struct Sums {
		uint64_t                                                        draws = 0, ns = 0;
		std::array<uint64_t, static_cast<size_t>(DispatchSub::Count)> sub {};
		uint32_t                                                        frame_start = 0;
		std::chrono::steady_clock::time_point                           window      = std::chrono::steady_clock::now();
	};
	static Sums s;
	s.draws++;
	s.ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
	for (size_t i = 0; i < s.sub.size(); i++) s.sub[i] += t_dispatch_sub_ns[i];
	if (now - s.window < std::chrono::seconds(5)) return;
	const uint32_t frame = DebugPresentCount().load();
	if (s.frame_start != 0 && frame > s.frame_start) {
		const double n  = static_cast<double>(s.draws);
		const auto   us = [&](DispatchSub k) { return static_cast<double>(s.sub[static_cast<size_t>(k)]) / 1e3 / n; };
		std::fprintf(stderr,
		             "PERF draw_census draws/frame=%.0f total_us=%.2f | front: pre=%.2f render_state=%.2f (programs=%.2f "
		             "targets=%.2f) pipeline=%.2f plan_setup=%.2f | wait=%.2f post=%.2f | commit=%.2f (replay=%.2f "
		             "acquire=%.2f vtx=%.2f bind=%.2f [pre=%.2f scan=%.2f desc=%.2f transit=%.2f emit=%.2f] index=%.2f dyn=%.2f) | "
		             "programs: lookup=%.2f take=%.2f mr=%.2f perm=%.2f stages/draw resolved=%.2f mr=%.2f\n",
		             n / static_cast<double>(frame - s.frame_start), static_cast<double>(s.ns) / 1e3 / n,
		             us(DispatchSub::DrawPre), us(DispatchSub::DrawRenderState), us(DispatchSub::DrawPrograms),
		             us(DispatchSub::DrawTargets), us(DispatchSub::DrawPipeline), us(DispatchSub::DrawPlanSetup),
		             us(DispatchSub::DrawWait), us(DispatchSub::DrawPost), us(DispatchSub::DrawCommit),
		             us(DispatchSub::CReplay), us(DispatchSub::CAcquire), us(DispatchSub::CVtx), us(DispatchSub::CBind),
		             us(DispatchSub::BPre), us(DispatchSub::BScan), us(DispatchSub::BDesc), us(DispatchSub::BTransit), us(DispatchSub::BEmit),
		             us(DispatchSub::CIndex), us(DispatchSub::CDyn), us(DispatchSub::GLookup), us(DispatchSub::GTake),
		             us(DispatchSub::GMr), us(DispatchSub::GPerm),
		             static_cast<double>(s.sub[static_cast<size_t>(DispatchSub::GPathResolved)]) / n,
		             static_cast<double>(s.sub[static_cast<size_t>(DispatchSub::GPathMr)]) / n);
	}
	const auto window = now;
	s                 = {};
	s.frame_start     = frame;
	s.window          = window;
}

// KYTY_TRACE_CLEAN_DRAWS (renderDrawTrace.h).
namespace C0 {
enum : uint32_t {
	// Elements deciding clean (first reason in this order).
	EBreak, EPrograms, EPipeline, ETargets, EDynamic, EImages, ESamplers,
	// Information.
	EOtherRegs, EBuffers, EUserData, EGeometry, EDispatch, EAsync, EMetaOp, ECmdBuf, EFirst, EObserverClean, EObserverWrong,
	// Register groups.
	RProg, RUdLayout, RRt, RBlend, RDepth, RRaster, RShIf, RViewport, RBlendColor, RStencil, RBounds, RPoly, RMisc, RUcfg,
	Count
};
constexpr uint64_t Bit(uint32_t e) { return uint64_t {1} << e; }
constexpr uint64_t DecideMask = Bit(EBreak) | Bit(EPrograms) | Bit(EPipeline) | Bit(ETargets) | Bit(EDynamic) | Bit(EImages) |
                                Bit(ESamplers);
constexpr uint64_t RegMask    = Bit(RProg) | Bit(RUdLayout) | Bit(RRt) | Bit(RBlend) | Bit(RDepth) | Bit(RRaster) | Bit(RShIf) |
                             Bit(RViewport) | Bit(RBlendColor) | Bit(RStencil) | Bit(RBounds) | Bit(RPoly) | Bit(RMisc) |
                             Bit(RUcfg);
constexpr const char* Names[Count] = {"break", "programs", "pipeline", "targets", "dynamic", "images", "samplers",
                                      "other_regs", "buffers", "user_data", "geometry", "dispatch", "async_dispatch", "meta_op", "cmd_buf",
                                      "first", "observer_clean", "observer_wrong", "r_prog", "r_ud_layout", "r_rt", "r_blend", "r_depth", "r_raster", "r_sh_if",
                                      "r_viewport", "r_blend_color", "r_stencil", "r_bounds", "r_poly", "r_misc", "r_ucfg"};
template <typename T>
bool Same(const T& a, const T& b) {
	return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// The plan's part (GPU thread): register groups, programs and pipeline against the previous plan's.
void Plan(DrawPlan& plan, const CommandBuffer& buffer, const DrawRenderState& state,
          std::span<ShaderVertexInputInfo> vertex_stages, const PipelineCache::Pipeline* pipeline) {
	struct Previous {
		bool                                                 valid = false;
		HW::Context                                          ctx;
		HW::UserConfig                                       ucfg;
		HW::Shader                                           sh;
		std::array<const void*, 4>                           programs {};
		const PipelineCache::Pipeline*                       pipeline = nullptr;
	};
	static Previous p;
	const auto&     ctx  = buffer.GetRegisters();
	const auto&     ucfg = buffer.GetUserConfig();
	const auto&     sh   = buffer.GetShaders();
	std::array<const void*, 4> programs {};
	for (size_t i = 0; i < vertex_stages.size() && i < 3u; i++) programs[i] = vertex_stages[i].stage.program;
	programs[3] = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	uint64_t m  = 0;
	if (!p.valid) {
		m |= Bit(EBreak) | Bit(EFirst);
	} else {
		const auto& pc = p.ctx;
		const auto& pv = p.sh.GetVs();
		const auto& v  = sh.GetVs();
		const auto& pp = p.sh.GetPs();
		const auto& ps = sh.GetPs();
		if (programs != p.programs) m |= Bit(EPrograms);
		if (pipeline != p.pipeline) m |= Bit(EPipeline);
		if (v.es_regs.data_addr != pv.es_regs.data_addr || v.ls_regs.data_addr != pv.ls_regs.data_addr ||
		    v.hs_regs.data_addr != pv.hs_regs.data_addr || !Same(v.hs_regs.rsrc1, pv.hs_regs.rsrc1) ||
		    !Same(v.hs_regs.rsrc2, pv.hs_regs.rsrc2) || v.gs_regs.data_addr != pv.gs_regs.data_addr ||
		    !Same(v.gs_regs.rsrc1, pv.gs_regs.rsrc1) || !Same(v.gs_regs.rsrc2, pv.gs_regs.rsrc2) ||
		    ps.ps_regs.data_addr != pp.ps_regs.data_addr || !Same(ps.ps_regs.rsrc1, pp.ps_regs.rsrc1) ||
		    !Same(ps.ps_regs.rsrc2, pp.ps_regs.rsrc2)) {
			m |= Bit(RProg);
		}
		const auto layout_same = [](const HW::UserSgprInfo& a, const HW::UserSgprInfo& b) {
			return a.count == b.count && Same(a.type, b.type);
		};
		const auto values_same = [](const HW::UserSgprInfo& a, const HW::UserSgprInfo& b) { return Same(a.value, b.value); };
		if (!layout_same(v.hs_user_sgpr, pv.hs_user_sgpr) || !layout_same(v.gs_user_sgpr, pv.gs_user_sgpr) ||
		    !layout_same(ps.ps_user_sgpr, pp.ps_user_sgpr)) {
			m |= Bit(RUdLayout);
		}
		if (!values_same(v.hs_user_sgpr, pv.hs_user_sgpr) || !values_same(v.gs_user_sgpr, pv.gs_user_sgpr) ||
		    !values_same(ps.ps_user_sgpr, pp.ps_user_sgpr) || v.hs_regs.user_data_addr != pv.hs_regs.user_data_addr ||
		    v.gs_regs.user_data_addr != pv.gs_regs.user_data_addr || ps.ps_regs.user_data_addr != pp.ps_regs.user_data_addr ||
		    ps.ps_regs.auxiliary_table_addr != pp.ps_regs.auxiliary_table_addr) {
			m |= Bit(EUserData);
		}
		bool rt = ctx.GetRenderTargetMask() == pc.GetRenderTargetMask() && Same(ctx.GetDepthRenderTarget(), pc.GetDepthRenderTarget());
		bool blend = true;
		for (uint32_t i = 0; i < 8u; i++) {
			rt    = rt && Same(ctx.GetRenderTarget(i), pc.GetRenderTarget(i));
			blend = blend && Same(ctx.GetBlendControl(i), pc.GetBlendControl(i));
		}
		if (!rt) m |= Bit(RRt);
		if (!blend) m |= Bit(RBlend);
		if (!Same(ctx.GetDepthControl(), pc.GetDepthControl()) || !Same(ctx.GetStencilControl(), pc.GetStencilControl()) ||
		    !Same(ctx.GetRenderControl(), pc.GetRenderControl()) ||
		    !Same(ctx.GetDepthRenderOverride(), pc.GetDepthRenderOverride()) ||
		    ctx.GetDepthClearValue() != pc.GetDepthClearValue() || ctx.GetStencilClearValue() != pc.GetStencilClearValue()) {
			m |= Bit(RDepth);
		}
		if (!Same(ctx.GetClipControl(), pc.GetClipControl()) || !Same(ctx.GetModeControl(), pc.GetModeControl()) ||
		    !Same(ctx.GetEqaaControl(), pc.GetEqaaControl()) || !Same(ctx.GetScanModeControl(), pc.GetScanModeControl()) ||
		    !Same(ctx.GetAaSampleControl(), pc.GetAaSampleControl()) || !Same(ctx.GetAaConfig(), pc.GetAaConfig()) ||
		    !Same(ctx.GetColorControl(), pc.GetColorControl())) {
			m |= Bit(RRaster);
		}
		if (!Same(ctx.GetShaderRegisters(), pc.GetShaderRegisters()) || ctx.GetShaderStages() != pc.GetShaderStages()) {
			m |= Bit(RShIf);
		}
		if (!Same(ctx.GetScreenViewport(), pc.GetScreenViewport())) m |= Bit(RViewport);
		if (!Same(ctx.GetBlendColor(), pc.GetBlendColor())) m |= Bit(RBlendColor);
		if (!Same(ctx.GetStencilMask(), pc.GetStencilMask())) m |= Bit(RStencil);
		if (ctx.GetDepthBoundsMin() != pc.GetDepthBoundsMin() || ctx.GetDepthBoundsMax() != pc.GetDepthBoundsMax()) {
			m |= Bit(RBounds);
		}
		if (!Same(ctx.GetPolyOffset(), pc.GetPolyOffset()) || ctx.GetLineWidth() != pc.GetLineWidth()) m |= Bit(RPoly);
		if (ctx.GetPrimitiveResetIndex() != pc.GetPrimitiveResetIndex() ||
		    ctx.GetDrawPayloadControl() != pc.GetDrawPayloadControl() ||
		    ctx.GetObjprimIdControl() != pc.GetObjprimIdControl()) {
			m |= Bit(RMisc);
		}
		const auto& pu = p.ucfg;
		if (ucfg.GetPrimType() != pu.GetPrimType() || ucfg.GetPrimitiveResetControl() != pu.GetPrimitiveResetControl() ||
		    !Same(ucfg.GetGeControl(), pu.GetGeControl()) || !Same(ucfg.GetGeUserVgprEn(), pu.GetGeUserVgprEn()) ||
		    !Same(ucfg.GetGdsOaState(), pu.GetGdsOaState()) || !Same(ucfg.GetFsrView(), pu.GetFsrView())) {
			m |= Bit(RUcfg);
		}
		if (ucfg.GetIndexOffset() != pu.GetIndexOffset() || ucfg.GetObjectId() != pu.GetObjectId()) m |= Bit(EGeometry);
		if ((m & Bit(RRt)) != 0) m |= Bit(ETargets);
		if ((m & (Bit(RViewport) | Bit(RBlendColor) | Bit(RStencil) | Bit(RBounds) | Bit(RPoly))) != 0) m |= Bit(EDynamic);
		if ((m & (Bit(RProg) | Bit(RUdLayout) | Bit(RBlend) | Bit(RDepth) | Bit(RRaster) | Bit(RShIf) | Bit(RMisc) | Bit(RUcfg))) != 0) {
			m |= Bit(EOtherRegs);
		}
	}
	// The draw-state observer's answer; clean while a register group changed is a fault of the observer.
	if (t_draw_observer_clean) {
		m |= Bit(EObserverClean);
		if ((m & RegMask) != 0 || (m & Bit(EFirst)) != 0) m |= Bit(EObserverWrong);
	}
	const uint32_t breaks = CleanDraws::g_breaks.exchange(0, std::memory_order_relaxed);
	if ((breaks & CleanDraws::BreakDispatch) != 0) m |= Bit(EBreak) | Bit(EDispatch);
	if ((breaks & CleanDraws::BreakMetaOp) != 0) m |= Bit(EBreak) | Bit(EMetaOp);
	if ((breaks & CleanDraws::BreakAsync) != 0) m |= Bit(EBreak) | Bit(EAsync);
	p.valid    = true;
	p.ctx      = ctx;
	p.ucfg     = ucfg;
	p.sh       = sh;
	p.programs = programs;
	p.pipeline = pipeline;
	plan.c0_mask     = m;
	plan.c0_front    = false;
	plan.c0_front_ns = 0;
	plan.c0_wait_ns  = 0;
	plan.c0_prep_ns  = 0;
	plan.c0_cause    = t_draw_observer_cause;
	plan.c0_programs_ns = CleanDraws::t_front_parts.programs_ns;
	plan.c0_targets_ns  = CleanDraws::t_front_parts.targets_ns;
	plan.c0_target_hit  = CleanDraws::t_front_parts.target_hit;
	plan.c0_pipeline_ns = CleanDraws::t_front_parts.pipeline_ns;
	CleanDraws::t_front_parts = {};
	t_current.plan   = &plan;
}

struct Stats {
	uint64_t draws = 0, clean = 0, strict = 0, state_clean = 0;
	std::array<uint64_t, Count> any {};   // draws with each change
	std::array<uint64_t, Count> first {}; // not clean draws by their first deciding change
	// Series: a not clean draw followed by k clean ones (k buckets 0, 1, 2-3, 4-7, 8-15, 16-63, 64+): series and draws.
	std::array<uint64_t, 7> series {};
	std::array<uint64_t, 7> series_draws {};
	uint64_t                run = 0;
	bool                    in_series = false;
	std::array<uint64_t, 7> strict_series {};
	uint64_t                strict_run = 0;
	bool                    strict_in  = false;
	// Per kind: draws, draws with a measured front; ns of the front (without the wait), the wait, the commit, the job.
	std::array<uint64_t, KindCount> n {}, n_front {}, front {}, wait {}, commit {}, prep {};
	std::array<uint64_t, KindCount> programs {}, targets {}, pipeline {}; // ns of the front's parts (CleanDraws::FrontParts)
	std::array<uint64_t, KindCount> target_hits {}, target_hit_ns {};     // of them, the target searches the memo answered
	// Strict clean draws the observer called dirty, by its cause of dirt (step K1).
	uint64_t                               missed = 0;
	std::unordered_map<uint32_t, uint64_t> missed_by;
	uint32_t                        frame_start = 0;
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
inline Stats g_stats;

uint32_t Bucket(uint64_t k) {
	return k == 0 ? 0u : k == 1 ? 1u : k < 4 ? 2u : k < 8 ? 3u : k < 16 ? 4u : k < 64 ? 5u : 6u;
}

uint64_t Mix(uint64_t h, uint64_t v) {
	return (h ^ v) * 0x100000001b3ull;
}

void Report(Stats& s, uint32_t frame) {
	const double frames = static_cast<double>(frame - s.frame_start);
	const double n      = static_cast<double>(s.draws);
	const auto   pct    = [&](uint64_t v) { return 100.0 * static_cast<double>(v) / n; };
	std::string  any, first, groups;
	for (uint32_t e = 0; e < Count; e++) {
		if (s.any[e] == 0) continue;
		auto& out = e >= RProg ? groups : any;
		out += fmt::format(" {}={:.1f}", Names[e], pct(s.any[e]));
		if ((DecideMask & Bit(e)) != 0 && s.first[e] != 0) first += fmt::format(" {}={:.1f}", Names[e], pct(s.first[e]));
	}
	std::string series, strict_series;
	for (size_t b = 0; b < s.series.size(); b++) {
		series += fmt::format(" {}/{:.1f}%", s.series[b], pct(s.series_draws[b]));
		strict_series += fmt::format(" {}", s.strict_series[b]);
	}
	const char* kinds[KindCount] = {"clean", "images_only", "rest"};
	std::string times;
	for (uint32_t k = 0; k < KindCount; k++) {
		const double c  = static_cast<double>(std::max<uint64_t>(s.n[k], 1));
		const double cf = static_cast<double>(std::max<uint64_t>(s.n_front[k], 1));
		times += fmt::format(" | {} n/frame={:.0f} front={:.2f} wait={:.2f} commit={:.2f} prep={:.2f} us (front measured {:.0f}%)"
		                     " programs={:.2f} targets={:.2f} (memo hit {:.0f}% at {:.2f}, miss at {:.2f}) pipeline={:.2f} us",
		                     kinds[k], static_cast<double>(s.n[k]) / frames, static_cast<double>(s.front[k]) / 1e3 / cf,
		                     static_cast<double>(s.wait[k]) / 1e3 / cf, static_cast<double>(s.commit[k]) / 1e3 / c,
		                     static_cast<double>(s.prep[k]) / 1e3 / c, 100.0 * static_cast<double>(s.n_front[k]) / c,
		                     static_cast<double>(s.programs[k]) / 1e3 / c, static_cast<double>(s.targets[k]) / 1e3 / c,
		                     100.0 * static_cast<double>(s.target_hits[k]) / c,
		                     static_cast<double>(s.target_hit_ns[k]) / 1e3 / static_cast<double>(std::max<uint64_t>(s.target_hits[k], 1)),
		                     static_cast<double>(s.targets[k] - s.target_hit_ns[k]) / 1e3 /
		                         static_cast<double>(std::max<uint64_t>(s.n[k] - s.target_hits[k], 1)),
		                     static_cast<double>(s.pipeline[k]) / 1e3 / c);
	}
	std::fprintf(stderr,
	             "PERF clean_draws draws/frame=%.0f clean=%.1f%% strict=%.1f%% state_clean=%.1f%% | first:%s | any:%s | regs:%s | "
	             "series(k=0,1,2-3,4-7,8-15,16-63,64+: series/draws%%):%s | strict series:%s%s\n",
	             n / frames, pct(s.clean), pct(s.strict), pct(s.state_clean), first.c_str(), any.c_str(), groups.c_str(),
	             series.c_str(), strict_series.c_str(), times.c_str());
	// The observer's misses: strict clean draws it called dirty, by the first cause of dirt (kind:detail, % of all draws).
	std::vector<std::pair<uint64_t, uint32_t>> by;
	for (const auto& [cause, count]: s.missed_by) by.emplace_back(count, cause);
	std::sort(by.begin(), by.end(), std::greater<>());
	constexpr const char* cause_kinds[] = {"none", "cx", "sh", "uc", "beyond", "packet", "nop", "predicated", "invalidate",
	                                       "nop_reset", "indirect_draw", "null_block", "short_packet"};
	std::string causes;
	for (size_t i = 0; i < by.size() && i < 16u; i++) {
		const uint32_t kind = by[i].second >> 16u;
		causes += fmt::format(" {}:0x{:x}={:.1f}", kind < std::size(cause_kinds) ? cause_kinds[kind] : "?", by[i].second & 0xffffu,
		                      pct(by[i].first));
	}
	std::fprintf(stderr, "PERF clean_draws observer_missed=%.1f%% (strict clean, observer dirty) causes:%s\n", pct(s.missed),
	             causes.c_str());
}

// The commit's part (the committing thread, in draw order): images, samplers, buffers and command buffer against the
// previous commit's; the draw's kind, the statistics. Returns the kind (for its commit time).
uint8_t Commit(DrawPlan& plan, std::span<PreparedBindings* const> stages, vk::CommandBuffer handle) {
	static uint64_t prev_images = 0, prev_samplers = 0, prev_buffers = 0, prev_cmd = 0;
	uint64_t images = 0xcbf29ce484222325ull, samplers = images, buffers = images;
	for (const auto* stage: stages) {
		images   = Mix(images, stage->images.size() | (stage->bindless_images.size() << 32u));
		samplers = Mix(samplers, stage->samplers.size());
		buffers  = Mix(buffers, stage->buffers.size());
		for (const auto& t: stage->images) {
			images = Mix(Mix(images, std::bit_cast<uint64_t>(static_cast<VkImageView>(t.image_view))), static_cast<uint64_t>(t.layout));
		}
		for (const auto& id: stage->bindless_images) images = Mix(images, (uint64_t {id.index} << 32u) | id.generation);
		for (const auto& smp: stage->samplers) samplers = Mix(samplers, std::bit_cast<uint64_t>(static_cast<VkSampler>(smp)));
		for (const auto& b: stage->buffers) {
			buffers = Mix(Mix(Mix(buffers, std::bit_cast<uint64_t>(static_cast<VkBuffer>(b.buffer))), b.offset), b.range);
		}
	}
	const uint64_t cmd = std::bit_cast<uint64_t>(static_cast<VkCommandBuffer>(handle));
	uint64_t       m   = plan.c0_mask;
	if (images != prev_images) m |= Bit(EImages);
	if (samplers != prev_samplers) m |= Bit(ESamplers);
	if (buffers != prev_buffers) m |= Bit(EBuffers);
	if (cmd != prev_cmd) m |= Bit(EBreak) | Bit(ECmdBuf);
	prev_images = images, prev_samplers = samplers, prev_buffers = buffers, prev_cmd = cmd;

	auto&      s           = g_stats;
	const bool clean       = (m & DecideMask) == 0;
	const bool strict      = clean && (m & RegMask) == 0;
	const bool state_clean = (m & (DecideMask & ~(Bit(EImages) | Bit(ESamplers)))) == 0;
	s.draws++;
	s.clean += clean ? 1u : 0u;
	s.strict += strict ? 1u : 0u;
	s.state_clean += state_clean ? 1u : 0u;
	for (uint32_t e = 0; e < Count; e++) {
		if ((m & Bit(e)) != 0) s.any[e]++;
	}
	if (!clean) s.first[static_cast<uint32_t>(std::countr_zero(m & DecideMask))]++;
	if (clean) {
		s.run++;
	} else {
		if (s.in_series) {
			s.series[Bucket(s.run)]++;
			s.series_draws[Bucket(s.run)] += s.run + 1u;
		}
		s.in_series = true;
		s.run       = 0;
	}
	if (strict) {
		s.strict_run++;
	} else {
		if (s.strict_in) s.strict_series[Bucket(s.strict_run)]++;
		s.strict_in  = true;
		s.strict_run = 0;
	}
	if (strict && (m & Bit(EObserverClean)) == 0) {
		s.missed++;
		s.missed_by[plan.c0_cause]++;
	}
	const uint8_t kind = clean ? KClean : state_clean ? KImagesOnly : KRest;
	s.n[kind]++;
	s.prep[kind] += plan.c0_prep_ns;
	s.programs[kind] += plan.c0_programs_ns;
	s.targets[kind] += plan.c0_targets_ns;
	if (plan.c0_target_hit) {
		s.target_hits[kind]++;
		s.target_hit_ns[kind] += plan.c0_targets_ns;
	}
	s.pipeline[kind] += plan.c0_pipeline_ns;
	if (plan.c0_front) {
		s.n_front[kind]++;
		s.front[kind] += plan.c0_front_ns;
		s.wait[kind] += plan.c0_wait_ns;
	} else if (t_current.plan == &plan) {
		t_current.plan      = nullptr; // committed inside its own front: FlushRunQueue adds the front
		t_current.committed = true;
		t_current.kind      = kind;
	}
	plan.c0_front = false;
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window >= std::chrono::seconds(5)) {
		const uint32_t frame = DebugPresentCount().load();
		if (s.frame_start != 0 && frame > s.frame_start && s.draws != 0) Report(s, frame);
		const bool in = s.in_series, strict_in = s.strict_in;
		const auto run = s.run, strict_run = s.strict_run;
		s             = {};
		s.frame_start = frame;
		s.window      = now;
		s.in_series = in, s.strict_in = strict_in, s.run = run, s.strict_run = strict_run; // a series goes on
	}
	return kind;
}

// After a draw's call: its front's time without the commits and waits inside it.
void FrontEnd(std::chrono::steady_clock::time_point start) {
	auto&          c     = t_current;
	const uint64_t total = Ns(std::chrono::steady_clock::now() - start);
	const uint64_t own   = total > c.commit_ns + c.wait_ns ? total - c.commit_ns - c.wait_ns : 0;
	if (c.committed) {
		auto& s = g_stats;
		s.n_front[c.kind]++;
		s.front[c.kind] += own;
		s.wait[c.kind] += c.wait_ns;
	} else if (c.plan != nullptr) {
		c.plan->c0_front    = true;
		c.plan->c0_front_ns = own;
		c.plan->c0_wait_ns  = c.wait_ns;
	}
	c = {};
}

CommitScope::~CommitScope() {
	const uint64_t ns = Ns(std::chrono::steady_clock::now() - start);
	g_stats.commit[kind] += ns;
	t_current.commit_ns += ns;
}
} // namespace C0

} // namespace Libs::Graphics
