#if defined(ANYPS5_KYTY)

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/guest_gpu/hardwareContext.h"

#include <cstring>

namespace Libs::Graphics {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

PipelineStaticParameters LibraryStaticParameters(const HW::Context& context, const HW::UserConfig& user_config,
                                                 bool tessellation, const ShaderTessellationInputInfo& tess, bool ps_active,
                                                 bool ps_sample_shading) {
	PipelineStaticParameters params {};
	const auto& mc           = context.GetModeControl();
	const auto& clip_control = context.GetClipControl();
	const bool  rect_list    = Prospero::IsRectList(user_config.GetPrimType());
	params.negative_one_to_one = !clip_control.dx_clip_space;
	params.depth_clip_enable   = clip_control.IsZClipEnabled();
	params.provoking_vtx_last  = mc.provoking_vtx_last;
	params.polygon_mode = ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	if (tessellation) {
		params.patch_control_points = tess.input_control_points;
		params.merged_hull          = tess.merged_hull;
	} else if (rect_list) {
		params.patch_control_points = 3;
	}
	uint32_t   samples = 0;
	const auto mask    = context.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < 8 && samples == 0; slot++) {
		const auto& target = context.GetRenderTarget(slot);
		if (((mask >> (slot * 4u)) & 0x0fu) != 0 && target.base.addr != 0) {
			samples = render_sample_count(target.attrib.num_fragments);
		}
	}
	if (samples == 0) {
		const auto& depth = context.GetDepthRenderTarget();
		samples           = depth.z_info.format != Prospero::DepthFormat::kInvalid ? render_sample_count(depth.z_info.num_samples)
		                                             : render_sample_count(context.GetAaConfig().msaa_num_samples);
	}
	params.samples                  = samples;
	params.sample_shading_enable    = ps_active && samples > 1 && ps_sample_shading;
	params.alpha_to_coverage_enable = ps_active && context.GetShaderRegisters().AlphaToMask();
	return params;
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& other) const {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

}

#endif
