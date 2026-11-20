#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_FUZZ_SHADERRECOMPILERRUN_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_FUZZ_SHADERRECOMPILERRUN_HPP

#include "common/assert.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "spirv-tools/libspirv.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <span>
#include <vector>

namespace KytyFuzz {

namespace Gfx        = Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Libs::Graphics::ShaderRecompiler::IR;
using Gfx::ShaderType;

inline Recompiler::Spirv::Profile Profile() {
	return {.maintenance8           = true,
	        .native_float_min_max   = false,
	        .device_clock_period_ps = 1000,
	        .robust_buffer_access   = true,
	        .r8_srgb_views          = true,
	        .r8g8_srgb_views        = true,
	        .relaxed_float          = false,
	        .ray_query              = true,
	        .lds_lane_sync          = true};
}

inline bool Valid(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char*) {});
	spvtools::ValidatorOptions options;
	options.SetAllowOffsetTextureOperand(true);
	options.SetUniformBufferStandardLayout(true);
	return tools.Validate(spirv.data(), spirv.size(), options);
}

inline void Run(std::span<const uint32_t> code, uint8_t mode, uint8_t shape) {
	static const auto profile = Profile();
	Recompiler::CompileOptions options;
	options.shader_hash        = 0x1234567812345678ull;
	options.dump_ir            = false;
	options.emit_debug_info    = false;
	options.native_ray_tracing = true;
	options.ray_stats          = false;
	options.profile            = &profile;
	const std::vector<uint32_t> user_data(16u + (shape & 15u), 0u);
	options.user_data = user_data;

	Gfx::ShaderComputeInputInfo compute {};
	Gfx::ShaderVertexInputInfo  vertex {};
	Gfx::ShaderPixelInputInfo   pixel {};
	const uint32_t              wave = (mode & 1u) != 0 ? 32u : 64u;
	switch (mode % 6u) {
		case 3:
			options.stage             = ShaderType::Vertex;
			options.wave_size         = wave;
			options.user_data_base    = 8;
			options.input_info.vertex = &vertex;
			break;
		case 4:
			options.stage            = ShaderType::Pixel;
			options.wave_size        = wave;
			options.input_info.pixel = &pixel;
			break;
		default:
			compute.wave_size          = wave;
			compute.host_subgroup_size = 32;
			compute.threads_num[0]     = 1u << (shape & 3u);
			compute.threads_num[1]     = 1u << ((shape >> 2) & 3u);
			compute.threads_num[2]     = 1;
			compute.thread_ids_num     = 2;
			compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
			compute.workgroup_register = 16;
			compute.scratch_size_dwords = (mode % 6u) == 2 ? 1024u : 0u;
			compute.lds_size_dwords     = (mode % 6u) == 2 ? 2048u : 0u;
			options.stage              = ShaderType::Compute;
			options.wave_size          = wave;
			options.input_info.compute = &compute;
			break;
	}

	try {
		Common::SoftExitScope soft;
		auto                  source = Recompiler::PrepareShaderSource(code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) return;
		auto translated = Recompiler::TranslateProgram(source.decoded, options);
		IR::ResourceSpecialization specialization;
		specialization.buffers.resize(translated.program.info.buffers.size());
		specialization.images.resize(translated.program.info.images.size());
		specialization.samplers.resize(translated.program.info.samplers.size());
		for (size_t k = 0; k < specialization.images.size(); ++k) {
			specialization.images[k].numeric_class = Gfx::Prospero::TextureNumericClass::Float;
			specialization.images[k].dimension     = translated.program.info.images[k].dimension;
		}
		if (!IR::ResourceSpecializationFits(translated.program, specialization)) return;
		auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0);
		if (!compiled.spirv.empty() && !Valid(compiled.spirv)) {
			std::fputs("kyty_shader_recompiler_fuzzer: the recompiler produced SPIR-V that spirv-val refuses\n", stderr);
			std::abort();
		}
	} catch (const std::exception&) {
	}
}


}

#endif
