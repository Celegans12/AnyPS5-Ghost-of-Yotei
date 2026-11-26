#pragma once

#include "graphics/host_gpu/renderer/pipeline/pipelineStoreTypes.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"

#include <type_traits>

namespace Libs::Graphics {

bool LdsLaneSyncEnabled();
bool F32ProductNoContractionEnabled();
void AppendNativeRayTracingKey(ShaderType stage, std::vector<uint32_t>& key, bool native);

template <typename InputInfo>
ShaderType ShaderStageForInput(const InputInfo& input) {
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		return input.logical_stage;
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		return ShaderType::Pixel;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		return ShaderType::Compute;
	}
}

template <typename InputInfo>
ShaderRecompiler::CompileOptions StageCompileOptions(const InputInfo& input, bool table_mode = false) {
	ShaderRecompiler::CompileOptions options {.stage = ShaderStageForInput(input), .wave_size = input.wave_size};
	options.table_mode = table_mode;
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		options.input_info.vertex = &input;
		if (options.stage != ShaderType::Mesh && options.stage != ShaderType::TessellationControl) {
			options.user_data_base = 8;
		}
		if (options.stage == ShaderType::Mesh) {
			options.wave_size = input.mesh.wave_size;
		}
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		options.input_info.pixel = &input;
	} else {
		options.input_info.compute = &input;
	}
	return options;
}

template <typename InputInfo>
void SetShaderProgramKey(ShaderProgramKey& key, const ShaderParams& params, const InputInfo& input,
                         bool native = true, bool lane_pairs = false) {
	key.stage           = ShaderStageForInput(input);
	key.hash            = params.hash;
	key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input, key.static_state);
	AppendNativeRayTracingKey(key.stage, key.static_state, native);
	if (lane_pairs) {
		constexpr uint32_t LanePairsKey = 0x574c5052u;
		key.static_state.push_back(LanePairsKey);
	}
}

void SetShaderCompileOptions(ShaderRecompiler::CompileOptions& options, uint64_t hash,
                             std::span<const uint32_t> user_data, std::span<const uint32_t> back_code,
                             bool native, bool lane_pairs = false);

}
