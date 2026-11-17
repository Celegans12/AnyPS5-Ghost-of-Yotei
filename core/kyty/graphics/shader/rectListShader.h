#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECTLISTSHADER_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECTLISTSHADER_H_

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct ShaderPixelInputInfo;

namespace ShaderRecompiler::IR {
struct CompiledShaderInfo;
}

struct RectListParameter {
	uint32_t input_location  = 0;
	uint32_t output_location = 0;
	bool     flat            = false;
};

struct RectListShaders {
	std::vector<uint32_t> control;
	std::vector<uint32_t> evaluation;
};

std::vector<RectListParameter> RectListParameters(
    const ShaderPixelInputInfo& pixel_info, const ShaderRecompiler::IR::CompiledShaderInfo& program);
RectListShaders BuildRectListShaders(std::span<const RectListParameter> parameters,
                                     uint32_t                           param_export_mask);

}

#endif
