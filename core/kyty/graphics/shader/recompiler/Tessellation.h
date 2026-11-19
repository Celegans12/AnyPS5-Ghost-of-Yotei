#pragma once

#include <cstdint>
#include <span>

namespace Libs::Graphics {
struct ShaderTessellationInputInfo;
}

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions;
namespace IR {
struct Program;
}

[[nodiscard]] bool AnalyzeTessellationPrograms(std::span<const uint32_t> local,
                                               std::span<const uint32_t> control,
                                               ShaderTessellationInputInfo& info);
[[nodiscard]] bool LowerTessellationMemory(IR::Program& program, const CompileOptions& options);

#if defined(ANYPS5_KYTY)
struct TessellationLimits {
	uint32_t per_vertex_input_components  = 0;
	uint32_t per_vertex_output_components = 0;
	uint32_t per_patch_output_components  = 0;
	uint32_t total_output_components      = 0;
};

void RequireMergedHullControlPoints(const ShaderTessellationInputInfo& info);
void RequireMergedHullLimits(const ShaderTessellationInputInfo& info,
                             const TessellationLimits&          limits);
#endif

}
