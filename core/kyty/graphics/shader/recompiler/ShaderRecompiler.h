#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bitset>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        emit_debug_info            = false;
	const Spirv::Profile*       profile                    = &Spirv::DefaultProfile;
	bool                        native_ray_tracing         = false;
	bool                        ray_stats                  = false;
	uint32_t                    rt_oracle                  = 0;
	bool                        table_mode                 = false;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
	bool                        plain_mip_stats_variant    = false;
	bool                        wave_lane_pairs            = false;
	bool                        uniform_scalar_loads       = true;
};

struct TranslateResult {
	IR::Program                        program;
	std::string                        decoded_dump;
	std::string                        cfg_dump;
	std::vector<IR::NativeTraversals>  oracle_sources;
	std::vector<IR::OracleImprovement> oracle_improvements;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            table_refusal;
	std::vector<uint32_t>  spirv_plain;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
	std::vector<IR::NativeTraversals> oracle_sources;
	std::vector<IR::OracleImprovement> oracle_improvements;
};

struct ShaderSource {
	struct Call {
		uint32_t instruction;
		std::array<IR::Value, 2> target;
		std::bitset<128> preserved_registers;
		uint64_t address = 0;
	};
	struct Linked {
		struct Function { uint32_t offset; uint32_t size; };
		std::vector<uint32_t> code;
		Decoder::Program decoded;
		std::vector<Function> functions;
		std::vector<uint32_t> observed_function;
	};
	std::vector<uint32_t> code;
	Decoder::Program decoded;
	IR::ResourcePlan call_targets;
	std::vector<Call> calls;
	uint32_t unsupported_call_pc = UINT32_MAX;
	bool expand_calls = false;
	uint32_t wave_size = 64;
	std::shared_ptr<Decoder::ShaderFunctionExpander> expander;
	bool refresh_failed = false;
	std::string refresh_failure;
	std::optional<Linked> linked;
	std::vector<uint32_t> call_words;
	uint64_t revision = 0;
	std::vector<std::pair<uint64_t, uint64_t>> reads;

	[[nodiscard]] bool HasCalls() const { return !calls.empty() || expand_calls; }
};

enum class RayTracingProgramKind : uint32_t { Native, Empty, AlwaysGuest, PixelGuest, Count };
struct RayTracingPrograms {
	uint64_t native            = 0;
	uint64_t empty             = 0;
	uint64_t always_guest      = 0;
	uint64_t pixel_guest       = 0;
	uint64_t last_empty        = 0;
	uint64_t last_always_guest = 0;
	uint64_t last_pixel_guest  = 0;
};
bool                             NoteRayTracingProgram(RayTracingProgramKind kind, uint64_t shader_hash);
[[nodiscard]] RayTracingPrograms GetRayTracingPrograms();

[[nodiscard]] constexpr const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "CS";
		case ShaderType::Vertex: return "VS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "TES";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Pixel: return "PS";
		default: return "unknown";
	}
}
[[nodiscard]] ShaderSource PrepareShaderSource(std::span<const uint32_t> code,
	                                           const CompileOptions& options);
const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime);
[[nodiscard]] TranslateResult TranslateProgram(const Decoder::Program& decoded,
	                                           const CompileOptions& options);

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

}

#endif
