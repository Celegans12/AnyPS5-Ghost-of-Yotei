#pragma once

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::PipelineTrace {

using Specialization = ShaderRecompiler::IR::ResourceSpecialization;

// GraphicsDebugDumpEnabled and KYTY_TRACE_SHADER_CODE_DIR (+KYTY_TRACE_DISASM): the guest code and the module of a compile.
void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv);
void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash, std::span<const uint32_t> code);
// --shader-validation: false when the module fails (logged, the caller stops the run).
bool ValidateShaderSpirv(const char* label, uint64_t shader_hash, const std::vector<uint32_t>& spirv, bool maintenance8_enabled);
// KYTY_TRACE_SPIRV_ALL_DIR, KYTY_TRACE_SHADER_CODE_DIR: the emitted module.
void DumpModule(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv);
// KYTY_DIAG_SPIRV_REPLACE: changes `spirv` for the listed shader.
void ReplaceSpirv(uint64_t shader_hash, std::vector<uint32_t>& spirv);
// KYTY_TRACE_SPIRV_TEXT.
void DumpSpirvText(const char* stage_name, uint64_t shader_hash, const std::vector<uint32_t>& spirv);
// KYTY_TRACE_PERMUTATIONS: `previous` as above.
void TracePermutation(uint64_t shader_hash, const Specialization* previous, size_t permutations, const Specialization& next);

} // namespace Libs::Graphics::PipelineTrace
