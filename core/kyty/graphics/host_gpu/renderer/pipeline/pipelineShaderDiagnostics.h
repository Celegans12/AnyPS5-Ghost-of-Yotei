#pragma once

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

void DumpShader(std::span<const uint32_t> words, const std::string& name);
void ValidateShaderSpirv(const char* stage, uint64_t shader_hash, const std::vector<uint32_t>& spirv,
                         bool maintenance8_enabled);
void NoteStoreRefused(ShaderType stage, uint64_t hash, const ShaderRecompiler::IR::EncodeFailure& failure);
#if defined(ANYPS5_KYTY)
void LogShaderVariantBuild(const char* stage, uint64_t shader_hash, uint32_t user_data_count,
                           const std::vector<uint32_t>& static_state, uint32_t push_data_cursor,
                           const ShaderModule& module);
#endif

}
