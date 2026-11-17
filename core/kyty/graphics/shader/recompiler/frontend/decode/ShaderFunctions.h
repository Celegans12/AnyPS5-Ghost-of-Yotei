#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERFUNCTIONS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERFUNCTIONS_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

using ShaderCodeReader = std::function<bool(uint64_t, std::span<uint32_t>)>;

constexpr uint32_t ShaderCallMissTrapCode = 0x7fu;

bool InlineShaderFunctions(std::span<const uint32_t> code, uint64_t base,
                           std::span<const uint32_t> user_data, const ShaderCodeReader& read,
                           std::vector<uint32_t>& expanded, std::vector<LongBranch>& long_branches,
                           std::string& reason, uint32_t wave_size = 64,
                           std::vector<uint32_t>* consulted_user_data = nullptr);

class ShaderFunctionExpander {
public:
	ShaderFunctionExpander();
	~ShaderFunctionExpander();
	ShaderFunctionExpander(const ShaderFunctionExpander&)            = delete;
	ShaderFunctionExpander& operator=(const ShaderFunctionExpander&) = delete;

	bool Expand(std::span<const uint32_t> code, uint64_t base, std::span<const uint32_t> user_data,
	            const ShaderCodeReader& read, std::vector<uint32_t>& expanded, std::string& reason,
	            uint32_t wave_size, uint64_t code_hash = 0);
	[[nodiscard]] const std::vector<InlinedCallBody>& Bodies() const;
	[[nodiscard]] const std::vector<LongBranch>& LongBranches() const;
	[[nodiscard]] const std::vector<RuntimeLiteral>& RuntimeLiterals() const;
	[[nodiscard]] const std::vector<uint32_t>&       RuntimeWords() const;
	[[nodiscard]] bool HasCalls(std::span<const uint32_t> code) const;
	[[nodiscard]] const std::vector<uint32_t>* ConsultedUserData(std::span<const uint32_t> code,
	                                                             uint64_t code_hash,
	                                                             uint32_t wave_size) const;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

}

#endif
