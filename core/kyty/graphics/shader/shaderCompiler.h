#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_

#include "graphics/shader/shader.h"

#include <array>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
}

struct ShaderParams {
	std::span<const uint32_t> code;
	std::array<uint32_t, 40>  user_data {};
	uint32_t                  user_data_count = 0;
	uint64_t                  hash = 0;
	std::span<const uint32_t> back_code;

	[[nodiscard]] uint64_t Base() const {
		return reinterpret_cast<uint64_t>(code.data());
	}
};

void BuildStageStaticKey(const ShaderVertexInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderPixelInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderComputeInputInfo& input_info, std::vector<uint32_t>& key);

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& input_info);
std::array<ShaderParams, 3>
PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            std::array<ShaderVertexInputInfo, 3>& input_info);
ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               input_info);
ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info);

using ShaderGuestReader = bool (*)(void* userdata, uint64_t address, void* data, uint64_t size);

class ShaderGuestReaderScope {
public:
	ShaderGuestReaderScope(ShaderGuestReader reader, void* userdata);
	~ShaderGuestReaderScope();
	ShaderGuestReaderScope(const ShaderGuestReaderScope&)            = delete;
	ShaderGuestReaderScope& operator=(const ShaderGuestReaderScope&) = delete;

private:
	ShaderGuestReader m_previous_reader;
	void*             m_previous_userdata;
};

}

#endif
