#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_PACKEDUNORMBLENDPROBE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_PACKEDUNORMBLENDPROBE_H_

#include <cstdint>
#include <vector>

#if defined(ANYPS5_KYTY)
namespace Libs::Graphics::ShaderRecompiler::Spirv {

std::vector<uint32_t> EmitPackedUnormBlendProbe(uint32_t blend, bool truncate);

}
#endif

#endif
