#ifndef CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBLOCKACCESS_HPP
#define CORE_SHADER_RECOMPILER_SPIRVBACKEND_SPIRVBLOCKACCESS_HPP

#include "SpirvBackend/SpirvEmitterState.hpp"
#include <cstdint>

namespace ShaderRecompiler {

void EmitBlockBase(SpirvEmitterState& state);
[[nodiscard]] std::uint32_t EmitBlockWord(SpirvEmitterState& state, std::uint32_t word);

struct BlockBufferOperands {
    std::uint32_t address = 0;
    std::uint32_t bytes = 0;
    std::uint32_t misalignment = 0;
    std::uint32_t present = 0;
};

[[nodiscard]] BlockBufferOperands EmitBlockBuffer(SpirvEmitterState& state, std::uint32_t buffer);
[[nodiscard]] std::uint32_t EmitBlockLoad32(SpirvEmitterState& state, const BlockBufferOperands& buffer, std::uint32_t byteOffset, std::uint32_t inBounds);

}

#endif
