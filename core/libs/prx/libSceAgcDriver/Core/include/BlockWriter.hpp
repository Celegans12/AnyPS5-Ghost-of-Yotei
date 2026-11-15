#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKWRITER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKWRITER_HPP

#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Optimization/BlockLayout.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"

#include <cstdint>
#include <span>

namespace AgcDriver::Core {

enum class BlockFault : std::uint32_t { None, Walk, Variant, RingFull, Unreadable, Count };

struct BufferRequest { std::uint64_t address = 0; std::uint64_t bytes = 0; std::uint32_t misalignment = 0; std::uint32_t buffer = 0; bool present = false; };
struct BufferAnswer { std::uint64_t deviceAddress = 0; std::uint32_t bytes = 0; bool deferred = false; };

struct BlockInputs {
    const ShaderRecompiler::IrResourcePlan* plan = nullptr;
    const ShaderRecompiler::BlockLayout* layout = nullptr;
    const ShaderRecompiler::Detail::SrtTape* tape = nullptr;
    std::span<const std::uint32_t> userData;
    std::uint64_t shaderBase = 0;
    std::uint64_t variantKey = 0;
    ShaderRecompiler::BlockWordReader read = nullptr;
    bool (*resolve)(void* context, const BufferRequest& request, BufferAnswer& answer) = nullptr;
    void* context = nullptr;
};

struct BlockOutput { RingSlice section; std::uint64_t deferred = 0; BlockFault fault = BlockFault::None; };

[[nodiscard]] BlockOutput WriteBlock(const BlockInputs& inputs, BlockRing& ring);
void CompleteBuffer(const ShaderRecompiler::BlockLayout& layout, const RingSlice& section, std::uint32_t buffer, const BufferAnswer& answer);
[[nodiscard]] BufferRequest DecodeBuffer(const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> words, std::uint32_t buffer);

}

#endif
