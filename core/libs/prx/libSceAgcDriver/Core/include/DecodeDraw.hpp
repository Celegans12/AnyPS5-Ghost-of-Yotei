#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODEDRAW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODEDRAW_HPP

#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockWriter.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Optimization/BlockLayout.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"

#include <array>
#include <cstdint>
#include <span>

namespace AgcDriver::Core {

enum class DrawLeft : std::uint32_t { None, Ring, Walk, Variant, Buffers, Count };

struct DrawStage {
    const ShaderRecompiler::IrResourcePlan* plan = nullptr;
    const ShaderRecompiler::BlockLayout* layout = nullptr;
    const ShaderRecompiler::Detail::SrtTape* tape = nullptr;
    std::span<const std::uint32_t> userData;
    std::uint64_t shaderBase = 0;
    std::uint64_t variantKey = 0;
};

struct DrawReads {
    ShaderRecompiler::BlockWordReader read = nullptr;
    bool (*resolve)(void* context, const BufferRequest& request, BufferAnswer& answer) = nullptr;
    void* context = nullptr;
    bool keyed = true;
};

struct DrawWritten {
    RingSlice slice;
    std::array<std::uint32_t, 4> section{};
    std::array<std::uint32_t, 4> driver{};
    std::uint64_t deferred = 0;
    std::array<std::uint64_t, 2> images{};
    std::array<std::uint64_t, 4> variant{};
    std::uint32_t extra = 0;
    DrawLeft left = DrawLeft::None;
    std::uint32_t stage = 0;
};

[[nodiscard]] std::uint32_t DrawBytes(std::span<const DrawStage> stages, std::uint32_t extraWords);
[[nodiscard]] DrawWritten WriteDraw(std::span<const DrawStage> stages, const DrawReads& reads, std::uint32_t extraWords, BlockRing& ring);
[[nodiscard]] std::uint32_t DrawDriverWords(const DrawStage& stage);
[[nodiscard]] std::span<const std::uint32_t> DrawDriverZone(const DrawStage& stage, const DrawWritten& written, std::uint32_t index);
void CompleteDrawBuffer(const DrawStage& stage, const DrawWritten& written, std::uint32_t index, std::uint32_t buffer, const BufferAnswer& answer);
[[nodiscard]] const char* DrawLeftName(DrawLeft left);

}

#endif
