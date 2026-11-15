#ifndef CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BLOCKLAYOUT_HPP
#define CORE_SHADER_RECOMPILIER_OPTIMIZATION_INCLUDE_OPTIMIZATION_BLOCKLAYOUT_HPP

#include "BlockAbi.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ShaderRecompiler {

struct IrResourcePlan;

enum class BlockRefusal : std::uint32_t { None, GpuDescriptor, ImageTable, WrittenBuffer, AtomicBuffer, UniformFill, UnguardedPoison, TooManyWords, TooManyBuffers, SrtPlanIncomplete, ResourceTrackingIncomplete, WordWithoutValue, WordOutsideBlock, WordShared, BufferSourceMissing, BufferFormatWordMissing, ImageModesMissing, ImageSourceMissing, ImageWordMissing, SamplerSourceMissing, Count };

struct BlockBuffer {
    std::uint32_t resource = 0;
    std::array<std::uint32_t, 4> descriptorWords{};
    bool formatFixed = false;
    std::uint32_t typedAlignment = 1;
};

struct BlockVariantBit { std::uint32_t word = 0; std::uint32_t mask = 0; };
struct BlockImageKeyWord { std::uint32_t word = 0; std::uint32_t mask = 0xffffffffu; };

struct BlockLayout {
    std::uint32_t abiVersion = BlockAbi::Version;
    std::uint32_t words = 0;
    std::vector<BlockBuffer> buffers;
    std::vector<BlockVariantBit> variantKey;
    std::vector<BlockImageKeyWord> imageKey;
    std::vector<BlockVariantBit> userVariantKey;
    std::vector<BlockImageKeyWord> userImageKey;
    BlockRefusal refusal = BlockRefusal::None;
    [[nodiscard]] bool Eligible() const { return refusal == BlockRefusal::None; }
    [[nodiscard]] std::uint32_t SectionBytes() const;
};

[[nodiscard]] BlockLayout BuildBlockLayout(const IrResourcePlan& plan);
[[nodiscard]] std::uint64_t BlockVariantKey(const BlockLayout& layout, std::span<const std::uint32_t> words);
[[nodiscard]] std::uint64_t BlockVariantKey(const BlockLayout& layout, std::span<const std::uint32_t> words, std::span<const std::uint32_t> userData);
[[nodiscard]] std::string_view BlockRefusalName(BlockRefusal refusal);

}

#endif
