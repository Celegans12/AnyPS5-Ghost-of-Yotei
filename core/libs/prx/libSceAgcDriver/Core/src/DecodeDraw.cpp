#include "prx/libSceAgcDriver/Core/include/DecodeDraw.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockRing.hpp"
#include "prx/libSceAgcDriver/Core/include/BlockWriter.hpp"
#include "prx/libSceAgcDriver/Core/include/ResourceCache.hpp"
#include "BlockAbi.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Optimization/BlockLayout.hpp"
#include "Optimization/SrtWalker/SrtTape.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Core {

namespace {

namespace Abi = ShaderRecompiler::BlockAbi;

constexpr std::size_t MaxStages = 4;
constexpr std::uint32_t MaxDrawBuffers = 64;
constexpr std::uint32_t LocatingWords = 3u;
constexpr std::uint64_t Golden = 0x9e3779b97f4a7c15ull;
constexpr std::uint64_t LowSeed = 0x452821e638d01377ull;
constexpr std::uint64_t HighSeed = 0xbe5466cf34e90c6cull;
constexpr std::uint64_t MostBytes = 0xffffffffull;

std::uint64_t extraBytes(std::uint32_t extraWords) {
    const std::uint64_t bytes = std::uint64_t{extraWords} * sizeof(std::uint32_t);
    return (bytes + Abi::SectionAlignmentBytes - 1u) / Abi::SectionAlignmentBytes * Abi::SectionAlignmentBytes;
}

void fold(std::array<std::uint64_t, 2>& images, std::uint64_t item) {
    std::uint64_t low = images[0] ^ item;
    low ^= low >> 33u;
    low *= 0xff51afd7ed558ccdull;
    low ^= low >> 33u;
    low *= 0xc4ceb9fe1a85ec53ull;
    low ^= low >> 33u;
    images[0] = low;
    std::uint64_t high = images[1] + std::rotl(item, 29) + Golden;
    high ^= high >> 30u;
    high *= 0xbf58476d1ce4e5b9ull;
    high ^= high >> 27u;
    high *= 0x94d049bb133111ebull;
    high ^= high >> 31u;
    images[1] = high;
}

DrawWritten leftAt(DrawLeft why, std::size_t stage) {
    DrawWritten written;
    written.left = why;
    written.stage = static_cast<std::uint32_t>(stage);
    return written;
}

DrawLeft refusalOf(const DrawStage& stage) {
    if (stage.plan == nullptr || stage.layout == nullptr || stage.tape == nullptr) return DrawLeft::Walk;
    const auto& layout = *stage.layout;
    if (!layout.Eligible() || layout.abiVersion != Abi::Version) return DrawLeft::Walk;
    if (layout.words > Abi::MaxWords || layout.buffers.size() > Abi::MaxBuffers) return DrawLeft::Walk;
    if (layout.words != stage.plan->srtReads.size()) return DrawLeft::Walk;
    for (const auto& buffer : layout.buffers) {
        for (std::uint32_t word = 0; word < LocatingWords; ++word) {
            if (buffer.descriptorWords[word] >= layout.words) return DrawLeft::Walk;
        }
    }
    for (const auto& bit : layout.variantKey) {
        if (bit.word >= layout.words) return DrawLeft::Variant;
    }
    for (const auto& bit : layout.userVariantKey) {
        if (bit.word >= stage.userData.size()) return DrawLeft::Variant;
    }
    return DrawLeft::None;
}

}

std::uint32_t DrawBytes(std::span<const DrawStage> stages, std::uint32_t extraWords) {
    std::uint64_t bytes = extraBytes(extraWords);
    for (const auto& stage : stages) {
        if (stage.layout != nullptr) bytes += stage.layout->SectionBytes();
    }
    return static_cast<std::uint32_t>(bytes > MostBytes ? MostBytes : bytes);
}

DrawWritten WriteDraw(std::span<const DrawStage> stages, const DrawReads& reads, std::uint32_t extraWords, BlockRing& ring) {
    if (stages.size() > MaxStages) return leftAt(DrawLeft::Walk, MaxStages);
    if (reads.read == nullptr || reads.resolve == nullptr) return leftAt(DrawLeft::Walk, 0);
    std::array<std::uint32_t, MaxStages> offsets{};
    std::uint64_t bytes = 0;
    std::size_t buffers = 0;
    for (std::size_t index = 0; index < stages.size(); ++index) {
        const auto refusal = refusalOf(stages[index]);
        if (refusal != DrawLeft::None) return leftAt(refusal, index);
        const auto& layout = *stages[index].layout;
        buffers += layout.buffers.size();
        if (buffers > MaxDrawBuffers) return leftAt(DrawLeft::Buffers, index);
        offsets[index] = static_cast<std::uint32_t>(bytes);
        bytes += layout.SectionBytes();
    }
    const auto extra = static_cast<std::uint32_t>(bytes);
    bytes += extraBytes(extraWords);
    if (bytes > MostBytes) bytes = MostBytes;
    const auto slice = ring.Take(static_cast<std::size_t>(bytes), Abi::SectionAlignmentBytes);
    if (slice.bytes == nullptr) return leftAt(DrawLeft::Ring, 0);
    DrawWritten written;
    std::array<std::uint64_t, 2> images{LowSeed, HighSeed};
    std::uint32_t rank = 0;
    for (std::size_t index = 0; index < stages.size(); ++index) {
        const auto& stage = stages[index];
        const auto& layout = *stage.layout;
        RingSlice part = slice;
        part.bytes += offsets[index];
        part.deviceAddress += offsets[index];
        auto* section = reinterpret_cast<std::uint32_t*>(part.bytes);
        const std::span<std::uint32_t> words(section, layout.words);
        if (stage.tape->RunInto(*stage.plan, stage.userData, stage.shaderBase, reads.read, reads.context, words) != ShaderRecompiler::BlockWalk::Done) return leftAt(DrawLeft::Walk, index);
        const auto variant = ShaderRecompiler::BlockVariantKey(layout, words, stage.userData);
        if (reads.keyed && variant != stage.variantKey) return leftAt(DrawLeft::Variant, index);
        written.variant[index] = variant;
        const auto count = static_cast<std::uint32_t>(layout.buffers.size());
        for (std::uint32_t buffer = 0; buffer < count; ++buffer, ++rank) {
            const auto request = DecodeBuffer(layout, words, buffer);
            BufferAnswer answer;
            if (request.present) {
                if (!reads.resolve(reads.context, request, answer)) return leftAt(DrawLeft::Buffers, index);
                if (answer.deferred) {
                    auto* entry = section + Abi::BufferEntryWord(layout.words, buffer);
                    for (std::uint32_t word = 0; word < Abi::BufferEntryWords; ++word) entry[word] = 0u;
                    written.deferred |= std::uint64_t{1} << rank;
                    continue;
                }
            }
            CompleteBuffer(layout, part, buffer, answer);
        }
        const auto sectionWords = Abi::SectionWords(layout.words, count);
        const auto paddedWords = Abi::SectionBytes(layout.words, count) / static_cast<std::uint32_t>(sizeof(std::uint32_t));
        for (std::uint32_t word = sectionWords; word < paddedWords; ++word) section[word] = 0u;
        const auto key = ImageSetKeyOf(index, layout, words);
        fold(images, key.digest[0]);
        fold(images, key.digest[1]);
        for (const auto& entry : layout.userImageKey) {
            const bool present = entry.word < stage.userData.size();
            const std::uint32_t masked = present ? stage.userData[entry.word] & entry.mask : 0u;
            fold(images, (std::uint64_t{entry.word} << 33u) | (std::uint64_t{present} << 32u) | masked);
        }
    }
    written.slice = slice;
    written.section = offsets;
    written.images = images;
    written.extra = extra;
    return written;
}

const char* DrawLeftName(DrawLeft left) {
    switch (left) {
        case DrawLeft::None: return "None";
        case DrawLeft::Ring: return "Ring";
        case DrawLeft::Walk: return "Walk";
        case DrawLeft::Variant: return "Variant";
        case DrawLeft::Buffers: return "Buffers";
        case DrawLeft::Count: break;
    }
    return "Unknown";
}

}
