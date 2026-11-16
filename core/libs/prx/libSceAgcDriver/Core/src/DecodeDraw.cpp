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
constexpr std::uint32_t MaxDriverWords = 8192;
constexpr std::uint64_t BaseMask = 0x0000ffffffffffffull;
constexpr std::uint32_t StrideShift = 16u;
constexpr std::uint32_t StrideMask = 0x3fffu;
constexpr std::uint64_t ImageTag = 0x1000000000000000ull;
constexpr std::uint64_t SamplerTag = 0x2000000000000000ull;
constexpr std::uint64_t ActiveTag = 0x3000000000000000ull;

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

std::uint64_t padded(std::uint64_t bytes) {
    return (bytes + Abi::SectionAlignmentBytes - 1u) / Abi::SectionAlignmentBytes * Abi::SectionAlignmentBytes;
}

bool driven(const DrawStage& stage) {
    return stage.tape != nullptr && stage.tape->Driver().complete && stage.tape->Driver().words != 0u && stage.tape->Driver().words <= MaxDriverWords;
}

bool inBlock(const ShaderRecompiler::BlockBuffer& buffer, std::uint32_t words) {
    for (std::uint32_t word = 0; word < LocatingWords; ++word) {
        if (buffer.descriptorWords[word] >= words) return false;
    }
    return true;
}

std::uint32_t zoneWord(const DrawStage& stage, const ShaderRecompiler::BlockBuffer& buffer) {
    if (!driven(stage)) return ShaderRecompiler::NoDriverWord;
    const auto& zone = stage.tape->Driver();
    if (buffer.resource >= zone.buffers.size()) return ShaderRecompiler::NoDriverWord;
    const auto rank = zone.buffers[buffer.resource];
    if (rank >= zone.sources.size() || zone.sources[rank].dwords < LocatingWords) return ShaderRecompiler::NoDriverWord;
    return zone.sources[rank].word;
}

BufferRequest requestOf(const DrawStage& stage, std::span<const std::uint32_t> words, std::span<const std::uint32_t> zone, std::uint32_t buffer) {
    const auto& layout = *stage.layout;
    if (buffer >= layout.buffers.size()) return {};
    if (inBlock(layout.buffers[buffer], layout.words)) return DecodeBuffer(layout, words, buffer);
    BufferRequest request;
    request.buffer = buffer;
    const auto first = zoneWord(stage, layout.buffers[buffer]);
    if (first == ShaderRecompiler::NoDriverWord || std::uint64_t{first} + LocatingWords > zone.size()) return request;
    const auto word0 = zone[first];
    const auto word1 = zone[first + 1u];
    const auto records = zone[first + 2u];
    const auto base = (static_cast<std::uint64_t>(word0) | (static_cast<std::uint64_t>(word1) << 32u)) & BaseMask;
    const std::uint64_t stride = (word1 >> StrideShift) & StrideMask;
    const std::uint64_t bytes = stride == 0u ? std::uint64_t{records} : stride * records;
    if (base == 0u || bytes == 0u) return request;
    request.address = base;
    request.bytes = bytes;
    request.misalignment = word0 & Abi::ServiceMisalignmentMask;
    request.present = true;
    return request;
}

void writeAnswer(std::uint32_t* entry, const BufferRequest& request, const BufferAnswer& answer) {
    const bool absent = !request.present || answer.deviceAddress == 0u || answer.bytes == 0u;
    entry[Abi::EntryAddressLow] = absent ? 0u : static_cast<std::uint32_t>(answer.deviceAddress);
    entry[Abi::EntryAddressHigh] = absent ? 0u : static_cast<std::uint32_t>(answer.deviceAddress >> 32u);
    entry[Abi::EntryBytes] = absent ? 0u : answer.bytes;
    entry[Abi::EntryService] = absent ? Abi::ServiceAbsent : request.misalignment & Abi::ServiceMisalignmentMask;
}

void foldSource(std::array<std::uint64_t, 2>& images, const ShaderRecompiler::BlockDriverZone& layout, std::span<const std::uint32_t> zone, std::uint64_t tag, std::size_t item, std::uint32_t rank) {
    if (rank >= layout.sources.size()) return;
    const auto& source = layout.sources[rank];
    for (std::uint32_t dword = 0; dword < source.dwords; ++dword) fold(images, tag | (static_cast<std::uint64_t>(item) << 36u) | (std::uint64_t{dword} << 32u) | zone[source.word + dword]);
}

void foldZone(std::array<std::uint64_t, 2>& images, const ShaderRecompiler::BlockDriverZone& layout, std::span<const std::uint32_t> zone) {
    for (std::size_t image = 0; image < layout.images.size(); ++image) {
        foldSource(images, layout, zone, ImageTag, image, layout.images[image]);
        fold(images, ActiveTag | (static_cast<std::uint64_t>(image) << 36u) | std::uint64_t{layout.Active(zone, layout.imageBase + static_cast<std::uint32_t>(image))});
    }
    for (std::size_t sampler = 0; sampler < layout.samplers.size(); ++sampler) foldSource(images, layout, zone, SamplerTag, sampler, layout.samplers[sampler]);
}

DrawLeft refusalOf(const DrawStage& stage) {
    if (stage.plan == nullptr || stage.layout == nullptr || stage.tape == nullptr) return DrawLeft::Walk;
    const auto& layout = *stage.layout;
    if (!layout.Eligible() || layout.abiVersion != Abi::Version) return DrawLeft::Walk;
    if (layout.words > Abi::MaxWords || layout.buffers.size() > Abi::MaxBuffers) return DrawLeft::Walk;
    if (layout.words != stage.plan->srtReads.size()) return DrawLeft::Walk;
    for (const auto& buffer : layout.buffers) {
        if (!inBlock(buffer, layout.words) && zoneWord(stage, buffer) == ShaderRecompiler::NoDriverWord) return DrawLeft::Walk;
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

std::uint32_t DrawDriverWords(const DrawStage& stage) {
    return driven(stage) ? stage.tape->Driver().words : 0u;
}

std::span<const std::uint32_t> DrawDriverZone(const DrawStage& stage, const DrawWritten& written, std::uint32_t index) {
    if (written.slice.bytes == nullptr || index >= written.driver.size()) return {};
    return {reinterpret_cast<const std::uint32_t*>(written.slice.bytes + written.driver[index]), DrawDriverWords(stage)};
}

void CompleteDrawBuffer(const DrawStage& stage, const DrawWritten& written, std::uint32_t index, std::uint32_t buffer, const BufferAnswer& answer) {
    if (written.slice.bytes == nullptr || index >= written.section.size() || stage.layout == nullptr || buffer >= stage.layout->buffers.size() || answer.deferred) return;
    const auto& layout = *stage.layout;
    auto* section = reinterpret_cast<std::uint32_t*>(written.slice.bytes + written.section[index]);
    const auto request = requestOf(stage, std::span<const std::uint32_t>(section, layout.words), DrawDriverZone(stage, written, index), buffer);
    writeAnswer(section + Abi::BufferEntryWord(layout.words, buffer), request, answer);
}

std::uint32_t DrawBytes(std::span<const DrawStage> stages, std::uint32_t extraWords) {
    std::uint64_t bytes = extraBytes(extraWords);
    std::uint64_t zones = 0;
    for (const auto& stage : stages) {
        if (stage.layout != nullptr) bytes += stage.layout->SectionBytes();
        zones += std::uint64_t{DrawDriverWords(stage)} * sizeof(std::uint32_t);
    }
    bytes += padded(zones);
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
    std::array<std::uint32_t, MaxStages> zones{};
    for (std::size_t index = 0; index < stages.size(); ++index) {
        zones[index] = static_cast<std::uint32_t>(bytes);
        bytes += std::uint64_t{DrawDriverWords(stages[index])} * sizeof(std::uint32_t);
    }
    bytes = padded(bytes);
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
        auto* section = reinterpret_cast<std::uint32_t*>(slice.bytes + offsets[index]);
        const std::span<std::uint32_t> words(section, layout.words);
        const std::span<std::uint32_t> zone(reinterpret_cast<std::uint32_t*>(slice.bytes + zones[index]), DrawDriverWords(stage));
        const auto walked = zone.empty() ? stage.tape->RunInto(*stage.plan, stage.userData, stage.shaderBase, reads.read, reads.context, words) : stage.tape->RunInto(*stage.plan, stage.userData, stage.shaderBase, reads.read, reads.context, words, zone);
        if (walked != ShaderRecompiler::BlockWalk::Done) return leftAt(DrawLeft::Walk, index);
        const auto variant = ShaderRecompiler::BlockVariantKey(layout, words, stage.userData);
        if (reads.keyed && variant != stage.variantKey) return leftAt(DrawLeft::Variant, index);
        written.variant[index] = variant;
        const auto count = static_cast<std::uint32_t>(layout.buffers.size());
        for (std::uint32_t buffer = 0; buffer < count; ++buffer, ++rank) {
            const auto request = requestOf(stage, words, zone, buffer);
            auto* entry = section + Abi::BufferEntryWord(layout.words, buffer);
            BufferAnswer answer;
            if (request.present) {
                if (!reads.resolve(reads.context, request, answer)) return leftAt(DrawLeft::Buffers, index);
                if (answer.deferred) {
                    for (std::uint32_t word = 0; word < Abi::BufferEntryWords; ++word) entry[word] = 0u;
                    written.deferred |= std::uint64_t{1} << rank;
                    continue;
                }
            }
            writeAnswer(entry, request, answer);
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
        if (!zone.empty()) foldZone(images, stage.tape->Driver(), zone);
    }
    written.slice = slice;
    written.section = offsets;
    written.driver = zones;
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
