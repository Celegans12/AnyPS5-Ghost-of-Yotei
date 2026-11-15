#include "Optimization/BlockLayout.hpp"
#include "IntermediateRepresentation/IrMetadata/ResourcePlan.hpp"
#include "Optimization/SrtWalker/SrtFlatSlotClasses.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ShaderRecompiler {

namespace {

constexpr std::uint32_t NoWord = 0xffffffffu;
constexpr std::uint32_t WholeWord = 0xffffffffu;
constexpr std::uint32_t BufferFormatWord = 3u;

std::uint32_t blockWord(const IrResourcePlan& plan, std::size_t source, std::uint32_t dword) {
    const auto slot = Detail::DescriptorDwordSlot(plan, source, dword);
    if (slot == Detail::NoDescriptorDwordSlot) return NoWord;
    const auto word = plan.srtReads[slot].flatOffset;
    return word < plan.srtReads.size() ? word : NoWord;
}

std::uint32_t userWord(const IrResourcePlan& plan, std::size_t source, std::uint32_t dword) {
    return Detail::DescriptorDwordUserWord(plan, source, dword);
}

bool constantWord(const IrResourcePlan& plan, std::size_t source, std::uint32_t dword) {
    if (source >= plan.descriptorSources.size()) return false;
    const auto& entry = plan.descriptorSources[source];
    if (dword >= entry.dwordCount || dword >= entry.dwords.size() || entry.dwords[dword] == nullptr) return false;
    const IrValue* value = entry.dwords[dword]->Resolve();
    return value->HasImmediate() && value->Type() == IrType::U32;
}

std::uint32_t sourceDwords(const IrResourcePlan& plan, std::size_t source) {
    if (source >= plan.descriptorSources.size()) return 0u;
    const auto& entry = plan.descriptorSources[source];
    return std::min(entry.dwordCount, static_cast<std::uint32_t>(entry.dwords.size()));
}

bool multipleModes(const IrResourcePlan& plan, std::size_t image) {
    return image < plan.info.runtimeImageModes.size() && plan.info.runtimeImageModes[image].size() > 1u;
}

bool readsTables(const IrResourcePlan& plan) {
    return std::ranges::any_of(plan.descriptorSources, [](const DescriptorSource& source) { return source.tableColumn.has_value(); })
        || std::ranges::any_of(plan.info.images, [](const ImageResource& image) { return image.table != NoTable; })
        || std::ranges::any_of(plan.info.samplers, [](const SamplerResource& sampler) { return sampler.table != NoTable; });
}

bool poisonsWithoutGuard(const IrResourcePlan& plan) {
    const auto poisonable = Detail::ComputeGuardedFlatSlots(plan);
    return std::ranges::any_of(poisonable, [&](std::uint32_t slot) { return std::ranges::find(plan.guardedSrtSlots, slot) == plan.guardedSrtSlots.end(); });
}

BlockRefusal wordsRefusal(const IrResourcePlan& plan) {
    std::vector<std::uint8_t> taken(plan.srtReads.size(), 0u);
    for (const auto& read : plan.srtReads) {
        if (read.value == nullptr) return BlockRefusal::WordWithoutValue;
        if (read.flatOffset >= taken.size()) return BlockRefusal::WordOutsideBlock;
        if (taken[read.flatOffset] != 0u) return BlockRefusal::WordShared;
        taken[read.flatOffset] = 1u;
    }
    return BlockRefusal::None;
}

BlockRefusal variantsRefusal(const IrResourcePlan& plan, const BlockLayout& layout) {
    for (const auto& buffer : layout.buffers) {
        if (plan.info.buffers[buffer.resource].source >= plan.descriptorSources.size()) return BlockRefusal::BufferSourceMissing;
        if (buffer.formatFixed && buffer.descriptorWords[BufferFormatWord] == NoWord) return BlockRefusal::BufferFormatWordMissing;
    }
    if (plan.info.runtimeImageModes.size() < plan.info.images.size()) return BlockRefusal::ImageModesMissing;
    for (std::size_t index = 0; index < plan.info.images.size(); ++index) {
        const auto source = plan.info.images[index].source;
        if (source >= plan.descriptorSources.size()) return BlockRefusal::ImageSourceMissing;
        if (!multipleModes(plan, index)) continue;
        const auto dwords = sourceDwords(plan, source);
        if (dwords == 0u) return BlockRefusal::ImageWordMissing;
        for (std::uint32_t dword = 0; dword < dwords; ++dword) {
            if (blockWord(plan, source, dword) == NoWord && userWord(plan, source, dword) == NoWord && !constantWord(plan, source, dword)) return BlockRefusal::ImageWordMissing;
        }
    }
    if (!std::ranges::all_of(plan.info.samplers, [&](const SamplerResource& sampler) { return sampler.source < plan.descriptorSources.size(); })) return BlockRefusal::SamplerSourceMissing;
    return BlockRefusal::None;
}

BlockRefusal refusalOf(const IrResourcePlan& plan, const BlockLayout& layout) {
    if (!plan.srtPlanComplete) return BlockRefusal::SrtPlanIncomplete;
    if (!plan.resourceTrackingComplete) return BlockRefusal::ResourceTrackingIncomplete;
    if (std::ranges::any_of(plan.memoryInfo, [](const MemoryInfo& memory) { return memory.gpuDescriptor; })) return BlockRefusal::GpuDescriptor;
    if (readsTables(plan)) return BlockRefusal::ImageTable;
    for (const auto& buffer : plan.info.buffers) {
        if (buffer.written) return BlockRefusal::WrittenBuffer;
        if (buffer.atomic) return BlockRefusal::AtomicBuffer;
    }
    if (plan.uniformFill.fill.kind != UniformFillKind::None) return BlockRefusal::UniformFill;
    if (poisonsWithoutGuard(plan)) return BlockRefusal::UnguardedPoison;
    if (layout.words > BlockAbi::MaxWords) return BlockRefusal::TooManyWords;
    if (layout.buffers.size() > BlockAbi::MaxBuffers) return BlockRefusal::TooManyBuffers;
    if (const auto words = wordsRefusal(plan); words != BlockRefusal::None) return words;
    return variantsRefusal(plan, layout);
}

}

std::uint32_t BlockLayout::SectionBytes() const {
    return BlockAbi::SectionBytes(words, static_cast<std::uint32_t>(buffers.size()));
}

BlockLayout BuildBlockLayout(const IrResourcePlan& plan) {
    BlockLayout layout;
    layout.words = static_cast<std::uint32_t>(plan.srtReads.size());
    layout.buffers.reserve(plan.info.buffers.size());
    for (std::size_t index = 0; index < plan.info.buffers.size(); ++index) {
        const auto& resource = plan.info.buffers[index];
        BlockBuffer buffer;
        buffer.resource = static_cast<std::uint32_t>(index);
        for (std::uint32_t dword = 0; dword < buffer.descriptorWords.size(); ++dword) buffer.descriptorWords[dword] = blockWord(plan, resource.source, dword);
        buffer.formatFixed = resource.descriptorFormatted;
        buffer.typedAlignment = resource.typedAlignment;
        if (buffer.formatFixed && buffer.descriptorWords[BufferFormatWord] != NoWord) layout.variantKey.push_back({buffer.descriptorWords[BufferFormatWord], WholeWord});
        layout.buffers.push_back(buffer);
    }
    for (std::size_t index = 0; index < plan.info.images.size(); ++index) {
        const auto& image = plan.info.images[index];
        if (image.table != NoTable) continue;
        const bool keyed = multipleModes(plan, index);
        const auto dwords = sourceDwords(plan, image.source);
        for (std::uint32_t dword = 0; dword < dwords; ++dword) {
            const auto word = blockWord(plan, image.source, dword);
            if (word != NoWord) {
                layout.imageKey.push_back({word, WholeWord});
                if (keyed) layout.variantKey.push_back({word, WholeWord});
                continue;
            }
            const auto user = userWord(plan, image.source, dword);
            if (user == NoWord) continue;
            layout.userImageKey.push_back({user, WholeWord});
            if (keyed) layout.userVariantKey.push_back({user, WholeWord});
        }
    }
    for (const auto& sampler : plan.info.samplers) {
        if (sampler.table != NoTable) continue;
        const auto dwords = sourceDwords(plan, sampler.source);
        for (std::uint32_t dword = 0; dword < dwords; ++dword) {
            const auto word = blockWord(plan, sampler.source, dword);
            if (word != NoWord) {
                layout.imageKey.push_back({word, WholeWord});
                continue;
            }
            const auto user = userWord(plan, sampler.source, dword);
            if (user != NoWord) layout.userImageKey.push_back({user, WholeWord});
        }
    }
    layout.refusal = refusalOf(plan, layout);
    return layout;
}

std::uint64_t BlockVariantKey(const BlockLayout& layout, std::span<const std::uint32_t> words, std::span<const std::uint32_t> userData) {
    std::uint64_t key = 0xcbf29ce484222325ull;
    for (const auto& bit : layout.variantKey) {
        if (bit.word >= words.size()) throw std::out_of_range("BlockVariantKey: a variant word lies outside the block words");
        key = (key ^ (words[bit.word] & bit.mask)) * 0x100000001b3ull;
    }
    for (const auto& bit : layout.userVariantKey) {
        if (bit.word >= userData.size()) throw std::out_of_range("BlockVariantKey: a variant word lies outside the user data words");
        key = (key ^ (userData[bit.word] & bit.mask)) * 0x100000001b3ull;
    }
    return key;
}

std::uint64_t BlockVariantKey(const BlockLayout& layout, std::span<const std::uint32_t> words) {
    if (!layout.userVariantKey.empty()) throw std::logic_error("BlockVariantKey: the layout keys its variant on user data words, which this call is not given");
    return BlockVariantKey(layout, words, {});
}

std::string_view BlockRefusalName(BlockRefusal refusal) {
    switch (refusal) {
        case BlockRefusal::None: return "None";
        case BlockRefusal::GpuDescriptor: return "GpuDescriptor";
        case BlockRefusal::ImageTable: return "ImageTable";
        case BlockRefusal::WrittenBuffer: return "WrittenBuffer";
        case BlockRefusal::AtomicBuffer: return "AtomicBuffer";
        case BlockRefusal::UniformFill: return "UniformFill";
        case BlockRefusal::UnguardedPoison: return "UnguardedPoison";
        case BlockRefusal::TooManyWords: return "TooManyWords";
        case BlockRefusal::TooManyBuffers: return "TooManyBuffers";
        case BlockRefusal::SrtPlanIncomplete: return "SrtPlanIncomplete";
        case BlockRefusal::ResourceTrackingIncomplete: return "ResourceTrackingIncomplete";
        case BlockRefusal::WordWithoutValue: return "WordWithoutValue";
        case BlockRefusal::WordOutsideBlock: return "WordOutsideBlock";
        case BlockRefusal::WordShared: return "WordShared";
        case BlockRefusal::BufferSourceMissing: return "BufferSourceMissing";
        case BlockRefusal::BufferFormatWordMissing: return "BufferFormatWordMissing";
        case BlockRefusal::ImageModesMissing: return "ImageModesMissing";
        case BlockRefusal::ImageSourceMissing: return "ImageSourceMissing";
        case BlockRefusal::ImageWordMissing: return "ImageWordMissing";
        case BlockRefusal::SamplerSourceMissing: return "SamplerSourceMissing";
        case BlockRefusal::Count: break;
    }
    return "Unknown";
}

}
