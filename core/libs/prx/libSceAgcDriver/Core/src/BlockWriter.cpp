#include "prx/libSceAgcDriver/Core/include/BlockWriter.hpp"
#include "BlockAbi.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Core {

namespace {

namespace Abi = ShaderRecompiler::BlockAbi;

constexpr std::uint64_t BaseMask = 0x0000ffffffffffffull;
constexpr std::uint32_t StrideShift = 16u;
constexpr std::uint32_t StrideMask = 0x3fffu;
constexpr std::uint32_t LocatingWords = 3u;

bool locatable(const ShaderRecompiler::BlockBuffer& buffer, std::size_t words) {
    for (std::uint32_t word = 0; word < LocatingWords; ++word) {
        if (buffer.descriptorWords[word] >= words) return false;
    }
    return true;
}

bool walkable(const BlockInputs& inputs) {
    if (inputs.plan == nullptr || inputs.layout == nullptr || inputs.tape == nullptr || inputs.read == nullptr || inputs.resolve == nullptr) return false;
    const auto& layout = *inputs.layout;
    if (!layout.Eligible() || layout.abiVersion != Abi::Version) return false;
    if (layout.words > Abi::MaxWords || layout.buffers.size() > Abi::MaxBuffers) return false;
    if (layout.words != inputs.plan->srtReads.size()) return false;
    for (const auto& buffer : layout.buffers) {
        if (!locatable(buffer, layout.words)) return false;
    }
    return true;
}

bool keyed(const ShaderRecompiler::BlockLayout& layout) {
    for (const auto& bit : layout.variantKey) {
        if (bit.word >= layout.words) return false;
    }
    return true;
}

void writeEntry(std::uint32_t* entry, std::uint64_t deviceAddress, std::uint32_t bytes, std::uint32_t service) {
    entry[Abi::EntryAddressLow] = static_cast<std::uint32_t>(deviceAddress);
    entry[Abi::EntryAddressHigh] = static_cast<std::uint32_t>(deviceAddress >> 32u);
    entry[Abi::EntryBytes] = bytes;
    entry[Abi::EntryService] = service;
}

void writeAnswer(std::uint32_t* entry, const BufferRequest& request, const BufferAnswer& answer) {
    if (!request.present || answer.deviceAddress == 0u || answer.bytes == 0u) {
        writeEntry(entry, 0u, 0u, Abi::ServiceAbsent);
        return;
    }
    writeEntry(entry, answer.deviceAddress, answer.bytes, request.misalignment & Abi::ServiceMisalignmentMask);
}

BlockOutput faulted(BlockFault fault) {
    BlockOutput output;
    output.fault = fault;
    return output;
}

}

BufferRequest DecodeBuffer(const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> words, std::uint32_t buffer) {
    BufferRequest request;
    request.buffer = buffer;
    if (buffer >= layout.buffers.size()) return request;
    const auto& entry = layout.buffers[buffer];
    if (!locatable(entry, words.size())) return request;
    const auto word0 = words[entry.descriptorWords[0]];
    const auto word1 = words[entry.descriptorWords[1]];
    const auto records = words[entry.descriptorWords[2]];
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

BlockOutput WriteBlock(const BlockInputs& inputs, BlockRing& ring) {
    if (!walkable(inputs)) return faulted(BlockFault::Walk);
    const auto& layout = *inputs.layout;
    if (!keyed(layout)) return faulted(BlockFault::Variant);
    const auto buffers = static_cast<std::uint32_t>(layout.buffers.size());
    const auto sectionWords = Abi::SectionWords(layout.words, buffers);
    const auto sectionBytes = Abi::SectionBytes(layout.words, buffers);
    if (sectionBytes == 0u) return ShaderRecompiler::BlockVariantKey(layout, {}) == inputs.variantKey ? BlockOutput{} : faulted(BlockFault::Variant);
    const auto slice = ring.Take(sectionBytes, Abi::SectionAlignmentBytes);
    if (slice.bytes == nullptr) return faulted(BlockFault::RingFull);
    auto* section = reinterpret_cast<std::uint32_t*>(slice.bytes);
    const std::span<std::uint32_t> words(section, layout.words);
    const auto walked = inputs.tape->RunInto(*inputs.plan, inputs.userData, inputs.shaderBase, inputs.read, inputs.context, words);
    if (walked == ShaderRecompiler::BlockWalk::Unreadable) return faulted(BlockFault::Unreadable);
    if (walked != ShaderRecompiler::BlockWalk::Done) return faulted(BlockFault::Walk);
    if (ShaderRecompiler::BlockVariantKey(layout, words) != inputs.variantKey) return faulted(BlockFault::Variant);
    BlockOutput output;
    for (std::uint32_t buffer = 0; buffer < buffers; ++buffer) {
        auto* entry = section + Abi::BufferEntryWord(layout.words, buffer);
        const auto request = DecodeBuffer(layout, words, buffer);
        BufferAnswer answer;
        if (request.present && !inputs.resolve(inputs.context, request, answer)) return faulted(BlockFault::Unreadable);
        if (request.present && answer.deferred) {
            writeEntry(entry, 0u, 0u, 0u);
            output.deferred |= std::uint64_t{1} << buffer;
            continue;
        }
        writeAnswer(entry, request, answer);
    }
    for (std::uint32_t word = sectionWords; word < sectionBytes / sizeof(std::uint32_t); ++word) section[word] = 0u;
    output.section = slice;
    return output;
}

void CompleteBuffer(const ShaderRecompiler::BlockLayout& layout, const RingSlice& section, std::uint32_t buffer, const BufferAnswer& answer) {
    if (section.bytes == nullptr || buffer >= layout.buffers.size() || answer.deferred) return;
    auto* words = reinterpret_cast<std::uint32_t*>(section.bytes);
    const auto request = DecodeBuffer(layout, std::span<const std::uint32_t>(words, layout.words), buffer);
    writeAnswer(words + Abi::BufferEntryWord(layout.words, buffer), request, answer);
}

}
