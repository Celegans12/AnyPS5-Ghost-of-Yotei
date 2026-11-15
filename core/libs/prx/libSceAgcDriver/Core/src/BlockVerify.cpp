#include "prx/libSceAgcDriver/Core/include/BlockVerify.hpp"
#include "BlockAbi.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Core {

namespace {

namespace Abi = ShaderRecompiler::BlockAbi;

bool entryAbsent(std::uint64_t address, std::uint32_t service) {
    return (service & Abi::ServiceAbsent) != 0 || address == 0;
}

const char* kindName(BlockMismatch::Kind kind) {
    switch (kind) {
    case BlockMismatch::Kind::Word: return "word";
    case BlockMismatch::Kind::Presence: return "presence";
    case BlockMismatch::Kind::Bytes: return "bytes";
    case BlockMismatch::Kind::Bound: return "bound";
    case BlockMismatch::Kind::Content: return "content";
    }
    return "unknown";
}

void add(std::vector<BlockMismatch>& out, BlockMismatch::Kind kind, std::uint32_t index, std::uint64_t expected, std::uint64_t found) {
    BlockMismatch mismatch{};
    mismatch.kind = kind;
    mismatch.index = index;
    mismatch.expected = expected;
    mismatch.found = found;
    out.push_back(mismatch);
}

void verifyWords(const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> section, std::span<const std::uint32_t> classicFlat, std::vector<BlockMismatch>& out) {
    if (classicFlat.size() < layout.words) {
        add(out, BlockMismatch::Kind::Word, layout.words, classicFlat.size(), layout.words);
    }
    const std::size_t compared = std::min<std::size_t>({layout.words, classicFlat.size(), section.size()});
    for (std::size_t k = 0; k < compared; ++k) {
        if (section[k] != classicFlat[k]) {
            add(out, BlockMismatch::Kind::Word, static_cast<std::uint32_t>(k), classicFlat[k], section[k]);
        }
    }
}

void verifyBuffer(std::uint32_t buffer, std::span<const std::uint32_t> entry, const BufferTruth* truth, std::span<const std::byte> (*bytesAt)(void* context, std::uint64_t deviceAddress, std::uint32_t bytes), void* context, std::vector<BlockMismatch>& out) {
    const std::uint64_t address = static_cast<std::uint64_t>(entry[Abi::EntryAddressLow]) | (static_cast<std::uint64_t>(entry[Abi::EntryAddressHigh]) << 32u);
    const std::uint32_t entryBytes = entry[Abi::EntryBytes];
    const bool found = !entryAbsent(address, entry[Abi::EntryService]);
    const bool expected = truth != nullptr && truth->present;
    if (found != expected) {
        add(out, BlockMismatch::Kind::Presence, buffer, expected ? 1u : 0u, found ? 1u : 0u);
        return;
    }
    if (!expected) return;
    bool addressRight = true;
    if (address != truth->address) {
        add(out, BlockMismatch::Kind::Word, buffer, truth->address, address);
        addressRight = false;
    }
    const std::uint64_t truthBytes = truth->bytes.size();
    if (entryBytes != truthBytes) {
        add(out, BlockMismatch::Kind::Bound, buffer, truthBytes, entryBytes);
    }
    if (!addressRight || truthBytes == 0) return;
    if (truthBytes > std::numeric_limits<std::uint32_t>::max()) {
        add(out, BlockMismatch::Kind::Bytes, buffer, truthBytes, 0);
        return;
    }
    const std::uint32_t want = static_cast<std::uint32_t>(truthBytes);
    const std::span<const std::byte> shadow = bytesAt != nullptr ? bytesAt(context, address, want) : std::span<const std::byte>{};
    if (shadow.size() < want) {
        add(out, BlockMismatch::Kind::Bytes, buffer, want, shadow.size());
        return;
    }
    for (std::uint32_t offset = 0; offset < want; ++offset) {
        if (shadow[offset] != truth->bytes[offset]) {
            add(out, BlockMismatch::Kind::Content, offset, std::to_integer<std::uint8_t>(truth->bytes[offset]), std::to_integer<std::uint8_t>(shadow[offset]));
            return;
        }
    }
}

}

std::vector<BlockMismatch> VerifyBlock(const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> section, std::span<const std::uint32_t> classicFlat, std::span<const BufferTruth> classicBuffers, std::span<const std::byte> (*bytesAt)(void* context, std::uint64_t deviceAddress, std::uint32_t bytes), void* context) {
    std::vector<BlockMismatch> out;
    const std::size_t buffers = layout.buffers.size();
    const std::size_t needed = static_cast<std::size_t>(Abi::SectionWords(layout.words, static_cast<std::uint32_t>(buffers)));
    if (section.size() < needed) {
        add(out, BlockMismatch::Kind::Bytes, static_cast<std::uint32_t>(section.size()), needed, section.size());
        return out;
    }
    verifyWords(layout, section, classicFlat, out);
    for (std::size_t i = 0; i < buffers; ++i) {
        const std::size_t first = static_cast<std::size_t>(Abi::BufferEntryWord(layout.words, static_cast<std::uint32_t>(i)));
        const BufferTruth* truth = i < classicBuffers.size() ? &classicBuffers[i] : nullptr;
        verifyBuffer(static_cast<std::uint32_t>(i), section.subspan(first, Abi::BufferEntryWords), truth, bytesAt, context, out);
    }
    return out;
}

std::string Describe(const BlockMismatch& mismatch) {
    char text[192];
    switch (mismatch.kind) {
    case BlockMismatch::Kind::Word:
        std::snprintf(text, sizeof(text), "%s %u: expected 0x%llx, block holds 0x%llx", kindName(mismatch.kind), mismatch.index, static_cast<unsigned long long>(mismatch.expected), static_cast<unsigned long long>(mismatch.found));
        break;
    case BlockMismatch::Kind::Presence:
        std::snprintf(text, sizeof(text), "%s of buffer %u: classic path binds it %s, block marks it %s", kindName(mismatch.kind), mismatch.index, mismatch.expected != 0 ? "present" : "absent", mismatch.found != 0 ? "present" : "absent");
        break;
    case BlockMismatch::Kind::Bytes:
        std::snprintf(text, sizeof(text), "%s unavailable at %u: needed %llu, got %llu", kindName(mismatch.kind), mismatch.index, static_cast<unsigned long long>(mismatch.expected), static_cast<unsigned long long>(mismatch.found));
        break;
    case BlockMismatch::Kind::Bound:
        std::snprintf(text, sizeof(text), "%s of buffer %u: classic size %llu bytes, block bound %llu bytes", kindName(mismatch.kind), mismatch.index, static_cast<unsigned long long>(mismatch.expected), static_cast<unsigned long long>(mismatch.found));
        break;
    case BlockMismatch::Kind::Content:
        std::snprintf(text, sizeof(text), "%s differs at byte %u: expected 0x%llx, shadow holds 0x%llx", kindName(mismatch.kind), mismatch.index, static_cast<unsigned long long>(mismatch.expected), static_cast<unsigned long long>(mismatch.found));
        break;
    default:
        std::snprintf(text, sizeof(text), "unknown mismatch %u", mismatch.index);
        break;
    }
    return std::string(text);
}

}
