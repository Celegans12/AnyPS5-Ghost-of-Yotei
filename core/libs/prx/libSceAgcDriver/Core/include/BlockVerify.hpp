#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKVERIFY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_BLOCKVERIFY_HPP

#include "Optimization/BlockLayout.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace AgcDriver::Core {

struct BufferTruth { std::uint64_t address = 0; std::span<const std::byte> bytes; bool present = false; };
struct BlockMismatch { enum class Kind : std::uint8_t { Word, Presence, Bytes, Bound, Content } kind; std::uint32_t index = 0; std::uint64_t expected = 0; std::uint64_t found = 0; };

[[nodiscard]] std::vector<BlockMismatch> VerifyBlock(const ShaderRecompiler::BlockLayout& layout, std::span<const std::uint32_t> section, std::span<const std::uint32_t> classicFlat, std::span<const BufferTruth> classicBuffers, std::span<const std::byte> (*bytesAt)(void* context, std::uint64_t deviceAddress, std::uint32_t bytes), void* context);
[[nodiscard]] std::string Describe(const BlockMismatch& mismatch);

}

#endif
