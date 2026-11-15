#ifndef CORE_SHADER_RECOMPILER_BLOCKABI_HPP
#define CORE_SHADER_RECOMPILER_BLOCKABI_HPP

#include <cstdint>

namespace ShaderRecompiler::BlockAbi {

inline constexpr std::uint32_t Version = 1;
inline constexpr std::uint32_t SectionAlignmentBytes = 16;
inline constexpr std::uint32_t MaxWords = 1024;
inline constexpr std::uint32_t MaxBuffers = 64;
inline constexpr std::uint32_t BufferEntryWords = 4;
inline constexpr std::uint32_t EntryAddressLow = 0;
inline constexpr std::uint32_t EntryAddressHigh = 1;
inline constexpr std::uint32_t EntryBytes = 2;
inline constexpr std::uint32_t EntryService = 3;
inline constexpr std::uint32_t ServiceMisalignmentMask = 3u;
inline constexpr std::uint32_t ServiceAbsent = 0x80000000u;
inline constexpr std::uint32_t PushAddressDwords = 2;

constexpr std::uint32_t BufferEntryWord(std::uint32_t words, std::uint32_t buffer) { return words + buffer * BufferEntryWords; }
constexpr std::uint32_t SectionWords(std::uint32_t words, std::uint32_t buffers) { return words + buffers * BufferEntryWords; }
constexpr std::uint32_t SectionBytes(std::uint32_t words, std::uint32_t buffers) { return (SectionWords(words, buffers) * 4u + SectionAlignmentBytes - 1u) / SectionAlignmentBytes * SectionAlignmentBytes; }

}

#endif
