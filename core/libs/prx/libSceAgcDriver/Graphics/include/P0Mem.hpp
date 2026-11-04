#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_P0MEM_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_P0MEM_HPP

#include <cstdint>
#include <span>
#include <utility>

namespace AgcDriver::Graphics::P0Mem {

[[nodiscard]] bool CeilingStaging();
[[nodiscard]] bool CeilingPublish();

enum class CeilingCount : std::uint8_t { CopyInSkipped, CopyInFirst, PristineSkipped, CopyBackSkipped, PublishSkipped, Count };
void CountCeiling(CeilingCount which, std::uint64_t bytes);

[[nodiscard]] bool CensusEnabled();

enum class Read : std::uint8_t { Bound, Reach, Count };
void NoteCpuWrite(std::uint64_t address, std::uint64_t bytes);
void NoteGpuRead(Read kind, std::uint64_t begin, std::uint64_t end);
void NoteGpuReads(Read kind, std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges);
void NoteGpuWrite(std::uint64_t begin, std::uint64_t end);
void NotePublish(std::uint64_t begin, std::uint64_t end, std::uint8_t reason, const char* reasonName);
void NoteUnboundedWriter(std::span<const std::pair<std::uint64_t, std::uint64_t>> writableLease, std::uint64_t boundWrittenBytes);
void NoteBdaReport(std::uint64_t bytes);

void Close(const char* window);

}

#endif
