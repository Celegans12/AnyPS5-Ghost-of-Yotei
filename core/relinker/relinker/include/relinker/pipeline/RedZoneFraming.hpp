#ifndef RELINKER_PIPELINE_REDZONEFRAMING_HPP
#define RELINKER_PIPELINE_REDZONEFRAMING_HPP

#include <codegen/CodegenTypes.hpp>
#include <codegen/x86/RedZoneFramer.hpp>
#include <domain/Types.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Relinker {

struct RedZoneAddressRange {
    std::uint64_t Begin = 0;
    std::uint64_t End = 0;
};

struct RedZoneRvaRange {
    std::uint32_t Begin = 0;
    std::uint32_t End = 0;
};

struct RedZoneFramingResult {
    std::size_t FirstSite = 0;
    std::vector<Codegen::RedZoneDisplacement> Displacements;
    std::size_t Skipped = 0;
    std::size_t Unexamined = 0;
    bool HasBounds = true;
    std::size_t SplitFunctions = 0;
    std::vector<RedZoneAddressRange> Unframed;
};

inline constexpr std::size_t RedZoneMarkOffset = 0x28;
inline constexpr std::array<std::uint8_t, 8> RedZoneMark = {'A', 'P', 'S', '5', 'R', 'Z', '0', '1'};
inline constexpr std::array<std::uint8_t, 8> RedZoneRangesMark = {'A', 'P', 'S', '5', 'R', 'Z', '0', '2'};
inline constexpr std::size_t RedZoneTableRvaOffset = 0x30;
inline constexpr std::size_t RedZoneTableCountOffset = 0x34;
inline constexpr std::size_t RedZoneTableEntryBytes = 8;

RedZoneFramingResult FrameRedZoneImage(std::vector<std::uint8_t>& bytes, const std::vector<Domain::ProgramHeader>& headers, const std::vector<std::uint8_t>& relaData, std::vector<Codegen::TrampolineSite>& trampolines, const std::string& prefix, const std::function<void(std::string_view)>& message);
bool RedZoneFramingComplete(const RedZoneFramingResult& result);
std::optional<std::vector<RedZoneRvaRange>> RedZoneRvaRanges(const std::vector<std::uint8_t>& bytes, const std::vector<Domain::ProgramHeader>& headers, const RedZoneFramingResult& result);
bool MarkRedZoneFramed(std::vector<std::uint8_t>& image, const std::vector<RedZoneRvaRange>& unframed = {});

}

#endif
