#ifndef RELINKER_PIPELINE_REDZONEFRAMING_HPP
#define RELINKER_PIPELINE_REDZONEFRAMING_HPP

#include <codegen/CodegenTypes.hpp>
#include <codegen/x86/RedZoneFramer.hpp>
#include <domain/Types.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace Relinker {

struct RedZoneFramingResult {
    std::size_t FirstSite = 0;
    std::vector<Codegen::RedZoneDisplacement> Displacements;
    std::size_t Skipped = 0;
    bool HasBounds = true;
    std::size_t SplitFunctions = 0;
};

RedZoneFramingResult FrameRedZoneImage(std::vector<std::uint8_t>& bytes, const std::vector<Domain::ProgramHeader>& headers, const std::vector<std::uint8_t>& relaData, std::vector<Codegen::TrampolineSite>& trampolines, const std::string& prefix, const std::function<void(std::string_view)>& message);

}

#endif
