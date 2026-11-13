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
    // Sites appended to the trampoline list, starting at this index.
    std::size_t FirstSite = 0;
    std::vector<Codegen::RedZoneDisplacement> Displacements;
    // Leaves that keep values below RSP and were not framed; zero is the phase 1 gate.
    std::size_t Skipped = 0;
    // False when the image has no .eh_frame function bounds, so nothing was framed.
    bool HasBounds = true;
};

// Frames the red-zone leaves of one image (the executable or a guest module). The image's
// displacements are applied to bytes, and the framing sites are appended to trampolines. Messages
// start with prefix ("Red-zone framing" or "Red-zone framing (<module>)").
RedZoneFramingResult FrameRedZoneImage(std::vector<std::uint8_t>& bytes, const std::vector<Domain::ProgramHeader>& headers, const std::vector<std::uint8_t>& relaData, std::vector<Codegen::TrampolineSite>& trampolines, const std::string& prefix, const std::function<void(std::string_view)>& message);

}

#endif
