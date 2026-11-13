#include <relinker/pipeline/RedZoneFraming.hpp>
#include <relinker/analysis/UnusedNidFilter.hpp>
#include <relinker/analysis/UnusedNidFilter/EhFrameReader.hpp>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>
#include <sstream>

namespace Relinker {

RedZoneFramingResult FrameRedZoneImage(std::vector<std::uint8_t>& bytes, const std::vector<Domain::ProgramHeader>& headers, const std::vector<std::uint8_t>& relaData, std::vector<Codegen::TrampolineSite>& trampolines, const std::string& prefix, const std::function<void(std::string_view)>& message) {
    RedZoneFramingResult result;
    result.FirstSite = trampolines.size();
    std::vector<Domain::ProgramHeader> codeSegments;
    std::vector<Domain::ProgramHeader> loadSegments;
    for (const auto& header : headers) if (header.Type == 1 && (header.Flags & 1) != 0) codeSegments.push_back(header);
    for (const auto& header : headers) if (header.Type == 1) loadSegments.push_back(header);
    std::vector<Codegen::RedZoneFunction> functions;
    for (const auto& region : UnusedNidFilter::ReadExceptionFunctions(bytes, headers, {}, {})) functions.push_back({region.Begin, region.End});
    if (functions.empty()) {
        result.HasBounds = false;
        return result;
    }
    // R_X86_64_RELATIVE addends: tables of absolute addresses are zero in the file.
    std::map<std::uint64_t, std::uint64_t> relativeAddends;
    for (std::size_t at = 0; at + 24 <= relaData.size(); at += 24) {
        std::uint64_t offset = 0;
        std::uint64_t info = 0;
        std::uint64_t addend = 0;
        std::memcpy(&offset, relaData.data() + at, 8);
        std::memcpy(&info, relaData.data() + at + 8, 8);
        std::memcpy(&addend, relaData.data() + at + 16, 8);
        if ((info & 0xffffffffu) == 8) relativeAddends.emplace(offset, addend);
    }
    Codegen::RedZoneFrameReport report;
    auto framed = Codegen::FrameRedZoneFunctions(bytes, codeSegments, functions, trampolines, report, result.Displacements, loadSegments, relativeAddends);
    for (const auto& moved : result.Displacements) {
        if (moved.Offset > bytes.size() || moved.Bytes.size() > bytes.size() - moved.Offset)
            throw Domain::RelinkerException("Red-zone displacement exceeds source image", moved.Offset);
        std::copy(moved.Bytes.begin(), moved.Bytes.end(), bytes.begin() + static_cast<std::ptrdiff_t>(moved.Offset));
    }
    const auto say = [&](const std::ostringstream& text) { message(text.str()); };
    { std::ostringstream text; text << prefix << ": " << report.Functions << " functions, " << report.RedZoneFunctions << " leaf functions keep values below RSP, " << report.Framed << " framed (" << report.FramedWithoutFramePointer << " without a frame pointer, " << report.MovedOperands << " rsp operands moved, " << report.StubbedOperands << " in " << report.OperandStubs << " operand stubs; " << report.Exits << " exits)\n"; say(text); }
    if (report.ReturnThunks != 0) { std::ostringstream text; text << prefix << ": " << report.ReturnThunks << " leaves return through a thunk, as their exits could not be patched\n"; say(text); }
    if (report.Relocated != 0) { std::ostringstream text; text << prefix << ": " << report.Relocated << " leaves relocated whole into a stub, as their frames could not be patched in place (" << report.RelocatedCases << " jump-table cases enter through a jump; " << report.RelocatedOpenJumps << " leaves have an indirect jump that leaves the stub)\n"; say(text); }
    if (report.WidenedPrologues != 0 || report.WidenedExits != 0) { std::ostringstream text; text << prefix << ": " << report.WidenedPrologues << " prologues widened over the instructions after them and " << report.WidenedExits << " exits over the instructions before them, as they were shorter than a jump (" << report.WidenedOperands << " rsp operands moved into their stubs)\n"; say(text); }
    for (const auto& [reason, count] : report.Skipped) {
        result.Skipped += count;
        std::ostringstream text; text << prefix << ": skipped " << count << ": " << reason << '\n'; say(text);
    }
    trampolines.insert(trampolines.end(), std::make_move_iterator(framed.begin()), std::make_move_iterator(framed.end()));
    return result;
}

}
