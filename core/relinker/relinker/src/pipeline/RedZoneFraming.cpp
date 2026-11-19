#include <relinker/pipeline/RedZoneFraming.hpp>
#include <relinker/analysis/UnusedNidFilter.hpp>
#include <relinker/analysis/UnusedNidFilter/EhFrameReader.hpp>
#include <elfpatcher/windows/WindowsLoadImage.hpp>
#include <io/BufferUtils.hpp>
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
    result.HasBounds = !functions.empty();
    const auto pieces = Codegen::SplitUncoveredCode(bytes, codeSegments, functions);
    result.SplitFunctions = pieces.size();
    if (!pieces.empty()) message(prefix + ": " + std::to_string(pieces.size()) + " functions taken from code outside .eh_frame bounds, split at int3 padding and call targets\n");
    functions.insert(functions.end(), pieces.begin(), pieces.end());
    if (functions.empty()) return result;
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
    Codegen::ApplyRedZoneSiteEdits(trampolines, report.SiteEdits);
    for (const auto& moved : result.Displacements) {
        if (moved.Offset > bytes.size() || moved.Bytes.size() > bytes.size() - moved.Offset)
            throw Domain::RelinkerException("Red-zone displacement exceeds source image", moved.Offset);
        std::copy(moved.Bytes.begin(), moved.Bytes.end(), bytes.begin() + static_cast<std::ptrdiff_t>(moved.Offset));
    }
    const auto say = [&](const std::ostringstream& text) { message(text.str()); };
    { std::ostringstream text; text << prefix << ": " << report.Functions << " functions, " << report.RedZoneFunctions << " leaf functions keep values below RSP, " << report.Framed << " framed (" << report.FramedWithoutFramePointer << " without a frame pointer, " << report.MovedOperands << " rsp operands moved, " << report.StubbedOperands << " in " << report.OperandStubs << " operand stubs; " << report.Exits << " exits)\n"; say(text); }
    if (report.LeaveExits != 0) { std::ostringstream text; text << prefix << ": " << report.LeaveExits << " exits leave through `leave` instead of an exit stub (frame pointer, nothing else pushed)\n"; say(text); }
    if (report.FramedWithCalls != 0) { std::ostringstream text; text << prefix << ": " << report.FramedWithCalls << " frame-pointer functions that call keep values below RSP and are framed too\n"; say(text); }
    if (report.ReturnThunks != 0) { std::ostringstream text; text << prefix << ": " << report.ReturnThunks << " leaves return through a thunk, as their exits could not be patched\n"; say(text); }
    if (report.Relocated != 0) { std::ostringstream text; text << prefix << ": " << report.Relocated << " leaves relocated whole into a stub, as their frames could not be patched in place (" << report.RelocatedCases << " jump-table cases enter through a jump; " << report.RelocatedOpenJumps << " leaves have an indirect jump that leaves the stub)\n"; say(text); }
    if (report.WidenedPrologues != 0 || report.WidenedExits != 0) { std::ostringstream text; text << prefix << ": " << report.WidenedPrologues << " prologues widened over the instructions after them and " << report.WidenedExits << " exits over the instructions before them, as they were shorter than a jump (" << report.WidenedOperands << " rsp operands moved into their stubs)\n"; say(text); }
    if (report.UncoveredRanges != 0) {
        std::ostringstream text;
        text << prefix << ": " << report.UncoveredRanges << " code ranges (" << report.UncoveredBytes << " bytes) outside every function's .eh_frame bounds, not examined; " << report.UncoveredPltRanges << " hold PLT-style jumps, " << report.UncoveredBelowRsp << " access memory below rsp";
        for (std::size_t index = 0; index < report.UncoveredBelowRspAddresses.size() && index < 8; ++index) text << (index == 0 ? " (at 0x" : ", 0x") << std::hex << report.UncoveredBelowRspAddresses[index] << std::dec;
        if (!report.UncoveredBelowRspAddresses.empty()) text << (report.UncoveredBelowRspAddresses.size() > 8 ? ", ...)" : ")");
        text << '\n';
        say(text);
    }
    if (report.EditedStubs != 0) { std::ostringstream text; text << prefix << ": " << report.EditedStubs << " rsp operands that Intel conversion stubs took in from the instructions after an instruction they lower were moved in those stubs\n"; say(text); }
    result.Unexamined = report.UncoveredBelowRsp;
    for (const auto& [reason, count] : report.Skipped) {
        result.Skipped += count;
        std::ostringstream text;
        text << prefix << ": skipped " << count << ": " << reason;
        std::size_t listed = 0;
        for (const auto& range : report.Unframed) {
            if (range.Reason != reason) continue;
            if (listed == 8) { text << ", ..."; break; }
            text << (listed == 0 ? " (at 0x" : ", 0x") << std::hex << range.Begin << std::dec;
            if (!range.FirstReason.empty() && range.FirstReason != reason) text << " after \"" << range.FirstReason << '"';
            ++listed;
        }
        if (listed != 0) text << ')';
        text << '\n';
        say(text);
    }
    std::vector<RedZoneAddressRange> ranges;
    for (const auto& range : report.Unframed) if (range.End > range.Begin) ranges.push_back({range.Begin, range.End});
    std::sort(ranges.begin(), ranges.end(), [](const RedZoneAddressRange& left, const RedZoneAddressRange& right) { return left.Begin < right.Begin; });
    for (const auto& range : ranges) {
        if (!result.Unframed.empty() && range.Begin <= result.Unframed.back().End) result.Unframed.back().End = std::max(result.Unframed.back().End, range.End);
        else result.Unframed.push_back(range);
    }
    if (!result.Unframed.empty()) {
        std::uint64_t bytesLeft = 0;
        for (const auto& range : result.Unframed) bytesLeft += range.End - range.Begin;
        std::ostringstream text;
        text << prefix << ": " << result.Unframed.size() << " code ranges (" << bytesLeft << " bytes) keep their red zone and go into the mark of the image: protection stays on elsewhere, and a fault taken inside them stops the title with a message\n";
        say(text);
    }
    trampolines.insert(trampolines.end(), std::make_move_iterator(framed.begin()), std::make_move_iterator(framed.end()));
    return result;
}

bool RedZoneFramingComplete(const RedZoneFramingResult& result) {
    return result.Skipped == 0 && result.Unexamined == 0;
}

std::optional<std::vector<RedZoneRvaRange>> RedZoneRvaRanges(const std::vector<std::uint8_t>& bytes, const std::vector<Domain::ProgramHeader>& headers, const RedZoneFramingResult& result) {
    std::vector<RedZoneRvaRange> ranges;
    if (result.Unframed.empty()) return ranges;
    try {
        const Elfpatcher::Windows::WindowsLoadImage image(bytes, headers, false);
        for (const auto& range : result.Unframed) {
            const auto begin = image.GetRva(range.Begin, range.End - range.Begin);
            ranges.push_back({begin, static_cast<std::uint32_t>(begin + (range.End - range.Begin))});
        }
    } catch (const Domain::RelinkerException&) {
        return std::nullopt;
    }
    std::sort(ranges.begin(), ranges.end(), [](const RedZoneRvaRange& left, const RedZoneRvaRange& right) { return left.Begin < right.Begin; });
    std::vector<RedZoneRvaRange> merged;
    for (const auto& range : ranges) {
        if (!merged.empty() && range.Begin <= merged.back().End) merged.back().End = std::max(merged.back().End, range.End);
        else merged.push_back(range);
    }
    return merged;
}

namespace {

constexpr std::size_t kSectionHeaderBytes = 40;
constexpr std::size_t kPeSignatureBytes = 4;
constexpr std::size_t kCoffHeaderBytes = 20;
constexpr std::uint32_t kRangesSectionCharacteristics = 0x40000040u;

bool appendRangeTable(std::vector<std::uint8_t>& image, const std::vector<RedZoneRvaRange>& ranges, std::uint32_t& tableRva) {
    const std::size_t peOffset = Io::ReadU32(image, 0x3c);
    const std::size_t coff = peOffset + kPeSignatureBytes;
    const std::size_t optional = coff + kCoffHeaderBytes;
    if (peOffset < 0x40 || optional + 64 > image.size() || Io::ReadU32(image, peOffset) != 0x4550 || Io::ReadU16(image, optional) != 0x20b) return false;
    const std::size_t sections = Io::ReadU16(image, coff + 2);
    const std::size_t optionalBytes = Io::ReadU16(image, coff + 16);
    const std::uint32_t sectionAlignment = Io::ReadU32(image, optional + 32);
    const std::uint32_t fileAlignment = Io::ReadU32(image, optional + 36);
    const std::uint32_t imageBytes = Io::ReadU32(image, optional + 56);
    const std::uint32_t headerBytes = Io::ReadU32(image, optional + 60);
    if (optionalBytes < 64 || sections >= 96 || sectionAlignment < 0x1000 || (sectionAlignment & (sectionAlignment - 1)) != 0 || fileAlignment < 0x200 || (fileAlignment & (fileAlignment - 1)) != 0 || imageBytes % sectionAlignment != 0) return false;
    const std::size_t table = optional + optionalBytes;
    const std::size_t slot = table + sections * kSectionHeaderBytes;
    if (slot + kSectionHeaderBytes > headerBytes || slot + kSectionHeaderBytes > image.size()) return false;
    if (!std::all_of(image.begin() + static_cast<std::ptrdiff_t>(slot), image.begin() + static_cast<std::ptrdiff_t>(slot + kSectionHeaderBytes), [](const std::uint8_t value) { return value == 0; })) return false;
    const std::uint64_t dataBytes = std::uint64_t{ranges.size()} * RedZoneTableEntryBytes;
    const std::uint64_t rawBytes = Io::AlignUp64(dataBytes, fileAlignment);
    const std::uint64_t rawOffset = Io::AlignUp64(std::uint64_t{image.size()}, fileAlignment);
    const std::uint64_t memoryBytes = Io::AlignUp64(dataBytes, sectionAlignment);
    if (dataBytes == 0 || rawOffset + rawBytes > 0x7fffffffu || std::uint64_t{imageBytes} + memoryBytes > 0x77000000u) return false;
    constexpr std::string_view name = ".rzgap";
    std::copy(name.begin(), name.end(), image.begin() + static_cast<std::ptrdiff_t>(slot));
    Io::WriteU32(image, slot + 8, static_cast<std::uint32_t>(dataBytes));
    Io::WriteU32(image, slot + 12, imageBytes);
    Io::WriteU32(image, slot + 16, static_cast<std::uint32_t>(rawBytes));
    Io::WriteU32(image, slot + 20, static_cast<std::uint32_t>(rawOffset));
    Io::WriteU32(image, slot + 36, kRangesSectionCharacteristics);
    image.resize(static_cast<std::size_t>(rawOffset), 0);
    for (const auto& range : ranges) {
        Io::AppendU32(image, range.Begin);
        Io::AppendU32(image, range.End);
    }
    image.resize(static_cast<std::size_t>(rawOffset + rawBytes), 0);
    Io::WriteU16(image, coff + 2, static_cast<std::uint16_t>(sections + 1));
    Io::WriteU32(image, optional + 8, static_cast<std::uint32_t>(Io::ReadU32(image, optional + 8) + rawBytes));
    Io::WriteU32(image, optional + 56, static_cast<std::uint32_t>(imageBytes + memoryBytes));
    tableRva = imageBytes;
    return true;
}

}

bool MarkRedZoneFramed(std::vector<std::uint8_t>& image, const std::vector<RedZoneRvaRange>& unframed) {
    constexpr std::size_t dosHeaderBytes = 0x40;
    if (image.size() < dosHeaderBytes || image[0] != 'M' || image[1] != 'Z' || Io::ReadU32(image, 0x3c) < dosHeaderBytes) return false;
    const auto mark = image.begin() + static_cast<std::ptrdiff_t>(RedZoneMarkOffset);
    const auto reserved = image.begin() + static_cast<std::ptrdiff_t>(RedZoneTableRvaOffset);
    if (!std::all_of(mark, mark + static_cast<std::ptrdiff_t>(RedZoneMark.size()), [](const std::uint8_t value) { return value == 0; })) return false;
    if (unframed.empty()) {
        std::copy(RedZoneMark.begin(), RedZoneMark.end(), mark);
        return true;
    }
    if (!std::all_of(reserved, reserved + 12, [](const std::uint8_t value) { return value == 0; })) return false;
    std::uint32_t tableRva = 0;
    if (!appendRangeTable(image, unframed, tableRva)) return false;
    std::copy(RedZoneRangesMark.begin(), RedZoneRangesMark.end(), image.begin() + static_cast<std::ptrdiff_t>(RedZoneMarkOffset));
    Io::WriteU32(image, RedZoneTableRvaOffset, tableRva);
    Io::WriteU32(image, RedZoneTableCountOffset, static_cast<std::uint32_t>(unframed.size()));
    return true;
}

}
