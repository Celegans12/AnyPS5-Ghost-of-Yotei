#include <relinker/pipeline/RedZoneFraming.hpp>
#include <elfpatcher/windows/WindowsPeFormat.hpp>
#include <io/BufferUtils.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace Relinker;

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint8_t> peImage() {
    std::vector<std::uint8_t> image(0x400);
    Io::WriteU16(image, 0, 0x5a4d);
    Io::WriteU32(image, 0x3c, 0x80);
    Io::WriteU32(image, 0x80, 0x4550);
    return image;
}

bool marked(const std::vector<std::uint8_t>& image) {
    return image.size() >= RedZoneMarkOffset + RedZoneMark.size() && std::equal(RedZoneMark.begin(), RedZoneMark.end(), image.begin() + static_cast<std::ptrdiff_t>(RedZoneMarkOffset));
}

void marksAPeImageOnce() {
    auto image = peImage();
    const auto before = image;
    require(MarkRedZoneFramed(image), "a PE image was not marked");
    require(marked(image), "the mark is not at its offset");
    for (std::size_t at = 0; at < image.size(); ++at) {
        const bool inMark = at >= RedZoneMarkOffset && at < RedZoneMarkOffset + RedZoneMark.size();
        require(inMark || image[at] == before[at], "marking changed byte " + std::to_string(at) + " outside the mark");
    }
    require(!MarkRedZoneFramed(image), "an image whose mark bytes are in use was marked again");
}

constexpr std::size_t PeOffset = 0x80;
constexpr std::size_t OptionalOffset = PeOffset + 24;
constexpr std::size_t SectionTable = OptionalOffset + 240;
constexpr std::uint32_t HeaderPage = 0x1000;
constexpr std::uint32_t ImageEnd = 0x13000;

std::vector<std::uint8_t> fullPeImage(const std::uint32_t headerBytes = HeaderPage, const std::size_t sections = 2) {
    std::vector<std::uint8_t> image(HeaderPage);
    Io::WriteU16(image, 0, 0x5a4d);
    Io::WriteU32(image, 0x3c, PeOffset);
    Io::WriteU32(image, PeOffset, 0x4550);
    Io::WriteU16(image, PeOffset + 4, 0x8664);
    Io::WriteU16(image, PeOffset + 6, static_cast<std::uint16_t>(sections));
    Io::WriteU16(image, PeOffset + 20, 240);
    Io::WriteU16(image, OptionalOffset, 0x20b);
    Io::WriteU32(image, OptionalOffset + 8, 0x400);
    Io::WriteU32(image, OptionalOffset + 32, 0x1000);
    Io::WriteU32(image, OptionalOffset + 36, 0x200);
    Io::WriteU32(image, OptionalOffset + 56, ImageEnd);
    Io::WriteU32(image, OptionalOffset + 60, headerBytes);
    for (std::size_t index = 0; index < sections; ++index) {
        const auto header = SectionTable + index * 40;
        image[header] = static_cast<std::uint8_t>('a' + index);
        Io::WriteU32(image, header + 8, 0x1000);
        Io::WriteU32(image, header + 12, Elfpatcher::Windows::LoadRva + static_cast<std::uint32_t>(index) * 0x1000);
        Io::WriteU32(image, header + 16, 0x200);
        Io::WriteU32(image, header + 20, static_cast<std::uint32_t>(image.size()));
        image.resize(image.size() + 0x200, static_cast<std::uint8_t>(0x90 + index));
    }
    return image;
}

void marksAPeImageWithTheRangesItLeavesUnframed() {
    auto image = fullPeImage();
    const auto before = image;
    const std::vector<RedZoneRvaRange> ranges{{0x11000, 0x11040}, {0x11100, 0x11110}, {0x12ff0, 0x13000}};
    require(MarkRedZoneFramed(image, ranges), "a PE image with ranges was not marked");
    require(std::equal(RedZoneRangesMark.begin(), RedZoneRangesMark.end(), image.begin() + static_cast<std::ptrdiff_t>(RedZoneMarkOffset)) && !marked(image), "the mark with ranges is not the ranges mark");
    require(Io::ReadU32(image, RedZoneTableRvaOffset) == ImageEnd && Io::ReadU32(image, RedZoneTableCountOffset) == ranges.size(), "the mark does not point at its table");
    require(Io::ReadU16(image, PeOffset + 6) == 3, "the section was not counted");
    require(Io::ReadU32(image, OptionalOffset + 56) == ImageEnd + 0x1000 && Io::ReadU32(image, OptionalOffset + 8) == 0x400 + 0x200, "the image size and the initialized data size do not include the table");
    const auto header = SectionTable + 2 * 40;
    require(std::string(image.begin() + static_cast<std::ptrdiff_t>(header), image.begin() + static_cast<std::ptrdiff_t>(header) + 6) == ".rzgap" && image[header + 6] == 0 && image[header + 7] == 0, "the section is not named .rzgap");
    require(Io::ReadU32(image, header + 8) == ranges.size() * RedZoneTableEntryBytes && Io::ReadU32(image, header + 12) == ImageEnd, "the section does not map the table at the end of the image");
    require(Io::ReadU32(image, header + 16) == 0x200 && Io::ReadU32(image, header + 36) == 0x40000040u, "the section is not read-only initialized data");
    const auto raw = Io::ReadU32(image, header + 20);
    require(raw == before.size() && image.size() == raw + 0x200 && raw % 0x200 == 0, "the table is not appended after the file, file aligned");
    for (std::size_t index = 0; index < ranges.size(); ++index)
        require(Io::ReadU32(image, raw + index * RedZoneTableEntryBytes) == ranges[index].Begin && Io::ReadU32(image, raw + index * RedZoneTableEntryBytes + 4) == ranges[index].End, "a range is not in the table");
    for (std::size_t at = 0; at < before.size(); ++at) {
        const bool changed = (at >= RedZoneMarkOffset && at < RedZoneMarkOffset + 8) || (at >= RedZoneTableRvaOffset && at < RedZoneTableRvaOffset + 8) || (at >= PeOffset + 6 && at < PeOffset + 8) || (at >= OptionalOffset + 8 && at < OptionalOffset + 12) || (at >= OptionalOffset + 56 && at < OptionalOffset + 60) || (at >= header && at < header + 40);
        require(changed || image[at] == before[at], "marking with ranges changed byte " + std::to_string(at) + " outside the mark, the counts and the new section header");
    }
    require(!MarkRedZoneFramed(image, ranges), "an image already marked was marked again");
}

void marksWithoutRangesAsBefore() {
    auto image = fullPeImage();
    const auto before = image;
    require(MarkRedZoneFramed(image, {}) && marked(image) && image.size() == before.size() && Io::ReadU16(image, PeOffset + 6) == 2, "an empty list did not give the plain mark");
}

void refusesRangesItCannotKeep() {
    const std::vector<RedZoneRvaRange> ranges{{0x11000, 0x11040}};
    auto crowded = fullPeImage(SectionTable + 2 * 40 + 39);
    const auto before = crowded;
    require(!MarkRedZoneFramed(crowded, ranges) && crowded == before, "an image without room for a section header was marked");
    auto inUse = fullPeImage();
    inUse[SectionTable + 2 * 40 + 3] = 1;
    const auto inUseBefore = inUse;
    require(!MarkRedZoneFramed(inUse, ranges) && inUse == inUseBefore, "an image whose next section header is in use was marked");
    auto reserved = fullPeImage();
    reserved[RedZoneTableRvaOffset + 2] = 1;
    const auto reservedBefore = reserved;
    require(!MarkRedZoneFramed(reserved, ranges) && reserved == reservedBefore, "an image whose table bytes are in use was marked");
    auto bare = peImage();
    const auto bareBefore = bare;
    require(!MarkRedZoneFramed(bare, ranges) && bare == bareBefore, "an image without an optional header was marked with ranges");
    auto unaligned = fullPeImage();
    Io::WriteU32(unaligned, OptionalOffset + 56, ImageEnd + 8);
    const auto unalignedBefore = unaligned;
    require(!MarkRedZoneFramed(unaligned, ranges) && unaligned == unalignedBefore, "an image whose size is not section aligned was marked");
    auto full = fullPeImage(HeaderPage, 95);
    const auto fullBefore = full;
    require(!MarkRedZoneFramed(full, ranges) && full == fullBefore, "an image with 95 sections was marked");
}

std::vector<std::uint8_t> elfImage() {
    std::vector<std::uint8_t> image(0x2000);
    image[0] = 0x7f;
    image[1] = 'E';
    image[2] = 'L';
    image[3] = 'F';
    image[4] = 2;
    image[5] = 1;
    image[6] = 1;
    Io::WriteU16(image, 16, 3);
    Io::WriteU16(image, 18, 62);
    return image;
}

void mapsUnframedAddressesToRvas() {
    const auto image = elfImage();
    const std::vector<Domain::ProgramHeader> headers{{1, 5, 0, 0x4000, 0x4000, 0x2000, 0x2000, 0x1000}};
    RedZoneFramingResult none;
    const auto empty = RedZoneRvaRanges(image, headers, none);
    require(empty && empty->empty(), "no unframed range gave a list");
    RedZoneFramingResult result;
    result.Unframed = {{0x4100, 0x4140}, {0x4140, 0x4150}, {0x5000, 0x5010}};
    const auto mapped = RedZoneRvaRanges(image, headers, result);
    require(mapped && mapped->size() == 2, "adjacent ranges were not merged");
    require((*mapped)[0].Begin == Elfpatcher::Windows::LoadRva + 0x100 && (*mapped)[0].End == Elfpatcher::Windows::LoadRva + 0x150, "a range was mapped to the wrong RVAs");
    require((*mapped)[1].Begin == Elfpatcher::Windows::LoadRva + 0x1000 && (*mapped)[1].End == Elfpatcher::Windows::LoadRva + 0x1010, "the second range was mapped to the wrong RVAs");
    result.Unframed = {{0x9000, 0x9010}};
    require(!RedZoneRvaRanges(image, headers, result), "a range outside the image was mapped");
}

void refusesOtherImages() {
    std::vector<std::uint8_t> elf(0x400);
    elf[0] = 0x7f;
    elf[1] = 'E';
    elf[2] = 'L';
    elf[3] = 'F';
    require(!MarkRedZoneFramed(elf) && !marked(elf), "an ELF image was marked");
    std::vector<std::uint8_t> shortImage(0x30);
    shortImage[0] = 'M';
    shortImage[1] = 'Z';
    require(!MarkRedZoneFramed(shortImage) && !marked(shortImage), "an image shorter than a DOS header was marked");
    auto inside = peImage();
    Io::WriteU32(inside, 0x3c, 0x20);
    require(!MarkRedZoneFramed(inside) && !marked(inside), "an image whose PE header overlaps the DOS header was marked");
    std::vector<std::uint8_t> empty;
    require(!MarkRedZoneFramed(empty), "an empty image was marked");
}

void completeOnlyWithNothingLeft() {
    RedZoneFramingResult result;
    require(RedZoneFramingComplete(result), "a framing with nothing left is not complete");
    result.Skipped = 1;
    require(!RedZoneFramingComplete(result), "a framing with a skipped leaf is complete");
    result.Skipped = 0;
    result.Unexamined = 1;
    require(!RedZoneFramingComplete(result), "a framing with an unexamined range below rsp is complete");
    result.Unexamined = 0;
    result.HasBounds = false;
    result.SplitFunctions = 3;
    require(RedZoneFramingComplete(result), "a framing of split functions with nothing left is not complete");
}

}

int main() {
    try {
        marksAPeImageOnce();
        marksAPeImageWithTheRangesItLeavesUnframed();
        marksWithoutRangesAsBefore();
        refusesRangesItCannotKeep();
        mapsUnframedAddressesToRvas();
        refusesOtherImages();
        completeOnlyWithNothingLeft();
        std::cout << "red-zone mark tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
