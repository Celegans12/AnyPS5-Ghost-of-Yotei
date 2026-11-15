#include <relinker/pipeline/RedZoneFraming.hpp>
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
        refusesOtherImages();
        completeOnlyWithNothingLeft();
        std::cout << "red-zone mark tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
