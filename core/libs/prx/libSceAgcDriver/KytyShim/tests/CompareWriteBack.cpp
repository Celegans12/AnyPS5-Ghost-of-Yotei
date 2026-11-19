#include "graphics/host_gpu/renderer/cache/compareWriteBack.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_compare_write_back: %s\n", what);
    std::abort();
}

constexpr uint32_t BlockSide    = 128;
constexpr uint32_t ElementBytes = 4;
constexpr uint64_t BlockBytes   = uint64_t {BlockSide} * BlockSide * ElementBytes;

uint32_t Spread(uint32_t value) {
    uint32_t result = 0;
    for (uint32_t bit = 0; bit < 16; bit++) result |= ((value >> bit) & 1u) << (2 * bit);
    return result;
}

struct TiledImage {
    uint32_t width          = 0;
    uint32_t height         = 0;
    uint32_t blocks_per_row = 0;
    uint64_t capacity       = 0;

    TiledImage(uint32_t w, uint32_t h, uint32_t tail_blocks): width(w), height(h) {
        blocks_per_row      = (w + BlockSide - 1) / BlockSide;
        const uint32_t rows = (h + BlockSide - 1) / BlockSide;
        capacity            = (uint64_t {blocks_per_row} * rows + tail_blocks) * BlockBytes;
    }
    uint64_t Offset(uint32_t x, uint32_t y) const {
        const uint64_t block = uint64_t {y / BlockSide} * blocks_per_row + x / BlockSide;
        const uint32_t inner = Spread(x % BlockSide) | (Spread(y % BlockSide) << 1);
        return block * BlockBytes + uint64_t {inner} * ElementBytes;
    }
    std::vector<uint8_t> TexelMask() const {
        std::vector<uint8_t> mask(capacity, 0);
        for (uint32_t y = 0; y < height; y++)
            for (uint32_t x = 0; x < width; x++)
                for (uint32_t b = 0; b < ElementBytes; b++) mask[Offset(x, y) + b] = 1;
        return mask;
    }
};

struct Guest {
    uint64_t             base = 0x200000000ull;
    std::vector<uint8_t> bytes;
    uint64_t             writes = 0;
    void Write(uint64_t address, const uint8_t* data, uint64_t size) {
        Require(address >= base && address - base + size <= bytes.size(), "write outside the image range");
        for (uint64_t i = 0; i < size; i++) bytes[address - base + i] = data[i];
        writes++;
    }
};

uint8_t Old(uint64_t i) { return static_cast<uint8_t>(i * 7u + 3u); }
uint8_t Produced(uint64_t i) { return static_cast<uint8_t>(~(i * 7u + 3u)); }
uint8_t Game(uint64_t i) { return static_cast<uint8_t>(Old(i) ^ 0x5au); }

void ProducedTexelsOnlyOverwriteTexels() {
    const TiledImage image(200, 60, 1);
    const auto       mask = image.TexelMask();
    Guest            guest;
    guest.bytes.resize(image.capacity);
    for (uint64_t i = 0; i < image.capacity; i++) guest.bytes[i] = Old(i);

    const std::vector<uint8_t> snapshot = guest.bytes;
    std::vector<uint8_t>       result   = snapshot;
    for (uint64_t i = 0; i < image.capacity; i++)
        if (mask[i] != 0) result[i] = Produced(i);

    std::vector<uint8_t> changed(image.capacity, 0);
    const uint64_t       padding_in_partial_block = image.Offset(150, 100);
    const uint64_t       column_past_width        = image.Offset(230, 10);
    const uint64_t       unused_tail              = image.capacity - BlockBytes + 4096;
    const uint64_t       game_texel               = image.Offset(20, 30);
    Require(mask[padding_in_partial_block] == 0 && mask[column_past_width] == 0 && mask[unused_tail] == 0,
            "the test layout has no padding where it expects some");
    Require(mask[game_texel] != 0, "the test layout misses the texel the game writes");
    for (const uint64_t first: {padding_in_partial_block, column_past_width, unused_tail})
        for (uint64_t i = first; i < first + 256; i++)
            if (mask[i] == 0) changed[i] = 1;
    for (uint64_t i = game_texel; i < game_texel + ElementBytes; i++) changed[i] = 1;
    for (uint64_t i = 0; i < image.capacity; i++)
        if (changed[i] != 0) guest.bytes[i] = Game(i);
    const std::vector<uint8_t> current = guest.bytes;

    const auto counts = Libs::Graphics::CompareWriteBack(
        guest.base, result.data(), snapshot.data(), current.data(), image.capacity,
        [&](uint64_t address, const uint8_t* data, uint64_t size) { guest.Write(address, data, size); });

    uint64_t game_bytes = 0;
    for (uint64_t i = 0; i < image.capacity; i++) {
        if (changed[i] != 0) {
            game_bytes++;
            Require(guest.bytes[i] == Game(i), "the write-back put back a byte the game wrote after the read");
        } else if (mask[i] != 0) {
            Require(guest.bytes[i] == Produced(i), "the write-back lost a texel the GPU produced");
        } else {
            Require(guest.bytes[i] == Old(i), "the write-back changed a byte outside the texels");
        }
    }
    Require(counts.kept == game_bytes, "the write-back did not count every byte it kept for the game");
    Require(counts.written < image.capacity, "the write-back wrote the whole image range");
}

void UnchangedGuestIsWrittenInFewRuns() {
    const TiledImage image(256, 256, 0);
    Guest            guest;
    guest.bytes.resize(image.capacity);
    for (uint64_t i = 0; i < image.capacity; i++) guest.bytes[i] = Old(i);
    const std::vector<uint8_t> snapshot = guest.bytes;
    std::vector<uint8_t>       result(image.capacity);
    for (uint64_t i = 0; i < image.capacity; i++) result[i] = Produced(i);

    const auto counts = Libs::Graphics::CompareWriteBack(
        guest.base, result.data(), snapshot.data(), snapshot.data(), image.capacity,
        [&](uint64_t address, const uint8_t* data, uint64_t size) { guest.Write(address, data, size); });
    Require(guest.bytes == result, "a fully produced image was not written back");
    Require(guest.writes == 1, "a fully produced image was not written back in one run");
    Require(counts.kept == 0 && counts.written == image.capacity, "wrong counts for a fully produced image");
}

void NothingProducedWritesNothing() {
    const TiledImage image(64, 64, 0);
    Guest            guest;
    guest.bytes.resize(image.capacity);
    for (uint64_t i = 0; i < image.capacity; i++) guest.bytes[i] = Old(i);
    const std::vector<uint8_t> snapshot = guest.bytes;
    for (uint64_t i = 0; i < image.capacity; i += 3) guest.bytes[i] = Game(i);
    const std::vector<uint8_t> current = guest.bytes;

    const auto counts = Libs::Graphics::CompareWriteBack(
        guest.base, snapshot.data(), snapshot.data(), current.data(), image.capacity,
        [&](uint64_t address, const uint8_t* data, uint64_t size) { guest.Write(address, data, size); });
    Require(guest.writes == 0 && guest.bytes == current, "an image the GPU left unchanged was written back");
    Require(counts.written == 0, "wrong written count for an unchanged image");
}

}

int main() {
    ProducedTexelsOnlyOverwriteTexels();
    UnchangedGuestIsWrittenInFewRuns();
    NothingProducedWritesNothing();
    std::printf("kyty_shim_compare_write_back: ok\n");
    return 0;
}
