#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using Libs::Graphics::ImageInfo;
using Libs::Graphics::Prospero::BufferFormat;
using Libs::Graphics::Prospero::ImageType;
using Libs::Graphics::Prospero::TileMode;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_tiled_layout: FAILED: %s\n", what);
    std::exit(1);
}

ImageInfo Texture(uint32_t format, uint32_t tile, uint32_t width, uint32_t height, uint32_t levels) {
    ImageInfo info;
    info.guest_format     = static_cast<BufferFormat>(format);
    info.tile_mode        = static_cast<TileMode>(tile);
    info.extent           = vk::Extent3D {width, height, 1};
    info.resources.levels = levels;
    return info;
}

ImageInfo Recycled() { return Texture(76, 24, 1009, 1, 2); }

void UnknownTileModes() {
    Require(!Texture(77, 19, 1009, 1, 16).HasTiledLayout(), "tile mode 19 has no layout");
    Require(!Texture(75, 22, 1009, 1, 5).HasTiledLayout(), "tile mode 22 has no layout");
    Require(!Texture(15, 15, 1, 1, 1).HasTiledLayout(), "tile mode 15 has no layout");
    Require(!Texture(15, 23, 25, 1, 9).HasTiledLayout(), "tile mode 23 has no layout");
}

void KnownTileModesWithoutBlocks() {
    Require(!Recycled().HasTiledLayout(), "the depth tile mode has no block for a 16-byte element");
    auto compressed = Texture(static_cast<uint32_t>(BufferFormat::kBc1UNorm), 24, 64, 64, 1);
    Require(!compressed.HasTiledLayout(), "the depth tile mode has no block for a compressed format");
    auto volume = Texture(static_cast<uint32_t>(BufferFormat::k8_8_8_8UNorm), 1, 16, 16, 1);
    volume.type = ImageType::kColor3D;
    Require(!volume.HasTiledLayout(), "the 256-byte tile mode has no block for a volume");
    volume.type = ImageType::kColor2D;
    Require(volume.HasTiledLayout(), "the 256-byte tile mode has a block for a 2D texture");
}

void FormatsWithoutElements() {
    Require(!Texture(static_cast<uint32_t>(BufferFormat::k32_32_32Float), 0, 64, 64, 1).HasElementLayout(), "a 12-byte element has no layout");
    Require(!Texture(static_cast<uint32_t>(BufferFormat::k32_32_32UInt), 21, 1009, 1, 14).HasElementLayout(), "a 12-byte element has no layout in an unknown tile mode");
    Require(Texture(static_cast<uint32_t>(BufferFormat::k32_32_32UInt), 21, 1009, 1, 14).HasTiledLayout(), "a 12-byte element is not declined for its tile mode");
    Require(!Texture(static_cast<uint32_t>(BufferFormat::k8_8_8_8UScaled), 9, 64, 64, 1).HasElementLayout(), "a scaled vertex format has no element layout");
    Require(Texture(static_cast<uint32_t>(BufferFormat::k32_32_32_32Float), 9, 64, 64, 1).HasElementLayout(), "a 16-byte element has a layout");
    Require(Texture(static_cast<uint32_t>(BufferFormat::kBc7UNorm), 9, 64, 64, 1).HasElementLayout(), "a compressed block has an element layout");
}

void LaidOut(ImageInfo info, const char* what) {
    Require(info.HasTiledLayout(), what);
    info.UpdateSize();
    Require(info.data.size != 0, what);
}

void TexturesWithLayouts() {
    const auto rgba = static_cast<uint32_t>(BufferFormat::k8_8_8_8UNorm);
    LaidOut(Texture(rgba, 0, 256, 256, 9), "a linear texture is laid out");
    LaidOut(Texture(rgba, 1, 16, 16, 1), "a 256-byte tiled texture is laid out");
    LaidOut(Texture(rgba, 5, 256, 256, 9), "a 4 KB tiled texture is laid out");
    LaidOut(Texture(rgba, 9, 256, 256, 9), "a 64 KB tiled texture is laid out");
    LaidOut(Texture(rgba, 17, 256, 256, 9), "a PRT texture is laid out");
    LaidOut(Texture(rgba, 24, 256, 256, 1), "a depth tiled texture of 4-byte elements is laid out");
    LaidOut(Texture(rgba, 27, 256, 256, 1), "a render target tiled texture is laid out");
    LaidOut(Texture(76, 9, 1009, 1, 2), "a 16-byte element is laid out in 64 KB tiles");
    LaidOut(Texture(static_cast<uint32_t>(BufferFormat::kBc1UNorm), 9, 256, 256, 9),
            "a compressed 64 KB tiled texture is laid out");
    auto volume = Texture(rgba, 9, 64, 64, 4);
    volume.type         = ImageType::kColor3D;
    volume.extent.depth = 16;
    LaidOut(volume, "a 64 KB tiled volume is laid out");
}

void OtherStopsStay() {
    auto multisampled = Texture(77, 19, 64, 64, 1);
    multisampled.samples = 4;
    Require(multisampled.HasTiledLayout(), "a multisampled texture is not declined for its tile mode");
    Require(Texture(0, 19, 64, 64, 1).HasTiledLayout(), "a format without an element size is not declined for its tile mode");
    auto depth = Texture(static_cast<uint32_t>(BufferFormat::k16UNorm), 24, 64, 64, 1);
    depth.pixel_format = vk::Format::eD16Unorm;
    Require(depth.HasTiledLayout(), "a single-level depth plane is not laid out as a tiled texture");
}

}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "stop") == 0) {
        auto info = Recycled();
        info.UpdateSize();
        std::fprintf(stderr, "kyty_shim_tiled_layout: the size update did not stop\n");
        return 0;
    }
    UnknownTileModes();
    KnownTileModesWithoutBlocks();
    FormatsWithoutElements();
    TexturesWithLayouts();
    OtherStopsStay();
    std::puts("kyty_shim_tiled_layout: ok");
    return 0;
}
