#pragma once

#include "common/liveAb.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// What textureCache.cpp and textureCacheShared.cpp (one guest memory under several images, off by default) both read: the
// gates stay here so that the main path's only cost is the one test it always had.
namespace Libs::Graphics {

// KYTY_FIX_IMAGE_MEMORY_OWNER (D4c step S1; 0 = KYTY_WORKAROUND_ALIAS_SYNC as before, 1 on, 2 on with checks): one guest
// memory under several images, every byte of it with one holder (see ImageMemoryOwners below). With it on,
// KYTY_WORKAROUND_ALIAS_SYNC and KYTY_FIX_SHARED_IMAGE_MEMORY are not used.
inline uint32_t ImageMemoryOwnerMode() {
	static const uint32_t mode = [] {
		const char*    value = std::getenv("KYTY_FIX_IMAGE_MEMORY_OWNER");
		const uint32_t m     = value != nullptr && (value[0] == '1' || value[0] == '2') ? uint32_t(value[0] - '0') : 0u;
		if (m != 0u) std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER %s\n", m == 1u ? "on" : "check");
		return m;
	}();
	return mode;
}

// KYTY_FIX_SHARED_IMAGE_MEMORY (D4 step K1; 0 = KYTY_WORKAROUND_ALIAS_SYNC as before, 1 on, 2 check; live A/B switch
// SharedImageMemory): one guest memory under several images. The game describes the same memory with other extents
// (dynamic resolution: one allocation of the largest size, smaller views of it) and with other formats (transient
// memory reused by later passes of the frame). Each description is its own host image here, so the bytes one of them
// writes are what the others must show over the same addresses, in their own layouts. When an image is bound for
// sampling, storage or rendering and images overlapping its memory were written after it (content serial above
// max(content_serial, alias_synced_serial)), their bytes are brought in, oldest first:
// - every newer image has the image's layout (one level and layer each, level 0 at the same address, same tile mode,
//   bytes per element, pitch, type and colour transform): an image copy of the common rectangle - texel (x, y) is at the
//   same guest address in both;
// - otherwise through guest memory: a scratch buffer takes the image's own bytes (its download when the GPU wrote it and
//   nothing newer reached its memory through buffers, else the cached buffer of guest memory), then each newer image's
//   download in its own layout, and the image is loaded from it (the tiler's conversions both ways).
// Not covered (counted, kept as before): depth and stencil images, multisampled and block-compressed images, video-out
// compression. =2 (sampled, the first 4 syncs of an (image, newest writer) pair then 1 in 64): the guest bytes merged
// as above are kept, the image is downloaded over a copy of them after the sync, and the two are compared once the GPU
// finished (diff must be 0: the image holds exactly the merged guest bytes).
// =3 (probe, measurement only - the picture is wrong where the game reads such bytes): instead of bringing the newer
// bytes in, the image is filled with a pattern (bytes 0xA5; 0xFF, a NaN, in float formats, so blending carries it on),
// and its next binds download it and count the texels still holding the pattern: a sampled or storage bind finding it
// means the game reads bytes another image wrote before writing them itself. =4: the same counts without the fill
// (the baseline of patterns and NaNs the game's own data holds).
inline uint32_t SharedImageMemoryMode() {
	static constinit Common::OnceValue<uint32_t> once;
	return Common::LiveAb::Gate(Common::LiveAb::Switch::SharedImageMemory, once.Get([] {
		const char* value = std::getenv("KYTY_FIX_SHARED_IMAGE_MEMORY");
		uint32_t    m     = value == nullptr ? 0u : (value[0] >= '1' && value[0] <= '4') ? uint32_t(value[0] - '0') : 0u;
		if (m != 0u && ImageMemoryOwnerMode() != 0u) {
			std::fprintf(stderr, "FIX_SHARED_IMAGE_MEMORY ignored (KYTY_FIX_IMAGE_MEMORY_OWNER is on)\n");
			m = 0u;
		}
		static constexpr const char* names[] = {"off", "on", "check", "probe", "probe baseline"};
		if (m != 0) std::fprintf(stderr, "FIX_SHARED_IMAGE_MEMORY %s\n", names[m]);
		return m;
	}));
}

// The images KYTY_FIX_SHARED_IMAGE_MEMORY keeps coherent: single-sample, not block-compressed colour images that are
// neither depth nor stencil planes and carry no video-out compression.
inline bool SharedImagePlain(const Image& image) {
	return !image.info.data.Empty() && !image.info.IsDepth() && !image.depth_id && !image.info.HasStencil() &&
	       image.backing.image != nullptr && image.info.samples == 1u && image.backing.samples == 1u &&
	       !image.info.IsBlock() && image.info.metadata.compression == VideoOutCompression::Uncompressed;
}

// An image in one log line.
std::string DescribeImage(const Image& image);

namespace ImageMemoryOwners {
bool     Followed(const Image& image);      // the images KYTY_FIX_IMAGE_MEMORY_OWNER keeps coherent
bool     KnownAlone(const Image& image);    // known to have no other image over its memory
uint32_t ExcludedIndex(const Image& image); // why it does not keep `image` coherent (Excluded index)
// KYTY_FIX_IMAGE_MEMORY_OWNER: KYTY_WORKAROUND_ALIAS_SYNC copies into images the map does not keep coherent, per reason.
inline std::array<std::atomic<uint64_t>, 12> g_alias_sync_copies {};
} // namespace ImageMemoryOwners

} // namespace Libs::Graphics
