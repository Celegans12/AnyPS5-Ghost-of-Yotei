#include "graphics/host_gpu/renderer/cache/textureCacheShared.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/envDefaults.h"
#include "common/liveAb.h"
#include "common/logging/log.h"
#include "common/lookupMemo.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>

// One guest memory under several images: KYTY_FIX_IMAGE_MEMORY_OWNER, KYTY_FIX_SHARED_IMAGE_MEMORY (+_DIRECT), both off by
// default and the proper implementations KYTY_WORKAROUND_ALIAS_SYNC waits for (D4). Nothing
// here runs while their gates (textureCacheShared.h, asked in textureCache.cpp) are off.
namespace Libs::Graphics {

// KYTY_FIX_SHARED_IMAGE_DIRECT (D4, the fast form of the merge through guest memory below; default 0, 1 on; checked by
// KYTY_FIX_SHARED_IMAGE_MEMORY=2, whose reference is that merge): when the image and every newer image are tiled, 2D,
// with one level and one layer (TileManager::AliasCopyFits) and the image's host contents are current (no newer guest
// bytes through buffers or the CPU), the image takes the newer images' bytes in place: its linear host copy is patched,
// newer image after newer image, by gpu_tiler_alias_copy (each texel's guest address looked up in the newer image's
// layout) and loaded back - no guest-layout download of the image itself and no detiling. The same guest addresses and
// the same order of writers as the merge, so the same bytes.
static bool SharedImageDirectEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FIX_SHARED_IMAGE_DIRECT");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "FIX_SHARED_IMAGE_DIRECT on\n");
		return set;
	}();
	return on;
}

// An image in one log line.
std::string DescribeImage(const Image& image) {
	return fmt::format("0x{:x} {}x{} L{} x{} {} tile={} pitch={} meta={}{}{}", image.info.data.address,
	                   image.info.extent.width, image.info.extent.height, image.info.resources.levels,
	                   image.info.resources.layers, vk::to_string(image.info.pixel_format),
	                   static_cast<uint32_t>(image.info.tile_mode), image.info.mip_layout[0].pitch,
	                   static_cast<uint32_t>(image.info.metadata.kind), image.info.IsDepth() ? " depth" : "",
	                   image.depth_id ? " stencil-assoc" : "");
}

// The guest bytes images sharing memory wrote: per 64 KiB granule, byte segments with the image holding their newest value
// (or guest memory) and the images whose host texels hold it too (KYTY_FIX_IMAGE_MEMORY_OWNER below).
namespace ImageMemorySegments {
constexpr uint64_t kGranule = uint64_t {1} << 16u;
enum class Excluded : uint32_t { None, Depth, Msaa, Block, VideoOut, Linear, Levels, Layers, Volume, Tail, Misaligned, Count };
using Key    = uint64_t; // generation << 32 | index (generations start at 1, so 0 is free for guest memory)
using Range  = std::pair<uint64_t, uint64_t>;
using Ranges = std::vector<Range>;
static Key KeyOf(ImageId id) {
	return (uint64_t {id.generation} << 32u) | id.index;
}
struct Segment {
	uint64_t         lo     = 0;
	uint64_t         hi     = 0;
	Key              holder = 0;     // the image holding the newest bytes; 0: guest memory
	bool             all    = false; // no image wrote them since the census took them (a gap being filled)
	std::vector<Key> current;        // the images whose host texels hold these bytes (sorted; the holder among them)
};
using GranuleMap = std::map<uint64_t, std::vector<Segment>>; // by granule base address
static bool Holds(const std::vector<Key>& keys, Key key) {
	return std::ranges::binary_search(keys, key);
}
static void AddKey(std::vector<Key>& keys, Key key) {
	const auto it = std::ranges::lower_bound(keys, key);
	if (it == keys.end() || *it != key) keys.insert(it, key);
}
static void RemoveKey(std::vector<Key>& keys, Key key) {
	const auto it = std::ranges::lower_bound(keys, key);
	if (it != keys.end() && *it == key) keys.erase(it);
}
static uint64_t Bytes(const Ranges& ranges) {
	uint64_t bytes = 0;
	for (const auto& [lo, hi]: ranges) bytes += hi - lo;
	return bytes;
}
// Every range of `inner` lies inside one range of `outer` (both sorted and disjoint).
static bool Covered(const Ranges& outer, const Ranges& inner) {
	for (const auto& [lo, hi]: inner) {
		auto it = std::ranges::upper_bound(outer, lo, {}, &Range::first);
		if (it == outer.begin()) return false;
		--it;
		if (hi > it->second) return false;
	}
	return true;
}
static void Merge(Ranges& into, const Ranges& add) {
	into.insert(into.end(), add.begin(), add.end());
	std::ranges::sort(into);
	size_t out = 0;
	for (size_t i = 0; i < into.size(); i++) {
		if (out != 0 && into[out - 1].second >= into[i].first) {
			into[out - 1].second = std::max(into[out - 1].second, into[i].second);
		} else {
			into[out++] = into[i];
		}
	}
	into.resize(out);
}
// One granule's segments split at lo and hi; with `create`, the parts of [lo, hi) no segment covers become segments
// (`all`: every image holds them).
static void SplitGranule(std::vector<Segment>& segs, uint64_t lo, uint64_t hi, bool create) {
	for (size_t i = 0; i < segs.size(); i++) {
		for (const uint64_t at: {lo, hi}) {
			if (segs[i].lo < at && at < segs[i].hi) {
				Segment tail = segs[i];
				tail.lo      = at;
				segs[i].hi   = at;
				segs.insert(segs.begin() + static_cast<std::ptrdiff_t>(i) + 1, std::move(tail));
			}
		}
	}
	if (!create) return;
	std::vector<Segment> gaps;
	uint64_t             cursor = lo;
	for (const auto& seg: segs) {
		if (seg.hi <= cursor || seg.lo >= hi) continue;
		if (seg.lo > cursor) gaps.push_back(Segment {cursor, seg.lo, 0, true, {}});
		cursor = std::max(cursor, seg.hi);
	}
	if (cursor < hi) gaps.push_back(Segment {cursor, hi, 0, true, {}});
	if (gaps.empty()) return;
	for (auto& gap: gaps) segs.push_back(std::move(gap));
	std::ranges::sort(segs, {}, &Segment::lo);
}
// Neighbours in the same state joined; `all` segments dropped (the same as no segment).
static void Coalesce(std::vector<Segment>& segs) {
	size_t out = 0;
	for (size_t i = 0; i < segs.size(); i++) {
		if (segs[i].all) continue;
		if (out != 0) {
			auto& last = segs[out - 1];
			if (last.hi == segs[i].lo && last.holder == segs[i].holder && last.current == segs[i].current) {
				last.hi = segs[i].hi;
				continue;
			}
		}
		if (out != i) segs[out] = std::move(segs[i]);
		out++;
	}
	segs.resize(out);
}
// `func` on every segment of [lo, hi); without `create` only the granules and bytes images wrote.
template <typename Func>
static void ForSegmentsIn(GranuleMap& granules, uint64_t lo, uint64_t hi, bool create, Func&& func) {
	if (hi <= lo) return;
	const auto visit = [&](GranuleMap::iterator it) {
		const uint64_t glo  = std::max(lo, it->first);
		const uint64_t ghi  = std::min(hi, it->first + kGranule);
		auto&          segs = it->second;
		SplitGranule(segs, glo, ghi, create);
		for (auto& seg: segs) {
			if (seg.lo >= glo && seg.hi <= ghi) func(seg);
		}
		Coalesce(segs);
		return segs.empty() ? granules.erase(it) : std::next(it);
	};
	if (create) {
		for (uint64_t g = lo & ~(kGranule - 1u); g < hi; g += kGranule) {
			visit(granules.try_emplace(g).first);
		}
		return;
	}
	for (auto it = granules.lower_bound(lo & ~(kGranule - 1u)); it != granules.end() && it->first < hi;) {
		it = visit(it);
	}
}
static Excluded Classify(const Image& image) {
	const auto& info = image.info;
	if (info.IsDepth() || image.depth_id || info.HasStencil()) return Excluded::Depth;
	if (info.samples != 1u || image.backing.samples != 1u) return Excluded::Msaa;
	if (info.IsBlock()) return Excluded::Block;
	if (info.metadata.compression != VideoOutCompression::Uncompressed) return Excluded::VideoOut;
	if (!info.IsTiled()) return Excluded::Linear;
	if (info.resources.levels != 1u) return Excluded::Levels;
	if (info.resources.layers != 1u) return Excluded::Layers;
	if (info.IsVolume()) return Excluded::Volume;
	if (info.first_tail_level == 0u) return Excluded::Tail;
	const auto& block = info.tiling.block;
	if (block.block_size == 0u || block.block_size > kGranule || block.block_width == 0u || block.block_height == 0u ||
	    (info.data.address + info.mip_layout[0].offset) % block.block_size != 0u) {
		return Excluded::Misaligned;
	}
	return Excluded::None;
}
} // namespace ImageMemorySegments

// D4c step S0: the facts of the written buffer binding whose InvalidateMemoryFromGPU runs next on this thread.
static thread_local const TextureCache::OwnerCensusBinding* t_owner_census_binding = nullptr;

void TextureCache::OwnerCensusSetBinding(const OwnerCensusBinding* binding) {
	t_owner_census_binding = binding;
}

// KYTY_FIX_IMAGE_MEMORY_OWNER (D4c step S1; 0 = KYTY_WORKAROUND_ALIAS_SYNC as before, 1 on, 2 on with checks): one guest
// memory under several images. The game describes the same memory with other extents and formats (dynamic resolution
// views of one allocation, transient memory reused by later passes); each description is its own host image here, so
// the bytes one of them writes are what the others must show over the same addresses, in their own layouts. A map of
// the guest bytes images wrote (per 64 KiB granule, byte segments) names for every such byte its holder - the image
// whose host texels hold its newest value, or guest memory - and the images whose texels hold it too:
// - writes: a render-target acquisition writes the blocks of level 0 its draw's rectangle touches (the union over the
//   draw's viewports of scissor and viewport bounds; the whole target when no rectangle is known), a storage binding
//   the whole image, a clear the whole image; an image copy gives the destination what the source holds;
// - before a write the writer takes the bytes of its write range it does not hold (pull), so its texels there that the
//   pass leaves alone hold memory's bytes too, and it becomes their only holder; a sampled binding takes every byte of
//   its memory another image or guest memory holds;
// - a pull moves bytes exactly through guest layout: the target's and the holder's blocks over the range are tiled into
//   guest-layout buffers (TileManager::TileRect), the range's bytes are copied from the holder's buffer into the
//   target's, and the target's blocks are detiled and loaded back - only blocks the range touches move;
// - a GPU buffer write over image-held bytes (InvalidateMemoryFromGPU, at the binding before the shader runs) first has
//   them written into guest memory's GPU copy (OwnerBufferWrite: within the binding's fixed write ranges when known;
//   none for fills and copies, which write their whole range), then leaves guest memory their holder (bytes it could not
//   write back: buffer_lost); an image loaded from guest memory holds guest memory's bytes (its own held bytes there are
//   lost - counted: refresh_lost); an unregistered image's bytes pass to an image that holds them too, else are written
//   into guest memory's GPU copy (unregister_written; lost when none holds the range: unregister_held); a CPU readback
//   of an image takes the bytes other holders hold from them (OwnerFixReadback);
// - images whose bytes another holder took are marked (Image::owner_stale, HeapChanged), so every memo that proves "no
//   work" for a binding (texture replay, table sets, the lazy commit wait, continued targets) takes the full path.
// Images it keeps coherent: single-sample, uncompressed, one level and layer, not a volume, texels of the host format's
// size, and either tiled (level 0 outside the mip tail and block-aligned, 1x1 texel elements; moved through the tiler)
// or guest-linear without a colour transform (rows moved by plain image/buffer copies with the guest pitch as row
// length; windows aligned to 256 bytes). Other images keep KYTY_WORKAROUND_ALIAS_SYNC's
// copy as before (AliasSyncSkipsImage skips only the images the map covers; copies counted per reason:
// alias_sync_excluded) and are counted when bytes images wrote reach their memory (excluded_stale).
// =2, all must stay 0:
// (ii) pull_diff - sampled (the first 4 pulls per target, then 1 in 64): after a pull the target is tiled again over
//      the window and compared with the bytes it was given;
// (i) escape - sampled draw runs (the first 4 per image, then 1 in 64): the image's guest bytes at the start of its
//      write interval and at its end (the next event that reads or replaces its bytes); a changed byte outside the
//      run's cover is an escape (the cover was too small);
// (iii) oracle - in a frame window over one range (OracleWindow): a reference memory built only from bytes that write
//      intervals changed (independent of holders and covers; bytes a buffer write may change become unknown), compared
//      at sampled binds with the image's guest bytes where it knows them;
// (iv) informational, at the same binds: k1_wrong - bytes where the whole-image merge (the image with the newest content
//      serial over each byte) differs from the reference.
// One "PERF image_memory_owner_fix" and one "PERF image_memory_owner_check" line about every 5 s.
namespace ImageMemoryOwners {
using ImageMemorySegments::AddKey;
using ImageMemorySegments::Bytes;
using ImageMemorySegments::Classify;
using ImageMemorySegments::Covered;
using ImageMemorySegments::Excluded;
using ImageMemorySegments::ForSegmentsIn;
using ImageMemorySegments::GranuleMap;
using ImageMemorySegments::Holds;
using ImageMemorySegments::Key;
using ImageMemorySegments::KeyOf;
using ImageMemorySegments::Merge;
using ImageMemorySegments::Range;
using ImageMemorySegments::Ranges;
using ImageMemorySegments::RemoveKey;
using ImageMemorySegments::Segment;
enum class Why : uint32_t { Rt, Storage, Read, Count };
struct Run {
	Ranges prepared; // the render-target cover of its current draw run (sorted, disjoint; the run: Image::owner_disturbed)
};
struct Stats {
	std::array<uint64_t, static_cast<size_t>(Why::Count)> pulls {}, pull_bytes {};
	uint64_t guest_bytes = 0, transfers = 0, moved_bytes = 0, writes_rt = 0, writes_storage = 0, writes_whole = 0;
	uint64_t writes_copy = 0, continue_refused = 0, excluded_stale = 0, unregister_held = 0, buffer_lost = 0;
	uint64_t refresh_lost = 0, holder_gone = 0, adopted = 0;
	// Buffer-path writes and CPU readback: image-held bytes written into guest memory's GPU copy before a buffer write
	// (and of them those under a write-only formatted binding, which a proof of a whole-range store would spare), bytes of
	// fills and copies (their whole range is written: nothing to write back), held bytes of unregistered images written
	// back, readback bytes taken from their holder instead of the downloaded image (and those that could not be).
	uint64_t writeback = 0, writeback_fill_like = 0, full_writes = 0, unregister_written = 0, readback_fixed = 0;
	uint64_t readback_unfixed = 0;
	// Cost: CPU time in the map's entry points (outermost calls) and in recording pulls, render passes a pull or a
	// write-back ended, bytes and copy regions moved image to image directly (same block arrangement).
	// (CPU by entry point: syncs at bindings, buffer writes, the rest; syncs a memo answered without work.)
	uint64_t cpu_sync_ns = 0, cpu_buffer_ns = 0, cpu_other_ns = 0, memo_hits = 0, pull_cpu_ns = 0, splits = 0;
	uint64_t direct_bytes = 0, direct_regions = 0, direct_checks = 0, buffer_quick = 0;
	uint64_t retile_bytes = 0, retile_jobs = 0; // (moved by TileManager::Retile)
};
// =2: one image's write interval - from a write's start (after its pulls) to the next event that reads or replaces its
// bytes (another image's write over its memory or pull from it, its own next write that is not the same draw run, a
// clear): its guest bytes at the start, and the bytes it may change (its run's cover; a storage binding: its memory).
struct Interval {
	ImageId                 id;
	uint64_t                lo = 0, hi = 0;
	std::shared_ptr<Buffer> before;
	Ranges                  allowed;
	bool                    escape = false; // check (i): changed bytes outside `allowed`
	bool                    oracle = false; // check (iii): changed bytes go to the reference memory
};
// =2, check (iii): reference guest memory of one range in frame windows, built only from bytes that write intervals
// changed (whatever the holders say), compared with images at their sampled binds. Windows: 3 frames every `period`
// frames from frame `period` on (the reference starts empty in each), or one window of given frames.
struct Oracle {
	std::mutex           mutex;
	uint64_t             lo = 0, hi = 0; // [lo, hi): the range (0, 0: not chosen yet)
	uint32_t             period = 300;   // 0: the single window [first_frame, last_frame]
	uint32_t             first_frame = 0, last_frame = 0;
	uint32_t             window_index = UINT32_MAX;
	std::vector<uint8_t> bytes, valid;
};
struct State {
	GranuleMap                     granules; // under the texture cache's lock (exclusive)
	std::unordered_map<Key, Run>   runs;
	std::unordered_map<Key, uint32_t> checks_done; // =2: pulls per target (sampling)
	std::unordered_map<Key, uint32_t> escape_done; // =2: write intervals per image (sampling)
	std::unordered_map<Key, Interval> intervals;
	Stats                          s;
	std::atomic<uint64_t>          pull_checks {0}, pull_diff_events {0}, pull_diff_bytes {0};
	std::atomic<uint64_t>          escape_checks {0}, escape_events {0}, escape_bytes {0};
	std::atomic<uint64_t>          oracle_checks {0}, oracle_events {0}, oracle_bytes {0}, oracle_valid {0};
	std::atomic<uint64_t>          k1_checks {0}, k1_wrong_bytes {0};
	Oracle                         oracle;
	// Check (iii)'s host-readable snapshots per frame (at most 3 GiB, else the reference stops: it would miss changes)
	// and its sampled binds per frame (at most 4; per image the first 2, then 1 in 16).
	uint32_t                          oracle_frame = 0, oracle_frame_reads = 0;
	uint64_t                          oracle_frame_bytes = 0;
	bool                              oracle_stopped = false;
	std::unordered_map<Key, uint32_t> oracle_done;
	uint32_t                       window_frame = 0;
	std::chrono::steady_clock::time_point window_time {};
};
static State& Get() {
	static State state;
	return state;
}
struct DrawRect {
	bool    valid = false;
	int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};
static thread_local DrawRect t_draw_rect;
static ImageId IdOf(Key key) {
	return ImageId {static_cast<uint32_t>(key), static_cast<uint32_t>(key >> 32u)};
}
// Guest-linear images the map keeps coherent: rows of the image's pitch from level 0's offset, one level and layer, 2D,
// single-sample, uncompressed, no colour transform, texels of the host format's size (a row copy moves guest bytes).
static bool LinearFollowed(const Image& image) {
	const auto& info = image.info;
	return !info.IsTiled() && !info.data.Empty() && !info.IsDepth() && !image.depth_id && !info.HasStencil() &&
	       info.samples == 1u && image.backing.samples == 1u && !info.IsBlock() &&
	       info.metadata.compression == VideoOutCompression::Uncompressed && info.resources.levels == 1u &&
	       info.resources.layers == 1u && !info.IsVolume() && info.GetColorTransform() == ColorTransform::None &&
	       std::has_single_bit(info.bytes_per_block) && info.bytes_per_block <= 16u &&
	       vk::blockSize(image.backing.format) == info.bytes_per_block &&
	       info.mip_layout[0].linear_pitch >= info.extent.width && image.backing.image != nullptr && image.registered;
}
// The images the map keeps coherent (see above).
bool Followed(const Image& image) {
	if (!image.info.IsTiled()) return LinearFollowed(image);
	return Classify(image) == Excluded::None && image.backing.image != nullptr && image.registered &&
	       image.info.tiling.texel_width == 1u && image.info.tiling.texel_height == 1u &&
	       vk::blockSize(image.backing.format) == image.info.tiling.block.bytes_per_element;
}
// Proven alone in its memory: its last sync found no other image there and the image registry is unchanged since
// (every registration or unregistration bumps it). Not proven: the caller takes the full path.
bool KnownAlone(const Image& image) {
	return image.owner_alone_registry != 0u &&
	       image.owner_alone_registry == DrawRepeat::g_image_registry_epoch.load(std::memory_order_relaxed);
}
// The unit a pull's window is aligned to: a tiled image's block, 256 bytes of a linear one.
static uint64_t WindowAlignment(const ImageInfo& info) {
	return info.IsTiled() ? info.tiling.block.block_size : 256u;
}
// CPU time of the outermost entry point on this thread (nested entry points add nothing of their own).
static thread_local uint32_t t_cpu_depth = 0;
struct CpuTimer {
	explicit CpuTimer(uint64_t& total_ns)
	    : total(total_ns), outer(t_cpu_depth++ == 0u),
	      start(outer ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {}) {}
	~CpuTimer() {
		t_cpu_depth--;
		if (outer) {
			total += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
		}
	}
	CpuTimer(const CpuTimer&)            = delete;
	CpuTimer& operator=(const CpuTimer&) = delete;
	uint64_t&                             total;
	bool                                  outer;
	std::chrono::steady_clock::time_point start;
};
// KYTY_TRACE_GPU_TIME labels of the map's GPU work (emulator-internal markers, component 8).
constexpr uint64_t kLabelPull       = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c1u; // pull through guest layout
constexpr uint64_t kLabelWriteBack  = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c2u; // before a buffer write
constexpr uint64_t kLabelUnregister = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c3u; // an unregistered holder
constexpr uint64_t kLabelReadback   = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c4u; // CPU readback fix
constexpr uint64_t kLabelCheck      = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c5u; // =2 snapshots
constexpr uint64_t kLabelDirect     = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c6u; // pull image to image
constexpr uint64_t kLabelRetile     = 0xFEED000000000000ull | (8ull << 16u) | 0xd4c7u; // pull in one retile pass
// Two tiled images whose blocks are arranged alike: the same block family, element size and block dimensions, and the
// same colour transform (their host texel bits stand for guest bytes the same way). A block of guest memory is then
// the same texel rectangle shape in both, its bytes placed by the in-block layout XOR a value of the block's position
// (TileGetBlockXor): where that value is equal, a texel copy of the block moves exactly its guest bytes.
static bool DirectPair(const Image& target, const Image& source) {
	const auto& a = target.info.tiling.block;
	const auto& b = source.info.tiling.block;
	return target.info.IsTiled() && source.info.IsTiled() && a.family == b.family &&
	       a.bytes_per_element == b.bytes_per_element && a.block_size == b.block_size && a.block_width == b.block_width &&
	       a.block_height == b.block_height && a.block_depth == b.block_depth &&
	       target.info.GetColorTransform() == source.info.GetColorTransform();
}
// The level-0 block at guest address `address` (block-aligned, inside level 0) and its position's XOR value.
static bool BlockAt(const ImageInfo& info, uint64_t address, uint32_t& bx, uint32_t& by, uint32_t& xor_value) {
	const auto&    block   = info.tiling.block;
	const auto&    mip     = info.mip_layout[0];
	const uint64_t base    = info.data.address + mip.offset;
	const uint64_t columns = (uint64_t {mip.pitch} + block.block_width - 1u) / block.block_width;
	if (address < base || columns == 0u) return false;
	const uint64_t index = (address - base) / block.block_size;
	bx                   = static_cast<uint32_t>(index % columns);
	by                   = static_cast<uint32_t>(index / columns);
	return TileGetBlockXor(block, bx, by, xor_value);
}
// A linear image's rows over the bytes [lo, hi), as texel rectangles clipped to the extent (full rows joined);
// `func(x0, y0, width, height)`.
template <typename Func>
static void ForLinearRects(const ImageInfo& info, uint64_t lo, uint64_t hi, Func&& func) {
	const auto&    mip  = info.mip_layout[0];
	const uint64_t e    = info.bytes_per_block;
	const uint64_t row  = uint64_t {mip.linear_pitch} * e;
	const uint64_t base = info.data.address + mip.linear_offset;
	lo                  = std::max(lo, base);
	hi                  = std::min({hi, info.data.End(), base + row * info.extent.height});
	if (hi <= lo || row == 0u) return;
	const uint64_t first = (lo - base) / row;
	const uint64_t last  = (hi - 1u - base) / row;
	struct Rect {
		uint32_t x0, y0, w, h;
	};
	std::optional<Rect> pending;
	for (uint64_t y = first; y <= last; y++) {
		const uint64_t xs = y == first ? ((lo - base) % row) / e : 0u;
		const uint64_t xe = std::min<uint64_t>(y == last ? ((hi - base) - y * row + e - 1u) / e : mip.linear_pitch,
		                                       info.extent.width);
		if (xe <= xs) continue;
		const Rect rect {static_cast<uint32_t>(xs), static_cast<uint32_t>(y), static_cast<uint32_t>(xe - xs), 1u};
		if (pending && pending->x0 == rect.x0 && pending->w == rect.w && pending->y0 + pending->h == rect.y0) {
			pending->h++;
			continue;
		}
		if (pending) func(pending->x0, pending->y0, pending->w, pending->h);
		pending = rect;
	}
	if (pending) func(pending->x0, pending->y0, pending->w, pending->h);
}
// Why the map does not keep `image` coherent: the census's reason, 11 for the other conditions of Followed.
uint32_t ExcludedIndex(const Image& image) {
	const auto why = Classify(image);
	return why == Excluded::None ? 11u : static_cast<uint32_t>(why);
}
static bool InRanges(const Ranges& ranges, uint64_t address) {
	auto it = std::ranges::upper_bound(ranges, address, {}, &Range::first);
	return it != ranges.begin() && address < std::prev(it)->second;
}
// =2, check (iii): the frame windows and the range (KYTY_FIX_IMAGE_MEMORY_OWNER_ORACLE=<hex address>[/<hex size>];
// KYTY_FIX_IMAGE_MEMORY_OWNER_ORACLE_FRAMES=<first>-<last> for one window instead of 3 frames every 300 from frame 300,
// which every run reaches; without an address the first image bound in a window with at least 3 others over its memory
// chooses it: their memory together, at most 256 MiB). Entering a new periodic window empties the reference: it knows
// only what writes in its window changed.
static bool OracleWindow(State& st) {
	static const bool parsed = [&] {
		if (const char* frames = std::getenv("KYTY_FIX_IMAGE_MEMORY_OWNER_ORACLE_FRAMES"); frames != nullptr) {
			unsigned a = 0, b = 0;
			if (std::sscanf(frames, "%u-%u", &a, &b) == 2 && a <= b) {
				st.oracle.period      = 0;
				st.oracle.first_frame = a;
				st.oracle.last_frame  = b;
			}
		}
		if (const char* value = std::getenv("KYTY_FIX_IMAGE_MEMORY_OWNER_ORACLE"); value != nullptr) {
			unsigned long long address = 0, size = 0x4000000ull;
			if (std::sscanf(value, "%llx/%llx", &address, &size) >= 1 && address != 0u && size != 0u) {
				std::scoped_lock lock(st.oracle.mutex);
				st.oracle.lo = address;
				st.oracle.hi = address + std::min<unsigned long long>(size, 0x10000000ull);
				st.oracle.bytes.assign(st.oracle.hi - st.oracle.lo, 0);
				st.oracle.valid.assign(st.oracle.hi - st.oracle.lo, 0);
			}
		}
		std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER check windows=%s%u-%u oracle=0x%llx+0x%llx\n",
		             st.oracle.period != 0u ? "every 300 frames from " : "", st.oracle.period != 0u ? 300u : st.oracle.first_frame,
		             st.oracle.period != 0u ? 302u : st.oracle.last_frame, static_cast<unsigned long long>(st.oracle.lo),
		             static_cast<unsigned long long>(st.oracle.hi - st.oracle.lo));
		return true;
	}();
	(void)parsed;
	if (ImageMemoryOwnerMode() != 2u || st.oracle_stopped) return false;
	const uint32_t frame = DebugPresentCount().load();
	if (st.oracle.period == 0u) return frame >= st.oracle.first_frame && frame <= st.oracle.last_frame;
	if (frame < st.oracle.period || frame % st.oracle.period >= 3u) return false;
	if (const uint32_t index = frame / st.oracle.period; index != st.oracle.window_index) {
		// A new window: the reference starts empty (updates still pending from before only add what they changed).
		std::scoped_lock lock(st.oracle.mutex);
		st.oracle.window_index = index;
		std::fill(st.oracle.valid.begin(), st.oracle.valid.end(), uint8_t {0});
		st.oracle_valid.store(0, std::memory_order_relaxed);
		st.oracle_done.clear();
	}
	return true;
}
// Check (iii)'s snapshot budget: false (and the reference stopped for good) past 3 GiB in one frame.
static bool OracleSpend(State& st, uint64_t bytes) {
	const uint32_t frame = DebugPresentCount().load();
	if (st.oracle_frame != frame) {
		st.oracle_frame       = frame;
		st.oracle_frame_bytes = 0;
		st.oracle_frame_reads = 0;
	}
	st.oracle_frame_bytes += bytes;
	if (st.oracle_frame_bytes > (uint64_t {3} << 30u) && !st.oracle_stopped) {
		st.oracle_stopped = true;
		std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER oracle stopped: more than 3 GiB of snapshots in frame %u\n", frame);
	}
	return !st.oracle_stopped;
}
// The level-0 blocks of `info` the bytes [lo, hi) touch, as texel rectangles clipped to the extent (block-aligned
// origins; full block rows joined); `func(x0, y0, width, height)`.
template <typename Func>
static void ForBlockRects(const ImageInfo& info, uint64_t lo, uint64_t hi, Func&& func) {
	const auto&    block   = info.tiling.block;
	const auto&    mip     = info.mip_layout[0];
	const uint64_t base    = info.data.address + mip.offset;
	const uint64_t bs      = block.block_size;
	const uint64_t columns = (uint64_t {mip.pitch} + block.block_width - 1u) / block.block_width;
	const uint64_t rows    = (uint64_t {mip.height} + block.block_height - 1u) / block.block_height;
	lo                     = std::max(lo, base);
	hi                     = std::min({hi, info.data.End(), base + columns * rows * bs});
	if (hi <= lo || columns == 0u) return;
	const uint64_t b0 = (lo - base) / bs;
	const uint64_t b1 = (hi - base + bs - 1u) / bs;
	struct Rect {
		uint32_t x0, y0, w, h;
	};
	std::optional<Rect> pending;
	const auto          flush = [&] {
        if (pending) func(pending->x0, pending->y0, pending->w, pending->h);
        pending.reset();
	};
	for (uint64_t r = b0 / columns; r <= (b1 - 1u) / columns; r++) {
		const uint64_t c0 = r == b0 / columns ? b0 % columns : 0u;
		const uint64_t c1 = r == (b1 - 1u) / columns ? (b1 - 1u) % columns + 1u : columns;
		const uint64_t x0 = c0 * block.block_width;
		const uint64_t x1 = std::min<uint64_t>(c1 * block.block_width, info.extent.width);
		const uint64_t y0 = r * block.block_height;
		const uint64_t y1 = std::min<uint64_t>((r + 1u) * block.block_height, info.extent.height);
		if (x1 <= x0 || y1 <= y0) continue;
		const Rect rect {static_cast<uint32_t>(x0), static_cast<uint32_t>(y0), static_cast<uint32_t>(x1 - x0),
		                 static_cast<uint32_t>(y1 - y0)};
		if (pending && pending->x0 == rect.x0 && pending->w == rect.w && pending->y0 + pending->h == rect.y0) {
			pending->h += rect.h;
			continue;
		}
		flush();
		pending = rect;
	}
	flush();
}
// The cover of a texel rectangle: the level-0 blocks it touches, as guest byte ranges.
static Ranges CoverOf(const ImageInfo& info, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
	Ranges cover;
	if (x1 <= x0 || y1 <= y0) return cover;
	if (!info.IsTiled()) { // a linear image: the rectangle's bytes row by row
		const uint64_t e    = info.bytes_per_block;
		const uint64_t row  = uint64_t {info.mip_layout[0].linear_pitch} * e;
		const uint64_t base = info.data.address + info.mip_layout[0].linear_offset;
		for (uint64_t y = y0; y < y1; y++) {
			const uint64_t lo = std::max(base + y * row + x0 * e, info.data.address);
			const uint64_t hi = std::min(base + y * row + x1 * e, info.data.End());
			if (hi <= lo) continue;
			if (!cover.empty() && cover.back().second >= lo) {
				cover.back().second = std::max(cover.back().second, hi);
			} else {
				cover.emplace_back(lo, hi);
			}
		}
		return cover;
	}
	const auto&    block   = info.tiling.block;
	const auto&    mip     = info.mip_layout[0];
	const uint64_t base    = info.data.address + mip.offset;
	const uint64_t columns = (uint64_t {mip.pitch} + block.block_width - 1u) / block.block_width;
	for (uint64_t by = y0 / block.block_height; by <= (y1 - 1u) / block.block_height; by++) {
		const uint64_t lo = std::max(base + (by * columns + x0 / block.block_width) * block.block_size, info.data.address);
		const uint64_t hi =
		    std::min(base + (by * columns + (x1 - 1u) / block.block_width + 1u) * block.block_size, info.data.End());
		if (hi <= lo) continue;
		if (!cover.empty() && cover.back().second >= lo) {
			cover.back().second = std::max(cover.back().second, hi);
		} else {
			cover.emplace_back(lo, hi);
		}
	}
	return cover;
}
static Buffer& Scratch(GraphicContext& graphics, CommandScheduler& scheduler, std::unique_ptr<Buffer>& buffer,
                       uint64_t size) {
	size = Common::AlignUp(std::max<uint64_t>(size, 4u), 4u);
	if (!buffer || buffer->Size() < size) {
		if (buffer) scheduler.DeferOperation([old = std::move(buffer)]() mutable { old.reset(); });
		buffer = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
		                                  std::max<uint64_t>(size, uint64_t {4} << 20u));
	}
	return *buffer;
}
static void MaybeReport(State& st) {
	const auto     now   = std::chrono::steady_clock::now();
	const uint32_t frame = DebugPresentCount().load();
	if (st.window_frame == 0u) {
		st.window_frame = frame == 0u ? 1u : frame;
		st.window_time  = now;
		return;
	}
	if (now - st.window_time < std::chrono::seconds(5)) return;
	const double frames = std::max(1.0, static_cast<double>(frame) - static_cast<double>(st.window_frame));
	const auto&  s      = st.s;
	const auto   pf     = [&](uint64_t v) { return static_cast<double>(v) / frames; };
	const auto   mib    = [&](uint64_t v) { return static_cast<double>(v) / 1048576.0 / frames; };
	std::fprintf(stderr,
	             "PERF image_memory_owner_fix frames=%u-%u per_frame pull_MiB rt=%.2f storage=%.2f read=%.2f guest=%.2f "
	             "moved_MiB=%.2f pulls rt=%.2f storage=%.2f read=%.2f transfers=%.1f writes rt=%.1f storage=%.1f "
	             "whole=%.2f copy=%.2f continue_refused=%.2f excluded_stale=%.2f lost_MiB unregister=%.3f buffer=%.3f "
	             "refresh=%.3f readback=%.3f holder_gone=%.2f adopted=%.2f writeback_MiB=%.3f fill_like=%.3f "
	             "full_writes_MiB=%.3f unregister_written_MiB=%.3f readback_fixed_MiB=%.3f cpu_ms=%.3f (sync=%.3f "
	             "buffer=%.3f other=%.3f memo_hits=%.0f buffer_quick=%.0f) pull_cpu_ms=%.3f "
	             "splits=%.2f direct_MiB=%.2f direct_regions=%.1f retile_MiB=%.2f retile_jobs=%.1f checks=%llu "
	             "pull_diff_events=%llu pull_diff_bytes=%llu "
	             "granules=%zu\n",
	             st.window_frame, frame, mib(s.pull_bytes[0]), mib(s.pull_bytes[1]), mib(s.pull_bytes[2]),
	             mib(s.guest_bytes), mib(s.moved_bytes), pf(s.pulls[0]), pf(s.pulls[1]), pf(s.pulls[2]), pf(s.transfers),
	             pf(s.writes_rt), pf(s.writes_storage), pf(s.writes_whole), pf(s.writes_copy), pf(s.continue_refused),
	             pf(s.excluded_stale), mib(s.unregister_held), mib(s.buffer_lost), mib(s.refresh_lost),
	             mib(s.readback_unfixed), pf(s.holder_gone), pf(s.adopted), mib(s.writeback), mib(s.writeback_fill_like),
	             mib(s.full_writes), mib(s.unregister_written), mib(s.readback_fixed),
	             static_cast<double>(s.cpu_sync_ns + s.cpu_buffer_ns + s.cpu_other_ns) / 1e6 / frames,
	             static_cast<double>(s.cpu_sync_ns) / 1e6 / frames, static_cast<double>(s.cpu_buffer_ns) / 1e6 / frames,
	             static_cast<double>(s.cpu_other_ns) / 1e6 / frames, pf(s.memo_hits), pf(s.buffer_quick),
	             static_cast<double>(s.pull_cpu_ns) / 1e6 / frames,
	             pf(s.splits), mib(s.direct_bytes), pf(s.direct_regions), mib(s.retile_bytes), pf(s.retile_jobs),
	             static_cast<unsigned long long>(st.pull_checks.load()),
	             static_cast<unsigned long long>(st.pull_diff_events.load()),
	             static_cast<unsigned long long>(st.pull_diff_bytes.load()), st.granules.size());
	const auto x = [](std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.load()); };
	std::array<double, 12> alias {};
	for (size_t i = 0; i < alias.size(); i++) alias[i] = pf(g_alias_sync_copies[i].exchange(0));
	std::fprintf(stderr,
	             "PERF image_memory_owner_check frames=%u-%u escape_checks=%llu escape_events=%llu escape_bytes=%llu "
	             "oracle=0x%llx+0x%llx oracle_checks=%llu oracle_events=%llu oracle_bytes=%llu oracle_valid_KiB=%llu "
	             "k1_checks=%llu k1_wrong_bytes=%llu alias_sync_excluded per_frame depth=%.2f msaa=%.2f bc=%.2f "
	             "videoout=%.2f linear=%.2f levels=%.2f layers=%.2f volume=%.2f tail=%.2f misaligned=%.2f other=%.2f\n",
	             st.window_frame, frame, x(st.escape_checks), x(st.escape_events), x(st.escape_bytes),
	             static_cast<unsigned long long>(st.oracle.lo), static_cast<unsigned long long>(st.oracle.hi - st.oracle.lo),
	             x(st.oracle_checks), x(st.oracle_events), x(st.oracle_bytes), x(st.oracle_valid) >> 10u, x(st.k1_checks),
	             x(st.k1_wrong_bytes), alias[1], alias[2], alias[3], alias[4], alias[5], alias[6], alias[7], alias[8],
	             alias[9], alias[10], alias[11]);
	st.s            = Stats {};
	st.window_frame = frame;
	st.window_time  = now;
}
} // namespace ImageMemoryOwners

uint32_t TextureCache::OwnerMode() {
	return ImageMemoryOwnerMode();
}

void TextureCache::OwnerSetDrawRect(bool valid, int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
	ImageMemoryOwners::t_draw_rect = {valid, x0, y0, x1, y1};
}

// The draw's cover of `image` from the rectangle set by OwnerSetDrawRect (the whole image without one).
static ImageMemoryOwners::Ranges OwnerTargetCover(const Image& image) {
	using namespace ImageMemoryOwners;
	const auto& rect  = t_draw_rect;
	const auto  clamp = [](int32_t v, uint32_t limit) {
        return static_cast<uint32_t>(std::clamp<int64_t>(v, 0, static_cast<int64_t>(limit)));
	};
	if (!rect.valid) return CoverOf(image.info, 0u, 0u, image.info.extent.width, image.info.extent.height);
	return CoverOf(image.info, clamp(rect.x0, image.info.extent.width), clamp(rect.y0, image.info.extent.height),
	               clamp(rect.x1, image.info.extent.width), clamp(rect.y1, image.info.extent.height));
}

bool TextureCache::OwnerDrawPrepared(std::span<const ImageId> ids) {
	using namespace ImageMemoryOwners;
	std::scoped_lock lock {m_lock};
	auto&            st = Get();
	for (const ImageId id: ids) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !Followed(*image)) continue;
		const auto run = st.runs.find(KeyOf(id));
		if (run == st.runs.end()) {
			// Never written by a draw run here: alone in its memory, or the full acquisition decides.
			if (image->owner_stale) {
				st.s.continue_refused++;
				return false;
			}
			continue;
		}
		const auto cover = OwnerTargetCover(*image);
		if (image->owner_stale || image->owner_run_disturbed != image->owner_disturbed ||
		    !Covered(run->second.prepared, cover)) {
			st.s.continue_refused++;
			return false;
		}
	}
	return true;
}

void TextureCache::OwnerDisturb(ImageId writer, uint64_t lo, uint64_t hi, std::span<const ImageId> candidates) {
	using namespace ImageMemoryOwners;
	auto& st = Get();
	for (const ImageId id: candidates) {
		if (id == writer) continue;
		auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered || !image->Overlaps(lo, hi - lo, false)) continue;
		if (ImageMemoryOwnerMode() == 2u) OwnerEndInterval(id, true); // its bytes before the other write
		if (!image->owner_stale) {
			image->owner_stale = true;
			image->HeapChanged();
		}
		image->owner_disturbed++;
		image->owner_sole = false;
	}
	(void)st;
}

// Another image took copies of `source`'s bytes (a pull or an image copy): its next write must reach them, so its draw
// run and its sole holding end.
static void OwnerSourceRead(Image& source) {
	source.owner_disturbed++;
	source.owner_sole = false;
}

// The guest bytes of `image`'s blocks over [lo, hi) to `tiled` (tile) or back (detile and load), `tiled`'s byte 0
// standing for guest address lo (block-aligned).
void TextureCache::OwnerTileRange(Image& image, uint64_t lo, uint64_t hi, Buffer& tiled, bool tile) {
	using namespace ImageMemoryOwners;
	if (!image.info.IsTiled()) {
		// A linear image: its rows copied straight between the image and their guest offsets (row length = pitch).
		const uint64_t e          = image.info.bytes_per_block;
		const uint64_t base       = image.info.data.address + image.info.mip_layout[0].linear_offset;
		const uint64_t tiled_size = Common::AlignUp(hi - lo, 4u);
		ForLinearRects(image.info, lo, hi, [&](uint32_t x0, uint32_t y0, uint32_t w, uint32_t h) {
			vk::BufferImageCopy copy {};
			copy.bufferOffset      = base + (uint64_t {y0} * image.info.mip_layout[0].linear_pitch + x0) * e - lo;
			copy.bufferRowLength   = image.info.mip_layout[0].linear_pitch;
			copy.bufferImageHeight = 0;
			copy.imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.imageOffset       = vk::Offset3D {static_cast<int32_t>(x0), static_cast<int32_t>(y0), 0};
			copy.imageExtent       = vk::Extent3D {w, h, 1};
			if (tile) {
				image.Download(std::span(&copy, 1), tiled.Handle(), 0, tiled_size);
			} else {
				image.Upload(std::span(&copy, 1), tiled.Handle(), 0, tiled_size);
			}
		});
		return;
	}
	const uint32_t bpe = image.info.tiling.block.bytes_per_element;
	ImageMemoryOwners::ForBlockRects(image.info, lo, hi, [&](uint32_t x0, uint32_t y0, uint32_t w, uint32_t h) {
		const uint64_t      bytes  = Common::AlignUp(uint64_t {w} * h * bpe, 4u);
		auto&               linear = Scratch(m_graphics, m_scheduler, m_owner_buffers[2], bytes);
		vk::BufferImageCopy copy {};
		copy.bufferOffset      = 0;
		copy.bufferRowLength   = w;
		copy.bufferImageHeight = h;
		copy.imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageOffset       = vk::Offset3D {static_cast<int32_t>(x0), static_cast<int32_t>(y0), 0};
		copy.imageExtent       = vk::Extent3D {w, h, 1};
		const uint64_t tiled_size = Common::AlignUp(hi - lo, 4u);
		if (tile) {
			image.Download(std::span(&copy, 1), linear.Handle(), 0, bytes);
			m_tiler.TileRect(true, image.info, x0, y0, w, h, linear.Handle(), 0, bytes, tiled.Handle(), 0, tiled_size, lo);
		} else {
			m_tiler.TileRect(false, image.info, x0, y0, w, h, linear.Handle(), 0, bytes, tiled.Handle(), 0, tiled_size, lo);
			image.Upload(std::span(&copy, 1), linear.Handle(), 0, bytes);
		}
	});
}

static void OwnerHostBarrier(CommandScheduler& scheduler) {
	auto              command = scheduler.Current().Handle();
	vk::MemoryBarrier host {};
	host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0,
	                        nullptr, 0, nullptr);
}

std::shared_ptr<Buffer> TextureCache::OwnerSnapshot(Image& image, uint32_t fill) {
	const uint64_t size   = Common::AlignUp(image.info.data.size, 4u);
	auto           buffer = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, size);
	m_scheduler.EndRendering();
	const GpuCheckpoints::FeedLabel label(m_scheduler.Current().Handle(), ImageMemoryOwners::kLabelCheck);
	// Bytes of no texel (outside the extent: block padding) are not written and keep `fill`: equal in every snapshot
	// taken with the same fill, told apart from texel bytes by two snapshots with different fills (OwnerOracleRead).
	buffer->Fill(0, size, fill);
	OwnerTileRange(image, image.info.data.address, image.info.data.End(), *buffer, true);
	return buffer;
}

void TextureCache::OwnerBeginInterval(ImageId id, std::span<const std::pair<uint64_t, uint64_t>> allowed, bool escape) {
	using namespace ImageMemoryOwners;
	auto& st    = Get();
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !Followed(*image)) return;
	const bool oracle = OracleWindow(st) && st.oracle.hi > st.oracle.lo &&
	                    image->Overlaps(st.oracle.lo, st.oracle.hi - st.oracle.lo, false) &&
	                    OracleSpend(st, 2u * image->info.data.size);
	if (!escape && !oracle) return;
	OwnerEndInterval(id, true);
	Interval interval;
	interval.id      = id;
	interval.lo      = image->info.data.address;
	interval.hi      = image->info.data.End();
	interval.before  = OwnerSnapshot(*image);
	interval.allowed = Ranges(allowed.begin(), allowed.end());
	interval.escape  = escape;
	interval.oracle  = oracle;
	st.intervals[KeyOf(id)] = std::move(interval);
}

void TextureCache::OwnerEndInterval(ImageId id, bool compare) {
	using namespace ImageMemoryOwners;
	auto&      st = Get();
	const auto it = st.intervals.find(KeyOf(id));
	if (it == st.intervals.end()) return;
	Interval interval = std::move(it->second);
	st.intervals.erase(it);
	auto* image = m_slot_images.try_get(id);
	if (!compare || image == nullptr || !image->registered) return;
	auto after = OwnerSnapshot(*image);
	OwnerHostBarrier(m_scheduler);
	std::string what = DescribeImage(*image);
	m_scheduler.DeferOperation([interval = std::move(interval), after, what = std::move(what)] {
		const uint64_t size = interval.hi - interval.lo;
		interval.before->Invalidate(0, size);
		after->Invalidate(0, size);
		const uint8_t* a       = interval.before->Mapped().data();
		const uint8_t* b       = after->Mapped().data();
		auto&          state   = Get();
		uint64_t       escaped = 0, first = 0;
		std::unique_lock lock(state.oracle.mutex, std::defer_lock);
		if (interval.oracle) lock.lock();
		const uint64_t olo = state.oracle.lo, ohi = state.oracle.hi;
		for (uint64_t n = 0; n < size; n++) {
			if (a[n] == b[n]) continue;
			const uint64_t address = interval.lo + n;
			if (interval.escape && !InRanges(interval.allowed, address)) {
				if (escaped == 0u) first = address;
				escaped++;
			}
			if (interval.oracle && address >= olo && address < ohi) {
				state.oracle.bytes[address - olo] = b[n];
				if (state.oracle.valid[address - olo] == 0u) {
					state.oracle.valid[address - olo] = 1u;
					state.oracle_valid.fetch_add(1, std::memory_order_relaxed);
				}
			}
		}
		if (!interval.escape) return;
		state.escape_checks.fetch_add(1, std::memory_order_relaxed);
		if (escaped == 0u) return;
		state.escape_bytes.fetch_add(escaped, std::memory_order_relaxed);
		if (state.escape_events.fetch_add(1, std::memory_order_relaxed) < 16u) {
			std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER escape bytes=%llu first=0x%llx cover_bytes=%llu image %s\n",
			             static_cast<unsigned long long>(escaped), static_cast<unsigned long long>(first),
			             static_cast<unsigned long long>(Bytes(interval.allowed)), what.c_str());
		}
	});
}

// =2: an open write interval brought up to now - its changes so far go to the reference and to check (i), and it goes
// on from here with the same allowed bytes (a draw run that continues writes into the new interval).
void TextureCache::OwnerFoldInterval(ImageId id) {
	using namespace ImageMemoryOwners;
	auto&      st = Get();
	const auto it = st.intervals.find(KeyOf(id));
	if (it == st.intervals.end()) return;
	const Ranges allowed = it->second.allowed;
	const bool   escape  = it->second.escape;
	OwnerEndInterval(id, true);
	OwnerBeginInterval(id, allowed, escape);
}

// =2, checks (iii) and (iv) at a sampled bind in the window: the image's guest bytes (after its pulls) against the
// reference where the reference knows them; and what the whole-image merge would show there - for each byte, the
// image (of this one and those over its memory) with the newest content serial covering it.
void TextureCache::OwnerOracleRead(ImageId id, std::span<const ImageId> overlaps) {
	using namespace ImageMemoryOwners;
	auto& st    = Get();
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !Followed(*image)) return;
	// Sampled (see State), and within the snapshot budget.
	auto&    done   = st.oracle_done[KeyOf(id)];
	uint64_t needed = 2u * image->info.data.size;
	for (const ImageId other_id: overlaps) {
		if (const auto* other = m_slot_images.try_get(other_id); other != nullptr && Followed(*other)) {
			needed += 2u * other->info.data.size;
		}
	}
	const bool sampled = done < 2u || done % 16u == 0u;
	done++;
	if (!sampled || st.oracle_frame_reads >= 4u || !OracleSpend(st, needed)) return;
	st.oracle_frame_reads++;
	// The reference first takes what the open write intervals of this image and of the images over its memory changed
	// so far: a write interval ends only when another image takes or reads its bytes, not at a read of its own image.
	OwnerFoldInterval(id);
	for (const ImageId other_id: overlaps) OwnerFoldInterval(other_id);
	// Each image twice, with fills 0x00 and 0xFF: a byte equal in both belongs to a texel of the image; a byte that is
	// not (padding of a partial block outside the extent) is no byte of the image and is not compared.
	struct Snap {
		std::shared_ptr<Buffer> buffer, mask;
		uint64_t                lo = 0, hi = 0, serial = 0;
		[[nodiscard]] bool      Has(uint64_t address) const {
            return buffer->Mapped()[address - lo] == mask->Mapped()[address - lo];
		}
	};
	std::vector<Snap> others;
	for (const ImageId other_id: overlaps) {
		auto* other = m_slot_images.try_get(other_id);
		if (other == nullptr || !Followed(*other)) continue;
		others.push_back({OwnerSnapshot(*other), OwnerSnapshot(*other, 0xffffffffu), other->info.data.address,
		                  other->info.data.End(), other->content_serial});
	}
	Snap self {OwnerSnapshot(*image), OwnerSnapshot(*image, 0xffffffffu), image->info.data.address,
	           image->info.data.End(), image->content_serial};
	OwnerHostBarrier(m_scheduler);
	std::string what = DescribeImage(*image);
	m_scheduler.DeferOperation([self, others = std::move(others), what = std::move(what)] {
		auto&            state = Get();
		std::scoped_lock lock(state.oracle.mutex);
		const uint64_t   lo = std::max(self.lo, state.oracle.lo);
		const uint64_t   hi = std::min(self.hi, state.oracle.hi);
		if (hi <= lo) return;
		self.buffer->Invalidate(0, self.hi - self.lo);
		self.mask->Invalidate(0, self.hi - self.lo);
		for (const auto& other: others) {
			other.buffer->Invalidate(0, other.hi - other.lo);
			other.mask->Invalidate(0, other.hi - other.lo);
		}
		uint64_t diff = 0, k1_wrong = 0, first = 0;
		for (uint64_t address = lo; address < hi; address++) {
			const uint64_t r = address - state.oracle.lo;
			if (state.oracle.valid[r] == 0u || !self.Has(address)) continue;
			const uint8_t expected = state.oracle.bytes[r];
			if (self.buffer->Mapped()[address - self.lo] != expected) {
				if (diff == 0u) first = address;
				diff++;
			}
			// The whole-image merge: of the images having this byte, the one with the newest content serial.
			const Snap* newest = &self;
			for (const auto& other: others) {
				if (address >= other.lo && address < other.hi && other.serial > newest->serial && other.Has(address)) {
					newest = &other;
				}
			}
			if (newest->buffer->Mapped()[address - newest->lo] != expected) k1_wrong++;
		}
		state.oracle_checks.fetch_add(1, std::memory_order_relaxed);
		state.k1_checks.fetch_add(1, std::memory_order_relaxed);
		state.k1_wrong_bytes.fetch_add(k1_wrong, std::memory_order_relaxed);
		if (diff == 0u) return;
		state.oracle_bytes.fetch_add(diff, std::memory_order_relaxed);
		if (state.oracle_events.fetch_add(1, std::memory_order_relaxed) < 16u) {
			std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER oracle MISMATCH bytes=%llu first=0x%llx image %s\n",
			             static_cast<unsigned long long>(diff), static_cast<unsigned long long>(first), what.c_str());
		}
	});
}

void TextureCache::OwnerPull(Image& target, ImageId target_id, uint64_t holder,
                             std::span<const std::pair<uint64_t, uint64_t>> all_ranges, uint32_t why) {
	using namespace ImageMemoryOwners;
	auto&          st = Get();
	const auto     pull_start = std::chrono::steady_clock::now();
	const struct PullTime {
		uint64_t&                             total;
		std::chrono::steady_clock::time_point start;
		~PullTime() {
			total += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
		}
	} pull_time {st.s.pull_cpu_ns, pull_start};
	if (m_scheduler.Current().IsRendering()) st.s.splits++;
	std::span<const std::pair<uint64_t, uint64_t>> ranges = all_ranges;
	Image* source = nullptr;
	if (holder != 0u) {
		source = m_slot_images.try_get(IdOf(holder));
		if (source == nullptr || !Followed(*source)) {
			// The holder left without passing its bytes on (counted); guest memory stands in.
			st.s.holder_gone++;
			source = nullptr;
		}
	}
	if (source != nullptr && ImageMemoryOwnerMode() == 2u) OwnerEndInterval(IdOf(holder), true); // its bytes are read
	if (source != nullptr) OwnerSourceRead(*source);
	const uint64_t bt = WindowAlignment(target.info);
	const uint64_t b  = source != nullptr ? std::max<uint64_t>(bt, WindowAlignment(source->info)) : bt;
	const uint64_t t_lo = target.info.data.address, t_hi = target.info.data.End();
	const uint64_t s_lo = source != nullptr ? source->info.data.address : 0u;
	const uint64_t s_hi = source != nullptr ? source->info.data.End() : UINT64_MAX;
	m_scheduler.EndRendering();
	// The fast form: whole blocks arranged alike in both images (DirectPair, equal position XOR) move as texel copies
	// image to image, all of them in one copy command; the rest (partial blocks, other arrangements, guest memory)
	// through guest layout below. Texels of the block that only one image has (outside the other's extent) stay as the
	// target holds them, as in the guest-layout form.
	Ranges slow;
	if (source != nullptr && DirectPair(target, *source)) {
		const auto&                   block = target.info.tiling.block;
		const uint64_t                bs    = block.block_size;
		std::vector<vk::ImageCopy>    regions;
		struct CheckedBlock {
			uint64_t address;
			uint32_t w, h, xor_value;
		};
		std::vector<CheckedBlock> checked;
		auto&                     done  = st.checks_done[KeyOf(target_id)];
		const bool                check = ImageMemoryOwnerMode() == 2u && (done < 4u || done % 64u == 0u);
		uint64_t                  bytes = 0;
		for (const auto& [r_lo, r_hi]: ranges) {
			const uint64_t e_lo = std::max({r_lo, t_lo, s_lo});
			const uint64_t e_hi = std::min({r_hi, t_hi, s_hi});
			if (e_hi <= e_lo) continue;
			const uint64_t b_lo = Common::AlignUp(e_lo, bs);
			const uint64_t b_hi = Common::AlignDown(e_hi, bs);
			if (b_lo >= b_hi) {
				slow.emplace_back(e_lo, e_hi);
				continue;
			}
			if (e_lo < b_lo) slow.emplace_back(e_lo, b_lo);
			for (uint64_t a = b_lo; a < b_hi; a += bs) {
				uint32_t tx = 0, ty = 0, txor = 0, sx = 0, sy = 0, sxor = 0;
				if (!BlockAt(target.info, a, tx, ty, txor) || !BlockAt(source->info, a, sx, sy, sxor) || txor != sxor) {
					if (!slow.empty() && slow.back().second == a) {
						slow.back().second = a + bs;
					} else {
						slow.emplace_back(a, a + bs);
					}
					continue;
				}
				const uint32_t tx0 = tx * block.block_width, ty0 = ty * block.block_height;
				const uint32_t sx0 = sx * block.block_width, sy0 = sy * block.block_height;
				if (tx0 >= target.info.extent.width || ty0 >= target.info.extent.height ||
				    sx0 >= source->info.extent.width || sy0 >= source->info.extent.height) {
					continue; // no texel of this block in one of them: nothing to move
				}
				const uint32_t w = std::min({block.block_width, target.info.extent.width - tx0, source->info.extent.width - sx0});
				const uint32_t h =
				    std::min({block.block_height, target.info.extent.height - ty0, source->info.extent.height - sy0});
				bytes += bs;
				if (check) checked.push_back({a, w, h, txor});
				if (!regions.empty()) {
					auto& last = regions.back();
					if (last.srcOffset.y == static_cast<int32_t>(sy0) && last.dstOffset.y == static_cast<int32_t>(ty0) &&
					    last.extent.height == h && last.extent.width % block.block_width == 0u &&
					    last.srcOffset.x + static_cast<int32_t>(last.extent.width) == static_cast<int32_t>(sx0) &&
					    last.dstOffset.x + static_cast<int32_t>(last.extent.width) == static_cast<int32_t>(tx0)) {
						last.extent.width += w;
						continue;
					}
				}
				vk::ImageCopy copy {};
				copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
				copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
				copy.srcOffset      = vk::Offset3D {static_cast<int32_t>(sx0), static_cast<int32_t>(sy0), 0};
				copy.dstOffset      = vk::Offset3D {static_cast<int32_t>(tx0), static_cast<int32_t>(ty0), 0};
				copy.extent         = vk::Extent3D {w, h, 1};
				regions.push_back(copy);
			}
			if (b_hi < e_hi) slow.emplace_back(b_hi, e_hi);
		}
		if (!regions.empty()) {
			// =2 (sampled with the guest-layout form's check): the source's guest bytes before, the target's after;
			// every byte of a moved texel must be equal.
			std::shared_ptr<Buffer> before_source;
			if (check) before_source = OwnerSnapshot(*source);
			auto                         command = m_scheduler.Current().Handle();
			const GpuCheckpoints::FeedLabel label(command, kLabelDirect);
			source->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
			target.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
			command.copyImage(source->backing.image, vk::ImageLayout::eTransferSrcOptimal, target.backing.image,
			                  vk::ImageLayout::eTransferDstOptimal, static_cast<uint32_t>(regions.size()), regions.data());
			st.s.direct_bytes += bytes;
			st.s.direct_regions += regions.size();
			st.s.moved_bytes += bytes;
			st.s.transfers++;
			if (check) {
				done++;
				st.s.direct_checks++;
				auto after_target = OwnerSnapshot(target);
				OwnerHostBarrier(m_scheduler);
				std::string what = fmt::format("direct target {} holder {}", DescribeImage(target),
				                               DescribeImage(*source));
				m_scheduler.DeferOperation([before_source, after_target, checked = std::move(checked), block, t_lo, s_lo,
				                            t_size = t_hi - t_lo, s_size = s_hi - s_lo, what = std::move(what)] {
					before_source->Invalidate(0, s_size);
					after_target->Invalidate(0, t_size);
					const uint8_t* s    = before_source->Mapped().data();
					const uint8_t* t    = after_target->Mapped().data();
					uint64_t       diff = 0, first = 0;
					for (const auto& b: checked) {
						for (uint32_t v = 0; v < b.h; v++) {
							for (uint32_t u = 0; u < b.w; u++) {
								uint32_t offset = 0;
								if (!TileGetBlockOffset(block, u, v, 0, offset)) continue;
								offset ^= b.xor_value;
								for (uint32_t k = 0; k < block.bytes_per_element; k++) {
									const uint64_t address = b.address + offset + k;
									if (s[address - s_lo] != t[address - t_lo]) {
										if (diff == 0u) first = address;
										diff++;
									}
								}
							}
						}
					}
					auto& state = Get();
					state.pull_checks.fetch_add(1, std::memory_order_relaxed);
					if (diff == 0u) return;
					state.pull_diff_bytes.fetch_add(diff, std::memory_order_relaxed);
					if (state.pull_diff_events.fetch_add(1, std::memory_order_relaxed) < 16u) {
						std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER pull MISMATCH bytes=%llu first=0x%llx %s\n",
						             static_cast<unsigned long long>(diff), static_cast<unsigned long long>(first),
						             what.c_str());
					}
				});
			}
		}
		ranges = slow;
	}
	// The one-pass form for whole blocks of other arrangements (other element sizes, block XOR values or families of
	// the same block size): one retile dispatch per pull, reading the source and writing the target directly
	// (TileManager::Retile); partial blocks and guest memory stay with the guest-layout form below.
	Ranges rest;
	if (source != nullptr && !ranges.empty() && TileManager::RetileFits(target, *source)) {
		const auto&                         tb = target.info.tiling.block;
		const auto&                         sb = source->info.tiling.block;
		const uint64_t                      bs = tb.block_size;
		std::vector<TileManager::RetileJob> jobs;
		std::vector<uint64_t>               job_addresses;
		for (const auto& [r_lo, r_hi]: ranges) {
			const uint64_t e_lo = std::max({r_lo, t_lo, s_lo});
			const uint64_t e_hi = std::min({r_hi, t_hi, s_hi});
			if (e_hi <= e_lo) continue;
			const uint64_t b_lo = Common::AlignUp(e_lo, bs);
			const uint64_t b_hi = Common::AlignDown(e_hi, bs);
			if (b_lo >= b_hi) {
				rest.emplace_back(e_lo, e_hi);
				continue;
			}
			if (e_lo < b_lo) rest.emplace_back(e_lo, b_lo);
			for (uint64_t a = b_lo; a < b_hi; a += bs) {
				uint32_t tx = 0, ty = 0, txor = 0, sx = 0, sy = 0, sxor = 0;
				if (!BlockAt(target.info, a, tx, ty, txor) || !BlockAt(source->info, a, sx, sy, sxor)) {
					rest.emplace_back(a, a + bs);
					continue;
				}
				const uint32_t tx0 = tx * tb.block_width, ty0 = ty * tb.block_height;
				const uint32_t sx0 = sx * sb.block_width, sy0 = sy * sb.block_height;
				if (tx0 >= target.info.extent.width || ty0 >= target.info.extent.height ||
				    sx0 >= source->info.extent.width || sy0 >= source->info.extent.height) {
					continue; // no texel of this block in one of them: nothing to move
				}
				TileManager::RetileJob job {};
				job.t_x0  = tx0;
				job.t_y0  = ty0;
				job.w     = std::min(tb.block_width, target.info.extent.width - tx0);
				job.h     = std::min(tb.block_height, target.info.extent.height - ty0);
				job.s_x0  = sx0;
				job.s_y0  = sy0;
				job.s_w   = std::min(sb.block_width, source->info.extent.width - sx0);
				job.s_h   = std::min(sb.block_height, source->info.extent.height - sy0);
				job.t_xor = txor;
				job.s_xor = sxor;
				jobs.push_back(job);
				job_addresses.push_back(a);
			}
			if (b_hi < e_hi) rest.emplace_back(b_hi, e_hi);
		}
		if (!jobs.empty()) {
			auto&      done  = st.checks_done[KeyOf(target_id)];
			const bool check = ImageMemoryOwnerMode() == 2u && (done < 4u || done % 64u == 0u);
			std::shared_ptr<Buffer> source_bytes, source_mask;
			if (check) {
				source_bytes = OwnerSnapshot(*source);
				source_mask  = OwnerSnapshot(*source, 0xffffffffu);
			}
			{
				const GpuCheckpoints::FeedLabel label(m_scheduler.Current().Handle(), kLabelRetile);
				for (size_t first = 0; first < jobs.size(); first += 0xffffu) {
					const size_t count = std::min<size_t>(0xffffu, jobs.size() - first);
					m_tiler.Retile(target, *source, std::span(jobs).subspan(first, count));
				}
			}
			st.s.retile_bytes += jobs.size() * bs;
			st.s.retile_jobs += jobs.size();
			st.s.moved_bytes += jobs.size() * bs;
			st.s.transfers++;
			if (check) {
				// =2: every byte of a moved target texel that a source texel has equals the source's byte.
				done++;
				auto after_target = OwnerSnapshot(target);
				OwnerHostBarrier(m_scheduler);
				std::string what = fmt::format("retile target {} holder {}", DescribeImage(target),
				                               DescribeImage(*source));
				m_scheduler.DeferOperation([source_bytes, source_mask, after_target, jobs, job_addresses, tb, t_lo, s_lo,
				                            t_size = t_hi - t_lo, s_size = s_hi - s_lo, what = std::move(what)] {
					source_bytes->Invalidate(0, s_size);
					source_mask->Invalidate(0, s_size);
					after_target->Invalidate(0, t_size);
					const uint8_t* s    = source_bytes->Mapped().data();
					const uint8_t* m    = source_mask->Mapped().data();
					const uint8_t* t    = after_target->Mapped().data();
					uint64_t       diff = 0, first = 0;
					for (size_t j = 0; j < jobs.size(); j++) {
						const auto& job = jobs[j];
						for (uint32_t v = 0; v < job.h; v++) {
							for (uint32_t u = 0; u < job.w; u++) {
								uint32_t offset = 0;
								if (!TileGetBlockOffset(tb, u, v, 0, offset)) continue;
								offset ^= job.t_xor;
								for (uint32_t k = 0; k < tb.bytes_per_element; k++) {
									const uint64_t address = job_addresses[j] + offset + k;
									if (s[address - s_lo] != m[address - s_lo]) continue; // no source texel there
									if (s[address - s_lo] != t[address - t_lo]) {
										if (diff == 0u) first = address;
										diff++;
									}
								}
							}
						}
					}
					auto& state = Get();
					state.pull_checks.fetch_add(1, std::memory_order_relaxed);
					if (diff == 0u) return;
					state.pull_diff_bytes.fetch_add(diff, std::memory_order_relaxed);
					if (state.pull_diff_events.fetch_add(1, std::memory_order_relaxed) < 16u) {
						std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER pull MISMATCH bytes=%llu first=0x%llx %s\n",
						             static_cast<unsigned long long>(diff), static_cast<unsigned long long>(first),
						             what.c_str());
					}
				});
			}
		}
		ranges = rest;
	}
	std::optional<GpuCheckpoints::FeedLabel> label;
	if (!ranges.empty()) label.emplace(m_scheduler.Current().Handle(), kLabelPull);
	size_t i = 0;
	while (i < ranges.size()) {
		// One window of whole blocks of both images over consecutive ranges.
		const uint64_t lo = Common::AlignDown(ranges[i].first, b);
		uint64_t       hi = Common::AlignUp(ranges[i].second, b);
		size_t         j  = i + 1;
		while (j < ranges.size() && Common::AlignDown(ranges[j].first, b) <= hi) {
			hi = std::max(hi, Common::AlignUp(ranges[j].second, b));
			j++;
		}
		const uint64_t size     = hi - lo;
		auto&          tiled_t  = Scratch(m_graphics, m_scheduler, m_owner_buffers[0], size);
		OwnerTileRange(target, lo, hi, tiled_t, true);
		std::pair<Buffer*, uint64_t> from {nullptr, 0};
		if (source != nullptr) {
			// The holder's texels over the target's bytes: bytes of texels the holder does not have (outside its
			// extent) keep the target's.
			auto& tiled_s = Scratch(m_graphics, m_scheduler, m_owner_buffers[1], size);
			tiled_s.CopyFrom(m_scheduler.Current(), tiled_t, 0, 0, size);
			OwnerTileRange(*source, lo, hi, tiled_s, true);
			from = {&tiled_s, 0u};
		} else {
			from = m_buffer_cache.ObtainBufferForImage(lo, size);
			if (from.first == nullptr) EXIT("TextureCache: no guest memory buffer for an image memory pull\n");
			// =2, check (iii): guest memory's bytes taken here are memory's bytes now (after the buffer writes that made
			// them unknown to the reference): the reference knows them again.
			if (OracleWindow(st) && st.oracle.hi > st.oracle.lo && lo < st.oracle.hi && hi > st.oracle.lo) {
				auto copy = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags,
				                                     Common::AlignUp(size, 4u));
				copy->CopyFrom(m_scheduler.Current(), *from.first, from.second, 0, size);
				OwnerHostBarrier(m_scheduler);
				Ranges known;
				for (size_t k = i; k < j; k++) {
					const uint64_t e_lo = std::max(ranges[k].first, t_lo);
					const uint64_t e_hi = std::min(ranges[k].second, t_hi);
					if (e_hi > e_lo) known.emplace_back(e_lo, e_hi);
				}
				m_scheduler.DeferOperation([copy, known, lo, size] {
					copy->Invalidate(0, size);
					auto&            state = Get();
					std::scoped_lock lock(state.oracle.mutex);
					for (const auto& [k_lo, k_hi]: known) {
						for (uint64_t a = std::max(k_lo, state.oracle.lo); a < std::min(k_hi, state.oracle.hi); a++) {
							state.oracle.bytes[a - state.oracle.lo] = copy->Mapped()[a - lo];
							if (state.oracle.valid[a - state.oracle.lo] == 0u) {
								state.oracle.valid[a - state.oracle.lo] = 1u;
								state.oracle_valid.fetch_add(1, std::memory_order_relaxed);
							}
						}
					}
				});
			}
		}
		for (size_t k = i; k < j; k++) {
			const uint64_t e_lo = std::max({ranges[k].first, t_lo, s_lo});
			const uint64_t e_hi = std::min({ranges[k].second, t_hi, s_hi});
			if (e_hi <= e_lo) continue;
			tiled_t.CopyFrom(m_scheduler.Current(), *from.first, from.second + (e_lo - lo), e_lo - lo, e_hi - e_lo);
			st.s.moved_bytes += e_hi - e_lo;
		}
		OwnerTileRange(target, lo, hi, tiled_t, false);
		st.s.transfers++;
		// =2: the target tiled again over the window against the bytes it was given (padding texels keep the given
		// bytes: the comparison buffer starts as a copy of them).
		auto& done = st.checks_done[KeyOf(target_id)];
		if (ImageMemoryOwnerMode() == 2u && (done < 4u || done % 64u == 0u)) {
			auto expected = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags,
			                                         Common::AlignUp(size, 4u));
			auto actual   = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags,
			                                         Common::AlignUp(size, 4u));
			expected->CopyFrom(m_scheduler.Current(), tiled_t, 0, 0, size);
			actual->CopyFrom(m_scheduler.Current(), tiled_t, 0, 0, size);
			OwnerTileRange(target, lo, hi, *actual, true);
			auto              command = m_scheduler.Current().Handle();
			vk::MemoryBarrier host {};
			host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
			host.dstAccessMask = vk::AccessFlagBits::eHostRead;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {}, 1, &host,
			                        0, nullptr, 0, nullptr);
			std::string what = fmt::format("target {} holder {} window=0x{:x}+0x{:x}", DescribeImage(target),
			                               source != nullptr ? DescribeImage(*source) : std::string("guest"),
			                               lo, size);
			m_scheduler.DeferOperation([expected, actual, size, what = std::move(what)] {
				expected->Invalidate(0, size);
				actual->Invalidate(0, size);
				const uint8_t* a    = expected->Mapped().data();
				const uint8_t* c    = actual->Mapped().data();
				uint64_t       diff = 0, first = 0;
				for (uint64_t n = 0; n < size; n++) {
					if (a[n] != c[n]) {
						if (diff == 0u) first = n;
						diff++;
					}
				}
				auto& state = Get();
				state.pull_checks.fetch_add(1, std::memory_order_relaxed);
				if (diff == 0u) return;
				state.pull_diff_bytes.fetch_add(diff, std::memory_order_relaxed);
				if (state.pull_diff_events.fetch_add(1, std::memory_order_relaxed) < 16u) {
					std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER pull MISMATCH bytes=%llu first=+0x%llx (%02x != %02x) %s\n",
					             static_cast<unsigned long long>(diff), static_cast<unsigned long long>(first), a[first],
					             c[first], what.c_str());
				}
			});
		}
		done++;
		i = j;
	}
	(void)why;
}

void TextureCache::OwnerSyncImage(ImageId id, uint32_t bind) {
	using namespace ImageMemoryOwners;
	auto& st = Get();
	const CpuTimer cpu_timer(st.s.cpu_sync_ns); // the map's CPU cost
	auto& t = m_slot_images[id];
	if (t.info.data.Empty() || !t.registered) return;
	if (!Followed(t)) {
		if (t.owner_stale) {
			st.s.excluded_stale++;
			t.owner_stale = false;
		}
		MaybeReport(st);
		return;
	}
	// Proofs of nothing to do (no map walk, no allocation): no pull pending, and
	// - alone in its memory, with the image registry unchanged since that was found;
	// - a sampled binding, its overlap set adopted at the current registry (=2: outside the reference's windows);
	// - a render target whose draw run is untouched since its last write, with the same draw rectangle;
	// - a storage binding of an image that holds all of its memory alone, untouched since its last write.
	const uint64_t registry = DrawRepeat::g_image_registry_epoch.load(std::memory_order_relaxed);
	if (!t.owner_stale) {
		const auto&                  rect = t_draw_rect;
		const std::array<int32_t, 5> now {1, rect.valid ? rect.x0 : INT32_MIN, rect.valid ? rect.y0 : INT32_MIN,
		                                  rect.valid ? rect.x1 : INT32_MAX, rect.valid ? rect.y1 : INT32_MAX};
		const bool untouched = t.owner_run_disturbed == t.owner_disturbed;
		if (t.owner_alone_registry == registry || (bind == 2u && untouched && t.owner_memo_rect == now) ||
		    (bind == 1u && untouched && t.owner_sole) ||
		    (bind == 0u && t.owner_adopt_registry == registry && (ImageMemoryOwnerMode() == 1u || !OracleWindow(st)))) {
			st.s.memo_hits++;
			MaybeReport(st);
			return;
		}
	}
	const auto& overlaps = SharedImageOverlaps(t, id); // (its own list: adoption refreshes only other images' lists)
	if (!overlaps.empty() && t.owner_adopt_registry != registry) {
		OwnerAdopt(id, overlaps);
		t.owner_adopt_registry = registry;
	}
	const Key      key = KeyOf(id);
	const uint64_t lo  = t.info.data.address;
	const uint64_t hi  = t.info.data.End();
	// The write range: the draw's cover (render target) or the whole image (storage).
	Ranges cover;
	if (bind == 2u) {
		cover = OwnerTargetCover(t);
	} else if (bind == 1u) {
		cover.emplace_back(lo, hi);
	}
	if (overlaps.empty() && !t.owner_stale) {
		// Alone in its memory: nobody else reads its bytes. A write here is not followed: the image is adopted again
		// (holding its whole memory) when another image comes over it.
		st.runs.erase(key);
		t.owner_alone_registry = registry;
		t.owner_adopt_registry = 0;
		t.owner_memo_rect      = {};
		t.owner_sole           = false;
		MaybeReport(st);
		return;
	}
	t.owner_alone_registry = 0;
	auto&      run     = st.runs[key];
	const bool running = !run.prepared.empty() && t.owner_run_disturbed == t.owner_disturbed;
	if (bind == 2u && !t.owner_stale && running && Covered(run.prepared, cover)) {
		const auto& rect  = t_draw_rect;
		t.owner_memo_rect = {1, rect.valid ? rect.x0 : INT32_MIN, rect.valid ? rect.y0 : INT32_MIN,
		                     rect.valid ? rect.x1 : INT32_MAX, rect.valid ? rect.y1 : INT32_MAX};
		MaybeReport(st); // the draw run goes on within what it holds
		return;
	}
	// =2: a pull into it, or a write that does not go on with its draw run, ends its write interval.
	const bool check      = ImageMemoryOwnerMode() == 2u;
	const bool continuing = bind == 2u && running;
	if (check && (t.owner_stale || (!cover.empty() && !continuing))) OwnerEndInterval(id, true);
	// Pulls: what it does not hold over the range it reads (the whole image) or writes (the cover).
	if (t.owner_stale) {
		const Ranges whole {Range {lo, hi}};
		const Ranges& reach = bind == 2u ? cover : whole;
		const auto    why   = bind == 2u ? Why::Rt : bind == 1u ? Why::Storage : Why::Read;
		std::map<Key, Ranges> by_holder;
		for (const auto& [rlo, rhi]: reach) {
			ForSegmentsIn(st.granules, rlo, rhi, false, [&](Segment& seg) {
				if (seg.holder == key || Holds(seg.current, key)) return;
				auto& list = by_holder[seg.holder];
				if (!list.empty() && list.back().second == seg.lo) {
					list.back().second = seg.hi;
				} else {
					list.emplace_back(seg.lo, seg.hi);
				}
				AddKey(seg.current, key);
			});
		}
		for (const auto& [holder, ranges]: by_holder) {
			const uint64_t bytes = Bytes(ranges);
			if (holder == 0u) {
				st.s.guest_bytes += bytes;
			} else {
				st.s.pull_bytes[static_cast<size_t>(why)] += bytes;
			}
			OwnerPull(t, id, holder, ranges, static_cast<uint32_t>(why));
		}
		if (!by_holder.empty()) st.s.pulls[static_cast<size_t>(why)]++;
		// Still not holding every byte of its memory (a render target pulls only its cover).
		bool rest = false;
		if (bind == 2u) {
			for (auto it = st.granules.lower_bound(lo & ~(ImageMemorySegments::kGranule - 1u));
			     it != st.granules.end() && it->first < hi && !rest; ++it) {
				for (const auto& seg: it->second) {
					if (seg.hi > lo && seg.lo < hi && seg.holder != key && !Holds(seg.current, key)) rest = true;
				}
			}
		}
		t.owner_stale = rest;
	}
	if (cover.empty()) {
		// =2, checks (iii) and (iv): a sampled bind in the window over the reference range (chosen here when no
		// address was given: this image and the images over its memory, when at least 3).
		if (check && OracleWindow(st)) {
			if (st.oracle.hi == 0u && overlaps.size() >= 3u) {
				uint64_t olo = lo, ohi = hi;
				for (const ImageId other_id: overlaps) {
					if (const auto* other = m_slot_images.try_get(other_id); other != nullptr) {
						olo = std::min(olo, other->info.data.address);
						ohi = std::max(ohi, other->info.data.End());
					}
				}
				ohi = std::min(ohi, olo + 0x10000000ull);
				std::scoped_lock lock(st.oracle.mutex);
				st.oracle.lo = olo;
				st.oracle.hi = ohi;
				st.oracle.bytes.assign(ohi - olo, 0);
				st.oracle.valid.assign(ohi - olo, 0);
				std::fprintf(stderr, "FIX_IMAGE_MEMORY_OWNER oracle range 0x%llx+0x%llx (image %s)\n",
				             static_cast<unsigned long long>(olo), static_cast<unsigned long long>(ohi - olo),
				             DescribeImage(t).c_str());
			}
			if (st.oracle.hi > st.oracle.lo && t.Overlaps(st.oracle.lo, st.oracle.hi - st.oracle.lo, false)) {
				OwnerOracleRead(id, overlaps);
			}
		}
		MaybeReport(st);
		return;
	}
	// The write: its only holder over the cover.
	for (const auto& [wlo, whi]: cover) {
		ForSegmentsIn(st.granules, wlo, whi, true, [&](Segment& seg) {
			seg.holder = key;
			seg.all    = false;
			seg.current.assign(1, key);
		});
	}
	OwnerDisturb(id, cover.front().first, cover.back().second, overlaps);
	if (bind == 2u) {
		st.s.writes_rt++;
		if (!running) run.prepared.clear();
		Merge(run.prepared, cover);
		const auto& rect  = t_draw_rect;
		t.owner_memo_rect = {1, rect.valid ? rect.x0 : INT32_MIN, rect.valid ? rect.y0 : INT32_MIN,
		                     rect.valid ? rect.x1 : INT32_MAX, rect.valid ? rect.y1 : INT32_MAX};
		t.owner_sole      = false; // (its cover only)
	} else {
		st.s.writes_storage++;
		run.prepared      = cover;
		t.owner_memo_rect = {};
		t.owner_sole      = true; // the whole memory, current = {it}
	}
	t.owner_run_disturbed = t.owner_disturbed;
	// =2: the write interval goes on (its allowed bytes grow with the run) or starts (check (i) sampled: the first 4
	// draw runs of an image, then 1 in 64; check (iii) for every write in the window over the reference range).
	if (check) {
		if (const auto interval = st.intervals.find(key); interval != st.intervals.end()) {
			interval->second.allowed = run.prepared;
		} else {
			bool escape = false;
			if (bind == 2u) {
				auto& done = st.escape_done[key];
				escape     = done < 4u || done % 64u == 0u;
				done++;
			}
			OwnerBeginInterval(id, run.prepared, escape);
		}
	}
	MaybeReport(st);
}

// Images that wrote their memory while no other image shared it hold their whole memory when sharing starts (the map
// takes them once, oldest content first; images never written by the GPU hold guest memory's bytes and need nothing).
void TextureCache::OwnerAdopt(ImageId id, std::span<const ImageId> overlaps) {
	using namespace ImageMemoryOwners;
	auto&                                      st = Get();
	std::vector<std::pair<uint64_t, ImageId>> pending;
	const auto                                 consider = [&](ImageId other_id) {
        const auto* other = m_slot_images.try_get(other_id);
        if (other == nullptr || other->info.data.Empty() || !Followed(*other)) return;
        if (!st.runs.try_emplace(KeyOf(other_id)).second) return;
        if (other->IsGpuModified()) pending.emplace_back(other->content_serial, other_id);
	};
	consider(id);
	for (const ImageId other_id: overlaps) consider(other_id);
	std::ranges::sort(pending);
	for (const auto& [serial, other_id]: pending) {
		auto&          other = m_slot_images[other_id];
		const Key      key   = KeyOf(other_id);
		const uint64_t lo    = other.info.data.address;
		const uint64_t hi    = other.info.data.End();
		ForSegmentsIn(st.granules, lo, hi, true, [&](Segment& seg) {
			seg.holder = key;
			seg.all    = false;
			seg.current.assign(1, key);
		});
		const auto others = SharedImageOverlaps(other, other_id);
		OwnerDisturb(other_id, lo, hi, others);
		st.s.adopted++;
	}
}

void TextureCache::OwnerWholeWrite(ImageId id) {
	using namespace ImageMemoryOwners;
	const CpuTimer cpu_timer(Get().s.cpu_other_ns); // the map's CPU cost
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || image->info.data.Empty() || !Followed(*image)) return;
	const auto& overlaps = SharedImageOverlaps(*image, id);
	if (overlaps.empty()) return;
	auto&          st  = Get();
	const Key      key = KeyOf(id);
	const uint64_t lo  = image->info.data.address;
	const uint64_t hi  = image->info.data.End();
	if (ImageMemoryOwnerMode() == 2u) OwnerEndInterval(id, true); // (called before the clear is recorded)
	ForSegmentsIn(st.granules, lo, hi, true, [&](Segment& seg) {
		seg.holder = key;
		seg.all    = false;
		seg.current.assign(1, key);
	});
	if (ImageMemoryOwnerMode() == 2u) {
		const Ranges whole {Range {lo, hi}};
		OwnerBeginInterval(id, whole, false); // check (iii): the clear's bytes go to the reference
	}
	OwnerDisturb(id, lo, hi, overlaps);
	image->owner_stale = false;
	auto& run          = st.runs[key];
	run.prepared.assign(1, Range {lo, hi});
	image->owner_run_disturbed = image->owner_disturbed;
	image->owner_memo_rect     = {};
	image->owner_sole          = true;
	st.s.writes_whole++;
}

void TextureCache::OwnerCopyWrite(ImageId destination_id, ImageId source_id) {
	using namespace ImageMemoryOwners;
	const CpuTimer cpu_timer(Get().s.cpu_other_ns); // the map's CPU cost
	auto*       destination = m_slot_images.try_get(destination_id);
	const auto* source      = m_slot_images.try_get(source_id);
	if (destination == nullptr || destination->info.data.Empty() || !Followed(*destination)) return;
	if (source == nullptr || !Followed(*source)) {
		OwnerWholeWrite(destination_id);
		return;
	}
	OwnerSourceRead(*m_slot_images.try_get(source_id));
	destination->owner_disturbed++;
	destination->owner_sole      = false;
	destination->owner_memo_rect = {};
	// The destination's texels now hold what the source's held: current where the source is, not where it is not.
	// (=2: no change of memory - its write interval ends without a comparison.)
	if (ImageMemoryOwnerMode() == 2u) OwnerEndInterval(destination_id, false);
	auto&          st  = Get();
	const Key      dst = KeyOf(destination_id);
	const Key      src = KeyOf(source_id);
	const uint64_t lo  = std::max(destination->info.data.address, source->info.data.address);
	const uint64_t hi  = std::min(destination->info.data.End(), source->info.data.End());
	bool           stale = destination->owner_stale;
	ForSegmentsIn(st.granules, lo, hi, false, [&](Segment& seg) {
		if (seg.holder == src || Holds(seg.current, src)) {
			AddKey(seg.current, dst);
		} else {
			if (seg.holder == dst) {
				st.s.refresh_lost += seg.hi - seg.lo;
				seg.holder = 0u;
			}
			RemoveKey(seg.current, dst);
			stale = true;
		}
	});
	if (stale && !destination->owner_stale) {
		destination->owner_stale = true;
		destination->HeapChanged();
	}
	st.s.writes_copy++;
}

void TextureCache::OwnerWriteBack(Image& holder, std::span<const std::pair<uint64_t, uint64_t>> ranges, Buffer& destination,
                                  uint64_t destination_offset, uint64_t destination_address) {
	using namespace ImageMemoryOwners;
	const uint64_t b     = WindowAlignment(holder.info);
	const uint64_t h_lo  = holder.info.data.address;
	const uint64_t h_hi  = holder.info.data.End();
	size_t         i     = 0;
	while (i < ranges.size()) {
		// One window of whole blocks of the holder over consecutive ranges, as in OwnerPull.
		const uint64_t lo = Common::AlignDown(ranges[i].first, b);
		uint64_t       hi = Common::AlignUp(ranges[i].second, b);
		size_t         j  = i + 1;
		while (j < ranges.size() && Common::AlignDown(ranges[j].first, b) <= hi) {
			hi = std::max(hi, Common::AlignUp(ranges[j].second, b));
			j++;
		}
		auto& tiled = Scratch(m_graphics, m_scheduler, m_owner_buffers[1], hi - lo);
		// The destination's bytes first: bytes of texels the holder does not have (outside its extent) keep them.
		const uint64_t available = destination_address + (destination.Size() - destination_offset);
		const uint64_t c_lo      = std::max(lo, destination_address);
		const uint64_t c_hi      = std::min(hi, available);
		if (c_hi > c_lo) {
			tiled.CopyFrom(m_scheduler.Current(), destination, destination_offset + (c_lo - destination_address), c_lo - lo,
			               c_hi - c_lo);
		}
		OwnerTileRange(holder, lo, hi, tiled, true);
		for (size_t k = i; k < j; k++) {
			const uint64_t e_lo = std::max(ranges[k].first, h_lo);
			const uint64_t e_hi = std::min(ranges[k].second, h_hi);
			if (e_hi <= e_lo || e_lo < destination_address) continue;
			destination.CopyFrom(m_scheduler.Current(), tiled, e_lo - lo, destination_offset + (e_lo - destination_address),
			                     e_hi - e_lo);
		}
		i = j;
	}
}

// Whether OwnerBufferWrite records GPU work for this write (image-held bytes to write back, or =2's snapshots).
bool TextureCache::OwnerBufferWriteRecords(uint64_t address, uint64_t size) const {
	using namespace ImageMemoryOwners;
	if (ImageMemoryOwnerMode() == 2u) return true;
	const auto& granules = Get().granules;
	for (auto it = granules.lower_bound(address & ~(ImageMemorySegments::kGranule - 1u));
	     it != granules.end() && it->first < address + size; ++it) {
		for (const auto& seg: it->second) {
			if (seg.holder != 0u && seg.hi > address && seg.lo < address + size) return true;
		}
	}
	return false;
}

// A GPU buffer write (InvalidateMemoryFromGPU: a written buffer binding or device-address range, at its binding before
// the shader runs; or a fill or copy). The bytes it may change: the binding's fixed write ranges when the shader fixes
// every write's address, else its whole range. Image-held bytes there are written into guest memory's GPU copy first
// (the cached buffer the binding uses; its pages are GPU-modified by the binding), so bytes the shader leaves alone keep
// their value; a fill or copy changes every byte of its range, so nothing is written back. Then guest memory holds them.
void TextureCache::OwnerBufferWrite(uint64_t address, uint64_t size) {
	using namespace ImageMemoryOwners;
	const CpuTimer cpu_timer(Get().s.cpu_buffer_ns); // the map's CPU cost
	auto&          st = Get();
	// Nothing to do (the common repeat of a binding) when no byte of the range is held or copied by an image and the
	// reference (=2) is not watching: one read-only walk of the granules there.
	if (!OracleWindow(st)) {
		bool any = false;
		for (auto it = st.granules.lower_bound(address & ~(ImageMemorySegments::kGranule - 1u));
		     it != st.granules.end() && it->first < address + size && !any; ++it) {
			for (const auto& seg: it->second) {
				if (seg.hi > address && seg.lo < address + size && (seg.holder != 0u || !seg.current.empty())) {
					any = true;
					break;
				}
			}
		}
		if (!any) {
			st.s.buffer_quick++;
			return;
		}
	}
	const auto* binding = t_owner_census_binding;
	Ranges      written;
	if (binding != nullptr && binding->kind == 1u && binding->fixed_ranges != nullptr) {
		for (const auto& [begin, end]: *binding->fixed_ranges) {
			const uint64_t lo = address + begin;
			const uint64_t hi = std::min<uint64_t>(address + end, address + size);
			if (hi > lo) written.emplace_back(lo, hi);
		}
	} else {
		written.emplace_back(address, address + size);
	}
	const bool full = binding == nullptr;
	// =2, check (iii): the bytes the write may change are unknown to the reference from here on (in command order, as
	// every reference update); a later pull from guest memory makes them known again (OwnerPull).
	if (OracleWindow(st) && st.oracle.hi > st.oracle.lo && address < st.oracle.hi && address + size > st.oracle.lo) {
		// The write intervals of images over the range end first: their changes come before the buffer write's.
		for (const ImageId id: FindImagesInRegion(address, size, true)) {
			if (st.intervals.contains(KeyOf(id))) OwnerEndInterval(id, true);
		}
		m_scheduler.DeferOperation([written] {
			auto&            state = Get();
			std::scoped_lock lock(state.oracle.mutex);
			for (const auto& [wlo, whi]: written) {
				const uint64_t lo = std::max(wlo, state.oracle.lo);
				const uint64_t hi = std::min(whi, state.oracle.hi);
				for (uint64_t a = lo; a < hi; a++) {
					if (state.oracle.valid[a - state.oracle.lo] != 0u) {
						state.oracle.valid[a - state.oracle.lo] = 0u;
						state.oracle_valid.fetch_sub(1, std::memory_order_relaxed);
					}
				}
			}
		});
	}
	bool                  touched = false;
	std::map<Key, Ranges> by_holder;
	for (const auto& [wlo, whi]: written) {
		ForSegmentsIn(st.granules, wlo, whi, false, [&](Segment& seg) {
			touched = true;
			if (seg.holder == 0u) return;
			if (full) {
				st.s.full_writes += seg.hi - seg.lo;
				return;
			}
			auto& list = by_holder[seg.holder];
			if (!list.empty() && list.back().second == seg.lo) {
				list.back().second = seg.hi;
			} else {
				list.emplace_back(seg.lo, seg.hi);
			}
		});
	}
	if (!touched) return;
	if (!by_holder.empty()) {
		const auto [buffer, offset] = m_buffer_cache.CachedBufferFor(address, size);
		const bool fill_like        = binding->formatted && !binding->read && !binding->atomic;
		if (m_scheduler.Current().IsRendering()) st.s.splits++;
		const GpuCheckpoints::FeedLabel label(m_scheduler.Current().Handle(), kLabelWriteBack);
		for (const auto& [holder, ranges]: by_holder) {
			auto*          image = m_slot_images.try_get(IdOf(holder));
			const uint64_t bytes = Bytes(ranges);
			if (buffer == nullptr || image == nullptr || !Followed(*image)) {
				st.s.buffer_lost += bytes; // no GPU copy of guest memory to write into, or the holder left
				continue;
			}
			OwnerWriteBack(*image, ranges, *buffer, offset, address);
			st.s.writeback += bytes;
			if (fill_like) st.s.writeback_fill_like += bytes;
		}
	}
	for (const auto& [wlo, whi]: written) {
		ForSegmentsIn(st.granules, wlo, whi, false, [&](Segment& seg) {
			seg.holder = 0u;
			seg.current.clear();
		});
	}
	std::vector<ImageId> images;
	for (const ImageId id: FindImagesInRegion(address, size, true)) images.push_back(id);
	OwnerDisturb(ImageId {}, address, address + size, images);
}

// A CPU readback of an image the map covers (DownloadImageMemory): the downloaded image's bytes that another image or
// guest memory holds are replaced by the holder's (guest memory's GPU copy for guest-held bytes) before they reach
// guest memory.
void TextureCache::OwnerFixReadback(ImageId id, Buffer& destination, uint64_t destination_offset) {
	using namespace ImageMemoryOwners;
	const CpuTimer cpu_timer(Get().s.cpu_other_ns); // the map's CPU cost
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !Followed(*image)) return;
	auto&                 st  = Get();
	const Key             key = KeyOf(id);
	const uint64_t        lo  = image->info.data.address;
	std::map<Key, Ranges> by_holder;
	for (auto it = st.granules.lower_bound(lo & ~(ImageMemorySegments::kGranule - 1u));
	     it != st.granules.end() && it->first < image->info.data.End(); ++it) {
		for (const auto& seg: it->second) {
			const uint64_t s_lo = std::max(seg.lo, lo);
			const uint64_t s_hi = std::min(seg.hi, image->info.data.End());
			if (s_hi <= s_lo || seg.holder == key || Holds(seg.current, key)) continue;
			auto& list = by_holder[seg.holder];
			if (!list.empty() && list.back().second == s_lo) {
				list.back().second = s_hi;
			} else {
				list.emplace_back(s_lo, s_hi);
			}
		}
	}
	if (by_holder.empty()) return;
	const GpuCheckpoints::FeedLabel label(m_scheduler.Current().Handle(), kLabelReadback);
	for (const auto& [holder, ranges]: by_holder) {
		const uint64_t bytes = Bytes(ranges);
		if (holder == 0u) {
			for (const auto& [r_lo, r_hi]: ranges) {
				const auto [buffer, offset] = m_buffer_cache.CachedBufferFor(r_lo, r_hi - r_lo);
				if (buffer == nullptr) {
					st.s.readback_unfixed += r_hi - r_lo;
					continue;
				}
				destination.CopyFrom(m_scheduler.Current(), *buffer, offset, destination_offset + (r_lo - lo), r_hi - r_lo);
				st.s.readback_fixed += r_hi - r_lo;
			}
			continue;
		}
		auto* source = m_slot_images.try_get(IdOf(holder));
		if (source == nullptr || !Followed(*source)) {
			st.s.readback_unfixed += bytes;
			continue;
		}
		OwnerWriteBack(*source, ranges, destination, destination_offset, lo);
		st.s.readback_fixed += bytes;
	}
}

void TextureCache::OwnerUnregister(ImageId id) {
	using namespace ImageMemoryOwners;
	const CpuTimer cpu_timer(Get().s.cpu_other_ns); // the map's CPU cost
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || image->info.data.Empty()) return;
	auto&     st  = Get();
	const Key key = KeyOf(id);
	if (ImageMemoryOwnerMode() == 2u) OwnerEndInterval(id, false);
	st.escape_done.erase(key);
	// Bytes only it holds go to guest memory's GPU copy when the cached buffer there holds them all and the
	// image can still be read here (no recording ahead of the previous commit); otherwise they are lost (counted).
	Ranges sole;
	ForSegmentsIn(st.granules, image->info.data.address, image->info.data.End(), false, [&](Segment& seg) {
		RemoveKey(seg.current, key);
		if (seg.holder != key) return;
		if (!seg.current.empty()) {
			seg.holder = seg.current.front(); // another image holds the same bytes
		} else {
			seg.holder = 0u;
			if (!sole.empty() && sole.back().second == seg.lo) {
				sole.back().second = seg.hi;
			} else {
				sole.emplace_back(seg.lo, seg.hi);
			}
		}
	});
	if (!sole.empty()) {
		const uint64_t lo    = sole.front().first;
		const uint64_t bytes = Bytes(sole);
		const auto [buffer, offset] = m_buffer_cache.CachedBufferFor(lo, sole.back().second - lo);
		if (buffer != nullptr && Followed(*image)) {
			if (m_scheduler.Current().IsRendering()) st.s.splits++;
			const GpuCheckpoints::FeedLabel label(m_scheduler.Current().Handle(), kLabelUnregister);
			OwnerWriteBack(*image, sole, *buffer, offset, lo);
			st.s.unregister_written += bytes;
		} else {
			st.s.unregister_held += bytes;
		}
	}
	st.runs.erase(key);
	st.checks_done.erase(key);
}

void TextureCache::OwnerAfterUpload(ImageId id) {
	using namespace ImageMemoryOwners;
	const CpuTimer cpu_timer(Get().s.cpu_other_ns); // the map's CPU cost
	auto& image = m_slot_images[id];
	if (image.info.data.Empty() || !Followed(image)) return;
	image.owner_disturbed++;
	image.owner_sole      = false;
	image.owner_memo_rect = {};
	// Its texels now hold guest memory's bytes: current where guest memory holds them, not where an image does.
	// (=2: not a change of memory - its write interval ends without a comparison.)
	if (ImageMemoryOwnerMode() == 2u) OwnerEndInterval(id, false);
	auto&     st    = Get();
	const Key key   = KeyOf(id);
	bool      stale = false;
	ForSegmentsIn(st.granules, image.info.data.address, image.info.data.End(), false, [&](Segment& seg) {
		if (seg.holder == key) {
			st.s.refresh_lost += seg.hi - seg.lo;
			seg.holder = 0u;
		}
		if (seg.holder == 0u) {
			AddKey(seg.current, key);
		} else {
			RemoveKey(seg.current, key);
			stale = true;
		}
	});
	if (stale && !image.owner_stale) {
		image.owner_stale = true;
		image.HeapChanged();
	}
}

namespace SharedImageMemoryStats {
struct Stats {
	std::atomic<uint64_t> binds {0}, overlapped {0}, fast {0}, general {0}, direct {0}, refused {0}, skipped {0}, bytes {0};
	std::atomic<uint64_t> checks {0}, check_diff_events {0}, check_diff_bytes {0}, check_skipped {0};
	std::atomic<uint32_t> window_start {0};
};
static Stats g_stats;
static void Report() {
	auto&          s     = g_stats;
	const uint32_t frame = DebugPresentCount().load();
	uint32_t       start = s.window_start.load(std::memory_order_relaxed);
	if (start == 0u) {
		s.window_start.compare_exchange_strong(start, frame == 0u ? 1u : frame);
		return;
	}
	if (frame < start + 300u || !s.window_start.compare_exchange_strong(start, frame)) return;
	const auto x = [](std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.exchange(0)); };
	std::fprintf(stderr,
	             "PERF shared_image_memory frames=%u-%u binds=%llu overlapped=%llu syncs fast=%llu general=%llu direct=%llu "
	             "refused=%llu skipped_writers=%llu MiB=%llu checks=%llu check_diff_events=%llu check_diff_bytes=%llu "
	             "check_skipped=%llu\n",
	             start, frame, x(s.binds), x(s.overlapped), x(s.fast), x(s.general), x(s.direct), x(s.refused), x(s.skipped),
	             x(s.bytes) >> 20u, x(s.checks), x(s.check_diff_events), x(s.check_diff_bytes), x(s.check_skipped));
}
} // namespace SharedImageMemoryStats

// KYTY_FIX_SHARED_IMAGE_MEMORY=3/4: at the first 3 binds after a probe fill, the texels of level 0 / layer 0 still
// holding the pattern (=3) or a NaN / the pattern bytes (=4, baseline). Counted per bind kind; a sampled or storage
// bind finding them is a read of bytes the image did not write itself (first 48 such events logged).
void TextureCache::ProbeSharedImageRead(ImageId id, uint32_t bind) {
	auto& image = m_slot_images[id];
	if (image.probe_pending == 0u || image.info.data.Empty()) return;
	if (++image.probe_checks > 3u) {
		image.probe_pending = 0;
		return;
	}
	struct ProbeStats {
		std::array<std::atomic<uint64_t>, 5> checks {}, found {}, texels {};
	};
	static ProbeStats stats;
	static std::atomic<uint32_t> window {0};
	const auto& info     = image.info;
	const uint64_t size  = info.LinearSize(1);
	auto           check = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, size);
	image.Download(info.BufferCopies(0, 1), check->Handle(), 0, size);
	auto              command = m_scheduler.Current().Handle();
	vk::MemoryBarrier host {};
	host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0,
	                        nullptr, 0, nullptr);
	const bool     floating = image.probe_pending == 2u;
	const uint32_t bpe      = vk::blockSize(image.backing.format);
	const uint32_t width    = info.extent.width;
	const uint32_t height   = info.extent.height;
	const uint64_t stride   = uint64_t {info.mip_layout[0].linear_pitch} * bpe;
	const auto     format   = image.backing.format;
	const uint32_t check_no = image.probe_checks;
	std::string    what     = fmt::format("image=0x{:x} {}x{} {}", info.data.address, width, height, vk::to_string(format));
	const uint32_t frame    = DebugPresentCount().load();
	if (bind == 0u || bind == 1u) image.probe_pending = 0; // a read: the question is answered
	m_scheduler.DeferOperation([check, floating, bpe, width, height, stride, format, bind, check_no, what = std::move(what),
	                            frame] {
		check->Invalidate(0, check->Size());
		const uint8_t* base     = check->Mapped().data();
		const auto     half_nan = [](uint16_t h) { return (h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0u; };
		uint64_t       found    = 0;
		for (uint32_t y = 0; y < height; ++y) {
			const uint8_t* row = base + y * stride;
			if (y * stride + uint64_t {width} * bpe > check->Size()) break;
			for (uint32_t x = 0; x < width; ++x) {
				const uint8_t* t   = row + uint64_t {x} * bpe;
				bool           hit = false;
				if (floating) {
					if (format == vk::Format::eB10G11R11UfloatPack32) {
						uint32_t v = 0;
						std::memcpy(&v, t, 4);
						const uint32_t r = v & 0x7ffu, g = (v >> 11u) & 0x7ffu, b = v >> 22u;
						hit = ((r >> 6u) == 0x1fu && (r & 0x3fu) != 0u) || ((g >> 6u) == 0x1fu && (g & 0x3fu) != 0u) ||
						      ((b >> 5u) == 0x1fu && (b & 0x1fu) != 0u);
					} else if (format == vk::Format::eR32Sfloat || format == vk::Format::eR32G32Sfloat ||
					           format == vk::Format::eR32G32B32A32Sfloat) {
						for (uint32_t c = 0; c < bpe / 4u && !hit; ++c) {
							uint32_t v = 0;
							std::memcpy(&v, t + c * 4u, 4);
							hit = (v & 0x7f800000u) == 0x7f800000u && (v & 0x7fffffu) != 0u;
						}
					} else {
						for (uint32_t c = 0; c < bpe / 2u && !hit; ++c) {
							uint16_t v = 0;
							std::memcpy(&v, t + c * 2u, 2);
							hit = half_nan(v);
						}
					}
				} else {
					hit = true;
					for (uint32_t c = 0; c < bpe; ++c) hit &= t[c] == 0xa5u;
				}
				found += hit ? 1u : 0u;
			}
		}
		stats.checks[bind].fetch_add(1, std::memory_order_relaxed);
		if (found != 0u) {
			stats.found[bind].fetch_add(1, std::memory_order_relaxed);
			stats.texels[bind].fetch_add(found, std::memory_order_relaxed);
		}
		static std::atomic_uint logged {0};
		if (found != 0u && (bind == 0u || bind == 1u) && logged.fetch_add(1) < 48u) {
			std::fprintf(stderr, "SHARED_IMAGE_PROBE read frame=%u bind=%u check=%u texels=%llu of %llu %s\n", frame, bind,
			             check_no, static_cast<unsigned long long>(found),
			             static_cast<unsigned long long>(uint64_t {width} * height), what.c_str());
		}
		uint32_t start = window.load(std::memory_order_relaxed);
		if (start == 0u) {
			window.compare_exchange_strong(start, frame == 0u ? 1u : frame);
		} else if (frame >= start + 300u && window.compare_exchange_strong(start, frame)) {
			const auto x = [](std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.exchange(0)); };
			std::fprintf(stderr,
			             "PERF shared_image_probe frames=%u-%u checks tex=%llu storage=%llu rt=%llu found tex=%llu storage=%llu "
			             "rt=%llu texels tex=%llu storage=%llu rt=%llu\n",
			             start, frame, x(stats.checks[0]), x(stats.checks[1]), x(stats.checks[2]), x(stats.found[0]),
			             x(stats.found[1]), x(stats.found[2]), x(stats.texels[0]), x(stats.texels[1]), x(stats.texels[2]));
		}
	});
}

bool TextureCache::SyncSharedImageMemory(ImageId id, uint32_t bind) {
	using namespace SharedImageMemoryStats;
	auto& s     = g_stats;
	auto& image = m_slot_images[id];
	if (!SharedImagePlain(image) || !image.registered) return false;
	s.binds.fetch_add(1, std::memory_order_relaxed);
	const auto& overlaps = SharedImageOverlaps(image, id);
	if (overlaps.empty()) {
		Report();
		return false;
	}
	s.overlapped.fetch_add(1, std::memory_order_relaxed);
	const uint64_t       merged = std::max(image.content_serial, image.alias_synced_serial);
	std::vector<ImageId> newer;
	for (const ImageId other_id: overlaps) {
		const auto* other = m_slot_images.try_get(other_id);
		if (other == nullptr || !other->registered || other->content_serial <= merged) continue;
		if (!SharedImagePlain(*other)) {
			s.skipped.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		newer.push_back(other_id);
	}
	if (newer.empty()) {
		Report();
		return false;
	}
	std::ranges::sort(newer, [&](ImageId a, ImageId b) {
		const auto sa = m_slot_images[a].content_serial;
		const auto sb = m_slot_images[b].content_serial;
		return sa != sb ? sa < sb : a.index < b.index;
	});
	const uint64_t newest = m_slot_images[newer.back()].content_serial;
	// Same layout: texel (x, y) at the same guest address in both, so the image copy of the common rectangle writes the
	// guest bytes the newer image left there.
	const auto same_layout = [&](const Image& other) {
		const auto& a = image.info;
		const auto& b = other.info;
		return a.resources == ImageSubresources {} && b.resources == ImageSubresources {} &&
		       a.data.address + a.mip_layout[0].offset == b.data.address + b.mip_layout[0].offset &&
		       a.tile_mode == b.tile_mode && a.bytes_per_block == b.bytes_per_block && a.type == b.type &&
		       a.mip_layout[0].pitch == b.mip_layout[0].pitch && a.GetColorTransform() == b.GetColorTransform() &&
		       vk::blockSize(image.backing.format) == vk::blockSize(other.backing.format);
	};
	const bool fast = std::ranges::all_of(newer, [&](ImageId other_id) { return same_layout(m_slot_images[other_id]); });
	// KYTY_FIX_SHARED_IMAGE_DIRECT: the in-place patch of the image's linear copy.
	const bool direct = !fast && SharedImageDirectEnabled() && TileManager::AliasCopyFits(image.info) &&
	                    !image.IsBufferModified() && !image.IsCpuDirty() &&
	                    vk::blockSize(image.backing.format) == image.info.bytes_per_block &&
	                    std::ranges::all_of(newer, [&](ImageId other_id) {
		                    const auto& other = m_slot_images[other_id];
		                    return TileManager::AliasCopyFits(other.info) &&
		                           vk::blockSize(other.backing.format) == other.info.bytes_per_block;
	                    });
	// What the merge through guest memory needs: the image's own bytes and every newer image downloadable.
	const bool own_gpu = image.IsGpuModified() && !image.IsBufferModified() && !image.IsCpuDirty();
	const auto mergeable = [&] {
		if (own_gpu && (!image.SafeToDownload() || !CanDownload(image))) return false;
		return std::ranges::all_of(newer, [&](ImageId other_id) {
			const auto& other = m_slot_images[other_id];
			return other.SafeToDownload() && CanDownload(other);
		});
	};
	if (const uint32_t probe = SharedImageMemoryMode(); probe >= 3u) {
		// The probe (=3/4): the bytes the merge would change (the image's own guest bytes against the merged ones), by
		// bind kind; a sampled bind then takes the merge, a storage or render-target bind gets the pattern instead (=3)
		// or nothing (=4) and its next binds look for it.
		using ProbeCounters = std::array<std::atomic<uint64_t>, 5>;
		static ProbeCounters events, changed, bytes;
		static std::atomic<uint32_t> window {0};
		if (!mergeable() || !CanDownload(image)) {
			s.refused.fetch_add(1, std::memory_order_relaxed);
			Report();
			return false;
		}
		m_scheduler.EndRendering();
		uint64_t lo = image.info.data.address;
		uint64_t hi = image.info.data.End();
		for (const auto other_id: newer) {
			lo = std::min(lo, m_slot_images[other_id].info.data.address);
			hi = std::max(hi, m_slot_images[other_id].info.data.End());
		}
		const uint64_t own = image.info.data.address - lo;
		const uint64_t sz  = image.info.data.size;
		if (!m_alias_scratch || m_alias_scratch->Size() < hi - lo) {
			if (m_alias_scratch) m_scheduler.DeferOperation([old = std::move(m_alias_scratch)]() mutable { old.reset(); });
			m_alias_scratch = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, hi - lo);
		}
		auto& scratch = *m_alias_scratch;
		if (own_gpu) {
			DownloadImage(image, scratch, own, sz, image.info.resources.levels);
		} else {
			const auto [source, source_offset] = m_buffer_cache.ObtainBufferForImage(image.info.data.address, sz);
			if (source == nullptr) {
				s.refused.fetch_add(1, std::memory_order_relaxed);
				return false;
			}
			scratch.CopyFrom(m_scheduler.Current(), *source, source_offset, own, sz);
		}
		auto before = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, sz);
		auto after  = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, sz);
		before->CopyFrom(m_scheduler.Current(), scratch, own, 0, sz);
		for (const auto other_id: newer) {
			auto& other = m_slot_images[other_id];
			DownloadImage(other, scratch, other.info.data.address - lo, other.info.data.size, other.info.resources.levels);
		}
		after->CopyFrom(m_scheduler.Current(), scratch, own, 0, sz);
		{
			auto              command = m_scheduler.Current().Handle();
			vk::MemoryBarrier host {};
			host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
			host.dstAccessMask = vk::AccessFlagBits::eHostRead;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {}, 1, &host,
			                        0, nullptr, 0, nullptr);
		}
		std::string what = fmt::format("image=0x{:x} {}x{} {} newer:", image.info.data.address, image.info.extent.width,
		                               image.info.extent.height, vk::to_string(image.backing.format));
		for (const auto other_id: newer) {
			const auto& other = m_slot_images[other_id];
			what += fmt::format(" 0x{:x} {}x{} {}", other.info.data.address, other.info.extent.width,
			                    other.info.extent.height, vk::to_string(other.backing.format));
		}
		const uint32_t frame = DebugPresentCount().load();
		m_scheduler.DeferOperation([before, after, bind, frame, what = std::move(what)] {
			before->Invalidate(0, before->Size());
			after->Invalidate(0, after->Size());
			const uint8_t* a    = before->Mapped().data();
			const uint8_t* b    = after->Mapped().data();
			uint64_t       diff = 0;
			for (uint64_t i = 0; i < before->Size(); ++i) diff += a[i] != b[i] ? 1u : 0u;
			events[bind].fetch_add(1, std::memory_order_relaxed);
			if (diff != 0u) {
				changed[bind].fetch_add(1, std::memory_order_relaxed);
				bytes[bind].fetch_add(diff, std::memory_order_relaxed);
			}
			static std::mutex                 mutex;
			static std::set<std::string>      seen;
			bool                              log = false;
			{
				std::scoped_lock lock(mutex);
				log = seen.size() < 64u && seen.insert(fmt::format("{} {} {}", bind, diff != 0u, what)).second;
			}
			if (log) {
				std::fprintf(stderr, "SHARED_IMAGE_PROBE stale frame=%u bind=%u changed_bytes=%llu of %llu %s\n", frame, bind,
				             static_cast<unsigned long long>(diff), static_cast<unsigned long long>(before->Size()),
				             what.c_str());
			}
			uint32_t start = window.load(std::memory_order_relaxed);
			if (start == 0u) {
				window.compare_exchange_strong(start, frame == 0u ? 1u : frame);
			} else if (frame >= start + 300u && window.compare_exchange_strong(start, frame)) {
				const auto x = [](std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.exchange(0)); };
				std::fprintf(stderr,
				             "PERF shared_image_stale frames=%u-%u events tex=%llu storage=%llu rt=%llu changed tex=%llu "
				             "storage=%llu rt=%llu MiB tex=%llu storage=%llu rt=%llu\n",
				             start, frame, x(events[0]), x(events[1]), x(events[2]), x(changed[0]), x(changed[1]),
				             x(changed[2]), x(bytes[0]) >> 20u, x(bytes[1]) >> 20u, x(bytes[2]) >> 20u);
			}
		});
		const bool floating = [&] {
			switch (image.backing.format) {
				case vk::Format::eR16Sfloat:
				case vk::Format::eR16G16Sfloat:
				case vk::Format::eR16G16B16A16Sfloat:
				case vk::Format::eR32Sfloat:
				case vk::Format::eR32G32Sfloat:
				case vk::Format::eR32G32B32A32Sfloat:
				case vk::Format::eB10G11R11UfloatPack32: return true;
				default: return false;
			}
		}();
		if (bind == 0u || bind == 4u) {
			UploadImage(image, scratch, own);
		} else {
			if (probe == 3u) {
				const uint64_t size    = image.info.LinearSize(1);
				auto&          pattern = m_probe_pattern[floating ? 1 : 0];
				if (!pattern || pattern->Size() < size) {
					if (pattern) m_scheduler.DeferOperation([old = std::move(pattern)]() mutable { old.reset(); });
					pattern = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0, AllFlags, size);
					std::memset(pattern->Mapped().data(), floating ? 0xFF : 0xA5, size);
					pattern->Flush(0, size);
				}
				image.Upload(image.info.BufferCopies(0, 1), pattern->Handle(), 0, size);
			}
			image.probe_pending = floating ? 2u : 1u;
			image.probe_checks  = 0;
		}
		image.ClearBufferModified();
		if (image.IsCpuDirty()) image.RefreshComplete();
		image.content_serial      = newest;
		image.alias_synced_serial = newest;
		s.general.fetch_add(1, std::memory_order_relaxed);
		Report();
		return true;
	}
	// =2: sampled checks.
	bool check = false;
	if (SharedImageMemoryMode() == 2u) {
		static std::map<std::pair<uint64_t, uint64_t>, uint64_t> seen;
		const auto n = seen[{image.info.data.address, m_slot_images[newer.back()].info.data.address}]++;
		check        = n < 4u || n % 64u == 0u;
		if (check && (!mergeable() || !CanDownload(image))) {
			check = false;
			s.check_skipped.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (!fast && !direct && !mergeable()) {
		s.refused.fetch_add(1, std::memory_order_relaxed);
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 16u) {
			std::fprintf(stderr, "FIX_SHARED_IMAGE_MEMORY refused frame=%u image=0x%llx %ux%u %s own_gpu=%d newer=%zu\n",
			             DebugPresentCount().load(), static_cast<unsigned long long>(image.info.data.address),
			             image.info.extent.width, image.info.extent.height,
			             vk::to_string(image.backing.format).c_str(), own_gpu ? 1 : 0, newer.size());
		}
		Report();
		return false;
	}
	m_scheduler.EndRendering();
	// The merged guest bytes, in a private scratch buffer (guest memory, the buffer cache and the tracker stay as they
	// are): [begin, end) spans the image and every newer image.
	uint64_t begin = image.info.data.address;
	uint64_t end   = image.info.data.End();
	for (const auto other_id: newer) {
		begin = std::min(begin, m_slot_images[other_id].info.data.address);
		end   = std::max(end, m_slot_images[other_id].info.data.End());
	}
	const uint64_t own_offset = image.info.data.address - begin;
	const auto     merge      = [&]() -> bool {
        const uint64_t size = end - begin;
        if (!m_alias_scratch || m_alias_scratch->Size() < size) {
            if (m_alias_scratch) m_scheduler.DeferOperation([old = std::move(m_alias_scratch)]() mutable { old.reset(); });
            m_alias_scratch = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
                                                       vk::BufferUsageFlagBits::eStorageBuffer |
                                                           vk::BufferUsageFlagBits::eTransferSrc |
                                                           vk::BufferUsageFlagBits::eTransferDst,
                                                       size);
        }
        auto& scratch = *m_alias_scratch;
        if (own_gpu) {
            DownloadImage(image, scratch, own_offset, image.info.data.size, image.info.resources.levels);
        } else {
            const auto [source, source_offset] =
                m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
            if (source == nullptr) return false;
            scratch.CopyFrom(m_scheduler.Current(), *source, source_offset, own_offset, image.info.data.size);
        }
        for (const auto other_id: newer) {
            auto& other = m_slot_images[other_id];
            DownloadImage(other, scratch, other.info.data.address - begin, other.info.data.size,
                          other.info.resources.levels);
        }
        return true;
	};
	std::shared_ptr<Buffer> expected, actual; // =2: the merged bytes, and the image's download over a copy of them
	if (check) {
		if (!merge()) {
			check = false;
			s.check_skipped.fetch_add(1, std::memory_order_relaxed);
		} else {
			const auto size = image.info.data.size;
			expected        = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, size);
			actual          = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, size);
			expected->CopyFrom(m_scheduler.Current(), *m_alias_scratch, own_offset, 0, size);
			actual->CopyFrom(m_scheduler.Current(), *m_alias_scratch, own_offset, 0, size);
		}
	}
	if (fast) {
		auto command = m_scheduler.Current().Handle();
		for (const auto other_id: newer) {
			auto& source = m_slot_images[other_id];
			source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
			image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
			vk::ImageCopy copy {};
			copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			copy.extent         = vk::Extent3D {std::min(source.backing.extent.width, image.backing.extent.width),
			                                    std::min(source.backing.extent.height, image.backing.extent.height), 1};
			command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.backing.image,
			                  vk::ImageLayout::eTransferDstOptimal, 1, &copy);
		}
		s.fast.fetch_add(1, std::memory_order_relaxed);
		s.bytes.fetch_add(image.info.data.size, std::memory_order_relaxed);
	} else if (direct) {
		const auto linear = [&](uint32_t slot, uint64_t size) -> Buffer& {
			auto& buffer = m_alias_linear[slot];
			if (!buffer || buffer->Size() < size) {
				if (buffer) m_scheduler.DeferOperation([old = std::move(buffer)]() mutable { old.reset(); });
				buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, size);
			}
			return *buffer;
		};
		const uint64_t target_size = image.info.LinearSize(1);
		auto&          target      = linear(0, target_size);
		image.Download(image.info.BufferCopies(0, 1), target.Handle(), 0, target_size);
		uint64_t bytes = 2u * target_size;
		for (const auto other_id: newer) {
			auto&          other       = m_slot_images[other_id];
			const uint64_t source_size = other.info.LinearSize(1);
			auto&          source      = linear(1, source_size);
			other.Download(other.info.BufferCopies(0, 1), source.Handle(), 0, source_size);
			m_tiler.AliasCopy(target.Handle(), target_size, image.info, source.Handle(), source_size, other.info);
			bytes += source_size;
		}
		image.Upload(image.info.BufferCopies(0, 1), target.Handle(), 0, target_size);
		s.direct.fetch_add(1, std::memory_order_relaxed);
		s.bytes.fetch_add(bytes, std::memory_order_relaxed);
	} else {
		if (!check && !merge()) {
			s.refused.fetch_add(1, std::memory_order_relaxed);
			Report();
			return false;
		}
		UploadImage(image, *m_alias_scratch, own_offset);
		// The host image holds the merged bytes: a later refresh must not load older guest memory over them.
		image.ClearBufferModified();
		if (image.IsCpuDirty()) image.RefreshComplete();
		uint64_t bytes = 2u * image.info.data.size;
		for (const auto other_id: newer) bytes += m_slot_images[other_id].info.data.size;
		s.general.fetch_add(1, std::memory_order_relaxed);
		s.bytes.fetch_add(bytes, std::memory_order_relaxed);
	}
	image.content_serial      = newest;
	image.alias_synced_serial = newest;
	if (check) {
		DownloadImage(image, *actual, 0, image.info.data.size, image.info.resources.levels);
		auto              command = m_scheduler.Current().Handle();
		vk::MemoryBarrier host {};
		host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
		host.dstAccessMask = vk::AccessFlagBits::eHostRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0,
		                        nullptr, 0, nullptr);
		std::string pair = fmt::format("image=0x{:x} {}x{} {} fast={} newer:", image.info.data.address,
		                               image.info.extent.width, image.info.extent.height,
		                               vk::to_string(image.backing.format), fast ? 1 : 0);
		for (const auto other_id: newer) {
			const auto& other = m_slot_images[other_id];
			pair += fmt::format(" 0x{:x} {}x{} {}", other.info.data.address, other.info.extent.width,
			                    other.info.extent.height, vk::to_string(other.backing.format));
		}
		const uint32_t frame = DebugPresentCount().load();
		m_scheduler.DeferOperation([expected, actual, pair = std::move(pair), frame] {
			const auto size = expected->Size();
			expected->Invalidate(0, size);
			actual->Invalidate(0, size);
			const uint8_t* a = expected->Mapped().data();
			const uint8_t* b = actual->Mapped().data();
			uint64_t       diff = 0, first = 0;
			for (uint64_t i = 0; i < size; ++i) {
				if (a[i] != b[i]) {
					if (diff == 0) first = i;
					++diff;
				}
			}
			g_stats.checks.fetch_add(1, std::memory_order_relaxed);
			if (diff == 0) return;
			g_stats.check_diff_events.fetch_add(1, std::memory_order_relaxed);
			g_stats.check_diff_bytes.fetch_add(diff, std::memory_order_relaxed);
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 16u) {
				std::fprintf(stderr, "FIX_SHARED_IMAGE_MEMORY check MISMATCH frame=%u bytes=%llu first=+0x%llx (%02x != %02x) %s\n",
				             frame, static_cast<unsigned long long>(diff), static_cast<unsigned long long>(first), a[first],
				             b[first], pair.c_str());
			}
		});
	}
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 24u) {
		std::fprintf(stderr, "FIX_SHARED_IMAGE_MEMORY sync frame=%u image=0x%llx %ux%u %s %s newer=%zu own_gpu=%d\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(image.info.data.address),
		             image.info.extent.width, image.info.extent.height, vk::to_string(image.backing.format).c_str(),
		             fast ? "fast" : "general", newer.size(), own_gpu ? 1 : 0);
	}
	Report();
	return true;
}

} // namespace Libs::Graphics
