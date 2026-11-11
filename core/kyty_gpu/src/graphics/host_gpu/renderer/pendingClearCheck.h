#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PENDINGCLEARCHECK_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PENDINGCLEARCHECK_H_

// KYTY_DIAG_PENDING_CLEAR_CHECK=1 (AnyPS5 lane v4x2; diagnostic, default off, read with getenv: not a shader variable,
// no pipeline changes). What a depth texture holds while a fast clear of its HTILE is pending.
//
// A fast clear (a compute shader or fill writing the HTILE) only sets a flag per slice (TextureCache::ClearMeta*); the
// host depth image is cleared at the surface's next bind as a depth target (renderDraw.cpp, load_clear_aspects), and
// the PS5-side HTILE expand draw that writes the clear value into the depth memory is left out
// (NoOutputDropReason). A texture bind in between samples the image as the last draws left it. This counts how far
// that is from the clear value, and whether the PS5 would see the clear at that bind:
//  - at the first texture bind of a slice with a pending clear, the slice's depth aspect is copied to host memory (one
//    render-pass end, one layout change and one copy; the image's contents and every later command are as without
//    the diagnostic);
//  - when the clear lands, the kept texels are compared with the clear value of that bind: one
//    "PENDING_CLEAR_CHECK episode" line with the texels that differ and the largest difference;
//  - every bind is classed by its texture descriptor: "htile" (MetaCompress set and the descriptor's metadata address
//    is this surface's HTILE: the PS5's texture unit reads through the HTILE and sees the fast clear at once),
//    "meta_other" (MetaCompress set with another address), "plain" (no MetaCompress, or a 128-bit descriptor: the
//    PS5 samples the raw depth memory, which a fast clear does not touch, so before an expand draw the old contents
//    are what the PS5 reads too). The first 6 binds per surface and class are logged (PENDING_CLEAR_CHECK bind);
//  - binds made for the very draw that lands the clear (the draw samples its own depth target; its textures are
//    resolved before its targets) are not stale reads: they are counted apart (same_draw_binds), and an episode with
//    only such binds is "same_draw_only". Binds since the previous depth-target bind count as that draw's when it
//    samples the surface, which can hide a stale read by a dispatch right before it;
//  - an expand draw left out before the first bind is noted with its clear value (after_expand, expand_clear,
//    differ_expand): from there on the PS5's depth memory holds the clear value whatever the descriptor says;
//  - how the surface's clear flag was last set (pendingClearNote.h: path, fill word, the depth clear register then);
//  - "PENDING_CLEAR_CHECK summary", "image" and "open" lines every 512 frames (totals, per depth image, and the episodes
//    whose clear has not landed for 600 frames, with the kept slice's minimum and maximum).
// Not covered: multisampled depth images, views of several layers (the base layer is checked, as lane v4x's
// KYTY_DIAG_PENDING_CLEAR_READS counts), slices above 64 MiB, more than 12 slices kept at once (counted as skipped).

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pendingClearNote.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::PendingClearCheck {

// A bind's texture descriptor: carries this surface's HTILE, carries other metadata, or plain.
inline constexpr uint32_t Classes = 3;
inline constexpr const char* ClassNames[Classes] = {"htile", "meta_other", "plain"};

struct Episode {
	uint64_t   id = 0, key = 0, depth_address = 0, htile = 0;
	uint32_t   slice = 0, level = 0, width = 0, height = 0, bytes = 0;
	vk::Format format      = vk::Format::eUndefined;
	uint32_t   first_frame = 0, last_frame = 0;
	uint64_t   binds_before[Classes] {}, binds_cur[Classes] {}, cur_serial = 0;
	bool       after_expand = false;
	float      expand_value = 0.0f;
	uint64_t   differ_expand = 0;
	int        slot  = -1;
	bool       ready = false, landed = false, self_sampled = false;
	float      landed_value = 0.0f, lo = 0.0f, hi = 0.0f;
	uint32_t   landed_frame = 0;
	uint64_t   stale_binds[Classes] {}, same_draw_binds = 0;
};

struct Slot {
	std::unique_ptr<Buffer> buffer;
	bool                    busy = false;
};

struct ImageTotals {
	uint32_t width = 0, height = 0;
	uint64_t stale_episodes = 0, same_draw_only = 0, texels = 0, differ = 0, episodes_with_differ = 0;
	uint64_t stale_binds[Classes] {};
	// By what the episode's stale binds were: at least one "htile" bind, else at least one "meta_other", else all plain.
	uint64_t episodes_by[Classes] {}, differ_by[Classes] {};
	double   max_diff = 0.0;
};

struct State {
	std::mutex                              mutex;
	std::unordered_map<uint64_t, uint64_t>  current;  // (htile, slice) -> the open episode's id
	std::unordered_map<uint64_t, Episode>   episodes; // by id: open ones, and landed ones whose copy is still on the GPU
	std::unordered_map<uint64_t, std::pair<float, uint32_t>> expands; // (htile, slice) -> clear value, frame of a left-out expand
	std::unordered_map<uint64_t, uint32_t>  bind_lines; // (htile, slice, class) -> PENDING_CLEAR_CHECK bind lines printed
	std::vector<Slot>                       slots;
	std::map<uint64_t, ImageTotals>         images; // by depth address
	uint64_t next_id = 1, depth_binds = 0, snapshots = 0, landings = 0, expands_left_out = 0;
	uint64_t pending_binds[Classes] {};
	uint64_t closed = 0, stale_episodes = 0, same_draw_only = 0, texels = 0, differ = 0, episodes_with_differ = 0;
	uint64_t skipped_msaa = 0, skipped_format = 0, skipped_size = 0, skipped_slots = 0;
	uint64_t episode_lines = 0;
	uint32_t summary_frame = 0;
};

// Never destroyed: its buffers must not outlive the device in a static destructor.
inline State& S() {
	static State* state = new State;
	return *state;
}

inline constexpr uint64_t MaxSliceBytes = 64ull << 20;
inline constexpr uint64_t MinSlotBytes  = 16ull << 20; // any slot holds a 2048x2048 slice: slots are interchangeable
inline constexpr size_t   MaxSlots      = 12;

[[nodiscard]] inline uint64_t Key(uint64_t htile, uint32_t slice) noexcept {
	return (htile << 12u) | (slice & 0xfffu);
}

// One texel of the copied depth aspect as a float, and whether it is the clear value in the format's own encoding.
struct Texel {
	float value;
	bool  equal;
};
[[nodiscard]] inline Texel ReadTexel(const uint8_t* data, uint64_t index, vk::Format transfer, float clear) noexcept {
	switch (transfer) {
		case vk::Format::eD16Unorm: {
			uint16_t v = 0;
			std::memcpy(&v, data + index * 2u, sizeof(v));
			const auto want = static_cast<uint16_t>(std::lround(std::clamp(clear, 0.0f, 1.0f) * 65535.0f));
			return {static_cast<float>(v) / 65535.0f, v == want};
		}
		case vk::Format::eX8D24UnormPack32: {
			uint32_t v = 0;
			std::memcpy(&v, data + index * 4u, sizeof(v));
			v &= 0x00ffffffu;
			const auto want = static_cast<uint32_t>(std::lround(std::clamp(clear, 0.0f, 1.0f) * 16777215.0f));
			return {static_cast<float>(v) / 16777215.0f, v == want};
		}
		default: {
			float v = 0.0f;
			std::memcpy(&v, data + index * 4u, sizeof(v));
			return {v, v == clear};
		}
	}
}

struct Counts {
	uint64_t differ   = 0;
	double   max_diff = 0.0;
	float    lo = 0.0f, hi = 0.0f;
};
[[nodiscard]] inline Counts Scan(const Episode& e, const uint8_t* data, float clear) noexcept {
	Counts         c;
	const auto     transfer = DepthAspectTransferFormat(e.format);
	const uint64_t texels   = uint64_t {e.width} * e.height;
	for (uint64_t i = 0; i < texels; i++) {
		const auto t = ReadTexel(data, i, transfer, clear);
		if (i == 0) c.lo = c.hi = t.value;
		c.lo = std::min(c.lo, t.value);
		c.hi = std::max(c.hi, t.value);
		if (!t.equal) {
			c.differ++;
			c.max_diff = std::max(c.max_diff, std::fabs(static_cast<double>(t.value) - static_cast<double>(clear)));
		}
	}
	return c;
}

// How the surface's clear flag was last set, as text for the episode and open lines.
struct SetText {
	char text[160];
};
[[nodiscard]] inline SetText DescribeSet(uint64_t htile) {
	SetText          out {};
	auto&            sets = Sets();
	std::scoped_lock lock(sets.mutex);
	const auto       found = sets.by_address.find(htile);
	if (found == sets.by_address.end()) {
		std::snprintf(out.text, sizeof(out.text), "set_path=none set_frame=0 set_counts=0/0/0 set_word=none set_register=none");
		return out;
	}
	const auto& set = found->second;
	char        word[24] = "none", reg[24] = "none";
	if (set.has_word) std::snprintf(word, sizeof(word), "0x%08x", set.word);
	if (set.has_register) std::snprintf(reg, sizeof(reg), "%g", set.depth_register);
	std::snprintf(out.text, sizeof(out.text), "set_path=%s set_frame=%u set_counts=%llu/%llu/%llu set_word=%s set_register=%s",
	              ClearPathNames[static_cast<uint32_t>(set.last_path)], set.frame,
	              static_cast<unsigned long long>(set.count[0]), static_cast<unsigned long long>(set.count[1]),
	              static_cast<unsigned long long>(set.count[2]), word, reg);
	return out;
}

// With the mutex held.
inline void PrintSummary(State& s) {
	const auto frame = DebugPresentCount().load();
	std::fprintf(stderr,
	             "PENDING_CLEAR_CHECK summary frame=%u closed=%llu stale_episodes=%llu same_draw_only=%llu texels=%llu "
	             "differ=%llu episodes_with_differ=%llu open=%zu pending_binds(htile/meta_other/plain)=%llu/%llu/%llu "
	             "snapshots=%llu landings=%llu expands_left_out=%llu skipped(msaa/format/size/slots)=%llu/%llu/%llu/%llu\n",
	             frame, static_cast<unsigned long long>(s.closed), static_cast<unsigned long long>(s.stale_episodes),
	             static_cast<unsigned long long>(s.same_draw_only), static_cast<unsigned long long>(s.texels),
	             static_cast<unsigned long long>(s.differ), static_cast<unsigned long long>(s.episodes_with_differ),
	             s.current.size(), static_cast<unsigned long long>(s.pending_binds[0]),
	             static_cast<unsigned long long>(s.pending_binds[1]), static_cast<unsigned long long>(s.pending_binds[2]),
	             static_cast<unsigned long long>(s.snapshots), static_cast<unsigned long long>(s.landings),
	             static_cast<unsigned long long>(s.expands_left_out), static_cast<unsigned long long>(s.skipped_msaa),
	             static_cast<unsigned long long>(s.skipped_format), static_cast<unsigned long long>(s.skipped_size),
	             static_cast<unsigned long long>(s.skipped_slots));
	for (const auto& [address, t]: s.images) {
		std::fprintf(stderr,
		             "PENDING_CLEAR_CHECK image depth=0x%llx %ux%u stale_episodes=%llu same_draw_only=%llu "
		             "stale_binds(htile/meta_other/plain)=%llu/%llu/%llu episodes(htile/meta_other/plain)=%llu/%llu/%llu "
		             "texels=%llu differ=%llu differ(htile/meta_other/plain)=%llu/%llu/%llu episodes_with_differ=%llu "
		             "max_diff=%g\n",
		             static_cast<unsigned long long>(address), t.width, t.height,
		             static_cast<unsigned long long>(t.stale_episodes), static_cast<unsigned long long>(t.same_draw_only),
		             static_cast<unsigned long long>(t.stale_binds[0]), static_cast<unsigned long long>(t.stale_binds[1]),
		             static_cast<unsigned long long>(t.stale_binds[2]), static_cast<unsigned long long>(t.episodes_by[0]),
		             static_cast<unsigned long long>(t.episodes_by[1]), static_cast<unsigned long long>(t.episodes_by[2]),
		             static_cast<unsigned long long>(t.texels), static_cast<unsigned long long>(t.differ),
		             static_cast<unsigned long long>(t.differ_by[0]), static_cast<unsigned long long>(t.differ_by[1]),
		             static_cast<unsigned long long>(t.differ_by[2]), static_cast<unsigned long long>(t.episodes_with_differ),
		             t.max_diff);
	}
	uint32_t printed = 0;
	for (const auto& [key, id]: s.current) {
		const auto found = s.episodes.find(id);
		if (found == s.episodes.end()) continue;
		const auto& e = found->second;
		if (frame - e.first_frame < 600u || printed++ >= 8u) continue;
		const auto set = DescribeSet(e.htile);
		std::fprintf(stderr,
		             "PENDING_CLEAR_CHECK open first_frame=%u age=%u depth=0x%llx htile=0x%llx slice=%u level=%u %ux%u "
		             "fmt=%s binds(htile/meta_other/plain)=%llu/%llu/%llu ready=%d min=%g max=%g after_expand=%d "
		             "expand_clear=%g differ_expand=%llu %s\n",
		             e.first_frame, frame - e.first_frame, static_cast<unsigned long long>(e.depth_address),
		             static_cast<unsigned long long>(e.htile), e.slice, e.level, e.width, e.height,
		             vk::to_string(e.format).c_str(), static_cast<unsigned long long>(e.binds_before[0] + e.binds_cur[0]),
		             static_cast<unsigned long long>(e.binds_before[1] + e.binds_cur[1]),
		             static_cast<unsigned long long>(e.binds_before[2] + e.binds_cur[2]), e.ready ? 1 : 0, e.lo, e.hi,
		             e.after_expand ? 1 : 0, e.expand_value, static_cast<unsigned long long>(e.differ_expand), set.text);
	}
}

// With the mutex held: the episode's copy is in host memory and its clear has landed.
inline void Close(State& s, uint64_t id) {
	const auto found = s.episodes.find(id);
	if (found == s.episodes.end()) return;
	const Episode  e      = found->second;
	auto&          slot   = s.slots[static_cast<size_t>(e.slot)];
	const auto     c      = Scan(e, slot.buffer->Mapped().data(), e.landed_value);
	const uint64_t texels = uint64_t {e.width} * e.height;
	const uint64_t binds  = e.stale_binds[0] + e.stale_binds[1] + e.stale_binds[2];
	const bool     stale  = binds != 0;
	auto&          image  = s.images[e.depth_address];
	image.width  = e.width;
	image.height = e.height;
	s.closed++;
	if (stale) {
		const uint32_t by = e.stale_binds[0] != 0 ? 0u : e.stale_binds[1] != 0 ? 1u : 2u;
		s.stale_episodes++;
		s.texels += texels;
		s.differ += c.differ;
		s.episodes_with_differ += c.differ != 0 ? 1u : 0u;
		image.stale_episodes++;
		for (uint32_t k = 0; k < Classes; k++) image.stale_binds[k] += e.stale_binds[k];
		image.episodes_by[by]++;
		image.differ_by[by] += c.differ;
		image.texels += texels;
		image.differ += c.differ;
		image.episodes_with_differ += c.differ != 0 ? 1u : 0u;
		image.max_diff = std::max(image.max_diff, c.max_diff);
	} else {
		s.same_draw_only++;
		image.same_draw_only++;
	}
	if ((stale && s.episode_lines < 300000u) || s.closed <= 32u) {
		const auto set = DescribeSet(e.htile);
		s.episode_lines++;
		std::fprintf(stderr,
		             "PENDING_CLEAR_CHECK episode frames=%u..%u depth=0x%llx htile=0x%llx slice=%u level=%u %ux%u fmt=%s "
		             "kind=%s stale_binds=%llu stale_htile=%llu stale_meta_other=%llu stale_plain=%llu same_draw_binds=%llu "
		             "clear=%g texels=%llu differ=%llu max_diff=%g min=%g max=%g after_expand=%d expand_clear=%g "
		             "differ_expand=%llu %s\n",
		             e.first_frame, e.landed_frame, static_cast<unsigned long long>(e.depth_address),
		             static_cast<unsigned long long>(e.htile), e.slice, e.level, e.width, e.height,
		             vk::to_string(e.format).c_str(), stale ? "stale" : "same_draw_only",
		             static_cast<unsigned long long>(binds), static_cast<unsigned long long>(e.stale_binds[0]),
		             static_cast<unsigned long long>(e.stale_binds[1]), static_cast<unsigned long long>(e.stale_binds[2]),
		             static_cast<unsigned long long>(e.same_draw_binds), e.landed_value,
		             static_cast<unsigned long long>(texels), static_cast<unsigned long long>(c.differ), c.max_diff, c.lo,
		             c.hi, e.after_expand ? 1 : 0, e.expand_value, static_cast<unsigned long long>(e.differ_expand),
		             set.text);
	}
	slot.busy = false;
	s.episodes.erase(found);
}

// A deferred operation of the command buffer the copy was recorded in: the copy is in host memory.
inline void OnReady(uint64_t id) {
	auto&            s = S();
	std::scoped_lock lock(s.mutex);
	const auto       found = s.episodes.find(id);
	if (found == s.episodes.end()) return;
	auto& e    = found->second;
	auto& slot = s.slots[static_cast<size_t>(e.slot)];
	slot.buffer->Invalidate(0, uint64_t {e.width} * e.height * e.bytes);
	const auto c    = Scan(e, slot.buffer->Mapped().data(), e.after_expand ? e.expand_value : 0.0f);
	e.lo            = c.lo;
	e.hi            = c.hi;
	e.differ_expand = e.after_expand ? c.differ : 0u;
	e.ready         = true;
	if (e.landed) Close(s, id);
}

// RenderExecutor::FindResolvedTexture, after the image is found (the render thread). `r128`, `meta_compress` and
// `meta_address` are the bind's texture descriptor: a 128-bit descriptor, its MetaCompress bit, its metadata address.
inline void OnTextureBind(GraphicContext& graphics, CommandScheduler& scheduler, TextureCache& cache, Image& image,
                          const TextureCache::ImageDesc& desc, bool storage, bool r128, bool meta_compress,
                          uint64_t meta_address) {
	if (storage || !image.info.IsDepth() || image.info.metadata.kind != ImageMetadataKind::Htile) return;
	const uint64_t htile = image.info.metadata.range.address;
	const uint32_t slice = desc.view_info.base_layer;
	if (!cache.IsMetaCleared(htile, slice)) return;
	auto&          s     = S();
	const auto     frame = DebugPresentCount().load();
	const uint64_t key   = Key(htile, slice);
	const uint32_t cls   = (!r128 && meta_compress) ? (meta_address == htile ? 0u : 1u) : 2u;
	Episode        e;
	vk::Buffer     staging;
	uint64_t       size   = 0;
	bool           create = false;
	{
		std::scoped_lock lock(s.mutex);
		s.pending_binds[cls]++;
		if (auto& lines = s.bind_lines[(key << 2u) | cls]; lines < 6u) {
			lines++;
			std::fprintf(stderr,
			             "PENDING_CLEAR_CHECK bind frame=%u depth=0x%llx htile=0x%llx slice=%u class=%s r128=%d "
			             "meta_compress=%d meta_address=0x%llx view_format=%s\n",
			             frame, static_cast<unsigned long long>(image.info.data.address),
			             static_cast<unsigned long long>(htile), slice, ClassNames[cls], r128 ? 1 : 0, meta_compress ? 1 : 0,
			             static_cast<unsigned long long>(meta_address), vk::to_string(desc.view_info.format).c_str());
		}
		if (const auto current = s.current.find(key); current != s.current.end()) {
			auto& open = s.episodes.at(current->second);
			if (open.cur_serial != s.depth_binds) {
				for (uint32_t k = 0; k < Classes; k++) {
					open.binds_before[k] += open.binds_cur[k];
					open.binds_cur[k] = 0;
				}
				open.cur_serial = s.depth_binds;
			}
			open.binds_cur[cls]++;
			open.last_frame = frame;
			return;
		}
		e.level  = desc.view_info.base_level;
		e.width  = std::max(1u, image.backing.extent.width >> e.level);
		e.height = std::max(1u, image.backing.extent.height >> e.level);
		e.format = image.backing.format;
		e.bytes  = DepthAspectTransferBytes(e.format);
		size     = uint64_t {e.width} * e.height * e.bytes;
		if (image.backing.samples != 1u) {
			s.skipped_msaa++;
			return;
		}
		if (e.bytes == 0u || e.level >= image.backing.mip_levels || slice >= image.backing.layers ||
		    image.backing.image_type != vk::ImageType::e2D) {
			s.skipped_format++;
			return;
		}
		if (size > MaxSliceBytes) {
			s.skipped_size++;
			return;
		}
		// A free slot whose buffer is large enough, else a new slot. Buffers are made outside the lock (below) and never
		// destroyed, so no buffer is created or freed while another thread waits for this state.
		for (size_t i = 0; i < s.slots.size(); i++) {
			if (!s.slots[i].busy && s.slots[i].buffer && s.slots[i].buffer->Size() >= size) {
				e.slot = static_cast<int>(i);
				break;
			}
		}
		if (e.slot < 0 && s.slots.size() < MaxSlots) {
			s.slots.emplace_back();
			e.slot = static_cast<int>(s.slots.size() - 1u);
			create = true;
		}
		if (e.slot < 0) {
			s.skipped_slots++;
			return;
		}
		s.slots[static_cast<size_t>(e.slot)].busy = true;
		e.id            = s.next_id++;
		e.key           = key;
		e.depth_address = image.info.data.address;
		e.htile         = htile;
		e.slice         = slice;
		e.first_frame = e.last_frame = frame;
		e.cur_serial                 = s.depth_binds;
		e.binds_cur[cls]             = 1;
		if (const auto expand = s.expands.find(key); expand != s.expands.end()) {
			e.after_expand = true;
			e.expand_value = expand->second.first;
		}
		s.current.emplace(key, e.id);
		s.episodes.emplace(e.id, e);
		s.snapshots++;
		if (s.snapshots <= 32u) {
			std::fprintf(stderr,
			             "PENDING_CLEAR_CHECK snapshot frame=%u depth=0x%llx htile=0x%llx slice=%u level=%u %ux%u fmt=%s "
			             "after_expand=%d class=%s\n",
			             frame, static_cast<unsigned long long>(e.depth_address), static_cast<unsigned long long>(htile),
			             slice, e.level, e.width, e.height, vk::to_string(e.format).c_str(), e.after_expand ? 1 : 0,
			             ClassNames[cls]);
		}
	}
	// The episode is open and its slot reserved; nothing reads the slot's buffer before OnReady, queued below.
	std::unique_ptr<Buffer> fresh;
	if (create) {
		fresh = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		                                 std::max(size, MinSlotBytes));
	}
	{
		std::scoped_lock lock(s.mutex);
		auto&            slot = s.slots[static_cast<size_t>(e.slot)];
		if (create) slot.buffer = std::move(fresh);
		staging = slot.buffer->Handle();
	}
	const std::array copies {vk::BufferImageCopy {
	    0, 0, 0, {vk::ImageAspectFlagBits::eDepth, e.level, slice, 1}, {}, vk::Extent3D {e.width, e.height, 1}}};
	image.Download(copies, staging, 0, size);
	scheduler.DeferOperation([id = e.id] { OnReady(id); });
}

// RenderExecutor::AcquireRenderTargets, for every depth target bound; `landing`: a pending clear of (htile, slice) is
// applied by this bind's load, with `clear_value`; `self_sampled`: the draw samples this depth image.
inline void OnDepthBind(uint64_t htile, uint32_t slice, bool landing, float clear_value, bool self_sampled) {
	auto&            s = S();
	std::scoped_lock lock(s.mutex);
	const uint64_t   serial = s.depth_binds++;
	const auto       frame  = DebugPresentCount().load();
	if (frame / 512u != s.summary_frame / 512u) {
		s.summary_frame = frame;
		PrintSummary(s);
	}
	if (!landing) return;
	const uint64_t key = Key(htile, slice);
	s.expands.erase(key);
	s.landings++;
	const auto current = s.current.find(key);
	if (current == s.current.end()) return;
	const uint64_t id = current->second;
	s.current.erase(current);
	auto& e        = s.episodes.at(id);
	e.landed       = true;
	e.landed_value = clear_value;
	e.landed_frame = frame;
	e.self_sampled = self_sampled;
	for (uint32_t k = 0; k < Classes; k++) {
		const uint64_t recent = (self_sampled && e.cur_serial == serial) ? e.binds_cur[k] : 0u;
		e.same_draw_binds += recent;
		e.stale_binds[k] = e.binds_before[k] + e.binds_cur[k] - recent;
	}
	if (e.ready) Close(s, id);
}

// ClassifyNoOutputDraw: an HTILE expand of a slice with a pending fast clear is left out; `clear_value` is the bound
// depth clear register's.
inline void OnExpandLeftOut(uint64_t htile, uint32_t slice, float clear_value) {
	auto&            s = S();
	std::scoped_lock lock(s.mutex);
	s.expands_left_out++;
	s.expands[Key(htile, slice)] = {clear_value, DebugPresentCount().load()};
}

} // namespace Libs::Graphics::PendingClearCheck

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PENDINGCLEARCHECK_H_ */
