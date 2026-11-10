#include "graphics/host_gpu/renderer/cache/textureCacheTrace.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// TextureCache's reports and logs (textureCacheTrace.h).
namespace Libs::Graphics {

// Logs (rate-limited) a colour-metadata slice that could not be read from the backing store.
void MetadataReadFailure(uint64_t address, uint64_t size) {
	static std::atomic_uint failures {0};
	if (const auto n = failures.fetch_add(1) + 1u; n <= 16u || (n % 1024u) == 0u) {
		std::fprintf(stderr, "META_READ_FAILED addr=0x%llx size=0x%llx count=%u (fast-clear check skipped)\n",
		             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size), n);
	}
}

// KYTY_FIX_DCC_KEY_0X10: one DCC_KEY_0X10 apply line per surface (address, extent, path), for the first 32 surfaces.
void LogDccKey10Apply(const char* path, const TextureCache::ImageDesc& desc, const vk::ClearColorValue& clear) {
	static std::mutex                                             mutex;
	static std::set<std::tuple<uint64_t, uint32_t, std::string>> logged;
	std::scoped_lock                                              lock {mutex};
	if (logged.size() >= 32u ||
	    !logged.emplace(desc.info.data.address, (desc.info.extent.width << 16u) | desc.info.extent.height, path)
	         .second) {
		return;
	}
	std::fprintf(stderr,
	             "DCC_KEY_0X10 apply frame=%u path=%s type=%u image=0x%llx %ux%u fmt=%s meta=0x%llx "
	             "color=%g,%g,%g,%g bits=%08x,%08x,%08x,%08x\n",
	             DebugPresentCount().load(), path, static_cast<uint32_t>(desc.type),
	             static_cast<unsigned long long>(desc.info.data.address), desc.info.extent.width,
	             desc.info.extent.height, vk::to_string(desc.view_info.format).c_str(),
	             static_cast<unsigned long long>(desc.info.metadata.range.address), clear.float32[0],
	             clear.float32[1], clear.float32[2], clear.float32[3], clear.uint32[0], clear.uint32[1], clear.uint32[2],
	             clear.uint32[3]);
}

std::mutex                                g_unregister_mutex; // KYTY_TRACE_PERF: PERF image_unregister
std::unordered_map<std::string, uint64_t> g_unregister_keys;
std::map<std::string, uint64_t>           g_unregister_reasons;

std::string TakeImageUnregisterStats() {
	std::vector<std::pair<uint64_t, std::string>> top;
	std::string                                   line = "reasons[";
	{
		std::lock_guard lock(g_unregister_mutex);
		for (const auto& [reason, n]: g_unregister_reasons) line += reason + "=" + std::to_string(n) + " ";
		for (const auto& [key, n]: g_unregister_keys) top.emplace_back(n, key);
		g_unregister_reasons.clear();
		g_unregister_keys.clear();
	}
	line += "] top[";
	std::sort(top.begin(), top.end(), std::greater<>());
	for (size_t i = 0; i < top.size() && i < 12u; ++i) {
		line += (i == 0 ? "" : " ") + top[i].second + "x" + std::to_string(top[i].first);
	}
	return line + "]";
}

namespace {

const char* DccGpuAllClassName(uint32_t value) {
	static const char* const names[] = {"no_resolver", "volume", "slices", "format", "depth_alias",
	                                    "no_backing",  "no_attachment", "layers", "init", "bind"};
	return value < std::size(names) ? names[value] : "?";
}

const char* BindingTypeName(TextureCache::BindingType type) {
	switch (type) {
		case TextureCache::BindingType::Texture: return "texture";
		case TextureCache::BindingType::Storage: return "storage";
		case TextureCache::BindingType::RenderTarget: return "render_target";
		case TextureCache::BindingType::DepthTarget: return "depth_target";
		case TextureCache::BindingType::VideoOut: return "video_out";
	}
	return "?";
}

} // namespace

void TextureCache::NoteDccGpuAllFallback(const ImageDesc& desc, DccGpuAllClass reason, uint32_t slices) {
	const auto index = static_cast<uint32_t>(reason);
	if (index < m_dcc_all.fallback.size()) m_dcc_all.fallback[index].fetch_add(1, std::memory_order_relaxed);
	// The first sighting of each (class, surface, binding type).
	static std::mutex                                            mutex;
	static std::set<std::tuple<uint32_t, uint64_t, uint32_t>>    seen;
	std::scoped_lock                                             lock(mutex);
	if (seen.size() >= 256u || !seen.insert({index, desc.info.data.address, static_cast<uint32_t>(desc.type)}).second) {
		return;
	}
	std::fprintf(stderr,
	             "DCC_GPU_ALL fallback=%s frame=%u image=0x%llx %ux%ux%u fmt=%s type=%s kind=%s slices=%u levels=%u "
	             "samples=%u volume=%d depth=%d meta=0x%llx size=0x%llx (keys read on the CPU)\n",
	             DccGpuAllClassName(index), DebugPresentCount().load(), static_cast<unsigned long long>(desc.info.data.address),
	             desc.info.extent.width, desc.info.extent.height, desc.info.extent.depth,
	             vk::to_string(desc.view_info.format).c_str(), BindingTypeName(desc.type),
	             desc.info.metadata.kind == ImageMetadataKind::Cmask ? "cmask" : "dcc", slices, desc.info.resources.levels,
	             static_cast<uint32_t>(desc.info.samples), desc.info.IsVolume() ? 1 : 0, desc.info.IsDepth() ? 1 : 0,
	             static_cast<unsigned long long>(desc.info.metadata.range.address),
	             static_cast<unsigned long long>(desc.info.metadata.range.size));
}

void TextureCache::VerifyDccGpuCheck(const DccClearResolver::VerifySlot& slot, uint64_t slice_size, uint32_t slices,
                                     uint32_t code_mask, uint64_t image_address, uint64_t keys_address) {
	const auto*     keys       = m_dcc_resolver->VerifyKeys(slot, slice_size * slices);
	const uint32_t* predicates = m_dcc_resolver->VerifyPredicates(slot, slices);
	for (uint32_t slice = 0; slice < slices; slice++) {
		// The CPU path's decision for these bytes (MaterializeColorClear): the first key byte, every byte equal to it, and
		// a code DecodeColorClear decodes for the view (code_mask holds exactly those of the five codes).
		const uint8_t* bytes   = keys + slice_size * slice;
		const uint8_t  code    = bytes[0];
		const bool     uniform = std::all_of(bytes, bytes + slice_size, [code](uint8_t byte) { return byte == code; });
		const uint32_t index   = DccClearResolver::CodeIndex(code);
		const bool     clear   = uniform && index < DccClearResolver::CodeCount && (code_mask & (1u << index)) != 0;
		bool           same    = true;
		for (uint32_t k = 0; k < DccClearResolver::CodeCount; k++) {
			const uint32_t expected = clear && k == index ? 1u : 0u;
			if (predicates[slice * DccClearResolver::CodeCount + k] != expected) same = false;
		}
		m_dcc_all.verified.fetch_add(1, std::memory_order_relaxed);
		if (same) continue;
		if (m_dcc_all.mismatches.fetch_add(1, std::memory_order_relaxed) < 16u) {
			const uint32_t* p = predicates + slice * DccClearResolver::CodeCount;
			std::fprintf(stderr,
			             "DCC_GPU_ALL_VERIFY_MISMATCH image=0x%llx keys=0x%llx slice=%u code=0x%02x uniform=%d mask=0x%x "
			             "cpu_clear=%d predicates=%u,%u,%u,%u,%u\n",
			             static_cast<unsigned long long>(image_address),
			             static_cast<unsigned long long>(keys_address + slice_size * slice), slice, code, uniform ? 1 : 0,
			             code_mask, clear ? 1 : 0, p[0], p[1], p[2], p[3], p[4]);
		}
	}
	m_dcc_resolver->ReleaseVerify(slot.end);
}


void TextureCache::ReportDccGpuAllPerf() {
	const int mode = DccGpuAllMode();
	if (mode == 0) return;
	auto&      s    = m_dcc_all;
	const auto take = [](std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.exchange(0)); };
	char       classes[512];
	size_t     used = 0;
	classes[0]      = '\0';
	for (uint32_t i = 0; i < s.fallback.size(); i++) {
		const auto n = s.fallback[i].exchange(0);
		if (n == 0) continue;
		const int written = std::snprintf(classes + used, sizeof(classes) - used, "%s%s=%llu", used == 0 ? "" : " ",
		                                  DccGpuAllClassName(i), static_cast<unsigned long long>(n));
		if (written > 0) used = std::min(sizeof(classes) - 1u, used + static_cast<size_t>(written));
	}
	// checks / slices / chunks recorded on the GPU; no_code: views no key clears (nothing read); already: slices checked
	// since their last write; cmask / msaa: checks of those kinds; fallback[...]: still read on the CPU (by class);
	// verify: slices compared / mismatches (must stay 0) / checks not copied (ring full).
	std::fprintf(stderr,
	             "PERF dcc_gpu_all mode=%d checks=%llu slices=%llu chunks=%llu no_code=%llu already=%llu cmask=%llu msaa=%llu "
	             "fallback[%s] verify(verified=%llu mismatches=%llu skipped=%llu)\n",
	             mode, take(s.checks), take(s.slices), take(s.chunks), take(s.no_code), take(s.already), take(s.cmask),
	             take(s.msaa), classes, take(s.verified), take(s.mismatches), take(s.verify_skipped));
}

void TextureCache::DebugDescribeRegion(const char* tag, uint64_t address, uint64_t size) const {
	const auto ids = FindImagesInRegion(address, size, true);
	std::fprintf(stderr, "SURFACE %s addr=0x%llx size=0x%llx images=%u\n", tag,
	             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
	             static_cast<uint32_t>(ids.size()));
	for (const ImageId id: ids) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr) continue;
		const auto& info = image->info;
		std::fprintf(stderr,
		             "  image id=%u addr=0x%llx size=0x%llx fmt=%u vk=%d extent=%ux%ux%u tile=%u "
		             "samples=%u gpu_modified=%d registered=%d\n",
		             static_cast<uint32_t>(id.index), static_cast<unsigned long long>(info.data.address),
		             static_cast<unsigned long long>(info.data.size), static_cast<uint32_t>(info.guest_format),
		             static_cast<int>(info.pixel_format), info.extent.width, info.extent.height,
		             info.extent.depth, static_cast<uint32_t>(info.tile_mode), info.samples,
		             image->IsGpuModified() ? 1 : 0, image->registered ? 1 : 0);
		std::fprintf(stderr, "    cpu_dirty=%d depth_id=%d\n", image->IsCpuDirty() ? 1 : 0, image->depth_id ? 1 : 0);
	}
	std::fflush(stderr);
}

std::vector<Image*> TextureCache::DebugImagesAt(uint64_t address) {
	std::vector<Image*> result;
	for (const ImageId id: FindImagesInRegion(address, 1, true)) {
		auto* image = m_slot_images.try_get(id);
		if (image != nullptr && image->info.data.address == address) result.push_back(image);
	}
	return result;
}

void TextureCache::DescribeImagesInRegion(uint64_t address, uint64_t size, uint32_t max_lines) {
	std::scoped_lock lock {m_lock};
	const auto ids = FindImagesInRegion(address, size, false);
	std::fprintf(stderr, "PERF_DIRECT_RANGES images in 0x%llx+0x%llx: %zu\n", static_cast<unsigned long long>(address),
	             static_cast<unsigned long long>(size), ids.size());
	uint32_t lines = 0;
	for (const auto id: ids) {
		if (lines++ >= max_lines) break;
		const auto& image = m_slot_images[id];
		const auto& i     = image.info;
		std::fprintf(stderr,
		             "PERF_DIRECT_RANGES image gpu_modified=%d cpu_dirty=%d buffer_modified=%d addr=0x%llx size=0x%llx "
		             "%ux%ux%u levels=%u layers=%u pixel_format=%u guest_format=%u type=%u tile_mode=%u\n",
		             image.IsGpuModified() ? 1 : 0, image.IsCpuDirty() ? 1 : 0, image.IsBufferModified() ? 1 : 0,
		             static_cast<unsigned long long>(i.data.address), static_cast<unsigned long long>(i.data.size),
		             i.extent.width, i.extent.height, i.extent.depth, i.resources.levels, i.resources.layers,
		             static_cast<uint32_t>(i.pixel_format), static_cast<uint32_t>(i.guest_format),
		             static_cast<uint32_t>(i.type), static_cast<uint32_t>(i.tile_mode));
	}
}

void TextureCache::NoteImageReadPlain(uint64_t bytes) {
	ReadStats().plain.fetch_add(bytes, std::memory_order_relaxed);
}

void TextureCache::NoteImageReadWait(uint64_t nanos) {
	ReadStats().wait_ns.fetch_add(nanos, std::memory_order_relaxed);
}

void TextureCache::NoteImageReadLift(uint64_t address, bool emulator_thread) {
	auto& stats = ReadStats();
	if (!emulator_thread) {
		stats.writes.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto count = stats.emu_lift.fetch_add(1, std::memory_order_relaxed);
	if (count >= 64u && count % 256u != 0u) {
		return;
	}
	std::fprintf(stderr, "IMAGE_READ_EMU_LIFT n=%llu frame=%u addr=0x%llx rip=0x%llx\n", static_cast<unsigned long long>(count + 1),
	             DebugPresentCount().load(), static_cast<unsigned long long>(address),
	             static_cast<unsigned long long>(CurrentFaultSite().rip));
}

void TextureCache::ReportImageReadServe() {
	static uint32_t calls = 0;
	if (!ImageReadTrace() || ++calls % 64u != 0u) {
		return;
	}
	std::scoped_lock lock {m_lock};
	auto&            s = ReadStats();
	const auto  n = [](const std::atomic<uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
	std::fprintf(stderr,
	             "IMAGE_READ_SERVE frame=%u serves=%llu images=%llu bytes=%llu record_ms=%llu wait_ms=%llu late=%llu orphan=%llu "
	             "emu_lift=%llu writes=%llu transferred=%llu detached=%llu freed=%llu plain_bytes=%llu",
	             DebugPresentCount().load(), n(s.serves), n(s.images), n(s.bytes), n(s.record_ns) / 1000000u, n(s.wait_ns) / 1000000u,
	             n(s.late), n(s.orphan), n(s.emu_lift), n(s.writes), n(s.transferred), n(s.detached), n(s.freed), n(s.plain));
	// The images that took the most bytes in the last 64 frames.
	std::vector<std::pair<uint64_t, ImageReadStats::Served>> top(s.served.begin(), s.served.end());
	std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
	for (size_t i = 0; i < top.size() && i < 8u; ++i) {
		const auto& served = top[i].second;
		std::fprintf(stderr, " [0x%llx %ux%u fmt=%u n=%u mb=%llu]", static_cast<unsigned long long>(top[i].first), served.width,
		             served.height, served.format, served.count, static_cast<unsigned long long>(served.bytes >> 20u));
	}
	std::fprintf(stderr, "\n");
	s.served.clear();
}


// At the first event of a new frame: one line with the previous frame's counts (first 300 frames with any) and a
// running total every 1000 frames.
void SliceStatsTick() {
	auto&          s     = SliceStats();
	const uint32_t frame = DebugPresentCount().load();
	const uint32_t prev  = s.frame.exchange(frame, std::memory_order_relaxed);
	if (prev == frame || prev == UINT32_MAX) return;
	const uint64_t now[6] = {s.clears.load(), s.base_partial.load(), s.layers.load(),
	                         s.partial.load(), s.consumed.load(),     s.diffs.load()};
	std::scoped_lock lock {s.report_lock};
	uint64_t         d[6];
	bool             any = false;
	for (int i = 0; i < 6; i++) {
		d[i]      = now[i] - s.last[i];
		s.last[i] = now[i];
		any |= d[i] != 0;
	}
	if (any && s.frame_lines < 300u) {
		s.frame_lines++;
		std::fprintf(stderr,
		             "HTILE_SLICE_CLEAR_FRAME frame=%u clears=%llu base_partial=%llu layers=%llu partial=%llu consumed=%llu "
		             "diffs=%llu\n",
		             prev, static_cast<unsigned long long>(d[0]), static_cast<unsigned long long>(d[1]),
		             static_cast<unsigned long long>(d[2]), static_cast<unsigned long long>(d[3]),
		             static_cast<unsigned long long>(d[4]), static_cast<unsigned long long>(d[5]));
	}
	if (frame / 1000u != prev / 1000u) {
		std::fprintf(stderr,
		             "HTILE_SLICE_CLEAR_TOTAL frame=%u clears=%llu base_partial=%llu layers=%llu partial=%llu consumed=%llu "
		             "checked=%llu diffs=%llu guard_inside=%llu\n",
		             frame, static_cast<unsigned long long>(now[0]), static_cast<unsigned long long>(now[1]),
		             static_cast<unsigned long long>(now[2]), static_cast<unsigned long long>(now[3]),
		             static_cast<unsigned long long>(now[4]), static_cast<unsigned long long>(s.checked.load()),
		             static_cast<unsigned long long>(now[5]), static_cast<unsigned long long>(s.guard_inside.load()));
	}
}

void TextureCache::TakeCensus(Census& census) {
	std::scoped_lock lock {m_lock};
	census = {};
	// Registered images the LRU has not seen for this many collector ticks (frames): the oldest age the
	// collector waits for before it frees an image.
	constexpr uint64_t LruOldTicks = 160;
	std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> by_address; // {images, bytes}
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		(void)id;
		const uint64_t bytes = m_graphics.GetAllocationSize(image.backing.allocation);
		census.all.Add(bytes);
		if (image.depth_id) {
			census.associations++;
			return;
		}
		if (!image.registered) {
			census.unregistered.Add(bytes);
			return;
		}
		census.registered.Add(bytes);
		if (image.bindless_pinned) {
			census.pinned.Add(bytes);
		}
		if (image.IsGpuModified()) {
			(image.info.IsTiled() ? census.gpu_tiled : census.gpu_linear).Add(bytes);
			if (image.info.HasMetadata()) {
				census.gpu_metadata.Add(bytes);
			}
		}
		if (image.info.IsDepth()) {
			census.depth.Add(bytes);
		}
		if (image.usage.render_target) {
			census.render_target.Add(bytes);
		}
		if (m_gc_tick >= LruOldTicks && m_lru_cache.GetTick(image.lru_id) + LruOldTicks <= m_gc_tick) {
			census.lru_old.Add(bytes);
		}
		if (!image.info.data.Empty()) {
			auto& entry = by_address[image.info.data.address];
			entry.first++;
			entry.second += bytes;
		}
	});
	for (const auto& [address, entry]: by_address) {
		(void)address;
		if (entry.first < 2u) {
			continue;
		}
		census.same_address_groups++;
		census.same_address.count += entry.first;
		census.same_address.bytes += entry.second;
	}
	const auto add_scratch = [&census](const std::unique_ptr<Buffer>& buffer) {
		if (buffer) {
			census.scratch_bytes += buffer->Size();
		}
	};
	add_scratch(m_saved_stencil);
	add_scratch(m_alias_scratch);
	add_scratch(m_stencil_sync_staging);
	for (const auto& [key, buffer]: m_stencil_planes) {
		(void)key;
		add_scratch(buffer);
	}
	census.pending_evictions = m_pending_evictions.size();
	census.budget_target     = m_budget_target;
	census.budget_used       = m_budget_used;
	census.recently_freed    = RecentlyFreedBytes();
	census.gc                = m_gc_stats;
	m_gc_stats               = {};
}

} // namespace Libs::Graphics
