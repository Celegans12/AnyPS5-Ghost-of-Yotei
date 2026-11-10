#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/cache/textureCacheShared.h"
#include "graphics/host_gpu/renderer/cache/textureCacheTrace.h"

#include "common/envDefaults.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/startupTrace.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "common/liveAb.h"
#include "common/lookupMemo.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <array>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <span>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>

#include "graphics/host_gpu/renderer/perfCounters.h"

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Largest GPU-written tiled image (in texels) that is written back to guest memory on submit.
constexpr uint64_t SmallTiledReadbackTexels = 64 * 64;

// Other threads touch the LRU list too (TouchImage runs
// outside m_lock), so its insertions, erasures and touches take this lock (the collector runs with the worker idle).
std::atomic_flag g_lru_lock = ATOMIC_FLAG_INIT;
class LruGuard {
public:
	LruGuard() {
		while (g_lru_lock.test_and_set(std::memory_order_acquire)) {
			while (g_lru_lock.test(std::memory_order_relaxed)) SpinPause();
		}
	}
	~LruGuard() { g_lru_lock.clear(std::memory_order_release); }
	LruGuard(const LruGuard&)            = delete;
	LruGuard& operator=(const LruGuard&) = delete;
};

// Image::plan_uses: counted with atomics.
uint16_t PlanUsesLoad(uint16_t& uses) {
	return std::atomic_ref<uint16_t>(uses).load(std::memory_order_relaxed);
}

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.info.IsDepth()
	                           ? TextureGetSurfaceFormatInfo(desc.info.guest_format).vk_format
	                           : desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodePackedColorClear(format, metadata.clear_word, clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

// Names the cache operation that unregisters images for TextureCache::on_bindless_unregister (the
// bindless table logs it); the innermost scope wins and the previous name returns on exit.
class UnregisterReason {
public:
	UnregisterReason(const char*& current, const char* reason): m_current(current), m_previous(current) {
		m_current = reason;
	}
	~UnregisterReason() { m_current = m_previous; }
	UnregisterReason(const UnregisterReason&)            = delete;
	UnregisterReason& operator=(const UnregisterReason&) = delete;

private:
	const char*& m_current;
	const char*  m_previous;
};

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()),
      m_readback_tiled_images(Config::ReadbackTiledImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
	if (DccGpuClearCheckLevel() > 0 || DccGpuAllMode() != 0) {
		m_dcc_resolver = std::make_unique<DccClearResolver>(graphics, scheduler, DccGpuAllMode() == 2);
		if (DccGpuClearCheckLevel() > 0) {
			std::fprintf(stderr, "FIX_DCC_GPU_CLEAR_CHECK level=%u conditional_rendering=%s\n", DccGpuClearCheckLevel(),
			             m_dcc_resolver->Available() ? "yes" : "no (CPU path)");
		}
		if (DccGpuAllMode() != 0) {
			std::fprintf(stderr, "PERF_DCC_GPU_ALL mode=%d conditional_rendering=%s\n", DccGpuAllMode(),
			             m_dcc_resolver->Available() ? "yes" : "no (every check falls back to the CPU read)");
		}
	}
	// The observer forgets checked slices whose keys are written again (OnDccMetaWrite): a check skipped as already
	// done is exact only with it (KYTY_PERF_DCC_GPU_ALL needs it even with KYTY_FIX_DCC_KNOWN_KEYS off).
	// KYTY_FIX_DCC_KEY_0X10: GPU writes over a color surface after its clear-to-pixel store end that store's entry.
	if (DccKnownKeysEnabled() || DccGpuClearCheckLevel() > 0 || DccGpuAllMode() != 0 || DccKey10FixEnabled()) {
		m_buffer_cache.SetGpuWriteObserver([this](uint64_t address, uint64_t size) {
			OnDccMetaWrite(address, size);
			const bool fills  = DccKnownKeysEnabled() && m_dcc_fill_count.load(std::memory_order_relaxed) != 0;
			const bool pixels = m_dcc_pixel_fill_count.load(std::memory_order_relaxed) != 0;
			if (!fills && !pixels) return;
			std::scoped_lock lock {m_dcc_pending_mutex};
			if (m_dcc_pending.size() < (1u << 16u)) m_dcc_pending.emplace_back(address, size);
		});
	}
}

TextureCache::~TextureCache() {
	UnregisterReason reason(m_unregister_reason, "shutdown");
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

// KYTY_TRACE_EVICT_REUSE=1 (diagnostic): images created again soon after the collector or the budget freed them
// (EVICT_REUSE, first 48), GPU-written images freed without their contents (EVICT_GPU_FREED, first 48), and both
// counts in the TEXGC summary.
static bool EvictReuseTrace() {
	static const bool on = std::getenv("KYTY_TRACE_EVICT_REUSE") != nullptr;
	return on;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	Common::StartupTrace::Scope startup_scope(Common::StartupTrace::Counter::ImageCreate); // KYTY_TRACE_STARTUP
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	m_slot_images[id].self_id = id; // bindless table refresh events (the bindless layout log)
	if (Common::StartupTrace::Enabled()) startup_scope.SetBytes(m_slot_images[id].AccountedSize());
	if (EvictReuseTrace() && !info.data.Empty()) {
		// KYTY_TRACE_EVICT_REUSE: an image created where the collector or the budget freed one 8 or fewer ticks ago.
		if (const auto found = m_evicted_at.find(info.data.address);
		    found != m_evicted_at.end() && m_gc_tick - found->second <= 8u) {
			// gpu_mod: the buffer cache holds GPU-written bytes of the image's memory (its re-upload must not take the
			// stale guest copy).
			const bool gpu_mod = m_buffer_cache.IsRegionGpuModified(info.data.address, info.data.size);
			m_evict_reused_gpu += gpu_mod ? 1u : 0u;
			if (m_evict_reused++ < 48u || (gpu_mod && m_evict_reused_gpu <= 48u))
				std::fprintf(stderr, "EVICT_REUSE addr=0x%llx %ux%u fmt=%u size=%llu after_ticks=%llu gpu_mod=%d\n",
				             static_cast<unsigned long long>(info.data.address), info.extent.width, info.extent.height,
				             static_cast<uint32_t>(info.pixel_format), static_cast<unsigned long long>(info.data.size),
				             static_cast<unsigned long long>(m_gc_tick - found->second), gpu_mod ? 1 : 0);
		}
	}
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
}

namespace {
// common/lookupMemo.h: the registered images whose first byte lies in a 64 KiB granule, hashed
// into the table (a collision only keeps a count above 0). Changed with the registration, under m_lock; read without it.
constexpr uint32_t    kImageStartSlots = 1u << 16u;
std::atomic<uint32_t> g_image_starts[kImageStartSlots] {};
std::atomic<uint32_t>& ImageStartCount(uint64_t address) {
	return g_image_starts[(address >> 16u) & (kImageStartSlots - 1u)];
}
} // namespace

TextureCache::RegisteredPageCounts::~RegisteredPageCounts() {
	for (auto& leaf: m_leaves) {
		delete leaf.load(std::memory_order_relaxed);
	}
}

void TextureCache::RegisteredPageCounts::Add(size_t page, int32_t delta) {
	auto& slot = m_leaves[page >> kLeafBits];
	auto* leaf = slot.load(std::memory_order_relaxed);
	if (leaf == nullptr) {
		leaf = new Leaf {};
		slot.store(leaf, std::memory_order_release);
	}
	(*leaf)[page & (kLeafSize - 1)].fetch_add(static_cast<uint32_t>(delta), std::memory_order_release);
}

bool TextureCache::RegisteredPageCounts::Any(size_t first, size_t last_exclusive) const noexcept {
	for (auto page = first; page < last_exclusive; page++) {
		const auto* leaf = m_leaves[page >> kLeafBits].load(std::memory_order_acquire);
		if (leaf != nullptr && (*leaf)[page & (kLeafSize - 1)].load(std::memory_order_acquire) != 0) {
			return true;
		}
	}
	return false;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_registered_pages.Add(page, 1);
		m_image_page_table[page].push_back(id);
	});
	ImageStartCount(image.info.data.address).fetch_add(1, std::memory_order_release); // common/lookupMemo.h
	image.registered = true;
	DrawRepeat::NoteImageRegistryChange(image.info.data.address, image.info.data.size); // the registry epoch (drawRepeat.h)
	{
		const LruGuard lru_guard;
		image.lru_id = m_lru_cache.Insert(id, m_gc_tick);
	}
	// Insert gave the item the current tick: a touch at it is a no-op (TouchImage).
	image.lru_touch_tick = m_gc_tick;
	m_total_used_memory += image.AccountedSize();
	image.registered_bytes = image.AccountedSize();
	m_registered_bytes += image.registered_bytes;
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (OwnerMode() != 0u) OwnerUnregister(id); // KYTY_FIX_IMAGE_MEMORY_OWNER
	DisarmImageRead(image);
	{
		const char* why = m_unregister_reason != nullptr ? m_unregister_reason : "other";
		char        key[160];
		std::snprintf(key, sizeof(key), "%s/%s/%ux%ux%u/l%u/m%u@%llx", why, vk::to_string(image.info.pixel_format).c_str(),
		              image.info.extent.width, image.info.extent.height, image.info.extent.depth,
		              image.info.resources.layers, image.info.resources.levels,
		              static_cast<unsigned long long>(image.info.data.address));
		std::lock_guard lock(g_unregister_mutex);
		g_unregister_reasons[why]++;
		g_unregister_keys[key]++;
	}
	if (image.bindless_pinned) {
		image.bindless_pinned = false;
		if (on_bindless_unregister)
			on_bindless_unregister(id, m_unregister_reason != nullptr ? m_unregister_reason : "other");
	}
	image.HeapChanged();
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_registered_pages.Add(page, -1);
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	});
	{
		const LruGuard lru_guard;
		m_lru_cache.Free(image.lru_id);
	}
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	m_registered_bytes -= std::min(m_registered_bytes, image.registered_bytes);
	image.registered_bytes = 0;
	image.registered = false;
	// The count it added at its registration (common/lookupMemo.h; its address cannot change while registered).
	if (ImageStartCount(image.info.data.address).fetch_sub(1, std::memory_order_release) == 0u) {
		ImageStartCount(image.info.data.address).fetch_add(1, std::memory_order_release);
		std::fprintf(stderr, "IMAGE_START_UNDERFLOW addr=0x%llx\n", static_cast<unsigned long long>(image.info.data.address));
	}
	DrawRepeat::NoteImageRegistryChange(image.info.data.address, image.info.data.size); // the registry epoch (drawRepeat.h)
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	if (image->depth_id) {
		std::erase(m_slot_images[image->depth_id].stencil_ids, id);
	} else {
		// Detach first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image->stencil_ids, {})) {
			FreeImage(association);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	// Our MetaDataInfo keeps its type (KYTY_FIX_HTILE_SLICE_CLEAR), so only an HTile registration is erased here;
	// upstream's bitset entry has no type (279bc9594, d8c4a9af3 + revert 2c730e58e) and erases by the image's kind.
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_images.erase(id); });
	} else {
		m_slot_images.erase(id);
	}
}

void TextureCache::FreeImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.IsGpuModified()) {
		// KYTY_TRACE_EVICT_REUSE: a GPU-written image freed without its contents (overlap resolution, expansion...).
		if (EvictReuseTrace() && m_evict_gpu_freed++ < 48u)
			std::fprintf(stderr, "EVICT_GPU_FREED addr=0x%llx %ux%u fmt=%u size=%llu reason=%s\n",
			             static_cast<unsigned long long>(image.info.data.address), image.info.extent.width,
			             image.info.extent.height, static_cast<uint32_t>(image.info.pixel_format),
			             static_cast<unsigned long long>(image.info.data.size),
			             m_unregister_reason != nullptr ? m_unregister_reason : "other");
		if (image.read_armed) DetachImageRead(id); // its bytes are not in guest memory yet
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	if (image.registered) {
		// m_gc_tick only rises (RunGarbageCollector) and an LRU item's tick never drops, so an image
		// touched (or registered) at this tick already holds it: Touch would return at its first check,
		// after a cache-missing deque lookup.
		if (image.lru_touch_tick == m_gc_tick) return;
		const LruGuard lru_guard;
		m_lru_cache.Touch(image.lru_id, m_gc_tick);
		image.lru_touch_tick = m_gc_tick;
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	image.HeapChanged();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::ArmWrittenImage(Image& image) {
	// Small tiled targets are read back too: games read GPU results such as reductions and
	// simulation state from them on the CPU (PPSA04263 keeps the origin of its ocean wave grid
	// in a 1x1 R32G32 target, and its water physics read it to sample the wave heights).
	const auto& info   = image.info;
	const auto  texels = static_cast<uint64_t>(info.extent.width) * info.extent.height *
	                     info.extent.depth * info.resources.layers;
	if (!info.data.Empty() && (info.IsTiled() ? m_readback_tiled_images && texels <= SmallTiledReadbackTexels
	                                          : m_readback_linear_images)) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot arm the read watch of a non-GPU-owned image\n");
		}
		ArmImageRead(image);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	return result;
}

static bool TraceImageSyncAddr(uint64_t address);

// KYTY_WORKAROUND_ALIAS_SYNC=1: logged APPROXIMATION for dynamic resolution. A guest surface used with
// several extents (or same-size formats) at one address is cached as separate images; before one
// of them is used, the overlapping top-left region is copied from the alias written most recently.
static bool AliasSyncEnabled() {
	static constinit Common::OnceValue<bool> enabled_once;
	const bool enabled = enabled_once.Get([] {
		// Default from Common::EnvDefaults; workaround: D4/D5 (shared image
		// memory), to be replaced by the proper implementation; =0 turns it off. With KYTY_FIX_IMAGE_MEMORY_OWNER on it
		// acts only on the images that map does not keep coherent (AliasSyncSkipsImage), counted there.
		const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_ALIAS_SYNC");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) std::fprintf(stderr, "WORKAROUND_ALIAS_SYNC APPROXIMATION enabled\n");
		return on;
	});
	return enabled;
}

// SyncExtentAliases' early return for the image itself (with KYTY_FIX_IMAGE_MEMORY_OWNER on, also every image that map
// keeps coherent).
static bool AliasSyncSkipsImage(const Image& image) {
	return image.info.data.Empty() || image.info.IsDepth() || image.backing.samples != 1 ||
	       image.backing.image_type != vk::ImageType::e2D || vk::isCompressed(image.backing.format) ||
	       (ImageMemoryOwnerMode() != 0u && ImageMemoryOwners::Followed(image));
}

// Whether SyncExtentAliases may copy into `image` from `other` (when other's content serial is higher).
static bool AliasSyncSource(const Image& image, const Image& other) {
	return other.info.data.address == image.info.data.address && !other.info.IsDepth() && other.backing.samples == 1 &&
	       other.backing.image_type == vk::ImageType::e2D && !vk::isCompressed(other.backing.format) &&
	       vk::blockSize(other.backing.format) == vk::blockSize(image.backing.format);
}

static uint64_t ImagePageEpochAt(uint64_t address);

// 1 + the exact image-page epoch sum over the image's memory: changes with every registration or unregistration of an
// image overlapping it (DrawRepeat::NoteImagePagesChange), so an overlap list kept with it is current while it holds.
static uint64_t SharedImageEpoch(const Image& image) {
	return 1u + DrawRepeat::ImagePageEpochSumExact(image.info.data.address, image.info.data.size);
}

// KYTY_FIX_SHARED_IMAGE_MEMORY: known to have no other image over its memory (a sync would do nothing).
static bool SharedImageAlone(const Image& image) {
	return image.shared_overlap_epoch == SharedImageEpoch(image) && image.shared_overlaps.empty();
}

// A pending rebind of an image (binding.needs_rebind) is one of FindImage's checks. Setting it bumps the image's page
// epochs as a registration does (DrawRepeat::NoteImagePagesChange), so a proof that stands on those epochs sees it
// (the other epoch memos only take their full path once more). Under m_lock, as registrations.
static void MarkNeedsRebind(Image& image) {
	if (image.binding.needs_rebind) return;
	image.binding.needs_rebind = true;
	DrawRepeat::NoteImagePagesChange(image.info.data.address, image.info.data.size);
}

bool TextureCache::SyncExtentAliases(ImageId id) {
	if (!AliasSyncEnabled()) return false;
	auto& image = m_slot_images[id];
	if (TraceImageSyncAddr(image.info.data.address)) {
		std::fprintf(stderr, "IMAGE_SYNC frame=%u sync_enter %ux%u fmt=%s serial=%llu others=%zu\n", DebugPresentCount().load(),
		             image.backing.extent.width, image.backing.extent.height,
		             vk::to_string(image.backing.format).c_str(), static_cast<unsigned long long>(image.content_serial),
		             FindImagesInRegion(image.info.data.address, 1, true).size());
	}
	if (AliasSyncSkipsImage(image)) {
		return false;
	}
	ImageId  best {};
	uint64_t best_serial = image.content_serial;
	bool     has_source  = false;
	for (const ImageId other_id: FindImagesInRegion(image.info.data.address, 1, true)) {
		if (other_id == id) continue;
		auto* other = m_slot_images.try_get(other_id);
		if (other != nullptr && TraceImageSyncAddr(image.info.data.address)) {
			std::fprintf(stderr, "IMAGE_SYNC frame=%u region_other addr=0x%llx %ux%u fmt=%s serial=%llu\n", DebugPresentCount().load(),
			             static_cast<unsigned long long>(other->info.data.address), other->backing.extent.width,
			             other->backing.extent.height, vk::to_string(other->backing.format).c_str(),
			             static_cast<unsigned long long>(other->content_serial));
		}
		if (other == nullptr || !AliasSyncSource(image, *other)) {
			continue;
		}
		has_source = true;
		if (TraceImageSyncAddr(image.info.data.address)) {
			std::fprintf(stderr, "IMAGE_SYNC frame=%u candidate %ux%u serial=%llu other %ux%u serial=%llu\n", DebugPresentCount().load(),
			             image.backing.extent.width, image.backing.extent.height,
			             static_cast<unsigned long long>(image.content_serial), other->backing.extent.width,
			             other->backing.extent.height, static_cast<unsigned long long>(other->content_serial));
		}
		if (other->content_serial > best_serial) {
			best        = other_id;
			best_serial = other->content_serial;
		}
	}
	if (!best) return has_source;
	if (ImageMemoryOwnerMode() != 0u) { // KYTY_FIX_IMAGE_MEMORY_OWNER: a copy into an image that map does not cover
		ImageMemoryOwners::g_alias_sync_copies[ImageMemoryOwners::ExcludedIndex(image)].fetch_add(1, std::memory_order_relaxed);
		// A source the map covers first takes the bytes others hold over its memory (as at its own sampled binding).
		if (ImageMemoryOwners::Followed(m_slot_images[best]) && m_slot_images[best].owner_stale) OwnerSyncImage(best, 0u);
	}
	auto& source = m_slot_images[best];
	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.extent         = vk::Extent3D {std::min(source.backing.extent.width, image.backing.extent.width),
	                                    std::min(source.backing.extent.height, image.backing.extent.height), 1};
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	image.content_serial = best_serial;
	if (TraceImageSyncAddr(image.info.data.address)) {
		std::fprintf(stderr, "IMAGE_SYNC frame=%u alias_copy addr=0x%llx %ux%u <- %ux%u buffer_modified=%d\n", DebugPresentCount().load(),
		             static_cast<unsigned long long>(image.info.data.address), image.backing.extent.width,
		             image.backing.extent.height, source.backing.extent.width, source.backing.extent.height,
		             image.IsBufferModified() ? 1 : 0);
	}
	static uint32_t logged = 0;
	if (logged < 24u) {
		++logged;
		std::fprintf(stderr, "WORKAROUND_ALIAS_SYNC addr=0x%llx %ux%u %s <- %ux%u %s\n",
		             static_cast<unsigned long long>(image.info.data.address), image.backing.extent.width,
		             image.backing.extent.height, vk::to_string(image.backing.format).c_str(),
		             source.backing.extent.width, source.backing.extent.height,
		             vk::to_string(source.backing.format).c_str());
	}
	return true;
}

static bool PreserveStencilEnabled() {
	static constinit Common::OnceValue<bool> enabled_once;
	const bool enabled = enabled_once.Get([] {
		// Default from Common::EnvDefaults; workaround: D8 (stencil / HTILE), to
		// be replaced by the proper implementation; =0 turns it off.
		const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_PRESERVE_STENCIL");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) std::fprintf(stderr, "WORKAROUND_PRESERVE_STENCIL APPROXIMATION enabled\n");
		return on;
	});
	return enabled;
}

static uint64_t StencilPlaneKey(uint64_t depth_address, uint64_t plane, uint32_t width) {
	return (depth_address * 31u) ^ plane ^ (uint64_t {width} << 48u);
}

// KYTY_FIX_STENCIL_PLANE_WRITES (M119, default off; =1 on, =2 on and every redirected write logged): PS5 keeps each
// stencil plane in its own memory, the host keeps one stencil aspect per depth image (KYTY_WORKAROUND_PRESERVE_STENCIL swaps
// planes in and out). A guest-side write to a plane - its guest bytes uploaded after a buffer or CPU write, or a fill
// turned into a clear - used to land on the depth image's stencil aspect whatever plane it held: at Atsu's house the
// game's fill of the interior-volume plane (0x5061360000) was uploaded over plane A (0x5004850000, the G-buffer classes
// and the water bits 0x48) right after Deferred wetness swapped A back in, so Water Lighting, SSR, SSDO and the shadow
// copy read an empty plane A (no water lighting: a flat lake). With the fix such a write goes to the plane it addresses:
// the stencil aspect when the depth image holds it, else the plane's saved copy; and a plane swapped in at a depth bind
// with buffer writes pending (the game filled it while it was not held) gets them before anything draws into it.
// Checked with dumps of the plane readers: the stencil each reader sees equals the plane as restored by the last swap
// (0 differences); before the fix plane A read back all zero.
static uint32_t StencilPlaneWritesMode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	return mode_once.Get([] {
		const char*    value = std::getenv("KYTY_FIX_STENCIL_PLANE_WRITES");
		// Default 1 since 22.10 (=0 off): gate 14.24 vs 14.15 fps, open world 20.13 vs 19.84 (second pair; the first,
		// 19.16 vs 22.22, was run noise), lake water near Atsu's house lit again.
		const uint32_t mode  = value == nullptr ? 1u : value[0] == '1' ? 1u : value[0] == '2' ? 2u : 0u;
		if (mode != 0) std::fprintf(stderr, "FIX_STENCIL_PLANE_WRITES mode %u\n", mode);
		return mode;
	});
}

static void NoteStencilPlaneRedirect(const char* what, const Image& depth, uint64_t plane, uint64_t bytes) {
	if (StencilPlaneWritesMode() != 2u) return;
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 256u) {
		std::fprintf(stderr, "FIX_STENCIL_PLANE_WRITES frame=%u %s plane=0x%llx depth=0x%llx %ux%u holds=0x%llx bytes=%llu\n",
		             DebugPresentCount().load(), what, static_cast<unsigned long long>(plane),
		             static_cast<unsigned long long>(depth.info.data.address), depth.backing.extent.width,
		             depth.backing.extent.height, static_cast<unsigned long long>(depth.stencil_plane_address),
		             static_cast<unsigned long long>(bytes));
	}
}

// KYTY_WORKAROUND_STENCIL_PLANE_SYNC: 1 = shader reads of a plane get its contents, 2 = also a plane written by a shader
// is loaded into the depth image at swap-in.
int TextureCache::StencilPlaneSyncLevel() {
	static constinit Common::OnceValue<int> level_once;
	const int level = level_once.Get([] {
		// Default from Common::EnvDefaults; workaround: D8 (stencil / HTILE), to
		// be replaced by the proper implementation; =0 turns it off.
		const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_STENCIL_PLANE_SYNC");
		const int   on    = value != nullptr && value[0] >= '1' && value[0] <= '9' ? value[0] - '0' : 0;
		if (on != 0) std::fprintf(stderr, "WORKAROUND_STENCIL_PLANE_SYNC level %d\n", on);
		return on;
	});
	return PreserveStencilEnabled() ? level : 0;
}

bool TextureCache::StencilPlaneSyncEnabled() {
	return StencilPlaneSyncLevel() >= 1;
}

ImageId TextureCache::NewerStencilPlaneImage(uint64_t plane_address, uint64_t saved_serial, vk::Extent3D extent) {
	ImageId best {};
	for (const auto id: FindImagesInRegion(plane_address, 1, false)) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || image->depth_id || image->info.IsDepth() || image->info.data.address != plane_address ||
		    image->info.bytes_per_block != 1u || image->backing.image == nullptr || image->backing.samples != 1u ||
		    image->shader_write_serial <= saved_serial) {
			continue;
		}
		// The plane's bytes map to the same pixels only at the depth image's own extent: a write at another
		// resolution (dynamic resolution) laid the same memory out with another pitch.
		if (image->backing.extent.width != extent.width || image->backing.extent.height != extent.height) {
			continue;
		}
		if (!best || image->shader_write_serial > m_slot_images[best].shader_write_serial) best = id;
	}
	return best;
}

bool TextureCache::SwapStencilPlane(Image& depth, uint64_t plane_address) {
	if (!PreserveStencilEnabled() || !depth.info.IsDepth() || !depth.info.HasStencil() ||
	    depth.backing.samples != 1 || plane_address == 0 || depth.stencil_plane_address == plane_address) {
		return false;
	}
	const auto extent = depth.backing.extent;
	const auto bytes  = uint64_t {extent.width} * extent.height;
	const auto key    = [&](uint64_t plane) { return StencilPlaneKey(depth.info.data.address, plane, extent.width); };
	m_scheduler.EndRendering();
	auto               command = m_scheduler.Current().Handle();
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {extent.width, extent.height, 1};
	const auto make_saved = [&](uint64_t plane) -> Buffer& {
		auto& saved = m_stencil_planes[key(plane)];
		if (!saved || saved->Size() < bytes) {
			saved = std::make_unique<Buffer>(
			    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
			    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
		}
		return *saved;
	};
	if (depth.stencil_plane_address != 0) {
		auto& saved = make_saved(depth.stencil_plane_address);
		command.pipelineBarrier2(dependency);
		depth.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
		command.copyImageToBuffer(depth.backing.image, vk::ImageLayout::eTransferSrcOptimal, saved.Handle(), 1,
		                          &copy);
		m_stencil_plane_saved_serial[key(depth.stencil_plane_address)] = Image::NextContentSerial();
	}
	const auto previous          = depth.stencil_plane_address;
	if (DebugStencilWindow()) {
		std::fprintf(stderr, "STENCIL_PLANE_SWAP frame=%u depth=0x%llx %ux%u 0x%llx -> 0x%llx\n", DebugPresentCount().load(),
		             static_cast<unsigned long long>(depth.info.data.address), extent.width, extent.height,
		             static_cast<unsigned long long>(previous), static_cast<unsigned long long>(plane_address));
	}
	depth.stencil_plane_address = plane_address;
	if (StencilPlaneSyncLevel() >= 2) {
		// A shader wrote the plane (a 1-byte image at its address) after it was saved: that is its contents now.
		const auto serial_it    = m_stencil_plane_saved_serial.find(key(plane_address));
		const auto saved_serial = serial_it != m_stencil_plane_saved_serial.end() ? serial_it->second : 0u;
		if (const auto newer_id = NewerStencilPlaneImage(plane_address, saved_serial, extent)) {
			auto&      newer  = m_slot_images[newer_id];
			auto&      saved  = make_saved(plane_address);
			const auto width  = std::min(extent.width, newer.backing.extent.width);
			const auto height = std::min(extent.height, newer.backing.extent.height);
			vk::BufferImageCopy color_copy {};
			color_copy.bufferRowLength  = extent.width;
			color_copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
			color_copy.imageExtent      = vk::Extent3D {width, height, 1};
			command.pipelineBarrier2(dependency);
			newer.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
			command.copyImageToBuffer(newer.backing.image, vk::ImageLayout::eTransferSrcOptimal, saved.Handle(), 1,
			                          &color_copy);
			m_stencil_plane_saved_serial[key(plane_address)] = newer.shader_write_serial;
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 8u) {
				std::fprintf(stderr, "WORKAROUND_STENCIL_PLANE_SYNC plane 0x%llx loaded from a shader-written %ux%u image\n",
				             static_cast<unsigned long long>(plane_address), newer.backing.extent.width,
				             newer.backing.extent.height);
			}
		}
	}
	const auto found             = m_stencil_planes.find(key(plane_address));
	if (found == m_stencil_planes.end() || !found->second) {
		return false;
	}
	command.pipelineBarrier2(dependency);
	depth.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyBufferToImage(found->second->Handle(), depth.backing.image, vk::ImageLayout::eTransferDstOptimal, 1,
	                          &copy);
	command.pipelineBarrier2(dependency);
	static uint32_t logged = 0;
	if (std::has_single_bit(++logged)) { // #1, 2, 4, 8...: that it happens and how often, not a line per 100
		std::fprintf(stderr, "WORKAROUND_PRESERVE_STENCIL plane swap #%u depth=0x%llx 0x%llx -> 0x%llx\n", logged,
		             static_cast<unsigned long long>(depth.info.data.address),
		             static_cast<unsigned long long>(previous), static_cast<unsigned long long>(plane_address));
	}
	return true;
}

bool TextureCache::SyncStencilPlaneReader(uint64_t plane_address, ImageId reader_id) {
	std::scoped_lock lock {m_lock};
	const auto       owner = m_stencil_plane_depth.find(plane_address);
	if (owner == m_stencil_plane_depth.end()) {
		return false;
	}
	auto*      reader   = m_slot_images.try_get(reader_id);
	// After a copy the reader holds the plane's current contents, which exist only on the GPU (stencil writes never
	// reach guest memory). Its guest-memory state is consumed as for a GPU write; otherwise the refresh when it is
	// bound (FindTexture) uploads the stale guest bytes over the copy: the game's reader image is created anew every
	// frame, and a new image starts CPU-dirty. The new content serial keeps an older same-address alias of another
	// extent from being copied over it (KYTY_WORKAROUND_ALIAS_SYNC).
	const auto synced = [&]() {
		reader->ClearBufferModified();
		if (reader->IsCpuDirty()) reader->RefreshComplete();
		reader->content_serial = Image::NextContentSerial();
	};
	if (reader == nullptr || !reader->registered || reader->backing.image == nullptr || reader->backing.samples != 1u ||
	    reader->info.bytes_per_block != 1u) {
		return false;
	}
	// The live depth/stencil image at the owner's address (depth images are replaced, e.g. on a dynamic resolution
	// switch) of the reader's extent: a reader at another extent sees the plane's memory with another pitch. Prefer
	// the one holding the plane, else one with a saved copy of it.
	// image.info.stencil only describes the last binding (a depth-only pass clears it); the stencil aspect of the
	// backing format is what holds the plane.
	const auto has_stencil_aspect = [](const Image& image) {
		switch (image.backing.format) {
			case vk::Format::eD32SfloatS8Uint:
			case vk::Format::eD24UnormS8Uint:
			case vk::Format::eD16UnormS8Uint: return true;
			default: return false;
		}
	};
	Image* depth = nullptr;
	for (const auto id: FindImagesInRegion(owner->second, 1, false)) {
		auto* candidate = m_slot_images.try_get(id);
		if (candidate == nullptr || !candidate->registered || !candidate->info.IsDepth() ||
		    !has_stencil_aspect(*candidate) || candidate->backing.samples != 1u ||
		    candidate->info.data.address != owner->second ||
		    candidate->backing.extent.width != reader->backing.extent.width ||
		    candidate->backing.extent.height != reader->backing.extent.height) {
			continue;
		}
		if (candidate->stencil_plane_address == plane_address) {
			depth = candidate;
			break;
		}
		const auto candidate_key =
		    StencilPlaneKey(candidate->info.data.address, plane_address, candidate->backing.extent.width);
		if (depth == nullptr && m_stencil_planes.contains(candidate_key)) depth = candidate;
	}
	if (depth == nullptr) {
		// No live depth image: it was replaced by a color view of its depth plane (a shader reads depth as R32F),
		// which saved its stencil first (SaveStencil), or the plane is only in a swap copy. Either buffer holds the
		// plane's bytes at the reader's extent.
		const auto        reader_extent = reader->backing.extent;
		vk::Buffer        copy_source   = nullptr;
		if (m_saved_stencil && m_saved_stencil_valid && m_saved_stencil_address == owner->second &&
		    m_saved_stencil_plane == plane_address && m_saved_stencil_extent.width == reader_extent.width &&
		    m_saved_stencil_extent.height == reader_extent.height) {
			copy_source = m_saved_stencil->Handle();
		} else {
			const auto swap_key = StencilPlaneKey(owner->second, plane_address, reader_extent.width);
			const auto swapped  = m_stencil_planes.find(swap_key);
			const auto serial   = m_stencil_plane_saved_serial.find(swap_key);
			if (swapped != m_stencil_planes.end() && swapped->second && serial != m_stencil_plane_saved_serial.end() &&
			    reader->shader_write_serial <= serial->second) {
				copy_source = swapped->second->Handle();
			}
		}
		if (copy_source == nullptr) {
			return false;
		}
		m_scheduler.EndRendering();
		auto               saved_command = m_scheduler.Current().Handle();
		vk::MemoryBarrier2 saved_barrier {};
		saved_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		saved_barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
		saved_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		saved_barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		vk::DependencyInfo saved_dependency {};
		saved_dependency.memoryBarrierCount = 1;
		saved_dependency.pMemoryBarriers    = &saved_barrier;
		vk::BufferImageCopy saved_copy {};
		saved_copy.bufferRowLength  = reader_extent.width;
		saved_copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		saved_copy.imageExtent      = vk::Extent3D {reader_extent.width, reader_extent.height, 1};
		saved_command.pipelineBarrier2(saved_dependency);
		reader->Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, saved_command);
		saved_command.copyBufferToImage(copy_source, reader->backing.image, vk::ImageLayout::eTransferDstOptimal, 1,
		                                &saved_copy);
		saved_command.pipelineBarrier2(saved_dependency);
		synced();
		return true;
	}
	const auto extent  = depth->backing.extent;
	const auto key     = StencilPlaneKey(depth->info.data.address, plane_address, extent.width);
	const bool current = depth->stencil_plane_address == plane_address;
	const auto saved   = m_stencil_planes.find(key);
	if (!current) {
		const auto serial = m_stencil_plane_saved_serial.find(key);
		// No saved copy yet, or the reader was written after the save: its own contents are the plane.
		if (saved == m_stencil_planes.end() || !saved->second || serial == m_stencil_plane_saved_serial.end() ||
		    reader->shader_write_serial > serial->second) {
			return true;
		}
	}
	const auto bytes = uint64_t {extent.width} * extent.height;
	m_scheduler.EndRendering();
	auto               command = m_scheduler.Current().Handle();
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	vk::Buffer source = nullptr;
	if (current) {
		if (!m_stencil_sync_staging || m_stencil_sync_staging->Size() < bytes) {
			m_stencil_sync_staging = std::make_unique<Buffer>(
			    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
			    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
		}
		vk::BufferImageCopy stencil_copy {};
		stencil_copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
		stencil_copy.imageExtent      = vk::Extent3D {extent.width, extent.height, 1};
		command.pipelineBarrier2(dependency);
		depth->Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
		command.copyImageToBuffer(depth->backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                          m_stencil_sync_staging->Handle(), 1, &stencil_copy);
		source = m_stencil_sync_staging->Handle();
	} else {
		source = saved->second->Handle();
	}
	const auto width  = std::min(extent.width, reader->backing.extent.width);
	const auto height = std::min(extent.height, reader->backing.extent.height);
	vk::BufferImageCopy color_copy {};
	color_copy.bufferRowLength  = extent.width;
	color_copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	color_copy.imageExtent      = vk::Extent3D {width, height, 1};
	command.pipelineBarrier2(dependency);
	reader->Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyBufferToImage(source, reader->backing.image, vk::ImageLayout::eTransferDstOptimal, 1, &color_copy);
	command.pipelineBarrier2(dependency);
	// Not a shader write: shader_write_serial stays, so a later swap-in does not take the reader for a newer plane.
	synced();
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 8u) {
		std::fprintf(stderr, "WORKAROUND_STENCIL_PLANE_SYNC read of plane 0x%llx from the %s (%ux%u -> %ux%u)\n",
		             static_cast<unsigned long long>(plane_address), current ? "depth stencil" : "saved copy",
		             extent.width, extent.height, reader->backing.extent.width, reader->backing.extent.height);
	}
	return true;
}

void TextureCache::EnsureStencilPlane(ImageId depth_id, uint64_t plane_address) {
	std::scoped_lock lock {m_lock};
	auto*            depth = m_slot_images.try_get(depth_id);
	if (depth == nullptr) return;
	(void)SwapStencilPlane(*depth, plane_address);
}

// KYTY_FIX_STENCIL_PLANE_WRITES: the saved copy of a plane holds what copyImageToBuffer of the stencil aspect (layer 0,
// level 0, tightly packed rows) gives; only a plane record of that shape is written there.
static bool StencilPlaneSavedShape(const Image& depth, const Image& plane) {
	const auto& extent = depth.backing.extent;
	return depth.backing.samples == 1u && plane.info.resources.layers == 1u &&
	       plane.stencil_subresources.base_layer == 0u && plane.info.extent.width == extent.width &&
	       plane.info.extent.height == extent.height;
}

bool TextureCache::UploadStencilPlaneToSaved(Image& depth, const Image& plane, Buffer& source, uint64_t source_offset) {
	if (!StencilPlaneSavedShape(depth, plane)) return false;
	const auto  extent = depth.backing.extent;
	const auto  bytes  = uint64_t {extent.width} * extent.height;
	const auto  linear = m_tiler.Detile(source.Handle(), source_offset, plane.info);
	const auto  copies = plane.info.BufferCopies(linear.offset, 1);
	if (copies.size() != 1u) return false;
	const auto& copy = copies[0];
	if (copy.imageOffset != vk::Offset3D {0, 0, 0} || copy.imageExtent.width != extent.width ||
	    copy.imageExtent.height != extent.height || copy.imageSubresource.mipLevel != 0u ||
	    copy.imageSubresource.baseArrayLayer != 0u) {
		return false;
	}
	const uint64_t row = copy.bufferRowLength != 0u ? copy.bufferRowLength : extent.width;
	std::vector<vk::BufferCopy> rows;
	if (row == extent.width) {
		rows.emplace_back(copy.bufferOffset, 0, bytes);
	} else {
		rows.reserve(extent.height);
		for (uint32_t y = 0; y < extent.height; y++) {
			rows.emplace_back(copy.bufferOffset + y * row, uint64_t {y} * extent.width, extent.width);
		}
	}
	const auto key   = StencilPlaneKey(depth.info.data.address, plane.info.data.address, extent.width);
	auto&      saved = m_stencil_planes[key];
	if (!saved || saved->Size() < bytes) {
		saved = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                                 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		                                 bytes);
	}
	m_scheduler.EndRendering();
	auto               command = m_scheduler.Current().Handle();
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	command.copyBuffer(linear.buffer, saved->Handle(), static_cast<uint32_t>(rows.size()), rows.data());
	command.pipelineBarrier2(dependency);
	// Newer than any 1-byte image a shader wrote at the plane address before (KYTY_WORKAROUND_STENCIL_PLANE_SYNC=2 loads
	// such an image at swap-in only when it is newer than the saved copy).
	m_stencil_plane_saved_serial[key] = Image::NextContentSerial();
	NoteStencilPlaneRedirect("upload", depth, plane.info.data.address, bytes);
	return true;
}

bool TextureCache::FillStencilPlaneSaved(CommandBuffer& command, Image& depth, const Image& plane, uint8_t value) {
	const auto extent = depth.backing.extent;
	const auto bytes  = uint64_t {extent.width} * extent.height;
	// vkCmdFillBuffer writes whole dwords.
	if (!StencilPlaneSavedShape(depth, plane) || bytes % 4u != 0u) return false;
	const auto key    = StencilPlaneKey(depth.info.data.address, plane.info.data.address, extent.width);
	auto&      saved  = m_stencil_planes[key];
	if (!saved || saved->Size() < bytes) {
		saved = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                                 vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		                                 bytes);
	}
	command.EndRendering();
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.Handle().pipelineBarrier2(dependency);
	command.Handle().fillBuffer(saved->Handle(), 0, bytes, uint32_t {value} * 0x01010101u);
	command.Handle().pipelineBarrier2(dependency);
	m_stencil_plane_saved_serial[key] = Image::NextContentSerial();
	NoteStencilPlaneRedirect("fill", depth, plane.info.data.address, bytes);
	return true;
}

void TextureCache::SaveStencil(Image& depth) {
	const auto extent = depth.backing.extent;
	const auto bytes  = uint64_t {extent.width} * extent.height;
	if (!m_saved_stencil || m_saved_stencil->Size() < bytes) {
		m_saved_stencil = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
	}
	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	depth.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {extent.width, extent.height, 1};
	command.copyImageToBuffer(depth.backing.image, vk::ImageLayout::eTransferSrcOptimal,
	                          m_saved_stencil->Handle(), 1, &copy);
	m_saved_stencil_address = depth.info.data.address;
	if (DebugStencilWindow()) {
		std::fprintf(stderr, "STENCIL_SAVE frame=%u depth=0x%llx %ux%u\n", DebugPresentCount().load(),
		             static_cast<unsigned long long>(depth.info.data.address), extent.width, extent.height);
	}
	m_saved_stencil_extent  = extent;
	m_saved_stencil_valid   = true;
	m_saved_stencil_plane   = depth.stencil_plane_address;
	if (DebugCaptureFrame().load()) {
		std::fprintf(stderr, "CAPTURE_STENCIL_SAVE cs_index=%u\n", DebugCaptureIndex().load());
	}
	static uint32_t saves = 0;
	if (std::has_single_bit(++saves)) { // #1, 2, 4, 8...
		std::fprintf(stderr, "WORKAROUND_PRESERVE_STENCIL save #%u 0x%llx %ux%u\n", saves,
		             static_cast<unsigned long long>(depth.info.data.address), extent.width, extent.height);
	}
}

void TextureCache::RestoreStencil(Image& depth) {
	if (!m_saved_stencil_valid || m_saved_stencil_address != depth.info.data.address ||
	    m_saved_stencil_extent.width != depth.backing.extent.width ||
	    m_saved_stencil_extent.height != depth.backing.extent.height) {
		static uint32_t misses = 0;
		if (++misses % 50u == 1u) {
			std::fprintf(stderr, "WORKAROUND_PRESERVE_STENCIL restore miss #%u valid=%d saved=0x%llx %ux%u want=0x%llx %ux%u\n",
			             misses, m_saved_stencil_valid ? 1 : 0, static_cast<unsigned long long>(m_saved_stencil_address),
			             m_saved_stencil_extent.width, m_saved_stencil_extent.height,
			             static_cast<unsigned long long>(depth.info.data.address), depth.backing.extent.width,
			             depth.backing.extent.height);
		}
		return;
	}
	m_scheduler.EndRendering();
	auto               command = m_scheduler.Current().Handle();
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	depth.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {m_saved_stencil_extent.width, m_saved_stencil_extent.height, 1};
	command.copyBufferToImage(m_saved_stencil->Handle(), depth.backing.image,
	                          vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	m_saved_stencil_valid = false;
	depth.stencil_restored = true;
	if (DebugStencilWindow()) {
		std::fprintf(stderr, "STENCIL_RESTORE frame=%u depth=0x%llx %ux%u\n", DebugPresentCount().load(),
		             static_cast<unsigned long long>(depth.info.data.address), depth.backing.extent.width,
		             depth.backing.extent.height);
	}
	depth.stencil_plane_address = m_saved_stencil_plane;
	if (DebugCaptureFrame().load()) {
		std::fprintf(stderr, "CAPTURE_STENCIL_RESTORE cs_index=%u\n", DebugCaptureIndex().load());
	}
	static uint32_t logged = 0;
	if (++logged % 50u == 1u) {
		std::fprintf(stderr, "WORKAROUND_PRESERVE_STENCIL restored stencil of 0x%llx %ux%u\n",
		             static_cast<unsigned long long>(depth.info.data.address), depth.backing.extent.width,
		             depth.backing.extent.height);
	}
}

ImageId TextureCache::GetStencilShadowImage(uint32_t width, uint32_t height) {
	std::scoped_lock lock {m_lock};
	const auto       key = (uint64_t {width} << 32u) | height;
	if (const auto found = m_stencil_shadow_images.find(key); found != m_stencil_shadow_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.guest_format    = Prospero::BufferFormat::k8UInt;
	info.pixel_format    = vk::Format::eR8Uint;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {width, height, 1};
	info.resources       = {1, 1};
	info.pitch           = width;
	info.bytes_per_block = 1;
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, uint64_t {width} * height, width, height};
	const auto id        = InsertImage(info);
	m_stencil_shadow_images.emplace(key, id);
	return id;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto format = desc.info.pixel_format;
	if (const auto found = m_null_images.find(format); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(format, id);
	// The RGBA32F null texture (every sampled null float T#, descriptors.cpp NullTextureDesc). KYTY_DIAG_NULL_TEXTURE_GRAY=1
	// (logged APPROXIMATION; default from Common::EnvDefaults; workaround: D1 (null
	// textures), to be replaced by the proper implementation): it holds (0.5, 0.5, 0.5, 1). =0: (0, 0, 0, 0), what the
	// hardware returns for a null T# (RDNA2 ISA 8.1.8: "A resource set to all zeros acts as an unbound texture or buffer
	// (return 0,0,0,0)"; its DST_SEL fields are 0 = constant 0). KYTY_TRACE_PERF: PERF workaround_hits
	// null_float_sampled counts the bindings that read it. (Earlier, =0 bound an R32F null image instead, whose
	// texel was never written: (x, 0, 0, 1).)
	const char* gray_null = Common::EnvOrDefault("KYTY_DIAG_NULL_TEXTURE_GRAY");
	if (format == vk::Format::eR32G32B32A32Sfloat) {
		const bool gray_on = gray_null != nullptr && gray_null[0] == '1';
		std::fprintf(stderr, gray_on ? "DIAG_NULL_TEXTURE_GRAY APPROXIMATION enabled\n"
		                             : "NULL_TEXTURE (0,0,0,0) as the hardware (KYTY_DIAG_NULL_TEXTURE_GRAY=0)\n");
		auto& image = m_slot_images[id];
		m_scheduler.EndRendering();
		auto command = m_scheduler.Current().Handle();
		image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
		// KYTY_DIAG_NULL_TEXTURE_COLOR=<r,g,b> (diagnostic): another color, e.g. 1,0,1, so surfaces
		// sampling null/unresolved textures stand out.
		vk::ClearColorValue gray {gray_on ? std::array<float, 4> {0.5f, 0.5f, 0.5f, 1.0f} : std::array<float, 4> {}};
		if (const char* color = std::getenv("KYTY_DIAG_NULL_TEXTURE_COLOR"); color != nullptr) {
			float r = 0.5f, g = 0.5f, b = 0.5f;
			if (std::sscanf(color, "%f,%f,%f", &r, &g, &b) == 3) gray = vk::ClearColorValue {std::array<float, 4> {r, g, b, 1.0f}};
		}
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		GpuCheckpoints::Mark(command, 0xFEED000000000000ull | (8ull << 16u) | 934u);
		command.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &gray, 1, &range);
		image.MarkGpuModified();
	}
	return id;
}

ImageId TextureCache::GetNullDepthImage(bool one) {
	std::unique_lock lock {m_lock};
	auto&            cached = m_null_depth_images[one ? 1u : 0u];
	if (cached) {
		return cached;
	}
	ImageInfo info {};
	info.pixel_format    = vk::Format::eD16Unorm;
	info.guest_format    = Prospero::BufferFormat::k16UNorm;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = 2;
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, 2, 1, 1};
	cached               = InsertImage(info);
	auto& image          = m_slot_images[cached];
	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	const vk::ClearDepthStencilValue value {one ? 1.0f : 0.0f, 0u};
	const vk::ImageSubresourceRange  range {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
	command.clearDepthStencilImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &value, 1, &range);
	image.MarkGpuModified();
	return cached;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	const std::array                 depth_copies {
	    vk::BufferImageCopy {0,
	                         depth.info.pitch,
	                         depth.info.extent.height,
	                         {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                         {},
	                         depth.info.extent}};
	const std::array color_copies {
	    vk::BufferImageCopy {0,
	                         color.info.pitch,
	                         color.info.extent.height,
	                         {vk::ImageAspectFlagBits::eColor, 0, 0, layers},
	                         {},
	                         color.info.extent}};

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (TraceImageSyncAddr(destination.info.data.address)) {
		std::fprintf(stderr, "IMAGE_SYNC frame=%u copy dst=%s %ux%u <- src=%s %ux%u src_buffer_modified=%d src_gpu_modified=%d same_data=%d\n",
		             DebugPresentCount().load(), vk::to_string(destination.backing.format).c_str(),
		             destination.backing.extent.width, destination.backing.extent.height,
		             vk::to_string(source.backing.format).c_str(), source.backing.extent.width,
		             source.backing.extent.height, source.IsBufferModified() ? 1 : 0, source.IsGpuModified() ? 1 : 0,
		             source.info.data == destination.info.data ? 1 : 0);
	}
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
		// The source's bytes are not in guest memory (armed): now they are the destination's, read back in its place
		// (M118: the hi-Z pyramid ends each frame as the D32 copy of its R32 one; an expanded or merged image likewise).
		if (source.read_armed) ArmWrittenImage(destination);
	}
	if (ImageMemoryOwnerMode() != 0u) OwnerCopyWrite(destination_id, source_id); // KYTY_FIX_IMAGE_MEMORY_OWNER
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	if (m_slot_images[destination_id].IsBufferModified()) {
		RefreshImage(destination_id);
	}
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	if (mip >= destination.backing.mip_levels && mip < destination.info.resources.levels) {
		// A guest level past the destination's 1x1 level has no host subresource (ImageInfo::HostLevels); those levels
		// are 1x1 as well, and the host's last level stands for them: the copy goes there (28.10; the
		// stop added that day ended Snake's house in RT mode).
		mip = destination.backing.mip_levels - 1u;
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
		if (source.read_armed) ArmWrittenImage(destination); // as in CopyImage
	}
	if (ImageMemoryOwnerMode() != 0u) OwnerCopyWrite(destination_id, source_id); // KYTY_FIX_IMAGE_MEMORY_OWNER
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	UnregisterReason reason(m_unregister_reason, "depth_overlap");
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	bool recreate = !cached.info.resources.Contains(requested.resources) ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout          = cached.info.mip_layout;
		info.tiling              = cached.info.tiling;
		info.tiled_slice_stride  = cached.info.tiled_slice_stride;
		info.linear_slice_stride = cached.info.linear_slice_stride;
		info.first_tail_level    = cached.info.first_tail_level;
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target || PlanUsesLoad(cached.plan_uses) != 0) {
		MarkNeedsRebind(cached);
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			if (PreserveStencilEnabled() && cached.info.IsDepth() && cached.info.HasStencil() &&
			    !replacement.info.IsDepth() && cached.backing.samples == 1) {
				SaveStencil(cached);
			}
			CopyImage(replacement_id, cached_id);
			if (PreserveStencilEnabled() && replacement.info.IsDepth() && replacement.info.HasStencil() &&
			    !cached.info.IsDepth() && replacement.backing.samples == 1) {
				RestoreStencil(replacement);
			}
			// The copy holds the GPU-written content now: read back in its place when the source was (CopyImage).
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

namespace BcAlias {

bool Matches(const ImageInfo& requested, const ImageInfo& cached) {
	return requested.guest_format != cached.guest_format &&
	       requested.IsBlock() != cached.IsBlock() &&
	       requested.tile_mode == cached.tile_mode &&
	       requested.bytes_per_block == cached.bytes_per_block &&
	       requested.samples == 1 && cached.samples == 1 &&
	       (requested.data != cached.data || requested.resources != cached.resources ||
	        requested.BlockExtent() != cached.BlockExtent()) &&
	       !TextureCache::SameBacking(cached, requested, false);
}

bool Publish(TextureCache& cache, Image& image) {
	if (!cache.CanDownload(image)) {
		return false;
	}
	// ObtainBuffer preserves padding and records ownership before the GPU-only transfer.
	auto [buffer, offset] =
	    cache.m_buffer_cache.ObtainBuffer(image.info.data.address, image.info.data.size, true);
	cache.DownloadImage(image, *buffer, offset, image.info.data.size, image.info.resources.levels);
	image.ClearGpuModified();
	image.MarkBufferModified();
	cache.DisarmImageRead(image); // the buffer cache watches these bytes now
	return true;
}

void Resolve(TextureCache& cache, ImageId cached_id, ImageId merged_id) {
	auto& cached = cache.m_slot_images[cached_id];
	auto* merged = merged_id ? &cache.m_slot_images[merged_id] : nullptr;
	// PS5 raw block images can span several packed BC mips. Transfer their complete
	// guest bytes, including tail padding, before retiring the incompatible backing.
	if (cached.SafeToDownload()) {
		if (merged && merged->SafeToDownload() && !Publish(cache, *merged)) {
			return;
		}
		if (!Publish(cache, cached)) {
			return;
		}
		if (merged) {
			merged->MarkBufferModified();
		}
	}
	if (cached.binding.is_bound || cached.binding.is_target || PlanUsesLoad(cached.plan_uses) != 0) MarkNeedsRebind(cached);
	cache.FreeImage(cached_id);
}

} // namespace BcAlias

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr || !owner->registered) {
		return {merged_id};
	}
	// An uncompressed alias of one level/layer of a compressed container binds as a block-texel view of the container
	// (below), not through BcAlias, which would download the container, free it and create both images again every frame
	// (the game's compressed cube array, ~9 MiB down and up per frame).
	if (const int32_t mip = requested.MipOf(owner->info); mip >= 0 && requested.BlockTexelAliasOf(owner->info) &&
	    static_cast<bool>(owner->backing.flags & vk::ImageCreateFlagBits::eBlockTexelViewCompatible) &&
	    (binding != BindingType::Storage || static_cast<bool>(owner->backing.usage & vk::ImageUsageFlagBits::eStorage))) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 24u) {
			std::fprintf(stderr, "BC_TEXEL_VIEW addr=0x%llx %ux%u fmt=%s -> container 0x%llx %ux%u fmt=%s mip=%d layer=%d binding=%u\n",
			             static_cast<unsigned long long>(requested.data.address), requested.extent.width,
			             requested.extent.height, vk::to_string(requested.pixel_format).c_str(),
			             static_cast<unsigned long long>(owner->info.data.address), owner->info.extent.width,
			             owner->info.extent.height, vk::to_string(owner->info.pixel_format).c_str(), mip,
			             requested.SliceOf(owner->info, mip), static_cast<uint32_t>(binding));
		}
		return {cached_id, mip, requested.SliceOf(owner->info, mip)};
	}
	if (BcAlias::Matches(requested, owner->info) && !owner->IsCpuDirty()) {
		BcAlias::Resolve(*this, cached_id, merged_id);
		return {merged_id};
	}
	UnregisterReason reason(m_unregister_reason, "overlap");
	auto&      cached       = *owner;
	const auto current_tick = m_scheduler.CurrentTick();
	// The list path of bindless tables over 1 MiB: an image such a table drew within as many frames is not old, as its
	// left-out touch would have made it.
	const bool fast_kept =
	    cached.bindless_pinned && bindless_fast_keep && bindless_fast_keep(cached_id, NumFramesBeforeRemoval);
	const bool safe_to_delete =
	    !fast_kept && current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    cached.heap_refs.load(std::memory_order_relaxed) == 0u;

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (TraceImageSyncAddr(requested.data.address) || TraceImageSyncAddr(cached.info.data.address)) {
		const auto re = requested.BlockExtent();
		const auto ce = cached.info.BlockExtent();
		std::fprintf(stderr, "IMAGE_SYNC frame=%u overlap req=0x%llx %ux%u tile=%u block=%u bext=%ux%u meta=%u bind=%u | cached=0x%llx %ux%u tile=%u block=%u bext=%ux%u meta=%u\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(requested.data.address),
		             requested.extent.width, requested.extent.height, static_cast<uint32_t>(requested.tile_mode),
		             requested_block, re.width, re.height, static_cast<uint32_t>(requested.metadata.kind),
		             static_cast<uint32_t>(binding), static_cast<unsigned long long>(cached.info.data.address),
		             cached.info.extent.width, cached.info.extent.height, static_cast<uint32_t>(cached.info.tile_mode),
		             cached_block, ce.width, ce.height, static_cast<uint32_t>(cached.info.metadata.kind));
	}
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     !std::equal(requested.mip_layout.begin(), requested.mip_layout.end(),
		                 cached.info.mip_layout.begin(),
		                 [](const ImageMipInfo& left, const ImageMipInfo& right) {
			                 return left.offset == right.offset && left.size == right.size &&
			                        left.pitch == right.pitch && left.height == right.height &&
			                        left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
			                        left.surface_z == right.surface_z;
		                 }))) {
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		// A request with layers the cached image lacks is no view of it, however few bytes it takes (a mip chain lives inside
		// each layer, so a shorter chain packs more layers into the same bytes: 12 layers of 1 level in an image of 9 layers of
		// 7): the tail below gives it its own image, from guest memory in its own layout.
		const bool layers_missing = !requested.IsVolume() && requested.resources.layers > cached.info.resources.layers;
		if (!layers_missing && (requested.pixel_format != cached.info.pixel_format ||
		                        requested.data.size <= cached.info.data.size)) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		// A larger request of the same type and format whose layout is none of the above (Demon's Souls' name-entry
		// keyboard, 28.10: a 1-level x 305-layer array over a 9-level x 64-layer one): another texture laid over the
		// same memory, not a view or a growth of the cached one. Like a request of another type, it is an image of its
		// own (the caller creates it from guest memory), where this used to stop the emulator.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
			std::fprintf(stderr,
			             "TextureCache: equal-address image of another layout, separate image: address=0x%016" PRIx64
			             " requested=%ux%u cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
			             " type=%u/%u tile=%u/%u\n",
			             requested.data.address, requested.resources.levels, requested.resources.layers,
			             cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
			             cached.info.data.size, static_cast<uint32_t>(requested.type),
			             static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
			             static_cast<uint32_t>(cached.info.tile_mode));
		}
		return {merged_id};
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	// A block-texel alias binds as a view of the compressed container, which needs block-texel view support (and
	// storage usage for a storage binding); otherwise it stays a separate image as before.
	const bool block_view =
	    requested_mip >= 0 && requested.BlockTexelAliasOf(cached.info) &&
	    static_cast<bool>(cached.backing.flags & vk::ImageCreateFlagBits::eBlockTexelViewCompatible) &&
	    (binding != BindingType::Storage || static_cast<bool>(cached.backing.usage & vk::ImageUsageFlagBits::eStorage));
	if (requested_mip >= 0 && requested.BlockTexelAliasOf(cached.info)) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 24u) {
			std::fprintf(stderr, "BLOCK_TEXEL_ALIAS view addr=0x%llx %ux%u fmt=%s -> container 0x%llx %ux%u fmt=%s mip=%d layer=%d binding=%u bound=%d\n",
			             static_cast<unsigned long long>(requested.data.address), requested.extent.width,
			             requested.extent.height, vk::to_string(requested.pixel_format).c_str(),
			             static_cast<unsigned long long>(cached.info.data.address), cached.info.extent.width,
			             cached.info.extent.height, vk::to_string(cached.info.pixel_format).c_str(), requested_mip,
			             requested.SliceOf(cached.info, requested_mip), static_cast<uint32_t>(binding), block_view ? 1 : 0);
		}
	}
	if (requested_mip >= 0 && (block_view || !requested.BlockTexelAliasOf(cached.info))) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0 && cached.info.BlockTexelAliasOf(requested)) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 24u) {
			std::fprintf(stderr, "BLOCK_TEXEL_ALIAS take-over container 0x%llx %ux%u fmt=%s <- alias 0x%llx %ux%u fmt=%s mip=%d layer=%d merged=%d\n",
			             static_cast<unsigned long long>(requested.data.address), requested.extent.width,
			             requested.extent.height, vk::to_string(requested.pixel_format).c_str(),
			             static_cast<unsigned long long>(cached.info.data.address), cached.info.extent.width,
			             cached.info.extent.height, vk::to_string(cached.info.pixel_format).c_str(), mip,
			             cached.info.SliceOf(requested, mip), merged_id ? 1 : 0);
		}
	}
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		if (cached.binding.is_bound || cached.binding.is_target || PlanUsesLoad(cached.plan_uses) != 0) MarkNeedsRebind(cached);
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		if (PlanUsesLoad(cached.plan_uses) != 0 && on_plan_target_transfer) on_plan_target_transfer(cached_id, merged_id); // the plan flags
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	// Diagnostic (2 GB buffer at full VRAM): an expansion to 512 MiB or more, with both images.
	if (info.data.size >= (uint64_t {512} << 20u)) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32u) {
			std::fprintf(stderr,
			             "BIG_IMAGE_EXPAND frame=%u requested 0x%llx+0x%llx %ux%ux%u L%u x%u guest_format=%u %s tile=%u "
			             "pitch=%u | source %s +0x%llx\n",
			             static_cast<unsigned>(DebugPresentCount().load()),
			             static_cast<unsigned long long>(info.data.address), static_cast<unsigned long long>(info.data.size),
			             info.extent.width, info.extent.height, info.extent.depth, info.resources.levels,
			             info.resources.layers, static_cast<uint32_t>(info.guest_format),
			             vk::to_string(info.pixel_format).c_str(), static_cast<uint32_t>(info.tile_mode),
			             info.mip_layout[0].pitch, DescribeImage(m_slot_images[source_id]).c_str(),
			             static_cast<unsigned long long>(m_slot_images[source_id].info.data.size));
		}
	}
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target || PlanUsesLoad(source.plan_uses) != 0) {
		MarkNeedsRebind(source);
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	UnregisterReason reason(m_unregister_reason, "expand");
	FreeImage(source_id);
	return expanded_id;
}

bool TextureCache::CanDownload(const Image& image) const {
	const auto& info = image.info;
	if (info.samples != 1 || image.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		return IsSupportedDepthPlaneReadback(info) &&
		       Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
	}
	return info.metadata.compression == VideoOutCompression::Uncompressed &&
	       (!info.IsTiled() || !Prospero::IsFmaskTextureFormat(info.guest_format)) &&
	       info.LinearSize() != 0;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	// KYTY_FIX_STENCIL_PLANE_WRITES: the guest bytes of a stencil plane the depth image does not hold go to that
	// plane's saved copy, not over the plane the image holds.
	if (image.depth_id && StencilPlaneWritesMode() != 0u && PreserveStencilEnabled() &&
	    destination.stencil_plane_address != 0 && destination.stencil_plane_address != image.info.data.address &&
	    UploadStencilPlaneToSaved(destination, image, source, source_offset)) {
		return;
	}
	const auto& info        = image.info;
	const auto  binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	if (DebugCaptureFrame().load() && destination.info.IsDepth()) {
		std::fprintf(stderr, "CAPTURE_DEPTH_UPLOAD cs_index=%u addr=0x%llx stencil_record=%d\n",
		             DebugCaptureIndex().load(), static_cast<unsigned long long>(image.info.data.address),
		             image.depth_id ? 1 : 0);
	}
	if (binding != BindingType::DepthTarget) {
		if ((binding == BindingType::RenderTarget || binding == BindingType::VideoOut) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		if (binding == BindingType::VideoOut &&
		    info.metadata.compression != VideoOutCompression::Uncompressed) {
			EXIT("TextureCache: invalid video-out upload\n");
		}
		const auto linear = m_tiler.Detile(source.Handle(), source_offset, info);
		const auto copies = info.BufferCopies(linear.offset);
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
		return;
	}

	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto layers          = info.resources.layers;
	const auto full_slice_size = info.linear_slice_stride;
	auto       linear          = m_tiler.Detile(source.Handle(), source_offset, info);
	auto       copies          = info.BufferCopies(linear.offset);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		copies.resize(1);
		copies[0].bufferOffset      = linear.offset;
		copies[0].bufferRowLength   = info.pitch;
		copies[0].bufferImageHeight = info.extent.height;
		copies[0].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, 0, layers};
		copies[0].imageExtent       = info.extent;
	}
	destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	// Our null images (GetNullImage) have no guest memory (upstream 892e70c8a removed them; not taken).
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		// KYTY_TRACE_STARTUP: guest bytes to the GPU (staging copy and detile) of every image upload.
		const Common::StartupTrace::Scope startup_scope(Common::StartupTrace::Counter::ImageUpload, image.info.data.size);
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		if (TraceImageSyncAddr(image.info.data.address)) {
			std::fprintf(stderr, "IMAGE_SYNC frame=%u upload addr=0x%llx image=%ux%u buffer_modified=%d\n", DebugPresentCount().load(),
			             static_cast<unsigned long long>(image.info.data.address), image.backing.extent.width,
			             image.backing.extent.height, image.IsBufferModified() ? 1 : 0);
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
		if (ImageMemoryOwnerMode() != 0u) OwnerAfterUpload(id); // KYTY_FIX_IMAGE_MEMORY_OWNER
		if (TraceImageSyncAddr(image.info.data.address) && DebugStencilWindow()) {
			if (const char* dir = Common::EnvRead::FrameDir(); dir != nullptr) {
				static uint32_t n = 0;
				// The image right after the upload, and the first 64 KiB of the upload source buffer.
				DebugQueueFullDump(m_graphics, m_scheduler.Current().Handle(), image,
				                   std::string(dir) + "/imgsync_after_upload_" + std::to_string(n), false);
				DebugQueueBufferDump(m_graphics, m_scheduler.Current().Handle(), source->Handle(), source_offset,
				                     65536u, std::string(dir) + "/imgsync_source_" + std::to_string(n) + ".bin");
				n++;
			}
		}
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

bool TextureCache::DccFastClearEnabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		// Default from Common::EnvDefaults; workaround: A5 (DCC metadata, M46),
		// to be replaced by the proper implementation; =0 turns it off.
		const char* value   = Common::EnvOrDefault("KYTY_WORKAROUND_DCC_FAST_CLEAR");
		const bool  enabled = value != nullptr && value[0] == '1';
		if (enabled) std::fprintf(stderr, "WORKAROUND_DCC_FAST_CLEAR enabled\n");
		return enabled;
	});
	return on;
}

bool TextureCache::DccKnownKeysEnabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		const char* value   = std::getenv("KYTY_FIX_DCC_KNOWN_KEYS"); // default on; =0 turns it off
		const bool  enabled = (value == nullptr || value[0] == '1') && DccFastClearEnabled();
		if (enabled) std::fprintf(stderr, "FIX_DCC_KNOWN_KEYS enabled\n");
		return enabled;
	});
	return on;
}

bool TextureCache::DccKey10FixEnabled() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		const char* value   = std::getenv("KYTY_FIX_DCC_KEY_0X10");
		const bool  enabled = value == nullptr || value[0] != '0'; // default on (=0 turns it off)
		if (enabled) std::fprintf(stderr, "FIX_DCC_KEY_0X10 enabled\n");
		return enabled;
	});
	return on;
}

void TextureCache::ApplyDccPendingWrites() {
	std::vector<std::pair<uint64_t, uint64_t>> pending;
	{
		std::scoped_lock lock {m_dcc_pending_mutex};
		pending.swap(m_dcc_pending);
	}
	// Another GPU writer (a dispatch, draw or BDA store) may have rewritten the keys: forget them.
	// KYTY_FIX_DCC_KEY_0X10: or the color surface after its clear-to-pixel store: forget that store. (Writes are
	// queued for the stores too, so the keys are matched only when KYTY_FIX_DCC_KNOWN_KEYS queued them.)
	const bool known_keys = DccKnownKeysEnabled();
	for (const auto& [address, size]: pending) {
		if (known_keys) {
			auto it = m_dcc_fills.upper_bound(address);
			if (it != m_dcc_fills.begin() && std::prev(it)->first + std::prev(it)->second.first > address) --it;
			while (it != m_dcc_fills.end() && it->first < address + size) it = m_dcc_fills.erase(it);
		}
		if (!m_dcc_pixel_fills.empty()) {
			auto it = m_dcc_pixel_fills.upper_bound(address);
			if (it != m_dcc_pixel_fills.begin() && std::prev(it)->first + std::prev(it)->second.size > address) --it;
			while (it != m_dcc_pixel_fills.end() && it->first < address + size) it = m_dcc_pixel_fills.erase(it);
		}
	}

	m_dcc_fill_count.store(m_dcc_fills.size(), std::memory_order_relaxed);
	m_dcc_pixel_fill_count.store(m_dcc_pixel_fills.size(), std::memory_order_relaxed);
}

void TextureCache::RegisterDccFill(uint64_t address, uint64_t size, uint8_t key) {
	if (address == 0 || size == 0) return;
	Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccFill); // KYTY_TRACE_PERF
	std::scoped_lock lock {m_lock};
	ApplyDccPendingWrites();
	// The fill rewrote every key in its range: registered clears overlapping it end.
	auto it = m_dcc_fills.upper_bound(address);
	if (it != m_dcc_fills.begin() && std::prev(it)->first + std::prev(it)->second.first > address) --it;
	while (it != m_dcc_fills.end() && it->first < address + size) it = m_dcc_fills.erase(it);
	const bool clear_key = key == 0x00u || key == 0x40u || key == 0x80u || key == 0xc0u;
	// Every key is kept: a clear key clears sampled surfaces, any key spares the metadata download.
	if (m_dcc_fills.size() < 4096u) m_dcc_fills[address] = {size, key};
	m_dcc_fill_count.store(m_dcc_fills.size(), std::memory_order_relaxed);
	// KYTY_FIX_DCC_KEY_0X10: a clear-to-pixel store counts for these keys only when it comes after them.
	if (key == 0x10u && DccKey10FixEnabled() &&
	    (m_dcc_key10_serials.size() < 4096u || m_dcc_key10_serials.contains(address))) {
		m_dcc_key10_serials[address] = ++m_dcc_fill_serial;
	}
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 32u) {
		std::fprintf(stderr, "WORKAROUND_DCC_FAST_CLEAR fill frame=%u meta=0x%llx size=0x%llx key=0x%02x clear=%d\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(address),
		             static_cast<unsigned long long>(size), key, clear_key ? 1 : 0);
	}
}

void TextureCache::RegisterDccPixelFill(uint64_t address, uint64_t size, const std::array<uint32_t, 4>& element,
                                        uint32_t element_bytes) {
	if (address == 0 || size == 0 || element_bytes == 0 || element_bytes > 16u) return;
	std::scoped_lock lock {m_lock};
	// The store's own write of the surface was reported when the dispatch bound its buffers, before this call:
	// applying it now ends the previous store's entry, and only later writes end this one.
	ApplyDccPendingWrites();
	auto it = m_dcc_pixel_fills.upper_bound(address);
	if (it != m_dcc_pixel_fills.begin() && std::prev(it)->first + std::prev(it)->second.size > address) --it;
	while (it != m_dcc_pixel_fills.end() && it->first < address + size) it = m_dcc_pixel_fills.erase(it);
	if (m_dcc_pixel_fills.size() < 4096u) {
		m_dcc_pixel_fills[address] = {size, element, element_bytes, ++m_dcc_fill_serial};
	}
	m_dcc_pixel_fill_count.store(m_dcc_pixel_fills.size(), std::memory_order_relaxed);
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 16u) {
		std::fprintf(stderr,
		             "DCC_KEY_0X10 store frame=%u addr=0x%llx size=0x%llx element=%08x,%08x,%08x,%08x bytes=%u\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(address),
		             static_cast<unsigned long long>(size), element[0], element[1], element[2], element[3],
		             element_bytes);
	}
}

void TextureCache::EraseDccPixelFill(uint64_t address) {
	m_dcc_pixel_fills.erase(address);
	m_dcc_pixel_fill_count.store(m_dcc_pixel_fills.size(), std::memory_order_relaxed);
}

// The fast path: every key of the surface is 0x10 (a registered 0x10 fill over the whole metadata, or every key byte
// read back) and one registered store put the same element in every block, so the surface is a clear to that element.
// TODO(KYTY_FIX_DCC_KEY_0X10, general case): blocks whose first elements differ, or were not stored by the Toolkit's
// clear to pixel, need a per-block expansion: a compute pass over a copy of the surface's raw bytes, before the
// detiling upload (guest memory stays as it is), that copies the first bytes_per_block bytes of every 256-byte block
// whose key is 0x10 over the whole block (elements of 1-4 dwords). It needs each block's key, i.e. the GFX10 DCC
// metadata addressing of the surface (the keys are not stored in block order), which the emulator does not have yet.
// Until then such surfaces keep their memory contents, logged here as DCC_KEY_0X10 skip.
bool TextureCache::DecodeDccKey10(const ImageDesc& desc, vk::ClearColorValue& clear, const char* path) {
	const auto  it     = m_dcc_pixel_fills.find(desc.info.data.address);
	const auto  keys   = m_dcc_key10_serials.find(desc.info.metadata.range.address);
	const char* reason = nullptr;
	if (it == m_dcc_pixel_fills.end()) {
		reason = "no_store";
	} else if (keys != m_dcc_key10_serials.end() && it->second.serial < keys->second) {
		reason = "store_before_keys";
	} else if (it->second.size < desc.info.data.size) {
		reason = "store_smaller_than_surface";
	} else if (desc.info.IsDepth() || desc.info.bytes_per_block == 0 ||
	           desc.info.bytes_per_block > it->second.element_bytes) {
		reason = "element_size";
	} else if (!DecodeDccElementClear(desc.view_info.format, it->second.element, desc.info.bytes_per_block, clear)) {
		reason = "format";
	}
	if (reason == nullptr) return true;
	static std::atomic_uint logged {0};
	if (const auto n = logged.fetch_add(1) + 1u; n <= 32u || (n % 1024u) == 0u) {
		std::fprintf(stderr,
		             "DCC_KEY_0X10 skip frame=%u path=%s type=%u image=0x%llx %ux%u fmt=%s bpp=%u size=0x%llx "
		             "meta=0x%llx reason=%s store_size=0x%llx element_bytes=%u count=%u\n",
		             DebugPresentCount().load(), path, static_cast<uint32_t>(desc.type),
		             static_cast<unsigned long long>(desc.info.data.address), desc.info.extent.width,
		             desc.info.extent.height, vk::to_string(desc.view_info.format).c_str(), desc.info.bytes_per_block,
		             static_cast<unsigned long long>(desc.info.data.size),
		             static_cast<unsigned long long>(desc.info.metadata.range.address), reason,
		             static_cast<unsigned long long>(it != m_dcc_pixel_fills.end() ? it->second.size : 0ull),
		             it != m_dcc_pixel_fills.end() ? it->second.element_bytes : 0u, n);
	}
	return false;
}

void TextureCache::ApplyDccFastClear(CommandBuffer& command, ImageId id, const ImageDesc& desc) {
	const auto& metadata = desc.info.metadata;
	if (metadata.kind != ImageMetadataKind::Dcc || metadata.range.size == 0 || !id) return;
	std::unique_lock lock {m_lock};
	if (DccKnownKeysEnabled() || DccKey10FixEnabled()) ApplyDccPendingWrites();
	const auto it = m_dcc_fills.find(metadata.range.address);
	if (it == m_dcc_fills.end()) return;
	if (desc.type != BindingType::Texture) {
		// Rendering or storage writes replace the cleared contents (and their keys). With
		// KYTY_FIX_DCC_KNOWN_KEYS the keys were expanded to 0xFF (MaterializeColorClear) and stay known.
		if (DccKnownKeysEnabled()) {
			it->second.second = 0xffu;
		} else {
			m_dcc_fills.erase(it);
		}
		return;
	}
	const auto key = it->second.second;
	// KYTY_FIX_DCC_KEY_0X10: keys 0x10 read as the element the Toolkit's clear to pixel stored after them.
	const bool pixel = key == 0x10u && DccKey10FixEnabled();
	if (key != 0x00u && key != 0x40u && key != 0x80u && key != 0xc0u && !pixel) return;
	auto& image = m_slot_images[id];
	vk::ClearValue clear {};
	if (it->second.first < metadata.range.size) {
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccPartialFill); // KYTY_TRACE_PERF
	}
	if (it->second.first < metadata.range.size || desc.info.resources.levels != 1 || image.info.IsDepth() ||
	    image.backing.format != desc.view_info.format ||
	    !(pixel ? DecodeDccKey10(desc, clear.color, "texture") : DecodeColorClear(desc, key, clear.color))) {
		return;
	}
	if (pixel) LogDccKey10Apply("texture", desc, clear.color);
	// The texture unit reads the clear color for these keys whatever the data memory holds (other
	// surfaces may alias it): clear the backing for this use without claiming a guest write, so
	// the aliased surfaces keep their contents.
	command.EndRendering();
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0,
	                                       VK_REMAINING_ARRAY_LAYERS};
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command.Handle());
	GpuCheckpoints::Mark(command.Handle(), 0xFEED000000000000ull | (8ull << 16u) | 1732u);
	command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &clear.color, 1, &range);
	image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite, {},
	              command.Handle());
	Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccFastTexClear); // KYTY_TRACE_PERF
	static std::set<std::pair<uint64_t, uint32_t>> logged;
	if (logged.size() < 64u &&
	    logged.insert({metadata.range.address, (desc.info.extent.width << 16u) | desc.info.extent.height}).second) {
		std::fprintf(stderr,
		             "WORKAROUND_DCC_FAST_CLEAR apply frame=%u image=0x%llx %ux%u fmt=%s meta=0x%llx key=0x%02x color=%g,%g,%g,%g\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(desc.info.data.address),
		             desc.info.extent.width, desc.info.extent.height, vk::to_string(desc.view_info.format).c_str(),
		             static_cast<unsigned long long>(metadata.range.address), it->second.second, clear.color.float32[0],
		             clear.color.float32[1], clear.color.float32[2], clear.color.float32[3]);
	}
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range = desc.info.metadata.range;
	// KYTY_WORKAROUND_DCC_FAST_CLEAR: a surface resolved for rendering or storage writes ends its
	// registered clear (its contents come from those writes from now on).
	// KYTY_FIX_DCC_KNOWN_KEYS keeps the registration: the known keys are expanded below instead.
	if (DccFastClearEnabled() && !DccKnownKeysEnabled() && desc.type != BindingType::Texture &&
	    desc.info.metadata.kind == ImageMetadataKind::Dcc) {
		std::scoped_lock lock {m_lock};
		m_dcc_fills.erase(range.address);
	}
	if (TraceImageSyncAddr(desc.info.data.address)) {
		std::fprintf(stderr, "IMAGE_SYNC frame=%u meta_enter kind=%u %ux%u fmt=%d levels=%u layers=%u meta=0x%llx size=0x%llx\n",
		             DebugPresentCount().load(), static_cast<uint32_t>(desc.info.metadata.kind), desc.info.extent.width,
		             desc.info.extent.height, static_cast<int>(desc.info.pixel_format), desc.info.resources.levels,
		             desc.info.TransferLayers(), static_cast<unsigned long long>(range.address),
		             static_cast<unsigned long long>(range.size));
	}
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			if (range.size != 0) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccLevels); // KYTY_TRACE_PERF
			return;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	const bool         whole_blocks      = range.Valid() && range.address % MetadataBlockSize == 0 && layers != 0 &&
	                                       range.size % layers == 0 && (range.size / layers) % MetadataBlockSize == 0;
	if (!whole_blocks) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccMetaUnaligned); // KYTY_TRACE_PERF
	// A5 step E1: the keys of each slice are read where they are, whatever their alignment (T# and CB place metadata on
	// 256-byte boundaries; the CPU read takes any size and the GPU check falls back to it for slices that are not whole
	// words). Only metadata that cannot be split into equal slices stops the emulator.
	if (!range.Valid() || layers == 0 || range.size % layers != 0) {
		EXIT("TextureCache: color metadata cannot be split into %u equal slices (image=0x%" PRIx64 " meta=0x%" PRIx64
		     " size=0x%" PRIx64 ")\n",
		     layers, static_cast<uint64_t>(desc.info.data.address), static_cast<uint64_t>(range.address),
		     static_cast<uint64_t>(range.size));
	}
	if (!whole_blocks) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u) {
			std::fprintf(stderr,
			             "COLOR_META_SLICES unaligned addr=0x%llx meta=0x%llx size=0x%llx layers=%u %ux%u type=%u\n",
			             static_cast<unsigned long long>(desc.info.data.address),
			             static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.size), layers,
			             desc.info.extent.width, desc.info.extent.height, static_cast<uint32_t>(desc.type));
		}
	}
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	// KYTY_FIX_DCC_KNOWN_KEYS=1: keys a registered fill wrote (and nothing has written since) are
	// known without the download. Sampled uses leave a clear key to ApplyDccFastClear; rendering and
	// storage uses get a clear key's color cleared into the image, then the keys become 0xFF
	// (uncompressed) on the GPU copy of the metadata and in the registration, as the CPU path below
	// publishes them.
	if (DccKnownKeysEnabled() && desc.info.metadata.kind == ImageMetadataKind::Dcc &&
	    m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		uint8_t key   = 0;
		bool    known = false;
		{
			std::scoped_lock lock {m_lock};
			ApplyDccPendingWrites();
			const auto it = m_dcc_fills.find(range.address);
			if (it != m_dcc_fills.end() && it->second.first >= range.size) {
				known = true;
				key   = it->second.second;
			} else if (it != m_dcc_fills.end()) {
				Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccPartialFill); // KYTY_TRACE_PERF
			}
		}
		if (known) {
			vk::ClearValue clear {};
			if (desc.type == BindingType::Texture || key == 0xffu) return;
			// KYTY_FIX_DCC_KEY_0X10: keys 0x10 read as the element the Toolkit's clear to pixel stored after them;
			// rendering into the surface starts from that color (Ghost of Yotei: motion vectors not written by
			// its G-buffer draws stay NaN).
			const bool pixel = key == 0x10u && DccKey10FixEnabled();
			if (pixel) {
				std::scoped_lock lock {m_lock};
				if (!DecodeDccKey10(desc, clear.color, "known")) return;
			} else if (!DecodeColorClear(desc, key, clear.color)) {
				return;
			}
			{
				// The keys decide, as on the hardware: buffer writes never change DCC keys, so a clear key reads as the
				// clear colour whatever was written through a buffer (on the object-ID target such contents were the
				// motion vectors' 1-in-64 NaN pattern).
				std::scoped_lock lock {m_lock};
				Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccKnownExpand); // KYTY_TRACE_PERF
				for (uint32_t slice = 0; slice < count; slice++) {
					ClearImage(m_scheduler.Current(), id, view.format,
					           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count, image_first + slice, 1},
					           clear);
				}
			}
			if (desc.type != BindingType::VideoOut) {
				const auto [buffer, offset] = m_buffer_cache.ObtainBuffer(range.address, range.size, true);
				auto& command = m_scheduler.Current();
				command.EndRendering();
				const auto native = command.Handle();
				vk::MemoryBarrier before {};
				before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
				before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1,
				                       &before, 0, nullptr, 0, nullptr);
				GpuCheckpoints::Mark(native, 0xFEED000000000000ull | (8ull << 16u) | 1882u);
				native.fillBuffer(buffer->Handle(), offset, range.size, 0xffffffffu);
				vk::MemoryBarrier after {};
				after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 1,
				                       &after, 0, nullptr, 0, nullptr);
			}
			{
				std::scoped_lock lock {m_lock};
				ApplyDccPendingWrites();
				if (m_dcc_fills.size() < 4096u) m_dcc_fills[range.address] = {range.size, uint8_t {0xff}};
				m_dcc_fill_count.store(m_dcc_fills.size(), std::memory_order_relaxed);
				if (pixel) EraseDccPixelFill(desc.info.data.address);
			}
			if (pixel) LogDccKey10Apply("known", desc, clear.color);
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 16u) {
				std::fprintf(stderr, "FIX_DCC_KNOWN_KEYS expand frame=%u image=0x%llx %ux%u type=%u key=0x%02x cleared=%d\n",
				             DebugPresentCount().load(), static_cast<unsigned long long>(desc.info.data.address),
				             desc.info.extent.width, desc.info.extent.height, static_cast<uint32_t>(desc.type), key,
				             1);
			}
			return;
		}
	}
	// KYTY_PERF_DCC_GPU_ALL (DccGpuAllMode): GPU-written DCC or CMASK keys of any binding are checked on the GPU, in
	// the command stream, instead of the readback below (a submit and a wait for all recorded work).
	if (DccGpuAllMode() != 0 && m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		const auto check_slice_size = range.size / layers;
		if (MaterializeDccClearOnGpu(id, desc, range.address + check_slice_size * first, check_slice_size, image_first,
		                             count, true)) {
			return;
		}
	}
	if (DccGpuAllMode() == 0 && DccGpuClearCheckLevel() > 0 && desc.info.metadata.kind == ImageMetadataKind::Dcc &&
	    m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		const auto check_slice_size = range.size / layers;
		if (MaterializeDccClearOnGpu(id, desc, range.address + check_slice_size * first, check_slice_size, image_first,
		                             count)) {
			return;
		}
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	if (m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		KYTY_PROFILER_BLOCK("DccMeta::Read");
		PerfReasonScope reason(PerfReason::DccMeta);
		// Upstream 5f5d6effa: the performance overlay's DCC/CMASK readback row.
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ColorMetadataReadbacks,
		                                         Common::DebugCounters::Counter::ColorMetadataReadbackNs);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCpuRead); // KYTY_TRACE_PERF
		m_buffer_cache.ReadMemory(range.address, range.size, false);
	}
	const auto slice_size    = range.size / layers;
	bool       pixel_cleared = false;
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			// Rare (2 of ~100 runs): the metadata is not readable from the backing store; skip the fast-clear
			// check for this slice instead of stopping the emulator (worst case: a missed clear). User-approved.
			MetadataReadFailure(address, slice_size);
			continue;
		}
		vk::ClearValue clear {};
		bool decoded = DecodeColorClear(desc, code, clear.color);
		// KYTY_FIX_DCC_KEY_0X10: keys 0x10 read as the element the Toolkit's clear to pixel stored.
		const bool pixel = !decoded && code == 0x10u && desc.info.metadata.kind == ImageMetadataKind::Dcc &&
		                   DccKey10FixEnabled();
		if (pixel) {
			std::scoped_lock lock {m_lock};
			ApplyDccPendingWrites();
			decoded = DecodeDccKey10(desc, clear.color, "readback");
		}
		if (TraceImageSyncAddr(desc.info.data.address)) {
			std::fprintf(stderr, "IMAGE_SYNC frame=%u meta_clear kind=%u meta=0x%llx size=0x%llx code=0x%02x decoded=%d\n",
			             DebugPresentCount().load(), static_cast<uint32_t>(desc.info.metadata.kind),
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(slice_size), code,
			             decoded ? 1 : 0);
		}
		if (!decoded) {
			continue;
		}
		std::vector<uint8_t> bytes(slice_size);
		if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			MetadataReadFailure(address, slice_size);
			continue;
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			if (TraceImageSyncAddr(desc.info.data.address)) {
				std::fprintf(stderr, "IMAGE_SYNC frame=%u meta_not_uniform code=0x%02x distinct_first=0x%02x\n",
				             DebugPresentCount().load(), code,
				             *std::find_if(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte != code; }));
			}
			continue;
		}
		// The keys decide, also for an image written through a buffer since they were set (buffer writes never change
		// DCC keys on the hardware).
		{
			std::scoped_lock lock {m_lock};
			const auto& image = m_slot_images[id];
			const bool depth = image.info.IsDepth();
			if (depth) {
				const auto value = clear.color.float32[0];
				clear.depthStencil = {value, 0u};
			}
			ClearImage(m_scheduler.Current(), id, depth ? image.backing.format : view.format,
			           {depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
			            view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
		if (pixel) {
			LogDccKey10Apply("readback", desc, clear.color);
			pixel_cleared = true;
		}
	}
	if (pixel_cleared) {
		std::scoped_lock lock {m_lock};
		EraseDccPixelFill(desc.info.data.address);
	}
}

// Default level 3 (=0 off; plan step 1 / A5): the remaining metadata
// downloads (~64 per 5 s, ~9.5 ms a frame of GPU-thread drains: the HDR target whose keys a whole-range write of the
// Forward Pre Water pass leaves unknown, and the two video-out surfaces after the Toolkit's retile) went to 0; 3 runs
// 20261014-2251..2257: images correct (kimono, landmarks), 6.27-6.55 fps.
// KYTY_FIX_DCC_GPU_CLEAR_CHECK=<level> (logged; port of BryanKAdams 8db16e32 + c41cebee): GPU-written
// DCC keys of a render target or sampled texture are checked by a compute pass that writes one predicate per clear
// code and consumes cleared slices (0xff); the candidate clears run under conditional rendering, so the CPU never
// drains the GPU to read the keys. 1 = render targets and textures, 2 = also storage images, 3 = also video-out
// surfaces (checked on every binding without consuming their keys). Needs
// VK_EXT_conditional_rendering (requested only with the switch); other cases keep the CPU readback.
uint32_t TextureCache::DccGpuClearCheckLevel() {
	static constinit Common::OnceValue<uint32_t> level_once;
	const uint32_t level = level_once.Get([] { return Common::EnvRead::DccGpuClearCheck(); });
	return level;
}

// KYTY_PERF_DCC_GPU_ALL=1 (default off): MaterializeColorClear checks GPU-written color metadata (DCC or CMASK keys)
// of every binding (render target, texture, storage, video out) on the GPU, in the command stream, instead of reading
// it back (the readback submitted and waited for everything recorded so far: ~2 per frame at ~20 ms each at the
// torii-gate scene). It extends KYTY_FIX_DCC_GPU_CLEAR_CHECK's check to every case that check left to the CPU read:
// any binding type, CMASK keys, more than DccClearResolver::MaxSlices slices (checked in chunks), multisampled images;
// and a view for which no key decodes to a clear (CMASK keys of anything but a render target with a valid clear
// register) needs no check at all. The decision is the CPU path's: a slice clears when every key byte equals its first
// byte and that byte is a code DecodeColorClear decodes for this view (the five DCC codes / CMASK code 0 are the only
// codes it decodes), read at the same point of the command stream; a cleared slice gets the same color over the same
// layer (a conditional clearAttachments instead of a transfer clear) and, for every binding but video out, its keys
// become 0xff (on the GPU copy instead of guest memory: the same bytes once read back). What still falls back (volume
// surfaces, depth associations, formats or images
// that cannot be color attachments, images that cannot get their guest contents first, keys that cannot be bound as one
// storage range, no VK_EXT_conditional_rendering) is counted and logged by class (DCC_GPU_ALL fallback=..., PERF
// dcc_gpu_all) and reads the keys on the CPU as before. =2 verify: the same, and every check also copies the keys it
// reads (before it consumes them) and the predicates it writes to a host-visible ring; once the GPU finished them a
// priority completion compares the predicates with the CPU path's decision for those bytes (PERF dcc_gpu_all
// verified/mismatches, DCC_GPU_ALL_VERIFY_MISMATCH lines). No CPU wait in either mode.
int TextureCache::DccGpuAllMode() {
	static constinit Common::OnceValue<int> mode_once;
	const int mode = mode_once.Get([] {
		const int m = Common::EnvRead::DccGpuAll();
		if (m == 1) std::fprintf(stderr, "PERF_DCC_GPU_ALL enabled\n");
		if (m == 2) std::fprintf(stderr, "PERF_DCC_GPU_ALL verify (checks copied aside and compared with the CPU decision)\n");
		return m;
	});
	return mode;
}

bool TextureCache::MaterializeDccClearOnGpu(ImageId id, const ImageDesc& desc, uint64_t slices_address,
                                            uint64_t slice_size, uint32_t image_first, uint32_t count, bool all) {
	// reason: KYTY_FIX_DCC_GPU_CLEAR_CHECK's fallback counter (DCC_GPU_CHECK line); cls: KYTY_PERF_DCC_GPU_ALL's class.
	const auto fallback = [&](uint32_t reason, DccGpuAllClass cls) {
		if (reason < m_dcc_gpu_fallbacks.size()) m_dcc_gpu_fallbacks[reason].fetch_add(1, std::memory_order_relaxed);
		if (all) NoteDccGpuAllFallback(desc, cls, count);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCheckFallback); // KYTY_TRACE_PERF
		return false;
	};
	if (m_dcc_resolver == nullptr || !m_dcc_resolver->Available()) {
		if (all) NoteDccGpuAllFallback(desc, DccGpuAllClass::NoResolver, count);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCheckFallback); // KYTY_TRACE_PERF
		return false;
	}
	const auto& view = desc.view_info;
	// The clear color of each code for this view, exactly as the CPU path decodes it.
	std::array<vk::ClearColorValue, DccClearResolver::CodeCount> colors {};
	uint32_t                                                     code_mask = 0;
	for (uint32_t index = 0; index < DccClearResolver::CodeCount; index++) {
		if (DecodeColorClear(desc, DccClearResolver::Code(index), colors[index])) code_mask |= 1u << index;
	}
	// KYTY_PERF_DCC_GPU_ALL: no key clears this view: the CPU path would read the keys and leave every slice as it is.
	if (all && code_mask == 0) {
		m_dcc_gpu_skips.fetch_add(1, std::memory_order_relaxed);
		m_dcc_all.no_code.fetch_add(1, std::memory_order_relaxed);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCheckSkip); // KYTY_TRACE_PERF
		return true;
	}
	const bool type_ok = all || desc.type == BindingType::RenderTarget || desc.type == BindingType::Texture ||
	                     (DccGpuClearCheckLevel() >= 2 && desc.type == BindingType::Storage) ||
	                     (DccGpuClearCheckLevel() >= 3 && desc.type == BindingType::VideoOut);
	// Video-out keys are not published as uncompressed (as on the CPU path): checked every time, never consumed.
	const bool consume = desc.type != BindingType::VideoOut;
	if (!type_ok) return fallback(0, DccGpuAllClass::Count);
	if (all && desc.info.IsVolume()) return fallback(1, DccGpuAllClass::Volume);
	if (desc.info.IsVolume() || count == 0 || (!all && count > DccClearResolver::MaxSlices) || view.base_level != 0 ||
	    slice_size % sizeof(uint32_t) != 0 || slices_address % sizeof(uint32_t) != 0) {
		return fallback(1, DccGpuAllClass::Slices);
	}
	if (!(m_graphics.GetFormatProperties(view.format).optimalTilingFeatures &
	      vk::FormatFeatureFlagBits::eColorAttachment)) {
		return fallback(2, DccGpuAllClass::Format);
	}
	if (code_mask == 0) {
		// No key clears this view: the CPU path would leave every slice as it is.
		m_dcc_gpu_skips.fetch_add(1, std::memory_order_relaxed);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCheckSkip); // KYTY_TRACE_PERF
		return true;
	}
	bool multisampled = false;
	{
		std::unique_lock lock {m_lock};
		auto&            image = m_slot_images[id];
		if (image.depth_id) return fallback(3, DccGpuAllClass::DepthAlias);
		if (image.backing.image == nullptr) return fallback(3, DccGpuAllClass::NoBacking);
		if (!(image.backing.usage & vk::ImageUsageFlagBits::eColorAttachment)) {
			return fallback(3, DccGpuAllClass::NoAttachment);
		}
		// KYTY_PERF_DCC_GPU_ALL: a multisampled attachment clears every sample, as the CPU path's transfer clear does.
		multisampled = image.backing.samples != 1;
		if (!all && multisampled) return fallback(3, DccGpuAllClass::Count);
		if (image_first >= image.backing.layers || count > image.backing.layers - image_first) {
			return fallback(3, DccGpuAllClass::Layers);
		}
		if (image.IsBufferModified() || image.IsCpuDirty()) {
			// A predicated clear may not happen: the image must already hold the guest contents.
			if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
				return fallback(4, DccGpuAllClass::Init);
			}
			InitializeImage(id);
			if (image.IsBufferModified() || image.IsCpuDirty()) return fallback(4, DccGpuAllClass::Init);
		}
	}
	if (consume && DccSlicesChecked(slices_address, slice_size, count, code_mask)) {
		m_dcc_gpu_skips.fetch_add(1, std::memory_order_relaxed);
		if (all) m_dcc_all.already.fetch_add(1, std::memory_order_relaxed);
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCheckSkip); // KYTY_TRACE_PERF
		return true;
	}
	// The check consumes cleared slices on the GPU, so the metadata stays GPU-owned.
	const auto [metadata, metadata_offset] = m_buffer_cache.ObtainBuffer(slices_address, slice_size * count, consume);
	if (metadata == nullptr) return fallback(5, DccGpuAllClass::Bind);
	// KYTY_PERF_DCC_GPU_ALL: chunks of at most MaxSlices slices (the predicate slots); every chunk must be recordable
	// before the first one is (nothing half done falls back to the CPU read).
	constexpr uint32_t ChunkSlices = DccClearResolver::MaxSlices;
	if (all) {
		for (uint32_t done = 0; done < count; done += ChunkSlices) {
			const uint32_t slices = std::min(ChunkSlices, count - done);
			if (!m_dcc_resolver->CanRecord(*metadata, metadata_offset + slice_size * done, slice_size, slices)) {
				return fallback(5, DccGpuAllClass::Bind);
			}
		}
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native    = command.Handle();
	const bool verifying = all && DccGpuAllMode() == 2;
	for (uint32_t done = 0; done < count;) {
		const uint32_t slices       = all ? std::min(ChunkSlices, count - done) : count;
		const uint64_t chunk_offset = metadata_offset + slice_size * done;
		// KYTY_PERF_DCC_GPU_ALL=2: the keys as the check reads them, then its predicates, copied aside.
		DccClearResolver::VerifySlot verify {};
		if (verifying) {
			verify = m_dcc_resolver->BeginVerify(native, *metadata, chunk_offset, slice_size, slices);
			if (!verify.ok) m_dcc_all.verify_skipped.fetch_add(1, std::memory_order_relaxed);
		}
		// KYTY_TRACE_GPU_TIME: the check's GPU time (the clears after it keep their own markers).
		GpuCheckpoints::Mark(native, 0xFEED000000000000ull | (8ull << 16u) | 0x0dccu);
		if (!m_dcc_resolver->Record(native, *metadata, chunk_offset, slice_size, slices, code_mask, consume)) {
			// Without `all` only (CanRecord proved every chunk above).
			return fallback(5, DccGpuAllClass::Bind);
		}
		if (verify.ok) {
			m_dcc_resolver->EndVerify(native, verify, slices);
			m_scheduler.DeferPriorityOperation(
			    [this, verify, slice_size, slices, code_mask, image_address = desc.info.data.address,
			     keys_address = slices_address + slice_size * done] {
				    VerifyDccGpuCheck(verify, slice_size, slices, code_mask, image_address, keys_address);
			    });
		}
		{
			std::scoped_lock lock {m_lock};
			auto&            image = m_slot_images[id];
			TrackImage(id);
			image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
			              vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite, {}, native);
			ImageViewInfo attachment_view {};
			attachment_view.format      = view.format;
			attachment_view.type        = slices == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
			attachment_view.base_layer  = image_first + done;
			attachment_view.layer_count = slices;
			attachment_view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
			vk::RenderingAttachmentInfo attachment {};
			attachment.imageView   = image.FindView(attachment_view);
			attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
			attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
			attachment.storeOp     = vk::AttachmentStoreOp::eStore;
			const vk::Extent2D extent {image.backing.extent.width, image.backing.extent.height};
			vk::RenderingInfo  rendering {};
			rendering.renderArea.extent    = extent;
			rendering.layerCount           = slices;
			rendering.colorAttachmentCount = 1;
			rendering.pColorAttachments    = &attachment;
			native.beginRendering(&rendering);
			for (uint32_t slice = 0; slice < slices; slice++) {
				for (uint32_t index = 0; index < DccClearResolver::CodeCount; index++) {
					if ((code_mask & (1u << index)) == 0) continue;
					vk::ConditionalRenderingBeginInfoEXT condition {};
					condition.buffer = m_dcc_resolver->PredicateBuffer();
					condition.offset = DccClearResolver::PredicateOffset(slice, index);
					native.beginConditionalRenderingEXT(&condition);
					const vk::ClearAttachment clear {vk::ImageAspectFlagBits::eColor, 0, vk::ClearValue {colors[index]}};
					const vk::ClearRect       rect {vk::Rect2D {{0, 0}, extent}, slice, 1};
					native.clearAttachments(1, &clear, 1, &rect);
					native.endConditionalRenderingEXT();
				}
			}
			native.endRendering();
			// Transit does not separate attachment writes in the same layout; order the clears before what follows.
			vk::MemoryBarrier2 ordered {};
			ordered.srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
			ordered.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
			ordered.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
			ordered.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
			vk::DependencyInfo dependency {};
			dependency.memoryBarrierCount = 1;
			dependency.pMemoryBarriers    = &ordered;
			native.pipelineBarrier2(dependency);
			CommitGpuWrite(image);
		}
		done += slices;
		if (all) m_dcc_all.chunks.fetch_add(1, std::memory_order_relaxed);
	}
	if (consume) MarkDccSlicesChecked(slices_address, slice_size, count, code_mask);
	if (all) {
		m_dcc_all.checks.fetch_add(1, std::memory_order_relaxed);
		m_dcc_all.slices.fetch_add(count, std::memory_order_relaxed);
		if (desc.info.metadata.kind == ImageMetadataKind::Cmask) m_dcc_all.cmask.fetch_add(1, std::memory_order_relaxed);
		if (multisampled) m_dcc_all.msaa.fetch_add(1, std::memory_order_relaxed);
	}
	Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::DccCheck); // KYTY_TRACE_PERF
	Common::DebugCounters::Add(Common::DebugCounters::Counter::ColorMetadataChecks); // the overlay's "DCC/CMASK GPU checks"
	const auto checks = m_dcc_gpu_checks.fetch_add(1, std::memory_order_relaxed) + 1;
	if (checks <= 16u || checks % 2000u == 0u) {
		std::fprintf(stderr,
		             "DCC_GPU_CHECK frame=%u checks=%llu skips=%llu fallback(type/view/format/image/init/bind)=%llu/%llu/"
		             "%llu/%llu/%llu/%llu image=0x%llx %ux%u type=%u slices=%u mask=0x%x\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(checks),
		             static_cast<unsigned long long>(m_dcc_gpu_skips.load()),
		             static_cast<unsigned long long>(m_dcc_gpu_fallbacks[0].load()),
		             static_cast<unsigned long long>(m_dcc_gpu_fallbacks[1].load()),
		             static_cast<unsigned long long>(m_dcc_gpu_fallbacks[2].load()),
		             static_cast<unsigned long long>(m_dcc_gpu_fallbacks[3].load()),
		             static_cast<unsigned long long>(m_dcc_gpu_fallbacks[4].load()),
		             static_cast<unsigned long long>(m_dcc_gpu_fallbacks[5].load()),
		             static_cast<unsigned long long>(desc.info.data.address), desc.info.extent.width,
		             desc.info.extent.height, static_cast<uint32_t>(desc.type), count, code_mask);
	}
	return true;
}

bool TextureCache::DccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count, uint32_t code_mask) {
	std::scoped_lock lock {m_dcc_checked_mutex};
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto found = m_dcc_checked.find(address + slice_size * slice);
		if (found == m_dcc_checked.end() || found->second.size != slice_size ||
		    (found->second.code_mask & code_mask) != code_mask) {
			return false;
		}
	}
	return true;
}

void TextureCache::MarkDccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count, uint32_t code_mask) {
	std::scoped_lock lock {m_dcc_checked_mutex};
	for (uint32_t slice = 0; slice < count; slice++) {
		m_dcc_checked[address + slice_size * slice] = {slice_size, code_mask};
	}
	m_dcc_checked_count.store(m_dcc_checked.size(), std::memory_order_relaxed);
}

void TextureCache::OnDccMetaWrite(uint64_t address, uint64_t size) {
	if (size == 0 || m_dcc_checked_count.load(std::memory_order_relaxed) == 0) return;
	std::scoped_lock lock {m_dcc_checked_mutex};
	auto             entry = m_dcc_checked.lower_bound(address);
	if (entry != m_dcc_checked.begin()) {
		const auto previous = std::prev(entry);
		if (previous->first + previous->second.size > address) entry = previous;
	}
	while (entry != m_dcc_checked.end() && entry->first < address + size) {
		entry = m_dcc_checked.erase(entry);
	}
	m_dcc_checked_count.store(m_dcc_checked.size(), std::memory_order_relaxed);
}

void TextureCache::RefreshImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		(void)image.ResolveMaybeCpuHash(image.HashGuestEdges());
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id);
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		ImageInfo plane {};
		plane.data         = stencil;
		plane.pixel_format = vk::Format::eS8Uint;
		plane.guest_format = Prospero::BufferFormat::k8UInt;
		plane.type         = depth.info.type;
		plane.extent       = depth.info.extent;
		plane.resources    = {depth.stencil_subresources.level_count,
		                      depth.stencil_subresources.layer_count};
		plane.tile_mode    = depth.info.tile_mode;
		plane.pitch =
		    plane.IsTiled() ? TileGetDepthPitch(plane.extent.width, 1, 0) : depth.info.pitch;
		plane.bytes_per_block = 1;
		plane.UpdateSize();
		if (record.depth_id != depth_id) {
			if (record.depth_id) {
				std::erase(m_slot_images[record.depth_id].stencil_ids, association);
			}
			depth.stencil_ids.push_back(association);
		}
		record.info                 = plane;
		record.depth_id             = depth_id;
		record.stencil_subresources = depth.stencil_subresources;
		DrawRepeat::NoteImageRegistryChange(plane.data.address, plane.data.size); // the registry epoch (drawRepeat.h)
	}

	return association;
}

// m_lock held (FindImage overlapping the other thread's draw work): FindImage's registry search
// for desc, read-only. The candidates are FindImagesInRegion's, in its order, without its query epoch (an image that
// does not overlap the range at its first page never does, so only overlapping ones need de-duplicating). Found only
// when FindImage would pick that image by SameBacking and do nothing else to the registry: no candidate with a
// stencil association (decided through its depth owner), none that BcAlias::Matches (FindImage resolves those whose
// dirty state allows it, which no epoch covers), and the found image passes the exact-format and mip/layer checks
// after the search.
// The image replay's hint: set by FindImage (LastFindImagePlain).
static thread_local bool g_last_find_plain = false;

static thread_local uint64_t g_full_searches = 0; // a draw's target memo

uint64_t TextureCache::FullSearches() {
	return g_full_searches;
}

bool TextureCache::TakeLastFindImagePlain() {
	const bool plain  = g_last_find_plain;
	g_last_find_plain = false;
	return plain;
}

bool TextureCache::ImageReplayable(ImageId id) const {
	const auto* image = m_slot_images.try_get(id);
	if (image != nullptr && image->info.data.Empty()) {
		return true; // null-image texture bindings take part in the image replay
	}
	return image != nullptr && image->registered && !image->depth_id && !image->info.IsDepth() &&
	       !image->info.HasStencil() && !image->info.HasMetadata() && image->info.bytes_per_block != 1u &&
	       image->stencil_ids.empty();
}

namespace {
// One read-only section of the cache's lock (lock(false): g_tc_write_seq stays as it is).
class TcReadSection {
public:
	explicit TcReadSection(TrackingSpinLock& lock) noexcept: m_lock(lock) { m_lock.lock(false); }
	~TcReadSection() { m_lock.unlock(); }
	KYTY_CLASS_NO_COPY(TcReadSection);

private:
	TrackingSpinLock& m_lock;
};

} // namespace

ImageId TextureCache::ReplayImage(ImageId id) {
	if (m_scheduler.Current().IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	const TcReadSection lock {m_lock}; // the access tick and the LRU touch only
	return ReplayImageLocked(id);
}

ImageId TextureCache::ReplayImageCheck(ImageId id) const {
	const auto* image = m_slot_images.try_get(id);
	if (image != nullptr && image->info.data.Empty()) {
		// A null image (GetNullImage): FindImage returns it without touching it.
		return id;
	}
	if (image == nullptr || !image->registered) {
		return {};
	}
	return id;
}

void TextureCache::TouchTableImages(std::span<const ImageId> ids) {
	if (ids.empty()) return;
	const TcReadSection lock {m_lock}; // the access tick and the LRU touch only, as ReplayImage
	for (const auto id: ids) (void)ReplayImageLocked(id);
}

ImageId TextureCache::ReplayImageLocked(ImageId id) {
	const ImageId result = ReplayImageCheck(id);
	if (!result) return {};
	auto& image = m_slot_images[result];
	if (image.info.data.Empty()) return result;
	image.tick_accessed_last = m_scheduler.CurrentTick();
	TouchImage(image);
	return result;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	g_last_find_plain = false;
	g_full_searches++;
	bool plain_source = false; // the image replay's hint: result from SameBacking
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		const auto       id = GetNullImage(desc);
		// The null image of a format never changes once made (m_null_images).
		g_last_find_plain = desc.type == BindingType::Texture;
		return id;
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	{
		std::scoped_lock lock {m_lock};
		KYTY_PROFILER_BLOCK("FindImage::Locked");
		const auto       candidates = [&] {
			KYTY_PROFILER_BLOCK("FindImage::Region");
			return FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
		}();
		bool block_alias = false;

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			// Stencil descriptors name the associated plane; its depth owner supplies
			// the native image and the canonical guest tiling.
			if ((desc.type == BindingType::Texture || desc.type == BindingType::Storage) &&
			    image.depth_id && image.info.data == desc.info.data &&
			    m_slot_images[image.depth_id].info.samples == desc.info.samples &&
			    image.info.extent == desc.info.extent && image.info.type == desc.info.type &&
			    image.info.resources.levels >= desc.info.resources.levels &&
			    image.info.resources.layers >= desc.info.resources.layers &&
			    desc.info.bytes_per_block == 1 &&
			    desc.info.guest_format == Prospero::BufferFormat::k8UInt) {
				result       = id;
				plain_source = false;
				break;
			}
			if (SameBacking(image.info, desc.info, exact_format)) {
				result       = id;
				plain_source = true;
			} else if (image.SafeToDownload() &&
			           BcAlias::Matches(desc.info, image.info)) {
				block_alias = true;
			}
		}
		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
		} else if (block_alias) {
			for (const auto candidate: candidates) {
				const auto* owner = m_slot_images.try_get(candidate);
				if (candidate != result && owner != nullptr && owner->registered &&
				    !owner->IsCpuDirty() &&
				    BcAlias::Matches(m_slot_images[result].info, owner->info)) {
					BcAlias::Resolve(*this, candidate, result);
				}
			}
		}

		if (result && !m_slot_images[result].depth_id) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (!resolved.info.resources.Contains(desc.info.resources)) {
				UnregisterReason reason(m_unregister_reason, "mip_growth");
				FreeImage(result);
				result = {};
			}
		}
		bool inserted_new = false;
		if (!result) {
			if (VramBudgetEnabled()) {
				MakeRoomForImage(desc.info);
			}
			result         = InsertImage(desc.info);
			inserted_new   = true;
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.info.data.address,
			                                    inserted.info.data.size)) {
				inserted.MarkBufferModified();
			}
		}
		if (TraceImageSyncAddr(desc.info.data.address)) {
			const auto& found = m_slot_images[result];
			std::fprintf(stderr, "IMAGE_SYNC frame=%u find type=%u exact=%d id=%u new=%d %ux%u fmt=%s candidates=%zu\n",
			             DebugPresentCount().load(), static_cast<uint32_t>(desc.type), exact_format ? 1 : 0,
			             static_cast<uint32_t>(result.index), inserted_new ? 1 : 0, found.backing.extent.width,
			             found.backing.extent.height, vk::to_string(found.backing.format).c_str(), candidates.size());
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
		g_last_find_plain = plain_source && !inserted_new && view_mip < 0 && view_layer < 0 && !block_alias &&
		                    (desc.type == BindingType::Texture || desc.type == BindingType::RenderTarget) &&
		                    desc.info.metadata.kind == ImageMetadataKind::None &&
		                    !TraceImageSyncAddr(desc.info.data.address);
	}
	{
		KYTY_PROFILER_BLOCK("FindImage::ColorClear");
		MaterializeColorClear(result, desc, metadata_base_layer);
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	// No registered image starts in the granule of `address` (common/lookupMemo.h), so none starts at it and the search
	// below keeps no match.
	if (ImageStartCount(address).load(std::memory_order_acquire) == 0u) {
		Common::LookupMemo::NoteHit(Common::LookupMemo::Kind::ImageStart);
		return {};
	}
	// Only images starting at `address` are kept, and an image is listed in the page of its first byte, so the first page
	// finds the same ones in the same order (a range the page table refuses stays {}).
	ImagePageTable::PageRange pages {};
	const bool                first_page = ImagePageTable::TryGetPageRange(address, size, pages);
	const uint64_t            search     = first_page ? uint64_t {1} : size;
	const TcReadSection       lock {m_lock}; // the search only (the page table's query epochs are its scratch)
	ImageIds                  matches;
	for (const auto id: FindImagesInRegion(address, search, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) continue;
		if (ensure_valid && owner->depth_id) owner = m_slot_images.try_get(owner->depth_id);
		if (owner == nullptr || (ensure_valid && !owner->SafeToDownload())) continue;
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto* image = m_slot_images.try_get(id);
			EXIT_IF(image == nullptr);
			if (image->info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) selected = owner->depth_id;
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::unique_lock lock {m_lock};
	auto& image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	const uint32_t shared_mode = SharedImageMemoryMode(); // KYTY_FIX_SHARED_IMAGE_MEMORY
	if (shared_mode != 0u) {
		if (shared_mode >= 3u) ProbeSharedImageRead(id, desc.type == BindingType::Storage ? 1u : 0u);
		SyncSharedImageMemory(id, desc.type == BindingType::Storage ? 1u : 0u);
	} else {
		SyncExtentAliases(id);
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
		image.shader_write_serial = image.content_serial;
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
	}
	// KYTY_FIX_IMAGE_MEMORY_OWNER: after the refresh from guest memory, the bytes other images hold (and for a storage
	// binding, the write).
	if (ImageMemoryOwnerMode() != 0u) OwnerSyncImage(id, desc.type == BindingType::Storage ? 1u : 0u);
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			ArmWrittenImage(image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	if (SharedImageMemoryMode() != 0u) { // KYTY_FIX_SHARED_IMAGE_MEMORY
		if (SharedImageMemoryMode() >= 3u) ProbeSharedImageRead(id, 2u);
		SyncSharedImageMemory(id, 2u);
	} else {
		SyncExtentAliases(id);
	}
	image.MarkGpuModified();
	image.usage.render_target = true;
	RefreshImage(id);
	if (ImageMemoryOwnerMode() != 0u) OwnerSyncImage(id, 2u); // KYTY_FIX_IMAGE_MEMORY_OWNER (after the refresh)
	CommitGpuWrite(image);
	ArmWrittenImage(image);
	return image.FindView(desc.view_info);
}

const std::vector<ImageId>& TextureCache::SharedImageOverlaps(Image& image, ImageId id) {
	const uint64_t epoch = SharedImageEpoch(image);
	if (image.shared_overlap_epoch != epoch) {
		image.shared_overlaps.clear();
		for (const ImageId other_id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			const auto* other = m_slot_images.try_get(other_id);
			if (other_id != id && other != nullptr && other->registered && !other->info.data.Empty()) {
				image.shared_overlaps.push_back(other_id);
			}
		}
		image.shared_overlap_epoch = epoch;
	}
	return image.shared_overlaps;
}

static uint64_t ImagePageEpochAt(uint64_t address) {
	return DrawRepeat::g_image_page_epochs[(address >> DrawRepeat::kImagePageEpochBits) &
	                                       (DrawRepeat::kImagePageEpochCount - 1u)]
	    .load(std::memory_order_relaxed);
}

// The table sets (pipeline\descriptors.cpp): FindImage's work for a stage set's sampled images, all or none
// under one lock: every image still exists with its generation and is registered without a pending rebind (a null
// image: exists); then the LRU touch and access tick of each (null images: nothing, as FindImage).
bool TextureCache::TableReplayStageChecks(std::span<const ImageId> ids) const {
	for (const auto id: ids) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr) return false;
		if (image->info.data.Empty()) continue;
		if (!image->registered || image->binding.needs_rebind) return false;
	}
	return true;
}

bool TextureCache::TableReplayStage(std::span<const ImageId> ids) {
	const TcReadSection lock {m_lock}; // access ticks and LRU touches only
	if (!TableReplayStageChecks(ids)) return false;
	const auto tick = m_scheduler.CurrentTick();
	for (const auto id: ids) {
		auto& image = m_slot_images[id];
		if (image.info.data.Empty()) continue;
		image.tick_accessed_last = tick;
		TouchImage(image);
	}
	return true;
}

// The table sets: FindTexture's work for a stage set's sampled images (views come from the set), all or none
// under one lock: FindTextureReplay's no-op predicate for each (see there), then the LRU touch.
bool TextureCache::TableTexturesReady(std::span<const ImageId> ids) {
	const TcReadSection lock {m_lock}; // LRU touches only
	return TableTexturesReadyLocked(ids);
}

bool TextureCache::ReplayTableSlotLocked(ImageId id) {
	if (m_scheduler.Current().IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	return ReplayImageLocked(id) && TableTexturesReadyLocked(std::span(&id, 1));
}

bool TextureCache::TableTexturesReadyLocked(std::span<const ImageId> ids) {
	if (!TableTexturesReadyChecks(ids)) return false;
	for (const auto id: ids) {
		auto& image = m_slot_images[id];
		if (!image.info.data.Empty()) TouchImage(image);
	}
	return true;
}

// FindTextureReplay's no-op predicate for one non-null image (TableTexturesReadyChecks, the fast path).
static bool TextureReplayReady(const Image& image) {
	return image.registered && !image.depth_id && !image.binding.needs_rebind && !image.info.IsDepth() &&
	       !image.info.HasStencil() && image.track_addr == image.info.data.address &&
	       image.track_addr_end == image.info.data.End() && !image.IsMaybeCpuDirty() && !image.IsBufferModified() &&
	       !image.IsDefinitelyCpuDirty() &&
	       (SharedImageMemoryMode() != 0u // KYTY_FIX_SHARED_IMAGE_MEMORY: no other image over its memory
	            ? !SharedImagePlain(image) || SharedImageAlone(image)
	            : !AliasSyncEnabled() || AliasSyncSkipsImage(image) ||
	                  image.alias_free_epoch == 1u + ImagePageEpochAt(image.info.data.address)) &&
	       // KYTY_FIX_IMAGE_MEMORY_OWNER: no pull pending, and no other image over its memory (whose writes in commits
	       // still to come would make one pending: answers taken ahead of those commits must not skip the sync).
	       !(ImageMemoryOwnerMode() != 0u &&
	         (image.owner_stale || (ImageMemoryOwners::Followed(image) && !ImageMemoryOwners::KnownAlone(image))));
}

bool TextureCache::TableTexturesReadyChecks(std::span<const ImageId> ids) const {
	for (const auto id: ids) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr) return false;
		if (image->info.data.Empty()) continue;
		if (!TextureReplayReady(*image)) return false;
	}
	return true;
}

// FindTexture of a sampled binding does, besides the LRU touch and the view lookup, the alias syncs and the refresh. They
// are skipped when known to do nothing:
// - SyncExtentAliases: off, or the image is one it skips, or its last full FindTexture found no image at its address it
//   could copy from and the image-page epoch of that address is unchanged since (every registration or unregistration
//   of an image overlapping the address bumps it, under m_lock, so the set of images there is the same);
// - the stencil refresh: not depth, no stencil;
// - RefreshImage: no depth association, tracked over exactly its range (TrackImage returns), not maybe-CPU-dirty, not
//   buffer-modified, not CPU-dirty (it returns before any upload).
// The checks run under m_lock, where the fault and upload paths set those flags.
vk::ImageView TextureCache::FindTextureReplay(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::Texture) {
		return FindTexture(id, desc);
	}
	vk::ImageView fast_view = nullptr;
	{
		const TcReadSection lock {m_lock}; // the LRU touch and the view (lock-free readers check neither)
		auto*               image = m_slot_images.try_get(id);
		const bool          ready = image != nullptr && !image->info.data.Empty() && TextureReplayReady(*image);
		if (ready) {
			TouchImage(*image);
			fast_view = image->FindView(desc.view_info);
		}
		if (ready) {
			return fast_view;
		}
	}
	const auto view = FindTexture(id, desc);
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	// Remember whether SyncExtentAliases has a source at this address (none: it stays a no-op while the epoch holds).
	image.alias_free_epoch = 0;
	if (AliasSyncEnabled() && image.registered && !AliasSyncSkipsImage(image)) {
		bool source = false;
		for (const ImageId other_id: FindImagesInRegion(image.info.data.address, 1, true)) {
			const auto* other = m_slot_images.try_get(other_id);
			source |= other_id != id && other != nullptr && AliasSyncSource(image, *other);
		}
		if (!source) image.alias_free_epoch = 1u + ImagePageEpochAt(image.info.data.address);
	}
	return view;
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.depth_target = true;
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasStencil()) m_stencil_plane_depth[desc.info.stencil.address] = image.info.data.address;
	if (desc.info.HasMetadata()) {
		const auto [entry, inserted] =
		    m_surface_metas.emplace(desc.info.metadata.range.address,
		                            MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                          .clear_mask = image.info.htile_clear_mask,
		                                          .size       = desc.info.metadata.range.size});
		if (!inserted) entry->second.size = desc.info.metadata.range.size;
		if (HtileSliceClearMode() != 0u) {
			// KYTY_FIX_HTILE_SLICE_CLEAR: one layer's HTILE bytes, and the layers of the whole cached image (a draw
			// into layer N describes layers 0..N only) when it is this target's image of the same extent.
			auto&      meta        = entry->second;
			const auto desc_layers = desc.info.resources.layers;
			if (inserted) {
				const auto mask = image.info.htile_clear_mask;
				meta.SetAllSlices(mask == UINT64_MAX);
				if (mask != 0 && mask != UINT64_MAX) {
					for (uint32_t s = 0; s < 64u; s++) meta.SetSlice(s, (mask & (uint64_t {1} << s)) != 0);
				}
			}
			if (desc_layers != 0 && desc.info.metadata.range.size % desc_layers == 0) {
				const uint64_t slice_size = desc.info.metadata.range.size / desc_layers;
				if (meta.slice_size != slice_size) {
					meta.slice_size = slice_size;
					meta.layers     = 0;
				}
				uint32_t layers = desc_layers;
				if (image.info.data.address == desc.info.data.address &&
				    image.info.extent.width == desc.info.extent.width &&
				    image.info.extent.height == desc.info.extent.height) {
					layers = std::max(layers, image.info.resources.layers);
				}
				meta.layers = std::max(meta.layers, layers);
			}
		}
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		const bool swapped_in = SwapStencilPlane(image, desc.info.stencil.address);
		if (swapped_in) {
			image.stencil_restored = true;
		}
		const auto association = AssociateStencil(id, desc.info.stencil);
		if (image.stencil_restored) {
			// The GPU copy is newer than guest memory; uploading it would erase the stencil.
			image.stencil_restored = false;
			// KYTY_FIX_STENCIL_PLANE_WRITES: except a GPU buffer write to the plane's memory made while the plane was
			// not held (the game's fill of the plane before drawing into it): it is newer than the saved copy, so the
			// plane gets the guest bytes now, before the draws (a CPU-dirty mark alone is not: a new plane record
			// starts CPU-dirty).
			if (swapped_in && StencilPlaneWritesMode() != 0u && m_slot_images[association].IsBufferModified()) {
				NoteStencilPlaneRedirect("swap-in upload", image, desc.info.stencil.address,
				                         uint64_t {image.backing.extent.width} * image.backing.extent.height);
				RefreshImage(association);
			}
			if (m_slot_images[association].IsCpuDirty()) {
				m_slot_images[association].RefreshComplete();
			}
		} else {
			RefreshImage(association);
		}
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	image.MarkGpuModified();
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	// KYTY_FIX_STENCIL_PLANE_WRITES: a fill of a plane the depth image does not hold clears that plane's saved copy.
	if (stencil_id && StencilPlaneWritesMode() != 0u && PreserveStencilEnabled() && image.stencil_plane_address != 0 &&
	    image.stencil_plane_address != m_slot_images[stencil_id].info.data.address &&
	    FillStencilPlaneSaved(command, image, m_slot_images[stencil_id],
	                          static_cast<uint8_t>(clear.depthStencil.stencil))) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
		return true;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	if (range.baseMipLevel >= image.backing.mip_levels) {
		// Only guest levels past the host image's 1x1 level (ImageInfo::HostLevels), which have no host subresource:
		// nothing to clear on the host (the last level keeps its own texels). A range that starts below it is cut
		// at the host's levels below.
		return;
	}
	TrackImage(id);
	if (ImageMemoryOwnerMode() != 0u) OwnerWholeWrite(id); // KYTY_FIX_IMAGE_MEMORY_OWNER (the same images)
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		BeginColorRendering(image, format, range.baseMipLevel, range.baseArrayLayer,
		                    range.layerCount, &clear, command.Handle());
		command.Handle().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	// The host image's levels end at its 1x1 level (baseMipLevel is below it here).
	native_range.levelCount = std::min(native_range.levelCount, image.backing.mip_levels - native_range.baseMipLevel);
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		GpuCheckpoints::Mark(command.Handle(), 0xFEED000000000000ull | (4ull << 16u) | 1666u);
		command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		GpuCheckpoints::Mark(command.Handle(), 0xFEED000000000000ull | (4ull << 16u) | 1669u);
		command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

vk::Rect2D TextureCache::BeginColorRendering(Image& image, vk::Format format, uint32_t level,
                                             uint32_t layer, uint32_t layers,
                                             const vk::ClearValue* clear,
                                             vk::CommandBuffer     command) {
	ImageViewInfo view {};
	view.format      = format;
	view.type        = layers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	view.base_level  = level;
	view.base_layer  = layer;
	view.layer_count = layers;
	view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	              vk::AccessFlagBits2::eColorAttachmentRead |
	                  vk::AccessFlagBits2::eColorAttachmentWrite,
	              {}, command);
	vk::RenderingAttachmentInfo attachment {};
	attachment.imageView   = image.FindView(view);
	attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
	attachment.storeOp     = vk::AttachmentStoreOp::eStore;
	if (clear != nullptr) {
		attachment.loadOp     = vk::AttachmentLoadOp::eClear;
		attachment.clearValue = *clear;
	}
	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {std::max(image.info.extent.width >> level, 1u),
	                                  std::max(image.info.extent.height >> level, 1u)};
	rendering.layerCount           = layers;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments    = &attachment;
	command.beginRendering(&rendering);
	return rendering.renderArea;
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	OnDccMetaWrite(address, size);
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	// Images register their pages before they are tracked, so CPU writes to pages without one have nothing to invalidate.
	// Most write faults land there and need not wait for the lock.
	if (ImagePageTable::PageRange pages {};
	    ImagePageTable::TryGetPageRange(address, size, pages) && !m_registered_pages.Any(pages.first, pages.last_exclusive)) {
		return;
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.linear_slice_stride;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			const auto copies = info.BufferCopies(destination_offset);
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
		} else {
			const auto copies = info.BufferCopies();
			m_tiler.TileImage(image, copies, destination.Handle(), destination_offset,
			                  info.data.size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	const std::array copies {vk::BufferImageCopy {0,
	                                              info.pitch,
	                                              info.extent.height,
	                                              {vk::ImageAspectFlagBits::eDepth, 0, 0, layers},
	                                              {},
	                                              info.extent}};
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto guest_linear = tiled ? m_tiler.GetScratchBuffer(info.LinearSize(), host_linear.buffer)
	                          : TileManager::Result {destination.Handle(), destination_offset,
	                                                 destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (tiled) {
		m_tiler.Tile(guest_linear.buffer, guest_linear.offset, guest_linear.size,
		             destination.Handle(), destination_offset, info.data.size, info);
	}
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                 uint64_t destination_size, uint32_t levels) {
	if (!CanDownload(image)) {
		EXIT("TextureCache: invalid image download\n");
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		EXIT_IF(destination_size != image.info.data.size || levels != image.info.resources.levels);
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	const auto& info      = image.info;
	const auto  transform = info.GetColorTransform();
	// Only the host image's levels are written back (ImageInfo::HostLevels): the guest bytes of levels past its 1x1
	// level keep what memory holds.
	const uint32_t guest_levels = levels == 0 ? info.resources.levels : levels;
	levels                      = std::min(guest_levels, image.backing.mip_levels);
	if (info.IsTiled()) {
		const auto copies = info.BufferCopies(0, levels);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, destination_size,
		                  levels);
	} else if (transform != ColorTransform::None) {
		auto       linear = m_tiler.GetScratchBuffer(destination_size);
		const auto copies = info.BufferCopies(0, levels);
		image.Download(copies, linear.buffer, 0, linear.size);
		if (levels < info.resources.levels && image.backing.mip_levels < info.resources.levels) {
			// (ImageInfo::HostLevels) A guest-linear chain stores its smallest levels first, so the levels past
			// the host image's 1x1 level lie before the copied ones: convert the copied levels slice by slice.
			for (uint32_t level = 0; level < levels; ++level) {
				const auto& mip    = info.mip_layout[level];
				const auto  slices = info.MipExtent(level).depth;
				for (uint32_t slice = 0; slice < slices; ++slice) {
					const uint64_t offset = mip.linear_offset + uint64_t {slice} * info.linear_slice_stride;
					if (offset >= destination_size || mip.linear_size > destination_size - offset) {
						break;
					}
					m_tiler.TransformColor({linear.buffer, linear.offset + offset, mip.linear_size},
					                       {destination.Handle(), destination_offset + offset, mip.linear_size},
					                       transform, false);
				}
			}
		} else {
			m_tiler.TransformColor(linear, {destination.Handle(), destination_offset, destination_size},
			                       transform, false);
		}
	} else {
		const auto copies = info.BufferCopies(destination_offset, levels);
		image.Download(copies, destination.Handle(), destination_offset, destination_size);
	}
}

bool TextureCache::DownloadImageMemory(ImageId id, bool evicting, std::shared_ptr<std::atomic_bool> done,
                                       std::shared_ptr<std::atomic_bool> cancelled) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	if (!image.SafeToDownload() || !CanDownload(image)) {
		return false;
	}
	const auto range    = image.info.data;
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	// An eviction does not wait for ring space (that waits for the GPU).
	auto [mapped, offset] =
	    download.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4), !evicting);
	// KYTY_VRAM_GC_FIX: when the ring cannot take the image now (busy, or the image is larger than the ring), the
	// copy goes to a buffer of its own, released once the bytes are published.
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		if (!evicting && !VramGcFixEnabled()) {
			EXIT("TextureCache: failed to map reusable download buffer\n");
		}
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, range.size);
		mapped    = temporary->Mapped().data();
		offset    = 0;
		if (mapped == nullptr) {
			return false;
		}
	} else {
		download.Commit();
	}
	Buffer& destination = temporary ? *temporary : static_cast<Buffer&>(download);
	if (!RecordImageDownload(id, destination, offset, mapped)) {
		return false;
	}
	m_scheduler.DeferPriorityOperation([&download, range, mapped, offset, owner = std::move(temporary), done = std::move(done),
	                                    cancelled = std::move(cancelled)] {
		    (owner ? *owner : static_cast<Buffer&>(download)).Invalidate(offset, range.size);
		    if (!cancelled || !cancelled->load(std::memory_order_acquire)) {
			    LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
		    }
		    if (done) {
			    done->store(true, std::memory_order_release);
		    }
	    });
	return true;
}

bool TextureCache::RecordImageDownload(ImageId id, Buffer& destination, uint64_t offset, uint8_t* mapped) {
	auto&      image = m_slot_images[id];
	const auto range = image.info.data;
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	destination.Flush(offset, range.size);

	DownloadImage(image, destination, offset, range.size, image.info.resources.levels);
	if (ImageMemoryOwnerMode() != 0u) OwnerFixReadback(id, destination, offset); // bytes other holders hold
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureDownloadBytes, range.size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = destination.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	return true;
}

// Image read coherence: the bytes of a GPU-written image reach guest memory when a guest thread reads them, so the CPU never
// reads a copy older than the labels it has already seen (the hi-Z pyramid before reprojection).
// - Every image the CPU may read back has its pages under the page manager's image read watch from the moment its write
//   enters the GPU thread's stream (ArmWrittenImage, ArmImageRead): no access, one bit per page apart from the watcher
//   counts. Only the host image's own levels: the guest bytes past them stay what memory holds.
// - A guest thread's read of such a page faults (RenderContext::ServeImageRead). The GPU thread, between packets, records
//   the copies into host buffers of their own and submits them (BeginCpuRead), and only that: the guest thread waits for
//   the GPU, publishes the bytes through the backing mapping and disarms (FinishCpuRead). The instruction then runs again.
//   Every armed image on the page is served, with the armed images overlapping those, oldest write first, so the newest
//   write wins on shared bytes. An image written again meanwhile stays armed: its next read is served anew.
// - An emulator thread never serves and never waits. Its own reads take armed pages from the backing mapping
//   (LibKernel::Memory::ReadGuestPlain), which holds what it would have held without the watch. A read that misses that
//   reader, and any write to an armed page, lifts the page's bit at once (no lock but the region's) and leaves the rest
//   to the GPU thread at a packet boundary, never in the fault (SettleImageReadLift): a write inside an image disarms it
//   without a download, because the image is CPU-dirty from then on and no download takes it (SafeToDownload);
//   any other armed image keeps the page open until its next write (ArmImageRead), so the access that lifted the page
//   runs exactly once, and until then a read of that page gets what guest memory holds, as without the watch. Lifts by
//   emulator threads are counted (emu_lift) with the faulting instruction: each is a missing reader.
// - An armed image freed before a guest thread read it (an overlap, an expansion, a merge, the collector) does not take
//   its bytes with it (DetachImageRead): an armed image written later over its whole range serves them (transferred: the
//   D32 copy of the hi-Z pyramid each frame; a copy's destination is armed when its source was); otherwise its copy is
//   recorded as an eviction's and published by the priority thread, and its pages stay closed until then (detached): a
//   guest read meanwhile waits for the copy's tick, the collector opens the pages once the copy landed. A write that gets
//   there first cancels the copy. The guard of this class: freed (no carrier for the bytes) is expected to stay 0.
// - A page whose image cannot be served any more (CPU- or buffer-dirty, unregistered) is opened (orphan).
// Line under KYTY_TRACE_READBACK=1 every 64 frames: IMAGE_READ_SERVE.

void TextureCache::ArmImageRead(Image& image) {
	if (image.read_armed) {
		if (image.read_lifted) {
			image.read_lifted = false;
			m_page_manager.ArmImageRead(image.read_address, image.read_size);
		}
		return;
	}
	if (image.depth_id || !CanDownload(image)) {
		return;
	}
	const auto& info  = image.info;
	uint64_t    begin = 0;
	uint64_t    end   = info.data.size;
	if (image.backing.mip_levels < info.resources.levels) {
		// (ImageInfo::HostLevels) Only the host image's levels are downloaded (DownloadImage).
		begin = info.data.size;
		end   = 0;
		for (uint32_t level = 0; level < image.backing.mip_levels; ++level) {
			begin = std::min(begin, info.mip_layout[level].offset);
			end   = std::max(end, info.mip_layout[level].offset + info.mip_layout[level].size);
		}
		end = std::min(end, info.data.size);
		if (begin >= end) {
			return;
		}
	}
	image.read_armed   = true;
	image.read_address = info.data.address + begin;
	image.read_size    = end - begin;
	m_page_manager.ArmImageRead(image.read_address, image.read_size);
}

void TextureCache::DisarmImageRead(Image& image) {
	if (!image.read_armed) {
		return;
	}
	image.read_armed  = false;
	image.read_lifted = false;
	OpenImageReadPages(image.read_address, image.read_size, &image);
}

void TextureCache::OpenImageReadPages(uint64_t address, uint64_t size, const Image* except) {
	// A page another armed image or a detached copy covers stays closed.
	const auto                                 begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto                                 end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	std::vector<std::pair<uint64_t, uint64_t>> closed;
	for (const auto id: FindImagesInRegion(begin, end - begin, true)) {
		const auto* other = m_slot_images.try_get(id);
		if (other != nullptr && other != except && other->read_armed) {
			closed.emplace_back(other->read_address, other->read_size);
		}
	}
	for (const auto& entry: m_detached_reads) {
		if (ImagePageRangesOverlap(entry.address, entry.size, begin, end - begin)) {
			closed.emplace_back(entry.address, entry.size);
		}
	}
	if (closed.empty()) {
		m_page_manager.DisarmImageRead(begin, end - begin);
		return;
	}
	for (uint64_t page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool covered = std::ranges::any_of(closed, [page](const auto& range) {
			return ImagePageRangesOverlap(range.first, range.second, page, TRACKER_PAGE_SIZE);
		});
		if (!covered) {
			m_page_manager.DisarmImageRead(page, TRACKER_PAGE_SIZE);
		}
	}
}

void TextureCache::DetachImageRead(ImageId id) {
	auto& image = m_slot_images[id];
	auto& stats = ReadStats();
	EXIT_IF(!image.read_armed);
	image.read_armed  = false;
	image.read_lifted = false;
	// An armed image written later over every byte of the range serves the reads (the D32 copy of the hi-Z pyramid that
	// replaces its R32 one each frame, an expanded image): nothing to copy, the pages stay closed.
	for (const auto other_id: FindImagesInRegion(image.read_address, image.read_size, false)) {
		const auto* other = m_slot_images.try_get(other_id);
		if (other != nullptr && other != &image && other->registered && other->read_armed &&
		    other->content_serial > image.content_serial && other->read_address <= image.read_address &&
		    other->read_address + other->read_size >= image.read_address + image.read_size) {
			stats.transferred.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	std::shared_ptr<std::atomic_bool> done;
	std::shared_ptr<std::atomic_bool> cancelled;
	// The collector's eviction already recorded the copy (FinishPendingEvictions frees the image once it landed; an
	// overlap or an unmap gets here first): the same copy, unless the image was written since.
	for (const auto& pending: m_pending_evictions) {
		if (pending.id == id && pending.content_serial == image.content_serial) {
			done = pending.done;
			break;
		}
	}
	if (!done && m_scheduler.Active() && image.backing.image != nullptr) {
		done      = std::make_shared<std::atomic_bool>(false);
		cancelled = std::make_shared<std::atomic_bool>(false);
		if (!DownloadImageMemory(id, true, done, cancelled)) {
			done = nullptr;
		}
	}
	if (!done) {
		// Nothing can carry the bytes (not downloadable, or the scheduler is gone): the pages open on what memory holds.
		const auto count = stats.freed.fetch_add(1, std::memory_order_relaxed);
		if (count < 16u) {
			std::fprintf(stderr, "IMAGE_READ_FREED n=%llu frame=%u addr=0x%llx %ux%u fmt=%u size=%llu reason=%s\n",
			             static_cast<unsigned long long>(count + 1), DebugPresentCount().load(),
			             static_cast<unsigned long long>(image.info.data.address), image.info.extent.width,
			             image.info.extent.height, static_cast<uint32_t>(image.info.pixel_format),
			             static_cast<unsigned long long>(image.info.data.size),
			             m_unregister_reason != nullptr ? m_unregister_reason : "other");
		}
		OpenImageReadPages(image.read_address, image.read_size, &image);
		return;
	}
	stats.detached.fetch_add(1, std::memory_order_relaxed);
	m_detached_reads.push_back({++m_detached_read_serial, image.read_address, image.read_size, m_scheduler.CurrentTick(),
	                            std::move(done), std::move(cancelled)});
}

void TextureCache::EraseDetachedRead(size_t index) {
	const auto entry = m_detached_reads[index];
	m_detached_reads.erase(m_detached_reads.begin() + static_cast<ptrdiff_t>(index));
	OpenImageReadPages(entry.address, entry.size, nullptr);
}

void TextureCache::FinishDetachedReads() {
	for (size_t i = 0; i < m_detached_reads.size();) {
		const auto& entry = m_detached_reads[i];
		if (entry.done->load(std::memory_order_acquire) ||
		    (entry.cancelled && entry.cancelled->load(std::memory_order_acquire))) {
			EraseDetachedRead(i);
		} else {
			i++;
		}
	}
}

Buffer* TextureCache::AcquireCpuReadBuffer(uint64_t size) {
	auto found = m_cpu_read_buffers.end();
	for (auto it = m_cpu_read_buffers.begin(); it != m_cpu_read_buffers.end(); ++it) {
		if ((*it)->Size() >= size && (found == m_cpu_read_buffers.end() || (*it)->Size() < (*found)->Size())) {
			found = it;
		}
	}
	std::unique_ptr<Buffer> buffer;
	if (found != m_cpu_read_buffers.end()) {
		buffer = std::move(*found);
		m_cpu_read_buffers.erase(found);
	} else {
		buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, size);
	}
	if (buffer->Mapped().empty()) {
		return nullptr;
	}
	m_cpu_read_buffers_busy.push_back(std::move(buffer));
	return m_cpu_read_buffers_busy.back().get();
}

TextureCache::CpuReadServe TextureCache::BeginCpuRead(uint64_t vaddr) {
	const auto       start = std::chrono::steady_clock::now();
	auto&            stats = ReadStats();
	CpuReadServe     serve;
	{
		std::scoped_lock lock {m_lock};
		const uint64_t   page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		// The armed images on the page and those overlapping them: a later serve of an older one must not reach bytes a
		// newer one already published.
		std::vector<ImageId> ids;
		size_t               visited = 0;
		const auto           add     = [&](uint64_t address, uint64_t size, bool pages) {
			for (const auto id: FindImagesInRegion(address, size, pages)) {
				const auto* image = m_slot_images.try_get(id);
				if (image != nullptr && image->read_armed && std::ranges::find(ids, id) == ids.end()) {
					ids.push_back(id);
				}
			}
		};
		// The detached copies on the page or under those images, and the images under the copies, to a fixed point.
		const auto add_detached = [&](uint64_t address, uint64_t size, bool pages) {
			for (const auto& entry: m_detached_reads) {
				const bool overlaps = pages ? ImagePageRangesOverlap(entry.address, entry.size, address, size)
				                            : ImageRangeOverlaps(entry.address, entry.size, address, size);
				if (overlaps && std::ranges::find(serve.detached, entry.id) == serve.detached.end()) {
					serve.detached.push_back(entry.id);
				}
			}
		};
		add(page, TRACKER_PAGE_SIZE, true);
		add_detached(page, TRACKER_PAGE_SIZE, true);
		for (size_t detached_visited = 0; visited < ids.size() || detached_visited < serve.detached.size();) {
			for (; visited < ids.size(); ++visited) {
				const auto range = m_slot_images[ids[visited]].info.data;
				add(range.address, range.size, false);
				add_detached(range.address, range.size, false);
			}
			for (; detached_visited < serve.detached.size(); ++detached_visited) {
				const auto found = std::ranges::find(m_detached_reads, serve.detached[detached_visited], &DetachedRead::id);
				add(found->address, found->size, false);
				add_detached(found->address, found->size, false);
			}
		}
		std::ranges::sort(ids, [this](ImageId a, ImageId b) {
			return m_slot_images[a].content_serial < m_slot_images[b].content_serial;
		});
		for (const auto id: ids) {
			auto&      image = m_slot_images[id];
			const auto range = image.info.data;
			auto*      buffer = image.SafeToDownload() && !image.depth_id && CanDownload(image) ? AcquireCpuReadBuffer(range.size)
			                                                                                   : nullptr;
			if (buffer == nullptr || !RecordImageDownload(id, *buffer, 0, buffer->Mapped().data())) {
				stats.orphan.fetch_add(1, std::memory_order_relaxed);
				DisarmImageRead(image);
				continue;
			}
			serve.items.push_back({id, image.content_serial, range.address, range.size, buffer});
			stats.images.fetch_add(1, std::memory_order_relaxed);
			stats.bytes.fetch_add(range.size, std::memory_order_relaxed);
			if (ImageReadTrace()) {
				auto& served  = stats.served[range.address];
				served.bytes += range.size;
				served.count++;
				served.width  = image.info.extent.width;
				served.height = image.info.extent.height;
				served.format = static_cast<uint32_t>(image.info.pixel_format);
			}
		}
		if (serve.items.empty() && serve.detached.empty() && m_page_manager.ImageReadArmed(page)) {
			stats.orphan.fetch_add(1, std::memory_order_relaxed);
			m_page_manager.DisarmImageRead(page, TRACKER_PAGE_SIZE);
		}
		// A detached copy lands with its tick (the priority thread); one recorded in the open command buffer needs the
		// submission below, as the items do.
		for (const auto detached_id: serve.detached) {
			const auto found = std::ranges::find(m_detached_reads, detached_id, &DetachedRead::id);
			serve.tick       = std::max(serve.tick, found->tick);
		}
	}
	stats.serves.fetch_add(1, std::memory_order_relaxed);
	if (!serve.items.empty() || serve.tick == m_scheduler.CurrentTick()) {
		serve.tick = m_scheduler.CurrentTick();
		m_scheduler.Flush();
	}
	stats.record_ns.fetch_add(
	    static_cast<uint64_t>(
	        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count()),
	    std::memory_order_relaxed);
	return serve;
}

void TextureCache::FinishCpuRead(CpuReadServe& serve) {
	std::scoped_lock lock {m_lock};
	// Under the lock: the image cannot be written again between the check and the publication, so nothing older than what
	// guest memory holds is published, whichever guest thread finishes first.
	for (const auto& item: serve.items) {
		auto* image = m_slot_images.try_get(item.id);
		if (image == nullptr || !image->registered || image->content_serial != item.serial || !image->SafeToDownload()) {
			ReadStats().late.fetch_add(1, std::memory_order_relaxed);
			continue;
		}
		item.buffer->Invalidate(0, item.size);
		LibKernel::Memory::WriteBacking(item.address, item.buffer->Mapped().data(), item.size);
	}
	for (const auto& item: serve.items) {
		auto* image = m_slot_images.try_get(item.id);
		if (image != nullptr && image->registered && image->content_serial == item.serial && image->SafeToDownload()) {
			DisarmImageRead(*image);
		}
		const auto owner = std::ranges::find_if(m_cpu_read_buffers_busy, [&](const auto& entry) { return entry.get() == item.buffer; });
		EXIT_IF(owner == m_cpu_read_buffers_busy.end());
		m_cpu_read_buffers.push_back(std::move(*owner));
		m_cpu_read_buffers_busy.erase(owner);
	}
	serve.items.clear();
	// The detached copies the caller waited for are in guest memory (the priority thread ran them before its tick), unless
	// the collector opened their pages first.
	for (const auto detached_id: serve.detached) {
		const auto found = std::ranges::find(m_detached_reads, detached_id, &DetachedRead::id);
		if (found != m_detached_reads.end() &&
		    (found->done->load(std::memory_order_acquire) ||
		     (found->cancelled && found->cancelled->load(std::memory_order_acquire)))) {
			EraseDetachedRead(static_cast<size_t>(found - m_detached_reads.begin()));
		}
	}
	serve.detached.clear();
}

void TextureCache::SettleImageReadLift(uint64_t address, uint64_t size, bool is_write) {
	std::scoped_lock lock {m_lock};
	const auto       begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto       end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	const auto       ids   = FindImagesInRegion(begin, end - begin, true);
	if (is_write) {
		for (const auto id: ids) {
			auto* image = m_slot_images.try_get(id);
			if (image != nullptr && image->read_armed && image->Overlaps(address, size)) {
				DisarmImageRead(*image);
			}
		}
		// A detached copy the write got ahead of is not published (its bytes are older than the write's).
		for (size_t i = 0; i < m_detached_reads.size();) {
			const auto& entry = m_detached_reads[i];
			if (ImageRangeOverlaps(entry.address, entry.size, address, size)) {
				if (entry.cancelled) entry.cancelled->store(true, std::memory_order_release);
				EraseDetachedRead(i);
			} else {
				i++;
			}
		}
	}
	for (const auto id: ids) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !image->read_armed) {
			continue;
		}
		const auto first = std::max(begin, Common::AlignDown(image->read_address, TRACKER_PAGE_SIZE));
		const auto last  = std::min(end, Common::AlignUp(image->read_address + image->read_size, TRACKER_PAGE_SIZE));
		if (first >= last) {
			continue;
		}
		m_slot_images[id].read_lifted = true;
	}
}

// KYTY_TRACE_IMAGE_SYNC_ADDR=<hex>: buffer-write invalidations, uploads and alias copies of the
// images at that address (frames 1100-1102 or the on-demand trace window, plus the first 40).
static bool TraceImageSyncAddr(uint64_t address) {
	static constinit Common::OnceValue<uint64_t> traced_once;
	const uint64_t traced = traced_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_IMAGE_SYNC_ADDR");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	});
	if (traced == 0 || address != traced) return false;
	static std::atomic_uint first {0};
	return first.fetch_add(1) < 40u || DebugStencilWindow() || DebugFrameRange();
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	if (ImageMemoryOwnerMode() != 0u) { // KYTY_FIX_IMAGE_MEMORY_OWNER
		std::unique_lock lock {m_lock};
		OwnerBufferWrite(address, size);
	}
	// Only an image whose first byte is `address` is acted on below; none is registered while the start count of its 64 KiB
	// granule (common/lookupMemo.h) is 0 (the count moves with every registration, under the lock), so the search would act
	// on nothing and it returns before the lock.
	if (ImageStartCount(address).load(std::memory_order_acquire) == 0u) return;
	std::scoped_lock lock {m_lock};
	// Only images that start exactly at the written address are acted on, and a registered image is listed in the page of
	// its first byte, so a lookup of the first page alone finds the same images in the same order.
	for (const auto id: FindImagesInRegion(address, 1, true)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
		DisarmImageRead(image); // the newest bytes are in a buffer
		if (TraceImageSyncAddr(address)) {
			std::fprintf(stderr, "IMAGE_SYNC frame=%u invalidate addr=0x%llx size=0x%llx image=%ux%u\n", DebugPresentCount().load(),
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
			             image.backing.extent.width, image.backing.extent.height);
		}
	}
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->Overlaps(address, size)) {
			if (TraceImageSyncAddr(owner->info.data.address)) {
				std::fprintf(stderr, "IMAGE_SYNC frame=%u cpu_invalidate addr=0x%llx write=0x%llx size=0x%llx image=%ux%u\n",
				             DebugPresentCount().load(), static_cast<unsigned long long>(owner->info.data.address),
				             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
				             owner->backing.extent.width, owner->backing.extent.height);
			}
			// (Image::InvalidateCpuWrite became MarkCpuDirty in upstream 8f7d74538: Overlaps is the exact byte test,
			// so its page-only branch never ran here.)
			owner->MarkCpuDirty();
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner->info.data.address;
		const auto image_end   = owner->info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	const TcReadSection lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

uint32_t TextureCache::HtileSliceClearMode() {
	static constinit Common::OnceValue<uint32_t> mode_once;
	return mode_once.Get([] {
		const char*    value = std::getenv("KYTY_FIX_HTILE_SLICE_CLEAR");
		// Default 1 since 22.10 (=0 off): =2 0 differences (2.18 M checks open world, 0.54 M gate), the far shadow-page
		// cache pages are really cleared before the game redraws them; gate 14.06 vs 13.32 fps (no cost).
		const uint32_t mode  = value == nullptr ? 1u : value[0] == '1' ? 1u : value[0] == '2' ? 2u : 0u;
		if (mode != 0u) std::fprintf(stderr, "FIX_HTILE_SLICE_CLEAR mode=%u\n", mode);
		return mode;
	});
}

std::map<uint64_t, TextureCache::MetaDataInfo>::iterator TextureCache::FindMetaInside(uint64_t address) {
	auto found = m_surface_metas.upper_bound(address);
	if (found == m_surface_metas.begin()) return m_surface_metas.end();
	--found;
	const auto& meta = found->second;
	if (meta.type != MetaDataInfo::Type::HTile || meta.slice_size == 0 || meta.layers == 0 || address <= found->first ||
	    address - found->first >= meta.slice_size * meta.layers) {
		return m_surface_metas.end();
	}
	return found;
}

bool TextureCache::IsMetaOrInside(uint64_t address) {
	const auto mode = HtileSliceClearMode();
	if (mode == 0u) return IsMeta(address);
	const TcReadSection lock {m_lock};
	return m_surface_metas.find(address) != m_surface_metas.end() || FindMetaInside(address) != m_surface_metas.end();
}

bool TextureCache::ClearMetaSlices(uint64_t address, uint64_t size, bool write, uint64_t hash) {
	const bool check = HtileSliceClearMode() == 2u;
	auto&      stats = SliceStats();
	if (check) SliceStatsTick();
	const auto found = m_surface_metas.find(address);
	if (found != m_surface_metas.end()) {
		auto& meta = found->second;
		if (write && meta.size != 0 && size > meta.size) {
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
				std::fprintf(stderr, "META_WRITE_NOT_CLEAR addr=0x%llx write=0x%llx meta=0x%llx: ordinary write\n",
				             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
				             static_cast<unsigned long long>(meta.size));
			}
			m_surface_metas.erase(found);
			return false;
		}
		meta.clear_mask = UINT64_MAX;
		const auto ss   = meta.slice_size;
		if (ss != 0 && meta.layers > 1 && size != 0 && size % ss == 0 && size / ss < meta.layers) {
			// The first layers only: the others keep their state (the old mask took every layer as cleared).
			const auto n = static_cast<uint32_t>(size / ss);
			for (uint32_t s = 0; s < n; s++) meta.SetSlice(s, true);
			const uint64_t low = n >= 64u ? UINT64_MAX : (uint64_t {1} << n) - 1u;
			meta.diverged      = ~low;
			stats.base_partial.fetch_add(1, std::memory_order_relaxed);
			stats.layers.fetch_add(n, std::memory_order_relaxed);
			static std::atomic_uint logged {0};
			if (check && logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
				std::fprintf(stderr,
				             "HTILE_SLICE_CLEAR frame=%u kind=%s-base hash=%016llx addr=0x%llx base=0x%llx size=0x%llx "
				             "slice_size=0x%llx layers=%u slices=0..%u\n",
				             DebugPresentCount().load(), write ? "write" : "rmw", static_cast<unsigned long long>(hash),
				             static_cast<unsigned long long>(address), static_cast<unsigned long long>(found->first),
				             static_cast<unsigned long long>(size), static_cast<unsigned long long>(ss), meta.layers,
				             n - 1u);
			}
		} else {
			meta.SetAllSlices(true);
			meta.diverged = 0;
		}
		return true;
	}
	const auto inside = FindMetaInside(address);
	if (inside == m_surface_metas.end()) return false;
	auto&          meta   = inside->second;
	const uint64_t ss     = meta.slice_size;
	const uint64_t offset = address - inside->first;
	const uint64_t total  = ss * meta.layers;
	if (offset % ss != 0 || size == 0 || size % ss != 0 || size > total - offset) {
		stats.partial.fetch_add(1, std::memory_order_relaxed);
		static std::atomic_uint logged {0};
		if (check && logged.fetch_add(1, std::memory_order_relaxed) < 32u) {
			std::fprintf(stderr,
			             "HTILE_SLICE_CLEAR_PARTIAL frame=%u kind=%s hash=%016llx addr=0x%llx base=0x%llx size=0x%llx "
			             "slice_size=0x%llx layers=%u: not whole layers, no change\n",
			             DebugPresentCount().load(), write ? "write" : "rmw", static_cast<unsigned long long>(hash),
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(inside->first),
			             static_cast<unsigned long long>(size), static_cast<unsigned long long>(ss), meta.layers);
		}
		return false;
	}
	const auto first = static_cast<uint32_t>(offset / ss);
	const auto n     = static_cast<uint32_t>(size / ss);
	for (uint32_t s = first; s < first + n; s++) {
		meta.SetSlice(s, true);
		if (s < 64u) meta.diverged |= uint64_t {1} << s;
	}
	stats.clears.fetch_add(1, std::memory_order_relaxed);
	stats.layers.fetch_add(n, std::memory_order_relaxed);
	static std::atomic_uint logged {0};
	if (check && logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
		std::fprintf(stderr,
		             "HTILE_SLICE_CLEAR frame=%u kind=%s hash=%016llx addr=0x%llx base=0x%llx size=0x%llx slice_size=0x%llx "
		             "layers=%u slices=%u..%u\n",
		             DebugPresentCount().load(), write ? "write" : "rmw", static_cast<unsigned long long>(hash),
		             static_cast<unsigned long long>(address), static_cast<unsigned long long>(inside->first),
		             static_cast<unsigned long long>(size), static_cast<unsigned long long>(ss), meta.layers, first,
		             first + n - 1u);
	}
	return true;
}

void TextureCache::NoteMetaReadGuard(uint64_t address, uint64_t hash) {
	if (HtileSliceClearMode() != 2u) return;
	SliceStats().guard_inside.fetch_add(1, std::memory_order_relaxed);
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
		std::fprintf(stderr, "HTILE_SLICE_CLEAR_GUARD frame=%u hash=%016llx addr=0x%llx: reads inside metadata\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(hash),
		             static_cast<unsigned long long>(address));
	}
}

bool TextureCache::ClearMetaRmw(uint64_t address, uint64_t size, uint64_t hash) {
	if (HtileSliceClearMode() == 0u) return ClearMeta(address);
	std::scoped_lock lock {m_lock};
	return ClearMetaSlices(address, size, false, hash);
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	const auto mode = HtileSliceClearMode();
	if (mode == 2u) SliceStatsTick();
	const TcReadSection lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (mode != 0u) {
		if (found == m_surface_metas.end()) return false;
		const auto& meta    = found->second;
		const bool  cleared = meta.SliceCleared(slice);
		if (mode == 2u && slice < 64u) {
			auto&      stats = SliceStats();
			const bool old   = (meta.clear_mask & (uint64_t {1} << slice)) != 0;
			stats.checked.fetch_add(1, std::memory_order_relaxed);
			if (old != cleared && (meta.diverged & (uint64_t {1} << slice)) == 0) {
				const auto n = stats.diffs.fetch_add(1, std::memory_order_relaxed);
				if (n < 16u) {
					std::fprintf(stderr, "HTILE_SLICE_CLEAR_DIFF frame=%u base=0x%llx slice=%u old=%d new=%d\n",
					             DebugPresentCount().load(), static_cast<unsigned long long>(address), slice, old ? 1 : 0,
					             cleared ? 1 : 0);
				}
			}
		}
		return cleared;
	}
	if (found == m_surface_metas.end() || slice >= 64) {
		return false;
	}
	return (found->second.clear_mask & (uint64_t {1} << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT64_MAX;
	if (HtileSliceClearMode() != 0u) {
		found->second.SetAllSlices(true);
		found->second.diverged = 0;
	}
	return true;
}

// Ghost of Yotei: the HTILE of the UI pass's depth target lies at the start of the contact-shadow mask
// (0x5062b40000, 0xa0000 bytes of HTILE in 0x870000 bytes of R8 mask). The game clears the mask every frame with a
// 0x870000-byte compute fill; taken as an HTILE clear, it was dropped, so the mask kept every earlier frame's shadows
// (dark trails and hard-edged black patches on clothes, the gate and the huts).
bool TextureCache::ClearMetaWrite(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	if (HtileSliceClearMode() != 0u) return ClearMetaSlices(address, size, true, 0);
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	if (found->second.size != 0 && size > found->second.size) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8u) {
			std::fprintf(stderr, "META_WRITE_NOT_CLEAR addr=0x%llx write=0x%llx meta=0x%llx: ordinary write\n",
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
			             static_cast<unsigned long long>(found->second.size));
		}
		m_surface_metas.erase(found);
		return false;
	}
	found->second.clear_mask = UINT64_MAX;
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (HtileSliceClearMode() != 0u) {
		if (found == m_surface_metas.end()) return false;
		auto&          meta = found->second;
		const uint64_t bit  = slice < 64u ? uint64_t {1} << slice : 0u;
		if (!is_clear && meta.SliceCleared(slice) && (slice >= 64u || (meta.clear_mask & bit) == 0)) {
			SliceStats().consumed.fetch_add(1, std::memory_order_relaxed); // only the new lookup knew this clear
		}
		meta.SetSlice(slice, is_clear);
		if (bit != 0) {
			meta.clear_mask = is_clear ? meta.clear_mask | bit : meta.clear_mask & ~bit;
			meta.diverged &= ~bit;
		}
		return true;
	}
	if (found == m_surface_metas.end() || slice >= 64) {
		return false;
	}
	if (is_clear) {
		found->second.clear_mask |= uint64_t {1} << slice;
	} else {
		found->second.clear_mask &= ~(uint64_t {1} << slice);
	}
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	OnDccMetaWrite(address, size);
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	UnregisterReason reason(m_unregister_reason, "unmap");
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
	// The detached copies under the range landed (RenderContext::UnmapMemory waited for the priority operations) or
	// are cancelled with the mapping: no page outlives it (the caller opened them).
	for (size_t i = 0; i < m_detached_reads.size();) {
		const auto& entry = m_detached_reads[i];
		if (ImageRangeOverlaps(entry.address, entry.size, address, size)) {
			if (entry.cancelled) entry.cancelled->store(true, std::memory_order_release);
			EraseDetachedRead(i);
		} else {
			i++;
		}
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	UnregisterReason reason(m_unregister_reason, "gc");
	FinishDetachedReads();
	const uint64_t   tick = m_gc_tick++;
	if (EvictReuseTrace() && tick % 64u == 0u)
		std::fprintf(stderr, "EVICT_TRACE tick=%llu reused_soon=%llu reused_gpu_mod=%llu gpu_freed=%llu\n",
		             static_cast<unsigned long long>(tick), static_cast<unsigned long long>(m_evict_reused),
		             static_cast<unsigned long long>(m_evict_reused_gpu),
		             static_cast<unsigned long long>(m_evict_gpu_freed));
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (VramGcFixEnabled() || VramBudgetEnabled()) {
		CollectGarbageFixed(tick);
		return;
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool           pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool           aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		const uint64_t age       = std::min<uint64_t>(aggressive ? 160 : pressured ? 80 : 16, tick);
		size_t         deletions = aggressive ? 40 : pressured ? 20 : 10;
		std::vector<ImageId> candidates;
		candidates.reserve(deletions);
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() == deletions;
		});
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			--deletions;
			m_gc_stats.examined++;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id || owner->bindless_pinned) {
				if (owner == nullptr || !owner->registered) {
					m_gc_stats.skip_unregistered++;
				} else if (owner->depth_id) {
					m_gc_stats.skip_association++;
				} else {
					m_gc_stats.skip_pinned++;
				}
				continue;
			}
			if (owner->heap_refs.load(std::memory_order_relaxed) != 0u) {
				// Resident heaps sample it without touching it.
				TouchImage(*owner);
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				if (safe && owner->info.IsTiled()) {
					m_gc_stats.skip_gpu_tiled++;
					continue;
				}
				if (safe && !pressured) {
					m_gc_stats.skip_not_pressured++;
					continue;
				}
				if (safe && !DownloadImageMemory(id)) {
					m_gc_stats.skip_download_failed++;
					continue;
				}
				if (safe) {
					m_gc_stats.downloaded++;
					m_gc_stats.downloaded_bytes += m_graphics.GetAllocationSize(owner->backing.allocation);
				}
			}
			m_gc_stats.freed++;
			m_gc_stats.freed_bytes += m_graphics.GetAllocationSize(owner->backing.allocation);
			FreeImage(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureEvictions);
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	m_gc_stats.runs++;
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
}

// KYTY_VRAM_GC_FIX=1: the collector of RunGarbageCollector without the blind spots of its legacy loop. That loop
// takes the 10-60 oldest LRU entries and spends its per-run count on each of them, freeable or not, and leaves the
// ones it skips where they are: once that many unfreeable images (bindless-pinned images, stencil associations,
// GPU-written tiled images) are the oldest, it frees nothing at all. It never frees a GPU-written tiled image,
// although DownloadImage writes tiled images back through the tiler. Here:
//  - entries that cannot be freed cost nothing; the ones that stay unfreeable move to the back of the LRU;
//  - a GPU-written color image whose bytes can be written back is downloaded to guest memory, and freed at a later
//    run once the download is published and the image was neither used nor written since: a later use re-creates
//    it from those bytes, as for any image that left the cache;
//  - under critical pressure, bindless-pinned images unused for the minimum age go too: the bindless table points
//    their slots back at the placeholder and resolves their keys again when a draw asks for them;
//  - the minimum age falls as the pressure rises (the legacy loop asked for older images under critical pressure).
// Depth/stencil images, stencil associations and images that cannot be downloaded stay as before.
// KYTY_VRAM_BUDGET=1: the pass runs against the live target instead of the thresholds the cache computed at start
// (see the budget branch below).
void TextureCache::CollectGarbageFixed(uint64_t tick) {
	FinishPendingEvictions();
	if (VramBudgetEnabled() && m_graphics.CanReportMemoryUsage()) {
		uint64_t target = 0, effective = 0;
		m_graphics.GetVramBudgetState(target, effective);
		if (target != 0) {
			// Images and buffers freed in command buffers the GPU has not retired yet (deferred deletion) and the
			// pending evictions are still in the measured usage.
			const uint64_t pending =
			    RecentlyFreedBytes() + PendingEvictionBytes() + m_buffer_cache.PendingReleaseBytes();
			const uint64_t used    = effective > pending ? effective - pending : 0;
			m_budget_target        = target;
			m_budget_used          = used;
			constexpr uint64_t MiB = 1024ull * 1024;
			// Below target - 512 MiB the cache keeps everything (re-uploads cost more than holding memory).
			if (used + 512 * MiB < target) {
				return;
			}
			// From there it frees the oldest images until the usage is 1 GiB under the target: images unused for
			// 64 frames, and once at the target or over it, images unused for 8 frames and bindless-pinned ones.
			const bool over = used >= target;
			m_gc_stats.runs++;
			GcPass pass;
			pass.min_age            = over ? 8 : 64;
			pass.usage              = used;
			pass.goal               = std::max<uint64_t>(target > 1024 * MiB ? target - 1024 * MiB : target / 2, 1);
			pass.allow_download     = true;
			pass.allow_pinned       = over;
			pass.max_download_bytes = over ? 256 * MiB : 128 * MiB;
			CollectImages(tick, pass);
			return;
		}
	}
	const uint64_t usage = m_total_used_memory;
	if (usage < m_trigger_gc_memory) {
		return;
	}
	const bool pressured = usage >= m_pressure_gc_memory;
	const bool critical  = usage >= m_critical_gc_memory;
	m_gc_stats.runs++;
	GcPass pass;
	pass.min_age            = critical ? 32 : pressured ? 80 : 16;
	pass.max_frees          = critical ? 60 : pressured ? 20 : 10;
	pass.usage              = usage;
	pass.allow_download     = pressured;
	pass.allow_pinned       = critical;
	pass.max_download_bytes = critical ? (256ull << 20u) : pressured ? (128ull << 20u) : 0;
	CollectImages(tick, pass);
}

void TextureCache::CollectImages(uint64_t tick, GcPass& pass) {
	// Bounds the LRU walk of one run (the walk itself is cheap; freeing scans every image for associations).
	constexpr size_t     MaxExamined = 8192;
	std::vector<ImageId> frees;     // nothing to keep: freed now
	std::vector<ImageId> downloads; // downloaded now, freed by a later run (FinishPendingEvictions)
	std::vector<ImageId> to_back;   // cannot be freed: moved to the back of the LRU
	uint64_t             download_bytes = 0;
	size_t               examined       = 0;
	const auto           enough         = [&] {
        return pass.goal != 0 ? pass.usage < pass.goal : frees.size() + downloads.size() >= pass.max_frees;
	};
	// Deleting a depth image deletes its stencil associations: finish the LRU walk first.
	m_lru_cache.ForEachItemBelow(tick - std::min(pass.min_age, tick), [&](ImageId id) {
		if (enough() || examined == MaxExamined) {
			return true;
		}
		examined++;
		m_gc_stats.examined++;
		auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered) {
			m_gc_stats.skip_unregistered++;
			return false;
		}
		if (owner->eviction_pending) {
			return false; // its download is being published (FinishPendingEvictions frees it)
		}
		if (owner->depth_id) {
			m_gc_stats.skip_association++; // freed with its depth image
			to_back.push_back(id);
			return false;
		}
		// The list path of bindless tables over 1 MiB: a pinned image such a table drew within min_age frames is not old
		// (its left-out touch); moved to the back as a pinned one is.
		if (owner->bindless_pinned &&
		    (!pass.allow_pinned || (bindless_fast_keep && bindless_fast_keep(id, pass.min_age)))) {
			m_gc_stats.skip_pinned++;
			to_back.push_back(id);
			return false;
		}
		if (owner->info.HasStencil()) {
			// Its stencil plane, a separate guest surface, lives in this image and in association records.
			m_gc_stats.skip_stencil++;
			to_back.push_back(id);
			return false;
		}
		const uint64_t bytes = owner->AccountedSize();
		if (owner->IsGpuModified() && owner->SafeToDownload()) {
			// The image holds the newest bytes of its memory: they go to guest memory before it goes.
			if (!pass.allow_download) {
				m_gc_stats.skip_not_pressured++;
				return false;
			}
			if (!CanEvictWithDownload(*owner)) {
				m_gc_stats.skip_no_download++;
				to_back.push_back(id);
				return false;
			}
			if (download_bytes + bytes > pass.max_download_bytes) {
				m_gc_stats.skip_download_budget++;
				return false;
			}
			download_bytes += bytes;
			downloads.push_back(id);
		} else if (owner->IsGpuModified() && !owner->IsBufferModified() && !owner->IsDefinitelyCpuDirty()) {
			// GPU-written and only maybe CPU-dirty (the CPU wrote a page it shares, perhaps not its bytes): its next
			// use settles that by hash; until then its bytes may be the newest, so it stays.
			m_gc_stats.skip_no_download++;
			to_back.push_back(id);
			return false;
		} else {
			// Not GPU-written, or guest memory or a buffer holds newer bytes than the image (its next use would
			// reload it from them): nothing to keep.
			frees.push_back(id);
		}
		pass.usage -= std::min(pass.usage, bytes);
		return false;
	});
	for (const auto id: to_back) {
		if (auto* image = m_slot_images.try_get(id); image != nullptr && image->registered) {
			TouchImage(*image);
		}
	}
	for (const auto id: frees) {
		const auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered) {
			continue; // freed with an earlier one
		}
		FreeCollectedImage(id);
	}
	for (const auto id: downloads) {
		auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered) {
			continue;
		}
		auto done = std::make_shared<std::atomic_bool>(false);
		if (!DownloadImageMemory(id, true, done)) {
			m_gc_stats.skip_download_failed++;
			TouchImage(*owner);
			continue;
		}
		owner->eviction_pending = true;
		const uint64_t bytes    = m_graphics.GetAllocationSize(owner->backing.allocation);
		m_pending_evictions.push_back({id, owner->content_serial, m_gc_tick, std::move(done), bytes});
		m_gc_stats.downloaded++;
		m_gc_stats.downloaded_bytes += bytes;
	}
}

// KYTY_VRAM_GC_FIX: frees the images whose download a previous run recorded, once the bytes are in guest memory.
void TextureCache::FinishPendingEvictions() {
	size_t kept = 0;
	for (size_t i = 0; i < m_pending_evictions.size(); i++) {
		auto& pending = m_pending_evictions[i];
		auto* owner   = m_slot_images.try_get(pending.id);
		if (owner == nullptr || !owner->registered) {
			continue; // freed meanwhile (unmap, overlap)
		}
		if (!pending.done->load(std::memory_order_acquire)) {
			if (kept != i) {
				m_pending_evictions[kept] = std::move(pending);
			}
			kept++;
			continue;
		}
		owner->eviction_pending = false;
		// Used or written since the download was recorded: it stays (guest memory holds the older copy, as for
		// any written-back image).
		// The list path of bindless tables over 1 MiB: a pinned image such a table drew since then is used as well.
		if (owner->content_serial != pending.content_serial || owner->lru_touch_tick >= pending.tick ||
		    (owner->bindless_pinned && bindless_fast_keep &&
		     bindless_fast_keep(pending.id, m_gc_tick - std::min(m_gc_tick, pending.tick)))) {
			m_gc_stats.evict_cancelled++;
			continue;
		}
		DisarmImageRead(*owner); // its bytes, unchanged since, are in guest memory
		FreeCollectedImage(pending.id);
	}
	m_pending_evictions.resize(kept);
}

void TextureCache::FreeCollectedImage(ImageId id) {
	auto&          image = m_slot_images[id];
	const uint64_t bytes = m_graphics.GetAllocationSize(image.backing.allocation);
	if (image.bindless_pinned) {
		m_gc_stats.freed_pinned++;
	}
	m_gc_stats.freed++;
	m_gc_stats.freed_bytes += bytes;
	NoteFreedBytes(bytes);
	image.eviction_pending = false;
	if (EvictReuseTrace() && !image.info.data.Empty()) {
		if (m_evicted_at.size() >= (size_t {1} << 20u)) m_evicted_at.clear();
		m_evicted_at[image.info.data.address] = m_gc_tick;
	}
	FreeImage(id);
}

// KYTY_VRAM_BUDGET: an image freed now is destroyed once the GPU retired the command buffer recording at this tick
// (deferred deletion); until then the measured usage still holds it. Bytes are kept per tick until the tick is free.
void TextureCache::NoteFreedBytes(uint64_t bytes) {
	if (!VramBudgetEnabled()) {
		return;
	}
	const uint64_t tick = m_scheduler.CurrentTick();
	if (!m_releasing.empty() && m_releasing.back().tick == tick) {
		m_releasing.back().bytes += bytes;
	} else {
		m_releasing.push_back({tick, bytes});
	}
}

uint64_t TextureCache::RecentlyFreedBytes() {
	uint64_t total = 0;
	size_t   kept  = 0;
	for (size_t i = 0; i < m_releasing.size(); i++) {
		const auto entry = m_releasing[i];
		if (m_scheduler.IsFree(entry.tick)) {
			continue;
		}
		m_releasing[kept++] = entry;
		total += entry.bytes;
	}
	m_releasing.resize(kept);
	return total;
}

uint64_t TextureCache::PendingEvictionBytes() const {
	uint64_t total = 0;
	for (const auto& pending: m_pending_evictions) {
		total += pending.bytes;
	}
	return total;
}

uint64_t TextureCache::PendingReleaseBytes() {
	std::scoped_lock lock {m_lock};
	return RecentlyFreedBytes() + PendingEvictionBytes();
}

// KYTY_VRAM_BUDGET: before a new image is created over the target, images unused for 4+ collector ticks that hold
// nothing the guest memory lacks (not GPU-written, not pinned, no stencil plane) are freed, oldest first. Images
// overlapping the new one are left to the overlap resolution that is creating it. Caller holds m_lock.
void TextureCache::MakeRoomForImage(const ImageInfo& info) {
	constexpr uint64_t MinAge  = 4;
	constexpr uint64_t Slack   = 64ull << 20u; // frees a little more than needed: fewer scans
	constexpr size_t   MaxScan = 4096;
	if (!m_graphics.CanReportMemoryUsage() || m_gc_tick <= MinAge) {
		return;
	}
	uint64_t target = 0, effective = 0;
	m_graphics.GetVramBudgetState(target, effective);
	if (target == 0) {
		return;
	}
	const uint64_t pending =
	    RecentlyFreedBytes() + PendingEvictionBytes() + m_buffer_cache.PendingReleaseBytes();
	const uint64_t used = effective > pending ? effective - pending : 0;
	// Over the target: a scan that found nothing to free at this collector tick is not repeated for every image.
	if (used + info.data.size <= target || m_make_room_dry_tick == m_gc_tick) {
		return;
	}
	uint64_t             need     = used + info.data.size - target + Slack;
	size_t               examined = 0;
	std::vector<ImageId> victims;
	std::vector<ImageId> fast_touched; // the list path of bindless tables over 1 MiB (below)
	m_lru_cache.ForEachItemBelow(m_gc_tick - MinAge, [&](ImageId id) {
		if (need == 0 || examined == MaxScan) {
			return true;
		}
		examined++;
		const auto* owner = m_slot_images.try_get(id);
		// The list path of bindless tables over 1 MiB: a pinned image such a table drew within MinAge frames gets the touch
		// it left out after the scan (to the back, out of the next scans' way); it is skipped as pinned anyway.
		if (owner != nullptr && owner->registered && owner->bindless_pinned && bindless_fast_keep &&
		    bindless_fast_keep(id, MinAge)) {
			fast_touched.push_back(id);
			return false;
		}
		if (owner == nullptr || !owner->registered || owner->depth_id || owner->bindless_pinned ||
		    owner->eviction_pending || owner->IsGpuModified() || owner->info.HasStencil() || owner->usage.video_out ||
		    owner->Overlaps(info.data.address, info.data.size)) {
			return false;
		}
		victims.push_back(id);
		need -= std::min(need, owner->AccountedSize());
		return false;
	});
	for (const auto id: fast_touched) {
		if (auto* image = m_slot_images.try_get(id); image != nullptr && image->registered) TouchImage(*image);
	}
	if (victims.empty()) {
		m_make_room_dry_tick = m_gc_tick;
		return;
	}
	UnregisterReason reason(m_unregister_reason, "budget");
	for (const auto id: victims) {
		const auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered) {
			continue;
		}
		m_gc_stats.alloc_evicted++;
		m_gc_stats.alloc_evicted_bytes += m_graphics.GetAllocationSize(owner->backing.allocation);
		FreeCollectedImage(id);
	}
}

// KYTY_VRAM_GC_FIX: a GPU-written image the collector may free after downloading it. Color images only: a depth
// image's HTILE state and stencil plane are not in the bytes a download writes back, and a video-out image may be
// on screen.
bool TextureCache::CanEvictWithDownload(const Image& image) const {
	return !image.depth_id && !image.info.IsDepth() && !image.info.HasStencil() && !image.usage.video_out &&
	       image.info.samples == 1 && image.backing.samples == 1 && image.backing.image != nullptr &&
	       CanDownload(image);
}

} // namespace Libs::Graphics
