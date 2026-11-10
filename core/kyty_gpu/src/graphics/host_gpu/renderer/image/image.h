#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <atomic>
#include <compare>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <atomic>
#include <vector>

namespace Libs::Graphics {


class Buffer;
class CommandScheduler;
class TileManager;
struct ImageTestAccess;

using ImageId = Common::SlotId;

struct CachedImageView {
	ImageViewInfo info;
	vk::ImageView view = nullptr;
};

struct ImageUsage {
	// texture / depth_target: kept although nothing in the renderer reads them (upstream 8f7d74538 removed them); our
	// tests check them.
	bool texture       = false;
	bool storage       = false;
	bool render_target = false;
	bool depth_target  = false;
	bool video_out     = false;
};

// The bindless layout memo (CommitBindings): bumped whenever the layout of an image with Image::bindless_watched changes.
inline std::atomic<uint64_t> g_bindless_layout_epoch {1};
// Bindless table refresh events (on with the tables): the images with
// Image::bindless_watched whose layout changed, in order (entry n is position base + n); a bindless table lists for a
// draw's layout walk only those of its images logged since its last listing (RenderExecutor::PrepareBindlessTable).
// Trimmed once per frame (base rises).
struct BindlessLayoutLog {
	std::mutex           mutex;
	std::vector<ImageId> ids;
	uint64_t             base = 0;
};
inline BindlessLayoutLog g_bindless_layout_log;
inline std::atomic<bool> g_bindless_layout_log_on {false};

struct ImageBinding {
	vk::ImageLayout  attachment_layout = vk::ImageLayout::eUndefined;
	vk::AccessFlags2 attachment_access;
	bool             is_bound      = false;
	bool             is_target     = false;
	bool             needs_rebind  = false;
	bool             force_general = false;
	bool             shader_write  = false;
};


class Image final {
public:
	Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& info);
	~Image();
	KYTY_CLASS_NO_COPY(Image);

	[[nodiscard]] vk::ImageView FindView(const ImageViewInfo& view_info);
	using Barriers = std::vector<vk::ImageMemoryBarrier2>;
	[[nodiscard]] Barriers GetBarriers(vk::ImageLayout                      destination_layout,
	                                   vk::AccessFlags2                     destination_access,
	                                   vk::PipelineStageFlags2              destination_stage,
	                                   std::optional<ImageSubresourceRange> range);
	void Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
	             std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	void MarkCpuDirty() {
		m_cpu_dirty       = true;
		m_maybe_cpu_dirty = false;
		HeapChanged();
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!IsCpuDirty()) {
			m_maybe_cpu_hash  = HashGuestEdges();
			m_maybe_cpu_dirty = true;
			HeapChanged();
		}
	}
	[[nodiscard]] bool ResolveMaybeCpuHash(uint64_t hash) {
		if (!m_maybe_cpu_dirty || m_cpu_dirty) {
			EXIT("image cannot resolve maybe-dirty hash\n");
		}
		m_maybe_cpu_dirty = false;
		m_cpu_dirty |= hash != m_maybe_cpu_hash;
		return m_cpu_dirty;
	}

	void RefreshComplete() {
		if (!IsCpuDirty()) {
			EXIT("clean image cannot complete a refresh\n");
		}
		m_cpu_dirty       = false;
		m_maybe_cpu_dirty = false;
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept {
		m_gpu_modified = true;
		content_serial = NextContentSerial();
	}
	// KYTY_WORKAROUND_ALIAS_SYNC: the write that produced this image's contents (monotonic over all
	// images), so same-address aliases of another extent can take over newer data.
	uint64_t content_serial = 0;
	// KYTY_WORKAROUND_STENCIL_PLANE_SYNC: content_serial of the last shader (storage) write, not of host-side copies.
	uint64_t shader_write_serial = 0;
	// content_serial last copied into a buffer that reads this image's memory.
	uint64_t buffer_sync_serial = 0;
	// KYTY_FIX_SHARED_IMAGE_MEMORY: content_serial of the newest overlapping image already merged into this one.
	uint64_t alias_synced_serial = 0;
	// The bytes [read_address, read_address + read_size) of this image are not in guest memory yet and its pages are under
	// the image read watch (TextureCache::ArmImageRead).
	bool     read_armed   = false;
	bool     read_lifted  = false; // armed, but a page of it was opened without a serve (the next write closes it)
	uint64_t read_address = 0;
	uint64_t read_size    = 0;
	// KYTY_WORKAROUND_PRESERVE_STENCIL: the stencil was restored on the GPU; skip the guest-memory refresh.
	bool stencil_restored = false;
	// KYTY_WORKAROUND_PRESERVE_STENCIL: guest stencil plane currently held by this depth image's stencil aspect.
	uint64_t stencil_plane_address = 0;
	static uint64_t NextContentSerial() noexcept { return ++s_content_serial; }
	static inline uint64_t s_content_serial = 0;
	void               ClearGpuModified() noexcept { m_gpu_modified = false; }

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		m_buffer_modified = true;
		HeapChanged();
	}
	void               ClearBufferModified() noexcept { m_buffer_modified = false; }

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size,
	                            bool pages = false) const noexcept {
		return pages ? ImagePageRangesOverlap(info.data.address, info.data.size, address, size)
		             : ImageRangeOverlaps(info.data.address, info.data.size, address, size);
	}
	[[nodiscard]] bool SafeToDownload() const noexcept {
		return IsGpuModified() && !IsBufferModified() && !IsCpuDirty();
	}
	[[nodiscard]] bool IsTracked() const noexcept { return track_addr != 0 && track_addr_end != 0; }
	[[nodiscard]] uint64_t AccountedSize() const noexcept {
		return backing.image == nullptr ? 0 : Common::AlignUp(info.data.size, 1024);
	}
	[[nodiscard]] uint64_t HashGuestEdges() const;
	[[nodiscard]] vk::ImageLayout ReadLayout() const noexcept {
		return info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
		                      : vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	// Resident image heaps sample an image without rediscovering it until one changes its
	// contents, tracking, identity or layout.
	void HeapChanged() const noexcept {
		// The bindless slot journal: a table slot's replay check reads the serial of the image it pins (journaled first).
		if (bindless_pinned && g_table_slot_journal_on.load(std::memory_order_relaxed))
			JournalTableImage((uint64_t {self_id.generation} << 32u) | self_id.index);
		change_serial++; // bindless table refresh events
		if (heap_refs.load(std::memory_order_relaxed) != 0u) {
			heap_epoch.fetch_add(1u, std::memory_order_relaxed);
		}
	}
	// Bindless table refresh events (E1 fast version): bumped with every change HeapChanged reports (CPU-dirty
	// marks, buffer writes, untracking, leaving the read layout), whether a heap references the image or not.
	mutable uint32_t change_serial = 0;

	static inline std::atomic<uint64_t> heap_epoch {0};
	static inline std::atomic<uint64_t> serials {0};

	// Unique per image, unlike ImageId: freed slots are reused.
	const uint64_t   serial = serials.fetch_add(1u, std::memory_order_relaxed) + 1u;
	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	// Referenced by a bindless table slot: the collector keeps it, and unregistering it tells the
	// table (TextureCache::on_bindless_unregister) to repoint the slot.
	bool             bindless_pinned = false;
	// Checked by CommitBindings' bindless layout walk; its layout changes bump
	// g_bindless_layout_epoch (GetBarriers, the only writer of backing.state).
	bool             bindless_watched = false;
	ImageId          self_id {}; // bindless table refresh events: its own id (set by TextureCache when inserted)
	mutable uint32_t query_epoch    = 0;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	ImageId          depth_id {};
	// Stencil associations that refer to this depth image.
	std::vector<ImageId> stencil_ids;
	// The current stencil plane's mapping into the depth image, also retained by its association.
	ImageSubresourceRange stencil_subresources;
	uint64_t         tick_accessed_last = 0;
	size_t           lru_id             = 0;
	// TextureCache::TouchImage: the GC tick of the image's last LRU touch (or registration); a touch
	// at the same tick changes nothing. ~0: not registered yet.
	uint64_t         lru_touch_tick     = ~uint64_t {0};
	// TextureCache::FindTextureReplay: 1 + the image-page epoch of the image's first page when
	// SyncExtentAliases found no same-address alias it could copy from (0: not known).
	uint64_t         alias_free_epoch   = 0;
	// KYTY_FIX_SHARED_IMAGE_MEMORY (TextureCache::SyncSharedImageMemory): the other registered images overlapping this
	// image's memory, valid while shared_overlap_epoch is 1 + the exact image-page epoch sum of that memory (0: not known).
	std::vector<ImageId> shared_overlaps;
	uint64_t             shared_overlap_epoch = 0;
	// KYTY_FIX_IMAGE_MEMORY_OWNER: another image or guest memory took bytes of this image's memory since its last sync
	// (TextureCache::OwnerSyncImage has work to check); set under m_lock exclusively, with HeapChanged().
	bool                 owner_stale = false;
	// KYTY_FIX_IMAGE_MEMORY_OWNER, the proofs that a sync has nothing to do (under m_lock exclusively):
	// owner_disturbed rises with every holder change over its memory by others and every read of its bytes by others
	// (OwnerDisturb, pulls and copies from it); owner_run_disturbed is its value at the image's own last write, so equal
	// values mean its draw run (or storage write) is untouched since. owner_memo_rect: that write's draw rectangle.
	// owner_sole: it holds every byte of its memory and no other image holds any of them. owner_adopt_registry /
	// owner_alone_registry: the image registry epoch at which its overlap set was adopted / found empty.
	uint64_t                owner_disturbed      = 0;
	uint64_t                owner_run_disturbed  = 0;
	std::array<int32_t, 5>  owner_memo_rect      {}; // valid flag, x0, y0, x1, y1 (valid 0: no memo)
	bool                    owner_sole           = false;
	uint64_t                owner_adopt_registry = 0;
	uint64_t                owner_alone_registry = 0;
	// KYTY_FIX_SHARED_IMAGE_MEMORY=3/4 (probe): the image was filled with the probe pattern (or, =4, would have been)
	// when other images had written its memory; the next binds look for it (TextureCache::ProbeSharedImageRead).
	uint8_t              probe_pending = 0;
	uint8_t              probe_checks  = 0;
	// TextureCache::RegisterImage: the AccountedSize() counted for it while registered (KYTY_TRACE_PERF images=).
	uint64_t         registered_bytes   = 0;
	// KYTY_VRAM_GC_FIX: the collector recorded its download and frees it once the bytes are in guest memory.
	bool             eviction_pending   = false;
	// The plan flags (S2 step B.3): draws being prepared that bind this image (as a texture or target); their
	// flags are in the draw's plan until it is committed. Counts as "bound" where the cache asks (needs_rebind).
	uint16_t         plan_uses          = 0;
	// Resident image heap elements that select this image.
	std::atomic<uint32_t> heap_refs {0};
	// Texture relocation: steady-clock ms of its last move to another VMA block (0: never), so a texture is not moved back and
	// forth between blocks.
	uint64_t         texmove_ms         = 0;

private:
	friend struct ImageTestAccess;

	[[nodiscard]] static vk::ImageAspectFlags FullAspectMask(vk::Format format) noexcept;
	[[nodiscard]] static uint32_t             CopyRows(uint64_t row_size, uint32_t rows,
	                                                   uint64_t capacity) noexcept;
	[[nodiscard]] static std::pair<uint32_t, uint32_t>
	SanitizeCopyLayers(const Image& source, const Image& destination, uint32_t depth);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;
	uint64_t          m_maybe_cpu_hash  = 0;
	bool              m_cpu_dirty       = false;
	bool              m_maybe_cpu_dirty = false;
	bool              m_gpu_modified    = false;
	bool              m_buffer_modified = false;
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

} // namespace ImageOps

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
