#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_H_

#include "common/alignment.h"
#include "common/assert.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <compare>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
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
	bool render_target = false;
	bool video_out     = false;
};

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
	             std::optional<ImageSubresourceRange> range, CommandRecorder command_buffer);
	void Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	            uint64_t size);
	void Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
	              uint64_t size);
	void CopyImage(Image& source);
	void Resolve(Image& source, const ImageSubresourceRange& source_range,
	             const ImageSubresourceRange& destination_range);
	void CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler);
	void CopyMip(Image& source, uint32_t mip, uint32_t layer);

	[[nodiscard]] static uint64_t CpuDirtyEpoch() noexcept {
		return s_cpu_dirty_epoch.load(std::memory_order_relaxed);
	}

	void MarkCpuDirty() {
		s_cpu_dirty_epoch.fetch_add(1, std::memory_order_relaxed);
		m_cpu_dirty       = true;
		m_maybe_cpu_dirty = false;
		DropPartialDirty();
		HeapChanged();
	}

	[[nodiscard]] bool CanTakePartialDirty() const noexcept {
		return !m_maybe_cpu_dirty && (!m_cpu_dirty || m_partial_dirty);
	}
	void InvalidateCpuWritePartial(uint64_t begin, uint64_t end) {
		if (!CanTakePartialDirty() || begin >= end) {
			EXIT("image cannot take a partial CPU write\n");
		}
		if (!m_cpu_dirty) {
			m_cpu_dirty     = true;
			m_partial_dirty = true;
			m_dirty_ranges.clear();
		}
		AddRange(m_dirty_ranges, begin, end);
		s_cpu_dirty_epoch.fetch_add(1, std::memory_order_relaxed);
		HeapChanged();
	}
	[[nodiscard]] bool IsPartiallyCpuDirty() const noexcept { return m_cpu_dirty && m_partial_dirty; }
	[[nodiscard]] const std::vector<std::pair<uint64_t, uint64_t>>& CpuDirtyRanges() const noexcept {
		return m_dirty_ranges;
	}
	void DropPartialDirty() noexcept {
		m_partial_dirty = false;
		m_dirty_ranges.clear();
	}
	static void AddRange(std::vector<std::pair<uint64_t, uint64_t>>& ranges, uint64_t begin,
	                     uint64_t end) {
		auto at = ranges.begin();
		while (at != ranges.end() && at->second < begin) {
			++at;
		}
		auto last = at;
		while (last != ranges.end() && last->first <= end) {
			begin = std::min(begin, last->first);
			end   = std::max(end, last->second);
			++last;
		}
		at = ranges.erase(at, last);
		ranges.insert(at, {begin, end});
	}

	[[nodiscard]] bool IsCpuDirty() const { return m_cpu_dirty || m_maybe_cpu_dirty; }
	[[nodiscard]] bool IsDefinitelyCpuDirty() const { return m_cpu_dirty; }
	[[nodiscard]] bool IsMaybeCpuDirty() const { return m_maybe_cpu_dirty; }
	void               MarkMaybeCpuDirty() {
		if (!IsCpuDirty()) {
			m_maybe_cpu_hash  = HashGuestEdges();
			m_maybe_cpu_dirty = true;
			s_cpu_dirty_epoch.fetch_add(1, std::memory_order_relaxed);
			HeapChanged();
		}
	}
	void ResolveMaybeCpuHash(uint64_t hash) {
		m_maybe_cpu_dirty = false;
		m_cpu_dirty       = hash != m_maybe_cpu_hash;
		StateChanged();
	}

	void RefreshComplete() {
		m_cpu_dirty       = false;
		m_maybe_cpu_dirty = false;
		DropPartialDirty();
		StateChanged();
	}

	[[nodiscard]] bool IsGpuModified() const noexcept { return m_gpu_modified; }
	void               MarkGpuModified() noexcept {
		if (!m_gpu_modified) {
			m_gpu_modified = true;
			StateChanged();
		}
	}
	void ClearGpuModified() noexcept {
		if (m_gpu_modified) {
			m_gpu_modified = false;
			StateChanged();
		}
	}

	[[nodiscard]] bool IsBufferModified() const noexcept { return m_buffer_modified; }
	void               MarkBufferModified() noexcept {
		s_cpu_dirty_epoch.fetch_add(1, std::memory_order_relaxed);
		m_buffer_modified = true;
		HeapChanged();
	}
	void ClearBufferModified() noexcept {
		if (m_buffer_modified) {
			m_buffer_modified = false;
			StateChanged();
		}
	}

	[[nodiscard]] bool Overlaps(uint64_t address, uint64_t size) const noexcept {
		return info.data.address < address + size && address < info.data.End();
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
	struct HeapChange {
		GuestRange data;
		uint64_t   serial = 0;
	};

	void StateChanged() const noexcept {
		state_epoch.fetch_add(1u, std::memory_order_relaxed);
		if (version != nullptr) {
			version->fetch_add(1u, std::memory_order_release);
		}
	}
	static inline std::atomic<uint64_t> state_epoch {0};
	[[nodiscard]] static uint64_t       StateEpoch() noexcept { return state_epoch.load(std::memory_order_relaxed); }
	void HeapChanged() const {
		StateChanged();
		if (heap_refs.load(std::memory_order_relaxed) != 0u) {
			std::scoped_lock lock {heap_mutex};
			heap_changes.push_back({info.data, serial});
		}
	}

	static inline std::mutex              heap_mutex;
	static inline std::vector<HeapChange> heap_changes;
	static inline std::atomic<uint64_t>   serials {0};
	static inline std::atomic<uint64_t>   gpu_write_sequences {0};

	const uint64_t   serial = serials.fetch_add(1u, std::memory_order_relaxed) + 1u;
	uint64_t         validity = 0;
	ImageInfo        info;
	VulkanImage      backing;
	std::vector<CachedImageView> views;
	ImageUsage       usage;
	ImageBinding     binding;
	bool             registered     = false;
	uint64_t         track_addr     = 0;
	uint64_t         track_addr_end = 0;
	std::vector<std::pair<uint64_t, uint64_t>> untracked_holes;
	bool             partial_candidate = false;
	ImageId          depth_id {};
	uint64_t         write_count        = 0;
	uint64_t         gpu_write_sequence = 0;
	uint64_t         candidate_clear_key = 0;
	bool             is_depth_copy      = false;
	ImageId          depth_copy_source {};
	uint64_t         depth_copy_writes  = UINT64_MAX;
	bool             is_stencil_copy     = false;
	bool             retile_stencil_copy = false;
	uint64_t         stencil_copy_writes = 0;
	std::vector<ImageId> stencil_ids;
	ImageSubresourceRange stencil_subresources;
	uint32_t         registrations      = 0;
	uint64_t         tick_accessed_last = 0;
	size_t           lru_id             = 0;
	std::atomic<uint32_t> heap_refs {0};
	uint64_t         lru_tick           = 0;
	uint32_t         table_pins         = 0;
	uint64_t         readback_begin     = 0;
	uint64_t         readback_end       = 0;
#if defined(ANYPS5_KYTY)
	std::array<uint64_t, 4> target_frames {};
#endif
	std::atomic<uint32_t>* version = nullptr;

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
	bool              m_partial_dirty   = false;
	std::vector<std::pair<uint64_t, uint64_t>> m_dirty_ranges;
	bool              m_gpu_modified    = false;
	bool              m_buffer_modified = false;

	inline static std::atomic<uint64_t> s_next_serial {0};
	inline static std::atomic<uint64_t> s_cpu_dirty_epoch {1};
};

namespace ImageOps {

void                                 Validate(const ImageInfo& info);
[[nodiscard]] Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element);

}

}

#endif
