#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/smallVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
}

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
		bool          heap = false;
		bool candidate = false;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	[[nodiscard]] ImageId FindImageSpeculative(ImageDesc& desc, bool exact_format = false,
	                                           ImageId* depth_source = nullptr);
	[[nodiscard]] ImageId FindDepthCopy(ImageId source_id, const ImageInfo& requested);
	[[nodiscard]] bool DepthCopyStale(const Image& image) const;
	[[nodiscard]] bool DepthCopyCurrent(const Image& image) const;
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] ImageId CopyDepthForFeedback(ImageId depth_id, const ImageSubresourceRange& range,
	                                           vk::ImageAspectFlags aspects);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc,
	                                            ImageId* stencil = nullptr);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);
	void CollectHotPagesOf(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	bool               InvalidateMemory(uint64_t address, uint64_t size);
	void               FlushLazyReadback(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	[[nodiscard]] bool CopyToImage(uint64_t address, uint64_t source, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);
	[[nodiscard]] bool HasGpuWrittenImages(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool CoversMeta(uint64_t address, uint64_t size);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void MapMemory(uint64_t address, uint64_t size);
	[[nodiscard]] bool HasPartialCandidates() const noexcept {
		return m_partial_candidates.load(std::memory_order_relaxed) != 0;
	}
	void ProcessDownloadImages();
	[[nodiscard]] bool StorageReadbackEnabled() const { return m_readback_storage_images; }
	[[nodiscard]] bool WriteBackImages();
	[[nodiscard]] uint64_t WriteBackTick(uint64_t address, uint64_t size);
	[[nodiscard]] uint64_t ImageSetEpoch() const { return m_image_set_epoch; }
	[[nodiscard]] uint64_t ImagesEpoch() const {
		return Image::StateEpoch() + m_image_set_epoch + m_meta_epoch.load(std::memory_order_relaxed) +
		       Image::CpuDirtyEpoch();
	}
	std::atomic<uint64_t> m_meta_epoch {0};
	void                    ForgetSettledMetadata(uint64_t vaddr, uint64_t size);
	void                    NoteSettledMetadata(GuestRange range);
	static constexpr size_t MaxSettledMetadata = 256;
	std::mutex              m_settled_lock;
	std::vector<GuestRange> m_settled_ranges;
	std::atomic<uint64_t>   m_settled_begin {UINT64_MAX};
	std::atomic<uint64_t>   m_settled_end {0};
	std::atomic<uint64_t>   m_color_clear_epoch {1};
	static constexpr uint32_t                 SettledGranuleBits = 16;
	std::array<std::atomic<uint64_t>, 64> m_settled_granules {};
	[[nodiscard]] bool MaybeSettled(uint64_t vaddr, uint64_t size) const {
		const auto first = vaddr >> SettledGranuleBits;
		const auto last  = (vaddr + size - 1) >> SettledGranuleBits;
		if (last - first >= 64) return true;
		for (auto granule = first; granule <= last; ++granule) {
			const auto bit = granule & 4095u;
			if (((m_settled_granules[bit >> 6u].load(std::memory_order_acquire) >> (bit & 63u)) & 1u) != 0) return true;
		}
		return false;
	}
	[[nodiscard]] uint64_t ResolutionEpoch() const { return m_image_set_epoch; }
	[[nodiscard]] uint64_t StartEpoch() const { return m_image_set_epoch; }
	[[nodiscard]] uint64_t GcTick() const noexcept { return m_gc_tick; }
	[[nodiscard]] uint64_t CurrentFrame() const noexcept { return m_gc_tick; }
	[[nodiscard]] bool     RegistrationsSince(uint64_t address, uint64_t size, uint64_t epoch) {
        return ImageEpochInRegion(address, size) > epoch;
	}
	[[nodiscard]] bool RegistrationsOverlapSince(uint64_t address, uint64_t size, uint64_t epoch, bool& known) {
		known  = ForEachImageSetChangeSince(epoch, [](uint64_t, uint64_t) {});
		if (!known) return true;
		bool overlap = false;
		(void)ForEachImageSetChangeSince(epoch, [&](uint64_t begin, uint64_t length) {
			overlap = overlap || (begin < address + size && address < begin + length);
		});
		return overlap;
	}
	[[nodiscard]] bool IsSampledImageCurrent(ImageId id, const ImageDesc& desc, bool storage_dcc);
	[[nodiscard]] bool ColorClearSettled(const ImageDesc& desc);
	[[nodiscard]] bool ColorClearSettled(const ImageDesc& desc, uint64_t& memo) {
		const auto epoch = m_color_clear_epoch.load(std::memory_order_acquire);
		if (memo == epoch) return true;
		const bool settled = ColorClearSettled(desc);
		if (settled) memo = epoch;
		return settled;
	}
	void NoteMetadataWrite(uint64_t vaddr, uint64_t size) {
		if (vaddr < m_settled_end.load(std::memory_order_acquire) &&
		    vaddr + size > m_settled_begin.load(std::memory_order_acquire) && MaybeSettled(vaddr, size)) {
			ForgetSettledMetadata(vaddr, size);
		}
	}
	[[nodiscard]] bool IsSampledDepthCurrent(ImageId, const ImageDesc&, uint64_t) { return false; }
	[[nodiscard]] static bool ViewFitsImage(const ImageDesc&) { return true; }
	void                      NoteMetaUse(uint64_t, uint32_t) {}
	[[nodiscard]] bool IsSampledImageCurrent(ImageId id, const ImageDesc& desc);
	[[nodiscard]] static bool RefreshPending(const Image& image) {
		return image.IsBufferModified() || image.IsCpuDirty();
	}
	[[nodiscard]] bool HasImageStartingAt(uint64_t address);
	[[nodiscard]] bool IsTargetCurrent(ImageId id, const ImageInfo& requested);
	[[nodiscard]] uint64_t ImageEpochInRegion(uint64_t address, uint64_t size);
	static constexpr uint32_t ImagePageBits = 20;
	template <typename Function>
	[[nodiscard]] bool ForEachImageSetChangeSince(uint64_t epoch, Function&& function) {
		std::scoped_lock lock {m_lock};
		if (m_image_set_epoch - epoch > ImageSetChangeHistory) {
			return false;
		}
		for (auto change = epoch + 1; change <= m_image_set_epoch; ++change) {
			const auto& range = m_image_set_changes[change % ImageSetChangeHistory];
			function(range.first, range.second);
		}
		return true;
	}
	[[nodiscard]] uint32_t ImageVersion(ImageId id) const noexcept {
		return m_image_versions.Load(id.index);
	}
	void PinImage(ImageId id);
	void UnpinImage(ImageId id);
	void RunGarbageCollector();

private:
	struct MetaDataInfo {
		std::bitset<8192> clear_mask;
		ImageId owner;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
		bool rejected = false;
		ImageId depth_source;
		bool    bound_depth = false;
	};

	struct WriteBack {
		GuestRange range;
		uint64_t   tick;
		bool       operator==(const WriteBack&) const = default;
	};

	struct BucketEntry {
		uint64_t address;
		uint32_t size;
		ImageId  id;

		[[nodiscard]] bool Overlaps(uint64_t addr, size_t size) const noexcept {
			return address < addr + size && addr < address + this->size;
		}
	};

	struct alignas(64) Bucket {
		Common::SmallVector<BucketEntry, 3, uint32_t> entries;
	};
	static_assert(sizeof(Bucket) == 64);

	using ImagePageTable = MultiLevelPageTable<Bucket, 20, 44, 14>;
	static_assert(ImagePageTable::kPageBits == ImagePageBits);

	class RegisteredPageCounts final {
	public:
		RegisteredPageCounts() = default;
		~RegisteredPageCounts();
		KYTY_CLASS_NO_COPY(RegisteredPageCounts);

		void               Add(size_t page, int32_t delta);
		[[nodiscard]] bool Any(size_t first, size_t last_exclusive) const noexcept;

	private:
		static constexpr size_t kLeafBits = 10;
		static constexpr size_t kLeafSize = size_t {1} << kLeafBits;
		using Leaf                        = std::array<std::atomic<uint32_t>, kLeafSize>;

		std::array<std::atomic<Leaf*>, ImagePageTable::kPageCount / kLeafSize> m_leaves {};
	};

	class ImageVersions final {
	public:
		ImageVersions() = default;
		~ImageVersions();
		KYTY_CLASS_NO_COPY(ImageVersions);

		[[nodiscard]] std::atomic<uint32_t>& Counter(uint32_t index);
		[[nodiscard]] uint32_t               Load(uint32_t index) const noexcept {
			if ((index >> kChunkBits) >= kMaxChunks) {
				return 0;
			}
			const auto* chunk = m_chunks[index >> kChunkBits].load(std::memory_order_acquire);
			return chunk == nullptr
			           ? 0
			           : (*chunk)[index & (kChunkSize - 1)].load(std::memory_order_acquire);
		}

	private:
		static constexpr size_t kChunkBits = 10;
		static constexpr size_t kChunkSize = size_t {1} << kChunkBits;
		static constexpr size_t kMaxChunks = 4096;
		using Chunk                        = std::array<std::atomic<uint32_t>, kChunkSize>;

		std::array<std::atomic<Chunk*>, kMaxChunks> m_chunks {};
	};
	void AttachImageVersion(ImageId id);

	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		static constexpr bool RETURNS_BOOL =
		    std::is_same_v<std::invoke_result_t<Func, uint64_t>, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	template <typename Func>
	void ForEachImageInRegion(uint64_t address, size_t size, Func&& func) {
		static constexpr bool BOOL_BREAK =
		    std::is_same_v<std::invoke_result_t<Func, ImageId, Image&>, bool>;
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
			return;
		}
		const uint64_t first_page = address >> ImagePageTable::kPageBits;
		ForEachPage(address, size, [&](uint64_t page) {
			const auto* bucket = m_image_page_table.Find(page);
			if (bucket == nullptr) {
				if constexpr (BOOL_BREAK) {
					return false;
				} else {
					return;
				}
			}
			for (const auto& entry: bucket->entries) {
				const uint64_t base_page = entry.address >> ImagePageTable::kPageBits;
				if (page != std::max(first_page, base_page)) {
					continue;
				}
				if (!entry.Overlaps(address, size)) {
					continue;
				}
				Image& image = m_slot_images[entry.id];
				if constexpr (BOOL_BREAK) {
					if (func(entry.id, image)) {
						return true;
					}
				} else {
					func(entry.id, image);
				}
			}
			if constexpr (BOOL_BREAK) {
				return false;
			}
		});
	}

	template <typename Func>
	void ForEachImageWithAddress(uint64_t address, Func&& func) {
		const auto* bucket = m_image_page_table.Find(address >> ImagePageTable::kPageBits);
		if (bucket == nullptr) {
			return;
		}
		for (const auto& entry: bucket->entries) {
			if (entry.address == address) {
				func(entry.id, m_slot_images[entry.id]);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      ReleaseDepthFeedbackCopy(ImageId depth_id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	[[nodiscard]] static bool PartialDirtyCandidate(const Image& image);
	[[nodiscard]] bool        TryInvalidatePartial(Image& image, uint64_t address, uint64_t size,
	                                                uint64_t granule);
	void                      UntrackImagePages(Image& image, uint64_t begin, uint64_t end);
	void                      RetrackHoles(Image& image);
	[[nodiscard]] bool        UploadImageLayers(Image& image);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged,
	                                           bool speculative = false, bool candidate = false);
	[[nodiscard]] ImageId       FindImageImpl(ImageDesc& desc, bool exact_format, bool speculative,
	                                          ImageId* depth_source = nullptr);
	[[nodiscard]] static bool   DepthOverlapNeedsRecreate(const ImageInfo& requested,
	                                                      BindingType binding, const Image& cached);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               const CommandRecorder& command);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] ImageId FindDepthCopyLocked(ImageId source_id, const ImageInfo& requested);
	void RefreshDepthCopy(ImageId copy_id);
	void ReleaseDepthCopies(ImageId color_id);
	void                  ParkImage(ImageId id);
	[[nodiscard]] ImageId UnparkImage(const ImageInfo& info);
	void                  DiscardParkedImage(ImageId id);
	void                  StencilPlanesChanged(const Image& depth);
	void                  DiscardParkedImages(uint64_t address, uint64_t size);
	[[nodiscard]] ImageId FindStencilCopy(ImageId association, const ImageInfo& requested);
	[[nodiscard]] ImageId FindLiveStencilStorageCopy(ImageId association, const ImageDesc& desc);
	void                  RefreshStencilCopy(ImageId copy_id);
	[[nodiscard]] bool    WriteBackStencilCopy(ImageId copy_id);
	void                  WriteBackStencilCopies(ImageId depth_id);
	void                  FinishStencilStorageWrite(ImageId copy_id);
	void                  ReleaseStencilCopies(ImageId association);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	[[nodiscard]] bool CopyDepthColor(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId FindStencilPlane(const Image& depth, GuestRange range) const;
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageViewDesc(const ImageDesc& desc) const;

	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	[[nodiscard]] bool DownloadQueuedImages();
	void               ArmLazyReadback(Image& image);
	void               DisarmLazyReadback(Image& image);
	void               FlushLazyImages(std::span<const std::pair<ImageId, uint64_t>> images);
	[[nodiscard]] static bool ReadbackArmedIn(const Image& image, uint64_t address,
	                                          uint64_t size) noexcept;
	[[nodiscard]] bool        HasArmedReadback(uint64_t address, uint64_t size);
	[[nodiscard]] bool        InvalidateLocked(uint64_t address, uint64_t size);
	[[nodiscard]] bool        InvalidateFromCompletion(uint64_t address, uint64_t size);
	void                      DeferInvalidation(uint64_t address, uint64_t size);
	[[nodiscard]] static uint64_t ReadbackConsumerKey(const ImageInfo& info) noexcept;
	void                      NoteDiscardedTarget(ImageId id, const Image& image);
	[[nodiscard]] static bool IsTargetImage(const Image& image) noexcept;
	[[nodiscard]] bool        WasDiscardedTarget(const Image& image) const;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	ImageVersions                                     m_image_versions;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	MultiLevelPageTable<uint64_t, ImagePageTable::kPageBits, ImagePageTable::kAddressSpaceBits,
	                    ImagePageTable::kFirstLevelBits>
	    m_image_page_epochs;
	RegisteredPageCounts                              m_registered_pages;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::mutex                                        m_write_back_mutex;
	std::vector<WriteBack>                            m_write_backs;
	std::unordered_set<uint64_t>                      m_readback_consumers;
	std::atomic<uint32_t>                             m_lazy_images {0};
	struct DiscardedTarget {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	static constexpr size_t DiscardedTargetCount = 32;
	std::array<DiscardedTarget, DiscardedTargetCount> m_discarded_targets {};
	size_t                                            m_discarded_next = 0;
	std::unordered_map<ImageId, std::vector<ImageId>> m_depth_copies;
	std::unordered_map<ImageId, ImageId>              m_depth_feedback_copies;
	std::unordered_map<ImageId, std::vector<ImageId>> m_stencil_copies;
	std::vector<std::pair<ImageId, uint64_t>>         m_parked_images;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	std::atomic<uint32_t> m_partial_candidates {0};
	uint64_t         m_image_set_epoch        = 0;
	static constexpr uint64_t ImageSetChangeHistory = 1024;
	std::array<std::pair<uint64_t, uint64_t>, ImageSetChangeHistory> m_image_set_changes {};
	bool             m_readback_linear_images = false;
	bool             m_readback_tiled_images  = false;
	bool             m_readback_storage_images = false;
#if defined(ANYPS5_KYTY)
	bool             m_readback_first_writes   = false;
#endif
	struct ExactLookup {
		ImageInfo info;
		ImageId   id;
		uint64_t  epoch        = 0;
		bool      exact_format = false;
		bool      valid        = false;
	};
	static constexpr size_t                                   ExactLookupWays = 1024;
	std::unique_ptr<std::array<ExactLookup, ExactLookupWays>> m_exact_lookups =
	    std::make_unique<std::array<ExactLookup, ExactLookupWays>>();
	[[nodiscard]] static size_t ExactLookupSlot(const ImageInfo& info) noexcept;
	[[nodiscard]] uint64_t      ImageEpochInRegionUnlocked(uint64_t address, uint64_t size) const;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

}

#endif
