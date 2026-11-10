#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/dccClearResolver.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"

#include <functional>
#include <array>
#include <map>
#include <string>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class BindlessTable;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class TextureCache;
struct TextureCacheTestAccess;

// KYTY_TRACE_PERF (PERF image_unregister): image unregistrations since the last call by reason, with the top
// reason/format/extent keys (GPU thread).
[[nodiscard]] std::string TakeImageUnregisterStats();
namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };
	// Called before a bindless-pinned image is unregistered, with the cache operation that drops
	// it (unmap, expand, mip_growth, overlap, depth_overlap, gc, shutdown, other) for the log.
	std::function<void(ImageId, const char*)> on_bindless_unregister;
	// The list path of bindless tables over 1 MiB (BindlessTable::FastTouchKeeps; null without the tables): such a table
	// leaves its unchanged slots' images untouched, so a bindless-pinned image is asked whether such a table drew within
	// `age` frames - kept then as the touch would have kept it (overlap deletion, collector, budget eviction).
	std::function<bool(ImageId id, uint64_t age)> bindless_fast_keep;
	// The plan flags: a merged image takes over the target flag of the image it replaces, in the draw's plan.
	std::function<void(ImageId from, ImageId to)> on_plan_target_transfer;

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	// The cache's lock taken by LockTableRefresh, and its release.
	void                               UnlockLookup() noexcept { m_lock.unlock(); }
	// The GPU thread waits for the lock a lookup batch holds.
	[[nodiscard]] bool                 LookupYieldRequested() const noexcept { return m_lock.GpuWaiting(); }
	// Diagnostic: print every cached image overlapping the range (KYTY_TRACE_FLIPS).
	void                        DebugDescribeRegion(const char* tag, uint64_t address, uint64_t size) const;
	// Diagnostic: the cached images starting exactly at address.
	[[nodiscard]] std::vector<Image*> DebugImagesAt(uint64_t address);
	// KYTY_WORKAROUND_STENCIL_STORAGE_SHADOW: a GPU-only R8_UINT image standing in for a stencil plane.
	[[nodiscard]] ImageId GetStencilShadowImage(uint32_t width, uint32_t height);
	// descriptors.cpp NullCompareBinding: the null image of a depth-compare binding, a 1x1 D16_UNORM image holding
	// 0.0 or (one) 1.0, created once each. Takes m_lock.
	[[nodiscard]] ImageId GetNullDepthImage(bool one);
	// Whether another image at the address can be a source (with or without a newer copy to take).
	bool SyncExtentAliases(ImageId id);
	// KYTY_FIX_SHARED_IMAGE_MEMORY (D4 step K1, textureCache.cpp); m_lock held exclusively. True when it recorded work.
	bool SyncSharedImageMemory(ImageId id, uint32_t bind);
	// KYTY_FIX_SHARED_IMAGE_MEMORY=3/4: the probe's check at a bind (bind: 0 texture, 1 storage, 2 render target, 3 depth target, 4 texture replay fast path); m_lock held exclusively.
	void ProbeSharedImageRead(ImageId id, uint32_t bind);
	std::unique_ptr<Buffer> m_probe_pattern[2]; // KYTY_FIX_SHARED_IMAGE_MEMORY=3: 0xA5 bytes, 0xFF bytes (float NaNs)
	std::unique_ptr<Buffer> m_alias_linear[2];  // KYTY_FIX_SHARED_IMAGE_DIRECT: the target's and a source's linear copy
	// The overlap list of an image, rebuilt when the images over its memory changed (m_lock held exclusively).
	const std::vector<ImageId>& SharedImageOverlaps(Image& image, ImageId id);
	// D4c step S0: what the shader recompiler knows of the written buffer binding whose InvalidateMemoryFromGPU follows on
	// this thread (set just before that call and cleared after it; none set: a fill or a copy).
	struct OwnerCensusBinding {
		uint32_t kind = 0; // 1 storage buffer binding, 2 written device-address range
		// fixed_writes: every write at an address fixed in the shader, the written bytes from the binding's start.
		const std::vector<std::pair<uint32_t, uint32_t>>* fixed_ranges = nullptr;
		bool     append         = false; // append_counter set: every write is an append record
		bool     append_tracked = false; // ... and the append windows track this pool
		uint32_t write_extent   = 0;     // the recompiler's write extent (UINT32_MAX: runtime-addressed or formatted)
		uint32_t max_byte_extent = 0;
		bool     read = false, atomic = false, formatted = false;
	};
	static void OwnerCensusSetBinding(const OwnerCensusBinding* binding);
	// KYTY_FIX_IMAGE_MEMORY_OWNER (D4c step S1, textureCache.cpp): 0 off, 1 on, 2 on with checks.
	[[nodiscard]] static uint32_t OwnerMode();
	// The colour-target rectangle of the draw whose targets this thread acquires next (pixels, [x0, x1) x [y0, y1));
	// cleared after the acquisition (none: the whole target).
	static void OwnerSetDrawRect(bool valid, int32_t x0, int32_t y0, int32_t x1, int32_t y1);
	// Whether a continued draw (targets not acquired again) writes only bytes its colour targets already hold in their
	// current draw run (the rectangle set above). Takes m_lock.
	[[nodiscard]] bool OwnerDrawPrepared(std::span<const ImageId> ids);
	// m_lock held exclusively. bind: 0 texture, 1 storage, 2 render target.
	void OwnerSyncImage(ImageId id, uint32_t bind);
	void OwnerWholeWrite(ImageId id);
	void OwnerCopyWrite(ImageId destination_id, ImageId source_id);
	void OwnerBufferWrite(uint64_t address, uint64_t size);
	void OwnerUnregister(ImageId id);
	void OwnerAfterUpload(ImageId id);
	void OwnerPull(Image& target, ImageId target_id, uint64_t holder, std::span<const std::pair<uint64_t, uint64_t>> ranges,
	               uint32_t why);
	void OwnerTileRange(Image& image, uint64_t lo, uint64_t hi, Buffer& tiled, bool tile);
	void OwnerDisturb(ImageId writer, uint64_t lo, uint64_t hi, std::span<const ImageId> candidates);
	void OwnerAdopt(ImageId id, std::span<const ImageId> overlaps);
	// The bytes `ranges` (sorted) of `holder` into `destination`, whose byte `destination_offset` stands for guest
	// address `destination_address`.
	void OwnerWriteBack(Image& holder, std::span<const std::pair<uint64_t, uint64_t>> ranges, Buffer& destination,
	                    uint64_t destination_offset, uint64_t destination_address);
	[[nodiscard]] bool OwnerBufferWriteRecords(uint64_t address, uint64_t size) const;
	void OwnerFixReadback(ImageId id, Buffer& destination, uint64_t destination_offset);
	// KYTY_FIX_IMAGE_MEMORY_OWNER=2: the image's guest bytes in a host-readable buffer; write intervals; the reference.
	std::shared_ptr<Buffer> OwnerSnapshot(Image& image, uint32_t fill = 0u);
	void OwnerBeginInterval(ImageId id, std::span<const std::pair<uint64_t, uint64_t>> allowed, bool escape);
	void OwnerEndInterval(ImageId id, bool compare);
	void OwnerFoldInterval(ImageId id);
	void OwnerOracleRead(ImageId id, std::span<const ImageId> overlaps);
	std::unique_ptr<Buffer> m_owner_buffers[3]; // KYTY_FIX_IMAGE_MEMORY_OWNER: target and source guest bytes, linear rows
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	// FindTexture for a sampled binding, skipping its alias sync and refresh when both are known no-ops (textureCache.cpp).
	[[nodiscard]] vk::ImageView FindTextureReplay(ImageId id, const ImageDesc& desc);
	// The image replay (pipeline\descriptors.cpp): whether this thread's last FindImage found an existing image by
	// the plain search and did nothing else; and FindImage's work for an image a caller proved it would find again: the
	// LRU touch and access tick (null id when it is gone).
	// TakeLastFindImagePlain also clears it. ImageReplayable: a registered image the replay may stand for (not depth,
	// no stencil plane or association, no metadata, not 1-byte texels).
	[[nodiscard]] static bool TakeLastFindImagePlain();
	// This thread's FindImage calls that went past the found-only answers (a full search; a draw's target memo).
	[[nodiscard]] static uint64_t FullSearches();
	[[nodiscard]] uint64_t        GcTick() const noexcept { return m_gc_tick; }
	[[nodiscard]] ImageId     ReplayImage(ImageId id);
	// Bindless table refresh events: what ReplayImage does to each registered, non-null image (access tick, LRU touch),
	// for a table's unchanged slots, under one lock.
	void TouchTableImages(std::span<const ImageId> ids);
	[[nodiscard]] bool        ImageReplayable(ImageId id) const;
	// The table sets (pipeline\descriptors.cpp): FindImage / FindTexture work of a stage image set, all or none.
	[[nodiscard]] bool TableReplayStage(std::span<const ImageId> ids);
	[[nodiscard]] bool TableTexturesReady(std::span<const ImageId> ids);
	// The cache's lock for a run of table slot refreshes (released with UnlockLookup), and
	// ReplayImage(id) && TableTexturesReady({id}) with the lock held by the caller.
	void               LockTableRefresh() noexcept { m_lock.lock(); }
	[[nodiscard]] bool ReplayTableSlotLocked(ImageId id);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);
	// KYTY_PERF_DIRECT_RANGES diagnostics: one stderr line per image registered in the region (at most max_lines).
	void DescribeImagesInRegion(uint64_t address, uint64_t size, uint32_t max_lines);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	// A write of [address, address + size) that starts at registered metadata clears it only when it stays inside
	// the metadata's bytes; a larger write is an ordinary memory write over memory another surface now uses, so the
	// registration is dropped and the write must run (false).
	[[nodiscard]] bool ClearMetaWrite(uint64_t address, uint64_t size);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);
	// KYTY_FIX_HTILE_SLICE_CLEAR (default off; =1 on; =2 on + check against the old decisions): HTILE clears of single
	// array layers. The game clears one layer's HTILE inside a layered depth target (far shadow-page cache 0x509e720000,
	// 256 layers); the old lookup took only writes at the metadata's first byte and
	// kept the cleared state of layers 0..63. 0: off, 1: on, 2: on + check.
	[[nodiscard]] static uint32_t HtileSliceClearMode();
	// The address is registered metadata, or (KYTY_FIX_HTILE_SLICE_CLEAR) lies inside a registered HTILE's layers.
	[[nodiscard]] bool IsMetaOrInside(uint64_t address);
	// A read-modify-write clear of [address, address + size) (shader b25b01ff1361e795): off, it is ClearMeta(address);
	// on, a write at the metadata's first byte clears every layer unless it covers a whole-layer part of it (then those
	// layers), and a write inside the metadata clears the whole layers it covers.
	[[nodiscard]] bool ClearMetaRmw(uint64_t address, uint64_t size, uint64_t hash);
	// KYTY_FIX_HTILE_SLICE_CLEAR=2: a dispatch read inside metadata (not at its first byte), so it is no clear.
	void NoteMetaReadGuard(uint64_t address, uint64_t hash);

	// KYTY_WORKAROUND_DCC_FAST_CLEAR=1: a compute fill wrote the uniform key byte over [address, address +
	// size). A DCC clear key there means every color surface whose metadata starts at address reads
	// as the clear color (the texture unit ignores its data memory) until it is rendered into or its
	// keys are written again; other keys end such a clear.
	[[nodiscard]] static bool DccFastClearEnabled();
	void                      RegisterDccFill(uint64_t address, uint64_t size, uint8_t key);
	// Before a sampling (or rendering/storage) use of the image: sampled images with a registered
	// clear key are cleared to its color; rendered or storage images end the registered clear.
	void ApplyDccFastClear(CommandBuffer& command, ImageId id, const ImageDesc& desc);
	// KYTY_FIX_DCC_KNOWN_KEYS=1 (with KYTY_WORKAROUND_DCC_FAST_CLEAR): metadata whose keys a registered
	// fill wrote is not downloaded (each download drained the GPU) for any use of the surface.
	[[nodiscard]] static bool DccKnownKeysEnabled();
	// KYTY_FIX_DCC_KEY_0X10=1 (default off): DCC key 0x10 means "the block reads as its first element". The AGC
	// Toolkit clears a color target to a color none of the fixed codes 00/40/80/C0 stands for (Ghost of Yotei: its
	// motion vectors to NaN) by filling the keys with 0x10 and then storing the packed color once at the start of
	// every 256-byte block (Agc::Toolkit::cs_dcc_cleartopixel_c, the same code as the fill shader above). The
	// dispatch registers that store here: [address, address + size) of the color surface, the element (1-4 dwords)
	// it stored in every block. A surface whose keys are all 0x10 is then cleared to that element at its next use,
	// as a fixed clear code is; guest memory keeps one element per block, as on the console.
	[[nodiscard]] static bool DccKey10FixEnabled();
	void RegisterDccPixelFill(uint64_t address, uint64_t size, const std::array<uint32_t, 4>& element,
	                          uint32_t element_bytes);
	// KYTY_PERF_DCC_GPU_ALL: 0 off, 1 on, 2 verify (see MaterializeColorClear in the .cpp).
	[[nodiscard]] static int DccGpuAllMode();
	// KYTY_TRACE_PERF: the PERF dcc_gpu_all line (KYTY_PERF_DCC_GPU_ALL counters; nothing when the switch is off).
	void ReportDccGpuAllPerf();

	void UnmapMemory(uint64_t address, uint64_t size);
	void RunGarbageCollector();
	// Texture relocation (textureMove.cpp): at a frame boundary, after the collector, the clean textures that keep nearly empty
	// VMA blocks alive move into fuller blocks (same ImageId, backing and views swapped, the bindless slots rewritten), so
	// whole blocks go back to the driver. GPU thread, m_lock not held (the batch waits for the GPU once).
	void RunTextureMove(BindlessTable& table);
	// Image read coherence (see the .cpp). A guest thread's read fault on an image-read-watched page: BeginCpuRead runs on the
	// GPU thread (it records the copies and submits them, it never waits for them), the faulting thread waits for serve.tick
	// and calls FinishCpuRead. Both take m_lock.
	struct CpuReadItem {
		ImageId  id;
		uint64_t serial  = 0; // content_serial the copy holds
		uint64_t address = 0; // what the copy publishes
		uint64_t size    = 0;
		Buffer*  buffer  = nullptr;
	};
	struct CpuReadServe {
		std::vector<CpuReadItem> items;
		std::vector<uint64_t>    detached; // DetachedRead::id of the copies on their way to the served pages
		uint64_t                 tick = 0; // 0: nothing was recorded
	};
	[[nodiscard]] CpuReadServe BeginCpuRead(uint64_t vaddr);
	void                       FinishCpuRead(CpuReadServe& serve);
	// Opens the pages of the detached copies that reached guest memory (FreeImage); the collector calls it every frame.
	void                       FinishDetachedReads();
	// After the page bits of [address, address + size) were lifted without a serve (a write, or an emulator thread's
	// access; RenderContext::LiftImageRead): disarms the images a write hit; the images that stay armed get the pages closed
	// by their next write. GPU thread, at a packet boundary, takes m_lock.
	void SettleImageReadLift(uint64_t address, uint64_t size, bool is_write);
	// Counters of the IMAGE_READ_SERVE line.
	void ReportImageReadServe();
	static void NoteImageReadPlain(uint64_t bytes);
	static void NoteImageReadWait(uint64_t nanos);
	static void NoteImageReadLift(uint64_t address, bool emulator_thread);
	// Bytes of registered images (KYTY_TRACE_PERF memory split).
	[[nodiscard]] uint64_t UsedMemory() const noexcept { return m_total_used_memory; }
	// Bytes of registered images (KYTY_TRACE_PERF images=): unlike UsedMemory(), which the collector replaces with
	// the device usage of the whole process at every run, only images count.
	[[nodiscard]] uint64_t RegisteredBytes() const noexcept { return m_registered_bytes; }
	// KYTY_TRACE_PERF / KYTY_TRACE_MEM_CENSUS: the collector's work since the previous census.
	struct GcStats {
		uint64_t runs = 0, examined = 0, freed = 0, freed_bytes = 0, downloaded = 0, downloaded_bytes = 0;
		uint64_t skip_unregistered = 0, skip_association = 0, skip_pinned = 0, skip_gpu_tiled = 0;
		uint64_t skip_not_pressured = 0, skip_no_download = 0, skip_download_failed = 0;
		// KYTY_VRAM_GC_FIX: depth/stencil images kept, downloads left for the next run (per-run byte limit),
		// evictions whose image was used or written while its download was published, freed pinned images.
		uint64_t skip_stencil = 0, skip_download_budget = 0, evict_cancelled = 0, freed_pinned = 0;
		// KYTY_VRAM_BUDGET: images freed to make room for a new one over the target.
		uint64_t alloc_evicted = 0, alloc_evicted_bytes = 0;
	};
	// Images by class as {count, bytes of their VMA allocations}; an image can be in several classes.
	struct Census {
		struct Class {
			uint64_t count = 0;
			uint64_t bytes = 0;
			void     Add(uint64_t size) noexcept {
				count++;
				bytes += size;
			}
		};
		Class    all, registered, pinned, gpu_tiled, gpu_linear, gpu_metadata, depth, render_target, lru_old,
		    same_address, unregistered;
		uint64_t associations        = 0; // stencil association records (they use their depth image's memory)
		uint64_t same_address_groups = 0; // guest addresses holding more than one image
		uint64_t scratch_bytes       = 0; // helper buffers the cache keeps at their largest size
		uint64_t pending_evictions   = 0; // KYTY_VRAM_GC_FIX: downloads not published yet
		// KYTY_VRAM_BUDGET at the last collector run: the target, the usage it counted (effective usage minus the
		// memory of images freed in the last ticks and of pending evictions), and that freed memory.
		uint64_t budget_target = 0, budget_used = 0, recently_freed = 0;
		GcStats  gc;
	};
	// Fills census and starts new collector statistics (GPU thread).
	void TakeCensus(Census& census);
	// KYTY_VRAM_BUDGET: memory of images freed in the last collector ticks (their deletion waits for the GPU) and
	// of images whose eviction is pending: the measured usage still holds it.
	[[nodiscard]] uint64_t PendingReleaseBytes();

private:
	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint64_t clear_mask = UINT64_MAX;
		uint64_t size       = 0; // bytes of the metadata at its address (0: not known)
		// KYTY_FIX_HTILE_SLICE_CLEAR: the HTILE bytes of one array layer, the layers known at this address, the
		// cleared state of every layer (layers past the vector read as slice_fill) and, for =2, the layers 0..63
		// whose new state may differ from clear_mask because the old lookup did not see the event.
		uint64_t             slice_size = 0;
		uint32_t             layers     = 0;
		uint8_t              slice_fill = 1;
		std::vector<uint8_t> slice_cleared;
		uint64_t             diverged = 0;

		[[nodiscard]] bool SliceCleared(uint32_t slice) const {
			return slice < slice_cleared.size() ? slice_cleared[slice] != 0 : slice_fill != 0;
		}
		void SetSlice(uint32_t slice, bool cleared) {
			if (slice >= slice_cleared.size()) slice_cleared.resize(static_cast<size_t>(slice) + 1u, slice_fill);
			slice_cleared[slice] = cleared ? 1u : 0u;
		}
		void SetAllSlices(bool cleared) {
			slice_cleared.clear();
			slice_fill = cleared ? 1u : 0u;
		}
	};
	// KYTY_FIX_HTILE_SLICE_CLEAR: the HTILE registration whose layers hold address (first byte excluded), or end.
	std::map<uint64_t, MetaDataInfo>::iterator FindMetaInside(uint64_t address);
	// KYTY_FIX_HTILE_SLICE_CLEAR (on, m_lock held): a clear of [address, address + size); write = ClearMetaWrite's
	// rules at the first byte (a larger write drops the registration), else ClearMeta's.
	bool ClearMetaSlices(uint64_t address, uint64_t size, bool write, uint64_t hash);

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Registered images per page of the image page table. Written under m_lock, read without it.
	// Based on TheCruZ/KytyPS5-GTA 3f9c73f46 (textureCache.h), GPL-2.0.
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
	RegisteredPageCounts m_registered_pages;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
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

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	// Our null images for unbound textures (upstream 892e70c8a binds native null descriptors instead; not taken).
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	// The image read watch (the caller holds m_lock). ArmWrittenImage: after a GPU write, for an image the CPU may read back.
	void                      ArmWrittenImage(Image& image);
	void                      ArmImageRead(Image& image);
	void                      DisarmImageRead(Image& image);
	// The armed image is freed (FreeImage): its bytes go to guest memory as an eviction's do, and its pages stay closed
	// until they are there (m_detached_reads), unless an armed image written later holds every byte of its range.
	void                      DetachImageRead(ImageId id);
	// Opens the pages of [address, address + size) that no other armed image (except is the one going) and no detached
	// copy keeps closed.
	void                      OpenImageReadPages(uint64_t address, uint64_t size, const Image* except);
	void                      EraseDetachedRead(size_t index); // and opens its pages
	[[nodiscard]] Buffer*     AcquireCpuReadBuffer(uint64_t size);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	// ReplayImage / TableTexturesReady with m_lock held by the caller.
	[[nodiscard]] ImageId ReplayImageLocked(ImageId id);
	[[nodiscard]] bool    TableTexturesReadyLocked(std::span<const ImageId> ids);
	// The checks only, without touches (the lock-free answers read them without the lock too).
	[[nodiscard]] bool    TableTexturesReadyChecks(std::span<const ImageId> ids) const;
	[[nodiscard]] bool    TableReplayStageChecks(std::span<const ImageId> ids) const;
	[[nodiscard]] ImageId ReplayImageCheck(ImageId id) const;
	// Caller holds m_lock: the touches of `touches` (cleared), and the calling job's own list before it may free images.
	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	// KYTY_FIX_DCC_GPU_CLEAR_CHECK (port of BryanKAdams 8db16e32 + c41cebee): GPU-written DCC keys are checked
	// on the GPU and the clears run under conditional rendering, instead of a CPU readback (a full drain).
	[[nodiscard]] static uint32_t DccGpuClearCheckLevel();
	// all (KYTY_PERF_DCC_GPU_ALL): every binding type, CMASK as well as DCC metadata (the caller), more than
	// DccClearResolver::MaxSlices slices (checked in chunks) and multisampled images; every fallback is counted and
	// logged by class (DccGpuAllClass).
	[[nodiscard]] bool MaterializeDccClearOnGpu(ImageId id, const ImageDesc& desc, uint64_t slices_address,
	                                            uint64_t slice_size, uint32_t image_first, uint32_t count,
	                                            bool all = false);
	enum class DccGpuAllClass : uint32_t {
		NoResolver,   // VK_EXT_conditional_rendering unavailable
		Volume,       // a volume surface (no 2D-array attachment view of its slices)
		Slices,       // no slice, a view base level > 0, or keys not in whole dwords
		Format,       // the view format cannot be a color attachment
		DepthAlias,   // the image has a depth association (the CPU path cannot clear it either)
		NoBacking,    // no native image
		NoAttachment, // the native image was created without color-attachment usage
		Layers,       // the view's layers are outside the native image
		Init,         // the image needs its guest contents first and cannot be initialized (compressed)
		Bind,         // the keys cannot be bound as one storage range
		Count
	};
	void NoteDccGpuAllFallback(const ImageDesc& desc, DccGpuAllClass reason, uint32_t slices);
	// KYTY_PERF_DCC_GPU_ALL=2: after the GPU finished a check copied aside (priority completion), compares its
	// predicates with the decision of the CPU path for the same key bytes.
	void VerifyDccGpuCheck(const DccClearResolver::VerifySlot& slot, uint64_t slice_size, uint32_t slices,
	                       uint32_t code_mask, uint64_t image_address, uint64_t keys_address);
	[[nodiscard]] bool DccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count, uint32_t code_mask);
	void               MarkDccSlicesChecked(uint64_t address, uint64_t slice_size, uint32_t count, uint32_t code_mask);
	// A write to checked metadata (GPU, CPU or unmap): its slices are checked again. Does not take m_lock.
	void               OnDccMetaWrite(uint64_t address, uint64_t size);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	// Caller holds m_lock and ends rendering. Begins rendering `layers` layers of one color mip
	// through a `format` view, clearing them or, when `clear` is null, loading them; returns the
	// render area.
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               vk::CommandBuffer command);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	// Records a copy of a GPU-written image into guest memory, published after the GPU ran it. evicting
	// (KYTY_VRAM_GC_FIX collector): never waits for space in the download ring, and done is set once the bytes
	// are in guest memory. An image the ring cannot hold uses a buffer of its own under KYTY_VRAM_GC_FIX (before,
	// the session ended).
	// cancelled (DetachImageRead): set by a CPU write that got to the bytes first, the copy is not published.
	[[nodiscard]] bool DownloadImageMemory(ImageId id, bool evicting = false,
	                                       std::shared_ptr<std::atomic_bool> done      = nullptr,
	                                       std::shared_ptr<std::atomic_bool> cancelled = nullptr);
	// The copy of DownloadImageMemory into destination at offset (mapped: its address there): the bytes the image does not
	// hold keep what guest memory holds. False when guest memory cannot be read. The image was checked by the caller.
	[[nodiscard]] bool RecordImageDownload(ImageId id, Buffer& destination, uint64_t offset, uint8_t* mapped);

	// KYTY_VRAM_GC_FIX: see CollectGarbageFixed (textureCache.cpp).
	struct GcPass {
		uint64_t min_age            = 0;     // collector ticks (frames) since the image's last use
		size_t   max_frees          = 0;     // images freed or downloaded this run when goal is 0
		uint64_t usage              = 0;     // estimate, lowered by every image the run gives up
		uint64_t goal               = 0;     // when not 0: stop once usage is below it
		bool     allow_download     = false; // GPU-written images go through guest memory
		bool     allow_pinned       = false; // bindless-pinned images may go
		uint64_t max_download_bytes = 0;
	};
	struct PendingEviction {
		ImageId                           id;
		uint64_t                          content_serial = 0; // the image's contents when the download was recorded
		uint64_t                          tick           = 0; // m_gc_tick then: a later use cancels the eviction
		std::shared_ptr<std::atomic_bool> done;               // the download is in guest memory
		uint64_t                          bytes = 0;          // its allocation (KYTY_VRAM_BUDGET)
	};
	void               CollectGarbageFixed(uint64_t tick);
	void               CollectImages(uint64_t tick, GcPass& pass);
	void               FinishPendingEvictions();
	void               FreeCollectedImage(ImageId id);
	[[nodiscard]] bool CanEvictWithDownload(const Image& image) const;
	// KYTY_VRAM_BUDGET (caller holds m_lock).
	void               MakeRoomForImage(const ImageInfo& info);
	void               NoteFreedBytes(uint64_t bytes);
	[[nodiscard]] uint64_t RecentlyFreedBytes();
	[[nodiscard]] uint64_t PendingEvictionBytes() const;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	// Own cache line (analysis C7): the lock is written by every acquisition, the references around
	// it are read around every acquisition.
	alignas(64) TrackingSpinLock                      m_lock {PerfLockSiteTextureCache};
	alignas(64) PageManager&                          m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::unordered_map<vk::Format, ImageId>           m_null_images;
	std::array<ImageId, 2>                            m_null_depth_images {}; // GetNullDepthImage: 0.0, 1.0
	std::unordered_map<uint64_t, ImageId>             m_stencil_shadow_images;
	// KYTY_WORKAROUND_PRESERVE_STENCIL: stencil of a depth image replaced by a color view (storage write
	// to the depth plane), restored when the depth image is recreated.
	std::unique_ptr<Buffer> m_saved_stencil;
	// KYTY_FIX_SHARED_IMAGE_MEMORY: private merge buffer.
	std::unique_ptr<Buffer> m_alias_scratch;
	uint64_t                m_saved_stencil_address = 0;
	vk::Extent3D            m_saved_stencil_extent {};
	bool                    m_saved_stencil_valid = false;
	uint64_t                m_saved_stencil_plane = 0;
	void SaveStencil(Image& depth);
	// One saved stencil aspect per (depth image address, guest stencil plane address).
	std::unordered_map<uint64_t, std::unique_ptr<Buffer>> m_stencil_planes;
	// KYTY_WORKAROUND_STENCIL_PLANE_SYNC: the depth address each guest stencil plane address was last bound with (an
	// address, not an image: depth images get replaced), and per saved plane (same key as m_stencil_planes) the
	// content serial taken when it was saved.
	std::unordered_map<uint64_t, uint64_t> m_stencil_plane_depth;
	std::unordered_map<uint64_t, uint64_t> m_stencil_plane_saved_serial;
	std::unique_ptr<Buffer>                m_stencil_sync_staging;
	bool SwapStencilPlane(Image& depth, uint64_t plane_address);
	// KYTY_FIX_STENCIL_PLANE_WRITES (M119): guest-side writes to a plane the depth image does not hold go to the
	// plane's saved copy (the guest bytes of an upload; the value of a fill turned into a clear). False when the
	// layout is not one the saved copy describes (one layer, the depth image's extent); the caller then writes as before.
	bool UploadStencilPlaneToSaved(Image& depth, const Image& plane, Buffer& source, uint64_t source_offset);
	bool FillStencilPlaneSaved(CommandBuffer& command, Image& depth, const Image& plane, uint8_t value);
	// KYTY_WORKAROUND_STENCIL_PLANE_SYNC: a plain 1-byte image at the plane address, of the depth image's extent, written
	// after the plane was saved.
	[[nodiscard]] ImageId NewerStencilPlaneImage(uint64_t plane_address, uint64_t saved_serial, vk::Extent3D extent);

public:
	// Makes a depth image's stencil aspect hold the given guest stencil plane (saved/restored on GPU).
	void EnsureStencilPlane(ImageId depth_id, uint64_t plane_address);
	// KYTY_WORKAROUND_STENCIL_PLANE_SYNC (default 2, Common::EnvDefaults; =0 off): PS5 keeps each stencil plane in its own memory, the host keeps one
	// stencil aspect per depth image. A 1-byte image at a guest stencil plane address that a shader reads gets that
	// plane's current contents: the owning depth image's stencil aspect when it holds the plane, else the copy saved
	// at its last plane swap (kept as is when the image itself was written after that save). Returns true when the
	// address is a known plane.
	[[nodiscard]] static bool StencilPlaneSyncEnabled();
	[[nodiscard]] static int  StencilPlaneSyncLevel();
	bool SyncStencilPlaneReader(uint64_t plane_address, ImageId reader_id);
	// KYTY_WORKAROUND_STENCIL_ALIAS_COPY: the stencil association record at this exact address (its
	// depth image holds the stencil), the last one found; empty when there is none.
	[[nodiscard]] ImageId FindStencilAssociationAt(uint64_t address) {
		ImageId best {};
		for (const auto id: FindImagesInRegion(address, 1, false)) {
			const auto* image = m_slot_images.try_get(id);
			if (image == nullptr || !image->depth_id || image->info.data.address != address) continue;
			best = id;
		}
		return best;
	}
	// KYTY_BINDLESS: re-uploads a resolved bindless texture the guest has written since.
	void RefreshBindlessImage(ImageId id) {
		if (const auto* image = m_slot_images.try_get(id); image != nullptr && image->registered) RefreshImage(id);
	}

private:
	void RestoreStencil(Image& depth);
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	// Host buffers of finished image read serves, for the next ones (under m_lock).
	std::vector<std::unique_ptr<Buffer>>              m_cpu_read_buffers;
	std::vector<std::unique_ptr<Buffer>>              m_cpu_read_buffers_busy;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	// KYTY_WORKAROUND_DCC_FAST_CLEAR: metadata start address -> (filled size, clear key).
	std::map<uint64_t, std::pair<uint64_t, uint8_t>> m_dcc_fills;
	// KYTY_FIX_DCC_KNOWN_KEYS: GPU writes reported by the buffer cache, applied to m_dcc_fills
	// under m_lock (the observer may run while m_lock is held).
	std::mutex                                        m_dcc_pending_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_dcc_pending;
	std::atomic<size_t>                               m_dcc_fill_count {0};
	void                                              ApplyDccPendingWrites(); // caller holds m_lock
	// KYTY_FIX_DCC_KEY_0X10 (RegisterDccPixelFill): color surface address -> the clear-to-pixel store over it, and
	// metadata address -> the serial of its last 0x10 key fill (a store completes only a key fill made before it). A
	// GPU write over the surface's memory ends its entry (ApplyDccPendingWrites), so does the clear it caused.
	struct DccPixelFill {
		uint64_t                size = 0;
		std::array<uint32_t, 4> element {};
		uint32_t                element_bytes = 0;
		uint64_t                serial        = 0;
	};
	std::map<uint64_t, DccPixelFill>                  m_dcc_pixel_fills;
	std::map<uint64_t, uint64_t>                      m_dcc_key10_serials;
	uint64_t                                          m_dcc_fill_serial = 0;
	std::atomic<size_t>                               m_dcc_pixel_fill_count {0};
	// Caller holds m_lock. The color a surface whose DCC keys are all 0x10 reads as (the element its clear to pixel
	// stored), decoded for desc's view; false (logged as DCC_KEY_0X10 skip) when no registered store explains it.
	[[nodiscard]] bool DecodeDccKey10(const ImageDesc& desc, vk::ClearColorValue& clear, const char* path);
	void               EraseDccPixelFill(uint64_t address); // caller holds m_lock
	// KYTY_FIX_DCC_GPU_CLEAR_CHECK: slices checked on the GPU since their last write.
	struct DccChecked {
		uint64_t size      = 0;
		uint32_t code_mask = 0;
	};
	std::unique_ptr<DccClearResolver>                 m_dcc_resolver;
	std::mutex                                        m_dcc_checked_mutex;
	std::map<uint64_t, DccChecked>                    m_dcc_checked;
	std::atomic<size_t>                               m_dcc_checked_count {0};
	std::atomic<uint64_t>                             m_dcc_gpu_checks {0};
	std::atomic<uint64_t>                             m_dcc_gpu_skips {0};
	std::array<std::atomic<uint64_t>, 6>              m_dcc_gpu_fallbacks {};
	// KYTY_PERF_DCC_GPU_ALL (ReportDccGpuAllPerf): checks recorded / their slices / chunks, views whose keys never clear
	// (no read needed), checks skipped as already done, CMASK and multisampled checks, fallbacks by class (each still
	// reads the keys on the CPU), and the verify mode's slices compared / mismatches / checks not copied (ring full).
	struct DccGpuAllStats {
		std::atomic<uint64_t> checks {0}, slices {0}, chunks {0}, no_code {0}, already {0}, cmask {0}, msaa {0};
		std::array<std::atomic<uint64_t>, static_cast<uint32_t>(DccGpuAllClass::Count)> fallback {};
		std::atomic<uint64_t> verified {0}, mismatches {0}, verify_skipped {0};
	};
	DccGpuAllStats                                    m_dcc_all;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	uint64_t         m_registered_bytes       = 0; // RegisteredBytes()
	GcStats          m_gc_stats;
	// KYTY_TRACE_EVICT_REUSE (diagnostic): guest address -> collector tick of images freed by the collector or the budget,
	// for images created again at the same address soon after, and GPU-written images freed without their contents.
	std::unordered_map<uint64_t, uint64_t> m_evicted_at;
	uint64_t m_evict_reused = 0, m_evict_gpu_freed = 0, m_evict_reused_gpu = 0;
	std::vector<PendingEviction> m_pending_evictions; // KYTY_VRAM_GC_FIX
	// Image read coherence: the copy of an armed image freed before a guest thread read it (DetachImageRead), on its way to
	// guest memory. Its pages stay closed until it is there: a guest read meanwhile waits for its tick (BeginCpuRead),
	// FinishDetachedReads opens the pages of the ones that landed.
	struct DetachedRead {
		uint64_t                          id      = 0;
		uint64_t                          address = 0; // the pages kept closed (the image's read range)
		uint64_t                          size    = 0;
		uint64_t                          tick    = 0; // the command buffer of the copy
		std::shared_ptr<std::atomic_bool> done;        // the bytes are in guest memory
		std::shared_ptr<std::atomic_bool> cancelled;   // a CPU write got there first (null: an eviction's copy)
	};
	std::vector<DetachedRead> m_detached_reads;
	uint64_t                  m_detached_read_serial = 0;
	// KYTY_VRAM_BUDGET: allocation bytes freed per command-buffer tick, kept until the GPU retired the tick (the
	// images are destroyed then); the collector tick at which a make-room scan found nothing to free; the
	// target/usage of the last collector run (census).
	struct ReleasingBytes {
		uint64_t tick  = 0;
		uint64_t bytes = 0;
	};
	std::vector<ReleasingBytes> m_releasing;
	uint64_t                    m_make_room_dry_tick = ~uint64_t {0};
	uint64_t                    m_budget_target      = 0;
	uint64_t                    m_budget_used        = 0;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;
	// The operation unregistering images now (see on_bindless_unregister); nullptr = other.
	const char*      m_unregister_reason      = nullptr;
	bool             m_readback_tiled_images  = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
