#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/inPlaceMemory.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class CommandRecorder;
class TextureCache;
class RenderContext;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

enum class HackMark : uint32_t {
	Count,
};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t NATIVE_RT_CONTEXT_PAGE = CACHING_NUMPAGES;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    (CACHING_NUMPAGES + 1) * sizeof(vk::DeviceAddress);
	static constexpr uint64_t WRITE_BITMAP_OFFSET = CACHING_NUMPAGES / 8;
	static constexpr uint64_t USE_BITMAP_OFFSET   = 2 * WRITE_BITMAP_OFFSET;
	static constexpr uint64_t HACK_MARKS_OFFSET   = 3 * WRITE_BITMAP_OFFSET;
	static constexpr uint32_t HACK_MARK_ENTRIES   = 256;
	static constexpr uint32_t HACK_MARK_DWORDS    = 16;
	static constexpr uint32_t HACK_MARK_COUNTS    = 4;
	static constexpr uint32_t HACK_MARK_GUEST_RAYS = HACK_MARK_COUNTS + static_cast<uint32_t>(HackMark::Count);
	static constexpr uint64_t HACK_MARKS_SIZE     = uint64_t {HACK_MARK_ENTRIES} * HACK_MARK_DWORDS * 4;
	static constexpr uint64_t FAULT_BUFFER_SIZE   = HACK_MARKS_OFFSET + HACK_MARKS_SIZE;
	static constexpr uint32_t HACK_MARK_ORACLE_RETURNED   = HACK_MARK_GUEST_RAYS + 1;
	static constexpr uint32_t HACK_MARK_ORACLE_RAYS       = HACK_MARK_GUEST_RAYS + 2;
	static constexpr uint32_t HACK_MARK_ORACLE_CANDIDATES = HACK_MARK_GUEST_RAYS + 3;
	static constexpr uint32_t HACK_MARK_ORACLE_ROUNDS     = HACK_MARK_GUEST_RAYS + 4;
	static constexpr uint32_t HACK_MARK_ORACLE_WAVES      = HACK_MARK_GUEST_RAYS + 5;
	static_assert(HACK_MARK_ORACLE_WAVES < HACK_MARK_DWORDS);

	static constexpr uint32_t HackMarkEntry(uint64_t shader_hash) {
		return (static_cast<uint32_t>(shader_hash) ^ static_cast<uint32_t>(shader_hash >> 32u)) %
		       HACK_MARK_ENTRIES;
	}

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache, RenderContext* resources = nullptr);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size, bool guest_fault = false);
	void                   InvalidateMemoryForWrite(uint64_t vaddr, uint64_t size);
	struct SyncRegionRequest {
		uint64_t address = 0, size = 0, cpu_epoch = 0, registration_epoch = 0;
	};
	void SynchronizeRegionRequest(SyncRegionRequest& request);
	[[nodiscard]] uint64_t RegistrationEpoch() const { return m_registration_epoch.load(std::memory_order_acquire); }
	[[nodiscard]] uint64_t RegionRegistrationEpoch(uint64_t vaddr) const {
		const auto region = vaddr / TRACKER_REGION_SIZE;
		return region < TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE ? m_region_registrations[region].load(std::memory_order_acquire)
		                                                           : 0;
	}
	[[nodiscard]] bool DeferredProtectionPending(uint64_t vaddr) const { return m_memory_tracker.DeferredProtectionPending(vaddr); }
	void CollectMappedRegisteredRanges(const RangeSet& mapped, uint64_t begin, uint64_t end,
	                                   std::vector<std::pair<uint64_t, uint64_t>>& ranges) const;
	[[nodiscard]] bool TakeRegistrationSpans(std::vector<GuestRange>& spans);
	void EnsureBdaReadable(std::span<const GuestRange> ranges);
	enum class WriteFaultOwner : uint8_t { CommandBuffer, CommandSegment, GpuNeighbor, Buffer, Other };
	WriteFaultOwner        InvalidateFaultedPage(uint64_t vaddr);
	static void            CountWriteFault(WriteFaultOwner owner, bool image);
	[[nodiscard]] static bool CommandMemoryInPlaceRequested() noexcept { return InPlaceMemory::Requested(); }
	[[nodiscard]] bool        ServesInPlace(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t    InPlaceGpuWriteTick(uint64_t vaddr, uint64_t size) const;
	void                      NoteInPlaceGpuWrite(uint64_t vaddr, uint64_t size);
	void                      ServeInPlaceConversions();
	void NoteCommandBuffer(uint64_t vaddr, uint64_t size) {
		if (size != 0 && GuestRange {vaddr, size}.Valid()) {
			m_memory_tracker.MarkCommandBlocks(vaddr, size);
		}
	}
	void NoteCommandSegment(uint64_t vaddr, uint64_t size) {
		if (size != 0 && GuestRange {vaddr, size}.Valid()) {
			m_memory_tracker.MarkSegmentBlocks(vaddr, size);
		}
	}
	void ForgetCommandBuffers(uint64_t vaddr, uint64_t size) {
		if (size != 0 && GuestRange {vaddr, size}.Valid()) {
			m_memory_tracker.ForgetCommandBlocks(vaddr, size);
		}
	}
	uint64_t               DeleteUnmappedBuffers(uint64_t vaddr, uint64_t size);
	using FreeBlocksQuery = bool (*)(uint64_t begin, uint64_t end, std::vector<std::pair<uint64_t, uint64_t>>* blocks);
	struct FreeBlockSplits {
		uint64_t lists = 0, cut = 0;
	};
	uint64_t               SplitUnmappedBuffers(uint64_t vaddr, uint64_t size, FreeBlocksQuery free_blocks = nullptr,
	                                            FreeBlockSplits* splits = nullptr);
	uint64_t               DropFreedRange(uint64_t vaddr, uint64_t size, bool read_back, FreeBlocksQuery free_blocks,
	                                      FreeBlockSplits* splits);
	void                   InvalidateUnlessGpuModified(uint64_t vaddr, uint64_t size) noexcept;
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false,
	                                  bool guest_fault = false);
	static void            LogReadbackSummary();
	static void            LogWriteBlockSummary();
	static void            NoteStagedCommandProcessorRead();
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	void CacheFaultedMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	bool PrepareDeviceAddressRead(uint64_t vaddr, uint64_t size);
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void StreamGuestData(uint8_t* destination, uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::optional<uint64_t> StreamGuestRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::optional<uint64_t> FindStreamCopy(uint64_t vaddr, uint64_t size);
	void NoteStreamCopy(uint64_t vaddr, uint64_t size, uint64_t offset);
	void SetStreamInPlace(bool in_place) noexcept { m_stream_in_place = in_place; }
	[[nodiscard]] uint64_t StreamUpload(const void* data, uint64_t size, uint64_t alignment);
	[[nodiscard]] vk::DeviceAddress TableUpload(const void* data, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	[[nodiscard]] bool ReplicateBuffer(uint64_t destination, uint64_t count, uint64_t source,
	                                   uint64_t pattern);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	void ScheduleCopyFeedback(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool NeverGpuWritten(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool NoGpuBytesNow(uint64_t vaddr, uint64_t size) const noexcept;
	[[nodiscard]] bool GpuLinesOverflowed() const noexcept { return m_gpu_line_overflow.load(std::memory_order_relaxed); }
	static void NoteReplayedRead();
	static void AllowReplayAfterReadback() noexcept;
	static bool TakeReplayAfterReadback(uint64_t* begin, uint64_t* end) noexcept;
	static void NoteReplayAfterReadback(bool replayed) noexcept;
	void        WaitReadbackRelease();
	[[nodiscard]] bool WriteAroundGpuWrites(uint64_t vaddr, const void* data, uint64_t size);
	[[nodiscard]] bool MayBeGpuModified(uint64_t vaddr, uint64_t size) const {
		return m_memory_tracker.MayBeGpuModified(vaddr, size) || InPlaceGpuWriteTick(vaddr, size) != 0;
	}
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	[[nodiscard]] Buffer* GetBufferIfLive(BufferId id) {
		return IsBufferInvalid(id) ? nullptr : &m_slot_buffers[id];
	}
	[[nodiscard]] bool TouchLiveBuffer(BufferId id, vk::Buffer handle, uint64_t vaddr, uint64_t size);
	void EnsureBufferContents(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t BufferSetEpoch() const noexcept { return m_buffer_set_epoch; }
	[[nodiscard]] static uint64_t BdaTableGeneration() noexcept { return 0; }
	[[nodiscard]] StreamBuffer& GetTableUploadBuffer();
	void               PrepareFaultBuffer() { m_fault_manager.PrepareFaultBuffer(); }
	void               ProcessFaultBuffer();
	void               ProcessUseBuffer();
	void ProcessWriteBuffer(FaultManager::PagesHandler&& handler) {
		m_fault_manager.ProcessWriteBuffer(std::move(handler));
	}
	void NoteGpuWrites(uint64_t vaddr, uint64_t size);
	template <typename Func>
	void ConsumeCpuWrites(CpuWriteConsumer consumer, uint64_t vaddr, uint64_t size, Func&& func) {
		m_memory_tracker.ConsumeCpuWrites(consumer, vaddr, size, std::forward<Func>(func));
	}
	template <typename Func>
	void ConsumeReleasedRanges(Func&& func) {
		m_released_ranges.ForEach(func);
		m_released_ranges.Clear();
		m_record_released = true;
	}
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	void SynchronizeBuffersInRanges(const RangeSet& ranges, bool all);
	void ForceSynchronization(uint64_t vaddr, uint64_t size);
	void SynchronizeBuffersInSpans(std::span<const std::pair<uint64_t, uint64_t>> spans);
	struct SetRegions {
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		uint32_t                                   pass = 0;
		std::vector<uint64_t>                      covered;
		std::vector<RegionBits>                    covered_pages;
		uint64_t                                   version = UINT64_MAX;
	};
	void SynchronizeBuffersInSet(const RangeSet& set, SetRegions& regions);
	void AddToSet(RangeSet& set, SetRegions& regions, uint64_t address, uint64_t size);
	void FlushUploadBatch();
	bool ScatterUploadBatch(const CommandRecorder& native);
	[[nodiscard]] uint64_t RegisterEpoch() const { return m_register_epoch; }
	[[nodiscard]] bool HasKeptPages() const noexcept { return m_kept_pages; }
	[[nodiscard]] uint64_t GpuWriteGeneration() const noexcept { return m_gpu_write_generation; }
	[[nodiscard]] uint64_t CpuDirtyEpoch(uint64_t vaddr, uint64_t size) const {
		uint64_t epoch = 0;
		for (auto index = vaddr / TRACKER_REGION_SIZE; index * TRACKER_REGION_SIZE < vaddr + size;
		     index++) {
			epoch += m_memory_tracker.RegionCpuDirtyEpoch(index);
		}
		return epoch;
	}
	[[nodiscard]] uint64_t CpuModificationEpoch(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || !GuestRange {vaddr, size}.Valid()) return 0;
		constexpr uint64_t MaxRegions = 64;
		const uint64_t     first      = vaddr / TRACKER_REGION_SIZE;
		const uint64_t     last       = (vaddr + size - 1) / TRACKER_REGION_SIZE;
		if (last - first >= MaxRegions) return 0;
		uint64_t sum = 0;
		for (uint64_t region = first; region <= last; ++region) {
			const uint64_t begin = std::max(vaddr, region * TRACKER_REGION_SIZE);
			const uint64_t end   = std::min(vaddr + size, (region + 1) * TRACKER_REGION_SIZE);
			const uint64_t epoch = m_memory_tracker.CpuModificationEpoch(begin, end - begin);
			if (epoch == 0) return 0;
			sum += epoch;
		}
		return sum;
	}
	void NoteHostWrite(uint64_t vaddr, uint64_t size) { m_memory_tracker.NoteHostWrite(vaddr, size); }
	[[nodiscard]] bool IsRegionCpuDirty(uint64_t vaddr, uint64_t size) { return m_memory_tracker.IsRegionCpuModified(vaddr, size); }
	[[nodiscard]] BufferId RegisteredBuffer(uint64_t vaddr, uint64_t size) const {
		if (vaddr == 0 || !GuestRange {vaddr, size}.Valid()) return {};
		const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
		if (owner == nullptr || !*owner) return {};
		const auto* buffer = m_slot_buffers.try_get(*owner);
		return buffer != nullptr && !buffer->is_deleted && buffer->IsInBounds(vaddr, size) ? *owner : BufferId {};
	}
	void               RunGarbageCollector();
	[[nodiscard]] bool CommitWriteTicks();
	[[nodiscard]] bool TryReadMirroredGpuWrites(uint64_t vaddr, void* data, uint64_t size);
	void RequestFlushForReadbacks() noexcept {
		if ((m_device_address_hot != 0 &&
		     m_readback_count - m_device_address_hot < HotReadbackLifetime) ||
		    m_mirror_uses - m_mirror_flush_use < MirrorFlushLifetime) {
			m_flush_requested = true;
		}
	}
	void NoteDeviceAddressUse() noexcept {
		m_device_address_pending = true;
		if (m_device_address_hot != 0 &&
		    m_readback_count - m_device_address_hot < HotReadbackLifetime) {
			m_flush_requested = true;
		}
	}

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }
	WriteFaultOwner InvalidateGuestWriteFault(uint64_t vaddr);

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	void DiscardGpuWrites(uint64_t vaddr, uint64_t size);
	void CollectStaleBuffers(uint64_t tick);
	[[nodiscard]] bool NoImageStartsAt(uint64_t vaddr) const noexcept;
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	void RecordBufferUpload(Buffer& buffer, std::span<const vk::BufferCopy> copies,
	                        vk::Buffer source);
	void               SynchronizeBuffers(std::span<Buffer* const>                       buffers,
	                                      std::span<const std::pair<uint64_t, uint64_t>> ranges);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);
	[[nodiscard]] uint64_t ReadAheadSize(const Buffer& buffer, uint64_t vaddr, uint64_t size) const;
	void ReadMemoryAsync(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault);
	void ReadMemoryAsyncTimed(uint64_t vaddr, uint64_t size, bool is_write, bool guest_fault, bool command_processor);
	[[nodiscard]] uint64_t InFlightReadbackTick(uint64_t begin, uint64_t end) const;
	void ReleaseReadback(uint64_t vaddr, uint64_t size, uint64_t pages_begin, uint64_t pages_end,
	                     uint64_t tick, bool is_write);
	std::unique_ptr<InPlaceMemory> m_in_place;
	struct CopyFeedback;
	std::unique_ptr<CopyFeedback> m_copy_feedback;
	size_t                        m_copy_feedback_entries = 0;
	void                          InvalidateCopyFeedback(uint64_t vaddr, uint64_t size) noexcept;
	[[nodiscard]] bool TryReadCopyFeedback(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool TryGuestReadFromFeedback(uint64_t vaddr, uint64_t size, bool is_write);
	uint64_t m_protect_jobs = 0;
	[[nodiscard]] bool DownloadOnReadbackQueue(Buffer& buffer, std::span<const vk::BufferCopy> copies,
	                                           uint64_t total_size);
	void CollectDownloadCopies(const Buffer& buffer, uint64_t vaddr, uint64_t size,
	                           std::vector<vk::BufferCopy>& copies, uint64_t& total_size);
	struct ReadbackQueueRead;
	[[nodiscard]] bool StartReadbackQueueRead(const Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                          uint64_t pages_begin, uint64_t pages_end,
	                                          ReadbackQueueRead& read, bool followers);
	struct GuestReadPage {
		uint64_t page = 0;
		uint64_t read = 0;
	};
	static constexpr size_t   MaxGuestReadPages     = 512;
	static constexpr uint64_t GuestReadPageLifetime = 16384;
	std::vector<GuestReadPage> m_guest_read_pages;
	uint64_t                   m_guest_reads = 0;
	void                       NoteGuestReadPages(uint64_t pages_begin, uint64_t pages_end);
	void FinishReadbackQueueRead(ReadbackQueueRead& read);
	void WaitReadbackQueueCopies(vk::Buffer buffer);
	void DrainHostCopies();
	void NoteWrite(uint64_t vaddr, uint64_t size) {
		if (m_copy_feedback_entries != 0) InvalidateCopyFeedback(vaddr, size);
		if (m_pending_writes.empty() || m_pending_writes.back() != std::pair {vaddr, size}) {
			m_pending_writes.emplace_back(vaddr, size);
			for (const auto& hot: m_readback_hot) {
				if (vaddr < hot.end && hot.begin < vaddr + size) {
					m_flush_requested = true;
					break;
				}
			}
		}
	}
	void NoteReadbackWait(uint64_t begin, uint64_t end, bool device_address, bool drained);
	[[nodiscard]] bool TakeFlushRequest();

	class WriteTicks {
	public:
		void Assign(uint64_t begin, uint64_t end, uint64_t tick);
		[[nodiscard]] uint64_t Latest(uint64_t begin, uint64_t end) const;

	private:
		struct Range {
			uint64_t end  = 0;
			uint64_t tick = 0;
		};
		std::map<uint64_t, Range> m_ranges;
		struct Recent {
			uint64_t begin = 0;
			uint64_t end   = 0;
			uint64_t tick  = 0;
		};
		std::array<Recent, 256> m_recent {};
	};
	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	RenderContext*                                    m_resources = nullptr;
	std::atomic<uint64_t>                             m_registration_epoch {1};
	std::unique_ptr<std::atomic<uint64_t>[]> m_region_registrations =
	    std::make_unique<std::atomic<uint64_t>[]>(TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE);
	std::vector<GuestRange> m_registration_spans;
	bool                    m_registration_spans_lost = true;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	struct DirtyPageMask {
		uint64_t                 page = UINT64_MAX;
		std::array<uint64_t, 64> bits {};
	};
	std::array<DirtyPageMask, 128> m_dirty_masks {};
	void NoteGpuModifiedChange(uint64_t vaddr, uint64_t size, bool dirty);
	struct GpuLineSlot {
		std::atomic<uint64_t> page {0};
		std::atomic<uint64_t> current {0};
		std::atomic<uint64_t> ever {0};
	};
	struct GpuLineRange {
		std::atomic<uint64_t> begin {0};
		std::atomic<uint64_t> end {0};
	};
	static constexpr size_t                       GpuLineSlots  = size_t {1} << 20u;
	static constexpr size_t                       GpuLineProbes = 16;
	static constexpr size_t                       GpuLineRanges = 64;
	std::unique_ptr<GpuLineSlot[]>                m_gpu_line_slots = std::make_unique<GpuLineSlot[]>(GpuLineSlots);
	std::array<GpuLineRange, GpuLineRanges>       m_gpu_line_current_ranges {};
	std::array<GpuLineRange, GpuLineRanges>       m_gpu_line_ever_ranges {};
	std::atomic<bool>                             m_gpu_line_overflow {false};
	static constexpr size_t                       GpuLineMemo = 8;
	std::array<GpuLineRange, GpuLineMemo>         m_gpu_line_memo {};
	size_t                                        m_gpu_line_memo_next = 0;
	void NoteGpuLines(uint64_t vaddr, uint64_t size) noexcept;
	void ForgetCurrentGpuLines(uint64_t vaddr, uint64_t size) noexcept;
	void UnmarkGpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool GpuLinesClean(uint64_t vaddr, uint64_t size, bool ever) const noexcept;
	std::array<RangeSet, 16>                          m_checked_color_metadata;
	RangeSet                                          m_released_ranges;
	bool                                              m_record_released = false;
	static constexpr size_t                           StaleRingSize = 32;
	std::array<std::atomic<uint64_t>, StaleRingSize * 2> m_stale_ring {};
	std::atomic<uint32_t>                             m_stale_ring_next {0};
	MemoryTracker                                     m_memory_tracker;
	uint64_t                                           m_gpu_write_generation = 0;
	WriteTicks                                        m_write_ticks;
	std::vector<std::pair<uint64_t, uint64_t>>        m_pending_writes;
	struct InFlightReadback {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::vector<InFlightReadback>                     m_inflight_readbacks;
	uint64_t                                          m_device_address_tick    = 0;
	bool                                              m_device_address_pending = false;
	struct HotReadback {
		uint64_t begin    = 0;
		uint64_t end      = 0;
		uint64_t last_hit = 0;
	};
	static constexpr uint64_t HotReadbackLifetime = 4096;
	static constexpr size_t   MaxHotReadbacks     = 32;
	std::vector<HotReadback>  m_readback_hot;
	uint64_t                  m_readback_count = 0;
	uint64_t                  m_device_address_hot = 0;
	bool                      m_flush_requested    = false;
	static constexpr std::chrono::microseconds MinRequestedFlushInterval {1000};
	std::chrono::steady_clock::time_point      m_last_requested_flush {};
	uint64_t                                   m_flush_deferred_tick = 0;
	struct ReadbackQueue {
		vk::CommandPool         pool    = nullptr;
		vk::CommandBuffer       command = nullptr;
		vk::Fence               fence   = nullptr;
		std::unique_ptr<Buffer> staging;
	};
	ReadbackQueue m_readback;
	struct ReadbackQueueSlot {
		ReadbackQueue     queue;
		vk::Buffer        source = nullptr;
		bool              many = false;
		std::atomic<bool> busy {false};
	};
	std::array<ReadbackQueueSlot, 4> m_readback_slots;
	struct ReadbackQueueRead {
		ReadbackQueueSlot*                 slot = nullptr;
		std::vector<vk::BufferCopy>        copies;
		uint64_t                           buffer_address = 0;
		uint64_t                           total_size     = 0;
		uint64_t                           needed   = 0;
		uint64_t                           recorded = 0;
		std::shared_ptr<std::atomic<bool>> published;
		struct Follower {
			uint64_t                    page           = 0;
			uint64_t                    buffer_address = 0;
			vk::Buffer                  source         = nullptr;
			std::vector<vk::BufferCopy> copies;
		};
		std::vector<Follower> followers;
	};
	struct MirrorLine {
		uint32_t slot       = 0;
		uint64_t tick       = 0;
		uint64_t dirty_tick = 0;
		uint64_t last_use   = 0;
	};
	static constexpr uint64_t MirrorLineSize      = 64;
	static constexpr uint64_t MirrorMaxRead       = 256;
	static constexpr uint32_t MaxMirrorLines      = 8192;
	static constexpr uint64_t MirrorFlushLifetime = 4096;
	void                      MarkMirrorLinesWritten(uint64_t vaddr, uint64_t size, uint64_t tick);
	void                      AddMirrorLine(uint64_t line);
	void                      RecordMirrorCopies(CommandBuffer& command);
	std::map<uint64_t, MirrorLine> m_mirror_lines;
	std::vector<uint64_t>          m_mirror_dirty;
	std::vector<uint32_t>          m_mirror_free_slots;
	std::unique_ptr<Buffer>        m_mirror;
	uint64_t                       m_mirror_uses      = 0;
	uint64_t                       m_mirror_flush_use = 0;
	StreamBuffer                                      m_staging_buffer;
	struct UploadBatch;
	bool                                              m_upload_batching   = false;
	uint64_t                                          m_upload_batch_tick = 0;
	struct UploadBatchBuffer {
		vk::Buffer        buffer;
		vk::DeviceAddress address = 0;
		uint32_t          count   = 0;
	};
	std::vector<UploadBatchBuffer>                    m_upload_batch_buffers;
	std::vector<vk::BufferCopy>                       m_upload_batch_copies;
	vk::PipelineLayout                                m_scatter_layout   = nullptr;
	vk::Pipeline                                      m_scatter_pipeline = nullptr;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_register_epoch     = 0;
	uint64_t m_buffer_set_epoch = 0;
	std::unique_ptr<StreamBuffer> m_table_upload;
	struct ObtainMemo {
		uint64_t vaddr        = 0;
		uint64_t size         = 0;
		uint64_t buffer_epoch = 0;
		uint64_t cpu_epoch    = 0;
		uint64_t offset       = 0;
		BufferId id;
		bool texel = false;
	};
	struct PageMemo {
		uint64_t page           = UINT64_MAX;
		uint64_t buffer_epoch   = 0;
		uint64_t cpu_epoch      = 0;
		uint64_t buffer_address = 0;
		BufferId id;
		uint32_t clean = 0;
	};
	static constexpr uint32_t AllPageBits = (1u << (CACHING_PAGESIZE / TRACKER_PAGE_SIZE)) - 1u;
	struct WrittenMemo {
		uint64_t vaddr        = 0;
		uint64_t size         = 0;
		uint64_t buffer_epoch = 0;
		uint64_t cpu_epoch    = 0;
		uint64_t gpu_epoch    = 0;
		BufferId id;
	};
	struct StreamMemo {
		uint64_t vaddr            = 0;
		uint64_t size             = 0;
		uint64_t window           = UINT64_MAX;
		uint64_t lap              = 0;
		uint64_t write_generation = 0;
		uint64_t offset           = 0;
	};
	static constexpr size_t                                  StreamMemoSlots = 1024;
	bool                                                     m_stream_in_place = false;
	std::unique_ptr<std::array<StreamMemo, StreamMemoSlots>> m_stream_memo =
	    std::make_unique<std::array<StreamMemo, StreamMemoSlots>>();
	static constexpr size_t                                  ObtainMemoSlots = 4096;
	std::unique_ptr<std::array<ObtainMemo, ObtainMemoSlots>> m_obtain_memo =
	    std::make_unique<std::array<ObtainMemo, ObtainMemoSlots>>();
	static constexpr size_t                              PageMemoSlots = 4096;
	std::unique_ptr<std::array<PageMemo, PageMemoSlots>> m_page_memo =
	    std::make_unique<std::array<PageMemo, PageMemoSlots>>();
	static constexpr size_t                                    WrittenMemoSlots = 1024;
	std::unique_ptr<std::array<WrittenMemo, WrittenMemoSlots>> m_written_memo =
	    std::make_unique<std::array<WrittenMemo, WrittenMemoSlots>>();
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	void                                       NoteBufferRegions(const Buffer& buffer, bool insert);
	void                                       GrowRegionTables(uint64_t region);
	std::vector<uint32_t>                      m_region_buffer_count;
	std::vector<uint64_t>                      m_buffer_region_bits;
	std::vector<uint64_t>                      m_forced_region_bits;
	std::vector<uint64_t>                      m_region_forces;
	void NoteUpload(uint64_t address, uint64_t bytes) noexcept {
		if (m_copy_feedback_entries != 0) InvalidateCopyFeedback(address, bytes);
	}
	std::vector<uint64_t>                      m_synced_region_epochs;
	std::vector<std::pair<uint64_t, uint64_t>> m_sync_regions;
	void CollectSpanBuffers(uint64_t span, uint64_t bytes, std::vector<Buffer*>& buffers,
	                        std::vector<std::pair<uint64_t, uint64_t>>& ranges);
	std::vector<Buffer*>                       m_span_buffers;
	std::vector<std::pair<uint64_t, uint64_t>> m_span_ranges;
	std::vector<std::vector<vk::BufferCopy>>   m_span_copies;
	std::vector<uint64_t>                      m_span_sizes;
	std::vector<std::pair<uint64_t, uint64_t>> m_sync_spans;
	struct SetChunk {
		std::vector<Buffer*>                       buffers;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};
	std::vector<std::pair<uint64_t, size_t>> m_set_visits;
	std::vector<SetChunk>                    m_set_chunks;
	uint32_t m_device_syncs     = 0;
	uint32_t m_device_sync_pass = 0;
	uint32_t m_device_full_pass = 0;
	bool     m_kept_pages       = false;
};

}

#endif
