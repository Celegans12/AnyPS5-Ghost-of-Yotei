#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RAYTRACING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RAYTRACING_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace ShaderRecompiler::IR {
struct ShaderInfo;
}

class BufferCache;
class CommandRecorder;
class CommandScheduler;
class RenderContext;
struct GraphicContext;
struct RayTracingTestAccess;


struct RayTracingGuestTlas {
	uint64_t end       = 0;
	uint32_t instances = 0;
};


class RayTracing {
public:
	explicit RayTracing(RenderContext& context);
	~RayTracing();
	KYTY_CLASS_NO_COPY(RayTracing);

	[[nodiscard]] vk::DeviceAddress Prepare(uint64_t tlas, uint64_t frame);

	void RunGarbageCollector();

	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }
	[[nodiscard]] static bool NativeTraversals() noexcept;
	static void SetDeviceSupport(bool supported) noexcept;

	struct TlasSource {
		uint32_t                user_sgpr     = 0;
		uint32_t                header_offset = 0;
		uint32_t                root          = 0;
		std::array<uint32_t, 4> layout {};
		std::array<uint32_t, 2> header_slots {};
		uint32_t                header_vsharp = 0xffffffffu;
		uint64_t                header        = 0;
	};
	static constexpr uint32_t PsrHeaderLayout = 1u << 30u;
	[[nodiscard]] static uint64_t UserPointer(const TlasSource& source, uint64_t pointer,
	                                          const std::function<bool(uint64_t, uint64_t)>& readable);
	[[nodiscard]] static bool HeaderInUserData(const TlasSource& source) noexcept;
	[[nodiscard]] static constexpr uint64_t UserDataHeader(uint64_t pointer) noexcept {
		return pointer & 0xffffffffffffull;
	}
	[[nodiscard]] static bool FindTlasSource(const ShaderRecompiler::IR::ShaderInfo& info,
	                                         TlasSource& source) noexcept;
	[[nodiscard]] static bool FindGuestTlasSource(const ShaderRecompiler::IR::ShaderInfo& info,
	                                              TlasSource& source) noexcept;

	using GuestTlas = RayTracingGuestTlas;
	[[nodiscard]] GuestTlas ReadGuestTlas(uint64_t header, const TlasSource& source,
	                                      const std::function<bool(uint64_t, uint64_t)>& readable,
	                                      std::vector<std::pair<uint64_t, uint64_t>>& blases);
	void PendingWalks(std::vector<uint64_t>& blases);
	void AnnounceWalks(std::span<const uint64_t> blases);
	void NoteTlasInstances(uint32_t count) noexcept { m_next_instances = count; }

	void Prepare(const CommandRecorder& command, uint64_t user_pointer, const TlasSource& source,
	             const vk::DescriptorBufferInfo& flattened_srt, uint64_t frame, uint64_t shader_hash,
	             bool count_rays = false, bool per_lane = false);

	struct Preparation {
		uint64_t                tick         = 0;
		uint64_t                frame        = 0;
		uint64_t                user_pointer = 0;
		uint64_t                header       = 0;
		uint64_t                generation   = 0;
		uint32_t                header_offset = 0;
		uint32_t                root         = 0;
		std::array<uint32_t, 4> layout {};

		bool operator==(const Preparation&) const = default;
	};
	[[nodiscard]] bool Prepared(const Preparation& preparation) const noexcept {
		return m_reusable && m_reuse == preparation;
	}
	void KeepPrepared(const Preparation& preparation) noexcept {
		m_reuse    = preparation;
		m_reusable = true;
	}
	void NoteGuestOperation() noexcept { m_generation.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Generation() const noexcept { return m_generation.load(std::memory_order_relaxed); }

	enum class HleTlasFormat : uint32_t {
		Base460,
		Base252,
	};
	void RecordHleTlas(const CommandRecorder& command, uint64_t input_va, uint32_t count,
	                   uint32_t stride, uint64_t header_va, HleTlasFormat format);
	static constexpr uint32_t HleTlasMaxInstances = 4096;
	static uint32_t HleTlasHeader(uint32_t count, HleTlasFormat format,
	                              std::array<uint32_t, 32>& header) noexcept;

	struct HleRefitGeometry {
		uint64_t                 blas     = 0;
		uint32_t                 geometry = 0;
		uint32_t                 pad      = 0;
		std::array<uint32_t, 4>  vsharp {};
		std::array<float, 16>    matrix {};
	};
	void RecordHleRefit(const CommandRecorder& command, std::span<const uint64_t> blases,
	                    std::span<const HleRefitGeometry> geometries, uint32_t header_layout,
	                    bool arguments = false);
	static constexpr uint32_t HleRefitBlasHeaderBytes = 256;
	[[nodiscard]] static std::vector<std::pair<uint64_t, uint32_t>> DropHleRefitOutsideGuest(
	    std::vector<uint64_t>& blases, std::vector<uint32_t>& sizes, std::vector<HleRefitGeometry>& geometries,
	    const std::function<uint32_t(uint64_t)>& read_size);

	struct GuestBlasBuild {
		uint64_t                               blas_va = 0;
		std::vector<std::pair<uint32_t, bool>> geometries;
		uint32_t                               header_layout = 0;
	};
	void NoteGuestBlasBuilds(std::span<const GuestBlasBuild> builds);

	void NoteGuestBlasDecompressed(uint64_t blas_va, uint64_t memory_size);

	enum class GuestDispatch : uint32_t { Empty, Count };
	static void NoteGuestDispatch(GuestDispatch kind, uint64_t shader_hash) noexcept;
	static void NoteGuestRays(uint64_t shader_hash, uint64_t rays);
	struct OracleCounts {
		uint64_t returned   = 0;
		uint64_t rays       = 0;
		uint64_t candidates = 0;
		uint64_t rounds     = 0;
		uint64_t waves      = 0;
	};
	static void NoteOracle(uint64_t shader_hash, const OracleCounts& counts);

	[[nodiscard]] uint64_t ContextAddress() const noexcept { return m_context_address; }
	[[nodiscard]] vk::Buffer ContextBuffer() const noexcept;
	struct PreparationOutcome {
		uint32_t missing;
		bool     uncached;
		bool     full;
	};
	[[nodiscard]] uint64_t LastPreparation() const noexcept { return m_preparations.load(std::memory_order_acquire); }
	[[nodiscard]] std::optional<PreparationOutcome> Outcome(uint64_t preparation) const;

	static constexpr uint32_t MaxContexts = 4;
	static constexpr uint64_t EvictFrames = 1800;

private:
	friend struct RayTracingTestAccess;


	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	BufferCache&                                      m_buffer_cache;
	RenderContext&                                    m_context;

	void InitializeSlots();
	void DestroySlots();

	struct Allocation {
		Buffer*           buffer  = nullptr;
		uint64_t          offset  = 0;
		vk::DeviceAddress address = 0;
		uint64_t          size    = 0;
	};

	class Arena {
	public:
		Arena(GraphicContext& graphics, CommandScheduler& scheduler): m_graphics(graphics),
		                                                              m_scheduler(scheduler) {}
		Allocation Allocate(uint64_t size, uint64_t alignment);
		void       Free(const Allocation& allocation);
		[[nodiscard]] uint64_t Used() const noexcept { return m_used; }
		[[nodiscard]] uint64_t Reserved() const noexcept { return m_reserved; }
		bool BeginDrain();
		void EndDrain() noexcept { m_draining = nullptr; }
		[[nodiscard]] const Buffer* Draining() const noexcept { return m_draining; }

	private:
		static constexpr uint64_t ChunkSize = uint64_t {64} << 20u;
		struct Chunk {
			std::unique_ptr<Buffer>      buffer;
			std::map<uint64_t, uint64_t> free_ranges;
			uint64_t                     free_bytes = 0;
			bool                         dedicated  = false;
		};
		GraphicContext&                     m_graphics;
		CommandScheduler&                   m_scheduler;
		std::vector<std::unique_ptr<Chunk>> m_chunks;
		uint64_t                            m_used     = 0;
		uint64_t                            m_reserved = 0;
		const Buffer*                       m_draining = nullptr;
	};

	struct SlotSizes {
		uint32_t capacity0 = 0;
		uint32_t capacity1 = 0;
		uint64_t storage   = 0;
		uint64_t vertices0 = 0;
		uint64_t vertices1 = 0;
		uint64_t entries   = 0;
		uint64_t scratch   = 0;
		[[nodiscard]] uint64_t Total() const noexcept { return storage + vertices0 + vertices1 + entries; }
	};

	struct Slot {
		uint64_t                     blas_va     = 0;
		uint32_t                     root        = 0;
		uint32_t                     table_index = 0;
		uint32_t                     capacity0   = 0;
		uint32_t                     capacity1   = 0;
		uint32_t                     leaf_count  = 0;
		Allocation                   vertices0;
		Allocation                   vertices1;
		Allocation                   entries;
		Allocation                   storage;
		vk::AccelerationStructureKHR handle      = nullptr;
		vk::DeviceAddress            as_address  = 0;
		uint64_t                     build_scratch = 0;
		bool                         built       = false;
		bool                         needs_build = false;
		uint32_t                     dynamic_until = 0;
		uint64_t                     built_frame   = UINT64_MAX;
		uint64_t                     refit_seq       = 0;
		uint64_t                     built_refit_seq = 0;
		bool                         fast_built      = false;
		uint32_t                     gathers         = 0;
		uint64_t                     last_use_frame  = 0;
		bool                         evicted         = false;
		std::vector<std::array<uint32_t, 4>> layout;
		uint32_t                     header_layout    = 0;
		uint32_t                     host_table_index = UINT32_MAX;
		bool                         replaced         = false;
	};

	struct Retired {
		uint64_t                     tick       = 0;
		uint64_t                     prepare_id = 0;
		vk::AccelerationStructureKHR handle     = nullptr;
		std::array<Allocation, 4>    allocations {};
		uint32_t                     slot       = UINT32_MAX;
	};

	struct PendingSlot {
		uint64_t blas_va     = 0;
		uint32_t root        = 0;
		uint32_t table_index = 0;
		uint32_t leaf_count  = 0;
		uint32_t opaque      = 0;
		uint32_t alpha       = 0;
	};

	struct Readback {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick  = 0;
		uint32_t                build_count = 0;
		std::vector<uint32_t>   build_slots;
		uint32_t                source_key  = 0;
		uint64_t                frame       = 0;
		uint64_t                shader_hash = 0;
		uint64_t                preparation = 0;
		uint32_t                capacity    = 0;
	};
	struct IncompleteRun {
		uint32_t count = 0;
		bool     noted = false;
	};

	void CreatePipelines();
	void CreateBuffers();
	void CreateTlas(uint32_t capacity);
	bool CreateSlot(const PendingSlot& pending);
	uint32_t CreateHostSlot(uint64_t blas_va, const SlotSizes& sizes);
	[[nodiscard]] uint32_t FindHostSlotIndex(uint64_t blas_va) const;
	bool ComputeSlotSizes(uint32_t opaque, uint32_t alpha, SlotSizes& sizes, bool exact = false);
	void AllocateSlot(Slot& slot, const SlotSizes& sizes);
	[[nodiscard]] static uint64_t SlotBytes(const Slot& slot) noexcept;
	void ReportAgain(uint64_t blas_va, uint32_t root, uint32_t table_index);
	void RetireResources(Slot& slot, uint32_t slot_index);
	void DestroyRetired();
	void EvictUnused(std::vector<uint32_t>& evicted);
	void EvictOverBudget(std::vector<uint32_t>& evicted);
	void EvictSlot(uint32_t index);
	void Defragment(std::vector<uint32_t>& moved);
	bool RecordBuilds(const CommandRecorder& command);
	bool RecordBuildBatch(const CommandRecorder& command, bool urgent_only, bool& gathered);
	void ProcessReadback(Readback& readback);

	bool              m_enabled = false;
	bool              m_cleared = false;

	vk::DescriptorSetLayout m_walk_descriptor_layout = nullptr;
	vk::PipelineLayout      m_tlas_layout       = nullptr;
	vk::PipelineLayout      m_blas_layout       = nullptr;
	vk::Pipeline            m_tlas_pipeline     = nullptr;
	vk::Pipeline            m_blas_pipeline     = nullptr;
	vk::PipelineLayout      m_hle_tlas_layout   = nullptr;
	vk::Pipeline            m_hle_tlas_pipeline = nullptr;
	std::unique_ptr<Buffer> m_hle_tlas_scratch;
	vk::PipelineLayout      m_hle_refit_layout   = nullptr;
	vk::Pipeline            m_hle_refit_pipeline = nullptr;
	std::unique_ptr<Buffer> m_hle_jobs;
	std::unique_ptr<Buffer> m_hle_pool;
	std::unique_ptr<Buffer> m_hle_state;

	std::unique_ptr<Buffer> m_walk_state;
	std::unique_ptr<Buffer> m_queue;
	std::unique_ptr<Buffer> m_instance_nodes;
	std::unique_ptr<Buffer> m_instances;
	std::unique_ptr<Buffer> m_table;
	std::unique_ptr<Buffer> m_slots;
	std::unique_ptr<Buffer> m_missing;
	std::unique_ptr<Buffer> m_context_table;
	std::unique_ptr<Buffer> m_pool;
	std::unique_ptr<Buffer> m_verify_list;
	std::unique_ptr<Buffer> m_changed;
	std::unique_ptr<Buffer> m_jobs;
	std::unique_ptr<Buffer> m_layouts;
	std::unique_ptr<Buffer> m_build_scratch;
	std::unique_ptr<Buffer> m_tlas_storage;
	vk::AccelerationStructureKHR m_tlas         = nullptr;
	vk::DeviceAddress            m_tlas_address = 0;
	uint64_t                     m_tlas_scratch_size = 0;
	uint32_t                     m_tlas_capacity     = 0;
	uint64_t                     m_context_address   = 0;
	Arena                        m_arena;

	std::array<Readback, 32> m_readbacks;
	Preparation              m_reuse {};
	bool                     m_reusable = false;
	std::atomic<uint64_t>    m_generation {0};
	uint32_t                m_readback_index = 0;
	std::atomic<uint64_t>   m_preparations {0};
	mutable std::mutex      m_outcome_mutex;
	std::array<std::pair<uint64_t, PreparationOutcome>, 64> m_outcomes {};

	Common::Mutex             m_mutex;
	std::vector<PendingSlot>  m_new_slots;
	std::vector<std::unique_ptr<Slot>> m_slot_list;
	std::unordered_map<uint64_t, std::vector<uint32_t>> m_slots_by_blas;
	uint64_t m_refit_seq = 0;
	std::vector<uint32_t>     m_build_queue;
	std::vector<uint32_t>     m_dynamic;
	std::vector<std::array<uint32_t, 3>> m_grow;
	std::vector<uint32_t>     m_retry;
	struct RetryState {
		uint32_t count       = 0;
		uint32_t table_index = 0;
	};
	std::unordered_map<uint64_t, RetryState> m_retries;
	std::vector<std::pair<uint64_t, uint32_t>> m_late_retries;
	std::vector<uint32_t>     m_free_slots;
	std::deque<Retired>       m_retired;
	uint64_t                  m_retired_bytes     = 0;
	uint64_t                  m_budget            = 0;
	uint64_t                  m_defrag_after      = 0;
	uint64_t                  m_defrag_started    = 0;
	uint64_t                  m_last_scan_prepare = 0;
	std::unordered_map<uint32_t, uint32_t> m_last_instance_counts;
	std::unordered_map<uint32_t, uint32_t> m_instance_floors;
	std::unordered_map<uint32_t, IncompleteRun> m_incomplete;
	std::unordered_set<uint32_t>           m_cpu_counted_sources;
	std::array<std::pair<uint64_t, GuestTlas>, 8> m_guest_tlases {};
	uint32_t                                      m_guest_tlas_next = 0;
	uint32_t                                      m_next_instances  = 0;
	uint32_t                  m_prepare_id          = 0;
	std::array<std::pair<uint64_t, uint64_t>, 8> m_checked_headers {};
	uint32_t                                     m_checked_next = 0;
	uint32_t                  m_blas_flags          = 0;
	uint64_t                  m_walk_frame               = 0;
	std::vector<uint64_t>                                 m_host_keys;
	std::vector<std::tuple<uint32_t, uint64_t, uint32_t>> m_host_table_writes;
	std::vector<uint32_t>     m_moved_slots;
	std::vector<uint32_t> m_give_up;
	std::vector<uint64_t> m_announced_walks;
	bool                  m_walks_announced = false;
};

}

#endif
