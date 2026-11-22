#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/command_processor/drawStateObserver.h"
#include "graphics/guest_gpu/command_processor/gpuOperation.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/guest_gpu/psrShaders.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <memory>
#include <span>
#include <unordered_map>
#include <array>
#include <vector>

namespace Libs::Graphics {

class GuestGpu;

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked, Yielded };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_yield         = false;
	bool                      m_made_progress = false;
	uint64_t                  m_scout_sequence = 0;

public:
	void SetScoutSequence(uint64_t sequence) noexcept { m_scout_sequence = sequence; }
};

class CommandProcessor {
public:
	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ResetQueue(uint32_t op);
	void ApplyContextStateOperation(ContextStateOperation operation);

	void AttachPipeline(GuestGpu* gpu) { m_pipeline = gpu; }

	void BufferInit();
	void BufferFlush();
	void BufferFlushIfBatchReady();
	void BufferWait();
	HW::Context& GetCtx() {
		m_ctx_dirty = true;
		return m_ctx;
	}
	HW::UserConfig& GetUcfg() {
		m_ucfg_dirty = true;
		return m_ucfg;
	}
	HW::Shader& GetShCtx() {
		m_sh_ctx_dirty = true;
		return m_sh_ctx;
	}

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       IndirectDrawRegisters registers, uint32_t draw_initiator, bool indexed,
	                       const uint32_t* run_offsets = nullptr);
	[[nodiscard]] bool TakesDrawRuns() const;
	void DrawIndirectRun(const uint32_t* packets, uint32_t count, IndirectDrawRegisters registers,
	                     uint32_t draw_initiator);
	void ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address, uint64_t data,
	                uint32_t interrupt_context_id, bool gl2_writeback);
	void WriteBackImages();
	void Flip(int handle, int index, int flip_mode, int64_t flip_arg);
	void PrepareCpuFlip(uint64_t request_id);
	void DeferEopInterrupt(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(const uint64_t* label);
	void ProcessPssr(const Pssr::Request& request);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);
	void WriteLodStats(void* dst, uint32_t size, uint32_t control);
	void RunGarbageCollector();
	void EmitHostCommand(Common::UniqueFunction<void>&& command);
	bool HleTlasDispatch(uint64_t data_addr);
	bool HleBlasDispatch(uint64_t data_addr);
	void EmitHleRefit();
	void EmitHleRefitArgs();
	void EmitHleTlas();
	void PsrBlasDispatch(uint64_t data_addr);
	void GuestTlasDispatch(uint64_t data_addr);
	void EmitPsrBlasBuilt();
	Psr::Shader ComputePsrShader(uint64_t data_addr);
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size);
	template <typename T>
	[[nodiscard]] T ReadDecisionValue(const volatile void* address) {
		T value {};
		ReadDecisionMemory(address, &value, sizeof(T));
		return value;
	}

	void Execute(GpuOperation& operation);

#if defined(ANYPS5_KYTY)
	void WaitForRewind(bool valid, const uint32_t* control);
#else
	void WaitForRewind(bool valid);
#endif

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op);
	void YieldIfWaitedOn(uint64_t address, uint64_t size);
	void MemSemaphore(uint64_t* counter, bool signal, bool write);
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);
	void SyncFetchWithExecution();
	void PrepareCommandFetch(std::span<const uint32_t> commands);
	const uint32_t* StageIndirectRegisters(uint32_t opcode, const uint32_t* packet, uint32_t packet_dw);

	[[nodiscard]] bool IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	void ProcessPm4(Pm4Execution& execution);
	void SuspendPm4();
	const void*         IndexAddress(uint32_t first_index) const;
	static const void*  IndexAddress(uint64_t base, uint32_t type_and_size, uint32_t first_index);
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	[[nodiscard]] GpuOperation MakeOperation(GpuOperationKind kind);
	[[nodiscard]] GpuRegisterState SnapshotState(uint64_t sequence);
	uint64_t                   Emit(GpuOperation&& operation);
	uint64_t                   EmitCallback(GpuCallback&& callback, const GpuGuestWrite& write = {});
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
	                    const void* value = nullptr);

	void BindState(const GpuRegisterState& state);
	void NoteRecordedWork();
	[[nodiscard]] uint32_t ResolveNumInstances(const GpuOperation& operation);
	void ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args);
	void ExecuteDrawAuto(const GpuOperation& operation, DrawAutoArgs args);
	void ExecuteDrawIndirect(const GpuOperation& operation);
	bool PatchIndirectDrawOffsets(const GpuOperation& operation, uint32_t vertex_offset,
	                              uint32_t instance_offset, uint32_t first_index,
	                              uint32_t draw_index);
	void ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x,
	                           uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void ExecuteDispatchIndirect(const GpuOperation& operation);
	void ExecuteReleaseMem(uint32_t data_sel, bool interrupt, void* address, uint64_t data,
	                       uint32_t interrupt_context_id);
	void ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address);
	void ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset, uint8_t src_sel,
	                    bool src_gds, uint64_t src_address_or_offset_or_immediate,
	                    uint32_t num_bytes);

	template <typename T>
	struct SnapshotRing {
		static constexpr size_t Size = 512;
		struct alignas(64) Slot {
			T        value;
			uint64_t id = 0;
		};
		std::unique_ptr<Slot[]>     slots;
		std::unique_ptr<uint64_t[]> last_use;
		size_t                      current = Size;
		size_t                      next    = 0;
	};
	static inline std::atomic_uint64_t s_snapshot_ids {0};
	template <typename T>
	const T* Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty, uint64_t sequence,
	                  uint64_t* id = nullptr);

	RenderContext& m_renderer;
	GuestGpu*      m_pipeline = nullptr;
	const int      m_interrupt_event_id;

	alignas(64) HW::Context m_ctx;
	std::optional<HW::Context> m_saved_ctx;
	HW::UserConfig m_ucfg;
	HW::Shader     m_sh_ctx;
	uint32_t       m_index_type_and_size              = 0;
	uint32_t       m_index_buffer_size                = 0;
	uint64_t       m_index_base_addr                  = 0;
	uint64_t       m_draw_indirect_args_base_addr     = 0;
	uint64_t       m_dispatch_indirect_args_base_addr = 0;
	uint64_t       m_fetch_sync_sequence              = 0;
	std::vector<uint32_t> m_indirect_stage;
	std::array<uint32_t, 5> m_indirect_packet {};
	GpuNumInstances m_num_instances;

	bool                         m_ctx_dirty    = true;
	bool                         m_ucfg_dirty   = true;
	bool                         m_sh_ctx_dirty = true;
	SnapshotRing<HW::Context>    m_ctx_snapshots;
	SnapshotRing<HW::UserConfig> m_ucfg_snapshots;
	SnapshotRing<HW::Shader>     m_sh_ctx_snapshots;

	uint64_t m_submit_id      = 0;
	uint64_t        m_scout_sequence = 0;
	bool     m_predicate_skip = false;
	std::unique_ptr<DrawStateObserver> m_draw_observer;
	bool     m_occlusion_wait_requested = false;
	std::atomic<uint32_t> m_deferred_labels {0};
#if defined(ANYPS5_KYTY)
	static bool                         LabelsAfterDumps();
	static bool                         LabelsAroundGpuWrites();
	static inline std::atomic<uint32_t> s_deferred_labels {0};
	const bool                          m_labels_after_dumps = LabelsAfterDumps();
	const bool                          m_labels_around_gpu_writes = LabelsAroundGpuWrites();
#endif
	bool m_wait_since_operation = false;
	bool m_acquire_since_operation = true;

public:
	void NoteAcquireMem() noexcept { m_acquire_since_operation = true; }
#if defined(ANYPS5_KYTY)
	static void LogLabelSummary();
#endif

private:

	struct HleTlasBuild {
		uint64_t                        input  = 0;
		uint64_t                        header = 0;
		uint32_t                        stride = 0;
		uint32_t                        count  = 0;
		RayTracing::HleTlasFormat format = RayTracing::HleTlasFormat::Base460;
		[[nodiscard]] bool Native() const;
	};
	std::vector<HleTlasBuild>              m_hle_tlas_builds;
	uint32_t                               m_hle_tlas_finished = 0;
	bool                                   m_hle_tlas_window   = false;
	uint32_t                               m_guest_tlas_builds = 0;
	bool                                   m_hle_tlas_observe  = false;
	bool                                   m_hle_tlas_known    = false;
	struct HleRefitGeometryRef {
		uint64_t blas     = 0;
		uint64_t table    = 0;
		uint32_t geometry = 0;
	};
	std::vector<uint64_t>            m_hle_refit_blases;
	std::vector<HleRefitGeometryRef> m_hle_refit_geometries;
	uint32_t                         m_hle_refit_finished = 0;
	bool                             m_hle_refit_window   = false;
	size_t                           m_hle_refit_args     = 0;
	struct PsrBlasBuildState {
		uint64_t             table = 0;
		std::vector<uint8_t> seen;
	};
	std::unordered_map<uint64_t, PsrBlasBuildState> m_psr_blas_builds;
	std::unordered_map<uint64_t, uint32_t> m_psr_captured;
	uint64_t                               m_psr_decompressed      = 0;
	uint32_t                               m_psr_decompressed_size = 0;
	RayTracing::GuestBlasBuild m_psr_blas_built;
	std::unordered_map<uint64_t, Psr::Shader> m_cs_psr_shaders;

	alignas(64) uint64_t m_synthetic_occlusion_counter = 0;
	uint64_t m_batch_tick = 0;
	uint32_t m_batch_work = 0;
	uint32_t m_indirect_instances         = 0;
	uint64_t m_indirect_instances_sequence = 0;
	bool     m_has_indirect_instances      = false;
	std::unique_ptr<HW::Shader> m_indirect_shaders;
	DrawIndirectArgs m_gpu_indirect_instances {};
	alignas(64) uint8_t m_end_padding = 0;
};

}

#endif
