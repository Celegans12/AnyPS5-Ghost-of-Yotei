#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/gpuOperation.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace Libs::Graphics {

class GuestGpu;

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

// The counters of KYTY_TRACE_PERF (PERF gpu_pipeline).
inline bool                  g_gpu_stage_stats        = false;

// The queue waits: a COND_EXEC block made only of waits runs its waits without its predicate
// being read (CpOpCondExec, CommandProcessor::ResolveAssumedCondBlock).


enum class Pm4ProcessResult { Complete, Blocked };

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

	// The queue waits: a COND_EXEC block of waits the stream entered without reading its predicate. It ends at dword
	// end_dw of the buffer `commands` at stack depth `depth` (a block of waits calls no buffer).
	struct CondBlock {
		bool            active       = false;
		bool            checked      = false; // the liveness guard read the predicate: non-zero, the waits stay
		bool            waited       = false; // a wait of the block suspended the stream
		bool            stuck_logged = false;
		uint64_t        predicate    = 0;
		const uint32_t* commands     = nullptr;
		size_t          depth        = 0;
		uint32_t        end_dw       = 0;
		uint32_t        frame        = 0; // presented frames when the stream entered the block
		int64_t         wait_ns      = 0; // steady clock: the first suspension inside the block
		int64_t         checked_ns   = 0; // steady clock: the guard's read
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
	CondBlock                 m_cond;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id);
	~CommandProcessor();

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush();
	// An end-of-pipe interrupt was queued: its tick must be submitted (CommandScheduler::FlushForInterrupt;
	// deferred while the GPU is busy, bounded).
	void            BufferFlushForInterrupt();
	void            BufferWait();
	// Register writes go through these; operations keep snapshots of the old values.
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
	void DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers, uint32_t draw_initiator,
	                  bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       IndirectDrawRegisters registers, uint32_t draw_initiator, bool indexed);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	// Kept: upstream 70668fbd9 removed the PM4 global barriers; our renderer still orders the game's work with them.
	void EmitGlobalBarrier();
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);
	void WriteLodStats(void* dst, uint32_t size);
	void RunGarbageCollector();
	// BDA sync epoch: the stream read guest memory that orders the guest's earlier writes.
	void PublishDirtyRegions();

	// Every effect of a packet beyond this processor's own state is an operation: built here and run by Execute(), on
	// the execution thread of a pipelined GPU, else when it is emitted.
	// Based on TheCruZ/KytyPS5-GTA 12ec19890 (commandProcessor.h), GPL-2.0.
	// Returns the operation's position in the stream.
	uint64_t EmitCallback(GpuCallback&& callback);
	// A callback that reads the registers of its point of the stream (t_draw_regs).
	uint64_t EmitStateCallback(GpuCallback&& callback);
	// Runs `callback` where the operations execute, after all the ones emitted so far, and waits for it.
	void RunOnExecution(GpuDrain reason, GpuCallback&& callback);
	// Executes an operation this command processor produced.
	void Execute(GpuOperation& operation);
	// Reads guest memory the stream decides on (COND_EXEC, branches): as a serial GPU would, after the earlier operations
	// that write it. Based on TheCruZ/KytyPS5-GTA 12ec19890 (commandProcessor.h), GPL-2.0.
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size);
	// The guest memory the stream reads directly (register tables, command buffers) is written by the operations before
	// this point once this returns.
	void WaitForPendingWrites(const void* address, uint64_t size);
	template <typename T>
	[[nodiscard]] T ReadDecisionValue(const volatile void* address) {
		T value {};
		ReadDecisionMemory(address, &value, sizeof(T));
		return value;
	}

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	void MemSemaphore(uint64_t* counter, bool signal, bool write); // upstream 1b7093805
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	// The queue waits: the stream enters a COND_EXEC block of waits without reading its predicate at
	// `predicate`; the block ends at dword `end_dw` of the current buffer.
	void EnterAssumedCondBlock(uint64_t predicate, uint32_t end_dw);
	// the queue waits liveness guard (GuestGpu::ThreadRun: no queue ran a packet for the guard delay): reads the
	// predicate of the block of waits `execution` is suspended in, as the console's CP does. Zero: the stream continues
	// after the block, as on the console (true). Non-zero: the console waits too, and the block stays (false).
	bool ResolveAssumedCondBlock(Pm4Execution& execution, uint32_t queue_id);
	// The execution is suspended in a block of waits whose predicate the guard has not read yet.
	[[nodiscard]] static bool IsCondGuardCandidate(const Pm4Execution& execution);
	// Logs (once per block) a block of waits that still waits 1 s after the guard read a non-zero predicate.
	static void NoteStuckCondBlock(Pm4Execution& execution, uint32_t queue_id, int64_t now_ns);

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	// The queue waits: clears the block of waits once the stream is past it (all its waits passed).
	static void LeaveCondBlockIfPassed(Pm4Execution& execution);
	void SuspendPm4();
	// True when the indirect draw was handed to the GPU path.
	[[nodiscard]] bool DrawIndirectGpu(const GpuOperation& operation);
	const void*         IndexAddress(uint32_t first_index) const;
	CommandScheduler& GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }
	// The dispatch's wave size (only a change dirties the shader registers).
	void SetCsWaveSize(uint32_t mode);

	// Producer side: operations carry the register state and the processor state they need.
	[[nodiscard]] GpuOperation     MakeOperation(GpuOperationKind kind);
	[[nodiscard]] GpuRegisterState SnapshotState(uint64_t sequence);
	uint64_t                       Emit(GpuOperation&& operation);
	// The guest memory an emitted operation writes, and the value when it is known at emission.
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value = nullptr);

	// Execution side, in operation order. Only these touch the renderer.
	enum class EndOfPipeWrite : uint8_t { None, Dword, Qword, Gds };
	template <typename T>
	[[nodiscard]] static EndOfPipeWrite ClassifyEndOfPipe(uint32_t event_write_source, uint32_t eop_event_type,
	                                                      uint32_t cache_action, uint32_t event_index, bool with_interrupt);
	void ExecuteEndOfPipe(EndOfPipeWrite write, uint64_t sequence, void* dst_gpu_addr, uint64_t value, bool gpu_clock,
	                      bool with_interrupt, uint32_t interrupt_context_id);
	void WriteIndirectOffsets(const GpuOperation& operation, uint32_t vertex_offset, uint32_t instance_offset,
	                          uint32_t first_index);
	void ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args);
	void ExecuteDrawAuto(const GpuOperation& operation, DrawAutoArgs args);
	void ExecuteDrawIndirect(const GpuOperation& operation);
	void ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x, uint32_t thread_group_y,
	                           uint32_t thread_group_z, uint32_t mode);
	void ExecuteDispatchIndirect(const GpuOperation& operation);
	void ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset, uint8_t src_sel, bool src_gds,
	                    uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes);
	void ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address);
	void RecordGlobalBarrier();
	void DeferEopInterrupt(uint32_t interrupt_context_id);
	void WriteLabelNow(void* dst_gpu_addr, const void* value, uint32_t size, bool with_interrupt,
	                   uint32_t interrupt_context_id);

	// Copies of the register state for the operations that read registers. A slot is overwritten only after the last
	// operation that uses it executed. Slots start on cache lines of their own (the resolve and execution threads read them
	// while this thread goes on).
	// Based on TheCruZ/KytyPS5-GTA 12ec19890 (commandProcessor.h), GPL-2.0.
	template <typename T>
	struct SnapshotRing {
		static constexpr size_t Size = 512;
		struct alignas(64) Slot {
			T value;
		};
		std::unique_ptr<Slot[]>     slots;
		std::unique_ptr<uint64_t[]> last_use;
		size_t                      current = Size;
		size_t                      next    = 0;
	};
	template <typename T>
	T* Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty, uint64_t sequence);

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	bool                         m_ctx_dirty    = true;
	bool                         m_ucfg_dirty   = true;
	bool                         m_sh_ctx_dirty = true;
	SnapshotRing<HW::Context>    m_ctx_snapshots;
	SnapshotRing<HW::UserConfig> m_ucfg_snapshots;
	SnapshotRing<HW::Shader>     m_sh_ctx_snapshots;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: NUM_INSTANCES as the packets set it. Indirect draws update it for subsequent draws when they
	// execute (m_indirect_instances).
	GpuNumInstances m_num_instances;
	// The instance count of the last executed CPU-expanded indirect draw, and the NUM_INSTANCES sequence it followed.
	uint32_t m_indirect_instances          = 0;
	uint64_t m_indirect_instances_sequence = 0;
	bool     m_has_indirect_instances      = false;
	// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS: the records of the last GPU-executed indirect packet while NUM_INSTANCES is
	// theirs (args 0: no such packet since the CPU expansion; another sequence: the count was set since).
	struct IndirectRecords {
		uint64_t args       = 0;
		uint64_t count_addr = 0;
		uint32_t max_count  = 0;
		uint32_t stride     = 0;
		uint64_t sequence   = 0;
		bool     indexed    = false;
	} m_num_instances_records;
	// The instance count a draw without its own inherits (DrawIndex, DrawIndexAuto).
	[[nodiscard]] uint32_t ResolveNumInstances(const GpuOperation& operation);
	// KYTY_UPSTREAM_GPU_INDIRECT_RECORDS=2: a draw of the stream (not of the CPU expansion) or a GPU-executed packet
	// (`named`: the GS user data it names itself) whose shader is handed a stale register.
	void CheckStaleIndirectSgprs(const GpuOperation& operation, uint32_t named, const char* kind);
	// The shader registers of a CPU-expanded indirect draw with its offsets written to user SGPRs.
	std::unique_ptr<HW::Shader> m_indirect_shaders;

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
