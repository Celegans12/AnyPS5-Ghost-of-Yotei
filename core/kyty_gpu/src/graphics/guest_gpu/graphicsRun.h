#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/spscQueue.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <utility>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	// On the GPU thread it runs in place.
	void               SendCommand(Common::UniqueFunction<void>&& command);
	// Always through the queue: it runs at a packet boundary, also when sent from the GPU thread (from a fault handler there,
	// where running in place would run under whatever the thread holds, before the faulting instruction runs again).
	void               PostCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	// Retries blocked submissions now instead of after the 100 ms stall timeout.
	void               Wake();

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> draw_commands,
	            std::span<const uint32_t> constant_commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	// The thread that owns the renderer: the execution thread.
	[[nodiscard]] static bool IsGpuThread() noexcept;
	// Command processor thread.
	void EmitOperation(GpuOperation&& operation);
	// Waits until the operations emitted so far executed.
	void WaitForExecution(GpuDrain reason);
	// The registers bound while nothing but an operation reads them.
	[[nodiscard]] const GpuRegisterState& NeutralState() const noexcept { return m_neutral_state; }
	// The thread that owns the renderer, while it waits for another thread (a compile), which may in
	// turn wait for a command: runs the commands queued so far.
	// Based on TheCruZ/KytyPS5-GTA e7036cb45 (graphicsRun.cpp), GPL-2.0.
	static void RunCommandsWhileWaiting();
	// That thread has commands to run.
	[[nodiscard]] static bool CommandsWaiting() noexcept;
	// A thread waits at all for such commands (false: its plain wait is fine).
	[[nodiscard]] static bool WaitsServeCommands() noexcept;
	// Numbers the next operation of the stream, across all command processors.
	[[nodiscard]] uint64_t TakeSequence() noexcept { return ++m_emitted; }
	[[nodiscard]] uint64_t NextSequence() const noexcept { return m_emitted + 1; }
	// The operation `sequence` and everything before it executed when this returns: their snapshots are free.
	void WaitForSnapshot(uint64_t sequence);

	// The journal of writes that are not in guest memory yet, for the reads the stream decides on. Oldest first; the
	// thread that makes the operations owns it. Based on TheCruZ/KytyPS5-GTA 12ec19890 (graphicsRun.h, graphicsRun.cpp),
	// GPL-2.0; the labels that land when the host GPU gets there (labels at completion) and the writes into direct ranges
	// are ours.
	// A write of an emitted operation; `value` null when it is known only when the operation executes.
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value);
	// A deferred label: in guest memory when its publish ran (MarkPublished), not when its operation executed.
	void NoteDeferredWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value);
	// Any thread, in sequence order (the host GPU's completions are): the label of `sequence` is in guest memory.
	void MarkPublished(uint64_t sequence) noexcept {
		if (sequence > m_published.load(std::memory_order_relaxed)) m_published.store(sequence, std::memory_order_release);
	}
	// A dword the operation records into the stream (a write into a direct range): in guest memory when the tick it was
	// recorded in finished.
	void NoteStreamWrite(uint64_t sequence, uint64_t address, uint32_t value);
	// Which pending writes a read sees. Stream: those of operations not executed yet, the writes that come before the
	// read in the stream and have not happened; not the labels, which land when the host GPU gets there. Overlay
	// (WAIT_REG_MEM): also the labels and direct-range writes recorded but not in guest memory yet (one in-order host
	// queue: a wait recorded after the label is ordered after its work).
	enum class DecisionRead { Stream, Overlay };
	// Reads `address` (at most 8 bytes) with the pending writes over it. `covered`: they cover all of it.
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size, DecisionRead scope,
	                        bool* covered = nullptr);
	// The writes of the operations not executed yet into the range are in guest memory when this returns.
	void WaitForPendingWrites(uint64_t address, uint64_t size, bool unknown_only);

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		uint64_t                  flip_request_id   = 0;
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	// Makes this thread the one that owns the renderer.
	void              BecomeRendererOwner();
	void              ExecutionThread();
	void              ResolveThread();
	// Execution thread: runs the queued commands, then lets the blocked queues try again.
	void              RunCommands();
	// Execution thread with no operation: waits for work; false when the GPU stops.
	bool              WaitForOperations();
	void              RingExecution() noexcept;
	void              StopExecution();
	// Command processor thread: runs a command on the execution thread, after the operations emitted so far, and waits.
	void              ExecuteSync(Common::UniqueFunction<void>&& command);
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);
	// the queue waits liveness guard, m_queue_mutex held, no queue runnable: the queue whose suspended stream sits
	// in a block of waits the guard has not checked, once no slice ran a packet for the guard delay; -1 otherwise.
	int CondGuardQueue();
	// The guard's exact read of that block's predicate (queue lock released: it may wait for the GPU). True when the
	// predicate was zero and the stream now continues after the block.
	bool ResolveCondGuard(Submission& submission);
	// Guest queue id of a queue index (0 = graphics, else the compute queue 0x20..0x57).
	static uint32_t GuestQueueId(uint32_t queue_index) {
		return queue_index == 0 ? 0u : ComputeQueueBase + queue_index - 1u;
	}

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_wake_pending      = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	uint64_t        m_emitted   = 0;
	Common::ProgressCounter m_executed;
	// The pipeline: operations ring from the command processor thread to the resolve thread and from there to the execution
	// thread, which also serves m_commands and so may sleep on the bell, not in the ring.
	static constexpr size_t         OperationQueueSize = 4096;
	static constexpr size_t         ResolvedQueueSize  = 256; // RenderExecutor::ResolvedDrawRing holds more
	Common::SpscQueue<GpuOperation> m_operations;
	Common::SpscQueue<GpuOperation> m_resolved;
	std::atomic_bool                m_execution_sleeping {false};
	std::atomic_uint32_t            m_execution_bell {0};
	std::atomic_bool                m_execution_stop {false};
	bool             m_commands_running = false; // m_queue_mutex
	// Game and host threads bump it when they queue work, under m_queue_mutex; the command processor spins on it.
	std::atomic_uint64_t m_work_epoch {0};
	HW::Context          m_neutral_context;
	HW::UserConfig       m_neutral_user_config;
	HW::Shader           m_neutral_shaders;
	GpuRegisterState     m_neutral_state;
	std::atomic_uint64_t    m_published {0}; // the newest label in guest memory
	struct GuestWrite {
		uint64_t sequence = 0;
		uint64_t address  = 0;
		uint64_t size     = 0;
		uint64_t tick     = 0; // recorded into the stream: an upper bound of its tick, once it executed
		bool     known      = false;
		bool     at_publish = false; // a deferred label
		bool     stream     = false; // recorded into the stream
		uint8_t  value[8]   = {};
	};
	// A ring, oldest first. It holds more writes than the host GPU has in flight.
	static constexpr size_t PendingWriteCapacity = 16384;
	std::unique_ptr<GuestWrite[]> m_pending_writes;
	size_t                        m_pending_first = 0;
	size_t                        m_pending_count = 0;
	size_t                        m_pending_ticked = 0; // writes from the oldest on that had their chance at a tick
	void                          ForgetExecutedWrites();
	[[nodiscard]] bool            WriteLanded(GuestWrite& write, uint64_t executed, uint64_t published, uint64_t known_tick);
	GuestWrite&                   NewWrite();
	void                          WaitForOldestWrite();
	// Puts the pending writes over `value`, read from `address`.
	void ApplyPendingWrites(const volatile void* address, void* value, size_t size, DecisionRead scope, bool* covered);
	// GPU thread, the queue waits guard: when a slice last ran a packet, and whether the current slice did.
	std::chrono::steady_clock::time_point m_last_progress {};
	bool                                  m_slice_progressed = false;
	std::jthread    m_executor;
	std::jthread    m_resolver;
	std::jthread    m_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
