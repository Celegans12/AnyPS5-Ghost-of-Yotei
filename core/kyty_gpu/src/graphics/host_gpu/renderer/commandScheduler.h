#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	// D4b step 1 (BufferCache::AppendCloseWindows): called at the start of every Submit, on the submitting thread,
	// while the command buffer can still take commands.
	void SetPreSubmitHook(std::function<void()> hook) { m_pre_submit_hook = std::move(hook); }

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	// Before a draw or dispatch, the open command buffer is submitted (no wait) once 128 draws/dispatches were recorded
	// into it, so the GPU starts on them while the CPU records the next ones.
	void           NoteGpuCommand();
	// An end-of-pipe interrupt was queued for the open command buffer's tick, which must be submitted for
	// it to fire (RELEASE_MEM): at once only when the GPU is idle (submission batching, commandScheduler.cpp).
	void           FlushForInterrupt();
	// An interrupt's submission is deferred; the command processor then calls
	// NoteDeferredPacket after each packet, which submits once the GPU is idle or at the bound.
	[[nodiscard]] bool SubmitDeferred() const noexcept { return m_batch_interrupt_pending.load(std::memory_order_relaxed); }
	void               NoteDeferredPacket();
	// KYTY_TRACE_PERF=1: one "PERF submit" line with this scheduler's submissions by reason since the last one.
	void               ReportSubmitPerf();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// True when WaitPriorityOperations(tick) would return at once: no priority completion queued at or
	// before tick is still waiting or running.
	[[nodiscard]] bool        PriorityOperationsDone(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	std::function<void()> m_pre_submit_hook;
	bool                  m_in_pre_submit_hook = false;
	uint32_t m_commands_since_submit = 0;

	// Why the next Submit happens (FlushFor), for the PERF submit line and the batching's idle arming.
	enum class SubmitReason : uint32_t {
		Other,          // slice end, flip, wait, readback, suspend point, ...
		Periodic,       // SubmitEvery boundary (the GPU was idle there)
		PeriodicLate,   // a deferred boundary, once the GPU was idle
		PeriodicBound,  // a deferred boundary at its bound (GPU still busy)
		Interrupt,      // RELEASE_MEM interrupt (the GPU was idle)
		InterruptLate,  // a deferred interrupt, once the GPU was idle
		InterruptBound, // a deferred interrupt at its packet/time bound (GPU still busy)
		IdleEarly,      // the GPU ran dry while this buffer was being recorded
		Count
	};
	void               FlushFor(SubmitReason reason);
	void               NoteGpuCommandBatched();
	// Every submitted tick is complete (the semaphore is re-read at most every 100 us).
	[[nodiscard]] bool GpuIdle();

	SubmitReason m_submit_reason      = SubmitReason::Other;
	SubmitReason m_last_submit_reason = SubmitReason::Other;
	// The batching's state, all reset by Submit.
	std::atomic_bool m_batch_interrupt_pending {false};
	uint32_t                              m_batch_interrupt_packets = 0;
	std::chrono::steady_clock::time_point m_batch_interrupt_since {};
	bool                                  m_batch_idle_armed = false;
	// KYTY_TRACE_PERF counters (ReportSubmitPerf).
	std::atomic<uint64_t> m_perf_submits[static_cast<uint32_t>(SubmitReason::Count)] {};
	std::atomic<uint64_t> m_perf_deferred_periodic {0};
	std::atomic<uint64_t> m_perf_deferred_interrupts {0};
	std::atomic<uint64_t> m_perf_interrupt_delay_us {0};
	std::atomic<uint64_t> m_perf_interrupt_delay_max_us {0};
	std::atomic<uint64_t> m_perf_interrupt_delays {0};

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
