#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/threadSafety.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/hostCopyQueue.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#if defined(ANYPS5_KYTY)
#include <deque>
#endif

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics,
	                 bool deferred_recording = false);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(const HW::Context& registers, const HW::UserConfig& user_config,
	                     const HW::Shader& shaders);
	void           EndRendering() { m_command.EndRendering(); }
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	[[nodiscard]] bool HasPendingGuestOperations(uint64_t address, uint64_t size);
	void               WaitPriorityOperations(uint64_t tick);
	void DeferOperation(Common::UniqueFunction<void>&& operation, uint64_t address = 0,
	                    uint64_t size = UINT64_MAX);
	void DeferPriorityOperation(Common::UniqueFunction<void>&& operation, uint64_t address = 0,
	                            uint64_t size = UINT64_MAX);
	void DeferPriorityOperation(Common::UniqueFunction<void>&&                     operation,
	                            const std::vector<std::pair<uint64_t, uint64_t>>& ranges);
#if defined(ANYPS5_KYTY)
	void DeferPriorityOperationAfter(uint64_t tick, Common::UniqueFunction<void>&& operation,
	                                 uint64_t address, uint64_t size);
	[[nodiscard]] uint64_t NewestPendingGuestOperation(uint64_t address, uint64_t size,
	                                                   uint64_t after);
#endif
	void DeferHostOperation(Common::UniqueFunction<void>&& operation, bool priority = false);
#if defined(ANYPS5_KYTY)
	void DeferFlipCompletion(Common::UniqueFunction<void>&& complete);
#endif
	[[nodiscard]] bool HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] uint64_t SubmittedTick() const noexcept {
		return m_submitted_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	[[nodiscard]] HostCopyQueue* HostCopies() noexcept { return m_host_copies.get(); }
	using SubmitHook = void (*)(void* userdata, CommandBuffer& command, SubmitInfo& submit);
	void SetSubmitHook(SubmitHook hook, void* userdata) noexcept {
		m_submit_hook      = hook;
		m_submit_hook_data = userdata;
	}

private:
	void QueueOperation(Common::UniqueFunction<void>&& operation);
	void QueuePriorityOperation(Common::UniqueFunction<void>&& operation);
	Common::UniqueFunction<void> TrackGuestOperation(Common::UniqueFunction<void>&& operation,
	                                                 uint64_t address, uint64_t size);
	Common::UniqueFunction<void> TrackGuestOperation(Common::UniqueFunction<void>&&                     operation,
	                                                 const std::vector<std::pair<uint64_t, uint64_t>>& ranges);

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();
		[[nodiscard]] std::pair<vk::CommandBuffer, size_t> CommitDeferred();
		void                                               Retire(size_t index, uint64_t tick);

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();
		size_t Acquire(uint64_t busy_until);

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

	struct RecordedSubmit {
		SubmitInfo info;
		uint64_t   tick            = 0;
		uint64_t   host_copies     = 0;
	};

	void BeginNext();
	void RecordingThread();
	void SubmitRecorded(vk::CommandBuffer& buffer, size_t buffer_index,
	                    const RecordedSubmit& recorded);
	void QueueSubmit(vk::CommandBuffer buffer, SubmitInfo& submit, uint64_t tick);
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);
#if defined(ANYPS5_KYTY)
	struct FlipCompletion {
		Common::UniqueFunction<void> complete;
		uint64_t                     submitted_tick = 0;
		uint64_t                     previous_tick  = 0;
	};
	void FlipCompletionThread(std::stop_token stop);
	void DrainFlipCompletions();
#endif

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	const bool                   m_deferred;
	CommandChunkQueue            m_chunk_queue;
	CommandStream                m_stream;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations KYTY_GUARDED_BY(m_operation_mutex);
	std::atomic<uint64_t>        m_pending_front_tick {UINT64_MAX};
	std::atomic<uint64_t>        m_submitted_tick {0};
	std::atomic<uint32_t>        m_pending_skips {0};
	std::queue<PendingOperation> m_priority_operations KYTY_GUARDED_BY(m_operation_mutex);
	Common::AnnotatedMutex       m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	OperationState               m_operation_state KYTY_GUARDED_BY(m_operation_mutex) = OperationState::Open;
	uint64_t                     m_priority_active_tick KYTY_GUARDED_BY(m_operation_mutex) = 0;
	struct GuestWrite {
		uint64_t id    = 0;
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	Common::AnnotatedMutex       m_guest_writes_mutex;
	std::vector<GuestWrite>      m_guest_writes KYTY_GUARDED_BY(m_guest_writes_mutex);
	uint64_t                     m_next_guest_write KYTY_GUARDED_BY(m_guest_writes_mutex) = 0;
	std::unique_ptr<HostCopyQueue> m_host_copies;
	SubmitHook                     m_submit_hook      = nullptr;
	void*                          m_submit_hook_data = nullptr;
#if defined(ANYPS5_KYTY)
	Common::AnnotatedMutex      m_flip_mutex;
	std::condition_variable_any m_flip_changed;
	std::deque<FlipCompletion>  m_flip_completions KYTY_GUARDED_BY(m_flip_mutex);
	std::atomic<bool>           m_flips_pending {false};
	bool                        m_flip_running KYTY_GUARDED_BY(m_flip_mutex) = false;
	uint64_t                    m_last_flip_tick = 0;
	bool                        m_flips_in_place = false;
	std::once_flag              m_flip_thread_once;
	std::jthread                m_flip_thread;
#endif
	std::jthread m_recording_thread;
};

}

#endif
