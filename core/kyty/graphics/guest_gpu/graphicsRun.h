#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/spscQueue.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/scout.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);
	void SendUrgentCommandSync(Common::UniqueFunction<void>&& command, bool drain_copies = true);

	void Submit(std::span<const uint32_t> commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	void SuspendPoint();
#if defined(ANYPS5_KYTY)
	void NoteFrameSubmitted();
#endif
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	[[nodiscard]] static bool IsCommandProcessorThread() noexcept;
	static void RunUrgentCommandsWhileWaiting();

	[[nodiscard]] bool Pipelined() const noexcept { return m_pipelined; }
	[[nodiscard]] uint64_t NextSequence() const noexcept { return m_emitted + 1; }
	uint64_t               EmitOperation(GpuOperation&& operation);
	void NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size, const void* value);
	void ReadDecisionMemory(const volatile void* address, void* value, size_t size);
	void                   WaitForExecution(uint64_t sequence) { m_executed.WaitFor(sequence); }
	void WaitForExecution() { m_executed.WaitFor(m_emitted); }
#if defined(ANYPS5_KYTY)
	void NoteBlockedWait(uint64_t address, uint64_t size, uint64_t reference, uint64_t mask,
	                     uint32_t function);
	void NoteBlockedRewind(uint64_t control);
#else
	void NoteBlockedWait(uint64_t address, uint64_t size);
#endif
	[[nodiscard]] bool OtherQueueWaitsOn(uint64_t address, uint64_t size) const;
	[[nodiscard]] const GpuRegisterState& NeutralState() const noexcept { return m_neutral_state; }
#if defined(ANYPS5_KYTY)

	struct BlockedWait {
		uint64_t address   = 0;
		uint64_t reference = 0;
		uint64_t mask      = 0;
		uint32_t function  = 0;
		uint32_t bytes     = 0;
		uint32_t queue     = 0;
	};
	[[nodiscard]] size_t BlockedWaits(BlockedWait* waits, size_t capacity) const;
	[[nodiscard]] bool   HoldSubmissions(uint32_t timeout_ms);
	void                 ReleaseSubmissions();
#endif

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Commands, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Commands;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		Pm4Execution              execution;
		bool                      blocked         = false;
		bool                      yielded         = false;
		uint64_t                  flip_request_id = 0;
		uint64_t                  scout_sequence  = 0;
#if defined(ANYPS5_KYTY)
		bool                      frame_end       = false;
		bool                      wait_flushed    = false;
#endif
	};

	void              Enqueue(Submission submission);
	void              ProcessCommands();
	void              ExecutionThread();
	void              ResolveThread();
	void              StopExecution();
	void              ExecuteSync(Common::UniqueFunction<void>&& command);
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

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
	bool                                           m_shutdown_complete = false;
#if defined(ANYPS5_KYTY)
	static constexpr ptrdiff_t SuspendPointDepthMax = 64;
	static ptrdiff_t           SuspendPointDepth();
	std::shared_ptr<std::counting_semaphore<SuspendPointDepthMax>> m_suspend_point_ready =
	    std::make_shared<std::counting_semaphore<SuspendPointDepthMax>>(SuspendPointDepth());
	std::shared_ptr<std::binary_semaphore> m_frame_ready = std::make_shared<std::binary_semaphore>(1);
	std::atomic_bool                       m_frame_submitted {false};
#else
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);
#endif

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	uint64_t        m_scout_sequence = 0;
	bool            m_scout_reset    = false;

	struct WaitRange {
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	std::array<WaitRange, QueueCount> m_wait_ranges {};
	uint32_t                          m_waiting_queues = 0;
	uint32_t                          m_current_queue  = 0;
#if defined(ANYPS5_KYTY)

	struct BlockedSlot {
		std::atomic_uint64_t sequence {0};
		std::atomic_uint64_t address {0};
		std::atomic_uint64_t reference {0};
		std::atomic_uint64_t mask {0};
		std::atomic_uint64_t shape {0};
	};
	std::array<BlockedSlot, QueueCount> m_blocked_slots;
	std::atomic_bool                    m_hold {false};
	bool                                m_in_submission = false;
	void RecordBlockedWait(uint64_t address, uint64_t reference, uint64_t mask, uint32_t function,
	                       uint32_t bytes);
#endif

	struct GuestWrite {
		uint64_t sequence = 0;
		uint64_t address  = 0;
		uint64_t size     = 0;
		bool     known    = false;
		uint8_t  value[8] = {};
	};

	const bool                      m_pipelined;
	const bool                      m_resolving;
	Common::SpscQueue<GpuOperation> m_operations;
	Common::SpscQueue<GpuOperation> m_resolved;
	static constexpr size_t         TableTicketRing = 1024;
	std::unique_ptr<Scout::Ticket[]> m_table_tickets = std::make_unique<Scout::Ticket[]>(TableTicketRing);
	uint64_t                        m_table_ticket_next = 0;
	Common::ProgressCounter         m_executed;
	uint64_t                        m_emitted = 0;
	static constexpr size_t PendingWriteCapacity = 16384;
	std::unique_ptr<GuestWrite[]> m_pending_writes;
	size_t                        m_pending_first = 0;
	size_t                        m_pending_count = 0;
	void                          ForgetExecutedWrites();
	void                                      RunUrgentCommands();
	std::mutex                                m_urgent_mutex;
	std::vector<Common::UniqueFunction<void>> m_urgent_commands;
	std::atomic_uint32_t                      m_urgent_pending {0};
	uint32_t                                  m_urgent_drains = 0;
	std::atomic_uint64_t m_work_epoch {0};
	HW::Context                     m_neutral_context;
	HW::UserConfig                  m_neutral_user_config;
	HW::Shader                      m_neutral_shaders;
	GpuRegisterState                m_neutral_state;
	std::jthread                    m_executor;
	std::jthread                    m_resolver;

	std::jthread m_thread;

	friend class CommandProcessor;
};
}

#endif
