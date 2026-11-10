#include "common/threads.h"
#include "common/profiler.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/graphicContext.h"

#include <type_traits>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <cstdio>
#include <optional>

#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"
#include "graphics/host_gpu/renderer/virtualGpuClock.h"
#include "graphics/host_gpu/renderer/waitSites.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/recordPackets.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/debugSample.h"

#include "graphics/host_gpu/queueLockSite.h"
namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

// One vkQueueSubmit of a render command buffer that signals `tick` (with the waits and signals of `submit`), under
// the queue lock. Called by the recording thread, or by the recorder's worker.
vk::Result QueueSubmitLocked(GraphicContext& graphics, vk::CommandBuffer buffer, SubmitInfo& submit) {
	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = 1;
	submit_info.pCommandBuffers      = &buffer;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	QueueLockGuard lock(graphics.queue_mutex, "commandScheduler.cpp:152");
	return graphics.queue.submit(1, &submit_info, nullptr);
}

// What the recorder's worker needs to submit an ended render buffer.
struct AsyncSubmitPacket {
	GraphicContext*   graphics = nullptr;
	vk::CommandBuffer buffer;
	SubmitInfo        submit;
	uint64_t          tick         = 0;
	uint32_t          debug_op     = 0;
	uint64_t          debug_submit = 0;
	uint32_t          debug_arg[4] {};
	uint64_t          debug_arg4 = 0;
};
static_assert(std::is_trivially_copyable_v<AsyncSubmitPacket>);

void AsyncSubmitOnWorker(const void* blob) {
	auto packet = *static_cast<const AsyncSubmitPacket*>(blob);
	if (GpuTimer::Enabled()) GpuTimer::NoteQueueSubmit(packet.tick); // KYTY_TRACE_GPU_TIMELINE
	const auto result = QueueSubmitLocked(*packet.graphics, packet.buffer, packet.submit);
	if (result != vk::Result::eSuccess) {
		GpuCheckpoints::Dump(packet.graphics->queue);
		ReportVulkanFatal("vkQueueSubmit (record thread)", result, packet.tick, packet.debug_op, packet.debug_submit,
		                  packet.debug_arg[0], packet.debug_arg[1], packet.debug_arg[2], packet.debug_arg[3],
		                  packet.debug_arg4);
		EXIT("vkQueueSubmit failed on the record thread\n");
	}
}

// Submission batching: fewer submissions while the GPU still runs work it was handed. A submission
// gives the GPU work; while the GPU is busy with earlier work, it gives it nothing it lacks and costs the
// recording thread a vkQueueSubmit (~28 us at the 4K test scene) plus the end of the rendering instance and
// the re-binding of all state in the next command buffer. Three rules, all from the GPU's state (no timing
// constant decides whether to batch):
//  - SubmitEvery boundary: submitted when the GPU is idle there; while it is busy the batch
//    goes on and is submitted at the first draw/dispatch that finds the GPU idle, at the latest after
//    BatchMaxBatches times the boundary.
//  - RELEASE_MEM interrupt: submitted at once when the GPU is idle. While it is busy the interrupt could
//    not fire before the GPU finished the earlier work anyway, so the submission waits for the first packet
//    that finds the GPU idle; what is recorded meanwhile delays the interrupt, so it is also forced after
//    BatchInterruptPackets packets or BatchInterruptUs microseconds. The end of every command-processor
//    slice still submits.
//  - Idle early: when the GPU had run dry before this command buffer's first draw/dispatch after a slice
//    end, wait, flip or other non-batch submission (or a wait for the newest submitted tick drained it while
//    this buffer was open), the buffer is
//    submitted once it holds BatchIdleCommands (4x inside a rendering
//    instance) draws/dispatches and the GPU is still idle: at most one extra submission per such
//    event, never in the steady draw stream.
// Only submission points move; what each command buffer records and the order of all GPU work are unchanged.
constexpr uint32_t SubmitEvery           = 128; // the open buffer is submitted every so many draws/dispatches
constexpr uint32_t BatchMaxBatches       = 8;
constexpr uint32_t BatchInterruptPackets = 256;
constexpr uint32_t BatchInterruptUs      = 1000;
constexpr uint32_t BatchIdleCommands     = 8;

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	{ PerfScope perf_wait(PerfKind::FenceWait); WaitSites::Scope wait_site; m_master.Wait(CurrentTick() - 1); }
	// The last buffer was ended and submitted above; the recording thread has
	// nothing left and is stopped.
	if (CommandRecorder::Enabled() && &m_context.GetCommandScheduler() == this) {
		CommandRecorder::Shutdown();
		RecordPackets::Shutdown(); // the record packets: the recording thread's descriptor heap (it has stopped)
	}
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
	// Upstream 4644c5292 (ported): the BDA page table and the fault buffer are zeroed in the render scheduler's first
	// command buffer, before any buffer registration writes into the table (device memory starts undefined).
	if (&m_context.GetCommandScheduler() == this) m_context.GetBufferCache().ClearDeviceTablesOnce();
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	const auto tick = Submit();
	{ PerfScope perf_wait(PerfKind::FenceWait); WaitSites::Scope wait_site; m_master.Wait(tick); }
	BeginNext();
}

void CommandScheduler::Finish() {
	SyncTimeline::SiteScope timeline_site(SyncTimeline::Site::Finish); // KYTY_TRACE_SYNC_TIMELINE
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	{ PerfScope perf_wait(PerfKind::FenceWait); WaitSites::Scope wait_site; m_master.Wait(CurrentTick() - 1); }
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	KYTY_PROFILER_BLOCK("Scheduler::Wait");
	const auto caller = reinterpret_cast<uint64_t>(__builtin_return_address(0));
	EXIT_IF(tick > CurrentTick());
	// KYTY_TRACE_SYNC_TIMELINE: the caller of a wait outside the tagged sites, and whether it submitted the open
	// command buffer.
	const bool timeline = SyncTimeline::Enabled();
	if (timeline) SyncTimeline::NoteCaller(caller);
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		if (timeline) SyncTimeline::NoteOpenFlush();
		{ PerfScope perf_wait(PerfKind::FenceWait); WaitSites::Scope wait_site; m_master.Wait(tick); }
		BeginNext();
	} else {
		// Waiting for the newest submitted work, not known finished, drains the GPU while the open buffer holds more; that
		// buffer may then be submitted early (NoteGpuCommandBatched).
		if (PerfIsGpuThread() && tick + 1u == CurrentTick() && !m_master.IsFree(tick)) {
			m_batch_idle_armed = true;
		}
		{ PerfScope perf_wait(PerfKind::FenceWait); WaitSites::Scope wait_site; m_master.Wait(tick); }
	}
	if (timeline) SyncTimeline::NoteCaller(0);
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	if (SyncTimeline::Enabled()) SyncTimeline::MarkPriorityThread();
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		{ PerfScope perf_wait(PerfKind::FenceWait); WaitSites::Scope wait_site; m_master.Wait(operation.tick); }
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	const auto       done = [this, tick] {
        const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
        const bool queued_before_or_at =
            !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
        return !active_before_or_at && !queued_before_or_at;
	};
	if (done()) return;
	// Blocked on a deferred priority operation (another thread's publish / callback): visible in Tracy.
	KYTY_PROFILER_BLOCK("Scheduler::WaitPriorityOps");
	SyncTimeline::WaitScope timeline(SyncTimeline::Site::PriorityOps, 0, 0, 0); // KYTY_TRACE_SYNC_TIMELINE
	m_operation_available.wait(lock, done);
}

bool CommandScheduler::PriorityOperationsDone(uint64_t tick) {
	std::lock_guard lock(m_operation_mutex);
	const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
	const bool queued_before_or_at = !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
	return !active_before_or_at && !queued_before_or_at;
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

bool CommandScheduler::GpuIdle() {
	return IsFree(CurrentTick() - 1);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	// Upstream 4644c5292 checks only for an open command buffer here, so the buffer cache's constructor can record
	// its table clears before Begin. Ours records them at the first Begin (BufferCache::ClearDeviceTablesOnce), so
	// Current() keeps requiring an active scheduler.
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	// Routed before it is begun, so the begin is a packet too (the worker owns the
	// pool while it records the previous buffers).
	const bool routed_async = CommandRecorder::AsyncSubmit() && &m_context.GetCommandScheduler() == this;
	if (routed_async) {
		CommandRecorder::Route(m_command.m_buffer);
	}
	m_command.Begin();
	// From here until vkEndCommandBuffer (CommandBuffer::End in Submit, which first waits for
	// the recording thread), the commands recorded into this buffer are copied into a ring and recorded by that thread.
	// Only the render scheduler's buffers; the presentation scheduler's are recorded and submitted as before.
	if (!routed_async && CommandRecorder::Enabled() && &m_context.GetCommandScheduler() == this) {
		CommandRecorder::Route(m_command.m_buffer);
	}
	if (GpuTimer::Enabled()) GpuTimer::BeginCommandBuffer(*this, m_command.m_buffer);
	if (VirtualGpuClock::Enabled()) VirtualGpuClock::BeginCommandBuffer(*this, m_command.m_buffer); // KYTY_FIX_VIRTUAL_GPU_CLOCK
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	KYTY_PROFILER_BLOCK("Scheduler::Submit");
	EXIT_IF(m_command.IsInvalid());
	if (m_pre_submit_hook && !m_in_pre_submit_hook) { // D4b step 1 (append windows)
		m_in_pre_submit_hook = true;
		m_pre_submit_hook();
		m_in_pre_submit_hook = false;
	}
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	if (GpuTimer::Enabled()) GpuTimer::EndCommandBuffer(m_command.m_buffer);
	// The host reads finished GPU writes straight from mapped buffers.
	// KYTY_PERF_DIRECT_RANGES: and from guest memory (direct buffers) once the submission finished.
	m_command.EndRendering();
	vk::MemoryBarrier host {};
	host.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	m_command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eHost, {}, 1,
	                                   &host, 0, nullptr, 0, nullptr);
	if (VirtualGpuClock::Enabled()) VirtualGpuClock::EndCommandBuffer(m_command.m_buffer); // KYTY_FIX_VIRTUAL_GPU_CLOCK
	m_command.End();
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result = vk::Result::eSuccess;
	uint64_t   tick   = 0;
	if (CommandRecorder::AsyncSubmit() && &m_context.GetCommandScheduler() == this) {
		// The buffer was ended by a packet (CommandBuffer::End above); the worker
		// submits it after that, in tick order (only this thread takes ticks of this scheduler).
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);
		AsyncSubmitPacket packet;
		packet.graphics     = &graphics;
		packet.buffer       = buffer;
		packet.submit       = submit;
		packet.tick         = tick;
		packet.debug_op     = m_command.m_debug_op;
		packet.debug_submit = m_command.m_debug_submit_id;
		packet.debug_arg[0] = m_command.m_debug_arg0;
		packet.debug_arg[1] = m_command.m_debug_arg1;
		packet.debug_arg[2] = m_command.m_debug_arg2;
		packet.debug_arg[3] = m_command.m_debug_arg3;
		packet.debug_arg4   = m_command.m_debug_arg4;
		CommandRecorder::EnqueueCall(&AsyncSubmitOnWorker, &packet, sizeof(packet));
	} else {
		QueueLockGuard lock(graphics.queue_mutex, "commandScheduler.cpp:781");
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);
		result = QueueSubmitLocked(graphics, buffer, submit);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		GpuCheckpoints::Dump(graphics.queue);
		ReportVulkanFatal("vkQueueSubmit", result, tick, m_command.m_debug_op,
		                  m_command.m_debug_submit_id, m_command.m_debug_arg0,
		                  m_command.m_debug_arg1, m_command.m_debug_arg2, m_command.m_debug_arg3,
		                  m_command.m_debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (GpuTimer::Enabled()) GpuTimer::Submitted(buffer, tick);
	if (VirtualGpuClock::Enabled()) VirtualGpuClock::Submitted(buffer, tick); // KYTY_FIX_VIRTUAL_GPU_CLOCK
	if (SyncTimeline::Enabled()) SyncTimeline::NoteSubmit(tick);

	m_command.m_buffer      = nullptr;
	m_commands_since_submit = 0;
	// What was recorded belongs to the submitted command buffer (KYTY_FIX_DRAW_STATE_FILTER).
	m_command.InvalidateGraphicsState();
	// KYTY_TRACE_PERF: the submission by reason, and how long a deferred interrupt
	// waited for it.
	if (PerfEnabled()) {
		m_perf_submits[static_cast<uint32_t>(m_submit_reason)].fetch_add(1, std::memory_order_relaxed);
		if (m_batch_interrupt_pending) {
			const auto us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                          std::chrono::steady_clock::now() - m_batch_interrupt_since)
			                                          .count());
			m_perf_interrupt_delay_us.fetch_add(us, std::memory_order_relaxed);
			m_perf_interrupt_delays.fetch_add(1, std::memory_order_relaxed);
			auto max = m_perf_interrupt_delay_max_us.load(std::memory_order_relaxed);
			while (us > max && !m_perf_interrupt_delay_max_us.compare_exchange_weak(max, us, std::memory_order_relaxed)) {
			}
		}
	}
	// Everything recorded, deferred interrupts included, is submitted now.
	m_last_submit_reason      = m_submit_reason;
	m_submit_reason           = SubmitReason::Other;
	m_batch_interrupt_pending = false;
	m_batch_idle_armed        = false;
	return tick;
}

void CommandScheduler::FlushFor(SubmitReason reason) {
	m_submit_reason = reason;
	Flush();
}

void CommandScheduler::NoteGpuCommand() {
	if (!Active() || m_command.IsInvalid()) return;
	NoteGpuCommandBatched();
}

// Called before each draw/dispatch is recorded; m_commands_since_submit counts it, so `count - 1` draws/dispatches are
// already in the open buffer.
void CommandScheduler::NoteGpuCommandBatched() {
	const auto count = ++m_commands_since_submit;
	if (count == 1 && !m_batch_idle_armed && m_last_submit_reason == SubmitReason::Other) {
		// The buffer's first draw/dispatch after a slice end, wait or flip: did the GPU run dry before it?
		// After a batch submission (periodic, interrupt, idle early) the GPU was just handed work, and an
		// idle GPU there only means the recording is the bottleneck, which more submissions would slow.
		m_batch_idle_armed = GpuIdle();
	}
	if (m_batch_idle_armed && count > (m_command.IsRendering() ? BatchIdleCommands * 4u : BatchIdleCommands)) {
		if (GpuIdle()) {
			FlushFor(SubmitReason::IdleEarly);
			return;
		}
	}
	if (count <= SubmitEvery) return;
	const bool boundary = count == SubmitEvery + 1u;
	if (GpuIdle()) {
		FlushFor(boundary ? SubmitReason::Periodic : SubmitReason::PeriodicLate);
		return;
	}
	if (boundary && PerfEnabled()) {
		m_perf_deferred_periodic.fetch_add(1, std::memory_order_relaxed);
	}
	if (uint64_t {count} > uint64_t {SubmitEvery} * BatchMaxBatches) {
		FlushFor(SubmitReason::PeriodicBound);
	}
}

void CommandScheduler::FlushForInterrupt() {
	if (GpuIdle()) {
		FlushFor(SubmitReason::Interrupt);
		return;
	}
	if (PerfEnabled()) {
		m_perf_deferred_interrupts.fetch_add(1, std::memory_order_relaxed);
	}
	if (!m_batch_interrupt_pending) {
		m_batch_interrupt_pending = true;
		m_batch_interrupt_packets = 0;
		m_batch_interrupt_since   = std::chrono::steady_clock::now();
	}
}

void CommandScheduler::NoteDeferredPacket() {
	if (GpuIdle()) {
		FlushFor(SubmitReason::InterruptLate);
		return;
	}
	if (++m_batch_interrupt_packets >= BatchInterruptPackets ||
	    std::chrono::steady_clock::now() - m_batch_interrupt_since >= std::chrono::microseconds(BatchInterruptUs)) {
		FlushFor(SubmitReason::InterruptBound);
	}
}

void CommandScheduler::ReportSubmitPerf() {
	if (!PerfEnabled()) return;
	const auto take = [](std::atomic<uint64_t>& value) {
		return static_cast<unsigned long long>(value.exchange(0, std::memory_order_relaxed));
	};
	unsigned long long submits[static_cast<uint32_t>(SubmitReason::Count)];
	unsigned long long total = 0;
	for (uint32_t i = 0; i < static_cast<uint32_t>(SubmitReason::Count); i++) {
		submits[i] = take(m_perf_submits[i]);
		total += submits[i];
	}
	const auto delays   = take(m_perf_interrupt_delays);
	const auto delay_us = take(m_perf_interrupt_delay_us);
	// Submissions by reason (periodic and interrupt: at once / deferred, then GPU idle / deferred, then bound),
	// boundaries and interrupts deferred, and the wait of deferred interrupts for their submission (avg/max us).
	std::fprintf(stderr,
	             "PERF submit total=%llu other=%llu periodic=%llu/%llu/%llu interrupt=%llu/%llu/%llu "
	             "idle_early=%llu deferred_periodic=%llu deferred_interrupts=%llu interrupt_delay_us=%llu/%llu\n",
	             total, submits[static_cast<uint32_t>(SubmitReason::Other)],
	             submits[static_cast<uint32_t>(SubmitReason::Periodic)],
	             submits[static_cast<uint32_t>(SubmitReason::PeriodicLate)],
	             submits[static_cast<uint32_t>(SubmitReason::PeriodicBound)],
	             submits[static_cast<uint32_t>(SubmitReason::Interrupt)],
	             submits[static_cast<uint32_t>(SubmitReason::InterruptLate)],
	             submits[static_cast<uint32_t>(SubmitReason::InterruptBound)],
	             submits[static_cast<uint32_t>(SubmitReason::IdleEarly)], take(m_perf_deferred_periodic),
	             take(m_perf_deferred_interrupts), delays != 0 ? delay_us / delays : 0ull,
	             take(m_perf_interrupt_delay_max_us));
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
