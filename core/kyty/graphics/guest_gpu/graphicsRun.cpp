#include "graphics/guest_gpu/graphicsRun.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#if defined(ANYPS5_KYTY)
#include "common/fallback.h"
#include "common/runtimeSwitches.h"
#endif
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/labelOrder.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/scout.h"
#include "graphics/guest_gpu/psrShaders.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/rayTracing.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <fmt/ranges.h>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <optional>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string_view>
#include <thread>
#include <utility>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandProcessor* g_current_processor = nullptr;
static thread_local Pm4Execution*     g_current_execution = nullptr;
static thread_local bool              g_gpu_thread        = false;
static thread_local GuestGpu*         g_executing_gpu     = nullptr;
static thread_local bool              g_cp_thread         = false;
static thread_local GuestGpu*         g_gpu_state         = nullptr;

#if defined(ANYPS5_KYTY)
namespace {
bool InPlaceLabelHeldBack(CommandScheduler& scheduler, uint64_t address, uint64_t size);
void SettleInPlaceLabels(CommandScheduler& scheduler, uint64_t address, uint64_t size);
}
#endif

constexpr size_t OperationQueueSize = 4096;
constexpr size_t ResolvedQueueSize = 256;
constexpr auto BlockedRetryInterval = std::chrono::microseconds(20);

static bool TableContinueEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TABLE_XPR");
		return value == nullptr || value[0] == 0 || value[0] == '2';
	}();
	return enabled;
}

static bool IsDrawOperation(GpuOperationKind kind) {
	return kind == GpuOperationKind::DrawIndex || kind == GpuOperationKind::DrawAuto ||
	       kind == GpuOperationKind::DrawIndirect;
}

static void CopyFromGuest(void* destination, const volatile void* source, size_t size) {
	auto*      out     = static_cast<uint8_t*>(destination);
	const auto address = reinterpret_cast<uint64_t>(source);
	if ((address & 7u) == 0 && (size & 7u) == 0) {
		const auto* words = static_cast<const volatile uint64_t*>(source);
		for (size_t i = 0; i < size / 8u; i++) {
			const uint64_t word = words[i];
			std::memcpy(out + i * 8u, &word, 8u);
		}
		return;
	}
	if ((address & 3u) == 0 && (size & 3u) == 0) {
		const auto* words = static_cast<const volatile uint32_t*>(source);
		for (size_t i = 0; i < size / 4u; i++) {
			const uint32_t word = words[i];
			std::memcpy(out + i * 4u, &word, 4u);
		}
		return;
	}
	const auto* bytes = static_cast<const volatile uint8_t*>(source);
	for (size_t i = 0; i < size; i++) {
		out[i] = bytes[i];
	}
}

GuestGpu::GuestGpu(RenderContext& renderer)
    : m_renderer(renderer),
      m_pipelined((Config::GpuPipelineStages() & Config::GPU_PIPELINE_COMMAND_PROCESSOR) != 0),
      m_resolving(m_pipelined && (Config::GpuPipelineStages() & Config::GPU_PIPELINE_RESOLVE) != 0),
      m_operations(OperationQueueSize), m_resolved(ResolvedQueueSize),
      m_neutral_state {&m_neutral_context, &m_neutral_user_config, &m_neutral_shaders} {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_gfx_cp->Reset();
	if (m_pipelined) {
		m_pending_writes = std::make_unique<GuestWrite[]>(PendingWriteCapacity);
		m_gfx_cp->AttachPipeline(this);
		m_executor = std::jthread([this] { ExecutionThread(); });
		if (m_resolving) {
			m_resolver = std::jthread([this] { ResolveThread(); });
		}
	}
	Scout::Start(renderer);
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_epoch.fetch_add(1, std::memory_order_release);
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	Scout::Stop();
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	if (g_cp_thread) {
		m_gfx_cp->EmitHostCommand(std::move(command));
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	EXIT_IF(!IsGpuThread() && !g_cp_thread);
	if (m_pending_commands.load(std::memory_order_acquire) != 0) {
		m_renderer.CollectHotPageWrites();
	}
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		if (m_pipelined) {
			m_gfx_cp->EmitHostCommand(std::move(command));
		} else {
			command();
		}
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	if (g_cp_thread) {
		ExecuteSync(std::move(command));
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::SendUrgentCommandSync(Common::UniqueFunction<void>&& command, bool drain_copies) {
	EXIT_IF(!command);
	if (!m_pipelined || IsGpuThread() || g_cp_thread) {
		SendCommandSync(std::move(command));
		return;
	}
	PostUrgentCommandSync(std::move(command), drain_copies);
}

void GuestGpu::SendReadbackCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (m_pipelined && g_cp_thread) {
		PostUrgentCommandSync(std::move(command), false);
		return;
	}
	SendUrgentCommandSync(std::move(command), false);
}

bool GuestGpu::PostReadbackCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (!m_pipelined || IsGpuThread()) {
		return false;
	}
	{
		std::lock_guard lock(m_urgent_mutex);
		m_urgent_commands.push_back(std::move(command));
		m_urgent_pending.fetch_add(1, std::memory_order_release);
	}
	(m_resolving ? m_resolved : m_operations).Interrupt();
	return true;
}

uint64_t GuestGpu::WaitForOrderedOperations(uint64_t begin, uint64_t end) {
	EXIT_IF(!g_cp_thread);
	ForgetExecutedWrites();
	uint64_t needed = m_cp_ordered;
	for (size_t i = 0; i < m_pending_count; i++) {
		const auto& write = m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write.address < end && begin < write.address + write.size) {
			needed = std::max(needed, write.sequence);
		}
	}
	needed = std::min(needed, m_emitted);
	if (needed != 0) {
		WaitForExecution(needed);
	}
	return m_emitted - std::max(needed, m_executed.Load());
}

void GuestGpu::PostUrgentCommandSync(Common::UniqueFunction<void>&& command, bool drain_copies) {
	std::binary_semaphore done {0};
	{
		std::lock_guard lock(m_urgent_mutex);
		m_urgent_commands.push_back([operation = std::move(command), &done]() mutable {
			operation();
			done.release();
		});
		m_urgent_drains += drain_copies ? 1 : 0;
		m_urgent_pending.fetch_add(1, std::memory_order_release);
	}
	(m_resolving ? m_resolved : m_operations).Interrupt();
	done.acquire();
}

void GuestGpu::RunUrgentCommands() {
	if (m_urgent_pending.load(std::memory_order_acquire) == 0) {
		return;
	}
	m_renderer.GetRenderExecutor().InvalidateRenderTargetMemo();
	m_renderer.GetRenderExecutor().FlushDeferredDispatchBarrier();
	std::vector<Common::UniqueFunction<void>> commands;
	bool                                      drain = false;
	{
		std::lock_guard lock(m_urgent_mutex);
		commands.swap(m_urgent_commands);
		drain = std::exchange(m_urgent_drains, 0) != 0;
		m_urgent_pending.store(0, std::memory_order_relaxed);
	}
	if (drain) {
		if (auto* copies = m_renderer.GetCommandScheduler().HostCopies(); copies != nullptr) {
			copies->Drain();
		}
	}
	for (auto& command: commands) {
		command();
	}
}

void GuestGpu::Submit(std::span<const uint32_t> commands) {
	if (commands.empty()) {
		return;
	}
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 0;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
#if defined(ANYPS5_KYTY)
	NoteFrameSubmitted();
#endif
}

#if defined(ANYPS5_KYTY)
ptrdiff_t GuestGpu::SuspendPointDepth() {
	static const ptrdiff_t depth = [] {
		constexpr ptrdiff_t DefaultDepth = 8;
		const char*         value        = std::getenv("KYTY_SUSPEND_POINT_DEPTH");
		if (value == nullptr || value[0] == '\0') {
			return DefaultDepth;
		}
		char*      end    = nullptr;
		const auto parsed = std::strtol(value, &end, 10);
		if (end == value || *end != '\0' || parsed < 1 || parsed > SuspendPointDepthMax) {
			EXIT("invalid KYTY_SUSPEND_POINT_DEPTH: %s (1 to %td)\n", value, SuspendPointDepthMax);
		}
		return static_cast<ptrdiff_t>(parsed);
	}();
	return depth;
}
#endif

#if defined(ANYPS5_KYTY)
void GuestGpu::NoteFrameSubmitted() {
	m_frame_submitted.store(true, std::memory_order_release);
}
#endif

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
#if defined(ANYPS5_KYTY)
	submission.frame_end = m_frame_submitted.exchange(false, std::memory_order_acq_rel);
	if (submission.frame_end) {
		m_frame_ready->acquire();
	}
#endif
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
		if (m_pipelined) {
			processor->AttachPipeline(this);
		}
	}
	return *processor;
}

uint64_t GuestGpu::EmitOperation(GpuOperation&& operation) {
	const auto sequence = ++m_emitted;
	operation.sequence  = sequence;
	m_operations.Push(std::move(operation));
	return sequence;
}

void GuestGpu::ForgetExecutedWrites() {
	const auto executed = m_executed.Load();
	while (m_pending_count != 0 && m_pending_writes[m_pending_first].sequence <= executed) {
		m_pending_first = (m_pending_first + 1) % PendingWriteCapacity;
		m_pending_count--;
	}
}

void GuestGpu::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
                              const void* value) {
	ForgetExecutedWrites();
	if (sequence <= m_executed.Load()) {
		return;
	}
	if (m_pending_count == PendingWriteCapacity) {
		WaitForExecution(m_pending_writes[m_pending_first].sequence);
		ForgetExecutedWrites();
	}
	auto& write = m_pending_writes[(m_pending_first + m_pending_count) % PendingWriteCapacity];
	write       = {.sequence = sequence, .address = address, .size = size};
	if (value != nullptr && size <= sizeof(write.value)) {
		write.known = true;
		std::memcpy(write.value, value, size);
	}
	m_pending_count++;
}

void GuestGpu::ReadDecisionMemory(const volatile void* address, void* value, size_t size) {
	ForgetExecutedWrites();
	const auto begin = reinterpret_cast<uint64_t>(address);
	const auto end   = begin + size;
	for (size_t i = m_pending_count; i-- > 0;) {
		const auto* write = &m_pending_writes[(m_pending_first + i) % PendingWriteCapacity];
		if (write->address >= end || write->address + write->size <= begin) {
			continue;
		}
		if (write->known && write->address <= begin && end <= write->address + write->size) {
			std::memcpy(value, write->value + (begin - write->address), size);
			NoteOrderedSequence(write->sequence);
			return;
		}
		WaitForExecution(write->sequence);
		break;
	}
#if defined(ANYPS5_KYTY)
	if (InPlaceLabelHeldBack(m_renderer.GetCommandScheduler(), begin, size)) {
		ExecuteSync([this, begin, size] { SettleInPlaceLabels(m_renderer.GetCommandScheduler(), begin, size); });
	}
#endif
	CopyFromGuest(value, address, size);
}

void GuestGpu::ExecuteSync(Common::UniqueFunction<void>&& command) {
	m_gfx_cp->EmitHostCommand(std::move(command));
	WaitForExecution();
}

void GuestGpu::ExecutionThread() {
	KYTY_PROFILER_THREAD("Thread_GpuExecute");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Execution);
	g_gpu_thread    = true;
	g_executing_gpu = this;
#if defined(ANYPS5_KYTY)
	PipelineCache::BeginExecuteThread();
#endif
	BdaDirtyRegions::g_translating = true;
	constexpr uint32_t NotifyInterval = 16;
	uint32_t           unnotified     = 0;
	auto&              operations     = m_resolving ? m_resolved : m_operations;
	for (;;) {
		if (m_urgent_pending.load(std::memory_order_relaxed) != 0) {
			RunUrgentCommands();
		}
		auto operation = operations.TryPop();
		if (!operation) {
			m_executed.Notify();
			unnotified = 0;
			operation  = operations.Pop();
			if (!operation) {
				if (operations.IsStopped()) {
					break;
				}
				continue;
			}
		}
		if (!m_resolving && operation->kind != GpuOperationKind::Callback) {
			const auto* next = operations.TryPeek();
			operation->stream_in_place = next != nullptr && !IsDrawOperation(next->kind);
		}
		operation->processor->Execute(*operation);
		const auto sequence = operation->sequence;
		operation.reset();
		m_executed.Publish(sequence);
		if ((sequence & 1023u) == 0) {
			BufferCache::LogReadbackSummary();
			m_renderer.GetRenderExecutor().LogResolveSummary();
#if defined(ANYPS5_KYTY)
			CommandProcessor::LogLabelSummary();
#endif
		}
		if (++unnotified >= NotifyInterval) {
			m_executed.Notify();
			unnotified = 0;
		}
	}
	m_executed.Notify();
	g_gpu_thread    = false;
	g_executing_gpu = nullptr;
}

static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets);

void GuestGpu::ResolveThread() {
	KYTY_PROFILER_THREAD("Thread_GpuResolve");
	Config::ConfigureGpuStageThread(Config::GpuStageThread::Resolve);
	auto& executor = m_renderer.GetRenderExecutor();
#if defined(ANYPS5_KYTY)
	if (PipelineCache::NarrowProgramLock()) {
		PipelineCache::BeginAheadThread(0);
	}
#endif
	for (;;) {
		auto operation = m_operations.TryPop();
		if (!operation) {
			operation = m_operations.Pop();
			if (!operation) {
				break;
			}
		}
		if (operation->kind == GpuOperationKind::Callback) {
			const auto& write = operation->guest_write;
			if (write.dwords != 0 && write.data != nullptr) {
				PipelineCache::NoteAheadGuestWrite(operation->sequence, write.address,
				                                   std::span(write.data, write.dwords),
				                                   write.one_address);
			}
		} else {
			PipelineCache::RetireAheadGuestWrites(m_executed.Load());
		}
		uint32_t table_hint = 0;
		if (operation->kind == GpuOperationKind::DrawIndex || operation->kind == GpuOperationKind::DrawIndirect) {
			table_hint = executor.TableHint(*operation->state.shaders);
			if ((table_hint & 1u) != 0 && Scout::Running()) {
				const auto& shaders = *operation->state.shaders;
				const auto& vs      = shaders.GetVs();
				const auto& ps      = shaders.GetPs();
				int32_t     start   = -1;
				if (operation->kind == GpuOperationKind::DrawIndirect) {
					ShaderIndirectOffsets offsets;
					if (IndirectDrawOffsets(operation->draw_indirect.registers, offsets) &&
					    (offsets.vertex != ShaderIndirectOffsets::None || offsets.instance != ShaderIndirectOffsets::None))
						start = static_cast<int32_t>(uint32_t {offsets.instance} << 8u | offsets.vertex);
				}
				const std::array<uint64_t, 3> key {vs.es_regs.data_addr, ps.ps_regs.data_addr, static_cast<uint32_t>(start)};
				auto& ticket = m_table_tickets[m_table_ticket_next & (TableTicketRing - 1)];
				if (Scout::EvaluateAhead(
				        key, XXH3_64bits(key.data(), sizeof(key)),
				        {vs.gs_user_sgpr.value, std::min<uint32_t>(vs.gs_regs.rsrc2.user_sgpr, HW::UserSgprInfo::SGPRS_MAX)},
				        {ps.ps_user_sgpr.value, std::min<uint32_t>(ps.ps_regs.rsrc2.user_sgpr, HW::UserSgprInfo::SGPRS_MAX)},
				        ticket)) {
					operation->table_ticket = &ticket;
					m_table_ticket_next++;
				}
			}
		}
		const bool resolve_dispatch = operation->kind == GpuOperationKind::DispatchDirect ||
		                              operation->kind == GpuOperationKind::DispatchIndirect;
		const auto resolve_begin    = operation->kind != GpuOperationKind::Callback ? Common::SteadyNs() : 0;
		if ((operation->kind == GpuOperationKind::DrawIndex ||
		     operation->kind == GpuOperationKind::DrawIndirect) &&
		    table_hint == 1u) {
			operation->ahead_note = RenderExecutor::AheadNote::TableSkip;
		} else if (operation->kind == GpuOperationKind::DrawIndex ||
		           operation->kind == GpuOperationKind::DrawAuto) {
			operation->ahead_note = RenderExecutor::AheadNote::NotPrepared;
			operation->resolved =
			    executor.ResolveDrawAhead(*operation->state.context, *operation->state.user_config,
			                              *operation->state.shaders, {}, false, operation->state.context_id,
			                              operation->state.user_config_id);
		} else if (operation->kind == GpuOperationKind::DrawIndirect) {
			ShaderIndirectOffsets offsets;
			if (IndirectDrawOffsets(operation->draw_indirect.registers, offsets)) {
				operation->ahead_note = RenderExecutor::AheadNote::NotPrepared;
				operation->resolved = executor.ResolveDrawAhead(
				    *operation->state.context, *operation->state.user_config,
				    *operation->state.shaders, offsets, true, operation->state.context_id,
				    operation->state.user_config_id);
			}
		} else if (operation->kind == GpuOperationKind::DispatchDirect ||
		           operation->kind == GpuOperationKind::DispatchIndirect) {
			operation->ahead_note = RenderExecutor::AheadNote::NotPrepared;
			if (operation->kind == GpuOperationKind::DispatchDirect) {
				const uint32_t groups[] = {operation->dispatch.thread_group_x,
				                           operation->dispatch.thread_group_y,
				                           operation->dispatch.thread_group_z};
				operation->resolved = executor.ResolveDispatchAhead(
				    *operation->state.context, *operation->state.shaders, groups,
				    operation->dispatch.mode, operation->processor->IsAsyncComputeQueue());
			} else {
				operation->resolved = executor.ResolveIndirectDispatchAhead(
				    *operation->state.context, *operation->state.shaders, operation->dispatch.mode,
				    operation->dispatch.args_addr,
				    operation->processor->IsAsyncComputeQueue());
			}
		}
		if (operation->kind != GpuOperationKind::Callback) {
			if (operation->resolved != nullptr) {
				operation->ahead_note = RenderExecutor::AheadNote::Prepared;
			}
			RenderExecutor::NoteAhead(resolve_dispatch, operation->ahead_note, Common::SteadyNs() - resolve_begin);
			const auto* next = m_operations.TryPeek();
			operation->stream_in_place = next != nullptr && !IsDrawOperation(next->kind);
		}
		m_resolved.Push(std::move(*operation));
	}
	m_resolved.Stop();
}

void GuestGpu::StopExecution() {
	WaitForExecution();
	m_operations.Stop();
	if (m_resolver.joinable()) {
		m_resolver.join();
	}
	if (m_executor.joinable()) {
		m_executor.join();
	}
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_ctx_dirty    = true;
	m_ucfg_dirty   = true;
	m_sh_ctx_dirty = true;
	m_saved_ctx.reset();
	m_index_type_and_size              = 0;
	m_index_buffer_size                = 0;
	m_index_base_addr                  = 0;
	m_num_instances.value              = 1;
	m_num_instances.sequence++;
	m_predicate_skip                   = false;
	m_draw_indirect_args_base_addr     = 0;
	m_dispatch_indirect_args_base_addr = 0;
	if (TableContinueEnabled()) {
		if (m_draw_observer == nullptr) m_draw_observer = std::make_unique<DrawStateObserver>();
		m_draw_observer->Invalidate();
	}
}

void CommandProcessor::ResetQueue(uint32_t op) {
	if ((op & Pm4::ResetQueueOp::PopContext) != 0 && m_saved_ctx.has_value()) {
		ApplyContextStateOperation(ContextStateOperation::Pop);
	}
	if ((op & Pm4::ResetQueueOp::CxState) != 0) {
		m_ctx.Reset();
	}
	if ((op & Pm4::ResetQueueOp::ShState) != 0) {
		m_sh_ctx.Reset();
	}
	if ((op & Pm4::ResetQueueOp::UcState) != 0) {
		m_ucfg.Reset();
	}
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	m_ctx_dirty = true;
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_saved_ctx.has_value());
			m_ctx = *m_saved_ctx;
			m_saved_ctx.reset();
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			m_ctx.Reset();
			break;
	}
}

GpuOperation CommandProcessor::MakeOperation(GpuOperationKind kind) {
	GpuOperation operation;
	operation.kind          = kind;
	operation.processor     = this;
	operation.submit_id     = m_submit_id;
	operation.num_instances = m_num_instances;
	operation.after_wait    = std::exchange(m_wait_since_operation, false);
	operation.guest_sync    = std::exchange(m_acquire_since_operation, false) || operation.after_wait;
	operation.clean         = m_draw_observer != nullptr &&
	                          (kind == GpuOperationKind::DrawIndex || kind == GpuOperationKind::DrawIndirect) &&
	                          m_draw_observer->LastClean();
	if (m_scout_packet != nullptr && (kind == GpuOperationKind::DrawIndex || kind == GpuOperationKind::DrawAuto ||
	                                  kind == GpuOperationKind::DrawIndirect)) {
		operation.scout_packet   = m_scout_packet;
		operation.scout_sequence = m_scout_sequence;
		operation.scout_ordinal  = m_scout_ordinal;
	}
	if (m_pipeline == nullptr) {
		operation.state = {&m_ctx, &m_ucfg, &m_sh_ctx};
	} else if (kind == GpuOperationKind::Callback) {
		operation.state = m_pipeline->NeutralState();
	} else {
		operation.state = SnapshotState(m_pipeline->NextSequence());
	}
	return operation;
}

template <typename T>
const T* CommandProcessor::Snapshot(SnapshotRing<T>& ring, const T& live, bool& dirty,
                                    uint64_t sequence, uint64_t* id) {
	static_assert(std::is_trivially_copyable_v<T>);
	if (dirty || ring.current == SnapshotRing<T>::Size) {
		if (ring.slots == nullptr) {
			ring.slots = std::make_unique<typename SnapshotRing<T>::Slot[]>(SnapshotRing<T>::Size);
			ring.last_use = std::make_unique<uint64_t[]>(SnapshotRing<T>::Size);
		}
		auto& slot = ring.slots[ring.next];
		m_pipeline->WaitForExecution(ring.last_use[ring.next]);
		slot.value   = live;
		slot.id      = s_snapshot_ids.fetch_add(1, std::memory_order_relaxed) + 1;
		ring.current = ring.next;
		ring.next    = (ring.next + 1) % SnapshotRing<T>::Size;
		dirty        = false;
	}
	ring.last_use[ring.current] = sequence;
	if (id != nullptr) {
		*id = ring.slots[ring.current].id;
	}
	return &ring.slots[ring.current].value;
}

GpuRegisterState CommandProcessor::SnapshotState(uint64_t sequence) {
	GpuRegisterState state;
	state.context     = Snapshot(m_ctx_snapshots, m_ctx, m_ctx_dirty, sequence, &state.context_id);
	state.user_config = Snapshot(m_ucfg_snapshots, m_ucfg, m_ucfg_dirty, sequence, &state.user_config_id);
	state.shaders     = Snapshot(m_sh_ctx_snapshots, m_sh_ctx, m_sh_ctx_dirty, sequence);
	return state;
}

uint64_t CommandProcessor::Emit(GpuOperation&& operation) {
	if (m_pipeline != nullptr) {
		EXIT_IF(operation.kind != GpuOperationKind::Callback &&
		        m_pipeline->NextSequence() !=
		            m_ctx_snapshots.last_use[m_ctx_snapshots.current]);
		return m_pipeline->EmitOperation(std::move(operation));
	}
	Execute(operation);
	return 0;
}

void CommandProcessor::NoteGuestWrite(uint64_t sequence, uint64_t address, uint64_t size,
                                      const void* value) {
	YieldIfWaitedOn(address, size);
	if (m_interrupt_event_id == 0 && size != 0 && Scout::Running()) Scout::NoteWrite(reinterpret_cast<const void*>(address), static_cast<uint32_t>(std::min<uint64_t>(size, UINT32_MAX)));
	if (m_pipeline != nullptr && size != 0) {
		m_pipeline->NoteGuestWrite(sequence, address, size, value);
	}
}

void CommandProcessor::EmitHostCommand(Common::UniqueFunction<void>&& command) {
	EmitCallback(std::move(command));
}

void CommandProcessor::ReadDecisionMemory(const volatile void* address, void* value,
                                          size_t size) {
	if (m_pipeline != nullptr) {
		m_pipeline->ReadDecisionMemory(address, value, size);
		return;
	}
#if defined(ANYPS5_KYTY)
	SettleInPlaceLabels(GetScheduler(), reinterpret_cast<uint64_t>(address), size);
#endif
	CopyFromGuest(value, address, size);
}

uint64_t CommandProcessor::EmitCallback(GpuCallback&& callback, const GpuGuestWrite& write) {
	auto operation        = MakeOperation(GpuOperationKind::Callback);
	operation.callback    = std::move(callback);
	operation.guest_write = write;
	return Emit(std::move(operation));
}

void CommandProcessor::Execute(GpuOperation& operation) {
	EXIT_IF(operation.processor != this);
	if (operation.scout_packet != nullptr && Scout::Running())
		Scout::NoteDrawAt(operation.scout_packet, operation.scout_sequence, operation.scout_ordinal);
	BindState(operation.state);
	m_renderer.GetRenderExecutor().SetTableTicket(operation.table_ticket);
	m_renderer.NoteOperation(operation.submit_id, operation.after_wait);
	auto& executor = m_renderer.GetRenderExecutor();
	executor.UseContextId(operation.state.context_id, operation.state.user_config_id);
	executor.BeginGuestOperation(static_cast<uint32_t>(m_interrupt_event_id),
	                             operation.kind == GpuOperationKind::DispatchDirect ||
	                                 operation.kind == GpuOperationKind::DispatchIndirect,
	                             operation.kind == GpuOperationKind::DrawIndex ||
	                                 operation.kind == GpuOperationKind::DrawAuto ||
	                                 operation.kind == GpuOperationKind::DrawIndirect,
	                             operation.guest_sync);
	if (operation.after_wait) {
		executor.InvalidateRenderTargetMemo();
	}
	if (operation.kind != GpuOperationKind::DrawIndex &&
	    operation.kind != GpuOperationKind::DrawAuto &&
	    operation.kind != GpuOperationKind::DrawIndirect) {
		executor.InvalidateRenderTargetMemo();
		if (auto* copies = GetScheduler().HostCopies();
		    copies != nullptr && operation.kind == GpuOperationKind::Callback) {
			copies->Drain();
		}
	}
	m_renderer.GetBufferCache().SetStreamInPlace(operation.stream_in_place);
#if defined(ANYPS5_KYTY)
	const bool submit_when_idle = operation.kind != GpuOperationKind::Callback;
#else
	const bool submit_when_idle = false;
#endif
	struct CommitWriteTicks {
		RenderContext& renderer;
		bool           submit_when_idle;
		~CommitWriteTicks() {
			auto& scheduler = renderer.GetCommandScheduler();
			if (renderer.GetBufferCache().CommitWriteTicks()) {
				scheduler.Flush();
			} else if (submit_when_idle) {
				const auto submitted = scheduler.SubmittedTick();
				if (submitted + 1 != scheduler.CurrentTick()) {
					return;
				}
				auto& master = scheduler.GetMasterSemaphore();
				bool  idle   = master.IsFree(submitted);
				if (!idle) {
					static thread_local std::chrono::steady_clock::time_point last_refresh {};
					const auto now = std::chrono::steady_clock::now();
					if (now - last_refresh >= std::chrono::microseconds(25)) {
						last_refresh = now;
						master.Refresh();
						idle = master.IsFree(submitted);
					}
				}
				if (idle) {
					scheduler.Flush();
				}
			}
		}
	} commit_write_ticks {m_renderer, submit_when_idle};
	switch (operation.kind) {
		case GpuOperationKind::DrawIndex:
		case GpuOperationKind::DrawAuto: {
			executor.BeginAheadOperation(operation.resolved, operation.ahead_note);
			if (operation.kind == GpuOperationKind::DrawIndex) {
				ExecuteDrawIndex(operation, operation.draw_index);
			} else {
				ExecuteDrawAuto(operation, operation.draw_auto);
			}
			executor.EndAheadOperation();
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		}
		case GpuOperationKind::DrawIndirect:
			executor.BeginAheadOperation(operation.resolved, operation.ahead_note);
			ExecuteDrawIndirect(operation);
			executor.EndAheadOperation();
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		case GpuOperationKind::DispatchDirect: {
			const auto& dispatch = operation.dispatch;
			executor.BeginAheadOperation(operation.resolved, operation.ahead_note);
			ExecuteDispatchDirect(operation, dispatch.thread_group_x, dispatch.thread_group_y,
			                      dispatch.thread_group_z, dispatch.mode);
			executor.EndAheadOperation();
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		}
		case GpuOperationKind::DispatchIndirect:
			executor.BeginAheadOperation(operation.resolved, operation.ahead_note);
			ExecuteDispatchIndirect(operation);
			executor.EndAheadOperation();
			if (operation.resolved != nullptr) {
				RenderExecutor::ReleaseResolvedDraw(operation.resolved);
			}
			break;
		case GpuOperationKind::Callback:
			if (operation.callback) {
				operation.callback();
			}
			break;
	}
	if (auto* copies = GetScheduler().HostCopies(); copies != nullptr) {
		(void)copies->Flush();
	}
}

void CommandProcessor::BindState(const GpuRegisterState& state) {
	GetScheduler().Begin(*state.context, *state.user_config, *state.shaders);
}

void CommandProcessor::BufferInit() {
	if (m_pipeline == nullptr) {
		Emit(MakeOperation(GpuOperationKind::Callback));
	}
}

void CommandProcessor::BufferFlush() {
	EmitCallback([this] { GetScheduler().Flush(); });
}

void CommandProcessor::NoteRecordedWork() {
	const auto tick = GetScheduler().CurrentTick();
	if (tick != m_batch_tick) {
		m_batch_tick = tick;
		m_batch_work = 0;
	}
	m_batch_work++;
}

void CommandProcessor::BufferFlushIfBatchReady() {
	EmitCallback([this] {
		constexpr uint32_t MinBatchWork = 64;
		if (GetScheduler().CurrentTick() == m_batch_tick && m_batch_work >= MinBatchWork) {
			GetScheduler().Flush();
		}
	});
}

void CommandProcessor::BufferWait() {
	EmitCallback([this] { GetScheduler().Finish(); });
}

void CommandProcessor::RunGarbageCollector() {
	EmitCallback([this] { m_renderer.RunGarbageCollector(); });
}

#if defined(ANYPS5_KYTY)
void CommandProcessor::WaitForRewind(bool valid, const uint32_t* control) {
	if (!valid) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->NoteBlockedRewind(reinterpret_cast<uint64_t>(control));
		}
		SuspendPm4();
	} else {
		m_renderer.CollectHotPageWrites();
	}
}
#else
void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
	} else {
		m_renderer.CollectHotPageWrites();
	}
}
#endif

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	m_wait_since_operation = true;
	if (TestWaitRegMemValue(ReadDecisionValue<T>(addr), ref, mask, func)) {
		m_renderer.CollectHotPageWrites();
	} else {
		if (g_gpu_state != nullptr) {
#if defined(ANYPS5_KYTY)
			g_gpu_state->NoteBlockedWait(reinterpret_cast<uint64_t>(addr), sizeof(T), ref, mask,
			                             func);
#else
			g_gpu_state->NoteBlockedWait(reinterpret_cast<uint64_t>(addr), sizeof(T));
#endif
		}
		SuspendPm4();
	}
}

void CommandProcessor::YieldIfWaitedOn(uint64_t address, uint64_t size) {
	if (g_gpu_state != nullptr && g_current_execution != nullptr &&
	    g_gpu_state->OtherQueueWaitsOn(address, size)) {
		g_current_execution->m_yield = true;
	}
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t);

void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
		YieldIfWaitedOn(reinterpret_cast<uint64_t>(counter), sizeof(*counter));
	} else if (*counter == 0) {
		if (g_gpu_state != nullptr) {
#if defined(ANYPS5_KYTY)
			g_gpu_state->NoteBlockedWait(reinterpret_cast<uint64_t>(counter), sizeof(*counter), 0,
			                             ~uint64_t {0}, 6);
#else
			g_gpu_state->NoteBlockedWait(reinterpret_cast<uint64_t>(counter), sizeof(*counter));
#endif
		}
		SuspendPm4();
	} else {
		(*counter)--;
		m_renderer.CollectHotPageWrites();
	}
}

void CommandProcessor::WriteData(uint64_t destination, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}

	if (dst_sel == 6) {
		EmitCallback([this, destination, src, dw_num, write_one_address] {
			auto offset = destination;
			for (uint32_t i = 0; i < dw_num; i++, offset += !write_one_address) {
				m_renderer.GetBufferCache().FillBuffer((offset & 0xffffu) * 4u, 4u, src[i], true);
			}
		});
		return;
	}

	auto* dst = reinterpret_cast<uint32_t*>(destination);
	const auto sequence = EmitCallback([this, dst, src, dw_num, write_one_address] {
		const auto* data  = write_one_address ? src + dw_num - 1 : src;
		const auto  bytes = uint64_t {write_one_address ? 1u : dw_num} * sizeof(uint32_t);
#if defined(ANYPS5_KYTY)
		SettleInPlaceLabels(GetScheduler(), reinterpret_cast<uint64_t>(dst), bytes);
#endif
		if (m_renderer.WriteAroundGpuWrites(dst, data, bytes)) {
			LibKernel::Memory::NoteEmulatorWrite(reinterpret_cast<uint64_t>(dst), bytes);
			return;
		}
		m_renderer.PrepareGpuWrite(dst, bytes);
		auto* output = static_cast<volatile uint32_t*>(dst);
		for (uint32_t i = 0; i < dw_num; i++) {
			*output = src[i];
			output += !write_one_address;
		}
		LibKernel::Memory::NoteEmulatorWrite(reinterpret_cast<uint64_t>(dst), bytes);
	}, {.address = reinterpret_cast<uint64_t>(dst), .data = src, .dwords = dw_num,
	    .one_address = write_one_address});
	if (write_one_address) {
		NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst), sizeof(uint32_t),
		               src + dw_num - 1);
	} else {
		NoteGuestWrite(sequence, reinterpret_cast<uint64_t>(dst),
		               uint64_t {dw_num} * sizeof(uint32_t), src);
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto sequence = EmitCallback([this, dst_address, num_bytes] {
#if defined(ANYPS5_KYTY)
		SettleInPlaceLabels(GetScheduler(), dst_address, num_bytes);
#endif
		auto& buffers = m_renderer.GetBufferCache();
		if (GuestRange {dst_address, num_bytes}.Valid() &&
		    !buffers.IsRegionGpuModified(dst_address, num_bytes)) {
			void* address = reinterpret_cast<void*>(dst_address);
			m_renderer.PrepareGpuWrite(address, num_bytes);
			m_deferred_labels.fetch_add(1, std::memory_order_relaxed);
#if defined(ANYPS5_KYTY)
			s_deferred_labels.fetch_add(1, std::memory_order_relaxed);
#endif
			GetScheduler().DeferPriorityOperation(
			    [this, address, num_bytes] {
				    const auto value = Sync::ReadReferenceClock();
				    m_renderer.WriteFromCompletion(address, &value, num_bytes);
				    m_deferred_labels.fetch_sub(1, std::memory_order_release);
#if defined(ANYPS5_KYTY)
				    s_deferred_labels.fetch_sub(1, std::memory_order_release);
#endif
			    },
			    dst_address, num_bytes);
			GetScheduler().Flush();
			return;
		}
		const auto value = Sync::ReadReferenceClock();
		std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
		LibKernel::Memory::NoteEmulatorWrite(dst_address, num_bytes);
		static std::atomic<uint32_t> clock_log_count {0};
		if (clock_log_count.fetch_add(1) < 64) {
			LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
			     " size=%u\n",
			     dst_address, value, num_bytes);
		}
	});
	NoteGuestWrite(sequence, dst_address, num_bytes);
}

void CommandProcessor::DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes) {
	if (num_bytes == 0) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	bool src_gds = false;
	if (src_sel != 2) {
		if (!decode_gds(src_sel, src_gds)) {
			EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
		}
		if (src_gds && dst_gds) {
			EXIT("unsupported dmaData GDS-to-GDS copy\n");
		}
	}
	const auto sequence = EmitCallback([=, this] {
		ExecuteDmaData(dst_gds, dst_address_or_offset, src_sel, src_gds,
		               src_address_or_offset_or_immediate, num_bytes);
	});
	if (!dst_gds) {
		NoteGuestWrite(sequence, dst_address_or_offset, num_bytes);
	}
}

void CommandProcessor::ExecuteDmaData(bool dst_gds, uint64_t dst_address_or_offset,
                                      uint8_t src_sel, bool src_gds,
                                      uint64_t src_address_or_offset_or_immediate,
                                      uint32_t num_bytes) {
	auto& buffer_cache = m_renderer.GetBufferCache();
#if defined(ANYPS5_KYTY)
	if (!dst_gds) {
		SettleInPlaceLabels(GetScheduler(), dst_address_or_offset, num_bytes);
	}
#endif
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	if (!src_gds) {
		m_renderer.GetOcclusionQueries().WaitForCounters(src_address_or_offset_or_immediate,
		                                                 num_bytes);
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	if (submission.queue_id == 0 && submission.type == SubmissionType::Commands) {
		if (Scout::Running()) {
			submission.scout_sequence = ++m_scout_sequence;
		}
	}
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_thread = !gpu->m_pipelined;
	g_cp_thread  = gpu->m_pipelined;
	g_gpu_state  = gpu;
	if (gpu->m_pipelined) {
		Config::ConfigureGpuStageThread(Config::GpuStageThread::CommandProcessor);
	}

	std::optional<std::chrono::steady_clock::time_point> blocked_since;
	bool                                                  spin_blocked = false;

	for (;;) {
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         should_stop    = false;
		if (spin_blocked) {
			spin_blocked      = false;
			const auto epoch  = gpu->m_work_epoch.load(std::memory_order_acquire);
			const auto target = std::chrono::steady_clock::now() + BlockedRetryInterval;
			while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch &&
			       std::chrono::steady_clock::now() < target) {
				Common::SpinPause();
			}
		}
		if (gpu->m_pipelined) {
			bool     no_work = false;
			uint64_t epoch   = 0;
			{
				Common::LockGuard lock(gpu->m_queue_mutex);
				no_work = gpu->m_commands.empty() && gpu->m_submission_count == 0 &&
				          !gpu->m_stopping;
			}
			if (no_work) {
				gpu->WaitForExecution();
				{
					Common::LockGuard lock(gpu->m_queue_mutex);
					epoch = gpu->m_work_epoch.load(std::memory_order_acquire);
					if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
						gpu->m_processing = false;
						gpu->m_idle.SignalAll();
					}
				}
				Common::SpinWait spin;
				while (gpu->m_work_epoch.load(std::memory_order_acquire) == epoch && spin.Spin()) {
				}
			}
		}
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
#if defined(ANYPS5_KYTY)
			} else if (gpu->m_hold.load(std::memory_order_acquire) && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
				continue;
#endif
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					const auto now    = std::chrono::steady_clock::now();
					if (gpu->m_pipelined && !blocked_since) {
						blocked_since = now;
					}
					if (gpu->m_pipelined && now - *blocked_since < Common::SpinBudget()) {
						spin_blocked = true;
					} else {
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					}
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
#if defined(ANYPS5_KYTY)
				gpu->m_in_submission = true;
#endif
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			if (gpu->m_pipelined) {
				gpu->StopExecution();
			}
			g_gpu_state  = nullptr;
			g_gpu_thread = false;
			g_cp_thread  = false;
			return;
		}

		if (command) {
			EXIT_IF(g_current_processor != nullptr);
			if (gpu->m_pipelined) {
				gpu->m_gfx_cp->EmitHostCommand(std::move(command));
				continue;
			}
			{
				Common::DebugCounters::ScopedTimer busy(
				    Common::DebugCounters::Counter::GpuThreadBusyNs);
				command();
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		const bool complete = [&] {
			Common::DebugCounters::ScopedTimer busy(
			    Common::DebugCounters::Counter::GpuThreadBusyNs);
			return gpu->Process(submission);
		}();
		if (complete) {
			blocked_since.reset();
		}

		Common::LockGuard lock(gpu->m_queue_mutex);
#if defined(ANYPS5_KYTY)
		gpu->m_in_submission = false;
		if (gpu->m_hold.load(std::memory_order_acquire)) {
			gpu->m_idle.SignalAll();
		}
#endif
		if (submission.yielded) {
			blocked_since.reset();
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		if (!complete) {
			submission.blocked = !submission.yielded;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		if (gpu->m_pipelined) {
			continue;
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

#if defined(ANYPS5_KYTY)
void GuestGpu::NoteBlockedWait(uint64_t address, uint64_t size, uint64_t reference,
                               uint64_t mask, uint32_t function) {
#else
void GuestGpu::NoteBlockedWait(uint64_t address, uint64_t size) {
#endif
	auto& range = m_wait_ranges[m_current_queue];
	if (range.begin == range.end) {
		m_waiting_queues++;
	}
	range = {address, address + size};
#if defined(ANYPS5_KYTY)
	RecordBlockedWait(address, reference, mask, function, static_cast<uint32_t>(size));
#endif
}
#if defined(ANYPS5_KYTY)

void GuestGpu::NoteBlockedRewind(uint64_t control) {
	RecordBlockedWait(control, 0, 0, 0, 0);
}

void GuestGpu::RecordBlockedWait(uint64_t address, uint64_t reference, uint64_t mask,
                                 uint32_t function, uint32_t bytes) {
	auto&      slot     = m_blocked_slots[m_current_queue];
	const auto sequence = slot.sequence.load(std::memory_order_relaxed);
	slot.sequence.store(sequence + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	slot.address.store(address, std::memory_order_relaxed);
	slot.reference.store(reference, std::memory_order_relaxed);
	slot.mask.store(mask, std::memory_order_relaxed);
	slot.shape.store(function | (uint64_t {bytes} << 32u), std::memory_order_relaxed);
	slot.sequence.store(sequence + 2, std::memory_order_release);
}

size_t GuestGpu::BlockedWaits(BlockedWait* waits, size_t capacity) const {
	size_t count = 0;
	for (uint32_t queue = 0; queue < QueueCount && count < capacity; queue++) {
		const auto& slot = m_blocked_slots[queue];
		for (int attempt = 0; attempt < 4; attempt++) {
			const auto before = slot.sequence.load(std::memory_order_acquire);
			if ((before & 1u) != 0) {
				continue;
			}
			BlockedWait wait;
			wait.address     = slot.address.load(std::memory_order_relaxed);
			wait.reference   = slot.reference.load(std::memory_order_relaxed);
			wait.mask        = slot.mask.load(std::memory_order_relaxed);
			const auto shape = slot.shape.load(std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_acquire);
			if (slot.sequence.load(std::memory_order_relaxed) != before) {
				continue;
			}
			if (wait.address != 0) {
				wait.function  = static_cast<uint32_t>(shape);
				wait.bytes     = static_cast<uint32_t>(shape >> 32u);
				wait.queue     = queue;
				waits[count++] = wait;
			}
			break;
		}
	}
	return count;
}

bool GuestGpu::HoldSubmissions(uint32_t timeout_ms) {
	EXIT_IF(IsGpuThread() || g_cp_thread);
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	Common::LockGuard lock(m_queue_mutex);
	m_hold.store(true, std::memory_order_release);
	while (m_in_submission) {
		if (std::chrono::steady_clock::now() >= deadline) {
			m_hold.store(false, std::memory_order_release);
			m_work_epoch.fetch_add(1, std::memory_order_release);
			m_work_available.Signal();
			return false;
		}
		m_idle.WaitFor(&m_queue_mutex, 50);
	}
	return true;
}

void GuestGpu::ReleaseSubmissions() {
	Common::LockGuard lock(m_queue_mutex);
	m_hold.store(false, std::memory_order_release);
	m_work_epoch.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}
#endif

bool GuestGpu::OtherQueueWaitsOn(uint64_t address, uint64_t size) const {
	if (m_waiting_queues == 0) {
		return false;
	}
	for (uint32_t queue = 0; queue < QueueCount; queue++) {
		const auto& range = m_wait_ranges[queue];
		if (queue != m_current_queue && range.begin < address + size && address < range.end) {
			return true;
		}
	}
	return false;
}

bool GuestGpu::Process(Submission& submission) {
	m_current_queue = submission.queue_id;
#if defined(ANYPS5_KYTY)
	if (m_blocked_slots[m_current_queue].address.load(std::memory_order_relaxed) != 0) {
		RecordBlockedWait(0, 0, 0, 0, 0);
	}
#endif
	if (auto& range = m_wait_ranges[m_current_queue]; range.begin != range.end) {
		range = {};
		m_waiting_queues--;
	}
	submission.yielded = false;
	auto& cp           = GetProcessor(submission.queue_id);

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Commands: {
			if (submission.scout_sequence != 0) submission.execution.SetScoutSequence(submission.scout_sequence);
			const auto result  = cp.Process(submission.execution, submission.commands);
			complete           = result == Pm4ProcessResult::Complete;
			submission.yielded = result == Pm4ProcessResult::Yielded;
			if (submission.execution.MadeProgress() && !submission.yielded) {
				cp.BufferFlush();
			}
#if defined(ANYPS5_KYTY)
			if (submission.execution.MadeProgress()) {
				submission.wait_flushed = 0;
			} else if (!complete && !submission.yielded) {
				const auto& range = m_wait_ranges[m_current_queue];
				if (range.begin != range.end) {
					const auto newest = m_renderer.GetCommandScheduler().NewestPendingGuestOperation(
					    range.begin, range.end - range.begin, submission.wait_flushed);
					if (newest != 0) {
						cp.BufferFlush();
						submission.wait_flushed = newest;
					}
				}
			}
#endif
			break;
		}
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			cp.RunGarbageCollector();
			cp.EmitHostCommand([this, ready = m_suspend_point_ready] {
				m_renderer.GetCommandScheduler().DeferHostOperation([ready] { ready->release(); },
				                                                    true);
			});
#if defined(ANYPS5_KYTY)
			if (submission.frame_end) {
				cp.EmitHostCommand([this, frame = m_frame_ready] {
					m_renderer.GetCommandScheduler().DeferHostOperation([frame] { frame->release(); },
					                                                    true);
				});
			}
#endif
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

static std::atomic<uint64_t> g_guest_submissions {0};

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		m_submit_id = g_guest_submissions.fetch_add(1, std::memory_order_relaxed) + 1;
		m_renderer.CollectHotPageWrites();
		m_renderer.GetBufferCache().NoteCommandBuffer(reinterpret_cast<uint64_t>(commands.data()), commands.size_bytes());
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_yield         = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		explicit ExecutionScope(Pm4Execution& execution) { g_current_execution = &execution; }
		~ExecutionScope() { g_current_execution = nullptr; }
	} execution_scope(execution);

	ProcessPm4(execution);
	if (execution.m_buffer_stack.empty()) {
		return Pm4ProcessResult::Complete;
	}
	return execution.m_suspended ? Pm4ProcessResult::Blocked : Pm4ProcessResult::Yielded;
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	m_renderer.GetBufferCache().NoteCommandBuffer(reinterpret_cast<uint64_t>(commands.data()), commands.size_bytes());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SyncFetchWithExecution() {
	if (m_pipeline != nullptr) {
		m_fetch_sync_sequence = m_pipeline->NextSequence() - 1;
		m_pipeline->NoteOrderedSequence(m_fetch_sync_sequence);
	}
}

const uint32_t* CommandProcessor::StageIndirectRegisters(uint32_t opcode, const uint32_t* packet,
                                                         uint32_t packet_dw) {
	if ((opcode != Pm4::IT_SET_CONTEXT_REG_INDIRECT && opcode != Pm4::IT_SET_SH_REG_INDIRECT &&
	     opcode != Pm4::IT_SET_UCONFIG_REG_INDIRECT) ||
	    packet_dw < 5 || m_pipeline == nullptr) {
		return packet;
	}
	const auto     address = (uint64_t {packet[1]} & 0xfffffffcu) | (uint64_t {packet[2]} << 32u);
	const uint32_t count   = packet[4] & 0x3fffu;
	const uint64_t bytes   = uint64_t {count} * ((packet[3] & Pm4::LOAD_REG_INDEX_OFFSET_AND_DATA) != 0 ? 8u : 4u);
	if (address == 0 || bytes == 0 || !GuestRange {address, bytes}.Valid() ||
	    !LibKernel::Memory::MayFaultOnGpuRead(address, bytes) ||
	    !m_renderer.GetBufferCache().NeverGpuWritten(address, bytes) ||
	    m_renderer.GetTextureCache().IsRegionGpuModified(address, bytes)) {
		return packet;
	}
	m_indirect_stage.assign(size_t {count} * 2u, 0u);
	if (!LibKernel::Memory::TryReadBacking(address, m_indirect_stage.data(), bytes)) {
		return packet;
	}
	const auto staged = reinterpret_cast<uint64_t>(m_indirect_stage.data());
	std::copy_n(packet, m_indirect_packet.size(), m_indirect_packet.begin());
	m_indirect_packet[1] = (static_cast<uint32_t>(staged) & 0xfffffffcu) | (packet[1] & 3u);
	m_indirect_packet[2] = static_cast<uint32_t>(staged >> 32u);
	BufferCache::NoteStagedCommandProcessorRead();
	return m_indirect_packet.data();
}

void CommandProcessor::PrepareCommandFetch(std::span<const uint32_t> commands) {
	if (m_fetch_sync_sequence == 0 || m_pipeline == nullptr) {
		return;
	}
	m_pipeline->WaitForExecution(std::exchange(m_fetch_sync_sequence, 0));
	const auto address = reinterpret_cast<uint64_t>(commands.data());
	const auto size    = commands.size_bytes();
	auto&      buffers = m_renderer.GetBufferCache();
	if (size != 0 && GuestRange {address, size}.Valid() && buffers.MayBeGpuModified(address, size) &&
	    buffers.IsRegionGpuModified(address, size)) {
		buffers.ReadMemory(address, size);
	}
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	const bool debug_dump = graphics_debug_dump_enabled();
	while (!execution.m_buffer_stack.empty()) {
		if (g_gpu_state != nullptr) {
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		if (remaining_dw < 2) {
			static std::atomic<uint32_t> tail_log_count {0};
			if (tail_log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("CP: trailing dword 0x%08" PRIx32 " at offset=0x%05" PRIx32 "\n", packet[0],
				     total_dw - remaining_dw);
			}
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		if (debug_dump) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     cursor.offset_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		const auto* const view = StageIndirectRegisters(opcode, packet, std::min(KYTY_PM4_LEN(packet_header), remaining_dw));

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			if (m_draw_observer != nullptr) m_draw_observer->ObservePacket(opcode, view, packet_dw, true);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			if (execution.m_scout_sequence != 0 && Scout::IsDraw(opcode)) execution.m_scout_ordinal++;
			continue;
		}

		auto handler = g_cp_op_func[opcode];
		if (m_draw_observer != nullptr) {
			m_draw_observer->ObservePacket(opcode, view, std::min(KYTY_PM4_LEN(packet_header), remaining_dw), false);
		}

		if (handler == nullptr) {
			const auto offset = cursor.offset_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n", offset, packet_header);
		}

		const bool scout_draw = execution.m_scout_sequence != 0 && Scout::IsDraw(opcode);
		if (scout_draw) {
			m_scout_packet   = packet;
			m_scout_sequence = execution.m_scout_sequence;
			m_scout_ordinal  = execution.m_scout_ordinal;
		}
		const auto packet_dw =
		    handler(*this, packet_header & ~1u, view + 1, remaining_dw, total_dw) + 1;
		EXIT_IF(packet_dw > remaining_dw);
		if (scout_draw) {
			m_scout_packet = nullptr;
			if (!execution.m_suspended) ++execution.m_scout_ordinal;
		}
		if (execution.m_suspended) {
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
		if (execution.m_yield) {
			return;
		}
	}
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	m_index_type_and_size = index_type_and_size & 0x3u;
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	m_index_base_addr = index_base_addr;
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	m_index_buffer_size = index_buffer_size;
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	m_draw_indirect_args_base_addr = draw_indirect_args_base_addr;
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	m_dispatch_indirect_args_base_addr = dispatch_indirect_args_base_addr;
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	m_num_instances.value = num_instances;
	m_num_instances.sequence++;
}

uint32_t CommandProcessor::ResolveNumInstances(const GpuOperation& operation) {
	if (m_has_indirect_instances &&
	    m_indirect_instances_sequence == operation.num_instances.sequence) {
		const auto& gpu = m_gpu_indirect_instances;
		if (gpu.address != 0) {
			uint32_t count = gpu.count;
			if (gpu.count_address != 0) {
				const uint32_t written =
				    *reinterpret_cast<const volatile uint32_t*>(gpu.count_address);
				count = std::min(written, count);
			}
			m_indirect_instances =
			    count != 0 ? *reinterpret_cast<const volatile uint32_t*>(
			                     gpu.address + uint64_t {count - 1} * gpu.stride +
			                     offsetof(vk::DrawIndirectCommand, instanceCount))
			               : m_indirect_instances;
			m_gpu_indirect_instances = {};
		}
		return m_indirect_instances;
	}
	return operation.num_instances.value;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address) {
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			constexpr uint64_t      ready_bit = 1ull << 63u;
			std::array<uint64_t, 32> results {};
			ReadDecisionMemory(address, results.data(), sizeof(results));
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						if (!m_occlusion_wait_requested) {
							m_occlusion_wait_requested = true;
							EmitCallback([this] {
								auto& queries = m_renderer.GetOcclusionQueries();
								if (queries.HasPendingDumps()) {
									GetScheduler().Finish();
									queries.PublishCompleted(true);
								}
							});
						}
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
			m_occlusion_wait_requested = false;
		} break;
		case 0x03:
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = ReadDecisionValue<uint64_t>(address);
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	args.index_type_and_size = m_index_type_and_size;
	auto operation       = MakeOperation(GpuOperationKind::DrawIndex);
	operation.draw_index = args;
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDrawIndex(const GpuOperation& operation, DrawIndexArgs args) {
	NoteRecordedWork();
	if (args.instance_count == 0) {
		args.instance_count = ResolveNumInstances(operation);
	}
	auto& executor = m_renderer.GetRenderExecutor();
	if (args.indirect.address == 0 &&
	    executor.TableXprTry(CurrentBuffer(), 0, 1, 0, 0, 0, vk::IndexType::eUint16, &args, operation.clean)) {
		executor.NoteTableServed();
		return;
	}
	executor.DrawIndex(CurrentBuffer(), args);
	if (args.indirect.address == 0) {
		executor.NativeXprEndDraw();
	}
}

const void* CommandProcessor::IndexAddress(uint64_t base, uint32_t type_and_size,
                                           uint32_t first_index) {
	uint64_t index_size = 0;
	switch (type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", type_and_size);
	}
	return reinterpret_cast<const void*>(base + first_index * index_size);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	return IndexAddress(m_index_base_addr, m_index_type_and_size, first_index);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets) {
	if (registers.index_offset != Pm4::SH_NOP || registers.draw_index != Pm4::SH_NOP) {
		return false;
	}
	for (const auto [location, slot]: {std::pair {registers.vertex_offset, &offsets.vertex},
	                                   std::pair {registers.instance_offset, &offsets.instance}}) {
		if (location == Pm4::SH_NOP) {
			continue;
		}
		const uint32_t index = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
		if (index >= HW::UserSgprInfo::SGPRS_MAX) {
			return false;
		}
		*slot = static_cast<uint8_t>(index);
	}
	return true;
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed, const uint32_t* run_offsets) {
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_draw_indirect_args_base_addr == 0);
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	auto operation          = MakeOperation(GpuOperationKind::DrawIndirect);
	operation.draw_indirect = {.args_base         = m_draw_indirect_args_base_addr,
	                           .data_offset       = data_offset,
	                           .max_count         = max_count_or_count,
	                           .count_addr        = count_addr,
	                           .stride            = stride_in_bytes,
	                           .draw_initiator    = draw_initiator,
	                           .index_buffer_size = m_index_buffer_size,
	                           .index_type        = static_cast<uint8_t>(m_index_type_and_size),
	                           .indexed           = indexed,
	                           .multi             = true,
	                           .index_base        = m_index_base_addr,
	                           .registers         = registers,
	                           .run_offsets       = run_offsets};
	Emit(std::move(operation));
}

bool CommandProcessor::TakesDrawRuns() const {
	const auto& executor = m_renderer.GetRenderExecutor();
	return executor.TableDrewDraws() && (executor.TableHint(m_sh_ctx) & 1u) != 0;
}

void CommandProcessor::DrawIndirectRun(const uint32_t* packets, uint32_t count, IndirectDrawRegisters registers,
                                       uint32_t draw_initiator) {
	DrawIndirectMulti(packets[0], count, nullptr, sizeof(vk::DrawIndexedIndirectCommand), registers, draw_initiator,
	                  true, packets);
}

static bool PatchUserSgpr(HW::Shader& shaders, uint32_t location, uint32_t value) {
	const auto in  = [location](uint32_t base) { return location >= base && location < base + 32u; };
	const auto set = [value](const HW::UserSgprInfo& sgprs, uint32_t id, auto&& setter) {
		if (sgprs.value[id] == value) {
			return false;
		}
		setter(id, value);
		return true;
	};
	if (in(Pm4::SPI_SHADER_USER_DATA_GS_0)) {
		return set(shaders.GetVs().gs_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_GS_0,
		           [&](uint32_t id, uint32_t v) { shaders.SetGsUserSgpr(id, v); });
	}
	if (in(Pm4::SPI_SHADER_USER_DATA_HS_0)) {
		return set(shaders.GetVs().hs_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_HS_0,
		           [&](uint32_t id, uint32_t v) { shaders.SetHsUserSgpr(id, v); });
	}
	if (in(Pm4::SPI_SHADER_USER_DATA_PS_0)) {
		return set(shaders.GetPs().ps_user_sgpr, location - Pm4::SPI_SHADER_USER_DATA_PS_0,
		           [&](uint32_t id, uint32_t v) { shaders.SetPsUserSgpr(id, v); });
	}
	EXIT("unsupported indirect draw offset register 0x%" PRIx32 "\n", location);
	return false;
}

bool CommandProcessor::PatchIndirectDrawOffsets(const GpuOperation& operation,
                                                uint32_t vertex_offset, uint32_t instance_offset,
                                                uint32_t first_index, uint32_t draw_index) {
	const auto& registers = operation.draw_indirect.registers;
	if (registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP && registers.draw_index == Pm4::SH_NOP) {
		return false;
	}
	const std::pair<uint32_t, uint32_t> writes[] = {{registers.vertex_offset, vertex_offset},
	                                                {registers.instance_offset, instance_offset},
	                                                {registers.index_offset, first_index},
	                                                {registers.draw_index, draw_index}};
	if (m_pipeline == nullptr) {
		for (const auto& [location, value]: writes) {
			if (location == Pm4::SH_NOP) {
				continue;
			}
			EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
			                     g_hw_sh_indirect_func[location] == nullptr);
			g_hw_sh_indirect_func[location](*this, location, value);
		}
		return true;
	}
	if (m_indirect_shaders == nullptr) {
		m_indirect_shaders = std::make_unique<HW::Shader>();
	}
	*m_indirect_shaders = *operation.state.shaders;
	bool changed        = false;
	for (const auto& [location, value]: writes) {
		if (location != Pm4::SH_NOP) {
			changed |= PatchUserSgpr(*m_indirect_shaders, location, value);
		}
	}
	if (changed) {
		m_renderer.GetRenderExecutor().UseResolvedDraw(nullptr);
	}
	GetScheduler().Begin(*operation.state.context, *operation.state.user_config,
	                     *m_indirect_shaders);
	return true;
}

void CommandProcessor::ExecuteDrawIndirect(const GpuOperation& operation) {
	const auto& indirect = operation.draw_indirect;
	if (indirect.max_count == 0) {
		return;
	}
	const auto  args_addr = indirect.args_base + indirect.data_offset;
	const auto& registers = indirect.registers;
	const auto record_address = [&](uint32_t i) {
		return indirect.run_offsets != nullptr ? indirect.args_base + indirect.run_offsets[5 * i]
		                                       : args_addr + uint64_t {i} * indirect.stride;
	};
	const auto count_address = reinterpret_cast<uint64_t>(indirect.count_addr);
	uint32_t   first_cpu_draw = 0;
	DrawIndirectArgs records {args_addr, indirect.max_count, indirect.stride};
	records.count_address = count_address;
	if ((count_address == 0 || (m_renderer.GetGraphics().indirect_draws_enabled &&
	                            (count_address & 3u) == 0)) &&
	    IndirectDrawOffsets(registers, records.offsets) &&
	    (!indirect.indexed || indirect.index_buffer_size != 0)) {
		m_gpu_indirect_instances = records;
		if (indirect.run_offsets != nullptr) {
			m_gpu_indirect_instances.address = record_address(indirect.max_count - 1);
			m_gpu_indirect_instances.count   = 1;
		}
		m_indirect_instances_sequence = operation.num_instances.sequence;
		m_has_indirect_instances      = true;
		NoteRecordedWork();
		auto& executor = m_renderer.GetRenderExecutor();
		if (indirect.indexed && count_address == 0 && indirect.max_count <= 64 &&
		    indirect.stride >= sizeof(vk::DrawIndexedIndirectCommand) && (indirect.stride & 3u) == 0 &&
		    (indirect.index_type == 0 || indirect.index_type == 1) &&
		    executor.TableXprTry(CurrentBuffer(), args_addr, indirect.max_count, indirect.stride,
		                         indirect.index_base,
		                         uint64_t {indirect.index_buffer_size} * (indirect.index_type == 0 ? 2u : 4u),
		                         indirect.index_type == 0 ? vk::IndexType::eUint16 : vk::IndexType::eUint32,
		                         nullptr, operation.clean, records.offsets, indirect.run_offsets)) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, indirect.max_count);
			executor.NoteTableServed();
			return;
		}
		while (indirect.run_offsets != nullptr && first_cpu_draw < indirect.max_count) {
			DrawIndirectArgs record = records;
			record.address          = record_address(first_cpu_draw);
			record.count            = 1;
			const bool drawn_one =
			    executor.DrawIndex(CurrentBuffer(), {.index_count         = indirect.index_buffer_size,
			                                         .index_addr          = IndexAddress(indirect.index_base,
			                                                                             indirect.index_type, 0),
			                                         .index_type_and_size = indirect.index_type,
			                                         .indirect            = record});
			executor.NativeXprEndDraw();
			if (!drawn_one) break;
			Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, 1);
			if (++first_cpu_draw == indirect.max_count) return;
		}
		if (indirect.run_offsets == nullptr) {
			const bool drawn =
			    indirect.indexed
			        ? executor.DrawIndex(CurrentBuffer(),
			                             {.index_count         = indirect.index_buffer_size,
			                              .index_addr          = IndexAddress(indirect.index_base,
			                                                                  indirect.index_type, 0),
			                              .index_type_and_size = indirect.index_type,
			                              .indirect            = records})
			        : executor.DrawAuto(CurrentBuffer(), {.indirect = records});
			executor.NativeXprEndDraw();
			if (drawn) {
				Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws,
				                           indirect.max_count);
				return;
			}
		}
	}

	uint32_t draw_count = indirect.max_count;
	if (indirect.count_addr != nullptr) {
		draw_count = *indirect.count_addr;
		if (draw_count > indirect.max_count) {
			draw_count = indirect.max_count;
		}
	}
	m_gpu_indirect_instances = {};
	if (draw_count <= first_cpu_draw) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count - first_cpu_draw);

	const uint32_t max_indices = indirect.index_buffer_size != 0 ? indirect.index_buffer_size : UINT32_MAX;
	for (uint32_t i = first_cpu_draw; i < draw_count; i++) {
		const auto* record_addr = reinterpret_cast<const void*>(record_address(i));

		if (!indirect.indexed) {
			vk::DrawIndirectCommand args {};
			std::memcpy(&args, record_addr, sizeof(args));
			m_indirect_instances          = args.instanceCount;
			m_indirect_instances_sequence = operation.num_instances.sequence;
			m_has_indirect_instances      = true;
			PatchIndirectDrawOffsets(operation, args.firstVertex, args.firstInstance, 0, i);
			ExecuteDrawAuto(operation, {.vertex_count   = args.vertexCount,
			                            .instance_count = args.instanceCount,
			                            .first_vertex   = args.firstVertex,
			                            .first_instance = args.firstInstance,
			                            .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		vk::DrawIndexedIndirectCommand args {};
		std::memcpy(&args, record_addr, sizeof(args));
		PatchIndirectDrawOffsets(operation, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);

		const auto*    index_addr = IndexAddress(indirect.index_base, indirect.index_type, args.firstIndex);
		const uint32_t index_count = std::min(args.indexCount, max_indices);
		m_indirect_instances          = args.instanceCount;
		m_indirect_instances_sequence = operation.num_instances.sequence;
		m_has_indirect_instances      = true;
		ExecuteDrawIndex(operation, {.index_count         = index_count,
		                             .index_addr          = index_addr,
		                             .instance_count      = args.instanceCount,
		                             .index_type_and_size = indirect.index_type,
		                             .base_vertex    = args.vertexOffset,
		                             .first_instance = args.firstInstance,
		                             .offset_source  = DrawOffsetSource::IndirectArgs});
	}
}

Psr::Shader CommandProcessor::ComputePsrShader(uint64_t data_addr) {
	auto [it, inserted] = m_cs_psr_shaders.try_emplace(data_addr, Psr::Shader::None);
	if (inserted) {
		it->second = Psr::IdentifyShader(data_addr);
	}
	return it->second;
}

static void ReadGuestAroundGpuWrites(uint64_t vaddr, void* data, uint64_t size) {
	if (!Libs::LibKernel::Memory::TryReadAroundGpuWrites(vaddr, data, size)) {
		std::memcpy(data, reinterpret_cast<const void*>(vaddr), size);
	}
}

static uint32_t ReadRefitBlasSize(uint64_t blas) {
	uint32_t size = 0;
	ReadGuestAroundGpuWrites(blas + 16, &size, sizeof(size));
	return size;
}

static void ReportRefitOutsideGuest(std::span<const std::pair<uint64_t, uint32_t>> dropped) {
	for (const auto& [blas, size]: dropped) {
		Common::HackNotice::Report("BLAS refit skipped for a BLAS outside guest memory", "CommandProcessor::EmitHleRefit",
		                           0, 0, fmt::format("BLAS 0x{:x}, {} bytes", blas, size));
	}
}

bool CommandProcessor::HleBlasDispatch(uint64_t data_addr) {
	if (data_addr == 0 || !m_renderer.GetRayTracing()->Enabled()) {
		return false;
	}
	using Psr::Shader;
	const auto shader       = ComputePsrShader(data_addr);
	const bool base515      = Psr::LibraryGeneration() == Psr::Generation::Base515;
	const bool refit_shader = shader == Shader::QbvhFanRefitCopyDispatchArgs ||
	                          shader == Shader::QbvhFanRefitAsync ||
	                          shader == Shader::QbvhFanRefitRecoverAabbs ||
	                          shader == Shader::QbvhRefitBottomLevel ||
	                          (base515 && shader == Shader::QbvhFanRefitAsyncShaderMesh);
	if (base515 && m_hle_refit_window && shader != Shader::QbvhFanRefitCopyDispatchArgs) {
		EmitHleRefitArgs();
	}
	if (!refit_shader) {
#if defined(ANYPS5_KYTY)
		if (shader == Shader::BasicsFillDword) {
			return false;
		}
#endif
		if (m_hle_refit_window && shader != Shader::None) {
			Psr::StopReplacement("a BLAS refit runs a pass the native refit does not replace",
			                     ShaderRegisteredName(data_addr));
			if (shader == Shader::MortonCaptureMemory) {
				m_hle_refit_window = false;
				EmitHleRefit();
			}
		}
		return false;
	}
	if (!m_hle_refit_window) {
		if (shader != Shader::QbvhFanRefitCopyDispatchArgs || !Psr::ReplacementEnabled()) {
			return false;
		}
		m_hle_refit_window = true;
		m_hle_refit_blases.clear();
		m_hle_refit_geometries.clear();
		m_hle_refit_finished = 0;
		m_hle_refit_args     = 0;
	}
	const auto& user = GetShCtx().GetCs().cs_user_sgpr.value;
#if defined(ANYPS5_KYTY)
	const bool base148 = Psr::LibraryGeneration() == Psr::Generation::Base148;
#endif
	if (shader == Shader::QbvhFanRefitCopyDispatchArgs) {
#if defined(ANYPS5_KYTY)
		const uint64_t blas = (uint64_t {user[0]} | (uint64_t {user[1]} << 32u)) & 0xffffffffffffull;
		if (!base148 || std::find(m_hle_refit_blases.begin(), m_hle_refit_blases.end(), blas) ==
		                    m_hle_refit_blases.end()) {
			m_hle_refit_blases.push_back(blas);
		}
#else
		m_hle_refit_blases.push_back((uint64_t {user[0]} | (uint64_t {user[1]} << 32u)) &
		                             0xffffffffffffull);
#endif
	} else if (shader == Shader::QbvhFanRefitAsyncShaderMesh) {
		return false;
	} else if (shader == Shader::QbvhFanRefitAsync) {
#if defined(ANYPS5_KYTY)
		const bool geometry_s6 = Psr::LibraryGeneration() == Psr::Generation::Base252 || base148;
#else
		const bool geometry_s6 = Psr::LibraryGeneration() == Psr::Generation::Base252;
#endif
		m_hle_refit_geometries.push_back(
		    {.blas     = uint64_t {user[4]} | (uint64_t {user[5] & 0xffffu} << 32u),
		     .table    = uint64_t {user[0]} | (uint64_t {user[1] & 0xffffu} << 32u),
		     .geometry = geometry_s6 ? user[6] : user[8]});
	} else if (shader == Shader::QbvhRefitBottomLevel &&
	           ++m_hle_refit_finished == m_hle_refit_blases.size()) {
		m_hle_refit_window = false;
		EmitHleRefit();
	}
	return true;
}

void CommandProcessor::EmitHleRefitArgs() {
	if (m_hle_refit_args >= m_hle_refit_blases.size()) {
		return;
	}
	std::vector<uint64_t> blases(m_hle_refit_blases.begin() + static_cast<ptrdiff_t>(m_hle_refit_args),
	                             m_hle_refit_blases.end());
	m_hle_refit_args = m_hle_refit_blases.size();
	EmitHostCommand([this, blases = std::move(blases)] {
		auto&                                     context = m_renderer;
		auto&                                     cache   = context.GetBufferCache();
		auto                                      refit   = blases;
		std::vector<uint32_t>                     sizes;
		std::vector<RayTracing::HleRefitGeometry> none;
		ReportRefitOutsideGuest(RayTracing::DropHleRefitOutsideGuest(refit, sizes, none, ReadRefitBlasSize));
		for (size_t i = 0; i < refit.size(); i++) {
			if (refit[i] != 0 && sizes[i] > RayTracing::HleRefitBlasHeaderBytes) {
				(void)cache.ObtainBuffer(refit[i], sizes[i], true);
			}
		}
		context.PrepareBda(true, false);
		auto& command = context.GetCommandScheduler().Current();
		command.EndRendering();
		context.GetRayTracing()->RecordHleRefit(command.Handle(), refit, {}, 2, true);
	});
}

void CommandProcessor::EmitHleRefit() {
	const auto     generation    = Psr::LibraryGeneration();
#if defined(ANYPS5_KYTY)
	const uint32_t header_layout = generation == Psr::Generation::Base252 || generation == Psr::Generation::Base148
	                                   ? 1u
	                               : generation == Psr::Generation::Base515 ? 2u
	                                                                        : 0u;
#else
	const uint32_t header_layout = generation == Psr::Generation::Base252   ? 1u
	                               : generation == Psr::Generation::Base515 ? 2u
	                                                                        : 0u;
#endif
	if (header_layout == 2) {
		EmitHleRefitArgs();
	}
	EmitHostCommand([this, blases = m_hle_refit_blases, refs = m_hle_refit_geometries, header_layout] {
		auto&                                          context = m_renderer;
		auto&                                          cache   = context.GetBufferCache();
		std::vector<RayTracing::HleRefitGeometry> geometries;
		geometries.reserve(refs.size());
		uint64_t last_vertices = 0;
		for (const auto& ref: refs) {
			if (ref.blas == 0 || ref.table == 0) {
				continue;
			}
			std::array<uint32_t, 30> entry {};
			ReadGuestAroundGpuWrites(ref.table + uint64_t {ref.geometry} * 120u, entry.data(),
			                         sizeof(entry));
			RayTracing::HleRefitGeometry geometry {.blas = ref.blas, .geometry = ref.geometry};
			std::memcpy(geometry.vsharp.data(), &entry[2], sizeof(geometry.vsharp));
			std::memcpy(geometry.matrix.data(), &entry[14], sizeof(geometry.matrix));
			ShaderBufferResource vsharp;
			std::memcpy(vsharp.fields, geometry.vsharp.data(), sizeof(vsharp.fields));
			if (vsharp.Base48() != 0 && vsharp.Base48() != last_vertices && vsharp.GetSize() != 0) {
				if (!cache.PrepareDeviceAddressRead(vsharp.Base48(), vsharp.GetSize())) {
					continue;
				}
				last_vertices = vsharp.Base48();
			}
			geometries.push_back(geometry);
		}
		auto                  refit = blases;
		std::vector<uint32_t> sizes;
		ReportRefitOutsideGuest(RayTracing::DropHleRefitOutsideGuest(refit, sizes, geometries, ReadRefitBlasSize));
		constexpr uint32_t header_bytes = RayTracing::HleRefitBlasHeaderBytes;
		for (size_t i = 0; i < refit.size(); i++) {
			const auto blas = refit[i];
			const auto size = sizes[i];
			if (blas != 0 && size > header_bytes && header_layout == 2) {
				(void)cache.ObtainBuffer(blas, size, true);
			} else if (blas != 0 && size > header_bytes) {
				(void)cache.PrepareDeviceAddressRead(blas, header_bytes);
				(void)cache.ObtainBuffer(blas + header_bytes, size - header_bytes, true);
			}
		}
		context.PrepareBda(true, false);
		auto& command = context.GetCommandScheduler().Current();
		command.EndRendering();
		context.GetRayTracing()->RecordHleRefit(command.Handle(), refit, geometries, header_layout);
	});
}

bool CommandProcessor::HleTlasBuild::Native() const {
	return count == 0 || (input != 0 && stride >= 96 && stride % 16 == 0 &&
	                      count <= RayTracing::HleTlasMaxInstances);
}

bool CommandProcessor::HleTlasDispatch(uint64_t data_addr) {
	if (data_addr == 0 || !m_renderer.GetRayTracing()->Enabled()) {
		return false;
	}
	using Psr::Shader;
	const auto shader = ComputePsrShader(data_addr);
	if (shader == Shader::None) {
		return false;
	}
	if (!Psr::IsTopLevelBuildShader(shader)) {
		if (m_hle_tlas_window) {
			Psr::StopReplacement("a top level build runs a pass the native build does not replace",
			                     ShaderRegisteredName(data_addr));
			if (shader == Shader::MortonCaptureMemory) {
				m_hle_tlas_window = false;
				if (!m_hle_tlas_observe) {
					m_hle_tlas_builds.resize(m_hle_tlas_finished);
					EmitHleTlas();
				}
			}
		}
		return false;
	}
	const auto& user = GetShCtx().GetCs().cs_user_sgpr.value;
	const auto  list = [&] {
		const bool base252 = Psr::LibraryGeneration() == Psr::Generation::Base252;
		const auto v       = base252 ? 4u : 6u;
		return HleTlasBuild {.input  = uint64_t {user[v]} | (uint64_t {user[v + 1] & 0xffffu} << 32u),
		                     .stride = (user[v + 1] >> 16u) & 0x3fffu,
		                     .count  = user[v + 2],
		                     .format = base252 ? RayTracing::HleTlasFormat::Base252
		                                       : RayTracing::HleTlasFormat::Base460};
	};
	if (m_guest_tlas_builds != 0) {
		if (shader == Shader::MortonComputeBoundingBoxForInstances) {
			m_guest_tlas_builds++;
		} else if (shader == Shader::CollapseToQbvhSimpleTop) {
			m_guest_tlas_builds--;
		}
		return false;
	}
	if (!m_hle_tlas_window) {
		if (shader != Shader::MortonComputeBoundingBoxForInstances || !Psr::ReplacementEnabled()) {
			return false;
		}
		if (!list().Native()) {
			m_guest_tlas_builds = 1;
			return false;
		}
		m_hle_tlas_window = true;
		m_hle_tlas_builds.clear();
		m_hle_tlas_finished = 0;
		m_hle_tlas_observe = !m_hle_tlas_known;
	}
	if (m_hle_tlas_observe) {
		if (shader == Shader::MortonComputeBoundingBoxForInstances) {
			m_hle_tlas_builds.emplace_back();
		} else if (shader == Shader::CollapseToQbvhSimpleTop && ++m_hle_tlas_finished >= m_hle_tlas_builds.size()) {
			m_hle_tlas_window = false;
			m_hle_tlas_known  = Psr::ReplacementEnabled();
		}
		return false;
	}
	if (shader == Shader::MortonComputeBoundingBoxForInstances) {
		m_hle_tlas_builds.push_back(list());
	}
	if (shader != Shader::CollapseToQbvhSimpleTop) {
		return true;
	}
	if (m_hle_tlas_finished < m_hle_tlas_builds.size()) {
		m_hle_tlas_builds[m_hle_tlas_finished].header =
		    (uint64_t {user[0]} | (uint64_t {user[1]} << 32u)) & 0xffffffffffffull;
	}
	if (++m_hle_tlas_finished < m_hle_tlas_builds.size()) {
		return true;
	}
	m_hle_tlas_window = false;
	EmitHleTlas();
	return true;
}

void CommandProcessor::EmitHleTlas() {
	EmitHostCommand([this, builds = m_hle_tlas_builds] {
		auto&                 context = m_renderer;
		auto&                 cache   = context.GetBufferCache();
		std::vector<uint8_t>  list;
		std::vector<uint64_t> blases;
		std::vector<uint8_t>  recorded(builds.size());
		for (size_t index = 0; index < builds.size(); index++) {
			const auto& build = builds[index];
			if (build.header == 0 || !build.Native()) {
				Common::HackNotice::Report(
				    "native TLAS build skipped for an instance list it cannot read",
				    "CommandProcessor::HleTlasDispatch", 0, build.stride,
				    fmt::format("{} instances of {} bytes at 0x{:x}, TLAS at 0x{:x}", build.count,
				                build.stride, build.input, build.header));
				continue;
			}
			std::array<uint32_t, 32> header {};
			const auto size = RayTracing::HleTlasHeader(build.count, build.format, header);
			if (!GuestRange {build.header, size}.Valid()) {
				Common::HackNotice::Report("native TLAS build skipped for a TLAS outside guest memory",
				                           "CommandProcessor::HleTlasDispatch", 0, build.stride,
				                           fmt::format("TLAS at 0x{:x}, {} bytes", build.header, size));
				continue;
			}
			if (build.count != 0) {
				const auto bytes = uint64_t {build.stride} * build.count;
				if (!cache.PrepareDeviceAddressRead(build.input, bytes)) {
					continue;
				}
				list.resize(bytes);
				ReadGuestAroundGpuWrites(build.input, list.data(), bytes);
				blases.clear();
				for (uint32_t i = 0; i < build.count; i++) {
					uint64_t blas = 0;
					std::memcpy(&blas, list.data() + uint64_t {i} * build.stride, sizeof(blas));
					if (blas != 0) {
						blases.push_back(blas);
					}
				}
				std::ranges::sort(blases);
				blases.erase(std::unique(blases.begin(), blases.end()), blases.end());
				constexpr uint64_t BlasReadSize = 256 + 64;
				for (const auto blas: blases) {
					if (context.IsMapped(blas, BlasReadSize) && !cache.IsRegionGpuModified(blas, BlasReadSize)) {
						(void)cache.PrepareDeviceAddressRead(blas, BlasReadSize);
					}
				}
			}
			Libs::LibKernel::Memory::WriteBacking(build.header, header.data(), sizeof(header));
			cache.InvalidateUnlessGpuModified(build.header, sizeof(header));
			(void)cache.ObtainBuffer(build.header + 256u, size - 256u, true);
			context.NoteHleTlas(build.header, size, build.count != 0 ? blases : std::vector<uint64_t> {});
			recorded[index] = 1;
		}
		context.PrepareBda(true, false);
		auto& command = context.GetCommandScheduler().Current();
		command.EndRendering();
		for (size_t index = 0; index < builds.size(); index++) {
			if (recorded[index] != 0) {
				const auto& build = builds[index];
				context.GetRayTracing()->RecordHleTlas(command.Handle(), build.input,
				                                            build.count, build.stride,
				                                            build.header, build.format);
			}
		}
	});
}

void CommandProcessor::PsrBlasDispatch(uint64_t data_addr) {
	if (data_addr == 0 || !m_renderer.GetRayTracing()->Enabled()) {
		return;
	}
	using Psr::Shader;
	const auto shader = ComputePsrShader(data_addr);
	if (shader == Shader::Trinity) {
		Psr::StopReplacement("the game builds kTrinity BVHs", ShaderRegisteredName(data_addr));
	}
	const auto generation = Psr::LibraryGeneration();
	if (!Psr::ReplacementEnabled() ||
	    (generation != Psr::Generation::Base460 && generation != Psr::Generation::Base252 &&
	     generation != Psr::Generation::Base148)) {
		m_psr_blas_builds.clear();
		m_psr_captured.clear();
		return;
	}
	const bool base252 = generation == Psr::Generation::Base252;
	const bool base148 = generation == Psr::Generation::Base148;
	auto       pass    = shader;
	if (base252 && shader == Shader::CollapseToQbvhSimpleTop && !m_hle_tlas_window && m_guest_tlas_builds == 0) {
		pass = Shader::CollapseToQbvhSimpleBottom;
	}
	if (pass != Shader::MortonComputeBoundingBoxForVertices && pass != Shader::MortonComputeCodeForMesh &&
	    pass != Shader::CollapseToQbvhSimpleBottom && pass != Shader::MortonCaptureMemory &&
	    pass != Shader::DecompressHeaderBase && pass != Shader::DecompressComputeFanNode) {
		return;
	}
	const auto& user = GetShCtx().GetCs().cs_user_sgpr.value;
	const auto address = [&](uint32_t sgpr) {
		return uint64_t {user[sgpr]} | (uint64_t {user[sgpr + 1] & 0xffffu} << 32u);
	};
	if (shader == Shader::MortonCaptureMemory) {
		if (m_psr_captured.size() >= 4096) {
			m_psr_captured.clear();
		}
		m_psr_captured[address(0)] = user[2];
		if (!base148) {
			EmitHostCommand([this, memory = address(0), size = user[2]] { m_renderer.NoteGuestBvhMemory(memory, size); });
		}
		return;
	}
	if (shader == Shader::DecompressHeaderBase || shader == Shader::DecompressComputeFanNode) {
		const auto blas = base148 ? uint64_t {user[7]} | (uint64_t {user[2] & 0xffffu} << 32u)
		                          : uint64_t {user[6]} | (uint64_t {user[1] & 0xffffu} << 32u);
		const auto it   = m_psr_captured.find(blas);
		if (it == m_psr_captured.end()) {
			return;
		}
		if (shader == Shader::DecompressComputeFanNode) {
			m_psr_decompressed      = blas;
			m_psr_decompressed_size = it->second;
		}
		return;
	}
	constexpr uint32_t GeometryStride = 120;
	constexpr uint32_t MaxGeometries  = 1024;
	constexpr size_t MaxBuilds = 4096;
	if (shader == Shader::MortonComputeBoundingBoxForVertices || shader == Shader::MortonComputeCodeForMesh) {
		const auto table = address(0);
		const auto count = user[2];
		const auto geometry = shader == Shader::MortonComputeBoundingBoxForVertices ? user[9]
		                      : base252 || base148                                  ? user[6] >> 18u
		                                                                            : user[7];
		if (m_psr_blas_builds.size() >= MaxBuilds) {
			m_psr_blas_builds.clear();
		}
		auto& state = m_psr_blas_builds[address(4)];
		if (table != state.table || count != state.seen.size()) {
			state.table = count != 0 && count <= MaxGeometries ? table : 0;
			state.seen.assign(state.table != 0 ? count : 0, 0);
		}
		if (geometry < state.seen.size()) {
			state.seen[geometry] = 1;
		}
		return;
	}
	const auto blas = address(0);
	const auto primitives = base252 || base148 ? user[14] + 1u : user[14];
	const auto it         = m_psr_blas_builds.find(address(2));
	if (it == m_psr_blas_builds.end()) {
		return;
	}
	const auto table = it->second.table;
	const auto seen  = std::move(it->second.seen);
	m_psr_blas_builds.erase(it);
	if (blas == 0 || table == 0 || std::ranges::find(seen, 0) != seen.end()) {
		return;
	}
	RayTracing::GuestBlasBuild build {.blas_va = blas, .header_layout = base148 ? 2u : base252 ? 1u : 0u};
	std::vector<uint32_t>            entries(seen.size() * GeometryStride / 4);
	ReadGuestAroundGpuWrites(table, entries.data(), entries.size() * 4);
	uint64_t triangles = 0;
	for (size_t g = 0; g < seen.size(); g++) {
		const auto* entry = &entries[g * GeometryStride / 4];
		if (entry[0] != 1) {
			return;
		}
		build.geometries.emplace_back(entry[8], (entry[1] & 1u) != 0);
		triangles += entry[8];
	}
	if (triangles == 0 || primitives < triangles) {
		return;
	}
	m_psr_blas_built = std::move(build);
}

void CommandProcessor::GuestTlasDispatch(uint64_t data_addr) {
	if (data_addr == 0 || !m_renderer.GetRayTracing()->Enabled()) {
		return;
	}
	using Psr::Shader;
	const auto shader = ComputePsrShader(data_addr);
	if (shader == Shader::MortonComputeBoundingBoxForInstances) {
		EmitHostCommand([this] { m_renderer.ForgetHleTlas(0); });
	} else if (shader == Shader::CollapseToQbvhSimpleTop) {
		const auto& user = GetShCtx().GetCs().cs_user_sgpr.value;
		EmitHostCommand([this, header = (uint64_t {user[0]} | (uint64_t {user[1]} << 32u)) & 0xffffffffffffull] {
			m_renderer.ForgetHleTlas(header);
		});
	}
}

void CommandProcessor::EmitPsrBlasBuilt() {
	if (m_psr_decompressed != 0) {
		EmitHostCommand([this, blas = m_psr_decompressed, size = m_psr_decompressed_size] {
			m_renderer.GetRayTracing()->NoteGuestBlasDecompressed(blas, size);
		});
		m_psr_decompressed = 0;
	}
	if (m_psr_blas_built.blas_va == 0) {
		return;
	}
	EmitHostCommand([this, build = std::move(m_psr_blas_built)] {
		m_renderer.GetRayTracing()->NoteGuestBlasBuilds(std::span(&build, 1));
	});
	m_psr_blas_built = {};
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	PsrBlasDispatch(GetShCtx().GetCs().cs_regs.data_addr);
	if (HleTlasDispatch(GetShCtx().GetCs().cs_regs.data_addr) ||
	    HleBlasDispatch(GetShCtx().GetCs().cs_regs.data_addr)) {
		return;
	}
	GuestTlasDispatch(GetShCtx().GetCs().cs_regs.data_addr);
	GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	auto operation     = MakeOperation(GpuOperationKind::DispatchDirect);
	operation.dispatch = {.thread_group_x = thread_group_x,
	                      .thread_group_y = thread_group_y,
	                      .thread_group_z = thread_group_z,
	                      .mode           = mode};
	Emit(std::move(operation));
	EmitPsrBlasBuilt();
}

void CommandProcessor::ExecuteDispatchDirect(const GpuOperation& operation, uint32_t thread_group_x,
                                             uint32_t thread_group_y, uint32_t thread_group_z,
                                             uint32_t mode) {
	NoteRecordedWork();

	m_renderer.GetRenderExecutor().DispatchDirect(CurrentBuffer(), thread_group_x,
	                                              thread_group_y, thread_group_z, mode,
	                                              IsAsyncComputeQueue());
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	EXIT_NOT_IMPLEMENTED(IsAsyncComputeQueue());
	EmitHostCommand([this, request] {
		m_renderer.GetRenderExecutor().ProcessPssr(CurrentBuffer(), request);
	});
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	PsrBlasDispatch(GetShCtx().GetCs().cs_regs.data_addr);
	if (HleTlasDispatch(GetShCtx().GetCs().cs_regs.data_addr) ||
	    HleBlasDispatch(GetShCtx().GetCs().cs_regs.data_addr)) {
		return;
	}
	GuestTlasDispatch(GetShCtx().GetCs().cs_regs.data_addr);
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	GetShCtx().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	auto operation     = MakeOperation(GpuOperationKind::DispatchIndirect);
	operation.dispatch = {.mode = mode, .args_addr = args_addr};
	Emit(std::move(operation));
	EmitPsrBlasBuilt();
}

void CommandProcessor::ExecuteDispatchIndirect(const GpuOperation& operation) {
	const auto& dispatch = operation.dispatch;
	NoteRecordedWork();
	if ((dispatch.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		m_renderer.GetRenderExecutor().DispatchIndirectThreads(CurrentBuffer(), dispatch.args_addr,
		                                                       dispatch.mode,
		                                                       IsAsyncComputeQueue());
		return;
	}
	m_renderer.GetRenderExecutor().DispatchIndirect(CurrentBuffer(),
	                                                dispatch.args_addr, dispatch.mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	auto operation      = MakeOperation(GpuOperationKind::DrawAuto);
	operation.draw_auto = args;
	Emit(std::move(operation));
}

void CommandProcessor::ExecuteDrawAuto(const GpuOperation& operation, DrawAutoArgs args) {
	NoteRecordedWork();
	if (args.instance_count == 0) {
		args.instance_count = ResolveNumInstances(operation);
	}
	m_renderer.GetRenderExecutor().DrawAuto(CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(const uint64_t* label) {
	EmitCallback([this, label] {
		GetScheduler().Flush();
		m_renderer.GetVideoOut().WaitFlipDone(label);
	});
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id,
                                  bool gl2_writeback) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	if (gl2_writeback) {
		WriteBackImages();
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	const auto sequence = EmitCallback([=, this] {
		ExecuteReleaseMem(data_sel, interrupt, address, data, interrupt_context_id);
	});
	const auto destination = reinterpret_cast<uint64_t>(address);
	switch (data_sel) {
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			NoteGuestWrite(sequence, destination, sizeof(value), &value);
			break;
		}
		case 2: NoteGuestWrite(sequence, destination, sizeof(data), &data); break;
		case 3: NoteGuestWrite(sequence, destination, sizeof(data)); break;
		case 5:
			NoteGuestWrite(sequence, destination,
			               uint64_t {static_cast<uint32_t>(data) >> 16u} * sizeof(uint32_t));
			break;
		default: break;
	}
}

#if defined(ANYPS5_KYTY)
bool CommandProcessor::LabelsAfterDumps() {
	static const bool enabled = [] {
		const char*            value = std::getenv("KYTY_LABELS_AFTER_DUMPS");
		const std::string_view text {value == nullptr ? "" : value};
		if (!text.empty() && text != "0" && text != "1") {
			EXIT("invalid KYTY_LABELS_AFTER_DUMPS: %s (0 or 1)\n", value);
		}
		return text != "0";
	}();
	return enabled;
}

namespace {
constexpr size_t LabelWaitBuckets = 8;
constexpr std::array<uint64_t, LabelWaitBuckets - 1> LabelWaitLimitsNs {100'000,   250'000,   500'000,  1'000'000,
                                                                       2'000'000, 4'000'000, 8'000'000};

struct LabelOrderStats {
	std::atomic<uint64_t> around {0};
	std::atomic<uint64_t> drains {0};
	std::atomic<uint64_t> drain_ns {0};
	std::atomic<int64_t>  last_line {0};
	std::atomic<uint64_t> deferred {0};
	std::atomic<uint64_t> immediate {0};
	std::atomic<uint64_t> covered {0};
	std::atomic<uint64_t> block_only {0};
	std::array<std::atomic<uint64_t>, LabelOrder::ReasonCount> reasons {};
	std::array<std::atomic<uint64_t>, LabelWaitBuckets>        waits {};
	std::array<std::atomic<uint64_t>, LabelWaitBuckets>        wait_ns {};
	std::mutex                                                 addresses_mutex;
	std::unordered_set<uint64_t>                               addresses;
	std::atomic<uint64_t> own_tick_block {0};
	std::atomic<uint64_t> own_tick_page {0};
	std::atomic<uint64_t> settles {0};
	std::atomic<uint64_t> verify_labels {0};
	std::atomic<uint64_t> verify_unread {0};
	std::atomic<uint64_t> verify_changed {0};
	std::atomic<uint64_t> verify_lost {0};
	std::atomic<uint64_t> verify_settles {0};
	std::atomic<uint64_t> verify_logged {0};
};
LabelOrderStats g_label_stats;

enum class InPlaceLabelDefer : uint8_t { Off, On, Verify };

InPlaceLabelDefer InPlaceLabelDeferMode() {
	static const auto mode = [] {
		const char*            value = std::getenv("KYTY_IN_PLACE_LABEL_DEFER");
		const std::string_view text {value == nullptr ? "" : value};
		if (text.empty() || text == "0") {
			return InPlaceLabelDefer::Off;
		}
		if (text == "1") {
			return InPlaceLabelDefer::On;
		}
		if (text != "verify") {
			EXIT("invalid KYTY_IN_PLACE_LABEL_DEFER: %s (0, 1 or verify)\n", value);
		}
		return InPlaceLabelDefer::Verify;
	}();
	return mode;
}

void NoteLabelViolation(const char* what, uint64_t address, uint64_t size, uint64_t before, uint64_t after) {
	if (g_label_stats.verify_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-label-drains] verify: label 0x{:x}+{}: {} (0x{:x} then 0x{:x})\n", address, size, what, before, after));
	}
}

bool InPlaceLabelHeldBack(CommandScheduler& scheduler, uint64_t address, uint64_t size) {
	const auto natural = scheduler.CurrentTick() - InPlaceMemory::LabelVerifyDrains();
	switch (InPlaceMemory::MeetsDeferredLabel(address, size, natural, true)) {
		case InPlaceMemory::LabelMeet::None: break;
		case InPlaceMemory::LabelMeet::WouldBePending:
			g_label_stats.verify_settles.fetch_add(1, std::memory_order_relaxed);
			break;
		case InPlaceMemory::LabelMeet::Pending: return true;
	}
	return false;
}

void SettleInPlaceLabels(CommandScheduler& scheduler, uint64_t address, uint64_t size) {
	if (InPlaceLabelHeldBack(scheduler, address, size)) {
		const auto tick = scheduler.CurrentTick();
		scheduler.Wait(tick);
		scheduler.WaitPriorityOperations(tick);
		g_label_stats.settles.fetch_add(1, std::memory_order_relaxed);
	}
}

void NoteLabelAddress(uint64_t address) {
	std::lock_guard lock(g_label_stats.addresses_mutex);
	g_label_stats.addresses.insert(address);
}

void NoteLabelDrain(const LabelOrder::Facts& facts, uint64_t waited_ns) {
	auto& stats = g_label_stats;
	stats.reasons[static_cast<size_t>(LabelOrder::DrainReason(facts))].fetch_add(1, std::memory_order_relaxed);
	if (facts.in_place_bytes) {
		stats.covered.fetch_add(1, std::memory_order_relaxed);
	} else if (facts.in_place_block) {
		stats.block_only.fetch_add(1, std::memory_order_relaxed);
	}
	const auto bucket = static_cast<size_t>(
	    std::upper_bound(LabelWaitLimitsNs.begin(), LabelWaitLimitsNs.end(), waited_ns) - LabelWaitLimitsNs.begin());
	stats.waits[bucket].fetch_add(1, std::memory_order_relaxed);
	stats.wait_ns[bucket].fetch_add(waited_ns, std::memory_order_relaxed);
}

int64_t LabelClockNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}
}

void CommandProcessor::LogLabelSummary() {
	auto&      stats = g_label_stats;
	const auto now   = LabelClockNs();
	auto       last  = stats.last_line.load(std::memory_order_relaxed);
	if (last == 0) {
		stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000 || !stats.last_line.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto take      = [](std::atomic<uint64_t>& value) { return value.exchange(0, std::memory_order_relaxed); };
	const auto around    = take(stats.around);
	const auto drains    = take(stats.drains);
	const auto drain_ns  = take(stats.drain_ns);
	const auto deferred  = take(stats.deferred);
	const auto immediate = take(stats.immediate);
	const auto covered   = take(stats.covered);
	const auto block     = take(stats.block_only);
	const auto over      = InPlaceMemory::TakeGpuWritesOverPending();
	std::array<uint64_t, LabelOrder::ReasonCount> reasons {};
	std::array<uint64_t, LabelWaitBuckets>        waits {};
	std::array<double, LabelWaitBuckets>          wait_ms {};
	for (size_t i = 0; i < reasons.size(); i++) {
		reasons[i] = take(stats.reasons[i]);
	}
	for (size_t i = 0; i < waits.size(); i++) {
		waits[i]   = take(stats.waits[i]);
		wait_ms[i] = static_cast<double>(take(stats.wait_ns[i])) / 1e6;
	}
	size_t addresses = 0;
	{
		std::lock_guard lock(stats.addresses_mutex);
		addresses = stats.addresses.size();
		stats.addresses.clear();
	}
	const auto own_block      = take(stats.own_tick_block);
	const auto own_page       = take(stats.own_tick_page);
	const auto settles        = take(stats.settles);
	const auto verify_labels  = take(stats.verify_labels);
	const auto verify_unread  = take(stats.verify_unread);
	const auto verify_changed = take(stats.verify_changed);
	const auto verify_lost    = take(stats.verify_lost);
	const auto verify_settles = take(stats.verify_settles);
	const auto verify_stops   = InPlaceMemory::TakeGpuWritesOverLabels();
	if (around + drains + own_block + own_page == 0) {
		return;
	}
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-labels] summary of the last {:.0f} s: {} RELEASE_MEM labels on a page the GPU wrote elsewhere deferred "
	    "behind the pending labels; the execute thread drained the GPU {} times to keep a label in order ({:.3f} s)\n",
	    static_cast<double>(now - last) / 1e9, around, drains, static_cast<double>(drain_ns) / 1e9));
	const auto reason = [&reasons](LabelOrder::Reason id) { return reasons[static_cast<size_t>(id)]; };
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-label-drains] summary of the last {:.0f} s: {} RELEASE_MEM labels at {} addresses: {} deferred to the end "
	    "of a tick, {} written at once, {} written after a drain; drained because: {} not deferable, {} own bytes under an "
	    "in-place GPU write in flight, {} own bytes GPU-written in the cache, {} around writes off, {} no backing view, {} "
	    "only the 64 KiB block under an in-place GPU write in flight, {} on a GPU-written page with no label pending, {} "
	    "other; of the drained labels {} lie under an in-place GPU write in flight and {} only share its block; drain "
	    "waits under 0.1/0.25/0.5/1/2/4/8 ms and above: {} ({:.1f} ms, in the same order {:.1f}); {} in-place GPU writes "
	    "noted over a pending completion write; KYTY_IN_PLACE_LABEL_DEFER {}: {} labels that only share the block of an "
	    "in-place write and {} on a GPU-written page with no label pending deferred to their own tick, {} drains before a "
	    "later write or decision read of the bytes of such a label; verify: {} drained labels the rule would defer ({} not compared), {} "
	    "whose bytes changed while the GPU finished, {} whose value was not found after the write, {} later writes or decision reads "
	    "that would drain first, {} later in-place GPU writes that would stop the emulator\n",
	    static_cast<double>(now - last) / 1e9, deferred + immediate + drains, addresses, deferred, immediate, drains,
	    reason(LabelOrder::Reason::NotDeferable), reason(LabelOrder::Reason::InPlaceBytes),
	    reason(LabelOrder::Reason::CacheBytes), reason(LabelOrder::Reason::AroundOff),
	    reason(LabelOrder::Reason::NoBacking), reason(LabelOrder::Reason::InPlaceBlock),
	    reason(LabelOrder::Reason::NoLabelPending), reason(LabelOrder::Reason::Other), covered, block,
	    fmt::join(waits, "/"), static_cast<double>(drain_ns) / 1e6, fmt::join(wait_ms, "/"), over,
	    InPlaceLabelDeferMode() == InPlaceLabelDefer::On ? "on" : InPlaceLabelDeferMode() == InPlaceLabelDefer::Verify ? "verify" : "off",
	    own_block, own_page, settles, verify_labels, verify_unread, verify_changed, verify_lost, verify_settles,
	    verify_stops));
}

bool CommandProcessor::LabelsAroundGpuWrites() {
	static const bool enabled = [] {
		const char*            value = std::getenv("KYTY_LABELS_AROUND_GPU_WRITES");
		const std::string_view text {value == nullptr ? "" : value};
		if (!text.empty() && text != "0" && text != "1") {
			EXIT("invalid KYTY_LABELS_AROUND_GPU_WRITES: %s (0 or 1)\n", value);
		}
		return text != "0";
	}();
	return enabled;
}

#endif
void CommandProcessor::ExecuteReleaseMem(uint32_t data_sel, bool interrupt, void* address,
                                         uint64_t data, uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();

	const auto write_label = [this, address](const void* value, uint64_t size) {
		const auto vaddr   = reinterpret_cast<uint64_t>(address);
		auto&      buffers = m_renderer.GetBufferCache();
		if (!GuestRange {vaddr, size}.Valid() || !buffers.IsRegionGpuModified(vaddr, size) ||
		    buffers.HasGpuDirtyBytes(vaddr, size) ||
		    !LibKernel::Memory::TryWriteBacking(vaddr, value, size)) {
			m_renderer.PrepareGpuWrite(address, size);
			std::memcpy(address, value, size);
			LibKernel::Memory::NoteEmulatorWrite(vaddr, size);
		}
	};
#if defined(ANYPS5_KYTY)
	const auto labels_pending = [this] {
		return (m_labels_after_dumps ? s_deferred_labels : m_deferred_labels)
		    .load(std::memory_order_acquire);
	};
#else
	const auto labels_pending = [this] { return m_deferred_labels.load(std::memory_order_acquire); };
#endif
	const bool at_completion = InPlaceMemory::Live();
#if defined(ANYPS5_KYTY)
	bool drained      = false;
	bool verify_write = false;
#endif
	if (data_sel != 0 &&
	    (at_completion || m_renderer.GetOcclusionQueries().HasPendingDumps() || labels_pending() != 0)) {
		const auto     vaddr = reinterpret_cast<uint64_t>(address);
		const uint64_t size  = data_sel == 1 ? sizeof(uint32_t) : sizeof(uint64_t);
		auto&          buffers   = m_renderer.GetBufferCache();
		const bool     deferable = data_sel <= 3 && GuestRange {vaddr, size}.Valid();
		const bool     gpu_page  = deferable && buffers.IsRegionGpuModified(vaddr, size);
		bool           plain     = deferable && !gpu_page;
#if defined(ANYPS5_KYTY)
		bool around = gpu_page && m_labels_around_gpu_writes && labels_pending() != 0 &&
		              !buffers.HasGpuDirtyBytes(vaddr, size) &&
		              LibKernel::Memory::FindBackingPointer(vaddr, size) != nullptr;
		LabelOrder::Facts facts {.deferable = deferable};
		if (deferable) {
			facts.around_enabled = m_labels_around_gpu_writes;
		}
		const auto gather = [&] {
			facts.tracked_page   = buffers.IsRegionTrackedGpuModified(vaddr, size);
			facts.tracked_bytes  = buffers.HasTrackedGpuDirtyBytes(vaddr, size);
			facts.in_place_block = buffers.InPlaceGpuWriteTick(vaddr, size) != 0;
			facts.in_place_bytes = facts.in_place_block && buffers.InPlaceGpuWriteBytesTick(vaddr, size) != 0;
			facts.backing        = LibKernel::Memory::FindBackingPointer(vaddr, size) != nullptr;
			facts.label_pending  = labels_pending() != 0;
		};
		const auto mode     = at_completion ? InPlaceLabelDeferMode() : InPlaceLabelDefer::Off;
		bool       gathered = false;
		bool       would    = false;
		uint64_t   own_tick = 0;
		if (mode != InPlaceLabelDefer::Off && gpu_page && !around) {
			gather();
			gathered         = true;
			const auto route = LabelOrder::AtCompletion(facts);
			would            = route != LabelOrder::Route::Drain;
			if (would && mode == InPlaceLabelDefer::On) {
				plain  = route == LabelOrder::Route::Defer;
				around = route == LabelOrder::Route::DeferAround;
				(plain ? g_label_stats.own_tick_block : g_label_stats.own_tick_page).fetch_add(1, std::memory_order_relaxed);
				own_tick = InPlaceMemory::NoteDeferredLabel(vaddr, size);
			}
		}
#else
		const bool around = false;
#endif
		if (deferable && (plain || around)) {
			if (plain) {
				m_renderer.PrepareGpuWrite(address, size);
			}
#if defined(ANYPS5_KYTY)
			else {
				g_label_stats.around.fetch_add(1, std::memory_order_relaxed);
			}
#endif
			if (at_completion) {
				InPlaceMemory::NoteLabelAtCompletion();
			}
#if defined(ANYPS5_KYTY)
			g_label_stats.deferred.fetch_add(1, std::memory_order_relaxed);
			NoteLabelAddress(vaddr);
#endif
			m_deferred_labels.fetch_add(1, std::memory_order_relaxed);
#if defined(ANYPS5_KYTY)
			s_deferred_labels.fetch_add(1, std::memory_order_relaxed);
#endif
#if defined(ANYPS5_KYTY)
			auto write_deferred = [this, address, data, data_sel, own_tick] {
#else
			auto write_deferred = [this, address, data, data_sel] {
#endif
				if (data_sel == 1) {
					const auto value = static_cast<uint32_t>(data);
					m_renderer.WriteFromCompletion(address, &value, sizeof(value));
				} else {
					const uint64_t value = data_sel == 3 ? Sync::ReadReferenceClock() : data;
					m_renderer.WriteFromCompletion(address, &value, sizeof(value));
				}
#if defined(ANYPS5_KYTY)
				if (own_tick != 0) {
					InPlaceMemory::ForgetDeferredLabel(own_tick);
				}
#endif
				m_deferred_labels.fetch_sub(1, std::memory_order_release);
#if defined(ANYPS5_KYTY)
				s_deferred_labels.fetch_sub(1, std::memory_order_release);
#endif
			};
#if defined(ANYPS5_KYTY)
			if (m_labels_after_dumps && !at_completion) {
				GetScheduler().DeferPriorityOperationAfter(
				    std::max<uint64_t>(m_renderer.GetOcclusionQueries().LastPendingDumpTick(), 1),
				    std::move(write_deferred), vaddr, size);
			} else {
				GetScheduler().DeferPriorityOperation(std::move(write_deferred), vaddr, size);
			}
#else
			GetScheduler().DeferPriorityOperation(std::move(write_deferred), vaddr, size);
#endif
			if (interrupt) {
				DeferEopInterrupt(interrupt_context_id);
			}
			return;
		}
		if (labels_pending() != 0 || at_completion) {
			auto&      scheduler = GetScheduler();
			const auto tick      = scheduler.CurrentTick();
#if defined(ANYPS5_KYTY)
			if (deferable && !gathered) {
				gather();
			}
			uint64_t   before  = 0;
			const bool read    = would && !scheduler.HasPendingGuestOperations(vaddr, size) &&
			                     LibKernel::Memory::TryReadBacking(vaddr, &before, size);
			const auto natural = tick - InPlaceMemory::LabelVerifyDrains();
			const auto started = LabelClockNs();
#endif
			scheduler.Wait(tick);
			scheduler.WaitPriorityOperations(tick);
#if defined(ANYPS5_KYTY)
			const auto waited = static_cast<uint64_t>(LabelClockNs() - started);
			if (would) {
				g_label_stats.verify_labels.fetch_add(1, std::memory_order_relaxed);
				uint64_t after = 0;
				if (!read || !LibKernel::Memory::TryReadBacking(vaddr, &after, size)) {
					g_label_stats.verify_unread.fetch_add(1, std::memory_order_relaxed);
				} else if (after != before) {
					g_label_stats.verify_changed.fetch_add(1, std::memory_order_relaxed);
					NoteLabelViolation("its bytes changed while the GPU finished the tick", vaddr, size, before, after);
				}
				InPlaceMemory::NoteLabelVerifyDrain();
				(void)InPlaceMemory::NoteDeferredLabel(vaddr, size, natural + 1);
				verify_write = read;
			}
			g_label_stats.drains.fetch_add(1, std::memory_order_relaxed);
			g_label_stats.drain_ns.fetch_add(waited, std::memory_order_relaxed);
			NoteLabelDrain(facts, waited);
			NoteLabelAddress(vaddr);
			drained = true;
#endif
		}
	}
#if defined(ANYPS5_KYTY)
	const auto verify_written = [&] {
		if (!verify_write || data_sel == 3) {
			return;
		}
		const auto     vaddr = reinterpret_cast<uint64_t>(address);
		const uint64_t size  = data_sel == 1 ? sizeof(uint32_t) : sizeof(uint64_t);
		const uint64_t value = data_sel == 1 ? static_cast<uint32_t>(data) : data;
		uint64_t       found = 0;
		if (!LibKernel::Memory::TryReadBacking(vaddr, &found, size)) {
			g_label_stats.verify_unread.fetch_add(1, std::memory_order_relaxed);
		} else if (found != value) {
			g_label_stats.verify_lost.fetch_add(1, std::memory_order_relaxed);
			NoteLabelViolation("the value written after the drain is not in memory", vaddr, size, value, found);
		}
	};
	if (data_sel != 0 && data_sel != 5 && !drained) {
		g_label_stats.immediate.fetch_add(1, std::memory_order_relaxed);
		NoteLabelAddress(reinterpret_cast<uint64_t>(address));
	}
#endif
	switch (data_sel) {
		case 0: break;
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			write_label(&value, sizeof(value));
			break;
		}
		case 3: data = Sync::ReadReferenceClock(); [[fallthrough]];
		case 2: write_label(&data, sizeof(data)); break;
		case 5: {
			GetScheduler().Finish();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}
#if defined(ANYPS5_KYTY)
	verify_written();
#endif

	if (interrupt) {
		DeferEopInterrupt(interrupt_context_id);
	}
}

void CommandProcessor::WriteBackImages() {
	if (!m_renderer.GetTextureCache().StorageReadbackEnabled()) {
		return;
	}
	EmitCallback([this] {
		if (m_renderer.GetTextureCache().WriteBackImages()) {
			GetScheduler().Flush();
		}
	});
}

void CommandProcessor::DeferEopInterrupt(uint32_t interrupt_context_id) {
	(void)CurrentBuffer().Handle();
	GetScheduler().DeferHostOperation(
	    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
		    renderer.TriggerInterrupt(event_id, interrupt_context_id);
	    },
	    true);
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto sequence = EmitCallback([this, event_type, event_index, event_address] {
		ExecuteTriggerEvent(event_type, event_index, event_address);
	});
	if (event_type == 0x00000039) {
		NoteGuestWrite(sequence, event_address, 16u * 2u * sizeof(uint64_t));
	}
}

void CommandProcessor::ExecuteTriggerEvent(uint32_t event_type, uint32_t event_index,
                                           uint64_t event_address) {
	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	switch (event_type) {
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: break;
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			auto& queries = m_renderer.GetOcclusionQueries();
			if (queries.Enabled()) {
				queries.Dump(GetScheduler(), event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
#if defined(ANYPS5_KYTY)
			if (Common::Fallback::Enabled()) {
				Common::HackNotice::Report("GPU event ignored: Kyty does not know the event type",
				                           "ExecuteTriggerEvent", 0, uint64_t {event_index} << 32u | event_type);
				break;
			}
#endif
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::WriteLodStats(void* dst, uint32_t size, uint32_t control) {
	EmitCallback([this, dst, size, control] {
		m_renderer.GetLodStats().Report(reinterpret_cast<uint64_t>(dst), size, control);
	});
}

#if defined(ANYPS5_KYTY)
static void NoteAbFlip() {
	if (!Common::Ab::Configured()) return;
	for (const auto& line: Common::Ab::AtFlip()) ::Log::WriteToConsoleAndLog(line);
}

#endif
void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}

	EmitCallback([this, handle, index, flip_mode, flip_arg] {
		auto& command = CurrentBuffer();
		m_renderer.GetOcclusionQueries().OnFrame();
#if defined(ANYPS5_KYTY)
		m_renderer.NoteFlip();
#endif
		auto  request = Sync::PrepareVideoOutFlip(command, handle, index, flip_mode, flip_arg);
#if defined(ANYPS5_KYTY)
		GetScheduler().DeferFlipCompletion(
		    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
#else
		GetScheduler().DeferHostOperation(
		    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); },
		    true);
#endif
		GetScheduler().Flush();
#if defined(ANYPS5_KYTY)
		NoteAbFlip();
#endif
	});
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	if (g_current_processor != nullptr) {
		EXIT("invalid graphics-thread CPU flip preparation\n");
	}
	m_renderer.CollectHotPageWrites();
	EmitCallback([this, request_id] {
		struct ProcessorScope {
			explicit ProcessorScope(CommandProcessor& processor) {
				g_current_processor = &processor;
			}
			~ProcessorScope() { g_current_processor = nullptr; }
		};
		ProcessorScope processor_scope(*this);

#if defined(ANYPS5_KYTY)
		m_renderer.NoteFlip();
#endif
		m_renderer.GetVideoOut().PrepareFlip(request_id, CurrentBuffer());
#if defined(ANYPS5_KYTY)
		GetScheduler().DeferFlipCompletion(
		    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
#else
		GetScheduler().DeferHostOperation(
		    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); }, true);
#endif
		GetScheduler().Flush();
#if defined(ANYPS5_KYTY)
		NoteAbFlip();
#endif
	});
}

void GuestGpu::RunUrgentCommandsWhileWaiting() {
	if (g_executing_gpu != nullptr) {
		g_executing_gpu->RunUrgentCommands();
	}
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_thread;
}

bool GuestGpu::IsCommandProcessorThread() noexcept {
	return g_cp_thread;
}

}
