#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/onceValue.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/gpuCheckpoints.h"
#include "graphics/host_gpu/renderer/gpuTimer.h"
#include "graphics/host_gpu/renderer/syncTimeline.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

MasterSemaphore::Stats& MasterSemaphore::GetStats() {
	static Stats stats;
	return stats;
}

namespace {
thread_local uint32_t t_query_site = 3; // KYTY_TRACE_PERF: who calls RefreshNow
struct QuerySite {
	explicit QuerySite(uint32_t site): previous(t_query_site) { t_query_site = site; }
	~QuerySite() { t_query_site = previous; }
	uint32_t previous;
};
} // namespace

void MasterSemaphore::Refresh() {
	const QuerySite site(0);
	constexpr int64_t PollNs = 100'000; // the semaphore is queried (a kernel call) at most every 100 us
	const auto        now    = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
	if (now - m_last_refresh_ns.load(std::memory_order_relaxed) < PollNs) {
		return;
	}
	m_last_refresh_ns.store(now, std::memory_order_relaxed);
	RefreshNow();
}

void MasterSemaphore::RefreshNow() {
	uint64_t   counter = 0;
	const auto start   = std::chrono::steady_clock::now();
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	{
		auto& s = GetStats();
		s.queries[t_query_site].fetch_add(1, std::memory_order_relaxed);
		s.query_ns[t_query_site].fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                              std::chrono::steady_clock::now() - start)
		                                                              .count()),
		                                   std::memory_order_relaxed);
	}
	if (result != vk::Result::eSuccess) {
		std::fprintf(stderr, "getSemaphoreCounterValue failed: %s\n", vk::to_string(result).c_str());
		GpuCheckpoints::Dump(m_graphics.queue);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	KYTY_PROFILER_BLOCK("MasterSemaphore::Wait");
	if (IsFree(tick)) {
		return;
	}
	{
		const QuerySite site(1);
		RefreshNow();
	}
	if (IsFree(tick)) {
		return;
	}
	const auto block_start = std::chrono::steady_clock::now();
	// KYTY_TRACE_SYNC_TIMELINE: a wait that blocks, attributed to its thread, site and reader.
	SyncTimeline::WaitScope timeline(SyncTimeline::Site::None, tick, CurrentTick(), KnownGpuTick());

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	PassCensus::t_wait_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - block_start).count()); // KYTY_TRACE_PASS_CENSUS
	if (GpuTimer::Enabled() && GpuTimer::TimelineCpuActive()) GpuTimer::NoteCpuWait('f', block_start); // KYTY_TRACE_GPU_TIMELINE
	if (result != vk::Result::eSuccess) {
		std::fprintf(stderr, "waitSemaphores failed: %s\n", vk::to_string(result).c_str());
		GpuCheckpoints::Dump(m_graphics.queue);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	{
		auto& s = GetStats();
		s.blocks.fetch_add(1, std::memory_order_relaxed);
		s.block_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                               std::chrono::steady_clock::now() - block_start)
		                                               .count()),
		                     std::memory_order_relaxed);
	}
	const QuerySite site(2);
	RefreshNow();
}

} // namespace Libs::Graphics
