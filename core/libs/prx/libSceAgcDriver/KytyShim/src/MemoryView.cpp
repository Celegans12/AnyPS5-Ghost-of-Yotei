#include "kernel/memory.h"

#include "MemoryShim.hpp"
#include "prx/libkernel/DirectMemory/DirectMemory.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace Libs::LibKernel::Memory {

namespace {

struct KernelStats {
	uint64_t direct_allocated = 0;
	uint64_t flexible_total = 0;
	uint64_t flexible_used = 0;
	uint64_t cpu_page_entries = 0;
	uint64_t gpu_page_entries = 0;
};

KernelStats SampleKernelStats() {
	KernelStats stats;
	int64_t start = 0;
	int64_t end = 0;
	int type = 0;
	for (int64_t offset = 0; DirectMemoryFind_nid_no_patch(offset, true, &start, &end, &type) && end > offset; offset = end) {
		stats.direct_allocated += static_cast<uint64_t>(end - start);
	}
	size_t configured = 0;
	size_t available = 0;
	if (sceKernelConfiguredFlexibleMemorySize(&configured) != 0 || sceKernelAvailableFlexibleMemorySize(&available) != 0) {
		Shim::Fail("KytyShim memory: libkernel did not report the flexible memory sizes\n");
	}
	stats.flexible_total = configured;
	stats.flexible_used = configured - std::min(configured, available);
	constexpr uint64_t TableGranularity = 2ull * 1024ull * 1024ull;
	GuestPageTableUsage_nid_no_patch(TableGranularity, &stats.cpu_page_entries, &stats.gpu_page_entries);
	return stats;
}

class KernelStatsSampler {
public:
	KernelStats Request() {
		std::call_once(m_started, [this] { std::thread([this] { Run(); }).detach(); });
		std::lock_guard lock(m_mutex);
		m_requested = true;
		m_wake.notify_one();
		return m_latest;
	}

private:
	[[noreturn]] void Run() {
		std::unique_lock lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_requested; });
			m_requested = false;
			lock.unlock();
			const auto sample = SampleKernelStats();
			lock.lock();
			m_latest = sample;
		}
	}

	std::once_flag m_started;
	std::mutex m_mutex;
	std::condition_variable m_wake;
	bool m_requested = false;
	KernelStats m_latest;
};

KernelStatsSampler& StatsSampler() {
	static auto* sampler = new KernelStatsSampler;
	return *sampler;
}

[[maybe_unused]] const bool g_view_enabled = (Shim::EnableView(), true);

}

DebugStats GetDebugStats() {
	DebugStats stats {};
	stats.direct_total = DirectMemorySize_nid_no_patch();
	const auto kernel = StatsSampler().Request();
	stats.direct_allocated = kernel.direct_allocated;
	stats.flexible_total = kernel.flexible_total;
	stats.flexible_used = kernel.flexible_used;
	stats.cpu_page_entries = kernel.cpu_page_entries;
	stats.gpu_page_entries = kernel.gpu_page_entries;
	stats.direct_mapped = Shim::DirectMappedBytes();
	return stats;
}

}
