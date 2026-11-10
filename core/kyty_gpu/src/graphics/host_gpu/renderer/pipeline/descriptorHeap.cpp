#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/envRead.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {
namespace {

// KYTY_TRACE_DISPATCH_CENSUS: the heap's work, one PERF descriptor_heap line every ~5 s per thread that commits (the GPU
// thread's heap, and the recording thread's with the record packets).
struct HeapCensus {
	uint64_t commits = 0, allocate_calls = 0, failed_calls = 0, pool_resets = 0, pool_creates = 0, alloc_ns = 0;
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
bool HeapCensusEnabled() {
	static const bool on = Common::EnvRead::DispatchCensus();
	return on;
}
thread_local HeapCensus g_heap_census;

constexpr uint32_t   DescriptorHeapCount = 1024;

// PoolScale: sets come from pools this many times larger than the sizes below (n x 1024 sets, every size n times); since
// Measured: set allocation 1.47 -> 0.45 ms per frame at the torii gate, pool resets 567 -> 12 and failed allocations
// 3402 -> 72 per 5 s; live A/B gate +0.55% (se 0.33), open world -0.10% (se 0.47); from the start without errors. Census
// Measured (torii gate, KYTY_TRACE_DISPATCH_CENSUS): ~2100 sets per frame, but each layout takes a batch of 32 and the
// batches are dropped at every pool change, so a 1024-set pool lasted ~190 sets: ~11 pool resets, ~70 failed and ~430
// allocation calls per frame, 1.47 ms per frame on the GPU thread. The sets themselves are the same (same layouts,
// written the same way): only how often pools turn over.
constexpr uint32_t PoolScale = 16;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
    // Uniform const buffers (ShaderIR.h): flattened SRT and shader-data uniform views.
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 2048},
};

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	if (m_current_pool) m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	return Commit(layout, m_master_semaphore.CurrentTick());
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout, uint64_t tick) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);

	const bool census = HeapCensusEnabled();
	if (census) {
		auto& c = g_heap_census;
		c.commits++;
		if ((c.commits & 0xfffu) == 0u && std::chrono::steady_clock::now() - c.window >= std::chrono::seconds(5)) {
			std::fprintf(stderr,
			             "PERF descriptor_heap commits=%llu allocate_calls=%llu failed=%llu pool_resets=%llu pool_creates=%llu "
			             "alloc_ms=%.2f (5 s)\n",
			             static_cast<unsigned long long>(c.commits), static_cast<unsigned long long>(c.allocate_calls),
			             static_cast<unsigned long long>(c.failed_calls), static_cast<unsigned long long>(c.pool_resets),
			             static_cast<unsigned long long>(c.pool_creates), static_cast<double>(c.alloc_ns) / 1e6);
			c = {};
		}
	}
	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	const auto alloc_start = census ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const auto alloc_end   = [&] {
		if (census) {
			g_heap_census.alloc_ns += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - alloc_start).count());
		}
	};
	if (Allocate(layout, batch)) {
		alloc_end();
		return batch.sets[--batch.size];
	}

	m_pending_pools.emplace_back(m_current_pool, tick);
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
		if (census) g_heap_census.pool_resets++;
	} else {
		CreateDescriptorPool();
		if (census) g_heap_census.pool_creates++;
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	EXIT_IF(!Allocate(layout, fresh_batch));
	alloc_end();
	return fresh_batch.sets[--fresh_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (HeapCensusEnabled()) { // KYTY_TRACE_DISPATCH_CENSUS
			g_heap_census.allocate_calls++;
			if (result != vk::Result::eSuccess) g_heap_census.failed_calls++;
		}
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool() {
	auto sizes = DescriptorPoolSizes;
	for (auto& size: sizes) size.descriptorCount *= PoolScale;
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount * PoolScale;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
