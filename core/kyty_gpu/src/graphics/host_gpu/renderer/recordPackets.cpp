#include "graphics/host_gpu/renderer/recordPackets.h"

#include "common/assert.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <memory>

// The record packets: the recording thread's descriptor heap and the statistics (recordPackets.h).
// Based on chenxiao07/KytyPS5 v20261020b (src/local/vulkan-recording.h, vulkan-descriptor-packet.inc), GPL-2.0.

namespace Libs::Graphics::RecordPackets {

namespace {

// The recording thread's heap: created and used by that thread only, destroyed by Shutdown after it stopped.
std::unique_ptr<DescriptorHeap> g_worker_heap;

std::array<std::atomic<uint64_t>, static_cast<size_t>(Stat::Count)> g_stats {};
std::array<uint64_t, static_cast<size_t>(Stat::Count)>              g_reported {};

} // namespace

bool On() {
	return CommandRecorder::AsyncSubmit();
}

vk::DescriptorSet WorkerHeapCommit(RenderContext& context, vk::DescriptorSetLayout layout, uint64_t tick) {
	if (g_worker_heap == nullptr) {
		g_worker_heap = std::make_unique<DescriptorHeap>(context.GetGraphics(),
		                                                 context.GetCommandScheduler().GetMasterSemaphore());
	}
	Note(Stat::HeapSets);
	return g_worker_heap->Commit(layout, tick);
}

void Shutdown() {
	g_worker_heap.reset();
}

void Note(Stat stat, uint64_t count) {
	// Statistics: the recording side and the recording thread each bump their own counters (relaxed adds).
	g_stats[static_cast<size_t>(stat)].fetch_add(count, std::memory_order_relaxed);
}

void Report(double seconds, uint64_t queued_writes, uint64_t write_flushes) {
	std::array<uint64_t, static_cast<size_t>(Stat::Count)> now {};
	std::array<uint64_t, static_cast<size_t>(Stat::Count)> d {};
	for (size_t i = 0; i < now.size(); i++) {
		now[i] = g_stats[i].load(std::memory_order_relaxed);
		d[i]   = now[i] - g_reported[i];
	}
	g_reported = now;
	const auto at = [&](Stat stat) { return d[static_cast<size_t>(stat)]; };
	// Per interval: commit packets, heap sets made by the recording thread, dynamic-state and pipeline packets, the
	// recording thread's state forgotten, bindless descriptor writes queued and moved into the ring.
	std::fprintf(stderr,
	             "RECORD_PACKETS s=%.1f commits=%" PRIu64 " heap_sets=%" PRIu64 " dyn=%" PRIu64 " pipelines=%" PRIu64
	             " invalidations=%" PRIu64 " queued_writes=%" PRIu64 " write_flushes=%" PRIu64 "\n",
	             seconds, at(Stat::CommitPackets), at(Stat::HeapSets), at(Stat::DynPackets), at(Stat::PipelinePackets),
	             at(Stat::Invalidations), queued_writes, write_flushes);
}

} // namespace Libs::Graphics::RecordPackets
