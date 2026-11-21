#include "graphics/host_gpu/renderer/lodStats.h"

#include "common/assert.h"
#include "common/hackNotice.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint64_t CounterBytes = uint64_t {LodStatsCounter::Entries} * 2u * sizeof(uint32_t);

}

LodStatsCounter::LodStatsCounter(RenderContext& context): m_context(context) {}

LodStatsCounter::~LodStatsCounter() = default;

Buffer& LodStatsCounter::CounterBuffer() {
	if (m_counters == nullptr) {
		const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
		                   vk::BufferUsageFlagBits::eTransferSrc |
		                   vk::BufferUsageFlagBits::eTransferDst;
		m_counters = std::make_unique<Buffer>(m_context.GetGraphics(),
		                                      m_context.GetCommandScheduler(),
		                                      MemoryUsage::DeviceLocal, 0, usage, CounterBytes);
	}
	return *m_counters;
}

uint32_t LodStatsCounter::ReportBytes(uint64_t destination, uint32_t size) {
	const uint32_t bytes = std::min<uint32_t>(size, ReportSize);
	if (destination == 0 || bytes < sizeof(uint32_t)) {
		return 0;
	}
	return LibKernel::Memory::IsFullyMapped(destination, bytes) ? bytes : 0;
}

void LodStatsCounter::Report(uint64_t destination, uint32_t size, uint32_t control) {
	const bool     reset_requested = ((control >> 18u) & 3u) != 0u;
	const uint32_t write_size      = ReportBytes(destination, size);
	const bool     writes_report   = write_size != 0;
	if (!writes_report && destination != 0 && size >= sizeof(uint32_t)) {
		Common::HackNotice::Report("GET_LOD_STATS report outside guest memory dropped", "LodStatsCounter::Report", 0,
		                           destination);
	}
	if (m_initialized && !writes_report && !reset_requested) {
		return;
	}
	if (m_initialized && writes_report) {
		if (GuestRange {destination, write_size}.Valid() &&
		    m_context.GetBufferCache().HasGpuDirtyBytes(destination, write_size)) {
			m_context.PrepareGpuWrite(reinterpret_cast<const void*>(destination), write_size);
		}
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	auto& counters = CounterBuffer();
	if (m_publish == nullptr) {
		m_publish = std::make_unique<Buffer>(m_context.GetGraphics(), scheduler,
		                                     MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst,
		                                     PublishSlots * PublishSlotSize);
	}
	const bool publish     = m_initialized && writes_report;
	uint64_t   slot_offset = 0;
	if (publish) {
		const auto slot = static_cast<uint32_t>(m_issued % PublishSlots);
		if (m_issued >= PublishSlots) {
			const auto tick = m_slot_ticks[slot];
			if (!scheduler.IsFree(tick)) {
				scheduler.Wait(tick);
			}
			scheduler.WaitPriorityOperations(tick);
		}
		slot_offset = uint64_t {slot} * PublishSlotSize;
	}
	const auto native = scheduler.Current().Handle();

	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &barrier, 0, nullptr, 0,
	                       nullptr);
	if (publish) {
		const vk::BufferCopy copy {0, slot_offset, CounterBytes};
		native.copyBuffer(counters.Handle(), m_publish->Handle(), 1, &copy);
	}
	const bool reset = !m_initialized || reset_requested;
	if (reset && publish) {
		vk::MemoryBarrier war {};
		war.srcAccessMask = vk::AccessFlagBits::eTransferRead;
		war.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &war, 0, nullptr, 0,
		                       nullptr);
	}
	if (reset) {
		constexpr uint64_t WordBytes = uint64_t {Entries} * sizeof(uint32_t);
		native.fillBuffer(counters.Handle(), 0, WordBytes, LodStatsReport::Unsampled);
		native.fillBuffer(counters.Handle(), WordBytes, WordBytes, 0u);
	}
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &barrier, 0, nullptr, 0, nullptr);
	m_initialized = true;

	if (!publish) {
		return;
	}
	m_slot_ticks[m_issued % PublishSlots] = scheduler.CurrentTick();
	++m_issued;
	scheduler.DeferPriorityOperation(
	    [this, slot_offset, destination, write_size] {
		    m_publish->Invalidate(slot_offset, CounterBytes);
		    const auto* words =
		        reinterpret_cast<const uint32_t*>(m_publish->Mapped().data() + slot_offset);
		    std::array<uint8_t, ReportSize> report {};
		    LodStatsReport::PackReport(words, report.data());
		    if (LibKernel::Memory::IsFullyMapped(destination, write_size)) {
			    m_context.WriteFromCompletion(reinterpret_cast<void*>(destination), report.data(),
			                                  write_size);
		    }
	    },
	    destination, write_size);
}

}
