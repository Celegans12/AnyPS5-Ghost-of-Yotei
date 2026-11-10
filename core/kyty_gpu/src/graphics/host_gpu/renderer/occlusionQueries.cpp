#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include "common/assert.h"
#include "common/onceValue.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics {

bool OcclusionQueriesRequested() {
	static constinit Common::OnceValue<bool> on_once;
	const bool on = on_once.Get([] {
		const char* value   = std::getenv("KYTY_FIX_OCCLUSION_QUERIES");
		const bool  enabled = value == nullptr || value[0] != '0'; // default on (=0 turns it off)
		if (enabled) {
			std::fprintf(stderr, "FIX_OCCLUSION_QUERIES enabled: host occlusion queries back the PIXEL_PIPE_STAT_DUMP "
			                     "counters, written by the GPU; draws without outputs count while a query is open\n");
		}
		return enabled;
	});
	return on;
}

OcclusionQueries::OcclusionQueries(RenderContext& context): m_context(context) {}

OcclusionQueries::~OcclusionQueries() {
	if (m_pool != nullptr) {
		m_device.destroyQueryPool(m_pool, nullptr);
	}
}

bool OcclusionQueries::CreatePool(vk::CommandBuffer recorder) {
	if (m_pool != nullptr) {
		return true;
	}
	if (m_failed || !OcclusionQueriesRequested()) {
		return false;
	}
	const auto& graphics = m_context.GetGraphics();
	m_device             = graphics.device;
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eOcclusion;
	info.queryCount = Capacity;
	if (m_device.createQueryPool(&info, nullptr, &m_pool) != vk::Result::eSuccess) {
		m_pool   = nullptr;
		m_failed = true;
		std::fprintf(stderr, "FIX_OCCLUSION_QUERIES: cannot create the query pool; the counters stay synthetic "
		                     "(always visible)\n");
		return false;
	}
	SetVulkanObjectNameF(m_device, m_pool, "Kyty.OcclusionQueries");
	if (graphics.occlusion_query_precise) {
		m_control = vk::QueryControlFlagBits::ePrecise;
	} else {
		std::fprintf(stderr, "FIX_OCCLUSION_QUERIES: the device lacks occlusionQueryPrecise; a visible query may read "
		                     "any nonzero count\n");
	}
	// Every slot starts reset; the caller ended rendering.
	recorder.resetQueryPool(m_pool, 0, Capacity);
	m_reset_end = Capacity;
	return true;
}

std::vector<OcclusionQueries::Open>::iterator OcclusionQueries::Pair(uint64_t address) {
	const auto frame = DebugPresentCount().load();
	std::erase_if(m_open, [&](const Open& open) { return open.frame + MaxOpenFrames < frame; });
	return std::find_if(m_open.begin(), m_open.end(),
	                    [&](const Open& open) { return open.address + sizeof(uint64_t) == address; });
}

void OcclusionQueries::NoteSyntheticDump(uint64_t address) {
	const auto open = Pair(address);
	if (open != m_open.end()) {
		m_open.erase(open);
		return;
	}
	std::erase_if(m_open, [&](const Open& entry) { return entry.address == address; });
	m_open.push_back({address, m_next_slot, DebugPresentCount().load(), false});
}

uint64_t OcclusionQueries::OldestNeeded() const {
	uint64_t oldest = m_next_slot;
	for (const auto& open: m_open) {
		oldest = std::min(oldest, open.first_slot);
	}
	return oldest;
}

void OcclusionQueries::ResetAhead(vk::CommandBuffer recorder) {
	// Slots are reset outside render pass instances only. A slot is reset for its next use when no open guest
	// query still needs the result of its previous use (that one is older than the oldest open guest query).
	if (m_pool == nullptr || m_active || m_reset_end - m_next_slot >= ResetLow) {
		return;
	}
	const uint64_t target = std::min<uint64_t>(m_next_slot + ResetBatch, OldestNeeded() + Capacity);
	while (m_reset_end < target) {
		const auto index = static_cast<uint32_t>(m_reset_end % Capacity);
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(target - m_reset_end, Capacity - index));
		recorder.resetQueryPool(m_pool, index, count);
		m_reset_end += count;
	}
}

void OcclusionQueries::BeforeDraw(vk::CommandBuffer recorder) {
	if (m_pool == nullptr || m_active || m_open.empty()) {
		return;
	}
	if (m_next_slot >= m_reset_end) {
		// No reset slot left until the next point outside rendering (a guest query open across thousands of render
		// pass instances): this instance is not counted.
		for (auto& open: m_open) {
			open.uncounted = true;
		}
		static std::atomic_bool logged {false};
		if (!logged.exchange(true)) {
			std::fprintf(stderr, "FIX_OCCLUSION_QUERIES: no free query slot, a render pass instance was not counted "
			                     "(open guest queries=%zu)\n",
			             m_open.size());
		}
		return;
	}
	m_active_slot = static_cast<uint32_t>(m_next_slot % Capacity);
	m_next_slot++;
	recorder.beginQuery(m_pool, m_active_slot, m_control);
	m_active = true;
}

void OcclusionQueries::BeforeEndRendering(vk::CommandBuffer recorder) {
	if (!m_active) {
		return;
	}
	recorder.endQuery(m_pool, m_active_slot);
	m_active = false;
}

void OcclusionQueries::AfterEndRendering(vk::CommandBuffer recorder) {
	if (m_pool != nullptr && !m_open.empty()) {
		ResetAhead(recorder);
	}
}

uint64_t OcclusionQueries::SumOnCpu(uint64_t first, uint64_t end) {
	// The host queries ended earlier in this or an older command buffer: finish the GPU work, then read them.
	m_context.GetCommandScheduler().Finish();
	std::vector<uint64_t> results;
	uint64_t              sum = 0;
	for (uint64_t slot = first; slot < end;) {
		const auto index = static_cast<uint32_t>(slot % Capacity);
		const auto count = static_cast<uint32_t>(std::min<uint64_t>(end - slot, Capacity - index));
		results.assign(count, 0);
		const auto result =
		    m_device.getQueryPoolResults(m_pool, index, count, results.size() * sizeof(uint64_t), results.data(),
		                                 sizeof(uint64_t), vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
		if (result == vk::Result::eSuccess) {
			for (const auto value: results) {
				sum += value;
			}
		} else {
			std::fprintf(stderr, "FIX_OCCLUSION_QUERIES: vkGetQueryPoolResults failed (%s)\n",
			             vk::to_string(result).c_str());
		}
		slot += count;
	}
	return sum;
}

void OcclusionQueries::WriteSlots(CommandBuffer& buffer, uint64_t address,
                                  const std::array<uint32_t, DbCount>& low_dwords, uint64_t first, uint64_t end) {
	EXIT_IF(end > first && end - first > MaxCopied);
	auto& cache = m_context.GetBufferCache();
	// Uploads what the CPU wrote around the slots and marks the range GPU-written: CPU readers download it.
	auto [destination, destination_offset] = cache.ObtainBuffer(address, DumpBytes, true);
	EXIT_IF(destination == nullptr);
	std::array<uint32_t, DbCount * 2u> pattern {};
	for (uint32_t db = 0; db < DbCount; db++) {
		pattern[db * 2u]      = low_dwords[db];
		pattern[db * 2u + 1u] = ReadyHigh;
	}
	auto&      stream         = cache.GetUtilityBuffer(MemoryUsage::Stream);
	const auto pattern_offset = stream.Copy(pattern.data(), sizeof(pattern), 16);
	std::array<vk::BufferCopy, DbCount> regions {};
	for (uint32_t db = 0; db < DbCount; db++) {
		regions[db].srcOffset = pattern_offset + db * sizeof(uint64_t);
		regions[db].dstOffset = destination_offset + db * DbStride;
		regions[db].size      = sizeof(uint64_t);
	}
	buffer.EndRendering();
	const auto              native = buffer.Handle();
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = destination->Handle();
	barrier.offset              = destination_offset;
	barrier.size                = DumpBytes;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &barrier, 0, nullptr);
	// The 16 slots of this half of the record only: the other half (interleaved) keeps what is there.
	native.copyBuffer(stream.Handle(), destination->Handle(), DbCount, regions.data());
	if (end > first) {
		// The low dwords of DB 1.. take the host results (32-bit), after the pattern copy (write after write).
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 0,
		                       nullptr, 1, &barrier, 0, nullptr);
		uint64_t db = 1;
		for (uint64_t slot = first; slot < end;) {
			const auto index = static_cast<uint32_t>(slot % Capacity);
			const auto count = static_cast<uint32_t>(std::min<uint64_t>(end - slot, Capacity - index));
			native.copyQueryPoolResults(m_pool, index, count, destination->Handle(),
			                            destination_offset + db * DbStride, DbStride, vk::QueryResultFlagBits::eWait);
			slot += count;
			db += count;
		}
	}
	// Whatever reads the counters next (a compute shader, a predicate, a CPU download) sees them.
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 0,
	                       nullptr, 1, &barrier, 0, nullptr);
}

bool OcclusionQueries::Dump(CommandBuffer& buffer, uint64_t address) {
	// The segment of the stream before this dump ends here: its host query ends inside its render pass instance.
	if (m_active) {
		buffer.Handle().endQuery(m_pool, m_active_slot);
		m_active = false;
	}
	// The guest writes are copies, recorded outside any render pass instance.
	buffer.EndRendering();
	if (!CreatePool(buffer.Handle())) {
		return false;
	}
	std::array<uint32_t, DbCount> low {};
	uint64_t                      first      = 0;
	uint64_t                      end        = 0;
	uint64_t                      cpu_summed = 0;
	bool                          uncounted  = false;
	const auto                    open       = Pair(address);
	const bool                    is_end     = open != m_open.end();
	const uint64_t                begin      = is_end ? open->address : address;
	if (is_end) {
		first     = open->first_slot;
		end       = m_next_slot;
		uncounted = open->uncounted;
		if (end - first > MaxCopied) {
			// More host queries than DB slots: their sum goes to DB 0 (a submit and wait, rare).
			const auto sum = SumOnCpu(first, end);
			low[0]         = static_cast<uint32_t>(std::min<uint64_t>(sum, UINT32_MAX));
			cpu_summed     = end - first;
			first          = end;
		}
	}
	WriteSlots(buffer, address, low, first, end);
	// The guest query stops needing its host queries only now that their results are copied (slot resets follow).
	std::erase_if(m_open, [&](const Open& entry) { return entry.address == begin; });
	if (!is_end) {
		m_open.push_back({address, m_next_slot, DebugPresentCount().load(), false});
	}
	ResetAhead(buffer.Handle());
	m_dumps++;
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 64u || (m_dumps % 4096u) == 0u || uncounted) {
		std::fprintf(stderr,
		             "OCCLUSION_DUMP frame=%u addr=0x%llx %s host_queries=%llu cpu_summed=%llu open=%zu%s dumps=%llu "
		             "marker=[%s]\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(address), is_end ? "end" : "begin",
		             static_cast<unsigned long long>(end - first), static_cast<unsigned long long>(cpu_summed),
		             m_open.size(), uncounted ? " UNCOUNTED" : "", static_cast<unsigned long long>(m_dumps),
		             DebugMarkerPath().c_str());
	}
	return true;
}

} // namespace Libs::Graphics
