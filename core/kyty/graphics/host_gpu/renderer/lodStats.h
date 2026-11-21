#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LODSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LODSTATS_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/lodStatsReport.h"

#include <array>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class Buffer;
class RenderContext;

class LodStatsCounter {
public:
	static constexpr uint32_t Counters   = LodStatsReport::Counters;
	static constexpr uint32_t Entries    = LodStatsReport::Entries;
	static constexpr uint32_t ReportSize = LodStatsReport::ReportSize;

	explicit LodStatsCounter(RenderContext& context);
	~LodStatsCounter();
	KYTY_CLASS_NO_COPY(LodStatsCounter);

	[[nodiscard]] static uint32_t ReportBytes(uint64_t destination, uint32_t size);
	[[nodiscard]] Buffer& CounterBuffer();
	void Report(uint64_t destination, uint32_t size, uint32_t control);

private:
	static constexpr uint32_t PublishSlots    = 64;
	static constexpr uint64_t PublishSlotSize = 4096;

	RenderContext&                     m_context;
	std::unique_ptr<Buffer>            m_counters;
	std::unique_ptr<Buffer>            m_publish;
	std::array<uint64_t, PublishSlots> m_slot_ticks {};
	uint64_t                           m_issued      = 0;
	bool                               m_initialized = false;
};

}

#endif
