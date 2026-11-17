#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STRICTWATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_STRICTWATCH_H_

#include "common/common.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class PageManager;

class StrictWatch {
public:
	explicit StrictWatch(PageManager& page_manager): m_page_manager(page_manager) {}
	~StrictWatch();
	KYTY_CLASS_NO_COPY(StrictWatch);

	static constexpr uint64_t MinSize = 256 * 1024;

	[[nodiscard]] uint64_t Watch(uint64_t address, uint64_t size);
	void Changed(uint64_t address, uint64_t size, uint64_t version,
	             std::vector<std::pair<uint64_t, uint64_t>>& runs);
	void InvalidateMemory(uint64_t address, uint64_t size) {
		if (address < m_end.load(std::memory_order_acquire) &&
		    address + size > m_begin.load(std::memory_order_acquire)) {
			Invalidate(address, size);
		}
	}
	void UnmapMemory(uint64_t address, uint64_t size);

private:
	struct Range {
		uint64_t              base = 0;
		uint64_t              end  = 0;
		std::vector<uint64_t> versions;
		uint64_t              used = 0;
	};
	static constexpr size_t MaxRanges = 16;

	Range& Find(uint64_t address, uint64_t size);
	void Arm(Range& range);
	void Unwatch(Range& range, size_t first, size_t last);
	void Invalidate(uint64_t address, uint64_t size);
	void UpdateBounds();

	PageManager&                        m_page_manager;
	std::mutex                          m_mutex;
	std::vector<std::unique_ptr<Range>> m_ranges;
	uint64_t                            m_version = 0;
	uint64_t                            m_uses    = 0;
	std::atomic<uint64_t> m_begin {UINT64_MAX};
	std::atomic<uint64_t> m_end {0};
};

}

#endif
