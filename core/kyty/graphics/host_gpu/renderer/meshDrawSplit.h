#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWSPLIT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWSPLIT_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics::MeshDrawSplit {

[[nodiscard]] inline uint32_t RowsPerDraw(uint64_t max_rows, uint64_t max_total, uint32_t groups) {
	return static_cast<uint32_t>(
	    std::max<uint64_t>(1, std::min<uint64_t>(max_rows, max_total / std::max(groups, 1u))));
}

template <typename Func>
void ForEachDraw(uint32_t instances, uint32_t rows_per_draw, Func&& func) {
	for (uint32_t first = 0; first < instances;) {
		const auto count = std::min(rows_per_draw, instances - first);
		func(first, count);
		first += count;
	}
}

}

#endif
