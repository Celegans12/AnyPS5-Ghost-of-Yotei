#include "common/diagFastPath.h"

namespace Common::DiagFastPath {

void NoteProfiler(bool running) {
	if (running) {
		g_state.fetch_or(kProfiler, std::memory_order_relaxed);
	} else {
		g_state.fetch_and(static_cast<uint8_t>(~kProfiler), std::memory_order_relaxed);
	}
}

} // namespace Common::DiagFastPath
