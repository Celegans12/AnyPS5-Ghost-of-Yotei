#pragma once

#include "common/onceValue.h"

#include <cstdint>
#include <cstdlib>

// BufferCache's diagnostic tools (bufferCacheTrace.cpp, with the members that hold the =2 checks of KYTY_D6_C0 and
// KYTY_RECORD_KEYED_VERIFY): every one is off unless its name is set, and a gate read once is the only cost then.
namespace Libs::Graphics::DownloadWaitTrace {

// KYTY_TRACE_DOWNLOAD_WAITS=1 (diagnostic): time of the GPU thread's synchronous readback waits by kind: why the BAR read
// declined (open command buffer / too large / ...), whether the newest write was already submitted, and eager readback
// waits on the open or an earlier command buffer.
enum Kind : uint32_t { BarNoTick, BarOpen, BarPending, BarLargeOld, BarLargeOpen, BarOutside, BarEmpty, NoBar, EagerOpen, EagerOld, Count };
[[nodiscard]] inline bool Enabled() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_DOWNLOAD_WAITS");
		return value != nullptr && value[0] == '1';
	});
}
inline int  g_last_bar_reason = -1;
inline bool g_last_bar_old    = false;
void        Add(Kind kind, uint64_t us, uint64_t bytes);

} // namespace Libs::Graphics::DownloadWaitTrace

namespace Libs::Graphics {
// BIG_BUFFER_CREATE: the host frames of the request (exe offsets, resolved with the link map).
void PrintBigBufferBacktrace();
} // namespace Libs::Graphics
