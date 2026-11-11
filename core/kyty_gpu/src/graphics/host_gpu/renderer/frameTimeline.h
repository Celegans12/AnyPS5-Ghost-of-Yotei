#ifndef KYTY_GRAPHICS_FRAME_TIMELINE_H_
#define KYTY_GRAPHICS_FRAME_TIMELINE_H_

#include "graphics/host_gpu/renderer/debugSample.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

// KYTY_TRACE_FRAME_TIMELINE=<frame>[,<period>] (diagnostic, default off; one once-read value per hook when off): the
// chain of a few frames on one clock. While the presented-frame counter is in <frame>-1 .. <frame>+3 (and, with a
// period, in the same five frames every <period> frames after), each of these events is kept as one line
//   FTL us=<steady clock, microseconds> frame=<presented frames> ev=<event> ...
//   ev=frame                      the execution thread passed a flip (the frame boundary)
//   ev=submit q=<queue>           a guest thread handed a command buffer to the command processor
//   ev=cbuf tick=<n>              the execution thread handed host command buffer <n> over; ev=qsubmit: the worker submits it
//   ev=tick tick=<n>              the priority thread saw the host GPU finish tick <n> and runs an operation deferred to it
//   ev=label rec_us= done_us= dst= value= irq= ctx=    an end-of-pipe label was published (recorded, GPU done, now)
//   ev=wait thr=<exec|other> kind=<sem|fence|direct> dur_us=<n> [tick=<n>]   a blocking wait for the host GPU ended
//   ev=idle dur_us=<n>            the execution thread had no operation for that long (commands served inside count)
// The lines are printed after the window, at the first frame boundary outside it (printing ~600 lines per frame inside
// the window made its frames ~60 ms longer). Nothing else changes. Offline: research/v4/e3/tools/ftl.py.
namespace Libs::Graphics::FrameTimeline {

struct Window {
	uint32_t first  = 0;
	uint32_t period = 0;
};

[[nodiscard]] inline Window Config() {
	static const Window window = [] {
		Window      w;
		const char* text = std::getenv("KYTY_TRACE_FRAME_TIMELINE");
		if (text == nullptr) return w;
		char* end = nullptr;
		w.first   = static_cast<uint32_t>(std::strtoul(text, &end, 10));
		if (end != nullptr && *end == ',') w.period = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
		if (w.period < 8u) w.period = 0;
		if (w.first != 0) {
			std::fprintf(stderr, "TRACE_FRAME_TIMELINE frames %u..%u%s\n", w.first - 1u, w.first + 3u,
			             w.period != 0 ? " and every period after" : "");
		}
		return w;
	}();
	return window;
}

[[nodiscard]] inline bool Active() {
	const Window w = Config();
	if (w.first == 0) return false;
	const uint32_t frame = DebugPresentCount().load(std::memory_order_relaxed);
	if (frame + 1u < w.first) return false;
	const uint32_t offset = frame + 1u - w.first;
	return (w.period != 0 ? offset % w.period : offset) < 5u;
}

[[nodiscard]] inline int64_t NowUs() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

[[nodiscard]] inline int64_t Us(std::chrono::steady_clock::time_point point) {
	return std::chrono::duration_cast<std::chrono::microseconds>(point.time_since_epoch()).count();
}

struct Lines {
	std::mutex               mutex;
	std::vector<std::string> kept;
};

[[nodiscard]] inline Lines& Kept() {
	static auto* lines = new Lines; // lives for the process
	return *lines;
}

// One line; `format` continues after "ev=". Kept until Flush.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void Event(const char* format, ...) {
	char      text[320];
	const int head = std::snprintf(text, sizeof(text), "FTL us=%lld frame=%u ev=", static_cast<long long>(NowUs()),
	                               DebugPresentCount().load(std::memory_order_relaxed));
	va_list   args;
	va_start(args, format);
	std::vsnprintf(text + head, sizeof(text) - static_cast<size_t>(head), format, args);
	va_end(args);
	auto&            lines = Kept();
	std::scoped_lock lock(lines.mutex);
	if (lines.kept.size() < 200000u) lines.kept.emplace_back(text);
}

// The execution thread, at a frame boundary: prints the kept lines once the window is over.
inline void Flush() {
	if (Config().first == 0 || Active()) return;
	auto&                    lines = Kept();
	std::vector<std::string> out;
	{
		std::scoped_lock lock(lines.mutex);
		if (lines.kept.empty()) return;
		out.swap(lines.kept);
	}
	for (const auto& line: out) {
		std::fputs(line.c_str(), stderr);
		std::fputc('\n', stderr);
	}
}

} // namespace Libs::Graphics::FrameTimeline

#endif // KYTY_GRAPHICS_FRAME_TIMELINE_H_
