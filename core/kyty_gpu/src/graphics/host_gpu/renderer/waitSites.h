#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WAITSITES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WAITSITES_H_

#include <chrono>
#include <cstdint>

// KYTY_TRACE_WAIT_SITES=1 (diagnostic, map 5.11 "ruch przez miasto": ~700 fence waits per frame in the town): every
// CommandScheduler wait for a fence, by thread (the GPU thread or another one) and by the call stack that asked for it,
// with its time. About every 5 s the top sites go to stderr as WAIT_SITE lines (exe-relative return addresses,
// symbolized offline with the build's link map) and one WAIT_SITES line with the totals per thread kind.
namespace Libs::Graphics::WaitSites {

[[nodiscard]] bool Enabled();
void               Record(uint64_t nanos);

class Scope {
public:
	Scope(): m_on(Enabled()) {
		if (m_on) m_start = std::chrono::steady_clock::now();
	}
	~Scope() {
		if (m_on)
			Record(static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - m_start).count()));
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
};

} // namespace Libs::Graphics::WaitSites

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WAITSITES_H_ */
