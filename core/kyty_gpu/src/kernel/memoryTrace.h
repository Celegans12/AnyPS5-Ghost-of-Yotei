#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace Libs::LibKernel::Memory {

// KYTY_TRACE_SRT_CLASSES (diagnostic, default off): the GPU thread's SRT reads, their waits for GPU writes and the Adopt
// work by what the word is for (shader-only or needed by the CPU) and by address region; one SRT_CLASSES line per ~5 s
// with per-frame averages. Decides whether letting the shaders read the shader-only words pays.
inline uint64_t SrtClassesNowNs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
void SrtClassesNoteRead(uint64_t vaddr, uint64_t size);
void SrtClassesWait(const char* where, uint32_t cls, uint64_t vaddr, uint64_t ns);

} // namespace Libs::LibKernel::Memory
