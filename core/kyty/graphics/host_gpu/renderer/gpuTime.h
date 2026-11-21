#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIME_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIME_H_

#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;

namespace GpuTime {

[[nodiscard]] bool ReadSwitch();

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = ReadSwitch();
	return enabled;
}

void BeginBuffer(CommandScheduler& scheduler, CommandBuffer& buffer);
void EndBuffer(CommandScheduler& scheduler, CommandBuffer& buffer);
void BeforeWork(CommandBuffer& buffer);
void AfterDraw(CommandBuffer& buffer, uint64_t vertex_hash, uint64_t pixel_hash);
void AfterDispatch(CommandBuffer& buffer, uint64_t compute_hash);
void NoteFlip();

[[nodiscard]] uint32_t InternMarker(const char* text, size_t max_bytes);
void                   SetMarker(uint32_t marker);

}

}

#endif
