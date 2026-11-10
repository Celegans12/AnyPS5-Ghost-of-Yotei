#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMER_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

#include <chrono>

namespace Libs::Graphics {

class CommandScheduler;

// KYTY_TRACE_GPU_TIME (diagnostic, default off): GPU timestamps measure where the GPU spends its
// time. Every ~5 s a GPUTIME line reports how long the GPU executed command buffers (busy), how
// long it sat idle between them (gap) and, with mode 1, the draws and dispatches that cost the most
// GPU time (GPUTIME_TOP).
//   1: timestamps at the start/end of every command buffer and around every draw and dispatch.
//      Timestamps wait for earlier work, so draws no longer overlap: totals are somewhat inflated.
//   2: only command-buffer start/end timestamps (busy/gap without per-command distortion).
//   3: alternates 1 and 2 every window (both measurements from one run).
namespace GpuTimer {

[[nodiscard]] int Mode();
[[nodiscard]] inline bool Enabled() {
	return Mode() != 0;
}

// Command buffer lifecycle, called by CommandScheduler on the GPU thread.
void BeginCommandBuffer(CommandScheduler& scheduler, vk::CommandBuffer buffer);
void EndCommandBuffer(vk::CommandBuffer buffer);
void Submitted(vk::CommandBuffer buffer, uint64_t tick);
// KYTY_TRACE_GPU_TIMELINE (CPU side): the recording worker hands command buffer `tick` to vkQueueSubmit now.
void NoteQueueSubmit(uint64_t tick);
// KYTY_TRACE_GPU_TIMELINE (CPU side): whether the GPU thread's waits are recorded now (the timeline's frames), and one
// wait of the GPU thread from `start` until now: 'f' GPU fence, 'b' every guest queue blocked (waiting for the game),
// 'r' the recording worker.
[[nodiscard]] bool TimelineCpuActive();
void               NoteCpuWait(char kind, std::chrono::steady_clock::time_point start);

// Around one draw ('d', a = vertex shader hash, b = pixel shader hash) or dispatch ('c', a = compute
// shader hash, b = 1 for indirect). Begin returns a slot for End (~0u when the command is untimed).
[[nodiscard]] uint32_t BeginCommand(vk::CommandBuffer buffer);
void                   EndCommand(vk::CommandBuffer buffer, uint32_t slot, char kind, uint64_t a, uint64_t b);
// Mode 1: GPU time from here to the next timestamp belongs to an emulator operation (tiler, image
// upload/download, buffer copies, clears...): kind and two values identify it in GPUTIME_TOP.
void Mark(vk::CommandBuffer buffer, char kind, uint64_t a, uint64_t b);

// KYTY_TRACE_VK_LABELS=1 (diagnostic): each draw and dispatch is wrapped in a VK_EXT_debug_utils
// label naming its shaders ("cs <hash>", "draw <vs> <ps>"), so external GPU profilers (Nsight
// Graphics GPU Trace, RenderDoc) report them per shader.
[[nodiscard]] bool LabelsEnabled();
void               LabelBegin(vk::CommandBuffer buffer, char kind, uint64_t a, uint64_t b);
void               LabelEnd(vk::CommandBuffer buffer);
// The emulator's own GPU work (uploads, downloads, barriers, image transitions) under the same switch: "emu <what> |
// <game marker path>", so a profiler shows what stands between two game draws.
void               LabelEmu(vk::CommandBuffer buffer, const char* what);

} // namespace GpuTimer

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUTIMER_H_
