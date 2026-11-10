#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_VIRTUALGPUCLOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_VIRTUALGPUCLOCK_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandScheduler;

// KYTY_FIX_VIRTUAL_GPU_CLOCK (default on; =0 turns it off): the 64-bit GPU-clock values a guest asks
// for with end-of-pipe events (RELEASE_MEM / EVENT_WRITE_EOP, data_sel 3) are GPU time, as on the console.
//
// Before: the value was the host reference clock when the emulator's command processor translated the packet, so a
// game timing its GPU work measured the emulator's translation time instead (Ghost of Yotei's dynamic resolution saw
// ~200 ms per frame and dropped to its lowest size).
//
// Now the clock counts the host GPU's busy time on the renderer's command buffers, at 100 MHz, starting from the
// reference clock when the renderer starts:
// - every command buffer gets a top-of-pipe timestamp at its start and an all-commands timestamp at its end;
// - the busy time is the union of those intervals in submission order, so the gaps where the host GPU waited for the
//   emulator to record work are not counted, and neither are overlaps;
// - an end-of-pipe clock event records an all-commands timestamp at its point of the stream; once its command buffer
//   finished, the clock value at that point is written to guest memory by the GPU thread (a deferred operation), the
//   way the console's command processor writes it when the GPU gets there.
// Until then guest memory keeps the slot's older value: games read GPU timers frames later, after the GPU passed them.
//
// Known biases:
// - emulator work recorded into the stream (uploads, tiling, conversions) counts as GPU time;
// - asynchronous compute runs on the same host queue and is counted inside the graphics work;
// - guest waits (WAIT_REG_MEM, flips) become host gaps and are not counted.
// Events with an interrupt, and events that find no room in their command buffer's query chunk, get the clock value of
// the work finished so far at once (same clock, never the host reference clock).
// A PERF vgpu_clock line every ~5 s with KYTY_TRACE_PERF.
// Measured (Ghost of Yotei, torii gate, Performance mode, 3 interleaved pairs): the game's dynamic
// resolution settles at 1920x1080 + FSR3 instead of 3840x2160 (it also lowers its volumetric mist to 1/4 resolution,
// its own rule for a slow GPU); 4.63 -> 5.40 fps; every event written, none dropped.
namespace VirtualGpuClock {

[[nodiscard]] bool Enabled();

// Command-buffer lifecycle of the renderer's scheduler (GPU thread; the presentation scheduler is ignored).
void BeginCommandBuffer(CommandScheduler& scheduler, vk::CommandBuffer buffer);
void EndCommandBuffer(vk::CommandBuffer buffer);
void Submitted(vk::CommandBuffer buffer, uint64_t tick);

// An end-of-pipe GPU-clock event that writes 8 bytes at `dst`, at this point of `buffer` (GPU thread).
// true: taken; the value is written later (deferred), or was written at once (no room, see above).
// false: not taken (the feature is off, or `scheduler` is not the renderer's); the caller writes as before.
[[nodiscard]] bool Stamp(CommandScheduler& scheduler, vk::CommandBuffer buffer, uint64_t dst, bool immediate);

} // namespace VirtualGpuClock

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_VIRTUALGPUCLOCK_H_
