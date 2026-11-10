#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BAR_GPU_COPY_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BAR_GPU_COPY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// KYTY_PERF_BAR_GPU_COPY=<KiB> (default 8; 0 off; live A/B switch; lane v4e of AnyPS5's PLAN v4): a host-mapped download
// (BufferCache::TryBarDownload) of at least that many GPU-written bytes is not read by the CPU from the resizable-BAR
// mapping but copied by the GPU into a host-cached staging buffer, on a second queue of the universal family, and
// published from there. On a laptop RTX 3080 a CPU read of BAR memory runs at ~50-90 MB/s (streaming loads, one line in
// flight; extra reader threads add nothing), so ~2 MB of such reads per frame cost ~38 ms of the GPU thread. The bytes
// are the same: the copy reads the buffer after the same write tick the host-mapped read waits for (its submission
// waits for that timeline value, which also makes the writes visible to it), no GPU work recorded or submitted later
// writes them (TryBarDownload's own conditions), and they are published to guest memory by the same calls in the same
// order. Only where the CPU reads them from changes.
//
// KYTY_PERF_BAR_GPU_LARGE=1 (default 1; 0 off; live A/B switch): the host-mapped download takes finished read windows of up
// to 8 MiB GPU-written bytes instead of 1 MiB (BufferCache::ReadRange, PlanCallerWait, TryBarDownload), and every
// download above 1 MiB is copied by the GPU as above. Such a window otherwise takes the copy path: the open command
// buffer is submitted with the copy and the whole queue is waited for. Which bytes a read publishes ahead changes; every
// read still gets the GPU's newest bytes and the rest of the window stays GPU-modified, as with the 1 MiB limit.
namespace Libs::Graphics::BarGpuCopy {

// Device creation: whether the device gets a second queue of the universal queue family for these copies (the switch is
// set at start or KYTY_LIVE_AB lists it). Without it the switch has no effect.
[[nodiscard]] bool WantQueue();

// After device creation: the second queue, or null when the family has only one.
void SetQueue(vk::Queue queue, uint32_t family);

} // namespace Libs::Graphics::BarGpuCopy

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BAR_GPU_COPY_H_
