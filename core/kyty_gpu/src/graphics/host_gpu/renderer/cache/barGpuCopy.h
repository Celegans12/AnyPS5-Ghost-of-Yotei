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
//
// KYTY_PERF_BAR_GPU_DIRECT=1 (default 0, off; live A/B switch; lane v4e3): the second-queue copy writes the download's
// GPU-written intervals straight into the guest backing (the imported 2 MiB granules the copy path's direct download
// uses, BufferCache::PlanDownloadToBacking) instead of into the staging buffer, so the GPU thread's copy of the staged
// bytes into guest memory (WriteBacking; 37 MB and 3.6 ms per frame in the river village) is not made. The bytes are
// the same: the same source ranges after the same write tick; only the intervals themselves are copied (never the
// gaps between them, where guest memory may hold newer CPU bytes); the pages stay closed to the guest and GPU-modified
// until the request ends, as during the staging copy; the emulator's own readers ask the GPU-dirty set or check their
// read logs at execution time. A download with an interval outside one guest mapping or a granule the driver does not
// import takes the staging buffer as before (PERF bar_gpu_copy direct_declined). Side effect, as the copy path's: a
// granule imported for a download stays imported and is added to the game's granule list.
// KYTY_PERF_BAR_GPU_COPY_VERIFY=1 compares these copies with the mapping too.
//
// KYTY_PERF_COPY_HANDBACK=1 (default 0, off; live A/B switch; lane v4e3): the GPU thread does not wait for the copy of a
// guest fault's download. The request of a retried guest fault from another thread (the caller wait's requests,
// BufferCache::PlanCallerWait) whose bytes are finished on the GPU and would be copied by the GPU now is served in two
// steps. First the GPU thread starts the copy into the backing on the second queue (as KYTY_PERF_BAR_GPU_DIRECT's, but
// signalling the next value of a copy timeline instead of a fence), takes the copied bytes out of the GPU-dirty set,
// lists the window as a pending eager readback and hands the request back; the faulting thread waits for that value
// and sends the request again, which finds the readback finished, applies it and releases the pages. Exact because
// the window is in the state an asynchronous (eager) copy leaves it in, which every reader already handles: its pages
// stay GPU-modified and closed to the guest until the readback is applied; the GPU thread's own readers wait for it
// (WaitPendingReadbacks); the resolve thread sees it as pending; the write-through declines; other faulting threads
// wait for the same value; a GPU write recorded meanwhile marks its bytes GPU-dirty again and their page stays
// modified. Two things an eager copy gets from the tick order are done by hand here, because this copy runs beside
// the main queue: a download whose range overlaps such a window waits for the copy before it publishes anything
// (BufferCache::WaitCopyReadbacks), and no second hand-back copy is started over a window that has one under way.
// Not handed back: downloads the GPU would not copy (small ones), windows the backing plan cannot place, a write fault
// whose page the dead-ring rule frees, and requests after the caller wait's two hand-backs.
namespace Libs::Graphics::BarGpuCopy {

// Device creation: whether the device gets a second queue of the universal queue family for these copies (the switch is
// set at start or KYTY_LIVE_AB lists it). Without it the switch has no effect.
[[nodiscard]] bool WantQueue();

// After device creation: the second queue, or null when the family has only one.
void SetQueue(vk::Queue queue, uint32_t family);

} // namespace Libs::Graphics::BarGpuCopy

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BAR_GPU_COPY_H_
