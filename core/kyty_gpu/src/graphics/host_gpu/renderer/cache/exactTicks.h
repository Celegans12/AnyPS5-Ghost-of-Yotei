#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_EXACT_TICKS_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_EXACT_TICKS_H_

#include <cstdint>

// Exact write ticks for reads of GPU-written bytes (lane v4e2 of AnyPS5's PLAN v4).
//
// A CPU read of GPU-written bytes (BufferCache::ReadMemory) waits for the newest write tick of the 64 KiB granules that
// hold them (TryBarDownload, PlanCallerWait), and the read window is narrowed to the access's pages, or taken whole by
// the copy path, when that granule tick is not finished. A granule's tick is the newest write of any of its bytes, so a
// read of bytes written long ago waits for, or submits and waits for, unrelated writes beside them. KYTY_TRACE_SYNC_TIMELINE
// in AnyPS5 (frames 4001-4003 of a world run, need_tick/exact_tick): the SRT walker's frame-start read of 36 bytes at
// 0x5000001424 waited 2.7 ms for granule tick +37 while its own bytes' tick was long finished; its reads in the two
// 9 MiB buffers waited 6.7-7.0 ms the same way; 0, 9.4 and 7.0 ms in those three frames.
//
// KYTY_PERF_EXACT_WALKER=1 (default 0, off; live A/B switch): an SRT-walker read waits for the newest write tick of the
// bytes it needs only. KYTY_PERF_EXACT_FAULTS=1 (default 0, off; live A/B switch): a guest read or write fault does the
// same for the 4 KiB pages it makes accessible. How:
//   - Exact ticks are kept per written byte range (an interval map, as KYTY_TRACE_SYNC_TIMELINE's), but only inside the
//     64 KiB granules such reads touched ("learned"; at most 4096 granules, 256 MiB, the least recently read one forgotten
//     with its entries at the limit): every granule tick raise (NoteGpuWriteTick: obtained writes, their final tick once
//     recorded, image copies into buffers) also notes its range clipped to the learned granules, at the same tick. Bytes
//     written before their granule was (last) learned have no entry, so their exact tick is unknown and the granule tick
//     is used for them; a covered byte's entry is never older than its newest write. They are kept whenever one of the
//     switches can be on (set at start or listed by KYTY_LIVE_AB), so either phase of a live A/B finds them.
//   - The read: every GPU-dirty interval of the read window (+-4 MiB, as ReadRange) gets its exact tick, else its granule
//     tick. The read waits for the newest tick of the intervals it needs (and for the buffer's own creation tick), when
//     that tick is submitted and none of them is a write obtained but not recorded; else it takes the usual path. Then it
//     publishes every interval finished by that tick: the window's when they fit the host-mapped read's limit
//     (BarReadMaxBytes), else those in the access's pages, else the needed ones only; by the BAR read or the GPU copy
//     (KYTY_PERF_BAR_GPU_COPY), with the same calls as TryBarDownload. A page left without GPU-dirty bytes is unmarked
//     (a written fault page is released to the CPU, as the usual path does); the others stay GPU-modified.
//   - Exact: each published interval's newest write is finished when it is read (its tick, exact or the granule's upper
//     bound, is at most the waited tick, and nothing recorded later writes it: a later write would have a newer tick),
//     so the published bytes are the ones the usual path publishes for them; bytes not published stay GPU-dirty and are
//     synchronized when accessed. Only which finished bytes a read publishes, and how long it waits, change.
//   - For a guest fault the caller wait (PlanCallerWait) hands the request back for the same exact tick.
// KYTY_PERF_EXACT_VERIFY=<n> (diagnostic): every n-th such read of each kind is compared with the bytes after all recorded
// work (TryExactDownload). KYTY_TRACE_PERF: the PERF exact_reads line every ~5 s.
namespace Libs::Graphics::ExactTicks {

// Whether the exact ticks are kept in this run (either switch set at start or listed by KYTY_LIVE_AB). Read once.
[[nodiscard]] bool Kept();
// The switches' values for this request (live A/B gates).
[[nodiscard]] bool WalkerOn();
[[nodiscard]] bool FaultsOn();
// Whether reads of that kind learn granules in this run.
[[nodiscard]] bool WalkerPossible();
[[nodiscard]] bool FaultsPossible();

} // namespace Libs::Graphics::ExactTicks

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_EXACT_TICKS_H_
