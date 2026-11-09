#ifndef KYTY_GRAPHICS_H_TRACE_H_
#define KYTY_GRAPHICS_H_TRACE_H_

#include <cstdint>

namespace Libs::Graphics::HTrace {

// KYTY_TRACE_H=1 (diagnostic, default off; one once-read flag per hook when off): the guest link of each presented frame,
// recorded the same way in Kyty standalone and in kyty_gpu.dll inside AnyPS5, so that a same-scene pair of runs can be
// compared hook by hook. One "HT" line per completed flip (times in microseconds of the QueryPerformanceCounter clock):
//   HT f=<flip> t=<now> rd=<ready->done> rp=<ready->present start> pd=<present start->done>
//      q0r=<the previous flip's ready->first queue-0 submission after it, before this flip's ready>
//      q0d=<previous done->first queue-0 submission after it, before this done>
//      anyr=<as q0r, for a submission of any queue> sub=<queue-0 submissions>/<other queues>
//      fg=<guest-thread faults>/<us in the handler> fe=<execution-thread faults>/<us> fo=<other threads' faults>/<us>
//      fx=<faults served by the outer handler>/<us there> fw=<flip-done waits>/<us> pac=<ranges changed after copy>/<ranges>
// where "ready" is the flip's GPU work done (CompleteFlip), "present start" the moment the presenter takes the frame
// and "done" the moment the flip is visible to the guest (its flip status counts it). The per-frame counts cover the
// time since the previous HT line. Every ~5 s an "HT_PAC" line lists the 64 KiB pages with the most changed ranges.
[[nodiscard]] bool Enabled();
[[nodiscard]] int64_t NowUs();

enum class FaultLayer : uint32_t { Handler, Outer };

void NoteSubmit(uint32_t queue);
void NoteFlipReady();
void NoteFlipPresentBegin();
void NoteFlipDone();
void NoteFault(FaultLayer layer, bool write, int64_t start_us);
void NoteFlipWait(int64_t start_us);
void NotePacRanges(uint64_t ranges);
void NotePacChanged(uint64_t address, uint64_t size);
void NotePacSample(uint64_t address, uint64_t size, uint64_t offset, uint64_t before, uint64_t after);

} // namespace Libs::Graphics::HTrace

#endif // KYTY_GRAPHICS_H_TRACE_H_
