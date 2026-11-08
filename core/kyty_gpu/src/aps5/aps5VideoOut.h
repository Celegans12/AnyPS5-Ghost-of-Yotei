#ifndef KYTY_APS5_APS5VIDEOOUT_H_
#define KYTY_APS5_APS5VIDEOOUT_H_

#include "aps5/GpuBackend.h"

#include <cstdint>

namespace Libs::Graphics {
class Presenter;
} // namespace Libs::Graphics

namespace Libs::VideoOut {
class VideoOutDriver;
} // namespace Libs::VideoOut

// Kyty's VideoOutDriver over AnyPS5's VideoOut (videoOutBridge.cpp). AnyPS5 keeps sceVideoOut*, the flip queue, the vblank
// pacing, the flip events and the window; Kyty records the copy of the display buffer into a presenter frame where the flip
// is in the command stream (as standalone), tells AnyPS5 the flip is ready when that GPU work completed, and presents
// the frame when AnyPS5's presenter thread asks for it.
namespace Aps5 {

[[nodiscard]] Libs::VideoOut::VideoOutDriver& VideoOutCreate(Libs::Graphics::Presenter& presenter, uint32_t width,
                                                             uint32_t height);
// A CPU flip (sceVideoOutSubmitFlip): reserved at AnyPS5 now, its copy ordered after the submissions already made.
int VideoOutSubmitCpuFlip(const Aps5GpuFlip& flip);
// AnyPS5's presenter thread: shows the next prepared frame (or a cleared one) and calls presented(context).
int VideoOutPresent(const Aps5GpuWindow* window, const Aps5GpuDisplayBuffer* buffer, int32_t opaque, int32_t dump_index,
                    void (*presented)(void* context), void* presented_context);

} // namespace Aps5

#endif // KYTY_APS5_APS5VIDEOOUT_H_
