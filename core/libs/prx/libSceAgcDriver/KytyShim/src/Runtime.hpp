#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_RUNTIME_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_RUNTIME_HPP

#include "kernel/eventQueue.h"

namespace Libs::Graphics {
class RenderContext;
struct GraphicContext;
}

namespace KytyShim {

inline constexpr Libs::LibKernel::EventQueue::KernelEqueue InterruptRoute = 0x4b59545900000000ll;
inline constexpr int GraphicsInterruptEvent = 0;
inline constexpr int FirstComputeInterruptEvent = 0x20;
inline constexpr int ComputeInterruptEvents = 7 * 8;

void StartRuntime();

void ReportNativeRayTracing(const Libs::Graphics::GraphicContext& graphics);

void RouteInterrupts(Libs::Graphics::RenderContext& renderer);

}

#endif
