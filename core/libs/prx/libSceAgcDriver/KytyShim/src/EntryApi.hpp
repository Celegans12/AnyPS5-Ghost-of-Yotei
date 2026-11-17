#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_ENTRYAPI_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_ENTRYAPI_HPP

#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include <cstdint>
#include <memory>

namespace Libs::Graphics {
class RenderContext;
class GuestGpu;
}

namespace KytyShim::Entry {

Libs::Graphics::RenderContext& Renderer();
Libs::Graphics::GuestGpu& Gpu();
bool GpuStarted();
std::shared_ptr<AgcDriver::IVideoOutput> VideoOutput(std::uint32_t handle);
bool OwnsFlipPacket(const std::uint32_t* packet);
void ReleaseFlipPacket(const std::uint32_t* packet);

}

#endif
