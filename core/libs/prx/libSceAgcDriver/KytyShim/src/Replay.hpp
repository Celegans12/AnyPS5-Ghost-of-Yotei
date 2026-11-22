#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_REPLAY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_REPLAY_HPP

#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"

#include <cstdint>

namespace Libs::Graphics {
class GuestGpu;
struct ImageInfo;
}

namespace KytyShim::Entry {

std::uint64_t* SerialLabels();

}

namespace KytyShim::Present {

Libs::Graphics::ImageInfo DisplayBufferImage(const AgcDriver::DisplayBuffer& buffer);

}

namespace KytyShim::Replay {

inline constexpr std::size_t SerialQueues = 1 + 7 * 8;

void NoteSubmission(Libs::Graphics::GuestGpu& gpu, std::uint32_t queue);
void RequireNotHeld(const char* what);
bool Tracking();
bool Replaying();

}

#endif
