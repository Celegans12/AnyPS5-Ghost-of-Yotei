#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTPAGETRACK_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTPAGETRACK_HPP

#ifdef _WIN32
#include <cstddef>
#include <cstdint>

namespace GuestArena {

extern "C" {

bool GuestPageTrackEnabled_nid_no_patch();
bool GuestPageTrackHandleWrite_nid_no_patch(std::uintptr_t address);
int GuestPageTrackState_nid_no_patch(std::uintptr_t address);

}

}
#endif

#endif
