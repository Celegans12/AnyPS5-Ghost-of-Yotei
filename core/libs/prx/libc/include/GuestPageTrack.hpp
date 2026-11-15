#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTPAGETRACK_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTPAGETRACK_HPP

#ifdef _WIN32
#include <cstddef>
#include <cstdint>

struct _MEMORY_BASIC_INFORMATION;

namespace GuestArena {

extern "C" {

bool GuestPageTrackEnabled_nid_no_patch();
bool GuestPageTrackHandleWrite_nid_no_patch(std::uintptr_t address);
int GuestPageTrackState_nid_no_patch(std::uintptr_t address);
bool GuestPageTrackQuery_nid_no_patch(const void* address, _MEMORY_BASIC_INFORMATION* memory);
bool GuestPageTrackQueryOpened_nid_no_patch(const void* address, _MEMORY_BASIC_INFORMATION* memory);
bool GuestPageTrackProtect_nid_no_patch(void* pointer, std::size_t bytes, std::uint32_t protection, bool requireCommitted);
void GuestPageTrackKeepOpen_nid_no_patch(const void* pointer, std::size_t bytes);
bool GuestPageTrackTokens_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* tokens, std::size_t count);
bool GuestPageTrackCleanSince_nid_no_patch(std::uintptr_t address, std::size_t bytes, const std::uint64_t* tokens, std::size_t count);
void GuestPageTrackFrame_nid_no_patch();
void GuestPageTrackCounts_nid_no_patch(std::uint64_t* counts, std::size_t capacity);

}

}
#endif

#endif
