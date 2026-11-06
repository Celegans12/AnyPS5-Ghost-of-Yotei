#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTDIRTY_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTDIRTY_HPP

#include <cstddef>
#include <cstdint>

namespace GuestArena {

struct GuestDirtyCounts {
    std::uint64_t arms = 0;
    std::uint64_t armedPages = 0;
    std::uint64_t protectCalls = 0;
    std::uint64_t armNs = 0;
    std::uint64_t refused = 0;
    std::uint64_t faults = 0;
    std::uint64_t faultNs = 0;
    std::uint64_t marks = 0;
    std::uint64_t markedBlocks = 0;
    std::uint64_t cleanQueries = 0;
    std::uint64_t cleanAnswers = 0;
};

struct GuestDirtyQuery {
    std::uintptr_t address;
    std::size_t bytes;
    std::uint64_t token;
};

extern "C" {

bool GuestDirtyEnabled_nid_postfix();
std::uint64_t GuestDirtyArm_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestDirtyCleanSince_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t token);
bool GuestDirtyCleanSinceAll_nid_postfix(const GuestDirtyQuery* queries, std::size_t count);
void GuestDirtyMark_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestDirtyHandleWrite_nid_postfix(std::uintptr_t address);
bool GuestDirtyArmed_nid_postfix(std::uintptr_t address);
void GuestDirtyBeginProtection_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyEndProtection_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyTakeCounts_nid_postfix(GuestDirtyCounts* counts);
void GuestDirtyForceEnable_nid_no_patch(bool enabled);
void GuestDirtyRunLocked_nid_no_patch(std::uintptr_t address, void (*body)(void*), void* argument);

}

}

#endif
