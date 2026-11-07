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
    std::uint64_t bumps = 0;
    std::uint64_t viewLosses = 0;
    std::uint64_t viewArms = 0;
    std::uint64_t cleanQueries = 0;
    std::uint64_t cleanAnswers = 0;
    std::uint64_t refusedBy[4][8] = {};
    std::uint64_t armsBy[4] = {};
    std::uint64_t importArmedPages = 0;
    std::uint64_t importMarks = 0;
    std::uint64_t renewRefused = 0;
    std::uint64_t lateFaults = 0;
    std::uint64_t declined = 0;
    std::uint64_t retriedFaults = 0;
    std::uint64_t lastDeclined = 0;
    std::uint64_t lastDeclinedState = 0;
    std::uint64_t armWaits = 0;
    std::uint64_t armWaitNs = 0;
};

enum GuestDirtyArmKind : unsigned { ArmOther, ArmTexture, ArmTarget, ArmStorage };

enum GuestDirtyRefusal : unsigned { RefusedEpoch, RefusedInFlight, RefusedGuards, RefusedBlock, RefusedUncommitted, RefusedPrivateNotWritable, RefusedOtherMemory, RefusedView, RefusedDisabled };

struct GuestDirtyQuery {
    std::uintptr_t address;
    std::size_t bytes;
    std::uint64_t token;
};

extern "C" {

bool GuestDirtyEnabled_nid_postfix();
std::uint64_t GuestDirtyArm_nid_postfix(std::uintptr_t address, std::size_t bytes);
// This thread's last Arm refusal reason + 1 (0: none since the last take; RefusedDisabled: tracker off), then reset.
unsigned GuestDirtyTakeLastRefusal_nid_postfix();
bool GuestDirtyCleanSince_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t token);
bool GuestDirtyCleanSinceAll_nid_postfix(const GuestDirtyQuery* queries, std::size_t count);
void GuestDirtyMark_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyBump_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestDirtyHandleWrite_nid_postfix(std::uintptr_t address);
bool GuestDirtyArmed_nid_postfix(std::uintptr_t address);
void GuestDirtyMappingChange_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyBeginProtection_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyBeginHostWrite_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyEndProtection_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyBeginHostImport_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyNoteImportArmed_nid_postfix(std::uintptr_t address);
unsigned GuestDirtySetArmKind_nid_postfix(unsigned kind);
std::uint64_t GuestDirtyOpenWindow_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestDirtyCloseWindow_nid_postfix();
std::uint64_t GuestDirtyRenew_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint64_t window);
void GuestDirtyTakeCounts_nid_postfix(GuestDirtyCounts* counts);
void GuestDirtyForceEnable_nid_no_patch(bool enabled);
void GuestDirtyRunLocked_nid_no_patch(std::uintptr_t address, void (*body)(void*), void* argument);

}

}

#endif
