#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTPAGETRACK_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTPAGETRACK_HPP

#ifdef _WIN32
#include "prx/libc/include/GuestArena.hpp"
#include <cstddef>
#include <cstdint>

struct _MEMORY_BASIC_INFORMATION;

namespace GuestArena {

enum class MappingLockRole : std::uint8_t { Game = 0, Committer, Finisher, QueueWorker, Walker, Presenter, Count };
enum class MappingLockCaller : std::uint8_t { Other = 0, Forget, Collect, Commit, KeepOpen, QueryOpened, Reset, MapZero, Map, SetProtection, Pin, Unpin, CompareWhenShared, ComparedPages, HandleWrite, HandleRead, ResidentSpan, ForgetSection, BeginHostWrite, EndHostWrite, MapAlias, UnmapAlias, Protection, CollectWork, ArmOwned, OwnedUnchanged, GpuWrite, ArmBlocks, RearmBlocks, Count };
inline constexpr std::size_t MappingLockRoles = static_cast<std::size_t>(MappingLockRole::Count);
inline constexpr std::size_t MappingLockCallers = static_cast<std::size_t>(MappingLockCaller::Count);
inline constexpr std::size_t MappingLockRoleFields = 3 + MappingLockRoles;
inline constexpr std::size_t MappingLockCallerFields = 4;
inline constexpr std::size_t MappingLockShares = MappingLockRoles * MappingLockRoleFields + MappingLockCallers * MappingLockCallerFields;

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
bool GuestPageTrackArm_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* tokens, std::uint64_t* reasons, std::size_t count);
bool GuestPageTrackArmsImports_nid_no_patch();
void GuestPageTrackGpuWrite_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint8_t origin = static_cast<std::uint8_t>(PageTrackOrigin::Announced));
void GuestPageTrackGpuUnwatched_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestPageTrackGpuWatched_nid_no_patch(std::uintptr_t address, std::size_t bytes);
std::uint64_t GuestPageTrackGpuFlightBegin_nid_no_patch();
void GuestPageTrackGpuFlightEnd_nid_no_patch(std::uint64_t flight);
bool GuestPageTrackReasons_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* reasons, std::size_t count);
bool GuestPageTrackRearm_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* tokens, std::uint16_t* written, std::uint64_t* reasons, std::size_t count);
void GuestPageTrackFrame_nid_no_patch();
void GuestPageTrackCounts_nid_no_patch(std::uint64_t* counts, std::size_t capacity);
void GuestPageTrackOrigins_nid_no_patch(std::uint64_t* counts, std::size_t capacity);
void GuestMappingLockRole_nid_no_patch(std::uint8_t role);
void GuestMappingLockShares_nid_no_patch(std::uint64_t* counts, std::size_t capacity);

}

}
#endif

#endif
