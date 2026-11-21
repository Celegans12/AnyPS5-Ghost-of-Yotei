#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP

#include <cstddef>
#include <cstdint>

#ifdef _WIN32
struct _EXCEPTION_POINTERS;
#endif

// Guest virtual memory is placed inside one reserved arena below the PS5 application map limit
// absolute address and breaks on host addresses outside that range.
namespace GuestArena {

#ifdef ANYPS5_ENABLE_ASAN
inline constexpr std::uintptr_t GuestAddressOffset = 0x0000100000000000ull;
#else
inline constexpr std::uintptr_t GuestAddressOffset = 0;
#endif

inline constexpr std::size_t MappingLockSites = 3;
inline constexpr std::size_t MappingForgetOutcomes = 4;
struct MappingLockSiteCounts {
    std::uint64_t acquisitions = 0;
    std::uint64_t waits = 0;
    std::uint64_t waitNs = 0;
    std::uint64_t behindNs[MappingLockSites] = {};
};
struct MappingLockCounts {
    MappingLockSiteCounts sites[MappingLockSites];
    std::uint64_t forgets[MappingForgetOutcomes] = {};
    std::uint64_t unlocked = 0;
    std::uint64_t unlockedHostWrites = 0;
};

enum class PageTrackReason : std::uint64_t { Armable = 0, Armed, Disabled, Edge, Held, Imported, Barred, Resting, Faulted, Busy, GpuWritten };
inline constexpr std::uint64_t PageTrackReasonMask = 0xffffffffu;
inline constexpr unsigned PageTrackReasonFramesShift = 32;
enum class PageTrackOrigin : std::uint8_t { None = 0, Commit, Recommit, Protect, KeepOpen, Resident, SetProtection, Import, Unwatch, MarkWritten, RecorderWrite, RecorderBatch, VerifyClean, Announced, Count };
inline constexpr std::size_t PageTrackOrigins = static_cast<std::size_t>(PageTrackOrigin::Count);
inline constexpr std::size_t PageTrackOriginFields = 5;

#ifndef _WIN32
using SharedBackingResolver = bool (*)(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
using SharedBackingWriter = bool (*)(std::uintptr_t address, const void* source, std::size_t bytes);
using SharedBackingHostFallback = bool (*)(std::uintptr_t address, std::size_t bytes);
struct SharedBackingRun {
    std::uint64_t offset;
    std::size_t bytes;
};
using SharedBackingRunsResolver = bool (*)(std::uintptr_t address, std::size_t bytes, int* file, SharedBackingRun* runs, std::size_t capacity, std::size_t* count);
#endif

extern "C" {

bool GuestArenaAvailable_nid_postfix();
bool GuestArenaContains_nid_postfix(const void* pointer, std::size_t bytes);
void* GuestArenaAllocate_nid_postfix(std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateAtOrAbove_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
void* GuestArenaAllocateFromHint_nid_postfix(std::uintptr_t hint, std::size_t bytes, std::size_t alignment);
std::uintptr_t GuestArenaHighestFree_nid_no_patch(std::size_t bytes, std::size_t alignment);
bool GuestArenaNextUsed_nid_no_patch(std::uintptr_t address, std::uintptr_t* start, std::uintptr_t* end);
void GuestArenaMarkUsed_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaRelease_nid_postfix(const void* pointer, std::size_t bytes);
// The reserved range, and whether it was reserved with page write watching (Windows MEM_WRITE_WATCH).
void GuestArenaRange_nid_postfix(std::uintptr_t* base, std::size_t* bytes);
bool GuestArenaWriteWatched_nid_postfix();
void GuestArenaSetProtection_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uint32_t protection);
void GuestArenaCommit_nid_postfix(void* pointer, std::size_t bytes, std::uint32_t protection, std::size_t granule);
void GuestArenaReset_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaSetAccessObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes));
#ifdef _WIN32
bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address);
bool GuestArenaHandleRead_nid_postfix(std::uintptr_t address);
void GuestArenaSetReadTrapResolver_nid_no_patch(bool (*resolve)(std::uintptr_t begin, std::uintptr_t end));
bool GuestArenaTrapReads_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaReleaseReads_nid_no_patch(std::uintptr_t address, std::size_t bytes);
bool GuestArenaReadsTrapped_nid_no_patch(std::uintptr_t address, std::size_t bytes);
bool GuestArenaCopyOwned_nid_no_patch(std::uintptr_t source, void* destination, std::size_t bytes);
bool GuestArenaCompareOwned_nid_no_patch(std::uintptr_t source, const void* expected, std::size_t bytes, bool* equal);
bool GuestArenaRecoverRead_nid_no_patch(_EXCEPTION_POINTERS* exception);
bool GuestArenaStoreOwned_nid_no_patch(std::uintptr_t destination, const void* source, std::size_t bytes);
bool GuestArenaRecoverWrite_nid_no_patch(_EXCEPTION_POINTERS* exception);
void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection);
bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectArmedWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectFreshWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
std::uint64_t GuestArenaCollectWatchCalls_nid_no_patch();
void GuestArenaMappingLockCounts_nid_no_patch(MappingLockCounts* counts);
void GuestArenaTimeWriteWatch_nid_no_patch(bool timed);
void GuestArenaWriteWatchCost_nid_no_patch(std::uint64_t* threadTicks, std::uint64_t* calls, std::uint64_t* ticks);
bool GuestArenaHostRegionOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes);
std::size_t GuestArenaHostRegions_nid_no_patch(std::uintptr_t* regions, std::size_t capacity);
bool GuestArenaHostRegionIndexActive_nid_no_patch();
std::uint64_t GuestArenaCommitGeneration_nid_postfix();
void GuestArenaSetPrivateMappingObserver_nid_postfix(void (*callback)(std::uintptr_t address, std::size_t bytes, std::uint64_t generation));
void GuestArenaMap_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection);
void GuestArenaMapDirect_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection);
void GuestArenaMapZero_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaMapFile_nid_postfix(void* pointer, std::size_t bytes, void* section, std::uint64_t offset, std::uint32_t protection, std::uint32_t viewProtection, std::size_t sectionBytes);
void* GuestArenaMapAlias_nid_postfix(std::uintptr_t address, std::size_t bytes);
void* GuestArenaMapAliasForImport_nid_no_patch(std::uintptr_t address, std::size_t bytes, bool* boundsRefused);
void GuestArenaUnmapAlias_nid_postfix(void* alias);
bool GuestArenaResidentSpan_nid_postfix(std::uintptr_t address, std::size_t bytes, std::uintptr_t* start, std::uintptr_t* end);
void GuestArenaCompareWhenShared_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaCompareResidentWhenShared_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaComparedPages_nid_no_patch(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count);
void GuestArenaTakeUnreportedPages_nid_no_patch(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count);
void GuestArenaKnownZero_nid_no_patch(std::uintptr_t address, std::size_t bytes, void (*zero)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
void GuestArenaForgetKnownZero_nid_no_patch(std::uintptr_t address, std::size_t bytes);
std::uint64_t GuestArenaKnownZeroMarks_nid_no_patch();
void GuestArenaForgetSection_nid_postfix(void* section);
bool GuestArenaBeginHostImport_nid_no_patch(void* pointer, std::size_t bytes);
void GuestArenaEndHostImport_nid_no_patch(void* pointer, std::size_t bytes);
bool GuestArenaForgetReusableSection_nid_no_patch(void* section);
#else
void GuestArenaSetSharedBacking_nid_postfix(SharedBackingResolver resolver);
void GuestArenaSetSharedBackingWriter_nid_no_patch(SharedBackingWriter writer);
bool GuestArenaWriteSharedBacking_nid_no_patch(std::uintptr_t address, const void* source, std::size_t bytes);
bool GuestArenaSharedBacking_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
void GuestArenaSetSharedBackingRuns_nid_postfix(SharedBackingRunsResolver resolver);
bool GuestArenaSharedBackingRuns_nid_postfix(std::uintptr_t address, std::size_t bytes, int* file, SharedBackingRun* runs, std::size_t capacity, std::size_t* count);
void GuestArenaSetSharedBackingHostFallback_nid_no_patch(SharedBackingHostFallback fallback);
bool GuestArenaSharedBackingHostFallback_nid_no_patch(std::uintptr_t address, std::size_t bytes);
bool GuestArenaOverlapsHostRegion_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaReleaseHostRegion_nid_no_patch(const void* pointer, std::size_t bytes);
#endif
bool GuestArenaArmOwned_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* words, std::size_t count, unsigned flags);
bool GuestArenaOwnedUnchanged_nid_no_patch(std::uintptr_t address, std::size_t bytes, const std::uint64_t* words, std::size_t count);
bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes);
std::uint64_t GuestArenaOpenHostWrite_nid_no_patch(void* pointer, std::size_t bytes);
void GuestArenaCloseHostWrite_nid_no_patch(void* pointer, std::size_t bytes, std::uint64_t token);
void GuestArenaSetHostWriteHook_nid_no_patch(void (*hook)(std::uintptr_t address, std::size_t bytes));
void GuestArenaSetHostReadHook_nid_no_patch(void (*hook)(std::uintptr_t address, std::size_t bytes));
void GuestArenaPrepareHostRead_nid_no_patch(const void* pointer, std::size_t bytes);
#if defined(ANYPS5_KYTY)
void GuestArenaSetHostWritePreparer_nid_no_patch(bool (*prepare)(std::uintptr_t address, std::size_t bytes));
#ifndef _WIN32
void GuestArenaSetHostWriteHold_nid_no_patch(void (*hold)(std::uintptr_t address, std::size_t bytes), bool (*release)(std::uintptr_t address, std::size_t bytes));
#endif
#ifdef _WIN32
void GuestArenaSetGpuFaultHandler_nid_no_patch(bool (*handler)(void* exceptionPointers));
bool GuestArenaHandleGpuFault_nid_no_patch(void* exceptionPointers);
#endif
#if defined(ANYPS5_KYTY)
void GuestArenaNoteHostProtection_nid_no_patch(std::uintptr_t address, std::size_t bytes, bool writable);
bool GuestArenaMayBeHostProtected_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaSetHostWriteFilter_nid_no_patch(bool (*filter)(std::uintptr_t address, std::size_t bytes));
void GuestArenaSetViewPlacer_nid_no_patch(void* (*placer)(void* pointer, std::size_t bytes, int protection));
bool GuestArenaPreparingHostWrite_nid_no_patch();
#endif
#endif

}

class HostWrite {
public:
    HostWrite(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), token(GuestArenaOpenHostWrite_nid_no_patch(pointer, bytes)) {}
    ~HostWrite() {
        if (token != 0) GuestArenaCloseHostWrite_nid_no_patch(pointer, bytes, token);
    }
    HostWrite(const HostWrite&) = delete;
    HostWrite& operator=(const HostWrite&) = delete;
    bool Open() const { return token != 0; }

private:
    void* pointer;
    std::size_t bytes;
    std::uint64_t token;
};

#ifdef _WIN32
class HostImportAccess {
public:
    HostImportAccess(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), open(GuestArenaBeginHostImport_nid_no_patch(pointer, bytes)) {}
    ~HostImportAccess() {
        if (open) GuestArenaEndHostImport_nid_no_patch(pointer, bytes);
    }
    HostImportAccess(const HostImportAccess&) = delete;
    HostImportAccess& operator=(const HostImportAccess&) = delete;

private:
    void* pointer;
    std::size_t bytes;
    bool open;
};
#endif

}

#endif
