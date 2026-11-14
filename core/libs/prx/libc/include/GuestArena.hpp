#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTARENA_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

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

#ifndef _WIN32
using SharedBackingResolver = bool (*)(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset);
using SharedBackingWriter = bool (*)(std::uintptr_t address, const void* source, std::size_t bytes);
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
bool GuestArenaHandleWrite_nid_postfix(std::uintptr_t address, std::uintptr_t instruction = 0);
bool GuestArenaHandleRead_nid_postfix(std::uintptr_t address);
// Called once the libc exception handler is installed: write faults on armed guest memory are then
// resumed, so collects may arm consumed private and resident memory (section A of the rewrite).
void GuestArenaEnableWriteFaults_nid_no_patch();
// Section A counters, in this order: collect walks (GetWriteWatch calls), armed runs, write faults
// on armed blocks, arm attempts skipped as hot, armed runs a store raced, armed bytes, armed
// ranges, submissions, bytes armed at submissions, import groups opened. Returns how many values
// it wrote.
std::size_t GuestArenaProtectCounts_nid_no_patch(std::uint64_t* values, std::size_t count);
// Around every vkQueueSubmit: makes each host import the submission stores through uniformly
// protected (a submission's GPU writes into an import partly read-only at the call can be dropped)
// and keeps it so until the call ends. stores: the guest ranges stored into through store imports
// (their groups of store imports are made uniform). regular: the guest ranges stored into through
// any other import (the armed pages of their import groups are opened). all: the stores are not
// known, every armed page inside an import is opened.
void GuestArenaBeginSubmit_nid_no_patch(const std::pair<std::uint64_t, std::uint64_t>* stores, std::size_t count, const std::pair<std::uint64_t, std::uint64_t>* regular, std::size_t regularCount, bool all);
void GuestArenaEndSubmit_nid_no_patch();
// Diagnostic only: a guest range the GPU will write (counted per import group under the walk probe).
void GuestArenaGpuWrites_nid_no_patch(std::uintptr_t address, std::size_t bytes, unsigned kind);
// A host import of guest pages (no alias) was made, known until
// GuestArenaForgetImport_nid_no_patch (same range and store, at its destruction). store: a store
// import, which submissions naming stores into it keep uniformly protected. With used false its
// pages stay writable until GuestArenaImportUsed_nid_no_patch (same range, after its first GPU
// use ended).
void GuestArenaRegisterImport_nid_no_patch(std::uintptr_t address, std::size_t bytes, bool used, bool store);
void GuestArenaImportUsed_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaForgetImport_nid_no_patch(std::uintptr_t address, std::size_t bytes, bool store);
// Changes whenever armed (CPU-clean) guest memory was opened to writes. Lock-free.
std::uint64_t GuestArenaArmedOpens_nid_no_patch();
// Whether the arena arms guest memory read-only (section A is on for this process).
bool GuestArenaArming_nid_no_patch();
// Guest memory a thread runs on as its stack: never armed read-only (an exception cannot be
// delivered on a read-only stack).
void GuestArenaNeverArm_nid_no_patch(std::uintptr_t address, std::size_t bytes);
// Whether the whole range is clean since its last resetting collect (a collect would report
// nothing): the dirty-event answer that lets the driver skip a walk.
bool GuestArenaClean_nid_no_patch(std::uintptr_t address, std::size_t bytes);
bool GuestArenaCopyOwned_nid_no_patch(std::uintptr_t source, void* destination, std::size_t bytes);
bool GuestArenaCompareOwned_nid_no_patch(std::uintptr_t source, const void* expected, std::size_t bytes, bool* equal);
bool GuestArenaRecoverRead_nid_no_patch(_EXCEPTION_POINTERS* exception);
void GuestArenaPinWritable_nid_postfix(const void* pointer, std::size_t bytes);
void GuestArenaUnpinWritable_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestArenaProtection_nid_postfix(std::uintptr_t address, std::uint32_t* protection);
bool GuestArenaCollectWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectArmedWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
bool GuestArenaCollectFreshWrites_nid_postfix(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear);
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
bool GuestArenaOverlapsHostRegion_nid_no_patch(std::uintptr_t address, std::size_t bytes);
void GuestArenaReleaseHostRegion_nid_no_patch(const void* pointer, std::size_t bytes);
#endif
bool GuestArenaArmOwned_nid_no_patch(std::uintptr_t address, std::size_t bytes, std::uint64_t* words, std::size_t count, unsigned flags);
bool GuestArenaOwnedUnchanged_nid_no_patch(std::uintptr_t address, std::size_t bytes, const std::uint64_t* words, std::size_t count);
bool GuestArenaBeginHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaEndHostWrite_nid_postfix(void* pointer, std::size_t bytes);
void GuestArenaSetHostWriteHook_nid_no_patch(void (*hook)(std::uintptr_t address, std::size_t bytes));
void GuestArenaSetHostReadHook_nid_no_patch(void (*hook)(std::uintptr_t address, std::size_t bytes));
void GuestArenaPrepareHostRead_nid_no_patch(const void* pointer, std::size_t bytes);

}

class HostWrite {
public:
    HostWrite(void* pointer, std::size_t bytes) : pointer(pointer), bytes(bytes), open(GuestArenaBeginHostWrite_nid_postfix(pointer, bytes)) {}
    ~HostWrite() {
        if (open) GuestArenaEndHostWrite_nid_postfix(pointer, bytes);
    }
    HostWrite(const HostWrite&) = delete;
    HostWrite& operator=(const HostWrite&) = delete;
    bool Open() const { return open; }

private:
    void* pointer;
    std::size_t bytes;
    bool open;
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

// Held around a vkQueueSubmit (GuestArenaBeginSubmit_nid_no_patch). Nothing else runs inside it.
class Submission {
public:
    // NoStores: the submission stores into no guest memory through an import (reads, device-local
    // work). The default constructor is for submissions whose stores are not known.
    struct NoStores {};
#ifdef _WIN32
    Submission() { GuestArenaBeginSubmit_nid_no_patch(nullptr, 0, nullptr, 0, true); }
    explicit Submission(NoStores) { GuestArenaBeginSubmit_nid_no_patch(nullptr, 0, nullptr, 0, false); }
    // stores: ranges stored into through store imports; regular: through any other import.
    Submission(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& stores, const std::vector<std::pair<std::uint64_t, std::uint64_t>>& regular) { GuestArenaBeginSubmit_nid_no_patch(stores.data(), stores.size(), regular.data(), regular.size(), false); }
    ~Submission() { GuestArenaEndSubmit_nid_no_patch(); }
#else
    Submission() {}
    explicit Submission(NoStores) {}
    Submission(const std::vector<std::pair<std::uint64_t, std::uint64_t>>&, const std::vector<std::pair<std::uint64_t, std::uint64_t>>&) {}
#endif
    Submission(const Submission&) = delete;
    Submission& operator=(const Submission&) = delete;
};

}

#endif
