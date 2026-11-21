#ifndef CORE_LIBS_PRX_LIBKERNEL_DIRECTMEMORY_DIRECTMEMORY_HPP
#define CORE_LIBS_PRX_LIBKERNEL_DIRECTMEMORY_DIRECTMEMORY_HPP

#include <cstdint>
#include <cstddef>
#include <mutex>
#include <vector>

#include "prx/libkernel/KernelErrors.hpp"

static constexpr size_t DIRECT_MEMORY_SIZE = 13824ULL * 1024 * 1024;
static constexpr size_t GAME_MEMORY_SIZE = 12736ULL * 1024 * 1024;
static constexpr size_t PS5_PAGE_SIZE = 0x4000;

#ifdef _WIN32
using HostSharedMemory = void*;
#else
using HostSharedMemory = int;
#endif

size_t DirectMemorySize();
int DirectMemoryAlloc(int64_t searchStart, int64_t searchEnd, size_t len, size_t alignment, int memoryType, int64_t* physOut);
void DirectMemoryFree(int64_t start, size_t len);
bool DirectMemoryCheckedFree(int64_t start, size_t len);
bool DirectMemoryClaimRelease(int64_t start, size_t len, bool requireAllocated);
void DirectMemoryFinishRelease(int64_t start, size_t len);
void DirectMemoryAbandonRelease(int64_t start, size_t len) noexcept;
void DirectMemoryWakeReleaseClaims();
void CreateDirectMemoryBacking(int64_t start, size_t len);
bool QueryDirectMapping(std::uintptr_t address, std::uintptr_t* start, std::uintptr_t* end, std::uint64_t* offset, int* memoryType);
bool DirectMemoryAliased(std::uintptr_t address, std::size_t bytes);
void ForgetDirectMemory(int64_t start, size_t len);

struct DirectMemoryView {
    std::uintptr_t address;
    std::size_t bytes;
};

void UnmapDirectMemoryRange(int64_t start, size_t len);
bool DirectMemoryReleaseClaimed(std::uint64_t first, std::uint64_t last, std::uint64_t* claimEnd);
bool DirectMemoryTryAddReleaseClaim(std::uint64_t first, std::uint64_t last);
void DirectMemoryRemoveReleaseClaim(std::uint64_t first);
bool DirectMemoryFind(int64_t offset, bool findNext, int64_t* start, int64_t* end, int* memoryType);
void DirectMemoryRetype(int64_t start, size_t len, int memoryType);
std::unique_lock<std::mutex> DirectMemoryPoolLock();
void DirectMemoryRetypeLocked(const std::unique_lock<std::mutex>& pool, int64_t start, size_t len, int memoryType);
void DirectMemoryPauseBeforeRetypeForTest(void (*pause)());
int DirectMemoryLargestFree(int64_t searchStart, int64_t searchEnd, size_t alignment, int64_t* startOut, size_t* sizeOut);
size_t DirectMemoryFreeRun(uint64_t offset, uint64_t limit);
int DoMapDirect(void** addr, size_t len, int prot, int flags, int64_t physStart, size_t alignment);
int DoMapAnon(void** addr, size_t len, int prot, int flags, size_t alignment = PS5_PAGE_SIZE, bool exclusive = false, bool allowStackReplacement = false);
int DoMapFlexibleAnon(void** addr, size_t len, int prot, int flags);
int DoMapZero(void** addr, size_t len, int prot, int flags);
int DoMapFileBacked_nid_no_patch(void** addr, size_t len, int prot, int flags, int descriptor, std::uint64_t offset, size_t alignment);
void ReleaseFlexibleMappingRange(std::uintptr_t address, size_t len);
int DoMapShared(void** addr, size_t len, int prot, HostSharedMemory object);
int DoMprotect(const void* addr, size_t len, int prot, bool requireMappedStart = true);
int DoMtypeprotect(const void* addr, size_t len, int type, int prot, bool requireMappedStart = true);
int DoMunmap(void* addr, size_t len);
int DoReserveVirtual(void** addr, size_t len, int flags, size_t alignment);
bool GuestProtection(uintptr_t addr, int* prot);
bool GuestReservation(std::uintptr_t addr, std::uintptr_t* start, std::uintptr_t* end);
bool GuestRangeMapped(std::uintptr_t start, std::uintptr_t end);
void ForgetJitSharedMemory(int descriptor);
void DuplicateJitSharedMemory(int from, int to);
bool IsJitSharedMemory(int descriptor);
void GuestPageTableUsage(std::uint64_t granularity, std::uint64_t* cpu_entries, std::uint64_t* gpu_entries);

#if defined(ANYPS5_KYTY)
inline constexpr std::uint64_t DirectMemoryViewUnmapped = ~std::uint64_t{0};
inline constexpr std::uint64_t DirectMemoryViewPrivate = ~std::uint64_t{0} - 1;
inline constexpr std::uint64_t DirectMemoryViewReplacing = ~std::uint64_t{0} - 2;

extern "C" {
using DirectMemoryViewObserver = void (*)(std::uintptr_t address, std::size_t bytes, std::uint64_t viewOffset);
bool DirectMemoryEnableView_nid_no_patch(DirectMemoryViewObserver observer, std::uint8_t** view, std::size_t* guestBytes, std::size_t* viewBytes);
bool DirectMemoryViewFiles_nid_no_patch(int* guestFile, int* directFile);
using DirectMemoryProtectionObserver = void (*)(std::uintptr_t address, std::size_t bytes, int protection);
void DirectMemoryWatchProtections_nid_no_patch(DirectMemoryProtectionObserver observer);
std::size_t DirectMemorySize_nid_no_patch();
bool DirectMemoryFind_nid_no_patch(std::int64_t offset, bool findNext, std::int64_t* start, std::int64_t* end, int* memoryType);
void GuestPageTableUsage_nid_no_patch(std::uint64_t granularity, std::uint64_t* cpuEntries, std::uint64_t* gpuEntries);
int DoMapAnon_nid_no_patch(void** addr, std::size_t len, int prot, int flags, std::size_t alignment, bool exclusive);
int DoMunmap_nid_no_patch(void* addr, std::size_t len);
void* DirectMemoryMapHostView_nid_no_patch(std::uintptr_t address, std::size_t bytes, int prot);
#ifdef _WIN32
bool DirectMemoryCommitGuestFile_nid_no_patch(void* addr, std::size_t len, std::uint32_t winProt);
#endif
}
#endif

#endif
