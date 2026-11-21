#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTWRITEWATCH_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTWRITEWATCH_HPP

#include <cstddef>
#include <cstdint>
#if defined(__linux__)
#include <csignal>
#endif

namespace GuestWriteWatch {

struct ReadTrapCounts {
    std::uint64_t trapped = 0;
    std::uint64_t trappedPages = 0;
    std::uint64_t refused = 0;
    std::uint64_t released = 0;
    std::uint64_t faults = 0;
    std::uint64_t resolved = 0;
    std::uint64_t resolveNs = 0;
    std::uint64_t waitNs = 0;
    std::uint64_t bypassOpens = 0;
    std::uint64_t ownOpens = 0;
    std::uint64_t forcedOpens = 0;
    std::uint64_t retries = 0;
    std::uint64_t settles = 0;
    std::uint64_t settleOpens = 0;
    std::uint64_t holds = 0;
    std::uint64_t dormant = 0;
    std::uint64_t revived = 0;
    std::uint64_t swept = 0;
    std::uint64_t dormantOpens = 0;
};

struct ReadTrapFault {
    std::uintptr_t page = 0;
    std::uintptr_t instruction = 0;
    int thread = 0;
    bool write = false;
};

#if defined(__linux__) && defined(ANYPS5_KYTY)
using GpuFaultHandler = bool (*)(int signal, siginfo_t* info, void* context);
#endif

extern "C" {

bool GuestWriteWatchAvailable_nid_postfix();
const char* GuestWriteWatchUnavailableReason_nid_postfix();
void GuestWriteWatchRegister_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestWriteWatchUnregister_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestWriteWatchAlias_nid_postfix(const void* first, const void* second, std::size_t bytes);
bool GuestWriteWatchCovers_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestWriteWatchOverlaps_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestWriteWatchHandleFault_nid_postfix(std::uintptr_t address);
bool GuestWriteWatchCollect_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
void GuestWriteWatchBeginHostWrite_nid_postfix(const void* pointer, std::size_t bytes);
void GuestWriteWatchEndHostWrite_nid_postfix(const void* pointer, std::size_t bytes);
void GuestWriteWatchProtectionChanged_nid_postfix(const void* pointer, std::size_t bytes, int protection);
int GuestWriteWatchProtect_nid_postfix(void* pointer, std::size_t bytes, int protection);
int GuestWriteWatchProtection_nid_postfix(std::uintptr_t address, std::uintptr_t limit, std::uintptr_t* runEnd);
bool GuestWriteWatchCollectArmed_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
bool GuestWriteWatchCollectFresh_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
bool GuestWriteWatchSetReadTrapResolver_nid_postfix(void (*resolve)(std::uintptr_t begin, std::uintptr_t end));
void GuestWriteWatchSetReadTrapBypass_nid_postfix(bool (*bypass)());
bool GuestWriteWatchTrapReads_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestWriteWatchReadTraps_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*trapped)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);
void GuestWriteWatchReleaseReads_nid_postfix(std::uintptr_t address, std::size_t bytes);
bool GuestWriteWatchTrapWrites_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestWriteWatchReleaseWrites_nid_postfix(std::uintptr_t address, std::size_t bytes);
void GuestWriteWatchSweepReadTraps_nid_postfix();
void GuestWriteWatchSettleReads_nid_postfix(const void* pointer, std::size_t bytes);
void GuestWriteWatchOpenReads_nid_postfix(const void* pointer, std::size_t bytes);
int GuestWriteWatchOpenIdleReads_nid_postfix(const void* pointer, std::size_t bytes);
bool GuestWriteWatchReadTrapsHeld_nid_postfix();
std::uint64_t GuestWriteWatchReadTrapEpoch_nid_postfix();
void GuestWriteWatchReadTrapCounts_nid_postfix(ReadTrapCounts* counts);
std::size_t GuestWriteWatchReadTrapFaults_nid_postfix(ReadTrapFault* faults, std::size_t capacity);
#if defined(__linux__)
bool GuestWriteWatchHandleReadTrap_nid_postfix(int signal, siginfo_t* info, void* context);
#endif
#if defined(__linux__) && defined(ANYPS5_KYTY)
void GuestWriteWatchSetGpuFaultHandler_nid_no_patch(GpuFaultHandler handler);
#endif

}

}

#endif
