#include "prx/libSceAgcDriver/KytyShim/src/Faults.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/MemoryShim.hpp"

#include "common/debugCounters.h"
#include "common/guardedCopy.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"

#include "prx/libSceAgcDriver/KytyShim/src/FaultReplay.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <ucontext.h>
#endif


namespace KytyShim {
namespace {

using GpuAccess = Libs::Graphics::PageFaultAccess;

#ifdef _WIN32
using FaultContext = CONTEXT;

void CompleteWalkerLoad(FaultContext& context, std::uint32_t length, std::uint32_t value) {
    context.Rax = value;
    context.Rip += length;
}

void RearmGuardPage(std::uintptr_t address) {
    MEMORY_BASIC_INFORMATION page{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &page, sizeof(page)) != sizeof(page) || page.State != MEM_COMMIT || (page.Protect & PAGE_GUARD) != 0) return;
    DWORD previous = 0;
    VirtualProtect(page.BaseAddress, 1, page.Protect | PAGE_GUARD, &previous);
}

std::atomic<bool> g_single_step_ready {false};
#else
using FaultContext = ucontext_t;

constexpr std::uint64_t PageFaultWrite = 1u << 1;
constexpr std::uint64_t PageFaultFetch = 1u << 4;

GpuAccess DecodeAccess(const ucontext_t& context) {
    const auto error = static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_ERR]);
    if ((error & PageFaultFetch) != 0) return GpuAccess::Execute;
    if ((error & PageFaultWrite) != 0) return GpuAccess::Write;
    return GpuAccess::Read;
}

void CompleteWalkerLoad(FaultContext& context, std::uint32_t length, std::uint32_t value) {
    context.uc_mcontext.gregs[REG_RAX] = static_cast<greg_t>(value);
    context.uc_mcontext.gregs[REG_RIP] += static_cast<greg_t>(length);
}

struct sigaction g_previous_trap {};
bool             g_trap_installed = false;
#endif

bool SwitchOn(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

bool ReplayReady() {
#ifdef _WIN32
    const bool stepping = g_single_step_ready.load(std::memory_order_acquire);
#else
    const bool stepping = g_trap_installed;
#endif
    return stepping && !FaultReplay::g_broken.load(std::memory_order_relaxed) && !Libs::Graphics::GuestGpu::IsGpuThread() &&
           !Libs::Graphics::GuestGpu::IsCommandProcessorThread();
}

bool ReplayEnabled() {
    static const bool on = SwitchOn("KYTY_REPLAY_GPU_PAGE_READS");
    return on && ReplayReady();
}

bool TryReplayOnBacking(FaultContext& context, std::uint64_t address) {
    namespace Memory = Libs::LibKernel::Memory;
    if (!ReplayEnabled()) return false;
    const auto replayable = [](std::uint64_t ea, std::uint64_t size) { return Memory::ReplayableGpuRead(ea, size); };
    const auto backing    = [](std::uint64_t ea, std::uint64_t size) { return Memory::FindBackingPointer(ea, size); };
    if (!FaultReplay::Start(context, address, replayable, backing)) return false;
    Memory::NoteReplayedGpuRead();
    return true;
}

bool ReplayAfterReadbackEnabled() {
    static const bool on = SwitchOn("KYTY_REPLAY_AFTER_READBACK");
    return on && ReplayReady();
}

void FinishReplayAfterReadback(FaultContext& context, std::uint64_t address) {
    namespace Memory = Libs::LibKernel::Memory;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
    if (!Memory::TakeReplayAfterReadback(&begin, &end)) return;
    const auto replayable = [begin, end](std::uint64_t ea, std::uint64_t size) {
        return ea + size <= end && (ea >= begin || Memory::ReplayableGpuRead(ea, begin - ea)) && !Memory::TextureGpuModified(ea, size);
    };
    const auto backing = [](std::uint64_t ea, std::uint64_t size) { return Memory::FindBackingPointer(ea, size); };
    Memory::FinishReplayAfterReadback(FaultReplay::Start(context, address, replayable, backing));
}

#ifndef _WIN32
void OnTrap(int signal, siginfo_t* info, void* native) {
    if (native != nullptr && FaultReplay::Finish(*static_cast<ucontext_t*>(native))) return;
    if ((g_previous_trap.sa_flags & SA_SIGINFO) != 0) {
        if (g_previous_trap.sa_sigaction != nullptr) {
            g_previous_trap.sa_sigaction(signal, info, native);
            return;
        }
    } else if (g_previous_trap.sa_handler == SIG_IGN) {
        return;
    } else if (g_previous_trap.sa_handler != SIG_DFL && g_previous_trap.sa_handler != nullptr) {
        g_previous_trap.sa_handler(signal);
        return;
    }
    ::signal(SIGTRAP, SIG_DFL);
    ::raise(SIGTRAP);
}
#endif

bool PrepareHostWrite(std::uintptr_t address, std::size_t bytes) {
    return Libs::LibKernel::Memory::TryPrepareHostWrite(address, bytes);
}

bool ResolveFault(FaultContext& context, GpuAccess access, std::uint64_t address, std::uint64_t rip) {
    namespace Memory = Libs::LibKernel::Memory;
    namespace Walker = Libs::Graphics::ShaderRecompiler::IR;
    if (access == GpuAccess::Read) {
        if (const auto length = Walker::SrtWalkerLoadLength(rip); length != 0) {
            std::uint32_t value = 0;
            if (Memory::MayFaultOnGpuRead(address, sizeof(value)) && Memory::TryReadAroundGpuWrites(address, &value, sizeof(value))) {
                CompleteWalkerLoad(context, length, value);
                return true;
            }
        }
    }
    if (const auto length = Walker::SrtWalkerAheadLoadLength(rip, address); length != 0) {
        CompleteWalkerLoad(context, length, 0);
        return true;
    }
    if (access == GpuAccess::Read && TryReplayOnBacking(context, address)) {
        return true;
    }
    const bool after = access == GpuAccess::Read && ReplayAfterReadbackEnabled();
    if (after) Memory::AllowReplayAfterReadback();
    if (Memory::HandleGpuFault(access, address)) {
        Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuFaults);
        if (after) FinishReplayAfterReadback(context, address);
        return true;
    }
    if (after) FinishReplayAfterReadback(context, address);
    if (const auto length = Walker::SrtWalkerLoadLength(rip); length != 0) {
        CompleteWalkerLoad(context, length, 0);
        return true;
    }
    return false;
}

}

#ifdef _WIN32
bool HandleHostFault(void* exceptionPointers) {
    auto* exception = static_cast<EXCEPTION_POINTERS*>(exceptionPointers);
    if (exception == nullptr || exception->ExceptionRecord == nullptr || exception->ContextRecord == nullptr) return false;
    const auto& record = *exception->ExceptionRecord;
    if (record.ExceptionCode == EXCEPTION_SINGLE_STEP) return FaultReplay::Finish(*exception->ContextRecord);
    const bool guardPage = record.ExceptionCode == STATUS_GUARD_PAGE_VIOLATION;
    if ((record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION && !guardPage) || record.NumberParameters < 2) return false;
    const auto access = record.ExceptionInformation[0] == 8 ? GpuAccess::Execute : record.ExceptionInformation[0] == 1 ? GpuAccess::Write : GpuAccess::Read;
    if (!guardPage && Common::ResumeGuardedCopy(exception->ContextRecord)) return true;
    const DWORD lastError = GetLastError();
    const bool handled = ResolveFault(*exception->ContextRecord, access, static_cast<std::uint64_t>(record.ExceptionInformation[1]), static_cast<std::uint64_t>(exception->ContextRecord->Rip));
    if (handled && guardPage) RearmGuardPage(static_cast<std::uintptr_t>(record.ExceptionInformation[1]));
    SetLastError(lastError);
    return handled;
}

void InstallFaultChain() {
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&PrepareHostWrite);
    GuestArena::GuestArenaSetGpuFaultHandler_nid_no_patch(&HandleHostFault);
    g_single_step_ready.store(true, std::memory_order_release);
}

void RemoveFaultChain() {
    g_single_step_ready.store(false, std::memory_order_release);
    GuestArena::GuestArenaSetGpuFaultHandler_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
}
#else
bool HandleHostFault(int signal, siginfo_t* info, void* native) {
    if ((signal != SIGSEGV && signal != SIGBUS) || info == nullptr || native == nullptr) return false;
    auto& context = *static_cast<ucontext_t*>(native);
    return ResolveFault(context, DecodeAccess(context), reinterpret_cast<std::uint64_t>(info->si_addr), static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_RIP]));
}

void InstallFaultChain() {
    if (!g_trap_installed) {
        struct sigaction action {};
        action.sa_sigaction = &OnTrap;
        action.sa_flags     = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        g_trap_installed = ::sigaction(SIGTRAP, &action, &g_previous_trap) == 0;
    }
    GuestArena::GuestArenaSetHostWriteHold_nid_no_patch(&Libs::LibKernel::Memory::Shim::HoldHostWrite, &Libs::LibKernel::Memory::Shim::ReleaseHostWrite);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&PrepareHostWrite);
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(&HandleHostFault);
}

void RemoveFaultChain() {
    if (g_trap_installed) {
        ::sigaction(SIGTRAP, &g_previous_trap, nullptr);
        g_trap_installed = false;
    }
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWriteHold_nid_no_patch(nullptr, nullptr);
}
#endif

}
