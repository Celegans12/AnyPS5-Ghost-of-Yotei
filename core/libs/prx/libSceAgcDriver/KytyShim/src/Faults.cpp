#include "prx/libSceAgcDriver/KytyShim/src/Faults.hpp"

#include "common/debugCounters.h"
#include "common/guardedCopy.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"

#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#include <windows.h>
#else
#include <ucontext.h>
#endif

#if !defined(ANYPS5_KYTY)
#error "KytyShim is built only with ANYPS5_GPU_BACKEND=kyty"
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
    if (Memory::HandleGpuFault(access, address)) {
        Common::DebugCounters::Add(Common::DebugCounters::Counter::GpuFaults);
        return true;
    }
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
    if (record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2) return false;
    const auto access = record.ExceptionInformation[0] == 8 ? GpuAccess::Execute : record.ExceptionInformation[0] == 1 ? GpuAccess::Write : GpuAccess::Read;
    if (Common::ResumeGuardedCopy(exception->ContextRecord)) return true;
    const DWORD lastError = GetLastError();
    const bool handled = ResolveFault(*exception->ContextRecord, access, static_cast<std::uint64_t>(record.ExceptionInformation[1]), static_cast<std::uint64_t>(exception->ContextRecord->Rip));
    SetLastError(lastError);
    return handled;
}

void InstallFaultChain() {
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&PrepareHostWrite);
    GuestArena::GuestArenaSetGpuFaultHandler_nid_no_patch(&HandleHostFault);
}

void RemoveFaultChain() {
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
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&PrepareHostWrite);
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(&HandleHostFault);
}

void RemoveFaultChain() {
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
}
#endif

}
