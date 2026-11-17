#include "prx/libSceAgcDriver/KytyShim/src/Faults.hpp"

#include "common/debugCounters.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"

#include <cstddef>
#include <cstdint>
#include <ucontext.h>

#if !defined(ANYPS5_KYTY)
#error "KytyShim is built only with ANYPS5_GPU_BACKEND=kyty"
#endif

namespace KytyShim {
namespace {

using GpuAccess = Libs::Graphics::PageFaultAccess;

constexpr std::uint64_t PageFaultWrite = 1u << 1;
constexpr std::uint64_t PageFaultFetch = 1u << 4;

GpuAccess DecodeAccess(const ucontext_t& context) {
    const auto error = static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_ERR]);
    if ((error & PageFaultFetch) != 0) return GpuAccess::Execute;
    if ((error & PageFaultWrite) != 0) return GpuAccess::Write;
    return GpuAccess::Read;
}

void CompleteWalkerLoad(ucontext_t& context, std::uint32_t length, std::uint32_t value) {
    context.uc_mcontext.gregs[REG_RAX] = static_cast<greg_t>(value);
    context.uc_mcontext.gregs[REG_RIP] += static_cast<greg_t>(length);
}

bool PrepareHostWrite(std::uintptr_t address, std::size_t bytes) {
    return Libs::LibKernel::Memory::TryPrepareHostWrite(address, bytes);
}

}

bool HandleHostFault(int signal, siginfo_t* info, void* native) {
    namespace Memory = Libs::LibKernel::Memory;
    namespace Walker = Libs::Graphics::ShaderRecompiler::IR;
    if ((signal != SIGSEGV && signal != SIGBUS) || info == nullptr || native == nullptr) return false;
    auto& context = *static_cast<ucontext_t*>(native);
    const auto address = reinterpret_cast<std::uint64_t>(info->si_addr);
    const auto rip = static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_RIP]);
    const auto access = DecodeAccess(context);
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

void InstallFaultChain() {
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&PrepareHostWrite);
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(&HandleHostFault);
}

void RemoveFaultChain() {
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
}

}
