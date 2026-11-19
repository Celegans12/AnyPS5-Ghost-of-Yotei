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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
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

void RearmGuardPage(std::uintptr_t address) {
    MEMORY_BASIC_INFORMATION page{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &page, sizeof(page)) != sizeof(page) || page.State != MEM_COMMIT || (page.Protect & PAGE_GUARD) != 0) return;
    DWORD previous = 0;
    VirtualProtect(page.BaseAddress, 1, page.Protect | PAGE_GUARD, &previous);
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

constexpr int GeneralRegisters[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                      REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
constexpr std::uint64_t TrapFlag    = 0x100;
constexpr std::uint64_t ReplaySpan  = 64;
constexpr std::size_t   ReplaySlots = 1024;

struct MemoryOperand {
    int           base  = -1;
    int           index = -1;
    std::uint32_t scale = 0;
    std::int64_t  disp  = 0;
};

struct ReplaySlot {
    std::atomic<std::int64_t> thread {0};
    int                       reg      = 0;
    std::uint64_t             delta    = 0;
    std::uint64_t             expected = 0;
    bool                      trap     = false;
};

ReplaySlot                g_replays[ReplaySlots];
std::atomic<bool>         g_replay_broken {false};
struct sigaction          g_previous_trap {};
bool                      g_trap_installed = false;

bool HasModRmOneByte(std::uint8_t op) {
    if (op < 0x40) return (op & 7u) < 4u;
    return op == 0x63 || op == 0x69 || op == 0x6B || (op >= 0x80 && op <= 0x8F) || op == 0xC0 || op == 0xC1 || op == 0xC6 ||
           op == 0xC7 || (op >= 0xD0 && op <= 0xD3) || (op >= 0xD8 && op <= 0xDF) || op == 0xF6 || op == 0xF7 || op == 0xFE ||
           op == 0xFF;
}

bool HasModRmTwoByte(std::uint8_t op) {
    if ((op >= 0x05 && op <= 0x09) || op == 0x0B || op == 0x0E || (op >= 0x30 && op <= 0x37) || op == 0x77) return false;
    if ((op >= 0x80 && op <= 0x8F) || (op >= 0xA0 && op <= 0xA2) || (op >= 0xA8 && op <= 0xAA) || (op >= 0xC8 && op <= 0xCF)) return false;
    return true;
}

bool DecodeMemoryOperand(const std::uint8_t* code, MemoryOperand& out) {
    std::size_t i = 0;
    for (; i < 14; i++) {
        const auto b = code[i];
        if (b == 0x66 || b == 0xF0 || b == 0xF2 || b == 0xF3 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x26) continue;
        if (b == 0x64 || b == 0x65 || b == 0x67) return false;
        break;
    }
    if (i >= 14) return false;
    bool rex   = false;
    bool rex_x = false;
    bool rex_b = false;
    if ((code[i] & 0xF0u) == 0x40u) {
        rex   = true;
        rex_x = (code[i] & 2u) != 0;
        rex_b = (code[i] & 1u) != 0;
        i++;
    }
    const auto b        = code[i];
    bool       modrm    = false;
    bool       implicit = false;
    if (b == 0xC5) {
        if (rex) return false;
        const auto opcode = code[i + 2];
        i += 3;
        modrm = opcode != 0x77;
    } else if (b == 0xC4) {
        if (rex) return false;
        const auto payload = code[i + 1];
        rex_x              = (payload & 0x40u) == 0;
        rex_b              = (payload & 0x20u) == 0;
        const auto map     = payload & 0x1Fu;
        const auto opcode  = code[i + 3];
        if (map < 1 || map > 3 || (map == 2 && opcode >= 0x90 && opcode <= 0x93)) return false;
        implicit = (map == 2 && opcode == 0xF6) || (map == 3 && (opcode == 0x60 || opcode == 0x61));
        i += 4;
        modrm = !(map == 1 && opcode == 0x77);
    } else if (b == 0x62) {
        return false;
    } else if (b == 0x0F) {
        const auto next = code[i + 1];
        if (next == 0x38 || next == 0x3A) {
            const auto opcode = code[i + 2];
            implicit          = (next == 0x38 && opcode == 0xF6) || (next == 0x3A && (opcode == 0x60 || opcode == 0x61));
            i += 3;
            modrm = true;
        } else {
            i += 2;
            modrm = HasModRmTwoByte(next);
        }
    } else {
        if (b == 0x8D) return false;
        i += 1;
        modrm = HasModRmOneByte(b);
    }
    if (!modrm) return false;
    const auto byte = code[i++];
    const auto mod  = byte >> 6u;
    const auto rm   = byte & 7u;
    if (mod == 3 || implicit || ((b == 0xF6 || b == 0xF7) && ((byte >> 3u) & 7u) >= 4)) return false;
    out = {};
    if (rm == 4) {
        const auto sib   = code[i++];
        const int  index = static_cast<int>(((sib >> 3u) & 7u) | (rex_x ? 8u : 0u));
        out.scale        = sib >> 6u;
        out.index        = index == 4 ? -1 : index;
        if ((sib & 7u) == 5 && mod == 0) {
            std::int32_t disp = 0;
            std::memcpy(&disp, code + i, sizeof(disp));
            out.disp = disp;
            return true;
        }
        out.base = static_cast<int>((sib & 7u) | (rex_b ? 8u : 0u));
    } else if (rm == 5 && mod == 0) {
        return false;
    } else {
        out.base = static_cast<int>(rm | (rex_b ? 8u : 0u));
    }
    if (mod == 1) {
        out.disp = static_cast<std::int8_t>(code[i]);
    } else if (mod == 2) {
        std::int32_t disp = 0;
        std::memcpy(&disp, code + i, sizeof(disp));
        out.disp = disp;
    }
    return true;
}

std::int64_t CurrentThread() {
    return static_cast<std::int64_t>(::syscall(SYS_gettid));
}

ReplaySlot* FindReplay(std::int64_t thread) {
    for (std::size_t probe = 0; probe < ReplaySlots; probe++) {
        auto& slot = g_replays[(static_cast<std::size_t>(thread) + probe) % ReplaySlots];
        if (slot.thread.load(std::memory_order_acquire) == thread) return &slot;
    }
    return nullptr;
}

ReplaySlot* ClaimReplay(std::int64_t thread) {
    for (std::size_t probe = 0; probe < ReplaySlots; probe++) {
        auto&        slot = g_replays[(static_cast<std::size_t>(thread) + probe) % ReplaySlots];
        std::int64_t free = 0;
        if (slot.thread.compare_exchange_strong(free, thread, std::memory_order_acq_rel)) return &slot;
    }
    return nullptr;
}

bool FinishReplay(FaultContext& context) {
    auto* slot = FindReplay(CurrentThread());
    if (slot == nullptr) return false;
    auto& reg = context.uc_mcontext.gregs[slot->reg];
    if (static_cast<std::uint64_t>(reg) == slot->expected) reg = static_cast<greg_t>(static_cast<std::uint64_t>(reg) - slot->delta);
    if (!slot->trap) context.uc_mcontext.gregs[REG_EFL] &= ~static_cast<greg_t>(TrapFlag);
    slot->thread.store(0, std::memory_order_release);
    return true;
}

bool ReplayEnabled() {
    static const bool on = [] {
        const char* value = std::getenv("KYTY_REPLAY_GPU_PAGE_READS");
        return value == nullptr || value[0] != '0';
    }();
    return on && g_trap_installed && !g_replay_broken.load(std::memory_order_relaxed);
}

bool TryReplayOnBacking(FaultContext& context, std::uint64_t address, std::uint64_t rip) {
    namespace Memory = Libs::LibKernel::Memory;
    if (!ReplayEnabled() || Libs::Graphics::GuestGpu::IsGpuThread() || Libs::Graphics::GuestGpu::IsCommandProcessorThread()) {
        return false;
    }
    if (FindReplay(CurrentThread()) != nullptr) {
        (void)FinishReplay(context);
        g_replay_broken.store(true, std::memory_order_relaxed);
        return true;
    }
    MemoryOperand operand;
    if (!DecodeMemoryOperand(reinterpret_cast<const std::uint8_t*>(rip), operand)) return false;
    auto*      gregs = context.uc_mcontext.gregs;
    const auto value = [&](int reg) { return static_cast<std::uint64_t>(gregs[GeneralRegisters[reg]]); };
    const auto ea    = (operand.base >= 0 ? value(operand.base) : 0) +
                    (operand.index >= 0 ? value(operand.index) << operand.scale : 0) + static_cast<std::uint64_t>(operand.disp);
    if (ea > address || address - ea >= ReplaySpan) return false;
    int adjusted = -1;
    if (operand.base >= 0 && operand.base != 4 && operand.base != operand.index) {
        adjusted = operand.base;
    } else if (operand.base < 0 && operand.index >= 0 && operand.index != 4 && operand.scale == 0) {
        adjusted = operand.index;
    }
    if (adjusted < 0 || !Memory::ReplayableGpuRead(ea, ReplaySpan)) return false;
    const auto* backing = static_cast<const std::uint8_t*>(Memory::FindBackingPointer(ea, ReplaySpan));
    if (backing == nullptr) return false;
    auto* slot = ClaimReplay(CurrentThread());
    if (slot == nullptr) return false;
    const auto delta = reinterpret_cast<std::uint64_t>(backing) - ea;
    auto&      reg   = gregs[GeneralRegisters[adjusted]];
    reg              = static_cast<greg_t>(static_cast<std::uint64_t>(reg) + delta);
    slot->reg        = GeneralRegisters[adjusted];
    slot->delta      = delta;
    slot->expected   = static_cast<std::uint64_t>(reg);
    slot->trap       = (static_cast<std::uint64_t>(gregs[REG_EFL]) & TrapFlag) != 0;
    gregs[REG_EFL] |= static_cast<greg_t>(TrapFlag);
    Memory::NoteReplayedGpuRead();
    return true;
}

void OnTrap(int signal, siginfo_t* info, void* native) {
    if (native != nullptr && FinishReplay(*static_cast<ucontext_t*>(native))) return;
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
#ifndef _WIN32
    if (access == GpuAccess::Read && TryReplayOnBacking(context, address, rip)) {
        return true;
    }
#endif
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
