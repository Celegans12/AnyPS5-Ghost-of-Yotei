#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>
#endif

namespace KytyShim::FaultReplay {

#ifdef _WIN32
using Context = CONTEXT;

inline std::uint64_t* Register(CONTEXT& context, int index) {
    DWORD64* const registers[16] = {&context.Rax, &context.Rcx, &context.Rdx, &context.Rbx, &context.Rsp, &context.Rbp,
                                    &context.Rsi, &context.Rdi, &context.R8,  &context.R9,  &context.R10, &context.R11,
                                    &context.R12, &context.R13, &context.R14, &context.R15};
    return reinterpret_cast<std::uint64_t*>(registers[index]);
}

inline std::uint64_t InstructionPointer(const CONTEXT& context) { return context.Rip; }
inline std::uint64_t StackPointer(const CONTEXT& context) { return context.Rsp; }
inline std::uint64_t Flags(const CONTEXT& context) { return context.EFlags; }
inline void          SetFlags(CONTEXT& context, std::uint64_t flags) { context.EFlags = static_cast<DWORD>(flags); }
#else
using Context = ucontext_t;

constexpr int GeneralRegisters[16] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
                                      REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};

inline std::uint64_t* Register(ucontext_t& context, int index) {
    return reinterpret_cast<std::uint64_t*>(&context.uc_mcontext.gregs[GeneralRegisters[index]]);
}

inline std::uint64_t InstructionPointer(const ucontext_t& context) { return static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_RIP]); }
inline std::uint64_t StackPointer(const ucontext_t& context) { return static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_RSP]); }
inline std::uint64_t Flags(const ucontext_t& context) { return static_cast<std::uint64_t>(context.uc_mcontext.gregs[REG_EFL]); }
inline void          SetFlags(ucontext_t& context, std::uint64_t flags) { context.uc_mcontext.gregs[REG_EFL] = static_cast<greg_t>(flags); }
#endif

constexpr std::uint64_t TrapFlag = 0x100;
constexpr std::uint64_t Span     = 64;
constexpr std::size_t   Slots    = 1024;

struct Operand {
    int           base  = -1;
    int           index = -1;
    std::uint32_t scale = 0;
    std::int64_t  disp  = 0;
    int           reg   = -1;
    int           vvvv  = -1;
};

struct Slot {
    std::atomic<std::int64_t> thread {0};
    int                       reg      = 0;
    std::uint64_t             delta    = 0;
    std::uint64_t             expected = 0;
    std::uint64_t             rip      = 0;
    std::uint64_t             rsp      = 0;
    bool                      trap     = false;
#ifndef _WIN32
    sigset_t                  mask {};
#endif
};

inline Slot              g_slots[Slots];
inline std::atomic<bool> g_broken {false};

enum class RegisterUse { None, FullWrite, Read };

struct Prefixes {
    bool operand16 = false;
    bool repne     = false;
    bool rep       = false;
};

inline bool HasModRmOneByte(std::uint8_t op) {
    if (op < 0x40) return (op & 7u) < 4u;
    return op == 0x63 || op == 0x69 || op == 0x6B || (op >= 0x80 && op <= 0x8F) || op == 0xC0 || op == 0xC1 || op == 0xC6 ||
           op == 0xC7 || (op >= 0xD0 && op <= 0xD3) || (op >= 0xD8 && op <= 0xDF) || op == 0xF6 || op == 0xF7 || op == 0xFE ||
           op == 0xFF;
}

inline bool HasModRmTwoByte(std::uint8_t op) {
    if ((op >= 0x05 && op <= 0x09) || op == 0x0B || op == 0x0E || (op >= 0x30 && op <= 0x37) || op == 0x77) return false;
    if ((op >= 0x80 && op <= 0x8F) || (op >= 0xA0 && op <= 0xA2) || (op >= 0xA8 && op <= 0xAA) || (op >= 0xC8 && op <= 0xCF)) return false;
    return true;
}

inline bool ClassifyOneByte(std::uint8_t op, std::uint32_t reg, const Prefixes& prefixes, RegisterUse& use, bool& byte_reg) {
    if (op < 0x40) {
        use      = RegisterUse::Read;
        byte_reg = (op & 1u) == 0;
        return (op & 7u) >= 2 || (op & 0x38u) == 0x38;
    }
    switch (op) {
        case 0x80:
        case 0x81:
        case 0x82:
        case 0x83: use = RegisterUse::None; return reg == 7;
        case 0x86:
        case 0x87:
        case 0x88:
        case 0x89:
        case 0x8F:
        case 0xC0:
        case 0xC1:
        case 0xC6:
        case 0xC7:
        case 0xD0:
        case 0xD1:
        case 0xD2:
        case 0xD3:
        case 0xFE: return false;
        case 0x63:
        case 0x69:
        case 0x6B:
        case 0x8B: use = prefixes.operand16 ? RegisterUse::Read : RegisterUse::FullWrite; return true;
        case 0x84:
        case 0x8A:
            use      = RegisterUse::Read;
            byte_reg = true;
            return true;
        case 0x85: use = RegisterUse::Read; return true;
        case 0x8C:
        case 0x8E: return false;
        case 0xDD: use = RegisterUse::None; return reg != 4;
        case 0xF6:
        case 0xF7: use = RegisterUse::None; return reg < 2;
        case 0xFF: use = RegisterUse::None; return reg == 2 || reg == 4 || reg == 6;
        default: use = RegisterUse::None; return true;
    }
}

inline bool ClassifyTwoByte(std::uint8_t op, std::uint32_t reg, const Prefixes& prefixes, RegisterUse& use) {
    use = RegisterUse::None;
    if (op == 0x0D || (op >= 0x18 && op <= 0x1F) || (op >= 0x50 && op <= 0x7F) || (op >= 0xD0 && op <= 0xFE)) return true;
    if ((op >= 0x10 && op <= 0x17) || (op >= 0x28 && op <= 0x2F)) {
        if ((op == 0x2C || op == 0x2D) && (prefixes.repne || prefixes.rep)) use = RegisterUse::FullWrite;
        return true;
    }
    if (op >= 0x40 && op <= 0x4F) {
        use = RegisterUse::Read;
        return true;
    }
    switch (op) {
        case 0xAE: return reg == 2;
        case 0xA3:
        case 0xAF: use = RegisterUse::Read; return true;
        case 0xB6:
        case 0xB7:
        case 0xBE:
        case 0xBF: use = prefixes.operand16 ? RegisterUse::Read : RegisterUse::FullWrite; return true;
        case 0xB8:
            use = prefixes.operand16 ? RegisterUse::Read : RegisterUse::FullWrite;
            return prefixes.rep;
        case 0xBC:
        case 0xBD: use = prefixes.rep && !prefixes.operand16 ? RegisterUse::FullWrite : RegisterUse::Read; return true;
        case 0xBA: return reg == 4;
        case 0xC2:
        case 0xC4:
        case 0xC5:
        case 0xC6: return true;
        default: return false;
    }
}

inline bool ClassifyThreeByte38(std::uint8_t op, const Prefixes& prefixes, RegisterUse& use) {
    use = RegisterUse::None;
    if (op < 0xF0) return true;
    if (op != 0xF0 && op != 0xF1) return false;
    use = op == 0xF0 && !prefixes.repne && !prefixes.operand16 ? RegisterUse::FullWrite : RegisterUse::Read;
    return true;
}

inline bool ClassifyVex(std::uint32_t map, std::uint8_t op, std::uint32_t reg, RegisterUse& use, bool& vvvv_read) {
    use = RegisterUse::None;
    if (map == 1) {
        if (op == 0x2C || op == 0x2D) use = RegisterUse::FullWrite;
        return op != 0xAE || reg == 2 || reg == 3;
    }
    if (map == 2) {
        if ((op >= 0x90 && op <= 0x93) || op == 0xF6) return false;
        if (op == 0xF2 || op == 0xF5 || op == 0xF7) {
            use       = RegisterUse::FullWrite;
            vvvv_read = true;
            return true;
        }
        if (op == 0xF3) return reg >= 1 && reg <= 3;
        return op < 0xF0;
    }
    if (map == 3) {
        if (op == 0x60 || op == 0x61) return false;
        if (op == 0xF0) use = RegisterUse::FullWrite;
        return op < 0xF0 || op == 0xF0;
    }
    return false;
}

inline bool Decode(const std::uint8_t* code, Operand& out) {
    std::size_t i = 0;
    Prefixes    prefixes;
    for (; i < 14; i++) {
        const auto b = code[i];
        if (b == 0x66) {
            prefixes.operand16 = true;
            continue;
        }
        if (b == 0xF2) {
            prefixes.repne = true;
            continue;
        }
        if (b == 0xF3) {
            prefixes.rep = true;
            continue;
        }
        if (b == 0xF0 || b == 0x2E || b == 0x36 || b == 0x3E || b == 0x26) continue;
        if (b == 0x64 || b == 0x65 || b == 0x67) return false;
        break;
    }
    if (i >= 14) return false;
    bool rex   = false;
    bool rex_r = false;
    bool rex_x = false;
    bool rex_b = false;
    if ((code[i] & 0xF0u) == 0x40u) {
        rex   = true;
        rex_r = (code[i] & 4u) != 0;
        rex_x = (code[i] & 2u) != 0;
        rex_b = (code[i] & 1u) != 0;
        i++;
    }
    const auto    b         = code[i];
    bool          vex       = false;
    std::uint32_t map       = 0;
    std::uint8_t  opcode    = 0;
    std::uint32_t vvvv      = 0;
    if (b == 0xC5) {
        if (rex) return false;
        vex    = true;
        map    = 1;
        rex_r  = (code[i + 1] & 0x80u) == 0;
        vvvv   = (~code[i + 1] >> 3u) & 0xFu;
        opcode = code[i + 2];
        i += 3;
    } else if (b == 0xC4) {
        if (rex) return false;
        vex    = true;
        rex_r  = (code[i + 1] & 0x80u) == 0;
        rex_x  = (code[i + 1] & 0x40u) == 0;
        rex_b  = (code[i + 1] & 0x20u) == 0;
        map    = code[i + 1] & 0x1Fu;
        vvvv   = (~code[i + 2] >> 3u) & 0xFu;
        opcode = code[i + 3];
        i += 4;
    } else if (b == 0x62 || b == 0x8D) {
        return false;
    } else if (b == 0x0F) {
        const auto next = code[i + 1];
        if (next == 0x38 || next == 0x3A) {
            map    = next == 0x38 ? 2u : 3u;
            opcode = code[i + 2];
            i += 3;
        } else {
            if (!HasModRmTwoByte(next)) return false;
            map    = 1;
            opcode = next;
            i += 2;
        }
    } else {
        if (!HasModRmOneByte(b)) return false;
        opcode = b;
        i += 1;
    }
    if (vex && map == 1 && opcode == 0x77) return false;
    const auto  byte      = code[i++];
    const auto  mod       = byte >> 6u;
    const auto  rm        = byte & 7u;
    const auto  reg_field = (byte >> 3u) & 7u;
    if (mod == 3) return false;
    RegisterUse use       = RegisterUse::None;
    bool        byte_reg  = false;
    bool        vvvv_read = false;
    bool        known     = false;
    if (vex) {
        known = ClassifyVex(map, opcode, reg_field, use, vvvv_read);
    } else if (map == 0) {
        known = ClassifyOneByte(opcode, reg_field, prefixes, use, byte_reg);
    } else if (map == 1) {
        known = ClassifyTwoByte(opcode, reg_field, prefixes, use);
    } else if (map == 2) {
        known = ClassifyThreeByte38(opcode, prefixes, use);
    } else {
        known = opcode != 0x60 && opcode != 0x61 && opcode < 0xF0;
    }
    if (!known) return false;
    out = {};
    if (use == RegisterUse::Read) {
        out.reg = byte_reg && !rex && reg_field >= 4 ? static_cast<int>(reg_field - 4) : static_cast<int>(reg_field | (rex_r ? 8u : 0u));
    }
    if (vvvv_read) out.vvvv = static_cast<int>(vvvv);
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

inline int ShiftedRegister(const Operand& operand) {
    int shifted = -1;
    if (operand.base >= 0 && operand.base != 4 && operand.base != operand.index) {
        shifted = operand.base;
    } else if (operand.base < 0 && operand.index >= 0 && operand.index != 4 && operand.scale == 0) {
        shifted = operand.index;
    }
    if (shifted < 0 || shifted == operand.reg || shifted == operand.vvvv) return -1;
    return shifted;
}

inline int ReplayRegister(const std::uint8_t* code) {
    Operand operand;
    return Decode(code, operand) ? ShiftedRegister(operand) : -1;
}

inline std::int64_t CurrentThread() {
#ifdef _WIN32
    return static_cast<std::int64_t>(GetCurrentThreadId());
#else
    return static_cast<std::int64_t>(::syscall(SYS_gettid));
#endif
}

inline Slot* Find(std::int64_t thread) {
    for (std::size_t probe = 0; probe < Slots; probe++) {
        auto& slot = g_slots[(static_cast<std::size_t>(thread) + probe) % Slots];
        if (slot.thread.load(std::memory_order_acquire) == thread) return &slot;
    }
    return nullptr;
}

inline Slot* Claim(std::int64_t thread) {
    for (std::size_t probe = 0; probe < Slots; probe++) {
        auto&        slot = g_slots[(static_cast<std::size_t>(thread) + probe) % Slots];
        std::int64_t free = 0;
        if (slot.thread.compare_exchange_strong(free, thread, std::memory_order_acq_rel)) return &slot;
    }
    return nullptr;
}

#ifndef _WIN32
inline void BlockAsynchronousSignals(sigset_t& mask) {
    std::memset(&mask, 0xFF, sizeof(mask));
    for (const int number : {SIGTRAP, SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        sigdelset(&mask, number);
    }
}
#endif

inline bool Finish(Context& context) {
    auto* slot = Find(CurrentThread());
    if (slot == nullptr) return false;
    auto* reg = Register(context, slot->reg);
    if (*reg == slot->expected) *reg -= slot->delta;
    SetFlags(context, slot->trap ? (Flags(context) | TrapFlag) : (Flags(context) & ~TrapFlag));
#ifndef _WIN32
    context.uc_sigmask = slot->mask;
#endif
    slot->thread.store(0, std::memory_order_release);
    return true;
}

template <typename Replayable, typename Backing>
bool Start(Context& context, std::uint64_t address, Replayable&& replayable, Backing&& backing) {
    const auto rip = InstructionPointer(context);
    const auto rsp = StackPointer(context);
    if (auto* active = Find(CurrentThread()); active != nullptr) {
        if (active->rip != rip || active->rsp != rsp) return false;
        (void)Finish(context);
        g_broken.store(true, std::memory_order_relaxed);
        return true;
    }
    Operand operand;
    if (!Decode(reinterpret_cast<const std::uint8_t*>(rip), operand)) return false;
    const auto value = [&](int reg) { return *Register(context, reg); };
    const auto ea    = (operand.base >= 0 ? value(operand.base) : 0) + (operand.index >= 0 ? value(operand.index) << operand.scale : 0) +
                    static_cast<std::uint64_t>(operand.disp);
    if (ea > address || address - ea >= Span) return false;
    const int shifted = ShiftedRegister(operand);
    if (shifted < 0 || !replayable(ea, Span)) return false;
    const auto target = reinterpret_cast<std::uint64_t>(backing(ea, Span));
    if (target == 0) return false;
    auto* slot = Claim(CurrentThread());
    if (slot == nullptr) return false;
    const auto delta = target - ea;
    auto*      reg   = Register(context, shifted);
    *reg += delta;
    slot->reg      = shifted;
    slot->delta    = delta;
    slot->expected = *reg;
    slot->rip      = rip;
    slot->rsp      = rsp;
    slot->trap     = (Flags(context) & TrapFlag) != 0;
#ifndef _WIN32
    slot->mask = context.uc_sigmask;
    BlockAsynchronousSignals(context.uc_sigmask);
#endif
    SetFlags(context, Flags(context) | TrapFlag);
    return true;
}

}
