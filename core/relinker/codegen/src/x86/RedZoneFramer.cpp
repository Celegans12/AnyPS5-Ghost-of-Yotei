#include <codegen/x86/RedZoneFramer.hpp>
#include <codegen/IInstructionScanner.hpp>
#include <codegen/x86/X64InstructionDecoder.hpp>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace Codegen {

namespace {

constexpr std::uint8_t kRegRsp = 4;
constexpr std::uint8_t kRegRbp = 5;
constexpr std::size_t kJumpBytes = 5;
constexpr std::int64_t kRedZoneBytes = 0x80;
constexpr std::int64_t kStackAlignment = 16;
constexpr std::uint8_t kJumpBack[] = {0xe9, 0x00, 0x00, 0x00, 0x00};

// A return thunk's gap holds the leaf's whole frame: at most this many bytes, so that the stack
// touches it adds stay within a guard page of the caller's.
constexpr std::int64_t kThunkGapLimit = 0x800;
constexpr std::int64_t kJumpTableEntryLimit = 4096;

std::vector<std::uint8_t> lowerRsp(const std::int64_t shift) {
    if (shift <= 0x80) return {0x48, 0x8d, 0x64, 0x24, static_cast<std::uint8_t>(-shift)};
    const auto value = static_cast<std::uint32_t>(-shift);
    return {0x48, 0x8d, 0xa4, 0x24, static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value >> 16), static_cast<std::uint8_t>(value >> 24)};
}

std::vector<std::uint8_t> raiseRsp(const std::int64_t shift) {
    const auto value = static_cast<std::uint32_t>(shift);
    return {0x48, 0x8d, 0xa4, 0x24, static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value >> 16), static_cast<std::uint8_t>(value >> 24)};
}

struct Insn {
    std::size_t Offset;
    std::size_t Length;
    DecodedInstructionInfo Info;
};

std::optional<std::uint8_t> pushedRegister(const std::uint8_t* code, const Insn& insn) {
    if (insn.Length == 1 && code[0] >= 0x50 && code[0] <= 0x57 && code[0] != 0x50 + kRegRsp) return static_cast<std::uint8_t>(code[0] - 0x50);
    if (insn.Length == 2 && code[0] == 0x41 && code[1] >= 0x50 && code[1] <= 0x57) return static_cast<std::uint8_t>(code[1] - 0x50 + 8);
    return std::nullopt;
}

std::optional<std::uint8_t> poppedRegister(const std::uint8_t* code, const Insn& insn) {
    if (insn.Length == 1 && code[0] >= 0x58 && code[0] <= 0x5f && code[0] != 0x58 + kRegRsp) return static_cast<std::uint8_t>(code[0] - 0x58);
    if (insn.Length == 2 && code[0] == 0x41 && code[1] >= 0x58 && code[1] <= 0x5f) return static_cast<std::uint8_t>(code[1] - 0x58 + 8);
    return std::nullopt;
}

bool pushesOrPops(const std::uint8_t* code, const Insn& insn) {
    if (pushedRegister(code, insn) || poppedRegister(code, insn)) return true;
    const auto& info = insn.Info;
    const auto lead = code[info.OpcodeOffset];
    if (info.RexPrefix == 0 && (lead == 0xc4 || lead == 0xc5 || lead == 0x62 || lead == 0x8f) && info.Opcode != 0x8f) return false;
    if (info.IsTwoByteOpcode) return info.Opcode == 0xa0 || info.Opcode == 0xa1 || info.Opcode == 0xa8 || info.Opcode == 0xa9;
    const auto reg = static_cast<std::uint8_t>((info.ModRmByte >> 3) & 7);
    return (info.Opcode >= 0x50 && info.Opcode <= 0x5f) || info.Opcode == 0x68 || info.Opcode == 0x6a || info.Opcode == 0x9c || info.Opcode == 0x9d || (info.Opcode == 0x8f && info.HasModRm && reg == 0) || (info.Opcode == 0xff && info.HasModRm && reg == 6);
}

std::optional<std::int64_t> rspAdjustment(const std::uint8_t* code, const Insn& insn) {
    std::int32_t value = 0;
    if (insn.Length == 4 && code[0] == 0x48 && code[1] == 0x83 && (code[2] == 0xec || code[2] == 0xc4)) {
        const std::int64_t amount = static_cast<std::int8_t>(code[3]);
        return code[2] == 0xec ? -amount : amount;
    }
    if (insn.Length == 7 && code[0] == 0x48 && code[1] == 0x81 && (code[2] == 0xec || code[2] == 0xc4)) {
        std::memcpy(&value, code + 3, 4);
        return code[2] == 0xec ? -std::int64_t{value} : std::int64_t{value};
    }
    if (insn.Length == 5 && std::memcmp(code, "\x48\x8d\x64\x24", 4) == 0) return static_cast<std::int8_t>(code[4]);
    if (insn.Length == 8 && std::memcmp(code, "\x48\x8d\xa4\x24", 4) == 0) {
        std::memcpy(&value, code + 4, 4);
        return value;
    }
    return std::nullopt;
}

struct MemoryUse {
    bool UsesRsp = false;
    bool RegisterRsp = false;
    bool RbpBased = false;
    std::int64_t RbpDisp = 0;
    bool RspBased = false;
    std::int64_t RspDisp = 0;
    std::uint8_t Mod = 0;
    std::size_t DispOffset = 0;
    bool Indexed = false;
    bool Address = false;
};

// A register among rax, rcx, rdx and rbx that a ModRM memory operand does not name as base or index.
std::optional<std::uint8_t> scratchBesideOperand(const std::uint8_t* code, const Insn& insn) {
    const auto& info = insn.Info;
    if (!info.HasModRm || (info.ModRmByte >> 6) == 3) return std::nullopt;
    std::set<std::uint8_t> named;
    const auto rm = static_cast<std::uint8_t>(info.ModRmByte & 7);
    if (rm == 4) {
        if (info.ModRmOffset + 1 >= insn.Length) return std::nullopt;
        const auto sib = code[info.ModRmOffset + 1];
        named.insert(static_cast<std::uint8_t>((sib & 7) | ((info.RexPrefix & 1) << 3)));
        named.insert(static_cast<std::uint8_t>(((sib >> 3) & 7) | ((info.RexPrefix & 2) << 2)));
    } else {
        named.insert(static_cast<std::uint8_t>(rm | ((info.RexPrefix & 1) << 3)));
    }
    for (std::uint8_t reg = 0; reg < 4; ++reg)
        if (!named.contains(reg)) return reg;
    return std::nullopt;
}

// The instruction with its ModRM reg field set to reg (below 8, so REX.R stays clear).
std::vector<std::uint8_t> withRegField(const std::uint8_t* code, const Insn& insn, const std::uint8_t reg) {
    std::vector<std::uint8_t> bytes(code, code + insn.Length);
    bytes[insn.Info.ModRmOffset] = static_cast<std::uint8_t>((bytes[insn.Info.ModRmOffset] & 0xc7) | ((reg & 7) << 3));
    return bytes;
}

bool xopPrefixed(const std::uint8_t* code, const Insn& insn) {
    const auto at = insn.Info.OpcodeOffset;
    return code[at] == 0x8f && at + 1 < insn.Length && (code[at + 1] & 0x1f) >= 8;
}

void extensionBits(const std::uint8_t* code, const Insn& insn, bool& r, bool& x, bool& b) {
    const auto& info = insn.Info;
    r = x = b = false;
    if (info.RexPrefix != 0) {
        r = (info.RexPrefix & 4) != 0;
        x = (info.RexPrefix & 2) != 0;
        b = (info.RexPrefix & 1) != 0;
        return;
    }
    const auto lead = code[info.OpcodeOffset];
    if ((lead == 0xc4 || lead == 0x62 || xopPrefixed(code, insn)) && info.OpcodeOffset + 1 < insn.Length) {
        r = (code[info.OpcodeOffset + 1] & 0x80) == 0;
        x = (code[info.OpcodeOffset + 1] & 0x40) == 0;
        b = (code[info.OpcodeOffset + 1] & 0x20) == 0;
    } else if (lead == 0xc5 && info.OpcodeOffset + 1 < insn.Length) {
        r = (code[info.OpcodeOffset + 1] & 0x80) == 0;
    }
}

std::uint8_t legacyPp(const std::uint8_t* code, const Insn& insn) {
    std::uint8_t repeat = 0;
    bool operandSize = false;
    for (std::size_t index = 0; index < insn.Info.OpcodeOffset; ++index) {
        if (code[index] == 0x66) operandSize = true;
        else if (code[index] == 0xf3) repeat = 2;
        else if (code[index] == 0xf2) repeat = 3;
    }
    return repeat != 0 ? repeat : (operandSize ? std::uint8_t{1} : std::uint8_t{0});
}

bool twoByteScalarRegisterIsGeneral(const std::uint8_t op) {
    return op == 0x02 || op == 0x03 || (op >= 0x40 && op <= 0x4f) || op == 0xa3 || op == 0xa4 || op == 0xa5 || op == 0xab || op == 0xac || op == 0xad || op == 0xaf || op == 0xb0 || op == 0xb1 || op == 0xb3 || op == 0xb6 || op == 0xb7 || op == 0xb8 || (op >= 0xbb && op <= 0xbf) || op == 0xc0 || op == 0xc1 || op == 0xc3;
}

bool map1RegisterIsGeneral(const std::uint8_t op, const std::uint8_t pp, const std::uint8_t mod, const bool vex, const bool evex) {
    if (!vex && !evex && twoByteScalarRegisterIsGeneral(op)) return true;
    if ((op == 0x2c || op == 0x2d) && (pp == 2 || pp == 3)) return true;
    if (op == 0x50 && !evex && mod == 3 && (pp == 0 || pp == 1)) return true;
    return (op == 0xc5 || (op == 0xd7 && !evex)) && mod == 3 && (pp == 1 || (!vex && !evex && pp == 0));
}

bool map1RmIsGeneral(const std::uint8_t op, const std::uint8_t pp, const std::uint8_t reg, const std::uint8_t rex, const bool vex, const bool evex) {
    if (!vex && !evex) {
        if (twoByteScalarRegisterIsGeneral(op) || op == 0xba) return true;
        if ((op == 0x00 && reg <= 1) || (op == 0x01 && reg == 4)) return true;
        if (op >= 0x90 && op <= 0x9f && pp == 0 && rex != 0) return true;
        if (op == 0xae && pp == 2 && reg <= 3) return true;
        if (op == 0xc7 && reg >= 6 && (pp <= 1 || (pp == 2 && reg == 7))) return true;
    }
    if (op == 0x2a && (pp == 2 || pp == 3)) return true;
    return (op == 0x6e || op == 0x7e || op == 0xc4) && (pp == 1 || (!vex && !evex && pp == 0));
}

bool extendedOpcodeUsesRsp(const std::uint8_t* code, const Insn& insn, const std::uint8_t mod, const std::uint8_t reg, const std::uint8_t rm, const bool r, const bool b) {
    const auto& info = insn.Info;
    const auto at = info.OpcodeOffset;
    const auto lead = code[at];
    const bool regRsp = reg == kRegRsp && !r;
    const bool rmRsp = mod == 3 && rm == kRegRsp && !b;
    if (info.RexPrefix == 0 && xopPrefixed(code, insn)) {
        if (at + 3 >= insn.Length) return false;
        const auto map = static_cast<std::uint8_t>(code[at + 1] & 0x1f);
        const auto vvvv = static_cast<std::uint8_t>((~code[at + 2] >> 3) & 0xf);
        const auto op = code[at + 3];
        if (map == 9 && (op == 0x01 || op == 0x02)) return rmRsp || vvvv == kRegRsp;
        if (map == 10 && op == 0x10) return regRsp || rmRsp;
        return false;
    }
    std::uint8_t map = 0;
    std::uint8_t pp = 0;
    std::uint8_t op = 0;
    std::uint8_t vvvv = 0;
    bool isVex = false;
    bool isEvex = false;
    if (info.RexPrefix == 0 && (lead == 0xc4 || lead == 0xc5)) {
        if (at + (lead == 0xc5 ? 2u : 3u) >= insn.Length) return false;
        isVex = true;
        map = lead == 0xc5 ? std::uint8_t{1} : static_cast<std::uint8_t>(code[at + 1] & 0x1f);
        const auto second = code[lead == 0xc5 ? at + 1 : at + 2];
        pp = static_cast<std::uint8_t>(second & 3);
        vvvv = static_cast<std::uint8_t>((~second >> 3) & 0xf);
        op = code[lead == 0xc5 ? at + 2 : at + 3];
    } else if (info.RexPrefix == 0 && lead == 0x62) {
        if (at + 4 >= insn.Length) return false;
        isEvex = true;
        map = static_cast<std::uint8_t>(code[at + 1] & 0x07);
        pp = static_cast<std::uint8_t>(code[at + 2] & 3);
        op = code[at + 4];
    } else if (info.IsTwoByteOpcode) {
        pp = legacyPp(code, insn);
        if (info.Opcode == 0x38 || info.Opcode == 0x3a) {
            if (at + 2 >= insn.Length) return false;
            map = info.Opcode == 0x38 ? std::uint8_t{2} : std::uint8_t{3};
            op = code[at + 2];
        } else {
            map = 1;
            op = info.Opcode;
        }
    } else {
        return false;
    }
    if (map == 1) return (regRsp && map1RegisterIsGeneral(op, pp, mod, isVex, isEvex)) || (rmRsp && map1RmIsGeneral(op, pp, reg, info.RexPrefix, isVex, isEvex));
    if (map == 2) {
        if (isVex) return (op == 0xf2 || op == 0xf3 || op == 0xf5 || op == 0xf6 || op == 0xf7) && ((op != 0xf3 && regRsp) || rmRsp || vvvv == kRegRsp);
        if (!isEvex) {
            const bool regGen = ((op == 0xf0 || op == 0xf1) && (pp <= 1 || pp == 3)) || (op == 0xf6 && (pp == 1 || pp == 2)) || (op == 0xf8 && pp == 1) || (op == 0xf9 && pp == 0);
            const bool rmGen = (op == 0xf0 && pp == 3 && info.RexPrefix != 0) || (op == 0xf1 && pp == 3) || (op == 0xf6 && (pp == 1 || pp == 2));
            return (regRsp && regGen) || (rmRsp && rmGen);
        }
        return false;
    }
    if (map == 3) {
        if (isVex && op == 0xf0 && pp == 3) return regRsp || rmRsp;
        return rmRsp && pp == 1 && ((op >= 0x14 && op <= 0x17) || op == 0x20 || op == 0x22);
    }
    return false;
}

MemoryUse memoryUse(const std::uint8_t* code, const Insn& insn) {
    MemoryUse use;
    const auto& info = insn.Info;
    if (!info.HasModRm) {
        const auto op = info.Opcode;
        if ((info.RexPrefix & 1) == 0
            && ((!info.IsTwoByteOpcode && (op == 0x54 || op == 0x5c || op == 0x94 || op == 0xbc || (op == 0xb4 && info.RexPrefix != 0)))
                || (info.IsTwoByteOpcode && op == 0xcc)))
            use.UsesRsp = use.RegisterRsp = true;
        return use;
    }
    if (info.ModRmOffset == 0 || info.ModRmOffset >= insn.Length) return use;
    bool r = false;
    bool x = false;
    bool b = false;
    extensionBits(code, insn, r, x, b);
    const auto modrm = info.ModRmByte;
    const auto mod = static_cast<std::uint8_t>(modrm >> 6);
    const auto reg = static_cast<std::uint8_t>((modrm >> 3) & 7);
    const auto rm = static_cast<std::uint8_t>(modrm & 7);
    const bool vex = info.RexPrefix == 0 && (code[info.OpcodeOffset] == 0xc4 || code[info.OpcodeOffset] == 0xc5 || code[info.OpcodeOffset] == 0x62 || xopPrefixed(code, insn));
    // Register 4 of a byte operand without a REX prefix is AH, not rsp.
    const auto op = info.Opcode;
    const bool ah = info.RexPrefix == 0 && !vex;
    const bool byteOneByte = !info.IsTwoByteOpcode && ((op <= 0x3f && (op & 7) <= 3 && (op & 1) == 0) || op == 0x80 || op == 0x82 || op == 0x84 || op == 0x86 || op == 0x88 || op == 0x8a || op == 0xc0 || op == 0xc6 || op == 0xd0 || op == 0xd2 || op == 0xf6 || op == 0xfe);
    const bool byteTwoByteReg = info.IsTwoByteOpcode && (op == 0xb0 || op == 0xc0);
    const bool byteTwoByteRm = info.IsTwoByteOpcode && (op == 0xb0 || op == 0xc0 || op == 0xb6 || op == 0xbe);
    if (ah && (byteOneByte || byteTwoByteReg) && reg == kRegRsp) {
    } else if (!vex && !info.IsTwoByteOpcode && reg == kRegRsp && !r) {
        if ((op <= 0x3f && (op & 7) <= 3) || (op >= 0x84 && op <= 0x8b) || op == 0x8d) use.UsesRsp = use.RegisterRsp = true;
    }
    const bool twoByteReg = reg == kRegRsp && !r && twoByteRegisterIsGeneral(op) && !(ah && byteTwoByteReg);
    const bool twoByteRm = mod == 3 && rm == kRegRsp && !b && twoByteRmIsGeneral(op) && !(ah && byteTwoByteRm);
    if (!vex && info.IsTwoByteOpcode && (twoByteReg || twoByteRm)) use.UsesRsp = use.RegisterRsp = true;
    if (vex && vexGeneralUsesRsp(code, insn, mod, reg, rm, r, b)) use.UsesRsp = use.RegisterRsp = true;
    if (mod == 3) {
        const bool generalPurpose = !vex && !info.IsTwoByteOpcode && !(info.Opcode >= 0xd8 && info.Opcode <= 0xdf) && !(ah && byteOneByte);
        if (generalPurpose && rm == kRegRsp && !b) use.UsesRsp = use.RegisterRsp = true;
        return use;
    }
    std::size_t at = info.ModRmOffset + 1;
    std::uint8_t base = rm;
    bool baseExtended = b;
    bool indexed = false;
    if (rm == 4) {
        if (at >= insn.Length) return use;
        const auto sib = code[at++];
        base = sib & 7;
        indexed = ((sib >> 3) & 7) != 4 || x;
        if (mod == 0 && base == 5) return use;
    } else if (mod == 0 && rm == 5) {
        return use;
    }
    std::int64_t disp = 0;
    if (mod == 1 && at < insn.Length) disp = static_cast<std::int8_t>(code[at]);
    if (mod == 2 && at + 4 <= insn.Length) {
        std::int32_t value = 0;
        std::memcpy(&value, code + at, 4);
        disp = value;
    }
    if (base == kRegRsp && !baseExtended) {
        use.UsesRsp = true;
        use.RspBased = true;
        use.RspDisp = disp;
        use.Mod = mod;
        use.DispOffset = at;
        use.Indexed = indexed;
        use.Address = !vex && !info.IsTwoByteOpcode && info.Opcode == 0x8d;
    }
    if (base == kRegRbp && !baseExtended && mod != 0) {
        use.RbpBased = true;
        use.RbpDisp = disp;
        use.Indexed = indexed;
    }
    return use;
}

std::optional<std::vector<std::uint8_t>> movedDisplacement(const MemoryUse& use, const std::int64_t shift) {
    const auto moved = use.RspDisp + shift;
    if (use.Mod == 1 && moved <= std::numeric_limits<std::int8_t>::max()) return std::vector<std::uint8_t>{static_cast<std::uint8_t>(moved)};
    if (use.Mod == 2 && moved <= std::numeric_limits<std::int32_t>::max()) {
        const auto value = static_cast<std::uint32_t>(static_cast<std::int32_t>(moved));
        return std::vector<std::uint8_t>{static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8), static_cast<std::uint8_t>(value >> 16), static_cast<std::uint8_t>(value >> 24)};
    }
    return std::nullopt;
}

// With wide the displacement is always 32 bits: an EVEX disp8 would be scaled.
std::optional<std::vector<std::uint8_t>> withDisplacement(const std::uint8_t* code, const Insn& insn, const MemoryUse& use, const std::int64_t disp, const bool wide = false) {
    if (disp < std::numeric_limits<std::int32_t>::min() || disp > std::numeric_limits<std::int32_t>::max()) return std::nullopt;
    const std::size_t oldBytes = use.Mod == 1 ? 1 : use.Mod == 2 ? 4 : 0;
    if (use.DispOffset + oldBytes > insn.Length) return std::nullopt;
    const bool narrow = !wide && disp >= std::numeric_limits<std::int8_t>::min() && disp <= std::numeric_limits<std::int8_t>::max();
    std::vector<std::uint8_t> bytes(code, code + use.DispOffset);
    bytes[insn.Info.ModRmOffset] = static_cast<std::uint8_t>((bytes[insn.Info.ModRmOffset] & 0x3f) | (narrow ? 0x40 : 0x80));
    const auto value = static_cast<std::uint32_t>(static_cast<std::int32_t>(disp));
    for (std::size_t index = 0; index < (narrow ? 1u : 4u); ++index) bytes.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
    bytes.insert(bytes.end(), code + use.DispOffset + oldBytes, code + insn.Length);
    return bytes;
}

// The reg and rm fields (with REX) of a one-byte-opcode instruction with this opcode, or nothing.
// With registerForm the instruction must name two registers (mod 3), else a memory operand.
std::optional<std::pair<std::uint8_t, std::uint8_t>> modRmFields(const std::uint8_t* code, const Insn& insn, const std::uint8_t opcode, const bool wide, const bool registerForm) {
    const auto& info = insn.Info;
    if (info.IsTwoByteOpcode || info.Opcode != opcode || code[info.OpcodeOffset] != opcode || !info.HasModRm) return std::nullopt;
    if (wide && (info.RexPrefix & 0x08) == 0) return std::nullopt;
    if (((info.ModRmByte >> 6) == 3) != registerForm) return std::nullopt;
    const auto reg = static_cast<std::uint8_t>(((info.ModRmByte >> 3) & 7) | ((info.RexPrefix & 0x04) << 1));
    const auto rm = static_cast<std::uint8_t>((info.ModRmByte & 7) | ((info.RexPrefix & 0x01) << 3));
    return std::pair{reg, rm};
}

// A switch's dispatch: `lea base, [rip+table]` ... `movsxd target, [base+index*4]` ...
// `add target, base` ... `jmp target`, the base possibly copied by movs. Returns the table's
// segment offset from the lea, or no table when the base was set further back (a lea hoisted
// out of a loop): the caller then takes every table the function's RIP-relative leas address.
struct JumpTableDispatch {
    std::optional<std::int64_t> Table;
};

std::optional<JumpTableDispatch> jumpTableDispatch(const std::vector<Insn>& body, const std::size_t k, const std::uint8_t* segment) {
    const auto at = [&](const std::size_t j) { return segment + body[j].Offset; };
    const auto& jump = body[k];
    if (jump.Info.FlowKind != ControlFlowKind::IndirectJump) return std::nullopt;
    const auto jumpForm = modRmFields(at(k), jump, 0xff, false, true);
    if (!jumpForm || (jump.Info.ModRmByte >> 3 & 7) != 4) return std::nullopt;
    const auto target = jumpForm->second;
    std::optional<std::uint8_t> addBase;
    bool loaded = false;
    bool sameBase = false;
    std::set<std::uint8_t> holders;
    // VEX vector instructions that write no general register do not count toward the window.
    const auto vectorOnly = [&](const std::uint8_t* code, const Insn& insn) {
        const auto at = insn.Info.OpcodeOffset;
        const auto lead = code[at];
        if (insn.Info.RexPrefix != 0 || (lead != 0xc4 && lead != 0xc5) || at + 3 >= insn.Length) return false;
        const auto map = lead == 0xc5 ? std::uint8_t{1} : static_cast<std::uint8_t>(code[at + 1] & 0x1f);
        const auto op = code[lead == 0xc5 ? at + 2 : at + 3];
        if (map == 1) return op != 0x7e && op != 0x50 && op != 0xd7 && op != 0xc5 && op != 0x2c && op != 0x2d;
        if (map == 2) return op < 0xf0;
        if (map == 3) return op < 0x14 || op > 0x17;
        return false;
    };
    for (std::size_t back = 1, counted = 0; back <= 64 && back <= k && counted < 16; ++back) {
        const auto j = k - back;
        const auto* code = at(j);
        const auto& insn = body[j];
        if (!vectorOnly(code, insn)) ++counted;
        if (!addBase) {
            if (const auto add = modRmFields(code, insn, 0x01, true, true); add && add->second == target) addBase = add->first;
            else if (const auto add3 = modRmFields(code, insn, 0x03, true, true); add3 && add3->first == target) addBase = add3->second;
            if (addBase) holders.insert(*addBase);
            continue;
        }
        if (!loaded) {
            const auto load = modRmFields(code, insn, 0x63, true, false);
            if (!load || load->first != target || (insn.Info.ModRmByte & 7) != 4) continue;
            if (insn.Info.ModRmOffset == 0 || insn.Info.ModRmOffset + 2 > insn.Length) return std::nullopt;
            const auto sib = code[insn.Info.ModRmOffset + 1];
            const auto mod = insn.Info.ModRmByte >> 6;
            const auto base = static_cast<std::uint8_t>((sib & 7) | ((insn.Info.RexPrefix & 0x01) << 3));
            const auto index = static_cast<std::uint8_t>(((sib >> 3) & 7) | ((insn.Info.RexPrefix & 0x02) << 2));
            if ((sib >> 6) != 2 || index == 4 || ((sib & 7) == 5 && mod == 0)) return std::nullopt;
            if (mod == 1 && code[insn.Info.ModRmOffset + 2] != 0) return std::nullopt;
            if (mod == 2) return std::nullopt;
            holders.insert(base);
            loaded = true;
            sameBase = base == *addBase;
            continue;
        }
        if (const auto copy = modRmFields(code, insn, 0x89, true, true); copy && holders.contains(copy->second)) {
            holders.erase(copy->second);
            holders.insert(copy->first);
            continue;
        }
        if (const auto copy = modRmFields(code, insn, 0x8b, true, true); copy && holders.contains(copy->first)) {
            holders.erase(copy->first);
            holders.insert(copy->second);
            continue;
        }
        if (const auto lea = modRmFields(code, insn, 0x8d, true, false); lea && insn.Info.HasRipRelativeDisp && holders.contains(lea->first)) {
            if (holders.size() != 1) return std::nullopt;
            std::int32_t disp = 0;
            std::memcpy(&disp, code + insn.Info.RipRelativeDispOffset, 4);
            return JumpTableDispatch{static_cast<std::int64_t>(insn.Offset + insn.Length) + disp};
        }
    }
    // The load and the add use one base register, set before the window.
    if (loaded && sameBase) return JumpTableDispatch{std::nullopt};
    return std::nullopt;
}

// `jmp [base+index*8]`: a jump through a table of absolute addresses.
bool absoluteTableJump(const std::uint8_t* code, const Insn& insn) {
    const auto& info = insn.Info;
    if (info.FlowKind != ControlFlowKind::IndirectJump || info.IsTwoByteOpcode || info.Opcode != 0xff || !info.HasModRm || ((info.ModRmByte >> 3) & 7) != 4) return false;
    const auto mod = info.ModRmByte >> 6;
    if ((info.ModRmByte & 7) != 4 || mod >= 2 || info.ModRmOffset == 0 || info.ModRmOffset + 1 >= insn.Length) return false;
    const auto sib = code[info.ModRmOffset + 1];
    if ((sib >> 6) != 3 || (((sib >> 3) & 7) == 4 && (info.RexPrefix & 2) == 0)) return false;
    if ((mod == 0 && (sib & 7) == 5) || ((sib & 7) == kRegRsp && (info.RexPrefix & 1) == 0)) return false;
    return mod == 0 || (info.ModRmOffset + 2 < insn.Length && code[info.ModRmOffset + 2] == 0);
}

// `op reg, [base+disp]` with REX.W, or without REX when the opcode takes no register size.
std::vector<std::uint8_t> memoryOperand(const std::uint8_t opcode, const std::uint8_t reg, const std::uint8_t base, const std::int32_t disp, const bool rexW = true) {
    const bool narrow = disp >= std::numeric_limits<std::int8_t>::min() && disp <= std::numeric_limits<std::int8_t>::max();
    std::vector<std::uint8_t> out;
    const auto rex = static_cast<std::uint8_t>((rexW ? 0x48 : 0x40) | ((reg & 8) >> 1) | ((base & 8) >> 3));
    if (rex != 0x40) out.push_back(rex);
    out.push_back(opcode);
    out.push_back(static_cast<std::uint8_t>((narrow ? 0x40 : 0x80) | ((reg & 7) << 3) | (base & 7)));
    if ((base & 7) == kRegRsp) out.push_back(0x24);
    const auto value = static_cast<std::uint32_t>(disp);
    for (std::size_t index = 0; index < (narrow ? 1u : 4u); ++index) out.push_back(static_cast<std::uint8_t>(value >> (8 * index)));
    return out;
}

// A relocated leaf runs this far below the rsp it keeps: the red zone's size.
constexpr std::int64_t kRelocatedShift = kRedZoneBytes;

// A RIP-relative instruction copied into a stub: Offset and End are within the copied bytes,
// Target is the segment offset it addresses.
struct CopiedRip {
    std::size_t Offset;
    std::size_t End;
    std::int64_t Target;
    bool AddressSize32;
};

struct Copied {
    std::vector<std::uint8_t> Bytes;
    std::vector<CopiedRip> Rips;

    void Append(const Copied& other) {
        for (auto rip : other.Rips) {
            rip.Offset += Bytes.size();
            rip.End += Bytes.size();
            Rips.push_back(rip);
        }
        Bytes.insert(Bytes.end(), other.Bytes.begin(), other.Bytes.end());
    }

    void Prepend(const Copied& other) {
        for (auto& rip : Rips) {
            rip.Offset += other.Bytes.size();
            rip.End += other.Bytes.size();
        }
        Rips.insert(Rips.begin(), other.Rips.begin(), other.Rips.end());
        Bytes.insert(Bytes.begin(), other.Bytes.begin(), other.Bytes.end());
    }
};

struct OperandStub {
    std::size_t First;
    std::size_t Last;
    std::size_t Begin;
    std::size_t End;
    Copied Body;
    std::size_t Operands = 0;
};

struct ExitSite {
    std::size_t First;
    std::size_t Tail;
    std::size_t Begin;
    std::size_t TailBegin;
    std::size_t End;
    Copied Lead;
    std::size_t Operands = 0;
    // The pops and the ret, or the pops and the tail call with its target relocated.
    Copied Trail;
};

void frameSegment(
    const std::vector<std::uint8_t>& fileBytes,
    const Domain::ProgramHeader& ph,
    const std::vector<RedZoneFunction>& functions,
    const std::vector<TrampolineSite>& existing,
    std::vector<TrampolineSite>& sites,
    std::vector<RedZoneDisplacement>& displacements,
    RedZoneFrameReport& report,
    const std::vector<Domain::ProgramHeader>& loadSegments,
    const std::map<std::uint64_t, std::uint64_t>& relativeAddends
) {
    const auto segOffset = static_cast<std::size_t>(ph.Offset);
    const auto segSize = static_cast<std::size_t>(ph.FileSize);
    if (segOffset > fileBytes.size() || segSize > fileBytes.size() - segOffset) return;
    const std::vector<std::uint8_t> seg(fileBytes.begin() + static_cast<std::ptrdiff_t>(segOffset), fileBytes.begin() + static_cast<std::ptrdiff_t>(segOffset + segSize));
    const auto scanner = MakeInstructionScanner();
    const X64InstructionDecoder decoder;

    std::set<std::size_t> targets;
    // Every branch as (target, source), by target: a relocated leaf checks that no branch from
    // outside it lands inside it.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> edges;
    for (const auto& match : scanner->ScanCodeSection(seg, 0, seg.size())) {
        const auto info = decoder.DecodeInstruction(seg.data() + match.Offset, match.Length);
        if (!info.HasBranchTarget || info.HasRipRelativeDisp || info.FlowKind == ControlFlowKind::Call) continue;
        const auto target = static_cast<std::int64_t>(match.Offset + match.Length) + info.BranchDisp;
        if (target >= 0 && static_cast<std::size_t>(target) < seg.size()) {
            targets.insert(static_cast<std::size_t>(target));
            edges.emplace_back(static_cast<std::uint32_t>(target), static_cast<std::uint32_t>(match.Offset));
        }
    }
    std::sort(edges.begin(), edges.end());
    std::vector<std::pair<std::size_t, std::size_t>> taken;
    for (const auto& site : existing) {
        if (site.Address < ph.MappedAddress || site.Address >= ph.MappedAddress + segSize) continue;
        const auto at = static_cast<std::size_t>(site.Address - ph.MappedAddress);
        taken.emplace_back(at, at + site.Length);
    }
    const auto overlapsExisting = [&](const std::size_t begin, const std::size_t end) {
        return std::any_of(taken.begin(), taken.end(), [&](const auto& range) { return begin < range.second && range.first < end; });
    };
    const auto targetInside = [&](const std::size_t begin, const std::size_t end) {
        const auto hit = targets.upper_bound(begin);
        return hit != targets.end() && *hit < end;
    };
    const auto readGuest32 = [&](const std::uint64_t address) -> std::optional<std::int32_t> {
        const auto read = [&](const Domain::ProgramHeader& segment) -> std::optional<std::int32_t> {
            if (address < segment.MappedAddress || address + 4 > segment.MappedAddress + segment.FileSize) return std::nullopt;
            const auto offset = static_cast<std::size_t>(segment.Offset + (address - segment.MappedAddress));
            if (offset + 4 > fileBytes.size()) return std::nullopt;
            std::int32_t value = 0;
            std::memcpy(&value, fileBytes.data() + offset, 4);
            return value;
        };
        if (const auto value = read(ph)) return value;
        for (const auto& segment : loadSegments)
            if (const auto value = read(segment)) return value;
        return std::nullopt;
    };
    // A tail call's target: another function's start, or code outside every function (a PLT stub).
    // A jump into the middle of another function (a cold part) may come back, so it is not one.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    for (const auto& function : functions) spans.emplace_back(function.Begin, function.End);
    std::sort(spans.begin(), spans.end());
    const auto tailCallTarget = [&](const std::int64_t target) {
        const auto address = static_cast<std::uint64_t>(static_cast<std::int64_t>(ph.MappedAddress) + target);
        auto span = std::upper_bound(spans.begin(), spans.end(), std::pair{address, std::numeric_limits<std::uint64_t>::max()});
        if (span == spans.begin()) return true;
        --span;
        return address == span->first || address >= span->second;
    };
    for (const auto& function : functions) {
        if (function.Begin < ph.MappedAddress || function.End <= function.Begin || function.End - ph.MappedAddress > segSize) continue;
        const auto start = static_cast<std::size_t>(function.Begin - ph.MappedAddress);
        const auto limit = static_cast<std::size_t>(function.End - ph.MappedAddress);
        ++report.Functions;
        const auto skip = [&](const char* reason) { ++report.Skipped[reason]; };

        const std::vector<std::uint8_t> code(seg.begin() + static_cast<std::ptrdiff_t>(start), seg.begin() + static_cast<std::ptrdiff_t>(limit));
        std::vector<Insn> body;
        for (const auto& match : scanner->ScanCodeSection(code, 0, code.size())) {
            if (code[match.Offset] == 0xcc && match.Length == 1) break;
            body.push_back({start + static_cast<std::size_t>(match.Offset), match.Length, decoder.DecodeInstruction(code.data() + match.Offset, match.Length)});
        }
        if (body.size() < 3) continue;
        const auto at = [&](const std::size_t k) { return seg.data() + body[k].Offset; };
        // Jump tables: entries are read until one does not land on an instruction of this function.
        // The real entries come first and all land, so every case is taken, and at worst a few
        // words past the table: those only add branch targets and paths to check.
        std::map<std::size_t, std::size_t> indexOf;
        for (std::size_t k = 0; k < body.size(); ++k) indexOf[body[k].Offset] = k;
        std::map<std::size_t, std::vector<std::size_t>> tables;
        // The scan stops at the first int3: an entry landing between it and the FDE end may be a
        // case the scan never saw, so that table is not resolved (returns false).
        const auto bodyEnd = static_cast<std::int64_t>(body.back().Offset + body.back().Length);
        const auto tableCases = [&](const std::int64_t table, std::set<std::size_t>& cases) {
            for (std::int64_t entry = 0; entry < kJumpTableEntryLimit; ++entry) {
                const auto value = readGuest32(static_cast<std::uint64_t>(static_cast<std::int64_t>(ph.MappedAddress) + table + 4 * entry));
                if (!value) break;
                const auto caseOffset = table + *value;
                if (caseOffset < static_cast<std::int64_t>(start) || caseOffset >= static_cast<std::int64_t>(limit)) break;
                if (caseOffset >= bodyEnd) return false;
                const auto found = indexOf.find(static_cast<std::size_t>(caseOffset));
                if (found == indexOf.end()) break;
                cases.insert(found->second);
            }
            return true;
        };
        // Every table the function's RIP-relative leas address, for a dispatch whose base was set
        // before its window: the real table is among them, the others only add cases.
        std::vector<std::int64_t> leaTables;
        for (std::size_t k = 0; k < body.size(); ++k)
            if (const auto lea = modRmFields(at(k), body[k], 0x8d, true, false); lea && body[k].Info.HasRipRelativeDisp) {
                std::int32_t disp = 0;
                std::memcpy(&disp, at(k) + body[k].Info.RipRelativeDispOffset, 4);
                leaTables.push_back(static_cast<std::int64_t>(body[k].Offset + body[k].Length) + disp);
            }
        for (std::size_t k = 0; k < body.size(); ++k) {
            const auto dispatch = jumpTableDispatch(body, k, seg.data());
            if (!dispatch) continue;
            std::set<std::size_t> found;
            bool complete = true;
            if (dispatch->Table) complete = tableCases(*dispatch->Table, found);
            else for (const auto table : leaTables) complete = tableCases(table, found) && complete;
            const std::vector<std::size_t> cases(found.begin(), found.end());
            if (!complete || cases.empty()) continue;
            for (const auto c : cases) targets.insert(body[c].Offset);
            tables.emplace(k, std::move(cases));
        }
        // A leaf whose frame cannot be patched in place is relocated whole into a stub that runs it
        // with rsp kRelocatedShift lower than the leaf keeps it. Every rsp operand moves up by the
        // shift and every push and pop is rewritten to keep its slot, so the leaf's stack data stays
        // at its addresses, which are now above rsp. Every way out raises rsp first, so code outside
        // the stub sees the stack as before. The leaf's start jumps to the stub, and each case of a
        // jump table gets a jump into it. Returns why the leaf could not be relocated, or nothing.
        // A branch from outside the leaf (another function, or code past its first int3) into
        // [begin, end).
        const auto enteredFromOutside = [&](const std::size_t begin, const std::size_t end) {
            auto edge = std::lower_bound(edges.begin(), edges.end(), std::pair<std::uint32_t, std::uint32_t>{static_cast<std::uint32_t>(begin), 0});
            for (; edge != edges.end() && edge->first < end; ++edge)
                if (edge->second < start || edge->second >= static_cast<std::size_t>(bodyEnd)) return true;
            return false;
        };
        const auto relocate = [&]() -> const char* {
            constexpr std::int64_t shift = kRelocatedShift;
            if (bodyEnd - static_cast<std::int64_t>(start) < static_cast<std::int64_t>(kJumpBytes)) return "leaf shorter than a jump";
            if (overlapsExisting(start, static_cast<std::size_t>(bodyEnd))) return "overlaps another stub";
            if (enteredFromOutside(start + 1, start + kJumpBytes)) return "branch from outside into the start of a relocated leaf";
            const auto inside = [&](const std::int64_t target) -> std::optional<std::size_t> {
                if (target < static_cast<std::int64_t>(start) || target >= bodyEnd) return std::nullopt;
                const auto found = indexOf.find(static_cast<std::size_t>(target));
                if (found == indexOf.end()) return std::nullopt;
                return found->second;
            };
            // Jump tables: relative ones as resolved above, and tables of absolute addresses, read
            // from the relocations that fill them (or the file) for every table a lea addresses.
            std::set<std::size_t> dispatches;
            std::set<std::size_t> cases;
            for (const auto& [k, found] : tables) {
                dispatches.insert(k);
                cases.insert(found.begin(), found.end());
            }
            // A table of other functions' starts (threaded code: each handler jumps to the next) makes
            // its jump a tail call, like a jump through a function pointer.
            std::set<std::size_t> tailJumps;
            for (std::size_t k = 0; k < body.size(); ++k) {
                if (!absoluteTableJump(at(k), body[k])) continue;
                std::set<std::size_t> found;
                std::size_t starts = 0;
                for (const auto table : leaTables)
                    for (std::int64_t entry = 0; entry < kJumpTableEntryLimit; ++entry) {
                        const auto address = static_cast<std::uint64_t>(static_cast<std::int64_t>(ph.MappedAddress) + table + 8 * entry);
                        std::optional<std::uint64_t> value;
                        if (const auto addend = relativeAddends.find(address); addend != relativeAddends.end()) value = addend->second;
                        else if (const auto low = readGuest32(address), high = readGuest32(address + 4); low && high && (*low != 0 || *high != 0))
                            value = static_cast<std::uint32_t>(*low) | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(*high)) << 32);
                        if (!value) break;
                        const auto offset = static_cast<std::int64_t>(*value) - static_cast<std::int64_t>(ph.MappedAddress);
                        // The leaf's own start is entered like a call, through its entry jump.
                        const auto c = offset == static_cast<std::int64_t>(start) ? std::nullopt : inside(offset);
                        if (c) found.insert(*c);
                        else if (offset >= 0 && static_cast<std::size_t>(offset) < seg.size() && (offset <= static_cast<std::int64_t>(start) || offset >= static_cast<std::int64_t>(limit)) && tailCallTarget(offset)) ++starts;
                        else break;
                    }
                if (found.empty() && starts > 0) tailJumps.insert(k);
                // A table with both cases and other functions' starts would need rsp both ways.
                if (!found.empty() && starts > 0) return "jump table of both cases and functions";
                if (found.empty()) continue;
                dispatches.insert(k);
                cases.insert(found.begin(), found.end());
            }
            Copied out{lowerRsp(shift), {}};
            const auto raise = raiseRsp(shift);
            const auto append = [&](const std::vector<std::uint8_t>& bytes) { out.Bytes.insert(out.Bytes.end(), bytes.begin(), bytes.end()); };
            const auto appendOriginal = [&](const std::size_t k) {
                const auto& insn = body[k];
                Copied copied{std::vector<std::uint8_t>(at(k), at(k) + insn.Length), {}};
                if (insn.Info.HasRipRelativeDisp && insn.Info.RipRelativeDispOffset + 4 <= insn.Length) {
                    std::int32_t disp = 0;
                    std::memcpy(&disp, at(k) + insn.Info.RipRelativeDispOffset, 4);
                    copied.Rips.push_back({insn.Info.RipRelativeDispOffset, insn.Length, static_cast<std::int64_t>(insn.Offset + insn.Length) + disp, insn.Info.AddressSize32});
                }
                out.Append(copied);
            };
            const auto leaveTo = [&](const std::int64_t target) {
                append(raise);
                out.Append(Copied{{0xe9, 0, 0, 0, 0}, {{1, 5, target, false}}});
            };
            std::vector<std::size_t> labelAt(body.size());
            std::vector<std::pair<std::size_t, std::size_t>> internal;
            std::map<std::int64_t, std::vector<std::size_t>> outside;
            bool openJump = false;
            bool leavesIntoFunction = false;
            const auto branchTo = [&](const std::int64_t target) {
                const auto field = out.Bytes.size() - 4;
                if (const auto k = inside(target)) internal.emplace_back(field, *k);
                else {
                    outside[target].push_back(field);
                    if (target >= static_cast<std::int64_t>(start) && target < static_cast<std::int64_t>(limit)) leavesIntoFunction = true;
                }
            };
            for (std::size_t k = 0; k < body.size(); ++k) {
                labelAt[k] = out.Bytes.size();
                const auto& insn = body[k];
                const auto* code = at(k);
                const auto& info = insn.Info;
                const auto flow = info.FlowKind;
                const auto use = memoryUse(code, insn);
                const auto target = static_cast<std::int64_t>(insn.Offset + insn.Length) + info.BranchDisp;
                if (flow == ControlFlowKind::ConditionalBranch) {
                    const auto op = info.Opcode;
                    const bool shortForm = !info.IsTwoByteOpcode && op >= 0x70 && op <= 0x7f;
                    const bool nearForm = info.IsTwoByteOpcode && op >= 0x80 && op <= 0x8f;
                    if (!info.HasBranchTarget || info.HasRipRelativeDisp || (!shortForm && !nearForm)) return "loop or jrcxz in a relocated leaf";
                    append({0x0f, static_cast<std::uint8_t>(0x80 | (op & 0x0f)), 0, 0, 0, 0});
                    branchTo(target);
                    continue;
                }
                if (flow == ControlFlowKind::UnconditionalJump && info.HasBranchTarget && !info.HasRipRelativeDisp) {
                    if (inside(target)) {
                        append({0xe9, 0, 0, 0, 0});
                        branchTo(target);
                    } else {
                        if (target >= static_cast<std::int64_t>(start) && target < static_cast<std::int64_t>(limit)) leavesIntoFunction = true;
                        leaveTo(target);
                    }
                    continue;
                }
                if (flow == ControlFlowKind::Return) {
                    append(raise);
                    append(std::vector<std::uint8_t>(code + info.OpcodeOffset, code + insn.Length));
                    continue;
                }
                if (flow == ControlFlowKind::IndirectJump || flow == ControlFlowKind::UnconditionalJump) {
                    // A jump table's dispatch lands on a case's jump into the stub; any other indirect
                    // jump leaves, its rsp operand read with rsp raised as in the leaf.
                    if (dispatches.contains(k)) {
                        if (use.RspBased) return "jump table through an rsp operand";
                    } else {
                        append(raise);
                        // Through memory without an index: a function pointer, so a tail call.
                        bool pointer = info.HasModRm && (info.ModRmByte >> 6) != 3 && !use.RspBased;
                        if (pointer && (info.ModRmByte & 7) == 4 && info.ModRmOffset + 1 < insn.Length) pointer = ((code[info.ModRmOffset + 1] >> 3) & 7) == 4 && (info.RexPrefix & 2) == 0;
                        if (!pointer && !tailJumps.contains(k)) openJump = true;
                    }
                    appendOriginal(k);
                    continue;
                }
                if (flow != ControlFlowKind::Sequential && flow != ControlFlowKind::Trap) return "call in a relocated leaf";
                if (!info.IsTwoByteOpcode && info.Opcode == 0xc9 && code[info.OpcodeOffset] == 0xc9) {
                    // leave: rsp from rbp, then pop rbp.
                    append(memoryOperand(0x8d, kRegRsp, kRegRbp, static_cast<std::int32_t>(-shift)));
                    append(memoryOperand(0x8b, kRegRbp, kRegRsp, static_cast<std::int32_t>(shift)));
                    append(raiseRsp(8));
                    continue;
                }
                if (pushesOrPops(code, insn)) {
                    if (std::find(code, code + info.OpcodeOffset, 0x66) != code + info.OpcodeOffset) return "16-bit push or pop in a relocated leaf";
                    if (const auto reg = pushedRegister(code, insn)) {
                        append(memoryOperand(0x89, *reg, kRegRsp, static_cast<std::int32_t>(shift - 8)));
                        append(lowerRsp(8));
                        continue;
                    }
                    if (const auto reg = poppedRegister(code, insn)) {
                        append(memoryOperand(0x8b, *reg, kRegRsp, static_cast<std::int32_t>(shift)));
                        append(raiseRsp(8));
                        continue;
                    }
                    const auto op = info.Opcode;
                    // pop rsp (a realigned frame restoring its saved rsp): the saved value sits at the
                    // real slot, rsp + shift. It is loaded into a scratch pushed below rsp, lowered by
                    // the shift and popped into rsp, so rsp is never above the leaf's lowered rsp.
                    if (!info.IsTwoByteOpcode && code[info.OpcodeOffset] == 0x5c && (info.RexPrefix & 1) == 0 && info.OpcodeOffset + 1 == insn.Length) {
                        append({0x50});
                        append(memoryOperand(0x8b, 0, kRegRsp, static_cast<std::int32_t>(8 + shift)));
                        append(memoryOperand(0x8d, 0, 0, static_cast<std::int32_t>(-shift)));
                        append(memoryOperand(0x87, 0, kRegRsp, 0));
                        append({0x5c});
                        continue;
                    }
                    if (use.RegisterRsp) return "push or pop of rsp in a relocated leaf";
                    const bool pop = info.IsTwoByteOpcode ? (op == 0xa1 || op == 0xa9) : ((op >= 0x58 && op <= 0x5f) || op == 0x9d || op == 0x8f);
                    // Any other push or pop runs as itself through a slot at rsp, then moves the value:
                    // a push's address is taken before it lowers rsp, a pop's after it raises it.
                    Copied moved;
                    if (use.RspBased) {
                        const auto bytes = withDisplacement(code, insn, use, use.RspDisp + shift + (pop ? 8 : 0));
                        if (!bytes) return "rsp operand that cannot move";
                        moved.Bytes = *bytes;
                    } else {
                        const auto mark = out.Bytes.size();
                        appendOriginal(k);
                        moved.Bytes.assign(out.Bytes.begin() + static_cast<std::ptrdiff_t>(mark), out.Bytes.end());
                        for (auto rip : out.Rips)
                            if (rip.Offset >= mark) moved.Rips.push_back({rip.Offset - mark, rip.End - mark, rip.Target, rip.AddressSize32});
                        out.Bytes.resize(mark);
                        out.Rips.erase(std::remove_if(out.Rips.begin(), out.Rips.end(), [&](const CopiedRip& rip) { return rip.Offset >= mark; }), out.Rips.end());
                    }
                    if (!pop) {
                        out.Append(moved);
                        append(memoryOperand(0x8f, 0, kRegRsp, static_cast<std::int32_t>(shift - 8), false));
                        append(lowerRsp(8));
                    } else {
                        append(memoryOperand(0xff, 6, kRegRsp, static_cast<std::int32_t>(shift), false));
                        out.Append(moved);
                        append(raiseRsp(8));
                    }
                    continue;
                }
                if (use.RegisterRsp) {
                    if (rspAdjustment(code, insn)) {
                        appendOriginal(k);
                        continue;
                    }
                    // mov reg, rsp reads the leaf's rsp; mov rsp, reg sets it.
                    const auto fromRsp = [&]() -> std::optional<std::uint8_t> {
                        if (const auto mov = modRmFields(code, insn, 0x89, true, true); mov && mov->first == kRegRsp && mov->second != kRegRsp) return mov->second;
                        if (const auto mov = modRmFields(code, insn, 0x8b, true, true); mov && mov->second == kRegRsp && mov->first != kRegRsp) return mov->first;
                        return std::nullopt;
                    };
                    const auto toRsp = [&]() -> std::optional<std::uint8_t> {
                        if (const auto mov = modRmFields(code, insn, 0x89, true, true); mov && mov->second == kRegRsp && mov->first != kRegRsp) return mov->first;
                        if (const auto mov = modRmFields(code, insn, 0x8b, true, true); mov && mov->first == kRegRsp && mov->second != kRegRsp) return mov->second;
                        return std::nullopt;
                    };
                    if (const auto reg = fromRsp()) {
                        append(memoryOperand(0x8d, *reg, kRegRsp, static_cast<std::int32_t>(shift)));
                        continue;
                    }
                    if (const auto reg = toRsp()) {
                        append(memoryOperand(0x8d, kRegRsp, *reg, static_cast<std::int32_t>(-shift)));
                        continue;
                    }
                    // and rsp, -2^n keeps the shift when 2^n divides it.
                    if (insn.Length == 4 && code[0] == 0x48 && code[1] == 0x83 && code[2] == 0xe4) {
                        const auto mask = -static_cast<std::int64_t>(static_cast<std::int8_t>(code[3]));
                        if (mask > 0 && (mask & (mask - 1)) == 0 && shift % mask == 0) {
                            appendOriginal(k);
                            continue;
                        }
                    }
                    // mov rsp, [mem] (longjmp) loads an rsp the leaf does not keep, and mov [mem], rsp
                    // stores the leaf's rsp, which is rsp + shift. Both go through a scratch pushed
                    // below rsp (free: the leaf's slots are at rsp and above), so no flags change and
                    // rsp never rises above the leaf's lowered rsp. An rsp operand moves by the shift
                    // and the push.
                    const auto load = modRmFields(code, insn, 0x8b, true, false);
                    const auto store = modRmFields(code, insn, 0x89, true, false);
                    if (((load && load->first == kRegRsp) || (store && store->first == kRegRsp)) && !info.HasRipRelativeDisp) {
                        const auto scratch = scratchBesideOperand(code, insn);
                        if (!scratch) return "rsp load or store with no scratch register";
                        const auto operand = withRegField(code, insn, *scratch);
                        std::vector<std::uint8_t> access = operand;
                        if (use.RspBased) {
                            const auto bytes = withDisplacement(operand.data(), insn, use, use.RspDisp + shift + 8);
                            if (!bytes) return "rsp operand that cannot move";
                            access = *bytes;
                        }
                        append({static_cast<std::uint8_t>(0x50 + *scratch)});
                        if (load && load->first == kRegRsp) {
                            append(access);
                            append(memoryOperand(0x8d, *scratch, *scratch, static_cast<std::int32_t>(-shift)));
                            append(memoryOperand(0x87, *scratch, kRegRsp, 0));
                            append({0x5c});
                        } else {
                            append(memoryOperand(0x8d, *scratch, kRegRsp, static_cast<std::int32_t>(8 + shift)));
                            append(access);
                            append({static_cast<std::uint8_t>(0x58 + *scratch)});
                        }
                        continue;
                    }
                    return "other use of rsp in a relocated leaf";
                }
                if (use.RspBased) {
                    const bool evex = info.RexPrefix == 0 && code[info.OpcodeOffset] == 0x62;
                    if (evex && use.Mod == 1) return "EVEX rsp operand with a scaled displacement";
                    if (info.AddressSize32) return "32-bit rsp operand";
                    const auto bytes = withDisplacement(code, insn, use, use.RspDisp + shift, evex);
                    if (!bytes) return "rsp operand that cannot move";
                    append(*bytes);
                    continue;
                }
                appendOriginal(k);
            }
            if (const auto last = body.back().Info.FlowKind; last == ControlFlowKind::Sequential || last == ControlFlowKind::ConditionalBranch || last == ControlFlowKind::Trap) {
                leavesIntoFunction = true;
                leaveTo(bodyEnd);
            }
            const auto patch = [&](const std::size_t field, const std::size_t label) {
                const auto value = static_cast<std::uint32_t>(static_cast<std::int32_t>(static_cast<std::int64_t>(label) - static_cast<std::int64_t>(field + 4)));
                for (std::size_t index = 0; index < 4; ++index) out.Bytes[field + index] = static_cast<std::uint8_t>(value >> (8 * index));
            };
            for (const auto& [target, fields] : outside) {
                const auto label = out.Bytes.size();
                leaveTo(target);
                for (const auto field : fields) patch(field, label);
            }
            for (const auto& [field, k] : internal) patch(field, labelAt[k]);
            // A case's jump assumes rsp is already lowered: only the stub may reach it, so the original
            // code must not run (no branch from outside into it, nothing in the stub going back to it).
            std::vector<std::size_t> pads;
            for (const auto c : cases) pads.push_back(body[c].Offset);
            if (!pads.empty()) {
                if (enteredFromOutside(start + 1, static_cast<std::size_t>(bodyEnd))) return "branch from outside into a relocated leaf with jump tables";
                if (openJump) return "indirect jump in a relocated leaf with jump tables";
                if (leavesIntoFunction) return "relocated leaf with jump tables leaves into its own code";
                for (std::size_t index = 0; index < pads.size(); ++index) {
                    const auto begin = pads[index];
                    if (begin < start + kJumpBytes || begin + kJumpBytes > limit || (index + 1 < pads.size() && pads[index + 1] < begin + kJumpBytes)) return "jump table cases closer than a jump";
                    if (overlapsExisting(begin, begin + kJumpBytes)) return "overlaps another stub";
                }
            }
            TrampolineSite site;
            site.Offset = ph.Offset + start;
            site.Address = ph.MappedAddress + start;
            site.Length = kJumpBytes;
            site.OriginalBytes.assign(seg.begin() + static_cast<std::ptrdiff_t>(start), seg.begin() + static_cast<std::ptrdiff_t>(start + kJumpBytes));
            site.ReturnBranchOffset = out.Bytes.size();
            for (const auto& rip : out.Rips) site.Relocations.push_back({rip.Offset, rip.End, rip.Target - static_cast<std::int64_t>(start), rip.AddressSize32});
            out.Bytes.insert(out.Bytes.end(), std::begin(kJumpBack), std::end(kJumpBack));
            site.Body = std::move(out.Bytes);
            taken.emplace_back(start, start + kJumpBytes);
            const auto main = sites.size();
            sites.push_back(std::move(site));
            for (std::size_t index = 0; index < pads.size(); ++index) {
                TrampolineSite pad;
                pad.Offset = ph.Offset + pads[index];
                pad.Address = ph.MappedAddress + pads[index];
                pad.Length = kJumpBytes;
                pad.OriginalBytes.assign(seg.begin() + static_cast<std::ptrdiff_t>(pads[index]), seg.begin() + static_cast<std::ptrdiff_t>(pads[index] + kJumpBytes));
                pad.ReturnBranchOffset = 0;
                pad.JoinBack = sites.size() - main;
                pad.JoinOffset = labelAt[indexOf.at(pads[index])];
                taken.emplace_back(pads[index], pads[index] + kJumpBytes);
                sites.push_back(std::move(pad));
            }
            ++report.Relocated;
            ++report.Framed;
            report.RelocatedCases += pads.size();
            if (openJump) ++report.RelocatedOpenJumps;
            return nullptr;
        };
        // A leaf without a frame pointer whose frame cannot be patched in place (a short exit, a
        // branch into one) is tried again as a return thunk, its pushes and allocation taken as
        // part of the body. If that fails too, the first reason is reported.
        const char* firstReason = nullptr;
        for (bool asThunk = false;; asThunk = true) {
            const bool rbpPrologue = body[0].Length == 1 && at(0)[0] == 0x55 && body[1].Length == 3 && std::memcmp(at(1), "\x48\x89\xe5", 3) == 0;
            std::vector<std::uint8_t> pushed;
            std::size_t p = rbpPrologue ? 2 : 0;
            while (p < body.size())
                if (const auto reg = pushedRegister(at(p), body[p])) { pushed.push_back(*reg); ++p; } else break;
            std::int64_t allocation = 0;
            if (!rbpPrologue && p < body.size())
                if (const auto adjustment = rspAdjustment(at(p), body[p]); adjustment && *adjustment < 0) { allocation = -*adjustment; ++p; }
            // A frame-pointer leaf that allocates with `sub rsp, N` after its pushes keeps [rbp-x] inside
            // that allocation, provided rsp only goes back up in epilogues: instructions that move rsp
            // up to a ret. Such slots are not below rsp.
            std::int64_t rbpAllocation = 0;
            if (rbpPrologue && p < body.size())
                if (const auto adjustment = rspAdjustment(at(p), body[p]); adjustment && *adjustment < 0) {
                    rbpAllocation = -*adjustment;
                    const auto raisesRsp = [&](const std::size_t k) {
                        return at(k)[body[k].Info.OpcodeOffset] == 0xc9 || poppedRegister(at(k), body[k]).has_value() || memoryUse(at(k), body[k]).RegisterRsp;
                    };
                    // An epilogue ends in a ret or in a tail call: a jump out of the function.
                    const auto leaves = [&](const std::size_t k) {
                        const auto& insn = body[k];
                        if (insn.Info.FlowKind == ControlFlowKind::Return || insn.Info.FlowKind == ControlFlowKind::IndirectJump) return true;
                        if (insn.Info.FlowKind != ControlFlowKind::UnconditionalJump || !insn.Info.HasBranchTarget || insn.Info.HasRipRelativeDisp) return false;
                        const auto target = static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp;
                        return target < static_cast<std::int64_t>(start) || target >= static_cast<std::int64_t>(limit);
                    };
                    for (std::size_t k = p + 1; k < body.size(); ++k) {
                        if (!raisesRsp(k)) continue;
                        auto q = k;
                        while (q < body.size() && raisesRsp(q)) ++q;
                        if (q >= body.size() || !leaves(q)) { rbpAllocation = 0; break; }
                        k = q;
                    }
                }
            std::vector<std::uint8_t> expected(pushed.begin(), pushed.end());
            if (rbpPrologue) expected.insert(expected.begin(), kRegRbp);
            const auto popsBefore = [&](const std::size_t k) -> std::optional<std::size_t> {
                std::size_t q = k;
                std::vector<std::uint8_t> popped;
                while (q > p && popped.size() < expected.size())
                    if (const auto reg = poppedRegister(at(q - 1), body[q - 1])) { popped.push_back(*reg); --q; } else break;
                if (popped != expected) return std::nullopt;
                return q;
            };
            const auto leavesByTailCall = [&](const std::size_t k) {
                const auto& insn = body[k];
                if (insn.Info.FlowKind == ControlFlowKind::IndirectJump) return !tables.contains(k);
                if (insn.Info.FlowKind != ControlFlowKind::UnconditionalJump || !insn.Info.HasBranchTarget) return false;
                const auto target = static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp;
                return (target < static_cast<std::int64_t>(start) || target >= static_cast<std::int64_t>(limit)) && tailCallTarget(target);
            };
            std::set<std::size_t> frees;
            if (allocation > 0)
                for (std::size_t k = p; k < body.size(); ++k)
                    if (body[k].Info.FlowKind == ControlFlowKind::Return || leavesByTailCall(k))
                        if (const auto q = popsBefore(k); q && *q > p && rspAdjustment(at(*q - 1), body[*q - 1]) == allocation) frees.insert(*q - 1);
            // A leaf that moves rsp anywhere but its prologue allocation and the frees before its exits
            // returns through a thunk, its pushes and allocations taken as part of the body.
            // A jmp through memory without an index ([reg+disp], [rip+disp]) loads a function
            // pointer: a tail call even in a leaf without pushes.
            // So is `mov reg, [base+disp]; jmp reg`.
            const auto pointerJumpAt = [&](const std::size_t k) {
                const auto& insn = body[k];
                if (insn.Info.FlowKind != ControlFlowKind::IndirectJump || !insn.Info.HasModRm) return false;
                if ((insn.Info.ModRmByte >> 6) == 3) {
                    const auto jump = modRmFields(at(k), insn, 0xff, false, true);
                    if (!jump || k == 0) return false;
                    const auto& prior = body[k - 1];
                    const auto load = modRmFields(at(k - 1), prior, 0x8b, true, false);
                    if (!load || load->first != jump->second || memoryUse(at(k - 1), prior).RspBased || prior.Info.HasRipRelativeDisp) return false;
                    if ((prior.Info.ModRmByte & 7) != 4) return true;
                    if (prior.Info.ModRmOffset + 1 >= prior.Length) return false;
                    const auto sib = at(k - 1)[prior.Info.ModRmOffset + 1];
                    return ((sib >> 3) & 7) == 4 && (prior.Info.RexPrefix & 0x02) == 0;
                }
                if ((insn.Info.ModRmByte & 7) != 4) return true;
                if (insn.Info.ModRmOffset + 1 >= insn.Length) return false;
                const auto sib = at(k)[insn.Info.ModRmOffset + 1];
                return ((sib >> 3) & 7) == 4 && (insn.Info.RexPrefix & 0x02) == 0;
            };
            // A tail call's jmp copied into a stub, its direct or RIP-relative target relocated.
            const auto copiedJump = [&](const std::size_t k) -> std::optional<Copied> {
                const auto& insn = body[k];
                if (insn.Info.FlowKind == ControlFlowKind::UnconditionalJump)
                    return Copied{{0xe9, 0, 0, 0, 0}, {{1, 5, static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp, false}}};
                Copied jump{std::vector<std::uint8_t>(at(k), at(k) + insn.Length), {}};
                if (insn.Info.HasRipRelativeDisp) {
                    const auto offset = insn.Info.RipRelativeDispOffset;
                    if (offset + 4 > insn.Length) return std::nullopt;
                    std::int32_t disp = 0;
                    std::memcpy(&disp, at(k) + offset, 4);
                    jump.Rips.push_back({offset, insn.Length, static_cast<std::int64_t>(insn.Offset + insn.Length) + disp, insn.Info.AddressSize32});
                }
                return jump;
            };
            // In a thunk, a tail call: a jmp to another function's start, through a function
            // pointer, or through a register once a leaf that had pushes has popped them.
            const bool prologueFrame = !pushed.empty() || allocation > 0;
            const auto thunkTailCall = [&](const std::size_t k) {
                if (!leavesByTailCall(k)) return false;
                if (body[k].Info.FlowKind != ControlFlowKind::IndirectJump) return true;
                return !memoryUse(at(k), body[k]).RspBased && (pointerJumpAt(k) || prologueFrame);
            };
            bool rspArithmetic = false;
            if (!rbpPrologue)
                for (std::size_t k = 0; k < body.size(); ++k)
                    if (rspAdjustment(at(k), body[k]) && !(allocation > 0 && k + 1 == p) && !frees.contains(k)) {
                        rspArithmetic = true;
                        p = 0;
                        pushed.clear();
                        expected.clear();
                        allocation = 0;
                        frees.clear();
                        break;
                    }
            if (asThunk) {
                p = 0;
                pushed.clear();
                expected.clear();
                allocation = 0;
                frees.clear();
            }

            bool leaf = true;
            bool redZone = false;
            bool called = false;
            bool redZoneBeforeCall = false;
            // Exits through `leave` (frame pointer, nothing else pushed): `pop rbp`s to rewrite.
            const bool leaveExits = rbpPrologue && pushed.empty() && !asThunk;
            std::vector<std::size_t> leaves;
            std::size_t leaveExitCount = 0;
            std::int64_t frameBeforeCall = static_cast<std::int64_t>(8 * pushed.size());
            const char* reason = nullptr;
            std::vector<ExitSite> exits;
            std::map<std::size_t, std::pair<std::size_t, std::vector<std::uint8_t>>> moves;
            std::vector<std::size_t> stubbed;
            std::vector<std::size_t> stackMoves;
            // Branches are checked against the scanned code: one past the first int3 reaches code
            // the scan never saw.
            const auto functionEnd = static_cast<std::size_t>(bodyEnd);
            std::int64_t depth = 0;
            bool wholeRedZone = rbpPrologue;
            for (std::size_t k = p; k < body.size(); ++k) {
                const auto use = memoryUse(at(k), body[k]);
                if (!use.RspBased) continue;
                if (use.RspDisp < 0) depth = std::max(depth, -use.RspDisp);
                if (use.Indexed || (use.Address && use.RspDisp < 0)) wholeRedZone = true;
            }
            auto shift = wholeRedZone ? kRedZoneBytes : std::min(kRedZoneBytes, (depth + kStackAlignment - 1) / kStackAlignment * kStackAlignment);
            // A leaf without pushes, allocation or frame pointer whose exits cannot be patched returns
            // through a thunk instead: the entry stub lowers rsp and calls the rest of itself, so every
            // ret lands on `lea rsp, [rsp+shift]; ret`. Its rsp operands also move over that slot.
            const bool frameless = !rbpPrologue && p == 0;
            const char* exitReason = nullptr;
            bool sawRet = false;
            std::int64_t bodyShift = shift;
            std::vector<std::size_t> rspOperands;
            // An instruction moved into a stub: rsp operands move with the shift, RIP-relative ones are
            // relocated to the stub.
            const auto copy = [&](const std::size_t k) -> std::optional<Copied> {
                const auto& insn = body[k];
                Copied out;
                if (insn.Info.FlowKind == ControlFlowKind::ConditionalBranch) {
                    // A jcc goes into the stub as its rel32 form, relocated to its target in the
                    // function; jcxz and loop have no rel32 form.
                    if (!insn.Info.HasBranchTarget || insn.Info.HasRipRelativeDisp || insn.Info.OpcodeOffset != 0) return std::nullopt;
                    const auto op = insn.Info.Opcode;
                    const bool shortForm = !insn.Info.IsTwoByteOpcode && op >= 0x70 && op <= 0x7f;
                    const bool nearForm = insn.Info.IsTwoByteOpcode && op >= 0x80 && op <= 0x8f;
                    if (!shortForm && !nearForm) return std::nullopt;
                    out.Bytes = {0x0f, static_cast<std::uint8_t>(0x80 | (op & 0x0f)), 0, 0, 0, 0};
                    out.Rips.push_back({2, 6, static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp, false});
                    return out;
                }
                if (insn.Info.FlowKind != ControlFlowKind::Sequential) return std::nullopt;
                const auto use = memoryUse(at(k), insn);
                if (use.RspBased) {
                    auto bytes = withDisplacement(at(k), insn, use, use.RspDisp + bodyShift);
                    if (!bytes) return std::nullopt;
                    out.Bytes = std::move(*bytes);
                    return out;
                }
                out.Bytes.assign(at(k), at(k) + insn.Length);
                if (insn.Info.HasRipRelativeDisp) {
                    const auto offset = insn.Info.RipRelativeDispOffset;
                    if (offset + 4 > insn.Length) return std::nullopt;
                    std::int32_t disp = 0;
                    std::memcpy(&disp, at(k) + offset, 4);
                    out.Rips.push_back({offset, insn.Length, static_cast<std::int64_t>(insn.Offset + insn.Length) + disp, insn.Info.AddressSize32});
                }
                return out;
            };
            // Inline assembly calls helpers inside its own function (AES key expansion: `call 1f` per
            // round): straight-line code after a jmp or ret, entered only by calls, ending in a ret
            // without touching the stack. Such calls keep the function a leaf, and the helper's ret is
            // not an exit; a lowered rsp only moves the return address further from the red zone.
            std::set<std::size_t> helperCalls;
            std::set<std::size_t> helperRets;
            for (std::size_t k = p; k < body.size(); ++k) {
                const auto& insn = body[k];
                if (insn.Info.FlowKind != ControlFlowKind::Call || !insn.Info.HasBranchTarget || insn.Info.HasRipRelativeDisp) continue;
                const auto target = static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp;
                if (target <= static_cast<std::int64_t>(start) || target >= static_cast<std::int64_t>(functionEnd)) continue;
                const auto found = indexOf.find(static_cast<std::size_t>(target));
                if (found == indexOf.end() || found->second == 0) continue;
                const auto prior = body[found->second - 1].Info.FlowKind;
                if (prior != ControlFlowKind::UnconditionalJump && prior != ControlFlowKind::Return) continue;
                auto j = found->second;
                while (j < body.size() && body[j].Info.FlowKind == ControlFlowKind::Sequential && !memoryUse(at(j), body[j]).UsesRsp && !pushesOrPops(at(j), body[j])) ++j;
                if (j >= body.size() || body[j].Info.FlowKind != ControlFlowKind::Return || body[j].Length != 1 || at(j)[0] != 0xc3) continue;
                helperCalls.insert(k);
                helperRets.insert(j);
            }
            for (std::size_t k = p; k < body.size(); ++k) {
                const auto& insn = body[k];
                const auto* code = at(k);
                const auto flow = insn.Info.FlowKind;
                if (helperCalls.contains(k) || helperRets.contains(k)) continue;
                // A function that calls is scanned on: some keep values below rsp before their calls
                // (libc's bounds-checked copies compare a slot at [rbp-8] with their own frame).
                if (flow == ControlFlowKind::Call || flow == ControlFlowKind::IndirectCall) { leaf = false; called = true; continue; }
                if (frees.contains(k)) continue;
                const auto use = memoryUse(code, insn);
                if (use.RbpBased && rbpPrologue && use.RbpDisp < -static_cast<std::int64_t>(8 * pushed.size()) - rbpAllocation) redZone = true;
                if (use.RspBased && use.RspDisp < 0) redZone = true;
                // For a function that calls, only a slot below the frame built so far (pushes and every
                // `sub rsp` before the first call) counts: after a call, nothing can live below rsp.
                if (!called) {
                    // `and rsp, -N` realigns the frame by an unknown amount: rbp slots are inside it.
                    if (insn.Length == 4 && code[0] == 0x48 && code[1] == 0x83 && code[2] == 0xe4) frameBeforeCall = std::numeric_limits<std::int64_t>::max() / 2;
                    else if (const auto adjustment = rspAdjustment(code, insn)) { if (*adjustment < 0) frameBeforeCall += -*adjustment; }
                    // An indexed operand ([rsp+rdx-0x10]) addresses an array in the frame: its
                    // displacement says nothing about the red zone. An `lea` stores nothing: below rsp
                    // it is an alloca's new rsp, or an array base biased below its first element.
                    else if (!use.Indexed && !(!insn.Info.IsTwoByteOpcode && insn.Info.Opcode == 0x8d) && ((use.RbpBased && rbpPrologue && use.RbpDisp < -frameBeforeCall) || (use.RspBased && use.RspDisp < 0))) {
                        redZoneBeforeCall = true;
                    }
                }
                if (reason != nullptr) continue;
                // With only rbp pushed, `leave` restores rsp from rbp however far rsp was lowered.
                if (code[insn.Info.OpcodeOffset] == 0xc9 && insn.Length == 1 && leaveExits) continue;
                if (code[insn.Info.OpcodeOffset] == 0xc9 || code[insn.Info.OpcodeOffset] == 0xc8) { reason = "leave or enter"; continue; }
                if (frameless && pushesOrPops(code, insn) && exitReason == nullptr) exitReason = "push or pop in a leaf without pushes";
                if (!rbpPrologue && pushesOrPops(code, insn)) stackMoves.push_back(k);
                if (rbpPrologue) {
                    if (use.UsesRsp) { reason = "other use of rsp"; continue; }
                } else {
                    if (use.RegisterRsp && frameless && rspAdjustment(code, insn)) {
                        if (exitReason == nullptr) exitReason = "rsp moved in a leaf without pushes";
                        continue;
                    }
                    if (use.RegisterRsp) { reason = "other use of rsp"; continue; }
                    if (use.RspBased) {
                        if (code[insn.Info.OpcodeOffset] == 0x62) { reason = "rsp operand that cannot move with rsp"; continue; }
                        if (overlapsExisting(insn.Offset, insn.Offset + insn.Length)) { reason = "overlaps another stub"; continue; }
                        rspOperands.push_back(k);
                    }
                }
                // A tail call: a jmp to another function's start (or to code outside every function,
                // such as a PLT stub), or an indirect jmp right after the pops that undo the prologue;
                // an indirect jmp in a leaf without pushes may be a jump table and is refused.
                // A jump table's cases are branch targets inside the leaf, like a jcc's.
                if (flow == ControlFlowKind::IndirectJump && tables.contains(k)) continue;
                const bool pointerJump = pointerJumpAt(k);
                bool tailCall = false;
                if (flow == ControlFlowKind::IndirectJump) {
                    if (!prologueFrame && !pointerJump) { reason = "indirect jump"; continue; }
                    if (use.RspBased) { reason = "indirect jump through an rsp operand"; continue; }
                    tailCall = true;
                }
                if ((flow == ControlFlowKind::UnconditionalJump || flow == ControlFlowKind::ConditionalBranch) && insn.Info.HasBranchTarget) {
                    const auto target = static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp;
                    if (insn.Info.HasRipRelativeDisp || target < static_cast<std::int64_t>(start) || target >= static_cast<std::int64_t>(functionEnd)) {
                        if (flow != ControlFlowKind::UnconditionalJump || !tailCallTarget(target)) { reason = "branch leaves the function"; continue; }
                        tailCall = true;
                    }
                }
                if (flow == ControlFlowKind::Return || tailCall) {
                    if (flow == ControlFlowKind::Return && (insn.Length != 1 || code[0] != 0xc3)) { reason = "ret with an immediate"; continue; }
                    if (flow == ControlFlowKind::Return) sawRet = true;
                    if (exitReason != nullptr) continue;
                    // A frame-pointer function that pushes only rbp leaves through `leave`: its 1-byte
                    // `pop rbp` becomes one, and an exit that already has one needs nothing.
                    if (leaveExits && k > p && body[k - 1].Length == 1 && body[k - 1].Offset + 1 == insn.Offset) {
                        const auto prior = at(k - 1)[0];
                        if (prior == 0xc9) { ++leaveExitCount; continue; }
                        if (prior == 0x5d) { leaves.push_back(k - 1); ++leaveExitCount; continue; }
                    }
                    const auto pops = popsBefore(k);
                    // A jump that does not undo the prologue first is not a tail call: a jump table, or
                    // a jump to a part that still uses the frame.
                    if (!pops) { reason = !tailCall ? "exit does not pop the prologue's registers" : flow == ControlFlowKind::IndirectJump ? "indirect jump" : "branch leaves the function"; continue; }
                    auto q = *pops;
                    if (allocation > 0) {
                        if (q == p || !frees.contains(q - 1)) { reason = "exit does not free the prologue's allocation"; continue; }
                        --q;
                    }
                    if (flow == ControlFlowKind::IndirectJump && q == k && !pointerJump) { reason = "indirect jump"; continue; }
                    ExitSite exit{q, q, body[q].Offset, body[q].Offset, insn.Offset + insn.Length, {}};
                    exit.Trail.Bytes.assign(at(q), at(k));
                    if (!tailCall) exit.Trail.Bytes.push_back(0xc3);
                    else if (const auto jump = copiedJump(k)) exit.Trail.Append(*jump);
                    else { reason = "indirect jump"; continue; }
                    while (exit.End - exit.Begin < kJumpBytes && exit.First > p) {
                        const auto j = exit.First - 1;
                        const auto& prior = body[j];
                        if (prior.Offset + prior.Length != exit.Begin || (prior.Info.FlowKind != ControlFlowKind::Sequential && prior.Info.FlowKind != ControlFlowKind::ConditionalBranch) || frees.contains(j) || pushesOrPops(at(j), prior)) break;
                        const auto copied = copy(j);
                        if (!copied) break;
                        exit.Lead.Prepend(*copied);
                        if (memoryUse(at(j), prior).RspBased) ++exit.Operands;
                        exit.Begin = prior.Offset;
                        exit.First = j;
                    }
                    const char* exitFailure = nullptr;
                    if (exit.End - exit.Begin < kJumpBytes) exitFailure = "exit shorter than a jump";
                    else if (targetInside(exit.Begin, exit.End)) exitFailure = exit.First == exit.Tail ? "branch into an exit" : "branch into the instructions before a short exit";
                    if (exitFailure != nullptr) {
                        if (frameless) exitReason = exitFailure;
                        else reason = exitFailure;
                        continue;
                    }
                    exits.push_back(std::move(exit));
                }
            }
            if (!redZone || (!leaf && !redZoneBeforeCall)) break;
            if (!asThunk) ++report.RedZoneFunctions;
            // A function that calls is framed only with a frame pointer: its unwind data finds the
            // caller's frame through rbp, and its callees build their frames below whatever rsp is.
            // Without one, outgoing stack arguments and unwinding are both relative to rsp.
            if (!leaf && !rbpPrologue) {
                skip("function with calls keeps values below rsp without a frame pointer");
                break;
            }
            // The thunk's slot sits where the leaf's own pushes go: they must balance on every path, so
            // that each ret finds the thunk's address (a fragment popping its caller's pushes would not).
            // The stack below the entry at each instruction, from the entry down: pushed slots and
            // allocated bytes.
            struct Slot {
                bool Pushed;
                std::int64_t Bytes;
                bool operator==(const Slot&) const = default;
            };
            std::map<std::size_t, std::vector<Slot>> layoutAt{{0, {}}};
            const auto bytesOf = [](const std::vector<Slot>& layout) {
                std::int64_t total = 0;
                for (const auto& slot : layout) total += slot.Bytes;
                return total;
            };
            const auto balanced = [&] {
                std::map<std::size_t, std::size_t> index;
                for (std::size_t k = 0; k < body.size(); ++k) index[body[k].Offset] = k;
                std::vector<std::size_t> pending{0};
                const auto reach = [&](const std::size_t k, const std::vector<Slot>& layout) {
                    const auto [it, inserted] = layoutAt.emplace(k, layout);
                    if (inserted) pending.push_back(k);
                    return it->second == layout;
                };
                while (!pending.empty()) {
                    const auto k = pending.back();
                    pending.pop_back();
                    const auto& insn = body[k];
                    auto layout = layoutAt.at(k);
                    if (pushesOrPops(at(k), insn)) {
                        // A 16-bit push or pop (66 prefix) moves rsp by 2: the slots would not add up.
                        if (std::find(at(k), at(k) + insn.Info.OpcodeOffset, 0x66) != at(k) + insn.Info.OpcodeOffset) return false;
                        const auto op = insn.Info.Opcode;
                        const bool pop = insn.Info.IsTwoByteOpcode ? (op == 0xa1 || op == 0xa9) : ((op >= 0x58 && op <= 0x5f) || op == 0x9d || op == 0x8f);
                        if (!pop) layout.push_back({true, 8});
                        else if (layout.empty() || !layout.back().Pushed) return false;
                        else layout.pop_back();
                    } else if (const auto adjustment = rspAdjustment(at(k), insn)) {
                        if (*adjustment < 0) layout.push_back({false, -*adjustment});
                        // An add frees whole slots, a pushed one included (`push rax` as a slot).
                        for (auto left = *adjustment; left > 0;) {
                            if (layout.empty() || layout.back().Bytes > left) return false;
                            left -= layout.back().Bytes;
                            layout.pop_back();
                        }
                    }
                    const auto flow = insn.Info.FlowKind;
                    // A ret, or a tail call that leaves through its own exit stub, with the frame undone.
                    if (flow == ControlFlowKind::Return || thunkTailCall(k)) {
                        if (!layout.empty()) return false;
                        continue;
                    }
                    if (flow == ControlFlowKind::IndirectJump) {
                        const auto table = tables.find(k);
                        if (table == tables.end()) return false;
                        for (const auto c : table->second)
                            if (!reach(c, layout)) return false;
                        continue;
                    }
                    if ((flow == ControlFlowKind::ConditionalBranch || flow == ControlFlowKind::UnconditionalJump) && insn.Info.HasBranchTarget) {
                        const auto target = index.find(static_cast<std::size_t>(static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp));
                        if (target == index.end() || !reach(target->second, layout)) return false;
                    }
                    if (flow == ControlFlowKind::Sequential || flow == ControlFlowKind::ConditionalBranch) {
                        if (k + 1 >= body.size() || !reach(k + 1, layout)) return false;
                    } else if (flow != ControlFlowKind::UnconditionalJump) {
                        return false;
                    }
                }
                return true;
            };
            const bool thunk = frameless && reason == nullptr && exitReason != nullptr && balanced();
            if (frameless && reason == nullptr && exitReason != nullptr && !thunk) reason = rspArithmetic ? "rsp arithmetic that does not balance" : exitReason;
            if (thunk) {
                exits.clear();
                // The leaf's pushes and allocations now go below the thunk's slot while its rsp operands
                // keep their addresses, so the gap must hold every frame address an operand reaches,
                // and no operand may read a pushed slot: the push went below the gap.
                std::int64_t need = 0;
                std::int64_t deepest = 0;
                for (const auto& [k, layout] : layoutAt) deepest = std::max(deepest, bytesOf(layout));
                for (const auto k : rspOperands) {
                    const auto use = memoryUse(at(k), body[k]);
                    if (pushesOrPops(at(k), body[k])) { reason = "push or pop with an rsp operand in a thunk"; break; }
                    const auto found = layoutAt.find(k);
                    const auto below = found != layoutAt.end() ? bytesOf(found->second) : deepest;
                    if (use.RspDisp >= below) continue;
                    if (use.RspDisp >= 0) {
                        if (found == layoutAt.end()) { reason = "rsp operand on a pushed register"; break; }
                        std::int64_t offset = 0;
                        bool pushedSlot = false;
                        for (auto slot = found->second.rbegin(); slot != found->second.rend(); ++slot) {
                            if (use.RspDisp < offset + slot->Bytes) { pushedSlot = slot->Pushed; break; }
                            offset += slot->Bytes;
                        }
                        if (pushedSlot) { reason = "rsp operand on a pushed register"; break; }
                    }
                    need = std::max(need, below - use.RspDisp);
                }
                if (wholeRedZone) need = std::max(need, kRedZoneBytes + deepest);
                shift = std::max(shift, (need + kStackAlignment - 1) / kStackAlignment * kStackAlignment);
                if (reason == nullptr && shift > kThunkGapLimit) reason = "frame too deep for a thunk";
                bodyShift = shift + 8;
                // A tail call leaves through an exit stub that raises rsp over the gap and the
                // thunk's slot, so the callee finds the caller's return address and stack arguments.
                for (const auto& [k, layout] : layoutAt) {
                    if (reason != nullptr) break;
                    if (!thunkTailCall(k)) continue;
                    const auto jump = copiedJump(k);
                    if (!jump) { reason = "indirect jump"; break; }
                    ExitSite exit{k, k, body[k].Offset, body[k].Offset, body[k].Offset + body[k].Length, {}};
                    exit.Trail = *jump;
                    while (exit.End - exit.Begin < kJumpBytes && exit.First > 0) {
                        const auto j = exit.First - 1;
                        const auto& prior = body[j];
                        if (prior.Offset + prior.Length != exit.Begin || (prior.Info.FlowKind != ControlFlowKind::Sequential && prior.Info.FlowKind != ControlFlowKind::ConditionalBranch) || pushesOrPops(at(j), prior) || rspAdjustment(at(j), prior)) break;
                        const auto copied = copy(j);
                        if (!copied) break;
                        exit.Lead.Prepend(*copied);
                        if (memoryUse(at(j), prior).RspBased) ++exit.Operands;
                        exit.Begin = prior.Offset;
                        exit.First = j;
                    }
                    if (exit.End - exit.Begin < kJumpBytes) reason = "exit shorter than a jump";
                    else if (targetInside(exit.Begin, exit.End)) reason = exit.First == exit.Tail ? "branch into an exit" : "branch into the instructions before a short exit";
                    else exits.push_back(std::move(exit));
                }
                if (reason == nullptr && !sawRet && exits.empty()) reason = "no ret exit";
            }
            for (const auto k : rspOperands) {
                const auto use = memoryUse(at(k), body[k]);
                if (auto moved = movedDisplacement(use, bodyShift)) moves.emplace(k, std::pair{body[k].Offset + use.DispOffset, std::move(*moved)});
                else stubbed.push_back(k);
            }
            const auto prologueEnd = p > 0 ? body[p - 1].Offset + body[p - 1].Length : start;
            if (reason == nullptr && !thunk && std::any_of(stackMoves.begin(), stackMoves.end(), [&](const std::size_t k) { return std::none_of(exits.begin(), exits.end(), [&](const ExitSite& exit) { return exit.Begin <= body[k].Offset && body[k].Offset < exit.End; }); })) reason = "push or pop inside the body";
            auto entryEnd = prologueEnd;
            auto entryLast = p;
            Copied entryTail;
            std::size_t entryOperands = 0;
            while (entryEnd - start < kJumpBytes && entryLast < body.size()) {
                const auto& covered = body[entryLast];
                if (covered.Offset != entryEnd || (covered.Info.FlowKind != ControlFlowKind::Sequential && covered.Info.FlowKind != ControlFlowKind::ConditionalBranch) || frees.contains(entryLast) || (!thunk && pushesOrPops(at(entryLast), covered))) break;
                if (std::any_of(exits.begin(), exits.end(), [&](const ExitSite& exit) { return covered.Offset < exit.End && exit.Begin < covered.Offset + covered.Length; })) break;
                const auto copied = copy(entryLast);
                if (!copied) break;
                entryTail.Append(*copied);
                if (memoryUse(at(entryLast), covered).RspBased) ++entryOperands;
                entryEnd += covered.Length;
                ++entryLast;
            }
            if (reason == nullptr && entryEnd - start < kJumpBytes) reason = "prologue shorter than a jump";
            if (reason == nullptr && targetInside(start, entryEnd)) reason = entryEnd == prologueEnd ? "branch into the prologue" : "branch into the instructions after a short prologue";
            // A jump from another function to the start enters like a call and runs the stub; only a
            // branch from inside the leaf would lower rsp a second time.
            const bool loopsToStart = std::any_of(body.begin(), body.end(), [&](const Insn& insn) {
                return insn.Info.HasBranchTarget && !insn.Info.HasRipRelativeDisp && insn.Info.FlowKind != ControlFlowKind::Call && static_cast<std::int64_t>(insn.Offset + insn.Length) + insn.Info.BranchDisp == static_cast<std::int64_t>(start);
            });
            if (reason == nullptr && entryEnd != prologueEnd && loopsToStart) reason = "branch to the start of a widened prologue";
            if (reason == nullptr && !thunk && exits.empty() && leaveExitCount == 0) reason = "no ret exit";
            if (reason == nullptr && std::any_of(leaves.begin(), leaves.end(), [&](const std::size_t k) { return body[k].Offset < entryEnd || overlapsExisting(body[k].Offset, body[k].Offset + 1); })) reason = "prologue shorter than a jump";
            if (reason == nullptr && std::any_of(exits.begin(), exits.end(), [&](const ExitSite& exit) { return exit.Begin < entryEnd; })) reason = "prologue shorter than a jump";
            if (reason == nullptr && (overlapsExisting(start, entryEnd) || std::any_of(exits.begin(), exits.end(), [&](const ExitSite& exit) { return overlapsExisting(exit.Begin, exit.End); }))) reason = "overlaps another stub";
            const auto carried = [&](const std::size_t k) {
                moves.erase(k);
                stubbed.erase(std::remove(stubbed.begin(), stubbed.end(), k), stubbed.end());
            };
            for (auto k = p; k < entryLast; ++k) carried(k);
            for (const auto& exit : exits)
                for (auto k = exit.First; k < exit.Tail; ++k) carried(k);
            std::vector<OperandStub> stubs;
            const auto overlapsSite = [&](const std::size_t begin, const std::size_t end) {
                if (begin < entryEnd && start < end) return true;
                if (std::any_of(exits.begin(), exits.end(), [&](const ExitSite& exit) { return begin < exit.End && exit.Begin < end; })) return true;
                return std::any_of(stubs.begin(), stubs.end(), [&](const OperandStub& stub) { return begin < stub.End && stub.Begin < end; });
            };
            for (const auto k : stubbed) {
                if (reason != nullptr) break;
                if (!stubs.empty() && k < stubs.back().Last) continue;
                OperandStub stub{k, k, body[k].Offset, body[k].Offset, {}};
                while (stub.Last < body.size() && stub.End - stub.Begin < kJumpBytes) {
                    const auto& covered = body[stub.Last];
                    if (covered.Offset != stub.End || covered.Info.FlowKind != ControlFlowKind::Sequential || frees.contains(stub.Last)) break;
                    const auto copied = copy(stub.Last);
                    if (!copied) break;
                    stub.Body.Append(*copied);
                    if (memoryUse(at(stub.Last), covered).RspBased) ++stub.Operands;
                    stub.End += covered.Length;
                    ++stub.Last;
                }
                while (stub.End - stub.Begin < kJumpBytes && stub.First > p) {
                    const auto j = stub.First - 1;
                    const auto& prior = body[j];
                    if (prior.Offset + prior.Length != stub.Begin || prior.Info.FlowKind != ControlFlowKind::Sequential || frees.contains(j)) break;
                    const auto copied = copy(j);
                    if (!copied) break;
                    stub.Body.Prepend(*copied);
                    if (memoryUse(at(j), prior).RspBased) ++stub.Operands;
                    stub.Begin = prior.Offset;
                    stub.First = j;
                }
                if (stub.End - stub.Begin < kJumpBytes || targetInside(stub.Begin, stub.End) || overlapsSite(stub.Begin, stub.End) || overlapsExisting(stub.Begin, stub.End)) reason = "rsp operand that cannot move and has no room for a stub";
                else stubs.push_back(std::move(stub));
            }
            // Code entered from outside its start runs there with rsp as the leaf keeps it, so the
            // leaf's own code must stay as it is: it is relocated whole instead.
            if (reason == nullptr && enteredFromOutside(start + 1, static_cast<std::size_t>(bodyEnd))) reason = "entered from outside its start";
            if (reason != nullptr) {
                // Relocation refuses calls: a function that calls reports why it was not framed in place.
                if (!leaf) {
                    skip(reason);
                    break;
                }
                if (!asThunk && !rbpPrologue && (!pushed.empty() || allocation > 0)) {
                    firstReason = reason;
                    continue;
                }
                const auto relocation = relocate();
                if (relocation == nullptr) break;
                skip(relocation);
                break;
            }
            for (const auto& stub : stubs)
                for (auto k = stub.First; k < stub.Last; ++k) moves.erase(k);

            const auto addSite = [&](const std::size_t begin, const std::size_t end, Copied code) {
                TrampolineSite site;
                site.Offset = ph.Offset + begin;
                site.Address = ph.MappedAddress + begin;
                site.Length = end - begin;
                site.OriginalBytes.assign(seg.begin() + static_cast<std::ptrdiff_t>(begin), seg.begin() + static_cast<std::ptrdiff_t>(end));
                site.ReturnBranchOffset = code.Bytes.size();
                for (const auto& rip : code.Rips) site.Relocations.push_back({rip.Offset, rip.End, rip.Target - static_cast<std::int64_t>(begin), rip.AddressSize32});
                code.Bytes.insert(code.Bytes.end(), std::begin(kJumpBack), std::end(kJumpBack));
                site.Body = std::move(code.Bytes);
                taken.emplace_back(begin, end);
                sites.push_back(std::move(site));
            };
            const auto original = [&](const std::size_t begin, const std::size_t end) { return Copied{std::vector<std::uint8_t>(seg.begin() + static_cast<std::ptrdiff_t>(begin), seg.begin() + static_cast<std::ptrdiff_t>(end)), {}}; };
            auto entry = original(start, prologueEnd);
            entry.Append(Copied{lowerRsp(shift), {}});
            if (thunk) {
                // call over the thunk: pushes the thunk's address as the leaf's return address.
                const auto raise = raiseRsp(shift);
                const auto over = static_cast<std::uint32_t>(raise.size() + 1);
                entry.Append(Copied{{0xe8, static_cast<std::uint8_t>(over), static_cast<std::uint8_t>(over >> 8), static_cast<std::uint8_t>(over >> 16), static_cast<std::uint8_t>(over >> 24)}, {}});
                entry.Append(Copied{raise, {}});
                entry.Append(Copied{{0xc3}, {}});
                ++report.ReturnThunks;
            }
            entry.Append(entryTail);
            addSite(start, entryEnd, std::move(entry));
            for (auto& exit : exits) {
                auto code = std::move(exit.Lead);
                code.Append(Copied{raiseRsp(thunk ? shift + 8 : shift), {}});
                code.Append(exit.Trail);
                addSite(exit.Begin, exit.End, std::move(code));
            }
            for (auto& stub : stubs) addSite(stub.Begin, stub.End, std::move(stub.Body));
            for (auto& [k, move] : moves) displacements.push_back({ph.Offset + move.first, std::move(move.second)});
            for (const auto k : leaves) displacements.push_back({ph.Offset + body[k].Offset, {0xc9}});
            report.LeaveExits += leaveExitCount;
            ++report.Framed;
            if (!rbpPrologue) ++report.FramedWithoutFramePointer;
            report.Exits += exits.size();
            if (entryLast != p) ++report.WidenedPrologues;
            report.WidenedOperands += entryOperands;
            for (const auto& exit : exits) {
                if (exit.First != exit.Tail) ++report.WidenedExits;
                report.WidenedOperands += exit.Operands;
            }
            report.MovedOperands += moves.size();
            if (!leaf) ++report.FramedWithCalls;
            report.OperandStubs += stubs.size();
            for (const auto& stub : stubs) report.StubbedOperands += stub.Operands;
            break;
        }
    }
}

// Decodes code outside every function's bounds one instruction at a time: visit(offset, length),
// with length 0 for a byte that does not decode (data), which is then stepped over.
template<typename TVisit>
void walkUncovered(const std::vector<std::uint8_t>& code, TVisit&& visit) {
    const X64InstructionDecoder decoder;
    for (std::size_t at = 0; at < code.size();) {
        std::size_t length = 0;
        try {
            length = decoder.Decode(code.data() + at, code.size() - at);
        } catch (const std::exception&) {
            length = 0;
        }
        visit(at, length);
        at += length == 0 ? 1 : length;
    }
}

// Padding between functions: int3, bytes that do not decode, and `add [rax], al` (zero fill) end
// a function; a nop does not start one (functions align loops with nops, so it ends none).
enum class UncoveredKind { Code, Nop, Break };

UncoveredKind uncoveredKind(const X64InstructionDecoder& decoder, const std::vector<std::uint8_t>& code, const std::size_t at, const std::size_t length) {
    if (length == 0 || (length == 1 && code[at] == 0xcc) || (length == 2 && code[at] == 0 && code[at + 1] == 0)) return UncoveredKind::Break;
    const auto info = decoder.DecodeInstruction(code.data() + at, length);
    if ((!info.IsTwoByteOpcode && info.Opcode == 0x90 && (info.RexPrefix & 1) == 0) || (info.IsTwoByteOpcode && info.Opcode == 0x1f)) return UncoveredKind::Nop;
    return UncoveredKind::Code;
}

// Code-segment bytes that no function's bounds cover, padding trimmed: the framer never examines
// them, so a red-zone leaf there would stay unframed.
void measureUncovered(const std::vector<std::uint8_t>& fileBytes, const Domain::ProgramHeader& ph, const std::vector<RedZoneFunction>& functions, RedZoneFrameReport& report) {
    const auto segOffset = static_cast<std::size_t>(ph.Offset);
    const auto segSize = static_cast<std::size_t>(ph.FileSize);
    if (segOffset > fileBytes.size() || segSize > fileBytes.size() - segOffset) return;
    const auto* seg = fileBytes.data() + segOffset;
    std::vector<std::pair<std::size_t, std::size_t>> spans;
    for (const auto& function : functions) {
        if (function.End <= ph.MappedAddress || function.Begin >= ph.MappedAddress + segSize) continue;
        spans.emplace_back(static_cast<std::size_t>(std::max(function.Begin, ph.MappedAddress) - ph.MappedAddress), static_cast<std::size_t>(std::min<std::uint64_t>(function.End, ph.MappedAddress + segSize) - ph.MappedAddress));
    }
    std::sort(spans.begin(), spans.end());
    const X64InstructionDecoder decoder;
    const auto gap = [&](const std::size_t begin, const std::size_t end) {
        const std::vector<std::uint8_t> code(seg + begin, seg + end);
        bool belowRsp = false;
        bool plt = false;
        std::optional<std::size_t> first;
        std::size_t last = 0;
        walkUncovered(code, [&](const std::size_t at, const std::size_t length) {
            if (uncoveredKind(decoder, code, at, length) != UncoveredKind::Code) return;
            if (!first) first = at;
            last = at + length;
            const Insn insn{at, length, decoder.DecodeInstruction(code.data() + at, length)};
            if (length == 6 && code[at] == 0xff && code[at + 1] == 0x25) plt = true;
            const auto use = memoryUse(code.data() + at, insn);
            if (use.RspBased && use.RspDisp < 0) belowRsp = true;
        });
        if (!first) return;
        ++report.UncoveredRanges;
        report.UncoveredBytes += last - *first;
        if (plt) ++report.UncoveredPltRanges;
        if (belowRsp) {
            ++report.UncoveredBelowRsp;
            report.UncoveredBelowRspAddresses.push_back(ph.MappedAddress + begin + *first);
        }    };
    std::size_t cursor = 0;
    for (const auto& [begin, end] : spans) {
        if (begin > cursor) gap(cursor, begin);
        cursor = std::max(cursor, end);
    }
    if (cursor < segSize) gap(cursor, segSize);
}

}

std::vector<RedZoneFunction> SplitUncoveredCode(const std::vector<std::uint8_t>& fileBytes, const std::vector<Domain::ProgramHeader>& codeSegments, const std::vector<RedZoneFunction>& functions) {
    std::vector<RedZoneFunction> pieces;
    const auto scanner = MakeInstructionScanner();
    const X64InstructionDecoder decoder;
    for (const auto& ph : codeSegments) {
        const auto segOffset = static_cast<std::size_t>(ph.Offset);
        const auto segSize = static_cast<std::size_t>(ph.FileSize);
        if (segOffset > fileBytes.size() || segSize > fileBytes.size() - segOffset) continue;
        const std::vector<std::uint8_t> seg(fileBytes.begin() + static_cast<std::ptrdiff_t>(segOffset), fileBytes.begin() + static_cast<std::ptrdiff_t>(segOffset + segSize));
        std::set<std::size_t> callTargets;
        for (const auto& match : scanner->ScanCodeSection(seg, 0, seg.size())) {
            const auto info = decoder.DecodeInstruction(seg.data() + match.Offset, match.Length);
            if (info.FlowKind != ControlFlowKind::Call || !info.HasBranchTarget || info.HasRipRelativeDisp) continue;
            const auto target = static_cast<std::int64_t>(match.Offset + match.Length) + info.BranchDisp;
            if (target >= 0 && static_cast<std::size_t>(target) < seg.size()) callTargets.insert(static_cast<std::size_t>(target));
        }
        std::vector<std::pair<std::size_t, std::size_t>> spans;
        for (const auto& function : functions) {
            if (function.End <= ph.MappedAddress || function.Begin >= ph.MappedAddress + segSize) continue;
            spans.emplace_back(static_cast<std::size_t>(std::max(function.Begin, ph.MappedAddress) - ph.MappedAddress), static_cast<std::size_t>(std::min<std::uint64_t>(function.End, ph.MappedAddress + segSize) - ph.MappedAddress));
        }
        std::sort(spans.begin(), spans.end());
        const auto gap = [&](const std::size_t begin, const std::size_t end) {
            const std::vector<std::uint8_t> code(seg.begin() + static_cast<std::ptrdiff_t>(begin), seg.begin() + static_cast<std::ptrdiff_t>(end));
            // A piece runs from its first instruction that is not padding to the end of its last one
            // that is not a nop.
            std::optional<std::size_t> start;
            std::size_t last = 0;
            // The furthest direct branch target in the piece so far, and whether it has an indirect
            // jump (a jump table's cases can sit anywhere after it).
            std::optional<std::int64_t> reach;
            bool opaque = false;
            const auto close = [&]() {
                if (start && last > *start) pieces.push_back({ph.MappedAddress + begin + *start, ph.MappedAddress + begin + last});
                start.reset();
                reach.reset();
                opaque = false;
            };
            // Functions follow each other with no padding (16-byte ones always do): after a ret, jmp
            // or ud2, a `push rbp; mov rbp, rsp` prologue starts the next one, and so does any code no
            // branch before it reaches (a shrink-wrapped function tests before it pushes).
            bool afterExit = false;
            walkUncovered(code, [&](const std::size_t at, const std::size_t length) {
                const auto kind = uncoveredKind(decoder, code, at, length);
                if (kind == UncoveredKind::Break) { close(); afterExit = false; return; }
                if (kind == UncoveredKind::Nop) return;
                const bool prologue = at + 4 <= code.size() && code[at] == 0x55 && code[at + 1] == 0x48 && code[at + 2] == 0x89 && code[at + 3] == 0xe5;
                const bool unreached = !opaque && (!reach || *reach < static_cast<std::int64_t>(at));
                if (start && (callTargets.contains(begin + at) || (afterExit && (prologue || unreached)))) close();
                if (!start) start = at;
                last = at + length;
                const auto info = decoder.DecodeInstruction(code.data() + at, length);
                if (info.FlowKind == ControlFlowKind::IndirectJump) opaque = true;
                if ((info.FlowKind == ControlFlowKind::ConditionalBranch || info.FlowKind == ControlFlowKind::UnconditionalJump) && info.HasBranchTarget && !info.HasRipRelativeDisp) {
                    const auto target = static_cast<std::int64_t>(at + length) + info.BranchDisp;
                    reach = reach ? std::max(*reach, target) : target;
                }
                afterExit = info.FlowKind == ControlFlowKind::Return || info.FlowKind == ControlFlowKind::UnconditionalJump || info.FlowKind == ControlFlowKind::IndirectJump || (length == 2 && code[at] == 0x0f && code[at + 1] == 0x0b);
            });
            close();
        };
        std::size_t cursor = 0;
        for (const auto& [begin, end] : spans) {
            if (begin > cursor) gap(cursor, begin);
            cursor = std::max(cursor, end);
        }
        if (cursor < segSize) gap(cursor, segSize);
    }
    return pieces;
}

std::vector<TrampolineSite> FrameRedZoneFunctions(
    const std::vector<std::uint8_t>& fileBytes,
    const std::vector<Domain::ProgramHeader>& codeSegments,
    const std::vector<RedZoneFunction>& functions,
    const std::vector<TrampolineSite>& existing,
    RedZoneFrameReport& report,
    std::vector<RedZoneDisplacement>& displacements,
    const std::vector<Domain::ProgramHeader>& loadSegments,
    const std::map<std::uint64_t, std::uint64_t>& relativeAddends
) {
    std::vector<TrampolineSite> sites;
    for (const auto& ph : codeSegments) {
        frameSegment(fileBytes, ph, functions, existing, sites, displacements, report, loadSegments, relativeAddends);
        measureUncovered(fileBytes, ph, functions, report);
    }
    return sites;
}

}
