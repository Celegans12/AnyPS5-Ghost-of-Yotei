#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include <cstdint>
#include <span>
#include <utility>

namespace AgcDriver::Core {

namespace {

constexpr std::uint32_t UserDataWords = 32;
constexpr std::uint32_t PixelUserData = 0x00cu;
constexpr std::uint32_t GeometryUserData = 0x08cu;
constexpr std::uint32_t HullUserData = 0x10cu;

enum class Opcode : std::uint32_t {
    Nop = 0x10,
    ClearState = 0x12,
    IndexBufferSize = 0x13,
    IndexBase = 0x26,
    DrawIndex2 = 0x27,
    IndexType = 0x2a,
    DrawIndexAuto = 0x2d,
    NumInstances = 0x2f,
    DrawIndexOffset2 = 0x35,
    SetShRegIndirect = 0x63,
    SetUconfigRegIndirect = 0x64,
    SetContextReg = 0x69,
    SetShReg = 0x76,
    SetUconfigReg = 0x79,
    SetUconfigRegIndex = 0x7a,
    IndexAttributesIndirect = 0x91,
    SetContextRegIndirect = 0x9f,
};

std::uint64_t splitMix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

std::uint64_t murmurMix(std::uint64_t value) {
    value ^= 0x6a09e667f3bcc909ull;
    value = (value ^ (value >> 33u)) * 0xff51afd7ed558ccdull;
    value = (value ^ (value >> 33u)) * 0xc4ceb9fe1a85ec53ull;
    return value ^ (value >> 33u);
}

void toggle(BankDigest& digest, std::uint32_t offset, std::uint32_t value) {
    const auto key = (static_cast<std::uint64_t>(offset) << 32u) | value;
    digest.low ^= splitMix(key);
    digest.high ^= murmurMix(key);
}

}

bool UserDataRegister(Bank bank, std::uint32_t offset) {
    return bank == Bank::Shader && (offset - PixelUserData < UserDataWords || offset - GeometryUserData < UserDataWords || offset - HullUserData < UserDataWords);
}

void DigestWrite(BankDigest& digest, std::uint32_t offset, bool had, std::uint32_t before, std::uint32_t after) {
    if (had) {
        if (before == after) return;
        toggle(digest, offset, before);
    }
    toggle(digest, offset, after);
}

BankDigest DigestOf(std::span<const std::pair<std::uint32_t, std::uint32_t>> registers, Bank bank) {
    BankDigest digest;
    for (const auto& [offset, value] : registers) {
        if (!UserDataRegister(bank, offset)) toggle(digest, offset, value);
    }
    return digest;
}

void RunMask::Packet(std::uint32_t opcode) {
    switch (static_cast<Opcode>(opcode)) {
        case Opcode::IndexBufferSize:
        case Opcode::IndexBase:
        case Opcode::NumInstances:
            bits |= Parameters;
            return;
        case Opcode::DrawIndex2:
        case Opcode::DrawIndexAuto:
        case Opcode::DrawIndexOffset2:
        case Opcode::SetShReg:
        case Opcode::SetShRegIndirect:
        case Opcode::SetContextReg:
        case Opcode::SetContextRegIndirect:
        case Opcode::SetUconfigReg:
        case Opcode::SetUconfigRegIndex:
        case Opcode::SetUconfigRegIndirect:
            return;
        case Opcode::Nop:
        case Opcode::ClearState:
        case Opcode::IndexType:
        case Opcode::IndexAttributesIndirect:
            bits |= Other | State;
            return;
        default:
            bits |= Other;
            return;
    }
}

std::uint64_t StateKey::Hash() const {
    std::uint64_t hash = 0x243f6a8885a308d3ull;
    const auto mix = [&hash](std::uint64_t value) { hash = splitMix(hash ^ value); };
    mix(shader.low);
    mix(shader.high);
    mix(context.low);
    mix(context.high);
    mix(userConfig.low);
    mix(userConfig.high);
    for (const auto program : programs) mix(program);
    for (const auto programHash : programHashes) mix(programHash);
    mix((static_cast<std::uint64_t>(indexType) << 32u) | topology);
    mix(device);
    return hash;
}

}
