#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

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

struct KeySet {
    std::array<std::vector<std::uint64_t>, 3> bits;
    bool shaped = false;
};

KeySet& keySet() {
    static KeySet held;
    return held;
}

std::size_t tableSlots(std::size_t slots) {
    return std::bit_ceil(std::clamp<std::size_t>(slots, 1, std::size_t{1} << 30u));
}

}

bool RuntimeRegister(std::uint32_t offset) {
    return offset - PixelUserData < UserDataWords || offset - 0x04cu < 16u || offset - GeometryUserData < UserDataWords || offset - HullUserData < UserDataWords || offset == 0x082u || offset == 0x083u || offset == 0x102u || offset == 0x103u;
}

bool KeyRegister(Bank bank, std::uint32_t offset) {
    const auto& set = keySet();
    if (!set.shaped) return !UserDataRegister(bank, offset);
    const auto& bits = set.bits[static_cast<std::size_t>(bank)];
    return offset / 64u < bits.size() && ((bits[offset / 64u] >> (offset % 64u)) & 1u) != 0u;
}

void SetKeyRegisters(Bank bank, std::span<const std::uint64_t> bits) {
    auto& set = keySet();
    set.bits[static_cast<std::size_t>(bank)].assign(bits.begin(), bits.end());
    set.shaped = true;
}

void ClearKeyRegisters() {
    auto& set = keySet();
    for (auto& bits : set.bits) bits.clear();
    set.shaped = false;
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
        if (KeyRegister(bank, offset)) toggle(digest, offset, value);
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

TeachFilter::TeachFilter(std::size_t slots) : entries(tableSlots(slots)) {}

bool TeachFilter::Ask(std::uint64_t hash, std::uint64_t now) {
    if (hash == 0) hash = 1;
    Entry& entry = entries[static_cast<std::size_t>(hash) & (entries.size() - 1u)];
    if (entry.hash != hash) {
        entry = Entry{hash, 0, 0};
        return false;
    }
    if (now < entry.until) return false;
    entry.until = now + (Wait << std::min(entry.asks, MostDoublings));
    ++entry.asks;
    return true;
}

void TeachFilter::Clear() {
    std::fill(entries.begin(), entries.end(), Entry{});
}

RefusalMemory::RefusalMemory(std::size_t count) : slots(std::make_unique<std::atomic<std::uint64_t>[]>(tableSlots(count))), mask(tableSlots(count) - 1u) {
    Clear();
}

bool RefusalMemory::Holds(std::uint64_t hash) const {
    if (hash == 0) hash = 1;
    return slots[static_cast<std::size_t>(hash) & mask].load(std::memory_order_relaxed) == hash;
}

void RefusalMemory::Keep(std::uint64_t hash) {
    if (hash == 0) hash = 1;
    slots[static_cast<std::size_t>(hash) & mask].store(hash, std::memory_order_relaxed);
}

void RefusalMemory::Clear() {
    for (std::size_t index = 0; index <= mask; ++index) slots[index].store(0, std::memory_order_relaxed);
}

}
