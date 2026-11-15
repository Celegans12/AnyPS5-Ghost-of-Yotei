#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_RUNKEY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_RUNKEY_HPP

#include <array>
#include <cstdint>
#include <span>
#include <utility>

namespace AgcDriver::Core {

struct BankDigest { std::uint64_t low = 0; std::uint64_t high = 0; bool operator==(const BankDigest&) const = default; };

enum class Bank : std::uint8_t { Shader, Context, UserConfig };

[[nodiscard]] bool UserDataRegister(Bank bank, std::uint32_t offset);
void DigestWrite(BankDigest& digest, std::uint32_t offset, bool had, std::uint32_t before, std::uint32_t after);
[[nodiscard]] BankDigest DigestOf(std::span<const std::pair<std::uint32_t, std::uint32_t>> registers, Bank bank);

struct RunMask {
    static constexpr std::uint32_t Words = 1u, Parameters = 2u, State = 4u, Other = 8u;
    std::uint32_t bits = 0;
    void Register(Bank bank, std::uint32_t offset) { bits |= UserDataRegister(bank, offset) ? Words : State; }
    void Packet(std::uint32_t opcode);
    [[nodiscard]] bool Continues() const { return (bits & (State | Other)) == 0; }
    void Drawn() { bits = 0; }
};

struct StateKey {
    BankDigest shader, context, userConfig;
    std::array<std::uint64_t, 4> programs{};
    std::array<std::uint64_t, 4> programHashes{};
    std::uint32_t indexType = 0;
    std::uint32_t topology = 0;
    std::uint64_t device = 0;
    bool operator==(const StateKey&) const = default;
    [[nodiscard]] std::uint64_t Hash() const;
};

}

#endif
