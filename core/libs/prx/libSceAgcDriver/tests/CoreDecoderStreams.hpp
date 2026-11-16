#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_COREDECODERSTREAMS_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_COREDECODERSTREAMS_HPP

#include "prx/libSceAgcDriver/Core/include/Decoder.hpp"
#include "prx/libSceAgcDriver/Core/include/RunKey.hpp"
#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

namespace DecoderTests {

using AgcDriver::QueueState;
using AgcDriver::Registers;
using AgcDriver::Core::Bank;

struct Random {
    std::uint64_t state = 0;

    std::uint64_t Next() {
        auto value = state += 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31u);
    }
    std::uint32_t Below(std::uint32_t limit) { return static_cast<std::uint32_t>(Next() % limit); }
    bool OneIn(std::uint32_t chances) { return Below(chances) == 0; }
};

struct Stream {
    std::vector<std::uint32_t> words;
    AgcDriver::Core::Decoder::Lists lists;
    std::size_t draws = 0;

    std::size_t Packet(std::uint32_t opcode, std::initializer_list<std::uint32_t> payload) { return Packet(opcode, std::span<const std::uint32_t>(payload.begin(), payload.size())); }
    std::size_t Packet(std::uint32_t opcode, std::span<const std::uint32_t> payload) {
        const auto at = words.size();
        words.push_back(0xc0000000u | (static_cast<std::uint32_t>(payload.size() - 1) << 16u) | (opcode << 8u));
        words.insert(words.end(), payload.begin(), payload.end());
        return at;
    }
    std::size_t Set(std::uint32_t opcode, std::uint32_t offset, std::span<const std::uint32_t> values) {
        std::vector<std::uint32_t> payload{offset};
        payload.insert(payload.end(), values.begin(), values.end());
        return Packet(opcode, payload);
    }
    std::size_t List(std::uint32_t opcode, std::vector<std::uint32_t> pairs) {
        const auto at = Packet(opcode, {0x1000u, 0x2u, 0x80000000u, static_cast<std::uint32_t>(pairs.size() / 2)});
        lists[at] = std::move(pairs);
        return at;
    }
    std::size_t DrawAuto(std::uint32_t count) {
        ++draws;
        return Packet(0x2d, {count, 2u});
    }
    std::size_t DrawOffset(std::uint32_t count) {
        ++draws;
        return Packet(0x35, {count, 0u, count, 0u});
    }
};

inline constexpr std::uint32_t HeavyClearState = 0x12u;
inline constexpr std::uint32_t HeavyIndexType = 0x2au;
inline constexpr std::uint32_t HeavyNop = 0x10u;

Bank BankOfOpcode(std::uint32_t opcode);
void ReferenceWrite(QueueState& queue, Bank kind, std::uint32_t offset, std::uint32_t value);
std::uint8_t ReferenceHeavy(QueueState& queue, std::span<const std::uint32_t> packet);
std::size_t ReferenceStep(const Stream& stream, std::size_t cursor, QueueState& queue, bool& draw);
std::vector<std::pair<QueueState, std::uint32_t>> ReferenceDraws(const Stream& stream, QueueState queue);
Stream RandomStream(std::uint64_t seed, std::size_t packets);
std::uint64_t HashState(const Registers& shader, const Registers& context, const Registers& userConfig);

}

#endif
