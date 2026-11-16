#include "CoreDecoderStreams.hpp"
#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace DecoderTests {

using AgcDriver::Core::DigestWrite;
using AgcDriver::Core::KeyRegister;
using AgcDriver::Core::Lane;
using AgcDriver::Core::LaneOf;
using AgcDriver::Core::LaneWords;

Bank BankOfOpcode(std::uint32_t opcode) {
    if (opcode == 0x69 || opcode == 0x9f) return Bank::Context;
    if (opcode == 0x76 || opcode == 0x63) return Bank::Shader;
    return Bank::UserConfig;
}

void ReferenceWrite(QueueState& queue, Bank kind, std::uint32_t offset, std::uint32_t value) {
    auto& bank = kind == Bank::Context ? queue.context : kind == Bank::Shader ? queue.shader : queue.userConfig;
    const bool had = bank.contains(offset);
    const auto before = had ? std::as_const(bank).at(offset) : 0u;
    if (!had || before != value) {
        queue.runMask.Register(kind, offset);
        if ((queue.staleDigests & (1u << static_cast<unsigned>(kind))) == 0 && KeyRegister(kind, offset)) DigestWrite(kind == Bank::Shader ? queue.shaderDigest : kind == Bank::Context ? queue.contextDigest : queue.userConfigDigest, offset, had, before, value);
    }
    bank.insert_or_assign(offset, value);
    if (kind == Bank::UserConfig && offset == 0x243) queue.indexType = value & 3u;
}

std::uint8_t ReferenceHeavy(QueueState& queue, std::span<const std::uint32_t> packet) {
    const auto opcode = (packet[0] >> 8u) & 0xffu;
    queue.runMask.Packet(opcode);
    if (opcode == HeavyClearState) {
        queue.ClearContext();
        return AgcDriver::Core::DecoderContext;
    }
    if (opcode == HeavyIndexType) {
        queue.indexType = packet[1];
        return 0;
    }
    return 0;
}

std::size_t ReferenceStep(const Stream& stream, std::size_t cursor, QueueState& queue, bool& draw) {
    const std::span<const std::uint32_t> words(stream.words);
    draw = false;
    const auto header = words[cursor];
    const auto lane = LaneOf(header);
    if (lane == Lane::Skip) return 1;
    const auto count = LaneWords(header);
    const auto packet = words.subspan(cursor, count);
    const auto opcode = (header >> 8u) & 0xffu;
    if (lane == Lane::Heavy) {
        ReferenceHeavy(queue, packet);
        return count;
    }
    queue.runMask.Packet(opcode);
    if (lane == Lane::Register) {
        if (!(opcode == 0x79 && packet[1] == 0x342)) {
            for (std::size_t index = 2; index < packet.size(); ++index) ReferenceWrite(queue, BankOfOpcode(opcode), (packet[1] & ~0x70000000u) + static_cast<std::uint32_t>(index - 2), packet[index]);
        }
    } else if (lane == Lane::RegisterList) {
        const auto& pairs = stream.lists.at(cursor);
        for (std::size_t index = 0; index < pairs.size(); index += 2) ReferenceWrite(queue, BankOfOpcode(opcode), pairs[index] & ~0x70000000u, pairs[index + 1]);
    } else if (lane == Lane::Parameter) {
        if (opcode == 0x13) queue.indexBufferSize = packet[1];
        else if (opcode == 0x26) queue.indexBase = packet[1] | (static_cast<std::uint64_t>(packet[2]) << 32u);
        else queue.instanceCount = packet[1];
    } else {
        draw = true;
    }
    return count;
}

std::vector<std::pair<QueueState, std::uint32_t>> ReferenceDraws(const Stream& stream, QueueState queue) {
    std::vector<std::pair<QueueState, std::uint32_t>> draws;
    for (std::size_t cursor = 0; cursor < stream.words.size();) {
        bool draw = false;
        const auto count = ReferenceStep(stream, cursor, queue, draw);
        if (draw) {
            draws.emplace_back(queue, queue.runMask.bits);
            queue.runMask.Drawn();
        }
        cursor += count;
    }
    return draws;
}

Stream RandomStream(std::uint64_t seed, std::size_t packets) {
    Random random{seed};
    Stream stream;
    constexpr std::array<std::uint32_t, 4> direct{0x69u, 0x76u, 0x79u, 0x7au};
    constexpr std::array<std::uint32_t, 3> listed{0x63u, 0x64u, 0x9fu};
    constexpr std::array<std::uint32_t, 6> bases{0x0cu, 0x8cu, 0x10cu, 0x200u, 0x240u, 0x318u};
    const auto value = [&] { return random.OneIn(3) ? random.Below(4) : static_cast<std::uint32_t>(random.Next()); };
    for (std::size_t made = 0; made < packets; ++made) {
        const auto choice = random.Below(100);
        if (choice < 45) {
            const auto opcode = direct[random.Below(static_cast<std::uint32_t>(direct.size()))];
            std::vector<std::uint32_t> values(1 + random.Below(8));
            for (auto& word : values) word = value();
            stream.Set(opcode, bases[random.Below(static_cast<std::uint32_t>(bases.size()))] + random.Below(24), values);
        } else if (choice < 60) {
            const auto opcode = listed[random.Below(static_cast<std::uint32_t>(listed.size()))];
            std::vector<std::uint32_t> pairs;
            const auto count = random.Below(12);
            for (std::uint32_t index = 0; index < count; ++index) {
                pairs.push_back(bases[random.Below(static_cast<std::uint32_t>(bases.size()))] + random.Below(24));
                pairs.push_back(value());
            }
            stream.List(opcode, std::move(pairs));
        } else if (choice < 68) {
            const auto which = random.Below(3);
            if (which == 0) stream.Packet(0x13, {random.Below(1u << 20u)});
            else if (which == 1) stream.Packet(0x26, {random.Below(1u << 28u) << 1u, random.Below(0x100u)});
            else stream.Packet(0x2f, {1 + random.Below(64)});
        } else if (choice < 93) {
            if (random.OneIn(2)) stream.DrawAuto(3 + random.Below(300));
            else stream.DrawOffset(3 + random.Below(300));
        } else if (choice < 96) {
            stream.words.push_back(random.OneIn(2) ? 0x80000000u : 0xffff1000u);
        } else {
            const auto which = random.Below(3);
            if (which == 0) stream.Packet(HeavyClearState, {0u});
            else if (which == 1) stream.Packet(HeavyIndexType, {random.Below(3)});
            else stream.Packet(HeavyNop, {0u});
        }
    }
    stream.DrawAuto(3);
    return stream;
}

std::uint64_t HashState(const Registers& shader, const Registers& context, const Registers& userConfig) {
    std::uint64_t hash = 0x243f6a8885a308d3ull;
    const auto mix = [&hash](std::uint64_t word) {
        hash += word + 0x9e3779b97f4a7c15ull;
        hash = (hash ^ (hash >> 30u)) * 0xbf58476d1ce4e5b9ull;
        hash = (hash ^ (hash >> 27u)) * 0x94d049bb133111ebull;
        hash ^= hash >> 31u;
    };
    const auto bank = [&](const Registers& registers, std::uint64_t tag) {
        mix(tag);
        registers.ForEach([&](std::uint32_t offset, std::uint32_t word) { mix((static_cast<std::uint64_t>(offset) << 32u) | word); });
    };
    bank(shader, 1);
    bank(context, 2);
    bank(userConfig, 3);
    return hash;
}

}
