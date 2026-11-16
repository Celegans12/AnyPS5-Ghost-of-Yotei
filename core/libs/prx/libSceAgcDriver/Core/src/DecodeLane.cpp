#include "prx/libSceAgcDriver/Core/include/DecodeLane.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Core {

namespace {

constexpr std::uint32_t FillerWord = 0xffff1000u;

enum class Opcode : std::uint32_t {
    IndexBufferSize = 0x13,
    IndexBase = 0x26,
    DrawIndex2 = 0x27,
    DrawIndexAuto = 0x2d,
    NumInstances = 0x2f,
    DrawIndexOffset2 = 0x35,
    SetShRegIndirect = 0x63,
    SetUconfigRegIndirect = 0x64,
    SetContextReg = 0x69,
    SetShReg = 0x76,
    SetUconfigReg = 0x79,
    SetUconfigRegIndex = 0x7a,
    SetContextRegIndirect = 0x9f,
};

constexpr Lane opcodeLane(std::uint32_t opcode) {
    switch (static_cast<Opcode>(opcode)) {
        case Opcode::SetShReg:
        case Opcode::SetContextReg:
        case Opcode::SetUconfigReg:
        case Opcode::SetUconfigRegIndex:
            return Lane::Register;
        case Opcode::SetShRegIndirect:
        case Opcode::SetContextRegIndirect:
        case Opcode::SetUconfigRegIndirect:
            return Lane::RegisterList;
        case Opcode::IndexBufferSize:
        case Opcode::IndexBase:
        case Opcode::NumInstances:
            return Lane::Parameter;
        case Opcode::DrawIndex2:
        case Opcode::DrawIndexAuto:
        case Opcode::DrawIndexOffset2:
            return Lane::Draw;
        default:
            return Lane::Heavy;
    }
}

constexpr std::array<Lane, 256> opcodeLanes() {
    std::array<Lane, 256> lanes{};
    for (std::size_t opcode = 0; opcode < lanes.size(); ++opcode) lanes[opcode] = opcodeLane(static_cast<std::uint32_t>(opcode));
    return lanes;
}

constexpr std::array<Lane, 256> OpcodeLanes = opcodeLanes();

constexpr bool filler(std::uint32_t header) {
    return (header >> 30u) == 2u || header == FillerWord;
}

}

Lane LaneOf(std::uint32_t header) {
    if (filler(header)) return Lane::Skip;
    if ((header >> 30u) != 3u || (header & 1u) != 0) return Lane::Heavy;
    return OpcodeLanes[(header >> 8u) & 0xffu];
}

std::size_t LaneWords(std::uint32_t header) {
    if (filler(header)) return 1u;
    return static_cast<std::size_t>((header >> 16u) & 0x3fffu) + 2u;
}

LaneRun ScanLane(std::span<const std::uint32_t> commands, std::size_t cursor, std::size_t limit, std::uint32_t most) {
    const auto stop = std::min(limit, commands.size());
    LaneRun run;
    run.end = cursor;
    while (run.end < stop && run.packets < most) {
        const auto header = commands[run.end];
        const auto lane = LaneOf(header);
        if (lane == Lane::Heavy) break;
        const auto words = LaneWords(header);
        if (words > stop - run.end) break;
        run.end += words;
        if (lane == Lane::Skip) continue;
        ++run.packets;
        if (lane == Lane::Draw) ++run.draws;
    }
    return run;
}

}
