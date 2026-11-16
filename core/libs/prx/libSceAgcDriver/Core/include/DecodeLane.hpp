#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODELANE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_DECODELANE_HPP

#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Core {

enum class Lane : std::uint8_t { Skip, Register, RegisterList, Parameter, Draw, Heavy };

[[nodiscard]] Lane LaneOf(std::uint32_t header);
[[nodiscard]] std::size_t LaneWords(std::uint32_t header);

struct LaneRun { std::size_t end = 0; std::uint32_t packets = 0; std::uint32_t draws = 0; };

[[nodiscard]] LaneRun ScanLane(std::span<const std::uint32_t> commands, std::size_t cursor, std::size_t limit, std::uint32_t most);

}

#endif
