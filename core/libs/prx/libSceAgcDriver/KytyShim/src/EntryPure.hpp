#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_ENTRYPURE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_KYTYSHIM_SRC_ENTRYPURE_HPP

#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/VideoOutput.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace KytyShim::Entry {

inline constexpr std::uint32_t FlipCustomCode = (AgcDriver::FlipPacketHeader >> 2u) & 0x3fu;
inline constexpr std::uint32_t RenderingWaitCustomCode = (AgcDriver::RenderingWaitPacketHeader >> 2u) & 0x3fu;
inline constexpr std::size_t PendingEopFlipLimit = 16;

struct RenderingWaitTarget {
    std::uint32_t handle = 0;
    std::uint32_t index = 0;
    std::uint32_t mode = 0;
};

struct EopFlipPoint {
    std::size_t end = 0;
    std::uint32_t token = 0;
};

std::size_t PacketWords(std::uint32_t header);
std::array<std::uint32_t, AgcDriver::FlipPacketWords> EncodeFlipPacket(const AgcDriver::FlipInfo& info);
std::optional<AgcDriver::FlipInfo> DecodeFlipPacket(std::span<const std::uint32_t> packet);
std::optional<RenderingWaitTarget> DecodeRenderingWaitPacket(std::span<const std::uint32_t> packet);
std::optional<std::uint32_t> EopReleaseToken(std::span<const std::uint32_t> packet);
std::vector<EopFlipPoint> FindEopFlipPoints(std::span<const std::uint32_t> commands, std::span<const std::uint32_t> pending);
std::optional<std::uint32_t> FindIndirectEopToken(std::span<const std::uint32_t> commands, std::span<const std::uint32_t> pending);
std::size_t DisplayBufferBytes(const AgcDriver::DisplayBuffer& buffer);
void CheckMappedRange(const void* pointer, std::size_t bytes, std::size_t alignment);

}

#endif
