#include "prx/libSceAgcDriver/KytyShim/src/EntryPure.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayFormat.hpp"
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace KytyShim::Entry {

namespace {

constexpr std::uint64_t PixelFormatUnormBit = 0x0100000000000000ull;
constexpr std::uint64_t PixelFormatB8G8R8A8 = 0x8000000000000000ull;
constexpr std::uint64_t PixelFormatR8G8B8A8 = 0x8000000022000000ull;
constexpr std::uint32_t IndirectBufferOpcode = 0x3fu;
constexpr std::uint32_t ReleaseMemOpcode = 0x49u;
constexpr std::size_t IndirectScanDepth = 8;
constexpr std::size_t IndirectScanWords = std::size_t{1} << 24u;

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

std::uint32_t Opcode(std::uint32_t header) {
    return (header >> 8u) & 0xffu;
}

bool Type3(std::uint32_t header) {
    return (header >> 30u) == 3u;
}

bool Pending(std::span<const std::uint32_t> pending, std::uint32_t token) {
    return std::find(pending.begin(), pending.end(), token) != pending.end();
}

const std::uint32_t* IndirectTarget(std::uint32_t low, std::uint32_t high) {
    return reinterpret_cast<const std::uint32_t*>(static_cast<std::uintptr_t>(low & ~3u) | (static_cast<std::uintptr_t>(high & 0xffffu) << 32u));
}

std::optional<std::uint32_t> ScanIndirect(std::span<const std::uint32_t> commands, std::span<const std::uint32_t> pending, std::size_t depth, std::size_t& budget) {
    for (std::size_t cursor = 0; cursor < commands.size();) {
        const auto header = commands[cursor];
        const auto words = PacketWords(header);
        if (words == 0 || cursor + words > commands.size()) break;
        const auto packet = commands.subspan(cursor, words);
        if (depth > 0) {
            if (const auto token = EopReleaseToken(packet); token && Pending(pending, *token)) return token;
        }
        if (Type3(header) && Opcode(header) == IndirectBufferOpcode && depth < IndirectScanDepth) {
            const std::uint32_t* target = nullptr;
            std::size_t targetWords = 0;
            if (words == 4) {
                target = IndirectTarget(packet[1], packet[2]);
                targetWords = packet[3] & 0xfffffu;
            } else if (words == 14) {
                target = IndirectTarget(packet[8], packet[9]);
                targetWords = packet[10] & 0xfffffu;
            }
            if (target != nullptr && targetWords != 0 && targetWords <= budget) {
                budget -= targetWords;
                if (const auto token = ScanIndirect(std::span(target, targetWords), pending, depth + 1, budget)) return token;
            }
        }
        cursor += words;
    }
    return std::nullopt;
}

}

std::size_t PacketWords(std::uint32_t header) {
    if ((header >> 30u) == 2u || header == 0xffff1000u) return 1;
    return static_cast<std::size_t>((header >> 16u) & 0x3fffu) + 2u;
}

std::array<std::uint32_t, AgcDriver::FlipPacketWords> EncodeFlipPacket(const AgcDriver::FlipInfo& info) {
    const auto argument = static_cast<std::uint64_t>(info.argument);
    return {AgcDriver::FlipPacketHeader, info.handle, std::bit_cast<std::uint32_t>(info.index), info.mode, static_cast<std::uint32_t>(argument), static_cast<std::uint32_t>(argument >> 32u)};
}

std::optional<AgcDriver::FlipInfo> DecodeFlipPacket(std::span<const std::uint32_t> packet) {
    if (packet.size() < AgcDriver::FlipPacketWords || packet[0] != AgcDriver::FlipPacketHeader) return std::nullopt;
    const auto argument = static_cast<std::uint64_t>(packet[4]) | (static_cast<std::uint64_t>(packet[5]) << 32u);
    return AgcDriver::FlipInfo{packet[1], std::bit_cast<std::int32_t>(packet[2]), packet[3], std::bit_cast<std::int64_t>(argument)};
}

bool HasFlipPacket(std::span<const std::uint32_t> commands) {
    for (std::size_t cursor = 0; cursor < commands.size(); cursor += PacketWords(commands[cursor])) {
        if (commands[cursor] == AgcDriver::FlipPacketHeader) return true;
    }
    return false;
}

std::optional<RenderingWaitTarget> DecodeRenderingWaitPacket(std::span<const std::uint32_t> packet) {
    if (packet.size() < AgcDriver::RenderingWaitPacketWords || packet[0] != AgcDriver::RenderingWaitPacketHeader) return std::nullopt;
    return RenderingWaitTarget{packet[1], packet[2], packet[3]};
}

std::optional<std::uint32_t> EopReleaseToken(std::span<const std::uint32_t> packet) {
    if (packet.size() < 8 || !Type3(packet[0]) || PacketWords(packet[0]) != 8u || Opcode(packet[0]) != ReleaseMemOpcode || ((packet[2] >> 24u) & 7u) != 2u) return std::nullopt;
    return packet[7];
}

std::vector<EopFlipPoint> FindEopFlipPoints(std::span<const std::uint32_t> commands, std::span<const std::uint32_t> pending) {
    std::vector<EopFlipPoint> points;
    if (pending.empty()) return points;
    for (std::size_t cursor = 0; cursor < commands.size();) {
        const auto words = PacketWords(commands[cursor]);
        if (cursor + words > commands.size()) break;
        if (const auto token = EopReleaseToken(commands.subspan(cursor, words)); token && Pending(pending, *token)) {
            require(std::none_of(points.begin(), points.end(), [&](const EopFlipPoint& point) { return point.token == *token; }), "EOP flip token repeated in a submission");
            points.push_back({cursor + words, *token});
        }
        cursor += words;
    }
    return points;
}

std::optional<std::uint32_t> FindIndirectEopToken(std::span<const std::uint32_t> commands, std::span<const std::uint32_t> pending) {
    if (pending.empty()) return std::nullopt;
    std::size_t budget = IndirectScanWords;
    return ScanIndirect(commands, pending, 0, budget);
}

std::size_t DisplayBufferBytes(const AgcDriver::DisplayBuffer& buffer) {
    require(buffer.width != 0 && buffer.height != 0 && buffer.width <= 16384 && buffer.height <= 16384, "VideoOut: invalid display buffer dimensions");
    const bool halfFloat = AgcDriver::DisplayHalfFloat(buffer.pixelFormat);
    const auto base = buffer.pixelFormat & ~PixelFormatUnormBit;
    if (!halfFloat && !AgcDriver::DisplayPq(buffer.pixelFormat) && base != PixelFormatB8G8R8A8 && base != PixelFormatR8G8B8A8) {
        char message[96];
        std::snprintf(message, sizeof(message), "VideoOut: unsupported display pixel format 0x%016llx", static_cast<unsigned long long>(buffer.pixelFormat));
        throw std::runtime_error(message);
    }
    require(!halfFloat || buffer.dccAddress == 0, "VideoOut: DCC metadata on a 16-bit float display buffer is not implemented");
    require(buffer.alignment == 65536u || (buffer.alignment == 4096u && (buffer.tilingMode == 1 || buffer.tilingMode == AgcDriver::DisplayTilingGcn)), "VideoOut: 4 KiB display buffer alignment is only supported for linear and PS4 buffers");
    require(buffer.address != 0 && (buffer.address & (buffer.alignment - 1u)) == 0, buffer.alignment == 65536u ? "VideoOut: display buffer requires 64 KiB alignment" : "VideoOut: display buffer requires 4 KiB alignment");
    require(buffer.tilingMode <= 1 || buffer.tilingMode == AgcDriver::DisplayTilingGcn, "VideoOut: unsupported display tiling mode");
    require(buffer.tilingMode != AgcDriver::DisplayTilingGcn, "VideoOut: PS4 tiled display buffers are not supported by the Kyty GPU driver");
    require(buffer.tilingMode != 0 || buffer.pitchInPixel == 0, "VideoOut: tiled display pitch is unsupported");
    require(buffer.dccAddress == 0 || buffer.tilingMode == 0, "VideoOut: a linear display buffer cannot carry DCC metadata");
    require(buffer.dccAddress != 0 || buffer.dccClearColor == 0, "VideoOut: a DCC clear color needs DCC metadata");
    const auto pitch = buffer.pitchInPixel == 0 ? buffer.width : buffer.pitchInPixel;
    require(pitch >= buffer.width && pitch <= 16384, "VideoOut: invalid linear display pitch");
    const auto texelBytes = AgcDriver::DisplayTexelBytes(buffer.pixelFormat);
    const auto blockHeight = 65536u / 128u / texelBytes;
    const auto size = buffer.tilingMode == 1 ? static_cast<std::uint64_t>(pitch) * buffer.height * texelBytes
        : static_cast<std::uint64_t>((buffer.width + 127u) / 128u) * ((buffer.height + blockHeight - 1u) / blockHeight) * 65536u;
    require(size <= std::numeric_limits<std::size_t>::max() && size <= std::numeric_limits<std::uintptr_t>::max() - buffer.address, "VideoOut: display buffer range overflow");
    return static_cast<std::size_t>(size);
}

void CheckMappedRange(const void* pointer, std::size_t bytes, std::size_t alignment) {
    require(alignment != 0, "zero guest memory alignment");
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    require(address != 0 && address % alignment == 0, "null or misaligned address");
#ifdef _WIN32
    const std::uintptr_t page = 0x1000;
#else
    const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
#endif
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - address - page, "address range overflow");
    if (bytes == 0) return;
    const auto begin = address & ~(page - 1u);
    const auto end = (address + bytes + page - 1u) & ~(page - 1u);
    char text[160];
#ifdef _WIN32
    for (auto cursor = begin; cursor < end;) {
        MEMORY_BASIC_INFORMATION memory{};
        if (VirtualQuery(reinterpret_cast<const void*>(cursor), &memory, sizeof(memory)) != sizeof(memory) || memory.State != MEM_COMMIT) {
            std::snprintf(text, sizeof(text), "guest memory is not mapped in 0x%llx+0x%zx (first unmapped page 0x%llx)", static_cast<unsigned long long>(address), bytes, static_cast<unsigned long long>(cursor));
            throw std::runtime_error(text);
        }
        cursor = reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize;
    }
#else
    if (msync(reinterpret_cast<void*>(begin), end - begin, MS_ASYNC) == 0) return;
    const int error = errno;
    std::snprintf(text, sizeof(text), "guest memory is not mapped in 0x%llx+0x%zx (%s)", static_cast<unsigned long long>(address), bytes, std::strerror(error));
    throw std::runtime_error(text);
#endif
}

}
