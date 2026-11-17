#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace Loader {
struct Program;
namespace DemonsSoulsPageFree {
void Install(Program* program);
void Clear();
bool FreeBlocksIn(uint64_t begin, uint64_t end, std::vector<std::pair<uint64_t, uint64_t>>* blocks);

inline constexpr std::string_view FreeBlock = "55 48 89 e5 41 57 41 56 53 48 83 ec 18 4c 8b 35 ?? ?? ?? ?? 48 89 fb 49 c7 c7 ff "
                                              "ff ff ff 31 c9 49 8b 06 48 89 45 e0 48 83 3d ?? ?? ?? ?? 00 74";
inline constexpr std::string_view FreeBlockBody =
    "48 8b 05 ?? ?? ?? ?? 48 8d 73 20 0f b6 4b 70 48 c7 43 20 00 00 00 00 4c 8b 4b 28 4c 8b 43 30 48 8d 14 89";
inline constexpr size_t FreeBlockBodyAt = 0x97;
inline constexpr size_t LockOperandAt = 0x2a, LockOperandEnd = 0x2f;
inline constexpr size_t StatsOperandAt = 0x9a, StatsOperandEnd = 0x9e;
inline constexpr size_t MovedPrologue = 6;
inline constexpr uint32_t SizeAt = 0x50, MappedAt = 0x58, AddressAt = 0x60, NodeAt = 0x20, NextAt = 0x30, ListAt = 0x38;
inline constexpr uint64_t Pools = 2, PoolStride = 0xa0, FreeHeadAt = 0x28, FreeFirstAt = 0x38, FreeCountAt = 0x48,
                          FreePagesAt = 0x70;
}
}
