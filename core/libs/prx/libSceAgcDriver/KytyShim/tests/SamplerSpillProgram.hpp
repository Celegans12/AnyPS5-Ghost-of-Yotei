#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace SamplerSpillProgram {

inline constexpr std::size_t RestoreCompare = 15;
inline constexpr std::size_t RestoreBranch  = 16;
inline constexpr std::array<uint32_t, 31> Words {
    0xbe980392u, 0xbe990380u, 0xbe9a0480u, 0xbe940382u, 0x7e1002f0u, 0x7e1202f0u, 0x7e140280u,
    0x3a200003u, 0xd7610018u, 0x00010a18u, 0xd7610018u, 0x00010c19u, 0x7e300510u, 0x7e320500u, 0x06101018u,
    0xbf068080u, 0xbf840004u, 0xd7600018u, 0x00010b18u, 0xd7600019u, 0x00010d18u,
    0xf0900f08u, 0x00c20408u, 0xbf8c3f70u, 0x34160082u, 0xe0701000u, 0x8001040bu,
    0x80948114u, 0xbf078014u, 0xbf85ffeau, 0xbf810000u,
};

}
