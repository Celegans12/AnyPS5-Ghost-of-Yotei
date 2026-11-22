#ifndef TOOLS_HW_ORACLE_KYTY_CODEOBJECT_HPP
#define TOOLS_HW_ORACLE_KYTY_CODEOBJECT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace KytyOracle {

struct Kernel {
    std::vector<std::uint32_t> text;
    std::uint32_t machine = 0;
    std::uint32_t lds_bytes = 0;
    std::uint32_t wave_size = 64;
    std::uint32_t float_mode = 0;
    bool dx10_clamp = false;
    bool ieee_mode = false;
    bool fp16_overflow = false;
    std::array<bool, 3> group_id {};
    bool tg_size = false;
    std::uint32_t thread_id_components = 1;
};

Kernel ParseCodeObject(std::span<const std::byte> bytes);

std::uint32_t ParseCount(std::string_view text, const char* what);

}

#endif
