#ifndef TOOLS_HW_ORACLE_KYTY_TEMPLATE_HPP
#define TOOLS_HW_ORACLE_KYTY_TEMPLATE_HPP

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace KytyOracle {

inline constexpr std::array<std::uint32_t, 24> TemplatePrologue {
    0xf4080100u, 0xfa000000u, 0x34020084u, 0x34040086u, 0xbf8cc07fu, 0xdc388000u, 0x04040001u, 0xbf8c3f70u,
    0x7e140280u, 0x7e160280u, 0x7e180280u, 0x7e1a0280u, 0x7e1c0280u, 0x7e1e0280u, 0x7e200280u, 0x7e220280u,
    0x7e240280u, 0x7e260280u, 0x7e280280u, 0x7e2a0280u, 0x7e2c0280u, 0x7e2e0280u, 0x7e300280u, 0x7e320280u,
};

inline constexpr std::array<std::uint32_t, 9> TemplateEpilogue {
    0xdc788000u, 0x00060a02u, 0xdc788010u, 0x00060e02u, 0xdc788020u, 0x00061202u, 0xdc788030u, 0x00061602u, 0xbf810000u,
};

inline constexpr std::uint32_t OutputDescriptorSgpr = 0;
inline constexpr std::uint32_t AddressSgpr = 4;
inline constexpr std::uint32_t InputDescriptorSgpr = 8;
inline constexpr std::uint32_t UserSgprs = 12;

struct Program {
    std::vector<std::uint32_t> code;
    std::uint32_t body_begin = 0;
    std::uint32_t body_end = 0;
};

Program RewriteTemplate(std::span<const std::uint32_t> text);

}

#endif
