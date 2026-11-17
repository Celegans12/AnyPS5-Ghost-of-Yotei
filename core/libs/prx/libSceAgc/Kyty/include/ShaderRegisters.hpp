#ifndef CORE_LIBS_PRX_LIBSCEAGC_KYTY_INCLUDE_SHADERREGISTERS_HPP
#define CORE_LIBS_PRX_LIBSCEAGC_KYTY_INCLUDE_SHADERREGISTERS_HPP

#include <cstdint>
#include <stdexcept>
#include <string>
#include "SceShaders.hpp"

namespace Agc::Kyty {

constexpr int OK = 0;

inline void NotImplemented(bool condition, const char* function, const char* expression) {
    if (condition) {
        throw std::runtime_error(std::string(function) + ": not implemented: " + expression);
    }
}

inline ShaderRegister* find_shader_register(ShaderRegister* regs, std::uint32_t num_regs, std::uint32_t offset, std::uint32_t occurrence = 0) {
    if (regs == nullptr) {
        return nullptr;
    }
    for (std::uint32_t i = 0; i < num_regs; i++) {
        if (regs[i].offset != offset) {
            continue;
        }
        if (occurrence == 0) {
            return regs + i;
        }
        occurrence--;
    }
    return nullptr;
}

inline ShaderRegister* require_shader_register(ShaderRegister* regs, std::uint32_t num_regs, std::uint32_t offset, const char* fn, const char* what, std::uint32_t occurrence = 0) {
    auto* reg = find_shader_register(regs, num_regs, offset, occurrence);
    NotImplemented(reg == nullptr, fn, what);
    return reg;
}

}

#endif
