#include <cstdint>
#include "SceTypes.hpp"
#include "SceShaders.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgc/Kyty/include/ShaderRegisters.hpp"
#include "prx/libSceAgc/Shader/include/InterpolantMapping.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"

namespace {

using namespace ShaderRegs;
using namespace Agc::Kyty;

std::uint32_t shader_semantic_word(const ShaderSemantic& semantic) {
    return ((semantic.semantic & 0xffu) << 0u) | ((semantic.hardware_mapping & 0xffu) << 8u) |
           ((semantic.size_in_elements & 0xfu) << 16u) | ((semantic.is_f16 & 0x3u) << 20u) |
           ((semantic.is_flat_shaded & 0x1u) << 22u) | ((semantic.is_linear & 0x1u) << 23u) |
           ((semantic.is_custom & 0x1u) << 24u) | ((semantic.static_vb_index & 0x1u) << 25u) |
           ((semantic.static_attribute & 0x1u) << 26u) | ((semantic.reserved & 0x1u) << 27u) |
           ((semantic.default_value & 0x3u) << 28u) | ((semantic.default_value_hi & 0x3u) << 30u);
}

std::uint32_t apply_interpolant_default_value(std::uint32_t value, std::uint32_t ps_word) {
    value &= ~0x00000300u;
    value |= ((ps_word >> 28u) & 0x3u) << 8u;
    return value;
}

std::uint32_t apply_interpolant_default_value_hi(std::uint32_t value, std::uint32_t ps_word) {
    value &= ~0x00600000u;
    value |= ((ps_word >> 30u) & 0x3u) << 21u;
    return value;
}

std::uint32_t create_interpolant_mapping_value(std::uint32_t value, std::uint32_t ps_word, std::uint32_t gs_word) {
    const std::uint32_t flat_shade = ((ps_word & 0x00400000u) != 0 || (ps_word & 0x01000000u) != 0 ? 0x00000400u : 0u);
    value &= ~0x0000001fu;
    value |= (gs_word >> 8u) & 0x1fu;
    value &= ~0x00000400u;
    value |= flat_shade;
    return apply_interpolant_default_value(value, ps_word);
}

std::uint32_t create_interpolant_default_value(std::uint32_t value, std::uint32_t ps_word) {
    value &= ~0x0000001fu;
    value &= ~0x00000400u;
    return apply_interpolant_default_value(value, ps_word);
}

std::uint32_t create_interpolant_f16_value(std::uint32_t ps_word, const ShaderSemantic* gs_semantic) {
    std::uint32_t value = (ps_word << 4u) & 0x03000000u;
    if (gs_semantic == nullptr) {
        value |= 0x00180020u;
    } else {
        const std::uint32_t common_word = ps_word & shader_semantic_word(*gs_semantic);
        value &= 0xfff7ffdfu;
        value |= (common_word >> 15u) & 0x20u;
        value ^= 0x00080020u;
        value &= ~0x00100000u;
        value |= (~common_word >> 1u) & 0x00100000u;
    }
    return apply_interpolant_default_value_hi(value, ps_word);
}

std::uint32_t create_interpolant_non_f16_value(std::uint32_t ps_word, const ShaderSemantic* gs_semantic) {
    std::uint32_t value = 0;
    if ((ps_word & 0x01000000u) != 0 || gs_semantic == nullptr) {
        value |= 0x20u;
    }
    return value;
}

const ShaderSemantic* find_interpolant_output_semantic(const Shader* gs, std::uint32_t semantic) {
    if (gs == nullptr || gs->output_semantics == nullptr) {
        return nullptr;
    }
    for (std::uint16_t i = 0; i < gs->num_output_semantics; i++) {
        if (gs->output_semantics[i].semantic == semantic) {
            return &gs->output_semantics[i];
        }
    }
    return nullptr;
}

void set_interpolant_register(ShaderRegister* regs, std::uint32_t index, std::uint32_t value) {
    regs[index].offset = SPI_PS_INPUT_CNTL_0 + index;
    regs[index].value = value;
}

void set_identity_interpolant_registers(ShaderRegister* regs, std::uint32_t first_index) {
    for (std::uint32_t i = first_index; i < 32u; i++) {
        set_interpolant_register(regs, i, i);
    }
}

std::uint32_t apply_interpolant_two_bit_field(std::uint32_t value, std::uint32_t field, std::uint32_t shift) {
    const auto mask = 0x3u << shift;
    return (value & ~mask) | ((field & 0x3u) << shift);
}

std::uint32_t apply_interpolant_final_mask(std::uint32_t flags, std::uint32_t source, std::uint32_t mask) {
    flags = (flags & 0xffffffe0u) | ((mask >> 8u) & 0x1fu);
    flags = (flags & 0xfffffbffu) | ((source & 0x400000u) != 0 ? 0x400u : ((source >> 14u) & 0x400u));
    return flags;
}

}

extern "C" {

int APS5_VABI sceAgcCreateInterpolantMapping_0100(ShaderRegister* regs, const Shader* gs, const Shader* ps) {
    constexpr auto fn = "sceAgcCreateInterpolantMapping_0100";
    NotImplemented(regs == nullptr, fn, "regs == nullptr");
    NotImplemented(ps != nullptr && ps->num_input_semantics != 0 && ps->input_semantics == nullptr, fn, "ps->input_semantics == nullptr");

    if (ps == nullptr || ps->num_input_semantics == 0) {
        set_identity_interpolant_registers(regs, 0);
        return OK;
    }

    NotImplemented(gs == nullptr, fn, "gs == nullptr");
    NotImplemented(gs->num_output_semantics != 0 && gs->output_semantics == nullptr, fn, "gs->output_semantics == nullptr");

    for (std::uint32_t ps_index = 0; ps_index < ps->num_input_semantics; ps_index++) {
        const auto& ps_semantic = ps->input_semantics[ps_index];
        const auto* gs_semantic = find_interpolant_output_semantic(gs, ps_semantic.semantic);
        const auto ps_word = shader_semantic_word(ps_semantic);

        auto value = ((ps_word & 0x00300000u) != 0 ? create_interpolant_f16_value(ps_word, gs_semantic)
                                                   : create_interpolant_non_f16_value(ps_word, gs_semantic));
        value = (gs_semantic == nullptr ? create_interpolant_default_value(value, ps_word)
                                        : create_interpolant_mapping_value(value, ps_word, shader_semantic_word(*gs_semantic)));

        set_interpolant_register(regs, ps_index, value);
    }

    set_identity_interpolant_registers(regs, ps->num_input_semantics);
    return OK;
}

int APS5_VABI sceAgcCreateInterpolantMapping_10000(ShaderRegister* regs, const Shader* gs, const Shader* ps) {
    constexpr auto fn = "sceAgcCreateInterpolantMapping_10000";
    NotImplemented(regs == nullptr, fn, "regs == nullptr");
    NotImplemented(ps != nullptr && ps->num_input_semantics != 0 && ps->input_semantics == nullptr, fn, "ps->input_semantics == nullptr");

    if (ps == nullptr || ps->num_input_semantics == 0) {
        set_identity_interpolant_registers(regs, 0);
        return OK;
    }

    NotImplemented(gs == nullptr, fn, "gs == nullptr");
    NotImplemented(gs->num_output_semantics != 0 && gs->output_semantics == nullptr, fn, "gs->output_semantics == nullptr");

    for (std::uint32_t i = 0; i < ps->num_input_semantics; i++) {
        const auto source = shader_semantic_word(ps->input_semantics[i]);
        auto mask_index = static_cast<std::uint32_t>(gs->num_output_semantics);

        for (std::uint32_t j = 0; j < gs->num_output_semantics; j++) {
            if (gs->output_semantics[j].semantic == static_cast<std::uint8_t>(source)) {
                mask_index = j;
                break;
            }
        }

        const bool has_mask = mask_index < gs->num_output_semantics;
        const auto mask = has_mask ? shader_semantic_word(gs->output_semantics[mask_index]) : 0u;
        const auto mode = (source >> 20u) & 0x3u;
        auto flags = 0u;

        if (mode == 0) {
            flags = (((source >> 24u) & 0x1u) | (has_mask ? 0u : 1u)) << 5u;
            flags = apply_interpolant_two_bit_field(flags, source >> 28u, 8u);
        } else {
            flags = ((source << 4u) & 0x03000000u) + 0x80000u;

            if (mode == 2) {
                flags &= 0xffefffdfu;
                flags |= has_mask ? ((~(mask & source) >> 16u) & 0x20u) : 0x20u;
                flags = apply_interpolant_two_bit_field(flags, source >> 30u, 8u);
                flags = apply_interpolant_two_bit_field(flags, source >> 30u, 21u);
            } else {
                if (has_mask) {
                    const auto masked = mask & source;
                    flags = (flags & 0xffffffdfu) | ((masked >> 15u) & 0x20u);
                    flags ^= 0x20u;
                    flags = (flags & 0xffefffffu) | ((~masked >> 1u) & 0x100000u);
                    flags = apply_interpolant_two_bit_field(flags, source >> 28u, 8u);
                } else {
                    flags |= 0x100020u;
                    flags = apply_interpolant_two_bit_field(flags, source >> 28u, 8u);
                }
                flags = apply_interpolant_two_bit_field(flags, source >> 30u, 21u);
            }
        }

        flags = has_mask ? apply_interpolant_final_mask(flags, source, mask) : (flags & 0xfffffbe0u);
        set_interpolant_register(regs, i, flags);
    }

    if (ps->num_input_semantics < 32u) {
        set_identity_interpolant_registers(regs, ps->num_input_semantics);
    }

    return OK;
}

}
