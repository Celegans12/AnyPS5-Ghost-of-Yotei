#include <algorithm>
#include <cmath>
#include <cstdint>
#include "SceTypes.hpp"
#include "SceShaders.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgc/Kyty/include/ShaderRegisters.hpp"
#include "prx/libSceAgc/Shader/include/GsOversubscription.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"

namespace {

using namespace ShaderRegs;
using namespace Agc::Kyty;

constexpr std::uint32_t UC_PARAMETER_OVERSUBSCRIPTION = 0x260u;

struct GsOccupancyLimits {
    std::uint32_t vertex;
    std::uint32_t export_count;
};

GsOccupancyLimits get_gs_occupancy_limits(const Shader* gs, std::uint32_t vertex_capacity, std::uint32_t export_capacity) {
    constexpr auto fn = "sceAgcGetGsOversubscription";
    const auto* onchip = require_shader_register(gs->cx_registers, gs->num_cx_registers, VGT_GS_ONCHIP_CNTL, fn, "gs lacks VGT_GS_ONCHIP_CNTL");
    const auto* subgroup = require_shader_register(gs->cx_registers, gs->num_cx_registers, GE_NGG_SUBGRP_CNTL, fn, "gs lacks GE_NGG_SUBGRP_CNTL");
    const auto* vs_out = require_shader_register(gs->cx_registers, gs->num_cx_registers, SPI_VS_OUT_CONFIG, fn, "gs lacks SPI_VS_OUT_CONFIG");
    const auto* cl_out = require_shader_register(gs->cx_registers, gs->num_cx_registers, PA_CL_VS_OUT_CNTL, fn, "gs lacks PA_CL_VS_OUT_CNTL");
    const auto* max_output = require_shader_register(gs->cx_registers, gs->num_cx_registers, GE_MAX_OUTPUT_PER_SUBGROUP, fn, "gs lacks GE_MAX_OUTPUT_PER_SUBGROUP");

    const std::uint32_t output_words = ((max_output->value & 0x3ffu) + 31u) >> 5u;
    NotImplemented(output_words == 0, fn, "GE_MAX_OUTPUT_PER_SUBGROUP is zero");
    const std::uint32_t subgroup_work = ((((onchip->value >> 11u) & 0x7ffu) * (subgroup->value & 0x1ffu)) + 31u) >> 5u;
    std::uint32_t waves = std::max(subgroup_work, output_words);
    if ((gs->specials->vgt_shader_stages_en.value & 0x00400000u) == 0) {
        waves >>= 1u;
    }
    waves = std::max(waves, 1u);

    const std::uint32_t exports = 1u + ((cl_out->value >> 21u) & 1u) + ((cl_out->value >> 22u) & 1u) + ((cl_out->value >> 23u) & 1u);
    const std::uint32_t export_limit = (export_capacity / exports) * 4u;
    const std::uint32_t vertex_limit = (vs_out->value & 0x80u) != 0 ? 2048u : vertex_capacity / (((vs_out->value >> 2u) & 0xfu) + 1u);

    return {waves * (vertex_limit / output_words), waves * (export_limit / output_words)};
}

}

extern "C" {

int APS5_VABI sceAgcGetGsOversubscription(ShaderRegister* regs, const Shader* gs, std::uint32_t budget, float factor) {
    constexpr std::uint32_t FULL_PC_OVERSUBSCRIPTION = 0x7ffu;
    constexpr std::uint32_t FULL_SH_OVERSUBSCRIPTION = 0x007f0000u;
    constexpr auto fn = "sceAgcGetGsOversubscription";
    NotImplemented(regs == nullptr, fn, "regs == nullptr");

    regs[0] = {UC_PARAMETER_OVERSUBSCRIPTION, 0u};
    regs[1] = {SPI_SHADER_PGM_RSRC4_GS, 0u};

    if (budget == 0u) {
        return OK;
    }
    if (budget == UINT32_MAX) {
        regs[0].value |= FULL_PC_OVERSUBSCRIPTION;
        regs[1].value |= FULL_SH_OVERSUBSCRIPTION;
        return OK;
    }

    NotImplemented(gs == nullptr || gs->specials == nullptr, fn, "gs or its special registers are null");
    const auto base = get_gs_occupancy_limits(gs, 1024u, 128u);
    const auto expanded = get_gs_occupancy_limits(gs, 2048u, 382u);
    const auto base_min = std::min(base.vertex, base.export_count);
    const auto limit_shift = (gs->specials->vgt_shader_stages_en.value & 0x00400000u) != 0 ? 5u : 6u;
    const auto expanded_limit = std::min({expanded.vertex, expanded.export_count, budget >> limit_shift, 1024u});
    const auto headroom = expanded_limit > base_min ? expanded_limit - base_min : 0u;
    const auto target = static_cast<std::uint32_t>(std::fma(factor, static_cast<float>(headroom), static_cast<float>(base_min)));

    if (target <= base_min) {
        return OK;
    }
    if (target < base.export_count) {
        const auto range = std::max(expanded.vertex - base.vertex, 1u);
        auto value = static_cast<std::uint32_t>(std::min<std::uint64_t>((static_cast<std::uint64_t>(target - base.vertex) << 10u) / range, 1024));
        value = std::max(value, 1u);
        regs[0].value = (regs[0].value & ~FULL_PC_OVERSUBSCRIPTION) | (((value << 1u) - 1u) & FULL_PC_OVERSUBSCRIPTION);
        regs[1].value |= FULL_SH_OVERSUBSCRIPTION;
    } else {
        const auto range = std::max(expanded.export_count - base.export_count, 1u);
        const auto value = static_cast<std::uint32_t>(std::min<std::uint64_t>((static_cast<std::uint64_t>(target - base.export_count) * 127u) / range, 127));
        regs[0].value |= FULL_PC_OVERSUBSCRIPTION;
        regs[1].value = (regs[1].value & ~FULL_SH_OVERSUBSCRIPTION) | (value << 16u);
    }

    return OK;
}

}
