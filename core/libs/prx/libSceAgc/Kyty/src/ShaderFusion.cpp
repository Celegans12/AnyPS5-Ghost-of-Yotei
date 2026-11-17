#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cstdint>
#include "SceTypes.hpp"
#include "SceShaders.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgc/Kyty/include/ShaderRegisters.hpp"
#include "prx/libSceAgc/Misc/include/ShaderFusion.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"

namespace {

using namespace ShaderRegs;
using namespace Agc::Kyty;

constexpr int GRAPHICS5_ERROR_INVALID_SHADER_HALVES = static_cast<int>(0x8a6c0008u);
constexpr std::uint32_t SPI_SHADER_PGM_CHKSUM_GS = 0x080u;
constexpr std::uint32_t SPI_SHADER_PGM_RSRC1_GS = 0x08Au;
constexpr std::uint32_t SPI_SHADER_PGM_RSRC2_GS = 0x08Bu;
constexpr std::uint32_t SPI_SHADER_PGM_CHKSUM_HS = 0x100u;
constexpr std::uint32_t SPI_SHADER_PGM_RSRC1_HS = 0x10Au;
constexpr std::uint32_t SPI_SHADER_PGM_RSRC2_HS = 0x10Bu;

void patch_shader_register_address(ShaderRegister* regs, std::uint32_t num_regs, std::uint32_t lo_offset, std::uint64_t address, const char* fn) {
    auto* lo = require_shader_register(regs, num_regs, lo_offset, fn, "fused half has no program address register");
    auto* hi = (lo + 1 < regs + num_regs && (lo + 1)->offset == lo_offset + 1u ? lo + 1 : nullptr);
    NotImplemented(hi == nullptr, fn, "fused half has no program address high register after the low one");
    lo->value = static_cast<std::uint32_t>((address >> 8u) & 0xffffffffu);
    hi->value &= 0xffffff00u;
    hi->value |= static_cast<std::uint32_t>((address >> 40u) & 0xffu);
}

bool valid_halves(const Shader* front, const Shader* back) {
    const auto front_type = static_cast<ShaderBinaryType>(front->type);
    const auto back_type = static_cast<ShaderBinaryType>(back->type);
    return (front_type == ShaderBinaryType::GsFront && back_type == ShaderBinaryType::GsBack) ||
           (front_type == ShaderBinaryType::HsFront && back_type == ShaderBinaryType::HsBack);
}

int get_fused_shader_size(SizeAlign* dst, const Shader* front, const Shader* back, const char* fn) {
    NotImplemented(dst == nullptr, fn, "dst == nullptr");
    NotImplemented(front == nullptr, fn, "front == nullptr");
    NotImplemented(back == nullptr, fn, "back == nullptr");
    if (!valid_halves(front, back)) {
        return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    }
    dst->m_size = static_cast<std::uint64_t>(back->num_sh_registers) * sizeof(ShaderRegister);
    dst->m_align = 4;
    return OK;
}

void merge_shader_register_max_field(ShaderRegister* dst, const ShaderRegister* src, std::uint32_t shift, std::uint32_t mask) {
    const auto dst_field = (dst->value >> shift) & mask;
    const auto src_field = (src->value >> shift) & mask;
    const auto field = std::max(dst_field, src_field);
    dst->value &= ~(mask << shift);
    dst->value |= field << shift;
}

int fuse_shader_halves(Shader* fused_result, const Shader* front, const Shader* back, void* scratch_mem, bool recompute_shared_vgprs, const char* fn) {
    NotImplemented(fused_result == nullptr, fn, "fused_result == nullptr");
    NotImplemented(front == nullptr, fn, "front == nullptr");
    NotImplemented(back == nullptr, fn, "back == nullptr");

    const auto front_type = static_cast<ShaderBinaryType>(front->type);
    const auto is_gs = front_type == ShaderBinaryType::GsFront;
    const auto is_hs = front_type == ShaderBinaryType::HsFront;
    if ((!is_gs && !is_hs) ||
        (is_gs && back->type != static_cast<std::uint8_t>(ShaderBinaryType::GsBack)) ||
        (is_hs && back->type != static_cast<std::uint8_t>(ShaderBinaryType::HsBack))) {
        return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    }

    NotImplemented(front->specials == nullptr || back->specials == nullptr, fn, "shader half has no special registers");

    *fused_result = *back;
    fused_result->type = static_cast<std::uint8_t>(is_gs ? ShaderBinaryType::Gs : ShaderBinaryType::Hs);

    const auto back_stages = back->specials->vgt_shader_stages_en.value;
    const auto front_stages = front->specials->vgt_shader_stages_en.value;
    const auto mismatch_bit = is_gs ? (1u << 22u) : (1u << 21u);
    if (((front_stages ^ back_stages) & mismatch_bit) != 0) {
        return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    }

    if (scratch_mem != nullptr) {
        auto* sh_registers = static_cast<ShaderRegister*>(scratch_mem);
        std::memcpy(sh_registers, back->sh_registers, static_cast<std::size_t>(back->num_sh_registers) * sizeof(ShaderRegister));
        fused_result->sh_registers = sh_registers;
    }

    auto* fused_regs = fused_result->sh_registers;
    const auto fused_reg_count = static_cast<std::uint32_t>(fused_result->num_sh_registers);
    const auto front_reg_count = static_cast<std::uint32_t>(front->num_sh_registers);
    const auto checksum_offset = is_gs ? SPI_SHADER_PGM_CHKSUM_GS : SPI_SHADER_PGM_CHKSUM_HS;
    for (std::uint32_t occurrence = 0; occurrence < 2; occurrence++) {
        const auto* src = require_shader_register(front->sh_registers, front_reg_count, checksum_offset, fn, "front half lacks a checksum register", occurrence);
        auto* dst = require_shader_register(fused_regs, fused_reg_count, checksum_offset, fn, "back half lacks a checksum register", occurrence);
        dst->value = src->value;
    }

    const auto rsrc1_offset = is_gs ? SPI_SHADER_PGM_RSRC1_GS : SPI_SHADER_PGM_RSRC1_HS;
    const auto rsrc2_offset = is_gs ? SPI_SHADER_PGM_RSRC2_GS : SPI_SHADER_PGM_RSRC2_HS;
    const auto* front_rsrc1 = require_shader_register(front->sh_registers, front_reg_count, rsrc1_offset, fn, "front half lacks SPI_SHADER_PGM_RSRC1");
    const auto* front_rsrc2 = require_shader_register(front->sh_registers, front_reg_count, rsrc2_offset, fn, "front half lacks SPI_SHADER_PGM_RSRC2");
    auto* fused_rsrc1 = require_shader_register(fused_regs, fused_reg_count, rsrc1_offset, fn, "back half lacks SPI_SHADER_PGM_RSRC1");
    auto* fused_rsrc2 = require_shader_register(fused_regs, fused_reg_count, rsrc2_offset, fn, "back half lacks SPI_SHADER_PGM_RSRC2");

    if (recompute_shared_vgprs) {
        const auto front_vgprs = ((front_rsrc1->value & 0x3fu) + 1u) * 4u;
        const auto back_vgprs = ((fused_rsrc1->value & 0x3fu) + 1u) * 4u;
        const auto front_total = front_vgprs + (front_rsrc2->value >> 28u) * 8u;
        const auto back_total = back_vgprs + (fused_rsrc2->value >> 28u) * 8u;
        const auto max_total = std::max(front_total, back_total);
        const auto shared = std::max(front_vgprs, back_vgprs) >= max_total ? 0u : (max_total - std::min(front_total, back_total) + 7u) / 64u;
        fused_rsrc2->value = (fused_rsrc2->value & 0x0fffffffu) | ((shared & 0xfu) << 28u);
    } else {
        merge_shader_register_max_field(fused_rsrc2, front_rsrc2, 28, 0x0fu);
    }
    merge_shader_register_max_field(fused_rsrc1, front_rsrc1, 0, 0x3fu);
    if (is_gs) {
        merge_shader_register_max_field(fused_rsrc1, front_rsrc1, 29, 0x03u);
        merge_shader_register_max_field(fused_rsrc2, front_rsrc2, 16, 0x03u);
        fused_rsrc2->value = (fused_rsrc2->value & 0xfffbffffu) | (front_rsrc2->value & 0x00040000u);
    } else {
        merge_shader_register_max_field(fused_rsrc1, front_rsrc1, 28, 0x03u);
    }
    fused_rsrc2->value = (fused_rsrc2->value & 0xf7ffffc1u) | (front_rsrc2->value & 0x0800003eu);

    patch_shader_register_address(fused_regs, fused_reg_count, is_gs ? SPI_SHADER_PGM_LO_ES : SPI_SHADER_PGM_LO_LS, reinterpret_cast<std::uint64_t>(front->code), fn);
    fused_result->user_data = recompute_shared_vgprs ? nullptr : front->user_data;
    return OK;
}

}

extern "C" {

int APS5_VABI sceAgcGetFusedShaderSize_0080(SizeAlign* dst, const Shader* front, const Shader* back) {
    return get_fused_shader_size(dst, front, back, "sceAgcGetFusedShaderSize_0080");
}

int APS5_VABI sceAgcGetFusedShaderSize(SizeAlign* dst, const Shader* front, const Shader* back) {
    if (front == nullptr || back == nullptr) APS5_INVALID_ARG_EX;
    if (!valid_halves(front, back)) return GRAPHICS5_ERROR_INVALID_SHADER_HALVES;
    if (dst == nullptr) APS5_INVALID_ARG_EX;
    dst->m_size = std::uint64_t{back->num_sh_registers} * sizeof(ShaderRegister);
    const std::uint32_t align = 4;
    std::memcpy(&dst->m_align, &align, sizeof(align));
    return OK;
}

int APS5_VABI sceAgcFuseShaderHalves_0200(Shader* fused_result, const Shader* front, const Shader* back, void* scratch_mem) {
    return fuse_shader_halves(fused_result, front, back, scratch_mem, true, "sceAgcFuseShaderHalves_0200");
}

int APS5_VABI sceAgcFuseShaderHalves_nid_postfix(Shader* fused_result, const Shader* front, const Shader* back, void* scratch_mem) {
    return fuse_shader_halves(fused_result, front, back, scratch_mem, false, "sceAgcFuseShaderHalves");
}

}
