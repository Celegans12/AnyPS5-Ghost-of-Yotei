#include <cstdint>
#include "SceTypes.hpp"
#include "SceShaders.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libSceAgc/Kyty/include/ShaderRegisters.hpp"
#include "prx/libSceAgc/Shader/include/PrimState.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"

namespace {

using namespace ShaderRegs;
using namespace Agc::Kyty;

std::uint32_t GraphicsPrimitiveTypeToGsOut(std::uint32_t prim_type) {
    switch (static_cast<PrimitiveType>(prim_type)) {
        case PrimitiveType::PointList:
            return static_cast<std::uint32_t>(GsOutputPrimitiveType::Points);
        case PrimitiveType::LineList:
        case PrimitiveType::LineStrip:
        case PrimitiveType::LineListAdjacency:
        case PrimitiveType::LineStripAdjacency:
        case PrimitiveType::LineLoop:
            return static_cast<std::uint32_t>(GsOutputPrimitiveType::Lines);
        case PrimitiveType::RectList:
            return static_cast<std::uint32_t>(GsOutputPrimitiveType::Rectangle2D);
        case PrimitiveType::RectListLegacy:
            return static_cast<std::uint32_t>(GsOutputPrimitiveType::RectList);
        default: return static_cast<std::uint32_t>(GsOutputPrimitiveType::Triangles);
    }
}

}

extern "C" {

int APS5_VABI sceAgcCreatePrimState(ShaderRegister* cx_regs, ShaderRegister* uc_regs, const Shader* hs, const Shader* gs, std::uint32_t prim_type) {
    constexpr auto fn = "sceAgcCreatePrimState";
    if (cx_regs == nullptr && uc_regs == nullptr) {
        return OK;
    }

    NotImplemented(gs == nullptr, fn, "gs == nullptr");
    NotImplemented(hs != nullptr && static_cast<ShaderBinaryType>(hs->type) != ShaderBinaryType::Hs, fn, "hs->type != kHs");
    NotImplemented(static_cast<ShaderBinaryType>(gs->type) != ShaderBinaryType::Gs, fn, "gs->type != kGs");
    NotImplemented(gs->specials == nullptr || (hs != nullptr && hs->specials == nullptr), fn, "shader has no special registers");

    if (cx_regs != nullptr) {
        NotImplemented(hs != nullptr && hs->specials->vgt_shader_stages_en.offset != VGT_SHADER_STAGES_EN, fn, "hs VGT_SHADER_STAGES_EN offset");
        NotImplemented(hs != nullptr && hs->specials->vgt_gs_out_prim_type.offset != VGT_GS_OUT_PRIM_TYPE, fn, "hs VGT_GS_OUT_PRIM_TYPE offset");
        NotImplemented(gs->specials->vgt_shader_stages_en.offset != VGT_SHADER_STAGES_EN, fn, "gs VGT_SHADER_STAGES_EN offset");
        NotImplemented(gs->specials->vgt_gs_out_prim_type.offset != VGT_GS_OUT_PRIM_TYPE, fn, "gs VGT_GS_OUT_PRIM_TYPE offset");

        cx_regs[0] = gs->specials->vgt_shader_stages_en;
        if ((cx_regs[0].value & 0x20u) != 0) {
            cx_regs[1] = gs->specials->vgt_gs_out_prim_type;
        } else {
            cx_regs[1].offset = VGT_GS_OUT_PRIM_TYPE;
            cx_regs[1].value = GraphicsPrimitiveTypeToGsOut(prim_type);
        }

        if (hs != nullptr) {
            cx_regs[0].value |= hs->specials->vgt_shader_stages_en.value;
            if ((cx_regs[0].value & 0x20u) == 0) {
                cx_regs[1] = hs->specials->vgt_gs_out_prim_type;
            }
        }
    }

    if (uc_regs != nullptr) {
        NotImplemented(gs->specials->ge_cntl.offset != GE_CNTL, fn, "gs GE_CNTL offset");
        NotImplemented(gs->specials->ge_user_vgpr_en.offset != GE_USER_VGPR_EN, fn, "gs GE_USER_VGPR_EN offset");
        NotImplemented(hs != nullptr && hs->specials->ge_user_vgpr_en.offset != GE_USER_VGPR_EN, fn, "hs GE_USER_VGPR_EN offset");

        uc_regs[0] = gs->specials->ge_cntl;
        uc_regs[1] = gs->specials->ge_user_vgpr_en;
        uc_regs[2].offset = VGT_PRIMITIVE_TYPE;
        uc_regs[2].value = prim_type;

        if (hs != nullptr) {
            uc_regs[1] = hs->specials->ge_user_vgpr_en;
        }
    }

    return OK;
}

int APS5_VABI sceAgcUpdatePrimState(ShaderRegister* cx_regs, ShaderRegister* uc_regs, std::uint32_t prim_type) {
    if (cx_regs != nullptr && (cx_regs[0].value & 0x24u) == 0) {
        cx_regs[1].value &= ~0x7u;
        cx_regs[1].value |= GraphicsPrimitiveTypeToGsOut(prim_type);
    }
    if (uc_regs != nullptr) {
        uc_regs[2].value &= ~0x1fu;
        uc_regs[2].value |= prim_type;
    }
    return OK;
}

}
