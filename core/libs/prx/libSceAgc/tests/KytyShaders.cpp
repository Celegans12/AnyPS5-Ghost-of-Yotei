#include "prx/libSceAgc/Misc/include/ShaderFusion.hpp"
#include "prx/libSceAgc/Shader/include/GsOversubscription.hpp"
#include "prx/libSceAgc/Shader/include/InterpolantMapping.hpp"
#include "prx/libSceAgc/Shader/include/PrimState.hpp"
#include "prx/libSceAgc/Shader/include/ShaderConstants.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {

using ShaderRegs::ShaderBinaryType;

constexpr int InvalidShaderHalves = static_cast<int>(0x8a6c0008u);

alignas(256) const std::array<std::uint32_t, 64> frontCode{0xbf810000u};
alignas(256) const std::array<std::uint32_t, 64> backCode{0xbf810000u};

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
bool fails(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

std::uint32_t lowAddress(const void* code) {
    return static_cast<std::uint32_t>(reinterpret_cast<std::uint64_t>(code) >> 8u);
}

std::uint32_t highByte(const void* code) {
    return static_cast<std::uint32_t>((reinterpret_cast<std::uint64_t>(code) >> 40u) & 0xffu);
}

const ShaderRegister* findRegister(const Shader& shader, std::uint32_t offset, std::uint32_t occurrence = 0) {
    for (std::uint32_t i = 0; i < shader.num_sh_registers; ++i) {
        if (shader.sh_registers[i].offset == offset && occurrence-- == 0) return &shader.sh_registers[i];
    }
    return nullptr;
}

struct Half {
    Shader shader{};
    std::array<ShaderRegister, 10> registers{};
    ShaderSpecialRegs specials{};
    ShaderUserData userData{};
};

void geometryFront(Half& half) {
    half.shader.type = static_cast<std::uint8_t>(ShaderBinaryType::GsFront);
    half.shader.code = frontCode.data();
    half.registers = {{{0x080u, 0x11111111u}, {0x080u, 0x22222222u}, {0x08Au, 0x40000005u}, {0x08Bu, 0xF807000Cu}}};
    half.shader.num_sh_registers = 4;
    half.shader.sh_registers = half.registers.data();
    half.specials.vgt_shader_stages_en = {ShaderRegs::VGT_SHADER_STAGES_EN, 0x00400000u};
    half.shader.specials = &half.specials;
    half.shader.user_data = &half.userData;
}

void geometryBack(Half& half) {
    half.shader.type = static_cast<std::uint8_t>(ShaderBinaryType::GsBack);
    half.shader.code = backCode.data();
    half.registers = {{{ShaderRegs::SPI_SHADER_PGM_LO_GS, lowAddress(backCode.data())}, {ShaderRegs::SPI_SHADER_PGM_LO_GS + 1u, 0xabcdef00u | highByte(backCode.data())},
                       {ShaderRegs::SPI_SHADER_PGM_LO_ES, 0u}, {ShaderRegs::SPI_SHADER_PGM_LO_ES + 1u, 0x12345600u},
                       {0x080u, 0xaaaau}, {0x080u, 0xbbbbu}, {0x08Au, 0x20ABC009u}, {0x08Bu, 0x200100BEu}}};
    half.shader.num_sh_registers = 8;
    half.shader.sh_registers = half.registers.data();
    half.specials.vgt_shader_stages_en = {ShaderRegs::VGT_SHADER_STAGES_EN, 0x00400000u};
    half.shader.specials = &half.specials;
    half.shader.user_data = &half.userData;
}

void checkGeometryFusion(bool reallocateShared) {
    Half front;
    Half back;
    geometryFront(front);
    geometryBack(back);
    const auto backRegisters = back.registers;
    std::array<ShaderRegister, 8> scratch{};
    Shader fused{};
    const int result = reallocateShared ? sceAgcFuseShaderHalves_0200(&fused, &front.shader, &back.shader, scratch.data())
                                        : sceAgcFuseShaderHalves_nid_postfix(&fused, &front.shader, &back.shader, scratch.data());
    check(result == 0, "geometry halves did not fuse");
    check(fused.type == static_cast<std::uint8_t>(ShaderBinaryType::Gs), "fused geometry halves are not a GS");
    check(fused.code == backCode.data() && fused.sh_registers == scratch.data() && fused.num_sh_registers == 8, "the fused GS is not the back half with its registers in the scratch memory");
    check(back.registers[2].value == backRegisters[2].value && back.registers[7].value == backRegisters[7].value, "fusion changed the back half's own registers");
    const auto* esLow = findRegister(fused, ShaderRegs::SPI_SHADER_PGM_LO_ES);
    check(esLow != nullptr && esLow[0].value == lowAddress(frontCode.data()) && esLow[1].value == (0x12345600u | highByte(frontCode.data())), "SPI_SHADER_PGM_LO_ES does not point at the front half's code in guest memory");
    const auto* gsLow = findRegister(fused, ShaderRegs::SPI_SHADER_PGM_LO_GS);
    check(gsLow != nullptr && gsLow[0].value == lowAddress(backCode.data()) && gsLow[1].value == (0xabcdef00u | highByte(backCode.data())), "SPI_SHADER_PGM_LO_GS no longer points at the back half's code");
    check(findRegister(fused, 0x080u, 0)->value == 0x11111111u && findRegister(fused, 0x080u, 1)->value == 0x22222222u, "the GS checksums are not the front half's");
    check(findRegister(fused, 0x08Au)->value == 0x40ABC009u, "SPI_SHADER_PGM_RSRC1_GS does not take the larger VGPR and wave fields");
    check(findRegister(fused, 0x08Bu)->value == (reallocateShared ? 0x1807008Cu : 0xF807008Cu), "SPI_SHADER_PGM_RSRC2_GS does not merge the shared VGPRs, LDS and user SGPR fields as Kyty does");
    check(fused.user_data == (reallocateShared ? nullptr : &front.userData), "the fused user data is not the one Kyty keeps for this export");
}

void testGeometryFusion() {
    checkGeometryFusion(true);
    checkGeometryFusion(false);

    Half front;
    Half back;
    geometryFront(front);
    geometryBack(back);
    std::array<ShaderRegister, 8> scratch{};
    Shader fused{};
    back.specials.vgt_shader_stages_en.value = 0;
    check(sceAgcFuseShaderHalves_0200(&fused, &front.shader, &back.shader, scratch.data()) == InvalidShaderHalves, "halves of different wave sizes fused");
    back.specials.vgt_shader_stages_en.value = 0x00400000u;
    front.shader.type = static_cast<std::uint8_t>(ShaderBinaryType::HsFront);
    check(sceAgcFuseShaderHalves_0200(&fused, &front.shader, &back.shader, scratch.data()) == InvalidShaderHalves, "a hull front fused with a geometry back");
    front.shader.type = static_cast<std::uint8_t>(ShaderBinaryType::GsFront);
    front.shader.num_sh_registers = 1;
    check(fails([&] { sceAgcFuseShaderHalves_0200(&fused, &front.shader, &back.shader, scratch.data()); }), "a front half without its second checksum register did not fail");

    SizeAlign size{};
    check(sceAgcGetFusedShaderSize_0080(&size, &front.shader, &back.shader) == 0 && size.m_size == 8 * sizeof(ShaderRegister) && size.m_align == 4, "the fused shader size is not the back half's register array");
}

void testHullFusion() {
    Half front;
    Half back;
    front.shader.type = static_cast<std::uint8_t>(ShaderBinaryType::HsFront);
    front.shader.code = frontCode.data();
    front.registers = {{{0x100u, 0x33u}, {0x100u, 0x44u}, {0x10Au, 0x00000003u}, {0x10Bu, 0x00000000u}}};
    front.shader.num_sh_registers = 4;
    front.shader.sh_registers = front.registers.data();
    front.shader.specials = &front.specials;
    back.shader.type = static_cast<std::uint8_t>(ShaderBinaryType::HsBack);
    back.shader.code = backCode.data();
    back.registers = {{{ShaderRegs::SPI_SHADER_PGM_LO_LS, 0u}, {ShaderRegs::SPI_SHADER_PGM_LO_LS + 1u, 0u},
                       {0x100u, 0x55u}, {0x100u, 0x66u}, {0x10Au, 0x00000001u}, {0x10Bu, 0x00000000u}}};
    back.shader.num_sh_registers = 6;
    back.shader.sh_registers = back.registers.data();
    back.shader.specials = &back.specials;
    std::array<ShaderRegister, 6> scratch{};
    Shader fused{};
    check(sceAgcFuseShaderHalves_0200(&fused, &front.shader, &back.shader, scratch.data()) == 0, "hull halves did not fuse");
    check(fused.type == static_cast<std::uint8_t>(ShaderBinaryType::Hs), "fused hull halves are not an HS");
    const auto* lsLow = findRegister(fused, ShaderRegs::SPI_SHADER_PGM_LO_LS);
    check(lsLow[0].value == lowAddress(frontCode.data()) && lsLow[1].value == highByte(frontCode.data()), "SPI_SHADER_PGM_LO_LS does not point at the front half's code");
    check(findRegister(fused, 0x100u, 0)->value == 0x33u && findRegister(fused, 0x100u, 1)->value == 0x44u, "the HS checksums are not the front half's");
    check(findRegister(fused, 0x10Au)->value == 0x00000003u, "SPI_SHADER_PGM_RSRC1_HS does not take the larger VGPR field");
}

struct Stage {
    Shader shader{};
    ShaderSpecialRegs specials{};
};

void stage(Stage& stage, ShaderBinaryType type, std::uint32_t stages, std::uint32_t outPrim, std::uint32_t userVgprs) {
    stage.shader.type = static_cast<std::uint8_t>(type);
    stage.specials.ge_cntl = {ShaderRegs::GE_CNTL, 0x1234u};
    stage.specials.vgt_shader_stages_en = {ShaderRegs::VGT_SHADER_STAGES_EN, stages};
    stage.specials.vgt_gs_out_prim_type = {ShaderRegs::VGT_GS_OUT_PRIM_TYPE, outPrim};
    stage.specials.ge_user_vgpr_en = {ShaderRegs::GE_USER_VGPR_EN, userVgprs};
    stage.shader.specials = &stage.specials;
}

void testPrimState() {
    Stage gs;
    stage(gs, ShaderBinaryType::Gs, 0u, 2u, 5u);
    std::array<ShaderRegister, 2> cx{};
    std::array<ShaderRegister, 3> uc{};
    check(sceAgcCreatePrimState(cx.data(), uc.data(), nullptr, &gs.shader, static_cast<std::uint32_t>(ShaderRegs::PrimitiveType::LineList)) == 0, "prim state failed");
    check(cx[0].offset == ShaderRegs::VGT_SHADER_STAGES_EN && cx[0].value == 0u && cx[1].offset == ShaderRegs::VGT_GS_OUT_PRIM_TYPE && cx[1].value == 1u, "a line list without a GS stage does not output lines");
    check(uc[0].offset == ShaderRegs::GE_CNTL && uc[0].value == 0x1234u && uc[1].offset == ShaderRegs::GE_USER_VGPR_EN && uc[1].value == 5u && uc[2].offset == ShaderRegs::VGT_PRIMITIVE_TYPE && uc[2].value == 2u, "the primitive registers are not GE_CNTL, GE_USER_VGPR_EN and VGT_PRIMITIVE_TYPE");

    gs.specials.vgt_shader_stages_en.value = ShaderRegs::VGT_SHADER_STAGES_GS_BIT;
    check(sceAgcCreatePrimState(cx.data(), uc.data(), nullptr, &gs.shader, 2u) == 0 && cx[1].value == 2u, "a GS stage does not keep its own output primitive type");

    Stage hs;
    stage(hs, ShaderBinaryType::Hs, ShaderRegs::VGT_SHADER_STAGES_HS_BIT, 7u, 9u);
    gs.specials.vgt_shader_stages_en.value = 0u;
    check(sceAgcCreatePrimState(cx.data(), uc.data(), &hs.shader, &gs.shader, static_cast<std::uint32_t>(ShaderRegs::PrimitiveType::Patch)) == 0, "prim state with a hull stage failed");
    check(cx[0].value == ShaderRegs::VGT_SHADER_STAGES_HS_BIT && cx[1].value == 7u && uc[1].value == 9u, "a hull stage does not give its stages, output primitive and user VGPRs");

    gs.specials.ge_cntl.offset = 0u;
    check(fails([&] { sceAgcCreatePrimState(nullptr, uc.data(), nullptr, &gs.shader, 4u); }), "a GS with a misplaced GE_CNTL did not fail");

    std::array<ShaderRegister, 2> update{{{ShaderRegs::VGT_SHADER_STAGES_EN, 0u}, {ShaderRegs::VGT_GS_OUT_PRIM_TYPE, 0x100u}}};
    std::array<ShaderRegister, 3> primitive{{{}, {}, {ShaderRegs::VGT_PRIMITIVE_TYPE, 0x80000004u}}};
    check(sceAgcUpdatePrimState(update.data(), primitive.data(), static_cast<std::uint32_t>(ShaderRegs::PrimitiveType::RectList)) == 0, "prim state update failed");
    check(update[1].value == 0x103u && primitive[2].value == 0x80000007u, "the update does not rewrite the output and primitive types");
    update[0].value = ShaderRegs::VGT_SHADER_STAGES_HS_BIT;
    check(sceAgcUpdatePrimState(update.data(), primitive.data(), 1u) == 0 && update[1].value == 0x103u, "the update rewrote the output primitive of a tessellated pipeline");
}

void testGsOversubscription() {
    std::array<ShaderRegister, 2> regs{};
    check(sceAgcGetGsOversubscription(regs.data(), nullptr, 0u, 1.0f) == 0, "a zero budget failed");
    check(regs[0].offset == 0x260u && regs[0].value == 0u && regs[1].offset == ShaderRegs::SPI_SHADER_PGM_RSRC4_GS && regs[1].value == 0u, "a zero budget does not clear the oversubscription");
    check(sceAgcGetGsOversubscription(regs.data(), nullptr, std::numeric_limits<std::uint32_t>::max(), 1.0f) == 0, "a full budget failed");
    check(regs[0].value == 0x7ffu && regs[1].value == 0x007f0000u, "a full budget does not oversubscribe fully");
    check(fails([&] { sceAgcGetGsOversubscription(regs.data(), nullptr, 64u, 1.0f); }), "a partial budget without a GS did not fail");
}

ShaderSemantic semantic(std::uint32_t id, std::uint32_t mapping, bool flat) {
    ShaderSemantic value{};
    value.semantic = id;
    value.hardware_mapping = mapping;
    value.size_in_elements = 4;
    value.is_flat_shaded = flat ? 1u : 0u;
    return value;
}

void checkInterpolants(decltype(&sceAgcCreateInterpolantMapping_0100) create, const char* name) {
    std::array<ShaderSemantic, 3> inputs{semantic(5, 0, false), semantic(6, 0, true), semantic(9, 0, false)};
    std::array<ShaderSemantic, 2> outputs{semantic(5, 3, false), semantic(6, 4, false)};
    Shader ps{};
    ps.input_semantics = inputs.data();
    ps.num_input_semantics = 3;
    Shader gs{};
    gs.output_semantics = outputs.data();
    gs.num_output_semantics = 2;
    std::array<ShaderRegister, 32> regs{};
    check(create(regs.data(), &gs, &ps) == 0, name);
    check(regs[0].offset == ShaderRegs::SPI_PS_INPUT_CNTL_0 && regs[0].value == 3u, "a matched interpolant does not read the GS output's slot");
    check(regs[1].value == 0x404u, "a flat interpolant is not flat shaded");
    check(regs[2].value == 0x20u, "an interpolant without a GS output does not take its default value");
    for (std::uint32_t i = 3; i < 32; ++i) check(regs[i].offset == ShaderRegs::SPI_PS_INPUT_CNTL_0 + i && regs[i].value == i, "the unused interpolants are not the identity mapping");
    check(create(regs.data(), &gs, nullptr) == 0 && regs[0].value == 0u && regs[31].value == 31u, "a missing pixel shader does not give the identity mapping");
}

void testInterpolantMapping() {
    checkInterpolants(sceAgcCreateInterpolantMapping_0100, "sceAgcCreateInterpolantMapping_0100 failed");
    checkInterpolants(sceAgcCreateInterpolantMapping_10000, "sceAgcCreateInterpolantMapping_10000 failed");
}

}

int main() {
    try {
        testGeometryFusion();
        testHullFusion();
        testPrimState();
        testGsOversubscription();
        testInterpolantMapping();
        std::puts("AGC Kyty shader register tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
