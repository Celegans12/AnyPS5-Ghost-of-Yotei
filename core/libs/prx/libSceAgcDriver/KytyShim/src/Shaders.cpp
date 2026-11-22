#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include "SceShaders.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/EntryApi.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/Replay.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#if !defined(ANYPS5_KYTY_DLL)
#include "prx/libSceAgcDriver/Execution/include/FrameRecorder.hpp"
#endif
#include "prx/libSceAgcDriver/Execution/include/ShaderPreparationScope.hpp"

#include "graphics/shader/shader.h"

namespace {

namespace Kyty = Libs::Graphics;

static_assert(sizeof(Shader) == sizeof(Kyty::Shader));
static_assert(offsetof(Shader, user_data) == offsetof(Kyty::Shader, user_data));
static_assert(offsetof(Shader, code) == offsetof(Kyty::Shader, code));
static_assert(offsetof(Shader, input_semantics) == offsetof(Kyty::Shader, input_semantics));
static_assert(offsetof(Shader, shader_size) == offsetof(Kyty::Shader, shader_size));
static_assert(offsetof(Shader, num_input_semantics) == offsetof(Kyty::Shader, num_input_semantics));
static_assert(offsetof(Shader, scratch_size_dw_per_thread) == offsetof(Kyty::Shader, scratch_size_dw_per_thread));
static_assert(offsetof(Shader, type) == offsetof(Kyty::Shader, type));
static_assert(sizeof(ShaderUserData) == sizeof(Kyty::ShaderUserData));
static_assert(offsetof(ShaderUserData, direct_resource_offset) == offsetof(Kyty::ShaderUserData, direct_resource_offset));
static_assert(offsetof(ShaderUserData, sharp_resource_offset) == offsetof(Kyty::ShaderUserData, sharp_resource_offset));
static_assert(offsetof(ShaderUserData, eud_size_dw) == offsetof(Kyty::ShaderUserData, eud_size_dw));
static_assert(offsetof(ShaderUserData, srt_size_dw) == offsetof(Kyty::ShaderUserData, srt_size_dw));
static_assert(offsetof(ShaderUserData, direct_resource_count) == offsetof(Kyty::ShaderUserData, direct_resource_count));
static_assert(offsetof(ShaderUserData, sharp_resource_count) == offsetof(Kyty::ShaderUserData, sharp_resource_count));
static_assert(sizeof(ShaderSemantic) == sizeof(Kyty::ShaderSemantic));
static_assert(sizeof(ShaderSharp) == sizeof(Kyty::ShaderSharp));

struct ShaderMap {
    ShaderMap() { Kyty::ShaderInit(); }
};

const ShaderMap shaderMap;

}

extern "C" void AgcDriverRegisterShader_nid_postfix(const Shader* shader) {
    const bool replay = KytyShim::Replay::Tracking();
    if (replay) KytyShim::Entry::CheckGuestRange(shader, sizeof(Shader), alignof(Shader));
    if (shader->code == nullptr || shader->shader_size == 0 || shader->shader_size % sizeof(std::uint32_t) != 0) {
        char text[160];
        std::snprintf(text, sizeof(text), "kyty GPU backend: shader header 0x%llx has no code (code 0x%llx, %u bytes)", static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(shader)),
                      static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(shader->code)), static_cast<unsigned>(shader->shader_size));
        throw std::runtime_error(text);
    }
    if (replay) KytyShim::Entry::CheckGuestRange(const_cast<const void*>(shader->code), shader->shader_size, sizeof(std::uint32_t));
    Kyty::ShaderMappedData map;
    map.type = static_cast<Kyty::Prospero::ShaderBinaryType>(shader->type);
    map.user_data = reinterpret_cast<Kyty::ShaderUserData*>(shader->user_data);
    map.input_semantics = reinterpret_cast<Kyty::ShaderSemantic*>(shader->input_semantics);
    map.num_input_semantics = shader->num_input_semantics;
    map.code_size_bytes = shader->shader_size;
    map.scratch_size_dwords = shader->scratch_size_dw_per_thread;
    const auto accept = [&] { Kyty::ShaderMapUserData(reinterpret_cast<std::uint64_t>(shader->code), map); };
#if !defined(ANYPS5_KYTY_DLL)
    if (auto* recorder = AgcDriver::FrameRecorder::Active()) {
        recorder->RegisterShader(shader, accept);
        return;
    }
#endif
    accept();
}

#if defined(ANYPS5_KYTY_DLL)
extern "C" void AgcDriverResolveShaderAbi_nid_postfix(const Shader*, std::span<const ShaderRegister>, std::span<const ShaderRegister>) {}

extern "C" void AgcDriverResolveGraphicsStagesAbi_nid_postfix(std::span<const Shader* const>, std::span<const ShaderRegister>, std::span<const ShaderRegister>) {}

extern "C" void AgcDriverResolveGraphicsAbi_nid_postfix(const Shader*, const Shader*, std::uint32_t) {}
#else
extern "C" void AgcDriverResolveShaderAbi_nid_postfix(const Shader* shader, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    if (auto* recorder = AgcDriver::FrameRecorder::Active()) recorder->ResolveShaderAbi(shader, context, primitive);
}

extern "C" void AgcDriverResolveGraphicsStagesAbi_nid_postfix(std::span<const Shader* const> stages, std::span<const ShaderRegister> context, std::span<const ShaderRegister> primitive) {
    if (auto* recorder = AgcDriver::FrameRecorder::Active()) recorder->ResolveGraphicsStagesAbi(stages, context, primitive);
}

extern "C" void AgcDriverResolveGraphicsAbi_nid_postfix(const Shader* vertex, const Shader* pixel, std::uint32_t primitiveType) {
    if (auto* recorder = AgcDriver::FrameRecorder::Active()) recorder->ResolveGraphicsAbi(vertex, pixel, primitiveType);
}
#endif

extern "C" AgcDriver::DriverDetail::ShaderPreparationTransaction* AgcDriverBeginShaderPreparation_nid_postfix(const Shader* const*, std::size_t) {
    return nullptr;
}

extern "C" void AgcDriverCommitShaderPreparation_nid_postfix(AgcDriver::DriverDetail::ShaderPreparationTransaction*) {}

extern "C" void AgcDriverEndShaderPreparation_nid_postfix(AgcDriver::DriverDetail::ShaderPreparationTransaction*) noexcept {}
