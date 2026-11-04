#ifndef CORE_SHADER_RECOMPILER_P0SCEILING_HPP
#define CORE_SHADER_RECOMPILER_P0SCEILING_HPP

// Lane P0-shader (PLAN v3 step 0): the "Kyty-equivalent ceiling" switches. Diagnostic only: every switch is off
// unless its variable is set, and an unset switch takes none of the new branches (byte-identical SPIR-V). Several are
// inexact by design (native float, unguarded bound buffers); they measure an upper bound on the request bench and in
// timed replays, never in the game. Each name is an APS5_* recompiler variable, so its value is in the shader disk
// cache key.
//   APS5_P0S_NATIVE_FLOAT=1     f32 ops without the RDNA flush, tiny-product and NaN-payload emulation; NMin/NMax
//   APS5_P0S_PAIR_MIN_WORDS=n   spanned (pair) wave64 only for programs of at least n guest code words
//   APS5_P0S_CONST_EXEC=1       a compute program whose waves all start full gets its initial EXEC as a constant
//   APS5_P0S_INLINE=1           the BDA, fault and runtime-V# helpers are emitted without DontInline
//   APS5_P0S_ROBUST=1           bound buffer accesses without the in-shader bound check (robustBufferAccess2 instead)
//   APS5_P0S_FLAT_BDA=1         read lookups try a one-entry-per-page flat table first (the driver appends it)
#include <cstdint>
#include <cstdlib>
#include <spirv/unified1/spirv.hpp>

namespace ShaderRecompiler {
namespace P0s {

inline std::uint32_t EnvNumber(const char* name) {
    const char* text = std::getenv(name);
    return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : 0u;
}

inline bool NativeFloat() {
    static const bool value = EnvNumber("APS5_P0S_NATIVE_FLOAT") != 0u;
    return value;
}

inline std::uint32_t PairMinWords() {
    static const std::uint32_t value = EnvNumber("APS5_P0S_PAIR_MIN_WORDS");
    return value;
}

inline bool ConstExec() {
    static const bool value = EnvNumber("APS5_P0S_CONST_EXEC") != 0u;
    return value;
}

// 1: every helper below; 2: the lookup and fault helpers only; 3: the span/dword and runtime-V# readers only.
inline std::uint32_t InlineHelpers() {
    static const std::uint32_t value = EnvNumber("APS5_P0S_INLINE");
    return value;
}

inline bool RobustBuffers() {
    static const bool value = EnvNumber("APS5_P0S_ROBUST") != 0u;
    return value;
}

inline bool FlatBda() {
    static const bool value = EnvNumber("APS5_P0S_FLAT_BDA") != 0u;
    return value;
}

}

// The function control of a BDA, fault or runtime-V# helper: DontInline as before, None where APS5_P0S_INLINE selects
// the helper's class (lookup = get/probe pointer and record_bda_fault; else a span/dword or runtime-V# reader).
inline spv::FunctionControlMask P0sHelperControl(bool lookup = false) {
    const auto level = P0s::InlineHelpers();
    const bool inlined = level == 1u || (level == 2u && lookup) || (level == 3u && !lookup);
    return inlined ? spv::FunctionControlMaskNone : spv::FunctionControlDontInlineMask;
}

}

#endif
