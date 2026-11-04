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

inline bool InlineHelpers() {
    static const bool value = EnvNumber("APS5_P0S_INLINE") != 0u;
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

// The function control of a BDA, fault or runtime-V# helper: DontInline as before, None with APS5_P0S_INLINE.
inline spv::FunctionControlMask P0sHelperControl() {
    return P0s::InlineHelpers() ? spv::FunctionControlMaskNone : spv::FunctionControlDontInlineMask;
}

}

#endif
