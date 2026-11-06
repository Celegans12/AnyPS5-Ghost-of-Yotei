#ifndef CORE_SHADER_RECOMPILER_V3TSWITCHES_HPP
#define CORE_SHADER_RECOMPILER_V3TSWITCHES_HPP

#include <cstdint>
#include <cstdlib>

namespace ShaderRecompiler {
namespace V3t {

inline std::uint32_t EnvNumber(const char* name) {
    const char* text = std::getenv(name);
    return text != nullptr ? static_cast<std::uint32_t>(std::strtoul(text, nullptr, 0)) : 0u;
}

inline bool FlatInline() {
    static const bool value = EnvNumber("APS5_V3T_FLAT") != 0u;
    return value;
}

inline std::uint32_t FlatLevel() {
    static const std::uint32_t value = EnvNumber("APS5_V3T_FLAT");
    return value;
}

inline bool SrtGuardHoist() {
    static const bool value = EnvNumber("APS5_V3T_SRT_GUARD") != 0u;
    return value;
}

inline bool SpecShape() {
    static const bool value = EnvNumber("APS5_V3T_SPEC_SHAPE") != 0u;
    return value;
}

inline bool SpecShapeChecked() {
    static const bool value = EnvNumber("APS5_V3T_SPEC_SHAPE") == 2u;
    return value;
}

inline bool OneLoad() {
    static const bool value = EnvNumber("APS5_V3T_ONE_LOAD") != 0u;
    return value;
}

inline bool BranchlessLoads() {
    static const bool value = EnvNumber("APS5_V3T_BRANCHLESS") != 0u;
    return value;
}

inline bool Census() {
    static const bool value = EnvNumber("APS5_V3T_CENSUS") != 0u;
    return value;
}

inline std::uint32_t PlainCoherentReads() {
    static const std::uint32_t value = EnvNumber("APS5_V3T_NOVOL");
    return value;
}

}
}

#endif
