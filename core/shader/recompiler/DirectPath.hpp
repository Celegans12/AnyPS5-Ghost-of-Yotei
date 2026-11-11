#ifndef CORE_SHADER_RECOMPILER_DIRECTPATH_HPP
#define CORE_SHADER_RECOMPILER_DIRECTPATH_HPP

#include <cstdlib>

// The direct path is the default on this branch. Every direct-path feature that is kept (measured
// faster, picture correct) is on without a switch; a feature joins here when it is proven, and
// until then it has its own opt-in switch. APS5_NO_DIRECT=1 restores the existing path as a whole.
namespace ShaderRecompiler::DirectPath {

inline bool All() {
    static const bool on = std::getenv("APS5_NO_DIRECT") == nullptr;
    return on;
}

// Plain vertex and pixel draws take the direct path. APS5_NO_DIRECT_DRAW=1 turns only this off;
// APS5_DIRECT_DRAW=1 turns it on under APS5_NO_DIRECT.
inline bool Draws() {
    static const bool on = (All() && std::getenv("APS5_NO_DIRECT_DRAW") == nullptr) || std::getenv("APS5_DIRECT_DRAW") != nullptr;
    return on;
}

// Stages with image tables go direct too. APS5_NO_DIRECT_TABLES=1 turns only this off;
// APS5_DIRECT_TABLES=1 turns it on under APS5_NO_DIRECT.
inline bool Tables() {
    static const bool on = (All() && std::getenv("APS5_NO_DIRECT_TABLES") == nullptr) || std::getenv("APS5_DIRECT_TABLES") != nullptr;
    return on;
}

}

#endif
