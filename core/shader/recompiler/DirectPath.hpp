#ifndef CORE_SHADER_RECOMPILER_DIRECTPATH_HPP
#define CORE_SHADER_RECOMPILER_DIRECTPATH_HPP

#include <cstdlib>

// The direct path is the default on this branch. APS5_NO_DIRECT=1 restores the existing path as a
// whole: direct draws, direct tables, and the changes that went with them (queued labels as
// pending writes, importable GDS, indirect draws in the open pass, skipped epoch bumps for
// unwritten waits, per-frame visibility of large clean ranges).
namespace ShaderRecompiler::DirectPath {

inline bool Enabled() {
    static const bool on = std::getenv("APS5_NO_DIRECT") == nullptr;
    return on;
}

}

#endif
