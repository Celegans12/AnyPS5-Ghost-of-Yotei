#pragma once

#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t OracleLoop = 16u;

[[nodiscard]] bool RewriteResultsWalk(Program& program, const NativeTraversalOptions& options, Block* header, Block* latch,
                                      std::span<Inst* const> served, NativeTraversals& result, uint32_t site);

}
