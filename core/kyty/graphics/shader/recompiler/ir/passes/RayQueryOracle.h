#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/RayQueryTraversal.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t OracleRemaining = 1u;
constexpr uint32_t OracleForms     = 2u;
constexpr uint32_t OracleOnly      = 4u;
constexpr uint32_t OraclePixel     = 8u;

[[nodiscard]] bool OracleLaneWalk(std::span<Inst* const> walk);
uint32_t           MarkOracleWalk(std::span<Inst* const> walk);
size_t             MarkEmptyWalk(const Program& program, std::span<Inst* const> intersections,
                                 const NativeTraversalOptions& options);

}
