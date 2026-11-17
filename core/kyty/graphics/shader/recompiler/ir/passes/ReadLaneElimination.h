#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ReadLaneStats {
	uint32_t rewritten_reads      = 0;
	uint32_t rewritten_reductions = 0;
};

[[nodiscard]] std::vector<Inst*> MarkUndefinedLaneSpills(Program& program);
void ReleaseUndefinedLaneSpills(const std::vector<Inst*>& undefined);

[[nodiscard]] ReadLaneStats EliminateReadLane(Program& program, uint32_t wave_size);

}
