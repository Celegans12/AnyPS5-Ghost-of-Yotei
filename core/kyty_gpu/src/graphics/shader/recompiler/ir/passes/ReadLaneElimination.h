#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct ReadLaneStats {
	uint32_t rewritten_reads      = 0;
	uint32_t rewritten_reductions = 0;
};

// rewrite_reductions: lower a wave64 UMin reduction (DPP rows, permlane, lane reads) to one SubgroupUMinU32 (upstream
// 302240da7). Off for paired shaders (WavePairRequested): their wave is two 32-wide subgroups, so a subgroup reduction
// covers half the wave (24.10: the volumetric fog shader c6b0a54e lit froxels again - glowing rectangles by fires).
[[nodiscard]] ReadLaneStats EliminateReadLane(Program& program, uint32_t wave_size, bool rewrite_reductions = true);

// Replaces Select(C, new, old) by `new` when every observation of the
// result happens on lanes where C is true (see ReadLaneElimination.cpp). Run after
// RemoveIdentities; the caller removes the Identity instructions and dead code afterwards.
struct ExecSelectElideStats {
	uint32_t selects    = 0; // SelectU32 instructions seen
	uint32_t rewritten  = 0; // stacked same-condition selects collapsed
	uint32_t restored   = 0; // of those: undone because the inner select stays alive anyway
	uint32_t elided     = 0; // selects replaced by their `new` operand
	uint32_t unobserved = 0; // of those: results nobody observes
	uint32_t top        = 0; // kept: observed on every lane
	uint32_t wider      = 0; // kept: observed on lanes outside the condition
	bool     skipped    = false;
};

[[nodiscard]] ExecSelectElideStats ElideRedundantExecSelects(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
