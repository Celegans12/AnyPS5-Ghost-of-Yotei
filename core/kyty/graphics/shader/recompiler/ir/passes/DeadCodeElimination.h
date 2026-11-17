#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;

void RemoveIdentities(Program& program);
void EliminateDeadCode(const BlockList& blocks);
uint32_t EliminateMaskedSelects(const BlockList& blocks);

}
