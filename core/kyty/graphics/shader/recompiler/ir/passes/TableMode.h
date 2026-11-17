#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_TABLEMODE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_TABLEMODE_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <string>

namespace Libs::Graphics::ShaderRecompiler::IR {

bool EnterTableMode(Program& program, std::string& reason);

}

#endif
