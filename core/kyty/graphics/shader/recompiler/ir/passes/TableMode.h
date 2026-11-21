#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_TABLEMODE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_TABLEMODE_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

bool EnterTableMode(Program& program, std::string& reason, bool gpu_reads = false);

struct GpuReadRoot {
	uint32_t low  = 0;
	uint32_t high = 0;
	bool     operator==(const GpuReadRoot&) const = default;
};

std::vector<GpuReadRoot> GpuReadRoots(const TablePlan& plan);
int32_t                  GpuReadRootOf(const TablePlan& plan, const std::vector<GpuReadRoot>& roots, size_t slot);
std::vector<uint32_t>    GpuReadRegisters(const TablePlan& plan);

}

#endif
