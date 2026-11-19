#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

bool SplitWaveAcrossSubgroups(std::vector<uint32_t>& module, std::string& refusal);

}
