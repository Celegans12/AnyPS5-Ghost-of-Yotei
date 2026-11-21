#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

struct SplitHints {
	enum class Update : uint32_t { FMin = 1, FMax, UMin, UMax, SMin, SMax, Add, Sub, And, Or, Xor };
	std::vector<std::pair<uint32_t, Update>> unordered_updates;
};

bool SplitWaveAcrossSubgroups(std::vector<uint32_t>& module, std::string& refusal, const SplitHints* hints = nullptr);

bool WaveSplitThreads(const uint32_t (&threads)[3]);

}
