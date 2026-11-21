#pragma once

#include "graphics/host_gpu/regionDefinitions.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
struct Program;

struct BdaReadWord {
	enum class Kind : uint8_t { Immediate, UserData, Flat };
	Kind     kind                                 = Kind::Immediate;
	uint32_t value                                = 0;
	bool     operator==(const BdaReadWord&) const = default;
};
struct BdaReadSpan {
	BdaReadWord low, high;
	int64_t     begin = 0, end = 0;
	bool        scalar_base = false;
	bool        operator==(const BdaReadSpan&) const = default;
};
struct BdaReadPlan {
	static constexpr uint32_t Capacity = 64;
	std::vector<BdaReadSpan>  spans;
	bool                      complete = false;
	bool                      operator==(const BdaReadPlan&) const = default;
};

[[nodiscard]] BdaReadPlan BuildBdaReadPlan(const Program& program);
}
