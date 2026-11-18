#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {
struct Program;
}

namespace Libs::Graphics::ShaderRecompiler::IR {

[[nodiscard]] uint32_t TraceTraversalsWithRayQueries(Program& program);

struct NativeTraversals {
	uint32_t                                   sites         = 0;
	bool                                       source_valid  = false;
	uint32_t                                   user_sgpr     = 0;
	uint32_t                                   header_offset = 0;
	uint32_t                                   header_vsharp = 0xffffffffu;
	uint32_t                                   root          = 0;
	std::array<uint32_t, 4>                    layout {};
	std::vector<std::pair<uint32_t, uint32_t>> reach;
	std::vector<NativeTraversals>              oracle_sources;
};

struct NativeTraversalOptions {
	uint32_t wave_size = 64;
	bool rewrite = false;
	uint32_t oracle = 0;
	const Decoder::Program* decoded = nullptr;
};

[[nodiscard]] NativeTraversals TraceNativeTraversals(Program&                      program,
                                                     const NativeTraversalOptions& options);

}
