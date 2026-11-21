#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_E2STATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_E2STATS_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics::E2Stats {

enum class Cause : uint8_t { None, AheadWalk, RevalidationTable, RevalidationWalk };

struct Refusal {
	uint64_t                                hash = 0;
	ShaderRecompiler::IR::SrtRefusalClasses classes;
	Cause                                   cause    = Cause::None;
	bool                                    dispatch = false;
};

[[nodiscard]] bool Enabled() noexcept;
void               Configure();

using ClassSet = uint8_t;
struct CoverageSet {
	const char* name    = "";
	ClassSet    classes = 0;
};
[[nodiscard]] std::array<CoverageSet, 5> CoverageSets();
[[nodiscard]] bool        Covers(ClassSet classes, const Refusal& refusal);
[[nodiscard]] const char* ClassName(ShaderRecompiler::IR::SrtLoadClass load_class);

void NoteRefusal(const Refusal& refusal);
void NoteExecuteResolution(const Refusal& refusal, uint64_t ns, uint64_t readbacks, uint64_t readback_ns);
void NoteArgsRead(bool gpu_written, uint64_t ns);
[[nodiscard]] std::vector<std::string> TakeSummary(double seconds, uint64_t frames);

}

#endif
