#include "common/fallback.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

struct PipelineCacheTestAccess {
	using Program     = PipelineCache::Program;
	using Speculation = PipelineCache::Program::Speculation;
};

}

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;
using Program        = PipelineCacheTestAccess::Program;
using Speculation    = PipelineCacheTestAccess::Speculation;

constexpr uint32_t ShiftLane      = 0x34020082u;
constexpr uint32_t LoadFormatX[]  = {0xe0001000u, 0x80000201u};
constexpr uint32_t WaitAll        = 0xbf8c0000u;
constexpr uint32_t StoreRaw[]     = {0xe0701000u, 0x80010201u};
constexpr uint32_t EndProgram     = 0xbf810000u;
constexpr uint32_t Format32UInt   = 20;
constexpr uint32_t SwizzleXyzw    = 0xfacu;
constexpr uint32_t ReservedX      = 0xfabu;
constexpr uint64_t SourceAddress  = 0x200810000ull;
constexpr uint64_t TargetAddress  = 0x200820000ull;
constexpr uint32_t Records        = 128u;
constexpr uint32_t Cursor         = 0u;
constexpr uint32_t Passes         = 5u;
constexpr uint64_t Hash           = 0x5bec0u;
constexpr const char* Site        = "kyty_shim_speculative_compile";

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

constexpr Sharp MakeSharp(uint64_t address, uint32_t swizzle) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu), Records * 4u,
	        swizzle | (Format32UInt << 12u) | (1u << 24u) | (3u << 28u)};
}

const std::vector<uint32_t> g_code {ShiftLane,  LoadFormatX[0], LoadFormatX[1], WaitAll,
                                    StoreRaw[0], StoreRaw[1],   EndProgram};
const std::vector<uint32_t> g_plan_user_data(16, 0u);
ShaderComputeInputInfo      g_info = [] {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	return info;
}();

Recompiler::CompileOptions Options() {
	Recompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = Hash;
	options.dump_ir            = false;
	options.user_data          = g_plan_user_data;
	options.input_info.compute = &g_info;
	return options;
}

Recompiler::TranslateResult Translate() {
	const auto options = Options();
	const auto source  = Recompiler::PrepareShaderSource(g_code, options);
	return Recompiler::TranslateProgram(source.decoded, options);
}

IR::ResourceSpecialization Specialize(uint32_t input_swizzle) {
	auto                     translated = Translate();
	const auto               plan       = IR::ExtractResourcePlan(translated.program);
	std::array<uint32_t, 16> user_data {};
	const auto               input  = MakeSharp(SourceAddress, input_swizzle);
	const auto               output = MakeSharp(TargetAddress, SwizzleXyzw);
	std::copy(input.begin(), input.end(), user_data.begin());
	std::copy(output.begin(), output.end(), user_data.begin() + 4);
	const IR::SrtRuntime       runtime {.user_data = user_data};
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	Check(IR::MaterializeResources(plan, runtime, snapshot, specialization),
	      "the resources of the program are materialized");
	return specialization;
}

std::vector<uint32_t> Compile(const IR::ResourceSpecialization& specialization) {
	return Recompiler::CompileProgram(Translate(), Options(), specialization, 0).spirv;
}

struct Tally {
	uint32_t attempts  = 0;
	uint32_t contained = 0;
	uint32_t kept      = 0;
	uint32_t refused   = 0;
};

Tally RunPasses(Program& program, const IR::ResourceSpecialization& specialization) {
	Tally tally;
	for (uint32_t pass = 0; pass < Passes; pass++) {
		switch (program.JudgeSpeculativeCompile(ShaderType::Compute, specialization, Cursor, true)) {
			case Speculation::Kept: tally.kept++; break;
			case Speculation::Refused: tally.refused++; break;
			case Speculation::Uncontained: break;
			case Speculation::Compile: {
				tally.attempts++;
				std::vector<uint32_t> spirv;
				if (!Common::Fallback::Run("test compile ahead dropped", Site, Hash, 0, true,
				                           [&] { spirv = Compile(specialization); })) {
					tally.contained++;
					program.NoteSpeculativeFailure(specialization, Cursor);
				}
				break;
			}
		}
	}
	return tally;
}

int Finish(const char* name) {
	if (g_failures != 0) {
		std::fprintf(stderr, "%s: %d failure(s)\n", name, g_failures);
		return 1;
	}
	std::printf("%s: ok\n", name);
	return 0;
}

}

extern "C" int RunKytySpeculativeCompileTest(int argc, char** argv) {
	const auto valid    = Specialize(SwizzleXyzw);
	const auto reserved = Specialize(ReservedX);
	Check(!valid.buffers.empty(), "the program has buffers");
	if (argc > 1 && std::strcmp(argv[1], "required") == 0) {
		std::vector<uint32_t> spirv;
		static_cast<void>(Common::Fallback::Run("test draw skipped", Site, Hash, 0, [&] { spirv = Compile(reserved); }));
		std::fprintf(stderr, "kyty_shim_speculative_compile: a required compile did not stop\n");
		return 1;
	}
	Check(!Common::Fallback::Enabled(), "the test runs with KYTY_FALLBACKS=0");
	Check(Common::Fallback::Containable(), "this platform does not catch a stop");

	Check(IR::SpecializedBuffersValid(valid, true), "a valid V# is refused");
	Check(!IR::SpecializedBuffersValid(reserved, true), "a reserved dst_sel is not detected");
	auto add_tid = valid;
	add_tid.buffers[0].packed_stride |= 1u << 20u;
	Check(IR::SpecializedBuffersValid(add_tid, true), "ADD_TID is refused in a compute program");
	Check(!IR::SpecializedBuffersValid(add_tid, false), "ADD_TID is not detected outside a compute program");
	{
		std::vector<uint32_t> spirv;
		Check(!Common::Fallback::Run("test compile ahead dropped", Site, Hash, 0, true,
		                             [&] { spirv = Compile(reserved); }),
		      "the recompiler does not stop on the reserved dst_sel the detector refuses");
	}

	Program    program;
	const auto refused = RunPasses(program, reserved);
	Check(refused.attempts == 0, "a variant with a reserved dst_sel was compiled");
	Check(refused.refused == 1 && refused.kept == Passes - 1,
	      "a refused variant is not refused once then kept: refused " + std::to_string(refused.refused) + ", kept " +
	          std::to_string(refused.kept));

	const IR::ResourceSpecialization truncated;
	Check(IR::SpecializedBuffersValid(truncated, true), "the truncated variant is refused before compiling");
	const auto failed = RunPasses(program, truncated);
	Check(failed.attempts == 1, "a failed compile ahead was tried " + std::to_string(failed.attempts) + " times");
	Check(failed.contained == 1, "a failed compile ahead stopped the test or did not fail");
	Check(failed.kept == Passes - 1, "a failed compile ahead is not kept for the next passes");

	Check(!program.Failed(reserved, Cursor) && !program.Failed(truncated, Cursor),
	      "a failure ahead refuses the variant to the draw");
	Check(program.JudgeSpeculativeCompile(ShaderType::Compute, valid, Cursor, true) == Speculation::Compile,
	      "a failure ahead refuses the other variants of the program");
	{
		std::vector<uint32_t> spirv;
		const bool compiled = Common::Fallback::Run("test draw skipped", Site, Hash, 0, [&] { spirv = Compile(valid); });
		Check(compiled && !spirv.empty(), "the variant the draw needs does not compile on the normal path");
	}

	Program capped;
	for (uint32_t index = 0; index < Program::SpeculativeFailureCap; index++) {
		auto variant = valid;
		variant.buffers[0].indirect_search_iterations = index + 1u;
		Check(capped.JudgeSpeculativeCompile(ShaderType::Compute, variant, Cursor, true) == Speculation::Compile,
		      "a new variant is refused before the cap");
		capped.NoteSpeculativeFailure(variant, Cursor);
	}
	Check(capped.JudgeSpeculativeCompile(ShaderType::Compute, valid, Cursor, true) == Speculation::Kept,
	      "a program past the cap of failures ahead is still compiled ahead");
	Check(capped.speculative_failures.size() == Program::SpeculativeFailureCap, "the failures ahead grow past the cap");

	return Finish("kyty_shim_speculative_compile");
}
