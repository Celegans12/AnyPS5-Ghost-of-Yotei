#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/RelativeVgprLowering.h"
#include "graphics/shader/shaderCompiler.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane   = 0x34020082u;
constexpr uint32_t LoadS1Lo    = 0xf4200042u;
constexpr uint32_t LoadS1Hi    = 0xfa000000u;
constexpr uint32_t WaitScalar  = 0xbf8cc07fu;
constexpr uint32_t M0FromS1    = 0xbefc0301u;
constexpr std::array<uint32_t, 8> FillV3ToV10 {0x4a060083u, 0x4a080084u, 0x4a0a0085u, 0x4a0c0086u, 0x4a0e0087u, 0x4a100088u, 0x4a120089u, 0x4a14008au};
constexpr uint32_t MoveRelSrc  = 0x7e048703u;
constexpr uint32_t StoreRawLo  = 0xe0701000u;
constexpr uint32_t StoreRawHi  = 0x80010201u;
constexpr uint32_t WaitAll     = 0xbf8c0000u;
constexpr uint32_t EndProgram  = 0xbf810000u;

int g_failures = 0;

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

uint32_t LoweredReads(ShaderType stage) {
	std::vector<uint32_t> code {ShiftLane};
	code.insert(code.end(), FillV3ToV10.begin(), FillV3ToV10.end());
	code.insert(code.end(), {LoadS1Lo, LoadS1Hi, WaitScalar, M0FromS1, MoveRelSrc, StoreRawLo, StoreRawHi, WaitAll, EndProgram});
	ShaderComputeInputInfo      info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x52u;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	translated.program.stage   = stage;
	return IR::LowerRelativeVgprs(translated.program);
}

}

int main() {
	Common::SoftExitScope soft;
	Check(LoweredReads(ShaderType::Compute) != 0, "a compute shader indexes its M0-relative VGPRs through the per-lane array");
	Check(LoweredReads(ShaderType::Vertex) != 0, "a vertex shader indexes its M0-relative VGPRs through the per-lane array");
	Check(LoweredReads(ShaderType::Pixel) == 0, "a pixel shader keeps its M0-relative VGPR select chains");
	if (g_failures != 0) {
		std::fprintf(stderr, "kyty_shim_relative_vgpr_stages: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_relative_vgpr_stages: ok\n");
	return 0;
}
