#include "common/assert.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Recompiler::IR;

constexpr uint32_t ShiftLane  = 0x34020082u;
constexpr uint32_t WaitAll    = 0xbf8c0000u;
constexpr uint32_t EndProgram = 0xbf810000u;

constexpr std::array<uint32_t, 2> LoadRaw {0xe0301000u, 0x80000201u};
constexpr std::array<uint32_t, 2> LoadFormatX {0xe0001000u, 0x80000201u};
constexpr std::array<uint32_t, 2> LoadFormatXyzw {0xe00c1000u, 0x80000201u};
constexpr std::array<uint32_t, 2> StoreRaw {0xe0701000u, 0x80010201u};

constexpr uint32_t Format32UInt  = 20;
constexpr uint32_t SwizzleXyzw   = 0xfacu;
constexpr uint64_t SourceAddress = 0x200810000ull;
constexpr uint64_t TargetAddress = 0x200820000ull;
constexpr uint32_t Records       = 128u;
constexpr uint32_t Threads       = 16u;
constexpr uint32_t Rounds        = 6u;

std::atomic<int> g_failures {0};

void Check(bool condition, const std::string& what) {
	if (condition) return;
	std::fprintf(stderr, "FAILED: %s\n", what.c_str());
	g_failures++;
}

using Sharp = std::array<uint32_t, 4>;

Sharp MakeSharp(uint64_t address, uint32_t records, uint32_t format, uint32_t swizzle, uint32_t select) {
	return {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu), records,
	        swizzle | (format << 12u) | (1u << 24u) | (select << 28u)};
}

std::vector<uint32_t> Program(std::span<const uint32_t, 2> load) {
	return {ShiftLane, load[0], load[1], WaitAll, StoreRaw[0], StoreRaw[1], EndProgram};
}

std::vector<uint32_t> CompileOnce(const std::vector<uint32_t>& code) {
	ShaderComputeInputInfo info {};
	info.wave_size          = 64;
	info.host_subgroup_size = 64;
	info.threads_num[0]     = 64;
	info.threads_num[1]     = 1;
	info.threads_num[2]     = 1;
	info.thread_ids_num     = 1;
	const std::vector<uint32_t> plan_user_data(16, 0u);
	Recompiler::CompileOptions  options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x51u;
	options.dump_ir            = false;
	options.user_data          = plan_user_data;
	options.input_info.compute = &info;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan            = IR::ExtractResourcePlan(translated.program);
	std::array<uint32_t, 16> user_data {};
	const auto               input  = MakeSharp(SourceAddress, Records * 4u, Format32UInt, SwizzleXyzw, 3);
	const auto               output = MakeSharp(TargetAddress, Records * 4u, Format32UInt, SwizzleXyzw, 3);
	std::copy(input.begin(), input.end(), user_data.begin());
	std::copy(output.begin(), output.end(), user_data.begin() + 4);
	const IR::SrtRuntime       runtime {.user_data = user_data, .runtime_stride = true};
	IR::ResourceSnapshot       snapshot;
	IR::ResourceSpecialization specialization;
	if (!IR::MaterializeResources(plan, runtime, snapshot, specialization)) return {};
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0);
	return std::move(compiled.spirv);
}

void Parallel(const std::vector<std::vector<uint32_t>>& programs) {
	std::vector<std::vector<std::vector<uint32_t>>> expected(programs.size());
	for (size_t index = 0; index < programs.size(); index++) {
		expected[index].resize(1);
		expected[index][0] = CompileOnce(programs[index]);
		Check(!expected[index][0].empty(), "program " + std::to_string(index) + " compiles on one thread");
	}
	std::atomic<uint32_t>    compiled {0};
	std::vector<std::thread> pool;
	for (uint32_t thread = 0; thread < Threads; thread++) {
		pool.emplace_back([&, thread] {
			for (uint32_t round = 0; round < Rounds; round++) {
				for (size_t offset = 0; offset < programs.size(); offset++) {
					const auto index  = (offset + thread + round) % programs.size();
					const auto result = CompileOnce(programs[index]);
					Check(result == expected[index][0], "program " + std::to_string(index) + " compiles to the same SPIR-V on every thread");
					compiled++;
				}
			}
		});
	}
	for (auto& worker: pool) worker.join();
	Check(compiled.load() == Threads * Rounds * programs.size(), "every thread finished every compilation");
}

}

int main() {
	Common::SoftExitScope soft;
	const std::vector<std::vector<uint32_t>> programs {Program(LoadRaw), Program(LoadFormatX), Program(LoadFormatXyzw)};
	Parallel(programs);
	if (g_failures.load() != 0) {
		std::fprintf(stderr, "kyty_shim_parallel_compile: %d failure(s)\n", g_failures.load());
		return 1;
	}
	std::printf("kyty_shim_parallel_compile: ok\n");
	return 0;
}
