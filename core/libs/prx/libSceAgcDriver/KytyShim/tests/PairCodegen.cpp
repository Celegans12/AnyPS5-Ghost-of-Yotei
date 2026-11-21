#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name           = "kyty_shim_pair_codegen";
constexpr uint32_t    UserDataDwords = 16u;

constexpr std::array<std::uint32_t, 37> LoadFormsCode{
    0xf4000200u, 0xfa000000u, 0x34020085u, 0x16040087u, 0x7e060200u, 0x7e080201u, 0xd70f6a05u, 0x00020503u, 0x500c0880u,
    0xdc308000u, 0x0a7d0005u, 0xdc288001u, 0x0b7d0005u, 0xdc208002u, 0x0c7d0005u, 0xdc308008u, 0x0d7d0003u, 0x7e0e0280u,
    0xdc30800cu, 0x0e000007u, 0xbf8c0070u, 0x93098308u, 0x7d8800a8u, 0xbe8a106au, 0x80090a09u, 0x4a1e1c09u, 0xe0701000u,
    0x80010a01u, 0xe0701004u, 0x80010b01u, 0xe0701008u, 0x80010c01u, 0xe070100cu, 0x80010d01u, 0xe0701010u, 0x80010f01u,
    0xbf810000u,
};

uint32_t g_failures = 0;

void Check(bool value, const std::string& what) {
	if (value) return;
	++g_failures;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

struct Module {
	std::vector<uint32_t>          words;
	std::map<uint32_t, std::string> names;
	std::map<spv::Op, uint32_t>     opcodes;
	std::map<std::string, uint32_t> calls;
	uint32_t                        physical_reads_in_loops = 0;
};

Module Inspect(std::vector<uint32_t> words) {
	Module module;
	module.words = std::move(words);
	std::vector<std::pair<uint32_t, uint32_t>> call_targets;
	std::vector<uint32_t>                      open_loops;
	for (size_t at = 5; at < module.words.size();) {
		const auto count  = module.words[at] >> 16u;
		const auto opcode = static_cast<spv::Op>(module.words[at] & 0xffffu);
		if (count == 0) break;
		++module.opcodes[opcode];
		if (opcode == spv::OpLoopMerge && count > 1) open_loops.push_back(module.words[at + 1]);
		if (opcode == spv::OpLabel && count > 1) std::erase(open_loops, module.words[at + 1]);
		if (opcode == spv::OpFunctionEnd) open_loops.clear();
		if (opcode == spv::OpConvertUToPtr && !open_loops.empty()) ++module.physical_reads_in_loops;
		if (opcode == spv::OpName && count > 2) {
			module.names[module.words[at + 1]] = reinterpret_cast<const char*>(&module.words[at + 2]);
		}
		if (opcode == spv::OpFunctionCall && count > 3) call_targets.emplace_back(module.words[at + 3], 0u);
		at += count;
	}
	for (const auto& [target, unused]: call_targets) {
		const auto found = module.names.find(target);
		++module.calls[found == module.names.end() ? std::string("?") : found->second];
	}
	return module;
}

uint32_t CallsStartingWith(const Module& module, const std::string& prefix) {
	uint32_t total = 0;
	for (const auto& [name, count]: module.calls)
		if (name.starts_with(prefix)) total += count;
	return total;
}

Module Compile(uint32_t host_subgroup) {
	ShaderComputeInputInfo compute;
	compute.threads_num[0]     = 64u;
	compute.threads_num[1]     = 1u;
	compute.threads_num[2]     = 1u;
	compute.wave_size          = 64u;
	compute.host_subgroup_size = host_subgroup;
	compute.thread_ids_num     = 1;
	static Recompiler::Spirv::Profile profile {.maintenance8 = true, .robust_buffer_access = true};
	alignas(256) static std::array<uint32_t, 1024> source_words {};
	alignas(256) static std::array<uint32_t, 1024> output_words {};
	const auto source_address = reinterpret_cast<uintptr_t>(source_words.data());
	const auto output_address = reinterpret_cast<uintptr_t>(output_words.data());
	std::array<uint32_t, UserDataDwords> user_data {};
	user_data[0] = static_cast<uint32_t>(source_address);
	user_data[1] = static_cast<uint32_t>(source_address >> 32u);
	user_data[4] = static_cast<uint32_t>(output_address);
	user_data[5] = static_cast<uint32_t>(output_address >> 32u) & 0xffffu;
	user_data[6] = static_cast<uint32_t>(sizeof(output_words));
	user_data[7] = (0x01016facu & ~(0x7fu << 12u)) | (20u << 12u) | (3u << 28u);
	Recompiler::CompileOptions options;
	options.stage               = ShaderType::Compute;
	options.wave_size           = 64u;
	options.user_data_base      = 0u;
	options.shader_hash         = 0x5a1f0001u;
	options.emit_debug_info     = true;
	options.profile             = &profile;
	options.user_data           = user_data;
	options.input_info.compute  = &compute;
	options.wave_lane_pairs     = host_subgroup == 32u;
	const auto source           = Recompiler::PrepareShaderSource(LoadFormsCode, options);
	auto       translated       = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan             = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime       runtime {.user_data = user_data};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	if (!Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization)) {
		std::fprintf(stderr, "%s: the program's resources were not materialized\n", Name);
		std::exit(1);
	}
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	spvtools::ValidatorOptions validator;
	validator.SetUniformBufferStandardLayout(true);
	Check(tools.Validate(compiled.spirv.data(), compiled.spirv.size(), validator),
	      "subgroup " + std::to_string(host_subgroup) + ": invalid SPIR-V: " + messages);
	return Inspect(std::move(compiled.spirv));
}

}

int main() {
	const auto pairs  = Compile(32u);
	const auto native = Compile(64u);
	const auto loops  = [](const Module& module) {
		const auto found = module.opcodes.find(spv::OpLoopMerge);
		return found == module.opcodes.end() ? 0u : found->second;
	};
	const auto lane_pointers   = CallsStartingWith(native, "get_bda_pointer");
	const auto paired_pointers = CallsStartingWith(pairs, "get_bda_pointer");
	std::printf("%s: address translations: native %u, lane pairs %u; loops: native %u, lane pairs %u; physical reads in loops: "
	            "native %u, lane pairs %u\n",
	            Name, lane_pointers, paired_pointers, loops(native), loops(pairs), native.physical_reads_in_loops,
	            pairs.physical_reads_in_loops);
	for (const auto& [name, count]: pairs.calls) std::printf("%s: lane pairs call %s x%u\n", Name, name.c_str(), count);
	Check(native.physical_reads_in_loops == 0u, "native wave64: a DMA read happens inside its address loop");
	Check(pairs.physical_reads_in_loops == 0u, "lane pairs: a DMA read happens inside its address loop");
	Check(lane_pointers == 4u && loops(native) == 4u,
	      "native wave64: one address loop for each of the four dword or ushort reads");
	Check(paired_pointers == 6u && loops(pairs) == 6u,
	      "lane pairs: the two reads at a lane-invariant address are translated once, the two per-lane reads once per half");
	Check(CallsStartingWith(native, "bda_load_u32") == 1u && CallsStartingWith(pairs, "bda_load_u32") == 2u,
	      "the byte read goes through the shared dword helper, once per half in lane pairs");
	if (g_failures != 0) return 1;
	std::printf("%s: ok\n", Name);
	return 0;
}
