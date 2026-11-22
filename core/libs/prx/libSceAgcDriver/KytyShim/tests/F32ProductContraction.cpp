#include "graphics/host_gpu/renderer/pipeline/pipelineStoreFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name           = "kyty_shim_f32_product_contraction";
constexpr uint32_t    UserDataDwords = 16u;
constexpr uint32_t    Threads        = 64u;

constexpr std::array<uint32_t, 25> FormsCode {
    0x34020084u, 0x34060086u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0x10100b04u, 0x08140d08u, 0x08161106u, 0x10120b05u,
    0xd503000cu, 0x20020909u, 0x061a0b04u, 0xd541000eu, 0x041a0b04u, 0xe0701000u, 0x80010a03u, 0xe0701004u, 0x80010b03u,
    0xe0701008u, 0x80010c03u, 0xe070100cu, 0x80010d03u, 0xe0701010u, 0x80010e03u, 0xbf810000u,
};

constexpr std::array<uint32_t, 16> ExecCode {
    0x34020084u, 0x34060086u, 0xe0381000u, 0x80000401u, 0xbf8c3f70u, 0x7c020b04u, 0xbe88246au, 0x10100b04u,
    0xbefe0408u, 0x08140d08u, 0x06160b04u, 0xe0701000u, 0x80010a03u, 0xe0701004u, 0x80010b03u, 0xbf810000u,
};

constexpr std::array<uint32_t, 2> LaneIndex {0xd7650014u, 0x000100c1u};

uint32_t g_failures = 0;

void Check(bool value, const std::string& what) {
	if (value) return;
	++g_failures;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

struct Module {
	std::map<spv::Op, std::vector<uint32_t>> f32_results;
	std::map<uint32_t, uint32_t>             no_contraction;
	std::set<uint32_t>                       product_sums;
	uint32_t                                 u32_selects = 0;
	bool                                     lane_local  = false;
};

enum class Path { Native, SubgroupSplit, LanePairs };

std::vector<uint32_t> WithLaneIndex(std::span<const uint32_t> code) {
	std::vector<uint32_t> crossing(code.begin(), code.end() - 1);
	crossing.insert(crossing.end(), LaneIndex.begin(), LaneIndex.end());
	crossing.push_back(code.back());
	return crossing;
}

std::vector<bool> Marks(const Module& module, spv::Op opcode) {
	std::vector<bool> marks;
	const auto        found = module.f32_results.find(opcode);
	if (found == module.f32_results.end()) return marks;
	for (const auto id: found->second) marks.push_back(module.no_contraction.contains(id));
	return marks;
}

uint32_t Marked(const Module& module, spv::Op opcode) {
	const auto marks = Marks(module, opcode);
	return static_cast<uint32_t>(std::ranges::count(marks, true));
}

Module Inspect(const std::vector<uint32_t>& words) {
	Module                                       module;
	std::set<uint32_t>                           f32_types;
	std::set<uint32_t>                           u32_types;
	std::map<uint32_t, std::vector<uint32_t>>    carried_from;
	std::vector<std::pair<uint32_t, std::array<uint32_t, 2>>> sums;
	for (size_t at = 5; at < words.size();) {
		const auto count  = words[at] >> 16u;
		const auto opcode = static_cast<spv::Op>(words[at] & 0xffffu);
		if (count == 0) break;
		if (opcode == spv::OpTypeFloat && words[at + 2] == 32u) f32_types.insert(words[at + 1]);
		if (opcode == spv::OpTypeInt && words[at + 2] == 32u && words[at + 3] == 0u) u32_types.insert(words[at + 1]);
		if (opcode == spv::OpDecorate && words[at + 2] == spv::DecorationNoContraction) ++module.no_contraction[words[at + 1]];
		if ((opcode == spv::OpFMul || opcode == spv::OpFAdd || opcode == spv::OpFSub) && f32_types.contains(words[at + 1])) {
			module.f32_results[opcode].push_back(words[at + 2]);
			if (opcode != spv::OpFMul) sums.push_back({words[at + 2], {words[at + 3], words[at + 4]}});
		}
		if (opcode == spv::OpFNegate || opcode == spv::OpBitcast) carried_from[words[at + 2]] = {words[at + 3]};
		if (opcode == spv::OpSelect) carried_from[words[at + 2]] = {words[at + 4], words[at + 5]};
		if (opcode == spv::OpSelect && u32_types.contains(words[at + 1])) ++module.u32_selects;
		at += count;
	}
	const auto products = module.f32_results[spv::OpFMul];
	std::function<bool(uint32_t)> from_product = [&](uint32_t id) {
		if (std::ranges::find(products, id) != products.end()) return true;
		const auto carried = carried_from.find(id);
		return carried != carried_from.end() && std::ranges::any_of(carried->second, from_product);
	};
	for (const auto& [sum, operands]: sums) {
		if (std::ranges::any_of(operands, from_product)) module.product_sums.insert(sum);
	}
	return module;
}

Module Compile(std::span<const uint32_t> code, bool exact, Path path, const std::string& what) {
	ShaderComputeInputInfo compute;
	compute.threads_num[0]     = Threads;
	compute.threads_num[1]     = 1u;
	compute.threads_num[2]     = 1u;
	compute.wave_size          = 64u;
	compute.host_subgroup_size = path == Path::Native ? 64u : 32u;
	compute.thread_ids_num     = 1;
	Recompiler::Spirv::Profile profile;
	profile.f32_product_no_contraction = exact;
	alignas(256) static std::array<uint32_t, Threads * 4u>  input {};
	alignas(256) static std::array<uint32_t, Threads * 16u> output {};
	const auto descriptor = [](const void* data, uint32_t bytes) {
		const auto address = reinterpret_cast<uintptr_t>(data);
		return std::array<uint32_t, 4> {static_cast<uint32_t>(address), static_cast<uint32_t>((address >> 32u) & 0xffffu),
		                                bytes, 0x31016facu};
	};
	std::array<uint32_t, UserDataDwords> user_data {};
	std::ranges::copy(descriptor(input.data(), sizeof(input)), user_data.begin());
	std::ranges::copy(descriptor(output.data(), sizeof(output)), user_data.begin() + 4);
	Recompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64u;
	options.user_data_base     = 0u;
	options.shader_hash        = 0xf32c0001u;
	options.profile            = &profile;
	options.user_data          = user_data;
	options.input_info.compute = &compute;
	options.wave_lane_pairs    = path == Path::LanePairs;
	const auto source          = Recompiler::PrepareShaderSource(code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	const bool lane_local      = translated.program.lane_local;
	const auto plan            = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime       runtime {.user_data = user_data};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	if (!Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization)) {
		std::fprintf(stderr, "%s: %s: the program's resources were not materialized\n", Name, what.c_str());
		std::exit(1);
	}
	auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	Check(tools.Validate(compiled.spirv.data(), compiled.spirv.size()), what + ": invalid SPIR-V: " + messages);
	auto module = Inspect(compiled.spirv);
	module.lane_local = lane_local;
	Check(std::ranges::all_of(module.no_contraction, [](const auto& entry) { return entry.second == 1u; }),
	      what + ": a result is decorated NoContraction more than once");
	return module;
}

using Marking = std::vector<bool>;

Marking Repeated(const Marking& marking, uint32_t times) {
	Marking repeated;
	for (uint32_t copy = 0; copy < times; ++copy) repeated.insert(repeated.end(), marking.begin(), marking.end());
	return repeated;
}

void CheckProvenance(const Module& module, const std::string& what) {
	for (const auto opcode: {spv::OpFAdd, spv::OpFSub}) {
		const auto found = module.f32_results.find(opcode);
		if (found == module.f32_results.end()) continue;
		for (const auto sum: found->second) {
			Check(module.no_contraction.contains(sum) == module.product_sums.contains(sum),
			      what + ": an f32 add or subtract is NoContraction exactly when an operand comes from a product");
		}
	}
}

Module CompileSplit(std::span<const uint32_t> code, bool exact, Path path, const std::string& what) {
	const auto crossing = WithLaneIndex(code);
	auto       module   = Compile(crossing, exact, path, what);
	Check(!module.lane_local, what + ": the program with v_mbcnt is lane-local, so no 32-lane split runs");
	return module;
}

void CheckForms() {
	const Marking subtracts {true, true};
	const Marking adds {true, false, true};
	const Marking multiplies {false, false, true};
	const auto exact = Compile(FormsCode, true, Path::Native, "forms, exact");
	Check(Marks(exact, spv::OpFSub) == subtracts, "exact: a subtract with a product as either operand must be NoContraction");
	Check(Marks(exact, spv::OpFAdd) == adds,
	      "exact: the add of a negated product and the v_mad_f32 sum are NoContraction, the add of two inputs is not");
	Check(Marks(exact, spv::OpFMul) == multiplies, "exact: only the v_mad_f32 product is NoContraction among the multiplies");
	CheckProvenance(exact, "forms, exact");
	const auto fused = Compile(FormsCode, false, Path::Native, "forms, switch off");
	Check(Marks(fused, spv::OpFSub) == Marking {false, false}, "switch off: no subtract is NoContraction");
	Check(Marks(fused, spv::OpFAdd) == Marking {false, false, true}, "switch off: only the v_mad_f32 sum is NoContraction");
	Check(Marks(fused, spv::OpFMul) == multiplies, "switch off: only the v_mad_f32 product is NoContraction");

	const auto split = CompileSplit(FormsCode, true, Path::SubgroupSplit, "forms, exact, split across subgroups");
	Check(Marks(split, spv::OpFSub) == subtracts && Marks(split, spv::OpFAdd) == adds && Marks(split, spv::OpFMul) == multiplies,
	      "exact, split across subgroups: the forms are marked as in the native wave64");
	CheckProvenance(split, "forms, exact, split across subgroups");
	const auto pairs = CompileSplit(FormsCode, true, Path::LanePairs, "forms, exact, lane pairs");
	Check(Marks(pairs, spv::OpFSub).size() == 2u * subtracts.size() && Marked(pairs, spv::OpFSub) == 2u * subtracts.size() &&
	          Marks(pairs, spv::OpFAdd).size() == 2u * adds.size() && Marked(pairs, spv::OpFAdd) == 4u &&
	          Marks(pairs, spv::OpFMul).size() == 2u * multiplies.size() && Marked(pairs, spv::OpFMul) == 2u,
	      "exact, lane pairs: both halves of the forms are marked as in the native wave64");
	CheckProvenance(pairs, "forms, exact, lane pairs");
	const auto pairs_fused = CompileSplit(FormsCode, false, Path::LanePairs, "forms, lane pairs, switch off");
	Check(Marks(pairs_fused, spv::OpFSub) == Repeated({false, false}, 2u),
	      "switch off, lane pairs: no subtract is NoContraction");
}

void CheckExec() {
	const auto exact = Compile(ExecCode, true, Path::Native, "EXEC, exact");
	Check(exact.u32_selects > 0u, "EXEC: the VGPR writes after s_and_saveexec_b64 are no OpSelect of 32-bit words");
	Check(Marks(exact, spv::OpFSub) == Marking {true} && Marks(exact, spv::OpFAdd) == Marking {false},
	      "exact: a subtract, after EXEC is restored, of a product written under EXEC must be NoContraction");
	CheckProvenance(exact, "EXEC, exact");
	const auto fused = Compile(ExecCode, false, Path::Native, "EXEC, switch off");
	Check(Marks(fused, spv::OpFSub) == Marking {false} && Marks(fused, spv::OpFAdd) == Marking {false},
	      "switch off: no add or subtract under EXEC is NoContraction");
	for (const auto path: {Path::SubgroupSplit, Path::LanePairs}) {
		const auto halves = path == Path::LanePairs ? 2u : 1u;
		const auto what   = std::string(path == Path::LanePairs ? "EXEC, exact, lane pairs" : "EXEC, exact, split across subgroups");
		const auto split  = CompileSplit(ExecCode, true, path, what);
		Check(split.u32_selects > 0u, what + ": the VGPR writes are no OpSelect of 32-bit words");
		Check(Marks(split, spv::OpFSub) == Repeated({true}, halves) && Marks(split, spv::OpFAdd) == Repeated({false}, halves),
		      what + ": a subtract of a product written under EXEC must be NoContraction in every half");
		CheckProvenance(split, what);
	}
}

std::vector<uint8_t> Header(bool exact) {
	Recompiler::Spirv::Profile profile;
	profile.f32_product_no_contraction = exact;
	const uint64_t             stamp = 1234;
	const uint32_t             limit = 65536;
	Recompiler::IR::Writer     writer;
	ShaderStoreFormat::IoStoreHeader(writer, stamp, profile, limit);
	return writer.bytes;
}

void CheckStoreHeader() {
	Check(Recompiler::Spirv::DefaultProfile.f32_product_no_contraction, "the default profile is not the exact one");
	Check(Header(true) != Header(false), "shader stores of both states have the same header");
	for (const bool expected: {false, true}) {
		const auto                 bytes = Header(expected);
		Recompiler::IR::Reader     reader {bytes};
		uint64_t                   stamp = 0;
		uint32_t                   limit = 0;
		Recompiler::Spirv::Profile profile;
		profile.f32_product_no_contraction = !expected;
		ShaderStoreFormat::IoStoreHeader(reader, stamp, profile, limit);
		Check(reader.ok && reader.bytes.empty() && profile.f32_product_no_contraction == expected,
		      "a store header lost the f32 product setting");
	}
}

}

int main() {
	CheckForms();
	CheckExec();
	CheckStoreHeader();
	if (g_failures != 0) return 1;
	std::printf("%s: ok\n", Name);
	return 0;
}
