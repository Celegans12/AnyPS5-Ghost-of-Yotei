#include "Wave64Programs.hpp"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvWaveSplit.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Wave64Programs;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;

constexpr const char* Name           = "kyty_shim_wave_split_codegen";
constexpr uint32_t    UserDataDwords = 8u;
constexpr uint32_t    WaveLanes      = 64u;

struct Program {
	const char*               name;
	std::span<const uint32_t> code;
	uint32_t                  threads;
	uint32_t                  lds_bytes;
	bool                      same_address_store;
};

const std::array Programs {
    Program {"lanes", LanesCode, 64, 0, false},
    Program {"reduce", ReduceCode, 1024, 0, false},
    Program {"vote", VoteCode, 64, 0, false},
    Program {"permlane", PermlaneCode, 64, 0, false},
    Program {"sdwa", SdwaCode, 64, 0, false},
    Program {"mask-scalar", MaskScalarCode, 64, 0, false},
    Program {"wave-branch", WaveBranchCode, 256, 1024, false},
    Program {"same-address", SameAddressCode, 64, 512, true},
    Program {"same-address-waves", SameAddressCode, 256, 512, true},
};

uint32_t g_failures = 0;

void Check(bool value, const std::string& what) {
	if (value) return;
	++g_failures;
	std::fprintf(stderr, "%s: FAILED: %s\n", Name, what.c_str());
}

bool Valid(const std::vector<uint32_t>& words, std::string& messages) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&, const char* message) {
		messages += message;
		messages += '\n';
	});
	spvtools::ValidatorOptions validator;
	validator.SetUniformBufferStandardLayout(true);
	return tools.Validate(words.data(), words.size(), validator);
}

struct Module {
	std::array<uint32_t, 3> local_size {};
	bool                    subgroup_id      = false;
	uint32_t                sync_functions   = 0;
	uint32_t                sync_calls       = 0;
	uint32_t                workgroup_stores = 0;
};

Module Inspect(const std::vector<uint32_t>& words) {
	Module                                 module;
	std::unordered_map<uint32_t, uint32_t> pointer_class;
	std::unordered_map<uint32_t, uint32_t> value_type;
	std::unordered_set<uint32_t>           sync_functions;
	std::vector<uint32_t>                  callees;
	uint32_t function = 0, parameters = 0;
	bool     atomic_add = false, atomic_load = false, loop = false;
	for (size_t at = 5; at < words.size();) {
		const auto count  = words[at] >> 16u;
		const auto opcode = static_cast<spv::Op>(words[at] & 0xffffu);
		if (count == 0 || at + count > words.size()) break;
		const auto word = [&](uint32_t index) { return words[at + index]; };
		switch (opcode) {
			case spv::OpExecutionMode:
				if (count == 6 && word(2) == spv::ExecutionModeLocalSize) module.local_size = {word(3), word(4), word(5)};
				break;
			case spv::OpDecorate:
				if (count == 4 && word(2) == spv::DecorationBuiltIn && word(3) == spv::BuiltInSubgroupId) {
					module.subgroup_id = true;
				}
				break;
			case spv::OpTypePointer: pointer_class[word(1)] = word(2); break;
			case spv::OpVariable:
			case spv::OpAccessChain:
			case spv::OpInBoundsAccessChain: value_type[word(2)] = word(1); break;
			case spv::OpStore:
				if (const auto type = value_type.find(word(1)); type != value_type.end()) {
					const auto storage = pointer_class.find(type->second);
					module.workgroup_stores +=
					    storage != pointer_class.end() && storage->second == spv::StorageClassWorkgroup ? 1u : 0u;
				}
				break;
			case spv::OpFunction:
				function   = word(2);
				parameters = 0;
				atomic_add = atomic_load = loop = false;
				break;
			case spv::OpFunctionParameter: ++parameters; break;
			case spv::OpAtomicIAdd: atomic_add = true; break;
			case spv::OpAtomicLoad: atomic_load = true; break;
			case spv::OpLoopMerge: loop = true; break;
			case spv::OpFunctionCall: callees.push_back(word(3)); break;
			case spv::OpFunctionEnd:
				if (parameters == 2 && atomic_add && atomic_load && loop) sync_functions.insert(function);
				break;
			default: break;
		}
		at += count;
	}
	module.sync_functions = static_cast<uint32_t>(sync_functions.size());
	for (const auto callee: callees) module.sync_calls += sync_functions.contains(callee) ? 1u : 0u;
	return module;
}

std::vector<uint32_t> Compile(const Program& program, uint64_t hash) {
	ShaderComputeInputInfo compute;
	compute.threads_num[0]     = program.threads;
	compute.threads_num[1]     = 1u;
	compute.threads_num[2]     = 1u;
	compute.lds_size_dwords    = program.lds_bytes / 4u;
	compute.wave_size          = WaveLanes;
	compute.host_subgroup_size = 32u;
	compute.thread_ids_num     = 1;
	compute.group_id[0]        = true;
	compute.workgroup_register = static_cast<int>(UserDataDwords);
	static Recompiler::Spirv::Profile profile {.maintenance8 = true, .robust_buffer_access = true};
	alignas(256) static std::array<uint32_t, 16384> output_words {};
	const auto output_address = reinterpret_cast<uintptr_t>(output_words.data());
	std::array<uint32_t, UserDataDwords> user_data {};
	user_data[4] = static_cast<uint32_t>(output_address);
	user_data[5] = static_cast<uint32_t>(output_address >> 32u) & 0xffffu;
	user_data[6] = static_cast<uint32_t>(sizeof(output_words));
	user_data[7] = (0x01016facu & ~(0x7fu << 12u)) | (20u << 12u) | (3u << 28u);
	Recompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = WaveLanes;
	options.user_data_base     = 0u;
	options.shader_hash        = hash;
	options.profile            = &profile;
	options.user_data          = user_data;
	options.input_info.compute = &compute;
	const auto source          = Recompiler::PrepareShaderSource(program.code, options);
	auto       translated      = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan            = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime       runtime {.user_data = user_data};
	Recompiler::IR::ResourceSnapshot       snapshot;
	Recompiler::IR::ResourceSpecialization specialization;
	if (!Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization)) {
		Check(false, std::string(program.name) + ": the program's resources were not materialized");
		return {};
	}
	auto        compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
	std::string messages;
	Check(Valid(compiled.spirv, messages), std::string(program.name) + ": invalid SPIR-V: " + messages);
	return std::move(compiled.spirv);
}

std::vector<uint32_t> ComputeModule(bool derivatives) {
	std::vector<uint32_t> words {spv::MagicNumber, 0x00010600u, 0u, 7u, 0u};
	const auto put = [&](spv::Op op, std::initializer_list<uint32_t> operands, const char* text = nullptr) {
		std::vector<uint32_t> packed;
		if (text != nullptr) {
			packed.assign(std::strlen(text) / 4u + 1u, 0u);
			std::memcpy(packed.data(), text, std::strlen(text));
		}
		words.push_back(static_cast<uint32_t>((operands.size() + packed.size() + 1u) << spv::WordCountShift) | op);
		words.insert(words.end(), operands);
		words.insert(words.end(), packed.begin(), packed.end());
	};
	put(spv::OpCapability, {spv::CapabilityShader});
	if (derivatives) {
		put(spv::OpCapability, {spv::CapabilityComputeDerivativeGroupQuadsKHR});
		put(spv::OpExtension, {}, "SPV_KHR_compute_shader_derivatives");
	}
	put(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
	put(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, 5u}, "main");
	put(spv::OpExecutionMode, {5u, spv::ExecutionModeLocalSize, 8u, 8u, 1u});
	if (derivatives) put(spv::OpExecutionMode, {5u, spv::ExecutionModeDerivativeGroupQuadsKHR});
	put(spv::OpTypeVoid, {1u});
	put(spv::OpTypeBool, {2u});
	put(spv::OpTypeInt, {3u, 32u, 0u});
	put(spv::OpTypeFunction, {4u, 1u});
	put(spv::OpFunction, {1u, 5u, spv::FunctionControlMaskNone, 4u});
	put(spv::OpLabel, {6u});
	put(spv::OpReturn, {});
	put(spv::OpFunctionEnd, {});
	return words;
}

std::vector<uint32_t> LoopModule(uint32_t threads, uint32_t values) {
	std::vector<uint32_t> words {spv::MagicNumber, 0x00010600u, 0u, 0u, 0u};
	uint32_t              next = 1;
	const auto put = [&](spv::Op op, std::initializer_list<uint32_t> operands, const char* text = nullptr) {
		std::vector<uint32_t> packed;
		if (text != nullptr) {
			packed.assign(std::strlen(text) / 4u + 1u, 0u);
			std::memcpy(packed.data(), text, std::strlen(text));
		}
		words.push_back(static_cast<uint32_t>((operands.size() + packed.size() + 1u) << spv::WordCountShift) | op);
		words.insert(words.end(), operands);
		words.insert(words.end(), packed.begin(), packed.end());
	};
	const auto type_void = next++, type_bool = next++, type_u32 = next++, type_function = next++, zero = next++,
	           one = next++, rounds = next++, main = next++, entry = next++, header = next++, body = next++,
	           latch = next++, merge = next++, counter = next++, counted = next++, again = next++;
	std::vector<uint32_t> carried(values), updated(values);
	for (uint32_t k = 0; k < values; k++) {
		carried[k] = next++;
		updated[k] = next++;
	}
	put(spv::OpCapability, {spv::CapabilityShader});
	put(spv::OpMemoryModel, {spv::AddressingModelLogical, spv::MemoryModelGLSL450});
	put(spv::OpEntryPoint, {spv::ExecutionModelGLCompute, main}, "main");
	put(spv::OpExecutionMode, {main, spv::ExecutionModeLocalSize, threads, 1u, 1u});
	put(spv::OpTypeVoid, {type_void});
	put(spv::OpTypeBool, {type_bool});
	put(spv::OpTypeInt, {type_u32, 32u, 0u});
	put(spv::OpTypeFunction, {type_function, type_void});
	put(spv::OpConstant, {type_u32, zero, 0u});
	put(spv::OpConstant, {type_u32, one, 1u});
	put(spv::OpConstant, {type_u32, rounds, 8u});
	put(spv::OpFunction, {type_void, main, spv::FunctionControlMaskNone, type_function});
	put(spv::OpLabel, {entry});
	put(spv::OpBranch, {header});
	put(spv::OpLabel, {header});
	put(spv::OpPhi, {type_u32, counter, zero, entry, counted, latch});
	for (uint32_t k = 0; k < values; k++) put(spv::OpPhi, {type_u32, carried[k], zero, entry, updated[k], latch});
	put(spv::OpLoopMerge, {merge, latch, spv::LoopControlMaskNone});
	put(spv::OpBranch, {body});
	put(spv::OpLabel, {body});
	for (uint32_t k = 0; k < values; k++) put(spv::OpIAdd, {type_u32, updated[k], carried[k], one});
	put(spv::OpIAdd, {type_u32, counted, counter, one});
	put(spv::OpBranch, {latch});
	put(spv::OpLabel, {latch});
	put(spv::OpULessThan, {type_bool, again, counted, rounds});
	put(spv::OpBranchConditional, {again, header, merge});
	put(spv::OpLabel, {merge});
	put(spv::OpReturn, {});
	put(spv::OpFunctionEnd, {});
	words[3] = next;
	return words;
}

void RegisterRelief() {
	struct Shape {
		uint32_t threads;
		uint32_t values;
		bool     split;
	};
	for (const auto& shape: {Shape {256u, 47u, true}, Shape {512u, 47u, true}, Shape {256u, 7u, false},
	                         Shape {1024u, 47u, false}, Shape {64u, 7u, true}}) {
		auto        module = LoopModule(shape.threads, shape.values);
		const auto  label  = std::to_string(shape.threads) + " threads, a loop carrying " + std::to_string(shape.values + 1u) +
		                     " values";
		std::string messages;
		Check(Valid(module, messages), label + ": invalid input module: " + messages);
		std::string refusal;
		const bool  split = Recompiler::Spirv::SplitWaveAcrossSubgroups(module, refusal);
		std::printf("%s: %s: %s\n", Name, label.c_str(), split ? "split" : ("kept in lane pairs: " + refusal).c_str());
		if (!shape.split) {
			Check(!split && refusal.find("without registers to gain") != std::string::npos,
			      label + ": a group of several waves stays in lane pairs unless the split gives its lanes registers they need");
			continue;
		}
		messages.clear();
		const auto inspected = Inspect(module);
		Check(split && inspected.local_size == std::array<uint32_t, 3> {shape.threads, 1u, 1u} && inspected.subgroup_id &&
		          inspected.sync_functions == (shape.threads > WaveLanes ? 1u : 0u) && Valid(module, messages),
		      label + ": the split runs one invocation per lane: " + refusal + messages);
	}
}

void ComputeDerivatives() {
	for (const bool derivatives: {false, true}) {
		auto        module = ComputeModule(derivatives);
		const auto  label  = std::string(derivatives ? "8x8 with compute derivatives" : "8x8 without compute derivatives");
		std::string messages;
		Check(Valid(module, messages), label + ": invalid input module: " + messages);
		std::string refusal;
		const bool  split = Recompiler::Spirv::SplitWaveAcrossSubgroups(module, refusal);
		std::printf("%s: %s: %s\n", Name, label.c_str(), split ? "split" : ("kept in lane pairs: " + refusal).c_str());
		if (derivatives) {
			Check(!split, label + ": derivative quads follow the 2D workgroup layout, which the split flattens");
			continue;
		}
		messages.clear();
		Check(split && Inspect(module).local_size == std::array<uint32_t, 3> {64u, 1u, 1u} && Valid(module, messages),
		      label + ": the split runs the group flat on 32-wide subgroups: " + refusal + messages);
	}
}

}

int main() {
	uint64_t hash = 0x5711c0de00u;
	for (const auto& program: Programs) {
		const auto words = Compile(program, ++hash);
		if (words.empty()) continue;
		const auto module = Inspect(words);
		const bool waves  = program.threads > WaveLanes;
		std::printf("%s: %s: local size %u, halves sync functions %u, calls %u, workgroup stores %u\n", Name, program.name,
		            module.local_size[0], module.sync_functions, module.sync_calls, module.workgroup_stores);
		if (waves && !program.same_address_store) {
			Check(module.local_size == std::array<uint32_t, 3> {program.threads / 2u, 1u, 1u} && module.sync_functions == 0u,
			      std::string(program.name) +
			          ": a group of several waves whose loops carry few values stays in lane pairs, with no halves to wait on");
			continue;
		}
		Check(module.local_size == std::array<uint32_t, 3> {program.threads, 1u, 1u} && module.subgroup_id,
		      std::string(program.name) + ": the workgroup runs one invocation per guest lane on 32-wide subgroups");
		Check(waves ? module.sync_functions == 1u && module.sync_calls != 0u : module.sync_functions == 0u,
		      std::string(program.name) +
		          (waves ? ": a group of several waves whose lanes store to one LDS word stays split, the last lane winning"
		                 : ": a one-wave workgroup synchronizes its halves with workgroup barriers"));
		if (program.same_address_store) {
			Check(module.workgroup_stores >= 2u,
			      std::string(program.name) + ": the wave's store to one LDS word goes through the exchange first");
		}
	}
	ComputeDerivatives();
	RegisterRelief();
	if (g_failures != 0) return 1;
	std::printf("%s: ok\n", Name);
	return 0;
}
