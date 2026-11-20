#include "common/assert.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/shaderCompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
using Attribute      = Recompiler::IR::TessellationAttribute;
using Opcode         = Recompiler::IR::ValueOpcode;

constexpr uint32_t SetExecAll        = 0xbefe04c1u;
constexpr uint32_t Barrier           = 0xbf8a0000u;
constexpr uint32_t WaitLds           = 0xbf8cc07fu;
constexpr uint32_t EndProgram        = 0xbf810000u;
constexpr uint32_t SetPcS6           = 0xbe802006u;
constexpr uint32_t BranchScc1ToPrior = 0xbf85fffeu;
constexpr uint32_t SwapPcS0          = 0xbe802100u;

constexpr std::array<uint32_t, 4> LaneIndex {0xd7650000u, 0x000100c1u, 0xd7660000u, 0x000200c1u};
constexpr std::array<uint32_t, 20> LocalBody {
    0xd5480004u, 0x02090502u, 0xd5480005u, 0x02090102u, 0xd5480006u, 0x02110902u, 0xd5480007u,
    0x02111102u, 0x7e080d04u, 0x7e0a0d05u, 0x7e0c0d06u, 0x7e0e0d07u, 0x10100cffu, 0x3f19999au,
    0x061010ffu, 0xbf666666u, 0x3e1008ffu, 0x3e800000u, 0xd54b0009u, 0x03c5e105u,
};
constexpr uint32_t                LocalAddress = 0x161406a0u;
constexpr std::array<uint32_t, 6> LocalWrites {0xd9340000u, 0x0000080au, 0xd8340008u,
                                               0x0000070au, 0xd8340010u, 0x0000070au};
constexpr std::array<uint32_t, 2> HullLanes {0x7da206f9u, 0x81060000u};
constexpr std::array<uint32_t, 17> HullPoints {
    0xd5480014u, 0x02151101u, 0x362402ffu, 0x000000ffu, 0x160028a0u, 0xd5430015u,
    0x040224ffu, 0x00000060u, 0xdbfc0000u, 0x02000015u, 0x162c2890u, 0xd5430016u,
    0x045a24ffu, 0x00000030u, WaitLds,     0xe0781000u, 0x02020216u,
};
constexpr std::array<uint32_t, 5> HullFirst {0x7da82881u, 0xbe94047eu, 0xb06a0060u, 0x162602f9u,
                                             0x0086066au};
constexpr std::array<uint32_t, 5> PatchReads {0xd8d80048u, 0x06000013u, 0xd8dc1008u, 0x07000013u,
                                              WaitLds};
constexpr std::array<uint32_t, 4> FactorMoves {0x7e300306u, 0x7e320306u, 0x7e340306u,
                                               0x7e360306u};
constexpr std::array<uint32_t, 11> HullStores {
    0x34142484u, 0xbefe0414u, 0xe0781000u, 0x0403180au, 0x4a162481u, 0x16161690u,
    0x4c1616ffu, 0x00008000u, 0xe0741000u, 0x0202070bu, EndProgram,
};

constexpr std::array<uint32_t, 3> RingRead {0xe0301010u, 0x02021813u, FactorMoves[1]};

struct Variant {
	uint32_t                local_write0  = LocalWrites[0];
	uint32_t                patch_read0   = PatchReads[0];
	uint32_t                patch_address = PatchReads[1];
	bool                    local_barrier = false;
	bool                    barrier_loop  = false;
	bool                    call          = false;
	bool                    implicit_handoff = false;
	std::array<uint32_t, 3> factor_head {FactorMoves[0], FactorMoves[1], FactorMoves[2]};
	std::array<uint32_t, 2> point_store {HullPoints[15], HullPoints[16]};
	std::vector<uint32_t>   first_head;
	std::vector<uint32_t>   stores;
	std::vector<uint32_t>   loop;
	std::vector<uint32_t>   factor_prefix;
};

enum class QuadFactor {
	Reference,
	Guarded,
	GuardedLoop,
	GuardedSelectArms,
	Observable,
	Count,
	RawMask,
};

Variant QuadFactors(QuadFactor factor) {
	constexpr uint32_t CompareFactor = 0x7d820c80u;
	constexpr uint32_t WholeQuadVcc  = 0xbe960a6au;
	constexpr uint32_t SelectFactor = 0xd5010006u;
	constexpr uint32_t VccFactors   = 0x01a9e8f2u;
	constexpr uint32_t WholeQuadFactors = 0x0059e8f2u;
	constexpr uint32_t InnerSelect  = 0xd501001fu;
	constexpr uint32_t GuardedValue = 0x020c3ef2u;
	constexpr uint32_t CountQuad    = 0xbe981016u;
	constexpr uint32_t CountFactor  = 0x7e0c0218u;
	switch (factor) {
		case QuadFactor::Reference:
			return {.factor_prefix = {CompareFactor, SelectFactor, VccFactors}};
		case QuadFactor::Guarded:
			return {.factor_prefix = {CompareFactor, WholeQuadVcc, InnerSelect, WholeQuadFactors,
			                          GuardedValue}};
		case QuadFactor::GuardedLoop:
			return {.factor_prefix = {0xb0180002u, 0x7d820e80u, WholeQuadVcc, InnerSelect,
			                          WholeQuadFactors, GuardedValue, 0x7e3e0280u, 0x8018c118u,
			                          0xbf078018u, 0xbf85fff7u}};
		case QuadFactor::GuardedSelectArms:
			return {.factor_prefix = {CompareFactor, WholeQuadVcc, 0xd501001au, WholeQuadFactors,
			                          0x020c34f2u}};
		case QuadFactor::Observable:
			return {.factor_prefix = {CompareFactor, WholeQuadVcc, SelectFactor, WholeQuadFactors}};
		case QuadFactor::Count:
			return {.factor_prefix = {CompareFactor, WholeQuadVcc, CountQuad, CountFactor}};
		case QuadFactor::RawMask:
			return {.factor_prefix = {CompareFactor, WholeQuadVcc, InnerSelect, WholeQuadFactors,
			                          GuardedValue, 0xbf07806au, 0x85188081u, 0x4a0c0c18u}};
	}
	return {};
}

void Append(std::vector<uint32_t>& code, std::span<const uint32_t> words) {
	code.insert(code.end(), words.begin(), words.end());
}

std::vector<uint32_t> HullPart(const Variant& variant) {
	std::vector<uint32_t> code;
	Append(code, HullLanes);
	Append(code, std::span(HullPoints).first(15));
	Append(code, variant.point_store);
	Append(code, variant.loop);
	if (variant.first_head.empty()) {
		Append(code, HullFirst);
	} else {
		Append(code, variant.first_head);
		Append(code, std::span(HullFirst).subspan(2));
	}
	code.push_back(variant.patch_read0);
	code.push_back(variant.patch_address);
	Append(code, std::span(PatchReads).subspan(2));
	Append(code, variant.factor_prefix);
	Append(code, variant.factor_head);
	code.push_back(FactorMoves[3]);
	if (variant.call) code.push_back(SwapPcS0);
	if (variant.stores.empty()) {
		Append(code, HullStores);
	} else {
		Append(code, variant.stores);
	}
	return code;
}

std::vector<uint32_t> LocalPart(const Variant& variant) {
	std::vector<uint32_t> code;
	Append(code, LocalBody);
	code.push_back(LocalAddress);
	code.push_back(variant.local_write0);
	Append(code, std::span(LocalWrites).subspan(1));
	if (variant.local_barrier) code.push_back(Barrier);
	if (!variant.implicit_handoff) code.push_back(WaitLds);
	return code;
}

std::vector<uint32_t> Merged(const Variant& variant = {}) {
	const auto local = LocalPart(variant);
	std::vector<uint32_t> code {SetExecAll};
	Append(code, LaneIndex);
	code.push_back(0x7da206f9u);
	code.push_back(0x80060000u);
	code.push_back(0xbf880000u | static_cast<uint32_t>(local.size()));
	code.insert(code.end(), local.begin(), local.end());
	code.push_back(SetExecAll);
	if (!variant.implicit_handoff) code.push_back(WaitLds);
	Append(code, LaneIndex);
	if (!variant.implicit_handoff) code.push_back(Barrier);
	if (variant.barrier_loop) code.push_back(BranchScc1ToPrior);
	const auto hull = HullPart(variant);
	code.insert(code.end(), hull.begin(), hull.end());
	return code;
}

std::vector<uint32_t> SplitFront() {
	auto code   = LocalPart({});
	code.back() = SetPcS6;
	return code;
}

std::vector<uint32_t> SplitBack() {
	std::vector<uint32_t> code {SetExecAll};
	Append(code, LaneIndex);
	const auto hull = HullPart({});
	code.insert(code.end(), hull.begin(), hull.end());
	return code;
}

void Require(bool value, const std::string& what) {
	if (value) return;
	std::fprintf(stderr, "kyty_shim_merged_hull: %s\n", what.c_str());
	std::abort();
}

std::string Refusal(const std::function<void()>& body) {
	try {
		const Common::SoftExitScope soft;
		body();
	} catch (const Common::SoftExitError& error) {
		return error.what();
	}
	return {};
}

void RequireRefusal(const std::function<void()>& body, const char* message, const char* what) {
	const auto refusal = Refusal(body);
	Require(refusal.find(message) != std::string::npos,
	        std::string(what) + ": expected \"" + message + "\", got \"" + refusal + "\"");
}

ShaderTessellationInputInfo Patches(uint32_t inputs, uint32_t outputs, bool merged) {
	return {.input_control_points  = inputs,
	        .output_control_points = outputs,
	        .domain                = 1u,
	        .partitioning          = 2u,
	        .output_topology       = 2u,
	        .merged_hull           = merged ? 1u : 0u};
}

ShaderTessellationInputInfo Analyzed(std::span<const uint32_t> local,
                                     std::span<const uint32_t> control, uint32_t inputs,
                                     uint32_t outputs, bool merged) {
	auto info = Patches(inputs, outputs, merged);
	Require(Recompiler::AnalyzeTessellationPrograms(local, control, info),
	        "the tessellation layout was not analyzed");
	return info;
}

struct Compiled {
	std::vector<uint32_t>   spirv;
	Recompiler::IR::Program program;
};

Compiled Compile(ShaderType stage, std::span<const uint32_t> code,
                 const ShaderTessellationInputInfo& tess) {
	ShaderVertexInputInfo info;
	info.logical_stage = stage;
	info.tess          = tess;
	const std::array<uint32_t, 24> users {};
	Recompiler::CompileOptions     options;
	options.stage             = stage;
	options.wave_size         = info.wave_size;
	options.user_data_base    = stage == ShaderType::TessellationControl ? 0u : 8u;
	options.shader_hash       = 0x31u;
	options.dump_ir           = false;
	options.user_data         = users;
	options.input_info.vertex = &info;
	const auto source         = Recompiler::PrepareShaderSource(code, options);
	auto       translated     = Recompiler::TranslateProgram(source.decoded, options);
	const auto plan           = Recompiler::IR::ExtractResourcePlan(translated.program);
	const Recompiler::IR::SrtRuntime           runtime {.user_data = users};
	Recompiler::IR::ResourceSnapshot           snapshot;
	Recompiler::IR::ResourceSpecialization     specialization;
	Require(Recompiler::IR::MaterializeResources(plan, runtime, snapshot, specialization),
	        "the test program's resources were not materialized");
	auto result = Recompiler::CompileProgram(std::move(translated), options, specialization, 0u);
	return {std::move(result.spirv), std::move(result.program)};
}

void RequireValid(const std::vector<uint32_t>& spirv, const char* what) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*, const spv_position_t&,
	                                     const char* message) {
		messages += message;
		messages += '\n';
	});
	Require(tools.Validate(spirv), std::string(what) + " is not valid SPIR-V: " + messages);
}

struct Module {
	struct Variable {
		uint32_t storage  = 0;
		uint32_t location = UINT32_MAX;
		bool     patch    = false;
		uint32_t loads    = 0;
		uint32_t stores   = 0;
	};
	std::map<uint32_t, Variable>               variables;
	std::vector<uint32_t>                      capabilities;
	std::vector<std::pair<spv::Op, uint32_t>> body;
	uint32_t                                   execution_model = UINT32_MAX;
	uint32_t                                   output_vertices = 0;

	[[nodiscard]] std::string Describe() const {
		std::string text = " (module:";
		for (const auto& [id, variable]: variables) {
			text += " [" + std::string(variable.storage == spv::StorageClassInput ? "in" : "out") +
			        (variable.location == UINT32_MAX ? std::string(" builtin")
			                                         : " location " + std::to_string(variable.location)) +
			        (variable.patch ? " patch" : "") + " loads " + std::to_string(variable.loads) +
			        " stores " + std::to_string(variable.stores) + "]";
		}
		text += " capabilities";
		for (const auto capability: capabilities) text += " " + std::to_string(capability);
		return text + ")";
	}
};

Module Parse(const std::vector<uint32_t>& words) {
	Module                       module;
	std::map<uint32_t, uint32_t> chains;
	const auto root = [&](uint32_t id) {
		const auto chain = chains.find(id);
		return chain != chains.end() ? chain->second : id;
	};
	for (size_t at = 5; at < words.size();) {
		const auto count = words[at] >> 16u;
		const auto op    = static_cast<spv::Op>(words[at] & 0xffffu);
		Require(count != 0u && at + count <= words.size(), "the SPIR-V module is malformed");
		const auto* w = &words[at];
		switch (op) {
			case spv::OpCapability: module.capabilities.push_back(w[1]); break;
			case spv::OpEntryPoint: module.execution_model = w[1]; break;
			case spv::OpExecutionMode:
				if (w[2] == spv::ExecutionModeOutputVertices) module.output_vertices = w[3];
				break;
			case spv::OpDecorate:
				if (w[2] == spv::DecorationLocation) module.variables[w[1]].location = w[3];
				if (w[2] == spv::DecorationPatch) module.variables[w[1]].patch = true;
				break;
			case spv::OpVariable:
				if (w[3] == spv::StorageClassInput || w[3] == spv::StorageClassOutput) {
					module.variables[w[2]].storage = w[3];
				}
				break;
			case spv::OpAccessChain: chains[w[2]] = root(w[3]); break;
			case spv::OpLoad:
				if (const auto found = module.variables.find(root(w[3]));
				    found != module.variables.end()) {
					found->second.loads++;
					module.body.emplace_back(op, found->first);
				}
				break;
			case spv::OpStore:
				if (const auto found = module.variables.find(root(w[1]));
				    found != module.variables.end()) {
					found->second.stores++;
					module.body.emplace_back(op, found->first);
				}
				break;
			case spv::OpControlBarrier:
			case spv::OpLoopMerge: module.body.emplace_back(op, 0u); break;
			default: break;
		}
		at += count;
	}
	return module;
}

bool Has(const Module& module, uint32_t capability) {
	return std::ranges::find(module.capabilities, capability) != module.capabilities.end();
}

struct Found {
	uint32_t         id = 0;
	Module::Variable variable;

	[[nodiscard]] bool Is(uint32_t loads, uint32_t stores) const {
		return id != 0u && variable.loads == loads && variable.stores == stores;
	}
};

Found Find(const Module& module, uint32_t storage, uint32_t location, bool patch) {
	for (const auto& [id, variable]: module.variables) {
		if (variable.storage == storage && variable.location == location &&
		    variable.patch == patch)
			return {id, variable};
	}
	return {};
}

struct Access {
	bool     write;
	uint64_t coefficient;
	uint64_t constant;
	bool     affine;
};

bool Affine(Recompiler::IR::Value value, uint64_t& coefficient, uint64_t& constant) {
	using namespace Recompiler::IR;
	value = value.Resolve();
	if (value.IsImmediate()) {
		coefficient = 0u;
		constant    = value.U32();
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return false;
	uint64_t lc = 0, lk = 0, rc = 0, rk = 0;
	switch (inst->GetOpcode()) {
		case Opcode::LaneId:
			coefficient = 1u;
			constant    = 0u;
			return true;
		case Opcode::GetBuiltin:
			coefficient = 1u;
			constant    = 0u;
			return inst->Arg(0).U32() == static_cast<uint32_t>(StageInputKind::InvocationId);
		case Opcode::IAdd32:
			if (!Affine(inst->Arg(0), lc, lk) || !Affine(inst->Arg(1), rc, rk)) return false;
			coefficient = lc + rc;
			constant    = lk + rk;
			return true;
		case Opcode::IMul32:
			if (!Affine(inst->Arg(0), lc, lk) || !Affine(inst->Arg(1), rc, rk)) return false;
			if (lc != 0u && rc != 0u) return false;
			coefficient = lc * rk + rc * lk;
			constant    = lk * rk;
			return true;
		case Opcode::ShiftLeftLogical32:
			if (!Affine(inst->Arg(0), lc, lk) || !Affine(inst->Arg(1), rc, rk) || rc != 0u)
				return false;
			coefficient = lc << rk;
			constant    = lk << rk;
			return true;
		case Opcode::BitFieldUExtract:
			if (!Affine(inst->Arg(0), lc, lk) || !Affine(inst->Arg(1), rc, rk) || rc != 0u ||
			    (lc | lk) % (uint64_t {1u} << rk) != 0u)
				return false;
			coefficient = lc >> rk;
			constant    = lk >> rk;
			return true;
		default: return false;
	}
}

std::vector<Access> SharedAccesses(const Recompiler::IR::Program& program, uint32_t& lds) {
	std::vector<Access> accesses;
	lds = 0;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto opcode = inst.GetOpcode();
			if (Recompiler::IR::SharedAccessOf(opcode) != Recompiler::IR::SharedAccess::None) lds++;
			const bool write = opcode == Opcode::SetTessellationAttribute;
			if ((!write && opcode != Opcode::GetTessellationAttribute) ||
			    inst.Arg(0).U32() != static_cast<uint32_t>(Attribute::ControlShared))
				continue;
			Access access {.write = write, .coefficient = 0u, .constant = 0u, .affine = false};
			access.affine = Affine(inst.Arg(1), access.coefficient, access.constant);
			accesses.push_back(access);
		}
	}
	return accesses;
}

void RequireMergedForm() {
	const auto code = Merged();
	const auto tess = Analyzed(code, code, 3u, 3u, true);
	Require(tess.ls_stride == 32u && tess.hs_stride == 16u && tess.patch_base == 0x7ff0u &&
	            tess.patch_size == 8u && tess.merged_hull == 1u,
	        "the merged hull layout is not LDS stride 32, control stride 16 and 8 patch bytes "
	        "at 0x7ff0");

	const auto local = Compile(ShaderType::Local, code, tess);
	RequireValid(local.spirv, "the local stage of a merged hull program");
	const auto vertex = Parse(local.spirv);
	Require(vertex.execution_model == spv::ExecutionModelVertex &&
	            Find(vertex, spv::StorageClassOutput, 0u, false).Is(0u, 2u) &&
	            vertex.variables.size() == 3u,
	        "the local stage of a merged hull program is not an empty program that passes the "
	        "vertex and instance index" + vertex.Describe());

	const auto hull = Compile(ShaderType::TessellationControl, code, tess);
	RequireValid(hull.spirv, "the control stage of a merged hull program");
	const auto control = Parse(hull.spirv);
	Require(control.execution_model == spv::ExecutionModelTessellationControl &&
	            control.output_vertices == 3u,
	        "a merged hull program did not compile to a control stage of 3 output vertices");
	Require(!Has(control, spv::CapabilityGroupNonUniform) &&
	            !Has(control, spv::CapabilityGroupNonUniformBallot),
	        "the control stage of a merged hull program uses subgroup operations instead of one "
	        "guest lane per invocation");
	Require(Find(control, spv::StorageClassInput, 0u, false).Is(2u, 0u),
	        "the control stage does not read the vertex and instance index of its control point" +
	            control.Describe());
	Require(Find(control, spv::StorageClassOutput, 0u, false).Is(0u, 4u),
	        "the control stage does not store its control point at location 0" +
	            control.Describe());
	Require(Find(control, spv::StorageClassOutput, 1u, true).Is(0u, 2u),
	        "the control stage does not store its patch constants at location 1" +
	            control.Describe());
	const auto shared = Find(control, spv::StorageClassOutput, 2u, false);
	Require(shared.Is(7u, 4u),
	        "the LDS of a merged hull program is not a per-control-point record at location 2 "
	        "that the local part writes (4 dwords) and the hull part reads back (7 dwords)" +
	            control.Describe());

	uint32_t   lds      = 0;
	const auto accesses = SharedAccesses(hull.program, lds);
	Require(lds == 0u && accesses.size() == 11u, "LDS accesses were left in the control stage");
	std::vector<uint64_t> written, own, fixed;
	for (const auto& access: accesses) {
		Require(access.affine, "an LDS record address is not affine in the invocation");
		if (access.write) {
			Require(access.coefficient == 32u, "an LDS write is not to the invocation's record");
			written.push_back(access.constant);
		} else if (access.coefficient == 32u) {
			own.push_back(access.constant);
		} else {
			Require(access.coefficient == 0u, "an LDS read is neither own nor patch relative");
			fixed.push_back(access.constant);
		}
	}
	std::ranges::sort(written);
	std::ranges::sort(own);
	std::ranges::sort(fixed);
	Require(written == std::vector<uint64_t> {0u, 4u, 8u, 16u},
	        "the local part does not write dwords 0, 1, 2 and 4 of its control point's record");
	Require(own == std::vector<uint64_t> {0u, 4u, 8u, 12u},
	        "the hull part does not read back the first four dwords of its control point's "
	        "record");
	Require(fixed == std::vector<uint64_t> {32u, 64u, 72u},
	        "the hull part does not read dword 0 of control points 1 and 2 and dword 2 of "
	        "control point 2 of its patch");

	bool passed = false, ordered = true;
	uint32_t barriers = 0;
	for (const auto& [op, variable]: control.body) {
		if (op == spv::OpControlBarrier) {
			passed = true;
			barriers++;
		} else if (variable == shared.id) {
			ordered = ordered && passed == (op == spv::OpLoad);
		}
	}
	Require(barriers == 1u && ordered,
	        "the control stage does not write its LDS records before the barrier and read them "
	        "after it");
}

void RequireWaveLocalLdsOrder() {
	const auto compile = [](const Variant& variant, uint32_t points) {
		const auto code = Merged(variant);
		return Compile(ShaderType::TessellationControl, code, Analyzed(code, code, points, points, true));
	};
	const auto implicit = compile({.implicit_handoff = true}, 4u);
	RequireValid(implicit.spirv, "the control stage with an implicit wave-local LDS handoff");
	const auto control = Parse(implicit.spirv);
	Require(control.output_vertices == 4u, "the wave-local hull lost its fourth control point");
	const auto shared = Find(control, spv::StorageClassOutput, 2u, false);
	uint32_t barriers = 0;
	Require(shared.Is(7u, 4u), "the wave-local hull lost LDS traffic");
	uint32_t peer_loads = 0;
	for (const auto& [op, variable]: control.body) {
		if (op == spv::OpControlBarrier) {
			barriers++;
		} else if (variable == shared.id && op == spv::OpStore) {
			Require(barriers == 0u, "a wave-local LDS record is written after the host barrier");
		} else if (variable == shared.id && op == spv::OpLoad && barriers != 0u) {
			peer_loads++;
		}
	}
	Require(peer_loads >= 3u && barriers == 1u, "the wave-local handoff does not order the peer LDS reads");

	RequireRefusal([&] { static_cast<void>(compile({.implicit_handoff = true}, 3u)); },
	               "no s_barrier has every LDS write before it and every such read after it",
	               "a barrierless hull whose patches can span waves");
	const Variant late_write {.implicit_handoff = true,
	                          .factor_head = {0xd8340000u, 0x00000215u, FactorMoves[2]}};
	const auto layout_code = Merged();
	const auto layout = Analyzed(layout_code, layout_code, 4u, 4u, true);
	RequireRefusal([&] {
	                   static_cast<void>(Compile(ShaderType::TessellationControl, Merged(late_write),
	                                             layout));
	               },
	               "no s_barrier has every LDS write before it and every such read after it",
	               "a wave-local hull that writes LDS after a cross-control-point read");
}

void RequireQuadFactors() {
	const auto code = Merged();
	const auto tess = Analyzed(code, code, 3u, 3u, true);
	const auto compile = [&](QuadFactor factor) {
		return Compile(ShaderType::TessellationControl, Merged(QuadFactors(factor)), tess);
	};
	constexpr auto Guarded = "a merged hull whose own predicate masks a quad-helper value";
	const auto lowered = compile(QuadFactor::Guarded);
	RequireValid(lowered.spirv, Guarded);
	const auto control = Parse(lowered.spirv);
	Require(control.execution_model == spv::ExecutionModelTessellationControl &&
	            !Has(control, spv::CapabilityGroupNonUniform),
	        "the guarded whole-quad factor was not lowered to a lane-local tessellation stage");
	const auto loop = compile(QuadFactor::GuardedLoop);
	RequireValid(loop.spirv, "a fixed loop whose own predicate masks a quad-helper value");
	const auto loop_control = Parse(loop.spirv);
	Require(std::ranges::find(loop_control.body, std::pair {spv::OpLoopMerge, 0u}) !=
	            loop_control.body.end(),
	        "the guarded whole-quad factor did not reach a loop");
	RequireValid(compile(QuadFactor::GuardedSelectArms).spirv,
	             "a merged hull masking the same quad-helper value with two selects");
	RequireRefusal([&] { static_cast<void>(compile(QuadFactor::Observable)); },
	               "uses a lane mask other than its own lane's bit",
	               "a merged hull storing a factor that a quad-helper lane can change");
	RequireRefusal([&] { static_cast<void>(compile(QuadFactor::Count)); },
	               "uses a lane mask other than its own lane's bit",
	               "a merged hull counting the bits of a whole-quad mask");
	RequireRefusal([&] { static_cast<void>(compile(QuadFactor::RawMask)); },
	               "uses a lane mask other than its own lane's bit",
	               "a merged hull testing whether any lane voted in a whole-quad source mask");
}

void RequirePatchConstantRoutes() {
	std::vector<uint32_t> shifted(HullStores.begin(), HullStores.end());
	shifted[5] = 0x34161684u;
	const std::vector<uint32_t> narrowed {0x34142484u, 0xe0781000u, 0x0403180au, 0x4a162481u,
	                                      0x34161684u, 0x4c1616ffu, 0x00008000u, 0xe0741000u,
	                                      0x0202070bu, 0xbefe0414u, EndProgram};
	const std::array<std::pair<const char*, Variant>, 3> routes {{
	    {"a patch constant address built with v_lshlrev_b32", {.stores = shifted}},
	    {"patch constants stored under s_and_saveexec_b64",
	     {.first_head = {0x7d882881u, 0xbe94246au}, .stores = narrowed}},
	    {"patch constants stored under s_and_saveexec_b64 and s_cbranch_execz",
	     {.first_head = {0x7d882881u, 0xbe94246au, 0xbf880015u}, .stores = narrowed}},
	}};
	for (const auto& [what, variant]: routes) {
		const auto code = Merged(variant);
		const auto tess = Analyzed(code, code, 3u, 3u, true);
		Require(tess.patch_base == 0x7ff0u && tess.patch_size == 8u,
		        std::string("the layout lost the patch constants of ") + what);
		const auto hull = Compile(ShaderType::TessellationControl, code, tess);
		RequireValid(hull.spirv, what);
		const auto control = Parse(hull.spirv);
		Require(Find(control, spv::StorageClassOutput, 0u, false).Is(0u, 4u) &&
		            Find(control, spv::StorageClassOutput, 1u, true).Is(0u, 2u) &&
		            Find(control, spv::StorageClassOutput, 2u, false).Is(7u, 4u),
		        std::string("a merged hull program with ") + what +
		            " did not store 4 control point dwords and 2 patch constant dwords" +
		            control.Describe());
	}
}

void RequireRingReadOrder() {
	const auto code = Merged({.factor_head = RingRead});
	const auto tess = Analyzed(Merged(), Merged(), 3u, 3u, true);
	const auto hull = Compile(ShaderType::TessellationControl, code, tess);
	RequireValid(hull.spirv, "the control stage of a merged hull program that reads a control point");
	const auto control = Parse(hull.spirv);
	const auto points  = Find(control, spv::StorageClassOutput, 0u, false);
	Require(points.Is(1u, 4u),
	        "the control stage does not store its control point and read one dword of another" +
	            control.Describe());
	uint32_t barriers = 0, stored = 0, loaded = 0;
	for (const auto& [op, variable]: control.body) {
		if (op == spv::OpControlBarrier) {
			barriers++;
		} else if (variable == points.id) {
			(op == spv::OpStore ? stored : loaded) |= 1u << barriers;
		}
	}
	Require(barriers == 2u && stored == 2u && loaded == 4u,
	        "the control stage does not put a barrier of its own between its control point "
	        "stores and its read of another control point's record (barriers " +
	            std::to_string(barriers) + ", stores " + std::to_string(stored) + ", loads " +
	            std::to_string(loaded) + ")");

	const auto plain = Parse(Compile(ShaderType::TessellationControl, Merged(), tess).spirv);
	Require(std::ranges::count(plain.body, std::pair {spv::OpControlBarrier, 0u}) == 1,
	        "a merged hull program that reads no other control point got a second barrier");
}

void RequireLimits() {
	const auto code = Merged();
	const auto tess = Analyzed(code, code, 3u, 3u, true);
	const auto hull = [](const Variant& variant, uint32_t inputs = 3u, uint32_t outputs = 3u) {
		return [=] {
			const auto refused = Merged(variant);
			auto       layout  = Analyzed(Merged(), Merged(), 3u, 3u, true);
			layout.input_control_points  = inputs;
			layout.output_control_points = outputs;
			static_cast<void>(Compile(ShaderType::TessellationControl, refused, layout));
		};
	};
	RequireRefusal(hull({}, 4u, 3u), "more input (4) than output (3) control points",
	               "a merged hull program with more input than output control points");
	RequireRefusal(
	    [&] {
		    auto info = Patches(4u, 3u, true);
		    static_cast<void>(Recompiler::AnalyzeTessellationPrograms(code, code, info));
	    },
	    "more input (4) than output (3) control points",
	    "the layout of a merged hull program with more input than output control points");
	RequireRefusal(hull({.local_barrier = true}), "s_barrier that some invocations do not reach",
	               "a merged hull program with an s_barrier in its local part");
	RequireRefusal(hull({.barrier_loop = true}), "s_barrier inside a loop",
	               "a merged hull program with an s_barrier in a loop");
	RequireRefusal(hull({.factor_head = {0xd7600010u, 0x02010106u, 0x7e300210u}}),
	               "reads or writes another lane's register",
	               "a merged hull program with v_readlane_b32");
	RequireRefusal(hull({.factor_head = {0xd8d48055u, 0x18000006u, FactorMoves[1]}}),
	               "reads or writes another lane's register",
	               "a merged hull program with ds_swizzle_b32");
	RequireRefusal(hull({.local_write0 = 0xd9340020u}),
	               "writes LDS outside its own control point's record",
	               "a merged hull program writing another control point's record");
	RequireRefusal(hull({.patch_read0 = 0xd8d80060u}),
	               "reads LDS outside its patch's control point records",
	               "a merged hull program reading past its patch");
	RequireRefusal(hull({}, 3u, 4u), "reads LDS outside its patch's control point records",
	               "a merged hull program whose fourth output reads a record no input wrote");
	RequireRefusal(hull({.patch_address = 0x06000002u}),
	               "through an address that is not a constant offset in a control point record",
	               "a merged hull program reading LDS at an address it loaded");
	RequireRefusal(hull({.point_store = {HullPoints[15], 0x02020202u}}),
	               "through an address that is neither a constant nor an offset in its own control point's record",
	               "a merged hull program storing a control point at an address it loaded");
	RequireRefusal(hull({.point_store = {0xe0781008u, HullPoints[16]}}),
	               "stores to the off-chip ring outside its own control point's record",
	               "a merged hull program storing past its control point's record");
	RequireRefusal(hull({.loop = {0x7e1402c0u, 0xb01c0002u, 0x7d88281cu, 0xbe98246au, 0xbe9a047eu,
	                              0xd8d80000u, 0x0600000au, 0xbefe0418u, 0x7e140280u, 0xbefe041au,
	                              0x7e1402c0u, 0xbefe0418u, 0x801cc11cu, 0xbf07801cu, 0xbf85fff3u}}),
	               "through an address that is not a constant offset in a control point record",
	               "a merged hull program reading LDS in a loop at an address the loop rewrites "
	               "under another exec mask");
	RequireRefusal(hull({.factor_head = {0xbe96107eu, 0x7e300216u, FactorMoves[1]}}),
	               "uses a lane mask other than its own lane's bit",
	               "a merged hull program counting the lanes of exec with s_bcnt1_i32_b64");
	std::vector<uint32_t> inside(HullStores.begin(), HullStores.end());
	inside[7] = 0x00000020u;
	RequireRefusal(hull({.stores = inside}),
	               "stores a control point's record at a constant off-chip address",
	               "a merged hull program storing at a constant address inside its control points");
	RequireRefusal(hull({.first_head = {0xbf800000u, 0xbe94047eu}}),
	               "is not written by control point 0 alone",
	               "a merged hull program storing its patch outputs from every invocation");
	RequireRefusal(hull({.factor_head = {0xe0301000u, 0x02021815u, FactorMoves[1]}}),
	               "reads off-chip memory outside its patch's control points",
	               "a merged hull program reading the ring at twice its control point stride");
	RequireRefusal(hull({.local_write0 = 0xd9340002u}),
	               "uses an unaligned tessellation ring or LDS address",
	               "a merged hull program writing LDS at an odd offset");
	RequireRefusal(hull({.point_store = {0xe0781002u, HullPoints[16]}}),
	               "uses an unaligned tessellation ring or LDS address",
	               "a merged hull program storing a control point at an odd offset");
	RequireRefusal(hull({.loop = {0xd8340000u, 0x00000215u}}),
	               "no s_barrier has every LDS write before it and every such read after it",
	               "a merged hull program writing its LDS record again after the s_barrier");
	auto late = std::vector<uint32_t>(HullStores.begin(), HullStores.end() - 1);
	Append(late, std::array {HullPoints[15], HullPoints[16], EndProgram});
	RequireRefusal(hull({.factor_head = RingRead, .stores = late}),
	               "comes after the last block every invocation reaches before the read",
	               "a merged hull program storing a control point after another one was read");
	RequireRefusal(hull({.call = true}), "calls a subroutine is unsupported",
	               "a merged hull program with s_swappc_b64");

	const Recompiler::TessellationLimits device {.per_vertex_input_components  = 64u,
	                                             .per_vertex_output_components = 64u,
	                                             .per_patch_output_components  = 120u,
	                                             .total_output_components      = 2048u};
	Require(Refusal([&] { Recompiler::RequireMergedHullLimits(tess, device); }).empty(),
	        "the merged hull program was refused within Vulkan's minimum tessellation limits");
	auto wide      = tess;
	wide.ls_stride = 16u * 16u;
	RequireRefusal([&] { Recompiler::RequireMergedHullLimits(wide, device); },
	               "exceed the device's tessellation limits (LDS stride 256",
	               "an LDS stride over the per-vertex output limit");
	auto narrow                        = device;
	narrow.total_output_components     = 39u;
	RequireRefusal([&] { Recompiler::RequireMergedHullLimits(tess, narrow); },
	               "exceed the device's tessellation limits",
	               "outputs over the total output limit");
	narrow                             = device;
	narrow.per_patch_output_components = 3u;
	RequireRefusal([&] { Recompiler::RequireMergedHullLimits(tess, narrow); },
	               "exceed the device's tessellation limits",
	               "patch constants over the per-patch output limit");
	narrow                             = device;
	narrow.per_vertex_input_components = 3u;
	RequireRefusal([&] { Recompiler::RequireMergedHullLimits(tess, narrow); },
	               "exceed the device's tessellation limits",
	               "a control point record over the per-vertex input limit");
}

void RequireSplitForm() {
	const auto front = SplitFront();
	const auto back  = SplitBack();
	const auto tess  = Analyzed(front, back, 3u, 3u, false);
	Require(tess.ls_stride == 32u && tess.hs_stride == 16u && tess.patch_base == 0x7ff0u &&
	            tess.patch_size == 8u && tess.merged_hull == 0u,
	        "the split hull layout changed");
	const auto local = Compile(ShaderType::Local, front, tess);
	RequireValid(local.spirv, "the local stage of a split hull program");
	const auto vertex = Parse(local.spirv);
	Require(Find(vertex, spv::StorageClassOutput, 0u, false).Is(0u, 4u),
	        "the local stage of a split hull program does not write its LDS as vertex outputs" +
	            vertex.Describe());
	const auto hull = Compile(ShaderType::TessellationControl, back, tess);
	RequireValid(hull.spirv, "the control stage of a split hull program");
	const auto control = Parse(hull.spirv);
	Require(Find(control, spv::StorageClassInput, 0u, false).Is(7u, 0u) &&
	            Find(control, spv::StorageClassOutput, 2u, false).id == 0u,
	        "the control stage of a split hull program does not read LDS as vertex inputs" +
	            control.Describe());
	uint32_t lds = 0;
	Require(SharedAccesses(hull.program, lds).empty() && lds == 0u,
	        "a split hull program got per-control-point shared records");
}

alignas(256) std::array<uint32_t, 128> g_merged {};
alignas(256) std::array<uint32_t, 64>  g_front {};
alignas(256) std::array<uint32_t, 128> g_back {};
alignas(256) std::array<uint32_t, 4>   g_domain {EndProgram};

uint64_t Register(std::span<uint32_t> storage, std::span<const uint32_t> code,
                  Prospero::ShaderBinaryType type, ShaderUserData* user_data) {
	Require(code.size() <= storage.size(), "a test program does not fit its storage");
	std::ranges::copy(code, storage.begin());
	const auto address = reinterpret_cast<uint64_t>(storage.data());
	ShaderMapUserData(address, {.type            = type,
	                            .user_data       = user_data,
	                            .code_size_bytes = static_cast<uint32_t>(code.size() * 4u)});
	return address;
}

void RequireDetection() {
	ShaderInit();
	ShaderUserData users {};
	const auto merged_code = Merged();
	const auto merged = Register(g_merged, merged_code, Prospero::ShaderBinaryType::kHs, &users);
	const auto front  = Register(g_front, SplitFront(), Prospero::ShaderBinaryType::kHsFront, &users);
	const auto back   = Register(g_back, SplitBack(), Prospero::ShaderBinaryType::kHsBack, &users);
	const auto domain = Register(g_domain, std::span(g_domain).first(1),
	                             Prospero::ShaderBinaryType::kGs, &users);
	HW::Context context;
	constexpr uint32_t HullStageEnabled = 1u << 2u;
	constexpr uint32_t ControlPoints    = 3u;
	constexpr uint32_t TriangleDomain   = 1u;
	constexpr uint32_t FractionalOdd    = 2u;
	constexpr uint32_t ClockwiseOutput  = 2u;
	context.SetShaderStages(HullStageEnabled);
	context.SetLsHsConfig((ControlPoints << 8u) | (ControlPoints << 14u));
	context.SetTfParam(TriangleDomain | (FractionalOdd << 2u) | (ClockwiseOutput << 5u));

	HW::VertexShaderInfo regs;
	regs.ls_regs.data_addr        = merged;
	regs.es_regs.data_addr        = domain;
	regs.hs_regs.rsrc2.user_sgpr  = 4u;
	regs.hs_user_sgpr.value[2]    = 0x1234u;
	std::array<ShaderVertexInputInfo, 3> info;
	const auto params = PrepareTessellationPrograms(regs, context, info);
	Require(info[0].tess.merged_hull == 1u && info[1].tess.merged_hull == 1u &&
	            info[2].tess.merged_hull == 1u && info[0].tess.input_control_points == 3u &&
	            info[0].tess.output_control_points == 3u,
	        "a registered type 3 program in the LS slot with no HS program was not prepared as "
	        "a merged hull");
	Require(params[0].code.data() == g_merged.data() && params[1].code.data() == g_merged.data() &&
	            params[1].code.size() == merged_code.size() && params[0].hash == params[1].hash,
	        "the local and control stages of a merged hull do not refer to the one program");
	Require(params[1].user_data_count == 12u && params[1].user_data[10] == 0x1234u &&
	            params[1].user_data[0] == 0u && params[1].user_data[1] == 0u &&
	            params[0].user_data_count == 4u && params[0].user_data[2] == 0x1234u,
	        "the stages of a merged hull lost the hull user data layout");
	Require(info[0].logical_stage == ShaderType::Local &&
	            info[1].logical_stage == ShaderType::TessellationControl &&
	            info[0].resources_num == 0 && !info[0].has_dynamic_vertex_fetch,
	        "the local stage of a merged hull was given vertex inputs");

	std::vector<uint32_t> merged_key, split_key;
	BuildStageStaticKey(info[1], merged_key);
	auto same_addresses             = info[1];
	same_addresses.tess.merged_hull = 0u;
	BuildStageStaticKey(same_addresses, split_key);
	Require(merged_key != split_key && merged_key.size() == split_key.size(),
	        "a merged and a split hull stage of the same program share a stage key, or the "
	        "merged flag changed the words of the key");

	GraphicsPipelineKey merged_pipeline {}, split_pipeline {};
	const HW::UserConfig user_config;
	merged_pipeline.static_params =
	    LibraryStaticParameters(context, user_config, true, info[1].tess, false, false);
	split_pipeline.static_params =
	    LibraryStaticParameters(context, user_config, true, same_addresses.tess, false, false);
	Require(merged_pipeline.static_params.merged_hull == 1u &&
	            split_pipeline.static_params.merged_hull == 0u &&
	            merged_pipeline.static_params.patch_control_points == 3u &&
	            !(merged_pipeline == split_pipeline),
	        "a merged and a split hull of the same stages share a pipeline key");

	regs.hs_regs.user_data_addr = 0x7654321000ull;
	const auto pointed = PrepareTessellationPrograms(regs, context, info);
	Require(pointed[1].user_data[0] == 0x54321000u && pointed[1].user_data[1] == 0x76u,
	        "a merged hull draw did not pass its hull user data address");

	auto bound              = regs;
	bound.hs_regs.data_addr = back;
	RequireRefusal(
	    [&] { static_cast<void>(PrepareTessellationPrograms(bound, context, info)); },
	    "is bound with a hull program", "a merged hull program with a program in the HS slot");

	HW::VertexShaderInfo split;
	split.ls_regs.data_addr      = front;
	split.hs_regs.data_addr      = back;
	split.es_regs.data_addr      = domain;
	split.hs_regs.user_data_addr = 0x7654321000ull;
	std::array<ShaderVertexInputInfo, 3> split_info;
	const auto halves = PrepareTessellationPrograms(split, context, split_info);
	Require(split_info[0].tess.merged_hull == 0u && halves[0].code.data() == g_front.data() &&
	            halves[1].code.data() == g_back.data() && halves[0].hash != halves[1].hash,
	        "split hull halves were not prepared as before");
	split.hs_regs.data_addr = 0u;
	RequireRefusal(
	    [&] { static_cast<void>(PrepareTessellationPrograms(split, context, split_info)); },
	    "ShaderGetInputInfoHS(): shader=0x0000000000000000 is missing from ShaderMap",
	    "a split front half without its back half");
}

constexpr std::array<uint32_t, 27> Domain {
    0x7e140280u, 0xe0381000u, 0x04020c0au, 0x7e140290u, 0xe0381000u, 0x0402100au, 0x7e1402a0u,
    0xe0381000u, 0x0402140au, 0xbf8c3f70u, 0x06120d05u, 0x081212f2u, 0x10000b0cu, 0xd54b0000u,
    0x04020d10u, 0xd54b0000u, 0x04021314u, 0x10020b0du, 0xd54b0001u, 0x04060d11u, 0xd54b0001u,
    0x04061315u, 0x7e040280u, 0x7e0602f2u, 0xf80008cfu, 0x03020100u, EndProgram,
};

constexpr std::array<uint32_t, 72> Fill {
    0x07230203u, 0x00010300u, 0x0008000bu, 0x0000000cu, 0x00000000u, 0x00020011u, 0x00000001u,
    0x0006000bu, 0x00000001u, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u, 0x0003000eu,
    0x00000000u, 0x00000001u, 0x0006000fu, 0x00000004u, 0x00000004u, 0x6e69616du, 0x00000000u,
    0x00000009u, 0x00030010u, 0x00000004u, 0x00000007u, 0x00040047u, 0x00000009u, 0x0000001eu,
    0x00000000u, 0x00020013u, 0x00000002u, 0x00030021u, 0x00000003u, 0x00000002u, 0x00030016u,
    0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u, 0x00000004u, 0x00040020u,
    0x00000008u, 0x00000003u, 0x00000007u, 0x0004003bu, 0x00000008u, 0x00000009u, 0x00000003u,
    0x0004002bu, 0x00000006u, 0x0000000au, 0x3f800000u, 0x0007002cu, 0x00000007u, 0x0000000bu,
    0x0000000au, 0x0000000au, 0x0000000au, 0x0000000au, 0x00050036u, 0x00000002u, 0x00000004u,
    0x00000000u, 0x00000003u, 0x000200f8u, 0x00000005u, 0x0003003eu, 0x00000009u, 0x0000000bu,
    0x000100fdu, 0x00010038u,
};

constexpr uint32_t                Side = 64u;
constexpr std::array<uint16_t, 6> PatchIndices {0x100u, 0x130u, 0x112u, 0x122u, 0x132u, 0x131u};
constexpr std::array<uint16_t, 8> WaveLocalPatchIndices {0x100u, 0x130u, 0x112u, 0x112u,
                                                        0x122u, 0x132u, 0x131u, 0x131u};

template <class T>
T Checked(vk::ResultValue<T> result, const char* what) {
	Require(result.result == vk::Result::eSuccess,
	        std::string(what) + " failed: " + vk::to_string(result.result));
	return std::move(result.value);
}

void Checked(vk::Result result, const char* what) {
	Require(result == vk::Result::eSuccess, std::string(what) + " failed: " + vk::to_string(result));
}

int NoDevice(const char* why) {
	const char* required = std::getenv("ANYPS5_REQUIRE_VULKAN");
	Require(required == nullptr || std::strcmp(required, "1") != 0,
	        std::string("ANYPS5_REQUIRE_VULKAN=1 and ") + why);
	std::printf("kyty_shim_merged_hull_draw: skipped, %s\n", why);
	return 77;
}

int DrawMergedHull() {
	using Stages      = std::array<std::vector<uint32_t>, 4>;
	const auto code   = Merged();
	const auto tess   = Analyzed(code, code, 3u, 3u, true);
	const auto front  = SplitFront();
	const auto back   = SplitBack();
	const auto halves = Analyzed(front, back, 3u, 3u, false);
	const Stages merged_stages {
	    Compile(ShaderType::Local, code, tess).spirv,
	    Compile(ShaderType::TessellationControl, code, tess).spirv,
	    Compile(ShaderType::TessellationEvaluation, Domain, tess).spirv,
	    {Fill.begin(), Fill.end()},
	};
	const Stages split_stages {
	    Compile(ShaderType::Local, front, halves).spirv,
	    Compile(ShaderType::TessellationControl, back, halves).spirv,
	    Compile(ShaderType::TessellationEvaluation, Domain, halves).spirv,
	    {Fill.begin(), Fill.end()},
	};
	const auto variant_stages = [](const Variant& variant, uint32_t points) -> Stages {
		const auto shader = Merged(variant);
		const auto info = Analyzed(shader, shader, points, points, true);
		return {
		    Compile(ShaderType::Local, shader, info).spirv,
		    Compile(ShaderType::TessellationControl, shader, info).spirv,
		    Compile(ShaderType::TessellationEvaluation, Domain, info).spirv,
		    {Fill.begin(), Fill.end()},
		};
	};
	const auto explicit_stages = variant_stages({}, 4u);
	const auto implicit_stages = variant_stages({.implicit_handoff = true}, 4u);
	const auto quad_reference_stages = variant_stages(QuadFactors(QuadFactor::Reference), 3u);
	const auto quad_guarded_stages   = variant_stages(QuadFactors(QuadFactor::Guarded), 3u);
	RequireValid(merged_stages[2], "the evaluation stage");

	const vk::detail::DynamicLoader loader;
	if (!loader.success()) return NoDevice("the Vulkan loader is missing");
	auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
	dispatch.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
	vk::ApplicationInfo application;
	application.apiVersion = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo instance_info;
	instance_info.pApplicationInfo = &application;
	const auto created             = vk::createInstance(instance_info);
	if (created.result != vk::Result::eSuccess) return NoDevice("no Vulkan 1.3 instance");
	const auto instance = created.value;
	dispatch.init(instance);

	vk::PhysicalDevice physical;
	uint32_t           family = 0;
	for (const auto candidate: Checked(instance.enumeratePhysicalDevices(), "device enumeration")) {
		vk::PhysicalDeviceVulkan13Features modern;
		vk::PhysicalDeviceFeatures2        features;
		features.pNext = &modern;
		candidate.getFeatures2(&features);
		const auto families = candidate.getQueueFamilyProperties();
		const auto graphics = std::ranges::find_if(families, [](const auto& queue) {
			return static_cast<bool>(queue.queueFlags & vk::QueueFlagBits::eGraphics);
		});
		if (!features.features.tessellationShader || !modern.dynamicRendering ||
		    graphics == families.end())
			continue;
		const bool software =
		    candidate.getProperties().deviceType == vk::PhysicalDeviceType::eCpu;
		if (physical && !software) continue;
		physical = candidate;
		family   = static_cast<uint32_t>(graphics - families.begin());
		if (software) break;
	}
	if (!physical) return NoDevice("no device with tessellation and dynamic rendering");

	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info;
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	vk::PhysicalDeviceVulkan13Features modern;
	modern.dynamicRendering = VK_TRUE;
	vk::PhysicalDeviceFeatures2 features;
	features.features.tessellationShader = VK_TRUE;
	features.pNext                       = &modern;
	vk::DeviceCreateInfo device_info;
	device_info.pNext                = &features;
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos    = &queue_info;
	const auto        device = Checked(physical.createDevice(device_info), "device creation");
	vk::PhysicalDeviceDriverProperties driver;
	vk::PhysicalDeviceProperties2      properties;
	properties.pNext = &driver;
	physical.getProperties2(&properties);
	const auto name = std::string(properties.properties.deviceName.data()) + " (" +
	                  driver.driverName.data() + ", " + driver.driverInfo.data() + ")";
	dispatch.init(device);
	const auto queue = device.getQueue(family, 0);

	const auto memory_types = physical.getMemoryProperties();
	const auto allocate     = [&](const vk::MemoryRequirements& needs, vk::MemoryPropertyFlags flags) {
		for (uint32_t type = 0; type < memory_types.memoryTypeCount; type++) {
			if ((needs.memoryTypeBits >> type & 1u) == 0u ||
			    (memory_types.memoryTypes[type].propertyFlags & flags) != flags)
				continue;
			vk::MemoryAllocateInfo info;
			info.allocationSize  = needs.size;
			info.memoryTypeIndex = type;
			return Checked(device.allocateMemory(info), "memory allocation");
		}
		Require(false, "the device has no memory type for the test");
		return vk::DeviceMemory {};
	};

	constexpr auto      Format = vk::Format::eR8G8B8A8Unorm;
	vk::ImageCreateInfo image_info;
	image_info.imageType   = vk::ImageType::e2D;
	image_info.format      = Format;
	image_info.extent      = vk::Extent3D {Side, Side, 1u};
	image_info.mipLevels   = 1;
	image_info.arrayLayers = 1;
	image_info.samples     = vk::SampleCountFlagBits::e1;
	image_info.usage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
	const auto image        = Checked(device.createImage(image_info), "image creation");
	const auto image_memory = allocate(device.getImageMemoryRequirements(image), {});
	Checked(device.bindImageMemory(image, image_memory, 0), "image binding");
	const vk::ImageSubresourceRange whole {vk::ImageAspectFlagBits::eColor, 0u, 1u, 0u, 1u};
	vk::ImageViewCreateInfo         view_info;
	view_info.image            = image;
	view_info.viewType         = vk::ImageViewType::e2D;
	view_info.format           = Format;
	view_info.subresourceRange = whole;
	const auto view = Checked(device.createImageView(view_info), "view creation");

	constexpr vk::DeviceSize PixelBytes = Side * Side * 4u;
	vk::BufferCreateInfo     buffer_info;
	buffer_info.size  = PixelBytes + sizeof(WaveLocalPatchIndices);
	buffer_info.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
	const auto buffer        = Checked(device.createBuffer(buffer_info), "buffer creation");
	const auto buffer_memory = allocate(
	    device.getBufferMemoryRequirements(buffer),
	    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	Checked(device.bindBufferMemory(buffer, buffer_memory, 0), "buffer binding");
	auto* bytes = static_cast<uint8_t*>(
	    Checked(device.mapMemory(buffer_memory, 0, VK_WHOLE_SIZE), "buffer mapping"));

	constexpr std::array StageBits {
	    vk::ShaderStageFlagBits::eVertex, vk::ShaderStageFlagBits::eTessellationControl,
	    vk::ShaderStageFlagBits::eTessellationEvaluation, vk::ShaderStageFlagBits::eFragment};
	const auto draw = [&](const Stages& spirv, std::span<const uint16_t> indices,
	                      uint32_t control_points, const char* what) {
	std::memcpy(bytes + PixelBytes, indices.data(), indices.size_bytes());
	std::array<vk::PipelineShaderStageCreateInfo, 4> stages;
	for (size_t index = 0; index < stages.size(); index++) {
		vk::ShaderModuleCreateInfo module_info;
		module_info.codeSize = spirv[index].size() * sizeof(uint32_t);
		module_info.pCode    = spirv[index].data();
		stages[index].stage  = StageBits[index];
		stages[index].module = Checked(device.createShaderModule(module_info), "module creation");
		stages[index].pName  = "main";
	}
	const auto all_stages = StageBits[0] | StageBits[1] | StageBits[2] | StageBits[3];
	const vk::PushConstantRange push_range {all_stages, 0u, Recompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo layout_info;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_range;
	const auto layout = Checked(device.createPipelineLayout(layout_info), "layout creation");

	const vk::PipelineVertexInputStateCreateInfo vertex_input;
	vk::PipelineInputAssemblyStateCreateInfo     assembly;
	assembly.topology = vk::PrimitiveTopology::ePatchList;
	vk::PipelineTessellationStateCreateInfo tessellation;
	tessellation.patchControlPoints = control_points;
	const vk::Viewport                  viewport {0.0f, 0.0f, Side, Side, 0.0f, 1.0f};
	const vk::Rect2D                    area {{0, 0}, {Side, Side}};
	vk::PipelineViewportStateCreateInfo viewports;
	viewports.viewportCount = 1;
	viewports.pViewports    = &viewport;
	viewports.scissorCount  = 1;
	viewports.pScissors     = &area;
	vk::PipelineRasterizationStateCreateInfo raster;
	raster.polygonMode = vk::PolygonMode::eFill;
	raster.cullMode    = vk::CullModeFlagBits::eNone;
	raster.lineWidth   = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample;
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	vk::PipelineColorBlendAttachmentState blend_target;
	blend_target.colorWriteMask =
	    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	    vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend;
	blend.attachmentCount = 1;
	blend.pAttachments    = &blend_target;
	vk::PipelineRenderingCreateInfo rendering;
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &Format;
	vk::GraphicsPipelineCreateInfo pipeline_info;
	pipeline_info.pNext               = &rendering;
	pipeline_info.stageCount          = static_cast<uint32_t>(stages.size());
	pipeline_info.pStages             = stages.data();
	pipeline_info.pVertexInputState   = &vertex_input;
	pipeline_info.pInputAssemblyState = &assembly;
	pipeline_info.pTessellationState  = &tessellation;
	pipeline_info.pViewportState      = &viewports;
	pipeline_info.pRasterizationState = &raster;
	pipeline_info.pMultisampleState   = &multisample;
	pipeline_info.pColorBlendState    = &blend;
	pipeline_info.layout              = layout;
	const auto pipeline =
	    Checked(device.createGraphicsPipeline(nullptr, pipeline_info), "pipeline creation");

	vk::CommandPoolCreateInfo pool_info;
	pool_info.queueFamilyIndex = family;
	const auto pool = Checked(device.createCommandPool(pool_info), "pool creation");
	vk::CommandBufferAllocateInfo command_info;
	command_info.commandPool        = pool;
	command_info.commandBufferCount = 1;
	const auto command =
	    Checked(device.allocateCommandBuffers(command_info), "command allocation").front();
	const auto move = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags source,
	                      vk::AccessFlags target, vk::PipelineStageFlags before,
	                      vk::PipelineStageFlags after) {
		vk::ImageMemoryBarrier barrier;
		barrier.srcAccessMask       = source;
		barrier.dstAccessMask       = target;
		barrier.oldLayout           = from;
		barrier.newLayout           = to;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image               = image;
		barrier.subresourceRange    = whole;
		command.pipelineBarrier(before, after, {}, 0, nullptr, 0, nullptr, 1, &barrier);
	};
	Checked(command.begin(vk::CommandBufferBeginInfo {}), "command begin");
	move(vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
	     vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
	     vk::PipelineStageFlagBits::eColorAttachmentOutput);
	vk::RenderingAttachmentInfo target;
	target.imageView   = view;
	target.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
	target.loadOp      = vk::AttachmentLoadOp::eClear;
	target.storeOp     = vk::AttachmentStoreOp::eStore;
	vk::RenderingInfo pass;
	pass.renderArea           = area;
	pass.layerCount           = 1;
	pass.colorAttachmentCount = 1;
	pass.pColorAttachments    = &target;
	command.beginRendering(&pass);
	command.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	command.bindIndexBuffer(buffer, PixelBytes, vk::IndexType::eUint16);
	const std::array<uint32_t, Recompiler::IR::NativePushConstantSize / 4u> push {};
	command.pushConstants(layout, all_stages, 0, Recompiler::IR::NativePushConstantSize,
	                      push.data());
	command.drawIndexed(static_cast<uint32_t>(indices.size()), 2, 0, 0, 0);
	command.endRendering();
	move(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal,
	     vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eTransferRead,
	     vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer);
	vk::BufferImageCopy copy;
	copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0u, 0u, 1u};
	copy.imageExtent      = vk::Extent3D {Side, Side, 1u};
	command.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &copy);
	Checked(command.end(), "command end");
	vk::SubmitInfo submit;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &command;
	Checked(queue.submit(1, &submit, nullptr), "submit");
	Checked(device.waitIdle(), "the draw");

	const auto lit = [&](uint32_t x, uint32_t y) { return bytes[(y * Side + x) * 4u] != 0u; };
	uint32_t   covered = 0;
	for (uint32_t y = 0; y < Side; y++) {
		for (uint32_t x = 0; x < Side; x++) covered += lit(x, y) ? 1u : 0u;
	}
	Require(lit(28u, 26u) && (lit(54u, 42u) || lit(54u, 21u)) && !lit(2u, 2u) && !lit(61u, 2u) &&
	            !lit(2u, 61u) && !lit(61u, 61u) && covered > 1000u && covered < 1160u,
	        std::string("the two patches of the ") + what +
	            " hull, drawn in two instances, did not cover their triangles (-0.9,-0.5) "
	            "(0.9,-0.5) (-0.3,0.5) and (0.3,0.5) (0.9,0.5) (0.9,0): " +
	            std::to_string(covered) + " of 4096 pixels, expected about 1076");

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(layout);
	for (const auto& stage: stages) device.destroyShaderModule(stage.module);
	device.destroyCommandPool(pool);
	return covered;
	};
	const auto merged_pixels = draw(merged_stages, PatchIndices, 3u, "merged");
	const auto split_pixels  = draw(split_stages, PatchIndices, 3u, "split");
	draw(explicit_stages, WaveLocalPatchIndices, 4u, "four-point explicit handoff");
	std::array<uint8_t, PixelBytes> reference;
	std::memcpy(reference.data(), bytes, reference.size());
	draw(implicit_stages, WaveLocalPatchIndices, 4u, "four-point implicit handoff");
	Require(std::memcmp(reference.data(), bytes, reference.size()) == 0,
	        "a wave-local implicit handoff changed the draw's pixels");
	draw(quad_reference_stages, PatchIndices, 3u, "direct own-predicate factor");
	std::memcpy(reference.data(), bytes, reference.size());
	draw(quad_guarded_stages, PatchIndices, 3u, "guarded whole-quad factor");
	Require(std::memcmp(reference.data(), bytes, reference.size()) == 0,
	        "guarded whole-quad factor lowering changed the draw's pixels");
	device.destroyBuffer(buffer);
	device.freeMemory(buffer_memory);
	device.destroyImageView(view);
	device.destroyImage(image);
	device.freeMemory(image_memory);
	device.destroy();
	instance.destroy();
	std::printf("kyty_shim_merged_hull_draw: ok, merged %u and split %u of 4096 pixels on %s\n",
	            merged_pixels, split_pixels, name.c_str());
	return 0;
}

}

int main(int argc, char** argv) {
	if (argc > 1 && std::strcmp(argv[1], "draw") == 0) return DrawMergedHull();
	if (argc > 1 && std::strcmp(argv[1], "wqm") == 0) {
		RequireQuadFactors();
		return 0;
	}
	RequireMergedForm();
	RequireWaveLocalLdsOrder();
	RequireQuadFactors();
	RequirePatchConstantRoutes();
	RequireRingReadOrder();
	RequireLimits();
	RequireSplitForm();
	RequireDetection();
	std::puts("kyty_shim_merged_hull: ok");
	return 0;
}
