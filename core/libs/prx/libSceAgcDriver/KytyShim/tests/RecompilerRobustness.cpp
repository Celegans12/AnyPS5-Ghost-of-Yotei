#include "SamplerSpillProgram.hpp"
#include "ShaderRecompilerRun.hpp"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "spirv/unified1/spirv.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Case {
	const char*           name;
	uint8_t               mode;
	uint8_t               shape;
	std::vector<uint32_t> words;
	bool                  requires_spirv = false;
	uint32_t              execution_mode = 0;
	bool                  quads_supported = true;
};

const std::vector<Case> Cases = {
    {"compute-image-lod-with-2x2-workgroup", 0x01, 0x05,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupQuadsKHR},
    {"compute-image-lod-with-4x2-workgroup", 0x01, 0x06,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupQuadsKHR},
    {"compute-image-lod-with-8x1-workgroup", 0x01, 0x03,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupLinearKHR},
    {"compute-image-lod-with-8x1-wave64-host32", 0x00, 0x03,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupQuadsKHR},
    {"compute-image-lod-with-2x2-without-quads-feature", 0x01, 0x05,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u}, true,
     spv::ExecutionModeDerivativeGroupLinearKHR, false},
    {"lds-append-in-a-vertex-shader", 0x03, 0x00, {0xd8f80000u, 0u, 0xbf810000u}, true},
    {"lds-consume-in-a-vertex-shader", 0x03, 0x00, {0xd8f40000u, 0u, 0xbf810000u}, true},
    {"lds-append-in-a-pixel-shader", 0x04, 0x00, {0xd8f80000u, 0u, 0xbf810000u}, true},
    {"lds-append-in-a-compute-shader", 0x02, 0x00, {0xd8f80000u, 0u, 0xbf810000u}, true},
    {"lds-consume-in-a-compute-shader", 0x02, 0x00, {0xd8f40000u, 0u, 0xbf810000u}, true},
    {"branch-target-outside-the-program", 0x00, 0x00, {0xbf827fffu, 0xbf810000u}},
    {"stale-lane-spill-after-read-lane-elimination", 0x32, 0x00, {0xd7610014u, 0x0201e205u, 0xbf810000u}},
    {"backward-branch-before-the-program", 0x00, 0x00, {0xbf828000u, 0xbf810000u}},
    {"setreg-float-mode-denorm", 0x00, 0x00, {0xb9801901u, 0xbf810000u}},
    {"empty-program-end", 0x00, 0x00, {0xbf810000u}},
    {"lds-atomic-in-a-pixel-shader", 0x04, 0x00, {0xd8000000u, 0x00000100u, 0xbf810000u}},
    {"lds-umin-in-a-vertex-shader", 0x03, 0x00, {0xd8070000u, 0x00000100u, 0xbf810000u}},
    {"image-get-lod-in-a-pixel-shader", 0x04, 0x00,
     {0xf1800108u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u}},
    {"image-gather-from-a-2d-image", 0x01, 0x00,
     {0xf35a218du, 0x7e000281u, 0x80000100u, 0xbf960001u, 0xbefe04c0u,
      0xea6f72eau, 0xbf8c0000u, 0xbf810000u}, true},
    {"null-export-in-a-compute-shader", 0x00, 0x00,
     {0xf8000891u, 0x03020100u, 0xbf810000u}},
    {"disabled-parameter-export-in-a-compute-shader", 0x00, 0x00,
     {0xf8000a00u, 0x03020100u, 0xbf810000u}},
};

struct Refusal {
	const char*           name;
	uint8_t               mode;
	uint8_t               shape;
	std::vector<uint32_t> words;
	const char*           reason;
	bool                  linear_supported = true;
	uint32_t              compute_local_x = 0u;
};

const std::vector<Refusal> Refusals = {
    {"compute-image-lod-with-1x1-workgroup", 0x01, 0x00,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u},
     "compute derivatives require four-lane workgroups"},
    {"compute-image-lod-with-3x1-workgroup", 0x01, 0x00,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u},
     "compute derivatives require four-lane workgroups", true, 3u},
    {"compute-image-lod-with-8x1-without-linear-feature", 0x01, 0x03,
     {0xf1800000u, 0x00000000u, 0xe8040000u, 0x00000000u, 0xbf810000u},
     "compute derivative group feature unavailable", false},
    {"vop3p-without-its-second-word", 0x3b, 0x07, {0x520a0001u, 0xce600001u}, "the code ends after"},
    {"literal-constant-without-its-word", 0x00, 0x00, {0x7e0002ffu}, "the code ends after"},
    {"vop1-dpp-without-its-modifier-word", 0x00, 0x00, {0x7e0002fau}, "the code ends after"},
    {"smem-without-its-second-word", 0x00, 0x00, {0xf4000000u}, "the code ends after"},
    {"mimg-nsa-without-its-address-words", 0x00, 0x00, {0xf0000002u, 0x00000000u}, "the code ends after"},
    {"position-export-from-a-pixel-shader", 0x04, 0x00, {0xf80008cfu, 0x03020100u, 0xbf810000u},
     "position export outside a vertex processing stage"},
    {"position-export-from-a-compute-shader", 0x00, 0x00, {0xf80008cfu, 0x03020100u, 0xbf810000u},
     "not valid in a compute shader"},
    {"parameter-export-from-a-compute-shader", 0x00, 0x00,
     {0xf8000a01u, 0x03020100u, 0xbf810000u},
     "not valid in a compute shader"},
    {"depth-compare-sample-on-a-3d-image", 0x04, 0x00, {0xf0a00110u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u},
     "depth comparison with a 3D image"},
    {"image-gather-from-a-3d-image", 0x01, 0x00,
     {0xf35a2195u, 0x7e000281u, 0x80000100u, 0xbf960001u, 0xbefe04c0u,
      0xea6f72eau, 0xbf8c0000u, 0xbf810000u}, "gathers from a 3D image"},
    {"depth-compare-sample-on-a-multisampled-image", 0x04, 0x00, {0xf0a00130u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u},
     "samples a multisampled image"},
    {"sample-on-a-multisampled-image", 0x04, 0x00, {0xf0800130u, 0x00400400u, 0xf8000801u, 0x00000004u, 0xbf810000u},
     "samples a multisampled image"},
    {"image-get-lod-in-a-vertex-shader", 0x03, 0x00,
     {0xf1800308u, 0x00820000u, 0xf80008cfu, 0x03020100u, 0xbf810000u},
     "image_get_lod requires a pixel or compute shader"},
};

struct ModeWrite {
	uint32_t word;
	uint32_t bits;
};

const std::vector<ModeWrite> ModeWrites = {
    {0xb9801801u, 0x0fu},
    {0xb9801901u, 0xf0u},
    {0xb980f801u, 0xffu},
    {0xb9801881u, 0x3cu},
    {0xb9800201u, 0x00u},
    {0xb9801803u, 0x00u},
    {0xb9001801u, 0x00u},
};

bool HasExecutionMode(const std::vector<uint32_t>& words, uint32_t expected) {
	for (size_t offset = 5; offset < words.size();) {
		const uint32_t count = words[offset] >> 16u;
		if (count == 0u || count > words.size() - offset) return false;
		if ((words[offset] & 0xffffu) == spv::OpExecutionMode && count >= 3u &&
		    words[offset + 2u] == expected) {
			return true;
		}
		offset += count;
	}
	return false;
}

bool FloatModeBits() {
	namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;
	bool ok = true;
	for (const auto& write: ModeWrites) {
		const std::vector<uint32_t> code = {write.word};
		Decoder::Instruction        inst;
		Decoder::DecodeInstruction(code, 0, inst);
		const auto bits = Decoder::FloatModeBitsWritten(inst);
		if (bits != write.bits) {
			std::printf("float-mode-bits: FAILED 0x%08x gives 0x%02x, expected 0x%02x\n", write.word, bits, write.bits);
			ok = false;
		}
	}
	return ok;
}

struct OffsetCase {
	uint32_t segment;
	uint32_t offset;
	uint32_t expected;
};

const std::vector<OffsetCase> OffsetCases = {
    {0u, 0x7ffu, 0x000007ffu},
    {0u, 0x800u, 0x00000000u},
    {0u, 0xfffu, 0x000007ffu},
    {1u, 0x7ffu, 0x000007ffu},
    {1u, 0x800u, 0xfffff800u},
    {1u, 0xfffu, 0xffffffffu},
};

bool FlatOffsets() {
	namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;
	bool ok = true;
	for (const auto& entry: OffsetCases) {
		const std::vector<uint32_t> code = {
		    (0x37u << 26u) | (entry.segment << 14u) | (0x0cu << 18u) | entry.offset,
		    (0x7fu << 16u),
		};
		Decoder::Instruction inst;
		Decoder::DecodeInstruction(code, 0, inst);
		if (inst.offset != entry.expected || inst.memory_segment != entry.segment) {
			std::printf("flat-offsets: FAILED segment %u offset 0x%03x gives 0x%08x/segment %u, expected 0x%08x/segment %u\n",
			            entry.segment, entry.offset, inst.offset, inst.memory_segment, entry.expected, entry.segment);
			ok = false;
		}
	}
	return ok;
}

bool SamplerSpillPhi() {
	const auto& shader = SamplerSpillProgram::Words;
	auto runtime_branch = shader;
	runtime_branch[SamplerSpillProgram::RestoreCompare] = 0xbf068002u;
	std::string refusal;
	KytyFuzz::Run(runtime_branch, 1, 0, &refusal, nullptr, nullptr, nullptr, 32u);
	if (refusal.find("GetSamplerResource dword 0 is not a valid runtime value") == std::string::npos) {
		std::printf("sampler-spill-phi: FAILED, unexpected result on the reachable reuse path: %s\n", refusal.c_str());
		return false;
	}

	auto unconditional_restore = shader;
	unconditional_restore[SamplerSpillProgram::RestoreBranch] = 0xbf800000u;
	const decltype(unconditional_restore)* cases[] {&unconditional_restore, &shader};
	for (const auto* code: cases) {
		bool emitted = false;
		std::vector<uint32_t> spirv;
		refusal.clear();
		KytyFuzz::Run(*code, 1, 0, &refusal, &emitted, nullptr, &spirv, 32u);
		if (!emitted || !KytyFuzz::Valid(spirv, refusal)) {
			std::printf("sampler-spill-phi: FAILED, %s restore: %s\n",
			            code == &shader ? "guarded" : "unconditional", refusal.c_str());
			return false;
		}
	}
	return true;
}

bool PhiReachability() {
	using namespace KytyFuzz::IR;
	using Term = KytyFuzz::Recompiler::CFG::TerminatorKind;
	enum class Input { Same, Conflict, Undef, Runtime, Unreachable, Unseeded };
	for (const auto input: {Input::Same, Input::Conflict, Input::Undef, Input::Runtime, Input::Unreachable,
	                        Input::Unseeded}) {
		Program program;
		program.user_data_count = 1;
		for (uint32_t id = 0; id < 8; ++id) {
			auto& block = *program.block_storage.emplace_back(std::make_unique<Block>());
			block.id = id;
			program.blocks.push_back(&block);
		}
		auto& entry       = *program.blocks[0];
		auto& outer       = *program.blocks[1];
		auto& inner       = *program.blocks[2];
		auto& keep        = *program.blocks[3];
		auto& update      = *program.blocks[4];
		auto& inner_latch = *program.blocks[5];
		auto& outer_latch = *program.blocks[6];
		auto& exit        = *program.blocks[7];
		const auto link = [](Block& from, Block& yes, Block* no = nullptr, Value condition = {}) {
			from.terminator.kind        = no == nullptr ? Term::Branch : Term::ConditionalBranch;
			from.terminator.true_block  = &yes;
			from.terminator.false_block = no;
			from.condition             = condition;
			from.AddBranch(&yes);
			if (no != nullptr) from.AddBranch(no);
		};
		const Value runtime(&entry.AppendNewInst(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(0u))}));
		const Value predicate(&entry.AppendNewInst(ValueOpcode::INotEqual32, {runtime, Value(0u)}));
		const auto phi_flags = static_cast<uint64_t>(Type::U32);
		auto& carried_outer = outer.AppendNewInst(ValueOpcode::Phi, {}, phi_flags);
		auto& outer_count   = outer.AppendNewInst(ValueOpcode::Phi, {}, phi_flags);
		auto& carried_inner = inner.AppendNewInst(ValueOpcode::Phi, {}, phi_flags);
		auto& inner_count   = inner.AppendNewInst(ValueOpcode::Phi, {}, phi_flags);
		auto& joined        = inner_latch.AppendNewInst(ValueOpcode::Phi, {}, phi_flags);
		constexpr auto exec_nonzero = KytyFuzz::Recompiler::CFG::BranchCondition::ExecNonZero;
		const auto exec = input == Input::Unreachable ? Value(true) : predicate;
		const Value condition(&inner.AppendNewInst(ValueOpcode::ConditionRef, {exec}, static_cast<uint64_t>(exec_nonzero)));
		link(entry, outer);
		link(outer, inner);
		link(inner, keep, &update, condition);
		link(keep, inner_latch);
		link(update, inner_latch);
		const auto repeat = [&](Block& first, Block& header, Block& latch, Block& end, Inst& count) {
			const Value next(&latch.AppendNewInst(ValueOpcode::IAdd32, {Value(&count), Value(1u)}));
			const Value more(&latch.AppendNewInst(ValueOpcode::INotEqual32, {next, Value(2u)}));
			link(latch, header, &end, more);
			count.AddPhiOperand(&first, Value(0u));
			count.AddPhiOperand(&latch, next);
		};
		repeat(outer, inner, inner_latch, outer_latch, inner_count);
		repeat(entry, outer, outer_latch, exit, outer_count);
		Value next(17u);
		switch (input) {
			case Input::Undef: next = Value(&update.AppendNewInst(ValueOpcode::UndefU32)); break;
			case Input::Runtime:
			case Input::Unseeded: next = runtime; break;
			case Input::Conflict:
			case Input::Unreachable: next = Value(29u); break;
			case Input::Same: break;
		}
		const auto seed = input == Input::Unseeded ? Value(&entry.AppendNewInst(ValueOpcode::UndefU32)) : Value(17u);
		carried_outer.AddPhiOperand(&entry, seed);
		carried_outer.AddPhiOperand(&outer_latch, Value(&joined));
		carried_inner.AddPhiOperand(&outer, Value(&carried_outer));
		carried_inner.AddPhiOperand(&inner_latch, Value(&joined));
		joined.AddPhiOperand(&keep, Value(&carried_inner));
		joined.AddPhiOperand(&update, next);
		auto& increment = exit.AppendNewInst(ValueOpcode::IAdd32, {Value(&joined), Value(1u)});
		auto& output = exit.AppendNewInst(ValueOpcode::ReferenceU32, {Value(&increment)});
		ValidateProgram(program);
		ConstantPropagationPass(program.blocks);
		RemoveIdentities(program);
		ValidateProgram(program);
		const bool fold = input == Input::Same || input == Input::Unreachable;
		const auto* add = output.Arg(0).TryInstruction();
		const auto* phi = add != nullptr && add->GetOpcode() == ValueOpcode::IAdd32
		                      ? add->Arg(0).TryInstruction() : nullptr;
		if (fold ? output.Arg(0) != Value(18u)
		         : phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->Arg(1) != next) {
			std::printf("phi-reachability: FAILED for nested loop input %u\n", static_cast<unsigned>(input));
			return false;
		}
	}

	return true;
}

struct MadCase {
	const char*           name;
	std::vector<uint32_t> words;
	bool                  fused;
};

const std::vector<MadCase> MadCases = {
    {"v_mac_f32", {0x7e080302u, 0x3e080300u, 0xf8000801u, 0x00000004u, 0xbf810000u}, false},
    {"v_mad_f32", {0xd5410004u, 0x040a0300u, 0xf8000801u, 0x00000004u, 0xbf810000u}, false},
    {"v_fmac_f32", {0x7e080302u, 0x56080300u, 0xf8000801u, 0x00000004u, 0xbf810000u}, true},
    {"v_fma_f32", {0xd54b0004u, 0x040a0300u, 0xf8000801u, 0x00000004u, 0xbf810000u}, true},
};

struct MadScan {
	uint32_t fma            = 0;
	uint32_t product_sums   = 0;
	uint32_t exact_products = 0;
	uint32_t output_stores  = 0;
	uint32_t output_type    = 0;
	uint32_t f32_type       = 0;
	size_t   output_store   = 0;
	size_t   first_function = 0;
};

MadScan ScanMad(const std::vector<uint32_t>& words) {
	MadScan                    scan;
	std::set<uint32_t>         glsl;
	std::set<uint32_t>         f32_types;
	std::set<uint32_t>         no_contraction;
	std::set<uint32_t>         location_zero;
	std::set<uint32_t>         output_variables;
	std::map<uint32_t, uint32_t> vector_sizes;
	std::map<uint32_t, size_t> definitions;
	std::vector<size_t>       stores;
	for (size_t offset = 5; offset < words.size();) {
		const uint32_t count = words[offset] >> 16u;
		if (count == 0u || count > words.size() - offset) return {};
		const auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
		if (opcode == spv::OpExtInstImport && count >= 3u &&
		    std::string_view(reinterpret_cast<const char*>(&words[offset + 2u])).starts_with("GLSL.std.450")) {
			glsl.insert(words[offset + 1u]);
		}
		if (opcode == spv::OpTypeFloat && count >= 3u && words[offset + 2u] == 32u) {
			f32_types.insert(words[offset + 1u]);
			scan.f32_type = words[offset + 1u];
		}
		if (opcode == spv::OpTypeVector && count >= 4u) vector_sizes.emplace(words[offset + 1u], words[offset + 3u]);
		if (opcode == spv::OpDecorate && count >= 3u) {
			if (words[offset + 2u] == spv::DecorationNoContraction) no_contraction.insert(words[offset + 1u]);
			if (count >= 4u && words[offset + 2u] == spv::DecorationLocation && words[offset + 3u] == 0u) {
				location_zero.insert(words[offset + 1u]);
			}
		}
		if (opcode == spv::OpVariable && count >= 4u && words[offset + 3u] == spv::StorageClassOutput) {
			output_variables.insert(words[offset + 2u]);
		}
		if (opcode == spv::OpStore && count >= 3u) stores.push_back(offset);
		if (opcode == spv::OpFunction && scan.first_function == 0u) scan.first_function = offset;
		if (count >= 3u) {
			switch (opcode) {
				case spv::OpBitcast:
				case spv::OpCopyObject:
				case spv::OpCompositeConstruct:
				case spv::OpCompositeExtract:
				case spv::OpCompositeInsert:
				case spv::OpVectorShuffle:
				case spv::OpFNegate:
				case spv::OpFAdd:
				case spv::OpFMul:
				case spv::OpExtInst:
				case spv::OpLoad:
				case spv::OpConstantComposite:
				case spv::OpConstantNull:
					definitions.emplace(words[offset + 2u], offset);
					break;
				default: break;
			}
		}
		offset += count;
	}
	uint32_t output = 0u;
	for (const auto variable: output_variables) {
		if (!location_zero.contains(variable)) continue;
		if (output != 0u) return {};
		output = variable;
	}
	if (output == 0u) return scan;
	for (const auto store: stores) {
		if (words[store + 1u] != output) continue;
		++scan.output_stores;
		scan.output_store = store;
	}
	if (scan.output_stores != 1u) return scan;
	const auto root = words[scan.output_store + 2u];
	if (const auto found = definitions.find(root); found != definitions.end()) {
		scan.output_type = words[found->second + 1u];
	}
	std::vector<std::pair<uint32_t, uint32_t>> pending {{root, 0u}};
	std::set<std::pair<uint32_t, uint32_t>>    visited;
	while (!pending.empty()) {
		const auto [id, lane] = pending.back();
		pending.pop_back();
		if (!visited.insert({id, lane}).second) continue;
		const auto found = definitions.find(id);
		if (found == definitions.end()) continue;
		const auto offset = found->second;
		const auto count = words[offset] >> 16u;
		const auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
		if (lane == UINT32_MAX && opcode == spv::OpExtInst && count >= 8u && f32_types.contains(words[offset + 1u]) &&
		    glsl.contains(words[offset + 3u]) && words[offset + 4u] == 50u) {
			++scan.fma;
		}
		if (lane == UINT32_MAX && opcode == spv::OpFAdd && count >= 5u &&
		    f32_types.contains(words[offset + 1u])) {
			for (const auto operand: {words[offset + 3u], words[offset + 4u]}) {
				const auto product = definitions.find(operand);
				if (product == definitions.end()) continue;
				const auto product_offset = product->second;
				if ((words[product_offset] & 0xffffu) != spv::OpFMul ||
				    !f32_types.contains(words[product_offset + 1u])) continue;
				++scan.product_sums;
				scan.exact_products += no_contraction.contains(operand) && no_contraction.contains(id);
			}
		}
		const auto push = [&](size_t index, uint32_t component) {
			if (index < count) pending.emplace_back(words[offset + index], component);
		};
		switch (opcode) {
			case spv::OpBitcast:
			case spv::OpCopyObject:
			case spv::OpFNegate: push(3u, lane); break;
			case spv::OpCompositeConstruct:
				if (lane != UINT32_MAX) push(3u + lane, UINT32_MAX);
				break;
			case spv::OpCompositeExtract:
				if (lane == UINT32_MAX && count == 5u) push(3u, words[offset + 4u]);
				break;
			case spv::OpCompositeInsert:
				if (lane != UINT32_MAX && count == 6u) {
					if (words[offset + 5u] == lane) push(3u, UINT32_MAX);
					else push(4u, lane);
				}
				break;
			case spv::OpVectorShuffle:
				if (lane != UINT32_MAX && 5u + lane < count) {
					const auto first = definitions.find(words[offset + 3u]);
					if (first == definitions.end()) break;
					const auto size = vector_sizes.find(words[first->second + 1u]);
					if (size == vector_sizes.end()) break;
					const auto selected = words[offset + 5u + lane];
					if (selected < size->second) push(3u, selected);
					else if (selected != UINT32_MAX) push(4u, selected - size->second);
				}
				break;
			case spv::OpFAdd:
			case spv::OpFMul:
				push(3u, lane);
				push(4u, lane);
				break;
			case spv::OpExtInst:
				for (size_t index = 5u; index < count; ++index) push(index, lane);
				break;
			default: break;
		}
	}
	return scan;
}

std::vector<uint32_t> MaskMadOutput(const std::vector<uint32_t>& words, const MadScan& scan) {
	if (scan.output_stores != 1u || scan.output_type == 0u || scan.first_function == 0u) return {};
	auto masked = words;
	const auto replacement = masked[3u]++;
	masked[scan.output_store + 2u] = replacement;
	masked.insert(masked.begin() + scan.first_function,
	              {(3u << 16u) | static_cast<uint32_t>(spv::OpUndef), scan.output_type, replacement});
	return masked;
}

std::vector<uint32_t> MaskMadComponent(const std::vector<uint32_t>& words, const MadScan& scan) {
	if (scan.output_stores != 1u || scan.output_type == 0u || scan.f32_type == 0u ||
	    scan.first_function == 0u || scan.first_function >= scan.output_store) return {};
	auto masked = words;
	const auto source = masked[scan.output_store + 2u];
	const auto undefined = masked[3u]++;
	const auto replacement = masked[3u]++;
	masked[scan.output_store + 2u] = replacement;
	masked.insert(masked.begin() + scan.first_function,
	              {(3u << 16u) | static_cast<uint32_t>(spv::OpUndef), scan.f32_type, undefined});
	masked.insert(masked.begin() + scan.output_store + 3u,
	              {(6u << 16u) | static_cast<uint32_t>(spv::OpCompositeInsert),
	               scan.output_type, replacement, undefined, source, 0u});
	return masked;
}

bool MadNotFused() {
	bool ok = true;
	for (const auto& entry: MadCases) {
		for (const bool exact: {true, false}) {
			const std::string what = std::string(entry.name) + (exact ? "" : " with f32_product_no_contraction off");
			auto profile = KytyFuzz::Profile();
			profile.f32_product_no_contraction = exact;
			std::vector<uint32_t> spirv;
			std::string           refusal;
			bool                  emitted = false;
			KytyFuzz::Run(entry.words, 0x04, 0x00, &refusal, &emitted, &profile, &spirv);
			if (!emitted) {
				std::printf("mad-not-fused: FAILED, %s did not emit SPIR-V: %s\n", what.c_str(), refusal.c_str());
				ok = false;
				continue;
			}
			std::string message;
			if (!KytyFuzz::Valid(spirv, message)) {
				std::printf("mad-not-fused: FAILED, %s is not valid SPIR-V: %s\n", what.c_str(), message.c_str());
				ok = false;
				continue;
			}
			const auto scan     = ScanMad(spirv);
			const bool expected = scan.output_stores == 1u &&
			                      (entry.fused ? scan.fma == 1u && scan.product_sums == 0u
			                                   : scan.fma == 0u && scan.product_sums == 1u && scan.exact_products == 1u);
			if (!expected) {
				std::printf("mad-not-fused: FAILED, %s: fma=%u product_sums=%u no_contraction_pairs=%u\n", what.c_str(), scan.fma,
				            scan.product_sums, scan.exact_products);
				ok = false;
				continue;
			}
			const auto CheckMask = [&](const std::vector<uint32_t>& masked, const char* name) {
				if (masked.empty() || !KytyFuzz::Valid(masked, message)) {
					std::printf("mad-not-fused: FAILED, %s %s control is invalid: %s\n", what.c_str(), name, message.c_str());
					return false;
				}
				const auto masked_scan = ScanMad(masked);
				if (masked_scan.output_stores != 1u || masked_scan.fma != 0u ||
				    masked_scan.product_sums != 0u || masked_scan.exact_products != 0u) {
					std::printf("mad-not-fused: FAILED, %s %s dead arithmetic reached the export scan\n", what.c_str(), name);
					return false;
				}
				return true;
			};
			ok = CheckMask(MaskMadOutput(spirv, scan), "whole-export") && ok;
			ok = CheckMask(MaskMadComponent(spirv, scan), "component") && ok;
		}
	}
	return ok;
}

}

int main(int argc, char** argv) {
	bool found = argc < 2;
	if (argc < 2 || std::string_view(argv[1]) == "float-mode-bits") {
		found = true;
		if (!FloatModeBits()) return 1;
		std::printf("float-mode-bits: ok\n");
	}
	if (argc < 2 || std::string_view(argv[1]) == "flat-offsets") {
		found = true;
		if (!FlatOffsets()) return 1;
		std::printf("flat-offsets: ok\n");
	}
	if (argc >= 2 && std::string_view(argv[1]) == "sampler-spill-phi") {
		found = true;
		if (!SamplerSpillPhi()) return 1;
		std::printf("sampler-spill-phi: ok\n");
	}
	if (argc >= 2 && std::string_view(argv[1]) == "phi-reachability") {
		found = true;
		Common::SoftExitScope soft;
		try {
			if (!PhiReachability()) return 1;
		} catch (const std::exception& error) {
			std::printf("phi-reachability: FAILED: %s\n", error.what());
			return 1;
		}
		std::printf("phi-reachability: ok\n");
	}
	for (const auto& entry: Cases) {
		if (argc >= 2 && std::string_view(argv[1]) != entry.name) continue;
		found = true;
		bool emitted = false;
		std::vector<uint32_t> spirv;
		auto profile = KytyFuzz::Profile();
		profile.compute_derivative_quads = entry.quads_supported;
		KytyFuzz::Run(entry.words, entry.mode, entry.shape, nullptr, &emitted, &profile,
		              entry.execution_mode == 0u ? nullptr : &spirv);
		if (entry.requires_spirv && !emitted) {
			std::printf("%s: FAILED, did not emit SPIR-V\n", entry.name);
			return 1;
		}
		if (entry.execution_mode != 0u && !HasExecutionMode(spirv, entry.execution_mode)) {
			std::printf("%s: FAILED, missing execution mode %u\n", entry.name, entry.execution_mode);
			return 1;
		}
		std::printf("%s: ok\n", entry.name);
	}
	for (const auto& entry: Refusals) {
		if (argc >= 2 && std::string_view(argv[1]) != entry.name) continue;
		found = true;
		std::string refusal;
		auto profile = KytyFuzz::Profile();
		profile.compute_derivative_linear = entry.linear_supported;
		KytyFuzz::Run(entry.words, entry.mode, entry.shape, &refusal, nullptr, &profile,
		              nullptr, entry.compute_local_x);
		if (refusal.find(entry.reason) == std::string::npos) {
			std::printf("%s: FAILED, refusal \"%s\" instead of \"%s\"\n", entry.name, refusal.c_str(), entry.reason);
			return 1;
		}
		std::printf("%s: ok\n", entry.name);
	}
	if (argc < 2 || std::string_view(argv[1]) == "mad-not-fused") {
		found = true;
		if (!MadNotFused()) return 1;
		std::printf("mad-not-fused: ok\n");
	}
	if (!found) {
		std::printf("kyty_shim_recompiler_robustness: unknown case %s\n", argv[1]);
		return 2;
	}
	std::printf("kyty_shim_recompiler_robustness: ok\n");
	return 0;
}
