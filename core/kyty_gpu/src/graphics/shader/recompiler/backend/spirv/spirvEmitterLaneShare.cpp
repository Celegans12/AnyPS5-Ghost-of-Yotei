// A wave64 compute shader whose host subgroup is 32 wide runs with lane_count == 2: every host invocation
// emulates guest lanes l (half 0) and l + 32 (half 1), and EmitBlock emits every IR instruction twice. A
// value that is the same in both halves (a wave-uniform value) therefore costs two registers and twice
// the code. This file proves which values are the same in both halves and lets the high half reuse the id
// the low half produced.
//
// Both halves run in ONE instruction stream: every branch has a single condition (ConditionRef combines
// both halves) and every phi selects the same incoming edge for both halves. So a value is the same in
// both halves when the instruction computing it
//   1. is a pure function of its IR operands (no lane id, no EXEC/half state, no cross-lane move, no
//      per-half storage such as scratch or compact LDS slots), and
//   2. has operands that are the same in both halves (constants, user SGPRs, workgroup ids, whole-wave
//      results, or the result of another such instruction; phi cycles are solved optimistically).
// Operands of a shared instruction are therefore shared ids themselves, so the high half's copy of the
// instruction is the same SPIR-V on the same ids: reusing the low half's result cannot change the program.
//
// What is deliberately NOT shared (and why), as established by reading the emitters:
//   * ConditionRef: the high half's result is the constant `true`, not the low half's condition
//     (EmitConditionRef returns `other_half->Def(IR::Value(&inst))`; `&inst` is a const pointer, so
//     IR::Value picks its bool constructor). Everything derived from it ends in a block condition (decided by
//     the low half alone) or in Reference, i.e. the high half's copy is dead code, but it is NOT the same
//     value as the low half's, so it is not "uniform" for the purpose of sharing.
//   * LaneId, GetBuiltin LocalInvocationId/Index and GlobalInvocationId: EmitSubgroupLocalInvocationId and
//     EmitLocalInvocationIndex add `lane_half * 32` (spirvEmitterHelpers.cpp). No IR operand carries it.
//   * DppMoveU32, DppUpdateU32, WriteLane, Permlane16U32, PermuteU32, BpermuteU32, SwizzleU32,
//     DataAppend/Consume: cross-lane operations, and EmitDppWriteCondition / WriteLane / Permlane16 read
//     the subgroup lane id inside the emitter.
//   * Vector buffer loads whose V# has ADD_TID (packed stride bit 20: BufferLane adds the lane id) and
//     every IndirectBuffer vector load (the descriptor, and so ADD_TID, is a run-time value).
//   * Scratch loads (scratch_variable[lane_half]), LDS loads (the compact-LDS slot is
//     `slot * lane_count + lane_half`; the contents are written per lane), FlatLocal loads (aperture
//     dispatch to LDS/scratch), image reads and samples (implicit derivatives, per-lane results).
//   * Atomics and every store: side effects that both halves must perform.
//   * Void operations, resource-handle producers (they emit nothing) and anything not listed below.
//
// The per-lane "thread bit" shadows that live in SGPR slots (Translate.cpp WriteMask / ReadMask) are plain
// U1 SSA values here: they are lane dependent because they derive from LaneId or a per-lane compare, and
// the data-flow below sees that; the rules never look at a register NAME, only at the producing operation.
#define SPV_ENABLE_UTILITY_CODE 1
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "common/shaderEnv.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

enum class Rule : uint8_t {
	Never,    // lane dependent, side effects or unknown
	Pure,     // the same in both halves when every operand is; emitted once
	Source,   // the same in both halves whatever the operands (user data, workgroup id, undef); emitted once
	Whole,    // the emitters already give both halves one result (Ballot, ReadFirstLane)
	ReadLane, // the same in both halves when the lane operand is (the value operand may be per lane)
	Handle,   // typed resource handle: emits nothing; the same when its operands are
	Phi,      // the same when every incoming value is
};

Rule MemoryRule(const EmitterState& state, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto index = inst.Flags<IR::MemoryFlags>().index;
	if (index >= state.program.memory_info.size()) return Rule::Never;
	const auto& mem = state.program.memory_info[index];
	switch (inst.GetOpcode()) {
		case Op::ReadConstBuffer:
			// EmitReadConstBuffer always loads one dword through the scalar path (scalar buffer, or the
			// GPU-selected descriptor of LoadIndirectScalarBuffer); neither reads lane state.
			return Rule::Pure;
		case Op::LoadAddressU8:
		case Op::LoadAddressU16:
		case Op::LoadAddressU32:
			// EmitLoadMemory: ScalarAddress, Flat and Global reads go through the BDA path with an address
			// built from the operands only. FlatLocal (aperture to LDS/scratch) and Scratch are per lane.
			switch (mem.kind) {
				case IR::ResourceKind::ScalarAddress: return Rule::Pure;
				case IR::ResourceKind::Flat:
				case IR::ResourceKind::Global: return Rule::Pure;
				default: return Rule::Never;
			}
		case Op::LoadBufferU8:
		case Op::LoadBufferU16:
		case Op::LoadBufferU32:
		case Op::LoadBufferU32x2:
		case Op::LoadBufferU32x3:
		case Op::LoadBufferU32x4:
			// Only a plain buffer whose V# does not add the lane id (ADD_TID, packed stride bit 20, a
			// property of the specialization known at emission). IndirectBuffer reads the descriptor at
			// run time, so ADD_TID is not known.
			if (mem.kind != IR::ResourceKind::Buffer ||
			    mem.resource >= state.program.info.buffers.size()) {
				return Rule::Never;
			}
			return (StorageBufferPackedStride(state, mem) & (1u << 20u)) == 0u ? Rule::Pure : Rule::Never;
		default: return Rule::Never;
	}
}

Rule RuleFor(const EmitterState& state, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	switch (inst.GetOpcode()) {
		case Op::Phi: return Rule::Phi;
		case Op::Ballot:
		case Op::ReadFirstLane: return Rule::Whole;
		case Op::ConditionRef:
			// The high half's ConditionRef is the constant true, not the low half's condition
			// (spirvEmitterFlow.cpp: IR::Value(&inst) on a const Inst converts the pointer to bool), so the
			// two halves do NOT hold the same value. Everything derived from it feeds branch conditions,
			// which only the low half's chain decides, and Reference; it stays as it is (token condref:
			// treat it as the whole-wave value it is meant to be).
			return Rule::Never;
		case Op::ReadLane: return Rule::ReadLane;
		case Op::UndefU1:
		case Op::UndefU32:
		case Op::GetUserData:
		case Op::GetShaderBase:
		case Op::GetDispatchThreadExtent: return Rule::Source;
		case Op::GetBuiltin: {
			if (!inst.Arg(0).IsImmediate()) return Rule::Never;
			const auto kind = static_cast<IR::StageInputKind>(inst.Arg(0).U32());
			return kind == IR::StageInputKind::WorkgroupId || kind == IR::StageInputKind::NumWorkgroups
			           ? Rule::Source
			           : Rule::Never;
		}
		case Op::GetSrtResource:
		case Op::GetBufferResource:
		case Op::GetAddressResource:
		case Op::GetImageResource:
		case Op::GetSamplerResource:
		case Op::MakeImageAddress: return Rule::Handle;
		case Op::ReadConst: return Rule::Pure;
		case Op::ReadConstBuffer:
		case Op::LoadAddressU8:
		case Op::LoadAddressU16:
		case Op::LoadAddressU32:
		case Op::LoadBufferU8:
		case Op::LoadBufferU16:
		case Op::LoadBufferU32:
		case Op::LoadBufferU32x2:
		case Op::LoadBufferU32x3:
		case Op::LoadBufferU32x4: return MemoryRule(state, inst);
		case Op::Identity:
		case Op::BitCastU16F16:
		case Op::BitCastF16U16:
		case Op::BitCastU32F32:
		case Op::BitCastF32U32:
		case Op::BitCastU64F64:
		case Op::BitCastF64U64:
		case Op::ConvertU16U32:
		case Op::ConvertU32U16:
		case Op::ConvertU8U32:
		case Op::ConvertU32U8:
		case Op::ConvertF32F16:
		case Op::ConvertF16F32:
		case Op::ConvertS32F32:
		case Op::ConvertU32F32:
		case Op::ConvertF32S32:
		case Op::ConvertF64S32:
		case Op::ConvertF32F64:
		case Op::ConvertF64F32:
		case Op::ConvertF32U32:
		case Op::ConvertF64U32:
		case Op::CompositeConstructU64:
		case Op::CompositeConstructU32x2:
		case Op::CompositeConstructU32x3:
		case Op::CompositeConstructF32x2:
		case Op::CompositeConstructU32x4:
		case Op::CompositeExtractU64:
		case Op::CompositeExtractU32x2:
		case Op::CompositeExtractU32x3:
		case Op::CompositeExtractU32x4:
		case Op::PackHalf2x16:
		case Op::PackSnorm2x16:
		case Op::PackUnorm2x16:
		case Op::PackFloat2x16Rtz:
		case Op::FPAbs32:
		case Op::FPNeg32:
		case Op::FPSaturate32:
		case Op::BitFieldInsert:
		case Op::BitFieldUExtract:
		case Op::BitFieldSExtract:
		case Op::WqmU64:
		case Op::SelectU1:
		case Op::SelectF32:
		case Op::SelectU32:
		case Op::IAdd32:
		case Op::IAdd64:
		case Op::ISub32:
		case Op::ISub64:
		case Op::IMul32:
		case Op::UDiv32:
		case Op::SMulHi:
		case Op::UMulHi:
		case Op::IAbs32:
		case Op::ShiftLeftLogical32:
		case Op::ShiftLeftLogical64:
		case Op::ShiftRightLogical32:
		case Op::ShiftRightLogical64:
		case Op::ShiftRightArithmetic32:
		case Op::ShiftRightArithmetic64:
		case Op::BitwiseAnd32:
		case Op::BitwiseAnd64:
		case Op::BitwiseOr32:
		case Op::BitwiseXor32:
		case Op::BitwiseNot32:
		case Op::BitReverse32:
		case Op::BitCount32:
		case Op::FindUMsb32:
		case Op::FindILsb32:
		case Op::SMin32:
		case Op::UMin32:
		case Op::SMax32:
		case Op::UMax32:
		case Op::SMinTri32:
		case Op::UMinTri32:
		case Op::SMaxTri32:
		case Op::UMaxTri32:
		case Op::SMedTri32:
		case Op::UMedTri32:
		case Op::SLessThan32:
		case Op::SLessThan64:
		case Op::SLessThanEqual64:
		case Op::ULessThanEqual64:
		case Op::UGreaterThanEqual64:
		case Op::ULessThan32:
		case Op::ULessThan64:
		case Op::IEqual32:
		case Op::IEqual64:
		case Op::SLessThanEqual32:
		case Op::ULessThanEqual32:
		case Op::SGreaterThan32:
		case Op::UGreaterThan32:
		case Op::UGreaterThan64:
		case Op::INotEqual32:
		case Op::INotEqual64:
		case Op::SGreaterThanEqual32:
		case Op::UGreaterThanEqual32:
		case Op::LogicalOr:
		case Op::LogicalAnd:
		case Op::LogicalXor:
		case Op::LogicalNot:
		case Op::FPOrdEqual32:
		case Op::FPOrdEqual64:
		case Op::FPOrdLessThanEqual64:
		case Op::FPOrdGreaterThanEqual64:
		case Op::FPUnordEqual32:
		case Op::FPOrdNotEqual32:
		case Op::FPUnordNotEqual32:
		case Op::FPOrdLessThan32:
		case Op::FPUnordLessThan32:
		case Op::FPOrdGreaterThan32:
		case Op::FPUnordGreaterThan32:
		case Op::FPOrdLessThanEqual32:
		case Op::FPUnordLessThanEqual32:
		case Op::FPOrdGreaterThanEqual32:
		case Op::FPUnordGreaterThanEqual32:
		case Op::FPIsNan32:
		case Op::FPCmpClass32:
		case Op::FPCmpClass16:
		case Op::FPAdd32:
		case Op::FPSub32:
		case Op::FPFma32:
		case Op::FPMad32:
		case Op::FPMul32:
		case Op::FPAdd64:
		case Op::FPMul64:
		case Op::FPFma64:
		case Op::FPRecip64:
		case Op::FPMin32:
		case Op::FPMax32:
		case Op::FPMin64:
		case Op::FPMax64:
		case Op::FPMinTri32:
		case Op::FPMaxTri32:
		case Op::FPMedTri32:
		case Op::FPRecip32:
		case Op::FPRecipIFlag32:
		case Op::FPRecipSqrt32:
		case Op::FPSqrt:
		case Op::FPSin:
		case Op::FPCos:
		case Op::FPExp2:
		case Op::FPLog2:
		case Op::FPLdexp:
		case Op::FPRoundEven32:
		case Op::FPFloor32:
		case Op::FPCeil32:
		case Op::FPTrunc32:
		case Op::FPFloor64:
		case Op::FPCeil64:
		case Op::FPTrunc64:
		case Op::FPFract32:
		case Op::FPFract64: return Rule::Pure;
		default: return Rule::Never;
	}
}

struct Candidate {
	const IR::Inst* inst = nullptr;
	Rule            rule = Rule::Never;
};

// Whether the operand `value` is the same in both halves under the current assumptions.
bool OperandUniform(const std::unordered_set<const IR::Inst*>& uniform, IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		// Constants are; a register-typed operand is not an IR value at all.
		const auto type = value.GetType();
		return type != IR::Type::ScalarReg && type != IR::Type::VectorReg;
	}
	const auto* operand = value.TryInstruction();
	return operand != nullptr && uniform.contains(operand);
}

bool StillUniform(const std::unordered_set<const IR::Inst*>& uniform, const Candidate& candidate) {
	const auto& inst = *candidate.inst;
	switch (candidate.rule) {
		case Rule::Source:
		case Rule::Whole: return true;
		case Rule::ReadLane: return inst.NumArgs() > 1 && OperandUniform(uniform, inst.Arg(1));
		case Rule::Never: return false;
		case Rule::Pure:
		case Rule::Handle:
		case Rule::Phi:
			for (size_t index = 0; index < inst.NumArgs(); index++) {
				if (!OperandUniform(uniform, inst.Arg(index))) return false;
			}
			return true;
	}
	return false;
}

} // namespace

void LaneShareClassify(EmitterState& state) {
	auto& share = state.lane_share;
	share       = {};
	if (state.lane_count != 2 || state.program.dispatcher_fallback ||
	    state.program.stage != ShaderType::Compute) {
		return;
	}
	share.active = true;

	std::vector<Candidate> candidates;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			share.instructions++;
			const auto rule = RuleFor(state, inst);
			// Only values and typed handles can be shared; stores, barriers and the like have no result.
			if (rule == Rule::Never || inst.GetType() == IR::Type::Void) continue;
			candidates.push_back({&inst, rule});
		}
	}
	// Optimistic fixed point: every candidate starts out uniform; an instruction with an operand that is
	// not uniform is dropped, and so on until nothing changes (a loop phi whose back-edge value depends on
	// itself and on uniform values only stays uniform).
	auto& uniform = share.uniform;
	uniform.reserve(candidates.size() * 2);
	for (const auto& candidate: candidates) uniform.insert(candidate.inst);
	for (bool changed = true; changed;) {
		changed = false;
		for (const auto& candidate: candidates) {
			if (uniform.contains(candidate.inst) && !StillUniform(uniform, candidate)) {
				uniform.erase(candidate.inst);
				changed = true;
			}
		}
	}
	for (const auto& candidate: candidates) {
		if (!uniform.contains(candidate.inst)) continue;
		share.candidates++;
		if (candidate.rule == Rule::Pure || candidate.rule == Rule::Source ||
		    candidate.rule == Rule::ReadLane || candidate.rule == Rule::Phi) {
			share.shared.insert(candidate.inst);
		}
	}
}

bool LaneShareReuse(ValueEmitContext& low, ValueEmitContext& high, const IR::Inst& inst,
                    size_t low_words) {
	auto& share = low.state.lane_share;
	if (!share.active || !share.shared.contains(&inst)) return false;
	// A shared instruction without a value of its own (a planning-only load) is emitted as before.
	const auto found = inst.Definition(low.half);
	if (found == 0u) return false;
	high.Define(inst, found);
	share.reused++;
	share.words_saved += low_words;
	return true;
}

void LaneShareReport(EmitterState& state) {
	const auto& share = state.lane_share;
	if (!share.active) return;
	static std::atomic_uint reports {0};
	if (reports.fetch_add(1) >= 64u) return; // a long game session compiles many permutations
	std::fprintf(stderr, "LANE2_UNIFORM_SHARE hash=%016llx instructions=%zu uniform=%zu shared=%zu reused=%llu words_saved=%llu\n",
	             static_cast<unsigned long long>(state.program.shader_hash), share.instructions, share.uniform.size(),
	             share.shared.size(), static_cast<unsigned long long>(share.reused),
	             static_cast<unsigned long long>(share.words_saved));
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
