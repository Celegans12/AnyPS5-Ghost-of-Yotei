#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

DppTargetLane EmitDppPermTargetLane(EmitterState& state, uint32_t subid, uint32_t control,
                                    uint32_t lane_bits) {
	// Each lane of a group takes its lane_bits-wide source index from control.
	const auto lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), subid,
	                         ConstantU32(state, (1u << lane_bits) - 1u));
	const auto selected = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
	    state, ConstantU32(state, control),
	    Binary(state, spv::OpIMul, TypeU32(state), lane, ConstantU32(state, lane_bits)),
	    ConstantU32(state, lane_bits));
	return {EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
	            state, subid, selected, ConstantU32(state, 0), ConstantU32(state, lane_bits)),
	        ConstantBool(state, true)};
}

DppTargetLane EmitDppRowShiftTargetLane(EmitterState& state, uint32_t subid, uint32_t amount,
                                        bool left) {
	const auto row          = state.builder.AllocateId();
	const auto lane         = state.builder.AllocateId();
	const auto lane_shifted = state.builder.AllocateId();
	const auto target       = state.builder.AllocateId();
	const auto valid        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	if (left) {
		state.builder.AddFunction(spv::OpIAdd, TypeU32(state), lane_shifted, lane,
		                          ConstantU32(state, amount));
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), valid, lane,
		                          ConstantU32(state, 16u - amount));
	} else {
		state.builder.AddFunction(spv::OpISub, TypeU32(state), lane_shifted, lane,
		                          ConstantU32(state, amount));
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), valid, lane,
		                          ConstantU32(state, amount));
	}
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row, lane_shifted);
	return {target, valid};
}

DppTargetLane EmitDppRowRotateRightTargetLane(EmitterState& state, uint32_t subid,
                                              uint32_t amount) {
	// Rotation wraps within the row, so only the low four lane bits change.
	const auto rotated = Binary(state, spv::OpISub, TypeU32(state), subid, ConstantU32(state, amount));
	return {EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(state, subid, rotated,
	                                                        ConstantU32(state, 0),
	                                                        ConstantU32(state, 4)),
	        ConstantBool(state, true)};
}

DppTargetLane EmitDppMirrorTargetLane(EmitterState& state, uint32_t subid, bool half_row) {
	// Mirroring lane i of a row or half row reads lane (size - 1 - i), which flips its low bits.
	return {Binary(state, spv::OpBitwiseXor, TypeU32(state), subid,
	               ConstantU32(state, half_row ? 7u : 15u)),
	        ConstantBool(state, true)};
}

DppTargetLane EmitDppTargetLane(EmitterState& state, const IR::DppMoveFlags& flags) {
	const auto subid = EmitSubgroupLocalInvocationId(state);
	const auto control = flags.control;
	if (flags.dpp8) {
		return EmitDppPermTargetLane(state, subid, control, 3u);
	}
	if (control <= 0xffu) {
		return EmitDppPermTargetLane(state, subid, control, 2u);
	}
	if (control >= 0x101u && control <= 0x10fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, true);
	}
	if (control >= 0x111u && control <= 0x11fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, false);
	}
	if (control >= 0x121u && control <= 0x12fu) {
		return EmitDppRowRotateRightTargetLane(state, subid, control & 0xfu);
	}
	if (control == 0x140u) {
		return EmitDppMirrorTargetLane(state, subid, false);
	}
	if (control == 0x141u) {
		return EmitDppMirrorTargetLane(state, subid, true);
	}
	// row_share:N reads lane N of the caller's row of 16.
	if (control >= 0x150u && control <= 0x15fu) {
		return {EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
		            state, subid, ConstantU32(state, control & 0xfu), ConstantU32(state, 0),
		            ConstantU32(state, 4)),
		        ConstantBool(state, true)};
	}
	if (control >= 0x160u && control <= 0x16fu) {
		const auto target = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), target, subid,
		                          ConstantU32(state, control & 0xfu));
		return {target, ConstantBool(state, true)};
	}
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 16u) {
		std::fprintf(stderr, "DPP_CONTROL_AS_IDENTITY hash=%016llx control=0x%x\n",
		             static_cast<unsigned long long>(state.program.shader_hash), control);
	}
	return {subid, ConstantBool(state, true)};
}

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state) {
	if (state.subgroup_local_invocation_id_variable == 0) {
		EXIT("SubgroupLocalInvocationId was not declared before SPIR-V function emission\n");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value,
	                          state.subgroup_local_invocation_id_variable);
	if (state.wave_pair) {
		// Guest lane = wave half * 32 + host subgroup lane.
		return EmitBinaryU32(state, spv::OpBitwiseOr, value,
		                     EmitBinaryU32(state, spv::OpShiftLeftLogical, EmitPairHalf(state),
		                                   ConstantU32(state, 5)));
	}
	return state.lane_half == 0 ? value : EmitAddU32(state, value, ConstantU32(state, 32));
}

	// Default: the shaders measured faster and verified against refinterp (d4288915 added after
	// the A16/D16 refinterp support: baseline and pair captures give equal statistics); =0 turns it off.
	// Also the five "Eval non-RT Probes" shaders (GPU-tail; refinterp as without pairs; live A/B
	// at the gate +1.13% (se 0.40), run 20261022-174659).
	// 6cc64dee (GPU tail, 64 threads, with the barrier exchanges): KYTY_DIAG_TWIN_DISPATCH 178 twins / 442.6M words / 0
	// differ; 1.21 -> 0.92 ms/frame; live A/B (this shader only) +0.40% se 0.48.
bool WavePairRequested(uint64_t shader_hash) {
	static constexpr std::array<std::string_view, 11> Prefixes = {"c6b0a54e", "1158e78f", "da4ff122", "54904fb4",
	                                                              "d4288915", "85312e99", "3ab9b653", "8368b02a",
	                                                              "2b018f33", "40395313", "6cc64dee"};
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(shader_hash));
	for (const auto prefix: Prefixes) {
		if (std::string_view(text, prefix.size()) == prefix) return true;
	}
	return false;
}

// The pair rule: besides the list above, every wave64 compute shader whose workgroup is one guest wave (64
// threads: the exchanges take the workgroup barrier) and whose guest code has 2048 words or more is paired - one guest
// lane per invocation halves the SPIR-V the driver compiles. Meter (the 48 compute records of a 29.10 play): 37.8 ->
// 28.2 s of driver time, the two giants 5.8 / 5.4 -> 2.7 / 2.7 s; live A/B open world +0.39 % +- 0.19 (all sizes: -2.46 %,
// the 1024..2048-word ones cost the card time; 1024: -1.69 %), gate +0.56 % +- 0.24 (1024). LDS writes are synced across
// the halves (EmitPairLdsWriteSync), without which the blur CS 099c56ee made black particles.
constexpr size_t PairRuleMinWords = 2048;

bool WavePairRule(uint32_t threads, size_t code_words) {
	return threads == 64u && code_words >= PairRuleMinWords;
}

// The shaders the rule paired (KYTY_DIAG_TWIN_DISPATCH=rule twins them).
namespace {
std::mutex                   g_rule_paired_mutex;
std::unordered_set<uint64_t> g_rule_paired;
} // namespace

void NoteRulePaired(uint64_t shader_hash) {
	std::lock_guard lock(g_rule_paired_mutex);
	g_rule_paired.insert(shader_hash);
}

uint32_t EmitSubgroupIdU32(EmitterState& state) {
	if (state.subgroup_id_variable == 0) {
		EXIT("SubgroupId was not declared before SPIR-V function emission\n");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, state.subgroup_id_variable);
	return value;
}

uint32_t EmitPairHalf(EmitterState& state) {
	return EmitBinaryU32(state, spv::OpBitwiseAnd, EmitSubgroupIdU32(state), ConstantU32(state, 1));
}

namespace {

uint32_t PairSlotPointer(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassWorkgroup, TypeU32(state)), pointer,
	                          state.pair_slots_variable, index);
	return pointer;
}

// Slot layout per guest wave (8 dwords): [parity][half] -> {value, sequence}.
uint32_t PairSlotIndex(EmitterState& state, uint32_t wave, uint32_t parity, uint32_t half,
                       uint32_t field) {
	auto index = EmitBinaryU32(state, spv::OpShiftLeftLogical, wave, ConstantU32(state, 3));
	index      = EmitAddU32(state, index,
	                        EmitBinaryU32(state, spv::OpShiftLeftLogical, parity, ConstantU32(state, 2)));
	index      = EmitAddU32(state, index,
	                        EmitBinaryU32(state, spv::OpShiftLeftLogical, half, ConstantU32(state, 1)));
	return field == 0u ? index : EmitAddU32(state, index, ConstantU32(state, field));
}

template <typename Fn>
void EmitIfElected(EmitterState& state, Fn&& fn) {
	const auto elected = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected,
	                          ConstantU32(state, spv::ScopeSubgroup));
	EmitIfCondition(state, elected, std::forward<Fn>(fn));
}

} // namespace

void EmitPairInitialize(EmitterState& state) {
	if (!state.wave_pair) return;
	state.builder.AddFunction(spv::OpStore, state.pair_seq_variable, ConstantU32(state, 0));
	const auto subgroup = EmitSubgroupIdU32(state);
	const auto wave     = EmitBinaryU32(state, spv::OpShiftRightLogical, subgroup, ConstantU32(state, 1));
	const auto half     = EmitBinaryU32(state, spv::OpBitwiseAnd, subgroup, ConstantU32(state, 1));
	EmitIfElected(state, [&]() {
		for (uint32_t parity = 0; parity < 2u; parity++) {
			state.builder.AddFunction(
			    spv::OpStore,
			    PairSlotPointer(state, PairSlotIndex(state, wave, ConstantU32(state, parity), half, 1)),
			    ConstantU32(state, 0));
		}
	});
	// Every invocation passes here before any guest code, so the barrier is in uniform control flow.
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
	                                                 spv::MemorySemanticsWorkgroupMemoryMask));
}

uint32_t EmitPairExchange(EmitterState& state, uint32_t value) {
	// Both halves of a guest wave execute the same exchanges in the same order (all branches are
	// whole-wave decisions), so a per-invocation sequence number pairs them. Two parity slots
	// suffice: a half overwrites a slot only after reading its partner's next value, which the
	// partner publishes only after it has read this slot.
	const auto previous = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), previous, state.pair_seq_variable);
	const auto sequence = EmitAddU32(state, previous, ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpStore, state.pair_seq_variable, sequence);
	const auto subgroup = EmitSubgroupIdU32(state);
	const auto wave     = EmitBinaryU32(state, spv::OpShiftRightLogical, subgroup, ConstantU32(state, 1));
	const auto half     = EmitBinaryU32(state, spv::OpBitwiseAnd, subgroup, ConstantU32(state, 1));
	const auto other    = EmitBinaryU32(state, spv::OpBitwiseXor, half, ConstantU32(state, 1));
	const auto parity   = EmitBinaryU32(state, spv::OpBitwiseAnd, sequence, ConstantU32(state, 1));
	const auto scope    = ConstantU32(state, spv::ScopeWorkgroup);
	const auto relaxed  = ConstantU32(state, spv::MemorySemanticsMaskNone);
	const auto release  = ConstantU32(state, spv::MemorySemanticsReleaseMask |
	                                             spv::MemorySemanticsWorkgroupMemoryMask);
	const auto acquire  = ConstantU32(state, spv::MemorySemanticsAcquireMask |
	                                             spv::MemorySemanticsWorkgroupMemoryMask);
	// The value (a vote, a ballot, one lane's value) and the sequence number are the same in the whole subgroup, so every
	// invocation publishes them: no branch. The order against the partner comes from the barrier or the spin below.
	state.builder.AddFunction(spv::OpAtomicStore,
	                          PairSlotPointer(state, PairSlotIndex(state, wave, parity, half, 0)),
	                          scope, relaxed, value);
	state.builder.AddFunction(spv::OpAtomicStore,
	                          PairSlotPointer(state, PairSlotIndex(state, wave, parity, half, 1)),
	                          scope, release, sequence);
	const auto partner_sequence = PairSlotPointer(state, PairSlotIndex(state, wave, parity, other, 1));
	const auto partner_value    = PairSlotPointer(state, PairSlotIndex(state, wave, parity, other, 0));
	// In a paired shader whose workgroup is one guest wave (64 threads: the two host subgroups are the whole workgroup),
	// outside per-invocation guards every invocation of the workgroup is here, so a workgroup barrier orders the elected
	// stores above before the partner's loads below (Volfog 0.99 -> 0.88 ms; 97.8M exchanges checked, 0 sequence
	// mismatches). Both halves reach every exchange (whole-wave decisions), so such a barrier is in dynamically uniform
	// control flow. A slot of parity p is rewritten two exchanges later, behind the next exchange's barrier, which the
	// partner reaches after its load.
	if (state.pair_barrier && state.guard_depth == 0u) {
		state.builder.AddFunction(spv::OpControlBarrier, scope, scope,
		                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
		                                                 spv::MemorySemanticsWorkgroupMemoryMask));
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), result, partner_value, scope, relaxed);
		return result;
	}
	// Spin until the partner publishes this sequence number; bounded so a broken pairing ends
	// with a wrong value instead of a GPU hang.
	const auto preheader      = state.current_label;
	const auto header         = state.builder.AllocateId();
	const auto body           = state.builder.AllocateId();
	const auto cont           = state.builder.AllocateId();
	const auto merge          = state.builder.AllocateId();
	const auto iteration      = state.builder.AllocateId();
	const auto next_iteration = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, header);
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), iteration, ConstantU32(state, 0), preheader,
	                          next_iteration, cont);
	state.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranch, body);
	EmitLabel(state, body);
	const auto observed = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), observed, partner_sequence, scope,
	                          acquire);
	const auto arrived = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), arrived, observed, sequence);
	state.builder.AddFunction(spv::OpIAdd, TypeU32(state), next_iteration, iteration,
	                          ConstantU32(state, 1));
	const auto over = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), over, iteration,
	                          ConstantU32(state, 1u << 20u));
	const auto done = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), done, arrived, over);
	state.builder.AddFunction(spv::OpBranchConditional, done, merge, cont);
	EmitLabel(state, cont);
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, merge);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), result, partner_value, scope, relaxed);
	return result;
}

uint32_t InputVariableForKind(const EmitterState& state, IR::StageInputKind kind) {
	for (const auto& input: state.inputs) {
		if (input.kind == kind) {
			return input.variable_id;
		}
	}
	return 0;
}

const ShaderIndirectDrawInfo* IndirectDrawVariant(const EmitterState& state) {
	if (state.program.stage != ShaderType::Vertex || state.input_info.vertex == nullptr ||
	    !state.input_info.vertex->indirect_draw.enabled) {
		return nullptr;
	}
	return &state.input_info.vertex->indirect_draw;
}

const InputBinding* InputBindingForParameter(const EmitterState& state, uint32_t location) {
	for (const auto& input: state.inputs) {
		if (input.kind == IR::StageInputKind::Parameter && input.location == location) {
			return &input;
		}
	}
	return nullptr;
}

uint32_t EmitInputComponentU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	const auto variable = InputVariableForKind(state, kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassInput, TypeU32(state)), pointer,
	                          variable, ConstantU32(state, component));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t EmitLocalInvocationIndex(EmitterState& state) {
	if (state.wave_pair) {
		// The flattened workgroup numbers guest invocations by host subgroup, so each guest
		// wave is exactly two host subgroups (EmitPairHalf).
		const auto lane = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), lane,
		                          state.subgroup_local_invocation_id_variable);
		return EmitAddU32(state, lane,
		                  EmitBinaryU32(state, spv::OpShiftLeftLogical, EmitSubgroupIdU32(state),
		                                ConstantU32(state, 5)));
	}
	const auto variable = InputVariableForKind(state, IR::StageInputKind::LocalInvocationIndex);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, variable);
	if (state.lane_count == 2) {
		const auto wave_base =
		    EmitBinaryU32(state, spv::OpBitwiseAnd, value, ConstantU32(state, ~31u));
		return EmitAddU32(state, EmitAddU32(state, value, wave_base),
		                  ConstantU32(state, state.lane_half * 32));
	}
	return value;
}

uint32_t EmitVertexParameterComponentU32(EmitterState& state, const InputBinding& input,
                                         uint32_t component) {
	const auto count = VertexParameterComponentCount(input);
	const auto kind  = VertexParameterScalarKind(state, input.location);
	const auto scalar_type = VertexParameterScalarType(state, kind);
	uint32_t   raw         = state.builder.AllocateId();
	if (count == 1u) {
		state.builder.AddFunction(spv::OpLoad, scalar_type, raw, input.variable_id);
	} else {
		const auto pointer_type = TypePointer(state, spv::StorageClassInput, scalar_type);
		const auto pointer      = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, input.variable_id,
		                          ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, scalar_type, raw, pointer);
	}

	if (kind == VertexInputScalarKind::Uint) {
		return raw;
	}

	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, raw);
	return bits;
}

uint32_t EmitSubgroupLaneActiveBool(EmitterState& state, uint32_t lane) {
	const auto active_ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), active_ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), ConstantBool(state, true));
	if (state.wave_pair) {
		// Callers read lanes of their own half (row and 32-lane swizzles): test the host bit.
		const auto word = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), word, active_ballot, 0);
		const auto bit = EmitBinaryU32(state, spv::OpShiftLeftLogical, ConstantU32(state, 1),
		                               EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31)));
		const auto active = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active,
		                          EmitBinaryU32(state, spv::OpBitwiseAnd, word, bit), ConstantU32(state, 0));
		return active;
	}
	return EmitBallotLaneActiveBool(state, active_ballot, lane);
}
uint32_t EmitBallotLaneActiveBool(EmitterState& state, uint32_t active_ballot, uint32_t lane) {
	const auto low = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, active_ballot, 0);
	uint32_t mask = low;
	if (state.program.wave_size == 64u) {
		const auto high     = state.builder.AllocateId();
		const auto in_high  = state.builder.AllocateId();
		const auto selected = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, active_ballot, 1);
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), in_high, lane,
		                          ConstantU32(state, 32));
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, in_high, high, low);
		mask = selected;
	}

	const auto lane_low = state.builder.AllocateId();
	const auto bit      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane_low, lane,
	                          ConstantU32(state, 31));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), bit, ConstantU32(state, 1),
	                          lane_low);

	const auto hit      = state.builder.AllocateId();
	const auto active   = state.builder.AllocateId();
	const auto in_range = state.builder.AllocateId();
	const auto ret      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), hit, mask, bit);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_range, lane,
	                          ConstantU32(state, state.program.wave_size));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), ret, active, in_range);
	return ret;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

namespace Libs::Graphics::ShaderRecompiler {

bool IR::WavePairPlanned(uint64_t shader_hash, uint32_t threads, size_t code_words) {
	return Spirv::Emitter::WavePairRequested(shader_hash) || Spirv::Emitter::WavePairRule(threads, code_words);
}

bool IR::WavePairRulePaired(uint64_t shader_hash) {
	std::lock_guard lock(Spirv::Emitter::g_rule_paired_mutex);
	return Spirv::Emitter::g_rule_paired.contains(shader_hash);
}

} // namespace Libs::Graphics::ShaderRecompiler
