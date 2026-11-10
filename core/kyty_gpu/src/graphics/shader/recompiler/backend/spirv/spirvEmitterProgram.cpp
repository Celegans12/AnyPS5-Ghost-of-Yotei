#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/assert.h"
#include "common/shaderEnv.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

void EmitKillIfBoolFalse(EmitterState& state, uint32_t active) {
	const auto kill_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, active, merge_label, kill_label);
	EmitLabel(state, kill_label);
	state.builder.AddFunction(spv::OpKill);
	EmitLabel(state, merge_label);
}

void EmitKillIfPixelValidMaskInactive(EmitterState& state) {
	if (state.pixel_valid_mask_variable == 0) {
		return;
	}

	const auto mask_value = state.builder.AllocateId();
	const auto active     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), mask_value,
	                          state.pixel_valid_mask_variable);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, mask_value,
	                          ConstantU32(state, 0));
	EmitKillIfBoolFalse(state, active);
}

uint32_t SpillPointerType(ValueEmitContext& ctx, IR::Type type) {
	const auto value_type = TypeId(ctx.state, type);
	return value_type == 0 ? 0 : TypePointer(ctx.state, spv::StorageClassFunction, value_type);
}

struct DeferredPhiPatch {
	DeferredPhi     phi;
	const IR::Inst* instruction = nullptr;
	uint32_t        half        = 0;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t> block_exit_labels;
	std::vector<DeferredPhiPatch>                  deferred_phis;
};

struct DispatcherFunctionState {
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2> spills;
	uint32_t                                      header_label       = 0;
	uint32_t                                      select_label       = 0;
	uint32_t                                      after_switch_label = 0;
	uint32_t                                      continue_label     = 0;
	uint32_t                                      merge_label        = 0;
};

void StoreDispatcherPhiEdge(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                            const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) {
			break;
		}
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == from) {
				ctx.state.builder.AddFunction(spv::OpStore, dispatcher.spills[ctx.half].at(&phi),
				                              ctx.Def(phi.Arg(index)));
				break;
			}
		}
	}
}

uint32_t LoopCapReached(ValueEmitContext& ctx, uint32_t variable) {
	auto&      state   = ctx.state;
	const auto counter = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), counter, variable);
	const auto over = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), over, counter,
	                          ConstantU32(state, ShaderLoopLimitFor(state.program.shader_hash)));
	return over;
}
// Force an exit the loop already has once its back edges have run ShaderLoopLimit() times.
uint32_t GuardLoopCondition(ValueEmitContext& ctx, const IR::Block* block, uint32_t condition) {
	auto&      state = ctx.state;
	const auto found = state.loop_counter_variables.find(block->id);
	if (found == state.loop_counter_variables.end()) {
		return condition;
	}
	const auto [variable, exit_block] = found->second;
	const auto over                   = LoopCapReached(ctx, variable);
	const auto* true_block            = block->terminator.true_block;
	const bool  exit_when_true        = true_block != nullptr && true_block->id == exit_block;
	const auto  guarded               = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeBool(state), guarded, over,
	                          ConstantBool(state, exit_when_true), condition);
	return guarded;
}
// A forced exit from an unconditional latch is a NEW edge into the exit block, so a phi
// there would be left one incoming value short and the module fails validation. Those loops
// keep whatever conditional exit they already had instead.
bool BlockStartsWithPhi(const IR::Block* block) {
	if (block == nullptr) {
		return true;
	}
	for (const auto& inst: *block) {
		return inst.GetOpcode() == IR::ValueOpcode::Phi;
	}
	return false;
}
const IR::Block* BlockById(const IR::Program& program, uint32_t id) {
	const auto found =
	    std::ranges::find_if(program.blocks, [&](const IR::Block* block) { return block->id == id; });
	return found == program.blocks.end() ? nullptr : *found;
}

void EmitReturn(ValueEmitContext& ctx) {
	EmitKillIfPixelValidMaskInactive(ctx.state);
	if (ctx.state.object_primitive_id_variable != 0) {
		auto& state = ctx.state;
		const auto primitive = EmitGetBuiltin(
		    ctx, IR::Value {static_cast<uint32_t>(IR::StageInputKind::PrimitiveId)}, IR::Value {0u});
		const auto object = EmitPushDataLoad(state, ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.object_primitive_id_variable,
		                          EmitAddU32(state, object, primitive));
	}
	ctx.state.builder.AddFunction(spv::OpReturn);
}

void EmitStructuredTerminator(ValueEmitContext& ctx, const IR::Block* block) {
	const auto& term       = block->terminator;
	const auto  emit_merge = [&]() {
		if (term.loop_header) {
			const auto* merge = term.merge_block;
			const auto* cont  = term.continue_block;
			if (merge != nullptr && cont != nullptr) {
				ctx.state.builder.AddFunction(spv::OpLoopMerge, merge->Definition(),
				                              cont->Definition(), spv::LoopControlMaskNone);
			}
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           term.merge_block != nullptr) {
			ctx.state.builder.AddFunction(spv::OpSelectionMerge, term.merge_block->Definition(),
			                              spv::SelectionControlMaskNone);
		}
	};

	switch (term.kind) {
		case CFG::TerminatorKind::Branch: {
			const auto* target = term.true_block;
			if (target == nullptr) {
				EmitReturn(ctx);
				return;
			}
			emit_merge();
			// An unconditional latch has no exit of its own, so give it one at the cap.
			const auto latch = ctx.state.loop_counter_variables.find(block->id);
			if (const auto* exit = latch != ctx.state.loop_counter_variables.end()
			                           ? BlockById(ctx.state.program, latch->second.second)
			                           : nullptr;
			    exit != nullptr && !BlockStartsWithPhi(exit)) {
				ctx.state.builder.AddFunction(spv::OpBranchConditional,
				                              LoopCapReached(ctx, latch->second.first),
				                              exit->Definition(), target->Definition());
				return;
			}
			ctx.state.builder.AddFunction(spv::OpBranch, target->Definition());
			return;
		}
		case CFG::TerminatorKind::ConditionalBranch: {
			const auto* true_block  = term.true_block;
			const auto* false_block = term.false_block;
			if (true_block == nullptr || false_block == nullptr || block->condition.IsEmpty()) {
				EmitReturn(ctx);
				return;
			}
			const auto condition = GuardLoopCondition(ctx, block, ctx.Def(block->condition));
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranchConditional, condition,
			                              true_block->Definition(), false_block->Definition());
			return;
		}
		default: EmitReturn(ctx); return;
	}
}

void EmitDispatcherTarget(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                          const IR::Block* from, const IR::Block* block) {
	if (block != nullptr) {
		StoreDispatcherPhiEdge(ctx, dispatcher, from, block);
		if (ctx.other_half != nullptr) {
			StoreDispatcherPhiEdge(*ctx.other_half, dispatcher, from, block);
		}
	}
}

uint32_t EmitDispatcherNextPc(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                              const IR::Block* block) {
	const auto& term = block->terminator;
	switch (term.kind) {
		case CFG::TerminatorKind::Branch:
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			return ConstantU32(ctx.state, term.true_block->id);
		case CFG::TerminatorKind::ConditionalBranch: {
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			EmitDispatcherTarget(ctx, dispatcher, block, term.false_block);
			const auto selected = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), selected,
			                              ctx.Def(block->condition),
			                              ConstantU32(ctx.state, term.true_block->id),
			                              ConstantU32(ctx.state, term.false_block->id));
			return selected;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			for (const auto* target: block->ImmSuccessors()) {
				EmitDispatcherTarget(ctx, dispatcher, block, target);
			}
			uint32_t selected = ConstantU32(ctx.state, UINT32_MAX);
			if (!block->indirect_target.IsEmpty()) {
				const auto selector = ctx.Def(block->indirect_target);
				for (const auto& branch: term.cases) {
					const auto match = ctx.state.builder.AllocateId();
					const auto next  = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), match,
					                              selector, ConstantU32(ctx.state, branch.value));
					ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), next, match,
					                              ConstantU32(ctx.state, branch.target->id),
					                              selected);
					selected = next;
				}
			}
			return selected;
		}
		default: return ConstantU32(ctx.state, UINT32_MAX);
	}
}

template <typename T>
decltype(auto) Arg(ValueEmitContext& ctx, const IR::Inst& inst, size_t index) {
	if constexpr (std::is_same_v<T, const IR::Inst&>) {
		return inst;
	} else if constexpr (std::is_same_v<T, IR::Value>) {
		return inst.Arg(index);
	} else if constexpr (std::is_same_v<T, IR::ScalarReg>) {
		return inst.Arg(index).ScalarRegister();
	} else {
		static_assert(std::is_same_v<T, uint32_t>);
		return ctx.Def(inst.Arg(index));
	}
}

template <typename Context, typename Return, typename... Args>
void Invoke(Return (*emit)(Context&, Args...), ValueEmitContext& ctx, const IR::Inst& inst) {
	// A full instruction keeps metadata and predicated/lane operand loads lazy.
	static_assert(std::is_same_v<Context, ValueEmitContext> ||
	              std::is_same_v<Context, EmitterState>);
	auto& context = [&]() -> Context& {
		if constexpr (std::is_same_v<Context, EmitterState>)
			return ctx.state;
		else
			return ctx;
	}();
	constexpr bool has_inst = (std::is_same_v<Args, const IR::Inst&> || ...);
	[&]<size_t... I>(std::index_sequence<I...>) {
		static_assert(((!std::is_same_v<Args, const IR::Inst&> || I == 0) && ...));
		const auto call = [&] {
			return emit(context, Arg<Args>(ctx, inst, I - (has_inst && I != 0))...);
		};
		if constexpr (std::is_void_v<Return>) {
			call();
		} else {
			static_assert(std::is_same_v<Return, uint32_t>);
			ctx.Define(inst, call());
		}
	}(std::index_sequence_for<Args...> {});
}

void EmitDirectInstructionImpl(ValueEmitContext& ctx, const IR::Inst& inst);

// KYTY_SWEEP_SPIRV_OPSTATS=1 (diagnostic): SPIR-V function words emitted per IR opcode, printed
// to stderr by PrintSpirvOpStats.
std::map<std::string, std::pair<uint64_t, uint64_t>>& SpirvOpStats() {
	static std::map<std::string, std::pair<uint64_t, uint64_t>> stats;
	return stats;
}

void PrintSpirvOpStats() {
	std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> sorted(SpirvOpStats().begin(),
	                                                                        SpirvOpStats().end());
	std::sort(sorted.begin(), sorted.end(),
	          [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
	for (size_t i = 0; i < sorted.size() && i < 40; i++)
		std::fprintf(stderr, "SPIRV_OPSTATS %s count=%llu words=%llu\n", sorted[i].first.c_str(),
		             static_cast<unsigned long long>(sorted[i].second.first),
		             static_cast<unsigned long long>(sorted[i].second.second));
	SpirvOpStats().clear();
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	static const bool stats = [] {
		const bool on = Common::ShaderEnv::Read("KYTY_SWEEP_SPIRV_OPSTATS") != nullptr;
		if (on) {
			(void)SpirvOpStats(); // construct first so it outlives the atexit report
			std::atexit(PrintSpirvOpStats);
		}
		return on;
	}();
	if (!stats) {
		EmitDirectInstructionImpl(ctx, inst);
		return;
	}
	const auto before = ctx.state.builder.FunctionWordCount();
	EmitDirectInstructionImpl(ctx, inst);
	auto& entry = SpirvOpStats()[std::to_string(static_cast<int>(inst.GetOpcode())) +
	                              (IR::BufferComponentCount(inst.GetOpcode()) != 0u && ctx.Memory(inst).formatted ? " formatted" : "")];
	// KYTY_SWEEP_SPIRV_SLICE=<opcode number>: prints the words of the 3rd emission of that opcode.
	static const int slice_opcode = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_SWEEP_SPIRV_SLICE");
		return value != nullptr ? std::atoi(value) : -1;
	}();
	if (static_cast<int>(inst.GetOpcode()) == slice_opcode)
		std::fprintf(stderr, "SPIRV_EMIT words=%zu kind=%u half=%u\n", ctx.state.builder.FunctionWordCount() - before,
		             IR::BufferComponentCount(inst.GetOpcode()) != 0u ? static_cast<uint32_t>(ctx.Memory(inst).kind) : 0u,
		             static_cast<uint32_t>(ctx.half));
	if (static_cast<int>(inst.GetOpcode()) == slice_opcode && entry.first == 2u) {
		const auto& words = ctx.state.builder.FunctionWords();
		std::fprintf(stderr, "SPIRV_SLICE");
		for (size_t i = before; i < words.size(); i++) std::fprintf(stderr, " %u", words[i]);
		std::fprintf(stderr, "\n");
	}
	entry.first++;
	entry.second += ctx.state.builder.FunctionWordCount() - before;
}

void EmitDirectInstructionImpl(ValueEmitContext& ctx, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto op = inst.GetOpcode();
	if (ctx.half != 0 &&
	    (op == Op::Ballot || op == Op::ReadFirstLane || op == Op::SubgroupUMinU32 || op == Op::SubgroupUMaxU32 ||
	     op == Op::SubgroupOrU32 || op == Op::ReadConstBuffer || op == Op::ReadRealtimeClock ||
	     (op == Op::LoadAddressU32 && ctx.Memory(inst).kind == IR::ResourceKind::ScalarAddress))) {
		// Whole-wave results and SMEM loads (SGPR operands, no EXEC) are identical in both halves.
		ctx.Define(inst, ctx.other_half->Result(inst));
		return;
	}
	switch (op) {
#define VALUE_OPCODE(name, ...)                                                                    \
	case IR::ValueOpcode::name: return Invoke(Emit##name, ctx, inst);
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: ctx.Fail(inst, "has no direct SPIR-V emitter");
	}
}

void EmitStructuredInstruction(ValueEmitContext& ctx, StructuredFunctionState& structured,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0 || inst.NumArgs() == 0) {
			ctx.Fail(inst, "has no native SPIR-V representation");
		}
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			const auto* predecessor = inst.PhiBlock(index);
			if (predecessor == nullptr || predecessor->Definition() == 0) {
				ctx.Fail(inst, "has a predecessor outside the structured function");
			}
		}
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), inst.NumArgs()), &inst,
		     ctx.half});
		return;
	}
	EmitDirectInstruction(ctx, inst);
}

void EmitDispatcherInstruction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0) {
			ctx.Fail(inst, "cannot be loaded by the dispatcher");
		}
		ctx.state.builder.AddFunction(spv::OpLoad, type, ctx.Result(inst),
		                              dispatcher.spills[ctx.half].at(&inst));
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (const auto found = dispatcher.spills[ctx.half].find(&inst);
	    found != dispatcher.spills[ctx.half].end()) {
		ctx.state.builder.AddFunction(spv::OpStore, found->second,
		                              ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))));
	}
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, block->Definition());
	bool emitted_non_phi = false;
	const bool lane_share = ctx.state.lane_share.active;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			emitted_non_phi = true;
		}
		size_t low_words = 0;
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::StoreCompletion &&
			                  inst.GetOpcode() != IR::ValueOpcode::DataCacheInvalidate &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				if (lane_share && half != 0 && LaneShareReuse(ctx, lane, inst, low_words)) continue;
				const size_t begin = ctx.state.builder.FunctionWordCount();
				emit_instruction(lane, inst);
				if (half == 0) low_words = ctx.state.builder.FunctionWordCount() - begin;
			}
		}
		ctx.state.lane_half = 0;
	}
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& deferred: structured.deferred_phis) {
		auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
		for (size_t index = 0; index < deferred.instruction->NumArgs(); index++) {
			const auto* predecessor = deferred.instruction->PhiBlock(index);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			ctx.state.builder.PatchDeferredPhi(
			    deferred.phi, index, lane.Def(deferred.instruction->Arg(index)), found->second);
		}
	}
}

// Every latch of a loop counts one iteration. GuardLoopCondition reads that count, so the
// forced exit is an edge the loop already has.
void EmitBackEdgeGuard(ValueEmitContext& ctx, const IR::Block* block) {
	auto&      state = ctx.state;
	const auto found = state.back_edge_counters.find(block->id);
	if (found == state.back_edge_counters.end()) {
		return;
	}
	const auto counter = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), counter, found->second);
	const auto next = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIAdd, TypeU32(state), next, counter, ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpStore, found->second, next);
}

// KYTY_DIAG_RT_TRIPS (DiagRtTripsFor): in a block that tests a loop's watchdog cap, an invocation that reached the
// cap counts the trip and reports the loop header's phi values (FaultManager logs WATCHDOG_TRIP). Emitted before the
// block's exit label is taken, so the selection it adds ends the block. Not in a loop header: its OpLoopMerge must stay
// in the block the back edge targets.
uint32_t PhiAsU32(ValueEmitContext& ctx, const IR::Inst& phi) {
	auto&      state = ctx.state;
	const auto value = ctx.Def(IR::Value(const_cast<IR::Inst*>(&phi)));
	switch (phi.GetType()) {
		case IR::Type::U32: return value;
		case IR::Type::F32: return Unary(state, spv::OpBitcast, TypeU32(state), value);
		case IR::Type::U64: return Unary(state, spv::OpUConvert, TypeU32(state), value);
		case IR::Type::U1: return Select(state, TypeU32(state), value, ConstantU32(state, 1), ConstantU32(state, 0));
		default: return ConstantU32(state, 0xdeadbeefu);
	}
}

void EmitWatchdogTripReport(ValueEmitContext& ctx, const IR::Block* block) {
	auto& state = ctx.state;
	if (!DiagRtTripsFor(state.program.shader_hash) || state.fault_buffer_variable == 0 || block->terminator.loop_header)
		return;
	const auto found = state.loop_counter_variables.find(block->id);
	if (found == state.loop_counter_variables.end()) return;
	const auto [variable, exit_id] = found->second;
	const auto& blocks             = state.program.blocks;
	const auto  index_of           = [&](const IR::Block* b) { return std::ranges::find(blocks, b) - blocks.begin(); };
	// The loop's header: an earlier loop header this block branches back to, else the one that merges at the exit.
	const IR::Block* header = nullptr;
	for (const auto* target: {block->terminator.true_block, block->terminator.false_block}) {
		if (target != nullptr && target->terminator.loop_header && index_of(target) <= index_of(block)) header = target;
	}
	if (header == nullptr) {
		for (const auto* candidate: blocks) {
			if (candidate->terminator.loop_header && candidate->terminator.merge_block != nullptr &&
			    candidate->terminator.merge_block->id == exit_id && index_of(candidate) <= index_of(block))
				header = candidate;
		}
	}
	std::vector<uint32_t> low, high;
	if (header != nullptr) {
		for (const auto& inst: *header) {
			if (inst.GetOpcode() != IR::ValueOpcode::Phi || low.size() == WatchdogTripValues()) break;
			low.push_back(PhiAsU32(ctx, inst));
			if (state.lane_count == 2 && ctx.other_half != nullptr) {
				high.push_back(PhiAsU32(*ctx.other_half, inst));
			} else {
				// One lane half: the second half of the report names each value's register (SsaRewrite's phi tag).
				high.push_back(ConstantU32(state, static_cast<uint32_t>(inst.Flags<uint64_t>() >> 32u)));
			}
		}
	}
	const auto counter = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), counter, variable);
	const auto group_variable = InputVariableForKind(state, IR::StageInputKind::WorkgroupId);
	uint32_t   group          = ConstantU32(state, 0xffffffffu);
	if (group_variable != 0) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassInput, TypeU32(state)), pointer,
		                          group_variable, ConstantU32(state, 0));
		group = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), group, pointer);
	}
	ReportWatchdogTrip(state, ShaderLoopLimitFor(state.program.shader_hash), block->id, counter,
	                   EmitLocalInvocationIndex(state), group, low, high);
}

void EmitStructuredFunction(ValueEmitContext& ctx) {
	const auto& program = ctx.state.program;
	StructuredFunctionState structured;
	ctx.state.builder.AddFunction(spv::OpBranch, program.blocks.front()->Definition());
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
		EmitBackEdgeGuard(ctx, block);
		EmitWatchdogTripReport(ctx, block); // KYTY_DIAG_RT_TRIPS
		structured.block_exit_labels.emplace(block, ctx.state.current_label);
		EmitStructuredTerminator(ctx, block);
	}
	PatchStructuredPhis(ctx, structured);
}

void EmitDispatcherFunction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher) {
	auto&       state = ctx.state;
	const auto* entry = state.program.blocks.front();
	state.builder.AddFunction(spv::OpBranch, entry->Definition());
	EmitBlock(ctx, entry, [&](ValueEmitContext& lane, const IR::Inst& inst) {
		EmitDispatcherInstruction(lane, dispatcher, inst);
	});
	const auto initial_pc     = EmitDispatcherNextPc(ctx, dispatcher, entry);
	const auto initial_parent = state.current_label;
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);

	state.current_block = nullptr;
	EmitLabel(state, dispatcher.header_label);
	const auto pc      = state.builder.AllocateId();
	const auto next_pc = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), pc, initial_pc, initial_parent, next_pc,
	                          dispatcher.continue_label);
	// Phis must lead the block, so the iteration counter is declared before the exit test.
	uint32_t iteration      = 0;
	uint32_t next_iteration = 0;
	if (ShaderLoopLimitFor(state.program.shader_hash) != 0) {
		iteration      = state.builder.AllocateId();
		next_iteration = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpPhi, TypeU32(state), iteration, ConstantU32(state, 0),
		                          initial_parent, next_iteration, dispatcher.continue_label);
	}
	auto done = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), done, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
	if (ShaderLoopLimitFor(state.program.shader_hash) != 0) {
		state.builder.AddFunction(spv::OpIAdd, TypeU32(state), next_iteration, iteration,
		                          ConstantU32(state, 1));
		const auto over = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), over, iteration,
		                          ConstantU32(state, ShaderLoopLimitFor(state.program.shader_hash)));
		const auto stop = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), stop, done, over);
		done = stop;
	}
	state.builder.AddFunction(spv::OpLoopMerge, dispatcher.merge_label, dispatcher.continue_label,
	                          spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, done, dispatcher.merge_label,
	                          dispatcher.select_label);

	EmitLabel(state, dispatcher.select_label);
	state.builder.AddFunction(spv::OpSelectionMerge, dispatcher.after_switch_label,
	                          spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, pc, dispatcher.after_switch_label};
	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		words.push_back(state.program.blocks[index]->id);
		words.push_back(state.program.blocks[index]->Definition());
	}
	state.builder.AddFunction(words);
	std::vector<uint32_t> next_pc_words {spv::OpPhi, TypeU32(state), next_pc,
	                                     ConstantU32(state, UINT32_MAX), dispatcher.select_label};

	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		EmitBlock(ctx, state.program.blocks[index],
		          [&](ValueEmitContext& lane, const IR::Inst& inst) {
			          EmitDispatcherInstruction(lane, dispatcher, inst);
		          });
		const auto selected = EmitDispatcherNextPc(ctx, dispatcher, state.program.blocks[index]);
		next_pc_words.push_back(selected);
		next_pc_words.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, dispatcher.after_switch_label);
	}
	state.current_block = nullptr;
	EmitLabel(state, dispatcher.after_switch_label);
	state.builder.AddFunction(next_pc_words);
	state.builder.AddFunction(spv::OpBranch, dispatcher.continue_label);
	EmitLabel(state, dispatcher.continue_label);
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);
	EmitLabel(state, dispatcher.merge_label);
	EmitReturn(ctx);
}

// The ObjPrimID write: what the hardware writes to the ObjPrimID colour slot, VGT_OBJECT_ID (push constant dword) plus
// the primitive index of the draw when ADD_PIPED_PRIM_ID is set. Stored in the entry block: a fragment the guest code
// discards writes nothing, like any colour output.
void EmitObjprimIdOutput(EmitterState& state) {
	const auto& pixel   = *state.input_info.pixel;
	const auto  pointer = state.builder.AllocateId();
	auto        value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 0),
	                          ConstantU32(state, pixel.objprim_push_dword));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	if (pixel.objprim_add_prim_id) {
		const auto variable = InputVariableForKind(state, IR::StageInputKind::PrimitiveId);
		EXIT_IF(variable == 0);
		const auto primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), primitive, variable);
		const auto primitive_u32 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), primitive_u32, primitive);
		value = EmitAddU32(state, value, primitive_u32);
	}
	state.builder.AddFunction(spv::OpStore, state.objprim_output_variable, value);
}

} // namespace

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16: return TypeU32(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
		case IR::Type::Bvh8Result: return TypeBvh8Result(state);
		case IR::Type::F32x2: return TypeF32Vector(state, 2);
		default: return 0;
	}
}

uint32_t ValueEmitContext::Def(IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case IR::Type::U1: return ConstantBool(state, value.U1());
			case IR::Type::U8: return ConstantU32(state, value.U8());
			case IR::Type::U16: return ConstantU32(state, value.U16());
			case IR::Type::U32: return ConstantU32(state, value.U32());
			case IR::Type::U64: return ConstantU64(state, value.U64());
			case IR::Type::F16: return ConstantU32(state, value.F16Bits());
			case IR::Type::F32:
				return ConstantF32(state, std::bit_cast<uint32_t>(value.F32Value()));
			default: break;
		}
	}
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr) {
		Fail("direct SPIR-V emitter received a non-value argument");
	}
	if (dispatcher_spills != nullptr && state.current_block != nullptr &&
	    inst->Parent() != state.current_block) {
		if (const auto found = dispatcher_spills->find(inst); found != dispatcher_spills->end()) {
			if (const auto loaded = dispatcher_block_loads.find(inst);
			    loaded != dispatcher_block_loads.end() &&
			    loaded->second.first == state.current_label) {
				return loaded->second.second;
			}
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), id,
			                          found->second);
			dispatcher_block_loads.insert_or_assign(inst, std::pair {state.current_label, id});
			return id;
		}
	}
	return Result(*inst);
}

uint32_t ValueEmitContext::Arg(const IR::Inst& inst, size_t index) {
	return Def(inst.Arg(index));
}

uint32_t ValueEmitContext::HalfArg(const IR::Inst& inst, size_t index, uint32_t lane_half) {
	return lane_half == half ? Arg(inst, index) : other_half->Arg(inst, index);
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto low         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, low, scope,
	                          other_half == nullptr || half == 0 ? Def(predicate)
	                                                             : other_half->Def(predicate));
	if (other_half == nullptr && state.wave_pair) {
		// Whole-wave ballot: this subgroup's word and the other half's, in guest lane order.
		const auto own = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), own, low, 0);
		const auto other  = EmitPairExchange(state, own);
		const auto is_low = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpIEqual, TypeBool(state), is_low, EmitPairHalf(state),
		                          ConstantU32(state, 0));
		const auto word0 = state.builder.AllocateId();
		const auto word1 = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), word0, is_low, own, other);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), word1, is_low, other, own);
		const auto ballot = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, word0, word1,
		                          ConstantU32(state, 0), ConstantU32(state, 0));
		return ballot;
	}
	if (other_half == nullptr) {
		return low;
	}
	const auto high      = state.builder.AllocateId();
	const auto low_word  = state.builder.AllocateId();
	const auto high_word = state.builder.AllocateId();
	const auto ballot    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, high, scope,
	                          half == 1 ? Def(predicate) : other_half->Def(predicate));
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low_word, low, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high_word, high, 0);
	state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, low_word, high_word,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	return ballot;
}

uint32_t ValueEmitContext::FirstLane(uint32_t ballot) {
	// An empty ballot (EXEC zero) gives lane 0, which V_READFIRSTLANE reads on hardware. FindILsb
	// returns -1 for zero, unlike BallotFindLSB, whose result is then undefined.
	const auto first = [&](uint32_t word) {
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits, ballot, word);
		return std::pair {bits, EmitGlsl<GLSLstd450FindILsb, IR::Type::U32>(state, bits)};
	};
	const auto [low, low_first] = first(0);
	if (state.program.wave_size == 32u) {
		return EmitGlsl<GLSLstd450SMax, IR::Type::U32>(state, low_first, ConstantU32(state, 0));
	}
	const auto high_first = first(1).second;
	const auto high_lane  = EmitGlsl<GLSLstd450SMax, IR::Type::U32>(
	    state, EmitBinaryU32(state, spv::OpBitwiseOr, high_first, ConstantU32(state, 32)),
	    ConstantU32(state, 0));
	return Select(state, TypeU32(state),
	              Binary(state, spv::OpINotEqual, TypeBool(state), low, ConstantU32(state, 0)),
	              low_first, high_lane);
}

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane, bool wave_uniform_lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto low   = state.builder.AllocateId();
	if (other_half == nullptr && state.wave_pair) {
		const auto physical = EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          physical);
		if (!wave_uniform_lane) {
			// Row, 16- and 32-lane permutations read a lane of the reader's own half.
			return low;
		}
		// One lane for the whole wave: the half that owns it supplies the value to the other.
		auto own = low;
		if (type != TypeU32(state)) {
			own = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, TypeU32(state), own, low);
		}
		const auto other    = EmitPairExchange(state, own);
		const auto lane_half = EmitBinaryU32(state, spv::OpBitwiseAnd,
		                                     EmitBinaryU32(state, spv::OpShiftRightLogical, lane,
		                                                   ConstantU32(state, 5)),
		                                     ConstantU32(state, 1));
		const auto mine = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpIEqual, TypeBool(state), mine, lane_half, EmitPairHalf(state));
		auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), value, mine, own, other);
		if (type != TypeU32(state)) {
			const auto cast = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitcast, type, cast, value);
			value = cast;
		}
		return value;
	}
	if (other_half == nullptr) {
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          lane);
		return low;
	}
	const auto physical_lane =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto high          = state.builder.AllocateId();
	const auto in_high       = state.builder.AllocateId();
	const auto value         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope,
	                          HalfArg(inst, index, 0), physical_lane);
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, high, scope,
	                          HalfArg(inst, index, 1), physical_lane);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), in_high,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, type, value, in_high, high, low);
	return value;
}

uint32_t ValueEmitContext::Result(const IR::Inst& inst) {
	if (const auto id = inst.Definition(half); id != 0) {
		return id;
	}
	const auto id = state.builder.AllocateId();
	inst.SetDefinition(half, id);
	return id;
}

uint32_t ValueEmitContext::Define(const IR::Inst& inst, uint32_t value) {
	if (const auto id = inst.Definition(half); id != 0) {
		if (id != value) {
			state.builder.AddFunction(spv::OpCopyObject, TypeId(state, inst.GetType()),
			                          id, value);
		}
		return id;
	}
	inst.SetDefinition(half, value);
	return value;
}

uint32_t ValueEmitContext::ResourceIndex(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != opcode) {
		Fail("typed resource handle has the wrong producer");
	}
	return inst->Flags<uint32_t>();
}

const IR::Inst* ValueEmitContext::ImageAddress(IR::Value value) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::MakeImageAddress) {
		Fail("typed image address was not constructed by MakeImageAddress");
	}
	return inst;
}

const IR::MemoryInfo& ValueEmitContext::Memory(const IR::Inst& inst) const {
	return state.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
}

const IR::ExportInfo& ValueEmitContext::Export(const IR::Inst& inst) const {
	return state.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
}

[[noreturn]] void ValueEmitContext::Fail(const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), reason);
	std::abort();
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
	std::abort();
}

// The loop watchdog gives every unconditional
// latch an extra exit edge to the loop's merge block. Every value that the code after the loop
// reads from inside the loop (the header phis of all registers the loop writes) must then stay
// live up to that latch, next to the new version the back edge carries, so a register loop-nest
// holds one extra live version per nesting level. When a conditional exit to the same merge
// already dominates the latch (every iteration passes through it), that exit carries the cap
// (GuardLoopCondition) and the extra latch edge adds nothing but live ranges, so it is dropped.
// Only a loop that hits the cap behaves differently: the forced exit now comes at the next pass
// through that exit instead of at the latch (the loop body before it runs once more).
// True when every path from the loop header to the latch (along emitted terminator edges) passes
// through `exit_index`.
static bool ExitDominatesLatch(const std::vector<std::array<size_t, 2>>& successors, size_t header,
                               size_t latch, size_t exit_index) {
	if (header == exit_index) return true;
	std::vector<bool>   seen(successors.size(), false);
	std::vector<size_t> stack {header};
	seen[header] = true;
	while (!stack.empty()) {
		const auto block = stack.back();
		stack.pop_back();
		if (block == latch) return false;
		for (const auto next: successors[block]) {
			if (next == SIZE_MAX || next == exit_index || seen[next]) continue;
			seen[next] = true;
			stack.push_back(next);
		}
	}
	return true;
}

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	LaneShareClassify(state);
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.program.info.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	for (const auto* block: program.blocks) {
		// Re-emission may use a different host subgroup width and ID allocation order.
		for (const auto& inst: *block) {
			inst.ResetDefinitions();
		}
		block->SetDefinition(state.builder.AllocateId());
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
		// Allocate both halves in program order so the SPIR-V is identical across runs.
		const auto add_spill = [&](const IR::Inst* inst) {
			for (uint32_t half = 0; half < state.lane_count; half++) {
				dispatch.spills[half].emplace(inst, state.builder.AllocateId());
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
					continue;
				}
				if (SpillPointerType(ctx, inst.GetType()) == 0) {
					ctx.Fail(inst, "cannot be stored by the dispatcher");
					break;
				}
				add_spill(&inst);
			}
		}
		const auto mark_cross_block = [&](IR::Value value, const IR::Block* consumer) {
			value                  = value.Resolve();
			const auto* definition = value.TryInstruction();
			if (definition == nullptr || definition->Parent() == consumer ||
			    definition->Parent() == program.blocks.front()) {
				return;
			}
			if (SpillPointerType(ctx, definition->GetType()) == 0) {
				ctx.Fail(*definition, "cannot be stored by the dispatcher");
				return;
			}
			if (!dispatch.spills[0].contains(definition)) {
				add_spill(definition);
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* consumer =
					    inst.GetOpcode() == IR::ValueOpcode::Phi ? inst.PhiBlock(index) : block;
					mark_cross_block(inst.Arg(index), consumer);
				}
			}
		}
		for (size_t index = 0; index < program.blocks.size(); index++) {
			mark_cross_block(program.blocks[index]->condition, program.blocks[index]);
			mark_cross_block(program.blocks[index]->indirect_target, program.blocks[index]);
		}
		dispatch.header_label       = state.builder.AllocateId();
		dispatch.select_label       = state.builder.AllocateId();
		dispatch.after_switch_label = state.builder.AllocateId();
		dispatch.continue_label     = state.builder.AllocateId();
		dispatch.merge_label        = state.builder.AllocateId();
		ctx.dispatcher_spills       = &dispatch.spills[0];
		high.dispatcher_spills      = &dispatch.spills[1];
	}
	DefineGetBdaPointer(state);
	FindScalarLoadGroups(state);
	DefineIndirectFormattedLoadFunctions(ctx);
	DefineIndirectRawDword(ctx);
	DefineIndirectStoreFunctions(ctx);
	DefineBvhIntersect(state);
	DefineTrinityFunctions(state); // upstream bfd4c34b8 / 5eab0ec73: BVH stack, BVH8
	if (program.info.uses_swizzle) {
		ctx.scratch_u32_variable = state.builder.AllocateId();
		if (state.lane_count == 2) {
			high.scratch_u32_variable = state.builder.AllocateId();
		}
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	if (state.program.info.function_lds) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.program.info.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
	}
	DeclareBdaEntryShare(state); // KYTY_PERF_BDA_ENTRY_SHARE
	state.loop_counter_variables.clear();
	state.back_edge_counters.clear();
	if (ShaderLoopLimitFor(state.program.shader_hash) != 0 && !state.program.dispatcher_fallback) {
		const auto new_counter = [&]() {
			const auto variable = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          variable, spv::StorageClassFunction, ConstantU32(state, 0));
			return variable;
		};
		// A latch is a block that branches back to an earlier loop header. A conditional
		// branch that leaves that loop for its merge block tests the latch's count.
		std::unordered_map<uint32_t, size_t> order;
		for (size_t index = 0; index < program.blocks.size(); index++) {
			order.emplace(program.blocks[index]->id, index);
		}
		const auto id_of = [](const IR::Block* block) { return block != nullptr ? block->id : UINT32_MAX; };
		std::unordered_map<uint32_t, uint32_t> loop_counters;
		// Unconditional latches whose forced exit is decided
		// after every conditional exit of the loops is known.
		struct DeferredLatch {
			size_t   latch_index;
			size_t   header_index;
			uint32_t counter;
			uint32_t merge;
		};
		std::vector<DeferredLatch> deferred_latches;
		for (size_t index = 0; index < program.blocks.size(); index++) {
			const auto* block = program.blocks[index];
			const auto& term  = block->terminator;
			for (const auto target: {id_of(term.true_block), id_of(term.false_block)}) {
				const auto found = order.find(target);
				if (found == order.end() || found->second > index ||
				    !program.blocks[found->second]->terminator.loop_header) {
					continue;
				}
				auto counter = loop_counters.find(target);
				if (counter == loop_counters.end()) {
					counter = loop_counters.emplace(target, new_counter()).first;
				}
				state.back_edge_counters.emplace(block->id, counter->second);
				const auto merge = id_of(program.blocks[found->second]->terminator.merge_block);
				if (term.kind == CFG::TerminatorKind::Branch && merge != UINT32_MAX) {
					deferred_latches.push_back({index, found->second, counter->second, merge});
				}
				break;
			}
		}
		for (const auto& [header, counter]: loop_counters) {
			const auto merge = id_of(program.blocks[order.at(header)]->terminator.merge_block);
			if (merge == UINT32_MAX) {
				continue;
			}
			for (const auto* block: program.blocks) {
				const auto& term = block->terminator;
				if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
				    (id_of(term.true_block) == merge) != (id_of(term.false_block) == merge) &&
				    !state.loop_counter_variables.contains(block->id)) {
					state.loop_counter_variables.emplace(block->id, std::pair {counter, merge});
				}
			}
		}
		if (!deferred_latches.empty()) {
			std::vector<std::array<size_t, 2>> successors(program.blocks.size(),
			                                              std::array<size_t, 2> {SIZE_MAX, SIZE_MAX});
			for (size_t index = 0; index < program.blocks.size(); index++) {
				const auto& term = program.blocks[index]->terminator;
				if (term.kind != CFG::TerminatorKind::Branch &&
				    term.kind != CFG::TerminatorKind::ConditionalBranch) {
					continue;
				}
				const auto true_block = order.find(id_of(term.true_block));
				if (true_block != order.end()) successors[index][0] = true_block->second;
				if (term.kind == CFG::TerminatorKind::ConditionalBranch) {
					const auto false_block = order.find(id_of(term.false_block));
					if (false_block != order.end()) successors[index][1] = false_block->second;
				}
			}
			for (const auto& latch: deferred_latches) {
				bool guarded = false;
				for (size_t index = 0; index < program.blocks.size() && !guarded; index++) {
					const auto* block = program.blocks[index];
					const auto  exit  = state.loop_counter_variables.find(block->id);
					guarded = exit != state.loop_counter_variables.end() &&
					          block->terminator.kind == CFG::TerminatorKind::ConditionalBranch &&
					          exit->second == std::pair {latch.counter, latch.merge} &&
					          ExitDominatesLatch(successors, latch.header_index, latch.latch_index,
					                             index);
				}
				if (!guarded) {
					state.loop_counter_variables.emplace(program.blocks[latch.latch_index]->id,
					                                     std::pair {latch.counter, latch.merge});
				}
			}
		}
	}
	for (uint32_t half = 0; half < state.lane_count; half++) {
		auto& lane = half == 0 ? ctx : high;
		if (state.program.dispatcher_fallback) {
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					if (const auto found = dispatcher->spills[half].find(&inst);
					    found != dispatcher->spills[half].end()) {
						state.builder.AddFunction(spv::OpVariable,
						                          SpillPointerType(lane, inst.GetType()),
						                          found->second, spv::StorageClassFunction);
					}
				}
			}
		}
		if (lane.scratch_u32_variable != 0) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
	}
	if (state.wave_pair) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pair_seq_variable, spv::StorageClassFunction);
	}
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	if (state.lds_storage_class == spv::StorageClassStorageBuffer && state.lds_variable != 0) {
		const auto group_x = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		const auto group_y = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto group_z = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 2);
		const auto count_x = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 0);
		const auto count_y = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 1);
		const auto row = EmitAddU32(state, group_y, EmitBinaryU32(state, spv::OpIMul, group_z, count_y));
		const auto index = EmitAddU32(state, group_x, EmitBinaryU32(state, spv::OpIMul, row, count_x));
		state.lds_base_dwords = EmitBinaryU32(state, spv::OpIMul, index,
		                                     ConstantU32(state, LdsDwordCount(state)));
	}
	EmitMemoryOffsets(state);
	EmitPairInitialize(state);
	if (state.objprim_output_variable != 0) {
		EmitObjprimIdOutput(state);
	}
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	LaneShareReport(state);
	state.current_block = nullptr;
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
