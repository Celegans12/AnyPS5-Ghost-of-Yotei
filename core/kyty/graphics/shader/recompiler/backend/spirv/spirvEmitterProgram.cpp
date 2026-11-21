#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#if defined(ANYPS5_KYTY)
#include "graphics/shader/recompiler/backend/spirv/packedUnormBlendProbe.h"
#endif

#include "common/assert.h"

#include <algorithm>
#include <string>
#include <cstdlib>
#include <bit>
#include <cmath>
#include <cstdlib>
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

struct OutlinedRegion {
	const IR::Block*                     entry = nullptr;
	const IR::Block*                     exit  = nullptr;
	std::unordered_set<const IR::Block*> blocks;
	std::vector<const IR::Inst*>         inputs;
	std::vector<const IR::Inst*>         outputs;
	size_t                               phi_outputs = 0;
	uint32_t                             function    = 0;
	uint32_t                             input_type  = 0;
	uint32_t                             output_type = 0;
	uint32_t                             call_label  = 0;
	std::vector<uint32_t>                output_ids;
};

struct CallLadder {
	struct Level {
		const IR::Block* a      = nullptr;
		const IR::Block* b      = nullptr;
		const IR::Block* m1     = nullptr;
		const IR::Block* m2     = nullptr;
		const IR::Block* fin    = nullptr;
		OutlinedRegion*  region = nullptr;
	};
	std::vector<Level>                   levels;
	const IR::Block*                     join = nullptr;
	const IR::Block*                     none = nullptr;
	std::unordered_set<const IR::Block*> skipped;
};

struct PhiSlot {
	size_t                arg    = 0;
	const OutlinedRegion* region = nullptr;
	size_t                output = 0;
};

struct DeferredPhiPatch {
	DeferredPhi          phi;
	const IR::Inst*      instruction = nullptr;
	uint32_t             half        = 0;
	std::vector<PhiSlot> slots;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t>              block_exit_labels;
	std::vector<DeferredPhiPatch>                               deferred_phis;
	std::unordered_map<const IR::Block*, const OutlinedRegion*> region_of;
	const CallLadder*                                           ladder = nullptr;
};

bool HasSpirvType(IR::Type type) {
	switch (type) {
		case IR::Type::U1:
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16:
		case IR::Type::U64:
		case IR::Type::U32x2:
		case IR::Type::F32:
		case IR::Type::F64:
		case IR::Type::U32x3:
		case IR::Type::U32x4:
		case IR::Type::F32x2: return true;
		default: return false;
	}
}

std::unordered_map<const IR::Block*, const IR::Block*> ImmediateDominators(const IR::Program& program) {
	std::vector<const IR::Block*>                 order;
	std::unordered_map<const IR::Block*, size_t>  rpo;
	std::unordered_set<const IR::Block*>          seen;
	std::vector<std::pair<const IR::Block*, size_t>> stack {{program.blocks.front(), 0}};
	seen.insert(program.blocks.front());
	while (!stack.empty()) {
		auto& [block, next] = stack.back();
		const auto successors = block->ImmSuccessors();
		if (next < successors.size()) {
			const auto* successor = successors[next++];
			if (seen.insert(successor).second) stack.emplace_back(successor, 0);
			continue;
		}
		order.push_back(block);
		stack.pop_back();
	}
	std::reverse(order.begin(), order.end());
	for (size_t i = 0; i < order.size(); i++)
		rpo[order[i]] = i;
	std::unordered_map<const IR::Block*, const IR::Block*> idom {{order.front(), order.front()}};
	const auto intersect = [&](const IR::Block* a, const IR::Block* b) {
		while (a != b) {
			while (rpo.at(a) > rpo.at(b)) a = idom.at(a);
			while (rpo.at(b) > rpo.at(a)) b = idom.at(b);
		}
		return a;
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (size_t i = 1; i < order.size(); i++) {
			const IR::Block* dominator = nullptr;
			for (const auto* predecessor: order[i]->ImmPredecessors()) {
				if (!idom.contains(predecessor)) continue;
				dominator = dominator == nullptr ? predecessor : intersect(predecessor, dominator);
			}
			if (dominator != nullptr && idom[order[i]] != dominator) {
				idom[order[i]] = dominator;
				changed = true;
			}
		}
	}
	return idom;
}

std::vector<OutlinedRegion> FindOutlinedRegions(const IR::Program& program) {
	std::vector<OutlinedRegion> regions;
	if (program.inlined_call_entries.empty() || program.dispatcher_fallback ||
	    program.stage != ShaderType::Compute || program.blocks.empty())
		return regions;
	const auto idom = ImmediateDominators(program);
	const auto dominates = [&](const IR::Block* dominator, const IR::Block* block) {
		if (!idom.contains(block)) return false;
		for (;;) {
			if (block == dominator) return true;
			const auto* parent = idom.at(block);
			if (parent == block) return false;
			block = parent;
		}
	};
	std::unordered_set<const IR::Block*> claimed;
	for (const auto pc: program.inlined_call_entries) {
		const auto found = std::ranges::find_if(program.blocks, [&](const IR::Block* block) {
			return block->start_pc == pc && block->end_pc > pc;
		});
		if (found == program.blocks.end() || *found == program.blocks.front() || claimed.contains(*found) ||
		    (*found)->ImmPredecessors().size() != 1u)
			continue;
		OutlinedRegion region;
		region.entry = *found;
		for (const auto* block: program.blocks)
			if (dominates(region.entry, block)) region.blocks.insert(block);
		std::unordered_set<const IR::Block*> exits;
		bool ok = true;
		for (const auto* block: program.blocks) {
			if (!region.blocks.contains(block)) continue;
			for (const auto* successor: block->ImmSuccessors())
				if (!region.blocks.contains(successor)) exits.insert(successor);
			const auto& term = block->terminator;
			ok &= (term.kind == CFG::TerminatorKind::Branch && term.true_block != nullptr) ||
			      (term.kind == CFG::TerminatorKind::ConditionalBranch && term.true_block != nullptr &&
			       term.false_block != nullptr && !block->condition.IsEmpty());
		}
		if (!ok || exits.size() != 1u) continue;
		region.exit = *exits.begin();
		for (const auto* block: program.blocks) {
			if (!region.blocks.contains(block)) continue;
			for (const auto* inst: {block->terminator.merge_block, block->terminator.continue_block})
				ok &= inst == nullptr || region.blocks.contains(inst) || inst == region.exit;
		}
		if (!ok) continue;
		std::unordered_set<const IR::Inst*> inputs, visited;
		const auto collect = [&](auto&& self, IR::Value value) -> void {
			value = value.Resolve();
			const auto* inst = value.TryInstruction();
			if (inst == nullptr || region.blocks.contains(inst->Parent()) || !visited.insert(inst).second) return;
			if (HasSpirvType(inst->GetType())) {
				if (inputs.insert(inst).second) region.inputs.push_back(inst);
				return;
			}
			for (size_t index = 0; index < inst->NumArgs(); index++) self(self, inst->Arg(index));
		};
		for (const auto* block: program.blocks) {
			if (!region.blocks.contains(block)) continue;
			for (const auto& inst: *block)
				for (size_t index = 0; index < inst.NumArgs(); index++) collect(collect, inst.Arg(index));
			collect(collect, block->condition);
			collect(collect, block->indirect_target);
		}
		for (const auto& phi: *region.exit) {
			if (phi.GetOpcode() != IR::ValueOpcode::Phi) break;
			for (size_t index = 0; index < phi.NumArgs(); index++)
				if (region.blocks.contains(phi.PhiBlock(index))) collect(collect, phi.Arg(index));
		}
		for (const auto& phi: *region.exit) {
			if (phi.GetOpcode() != IR::ValueOpcode::Phi) break;
			bool from_region = false;
			for (size_t index = 0; index < phi.NumArgs(); index++)
				from_region |= region.blocks.contains(phi.PhiBlock(index));
			if (!from_region) continue;
			ok &= HasSpirvType(phi.GetType());
			region.outputs.push_back(&phi);
		}
		region.phi_outputs = region.outputs.size();
		std::unordered_set<const IR::Inst*> direct;
		const auto used_outside = [&](const IR::Inst& inst, const IR::Block* consumer) {
			if (!region.blocks.contains(consumer) && direct.insert(&inst).second) {
				ok &= HasSpirvType(inst.GetType());
				region.outputs.push_back(&inst);
			}
		};
		for (const auto* block: program.blocks) {
			if (region.blocks.contains(block)) continue;
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* arg = inst.Arg(index).Resolve().TryInstruction();
					if (arg == nullptr || !region.blocks.contains(arg->Parent())) continue;
					if (inst.GetOpcode() == IR::ValueOpcode::Phi && region.blocks.contains(inst.PhiBlock(index))) {
						ok &= block == region.exit;
						continue;
					}
					used_outside(*arg, block);
				}
			}
			for (const auto& value: {block->condition, block->indirect_target}) {
				const auto* arg = value.IsEmpty() ? nullptr : value.Resolve().TryInstruction();
				if (arg != nullptr && region.blocks.contains(arg->Parent())) used_outside(*arg, block);
			}
		}
		if (!ok) continue;
		for (const auto* block: region.blocks) claimed.insert(block);
		regions.push_back(std::move(region));
	}
	return regions;
}

std::vector<PhiSlot> PhiSlots(const StructuredFunctionState& structured, const IR::Inst& phi) {
	std::vector<PhiSlot> slots;
	if (structured.ladder != nullptr && phi.Parent() == structured.ladder->join) {
		for (size_t level = 0; level < structured.ladder->levels.size(); level++)
			slots.push_back({level, structured.ladder->levels[level].region, 0});
		return slots;
	}
	for (size_t index = 0; index < phi.NumArgs(); index++) {
		const auto found = structured.region_of.find(phi.PhiBlock(index));
		if (found == structured.region_of.end()) {
			slots.push_back({index});
			continue;
		}
		const auto* region = found->second;
		if (std::ranges::any_of(slots, [&](const PhiSlot& slot) { return slot.region == region; })) continue;
		const auto output = std::ranges::find(region->outputs, &phi) - region->outputs.begin();
		slots.push_back({index, region, static_cast<size_t>(output)});
	}
	return slots;
}

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

void EmitReturn(ValueEmitContext& ctx) {
#if defined(ANYPS5_KYTY)
	if (ctx.state.pixel_guest_func == 0) {
		EmitKillIfPixelValidMaskInactive(ctx.state);
	}
#else
	EmitKillIfPixelValidMaskInactive(ctx.state);
#endif
	if (ctx.state.object_primitive_id_variable != 0) {
		auto& state = ctx.state;
		const auto primitive = EmitGetBuiltin(
		    ctx, IR::Value {static_cast<uint32_t>(IR::StageInputKind::PrimitiveId)}, IR::Value {0u});
		const auto object = EmitPushDataLoad(state, ConstantU32(state, 0));
		state.builder.AddFunction(spv::OpStore, state.object_primitive_id_variable,
		                          EmitAddU32(state, object, primitive));
	}
	FlushOracleCounts(ctx.state);
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
			const auto condition = ctx.Def(block->condition);
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

bool PairPureOpcode(IR::ValueOpcode op) {
	using Op = IR::ValueOpcode;
	switch (op) {
		case Op::UndefU1:
		case Op::UndefU32:
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
		case Op::ConvertF32F16Lane:
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
		case Op::SelectU32:
		case Op::SelectF32:
		case Op::SelectF64:
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
		case Op::FPOrdNotEqual64:
		case Op::FPOrdLessThan64:
		case Op::FPOrdGreaterThan64:
		case Op::FPUnordEqual64:
		case Op::FPUnordNotEqual64:
		case Op::FPUnordLessThan64:
		case Op::FPUnordGreaterThan64:
		case Op::FPUnordLessThanEqual64:
		case Op::FPUnordGreaterThanEqual64:
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
		case Op::FPIsNan64:
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
		case Op::PackedFPAdd16x2:
		case Op::PackedFPMul16x2:
		case Op::PackedFPFma16x2:
		case Op::PackedFPMin16x2:
		case Op::PackedFPMax16x2:
		case Op::ExtractF16Lane:
		case Op::InsertF16Lane:
		case Op::FPAdd16:
		case Op::FPSub16:
		case Op::FPMul16:
		case Op::FPFma16:
		case Op::FPMin16:
		case Op::FPMax16:
		case Op::FPAbs16:
		case Op::FPNeg16:
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
		case Op::FPFract64:
		case Op::GetUserData:
		case Op::GetShaderBase:
		case Op::GetDispatchThreadExtent:
		case Op::ReadConst:
		case Op::LoadDeviceAddressU32: return true;
		default: return false;
	}
}

bool PairSharedRoot(const IR::Program& program, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto op = inst.GetOpcode();
	return op == Op::Ballot || op == Op::ReadFirstLane || op == Op::SubgroupUMinU32 || op == Op::SubgroupUMaxU32 ||
	       op == Op::SubgroupOrU32 || op == Op::RayQueryNext || op == Op::ReadConstBuffer ||
	       op == Op::ReadRealtimeClock || op == Op::ReadCallWord ||
	       (op == Op::LoadAddressU32 &&
	        program.memory_info.at(inst.Flags<IR::MemoryFlags>().index).kind == IR::ResourceKind::ScalarAddress);
}

bool PairSharedCandidate(const IR::Program& program, const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto op = inst.GetOpcode();
	if (op == Op::Phi) return inst.NumArgs() != 0;
	if (PairPureOpcode(op)) return true;
	switch (op) {
		case Op::GetBuiltin: {
			if (!inst.Arg(0).IsImmediate()) return false;
			const auto kind = static_cast<IR::StageInputKind>(inst.Arg(0).U32());
			return kind == IR::StageInputKind::WorkgroupId || kind == IR::StageInputKind::NumWorkgroups;
		}
		case Op::LoadAddressU8:
		case Op::LoadAddressU16:
		case Op::LoadAddressU32: {
			const auto& memory = program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
			return !memory.planning_only &&
			       (memory.kind == IR::ResourceKind::Global || memory.kind == IR::ResourceKind::Flat ||
			        memory.kind == IR::ResourceKind::ScalarAddress);
		}
		case Op::GetAddressResource:
		case Op::GetSrtResource: return true;
		default: return false;
	}
}

bool PairSharedEmission(const IR::Inst& inst) {
	using Op = IR::ValueOpcode;
	const auto op = inst.GetOpcode();
	return op != Op::GetAddressResource && op != Op::GetSrtResource && op != Op::Phi;
}

std::unordered_set<const IR::Inst*> FindPairSharedValues(const IR::Program& program) {
	std::unordered_set<const IR::Inst*> shared;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (PairSharedRoot(program, inst) || PairSharedCandidate(program, inst)) shared.insert(&inst);
		}
	}
	const auto invariant = [&](IR::Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) return true;
		const auto* definition = value.TryInstruction();
		return definition != nullptr && shared.contains(definition);
	};
	for (bool changed = true; changed;) {
		changed = false;
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (!shared.contains(&inst) || PairSharedRoot(program, inst)) continue;
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					if (!invariant(inst.Arg(index))) {
						shared.erase(&inst);
						changed = true;
						break;
					}
				}
			}
		}
	}
	return shared;
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto op = inst.GetOpcode();
	if (ctx.half != 0 &&
	    (PairSharedRoot(ctx.state.program, inst) ||
	     (ctx.state.pair_shared.contains(&inst) && PairSharedEmission(inst)))) {
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
		if (ctx.half != 0 && ctx.state.pair_shared.contains(&inst)) {
			ctx.Define(inst, ctx.other_half->Result(inst));
			return;
		}
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
		auto slots = PhiSlots(structured, inst);
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), slots.size()), &inst, ctx.half,
		     std::move(slots)});
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
void EmitBlockInstructions(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	bool emitted_non_phi = false;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			emitted_non_phi = true;
		}
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::StoreCompletion &&
			                  inst.GetOpcode() != IR::ValueOpcode::DataCacheInvalidate &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				emit_instruction(lane, inst);
			}
		}
		ctx.state.lane_half = 0;
	}
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, block->Definition());
	EmitBlockInstructions(ctx, block, emit_instruction);
}

uint32_t ResolveLadderValue(ValueEmitContext& lane, const CallLadder& ladder, size_t k, IR::Value value) {
	value = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::Phi) return lane.Def(value);
	const auto& level = ladder.levels[k];
	const auto  incoming = [&](const IR::Block* from) {
		for (size_t index = 0; index < inst->NumArgs(); index++)
			if (inst->PhiBlock(index) == from) return inst->Arg(index);
		lane.Fail(*inst, "has no operand for the call of its level");
	};
	for (size_t j = 1; j <= k; j++) {
		if (inst->Parent() == ladder.levels[j].fin)
			return ResolveLadderValue(lane, ladder, k,
			                          incoming(j == k ? ladder.levels[k].m2 : ladder.levels[j + 1].fin));
	}
	if (inst->Parent() == level.m2) return ResolveLadderValue(lane, ladder, k, incoming(level.m1));
	if (inst->Parent() != level.m1) return lane.Def(value);
	const auto output = std::ranges::find(level.region->outputs, inst) - level.region->outputs.begin();
	if (output >= static_cast<ptrdiff_t>(level.region->phi_outputs))
		lane.Fail(*inst, "is not an output of the call of its level");
	return level.region->output_ids.at(static_cast<size_t>(output) * lane.state.lane_count + lane.half);
}

void PatchLadderPhi(ValueEmitContext& ctx, const CallLadder& ladder, const DeferredPhiPatch& deferred) {
	auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
	for (size_t level = 0; level < ladder.levels.size(); level++) {
		const auto& entry = ladder.levels[level];
		const auto& phi   = *deferred.instruction;
		const auto* from  = level == 0 ? ladder.levels[0].m2 : ladder.levels[1].fin;
		size_t      index = 0;
		while (index < phi.NumArgs() && phi.PhiBlock(index) != from) index++;
		if (index == phi.NumArgs()) ctx.Fail(phi, "has no operand for a level of its call table");
		ctx.state.builder.PatchDeferredPhi(deferred.phi, level,
		                                   ResolveLadderValue(lane, ladder, level, phi.Arg(index)),
		                                   entry.region->call_label);
	}
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& deferred: structured.deferred_phis) {
		auto& lane = deferred.half == 0 ? ctx : *ctx.other_half;
		if (structured.ladder != nullptr && deferred.instruction->Parent() == structured.ladder->join) {
			PatchLadderPhi(ctx, *structured.ladder, deferred);
			continue;
		}
		for (size_t slot = 0; slot < deferred.slots.size(); slot++) {
			const auto& source = deferred.slots[slot];
			if (source.region != nullptr) {
				ctx.state.builder.PatchDeferredPhi(
				    deferred.phi, slot,
				    source.region->output_ids.at(source.output * ctx.state.lane_count + deferred.half),
				    source.region->call_label);
				continue;
			}
			const auto* predecessor = deferred.instruction->PhiBlock(source.arg);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			ctx.state.builder.PatchDeferredPhi(
			    deferred.phi, slot, lane.Def(deferred.instruction->Arg(source.arg)), found->second);
		}
	}
}

void EmitFunctionEntryValues(EmitterState& state) {
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
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
}

void EmitOutlinedFunction(ValueEmitContext& ctx, OutlinedRegion& region) {
	auto&       state   = ctx.state;
	const auto& program = state.program;
	const auto  lanes   = state.lane_count;
	std::vector<uint32_t> input_members, output_members;
	for (const auto* input: region.inputs)
		for (uint32_t half = 0; half < lanes; half++) input_members.push_back(TypeId(state, input->GetType()));
	for (const auto* output: region.outputs)
		for (uint32_t half = 0; half < lanes; half++) output_members.push_back(TypeId(state, output->GetType()));
	region.input_type  = input_members.empty() ? 0u
	    : state.builder.Type(spv::OpTypeStruct, std::span<const uint32_t>(input_members));
	region.output_type = output_members.empty() ? TypeVoid(state)
	    : state.builder.Type(spv::OpTypeStruct, std::span<const uint32_t>(output_members));
	const auto function_type = region.input_type == 0u
	    ? state.builder.Type(spv::OpTypeFunction, region.output_type)
	    : state.builder.Type(spv::OpTypeFunction, region.output_type, region.input_type);
	region.function = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpFunction, region.output_type, region.function,
	                          spv::FunctionControlDontInlineMask, function_type);
	const auto parameter = region.input_type == 0u ? 0u : state.builder.AllocateId();
	if (parameter != 0u) state.builder.AddFunction(spv::OpFunctionParameter, region.input_type, parameter);
	state.current_block = nullptr;
	EmitLabel(state, state.builder.AllocateId());
	EmitFunctionEntryValues(state);
	EmitMemoryBindings(state);
	uint32_t member = 0;
	for (const auto* input: region.inputs) {
		for (uint32_t half = 0; half < lanes; half++) {
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeId(state, input->GetType()), id,
			                          parameter, member++);
			input->SetDefinition(half, id);
		}
	}
	const auto main_entry = region.entry->Definition();
	const auto main_exit  = region.exit->Definition();
	const auto return_label = state.builder.AllocateId();
	region.entry->SetDefinition(state.builder.AllocateId());
	region.exit->SetDefinition(return_label);
	state.builder.AddFunction(spv::OpBranch, region.entry->Definition());
	StructuredFunctionState structured;
	for (const auto* block: program.blocks) {
		if (!region.blocks.contains(block)) continue;
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			if (block == region.entry && inst.GetOpcode() == IR::ValueOpcode::Phi) {
				lane.Define(inst, lane.Def(inst.Arg(0)));
				return;
			}
			EmitStructuredInstruction(lane, structured, inst);
		});
		structured.block_exit_labels.emplace(block, state.current_label);
		EmitStructuredTerminator(ctx, block);
	}
	state.current_block = nullptr;
	EmitLabel(state, return_label);
	std::vector<const IR::Block*> exit_edges;
	for (const auto* block: program.blocks)
		if (region.blocks.contains(block) && std::ranges::find(block->ImmSuccessors(), region.exit) !=
		                                         block->ImmSuccessors().end())
			exit_edges.push_back(block);
	std::vector<uint32_t> results;
	for (size_t output = 0; output < region.outputs.size(); output++) {
		const auto* inst = region.outputs[output];
		for (uint32_t half = 0; half < lanes; half++) {
			auto& lane = half == 0 ? ctx : *ctx.other_half;
			if (output >= region.phi_outputs) {
				results.push_back(0u);
				continue;
			}
			std::vector<std::pair<uint32_t, uint32_t>> incoming;
			for (const auto* edge: exit_edges) {
				for (size_t index = 0; index < inst->NumArgs(); index++) {
					if (inst->PhiBlock(index) == edge) {
						incoming.emplace_back(lane.Def(inst->Arg(index)), structured.block_exit_labels.at(edge));
						break;
					}
				}
			}
			if (incoming.size() == 1u) {
				results.push_back(incoming.front().first);
				continue;
			}
			std::vector<uint32_t> words {spv::OpPhi, TypeId(state, inst->GetType()), state.builder.AllocateId()};
			for (const auto& [value, label]: incoming) {
				words.push_back(value);
				words.push_back(label);
			}
			state.builder.AddFunction(words);
			results.push_back(words[2]);
		}
	}
	for (size_t output = region.phi_outputs; output < region.outputs.size(); output++)
		for (uint32_t half = 0; half < lanes; half++)
			results[output * lanes + half] = (half == 0 ? ctx : *ctx.other_half).Def(IR::Value(const_cast<IR::Inst*>(region.outputs[output])));
	if (results.empty()) {
		state.builder.AddFunction(spv::OpReturn);
	} else {
		const auto composite = state.builder.AllocateId();
		std::vector<uint32_t> words {spv::OpCompositeConstruct, region.output_type, composite};
		words.insert(words.end(), results.begin(), results.end());
		state.builder.AddFunction(words);
		state.builder.AddFunction(spv::OpReturnValue, composite);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	PatchStructuredPhis(ctx, structured);
	region.entry->SetDefinition(main_entry);
	region.exit->SetDefinition(main_exit);
	state.gds_length      = 0;
	state.lds_base_dwords = 0;
	state.memory_byte_offsets.clear();
	state.memory_byte_sizes.clear();
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (!region.blocks.contains(block) && inst.Definition(0) != 0u &&
			    std::ranges::find(region.inputs, &inst) == region.inputs.end()) {
				ctx.Fail(inst, "is read by an outlined call target without being passed to it");
			}
			inst.ResetDefinitions();
		}
	}
}

void EmitOutlinedCall(ValueEmitContext& ctx, OutlinedRegion& region, uint32_t exit_label = 0) {
	auto& state = ctx.state;
	const auto lanes = state.lane_count;
	state.current_block = region.entry;
	EmitLabel(state, region.entry->Definition());
	uint32_t argument = 0;
	if (region.input_type != 0u) {
		argument = state.builder.AllocateId();
		std::vector<uint32_t> words {spv::OpCompositeConstruct, region.input_type, argument};
		for (const auto* input: region.inputs)
			for (uint32_t half = 0; half < lanes; half++)
				words.push_back((half == 0 ? ctx : *ctx.other_half).Def(IR::Value(const_cast<IR::Inst*>(input))));
		state.builder.AddFunction(words);
	}
	const auto result = state.builder.AllocateId();
	if (argument != 0u) {
		state.builder.AddFunction(spv::OpFunctionCall, region.output_type, result, region.function, argument);
	} else {
		state.builder.AddFunction(spv::OpFunctionCall, region.output_type, result, region.function);
	}
	uint32_t member = 0;
	region.output_ids.clear();
	for (size_t output = 0; output < region.outputs.size(); output++) {
		const auto* inst = region.outputs[output];
		for (uint32_t half = 0; half < lanes; half++) {
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeId(state, inst->GetType()), id, result,
			                          member++);
			region.output_ids.push_back(id);
			if (output >= region.phi_outputs) (half == 0 ? ctx : *ctx.other_half).Define(*inst, id);
		}
	}
	region.call_label = state.current_label;
	state.builder.AddFunction(spv::OpBranch, exit_label != 0u ? exit_label : region.exit->Definition());
}

std::optional<CallLadder> FindCallLadder(const IR::Program& program, std::vector<OutlinedRegion>& regions) {
	using Op = IR::ValueOpcode;
	std::unordered_map<const IR::Block*, CallLadder::Level> by_a;
	const auto only_phis = [](const IR::Block* block) {
		return std::ranges::all_of(*block, [](const IR::Inst& inst) {
			return inst.GetOpcode() == Op::Phi || inst.GetOpcode() == Op::Reference ||
			       inst.GetOpcode() == Op::ReferenceU32;
		});
	};
	const auto only_reads = [](const IR::Block* block) {
		return std::ranges::all_of(*block, [](const IR::Inst& inst) {
			const auto op = inst.GetOpcode();
			return op != Op::Phi && (!inst.MayHaveSideEffects() || op == Op::Reference ||
			                         op == Op::ReferenceU32 || op == Op::ConditionRef);
		});
	};
	const auto branches = [](const IR::Block* block, const IR::Block* on_true, const IR::Block* on_false) {
		const auto& term = block->terminator;
		return term.kind == CFG::TerminatorKind::ConditionalBranch && term.true_block == on_true &&
		       term.false_block == on_false && !block->condition.IsEmpty();
	};
	const auto negation_of = [](IR::Value value, IR::Value of) {
		const auto* inst = value.Resolve().TryInstruction();
		return inst != nullptr && inst->GetOpcode() == Op::LogicalNot && !of.IsEmpty() &&
		       inst->Arg(0).Resolve().TryInstruction() == of.Resolve().TryInstruction() &&
		       of.Resolve().TryInstruction() != nullptr;
	};
	for (auto& region: regions) {
		const auto* c = region.entry;
		if (c->ImmPredecessors().size() != 1u) continue;
		const auto* b = c->ImmPredecessors()[0];
		if (b->ImmPredecessors().size() != 1u) continue;
		const auto* a  = b->ImmPredecessors()[0];
		const auto* m1 = region.exit;
		if (m1->terminator.kind != CFG::TerminatorKind::Branch || m1->terminator.true_block == nullptr) continue;
		const auto* m2 = m1->terminator.true_block;
		const auto& m2_term = m2->terminator;
		if (!branches(b, c, m1) || !branches(a, b, m2) || m2_term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    m2_term.true_block == nullptr || m2_term.false_block == nullptr || m2->condition.IsEmpty() ||
		    !only_phis(m1) || !only_phis(m2) || !only_reads(b) ||
		    m2->ImmPredecessors().size() != 2u)
			continue;
		bool exits_ok = true;
		for (const auto* predecessor: m1->ImmPredecessors())
			exits_ok &= predecessor == b || region.blocks.contains(predecessor);
		for (const auto* predecessor: m2->ImmPredecessors()) exits_ok &= predecessor == a || predecessor == m1;
		const auto* condition = m2->condition.Resolve().TryInstruction();
		exits_ok &= condition != nullptr && condition->GetOpcode() == Op::Phi && condition->Parent() == m2 &&
		            condition->NumArgs() == 2u;
		if (exits_ok) {
			for (size_t index = 0; index < 2u; index++) {
				const auto* from = condition->PhiBlock(index);
				exits_ok &= (from == a && negation_of(condition->Arg(index), a->condition)) ||
				            (from == m1 && negation_of(condition->Arg(index), b->condition));
			}
		}
		if (!exits_ok) continue;
		by_a.emplace(a, CallLadder::Level {a, b, m1, m2, nullptr, &region});
	}
	if (by_a.size() < 2u) return std::nullopt;
	std::unordered_set<const IR::Block*> nexts;
	for (const auto& [a, level]: by_a)
		if (by_a.contains(level.m2->terminator.true_block)) nexts.insert(level.m2->terminator.true_block);
	CallLadder ladder;
	for (const auto& [a, level]: by_a)
		if (!nexts.contains(a)) {
			if (!ladder.levels.empty()) return std::nullopt;
			ladder.levels.push_back(level);
		}
	if (ladder.levels.empty()) return std::nullopt;
	while (by_a.contains(ladder.levels.back().m2->terminator.true_block))
		ladder.levels.push_back(by_a.at(ladder.levels.back().m2->terminator.true_block));
	if (ladder.levels.size() != by_a.size()) return std::nullopt;
	ladder.none = ladder.levels.back().m2->terminator.true_block;
	if (std::ranges::any_of(*ladder.none, [](const IR::Inst& inst) { return inst.GetOpcode() == Op::Phi; }) ||
	    ladder.none->ImmPredecessors().size() != 1u)
		return std::nullopt;
	std::unordered_set<const IR::Block*> members;
	for (size_t k = 0; k < ladder.levels.size(); k++) {
		auto& level = ladder.levels[k];
		level.fin   = level.m2->terminator.false_block;
		for (const auto* block: {level.a, level.b, level.m1, level.m2, level.fin}) members.insert(block);
		const bool last = k + 1u == ladder.levels.size();
		const auto expected = last ? 1u : 2u;
		if (level.fin == ladder.none || level.fin->ImmPredecessors().size() != expected ||
		    std::ranges::find(level.fin->ImmPredecessors(), level.m2) == level.fin->ImmPredecessors().end() ||
		    (!last && std::ranges::find(level.fin->ImmPredecessors(), ladder.levels[k + 1].m2->terminator.false_block) ==
		                  level.fin->ImmPredecessors().end()))
			return std::nullopt;
		if (k > 0) {
			if (!only_phis(level.fin) || level.fin->terminator.kind != CFG::TerminatorKind::Branch ||
			    level.fin->terminator.true_block != ladder.levels[k - 1].fin)
				return std::nullopt;
			ladder.skipped.insert(level.fin);
		}
		if (k > 0 && !only_reads(level.a)) return std::nullopt;
		for (const auto* block: {level.b, level.m1, level.m2}) ladder.skipped.insert(block);
		if (level.a != ladder.levels.front().a) ladder.skipped.insert(level.a);
	}
	ladder.join = ladder.levels.front().fin;
	if (ladder.levels.front().a->ImmPredecessors().size() != 1u)
		return std::nullopt;
	members.erase(ladder.join);
	for (const auto* block: program.blocks) {
		if (members.contains(block)) continue;
		if (std::ranges::any_of(ladder.levels, [&](const auto& level) { return level.region->blocks.contains(block); }))
			continue;
		for (const auto& inst: *block) {
			if (block == ladder.join && inst.GetOpcode() == Op::Phi) continue;
			for (size_t index = 0; index < inst.NumArgs(); index++) {
				const auto* arg = inst.Arg(index).Resolve().TryInstruction();
				if (arg != nullptr && members.contains(arg->Parent()) && arg->GetOpcode() == Op::Phi)
					return std::nullopt;
			}
		}
		for (const auto& value: {block->condition, block->indirect_target}) {
			const auto* arg = value.IsEmpty() ? nullptr : value.Resolve().TryInstruction();
			if (arg != nullptr && members.contains(arg->Parent()) && arg->GetOpcode() == Op::Phi)
				return std::nullopt;
		}
	}
	return ladder;
}

void EmitCallLadder(ValueEmitContext& ctx, StructuredFunctionState& structured, const CallLadder& ladder) {
	auto& state = ctx.state;
	const auto instructions = [&](const IR::Block* block) {
		EmitBlockInstructions(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
	};
	EmitLabel(state, ladder.levels.front().a->Definition());
	for (const auto& level: ladder.levels) {
		instructions(level.a);
		instructions(level.b);
		for (uint32_t half = 0; half < state.lane_count; half++) {
			auto& lane = half == 0 ? ctx : *ctx.other_half;
			for (const auto* merge: {level.m1, level.m2}) {
				const auto* from = merge == level.m1 ? level.b : level.a;
				for (const auto& phi: *merge) {
					if (phi.GetOpcode() != IR::ValueOpcode::Phi) continue;
					size_t index = 0;
					while (index < phi.NumArgs() && phi.PhiBlock(index) != from) index++;
					if (index == phi.NumArgs()) ctx.Fail(phi, "has no operand for a level of its call table");
					lane.Define(phi, lane.Def(phi.Arg(index)));
				}
			}
		}
	}
	uint32_t selected = ConstantU32(state, 0);
	for (size_t level = ladder.levels.size(); level-- > 0;) {
		const auto& entry   = ladder.levels[level];
		const auto  matched = Binary(state, spv::OpLogicalAnd, TypeBool(state), ctx.Def(entry.a->condition),
		                             ctx.Def(entry.b->condition));
		selected = Select(state, TypeU32(state), matched, ConstantU32(state, static_cast<uint32_t>(level) + 1u),
		                  selected);
	}
	structured.block_exit_labels.emplace(ladder.levels.front().a, state.current_label);
	state.current_block = nullptr;
	state.builder.AddFunction(spv::OpSelectionMerge, ladder.join->Definition(), spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, selected, ladder.none->Definition()};
	for (size_t level = 0; level < ladder.levels.size(); level++) {
		words.push_back(static_cast<uint32_t>(level) + 1u);
		words.push_back(ladder.levels[level].region->entry->Definition());
	}
	state.builder.AddFunction(words);
	for (const auto& level: ladder.levels) EmitOutlinedCall(ctx, *level.region, ladder.join->Definition());
}

void EmitStructuredFunction(ValueEmitContext& ctx, std::vector<OutlinedRegion>& regions) {
	const auto& program = ctx.state.program;
	StructuredFunctionState structured;
	for (const auto& region: regions)
		for (const auto* block: region.blocks) structured.region_of.emplace(block, &region);
	const auto ladder = FindCallLadder(program, regions);
	std::unordered_set<const OutlinedRegion*> ladder_regions;
	if (ladder) {
		LOGF("ShaderRecompiler hash=0x%016" PRIx64 " emitted the %" PRIu64 " levels of its call table as one switch\n",
		     ctx.state.program.shader_hash, static_cast<uint64_t>(ladder->levels.size()));
		structured.ladder = &*ladder;
		for (const auto& level: ladder->levels) ladder_regions.insert(level.region);
	}
	ctx.state.builder.AddFunction(spv::OpBranch, program.blocks.front()->Definition());
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		if (const auto found = structured.region_of.find(block); found != structured.region_of.end()) {
			if (found->second->entry == block && !ladder_regions.contains(found->second))
				EmitOutlinedCall(ctx, *const_cast<OutlinedRegion*>(found->second));
			continue;
		}
		if (ladder && ladder->skipped.contains(block)) continue;
		if (ladder && block == ladder->levels.front().a) {
			EmitCallLadder(ctx, structured, *ladder);
			continue;
		}
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
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
	const auto done = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), done, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
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

}

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32: return TypeU32(state);
		case IR::Type::F16: return TypeF16(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::Bvh8Result: return TypeBvh8Result(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
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
			case IR::Type::F16:
				return state.builder.Constant(spv::OpConstant, TypeF16(state), value.F16Bits());
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

bool QuadLanes(const EmitterState& state) {
	return state.quad_lanes && state.helper_invocation_variable != 0;
}

uint32_t QuadBallot(EmitterState& state, uint32_t predicate) {
	state.builder.RequireCapability(spv::CapabilityGroupNonUniformQuad);
	const auto vec   = TypeU32Vector(state, 4);
	const auto u32   = TypeU32(state);
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto splat = [&](uint32_t value) {
		const auto c  = ConstantU32(state, value);
		const auto id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, vec, id, c, c, c, c);
		return id;
	};
	const auto ballot = [&](uint32_t value) {
		const auto id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, vec, id, scope, value);
		return id;
	};
	const auto helper = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeBool(state), helper, state.helper_invocation_variable);
	const auto helper_true = Binary(state, spv::OpLogicalAnd, TypeBool(state), predicate, helper);
	const auto word   = Select(state, u32, helper_true, ConstantU32(state, 1), ConstantU32(state, 0));
	auto       result = ballot(predicate);
	const auto nibble = splat(0x11111111u);
	for (uint32_t k = 0; k < 4; k++) {
		const auto mate = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformQuadBroadcast, u32, mate, scope, word,
		                          ConstantU32(state, k));
		const auto bits =
		    ballot(Binary(state, spv::OpINotEqual, TypeBool(state), mate, ConstantU32(state, 0)));
		auto any = bits;
		for (uint32_t shift = 1; shift < 4; shift++) {
			any = Binary(state, spv::OpBitwiseOr, vec, any,
			             Binary(state, spv::OpShiftRightLogical, vec, bits, splat(shift)));
		}
		any = Binary(state, spv::OpBitwiseAnd, vec, any, nibble);
		if (k != 0) {
			any = Binary(state, spv::OpShiftLeftLogical, vec, any, splat(k));
		}
		result = Binary(state, spv::OpBitwiseOr, vec, result, any);
	}
	return result;
}

uint32_t QuadShuffle(EmitterState& state, uint32_t type, uint32_t value, uint32_t lane) {
	state.builder.RequireCapability(spv::CapabilityGroupNonUniformQuad);
	const auto u32     = TypeU32(state);
	const auto boolean = TypeBool(state);
	const auto scope   = ConstantU32(state, spv::ScopeSubgroup);
	const auto index   = Binary(state, spv::OpBitwiseAnd, u32, lane, ConstantU32(state, 3));
	const auto base    = Binary(state, spv::OpBitwiseAnd, u32, lane, ConstantU32(state, ~3u));
	const auto helper  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, boolean, helper, state.helper_invocation_variable);
	const auto live = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), live, scope,
	                          Unary(state, spv::OpLogicalNot, boolean, helper));
	const auto extract = [&](uint32_t word) {
		const auto id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, u32, id, live, word);
		return id;
	};
	auto live_word = extract(0);
	if (state.program.wave_size == 64u) {
		live_word = Select(state, u32,
		                   Binary(state, spv::OpUGreaterThanEqual, boolean, lane, ConstantU32(state, 32)),
		                   extract(1), live_word);
	}
	const auto quad_live = Binary(
	    state, spv::OpBitwiseAnd, u32,
	    Binary(state, spv::OpShiftRightLogical, u32, live_word,
	           Binary(state, spv::OpBitwiseAnd, u32, base, ConstantU32(state, 31))),
	    ConstantU32(state, 15));
	const auto source = Select(
	    state, u32, Binary(state, spv::OpINotEqual, boolean, quad_live, ConstantU32(state, 0)),
	    Binary(state, spv::OpIAdd, u32, base,
	           EmitGlsl<GLSLstd450FindILsb, IR::Type::U32>(state, quad_live)),
	    lane);
	uint32_t picked = 0;
	for (uint32_t k = 0; k < 4; k++) {
		const auto mate = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformQuadBroadcast, type, mate, scope, value,
		                          ConstantU32(state, k));
		picked = k == 0 ? mate
		                : Select(state, type,
		                         Binary(state, spv::OpIEqual, boolean, index, ConstantU32(state, k)),
		                         mate, picked);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, result, scope, picked, source);
	return result;
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	if (other_half == nullptr && quad_read && QuadLanes(state)) {
		return QuadBallot(state, Def(predicate));
	}
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto low         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, low, scope,
	                          other_half == nullptr || half == 0 ? Def(predicate)
	                                                             : other_half->Def(predicate));
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

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	if (other_half == nullptr && quad_read && QuadLanes(state)) {
		return QuadShuffle(state, type, Arg(inst, index), lane);
	}
	const auto low   = state.builder.AllocateId();
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
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
}

#if defined(ANYPS5_KYTY)
uint32_t EmitPackedUnormBlend(EmitterState& state, const ShaderPackedBlend& blend, bool truncate,
                              uint32_t source, uint32_t destination) {
	using Factor = Prospero::BlendFactor;
	const auto info = Format::GetFormatInfo(Prospero::BufferFormat::k11_11_10UNorm);
	const auto f32  = TypeF32(state);
	const auto u32  = TypeU32(state);
	const auto zero = ConstantF32Value(state, 0.0f);
	const auto one  = ConstantF32Value(state, 1.0f);
	const auto exact = [&](spv::Op opcode, uint32_t lhs, uint32_t rhs) {
		const auto result = Binary(state, opcode, f32, lhs, rhs);
		state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
		return result;
	};
	const auto saturate = [&](uint32_t value) {
		const auto bits = Unary(state, spv::OpBitcast, u32, value);
		value = Select(state, f32, EmitClassifyF32Bits(state, bits).nan, zero, value);
		return EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, value, zero, one);
	};
	const auto extract = [&](uint32_t composite, uint32_t index) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, f32, result, composite, index);
		return result;
	};
	auto source_alpha = extract(source, 3u);
	if (blend.blend && blend.clamp) {
		source_alpha = saturate(source_alpha);
	}
	uint32_t packed = 0;
	for (uint32_t component = 0; component < info.component_count; component++) {
		const auto bits       = info.component_bits[component];
		const auto offset     = info.component_bit_offset[component];
		const auto field_mask = (1u << bits) - 1u;
		uint32_t   field      = 0;
		if ((blend.write_mask & (1u << component)) == 0u) {
			field = Binary(state, spv::OpBitwiseAnd, u32, destination,
			               ConstantU32(state, field_mask << offset));
		} else {
			auto src    = extract(source, component);
			auto result = src;
			if (blend.blend) {
				if (blend.clamp) {
					src = saturate(src);
				}
				const auto stored = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpBitFieldUExtract, u32, stored, destination,
				                          ConstantU32(state, offset), ConstantU32(state, bits));
				const auto dst = exact(spv::OpFMul, Unary(state, spv::OpConvertUToF, f32, stored),
				                       ConstantF32Value(state, std::nextafter(1.0f / static_cast<float>(field_mask), 1.0f)));
				const auto weighted = [&](uint8_t factor, uint32_t operand) {
					switch (static_cast<Factor>(factor)) {
						case Factor::kZero: return zero;
						case Factor::kOne: return operand;
						case Factor::kSrcColor: return exact(spv::OpFMul, operand, src);
						case Factor::kOneMinusSrcColor:
							return exact(spv::OpFMul, operand, exact(spv::OpFSub, one, src));
						case Factor::kSrcAlpha: return exact(spv::OpFMul, operand, source_alpha);
						case Factor::kOneMinusSrcAlpha:
							return exact(spv::OpFMul, operand, exact(spv::OpFSub, one, source_alpha));
						case Factor::kDstAlpha: return exact(spv::OpFMul, operand, one);
						case Factor::kOneMinusDstAlpha: return exact(spv::OpFMul, operand, zero);
						case Factor::kDstColor: return exact(spv::OpFMul, operand, dst);
						case Factor::kOneMinusDstColor:
							return exact(spv::OpFMul, operand, exact(spv::OpFSub, one, dst));
						case Factor::kSrcAlphaSaturate:
							return exact(spv::OpFMul, operand,
							             EmitGlsl<GLSLstd450FMin, IR::Type::F32>(state, source_alpha, zero));
						default: EXIT("unsupported packed UNorm blend factor %u\n", factor);
					}
					return zero;
				};
				switch (static_cast<Prospero::BlendOp>(blend.op)) {
					case Prospero::BlendOp::kAdd:
						result = exact(spv::OpFAdd, weighted(blend.src_factor, src),
						               weighted(blend.dst_factor, dst));
						break;
					case Prospero::BlendOp::kSubtract:
						result = exact(spv::OpFSub, weighted(blend.src_factor, src),
						               weighted(blend.dst_factor, dst));
						break;
					case Prospero::BlendOp::kReverseSubtract:
						result = exact(spv::OpFSub, weighted(blend.dst_factor, dst),
						               weighted(blend.src_factor, src));
						break;
					case Prospero::BlendOp::kMin:
						result = EmitGlsl<GLSLstd450FMin, IR::Type::F32>(state, src, dst);
						break;
					case Prospero::BlendOp::kMax:
						result = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(state, src, dst);
						break;
					default: EXIT("unsupported packed UNorm blend operation %u\n", blend.op);
				}
			}
			auto scaled = exact(spv::OpFMul, saturate(result),
			                    ConstantF32Value(state, static_cast<float>(field_mask)));
			if (!truncate) {
				scaled = exact(spv::OpFAdd, scaled, ConstantF32Value(state, 0.5f));
			}
			field = Unary(state, spv::OpConvertFToU, u32, scaled);
			if (offset != 0u) {
				field = Binary(state, spv::OpShiftLeftLogical, u32, field, ConstantU32(state, offset));
			}
		}
		packed = packed == 0 ? field : Binary(state, spv::OpBitwiseOr, u32, packed, field);
	}
	return packed;
}

void EmitPixelEntryPoint(EmitterState& state) {
	const auto& pixel = *state.input_info.pixel;
	const auto  u32   = TypeU32(state);
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.builder.AllocateId());
	for (const auto pending: state.packed_pending_variables) {
		if (pending != 0) {
			state.builder.AddFunction(spv::OpStore, pending, ConstantU32(state, 0u));
		}
	}
	state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
	                          state.pixel_guest_func);
	state.builder.AddFunction(spv::OpBeginInvocationInterlockEXT);
	const auto frag_coord = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), frag_coord,
	                          state.packed_frag_coord_variable);
	const auto position = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpVectorShuffle, TypeF32Vector(state, 2), position, frag_coord,
	                          frag_coord, 0u, 1u);
	const auto coord = Unary(state, spv::OpConvertFToU, TypeU32Vector(state, 2), position);
	uint32_t   valid = 0;
	if (state.pixel_valid_mask_variable != 0) {
		const auto mask = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, u32, mask, state.pixel_valid_mask_variable);
		valid = Binary(state, spv::OpINotEqual, TypeBool(state), mask, ConstantU32(state, 0u));
	}
	const auto image_pointer =
	    TypePointer(state, spv::StorageClassUniformConstant, state.packed_target_type);
	uint32_t binding = 0;
	for (uint32_t index = 0; index < 8u; index++) {
		if ((pixel.target_packed_blend_mask & (1u << index)) == 0u) continue;
		const auto pending = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, u32, pending, state.packed_pending_variables[index]);
		auto write = Binary(state, spv::OpINotEqual, TypeBool(state), pending, ConstantU32(state, 0u));
		if (valid != 0) {
			write = Binary(state, spv::OpLogicalAnd, TypeBool(state), write, valid);
		}
		const auto element = binding++;
		EmitIfCondition(state, write, [&] {
			const auto pointer = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAccessChain, image_pointer, pointer,
			                          state.packed_target_variable, ConstantU32(state, element));
			const auto image = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, state.packed_target_type, image, pointer);
			const auto texel = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpImageRead, TypeU32Vector(state, 4), texel, image, coord);
			const auto destination = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, u32, destination, texel, 0u);
			const auto source = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeF32Vector(state, 4), source,
			                          state.packed_source_variables[index]);
			const auto word = EmitPackedUnormBlend(
			    state, ShaderPackedBlend::Decode(pixel.target_packed_blend[index]),
			    (pixel.target_round_mode & (1u << index)) != 0u, source, destination);
			const auto result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, word,
			                          ConstantU32(state, 0u), ConstantU32(state, 0u), ConstantU32(state, 0u));
			state.builder.AddFunction(spv::OpImageWrite, image, coord, result);
		});
	}
	state.builder.AddFunction(spv::OpEndInvocationInterlockEXT);
	EmitKillIfPixelValidMaskInactive(state);
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}
#endif

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.program.info.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			inst.ResetDefinitions();
		}
		block->SetDefinition(state.builder.AllocateId());
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
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
	state.converged_calls = program.stage == ShaderType::Compute && !program.inlined_call_entries.empty() &&
	                        !program.dispatcher_fallback && !FindOutlinedRegions(program).empty();
	if (state.lane_count == 2 && !program.dispatcher_fallback && program.inlined_call_entries.empty()) {
		state.pair_shared = FindPairSharedValues(program);
	}
	DefineGetBdaPointer(state);
	DefineIndirectBufferLoads(ctx);
	DefineBvhIntersect(state);
	DefineTrinityFunctions(state);
	if (program.info.uses_swizzle) {
		ctx.scratch_u32_variable = state.builder.AllocateId();
		if (state.lane_count == 2) {
			high.scratch_u32_variable = state.builder.AllocateId();
		}
	}
	if (program.info.relative_vgprs) {
		ctx.relative_vgpr_variable = state.builder.AllocateId();
		if (state.lane_count == 2) {
			high.relative_vgpr_variable = state.builder.AllocateId();
		}
	}
	auto regions = FindOutlinedRegions(program);
	const auto per_invocation = regions.empty() ? spv::StorageClassFunction : spv::StorageClassPrivate;
	if (!regions.empty()) {
		state.scratch_storage_class = spv::StorageClassPrivate;
		if (state.program.info.function_lds) {
			state.lds_storage_class = spv::StorageClassPrivate;
			state.builder.DefineGlobalVariable(
			    state.lds_variable, TypeU32ArrayPointer(state, per_invocation, LdsDwordCount(state)),
			    per_invocation);
		}
		if (state.program.info.function_scratch) {
			for (uint32_t half = 0; half < state.lane_count; half++) {
				state.builder.DefineGlobalVariable(
				    state.scratch_variable[half],
				    TypeU32ArrayPointer(state, per_invocation, state.program.scratch_dwords), per_invocation);
			}
		}
		for (auto* lane: {&ctx, &high}) {
			if (lane->scratch_u32_variable != 0) {
				state.builder.DefineGlobalVariable(lane->scratch_u32_variable,
				                                   TypePointer(state, per_invocation, TypeU32(state)),
				                                   per_invocation);
			}
			if (lane->relative_vgpr_variable != 0) {
				state.builder.DefineGlobalVariable(
				    lane->relative_vgpr_variable,
				    TypeU32ArrayPointer(state, per_invocation, state.program.relative_vgpr_dwords), per_invocation);
			}
		}
		for (auto& region: regions) {
			EmitOutlinedFunction(ctx, region);
		}
		LOGF("ShaderRecompiler hash=0x%016" PRIx64 " outlined %" PRIu64 " inlined call targets into functions\n",
		     program.shader_hash, static_cast<uint64_t>(regions.size()));
	}
#if defined(ANYPS5_KYTY)
	const auto guest_func = state.mesh_guest_func != 0    ? state.mesh_guest_func
	                        : state.pixel_guest_func != 0 ? state.pixel_guest_func
	                                                      : state.main_func;
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), guest_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
#else
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
#endif
	EmitLabel(state, state.entry_label);
	if (state.program.info.function_lds && per_invocation == spv::StorageClassFunction) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.program.info.function_scratch && per_invocation == spv::StorageClassFunction) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	DeclareRayQueryVariables(state);
#if defined(ANYPS5_KYTY)
	if (state.pixel_valid_mask_variable != 0 && state.pixel_guest_func != 0) {
		state.builder.DefineGlobalVariable(state.pixel_valid_mask_variable,
		                                   TypePointer(state, spv::StorageClassPrivate, TypeU32(state)),
		                                   spv::StorageClassPrivate);
	} else if (state.pixel_valid_mask_variable != 0) {
#else
	if (state.pixel_valid_mask_variable != 0) {
#endif
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
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
		if (lane.scratch_u32_variable != 0 && per_invocation == spv::StorageClassFunction) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
		if (lane.relative_vgpr_variable != 0 && per_invocation == spv::StorageClassFunction) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.relative_vgpr_dwords),
			    lane.relative_vgpr_variable, spv::StorageClassFunction);
		}
	}
	EmitFunctionEntryValues(state);
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	InitializeRayQueryVariables(state);
	EmitTableMode(state);
	EmitMemoryBindings(state);
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx, regions);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	state.current_block = nullptr;
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
#if defined(ANYPS5_KYTY)
	if (state.pixel_guest_func != 0) {
		EmitPixelEntryPoint(state);
	}
#endif
	DefineBdaLoads(state);
}

}

#if defined(ANYPS5_KYTY)
namespace Libs::Graphics::ShaderRecompiler::Spirv {

std::vector<uint32_t> EmitPackedUnormBlendProbe(uint32_t blend, bool truncate) {
	using namespace Emitter;
	constexpr uint32_t RecordWords = 5u;
	IR::Program        program;
	program.stage = ShaderType::Compute;
	ShaderComputeInputInfo compute {};
	ShaderStageInputInfo   input {};
	input.compute = &compute;
	EmitterState state(program, input, DefaultProfile);
	auto&        builder = state.builder;
	const auto   u32     = TypeU32(state);
	const auto   f32     = TypeF32(state);
	const auto   array   = builder.DecoratedType(
	    spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, u32);
	const auto block = builder.DecoratedType(
	    spv::OpTypeStruct,
	    {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}}, {spv::OpDecorate, {spv::DecorationBlock}}},
	    array);
	const auto              element_pointer = TypePointer(state, spv::StorageClassStorageBuffer, u32);
	std::array<uint32_t, 2> buffers {};
	for (uint32_t binding = 0; binding < buffers.size(); binding++) {
		buffers[binding] = builder.DefineGlobalVariable(TypePointer(state, spv::StorageClassStorageBuffer, block),
		                                                spv::StorageClassStorageBuffer);
		builder.AddAnnotation(spv::OpDecorate, buffers[binding], spv::DecorationDescriptorSet, 0u);
		builder.AddAnnotation(spv::OpDecorate, buffers[binding], spv::DecorationBinding, binding);
	}
	const auto invocation = DefineInterfaceVariable(state, TypeU32Vector(state, 3), spv::StorageClassInput,
	                                                "gl_GlobalInvocationID");
	builder.AddAnnotation(spv::OpDecorate, invocation, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);
	state.main_func = builder.AllocateId();
	builder.RequireCapability(spv::CapabilityShader);
	builder.RequireCapability(spv::CapabilitySignedZeroInfNanPreserve);
	builder.RequireExtension("SPV_KHR_float_controls");
	builder.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
	builder.AddExecutionMode(state.main_func, spv::ExecutionModeSignedZeroInfNanPreserve, 32u);
	builder.AddExecutionMode(state.main_func, spv::ExecutionModeLocalSize, 64u, 1u, 1u);
	builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func, spv::FunctionControlMaskNone,
	                    TypeFunction(state));
	EmitLabel(state, builder.AllocateId());
	const auto ids = builder.AllocateId();
	builder.AddFunction(spv::OpLoad, TypeU32Vector(state, 3), ids, invocation);
	const auto index = builder.AllocateId();
	builder.AddFunction(spv::OpCompositeExtract, u32, index, ids, 0u);
	const auto word_pointer = [&](uint32_t buffer, uint32_t element) {
		const auto pointer = builder.AllocateId();
		builder.AddFunction(spv::OpAccessChain, element_pointer, pointer, buffer, ConstantU32(state, 0u), element);
		return pointer;
	};
	const auto base = Binary(state, spv::OpIMul, u32, index, ConstantU32(state, RecordWords));
	std::array<uint32_t, RecordWords> words {};
	for (uint32_t word = 0; word < RecordWords; word++) {
		words[word] = builder.AllocateId();
		builder.AddFunction(spv::OpLoad, u32, words[word],
		                    word_pointer(buffers[0], Binary(state, spv::OpIAdd, u32, base, ConstantU32(state, word))));
	}
	const auto source = builder.AllocateId();
	builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), source,
	                    Unary(state, spv::OpBitcast, f32, words[0]), Unary(state, spv::OpBitcast, f32, words[1]),
	                    Unary(state, spv::OpBitcast, f32, words[2]), Unary(state, spv::OpBitcast, f32, words[3]));
	const auto result =
	    EmitPackedUnormBlend(state, ShaderPackedBlend::Decode(blend), truncate, source, words[4]);
	builder.AddFunction(spv::OpStore, word_pointer(buffers[1], index), result);
	builder.AddFunction(spv::OpReturn);
	builder.AddFunction(spv::OpFunctionEnd);
	builder.AddEntryPoint(spv::ExecutionModelGLCompute, state.main_func, "main", state.interface_variables);
	return builder.Build();
}

}
#endif
