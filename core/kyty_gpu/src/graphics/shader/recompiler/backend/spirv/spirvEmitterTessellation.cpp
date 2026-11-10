#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "common/workaroundHits.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t TessellationPointer(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state          = ctx.state;
	using Attribute      = IR::TessellationAttribute;
	const auto  kind     = static_cast<Attribute>(inst.Arg(0).U32());
	const auto& tess     = state.input_info.vertex->tess;
	const auto  variable = state.tess_variables.at(static_cast<uint32_t>(kind));
	EXIT_IF(variable == 0);
	const auto input   = kind == Attribute::ControlInput || kind == Attribute::EvaluationInput ||
	                   kind == Attribute::PatchInput || kind == Attribute::CarriedInput;
	const auto storage = input ? spv::StorageClassInput : spv::StorageClassOutput;
	const auto pointer = state.builder.AllocateId();
	if (kind == Attribute::CarriedOutput || kind == Attribute::CarriedInput) {
		// One u32 per carried VGPR, four per vec4; the HS invocation reads its own control point's (the lane the LS
		// part ran in for it).
		EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
		const auto rank      = inst.Arg(1).U32() / 4u;
		const auto attribute = ConstantU32(state, rank / 4u);
		const auto component = ConstantU32(state, rank % 4u);
		const auto type      = TypePointer(state, storage, TypeU32(state));
		if (kind == Attribute::CarriedOutput) {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute, component);
		} else {
			state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, EmitLaneId(state), attribute,
			                          component);
		}
		return pointer;
	}
	if (kind == Attribute::Factor) {
		// Hardware tess-factor layout: the outer factors of the domain, then its inner ones (isoline 2+0, triangle
		// 3+1, quad 4+2).
		EXIT_NOT_IMPLEMENTED(!inst.Arg(1).IsImmediate());
		const auto index       = inst.Arg(1).U32() / 4u;
		const auto outer_count = tess.domain == 2u ? 4u : tess.domain == 1u ? 3u : 2u;
		const auto inner_count = tess.domain == 2u ? 2u : tess.domain == 1u ? 1u : 0u;
		const auto outer       = index < outer_count;
		EXIT_NOT_IMPLEMENTED(index >= outer_count + inner_count);
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, TypeF32(state)),
		                          pointer, outer ? variable : state.tess_inner_variable,
		                          ConstantU32(state, outer ? index : index - outer_count));
		return pointer;
	}
	auto address = ctx.Arg(inst, 1);
	if (kind == Attribute::PatchOutput || kind == Attribute::PatchInput) {
		// Fixed per-patch layout shared by both stages: 16 vec4 slots, slot = address / 16 % 16.
		const auto slot = EmitBinaryU32(state, spv::OpBitwiseAnd,
		                                EmitBinaryU32(state, spv::OpShiftRightLogical, address, ConstantU32(state, 4)),
		                                ConstantU32(state, 15));
		const auto component =
		    EmitBinaryU32(state, spv::OpBitwiseAnd,
		                  EmitBinaryU32(state, spv::OpShiftRightLogical, address, ConstantU32(state, 2)),
		                  ConstantU32(state, 3));
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, TypeU32(state)), pointer, variable,
		                          slot, component);
		return pointer;
	}
	const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
	const auto stride = local ? tess.ls_stride : tess.hs_stride;
	const auto offset = EmitBinaryU32(state, spv::OpUMod, address, ConstantU32(state, stride));
	const auto attribute =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 4));
	const auto component =
	    EmitBinaryU32(state, spv::OpBitwiseAnd,
	                  EmitBinaryU32(state, spv::OpShiftRightLogical, offset, ConstantU32(state, 2)),
	                  ConstantU32(state, 3));
	const auto type = TypePointer(state, storage, TypeU32(state));
	if (kind == Attribute::LocalOutput || kind == Attribute::PatchOutput) {
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, attribute,
		                          component);
	} else {
		const auto vertex =
		    kind == Attribute::ControlOutput
		        ? EmitLaneId(state)
		        : EmitBinaryU32(state, spv::OpUDiv, address, ConstantU32(state, stride));
		state.builder.AddFunction(spv::OpAccessChain, type, pointer, variable, vertex, attribute,
		                          component);
	}
	return pointer;
}

} // namespace

void DefineTessellationInterfaces(EmitterState& state) {
	using Attribute = IR::TessellationAttribute;
	std::array<bool, IR::NumTessellationAttributes> used {};
	// KYTY_TRACE_PERF (PERF workaround_hits tess_ring): the 16-byte blocks of per-patch data in the fixed layout, by slot.
	const bool          count_slots = Common::WorkaroundHits::Enabled();
	std::array<uint32_t, 16> slot_block {};
	slot_block.fill(UINT32_MAX);
	bool slot_shared = false, dynamic_patch = false;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::GetTessellationAttribute &&
			    inst.GetOpcode() != IR::ValueOpcode::SetTessellationAttribute)
				continue;
			const auto kind = inst.Arg(0).U32();
			used.at(kind)   = true;
			if (count_slots &&
			    (kind == static_cast<uint32_t>(Attribute::PatchOutput) || kind == static_cast<uint32_t>(Attribute::PatchInput))) {
				// A constant, or a constant plus a component's byte offset (LowerTessellationMemory's IAdd32).
				uint32_t   address = 0;
				const auto value   = inst.Arg(1).Resolve();
				const auto* add    = value.IsImmediate() ? nullptr : value.TryInstruction();
				if (value.IsImmediate()) {
					address = value.U32();
				} else if (add != nullptr && add->GetOpcode() == IR::ValueOpcode::IAdd32 && add->Arg(0).Resolve().IsImmediate() &&
				           add->Arg(1).Resolve().IsImmediate()) {
					address = add->Arg(0).Resolve().U32() + add->Arg(1).Resolve().U32();
				} else {
					dynamic_patch = true;
					continue;
				}
				// Two different 16-byte blocks in one of the 16 slots (slot = address / 16 % 16) would overwrite each other.
				const auto block_index = address / 16u;
				auto&      slot        = slot_block[block_index % 16u];
				if (slot != UINT32_MAX && slot != block_index) slot_shared = true;
				slot = block_index;
			}
		}
	}
	if (slot_shared) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::TessPatchSlotShared);
	if (dynamic_patch) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::TessPatchDynamic);
	if (std::ranges::none_of(used, [](bool value) { return value; })) return;
	const auto& tess  = state.input_info.vertex->tess;
	const auto  array = [&](uint32_t type, uint32_t count) {
		return state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	};
	for (uint32_t index = 0; index < used.size(); index++) {
		if (!used[index]) continue;
		const auto kind    = static_cast<Attribute>(index);
		const bool input   = kind == Attribute::ControlInput || kind == Attribute::EvaluationInput ||
		                   kind == Attribute::PatchInput || kind == Attribute::CarriedInput;
		const auto storage = input ? spv::StorageClassInput : spv::StorageClassOutput;
		const bool carried = kind == Attribute::CarriedOutput || kind == Attribute::CarriedInput;
		uint32_t   type;
		if (kind == Attribute::Factor) {
			type = array(TypeF32(state), 4u);
		} else if (carried) {
			uint32_t count = 0;
			for (const auto word: tess.carried_vgprs) count += static_cast<uint32_t>(std::popcount(word));
			EXIT_IF(count == 0u);
			type = array(TypeU32Vector(state, 4), (count + 3u) / 4u);
			if (kind == Attribute::CarriedInput) type = array(type, tess.input_control_points);
		} else if (kind == Attribute::PatchOutput || kind == Attribute::PatchInput) {
			type = array(TypeU32Vector(state, 4), 16u);
		} else {
			const bool local  = kind == Attribute::LocalOutput || kind == Attribute::ControlInput;
			const auto stride = local ? tess.ls_stride : tess.hs_stride;
			type              = array(TypeU32Vector(state, 4), (stride + 15u) / 16u);
			if (kind != Attribute::LocalOutput) {
				type = array(type, local ? tess.input_control_points : tess.output_control_points);
			}
		}
		auto& variable = state.tess_variables[index];
		variable       = DefineInterfaceVariable(state, type, storage, "tess_attributes");
		if (kind == Attribute::Factor) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
			                            spv::BuiltInTessLevelOuter);
			state.tess_inner_variable =
			    DefineInterfaceVariable(state, array(TypeF32(state), 2u), storage, "tess_inner");
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationBuiltIn, spv::BuiltInTessLevelInner);
			state.builder.AddAnnotation(spv::OpDecorate, state.tess_inner_variable,
			                            spv::DecorationPatch);
		} else {
			state.builder.AddAnnotation(
			    spv::OpDecorate, variable, spv::DecorationLocation,
			    kind == Attribute::PatchOutput || kind == Attribute::PatchInput ? (tess.hs_stride + 15u) / 16u
			    : carried                                                       ? (tess.ls_stride + 15u) / 16u
			                                                                    : 0u);
		}
		if (kind == Attribute::Factor || kind == Attribute::PatchOutput || kind == Attribute::PatchInput) {
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationPatch);
		}
	}
}

void DefineTessellationExecutionModes(EmitterState& state) {
	const auto& tess = state.input_info.vertex->tess;
	// VGT_TF_PARAM: TYPE 0 isoline / 1 triangle / 2 quad; PARTITIONING 0 integer / 1 pow2 (closest Vulkan spacing:
	// equal) / 2 fractional odd / 3 fractional even; TOPOLOGY 0 point / 1 line / 2 triangle cw / 3 triangle ccw.
	EXIT_NOT_IMPLEMENTED(tess.domain > 2u || tess.partitioning > 3u || tess.output_topology > 3u);
	state.builder.RequireCapability(spv::CapabilityTessellation);
	if (state.program.stage == ShaderType::TessellationControl) {
		state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeOutputVertices,
		                               tess.output_control_points);
	} else {
		state.builder.AddExecutionMode(state.main_func, tess.domain == 0u   ? spv::ExecutionModeIsolines
		                                                : tess.domain == 1u ? spv::ExecutionModeTriangles
		                                                                    : spv::ExecutionModeQuads);
		state.builder.AddExecutionMode(state.main_func,
		                               tess.partitioning == 2u   ? spv::ExecutionModeSpacingFractionalOdd
		                               : tess.partitioning == 3u ? spv::ExecutionModeSpacingFractionalEven
		                                                         : spv::ExecutionModeSpacingEqual);
		if (tess.output_topology == 0u) {
			state.builder.AddExecutionMode(state.main_func, spv::ExecutionModePointMode);
		} else if (tess.output_topology >= 2u) {
			state.builder.AddExecutionMode(state.main_func, tess.output_topology == 2u ? spv::ExecutionModeVertexOrderCw
			                                                                           : spv::ExecutionModeVertexOrderCcw);
		}
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1) < 8u) {
		std::fprintf(stderr, "TESS_MODE stage=%u domain=%u partitioning=%u topology=%u input_cp=%u output_cp=%u ls=%u hs=%u\n",
		             static_cast<uint32_t>(state.program.stage), tess.domain, tess.partitioning, tess.output_topology,
		             tess.input_control_points, tess.output_control_points, tess.ls_stride, tess.hs_stride);
	}
}

uint32_t EmitGetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitValueOrZeroIfCondition(ctx.state, ctx.Arg(inst, 2), [&] {
		const auto value = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpLoad, TypeU32(ctx.state), value,
		                              TessellationPointer(ctx, inst));
		return value;
	});
}

void EmitSetTessellationAttribute(ValueEmitContext& ctx, const IR::Inst& inst) {
	EmitIfCondition(ctx.state, ctx.Arg(inst, 3), [&] {
		auto value = ctx.Arg(inst, 2);
		if (inst.Arg(0).U32() == static_cast<uint32_t>(IR::TessellationAttribute::Factor)) {
			const auto floating = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpBitcast, TypeF32(ctx.state), floating, value);
			value = floating;
		}
		ctx.state.builder.AddFunction(spv::OpStore, TessellationPointer(ctx, inst), value);
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
