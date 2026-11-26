#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t MeshArray(EmitterState& state, spv::StorageClass storage, uint32_t type, uint32_t count) {
	const auto array = state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	return state.builder.DefineGlobalVariable(TypePointer(state, storage, array), storage);
}

uint32_t MeshElement(EmitterState& state, uint32_t variable, spv::StorageClass storage,
                     uint32_t type, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, type), pointer,
	                          variable, index);
	return pointer;
}

uint32_t MeshLoad(EmitterState& state, uint32_t variable, spv::StorageClass storage, uint32_t type,
                  uint32_t index) {
	const auto pointer = MeshElement(state, variable, storage, type, index);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, value, pointer);
	return value;
}

uint32_t MeshOutputType(EmitterState& state, IR::StageOutputKind kind) {
	switch (kind) {
		case IR::StageOutputKind::Layer: return TypeU32(state);
		case IR::StageOutputKind::ClipDistance:
		case IR::StageOutputKind::CullDistance: return TypeF32(state);
		case IR::StageOutputKind::Position:
		case IR::StageOutputKind::Parameter: return TypeF32Vector(state, 4);
		default: EXIT("unsupported mesh output kind=%u\n", static_cast<uint32_t>(kind));
	}
}

uint32_t MeshPrimitiveType(EmitterState& state) {
	const auto size = state.input_info.vertex->mesh.output_primitive_size;
	return size == 1u ? TypeU32(state) : TypeU32Vector(state, size);
}

uint32_t MeshPerPrimitiveType(EmitterState& state, bool has_layer) {
	if (has_layer) {
		return state.builder.DecoratedType(
		    spv::OpTypeStruct,
		    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInCullPrimitiveEXT}},
		     {spv::OpMemberDecorate, {0, spv::DecorationPerPrimitiveEXT}},
		     {spv::OpMemberDecorate, {1, spv::DecorationBuiltIn, spv::BuiltInLayer}},
		     {spv::OpMemberDecorate, {1, spv::DecorationPerPrimitiveEXT}},
		     {spv::OpDecorate, {spv::DecorationBlock}}},
		    TypeBool(state), TypeU32(state));
	}
	return state.builder.DecoratedType(
	    spv::OpTypeStruct,
	    {{spv::OpMemberDecorate, {0, spv::DecorationBuiltIn, spv::BuiltInCullPrimitiveEXT}},
	     {spv::OpMemberDecorate, {0, spv::DecorationPerPrimitiveEXT}},
	     {spv::OpDecorate, {spv::DecorationBlock}}},
	    TypeBool(state));
}

uint32_t MeshPrimitiveMember(EmitterState& state, uint32_t type, uint32_t index, uint32_t member) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, type),
	                          pointer, state.mesh_per_primitive, index, ConstantU32(state, member));
	return pointer;
}

}

void DefineMeshOutputs(EmitterState& state, uint32_t clip_distance_count,
                       uint32_t cull_distance_count) {
	const auto& mesh      = state.input_info.vertex->mesh;
	const bool  has_layer = std::ranges::any_of(state.outputs, [](const OutputBinding& output) {
		return output.kind == IR::StageOutputKind::Layer;
	});
	const auto per_primitive = MeshPerPrimitiveType(state, has_layer);
	state.builder.AddName(per_primitive, "gl_MeshPerPrimitiveEXT");
	state.mesh_per_primitive =
	    MeshArray(state, spv::StorageClassOutput, per_primitive, mesh.max_primitives);
	state.builder.AddName(state.mesh_per_primitive, "gl_MeshPrimitivesEXT");
	for (auto& output: state.outputs) {
		const auto type  = MeshOutputType(state, output.kind);
		const bool clip  = output.kind == IR::StageOutputKind::ClipDistance;
		const bool cull  = output.kind == IR::StageOutputKind::CullDistance;
		const bool layer = output.kind == IR::StageOutputKind::Layer;
		if (layer) {
			output.variable_id = state.mesh_per_primitive;
		} else {
			auto& variable = clip ? state.clip_distance_variable
			                 : cull ? state.cull_distance_variable : output.variable_id;
			if (variable == 0) {
				const auto element_type =
				    clip || cull
				        ? state.builder.Type(spv::OpTypeArray, type,
				                             ConstantU32(state, clip ? clip_distance_count
				                                                     : cull_distance_count))
				        : type;
				variable = MeshArray(state, spv::StorageClassOutput, element_type, mesh.max_vertices);
				state.interface_variables.push_back(variable);
				state.builder.AddName(variable, output.debug_name.c_str());
				if (output.kind == IR::StageOutputKind::Parameter) {
					state.builder.AddAnnotation(spv::OpDecorate, variable,
					                            spv::DecorationLocation, output.location);
				} else {
					const auto builtin = clip ? spv::BuiltInClipDistance
					                     : cull ? spv::BuiltInCullDistance
					                            : spv::BuiltInPosition;
					state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
					                            builtin);
				}
			}
			output.variable_id = variable;
		}
		output.mesh_data_variable =
		    MeshArray(state, layer ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, type,
		              layer ? mesh.max_vertices : state.lane_count);
	}
	state.mesh_allocation = MeshArray(state, spv::StorageClassWorkgroup, TypeU32(state), 2);
	state.mesh_primitive_data =
	    MeshArray(state, spv::StorageClassPrivate, TypeU32(state), state.lane_count);
	state.mesh_primitives = MeshArray(state, spv::StorageClassOutput, MeshPrimitiveType(state),
	                                  mesh.max_primitives);
	state.interface_variables.push_back(state.mesh_primitives);
	state.interface_variables.push_back(state.mesh_per_primitive);
	constexpr spv::BuiltIn indices[] = {spv::BuiltInPrimitivePointIndicesEXT,
	                                    spv::BuiltInPrimitiveLineIndicesEXT,
	                                    spv::BuiltInPrimitiveTriangleIndicesEXT};
	state.builder.AddAnnotation(spv::OpDecorate, state.mesh_primitives, spv::DecorationBuiltIn,
	                            indices[mesh.output_primitive_size - 1u]);
}

uint32_t MeshOutputPointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index) {
	const auto output = std::ranges::find_if(state.outputs, [=](const OutputBinding& binding) {
		return binding.kind == kind && binding.index == index;
	});
	if (output == state.outputs.end()) {
		EXIT("mesh export has no output binding: kind=%u index=%u\n", static_cast<uint32_t>(kind),
		     index);
	}
	const bool shared = kind == IR::StageOutputKind::Layer;
	return MeshElement(
	    state, output->mesh_data_variable,
	    shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, MeshOutputType(state, kind),
	    shared ? EmitLocalInvocationIndex(state) : ConstantU32(state, state.lane_half));
}

uint32_t MeshPrimitivePointer(EmitterState& state) {
	return MeshElement(state, state.mesh_primitive_data, spv::StorageClassPrivate, TypeU32(state),
	                   ConstantU32(state, state.lane_half));
}

void EmitMeshAllocate(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto first = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first,
	                          EmitLocalInvocationIndex(state), ConstantU32(state, 0));
	EmitIfCondition(state, first, [&] {
		const auto allocation = ctx.Arg(inst, 0);
		for (uint32_t field = 0; field < 2; field++) {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), value, allocation,
			                          ConstantU32(state, field * 12u),
			                          ConstantU32(state, field == 0 ? 10u : 11u));
			const auto pointer =
			    MeshElement(state, state.mesh_allocation, spv::StorageClassWorkgroup,
			                TypeU32(state), ConstantU32(state, field));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		}
	});
}

void EmitMeshEntryPoint(EmitterState& state) {
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.builder.AllocateId());
	state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
	                          state.mesh_guest_func);
	EmitBarrier(state);
	const auto vertices   = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 0));
	const auto primitives = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 1));
	state.builder.AddFunction(spv::OpSetMeshOutputsEXT, vertices,
	                          primitives);
	for (uint32_t half = 0; half < state.lane_count; half++) {
		state.lane_half      = half;
		const auto index     = EmitLocalInvocationIndex(state);
		const auto is_vertex = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
		EmitIfCondition(state, is_vertex, [&] {
			for (const auto& output: state.outputs) {
				if (output.kind == IR::StageOutputKind::Layer) {
					continue;
				}
				const auto type  = MeshOutputType(state, output.kind);
				const auto value =
				    MeshLoad(state, output.mesh_data_variable, spv::StorageClassPrivate, type,
				             ConstantU32(state, half));
				uint32_t pointer;
				if (output.kind == IR::StageOutputKind::ClipDistance ||
				    output.kind == IR::StageOutputKind::CullDistance) {
					pointer = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, type), pointer,
					    output.variable_id, index, ConstantU32(state, output.index));
				} else {
					pointer = MeshElement(state, output.variable_id, spv::StorageClassOutput, type, index);
				}
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
		});
		const auto is_primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index,
		                          primitives);
		EmitIfCondition(state, is_primitive, [&] {
			const auto packed = MeshLoad(state, state.mesh_primitive_data, spv::StorageClassPrivate,
			                             TypeU32(state), ConstantU32(state, half));
			const auto size = state.input_info.vertex->mesh.output_primitive_size;
			const auto type = MeshPrimitiveType(state);
			uint32_t   vertex[3] {};
			for (uint32_t component = 0; component < size; component++) {
				vertex[component] = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpBitFieldUExtract, TypeU32(state), vertex[component], packed,
				    ConstantU32(state, component * 10u), ConstantU32(state, 9));
			}
			auto primitive = vertex[0];
			if (size > 1u) {
				primitive = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeConstruct, type, primitive,
				                          std::span<const uint32_t>(vertex, size));
			}
			state.builder.AddFunction(
			    spv::OpStore,
			    MeshElement(state, state.mesh_primitives, spv::StorageClassOutput, type, index),
			    primitive);
			const auto null_bit =
			    EmitBinaryU32(state, spv::OpBitwiseAnd, packed, ConstantU32(state, 0x80000000u));
			const auto culled = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), culled, null_bit,
			                          ConstantU32(state, 0));
			state.builder.AddFunction(
			    spv::OpStore, MeshPrimitiveMember(state, TypeBool(state), index, 0), culled);
			for (const auto& output: state.outputs) {
				if (output.kind != IR::StageOutputKind::Layer) {
					continue;
				}
				const auto layer = MeshLoad(state, output.mesh_data_variable,
				                            spv::StorageClassWorkgroup, TypeU32(state),
				                            vertex[state.input_info.vertex->mesh.provoking_vertex]);
				state.builder.AddFunction(
				    spv::OpStore, MeshPrimitiveMember(state, TypeU32(state), index, 1), layer);
			}
		});
	}
	state.lane_half = 0;
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

}
