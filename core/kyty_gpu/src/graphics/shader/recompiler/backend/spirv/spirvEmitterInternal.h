#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_INTERNAL_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_INTERNAL_H_

#include "common/common.h"
#include "common/envRead.h"
#include "common/stringUtils.h"
#include "common/shaderEnv.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iterator>
#include <map>
#include <spirv/unified1/GLSL.std.450.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

// spirvEmitterLaneShare.cpp: with lane_count == 2 every IR
// instruction is emitted once per emulated wave half. A value that is provably the same in both halves
// is emitted by the low half only; the high half reuses its id.
struct LaneShareState {
	bool active = false; // a lane_count == 2 compute shader
	// Instructions the high half does not emit (subset of `uniform`: not Ballot/ReadFirstLane/handles).
	std::unordered_set<const IR::Inst*> shared;
	// Every instruction proven to have the same value in both halves.
	std::unordered_set<const IR::Inst*> uniform;
	size_t   instructions = 0;
	size_t   candidates   = 0;
	uint64_t reused       = 0;
	uint64_t words_saved  = 0;
};

struct InputBinding : IR::StageInput {
	uint32_t variable_id = 0;
};

struct OutputBinding : IR::StageOutput {
	uint32_t variable_id        = 0;
	uint32_t mesh_data_variable = 0;
};

using ImageDimension = Decoder::ImageDimension;

struct ImageDimensionInfo {
	ImageDimension dimension;
	uint32_t       spirv_dimension;
	uint32_t       coordinate_components;
	uint32_t       spatial_components;
	uint32_t       arrayed;
	uint32_t       multisampled;
};

constexpr std::array<ImageDimensionInfo, 7> ImageDimensions {{
    {ImageDimension::Dim1D, spv::Dim1D, 1, 1, 0, 0},
    {ImageDimension::Dim1DArray, spv::Dim1D, 2, 1, 1, 0},
    {ImageDimension::Dim2D, spv::Dim2D, 2, 2, 0, 0},
    {ImageDimension::Dim2DArray, spv::Dim2D, 3, 2, 1, 0},
    {ImageDimension::Dim3D, spv::Dim3D, 3, 3, 0, 0},
    {ImageDimension::Dim2DMsaa, spv::Dim2D, 2, 2, 0, 1},
    {ImageDimension::Dim2DMsaaArray, spv::Dim2D, 3, 2, 1, 1},
}};

const ImageDimensionInfo& ImageDimensionInfoFor(ImageDimension dimension);

struct BufferDefinition {
	uint32_t variable             = 0;
	uint32_t type                 = 0;
	uint32_t element_type         = 0;
	uint32_t pointer_type         = 0;
	uint32_t element_pointer_type = 0;
};

struct ImageDefinition {
	uint32_t variable     = 0;
	uint32_t type         = 0;
	uint32_t pointer_type = 0;
	uint32_t sampled_type = 0;
};

struct EmitterState {
	EmitterState(IR::Program& program_, ShaderStageInputInfo input_info_, const Profile& profile_)
	    : builder(program_.stage == ShaderType::Mesh ? 0x00010400u : 0x00010300u),
	      program(program_), input_info(input_info_), profile(profile_) {
		if (ShaderWorkgroupInput(program.stage, input_info) != nullptr) {
			lds_storage_class = spv::StorageClassWorkgroup;
		}
		if (program.info.uses_lds && program.stage == ShaderType::Compute &&
		    input_info.compute != nullptr && input_info.compute->lds_storage) {
			lds_storage_class = spv::StorageClassStorageBuffer;
		}
	}

	Builder                                          builder;
	IR::Program&                                     program;
	ShaderStageInputInfo                             input_info;
	const Profile&                                      profile;
	uint32_t                                        debug_source = 0;
	uint32_t                                        void_type = 0;
	uint32_t                                        bool_type = 0;
	uint32_t                                        u32_type = 0;
	uint32_t                                        u64_type = 0;
	uint32_t                                        i32_type = 0;
	uint32_t                                        f32_type = 0;
	uint32_t                                        f64_type = 0;
	// 16-bit floats, only for exact half conversions in hardware (native_half, DeclareNativeHalf).
	uint32_t                                        f16_type      = 0;
	uint32_t                                        f16_pair_type = 0;
	bool                                            native_half   = false;
	uint32_t                                        u32_pair_type = 0;
	uint32_t                                        i32_pair_type = 0;
	uint32_t                                        function_type = 0;
	std::array<uint32_t, 3>                          bool_vector_types {};
	std::array<uint32_t, 3>                          u32_vector_types {};
	std::array<uint32_t, 3>                          i32_vector_types {};
	std::array<uint32_t, 3>                          f32_vector_types {};
	std::array<uint32_t, IR::NumTessellationAttributes> tess_variables {};
	uint32_t                                         tess_inner_variable = 0;

	uint32_t                                         lane_count              = 1;
	uint32_t                                         lane_half               = 0;
	LaneShareState                                   lane_share;
	std::array<BufferDefinition, 4>                  storage_buffers {};
	// The 32-bit buffers alias decorated Coherent that only the cache-bypassing (GLC)
	// accesses of a buffer-read-only shader use (variable 0: not used).
	BufferDefinition                                 coherent_buffers_alias {};
	// KYTY_PERF_BDA_ENTRY_SHARE: per lane half, Function variables holding the page (u64, address >> page bits) and the
	// page's base pointer (u64, 0 = none) of the last scalar load through a GPU-built V# (0: not declared).
	std::array<uint32_t, 2>                          bda_memo_page {};
	std::array<uint32_t, 2>                          bda_memo_base {};
	// Dword loads through one GPU-built V# (same descriptor words and dynamic offset) in one
	// block with no side effect between them; the first in block order (the leader) loads the whole group with one
	// bounds check and one page lookup, the others take the leader's values (per lane half).
	struct ScalarLoadGroup {
		std::vector<const IR::Inst*> members;
		std::vector<uint64_t>        deltas; // member byte offset - c_min
		uint64_t                     c_min = 0;
		uint64_t                     c_max = 0;
	};
	std::vector<ScalarLoadGroup>                                  scalar_groups;
	std::unordered_map<const IR::Inst*, uint32_t>                 scalar_group_leader; // leader -> scalar_groups index
	std::unordered_set<const IR::Inst*>                           scalar_group_members; // non-leader members
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2>  scalar_group_values;
	std::array<uint32_t, IR::ShaderInfo::MaxBuffers> memory_byte_offsets {};
	uint32_t                                         bda_pagetable_variable  = 0;
	uint32_t                                         fault_buffer_variable   = 0;
	uint32_t                                         bda_pointer_function    = 0;
	// One function per formatted load shape through a GPU-selected
	// descriptor, the whole load in one call (shape key: channels | components << 4 | d16 << 8 -> function).
	std::map<uint32_t, uint32_t>                     indirect_formatted_load_functions;
	uint32_t                                         indirect_raw_function = 0;
	uint32_t                                         indirect_raw_coherent_function = 0; // GLC accesses (MemoryInfo::coherent)
	// F1 (grouped IndirectBuffer reads): indirect_raw_x<N>[_coherent], index (N - 2) + 3 * coherent (0: not defined).
	std::array<uint32_t, 6>                          indirect_raw_group_functions {};
	// KYTY_PERF_SPIRV_STORE_OUTLINE (DefineIndirectStoreFunctions): one function per shape of the stores through a
	// GPU-selected descriptor, the formatted X..XYZW stores by component count (1..4 -> function) and the raw dword
	// store; 0 / absent: the store is expanded at its instruction.
	std::map<uint32_t, uint32_t>                     indirect_formatted_store_functions;
	uint32_t                                         indirect_raw_store_function = 0;
	uint32_t                                         indirect_raw_store_coherent_function = 0; // GLC accesses (MemoryInfo::coherent)
	uint32_t                                         bvh_intersect_function  = 0;
	uint32_t                                         bvh8_result_type       = 0; // upstream 5eab0ec73
	uint32_t                                         bvh8_intersect_function = 0;
	uint32_t                                         bvh_stack_function      = 0; // upstream bfd4c34b8
	uint32_t                                         gds_variable            = 0;
	uint32_t                                         gds_length              = 0;
	uint32_t                                         push_constant_variable  = 0;
	uint32_t                                         flattened_srt_variable  = 0;
	// Uniform const buffers (ShaderIR.h): the uniform blocks of ReadConst and of the shader data.
	uint32_t                                         flattened_srt_uniform_variable = 0;
	uint32_t                                         shader_data_uniform_variable   = 0;
	// KYTY_BINDLESS (set 1): one runtime image array per image type, the key -> slot translation
	// and the feedback flags; bindless_slot is the slot of the image instruction being emitted.
	std::map<uint32_t, uint32_t>                     bindless_image_variables;
	uint32_t                                         bindless_translation_variable = 0;
	uint32_t                                         bindless_feedback_variable    = 0;
	uint32_t                                         bindless_slot                 = 0;
	spv::StorageClass                                lds_storage_class = spv::StorageClassFunction;
	// Compact function LDS: per-invocation (function) LDS whose every access is the
	// DS_*_ADDTID pattern (constant base + 4 * lane) keeps one dword per distinct base (byte base -> slot).
	bool                                             compact_lds = false;
	std::unordered_map<uint32_t, uint32_t>           compact_lds_slots;
	uint32_t                                         lds_base_dwords         = 0;
	uint32_t                                         lds_variable            = 0;
	uint32_t                                         lds_u64_variable        = 0;
	std::array<uint32_t, 2>                          scratch_variable {};
	std::array<ImageDefinition, IR::ImageBindingCount> images {};
	std::array<ImageDefinition, IR::ImageBindingCount> heap_images {};
	uint32_t                   sampler_variable                      = 0;
	uint32_t                   sampler_type                          = 0;
	uint32_t                   sampler_pointer_type                  = 0;
	uint32_t                   main_func                             = 0;
	uint32_t                   mesh_guest_func                       = 0;
	uint32_t                   mesh_allocation                       = 0;
	uint32_t                   mesh_primitive_data                   = 0;
	uint32_t                   mesh_primitives                       = 0;
	uint32_t                   mesh_cull                             = 0;
	uint32_t                   entry_label                           = 0;
	uint32_t                   current_label                         = 0;
	const IR::Block*           current_block                         = nullptr;
	uint32_t                   pixel_valid_mask_variable             = 0;
	// Loop watchdog: function-storage counter per structured loop header (by block id),
	// and the latch block that increments it.
	std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> loop_counter_variables;
	std::unordered_map<uint32_t, uint32_t>                      back_edge_counters;
	uint32_t                   subgroup_local_invocation_id_variable = 0;
	uint32_t                   helper_invocation_variable            = 0; // upstream 5429888d4
	// Paired shaders: a wave64 compute shader runs one guest lane per host invocation on a
	// 32-wide host subgroup; the two subgroups of a guest wave exchange whole-wave values
	// (ballots, lane reads) through workgroup memory (see EmitPairExchange).
	bool                       wave_pair                             = false;
	uint32_t                   subgroup_id_variable                  = 0;
	uint32_t                   pair_slots_variable                   = 0;
	uint32_t                   pair_seq_variable                     = 0;
	// Exchanges outside per-invocation guards (guard_depth 0, counted by
	// EmitIfCondition / EmitValueOrDefaultIfCondition) are workgroup barriers.
	bool                       pair_barrier                          = false;
	uint32_t                   guard_depth                           = 0;
	bool                       guardless                             = false; // compute stages: RobustBuffer (spirvEmitterMemory.cpp)
	uint32_t                   indirect_guard                        = 0;     // IndirectCallWord3
	// EmitPairLdsWriteSync: no LDS read since the last pair LDS barrier (false after a read and at every label).
	bool                       lds_pair_synced                       = true;
	uint32_t                   per_vertex_variable                   = 0;
	uint32_t                   point_size_variable                   = 0;
	uint32_t                   clip_distance_variable                = 0;
	uint32_t                   invalid_position_clip_distance        = UINT32_MAX;
	uint32_t                   user_clip_planes                      = 0;
	uint32_t                   cull_distance_variable                = 0;
	uint32_t                   layer_variable                        = 0;
	uint32_t                   viewport_index_variable               = 0;
	uint32_t                   depth_variable                        = 0;
	uint32_t                   object_primitive_id_variable          = 0;
	uint32_t                   sample_mask_variable                  = 0;
	// The ObjPrimID write: the pixel shader's ObjPrimID output (StageOutputKind::ObjPrimId).
	uint32_t                   objprim_output_variable               = 0;
	std::vector<InputBinding>  inputs;
	std::vector<OutputBinding> outputs;
	std::vector<uint32_t>      interface_variables;
};

uint32_t TypeVoid(EmitterState& state);
uint32_t TypeBool(EmitterState& state);
uint32_t TypeBoolVector(EmitterState& state, uint32_t components);
uint32_t TypeU32(EmitterState& state);
uint32_t TypeU64(EmitterState& state);
uint32_t TypeU32Pair(EmitterState& state);
uint32_t TypeBvh8Result(EmitterState& state);
uint32_t TypeI32(EmitterState& state);
uint32_t TypeI32Pair(EmitterState& state);
uint32_t TypeF32(EmitterState& state);
uint32_t TypeF64(EmitterState& state);
uint32_t TypeF16(EmitterState& state);
uint32_t TypeF16Pair(EmitterState& state);
uint32_t TypeU32Vector(EmitterState& state, uint32_t components);

uint32_t TypeU32Composite(EmitterState& state, uint32_t components);
uint32_t TypeI32Vector(EmitterState& state, uint32_t components);
uint32_t TypeF32Vector(EmitterState& state, uint32_t components);
uint32_t TypePointer(EmitterState& state, spv::StorageClass storage_class, uint32_t pointee);
uint32_t TypeFunction(EmitterState& state);
uint32_t TypeStorageBufferElementPointer(EmitterState& state, uint32_t bits = 32);
// The block type of a storage buffer alias (runtime array of `bits`-bit elements), made on first use.
uint32_t StorageBufferType(EmitterState& state, BufferDefinition& buffer, uint32_t bits);
uint32_t TypePhysicalU32Pointer(EmitterState& state);
uint32_t TypePushConstantElementPointer(EmitterState& state);
uint32_t TypeU32ArrayPointer(EmitterState& state, spv::StorageClass storage_class, uint32_t dwords);
uint32_t TypeU32ElementPointer(EmitterState& state, spv::StorageClass storage_class);

inline void EmitLabel(EmitterState& state, uint32_t label) {
	state.current_label   = label;
	state.lds_pair_synced = false; // (EmitPairLdsWriteSync: an LDS read may reach this block from another path)
	state.builder.AddFunction(spv::OpLabel, label);
	if (state.debug_source != 0 && state.current_block != nullptr) {
		// Source lines encode the containing native block's byte offset plus one.
		state.builder.AddFunction(spv::OpLine, state.debug_source,
		                          state.current_block->start_pc + 1u, 0u);
	}
}

uint32_t TypeId(EmitterState& state, IR::Type type);

// Shared instruction construction; typed aliases add no forwarding functions.
template <spv::Op opcode, IR::Type type, typename... Args>
uint32_t EmitNative(EmitterState& state, Args... args) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(opcode, TypeId(state, type), result, args...);
	if constexpr (type == IR::Type::F64 &&
	              (opcode == spv::OpFMul || opcode == spv::OpFDiv || opcode == spv::OpExtInst)) {
		state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	}
	return result;
}

uint32_t GlslStd450(EmitterState& state);

template <GLSLstd450 opcode, IR::Type type, typename... Args>
uint32_t EmitGlsl(EmitterState& state, Args... args) {
	return EmitNative<spv::OpExtInst, type>(state, GlslStd450(state), opcode, args...);
}

inline uint32_t Unary(EmitterState& state, spv::Op opcode, uint32_t type, uint32_t value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(opcode, type, result, value);
	return result;
}

inline uint32_t PackU64(EmitterState& state, uint32_t low, uint32_t high) {
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 2), pair, low, high);
	return Unary(state, spv::OpBitcast, TypeU64(state), pair);
}

inline uint32_t Binary(EmitterState& state, spv::Op opcode, uint32_t type, uint32_t lhs,
                       uint32_t rhs) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(opcode, type, result, lhs, rhs);
	return result;
}

inline uint32_t Select(EmitterState& state, uint32_t type, uint32_t condition, uint32_t true_value,
                       uint32_t false_value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, type, result, condition, true_value, false_value);
	return result;
}

struct ValueEmitContext {
	explicit ValueEmitContext(EmitterState& state_): state(state_) {}

	uint32_t              Def(IR::Value value);
	uint32_t              Arg(const IR::Inst& inst, size_t index);
	uint32_t              HalfArg(const IR::Inst& inst, size_t index, uint32_t half);
	uint32_t              Ballot(IR::Value predicate);
	uint32_t              FirstLane(uint32_t ballot);
	// wave_uniform_lane: every invocation reads the same guest lane (ReadLane, ReadFirstLane).
	uint32_t              Shuffle(const IR::Inst& inst, size_t index, uint32_t lane, bool wave_uniform_lane = false);
	uint32_t              Result(const IR::Inst& inst);
	uint32_t              Define(const IR::Inst& inst, uint32_t value);
	uint32_t              ResourceIndex(IR::Value value, IR::ValueOpcode opcode);
	const IR::Inst*       ImageAddress(IR::Value value);
	const IR::MemoryInfo& Memory(const IR::Inst& inst) const;
	const IR::ExportInfo& Export(const IR::Inst& inst) const;
	[[noreturn]] void     Fail(const char* reason) const;
	[[noreturn]] void     Fail(const IR::Inst& inst, const char* reason) const;

	EmitterState&                                                      state;
	const std::unordered_map<const IR::Inst*, uint32_t>*               dispatcher_spills = nullptr;
	std::unordered_map<const IR::Inst*, std::pair<uint32_t, uint32_t>> dispatcher_block_loads;
	uint32_t                                                           scratch_u32_variable = 0;
	ValueEmitContext*                                                  other_half = nullptr;
	uint32_t                                                           half       = 0;
};

// spirvEmitterLaneShare.cpp.
// Classify the program (state.lane_share.shared / uniform); a no-op unless the flag selects this shader.
void LaneShareClassify(EmitterState& state);
// High half of `inst`: if the low half already emitted it for both halves, bind the low id and return true.
// `low_words` is the number of function words the low half's emission took (statistics).
bool LaneShareReuse(ValueEmitContext& low, ValueEmitContext& high, const IR::Inst& inst,
                    size_t low_words);
void LaneShareReport(EmitterState& state);

enum class VertexInputScalarKind { Float, Sint, Uint };

constexpr uint32_t NoImageComponent = 0xffffffffu;

struct DppTargetLane {
	uint32_t lane  = 0;
	uint32_t valid = 0;
};

struct ImageSampleLayout {
	uint32_t offset = NoImageComponent;
	uint32_t dref   = NoImageComponent;
	uint32_t bias   = NoImageComponent;
	uint32_t coord  = 0;
	uint32_t lod    = NoImageComponent;
	uint32_t grad_x = NoImageComponent;
	uint32_t grad_y = NoImageComponent;
	uint32_t clamp  = NoImageComponent; // IMAGE_SAMPLE*_CL minimum LOD, the last component
};

struct F32Class {
	uint32_t nan  = 0;
	uint32_t zero = 0;
};

uint32_t PixelParameterLocation(const EmitterState& state, uint32_t attr);

bool PixelParameterIsFlat(const EmitterState& state, uint32_t attr);

bool PixelParameterIsCustom(const EmitterState& state, uint32_t attr);

VertexInputScalarKind VertexParameterScalarKind(const EmitterState& state, uint32_t location);

uint32_t VertexParameterComponentCount(const InputBinding& input);

uint32_t VertexParameterScalarType(EmitterState& state, VertexInputScalarKind kind);


uint32_t OutputVariableForExport(const EmitterState& state, const IR::ExportInfo& exp);

uint32_t ConstantU32(EmitterState& state, uint32_t value);

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state);

[[noreturn]] void ExitDescriptorBindingFailure(const EmitterState&       state,
                                               IR::DescriptorBindingKind kind, uint32_t resource,
                                               const char* reason);

uint32_t DescriptorElementPointer(EmitterState& state, uint32_t result_ptr_type,
                                  uint32_t variable_id, uint32_t array_index,
                                  IR::DescriptorBindingKind kind, uint32_t resource,
                                  const char* variable_name);

uint32_t ImageScalarType(EmitterState& state, Prospero::TextureNumericClass numeric_class);

uint32_t ImageVectorType(EmitterState& state, Prospero::TextureNumericClass numeric_class,
                         uint32_t components);

uint32_t ImageType(EmitterState& state, const IR::ImageResource& image);

uint32_t ImageViewSizeType(EmitterState& state, ImageDimension dimension);

uint32_t ImageDescriptorPointer(EmitterState& state, uint32_t resource, uint32_t mip = 0,
                                uint32_t array_index = 0);

uint32_t LoadImageDescriptor(EmitterState& state, uint32_t resource, uint32_t mip = 0,
                             uint32_t array_index = 0);

uint32_t LoadSamplerDescriptor(EmitterState& state, uint32_t sampler, uint32_t array_index = 0);
uint32_t LoadSamplerDescriptorAt(EmitterState& state, uint32_t sampler, uint32_t slot);
// The descriptor array slot of a resource of that binding kind: buffers and images carry their descriptor_index,
// samplers bind in resource order.
inline uint32_t ResourceForDescriptor(const EmitterState& state, IR::DescriptorBindingKind kind, uint32_t resource) {
	switch (kind) {
		case IR::DescriptorBindingKind::Buffers: return state.program.info.buffers.at(resource).descriptor_index;
		case IR::DescriptorBindingKind::Samplers: return resource;
		default: return state.program.info.images.at(resource).descriptor_index;
	}
}

uint32_t MakeSampledImage(EmitterState& state, uint32_t resource, uint32_t sampler_id,
                          uint32_t mip = 0, uint32_t array_index = 0,
                          bool sampler_dynamic = false);

// sample is 0 (no id) for single-sampled images.
void EmitStorageImageWrite(EmitterState& state, uint32_t resource, uint32_t mip_lod, uint32_t coord,
                           uint32_t texel, uint32_t array_index, uint32_t sample);

spv::ExecutionModel ExecutionModelForStage(ShaderType stage);

uint32_t ConstantU32(EmitterState& state, uint32_t value);

uint32_t ConstantI32(EmitterState& state, int32_t value);

uint32_t ConstantF32(EmitterState& state, uint32_t bits);

uint32_t ConstantF32Value(EmitterState& state, float value);

uint32_t ConstantBool(EmitterState& state, bool value);

// Iteration cap for emitted loops (KYTY_SHADER_LOOP_LIMIT, default 1 << 16; 0 disables).
// A loop that runs past it takes an exit it already has, so a shader fed bad data cannot
// hang the GPU. 65536 is the cap measured against a pinned loop (~50 s/frame); a million
// iterations would turn that same shader into many minutes per frame.
inline uint32_t ShaderLoopLimit() {
	constexpr uint32_t kDefaultLimit = 1u << 16u;
	const char*        value         = Common::ShaderEnv::Read("KYTY_SHADER_LOOP_LIMIT");
	if (value == nullptr || *value == '\0') {
		return kDefaultLimit;
	}
	// Only a plain decimal number overrides the default; anything else keeps the cap on
	// instead of silently turning the watchdog off. strtoull would accept leading
	// whitespace and a sign, so the first character has to be a digit.
	if (*value < '0' || *value > '9') {
		return kDefaultLimit;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	if (*end != '\0' || parsed > UINT32_MAX) {
		return kDefaultLimit;
	}
	return static_cast<uint32_t>(parsed);
}

// KYTY_DIAG_RT_TRIPS=1 or =<hex hash>[,...] (diagnostic, default off; FaultManager reads the same variable): the
// listed shaders (every shader with =1) that have the fault buffer count and report, in its diagnostic tail, each
// invocation that reaches a loop watchdog cap (with the loop header's phi values) and each BDA store to a page
// without a host buffer (which is dropped). Off: no code is emitted.
// KYTY_FIX_CLAMP_NAN=2: a program with a saturate gets the fault buffer for the counts, even without DMA.
inline bool ClampNanCheckBinding(const IR::Program& program) {
	if (IR::ClampNanModeFor(program.shader_hash) != 2) return false;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto op = inst.GetOpcode();
			if (op == IR::ValueOpcode::FPSaturate32 || op == IR::ValueOpcode::HalfPkSaturate) return true;
		}
	}
	return false;
}

// DX10_CLAMP of the program: compute from COMPUTE_PGM_RSRC1 (ShaderComputeInputInfo::dx10_clamp); the PS and GS/NGG
// checks (shader.cpp ps_check / the GS check) stop the emulator unless SPI_SHADER_PGM_RSRC1_PS/GS.DX10_CLAMP is 1;
// the other graphics stages' registers are not passed to the recompiler and are taken as 1 (the PS5 compiler's value).
inline bool ProgramDx10Clamp(const EmitterState& state) {
	if (state.program.stage == ShaderType::Compute) {
		return state.input_info.compute == nullptr || state.input_info.compute->dx10_clamp;
	}
	return true;
}

// KYTY_FIX_CLAMP_NAN=2: `is_nan` (bool id) true -> count it in ClampNanDiagWord + `counter` and record the shader.
void ClampNanCheck(EmitterState& state, uint32_t is_nan, uint32_t counter);

inline bool DiagRtTripsFor(uint64_t shader_hash) {
	static const auto config = [] {
		std::pair<bool, std::vector<uint64_t>> result {false, {}};
		const char* value = Common::EnvRead::RtTrips();
		if (value == nullptr) return result;
		if (value[0] == '1' && value[1] == '\0') {
			result.first = true;
			return result;
		}
		while (*value != '\0') {
			char*      end  = nullptr;
			const auto hash = std::strtoull(value, &end, 16);
			if (end == value) break;
			result.second.push_back(hash);
			if (*end != ',') break;
			value = end + 1;
		}
		return result;
	}();
	return config.first || std::ranges::find(config.second, shader_hash) != config.second.end();
}

// KYTY_DIAG_RT_TRIPS (diagnostic, stage 2 pack 2 / D7): a program it names (=1: every program) that has a structured
// loop gets the fault buffer, so its loop-cap trips are counted and reported even without DMA. Before, only programs
// with DMA had it: the terrain PS 7de43c82ec18f609 (cap 256) had none, and its RT_TRIPS run of 16.10 ("0 trips") could
// not have reported any. The dispatcher form (dispatcher_fallback) counts no trips. RT_TRIPS off: no change.
inline bool RtTripsCheckBinding(const IR::Program& program) {
	if (program.dispatcher_fallback || !DiagRtTripsFor(program.shader_hash)) return false;
	return std::ranges::any_of(program.blocks, [](const IR::Block* block) { return block->terminator.loop_header; });
}

// KYTY_SHADER_LOOP_LIMIT_HASHES=<hex hash>:<limit>[,...] (logged APPROXIMATION): a lower loop
// cap for the listed shaders, e.g. a per-pixel loop that would otherwise run past the GPU
// timeout at the default cap.
inline uint32_t ShaderLoopLimitFor(uint64_t shader_hash) {
	static const auto overrides = [] {
		std::vector<std::pair<uint64_t, uint32_t>> result;
		// Default from Common::EnvDefaults; workaround: D7 (skipped shaders,
		// loop caps), to be replaced by the proper implementation; =0 turns it off.
		const char*                                value = Common::ShaderEnv::Read("KYTY_SHADER_LOOP_LIMIT_HASHES");
		while (value != nullptr && *value != '\0') {
			char*      end   = nullptr;
			const auto hash  = std::strtoull(value, &end, 16);
			if (end == value || *end != ':') break;
			const auto limit = std::strtoul(end + 1, &end, 10);
			result.emplace_back(hash, static_cast<uint32_t>(limit));
			value = *end == ',' ? end + 1 : nullptr;
		}
		return result;
	}();
	for (const auto& [hash, limit]: overrides) {
		if (hash != shader_hash) continue;
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u)
			std::fprintf(stderr, "DIAG_SHADER_LOOP_LIMIT APPROXIMATION hash=0x%016llx limit=%u\n",
			             static_cast<unsigned long long>(shader_hash), limit);
		return limit;
	}
	return ShaderLoopLimit();
}

uint32_t ConstantU64(EmitterState& state, uint64_t value);

uint32_t ConstantU32CompositeZero(EmitterState& state, uint32_t components);

uint32_t DefineInterfaceVariable(EmitterState& state, uint32_t type, spv::StorageClass storage,
                                 const char* name);
void     DefineModule(EmitterState& state);
void     DefineTessellationInterfaces(EmitterState& state);
void     DefineTessellationExecutionModes(EmitterState& state);
void     DefineMeshOutputs(EmitterState& state, uint32_t clip_distance_count,
                           uint32_t cull_distance_count);
void     EmitMeshEntryPoint(EmitterState& state);
void     EmitMeshAllocate(ValueEmitContext& ctx, const IR::Inst& inst);
uint32_t MeshOutputPointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index = 0);
uint32_t MeshPrimitivePointer(EmitterState& state);

DppTargetLane EmitDppPermTargetLane(EmitterState& state, uint32_t subid, uint32_t control,
                                    uint32_t lane_bits);

DppTargetLane EmitDppRowShiftTargetLane(EmitterState& state, uint32_t subid, uint32_t amount,
                                        bool left);

DppTargetLane EmitDppRowRotateRightTargetLane(EmitterState& state, uint32_t subid, uint32_t amount);

DppTargetLane EmitDppMirrorTargetLane(EmitterState& state, uint32_t subid, bool half_row);

DppTargetLane EmitDppTargetLane(EmitterState& state, const IR::DppMoveFlags& flags);

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state);

uint32_t InputVariableForKind(const EmitterState& state, IR::StageInputKind kind);

// GPU indirect draws: the vertex variant whose user SGPRs come from SPIR-V draw parameters
// (nullptr for every other program).
const ShaderIndirectDrawInfo* IndirectDrawVariant(const EmitterState& state);

const InputBinding* InputBindingForParameter(const EmitterState& state, uint32_t location);

uint32_t EmitVertexParameterComponentU32(EmitterState& state, const InputBinding& input,
                                         uint32_t component);

uint32_t EmitInputComponentU32(EmitterState& state, IR::StageInputKind kind, uint32_t component);

uint32_t EmitLocalInvocationIndex(EmitterState& state);

uint32_t EmitBallotLaneActiveBool(EmitterState& state, uint32_t ballot, uint32_t lane);

uint32_t EmitSubgroupLaneActiveBool(EmitterState& state, uint32_t lane);

// Paired shader helpers: host subgroup index, guest wave half (0/1), and the exchange of
// a subgroup-uniform U32 with the other half of the same guest wave (returns the partner's value).
uint32_t EmitSubgroupIdU32(EmitterState& state);
uint32_t EmitPairHalf(EmitterState& state);
uint32_t EmitPairExchange(EmitterState& state, uint32_t value);
void     EmitPairInitialize(EmitterState& state);
bool     WavePairRequested(uint64_t shader_hash);
// The pair rule: a compute workgroup of `threads` (64: one guest wave) and `code_words` of guest code it pairs.
bool     WavePairRule(uint32_t threads, size_t code_words);
void     NoteRulePaired(uint64_t shader_hash);

inline constexpr auto EmitAddU32 = EmitNative<spv::OpIAdd, IR::Type::U32, uint32_t, uint32_t>;

uint32_t EmitBinaryU32(EmitterState& state, spv::Op opcode, uint32_t lhs, uint32_t rhs);

uint32_t EmitShaderDataDwordLoad(EmitterState& state, uint32_t dword_index);

// Uniform const buffers (ShaderIR.h): a uniform block of `dwords` dwords stored as uvec4 elements (std140, stride 16),
// and dword `index` of such a block.
uint32_t UniformDwordBlockType(EmitterState& state, uint32_t dwords);
uint32_t LoadUniformDword(EmitterState& state, uint32_t variable, uint32_t index);

// Push-constant dword `index` (a u32 id), e.g. the user clip planes (upstream feed075a9).
uint32_t EmitPushDataLoad(EmitterState& state, uint32_t index);

uint32_t StorageBufferPackedStride(const EmitterState& state, const IR::MemoryInfo& mem);

Prospero::BufferFormat StorageBufferFormat(const EmitterState& state, const IR::MemoryInfo& mem);

void EmitMemoryOffsets(EmitterState& state);

uint32_t LdsDwordCount(const EmitterState& state);
void     PlanCompactFunctionLds(EmitterState& state);
bool     CompactLdsBase(const EmitterState& state, const IR::Inst& inst, uint32_t& base);
void EnsureLdsStorage(EmitterState& state);

struct MemoryResourceAccess {
	IR::ResourceKind      kind             = IR::ResourceKind::None;
	uint32_t              object_pointer   = 0;
	uint32_t              length           = 0;
	uint32_t              byte_offset      = 0;
	uint32_t              element_bits     = 32;
	spv::MemoryAccessMask memory_access    = spv::MemoryAccessMaskNone;
};

// The element `index` of a prepared buffer (or LDS/GDS/scratch) access through `pointer` (EmitMemoryElementPointer of
// the same index) into the id `value`, as a value of `type`. The caller allocates `value` where the load it replaces
// did, and mask_operand says whether that load has a memory-access operand.
uint32_t LoadResourceElement(EmitterState& state, uint32_t type, uint32_t value, uint32_t pointer,
                             const MemoryResourceAccess& access, uint32_t index, bool mask_operand = true);

uint32_t EmitIndirectResourceIndex(EmitterState& state, uint32_t key, uint32_t mapping_offset,
                                   uint32_t search_iterations, uint32_t default_resource);
// Slot word of an image heap byte offset; offsets past the heap select the null element.
uint32_t EmitImageHeapEntry(EmitterState& state, uint32_t key, uint32_t mapping_offset);
MemoryResourceAccess PrepareMemoryResourceAccess(EmitterState& state, const IR::MemoryInfo& mem);

MemoryResourceAccess PrepareStorageBufferResourceAccess(EmitterState&         state,
                                                        const IR::MemoryInfo& mem,
                                                        BufferDefinition&     buffer);

uint32_t EmitMemoryElementInBounds(EmitterState& state, const MemoryResourceAccess& access,
                                   uint32_t index);

uint32_t EmitMemoryElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                  uint32_t index);

uint32_t EmitStorageBufferElementPointer(EmitterState& state, const MemoryResourceAccess& access,
                                         uint32_t index, uint32_t pointer_type);

uint32_t EmitTBufferBitcastU32ToI32(EmitterState& state, uint32_t value);

inline constexpr auto EmitTBufferSelectF32 =
    EmitNative<spv::OpSelect, IR::Type::F32, uint32_t, uint32_t, uint32_t>;

bool IsSignedFormatComponent(Format::ComponentType type);

uint32_t EmitUFloatToF32Bits(EmitterState& state, uint32_t raw, uint32_t bits);

uint32_t NormalizeFormatComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                  uint32_t component, uint32_t raw);

enum class UnormRounding { RoundEven, HalfUp, Truncate };

uint32_t EncodeFormattedStoreComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                       uint32_t component, uint32_t data, uint32_t data_bits,
                                       UnormRounding rounding = UnormRounding::RoundEven);

uint32_t EncodePackedColorComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                    uint32_t component, uint32_t data, uint32_t data_bits,
                                    UnormRounding rounding = UnormRounding::RoundEven);

spv::Op SpirvAtomicOpcode(IR::ValueOpcode opcode);

uint32_t EmitAtomicOperation(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t pointer,
                             uint32_t scope);

void EmitAtomicMemoryBarrier(EmitterState& state, IR::ResourceKind kind);

uint32_t EmitFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                    bool max_value);

uint32_t EmitDsFloatAtomicReplacement(EmitterState& state, uint32_t old, uint32_t source,
                                      bool max_value);

uint32_t EmitDsSwizzleTargetLane(EmitterState& state, uint32_t subid, uint32_t control);

inline constexpr auto EmitSelectValueU32 =
    EmitNative<spv::OpSelect, IR::Type::U32, uint32_t, uint32_t, uint32_t>;

uint32_t EmitAndConstant(EmitterState& state, uint32_t value, uint32_t mask);

inline constexpr auto EmitOrU32 = EmitNative<spv::OpBitwiseOr, IR::Type::U32, uint32_t, uint32_t>;

uint32_t EmitCompareU32Constant(EmitterState& state, spv::Op opcode, uint32_t value,
                                uint32_t constant);

uint32_t EmitF32ToF16RtzBits(EmitterState& state, uint32_t f32);
// Both lanes of V_CVT_PKRTZ in hardware (spirvEmitterAluHelpers.cpp); 0 when it does not apply.
uint32_t EmitPackF16RtzNative(EmitterState& state, uint32_t a, uint32_t b);

// A float as an IEEE half in the low 16 bits (high bits zero), as GLSL PackHalf2x16 of (f32, 0); exact under
// (spirvEmitterAluHelpers.cpp)
uint32_t EmitF32ToF16Bits(EmitterState& state, uint32_t f32);

// The capabilities and 16-bit float execution modes the hardware half conversions rely on (once per module that
// emitted one: state.native_half): RoundingModeRTE and SignedZeroInfNanPreserve for 16 bits, never DenormPreserve 16
// (it stops the host's f32 denormal flush the guest's FLOAT_MODE 0xc0 needs).
void DeclareNativeHalf(EmitterState& state);

inline constexpr auto EmitBitcastF32ToU32 = EmitNative<spv::OpBitcast, IR::Type::U32, uint32_t>;

inline constexpr auto EmitBitcastU32ToF32 = EmitNative<spv::OpBitcast, IR::Type::F32, uint32_t>;

inline constexpr auto EmitAndU32 = EmitNative<spv::OpBitwiseAnd, IR::Type::U32, uint32_t, uint32_t>;

inline constexpr auto EmitLogicalAndBool =
    EmitNative<spv::OpLogicalAnd, IR::Type::U1, uint32_t, uint32_t>;

inline constexpr auto EmitLogicalOrBool =
    EmitNative<spv::OpLogicalOr, IR::Type::U1, uint32_t, uint32_t>;

inline constexpr auto EmitLogicalNotBool = EmitNative<spv::OpLogicalNot, IR::Type::U1, uint32_t>;

F32Class EmitClassifyF32Bits(EmitterState& state, uint32_t bits);

uint32_t EmitClassMaskBitMatch(EmitterState& state, uint32_t mask, uint32_t bit,
                               uint32_t class_match);

uint32_t EmitClassMaskF32(EmitterState& state, uint32_t value, uint32_t mask);
uint32_t EmitClassMaskF16(EmitterState& state, uint32_t bits, uint32_t mask);

uint32_t EmitMinMaxF32Value(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value);

uint32_t EmitFlushF32DenormToSignedZero(EmitterState& state, uint32_t value);
// The input flush of 1/x, InverseSqrt, Sqrt, Exp2, Log2 and f32 compares, left out where the
// host already flushes those inputs (EmitFlushF32DenormToSignedZero otherwise).
uint32_t EmitFlushF32InputUnlessHost(EmitterState& state, uint32_t value);

uint32_t EmitTrigCycleF32(EmitterState& state, uint32_t src, bool preserve_signed_zero);

inline constexpr auto EmitFNegateValue = EmitNative<spv::OpFNegate, IR::Type::F32, uint32_t>;

inline constexpr auto EmitFAbsValue = EmitGlsl<GLSLstd450FAbs, IR::Type::F32, uint32_t>;

uint32_t EmitF16BitsToF32(EmitterState& state, uint32_t bits);

void EmitProgram(EmitterState& state);

void DefineGetBdaPointer(EmitterState& state);
// KYTY_PERF_BDA_ENTRY_SHARE: the Function variables of state.bda_memo_page / bda_memo_base, in the entry block of the
// function the guest program is emitted into (after DefineGetBdaPointer; a no-op when it defined nothing or the switch
// is off).
void DeclareBdaEntryShare(EmitterState& state);
// Fills state.scalar_groups (after DefineGetBdaPointer; none when it defined nothing).
void FindScalarLoadGroups(EmitterState& state);
void DefineIndirectFormattedLoadFunctions(ValueEmitContext& ctx);
void DefineIndirectRawDword(ValueEmitContext& ctx);
void DefineIndirectStoreFunctions(ValueEmitContext& ctx);
void DefineBvhIntersect(EmitterState& state);
// Upstream bfd4c34b8 / 5eab0ec73 (spirvEmitterTrinityBvh.cpp): BVH8 intersections and the BVH stack.
void DefineTrinityBvhIntersect(EmitterState& state);
void DefineTrinityFunctions(EmitterState& state); // spirvEmitterTrinity.cpp: the BVH stack and the BVH8 intersect
uint32_t EmitBvhTriangle(EmitterState& state, const std::array<std::array<uint32_t, 3>, 3>& positions,
                         std::array<uint32_t, 3> origin, std::array<uint32_t, 3> direction,
                         uint32_t remap, uint32_t triangle_id, uint32_t bary);
uint32_t GetBdaPointer(EmitterState& state, uint32_t address);
// KYTY_DIAG_RT_TRIPS (DiagRtTripsFor; needs state.fault_buffer_variable): the fault buffer's diagnostic tail
// (BufferCache::FAULT_DIAG_*). FaultDiagAdd: atomic add of `value` to tail word `word`, returns the old value.
// FaultDiagStore: `value` (u32 id) to tail word FAULT_DIAG_BASE + `first` + `slot` * `stride` + `field`.
uint32_t FaultDiagAdd(EmitterState& state, uint32_t word, uint32_t value);
void     FaultDiagStore(EmitterState& state, uint32_t first, uint32_t slot, uint32_t stride, uint32_t field, uint32_t value);
// A store through BDA whose page has no host buffer (`bda` == 0) is dropped: counted and reported.
void     ReportBdaStoreMiss(EmitterState& state, uint32_t address, uint32_t bda);
// When `near` (the loop is in its last FAULT_DIAG_TRIP_LAST iterations before the watchdog cap): reports the loop
// (block id), its counter, the invocation (`lane`, `group`) and up to FAULT_DIAG_TRIP_PHIS u32 values per lane half
// (the loop header's phis); when `over` (the cap is reached) it also counts the trip.
// `cap` is the loop's watchdog limit, `counter` its current count (u32 ids: lane, group).
void     ReportWatchdogTrip(EmitterState& state, uint32_t cap, uint32_t block_id, uint32_t counter, uint32_t lane,
                            uint32_t group, const std::vector<uint32_t>& low, const std::vector<uint32_t>& high);
// The number of loop header values a watchdog report carries per lane half.
uint32_t WatchdogTripValues();

// These templates accept local lambdas from several emitter translation units.
template <typename Fn>
auto EmitIndexSwitch(EmitterState& state, uint32_t index, uint32_t count,
                        uint32_t result_type, Fn&& emit) {
	constexpr bool has_result = !std::is_void_v<std::invoke_result_t<Fn, uint32_t>>;
	EXIT_IF(count == 0u);
	const auto merge_label = state.builder.AllocateId();
	std::vector<uint32_t> labels(count);
	std::vector<uint32_t> words {spv::OpSwitch, index, merge_label};
	for (uint32_t item = 0; item < count; item++) {
		labels[item] = state.builder.AllocateId();
		words.push_back(item);
		words.push_back(labels[item]);
	}
	if constexpr (has_result) {
		words[2] = labels.front();
	}
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(words);
	std::vector<uint32_t> phi;
	if constexpr (has_result) {
		phi = {spv::OpPhi, result_type, state.builder.AllocateId()};
	}
	for (uint32_t item = 0; item < count; item++) {
		EmitLabel(state, labels[item]);
		if constexpr (has_result) {
			phi.push_back(emit(item));
			phi.push_back(state.current_label);
		} else {
			emit(item);
		}
		state.builder.AddFunction(spv::OpBranch, merge_label);
	}
	EmitLabel(state, merge_label);
	if constexpr (has_result) {
		state.builder.AddFunction(phi);
		return phi[2];
	}
}

template <typename Fn>
void EmitIfCondition(EmitterState& state, uint32_t condition, Fn&& fn) {
	if (condition == ConstantBool(state, true)) {
		fn();
		return;
	}
	const auto then_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, then_label, merge_label);
	EmitLabel(state, then_label);
	state.guard_depth++;
	fn();
	state.guard_depth--;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
}

template <typename Fn>
uint32_t EmitValueOrDefaultIfCondition(EmitterState& state, uint32_t condition, uint32_t type,
                                       uint32_t default_value, Fn&& fn) {
	if (condition == ConstantBool(state, true)) return fn();
	const auto then_label  = state.builder.AllocateId();
	const auto header     = state.current_label;
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, condition, then_label, merge_label);
	EmitLabel(state, then_label);
	state.guard_depth++;
	const auto then_value = fn();
	state.guard_depth--;
	const auto then_exit  = state.current_label;
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, type, value, then_value, then_exit, default_value,
	                          header);
	return value;
}

template <typename Fn>
uint32_t EmitValueOrZeroIfCondition(EmitterState& state, uint32_t condition, Fn&& fn) {
	return EmitValueOrDefaultIfCondition(state, condition, TypeU32(state), ConstantU32(state, 0),
	                                     std::forward<Fn>(fn));
}

template <typename Fn>
uint32_t AtomicUpdate(EmitterState& state, uint32_t pointer, IR::ResourceKind kind, Fn&& desired) {
	const auto scope     = kind == IR::ResourceKind::Lds ? spv::ScopeWorkgroup : spv::ScopeDevice;
	const auto preheader = state.builder.AllocateId();
	const auto header    = state.builder.AllocateId();
	const auto cont      = state.builder.AllocateId();
	const auto merge     = state.builder.AllocateId();
	const auto initial   = state.builder.AllocateId();
	const auto observed  = state.builder.AllocateId();
	const auto exchanged = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBranch, preheader);
	EmitLabel(state, preheader);
	state.builder.AddFunction(spv::OpAtomicLoad, TypeU32(state), initial, pointer,
	                          ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone));
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, header);
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), observed, initial, preheader, exchanged,
	                          cont);
	// Give up after ShaderLoopLimit() failed exchanges so a CAS that never lands cannot spin.
	const auto iteration      = state.builder.AllocateId();
	const auto next_iteration = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), iteration, ConstantU32(state, 0),
	                          preheader, next_iteration, cont);
	const auto body = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranch, body);
	EmitLabel(state, body);
	const auto next = desired(observed);
	// An update that leaves the value unchanged (e.g. a min/max that does not win) needs no
	// exchange: skipping it keeps contended min/max loops from retrying no-op writes.
	const auto unchanged = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), unchanged, next, observed);
	const auto exchange_label = state.builder.AllocateId();
	const auto exchange_merge = state.builder.AllocateId();
	const auto exchange_from  = state.current_label;
	state.builder.AddFunction(spv::OpSelectionMerge, exchange_merge, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, unchanged, exchange_merge, exchange_label);
	EmitLabel(state, exchange_label);
	const auto swapped = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), swapped, pointer,
	                          ConstantU32(state, scope),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone), next, observed);
	state.builder.AddFunction(spv::OpBranch, exchange_merge);
	EmitLabel(state, exchange_merge);
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), exchanged, observed, exchange_from, swapped,
	                          exchange_label);
	auto success = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), success, exchanged, observed);
	state.builder.AddFunction(spv::OpIAdd, TypeU32(state), next_iteration, iteration,
	                          ConstantU32(state, 1));
	if (ShaderLoopLimit() != 0) {
		const auto over = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), over, iteration,
		                          ConstantU32(state, ShaderLoopLimit()));
		const auto stop = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), stop, success, over);
		success = stop;
	}
	state.builder.AddFunction(spv::OpBranchConditional, success, merge, cont);
	EmitLabel(state, cont);
	state.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(state, merge);
	EmitAtomicMemoryBarrier(state, kind);
	return observed;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_INTERNAL_H_ */
