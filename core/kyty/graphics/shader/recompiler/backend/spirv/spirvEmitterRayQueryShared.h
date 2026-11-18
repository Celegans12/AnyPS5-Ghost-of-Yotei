#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_SPIRVEMITTERRAYQUERYSHARED_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_SPIRVEMITTERRAYQUERYSHARED_H_

#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter::RayQueryEmit {

constexpr uint32_t DoneBit       = 1u << 0u;
constexpr uint32_t TraversalBit  = 1u << 3u;
constexpr uint32_t MatchedBit    = 1u << 4u;
constexpr uint32_t CommittedBit  = 1u << 6u;
constexpr uint32_t None          = 0xffffffffu;
constexpr uint32_t ListSize      = IR::RayQueryListSize;

constexpr uint32_t ContextHeader        = 0;
constexpr uint32_t ContextTlas          = 8;
constexpr uint32_t ContextInstanceNodes = 16;
constexpr uint32_t ContextSlots         = 24;
constexpr uint32_t ContextValid         = 36;
constexpr uint32_t ContextInstanceBase  = 40;
constexpr uint32_t SlotStride           = 64;
constexpr uint32_t SlotEntryTable       = 8;
constexpr uint32_t SlotCapacity0        = 16;
constexpr uint32_t SlotCapacity1        = 20;

constexpr uint32_t CandidateIntersection = 0;
constexpr uint32_t CommittedIntersection = 1;

constexpr uint32_t RayFlagOpaque              = 0x1;
constexpr uint32_t RayFlagNoOpaque            = 0x2;
constexpr uint32_t RayFlagCullBackFacing      = 0x10;
constexpr uint32_t RayFlagCullFrontFacing     = 0x20;
constexpr uint32_t RayFlagCullNoOpaque        = 0x80;

struct LaneVariables {
	uint32_t query   = 0;
	uint32_t state   = 0;
	uint32_t nodes   = 0;
	uint32_t slots   = 0;
	uint32_t list    = 0;
	uint32_t count   = 0;
	uint32_t current = 0;
	uint32_t base    = 0;
	uint32_t result_t        = 0;
	uint32_t result_word     = 0;
	uint32_t result_instance = 0;
	uint32_t result_u        = 0;
	uint32_t result_v        = 0;
	uint32_t result_node     = 0;
	uint32_t alpha           = 0;
	uint32_t alpha_cache     = 0;
	uint32_t inline_record   = 0;
};

struct InlineLayout {
	bool     valid  = false;
	uint32_t type   = 0;
	uint32_t offset = 0;
};

inline InlineLayout ProgramInlineLayout(const EmitterState& s) {
	const auto word = s.program.info.ray_query_layout[3];
	return {((word >> 19u) & 1u) != 0, (word >> 16u) & 7u, (word & 0xffu) * 8u + ((word >> 28u) & 1u) * 4u};
}

enum class AlphaArgument : uint32_t {
	Lod,
	Spread,
	Factor,
	Material,
	Buffers   = 7,
	Threshold = 11,
	LodOffset = 12,
	Count     = 13,
};

inline uint32_t SiteOf(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto site = IR::RayQuerySite(inst.Flags<uint32_t>());
	if (site >= IR::MaxRayQuerySites || ctx.state.ray_query_variables[site][0] == 0) {
		ctx.Fail(inst, "ray query site without variables");
	}
	return site;
}

inline LaneVariables Lane(EmitterState& s, uint32_t site, uint32_t half) {
	return {s.ray_query_variables[site][half], s.ray_query_states[site][half],
	        s.ray_query_nodes[site][half],     s.ray_query_slots[site][half],
	        s.ray_query_lists[site][half],     s.ray_query_counts[site][half],
	        s.ray_query_currents[site][half],  s.ray_query_bases[site][half],
	        s.ray_query_result_t[site][half],  s.ray_query_result_word[site][half],
	        s.ray_query_result_instance[site][half], s.ray_query_result_u[site][half],
	        s.ray_query_result_v[site][half],  s.ray_query_result_node[site][half],
	        s.ray_query_alpha[site][half],     s.ray_query_alpha_cache[site][half],
	        s.ray_query_inline_records[site][half]};
}

inline uint32_t Load(EmitterState& s, uint32_t type, uint32_t pointer) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, type, result, pointer);
	return result;
}

inline void Store(EmitterState& s, uint32_t pointer, uint32_t value) {
	s.builder.AddFunction(spv::OpStore, pointer, value);
}

inline uint32_t U32(EmitterState& s, uint32_t value) {
	return ConstantU32(s, value);
}

inline uint32_t InstanceRecordAddress(EmitterState& s, uint32_t base, uint32_t node) {
	const auto wide   = TypeU64(s);
	const auto index  = Unary(s, spv::OpUConvert, wide, Binary(s, spv::OpShiftRightLogical, TypeU32(s), node, U32(s, 3)));
	const auto stride = 128u + ((s.program.info.ray_query_layout[1] >> 30u) & 3u) * 32u;
	const auto offset = stride == 128u ? Binary(s, spv::OpShiftLeftLogical, wide, index, ConstantU64(s, 7))
	                                   : Binary(s, spv::OpIMul, wide, index, ConstantU64(s, stride));
	return Binary(s, spv::OpIAdd, wide, Load(s, wide, base), offset);
}

inline uint32_t And(EmitterState& s, uint32_t value, uint32_t mask) {
	return Binary(s, spv::OpBitwiseAnd, TypeU32(s), value, U32(s, mask));
}

inline uint32_t Or(EmitterState& s, uint32_t value, uint32_t bits) {
	return Binary(s, spv::OpBitwiseOr, TypeU32(s), value, bits);
}

inline uint32_t BitSet(EmitterState& s, uint32_t value, uint32_t bit) {
	return Binary(s, spv::OpINotEqual, TypeBool(s), And(s, value, bit), U32(s, 0));
}

inline uint32_t Equal(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpIEqual, TypeBool(s), a, b);
}

inline uint32_t Less(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpULessThan, TypeBool(s), a, b);
}

inline uint32_t LogicalAnd(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpLogicalAnd, TypeBool(s), a, b);
}

inline uint32_t LogicalOr(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpLogicalOr, TypeBool(s), a, b);
}

inline uint32_t LogicalNot(EmitterState& s, uint32_t a) {
	return Unary(s, spv::OpLogicalNot, TypeBool(s), a);
}

inline uint32_t Add(EmitterState& s, uint32_t a, uint32_t b) {
	return Binary(s, spv::OpIAdd, TypeU32(s), a, b);
}

inline uint32_t KeyLess(EmitterState& s, uint32_t a_instance, uint32_t a_leaf, uint32_t b_instance,
                 uint32_t b_leaf) {
	return LogicalOr(s, Less(s, a_instance, b_instance),
	                 LogicalAnd(s, Equal(s, a_instance, b_instance), Less(s, a_leaf, b_leaf)));
}

inline uint32_t ListWord(EmitterState& s, const LaneVariables& lane, uint32_t index) {
	const auto pointer = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypePointer(s, spv::StorageClassFunction, TypeU32(s)),
	                      pointer, lane.list, index);
	return pointer;
}

inline uint32_t LoadAddressU32(EmitterState& s, uint32_t address) {
	const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
	const auto result  = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, TypeU32(s), result, pointer, spv::MemoryAccessAlignedMask, 4u);
	return result;
}

inline void StoreAddressU32(EmitterState& s, uint32_t address, uint32_t value) {
	const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
	s.builder.AddFunction(spv::OpStore, pointer, value, spv::MemoryAccessAlignedMask, 4u);
}

inline uint32_t LoadAddressU64(EmitterState& s, uint32_t address) {
	const auto low  = LoadAddressU32(s, address);
	const auto high = LoadAddressU32(s, Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, 4)));
	return PackU64(s, low, high);
}

inline uint32_t AddressPlus(EmitterState& s, uint32_t address, uint32_t offset) {
	return Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, offset));
}

inline uint32_t ScaledAddress(EmitterState& s, uint32_t base, uint32_t index, uint32_t scale) {
	const auto wide = TypeU64(s);
	return Binary(s, spv::OpIAdd, wide, base,
	              Binary(s, spv::OpIMul, wide, Unary(s, spv::OpUConvert, wide, index),
	                     ConstantU64(s, scale)));
}

inline uint32_t Finite(EmitterState& s, uint32_t value) {
	const auto b = TypeBool(s);
	return LogicalAnd(s, LogicalNot(s, Unary(s, spv::OpIsNan, b, value)),
	                  LogicalNot(s, Unary(s, spv::OpIsInf, b, value)));
}

template <typename Body>
void EmitLoop(EmitterState& s, Body&& body) {
	const auto header = s.builder.AllocateId(), start = s.builder.AllocateId();
	const auto cont = s.builder.AllocateId(), merge = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, header);
	s.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
	s.builder.AddFunction(spv::OpBranch, start);
	EmitLabel(s, start);
	body(merge, cont);
	EmitLabel(s, cont);
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, merge);
}

inline uint32_t LoadWord(EmitterState& s, const LaneVariables& lane, uint32_t index) {
	return Load(s, TypeU32(s), ListWord(s, lane, index));
}

inline void InsertKey(EmitterState& s, const LaneVariables& lane, uint32_t count, uint32_t instance,
               uint32_t leaf) {
	const auto u = TypeU32(s), b = TypeBool(s);
	const auto position = ListWord(s, lane, U32(s, 2 * ListSize));
	Store(s, position, count);
	EmitLoop(s, [&](uint32_t merge, uint32_t cont) {
		const auto i     = Load(s, u, position);
		const auto above = Binary(s, spv::OpUGreaterThan, b, i, U32(s, 0));
		const auto move  = EmitValueOrDefaultIfCondition(s, above, b, ConstantBool(s, false), [&] {
			const auto at = Binary(s, spv::OpShiftLeftLogical, u, Binary(s, spv::OpISub, u, i, U32(s, 1)),
			                       U32(s, 1));
			const auto previous_instance = LoadWord(s, lane, at);
			const auto previous_leaf     = LoadWord(s, lane, Add(s, at, U32(s, 1)));
			const auto larger = KeyLess(s, instance, leaf, previous_instance, previous_leaf);
			EmitIfCondition(s, larger, [&] {
				const auto to = Binary(s, spv::OpShiftLeftLogical, u, i, U32(s, 1));
				Store(s, ListWord(s, lane, to), previous_instance);
				Store(s, ListWord(s, lane, Add(s, to, U32(s, 1))), previous_leaf);
				Store(s, position, Binary(s, spv::OpISub, u, i, U32(s, 1)));
			});
			return larger;
		});
		s.builder.AddFunction(spv::OpBranchConditional, move, cont, merge);
	});
	const auto at = Binary(s, spv::OpShiftLeftLogical, u, Load(s, u, position), U32(s, 1));
	Store(s, ListWord(s, lane, at), instance);
	Store(s, ListWord(s, lane, Add(s, at, U32(s, 1))), leaf);
}

inline void PopKey(EmitterState& s, const LaneVariables& lane) {
	Store(s, lane.current, Add(s, Load(s, TypeU32(s), lane.current), U32(s, 1)));
}

inline void ClearKeys(EmitterState& s, const LaneVariables& lane) {
	Store(s, lane.count, U32(s, 0));
	Store(s, lane.current, U32(s, 0));
}

struct GuestEntry {
	uint32_t entry  = 0;
	uint32_t word   = 0;
	uint32_t node   = 0;
	uint32_t remap  = 0;
	uint32_t valid  = 0;
	uint32_t record = 0;
};

constexpr uint32_t AlphaRecordStride = 32;

template <class Slots>
GuestEntry TriangleEntry(EmitterState& s, uint32_t ray_query, uint32_t intersection, Slots&& slots) {
	const auto u = TypeU32(s), wide = TypeU64(s);
	const auto query = [&](spv::Op op) {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(op, u, result, ray_query, U32(s, intersection));
		return result;
	};
	const auto slot      = query(spv::OpRayQueryGetIntersectionInstanceCustomIndexKHR);
	const auto geometry  = query(spv::OpRayQueryGetIntersectionGeometryIndexKHR);
	const auto primitive = query(spv::OpRayQueryGetIntersectionPrimitiveIndexKHR);
	const auto info      = ScaledAddress(s, slots(), slot, SlotStride);
	const auto table     = LoadAddressU64(s, AddressPlus(s, info, SlotEntryTable));
	const auto capacity0 = LoadAddressU32(s, AddressPlus(s, info, SlotCapacity0));
	const auto capacity1 = LoadAddressU32(s, AddressPlus(s, info, SlotCapacity1));
	const auto opaque    = Equal(s, geometry, U32(s, 0));
	const auto index     = Select(s, u, opaque, primitive, Add(s, capacity0, primitive));
	const auto valid     = LogicalAnd(
	    s, Binary(s, spv::OpINotEqual, TypeBool(s), table, ConstantU64(s, 0)),
	    Less(s, primitive, Select(s, u, opaque, capacity0, capacity1)));
	const auto entry = ScaledAddress(s, table, Select(s, u, valid, index, U32(s, 0)), 16);
	const auto field = [&](uint32_t offset) {
		return EmitValueOrZeroIfCondition(
		    s, valid, [&] { return LoadAddressU32(s, AddressPlus(s, entry, offset)); });
	};
	const auto records =
	    And(s, Add(s, Binary(s, spv::OpIMul, u, Add(s, capacity0, capacity1), U32(s, 16)), U32(s, AlphaRecordStride - 1)),
	        ~(AlphaRecordStride - 1));
	const auto record = ScaledAddress(s, Binary(s, spv::OpIAdd, wide, table, Unary(s, spv::OpUConvert, wide, records)),
	                                  Select(s, u, valid, primitive, U32(s, 0)), AlphaRecordStride);
	return {field(0), field(4), field(8), field(12), valid, record};
}

inline GuestEntry TriangleEntry(EmitterState& s, const LaneVariables& lane, uint32_t intersection) {
	return TriangleEntry(s, lane.query, intersection, [&] { return Load(s, TypeU64(s), lane.slots); });
}

inline uint32_t InstanceNode(EmitterState& s, const LaneVariables& lane, uint32_t intersection) {
	const auto instance = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpRayQueryGetIntersectionInstanceIdKHR, TypeU32(s), instance, lane.query,
	                      U32(s, intersection));
	return LoadAddressU32(s, ScaledAddress(s, Load(s, TypeU64(s), lane.nodes), instance, 4));
}

inline uint32_t GroupMin(EmitterState& s, uint32_t value) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformUMin, TypeU32(s), result, U32(s, spv::ScopeSubgroup),
	                      spv::GroupOperationReduce, value);
	return result;
}

}

#endif
