#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterRayQueryShared.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#undef near
#undef far

#include <algorithm>
#include <array>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

using namespace RayQueryEmit;

constexpr uint32_t Active       = 1u;
constexpr uint32_t HasCandidate = 2u;
constexpr uint32_t Proposed     = 4u;
constexpr uint32_t Consumed     = 8u;
constexpr uint32_t Dead         = 16u;

constexpr uint32_t ContextRoot      = 48;
constexpr uint32_t ContextServed    = 52;
constexpr uint32_t ContextReturned  = 56;
constexpr uint32_t ContextCounting  = 60;
constexpr uint32_t ContextRounds    = 64;
constexpr uint32_t ContextWaves     = 68;
constexpr uint32_t ContextProposed  = 72;
constexpr uint32_t SlotBlas         = 40;
constexpr uint32_t SlotRoot         = 48;
constexpr uint32_t OracleFlags      = IR::BvhIntersectNative | IR::BvhIntersectOracle | IR::BvhIntersectOracleLane;

struct Call {
	uint32_t descriptor = 0;
	uint32_t node       = 0;
	uint32_t wide       = 0;
	uint32_t extent     = 0;
	uint32_t origin     = 0;
	uint32_t direction  = 0;
	uint32_t inverse    = 0;
	uint32_t exec       = 0;
	uint32_t base       = 0;
	uint32_t address    = 0;
	uint32_t box        = 0;
	uint32_t start      = 0;
	uint32_t resume     = 0;
	uint32_t drained    = 0;
	uint32_t top        = 0;
	uint32_t state      = 0;
	uint32_t leaf       = 0;
	uint32_t entry      = 0;
	uint32_t blas       = 0;
	uint32_t root       = 0;
	uint32_t has        = 0;
};

LaneVariables QueryLane(EmitterState& s, uint32_t half) {
	LaneVariables lane;
	lane.query = s.oracle.lanes[half].query;
	lane.state = s.oracle.lanes[half].state;
	lane.nodes = s.oracle.nodes;
	lane.slots = s.oracle.slots;
	return lane;
}

uint32_t Wide(EmitterState& s, uint32_t value) {
	return Unary(s, spv::OpUConvert, TypeU64(s), value);
}

uint32_t NodeAddress(EmitterState& s, uint32_t base, uint32_t node) {
	const auto wide = TypeU64(s);
	return Binary(s, spv::OpIAdd, wide, base,
	              Binary(s, spv::OpShiftLeftLogical, wide, Binary(s, spv::OpShiftRightLogical, wide, node, U32(s, 3)),
	                     U32(s, 6)));
}

uint32_t Low(EmitterState& s, uint32_t value) {
	return Unary(s, spv::OpUConvert, TypeU32(s), value);
}

uint32_t High(EmitterState& s, uint32_t value) {
	return Low(s, Binary(s, spv::OpShiftRightLogical, TypeU64(s), value, U32(s, 32)));
}

void AtomicAdd(EmitterState& s, uint32_t address, uint32_t value) {
	const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
	s.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(s), s.builder.AllocateId(), pointer, U32(s, spv::ScopeDevice),
	                      U32(s, spv::MemorySemanticsMaskNone), value);
}

void Accumulate(EmitterState& s, uint32_t variable, const std::array<uint32_t, 2>& predicates) {
	auto total = Select(s, TypeU32(s), predicates[0], U32(s, 1), U32(s, 0));
	if (s.lane_count == 2) total = Add(s, total, Select(s, TypeU32(s), predicates[1], U32(s, 1), U32(s, 0)));
	Store(s, variable, Add(s, Load(s, TypeU32(s), variable), total));
}

uint32_t SubgroupSum(EmitterState& s, uint32_t variable) {
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpGroupNonUniformIAdd, TypeU32(s), result, U32(s, spv::ScopeSubgroup),
	                      spv::GroupOperationReduce, Load(s, TypeU32(s), variable));
	return result;
}

}

bool ProgramUsesOracle(const IR::Program& program) {
	return std::ranges::any_of(program.blocks, [](const IR::Block* block) {
		return std::ranges::any_of(*block, [](const IR::Inst& inst) {
			return inst.GetOpcode() == IR::ValueOpcode::BvhIntersect &&
			       (inst.Flags<uint32_t>() & IR::BvhIntersectOracle) != 0;
		});
	});
}

void DeclareOracleVariables(EmitterState& s) {
	if (!ProgramUsesOracle(s.program)) return;
	const auto u = TypeU32(s), wide = TypeU64(s);
	const auto declare = [&](uint32_t pointee, uint32_t& id) {
		id = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpVariable, TypePointer(s, spv::StorageClassFunction, pointee), id,
		                      spv::StorageClassFunction);
	};
	for (uint32_t half = 0; half < s.lane_count; half++) {
		auto& lane = s.oracle.lanes[half];
		declare(s.builder.Type(spv::OpTypeRayQueryKHR), lane.query);
		for (auto* variable: {&lane.state, &lane.leaf, &lane.entry, &lane.t, &lane.root, &lane.proposed, &lane.proposed_root}) {
			declare(u, *variable);
		}
		declare(wide, lane.blas);
		declare(wide, lane.proposed_blas);
	}
	for (auto* variable: {&s.oracle.valid, &s.oracle.root, &s.oracle.counting, &s.oracle.served, &s.oracle.returned,
	                      &s.oracle.proposed, &s.oracle.resumed}) {
		declare(u, *variable);
	}
	for (auto* variable: {&s.oracle.context, &s.oracle.header, &s.oracle.tlas, &s.oracle.nodes, &s.oracle.slots}) {
		declare(wide, *variable);
	}
}

void InitializeOracleVariables(EmitterState& s) {
	if (s.oracle.valid == 0) return;
	const auto u = TypeU32(s), b = TypeBool(s), wide = TypeU64(s);
	const auto entry = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(s, 64), entry, s.bda_pagetable_variable,
	                      U32(s, 0), U32(s, static_cast<uint32_t>(BufferCache::NATIVE_RT_CONTEXT_PAGE)));
	const auto context = Load(s, wide, entry);
	const auto present = Binary(s, spv::OpINotEqual, b, context, ConstantU64(s, 0));
	const auto valid   = EmitValueOrDefaultIfCondition(s, present, b, ConstantBool(s, false), [&] {
		return Binary(s, spv::OpINotEqual, b, LoadAddressU32(s, AddressPlus(s, context, ContextValid)), U32(s, 0));
	});
	Store(s, s.oracle.valid, Select(s, u, valid, U32(s, 1), U32(s, 0)));
	Store(s, s.oracle.context, context);
	for (const auto variable: {s.oracle.header, s.oracle.tlas, s.oracle.nodes, s.oracle.slots}) {
		Store(s, variable, ConstantU64(s, 0));
	}
	Store(s, s.oracle.root, U32(s, None));
	for (const auto variable: {s.oracle.counting, s.oracle.served, s.oracle.returned, s.oracle.proposed, s.oracle.resumed}) {
		Store(s, variable, U32(s, 0));
	}
	EmitIfCondition(s, present, [&] {
		Store(s, s.oracle.counting, LoadAddressU32(s, AddressPlus(s, context, ContextCounting)));
	});
	EmitIfCondition(s, valid, [&] {
		Store(s, s.oracle.header, LoadAddressU64(s, AddressPlus(s, context, ContextHeader)));
		Store(s, s.oracle.tlas, LoadAddressU64(s, AddressPlus(s, context, ContextTlas)));
		Store(s, s.oracle.nodes, LoadAddressU64(s, AddressPlus(s, context, ContextInstanceNodes)));
		Store(s, s.oracle.slots, LoadAddressU64(s, AddressPlus(s, context, ContextSlots)));
		Store(s, s.oracle.root, LoadAddressU32(s, AddressPlus(s, context, ContextRoot)));
	});
	for (uint32_t half = 0; half < s.lane_count; half++) {
		const auto& lane = s.oracle.lanes[half];
		Store(s, lane.state, U32(s, 0));
		for (const auto variable: {lane.leaf, lane.entry, lane.root, lane.proposed, lane.proposed_root}) {
			Store(s, variable, U32(s, None));
		}
		Store(s, lane.t, U32(s, 0));
		Store(s, lane.blas, ConstantU64(s, 0));
		Store(s, lane.proposed_blas, ConstantU64(s, 0));
	}
}

uint32_t EmitBvhOracle(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	if (ctx.half != 0) return s.oracle.results[ctx.half];
	const auto  u = TypeU32(s), f = TypeF32(s), b = TypeBool(s), wide = TypeU64(s);
	const auto  vec3 = TypeF32Vector(s, 3), vec4 = TypeU32Vector(s, 4);
	const auto* ray        = ctx.ImageAddress(inst.Arg(1));
	const auto  node_words = inst.Flags<uint32_t>() & ~OracleFlags;
	const bool  per_lane   = (inst.Flags<uint32_t>() & IR::BvhIntersectOracleLane) != 0;
	if (ray == nullptr || (node_words != 1u && node_words != 2u) || ray->NumArgs() < node_words + 10u ||
	    s.bvh_intersect_function == 0) {
		ctx.Fail(inst, "BVH oracle requires its node pointer, ray components and the shared function");
	}
	std::array<Call, 2> calls {};
	const auto none   = U32(s, None);
	const auto valid  = Binary(s, spv::OpINotEqual, b, Load(s, u, s.oracle.valid), U32(s, 0));
	const auto root   = Load(s, u, s.oracle.root);
	const auto first  = NodeAddress(s, Load(s, wide, s.oracle.header), Wide(s, root));
	const auto again  = Binary(s, spv::OpIAdd, wide, first, ConstantU64(s, 64));
	const auto slack  = ConstantF32Value(s, 1.0001f);
	for (uint32_t half = 0; half < s.lane_count; half++) {
		auto&       c    = calls[half];
		const auto& vars = s.oracle.lanes[half];
		const auto  lane = QueryLane(s, half);
		c.descriptor     = ctx.HalfArg(inst, 0, half);
		c.node           = ctx.HalfArg(*ray, 0, half);
		c.wide           = node_words == 2u ? PackU64(s, c.node, ctx.HalfArg(*ray, 1, half)) : Wide(s, c.node);
		c.extent         = Unary(s, spv::OpBitcast, f, ctx.HalfArg(*ray, node_words, half));
		c.exec           = ctx.HalfArg(inst, 2, half);
		std::array<uint32_t, 9> parts {};
		for (uint32_t i = 0; i < 9; i++) {
			parts[i] = Unary(s, spv::OpBitcast, f, ctx.HalfArg(*ray, node_words + 1u + i, half));
		}
		const auto vector = [&](uint32_t at) {
			const auto result = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeConstruct, vec3, result, parts[at], parts[at + 1], parts[at + 2]);
			return result;
		};
		c.origin    = vector(0);
		c.direction = vector(3);
		c.inverse   = vector(6);
		const auto word = [&](uint32_t index) {
			const auto result = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeExtract, u, result, c.descriptor, index);
			return result;
		};
		c.base          = Binary(s, spv::OpShiftLeftLogical, wide, PackU64(s, word(0), And(s, word(1), 0xffu)), U32(s, 8));
		c.address       = NodeAddress(s, c.base, c.wide);
		const auto kind = And(s, c.node, 7u);
		c.box           = LogicalOr(s, Equal(s, kind, U32(s, 4)), Equal(s, kind, U32(s, 5)));
		const auto on   = LogicalAnd(s, LogicalAnd(s, c.exec, valid), c.box);
		const auto before = Load(s, u, vars.state);
		const auto at_again = LogicalAnd(s, on, Binary(s, spv::OpIEqual, b, c.address, again));
		c.start   = LogicalAnd(s, on, Binary(s, spv::OpIEqual, b, c.address, first));
		c.resume  = LogicalAnd(s, at_again, BitSet(s, before, Active));
		c.drained = LogicalAnd(s, at_again, LogicalNot(s, BitSet(s, before, Active)));
		c.top     = LogicalOr(s, c.start, c.resume);
		EmitIfCondition(s, c.start, [&] {
			auto traced = Binary(s, spv::OpFOrdGreaterThanEqual, b, c.extent, ConstantF32Value(s, 0.0f));
			for (uint32_t i = 0; i < 6; i++) {
				traced = LogicalAnd(s, traced, Finite(s, parts[i]));
			}
			Store(s, vars.state, Select(s, u, traced, U32(s, Active), U32(s, Active | Dead)));
			EmitIfCondition(s, traced, [&] {
				const auto structure = Unary(s, spv::OpConvertUToAccelerationStructureKHR,
				                             s.builder.Type(spv::OpTypeAccelerationStructureKHR), Load(s, wide, s.oracle.tlas));
				s.builder.AddFunction(spv::OpRayQueryInitializeKHR, lane.query, structure, U32(s, RayFlagNoOpaque),
				                      U32(s, 0xffu), c.origin, ConstantF32Value(s, 0.0f), c.direction, c.extent);
			});
		});
		const auto entered = Load(s, u, vars.state);
		const auto holds   = BitSet(s, entered, HasCandidate);
		const auto held_t  = Unary(s, spv::OpBitcast, f, Load(s, u, vars.t));
		const auto beyond  = LogicalAnd(s, holds, Binary(s, spv::OpFOrdGreaterThan, b, held_t, c.extent));
		const auto shrunk  = LogicalAnd(s, LogicalAnd(s, c.resume, holds),
		                                Binary(s, spv::OpFOrdLessThanEqual, b, c.extent, Binary(s, spv::OpFMul, f, held_t, slack)));
		EmitIfCondition(s, shrunk, [&] { s.builder.AddFunction(spv::OpRayQueryConfirmIntersectionKHR, lane.query); });
		const auto advance = LogicalAnd(
		    s, LogicalNot(s, BitSet(s, entered, Dead)),
		    LogicalOr(s, c.start, LogicalAnd(s, LogicalAnd(s, c.resume, holds), LogicalOr(s, BitSet(s, entered, Consumed), beyond))));
		EmitIfCondition(s, advance, [&] {
			Store(s, vars.state, And(s, entered, ~(HasCandidate | Consumed)));
			const auto reach = Binary(s, spv::OpFMul, f, c.extent, slack);
			EmitLoop(s, [&](uint32_t merge, uint32_t cont) {
				const auto proceed = s.builder.AllocateId();
				s.builder.AddFunction(spv::OpRayQueryProceedKHR, b, proceed, lane.query);
				const auto more = EmitValueOrDefaultIfCondition(s, proceed, b, ConstantBool(s, false), [&] {
					const auto type = s.builder.AllocateId();
					s.builder.AddFunction(spv::OpRayQueryGetIntersectionTypeKHR, u, type, lane.query, U32(s, CandidateIntersection));
					const auto triangle =
					    Equal(s, type, U32(s, spv::RayQueryCandidateIntersectionTypeRayQueryCandidateIntersectionTriangleKHR));
					return EmitValueOrDefaultIfCondition(s, triangle, b, ConstantBool(s, true), [&] {
						const auto t = s.builder.AllocateId();
						s.builder.AddFunction(spv::OpRayQueryGetIntersectionTKHR, f, t, lane.query, U32(s, CandidateIntersection));
						const auto near = Binary(s, spv::OpFOrdLessThanEqual, b, t, reach);
						return EmitValueOrDefaultIfCondition(s, near, b, ConstantBool(s, true), [&] {
							const auto candidate = TriangleEntry(s, lane, CandidateIntersection);
							const auto usable =
							    LogicalAnd(s, candidate.valid, Binary(s, spv::OpINotEqual, b, candidate.entry, none));
							EmitIfCondition(s, usable, [&] {
								const auto slot = s.builder.AllocateId();
								s.builder.AddFunction(spv::OpRayQueryGetIntersectionInstanceCustomIndexKHR, u, slot, lane.query,
								                      U32(s, CandidateIntersection));
								const auto info = ScaledAddress(s, Load(s, wide, s.oracle.slots), slot, SlotStride);
								Store(s, vars.leaf, InstanceNode(s, lane, CandidateIntersection));
								Store(s, vars.entry, candidate.entry);
								Store(s, vars.t, Unary(s, spv::OpBitcast, u, t));
								Store(s, vars.blas, LoadAddressU64(s, AddressPlus(s, info, SlotBlas)));
								Store(s, vars.root, LoadAddressU32(s, AddressPlus(s, info, SlotRoot)));
								Store(s, vars.state, Or(s, Load(s, u, vars.state), U32(s, HasCandidate)));
							});
							return LogicalNot(s, usable);
						});
					});
				});
				s.builder.AddFunction(spv::OpBranchConditional, more, cont, merge);
			});
		});
		c.state         = Load(s, u, vars.state);
		const auto mine = LogicalAnd(s, c.top, BitSet(s, c.state, HasCandidate));
		c.leaf          = Select(s, u, mine, Load(s, u, vars.leaf), none);
		c.entry         = Select(s, u, mine, Load(s, u, vars.entry), none);
		c.blas          = Load(s, wide, vars.blas);
		c.root          = Load(s, u, vars.root);
	}
	if (!per_lane) {
		auto local_leaf = calls[0].leaf, local_entry = calls[0].entry;
		if (s.lane_count == 2) {
			const auto smaller = KeyLess(s, calls[1].leaf, calls[1].entry, local_leaf, local_entry);
			local_leaf         = Select(s, u, smaller, calls[1].leaf, local_leaf);
			local_entry        = Select(s, u, smaller, calls[1].entry, local_entry);
		}
		const auto leaf  = GroupMin(s, local_leaf);
		const auto entry = GroupMin(s, Select(s, u, Equal(s, local_leaf, leaf), local_entry, none));
		std::array<uint32_t, 3> pick {none, none, none};
		for (uint32_t half = 0; half < s.lane_count; half++) {
			const auto& c     = calls[half];
			const auto  match = LogicalAnd(s, Equal(s, c.leaf, leaf), Equal(s, c.entry, entry));
			const std::array<uint32_t, 3> parts {Low(s, c.blas), High(s, c.blas), c.root};
			for (uint32_t i = 0; i < 3; i++) {
				const auto value = Select(s, u, match, parts[i], none);
				pick[i]          = half == 0 ? value : Select(s, u, Less(s, value, pick[i]), value, pick[i]);
			}
		}
		std::array<uint32_t, 3> shared {};
		for (uint32_t i = 0; i < 3; i++) {
			shared[i] = GroupMin(s, pick[i]);
		}
		const auto has  = Binary(s, spv::OpINotEqual, b, leaf, none);
		const auto blas = PackU64(s, shared[0], shared[1]);
		for (uint32_t half = 0; half < s.lane_count; half++) {
			auto& c = calls[half];
			c.leaf  = leaf;
			c.entry = entry;
			c.blas  = blas;
			c.root  = shared[2];
			c.has   = has;
		}
	} else {
		for (uint32_t half = 0; half < s.lane_count; half++) {
			calls[half].has = Binary(s, spv::OpINotEqual, b, calls[half].leaf, none);
		}
	}
	std::array<uint32_t, 2> served {}, returned {}, candidates {};
	for (uint32_t half = 0; half < s.lane_count; half++) {
		auto&       c    = calls[half];
		const auto& vars = s.oracle.lanes[half];
		const auto  held = LogicalAnd(s, BitSet(s, c.state, HasCandidate),
		                              LogicalAnd(s, Equal(s, Load(s, u, vars.leaf), c.leaf), Equal(s, Load(s, u, vars.entry), c.entry)));
		auto next = Or(s, And(s, c.state, ~(Consumed | Proposed)),
		               Or(s, Select(s, u, held, U32(s, Consumed), U32(s, 0)), Select(s, u, c.has, U32(s, Proposed), U32(s, 0))));
		next      = Select(s, u, c.has, next, And(s, next, ~Active));
		const auto proposed      = Load(s, u, vars.proposed);
		const auto proposed_blas = Load(s, wide, vars.proposed_blas);
		const auto proposed_root = Load(s, u, vars.proposed_root);
		const auto blas_base     = Binary(s, spv::OpBitwiseAnd, wide, proposed_blas, ConstantU64(s, ~uint64_t {0xff}));
		const auto ongoing = LogicalAnd(s, LogicalAnd(s, c.exec, LogicalAnd(s, valid, c.box)),
		                                LogicalAnd(s, LogicalNot(s, LogicalOr(s, c.top, c.drained)),
		                                           LogicalAnd(s, BitSet(s, c.state, Active), BitSet(s, c.state, Proposed))));
		const auto nested  = LogicalAnd(s, ongoing,
		                                Binary(s, spv::OpIEqual, b, c.address, NodeAddress(s, blas_base, Wide(s, proposed_root))));
		const auto inside  = LogicalAnd(s, LogicalAnd(s, ongoing, LogicalNot(s, nested)),
		                                Binary(s, spv::OpIEqual, b, c.base, blas_base));
		const auto foreign = LogicalAnd(s, LogicalAnd(s, c.exec, c.box),
		                                LogicalNot(s, LogicalOr(s, LogicalOr(s, c.top, c.drained), LogicalOr(s, nested, inside))));
		Store(s, vars.state, Select(s, u, c.top, next, Select(s, u, foreign, U32(s, 0), c.state)));
		Store(s, vars.proposed, Select(s, u, c.top, c.entry, proposed));
		Store(s, vars.proposed_blas, Select(s, wide, c.top, c.blas, proposed_blas));
		Store(s, vars.proposed_root, Select(s, u, c.top, c.root, proposed_root));
		const auto guest  = LogicalAnd(s, c.exec, LogicalNot(s, LogicalOr(s, LogicalOr(s, c.top, c.drained), nested)));
		const auto walked = EmitValueOrDefaultIfCondition(s, guest, vec4, ConstantU32CompositeZero(s, 4), [&] {
			const auto result = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpFunctionCall, vec4, result, s.bvh_intersect_function, c.descriptor, c.wide,
			                      c.extent, c.origin, c.direction, c.inverse);
			return result;
		});
		const std::array<uint32_t, 4> top {Select(s, u, c.has, c.leaf, none), Select(s, u, c.has, Add(s, root, U32(s, 8)), none),
		                                   none, none};
		std::array<uint32_t, 4> words {};
		for (uint32_t i = 0; i < 4; i++) {
			const auto component = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeExtract, u, component, walked, i);
			const auto below = i == 0 ? proposed : none;
			words[i] = Select(s, u, c.top, top[i], Select(s, u, c.drained, none, Select(s, u, nested, below, component)));
		}
		s.oracle.results[half] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, vec4, s.oracle.results[half], words[0], words[1], words[2], words[3]);
		served[half]     = c.start;
		returned[half]   = LogicalAnd(s, guest, c.box);
		candidates[half] = LogicalAnd(s, c.top, c.has);
	}
	Accumulate(s, s.oracle.served, served);
	Accumulate(s, s.oracle.returned, returned);
	Accumulate(s, s.oracle.proposed, candidates);
	const auto resumed = s.lane_count == 2 ? LogicalOr(s, calls[0].resume, calls[1].resume) : calls[0].resume;
	Store(s, s.oracle.resumed, Add(s, Load(s, TypeU32(s), s.oracle.resumed), Select(s, TypeU32(s), resumed, U32(s, 1), U32(s, 0))));
	return s.oracle.results[0];
}

void FlushOracleCounts(EmitterState& s) {
	if (s.oracle.valid == 0) return;
	const auto b        = TypeBool(s);
	const auto counting = Binary(s, spv::OpINotEqual, b, Load(s, TypeU32(s), s.oracle.counting), U32(s, 0));
	EmitIfCondition(s, s.ray_stats ? ConstantBool(s, true) : counting, [&] {
		const auto rays     = SubgroupSum(s, s.oracle.served);
		const auto guest    = SubgroupSum(s, s.oracle.returned);
		const auto proposed = SubgroupSum(s, s.oracle.proposed);
		const auto rounds   = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformUMax, TypeU32(s), rounds, U32(s, spv::ScopeSubgroup),
		                      spv::GroupOperationReduce, Load(s, TypeU32(s), s.oracle.resumed));
		const auto waves    = Select(s, TypeU32(s), Binary(s, spv::OpINotEqual, b, rays, U32(s, 0)), U32(s, 1), U32(s, 0));
		const auto elected  = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformElect, b, elected, U32(s, spv::ScopeSubgroup));
		const auto any = Binary(s, spv::OpINotEqual, b, Or(s, rays, guest), U32(s, 0));
		EmitIfCondition(s, LogicalAnd(s, LogicalAnd(s, elected, counting), any), [&] {
			const auto context = Load(s, TypeU64(s), s.oracle.context);
			AtomicAdd(s, AddressPlus(s, context, ContextServed), rays);
			AtomicAdd(s, AddressPlus(s, context, ContextReturned), guest);
			AtomicAdd(s, AddressPlus(s, context, ContextRounds), rounds);
			AtomicAdd(s, AddressPlus(s, context, ContextWaves), waves);
			AtomicAdd(s, AddressPlus(s, context, ContextProposed), proposed);
		});
		if (!s.ray_stats) return;
		const auto marked = Binary(s, spv::OpINotEqual, b, Or(s, Or(s, rays, guest), proposed), U32(s, 0));
		EmitIfCondition(s, LogicalAnd(s, elected, marked), [&] {
			EmitRayMarkCount(s, BufferCache::HACK_MARK_ORACLE_RETURNED, guest);
			EmitRayMarkCount(s, BufferCache::HACK_MARK_ORACLE_RAYS, rays);
			EmitRayMarkCount(s, BufferCache::HACK_MARK_ORACLE_CANDIDATES, proposed);
			EmitRayMarkCount(s, BufferCache::HACK_MARK_ORACLE_ROUNDS, rounds);
			EmitRayMarkCount(s, BufferCache::HACK_MARK_ORACLE_WAVES, waves);
		});
	});
}

}
