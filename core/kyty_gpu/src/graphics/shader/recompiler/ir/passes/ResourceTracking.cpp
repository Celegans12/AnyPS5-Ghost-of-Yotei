#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include "common/assert.h"
#include "common/envRead.h"
#include "common/logging/log.h"
#include "common/workaroundHits.h"
#include "common/shaderEnv.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <atomic>
#include <set>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <map>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <span>
#include <tuple>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

constexpr uint32_t SamplerBorderClampMask    = (1u << 2u) | (1u << 5u) | (1u << 8u);
constexpr uint32_t SamplerDword3ReservedMask = 0x3ffff000u;

uint32_t PossibleU32Bits(Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32 ? value.U32() : UINT32_MAX;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return UINT32_MAX;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::BitwiseAnd32:
			return PossibleU32Bits(inst->Arg(0)) & PossibleU32Bits(inst->Arg(1));
		case ValueOpcode::BitwiseOr32:
			return PossibleU32Bits(inst->Arg(0)) | PossibleU32Bits(inst->Arg(1));
		case ValueOpcode::ShiftLeftLogical32: {
			const auto shift = inst->Arg(1).Resolve();
			return shift.IsImmediate() && shift.GetType() == Type::U32
			           ? PossibleU32Bits(inst->Arg(0)) << (shift.U32() & 31u)
			           : UINT32_MAX;
		}
		default: return UINT32_MAX;
	}
}

// Composed V#s: an immediate 0xffffffff, or a select between two (Translator::ThreadBit's word of a
// 64-lane mask whose halves are both all ones), or a phi of such (the back edge carrying the phi itself aside).
bool AllOnesWord(Value value, uint32_t depth = 0) {
	if (depth > 8u) return false;
	value = value.Resolve();
	if (value.IsImmediate()) return value.GetType() == Type::U32 && value.U32() == UINT32_MAX;
	const auto* inst = value.TryInstruction();
	if (inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi) {
		bool any = false;
		for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
			const auto operand = inst->Arg(arg).Resolve();
			if (operand == value) continue;
			if (!AllOnesWord(operand, depth + 1u)) return false;
			any = true;
		}
		return any;
	}
	return inst != nullptr && inst->GetOpcode() == ValueOpcode::SelectU32 && inst->NumArgs() == 3u &&
	       AllOnesWord(inst->Arg(1), depth + 1u) && AllOnesWord(inst->Arg(2), depth + 1u);
}

// Composed V#s: one bit of an all-ones word at a position below 32: (word >> shift) & 1 (either operand
// order of the and) or BitFieldUExtract(word, offset, 1). It is 1 whatever the lane the position depends on.
bool BitOfAllOnes(Value bit) {
	bit             = bit.Resolve();
	const auto* inst = bit.TryInstruction();
	if (inst == nullptr) return false;
	if (inst->GetOpcode() == ValueOpcode::BitFieldUExtract && inst->NumArgs() == 3u) {
		const auto count = inst->Arg(2).Resolve();
		return count.IsImmediate() && count.GetType() == Type::U32 && count.U32() == 1u && AllOnesWord(inst->Arg(0)) &&
		       PossibleU32Bits(inst->Arg(1)) < 32u;
	}
	if (inst->GetOpcode() != ValueOpcode::BitwiseAnd32 || inst->NumArgs() != 2u) return false;
	for (uint32_t side = 0; side < 2u; ++side) {
		const auto one = inst->Arg(side).Resolve();
		if (!one.IsImmediate() || one.GetType() != Type::U32 || one.U32() != 1u) continue;
		const auto* shift = inst->Arg(side ^ 1u).Resolve().TryInstruction();
		if (shift != nullptr && shift->GetOpcode() == ValueOpcode::ShiftRightLogical32 && shift->NumArgs() == 2u &&
		    AllOnesWord(shift->Arg(0)) && PossibleU32Bits(shift->Arg(1)) < 32u)
			return true;
	}
	return false;
}

// Composed V#s: `value` (U1) is true on every lane. An immediate true; Translator::ThreadBit of a mask
// whose words are all ones (((word >> (lane & 31)) & 1) != 0, or == 1: S_MOV_B64 exec, -1 and its like); a conjunction
// of such; a disjunction with one; a select whose condition is one and whose first arm is, or whose arms both are; a
// phi whose operands all are (EXEC set to all ones on every path into a merge, as 32acc45f's S_MOV_B64 exec, -1 before
// and inside an if; a loop's back edge carrying the phi itself keeps the entry value).
bool TrueOnEveryLane(Value value, uint32_t depth = 0) {
	if (depth > 16u) return false;
	value = value.Resolve();
	if (value.IsImmediate()) return value.GetType() == Type::U1 && value.U1();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return false;
	switch (inst->GetOpcode()) {
		case ValueOpcode::Phi: {
			bool any = false;
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
				const auto operand = inst->Arg(arg).Resolve();
				if (operand == value) continue;
				if (!TrueOnEveryLane(operand, depth + 1u)) return false;
				any = true;
			}
			return any;
		}
		case ValueOpcode::LogicalAnd:
			return inst->NumArgs() == 2u && TrueOnEveryLane(inst->Arg(0), depth + 1u) &&
			       TrueOnEveryLane(inst->Arg(1), depth + 1u);
		case ValueOpcode::LogicalOr:
			return inst->NumArgs() == 2u &&
			       (TrueOnEveryLane(inst->Arg(0), depth + 1u) || TrueOnEveryLane(inst->Arg(1), depth + 1u));
		case ValueOpcode::SelectU1:
			if (inst->NumArgs() != 3u) return false;
			if (TrueOnEveryLane(inst->Arg(0), depth + 1u)) return TrueOnEveryLane(inst->Arg(1), depth + 1u);
			return TrueOnEveryLane(inst->Arg(1), depth + 1u) && TrueOnEveryLane(inst->Arg(2), depth + 1u);
		case ValueOpcode::INotEqual32:
		case ValueOpcode::IEqual32: {
			if (inst->NumArgs() != 2u) return false;
			const uint32_t wanted = inst->GetOpcode() == ValueOpcode::INotEqual32 ? 0u : 1u; // bit != 0, bit == 1
			for (uint32_t side = 0; side < 2u; ++side) {
				const auto constant = inst->Arg(side).Resolve();
				if (constant.IsImmediate() && constant.GetType() == Type::U32 && constant.U32() == wanted &&
				    BitOfAllOnes(inst->Arg(side ^ 1u)))
					return true;
			}
			return false;
		}
		default: return false;
	}
}

// Value bounds: the largest value `value` can take when every value it can take is a shader constant
// (immediates through the arms of a select, the operands of a phi and a read-first-lane); UINT32_MAX otherwise. Only
// constants: the records they name are ones the shader selects on purpose, while the width of a field or mask (e.g.
// Update Oct Maps' 5-bit slot field) also admits stale records. Cycles (loop phis) end at the depth limit.
// `active`: the mask of the ReadFirstLane that reads `value`, whose lane is known active. A select on exactly that mask
// (a predicated VGPR write) gives its new value on that lane, so its old value needs no bound. A phi arm on a loop back
// edge (`back_edge`; none given: every arm) and a nested lane read drop the mask (another iteration's or lane's value).
// `memo` (per top-level call): each (instruction, mask kept) is bounded once; an entry being computed reads UINT32_MAX
// (a cycle bounds nothing), and a bound found with more depth left stays a true bound deeper down.
using BoundMemo = std::unordered_map<const Inst*, std::array<uint32_t, 2>>;
uint32_t BoundedU32Max(Value value, uint32_t depth, Value active, const std::function<bool(const Inst&, size_t)>& back_edge,
                       BoundMemo& memo) {
	if (depth > 24u) return UINT32_MAX;
	value = value.Resolve();
	if (value.IsImmediate()) return value.GetType() == Type::U32 ? value.U32() : UINT32_MAX;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return UINT32_MAX;
	constexpr uint32_t Unknown = UINT32_MAX - 1u; // not computed yet (no real bound: keys are at most 65535)
	auto [entry, inserted] = memo.try_emplace(inst, std::array<uint32_t, 2> {Unknown, Unknown});
	auto& slot = entry->second[active.IsEmpty() ? 0u : 1u];
	if (slot != Unknown) return slot;
	slot = UINT32_MAX; // in progress
	const auto arg = [&](size_t index, Value mask) {
		return BoundedU32Max(inst->Arg(index), depth + 1u, mask, back_edge, memo);
	};
	uint32_t result = UINT32_MAX;
	switch (inst->GetOpcode()) {
		case ValueOpcode::SelectU32:
			result = !active.IsEmpty() && inst->Arg(0).Resolve() == active ? arg(1, active)
			                                                                : std::max(arg(1, active), arg(2, active));
			break;
		case ValueOpcode::ReadFirstLane: result = arg(0, {}); break;
		case ValueOpcode::Phi: {
			result = inst->NumArgs() == 0u ? UINT32_MAX : 0u;
			for (size_t index = 0; index < inst->NumArgs() && result != UINT32_MAX; ++index) {
				const bool back = !back_edge || back_edge(*inst, index);
				result          = std::max(result, arg(index, back ? Value {} : active));
			}
			break;
		}
		default: break;
	}
	memo[inst][active.IsEmpty() ? 0u : 1u] = result;
	return result;
}

bool IsNoncanonicalFlatAddress(Value high, Value active) {
	high = high.Resolve();
	active = active.Resolve();
	// A predicated VGPR write supplies its new value on exactly the lanes which
	// execute this access. Do not discard a selection under a different mask.
	while (const auto* select = high.TryInstruction()) {
		if (select->GetOpcode() != ValueOpcode::SelectU32 || select->Arg(0).Resolve() != active) break;
		high = select->Arg(1).Resolve();
	}
	uint32_t lower = 0, upper = UINT32_MAX;
	if (high.IsImmediate() && high.GetType() == Type::U32) {
		lower = upper = high.U32();
	} else if (const auto* inst = high.TryInstruction();
	           inst != nullptr && inst->GetOpcode() == ValueOpcode::UMedTri32) {
		// The median lies between any two constant inputs, regardless of the third.
		uint32_t count = 0;
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			const auto value = inst->Arg(index).Resolve();
			if (!value.IsImmediate() || value.GetType() != Type::U32) continue;
			if (count++ == 0) lower = upper = value.U32();
			else { lower = std::min(lower, value.U32()); upper = std::max(upper, value.U32()); }
		}
		if (count < 2) return false;
	}
	// Leave one high DWORD on each side for the signed instruction offset/carry.
	return lower > 0x00008000u && upper < 0xffff7fffu;
}

Value CanonicalizeSampleAdjustDword3(Value value) {
	for (;;) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseOr32) {
			if ((PossibleU32Bits(value) & ~SamplerDword3ReservedMask) == 0)
				return Value(0u);
			return value;
		}
		const auto left           = inst->Arg(0).Resolve();
		const auto right          = inst->Arg(1).Resolve();
		const bool left_reserved  = (PossibleU32Bits(left) & ~SamplerDword3ReservedMask) == 0;
		const bool right_reserved = (PossibleU32Bits(right) & ~SamplerDword3ReservedMask) == 0;
		if (left_reserved && right_reserved) {
			return Value(0u);
		}
		if (left_reserved) {
			value = right;
		} else if (right_reserved) {
			value = left;
		} else {
			return value;
		}
	}
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

uint32_t ByteExtent(const MemoryInfo& memory) {
	const auto bytes = std::max((memory.data_bits + 7u) / 8u, 1u);
	const auto count = std::max(memory.data_dwords, 1u);
	const auto end   = static_cast<uint64_t>(memory.offset) + static_cast<uint64_t>(bytes) * count;
	return end > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(end);
}

// Beyond this many separate fixed write ranges a buffer counts as written over its whole range.
constexpr size_t MaxFixedWriteRanges = 16;

// The bytes [first, second) from the descriptor base that a raw buffer store or atomic writes when its address is
// fixed in the shader: index and vector offset zero, scalar offset immediate (args: resource, index, offset,
// soffset). The descriptor's add-TID and swizzle bits are checked where the ranges are used.
std::optional<std::pair<uint32_t, uint32_t>> FixedWriteRange(const Inst& inst, const MemoryInfo& memory,
                                                             ValueOpcode op) {
	if (memory.kind != ResourceKind::Buffer || memory.formatted || memory.typed || inst.NumArgs() < 4) {
		return std::nullopt;
	}
	const auto immediate = [&](size_t index) -> std::optional<uint32_t> {
		const auto value = inst.Arg(index).Resolve();
		if (!value.IsImmediate() || value.GetType() != Type::U32) return std::nullopt;
		return value.U32();
	};
	const auto index   = immediate(1);
	const auto offset  = immediate(2);
	const auto soffset = immediate(3);
	if (!index || *index != 0u || !offset || *offset != 0u || !soffset) return std::nullopt;
	const uint32_t bytes = op == ValueOpcode::StoreBufferU8    ? 1u
	                       : op == ValueOpcode::StoreBufferU16 ? 2u
	                                                           : 4u * BufferComponentCount(op);
	const auto     begin = uint64_t {*soffset} + memory.offset;
	if (bytes == 0u || begin + bytes > UINT32_MAX) return std::nullopt;
	return std::pair {static_cast<uint32_t>(begin), static_cast<uint32_t>(begin + bytes)};
}

// Hashes of the LoopBoundProof decision-diagram tables. Ordered maps of these keys took most of the
// translation time of the shaders with record-keyed buffers (40395313 6 s, 7ff41f74 23 s), in
// lexicographic key compares and tree teardown.
inline uint64_t ProofHashMix(uint64_t value) {
	value ^= value >> 30u;
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= value >> 27u;
	value *= 0x94d049bb133111ebull;
	return value ^ (value >> 31u);
}

struct ProofTripleHash {
	size_t operator()(const std::array<uint32_t, 3>& key) const noexcept {
		return static_cast<size_t>(
		    ProofHashMix(((uint64_t {key[0]} << 32u) | key[1]) ^ ProofHashMix(uint64_t {key[2]} + 0x9e3779b97f4a7c15ull)));
	}
};

struct ProofPredicateHash {
	size_t operator()(const std::tuple<ValueOpcode, uint64_t, uint32_t, uint32_t>& key) const noexcept {
		const auto& [opcode, flags, left, right] = key;
		return static_cast<size_t>(ProofHashMix(
		    ProofHashMix(flags ^ (uint64_t {static_cast<uint32_t>(opcode)} << 48u)) ^
		    ((uint64_t {left} << 32u) | right)));
	}
};

// Prove a loop cannot continue after its bound fails, both on entry and after an
// arbitrary previous iteration. Header Phis are substituted simultaneously; all
// unsupported expressions remain unconstrained. This state exists only while tracking.
class LoopBoundProof {
public:
	LoopBoundProof(const Program& program, const Inst& induction, const Inst& bound)
	    : m_program(program), m_induction(induction), m_bound(bound) {}

	bool Excludes(Value condition, bool positive) {
		for (uint32_t incoming = 0; incoming < m_induction.NumArgs(); ++incoming) {
			m_incoming = m_induction.PhiBlock(incoming);
			m_values[1].clear();
			if (Evaluate(condition, true) != (positive ? 0u : 1u)) return false;
		}
		return true;
	}

	bool SelectsRange(const Inst& selected, Value active, const Inst& candidate,
	                  const Inst& carried, const Inst& mask,
	                  std::span<const std::pair<Value, bool>> exit_conditions) {
		// Inactive lanes retain a previously selected index. Initially every lane is
		// active; the bounded assignment establishes the invariant on each backedge.
		m_exclude_bound = false;
		std::vector<uint32_t> range_values;
		const auto in_range = [&] {
			const auto value = Unknown(Type::U32);
			range_values.push_back(value);
			return value;
		};
		const auto contains = [&](const auto& self, uint32_t node) -> uint32_t {
			if (std::ranges::find(range_values, node) != range_values.end()) return 1u;
			const auto entry = m_nodes[node];
			if (entry.variable == UINT32_MAX) return 0u;
			return Select(NodeFor(entry.variable, 0u, 1u), self(self, entry.high),
			              self(self, entry.low));
		};
		const auto state = [&](bool in_body) {
			m_values[1].clear();
			const auto enabled = Unknown(Type::U1);
			for (const auto& phi: *m_induction.Parent()) {
				if (phi.GetOpcode() != ValueOpcode::Phi) continue;
				m_values[1][&phi] = EquivalentValue(m_program, Value(const_cast<Inst*>(&phi)),
				                                      Value(const_cast<Inst*>(&mask)))
				                         ? enabled : Unknown(phi.GetType());
			}
			m_values[1][&carried] = Select(enabled, Unknown(Type::U32), in_range());
			if (in_body) m_values[1][&m_bound] = 1u;
			m_values[1][&candidate] = Select(Evaluate(Value(const_cast<Inst*>(&m_bound)), true),
			                                  in_range(), Unknown(Type::U32));
		};
		for (uint32_t arm = 0; arm < carried.NumArgs(); ++arm) {
			const auto initial = mask.Arg(arm).Resolve();
			if (initial.IsImmediate() && initial.GetType() == Type::U1 && initial.U1()) continue;
			state(true);
			const auto retained = contains(contains, Evaluate(carried.Arg(arm), true));
			if (Select(Evaluate(mask.Arg(arm), true), 1u, retained) != 1u) return false;
		}
		for (uint32_t arm = 0; arm < selected.NumArgs(); ++arm) {
			state(false);
			const auto value = Evaluate(selected.Arg(arm), true);
			m_values[1][&selected] = value;
			auto allowed = Evaluate(exit_conditions[arm].first, true);
			if (!exit_conditions[arm].second) allowed = Select(allowed, 0u, 1u);
			allowed = Select(allowed, Evaluate(active, true), 0u);
			if (Select(allowed, contains(contains, value), 1u) != 1u) return false;
		}
		return true;
	}

private:
	struct Node {
		uint32_t variable = UINT32_MAX;
		uint32_t low = 0;
		uint32_t high = 0;
	};

	uint32_t NodeFor(uint32_t variable, uint32_t low, uint32_t high) {
		if (low == high) return low;
		const auto [it, inserted] = m_nodes_by_key.try_emplace(
		    std::array {variable, low, high}, static_cast<uint32_t>(m_nodes.size()));
		if (inserted) m_nodes.push_back({variable, low, high});
		return it->second;
	}

	uint32_t Unknown(Type type) {
		if (type == Type::U1) return NodeFor(m_variables++, 0u, 1u);
		m_nodes.emplace_back();
		return static_cast<uint32_t>(m_nodes.size() - 1u);
	}

	uint32_t Select(uint32_t condition, uint32_t yes, uint32_t no) {
		if (condition == 0u) return no;
		if (condition == 1u || yes == no) return yes;
		if (yes == 1u && no == 0u) return condition;
		const std::array key {condition, yes, no};
		if (const auto found = m_choices.find(key); found != m_choices.end()) return found->second;
		const auto variable = std::min({m_nodes[condition].variable, m_nodes[yes].variable,
		                                m_nodes[no].variable});
		const auto arm = [&](uint32_t value, bool high) {
			const auto node = m_nodes[value];
			return node.variable == variable ? (high ? node.high : node.low) : value;
		};
		const auto low = Select(arm(condition, false), arm(yes, false), arm(no, false));
		const auto high = Select(arm(condition, true), arm(yes, true), arm(no, true));
		const auto result = NodeFor(variable, low, high);
		m_choices.emplace(key, result);
		return result;
	}

	uint32_t Compare(const Inst& inst, uint32_t left, uint32_t right) {
		if (left == right) {
			if (inst.GetOpcode() == ValueOpcode::IEqual32) return 1u;
			if (inst.GetOpcode() == ValueOpcode::INotEqual32) return 0u;
		}
		const auto key = std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), left, right};
		if (const auto found = m_predicates.find(key); found != m_predicates.end()) return found->second;
		const auto variable = std::min(m_nodes[left].variable, m_nodes[right].variable);
		uint32_t result;
		if (variable == UINT32_MAX) {
			result = Unknown(Type::U1);
		} else {
			const auto lhs = m_nodes[left];
			const auto rhs = m_nodes[right];
			const auto low = Compare(inst, lhs.variable == variable ? lhs.low : left,
			                         rhs.variable == variable ? rhs.low : right);
			const auto high = Compare(inst, lhs.variable == variable ? lhs.high : left,
			                          rhs.variable == variable ? rhs.high : right);
			result = Select(NodeFor(variable, 0u, 1u), high, low);
		}
		m_predicates.emplace(key, result);
		return result;
	}

	uint32_t Evaluate(Value value, bool current) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			if (value.GetType() == Type::U1) return value.U1() ? 1u : 0u;
			for (const auto& [literal, node]: m_literals) {
				if (literal == value) return node;
			}
			const auto node = Unknown(value.GetType());
			m_literals.emplace_back(value, node);
			return node;
		}
		const auto* inst = value.TryInstruction();
		if (current && m_exclude_bound && inst == &m_bound) return 0u;
		auto& values = m_values[current];
		if (const auto found = values.find(inst); found != values.end()) {
			if (found->second == UINT32_MAX) found->second = Unknown(value.GetType());
			return found->second;
		}
		const auto cached = values.emplace(inst, UINT32_MAX).first;
		auto result = UINT32_MAX;
		const auto arg = [&](uint32_t index) { return Evaluate(inst->Arg(index), current); };
		switch (inst->GetOpcode()) {
			case ValueOpcode::Phi: {
				const auto invariant = ResolveInvariantPhi(m_program, value);
				if (!invariant.IsEmpty()) {
					result = Evaluate(invariant, current);
				} else if (inst->Parent() == m_induction.Parent()) {
					if (current) {
						for (uint32_t i = 0; i < inst->NumArgs(); ++i) {
							if (inst->PhiBlock(i) == m_incoming) result = Evaluate(inst->Arg(i), false);
						}
					}
				} else if (inst->NumArgs() != 0u) {
					result = arg(0);
					for (uint32_t i = 1; i < inst->NumArgs(); ++i) {
						if (arg(i) != result) { result = UINT32_MAX; break; }
					}
				}
				break;
			}
			case ValueOpcode::LogicalNot: result = Select(arg(0), 0u, 1u); break;
			case ValueOpcode::LogicalAnd: {
				const auto left = arg(0);
				result = left == 0u ? 0u : Select(left, arg(1), 0u);
				break;
			}
			case ValueOpcode::LogicalOr: {
				const auto left = arg(0);
				result = left == 1u ? 1u : Select(left, 1u, arg(1));
				break;
			}
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectU32: {
				const auto condition = arg(0);
				result = condition == 0u ? arg(2) : condition == 1u ? arg(1)
				                                                   : Select(condition, arg(1), arg(2));
				break;
			}
			default:
				if (inst->GetType() == Type::U1 && inst->NumArgs() == 2u &&
				    inst->Arg(0).GetType() == Type::U32 && inst->Arg(1).GetType() == Type::U32)
					result = Compare(*inst, arg(0), arg(1));
				break;
		}
		// Only unsupported values and cycles need free variables.
		if (result == UINT32_MAX) {
			if (cached->second == UINT32_MAX) cached->second = Unknown(value.GetType());
			return cached->second;
		}
		cached->second = result;
		return result;
	}

	const Program& m_program;
	const Inst& m_induction;
	const Inst& m_bound;
	const Block* m_incoming = nullptr;
	uint32_t m_variables = 0;
	bool m_exclude_bound = true;
	std::vector<Node> m_nodes {{}, {}};
	std::vector<std::pair<Value, uint32_t>> m_literals;
	// Ordered: Evaluate keeps an iterator into its map across the recursive evaluation of the operands.
	std::array<std::map<const Inst*, uint32_t>, 2> m_values;
	std::unordered_map<std::array<uint32_t, 3>, uint32_t, ProofTripleHash> m_nodes_by_key;
	std::unordered_map<std::array<uint32_t, 3>, uint32_t, ProofTripleHash> m_choices;
	std::unordered_map<std::tuple<ValueOpcode, uint64_t, uint32_t, uint32_t>, uint32_t, ProofPredicateHash> m_predicates;
};

// ---- D4b step 1 (append windows): append buffers ------------------------------------------------------------
// A wave appends records to a pool: its first active lane adds the wave's lane count to a 4-byte counter (one buffer
// atomic, result returned), every lane takes record index = ReadFirstLane(old value) + its rank among the wave's
// active lanes, and stores whole records through the pool at that index. Each dispatch then writes only the records
// [counter before, counter after) of the pool, which the renderer reads back after the dispatch (bufferCache.cpp
// "append windows"). The proof below is structural; anything else leaves the pool untagged (whole binding, as before).
namespace AppendRange {

std::optional<uint32_t> ImmediateU32(Value value) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) return std::nullopt;
	return value.U32();
}

const Inst* InstOf(Value value, ValueOpcode op) {
	const auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == op ? inst : nullptr;
}

// x for BitFieldSExtract(x, 0, 24) (V_MUL_I32_I24 by 1: the values here are lane counts, at most 64), else the value.
Value StripSext24(Value value) {
	value = value.Resolve();
	if (const auto* inst = InstOf(value, ValueOpcode::BitFieldSExtract);
	    inst != nullptr && ImmediateU32(inst->Arg(1)) == 0u && ImmediateU32(inst->Arg(2)) == 24u) {
		return inst->Arg(0).Resolve();
	}
	return value;
}

bool IsLaneId(Value value) {
	return InstOf(value, ValueOpcode::LaneId) != nullptr;
}

// The value on the lanes of `mask`: predicated VGPR writes under that exec (SelectU32(mask, new, old)) give `new` there.
// Every use checked below is evaluated on lanes of `mask` only.
Value OnMask(Value value, Value mask) {
	value = value.Resolve();
	for (uint32_t depth = 0; depth < 8u && !mask.IsEmpty(); depth++) {
		const auto* sel = InstOf(value, ValueOpcode::SelectU32);
		if (sel == nullptr || !(sel->Arg(0).Resolve() == mask)) break;
		value = sel->Arg(1).Resolve();
	}
	return value;
}

// UGreaterThanEqual32(LaneId, 32): the lane is in the upper half of the wave.
bool IsHighLane(Value value) {
	const auto* inst = InstOf(value, ValueOpcode::UGreaterThanEqual32);
	return inst != nullptr && IsLaneId(inst->Arg(0)) && ImmediateU32(inst->Arg(1)) == 32u;
}

// BitFieldInsert(0, ~0, 0, LaneId & 31): the bits of the lanes below this one in its half.
bool IsLanesBelow(Value value) {
	const auto* inst = InstOf(value, ValueOpcode::BitFieldInsert);
	if (inst == nullptr || ImmediateU32(inst->Arg(0)) != 0u || ImmediateU32(inst->Arg(1)) != 0xffffffffu ||
	    ImmediateU32(inst->Arg(2)) != 0u) {
		return false;
	}
	const auto* local = InstOf(inst->Arg(3), ValueOpcode::BitwiseAnd32);
	return local != nullptr && ((IsLaneId(local->Arg(0)) && ImmediateU32(local->Arg(1)) == 31u) ||
	                            (IsLaneId(local->Arg(1)) && ImmediateU32(local->Arg(0)) == 31u));
}

// CompositeExtractU32x4(Ballot(mask), half).
bool IsBallotHalf(Value value, Value mask, uint32_t half) {
	const auto* extract = InstOf(value, ValueOpcode::CompositeExtractU32x4);
	if (extract == nullptr || ImmediateU32(extract->Arg(1)) != half) return false;
	const auto* ballot = InstOf(extract->Arg(0), ValueOpcode::Ballot);
	return ballot != nullptr && ballot->Arg(0).Resolve() == mask;
}

// BitCount32(BitwiseAnd32(half of Ballot(mask), lane mask)) with the lane mask V_MBCNT_LO/HI build for that half.
bool IsHalfRank(Value value, Value mask, uint32_t half) {
	const auto* count = InstOf(value, ValueOpcode::BitCount32);
	if (count == nullptr) return false;
	const auto* masked = InstOf(count->Arg(0), ValueOpcode::BitwiseAnd32);
	if (masked == nullptr) return false;
	for (int order = 0; order < 2; order++) {
		const auto bits = masked->Arg(order == 0 ? 0 : 1);
		const auto sel  = InstOf(masked->Arg(order == 0 ? 1 : 0), ValueOpcode::SelectU32);
		if (sel == nullptr || !IsBallotHalf(bits, mask, half) || !IsHighLane(sel->Arg(0))) continue;
		if (half == 0u && ImmediateU32(sel->Arg(1)) == 0xffffffffu && IsLanesBelow(sel->Arg(2))) return true;
		if (half == 1u && IsLanesBelow(sel->Arg(1)) && ImmediateU32(sel->Arg(2)) == 0u) return true;
	}
	return false;
}

// The lane's rank among the active lanes of `mask` (V_MBCNT_HI + V_MBCNT_LO over exec): lanes of `mask` below it.
bool IsRank(Value value, Value mask) {
	const auto* add = InstOf(OnMask(StripSext24(OnMask(value, mask)), mask), ValueOpcode::IAdd32);
	if (add == nullptr) return false;
	const auto a = OnMask(add->Arg(0), mask);
	const auto b = OnMask(add->Arg(1), mask);
	return (IsHalfRank(a, mask, 0u) && IsHalfRank(b, mask, 1u)) || (IsHalfRank(a, mask, 1u) && IsHalfRank(b, mask, 0u));
}

// The number of active lanes of `mask` (S_BCNT1_I32_B64 exec).
bool IsLaneCount(Value value, Value mask) {
	const auto* add = InstOf(OnMask(StripSext24(OnMask(value, mask)), mask), ValueOpcode::IAdd32);
	if (add == nullptr) return false;
	const auto half = [&](Value v, uint32_t h) {
		const auto* count = InstOf(OnMask(v, mask), ValueOpcode::BitCount32);
		return count != nullptr && IsBallotHalf(count->Arg(0), mask, h);
	};
	return (half(add->Arg(0), 0u) && half(add->Arg(1), 1u)) || (half(add->Arg(0), 1u) && half(add->Arg(1), 0u));
}

// `pred` is false on every lane where `mask` is false: `mask` itself or a LogicalAnd chain with it.
bool Implies(Value pred, Value mask, uint32_t depth = 0) {
	pred = pred.Resolve();
	if (pred == mask) return true;
	if (depth > 8u) return false;
	const auto* conj = InstOf(pred, ValueOpcode::LogicalAnd);
	return conj != nullptr && (Implies(conj->Arg(0), mask, depth + 1u) || Implies(conj->Arg(1), mask, depth + 1u));
}

struct Append {
	const Inst* atomic = nullptr; // BufferAtomicIAdd32 on the counter
	Value       mask;             // the wave's active lanes (E)
	Value       rank;             // the lane's rank among them (R)
};

// index = IAdd32(ReadFirstLane(V, E), R) with R the rank in E, V the atomic's result on the lane where R == 0, and the
// atomic run by exactly that lane (exec LogicalAnd(E, R == 0)) with data = lane count of E. V is that atomic's
// predicated result (SelectU32(Pa, atomic, any)), possibly through a two-arm phi whose other arm is the branch taken
// when Pa is empty: Pa holds E's first lane whenever E has one, so that arm only runs with E empty, and then no lane
// stores (every store's exec implies E).
std::optional<Append> MatchAppendIndex(Value index) {
	// The index register written under exec E: SelectU32(E, IAdd32(...), old) (the stores run on lanes of E only).
	Value outer;
	index = index.Resolve();
	if (const auto* sel = InstOf(index, ValueOpcode::SelectU32); sel != nullptr) {
		outer = sel->Arg(0).Resolve();
		index = sel->Arg(1).Resolve();
	}
	const auto* add = InstOf(index, ValueOpcode::IAdd32);
	if (add == nullptr) return std::nullopt;
	for (int order = 0; order < 2; order++) {
		const auto* first = InstOf(add->Arg(order == 0 ? 0 : 1), ValueOpcode::ReadFirstLane);
		if (first == nullptr) continue;
		const auto mask = first->Arg(1).Resolve();
		if (!outer.IsEmpty() && !(outer == mask)) continue;
		const auto rank = OnMask(add->Arg(order == 0 ? 1 : 0), mask);
		if (!IsRank(rank, mask)) continue;
		// The predicated result: SelectU32(Pa, atomic, other).
		const auto result_of = [&](Value v) -> const Inst* {
			const auto* sel = InstOf(v, ValueOpcode::SelectU32);
			if (sel == nullptr) return nullptr;
			const auto* atomic = InstOf(sel->Arg(1), ValueOpcode::BufferAtomicIAdd32);
			if (atomic == nullptr || atomic->NumArgs() < 6u || atomic->Arg(5).Resolve() != sel->Arg(0).Resolve()) {
				return nullptr;
			}
			return atomic;
		};
		const Inst* atomic = nullptr;
		const auto  value  = first->Arg(0).Resolve();
		if (const auto* phi = InstOf(value, ValueOpcode::Phi); phi != nullptr) {
			if (phi->NumArgs() != 2u) continue;
			const auto* a = result_of(phi->Arg(0));
			const auto* b = result_of(phi->Arg(1));
			if ((a == nullptr) == (b == nullptr)) continue; // exactly one arm carries the atomic
			atomic = a != nullptr ? a : b;
		} else {
			atomic = result_of(value);
		}
		if (atomic == nullptr) continue;
		// The atomic's exec: LogicalAnd(E, IEqual32(R, 0)) on the same rank value.
		const auto* exec = InstOf(atomic->Arg(5), ValueOpcode::LogicalAnd);
		if (exec == nullptr) continue;
		bool first_lane = false;
		for (int side = 0; side < 2 && !first_lane; side++) {
			const auto* eq = InstOf(exec->Arg(side == 0 ? 1 : 0), ValueOpcode::IEqual32);
			first_lane = exec->Arg(side == 0 ? 0 : 1).Resolve() == mask && eq != nullptr &&
			             ((OnMask(eq->Arg(0), mask) == rank && ImmediateU32(eq->Arg(1)) == 0u) ||
			              (OnMask(eq->Arg(1), mask) == rank && ImmediateU32(eq->Arg(0)) == 0u));
		}
		if (!first_lane) continue;
		// Index 0, offset 0, immediate soffset; data = E's lane count on that lane (directly or the selected arm).
		if (ImmediateU32(atomic->Arg(1)) != 0u || ImmediateU32(atomic->Arg(2)) != 0u || !ImmediateU32(atomic->Arg(3))) {
			continue;
		}
		auto data = atomic->Arg(4).Resolve();
		if (const auto* sel = InstOf(data, ValueOpcode::SelectU32);
		    sel != nullptr && sel->Arg(0).Resolve() == atomic->Arg(5).Resolve()) {
			data = sel->Arg(1).Resolve();
		}
		if (!IsLaneCount(data, mask)) continue;
		return Append {atomic, mask, rank};
	}
	return std::nullopt;
}

bool IsRawBufferStore(ValueOpcode op) {
	return op == ValueOpcode::StoreBufferU32 || op == ValueOpcode::StoreBufferU32x2 ||
	       op == ValueOpcode::StoreBufferU32x3 || op == ValueOpcode::StoreBufferU32x4;
}

// Tags each pool whose every write is such an append store (one counter, one index value) with its counter. Compute
// shaders only: the renderer copies the counter around dispatches (bufferCache.cpp "append windows").
void DetectAppendWrites(const Program& program, std::vector<BufferResource>& buffers) {
	if (program.stage != ShaderType::Compute) return;
	struct PoolState {
		bool        ok      = true;
		const Inst* atomic  = nullptr;
		Value       index;
		Value       mask;
		uint32_t    min_off = UINT32_MAX, max_end = 0;
	};
	std::vector<PoolState> pools(buffers.size());
	std::vector<uint32_t>  writes(buffers.size(), 0u);
	std::vector<const Inst*> stores;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) continue;
			const auto& memory = program.memory_info[inst.Flags<MemoryFlags>().index];
			if (memory.kind != ResourceKind::Buffer || memory.resource >= buffers.size()) continue;
			writes[memory.resource]++;
			if (access == BufferAccess::Write) stores.push_back(&inst);
			else pools[memory.resource].ok = false; // an atomic on a pool: not an append pool
		}
	}
	for (const auto* store: stores) {
		const auto& memory = program.memory_info[store->Flags<MemoryFlags>().index];
		auto&       pool   = pools[memory.resource];
		if (!pool.ok) continue;
		const auto soffset = ImmediateU32(store->Arg(3));
		if (!IsRawBufferStore(store->GetOpcode()) || memory.formatted || memory.typed || !soffset ||
		    ImmediateU32(store->Arg(2)) != 0u || store->NumArgs() < 6u) {
			pool.ok = false;
			continue;
		}
		const auto index = store->Arg(1).Resolve();
		if (pool.atomic == nullptr) {
			const auto append = MatchAppendIndex(index);
			if (!append) {
				pool.ok = false;
				continue;
			}
			pool.atomic = append->atomic;
			pool.index  = index;
			pool.mask   = append->mask;
		}
		// One index value for every store of the pool, and only lanes of the appending wave store.
		if (!(index == pool.index) || !Implies(store->Arg(store->NumArgs() - 1u), pool.mask)) {
			pool.ok = false;
			continue;
		}
		const uint64_t begin = uint64_t {*soffset} + memory.offset;
		const uint64_t end   = begin + 4u * BufferComponentCount(store->GetOpcode());
		if (end > UINT32_MAX) {
			pool.ok = false;
			continue;
		}
		pool.min_off = std::min(pool.min_off, static_cast<uint32_t>(begin));
		pool.max_end = std::max(pool.max_end, static_cast<uint32_t>(end));
	}
	for (uint32_t p = 0; p < buffers.size(); p++) {
		const auto& pool = pools[p];
		if (!pool.ok || pool.atomic == nullptr || writes[p] == 0u) continue;
		const auto& counter_memory = program.memory_info[pool.atomic->Flags<MemoryFlags>().index];
		const auto  c              = counter_memory.resource;
		if (counter_memory.kind != ResourceKind::Buffer || c >= buffers.size() || c == p) continue;
		// The counter: written only by this atomic, at a fixed 4-byte address.
		const auto& counter = buffers[c];
		if (writes[c] != 1u || !counter.fixed_writes || counter.fixed_write_ranges.size() != 1u ||
		    counter.fixed_write_ranges[0].second - counter.fixed_write_ranges[0].first != 4u) {
			continue;
		}
		buffers[p].append_counter    = c;
		buffers[p].append_min_offset = pool.min_off;
		buffers[p].append_max_end    = pool.max_end;
		std::fprintf(stderr, "APPEND_WRITES hash=0x%016llx pool=%u counter=%u record_bytes=[%u,%u)\n",
		             static_cast<unsigned long long>(program.shader_hash), p, c, pool.min_off, pool.max_end);
	}
}

} // namespace AppendRange

class Tracker {
public:
	Tracker(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg)
	    : m_program(program), m_decoded(decoded), m_native_cfg(native_cfg),
	      m_scalar_writes(std::move(program.scalar_writes)), m_info(program.info) {
		std::ranges::sort(m_scalar_writes, {}, &Program::ScalarWrite::pc);
		m_info.buffers.clear();
		m_info.images.clear();
		m_info.samplers.clear();
		m_info.sampled_pairs.clear();
		m_info.indirect_ranges.clear();
		m_info.uses_dma = false;
		m_shader_writes = HasShaderMemoryWrites(program);
	}

	// Record-keyed V#s: a V# that a bounded loop reads from the
	// records of a table V# (every dword ReadConstBuffer(table, key * stride + field), key = 0, 1, .. below a runtime
	// count) and that feeds raw BDA stores or atomics is tracked instead of making the shader write through unknown
	// addresses: the GPU keeps the V# words and the BDA accesses, and every materialization reads the count's
	// candidate V#s and lists the guest ranges their accesses can touch (ResourceSnapshot::indirect_ranges), which
	// the renderer makes resident, marks GPU-written and tests against captured scalar reads. Ghost of Yotei: the
	// probe flags of RTGI "Light Rays" 7ff41f74 (without it the emulator stopped with code 321 at the Snake's-house
	// cutscene) and of the probe evaluation shaders (a tree behind the torii gate is visible only with it).
	//
	// An S# that the shader reads from the records of
	// a table V# (four consecutive dwords ReadConstBuffer(table, key * stride + field), e.g. the decal record of the
	// Tiled Decals compute shaders or the region record of the terrain pixel shader) is GPU-selected among the S# of
	// every record inside the V#: the GPU keeps computing the record index, and every materialization reads each
	// record's S# exactly (RecordTableSampler). Without it such a sampler stays unresolved
	// and stops translation.

	static bool WorkaroundWaterfallPointerRecord0() {
		static const bool enabled = [] {
			// Default from Common::EnvDefaults; workaround: M42 (lane exchange /
			// terrain material pointer), to be replaced by the proper implementation; =0 turns it off.
			const char* value = Common::ShaderEnv::Read("KYTY_WORKAROUND_WATERFALL_POINTER_RECORD0");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}

	// A uniform copy of a lane-varying value whose only lane-varying input is the record index of
	// an idxen buffer load: the load reads record 0 instead (APPROXIMATION).
	Value UniformRecord0(Value value, Block& block, Block::iterator where, uint32_t depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate()) return value;
		auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 8u) return {};
		const auto op = inst->GetOpcode();
		if (Common::EnvRead::IndirectImage())
			std::fprintf(stderr, "TRACE_WATERFALL_POINTER depth=%u op=%s\n", depth,
			             ValueOpcodeName(op).data());
		if (op == ValueOpcode::Phi) {
			Value same;
			for (size_t i = 0; i < inst->NumArgs(); ++i) {
				const auto arg = inst->Arg(i).Resolve();
				if (arg.TryInstruction() == inst) continue;
				if (!same.IsEmpty() && !EquivalentValue(m_program, same, arg)) return {};
				same = arg;
			}
			return same.IsEmpty() ? Value {} : UniformRecord0(same, block, where, depth + 1u);
		}
		if (op == ValueOpcode::SelectU32 && inst->NumArgs() == 3u) {
			// An exec-masked write: the waterfall lane was active when the value was written.
			return UniformRecord0(inst->Arg(1), block, where, depth + 1u);
		}
		uint32_t component = 0;
		Inst* load = inst;
		if (op == ValueOpcode::CompositeExtractU32x2 || op == ValueOpcode::CompositeExtractU32x3 ||
		    op == ValueOpcode::CompositeExtractU32x4) {
			if (!ImmediateU32(inst->Arg(1), component)) return {};
			load = inst->Arg(0).Resolve().TryInstruction();
			if (load == nullptr) return {};
		}
		const auto load_op = load->GetOpcode();
		if (load_op == ValueOpcode::LoadBufferU32 || load_op == ValueOpcode::LoadBufferU32x2 ||
		    load_op == ValueOpcode::LoadBufferU32x3 || load_op == ValueOpcode::LoadBufferU32x4) {
			const auto flags = load->Flags<MemoryFlags>();
			if (flags.index >= m_program.memory_info.size()) return {};
			auto memory = m_program.memory_info[flags.index];
			if (memory.kind != ResourceKind::Buffer || !memory.idxen || memory.offen ||
			    memory.typed || memory.formatted || memory.data_bits != 32u ||
			    component >= memory.data_dwords)
				return {};
			memory.kind            = ResourceKind::ScalarBuffer;
			memory.idxen           = false;
			memory.offset         += component * sizeof(uint32_t);
			memory.data_dwords     = 1u;
			memory.component_index = component;
			memory.component_count = 1u;
			const auto index = static_cast<uint32_t>(m_program.memory_info.size());
			m_program.memory_info.push_back(memory);
			const MemoryFlags read_flags {.index = index, .pc = flags.pc};
			uint64_t packed = 0;
			std::memcpy(&packed, &read_flags, sizeof(packed));
			auto read = block.PrependNewInst(where, ValueOpcode::ReadConstBuffer,
			                                 {load->Arg(0), load->Arg(3)}, packed);
			return Value(&*read);
		}
		switch (op) {
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::IAdd32: break;
			default: return {};
		}
		std::array<Value, 3> args {};
		if (inst->NumArgs() > args.size()) return {};
		for (size_t i = 0; i < inst->NumArgs(); ++i) {
			args[i] = UniformRecord0(inst->Arg(i), block, where, depth + 1u);
			if (args[i].IsEmpty()) return {};
		}
		auto copy = inst->NumArgs() == 3u
		                ? block.PrependNewInst(where, op, {args[0], args[1], args[2]})
		                : block.PrependNewInst(where, op, {args[0], args[1]});
		return Value(&*copy);
	}

	// KYTY_WORKAROUND_WATERFALL_POINTER_RECORD0=1: logged APPROXIMATION. A scalar-load base address that a
	// waterfall reads with v_readfirstlane from an idxen buffer record (e.g. a per-tile material
	// table pointer) is replaced by the same field of record 0, so the descriptor table becomes a
	// snapshot source. Only the address operands of scalar loads change; the waterfall compares
	// keep the lane values, so the loop still terminates.
	// KYTY_DIAG_WATERFALL_RECORD0_CENSUS=1 (diagnostic): before each first-form replacement a
	// WaterfallRecord0Census compares the lane's address (the ReadFirstLane results) with record 0's
	// and counts on the GPU the lanes and those that differ (BindlessBindings.h).
	void UniformizeWaterfallPointers() {
		if (!WorkaroundWaterfallPointerRecord0()) return;
		uint32_t rewritten = 0;
		uint32_t census = 0;
		for (auto* block: m_program.blocks) {
			for (auto it = block->begin(); it != block->end(); ++it) {
				auto& inst = *it;
				if (inst.GetOpcode() != ValueOpcode::GetAddressResource || inst.NumArgs() != 2u)
					continue;
				std::array<Value, 2> args {};
				bool lane_read = false;
				Value lane_mask;
				for (uint32_t i = 0; i < 2u; ++i) {
					args[i] = inst.Arg(i).Resolve();
					const auto* first = args[i].TryInstruction();
					if (first == nullptr || first->GetOpcode() != ValueOpcode::ReadFirstLane) continue;
					if (lane_mask.IsEmpty()) lane_mask = first->Arg(1);
					args[i] = UniformRecord0(first->Arg(0), *block, it);
					if (args[i].IsEmpty()) break;
					lane_read = true;
				}
				if (!lane_read) continue;
				if (args[0].IsEmpty() || args[1].IsEmpty()) continue;
				if (WaterfallRecord0Census()) {
					block->PrependNewInst(it, ValueOpcode::WaterfallRecord0Census,
					                      {inst.Arg(0), inst.Arg(1), args[0], args[1], lane_mask});
					++census;
				}
				auto address = block->PrependNewInst(it, ValueOpcode::GetAddressResource,
				                                     {args[0], args[1]});
				const auto uses = inst.Uses();
				for (const auto& use: uses) {
					if (use.user->GetOpcode() == ValueOpcode::LoadAddressU32 && use.operand == 0u)
						use.user->SetArg(0, Value(&*address));
				}
				++rewritten;
			}
		}
		if (rewritten != 0u) {
			// KYTY_DIAG_WATERFALL_RECORD0_CENSUS: the line also gives the addresses counted (without it, today's line).
			char suffix[24] = "";
			if (WaterfallRecord0Census()) std::snprintf(suffix, sizeof(suffix), " census=%u", census);
			std::fprintf(stderr,
			             "WORKAROUND_WATERFALL_POINTER_RECORD0 APPROXIMATION hash=0x%016llx addresses=%u%s\n",
			             static_cast<unsigned long long>(m_program.shader_hash), rewritten, suffix);
		}
	}

	void Run() {
		if (m_program.resource_tracking_complete) {
			Fail(0, "resources already tracked");
		}
		UniformizeWaterfallPointers();
		FoldBoundedLoopSelectors();
		PlanScalarReads();
		EliminateDeadCode(m_program.blocks);
		PlanIndirectDescriptors();
		// KYTY_PERF_BINDLESS_CLASSES: a resource per table field can take a shader past its resource limits (MaxImages
		// and MaxSampledPairs end the translation). Such a shader is tracked again from the same state without field
		// resources, as with the switch off (the handle lookups of the first pass are memoized and come out the same).
		m_split_fields = BindlessClassesRequested();
		std::optional<SplitRetryState> retry;
		if (m_split_fields)
			retry.emplace(SplitRetryState {m_info, m_program.memory_info, m_program.has_address_writes,
			                               m_handle_patches, m_memory_patches});
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				Collect(inst);
			}
		}
		if (m_split_overflow && retry.has_value()) {
			auto& counters = GetBindlessClassCounters();
			counters.split_resources.fetch_sub(m_splits_made, std::memory_order_relaxed);
			counters.split_retries.fetch_add(1, std::memory_order_relaxed);
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 16u)
				std::fprintf(stderr,
				             "BINDLESS_CLASS split_retry hash=0x%016llx (%u field resources exceed the resource limits: "
				             "tracked again without them)\n",
				             static_cast<unsigned long long>(m_program.shader_hash), m_splits_made);
			m_info                       = std::move(retry->info);
			m_program.memory_info        = std::move(retry->memory_info);
			m_program.has_address_writes = retry->has_address_writes;
			m_handle_patches             = std::move(retry->handle_patches);
			m_memory_patches             = std::move(retry->memory_patches);
			m_split_fields               = false;
			m_split_overflow             = false;
			m_splits_made                = 0;
			for (auto* block: m_program.blocks) {
				for (auto& inst: *block) {
					Collect(inst);
				}
			}
		}
		DropRecordKeyedRanges();
		LinkImageAliases();
		for (const auto& patch: m_handle_patches) {
			patch.handle->SetFlags<uint32_t>(patch.resource);
		}
		for (const auto& patch: m_memory_patches) {
			auto& memory    = m_program.memory_info[patch.index];
			memory.resource = patch.resource;
			if (patch.has_sampler) {
				memory.sampler = patch.sampler;
			}
		}
		for (const auto& plan: m_indirect_descriptors) {
			// Record-keyed V#s: the GPU keeps reading the V# words; only the host lists their ranges.
			if (plan.range_only) continue;
			auto key = plan.key;
			if (plan.table_read != nullptr) {
				const auto* anchor = plan.table_indexed ? plan.handle : plan.table_read;
				auto* block = anchor->Parent();
				const auto where = std::ranges::find_if(block->Instructions(),
				    [&](const Inst& inst) { return &inst == anchor; });
				const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
					return Value(&*block->PrependNewInst(where, op, args));
				};
				const auto& read = *plan.table_read;
				const auto& memory = m_program.memory_info[read.Flags<MemoryFlags>().index];
				Value offset;
				Value valid(true);
				if (plan.table_indexed) {
					const auto* table = read.Arg(0).ResolveInstruction();
					const auto stride = emit(ValueOpcode::BitFieldUExtract,
					    {table->Arg(1), Value(16u), Value(14u)});
					const auto format = emit(ValueOpcode::BitFieldUExtract,
					    {table->Arg(3), Value(12u), Value(7u)});
					offset = emit(ValueOpcode::IAdd32,
					    {emit(ValueOpcode::IMul32, {read.Arg(1), stride}), Value(memory.offset)});
					valid = emit(ValueOpcode::LogicalAnd,
					    {emit(ValueOpcode::ULessThan32, {read.Arg(1), table->Arg(2)}),
					     emit(ValueOpcode::INotEqual32, {format, Value(0u)})});
				} else {
					const auto dynamic = emit(ValueOpcode::BitwiseAnd32, {read.Arg(1), Value(~3u)});
					offset = emit(ValueOpcode::IAdd32, {dynamic, Value(memory.offset & ~3u)});
					valid = emit(ValueOpcode::ULessThanEqual32, {dynamic, offset});
				}
				key = emit(ValueOpcode::SelectU32, {valid, offset, Value(UINT32_MAX)});
				if (!plan.table_indexed) {
					// NativeDescriptorSource proved this read reaches the image. Preserve its
					// SSA Phi edges while retaining descriptor DWORDs for other consumers.
					std::map<const Inst*, Value> projected;
					const auto project = [&](auto&& self, Value value) -> Value {
						value = value.Resolve();
						const auto* phi = value.TryInstruction();
						if (phi == plan.table_read) return key;
						if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi) return Value(UINT32_MAX);
						if (const auto found = projected.find(phi); found != projected.end()) return found->second;
						auto* parent = phi->Parent();
						const auto position = std::ranges::find_if(parent->Instructions(),
						    [&](const Inst& inst) { return &inst == phi; });
						auto& selected = *parent->PrependNewInst(position, ValueOpcode::Phi);
						selected.SetFlags(Type::U32);
						projected.emplace(phi, Value(&selected));
						for (uint32_t arm = 0; arm < phi->NumArgs(); ++arm)
							selected.AddPhiOperand(phi->PhiBlock(arm), self(self, phi->Arg(arm)));
						return Value(&selected);
					};
					key = project(project, plan.handle->Arg(0));
				}
				plan.handle->SetArg(0, key);
				for (uint32_t word = 1; word < plan.handle->NumArgs(); ++word)
					plan.handle->SetArg(word, Value(0u));
				continue;
			}
			if (plan.handle->NumArgs() == 4u) {
				plan.handle->SetArg(0, plan.key);
				for (uint32_t word = 1; word < 4u; ++word) plan.handle->SetArg(word, Value(0u));
			} else {
				if (plan.reads[0] == nullptr) plan.handle->SetArg(0, key);
				for (uint32_t dword = 0; dword < 4u; dword++) {
					plan.handle->SetArg(dword + 1u, plan.roots[dword + 4u]);
				}
				for (uint32_t dword = 5u; dword < plan.roots.size(); dword++) {
					plan.handle->SetArg(dword, plan.key);
				}
			}
			if (plan.reads[0] != nullptr) {
				for (uint32_t word = 0; word < plan.handle->NumArgs(); ++word)
					if (plan.reads[word] != nullptr)
						m_program.memory_info[plan.memory[word]].planning_only = true;
			}
		}
		// Project descriptor-only SSA onto its key. Existing Phis preserve dominance;
		// native descriptor resolution excludes their stale incoming values at each use.
		for (const auto& plan: m_indirect_descriptors) {
			if ((plan.handle->GetOpcode() != ValueOpcode::GetImageResource &&
			     plan.handle->GetOpcode() != ValueOpcode::GetSamplerResource) ||
			    plan.reads[0] == nullptr)
				continue;
			for (const auto* read: plan.reads) {
				if (read == nullptr) continue; // R128 words 4..7 are constant zero, not table reads.
				const auto first = std::ranges::find_if(m_indirect_descriptors, [&](const auto& candidate) {
					return candidate.reads[0] == read;
				});
				const auto key = first != m_indirect_descriptors.end() ? first->key : Value(0u);
				const auto uses = read->Uses();
				for (const auto& use: uses) use.user->SetArg(use.operand, key);
			}
		}
		AppendRange::DetectAppendWrites(m_program, m_info.buffers); // D4b step 1 (append windows)
		m_program.descriptor_sources         = std::move(m_sources);
		m_program.info                       = std::move(m_info);
		m_program.resource_tracking_complete = true;
	}

private:
	struct HandlePatch {
		Inst*    handle   = nullptr;
		uint32_t resource = 0;
	};

	struct MemoryPatch {
		uint32_t index       = 0;
		uint32_t resource    = 0;
		uint32_t sampler     = 0;
		bool     has_sampler = false;
	};

	// KYTY_PERF_BINDLESS_CLASSES: what Collect changes, saved before it runs (Run tracks a shader again without field
	// resources when they exceed the resource limits).
	struct SplitRetryState {
		ShaderInfo               info;
		std::vector<MemoryInfo>  memory_info;
		bool                     has_address_writes = false;
		std::vector<HandlePatch> handle_patches;
		std::vector<MemoryPatch> memory_patches;
	};

	struct MemoryIndexUse {
		const Inst* owner = nullptr;
		uint32_t    count = 0;
	};

	struct ResolvedHandle {
		const Inst* handle;
		uint32_t pc;
		DescriptorSource source;
		Value planning_handle;
	};

	struct IndirectDescriptorPlan {
		Inst*                      handle = nullptr;
		uint32_t                   source = 0;
		Value                      key;
		std::array<Value, 8>       roots {};
		std::array<uint32_t, 8>    memory {};
		std::array<const Inst*, 8> reads {};
		bool split_offsets = false;
		bool retain_reads = false;
		bool table_indexed = false;
		// Record-keyed V#s: the handle keeps its V# words; the source only lists
		// the candidates whose guest ranges the host makes known.
		bool range_only = false;
		const Inst* table_read = nullptr;
		// Two-level table: the descriptor's base address is a pointer read from buffer records.
		bool        pointer_record        = false;
		const Inst* pointer_read          = nullptr; // the low-dword ReadConstBuffer of the pointer
		uint32_t    pointer_stride        = 0;
		uint32_t    pointer_record_offset = 0;
		// The bindless tables: for a bounded heap table (table_read set), the record field the key points at:
		// record stride in bytes (0: the V#'s own stride) and byte offset of the descriptor in the record.
		// field_known is false when the key is not of the form record * stride + offset.
		bool     field_known  = false;
		uint32_t field_stride = 0;
		uint32_t field_offset = 0;
	};

	// Pointer record tables: a descriptor read from the address held in a buffer record that a lane value selects (lo/hi =
	// ReadConstBuffer(V#, key * stride + o [+ 4])) becomes a GPU-selected descriptor with every record of the buffer as a
	// candidate.
	bool MatchPointerRecord(const Inst& address, const Inst*& lo_read, Value& key, uint32_t& stride,
	                        uint32_t& record_offset) const {
		if (address.GetOpcode() != ValueOpcode::GetAddressResource || address.NumArgs() != 2u) return false;
		const auto* lo = address.Arg(0).Resolve().TryInstruction();
		const auto* hi = address.Arg(1).Resolve().TryInstruction();
		if (lo == nullptr || hi == nullptr || lo->GetOpcode() != ValueOpcode::ReadConstBuffer ||
		    hi->GetOpcode() != ValueOpcode::ReadConstBuffer || lo->NumArgs() != 2u || hi->NumArgs() != 2u ||
		    !EquivalentValue(m_program, lo->Arg(0), hi->Arg(0)))
			return false;
		Value    key_lo, key_hi;
		uint32_t offset_lo = 0, offset_hi = 0, stride_lo = 0, stride_hi = 0;
		if (!MatchTableOffset(lo->Arg(1), key_lo, offset_lo, stride_lo) ||
		    !MatchTableOffset(hi->Arg(1), key_hi, offset_hi, stride_hi) || stride_lo != stride_hi ||
		    !EquivalentValue(m_program, key_lo, key_hi))
			return false;
		const auto index_lo = lo->Flags<MemoryFlags>().index;
		const auto index_hi = hi->Flags<MemoryFlags>().index;
		if (index_lo >= m_program.memory_info.size() || index_hi >= m_program.memory_info.size()) return false;
		const auto lo_offset = uint64_t {offset_lo} + m_program.memory_info[index_lo].offset;
		const auto hi_offset = uint64_t {offset_hi} + m_program.memory_info[index_hi].offset;
		if (hi_offset != lo_offset + 4u || (lo_offset & 3u) != 0u || lo_offset > UINT32_MAX) return false;
		lo_read       = lo;
		key           = key_lo;
		stride        = stride_lo;
		record_offset = static_cast<uint32_t>(lo_offset);
		return true;
	}

	[[noreturn]] void Fail(uint32_t pc, const std::string& reason) const {
		const auto message =
		    fmt::format("shader resource tracking: hash=0x{:016x} stage={} pc=0x{:08x} {}",
		                m_program.shader_hash, StageName(m_program.stage), pc, reason);
		EXIT("%s", message.c_str());
		std::abort();
	}

	Value NativeDescriptorSource(Value value, uint32_t reg, uint32_t use_pc) const {
		value = value.Resolve();
		const auto* phi = value.TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || m_native_cfg.blocks.empty())
			return value;
		std::vector<const Inst*> candidates;
		std::vector<const Inst*> visited;
		std::vector<const Inst*> pending {phi};
		while (!pending.empty()) {
			const auto* inst = pending.back();
			pending.pop_back();
			if (std::ranges::find(visited, inst) != visited.end()) continue;
			visited.push_back(inst);
			if (inst->GetOpcode() == ValueOpcode::Phi) {
				for (size_t i = 0; i < inst->NumArgs(); ++i) {
					const auto* arg = inst->Arg(i).Resolve().TryInstruction();
					if (arg != nullptr) pending.push_back(arg);
				}
			} else if (inst->GetOpcode() == ValueOpcode::ReadConst ||
			           inst->GetOpcode() == ValueOpcode::LoadAddressU32 ||
			           inst->GetOpcode() == ValueOpcode::ReadConstBuffer ||
			           inst->GetOpcode() == ValueOpcode::GetUserData) {
				candidates.push_back(inst);
			}
		}
		if (candidates.empty()) return value;
		const auto source_at = [&](uint32_t pc) {
			Value source;
			const auto native = std::ranges::lower_bound(m_decoded.instructions, pc, {},
			                                            &Decoder::Instruction::pc);
			for (const auto* candidate: candidates) {
				if (pc == UINT32_MAX) {
					if (candidate->GetOpcode() != ValueOpcode::GetUserData ||
					    RegIndex(candidate->Arg(0).ScalarRegister()) != reg)
						continue;
				} else {
					if (candidate->GetOpcode() != ValueOpcode::ReadConst &&
					    candidate->GetOpcode() != ValueOpcode::LoadAddressU32 &&
					    candidate->GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
					const auto flags = candidate->Flags<MemoryFlags>();
					if (flags.pc != pc || flags.index >= m_program.memory_info.size()) continue;
					if (native == m_decoded.instructions.end() || native->pc != pc ||
					    native->dst.kind != Decoder::OperandKind::Sgpr ||
					    native->dst.reg + m_program.memory_info[flags.index].component_index != reg)
						continue;
				}
				const Value current(const_cast<Inst*>(candidate));
				if (!source.IsEmpty() && !EquivalentValue(m_program, source, current)) return Value {};
				source = current;
			}
			return source;
		};
		const auto use = std::ranges::find_if(m_native_cfg.blocks, [&](const auto& block) {
			return block.start_pc <= use_pc && use_pc < block.end_pc;
		});
		if (use == m_native_cfg.blocks.end()) return value;
		struct Position { uint32_t block; uint32_t before; };
		std::vector<Position> positions {{use->id, use_pc}};
		std::vector<bool> reached(m_native_cfg.blocks.size());
		Value selected;
		uint32_t selected_pc = UINT32_MAX;
		const auto select = [&](uint32_t pc) {
			const auto source = source_at(pc);
			if (source.IsEmpty()) return false;
			if (!selected.IsEmpty()) {
				const auto op = source.TryInstruction()->GetOpcode();
				if ((op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer) &&
				    selected_pc != pc) return false;
				if (!EquivalentValue(m_program, selected, source)) return false;
			}
			selected = source;
			selected_pc = pc;
			return true;
		};
		while (!positions.empty()) {
			const auto position = positions.back();
			positions.pop_back();
			const auto& block = m_native_cfg.blocks[position.block];
			// A backedge may revisit the use block after a write later than the original use.
			if (position.before == block.end_pc) {
				if (reached[block.id]) continue;
				reached[block.id] = true;
			}
			auto write = std::ranges::lower_bound(m_scalar_writes, position.before, {},
			                                    &Program::ScalarWrite::pc);
			bool found = false;
			while (write != m_scalar_writes.begin()) {
				--write;
				if (write->pc < block.start_pc) break;
				if (RegIndex(write->reg) != reg) continue;
				if (!select(write->pc)) return value;
				found = true;
				break;
			}
			if (found) continue;
			if (block.id == m_native_cfg.entry_block && !select(UINT32_MAX)) return value;
			for (const auto pred: block.predecessors)
				positions.push_back({pred, m_native_cfg.blocks[pred].end_pc});
		}
		return selected.IsEmpty() ? value : selected;
	}

	Value LowerDescriptorValue(Value value) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return value;
		// The scalar planner owns reads and lowers their address handles separately.
		uint32_t memory_index = 0;
		if (ScalarReadMemory(*inst, memory_index) != nullptr) return value;
		const auto [cached, inserted] = m_descriptor_values.emplace(inst, value);
		if (!inserted) return cached->second;
		if (inst->GetOpcode() != ValueOpcode::Phi) {
			Inst* projected = nullptr;
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
				const auto lowered = LowerDescriptorValue(inst->Arg(arg));
				if (lowered == inst->Arg(arg).Resolve()) continue;
				if (projected == nullptr) {
					projected = &m_program.value_storage.emplace_back(inst->GetOpcode(), inst->Flags<uint64_t>());
					for (size_t i = 0; i < inst->NumArgs(); ++i) projected->SetArg(i, inst->Arg(i));
				}
				projected->SetArg(arg, lowered);
			}
			if (projected != nullptr) cached->second = Value(projected);
			return cached->second;
		}
		if (inst->NumArgs() != 2u || inst->NumPhiBlocks() != 2u || inst->GetType() != Type::U32) {
			return value;
		}
		const auto* merge = inst->Parent();
		const auto* branch = inst->PhiBlock(0);
		if (merge == nullptr || branch == nullptr || inst->PhiBlock(1) == nullptr ||
		    branch == inst->PhiBlock(1)) {
			return value;
		}
		if (branch->ImmSuccessors().size() != 2u) {
			if (branch->ImmPredecessors().size() != 1u) {
				return value;
			}
			branch = branch->ImmPredecessors()[0];
		}
		if (branch == merge || branch->ImmSuccessors().size() != 2u) {
			return value;
		}
		std::array<const Block*, 2> targets;
		for (uint32_t arm = 0; arm < 2; arm++) {
			const auto* incoming = inst->PhiBlock(arm);
			if (incoming == merge ||
			    (incoming != branch &&
			     (incoming->ImmPredecessors().size() != 1u ||
			      incoming->ImmPredecessors()[0] != branch ||
			      incoming->ImmSuccessors().size() != 1u ||
			      incoming->ImmSuccessors()[0] != merge))) {
				return value;
			}
			targets[arm] = incoming == branch ? merge : incoming;
		}
		const auto& info = *branch;
		const auto& term = info.terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    !((term.true_block == targets[0] && term.false_block == targets[1]) ||
		      (term.false_block == targets[0] && term.true_block == targets[1])) ||
		    !ValidateRuntimeValue(m_program, info.condition, m_shader_writes
		        ? RuntimeValueType::ImmutableInteger : RuntimeValueType::Integer) ||
		    !ValidateRuntimeValue(m_program, inst->Arg(0)) ||
		    !ValidateRuntimeValue(m_program, inst->Arg(1))) {
			return value;
		}
		// Retain a host expression; replacing the GPU Phi would break SSA dominance.
		const auto true_arg = term.true_block == targets[0] ? 0u : 1u;
		auto&      selected = m_program.value_storage.emplace_back(ValueOpcode::SelectU32);
		selected.SetArg(0, info.condition);
		selected.SetArg(1, inst->Arg(true_arg));
		selected.SetArg(2, inst->Arg(true_arg ^ 1u));
		cached->second = Value(&selected);
		return cached->second;
	}

	void MakeSource(const Inst& handle, uint32_t width, bool sampler, bool sample_adjust,
	                uint32_t base_reg, DescriptorSource& descriptor, uint32_t pc) {
		const auto resolved = std::ranges::find_if(m_resolved_handles, [&](const auto& entry) {
			return entry.handle == &handle && entry.pc == pc;
		});
		if (resolved != m_resolved_handles.end()) {
			descriptor = resolved->source;
			return;
		}
		if (handle.NumArgs() != width) {
			Fail(pc, fmt::format("{} has {} descriptor dwords, expected {}",
			                     ValueOpcodeName(handle.GetOpcode()), handle.NumArgs(), width));
		}
		descriptor.dword_count = width;
		for (uint32_t i = 0; i < width; i++) {
			const auto value = base_reg != UINT32_MAX
			    ? NativeDescriptorSource(handle.Arg(i), base_reg + i, pc) : handle.Arg(i);
			descriptor.dwords[i] = LowerDescriptorValue(value);
		}
		if (sample_adjust) {
			descriptor.dwords[3] = CanonicalizeSampleAdjustDword3(descriptor.dwords[3]);
		}
		const auto dword0 = descriptor.dwords[0].Resolve();
		if (sampler && dword0.IsImmediate() && dword0.GetType() == Type::U32 &&
		    (dword0.U32() & SamplerBorderClampMask) == 0) {
			// Border color and its table index are unused unless a clamp axis selects border mode.
			descriptor.dwords[3] = Value(0u);
		}
		m_resolved_handles.push_back({&handle, pc, descriptor, {}});
	}

	uint32_t ScalarReadBase(const Inst& read) const {
		const auto flags = read.Flags<MemoryFlags>();
		if (m_program.memory_info[flags.index].kind == ResourceKind::ScalarBuffer)
			return m_program.memory_info[flags.index].resource * 4u;
		const auto native = std::ranges::lower_bound(m_decoded.instructions, flags.pc, {},
		                                            &Decoder::Instruction::pc);
		return native != m_decoded.instructions.end() && native->pc == flags.pc &&
		               native->src0.kind == Decoder::OperandKind::Sgpr
		           ? native->src0.reg : UINT32_MAX;
	}

	void CollectScalarRead(Value value, uint32_t use_pc) {
		value = value.Resolve();
		if (value.IsImmediate()) return;
		auto* inst = value.TryInstruction();
		if (inst == nullptr) Fail(use_pc, "invalid typed planning value");
		if (m_srt_visited.contains(inst)) return;
		const auto cycle = std::ranges::find(m_srt_visiting, inst);
		if (cycle != m_srt_visiting.end()) {
			if (std::any_of(cycle, m_srt_visiting.end(), [](const Inst* value) {
				return value->GetOpcode() == ValueOpcode::Phi;
			})) return;
			Fail(use_pc, "cyclic typed planning value without a phi");
		}
		m_srt_visiting.push_back(inst);
		uint32_t memory_index = 0;
		const auto* memory = ScalarReadMemory(*inst, memory_index);
		DescriptorSource source;
		if (memory != nullptr) {
			const auto* handle = inst->Arg(0).Resolve().TryInstruction();
			const auto width = memory->kind == ResourceKind::ScalarBuffer ? 4u : 2u;
			if (handle == nullptr || handle->GetOpcode() !=
			        (width == 4u ? ValueOpcode::GetBufferResource : ValueOpcode::GetAddressResource))
				Fail(use_pc, "scalar read has an invalid resource handle");
			MakeSource(*handle, width, false, false, ScalarReadBase(*inst), source,
			           inst->Flags<MemoryFlags>().pc);
			for (uint32_t word = 0; word < width; ++word)
				CollectScalarRead(source.dwords[word], inst->Flags<MemoryFlags>().pc);
			for (size_t arg = 1; arg < inst->NumArgs(); ++arg)
				CollectScalarRead(inst->Arg(arg), use_pc);
		} else {
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg)
				CollectScalarRead(inst->Arg(arg), use_pc);
		}
		m_srt_visiting.pop_back();
		m_srt_visited.insert(inst);
		if (memory == nullptr) return;
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return;
		m_scalar_reads.push_back(inst);
	}

	void PlanScalarReads() {
		m_program.srt_plan_complete = false;
		m_program.srt_reads.clear();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				const auto image = ImageOpcodeInfoOf(op);
				if (BufferAccessOf(op) == BufferAccess::None &&
				    AddressOpcodeInfoOf(op).access == AddressAccess::None &&
				    image.access == ImageAccess::None) continue;
				const auto flags = inst.Flags<MemoryFlags>();
				if (flags.index >= m_program.memory_info.size())
					Fail(flags.pc, "memory metadata index is out of range");
				if (inst.NumArgs() < (image.needs_sampler ? 2u : 1u))
					Fail(flags.pc, "memory operation has no resource handle");
				const auto& memory = m_program.memory_info[flags.index];
				if ((op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarBuffer) ||
				    (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarAddress))
					Fail(flags.pc, "scalar read has incompatible scalar memory metadata");
				for (uint32_t arg = 0; arg < (image.needs_sampler ? 2u : 1u); ++arg) {
					const auto* handle = inst.Arg(arg).Resolve().TryInstruction();
					if (handle == nullptr) continue;
					const auto kind = handle->GetOpcode();
					const bool sampler = kind == ValueOpcode::GetSamplerResource;
					const uint32_t width = kind == ValueOpcode::GetImageResource ? 8u
					                     : kind == ValueOpcode::GetAddressResource ? 2u
					                     : kind == ValueOpcode::GetBufferResource || sampler ? 4u : 0u;
					if (width == 0u) continue;
					const auto base = width == 2u
					    ? (memory.kind == ResourceKind::ScalarAddress ? ScalarReadBase(inst) : UINT32_MAX)
					    : (sampler ? memory.sampler : memory.resource) * 4u;
					DescriptorSource source;
					MakeSource(*handle, width, sampler,
					           sampler && (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0,
					           base, source, flags.pc);
					for (uint32_t word = 0; word < width; ++word)
						CollectScalarRead(source.dwords[word], flags.pc);
				}
			}
		}
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				uint32_t index = 0;
				if (inst.GetOpcode() == ValueOpcode::LoadAddressU32 &&
				    ScalarReadMemory(inst, index) != nullptr && inst.Arg(1).Resolve().IsImmediate() &&
				    ValidateRuntimeValue(m_program, Value(&inst)))
					CollectScalarRead(Value(&inst), inst.Flags<MemoryFlags>().pc);
			}
		}
		for (auto* read: m_scalar_reads) {
			const auto flags = read->Flags<MemoryFlags>();
			auto& memory = m_program.memory_info[flags.index];
			const auto* handle = read->Arg(0).Resolve().TryInstruction();
			auto resolved = std::ranges::find_if(m_resolved_handles, [&](const auto& entry) {
				return entry.handle == handle && entry.pc == flags.pc;
			});
			EXIT_IF(resolved == m_resolved_handles.end());
			uint32_t bad_dword = 0;
			if (!ValidateSource(resolved->source, bad_dword)) continue;
			// A descriptor word that also feeds shader arithmetic, at an offset only the GPU
			// knows (e.g. a waterfall key), cannot be snapshotted: it stays a runtime load.
			if (!ValidateRuntimeValue(m_program, Value(read)) && !UsesOnlyDescriptorHandles(*read)) continue;
			memory.planning_only = true;
			uint32_t slot = 0;
			for (; slot < m_program.srt_reads.size(); ++slot) {
				const auto* other = m_program.srt_reads[slot].value.Resolve().TryInstruction();
				if (other->GetOpcode() != read->GetOpcode() ||
				    m_program.memory_info[other->Flags<MemoryFlags>().index] != memory) continue;
				const auto* other_handle = other->Arg(0).Resolve().TryInstruction();
				bool same = true;
				for (uint32_t word = 0; word < resolved->source.dword_count; ++word)
					same &= EquivalentValue(m_program, resolved->source.dwords[word], other_handle->Arg(word));
				for (size_t arg = 1; arg < read->NumArgs(); ++arg)
					same &= EquivalentValue(m_program, read->Arg(arg), other->Arg(arg));
				if (same) break;
			}
			const bool keep = slot == m_program.srt_reads.size();
			if (keep) m_program.srt_reads.push_back({Value(read), slot});
			auto* block = read->Parent();
			auto& list = block->Instructions();
			const auto where = std::ranges::find_if(list, [&](const Inst& inst) {
				return &inst == read;
			});
			const auto resource = Value(&*block->PrependNewInst(where, ValueOpcode::GetSrtResource));
			const auto flat = Value(&*block->PrependNewInst(where, ValueOpcode::ReadConst,
			    {resource, Value(slot)}, read->Flags<uint64_t>()));
			const auto uses = read->Uses();
			for (const auto& use: uses) use.user->SetArg(use.operand, flat);
			for (auto& entry: m_resolved_handles) {
				for (uint32_t word = 0; word < entry.source.dword_count; ++word)
					if (entry.source.dwords[word].Resolve() == Value(read))
						entry.source.dwords[word] = flat;
			}
			for (auto* block: m_program.blocks) {
				if (block->condition.Resolve() == Value(read)) block->condition = flat;
				if (block->indirect_target.Resolve() == Value(read)) block->indirect_target = flat;
			}
			if (keep) {
				if (resolved->planning_handle.IsEmpty()) {
					bool unchanged = true;
					for (uint32_t word = 0; word < resolved->source.dword_count; ++word)
						unchanged &= handle->Arg(word).Resolve() == resolved->source.dwords[word].Resolve();
					resolved->planning_handle = read->Arg(0);
					if (!unchanged) {
						auto& retained = m_program.value_storage.emplace_back(handle->GetOpcode());
						for (uint32_t word = 0; word < resolved->source.dword_count; ++word)
							retained.SetArg(word, resolved->source.dwords[word]);
						resolved->planning_handle = Value(&retained);
					}
				}
				read->SetArg(0, resolved->planning_handle);
				read->SetParent(nullptr);
				m_program.value_storage.splice(m_program.value_storage.end(), list, where);
			} else {
				list.erase(where);
			}
		}
		m_program.srt_plan_complete = true;
	}

	bool ValidateSource(const DescriptorSource& descriptor, uint32_t& bad_dword) const {
		for (uint32_t i = 0; i < descriptor.dword_count; i++) {
			bad_dword = i;
			if (descriptor.dwords[i].Resolve().GetType() != Type::U32) {
				return false;
			}
			if (!ValidateRuntimeValue(m_program, descriptor.dwords[i])) {
				return false;
			}
		}
		return true;
	}

	uint32_t InternSource(const DescriptorSource& descriptor) {
		for (uint32_t candidate = 0; candidate < m_sources.size(); candidate++) {
			auto& current = m_sources[candidate];
			if (current.dword_count != descriptor.dword_count ||
			    current.indirect_descriptor.has_value() != descriptor.indirect_descriptor.has_value()) {
				continue;
			}
			if (current.indirect_descriptor.has_value()) {
				const auto& a = *current.indirect_descriptor;
				const auto& b = *descriptor.indirect_descriptor;
				if (a.selector != b.selector || a.table_source != b.table_source ||
				    a.table_offset != b.table_offset || a.table_immediate != b.table_immediate ||
				    a.table_stride != b.table_stride || a.record_keyed != b.record_keyed ||
				    a.unsigned_bound != b.unsigned_bound || a.all_records != b.all_records || a.widened != b.widened ||
				    a.count_pointer != b.count_pointer || a.count_field != b.count_field || a.count_check != b.count_check ||
				    a.count_limit != b.count_limit || a.composed != b.composed || a.pointer_offset != b.pointer_offset ||
				    a.key_count.IsEmpty() != b.key_count.IsEmpty() ||
				    a.workgroup_axis != b.workgroup_axis || a.sources != b.sources ||
				    !EquivalentValue(m_program, a.key_count, b.key_count) ||
				    a.selector_first.IsEmpty() != b.selector_first.IsEmpty() ||
				    (!a.selector_first.IsEmpty() &&
				     !EquivalentValue(m_program, a.selector_first, b.selector_first)) ||
				    a.selector_mask.IsEmpty() != b.selector_mask.IsEmpty() ||
				    (!a.selector_mask.IsEmpty() &&
				     !EquivalentValue(m_program, a.selector_mask, b.selector_mask))) continue;
			}
			bool same = true;
			for (uint32_t i = 0; i < descriptor.dword_count; i++) {
				same = same && EquivalentValue(m_program, current.dwords[i], descriptor.dwords[i]);
			}
			if (same) {
				if (current.indirect_descriptor) {
					current.indirect_descriptor->table_scalar |= descriptor.indirect_descriptor->table_scalar;
					current.indirect_descriptor->table_record_bytes = std::max(
					    current.indirect_descriptor->table_record_bytes,
					    descriptor.indirect_descriptor->table_record_bytes);
				}
				return candidate;
			}
		}
		m_sources.push_back(descriptor);
		return static_cast<uint32_t>(m_sources.size() - 1);
	}

	static bool ImmediateU32(Value value, uint32_t& result) {
		value = value.Resolve();
		if (!value.IsImmediate() || value.GetType() != Type::U32) {
			return false;
		}
		result = value.U32();
		return true;
	}

	// Iterative dominators over the value CFG (entry = first block). An unbounded scalar-address
	// table read may serve a handle in another block only when every path to it passes the read.
	bool BlockDominates(const Block* dominator, const Block* block) const {
		if (dominator == block) return true;
		const auto& blocks = m_program.blocks;
		if (blocks.empty()) return false;
		if (m_dominators.size() != blocks.size()) {
			std::unordered_map<const Block*, size_t> index;
			for (size_t i = 0; i < blocks.size(); ++i) index.emplace(blocks[i], i);
			const size_t n = blocks.size();
			m_dominators.assign(n, std::vector<uint8_t>(n, 1u));
			m_dominators[0].assign(n, 0u);
			m_dominators[0][0] = 1u;
			for (bool changed = true; changed;) {
				changed = false;
				for (size_t i = 1; i < n; ++i) {
					std::vector<uint8_t> next(n, 1u);
					bool any = false;
					for (const auto* pred: blocks[i]->ImmPredecessors()) {
						const auto it = index.find(pred);
						if (it == index.end()) continue;
						any = true;
						for (size_t k = 0; k < n; ++k) next[k] &= m_dominators[it->second][k];
					}
					if (!any) next.assign(n, 0u);
					next[i] = 1u;
					if (next != m_dominators[i]) {
						m_dominators[i] = std::move(next);
						changed = true;
					}
				}
			}
			m_dominator_index = std::move(index);
		}
		const auto a = m_dominator_index.find(dominator);
		const auto b = m_dominator_index.find(block);
		return a != m_dominator_index.end() && b != m_dominator_index.end() &&
		       m_dominators[b->second][a->second] != 0u;
	}

	bool R128ImageHandle(const Inst& handle) const {
		if (handle.Uses().empty()) return false;
		for (const auto& use: handle.Uses()) {
			const auto* user = use.user;
			if (ImageOpcodeInfoOf(user->GetOpcode()).access == ImageAccess::None || use.operand != 0u)
				return false;
			const auto index = user->Flags<MemoryFlags>().index;
			if (index >= m_program.memory_info.size() || !m_program.memory_info[index].image_r128)
				return false;
		}
		return true;
	}

	static bool UsesOnlyDescriptorHandles(const Inst& value) {
		if (value.Uses().empty()) return false;
		std::vector<const Inst*> values {&value};
		for (size_t index = 0; index < values.size(); ++index) {
			for (const auto& use: values[index]->Uses()) {
				const auto op = use.user->GetOpcode();
				if (op == ValueOpcode::GetImageResource || op == ValueOpcode::GetSamplerResource ||
				    op == ValueOpcode::GetBufferResource || op == ValueOpcode::GetAddressResource) continue;
				if (op != ValueOpcode::Phi) return false;
				if (std::ranges::find(values, use.user) == values.end()) values.push_back(use.user);
			}
		}
		return true;
	}

	static bool UsesOnlySamplerDescriptors(const Inst& value) {
		if (value.Uses().empty()) return false;
		std::vector<const Inst*> values {&value};
		for (size_t index = 0; index < values.size(); ++index) {
			for (const auto& use: values[index]->Uses()) {
				if (use.user->GetOpcode() == ValueOpcode::GetSamplerResource) continue;
				if (use.user->GetOpcode() != ValueOpcode::Phi) return false;
				if (std::ranges::find(values, use.user) == values.end()) values.push_back(use.user);
			}
		}
		return true;
	}

	static bool UsesOnlyImageDescriptors(const Inst& value) {
		if (value.Uses().empty()) return false;
		std::vector<const Inst*> values {&value};
		for (size_t index = 0; index < values.size(); ++index) {
			for (const auto& use: values[index]->Uses()) {
				if (use.user->GetOpcode() == ValueOpcode::GetImageResource) continue;
				if (use.user->GetOpcode() != ValueOpcode::Phi) return false;
				if (std::ranges::find(values, use.user) == values.end()) values.push_back(use.user);
			}
		}
		return true;
	}

	const MemoryInfo* ScalarReadMemory(const Inst& read, uint32_t& index) const {
		const bool address = read.GetOpcode() == ValueOpcode::LoadAddressU32;
		if (!(address ? read.NumArgs() == 4u
		              : read.GetOpcode() == ValueOpcode::ReadConstBuffer && read.NumArgs() == 2u)) {
			return nullptr;
		}
		if (address) {
			const auto high = read.Arg(2).Resolve();
			const auto enabled = read.Arg(3).Resolve();
			if (!high.IsImmediate() || high.GetType() != Type::U32 || high.U32() != 0u ||
			    !enabled.IsImmediate() || enabled.GetType() != Type::U1 || !enabled.U1()) {
				return nullptr;
			}
		}
		index = read.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) {
			return nullptr;
		}
		const auto& memory = m_program.memory_info[index];
		return memory.kind == (address ? ResourceKind::ScalarAddress : ResourceKind::ScalarBuffer) &&
		               memory.data_bits == 32u && memory.data_dwords == 1u
		           ? &memory
		           : nullptr;
	}

	// Upstream 216bd28bf: the memory instructions per memory index, counted once at the first question (only descriptor
	// planning asks, and it adds no memory instructions).
	bool MemoryIndexBelongsTo(uint32_t index, const Inst& owner) {
		if (m_memory_index_uses.empty()) {
			m_memory_index_uses.resize(m_program.memory_info.size());
			for (const auto* block: m_program.blocks) {
				for (const auto& inst: *block) {
					const auto op = inst.GetOpcode();
					if (BufferAccessOf(op) != BufferAccess::None ||
					    AddressOpcodeInfoOf(op).access != AddressAccess::None ||
					    ImageOpcodeInfoOf(op).access != ImageAccess::None) {
						auto& use = m_memory_index_uses[inst.Flags<MemoryFlags>().index];
						use.owner = &inst;
						use.count++;
					}
				}
			}
		}
		const auto& use = m_memory_index_uses[index];
		return use.count == 0 || (use.count == 1 && use.owner == &owner);
	}

	bool MakeRuntimeTableSource(const Inst& read, DescriptorSource& descriptor) {
		const auto* handle = read.Arg(0).Resolve().TryInstruction();
		if (handle == nullptr) return false;
		const auto width = handle->GetOpcode() == ValueOpcode::GetBufferResource ? 4u
		                 : handle->GetOpcode() == ValueOpcode::GetAddressResource ? 2u : 0u;
		if (width == 0u) {
			return false;
		}
		const auto flags = read.Flags<MemoryFlags>();
		const auto kind = m_program.memory_info[flags.index].kind;
		const auto base = kind == ResourceKind::ScalarAddress || kind == ResourceKind::ScalarBuffer
		    ? ScalarReadBase(read) : UINT32_MAX;
		MakeSource(*handle, width, false, false, base, descriptor, flags.pc);
		uint32_t bad_dword = 0;
		return ValidateSource(descriptor, bad_dword);
	}

	enum class LaneQuantifier { Any, All };
	struct EdgePredicate {
		Value condition;
		bool positive;
		LaneQuantifier lanes = LaneQuantifier::All;
	};

	bool NonzeroOnEntry(Value value, const Block* block) const {
		// Each unique predecessor must execute before this use. Stop at joins: an
		// unrelated comparison is not a bound on FindILsb's zero-input sentinel.
		for (size_t depth = 0; block != nullptr && depth < m_program.blocks.size(); ++depth) {
			if (block->ImmPredecessors().size() != 1u) {
				return false;
			}
			const auto* previous = block->ImmPredecessors()[0];
			const auto edge = ConditionalEdge(previous, block);
			if (edge && edge->lanes == LaneQuantifier::All) {
				const auto* test = edge->condition.TryInstruction();
				if (test != nullptr && test->NumArgs() == 2u &&
				    ((test->GetOpcode() == ValueOpcode::INotEqual32 && edge->positive) ||
				     (test->GetOpcode() == ValueOpcode::IEqual32 && !edge->positive))) {
					for (uint32_t arg = 0; arg < 2u; ++arg) {
						uint32_t immediate;
						if (ImmediateU32(test->Arg(arg), immediate) && immediate == 0u &&
						    EquivalentValue(m_program, test->Arg(arg ^ 1u), value)) {
							return true;
						}
					}
				}
			}
			block = previous;
		}
		return false;
	}

	uint32_t WorkgroupAxis(Value key) const {
		const auto* builtin = key.Resolve().TryInstruction();
		uint32_t axis = UINT32_MAX;
		if (m_program.stage != ShaderType::Compute || builtin == nullptr ||
		    builtin->GetOpcode() != ValueOpcode::GetBuiltin || builtin->NumArgs() != 2u ||
		    builtin->Arg(0) != Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) ||
		    !ImmediateU32(builtin->Arg(1), axis) || axis >= 3u) return UINT32_MAX;
		return axis;
	}

	bool MatchTableOffset(Value value, Value& key, uint32_t& offset, uint32_t& stride) const {
		offset = 0;
		for (;;) {
			const auto* inst = value.Resolve().TryInstruction();
			if (inst == nullptr || inst->NumArgs() != 2u) {
				return false;
			}
			uint32_t immediate;
			if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    ImmediateU32(inst->Arg(1), immediate) && immediate < 32u) {
				key = inst->Arg(0).Resolve();
				stride = 1u << immediate;
				return key.GetType() == Type::U32;
			}
			if (inst->GetOpcode() == ValueOpcode::IMul32) {
				if (ImmediateU32(inst->Arg(0), stride)) key = inst->Arg(1).Resolve();
				else if (ImmediateU32(inst->Arg(1), stride)) key = inst->Arg(0).Resolve();
				else return false;
				return stride != 0u && key.GetType() == Type::U32;
			}
			if (inst->GetOpcode() != ValueOpcode::IAdd32) {
				return false;
			}
			if (ImmediateU32(inst->Arg(0), immediate)) {
				value = inst->Arg(1);
			} else if (ImmediateU32(inst->Arg(1), immediate)) {
				value = inst->Arg(0);
			} else {
				return false;
			}
			// These additions are shader U32 arithmetic, before the scalar memory offset.
			offset += immediate;
		}
	}

	std::optional<EdgePredicate> ConditionalEdge(const Block* from, const Block* to) const {
		const auto& term = from->terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    (term.true_block == to) == (term.false_block == to))
			return {};
		EdgePredicate edge {from->condition, term.true_block == to};
		while (const auto* inst = edge.condition.Resolve().TryInstruction()) {
			if (inst->GetOpcode() == ValueOpcode::LogicalNot) {
				edge.positive = !edge.positive;
			} else if (inst->GetOpcode() == ValueOpcode::ConditionRef) {
				const auto kind = inst->Flags<CFG::BranchCondition>();
				const bool scalar = kind == CFG::BranchCondition::SccZero ||
				                    kind == CFG::BranchCondition::SccNonZero;
				const bool zero = kind == CFG::BranchCondition::ExecZero ||
				                  kind == CFG::BranchCondition::VccZero;
				const bool nonzero = kind == CFG::BranchCondition::ExecNonZero ||
				                     kind == CFG::BranchCondition::VccNonZero;
				if (!scalar && !zero && !nonzero) break;
				// SCC is uniform. Negating a lane reduction exchanges all and any.
				edge.lanes = scalar || (zero == edge.positive) ? LaneQuantifier::All
				                                              : LaneQuantifier::Any;
			} else {
				break;
			}
			edge.condition = inst->Arg(0);
		}
		edge.condition = edge.condition.Resolve();
		return edge;
	}

	Value PositiveLaneWitness(const Block* use) const {
		if (use == nullptr || use->ImmPredecessors().size() != 1u) return {};
		const auto edge = ConditionalEdge(use->ImmPredecessors()[0], use);
		return edge && edge->positive ? edge->condition : Value {};
	}

	bool GuardedOnEntry(const Block* block, const auto& stops, const auto& accepts) const {
		std::vector<const Block*>        pending {block};
		std::unordered_set<const Block*> queued {block}; // a linear search of `pending` was quadratic
		for (size_t i = 0; i < pending.size(); ++i) {
			const auto* current = pending[i];
			if (stops(current) || current->ImmPredecessors().empty()) return false;
			for (const auto* previous: current->ImmPredecessors()) {
				const auto edge = ConditionalEdge(previous, current);
				if (edge && accepts(*edge)) continue;
				if (queued.insert(previous).second) pending.push_back(previous);
			}
		}
		return true;
	}

	bool HasActiveLane(Value mask, const Block* use) const {
		mask = mask.Resolve();
		const auto incoming_is_nonempty = [&](const Block* from, const Block* to,
		                                     Value incoming, const Block* header) {
			const auto accepts = [&](const EdgePredicate& edge) {
				return edge.positive && Implies(edge.condition, incoming);
			};
			const auto edge = ConditionalEdge(from, to);
			if (edge && accepts(*edge)) return true;
			const auto* definition = incoming.Resolve().TryInstruction();
			return GuardedOnEntry(from, [&](const Block* block) {
				// A witness before this definition may belong to an earlier loop iteration.
				return block == header || (definition != nullptr && block == definition->Parent());
			}, accepts);
		};
		const auto* phi = mask.TryInstruction();
		if (phi != nullptr && phi->GetOpcode() == ValueOpcode::Phi &&
		    phi->GetType() == Type::U1 && phi->NumArgs() != 0u) {
			for (size_t arm = 0; arm < phi->NumArgs(); ++arm) {
				if (!incoming_is_nonempty(phi->PhiBlock(arm), phi->Parent(),
				                          phi->Arg(arm), phi->Parent())) return false;
			}
			return true;
		}
		return use != nullptr && use->ImmPredecessors().size() == 1u &&
		       incoming_is_nonempty(use->ImmPredecessors()[0], use, mask, nullptr);
	}

	Value SimplifyGuard(Value guard) const {
		const auto invariant = ResolveInvariantPhi(m_program, guard);
		return (invariant.IsEmpty() ? guard : invariant).Resolve();
	}

	bool Implies(Value guard, Value required) const {
		guard = SimplifyGuard(guard);
		required = required.Resolve();
		if (EquivalentValue(m_program, guard, required)) return true;
		const auto* inst = guard.TryInstruction();
		return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalAnd &&
		       (Implies(inst->Arg(0), required) || Implies(inst->Arg(1), required));
	}

	Value EqualLocalKey(Value guard, Value key) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return {};
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			const auto left = EqualLocalKey(inst->Arg(0), key);
			return left.IsEmpty() ? EqualLocalKey(inst->Arg(1), key) : left;
		}
		if (inst->GetOpcode() != ValueOpcode::IEqual32 || inst->NumArgs() != 2u) return {};
		if (EquivalentValue(m_program, inst->Arg(0), key)) return inst->Arg(1).Resolve();
		if (EquivalentValue(m_program, inst->Arg(1), key)) return inst->Arg(0).Resolve();
		return {};
	}

	struct AffineOffset {
		Value    index;
		uint64_t stride = 0;
		uint64_t offset = 0;
	};

	bool MatchAffineOffset(Value value, Value guard, AffineOffset& out,
	                       uint32_t depth = 0) const {
		if (depth > 16u) return false;
		value = value.Resolve();
		uint32_t immediate = 0;
		if (ImmediateU32(value, immediate)) {
			out.offset = immediate;
			return true;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::SelectU32 && inst->NumArgs() == 3u &&
		    Implies(guard, inst->Arg(0))) {
			return MatchAffineOffset(inst->Arg(1), guard, out, depth + 1u);
		}
		if (inst->GetOpcode() == ValueOpcode::IAdd32 && inst->NumArgs() == 2u) {
			AffineOffset left, right;
			if (!MatchAffineOffset(inst->Arg(0), guard, left, depth + 1u) ||
			    !MatchAffineOffset(inst->Arg(1), guard, right, depth + 1u) ||
			    (!left.index.IsEmpty() && !right.index.IsEmpty() &&
			     !EquivalentValue(m_program, left.index, right.index))) return false;
			out.index = left.index.IsEmpty() ? right.index : left.index;
			out.stride = left.stride + right.stride;
			out.offset = left.offset + right.offset;
			return out.stride <= UINT32_MAX && out.offset <= UINT32_MAX;
		}
		if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 && inst->NumArgs() == 2u &&
		    ImmediateU32(inst->Arg(1), immediate) && immediate < 32u) {
			if (!MatchAffineOffset(inst->Arg(0), guard, out, depth + 1u)) return false;
			out.stride <<= immediate;
			out.offset <<= immediate;
			return out.stride <= UINT32_MAX && out.offset <= UINT32_MAX;
		}
		out.index = value;
		out.stride = 1u;
		return value.GetType() == Type::U32;
	}

	bool ImpliesNonzero(Value guard, Value value) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return ImpliesNonzero(inst->Arg(0), value) ||
			       ImpliesNonzero(inst->Arg(1), value);
		}
		if (inst->GetOpcode() != ValueOpcode::INotEqual32 || inst->NumArgs() != 2u)
			return false;
		uint32_t zero = 1u;
		return (ImmediateU32(inst->Arg(0), zero) && zero == 0u &&
		        EquivalentValue(m_program, inst->Arg(1), value)) ||
		       (ImmediateU32(inst->Arg(1), zero) && zero == 0u &&
		        EquivalentValue(m_program, inst->Arg(0), value));
	}

	bool ImpliesIndexBelow32(Value guard, Value index) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return ImpliesIndexBelow32(inst->Arg(0), index) ||
			       ImpliesIndexBelow32(inst->Arg(1), index);
		}
		if (inst->NumArgs() != 2u) return false;
		uint32_t limit = 0;
		return (inst->GetOpcode() == ValueOpcode::SLessThan32 &&
		        EquivalentValue(m_program, inst->Arg(0), index) &&
		        ImmediateU32(inst->Arg(1), limit) && limit == 32u) ||
		       (inst->GetOpcode() == ValueOpcode::SGreaterThan32 &&
		        ImmediateU32(inst->Arg(0), limit) && limit == 32u &&
		        EquivalentValue(m_program, inst->Arg(1), index));
	}

	bool MaskOnlyLosesBits(Value value, Value mask) const {
		const auto* update = value.Resolve().TryInstruction();
		if (update == nullptr || update->NumArgs() != 2u) return false;
		if (update->GetOpcode() == ValueOpcode::BitwiseAnd32) {
			return EquivalentValue(m_program, update->Arg(0), mask) ||
			       EquivalentValue(m_program, update->Arg(1), mask);
		}
		if (update->GetOpcode() != ValueOpcode::BitwiseXor32) return false;
		Value bit;
		if (EquivalentValue(m_program, update->Arg(0), mask)) bit = update->Arg(1);
		else if (EquivalentValue(m_program, update->Arg(1), mask)) bit = update->Arg(0);
		else return false;
		const auto* shift = bit.Resolve().TryInstruction();
		uint32_t one = 0;
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    shift->NumArgs() != 2u || !ImmediateU32(shift->Arg(0), one) || one != 1u)
			return false;
		Value position = shift->Arg(1).Resolve();
		const auto* masked = position.TryInstruction();
		uint32_t lane_mask = 0;
		if (masked != nullptr && masked->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
		    masked->NumArgs() == 2u) {
			if (ImmediateU32(masked->Arg(0), lane_mask) && lane_mask == 31u)
				position = masked->Arg(1);
			else if (ImmediateU32(masked->Arg(1), lane_mask) && lane_mask == 31u)
				position = masked->Arg(0);
		}
		const auto* first = position.Resolve().TryInstruction();
		return first != nullptr && first->GetOpcode() == ValueOpcode::FindILsb32 &&
		       first->NumArgs() == 1u && EquivalentValue(m_program, first->Arg(0), mask);
	}

	Value InitialCandidateMask(Value value, const Block* update_block) const {
		const auto* phi = value.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 2u || phi->GetType() != Type::U32) return {};
		for (uint32_t back = 0; back < 2u; ++back) {
			if (phi->PhiBlock(back) != update_block ||
			    !MaskOnlyLosesBits(phi->Arg(back), value)) continue;
			const auto initial = phi->Arg(back ^ 1u).Resolve();
			return ValidateRuntimeValue(m_program, initial, RuntimeValueType::Integer)
			           ? initial : Value {};
		}
		return {};
	}

	bool MaskEdge(const Block* from, const Block* to, Value predicate, bool positive) const {
		const auto edge = ConditionalEdge(from, to);
		// Continuation needs an active lane; an inactive exit must include every lane.
		return edge && edge->positive == positive &&
		       (positive || edge->lanes == LaneQuantifier::All) &&
		       EquivalentValue(m_program, edge->condition, predicate);
	}

	Value BoundedSetBitMask(Value index, Value guard) const {
		const auto* phi = index.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 3u || phi->GetType() != Type::U32 ||
		    !ImpliesIndexBelow32(guard, index)) return {};
		for (uint32_t bit_arm = 0; bit_arm < 3u; ++bit_arm) {
			const auto bit_guard = PositiveLaneWitness(phi->PhiBlock(bit_arm));
			const auto* selected = phi->Arg(bit_arm).Resolve().TryInstruction();
			if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32 ||
			    selected->NumArgs() != 3u ||
			    !Implies(bit_guard, selected->Arg(0))) continue;
			const auto* first = selected->Arg(1).Resolve().TryInstruction();
			if (first == nullptr || first->GetOpcode() != ValueOpcode::FindILsb32 ||
			    first->NumArgs() != 1u || !ImpliesNonzero(bit_guard, first->Arg(0)))
				continue;
			const auto initial_mask = InitialCandidateMask(first->Arg(0), phi->PhiBlock(bit_arm));
			if (initial_mask.IsEmpty()) continue;
			for (uint32_t sentinel_arm = 0; sentinel_arm < 3u; ++sentinel_arm) {
				if (sentinel_arm == bit_arm) continue;
				const auto sentinel_guard = PositiveLaneWitness(phi->PhiBlock(sentinel_arm));
				const auto* sentinel = phi->Arg(sentinel_arm).Resolve().TryInstruction();
				uint32_t bound = 0;
				if (sentinel == nullptr || sentinel->GetOpcode() != ValueOpcode::SelectU32 ||
				    sentinel->NumArgs() != 3u ||
				    !Implies(sentinel_guard, sentinel->Arg(0)) ||
				    !ImmediateU32(sentinel->Arg(1), bound) || bound != 32u) continue;
				const auto loop_active = sentinel->Arg(0).Resolve();
				const auto* active_phi = loop_active.TryInstruction();
				if (active_phi == nullptr || active_phi->GetOpcode() != ValueOpcode::Phi ||
				    active_phi->NumArgs() != 2u) continue;
				const auto other_arm = 3u - bit_arm - sentinel_arm;
				const auto* carried = phi->Arg(other_arm).Resolve().TryInstruction();
				const auto* bit_block = phi->PhiBlock(bit_arm);
				if (carried == nullptr || carried->GetOpcode() != ValueOpcode::Phi ||
				    carried->NumArgs() != 2u || carried->Parent() != active_phi->Parent() ||
				    selected->Arg(2).Resolve() != phi->Arg(sentinel_arm).Resolve() ||
				    sentinel->Arg(2).Resolve() != phi->Arg(other_arm).Resolve()) continue;
				const uint32_t back = carried->PhiBlock(0) == bit_block ? 0u : 1u;
				if (carried->PhiBlock(back) != bit_block ||
				    carried->Arg(back).Resolve() != phi->Arg(bit_arm).Resolve()) continue;
				bool invariant = false;
				for (uint32_t initial = 0; initial < 2u; ++initial) {
					invariant = active_phi->PhiBlock(initial) == carried->PhiBlock(back ^ 1u) &&
					            active_phi->PhiBlock(initial ^ 1u) == bit_block &&
					            Implies(guard, active_phi->Arg(initial)) &&
					            MaskEdge(bit_block, active_phi->Parent(),
					                     active_phi->Arg(initial ^ 1u), true);
					if (invariant) break;
				}
				if (!invariant) continue;
				if (MaskEdge(phi->PhiBlock(other_arm), phi->Parent(), loop_active, false))
					return initial_mask;
			}
		}
		return {};
	}

	const Inst* UniformizedMaterialValue(Value key, const Inst& use, Value executing = {}) const {
		Value guard;
		Value local;
		const auto* first = key.Resolve().TryInstruction();
		if (first != nullptr && first->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    first->NumArgs() == 2u) {
			guard = first->Arg(1).Resolve();
			// Empty EXEC selects lane zero, which may not have loaded a material key.
			// An executing store also witnesses nonempty EXEC at the lane read.
			if (executing.IsEmpty() ? !HasActiveLane(guard, first->Parent())
			                        : !Implies(executing, guard)) return nullptr;
			local = first->Arg(0).Resolve();
			const auto* phi = guard.TryInstruction();
			if (phi != nullptr && phi->GetOpcode() == ValueOpcode::Phi &&
			    phi->GetType() == Type::U1 && phi->NumArgs() == 2u) {
				for (uint32_t initial = 0; initial < 2u; ++initial) {
					if (Implies(phi->Arg(initial ^ 1u), guard)) {
						// The backedge only removes lanes from the initial mask.
						guard = phi->Arg(initial).Resolve();
						break;
					}
				}
			}
		} else {
			guard = PositiveLaneWitness(use.Parent());
			if (guard.IsEmpty()) return nullptr;
			local = EqualLocalKey(guard, key);
		}
		const auto* selected = local.Resolve().TryInstruction();
		if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32 ||
		    selected->NumArgs() != 3u || !Implies(guard, selected->Arg(0))) return nullptr;
		return selected;
	}

	bool MatchUniformizedMaterialKey(Value key, const Inst& image,
	                                DescriptorSource::IndirectDescriptor& indirect,
	                                DescriptorSource& material_source) {
		const auto* selected = UniformizedMaterialValue(key, image);
		if (selected == nullptr) return false;
		const auto active = selected->Arg(0).Resolve();
		const auto* read = selected->Arg(1).Resolve().TryInstruction();
		if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    read->NumArgs() != 4u || !EquivalentValue(m_program, read->Arg(3), active))
			return false;
		uint32_t high = 1u;
		if (!ImmediateU32(read->Arg(2), high) || high != 0u) return false;
		const auto memory_index = read->Flags<MemoryFlags>().index;
		if (memory_index >= m_program.memory_info.size()) return false;
		const auto& memory = m_program.memory_info[memory_index];
		if (memory.kind != ResourceKind::Global || memory.data_bits != 32u ||
		    memory.data_dwords != 1u || !MemoryIndexBelongsTo(memory_index, *read)) return false;
		const auto* material_handle = read->Arg(0).Resolve().TryInstruction();
		if (material_handle == nullptr ||
		    material_handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    !MakeRuntimeTableSource(*read, material_source)) return false;
		AffineOffset offset;
		if (!MatchAffineOffset(read->Arg(1), active, offset) || offset.index.IsEmpty() ||
		    offset.stride == 0u || offset.offset + memory.offset > UINT32_MAX ||
		    offset.offset + memory.offset + 31u * offset.stride + 4u >
		        static_cast<uint64_t>(UINT32_MAX) + 1u) return false;
		const auto mask = BoundedSetBitMask(offset.index, active);
		if (mask.IsEmpty()) return false;
		indirect.selector.emplace(DescriptorSource::IndirectDescriptor::SelectorRead{.source = InternSource(material_source),
		                              .stride = static_cast<uint32_t>(offset.stride),
		                              .offset = static_cast<uint32_t>(offset.offset + memory.offset)});
		indirect.key_count = Value(32u);
		indirect.selector_mask = mask;
		return true;
	}

	const Inst* BoundedLoop(Value key, const Block* use, const auto& accepts_bound) const {
		const auto* phi = key.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->GetType() != Type::U32 ||
		    phi->NumArgs() != 2u || phi->NumPhiBlocks() != 2u)
			return {};
		const Block* increment_block = nullptr;
		uint32_t initial_arm = 0;
		for (uint32_t initial = 0; initial < 2u; ++initial) {
			const auto zero = phi->Arg(initial).Resolve();
			const auto* step = phi->Arg(initial ^ 1u).Resolve().TryInstruction();
			if (!zero.IsImmediate() || zero.GetType() != Type::U32 || zero.U32() != 0u ||
			    step == nullptr || step->GetOpcode() != ValueOpcode::IAdd32 ||
			    step->Parent() == nullptr) continue;
			// Structurization can forward the native increment through a synthetic continue block.
			const auto* incoming = phi->PhiBlock(initial ^ 1u);
			const auto* next = phi->Parent();
			for (size_t edges = 0; incoming != step->Parent() && edges < m_program.blocks.size(); ++edges) {
				if (incoming == nullptr || incoming->terminator.kind != CFG::TerminatorKind::Branch ||
				    incoming->terminator.true_block != next || incoming->ImmPredecessors().size() != 1u)
					break;
				next = incoming;
				incoming = incoming->ImmPredecessors()[0];
			}
			if (incoming != step->Parent()) continue;
			uint32_t increment = 0;
			if ((step->Arg(0).Resolve() == key &&
			     ImmediateU32(step->Arg(1), increment) && increment == 1u) ||
			    (step->Arg(1).Resolve() == key &&
			     ImmediateU32(step->Arg(0), increment) && increment == 1u)) {
				increment_block = step->Parent();
				initial_arm = initial;
				break;
			}
		}
		if (increment_block == nullptr) return {};

		const auto stops = [&](const Block* block) { return block == phi->Parent(); };
		for (const auto& use_of_key: phi->Uses()) {
			const auto* compare = use_of_key.user;
			if ((compare->GetOpcode() != ValueOpcode::SLessThan32 &&
			     compare->GetOpcode() != ValueOpcode::ULessThan32) || use_of_key.operand != 0u ||
			    !accepts_bound(*compare, initial_arm)) continue;
			if (!GuardedOnEntry(use != nullptr ? use : increment_block, stops, [&](const EdgePredicate& edge) {
				return edge.positive && Implies(edge.condition, Value(use_of_key.user));
			})) continue;
			LoopBoundProof proof(m_program, *phi, *compare);
			// The image bound and the increment guard are separate obligations: a
			// skipped image alone does not prevent signed induction wraparound.
			if (GuardedOnEntry(increment_block, stops, [&](const EdgePredicate& edge) {
				return proof.Excludes(edge.condition, edge.positive);
			})) return compare;
		}
		return {};
	}

	// do { ... key ... } while (++key < N) with key starting at 0 and an immediate N: the
	// back edge only continues while the incremented key is below N, so key < N everywhere.
	// Every path to `use` passes a uniform branch edge that holds only when key < N; returns the
	// largest such N over the guarding edges (0 when some path is unguarded).
	uint32_t GuardedKeyBound(Value key, const Block* use) const {
		if (use == nullptr) return 0u;
		uint32_t bound = 0;
		const auto accepts = [&](const EdgePredicate& edge) {
			const auto* compare = edge.condition.Resolve().TryInstruction();
			if (compare == nullptr || compare->NumArgs() != 2u || edge.lanes != LaneQuantifier::All)
				return false;
			uint32_t limit = 0;
			const bool below =
			    (compare->GetOpcode() == ValueOpcode::UGreaterThanEqual32 && !edge.positive) ||
			    (compare->GetOpcode() == ValueOpcode::ULessThan32 && edge.positive);
			if (!below || !EquivalentValue(m_program, compare->Arg(0), key) ||
			    !ImmediateU32(compare->Arg(1), limit) || limit == 0u || limit > 1024u)
				return false;
			bound = std::max(bound, limit);
			return true;
		};
		return GuardedOnEntry(use, [](const Block*) { return false; }, accepts) ? bound : 0u;
	}

	uint32_t DoWhileImmediateBound(Value key) const {
		const auto* phi = key.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->GetType() != Type::U32 ||
		    phi->NumArgs() != 2u)
			return 0u;
		for (uint32_t initial = 0; initial < 2u; ++initial) {
			const auto zero = phi->Arg(initial).Resolve();
			const auto* step = phi->Arg(initial ^ 1u).Resolve().TryInstruction();
			uint32_t increment = 0;
			if (!zero.IsImmediate() || zero.GetType() != Type::U32 || zero.U32() != 0u || step == nullptr ||
			    step->GetOpcode() != ValueOpcode::IAdd32 ||
			    !((step->Arg(0).Resolve() == key && ImmediateU32(step->Arg(1), increment)) ||
			      (step->Arg(1).Resolve() == key && ImmediateU32(step->Arg(0), increment))) ||
			    increment != 1u)
				continue;
			for (const auto& use: step->Uses()) {
				const auto* compare = use.user;
				uint32_t bound = 0;
				if ((compare->GetOpcode() == ValueOpcode::SLessThan32 ||
				     compare->GetOpcode() == ValueOpcode::ULessThan32) &&
				    use.operand == 0u && ImmediateU32(compare->Arg(1), bound) && bound != 0u &&
				    bound <= 1024u)
					return bound;
			}
		}
		return 0u;
	}

	const Inst* BoundedLoop(Value key, const Block* use) const {
		return BoundedLoop(key, use, [&](const Inst& compare, uint32_t) {
			return ValidateRuntimeValue(m_program, compare.Arg(1), RuntimeValueType::Integer);
		});
	}

	bool MatchScalarMaterialKey(Value key, DescriptorSource::IndirectDescriptor& indirect,
	                            DescriptorSource& material_source) {
		const auto* read = key.Resolve().TryInstruction();
		if (read == nullptr || read->GetOpcode() != ValueOpcode::ReadConstBuffer) return false;
		uint32_t memory_index = 0;
		const auto* memory = ScalarReadMemory(*read, memory_index);
		Value index;
		uint32_t offset = 0, stride = 0;
		if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
		    memory->offset > INT32_MAX || !MemoryIndexBelongsTo(memory_index, *read) ||
		    !MatchTableOffset(read->Arg(1), index, offset, stride) ||
		    ((offset | stride | memory->offset) & 3u) != 0u ||
		    uint64_t {offset} + memory->offset > UINT32_MAX) return false;
		const auto* bound = BoundedLoop(index, read->Parent());
		if (bound == nullptr || bound->GetOpcode() != ValueOpcode::ULessThan32 ||
		    !MakeRuntimeTableSource(*read, material_source)) return false;
		indirect.selector.emplace(DescriptorSource::IndirectDescriptor::SelectorRead {
		    .source = InternSource(material_source), .stride = stride,
		    .offset = offset + memory->offset});
		indirect.selector_first = Value(0u);
		indirect.key_count = bound->Arg(1);
		return true;
	}

	// Game counts: Ghost of Yotei record tables whose filled record count the game keeps in CPU memory
	// the shader does not read (the layout of the game's data in guest memory; no game code). Update Oct Maps 34e090c6:
	// 32 GI-volume V#s at T+48 (stride 16); the CPU fills slots 0..count-1 and the job list names only those;
	// count =mem32[mem64[T + 0x238] + 0x30], at most 32; the per-volume V# at +0x20 of the same block has count records
	// (its NUM_RECORDS at +0x28), which a block the game reused for a later frame would not match.
	// Hardware-checked game data, not a workaround (D6 step K3; the user's decision: kept as described
	// knowledge of Yotei's data, no game code): game counts compared every job-list key the GPU used
	// with this count - gate 940,842 keys in 2,945 dispatches, night 582,844 in 4,289: 0 violations (largest key 8,
	// largest count 9); the open world dispatches no 34e090c6.
	struct GameRecordCount {
		uint64_t shader_hash;
		uint64_t field;
		uint32_t stride, pointer, count_field, count_check, limit;
	};
	static constexpr GameRecordCount GameRecordCounts[] = {
	    {0x34e090c623ad611cull, 48u, 16u, 0x238u, 0x30u, 0x28u, 32u}};

	bool MatchGameRecordCount(uint64_t field, uint32_t stride, DescriptorSource::IndirectDescriptor& indirect) const {
		for (const auto& count: GameRecordCounts) {
			if (count.shader_hash != m_program.shader_hash || count.field != field || count.stride != stride) continue;
			indirect.count_pointer = count.pointer;
			indirect.count_field   = count.count_field;
			indirect.count_check   = count.count_check;
			indirect.count_limit   = count.limit;
			return true;
		}
		return false;
	}

	void FoldBoundedLoopSelectors() {
		for (auto* header: m_program.blocks) {
			for (auto& induction: *header) {
				const Inst* carried = nullptr;
				uint32_t maximum = 0;
				const auto* compare = BoundedLoop(Value(&induction), nullptr,
				    [&](const Inst& test, uint32_t initial) {
					if (test.GetOpcode() != ValueOpcode::ULessThan32) return false;
					carried = test.Arg(1).Resolve().TryInstruction();
					if (carried == nullptr || carried->GetOpcode() != ValueOpcode::Phi ||
					    carried->GetType() != Type::U32 || carried->Parent() != header ||
					    carried->NumArgs() != 2u || carried->NumPhiBlocks() != 2u ||
					    carried->PhiBlock(0) != induction.PhiBlock(0) ||
					    carried->PhiBlock(1) != induction.PhiBlock(1)) return false;
					maximum = SelectorMaximum(carried->Arg(initial));
					if (maximum == 0u || maximum == UINT32_MAX) return false;
					const std::array assumptions {std::pair {static_cast<const Inst*>(&induction), maximum - 1u},
					                             std::pair {carried, maximum}};
					// Close the carried-bound invariant before using either assumption.
					return SelectorMaximum(carried->Arg(initial ^ 1u), assumptions) <= maximum;
				});
				if (compare == nullptr) continue;
				const std::array assumptions {std::pair {static_cast<const Inst*>(&induction), maximum - 1u},
				                             std::pair {carried, maximum}};
				// Native scalar branches advance the zero/+1 induction uniformly, so
				// even an any-lane bound test limits it by the largest carried bound.
				// Per-lane edges instead require the comparison on every executing lane.
				for (auto* block: m_program.blocks) {
					if (!GuardedOnEntry(block, [&](const Block* at) { return at == header; },
					    [&](const EdgePredicate& edge) {
						return edge.positive && Implies(edge.condition, Value(const_cast<Inst*>(compare)));
					})) continue;
					for (auto& inst: *block) {
						if (inst.GetOpcode() == ValueOpcode::SelectU32 &&
						    SelectorPredicateFalse(inst.Arg(0), assumptions))
							inst.ReplaceUsesWith(inst.Arg(2));
					}
				}
			}
		}
	}

	bool BoundedSelectedIndex(Value index, Value active,
	                          DescriptorSource::IndirectDescriptor& indirect) const {
		const auto* selected = index.Resolve().TryInstruction();
		if (selected == nullptr || selected->GetOpcode() != ValueOpcode::Phi ||
		    selected->NumArgs() == 0u) return false;
		std::vector<const Inst*> values {selected};
		for (size_t i = 0; i < values.size(); ++i) {
			const auto* value = values[i];
			if (value->GetOpcode() != ValueOpcode::Phi &&
			    value->GetOpcode() != ValueOpcode::SelectU32) continue;
			for (size_t arm = value->GetOpcode() == ValueOpcode::SelectU32 ? 1u : 0u;
			     arm < value->NumArgs(); ++arm) {
				const auto* next = value->Arg(arm).Resolve().TryInstruction();
				if (next != nullptr && std::ranges::find(values, next) == values.end())
					values.push_back(next);
			}
		}
		for (const auto* candidate: values) {
			if (candidate->GetOpcode() != ValueOpcode::IAdd32) continue;
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				const auto* bound = BoundedLoop(candidate->Arg(arg), candidate->Parent());
				const auto first = candidate->Arg(arg ^ 1u);
				if (bound == nullptr || !ValidateRuntimeValue(m_program, first, RuntimeValueType::Integer))
					continue;
				const auto* induction = candidate->Arg(arg).ResolveInstruction();
				const Inst* carried = nullptr;
				for (const auto* value: values) {
					if (value->GetOpcode() == ValueOpcode::Phi && value->Parent() == induction->Parent()) {
						if (carried != nullptr) { carried = nullptr; break; }
						carried = value;
					}
				}
				if (carried == nullptr || carried->NumArgs() != 2u ||
				    carried->PhiBlock(0) != induction->PhiBlock(0) ||
				    carried->PhiBlock(1) != induction->PhiBlock(1)) continue;
				std::vector<std::pair<Value, bool>> exits;
				for (size_t arm = 0; arm < selected->NumArgs(); ++arm) {
					const auto edge = ConditionalEdge(selected->PhiBlock(arm), selected->Parent());
					if (!edge || edge->lanes != LaneQuantifier::All) break;
					exits.emplace_back(edge->condition, edge->positive);
				}
				if (exits.size() != selected->NumArgs()) continue;
				for (const auto& mask: *induction->Parent()) {
					if (mask.GetOpcode() != ValueOpcode::Phi || mask.GetType() != Type::U1 ||
					    mask.NumArgs() != 2u || mask.PhiBlock(0) != carried->PhiBlock(0) ||
					    mask.PhiBlock(1) != carried->PhiBlock(1)) continue;
					const auto initial = induction->Arg(0).Resolve().IsImmediate() ? 0u : 1u;
					const auto enabled = mask.Arg(initial).Resolve();
					if (!enabled.IsImmediate() || enabled.GetType() != Type::U1 || !enabled.U1()) continue;
					LoopBoundProof proof(m_program, *induction, *bound);
					if (!proof.SelectsRange(*selected, active, *candidate, *carried, mask, exits)) continue;
					indirect.selector_first = first;
					indirect.key_count = bound->Arg(1);
					return true;
				}
			}
		}
		return false;
	}

	void TraceIndirectReject(int line) const {
		if (Common::EnvRead::IndirectImage())
			std::fprintf(stderr, "TRACE_INDIRECT_IMAGE reject hash=0x%016llx line=%d\n",
			     static_cast<unsigned long long>(m_program.shader_hash), line);
	}

	bool MatchDescriptorTable(Inst& handle, const DescriptorSource& descriptor,
	                          IndirectDescriptorPlan& plan,
	                          DescriptorSource& table_source, Value& key, uint32_t& table_offset,
	                          uint32_t& table_stride, uint32_t& table_immediate) {
		Inst* table_handle = nullptr;
		Value table_guard;
		// An R128 image reads only words 0..3; the translator supplies constant zero above.
		const bool r128 = handle.GetOpcode() == ValueOpcode::GetImageResource && R128ImageHandle(handle);
		for (uint32_t dword = 0; dword < handle.NumArgs(); ++dword) {
			if (r128 && dword >= 4u) {
				uint32_t word = 1u;
				if (!ImmediateU32(descriptor.dwords[dword], word) || word != 0u)
					{ TraceIndirectReject(__LINE__); return false; }
				continue;
			}
			auto* read = descriptor.dwords[dword].Resolve().TryInstruction();
			if (read == nullptr) {
				{ TraceIndirectReject(__LINE__); return false; }
			}
			uint32_t memory_index = 0;
			const auto* memory = ScalarReadMemory(*read, memory_index);
			uint32_t component = 0;
			const bool indexed = memory == nullptr;
			if (indexed) {
				if (handle.NumArgs() != 8u) return false;
				const auto* selected = UniformizedMaterialValue(descriptor.dwords[dword], handle);
				if (selected == nullptr) return false;
				const auto* extract = selected->Arg(1).Resolve().TryInstruction();
				if (extract == nullptr || extract->GetOpcode() != ValueOpcode::CompositeExtractU32x4 ||
				    !ImmediateU32(extract->Arg(1), component) || component >= 4u) return false;
				read = extract->Arg(0).Resolve().TryInstruction();
				if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadBufferU32x4) return false;
				memory_index = read->Flags<MemoryFlags>().index;
				if (memory_index >= m_program.memory_info.size()) return false;
				memory = &m_program.memory_info[memory_index];
				uint32_t offset = 1u, scalar = 1u;
				if (memory->kind != ResourceKind::Buffer || memory->typed || memory->formatted ||
				    !memory->idxen || memory->offen || memory->data_bits != 32u ||
				    memory->data_dwords != 4u ||
				    !ImmediateU32(read->Arg(2), offset) || offset != 0u ||
				    !ImmediateU32(read->Arg(3), scalar) || scalar != 0u ||
				    !EquivalentValue(m_program, read->Arg(4), selected->Arg(0)) ||
				    (dword != 0u && !EquivalentValue(m_program, table_guard, read->Arg(4)))) return false;
				table_guard = read->Arg(4);
			}
			if (memory->offset > INT32_MAX || (memory->offset & 3u) != 0u ||
			    !MemoryIndexBelongsTo(memory_index, *read)) {
				{ TraceIndirectReject(__LINE__); return false; }
			}
			auto* current_handle = read->Arg(0).Resolve().TryInstruction();
			Value current_key;
			uint32_t offset = 0;
			uint32_t stride = 0;
			if (current_handle == nullptr ||
			    current_handle->GetOpcode() != (memory->kind == ResourceKind::ScalarAddress
			                                      ? ValueOpcode::GetAddressResource
			                                      : ValueOpcode::GetBufferResource) ||
			    (memory->kind == ResourceKind::ScalarAddress && read->Parent() != handle.Parent() &&
			     !BlockDominates(read->Parent(), handle.Parent())) ||
			    (table_handle != nullptr &&
			     !EquivalentValue(m_program, Value(table_handle), Value(current_handle)))) {
				return false;
			}
			if (!indexed && !MatchTableOffset(read->Arg(1), current_key, offset, stride)) {
				// Two-level table: a constant offset from a base address that buffer records hold.
				uint32_t    inner = 1u;
				const Inst* pointer_read = nullptr;
				Value       pointer_key;
				uint32_t    pointer_stride = 0, pointer_record_offset = 0;
				if (memory->kind != ResourceKind::ScalarAddress ||
				    !ImmediateU32(read->Arg(1), inner) || inner != 0u ||
				    !MatchPointerRecord(*current_handle, pointer_read, pointer_key, pointer_stride,
				                        pointer_record_offset) ||
				    (plan.pointer_record && (plan.pointer_stride != pointer_stride ||
				                             plan.pointer_record_offset != pointer_record_offset)))
					return false;
				plan.pointer_record        = true;
				plan.pointer_read          = pointer_read;
				plan.pointer_stride        = pointer_stride;
				plan.pointer_record_offset = pointer_record_offset;
				current_key                = pointer_key;
			}
			if (indexed) {
				current_key = read->Arg(1);
				offset = component * 4u;
			}
			if (dword == 0u) {
				if (component != 0u) return false;
				plan.table_indexed = indexed;
				key = current_key;
				table_offset = offset;
				table_immediate = memory->offset;
				table_stride = stride;
			} else if (plan.table_indexed != indexed || table_stride != stride ||
			           !EquivalentValue(m_program, key, current_key) ||
			           uint64_t {table_offset} + table_immediate + dword * sizeof(uint32_t) !=
			               uint64_t {offset} + memory->offset) {
				return false;
			}
			plan.split_offsets |= !indexed && offset != table_offset;
			table_handle = current_handle;
			if (handle.NumArgs() == 8u || handle.GetOpcode() == ValueOpcode::GetSamplerResource) {
				if (memory->kind == ResourceKind::ScalarAddress)
					plan.retain_reads |= handle.NumArgs() == 8u ? !UsesOnlyImageDescriptors(*read)
					                                            : !UsesOnlySamplerDescriptors(*read);
			} else if (!plan.range_only && std::ranges::any_of(read->Uses(), [](const Use& use) {
				return use.user->GetOpcode() != ValueOpcode::GetBufferResource ||
				       std::ranges::any_of(use.user->Uses(), [](const Use& consumer) {
					       return consumer.user->GetOpcode() != ValueOpcode::StoreBufferU32;
				       });
			    })) {
				{ TraceIndirectReject(__LINE__); return false; }
			}
			plan.memory[dword] = memory_index;
			plan.reads[dword] = read;
		}

		if (!MakeRuntimeTableSource(plan.pointer_record ? *plan.pointer_read : *plan.reads[0], table_source)) {
			{ TraceIndirectReject(__LINE__); return false; }
		}
		return true;
	}

	// Whether an explicit-LOD gather samples through `handle`: MaterializeResources refuses a GPU-selected sampler
	// there (gather_lod), so such a sampler keeps its previous path.
	bool FeedsExplicitLodGather(const Inst& handle) const {
		return std::ranges::any_of(handle.Uses(), [&](const Use& use) {
			if (use.user->GetOpcode() != ValueOpcode::ImageGatherRaw) return false;
			const auto index = use.user->Flags<MemoryFlags>().index;
			return index >= m_program.memory_info.size() ||
			       (m_program.memory_info[index].image_sample_flags & Decoder::ImageSampleFlagLod) != 0u;
		});
	}

	// Table-record S#s: the four S# words are ReadConstBuffer(table V#, key * stride + offset) at
	// consecutive fields (MatchDescriptorTable) and the GPU computes `key`: the decal index of a Tiled Decals record
	// (s_buffer_load_dwordx8 s[12:19], decals, key * 0x368 offset:136) or the waterfall-selected region of the
	// terrain pixel shader (s_buffer_load_dwordx4 s[56:59], regions, readfirstlane * 0x88 offset:80). The hardware
	// reads the S# of record `key`. Each candidate here is exactly that S#, read from guest memory at every
	// materialization for every record inside the V# (MaterializeIndirectDescriptor, all_records), and the GPU maps
	// its key to the candidate. Dword-aligned fields only, so the S_BUFFER_LOAD offset alignment changes nothing.
	// A key past the table reads a zero S# on the GPU and selects the root (record 0's S#) here.
	bool RecordTableSampler(const Inst& handle, IndirectDescriptorPlan& plan, uint32_t table_offset,
	                        uint32_t table_immediate, uint32_t table_stride,
	                        DescriptorSource::IndirectDescriptor& indirect) const {
		if (plan.table_indexed || plan.pointer_record || plan.split_offsets || plan.reads[0] == nullptr ||
		    table_stride < 16u || ((table_stride | table_offset | table_immediate) & 3u) != 0u ||
		    uint64_t {table_offset} + table_immediate > UINT32_MAX - 16u || FeedsExplicitLodGather(handle))
			return false;
		indirect.table_offset    = table_offset;
		indirect.table_immediate = table_immediate;
		indirect.table_stride    = table_stride;
		indirect.all_records     = true;
		indirect.workgroup_axis  = UINT32_MAX;
		// The S# words stay GPU loads for any other consumer (e.g. a gather also reads two of them); the handle
		// carries the record index instead (Run).
		plan.retain_reads = true;
		// Once per shader and table field: which samplers took this form.
		static std::mutex                                           mutex;
		static std::set<std::tuple<uint64_t, uint32_t, uint32_t>> seen;
		const uint32_t field = table_offset + table_immediate;
		std::scoped_lock lock(mutex);
		if (seen.size() < 256u &&
		    seen.insert(std::make_tuple(uint64_t {m_program.shader_hash}, table_stride, field)).second)
			std::fprintf(stderr, "FIX_SAMPLER_TABLE_RECORDS hash=0x%016llx pc=0x%08x stride=%u field=%u\n",
			             static_cast<unsigned long long>(m_program.shader_hash),
			             plan.reads[0]->Flags<MemoryFlags>().pc, table_stride, field);
		return true;
	}

	bool TryMakeIndirectImage(Inst& handle, const DescriptorSource& descriptor,
	                          IndirectDescriptorPlan& plan) {
		// Samplers use the same keyed descriptor-table forms as images (four words).
		const bool sampler = handle.GetOpcode() == ValueOpcode::GetSamplerResource;
		const uint32_t words = sampler ? 4u : 8u;
		if ((!sampler && handle.GetOpcode() != ValueOpcode::GetImageResource) || handle.NumArgs() != words)
			{ TraceIndirectReject(__LINE__); return false; }
		DescriptorSource table_source;
		Value key;
		uint32_t table_offset = 0;
		uint32_t table_stride = 0;
		uint32_t table_immediate = 0;
		if (!MatchDescriptorTable(handle, descriptor, plan, table_source, key, table_offset,
		                          table_stride, table_immediate)) return false;
		DescriptorSource material_source;
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_offset = table_offset;
		indirect.table_immediate = table_immediate;
		indirect.table_stride = table_stride;
		indirect.workgroup_axis = WorkgroupAxis(key);
		if (plan.pointer_record) {
			// Two-level table: the key is a lane value; every record of the buffer is a candidate.
			indirect.pointer_offset  = table_offset + table_immediate;
			std::fprintf(stderr, "POINTER_RECORD_TABLE hash=0x%016llx %s stride=%u record_offset=%u pointer_offset=%u\n",
			             static_cast<unsigned long long>(m_program.shader_hash), sampler ? "sampler" : "image",
			             plan.pointer_stride, plan.pointer_record_offset, indirect.pointer_offset);
			indirect.table_offset    = plan.pointer_record_offset;
			indirect.table_immediate = 0u;
			indirect.table_stride    = plan.pointer_stride;
			indirect.all_records     = true;
			indirect.workgroup_axis  = UINT32_MAX;
			if (plan.pointer_stride < 8u || (plan.pointer_stride & 3u) != 0u) {
				TraceIndirectReject(__LINE__);
				return false;
			}
		} else if (indirect.workgroup_axis != UINT32_MAX && !plan.table_indexed) {
			// The descriptor also supplies dimensions to shader arithmetic. Keep its reads;
			// only the image handle is projected onto the bounded workgroup key.
			plan.retain_reads = true;
		} else if (table_source.dword_count == 2u) {
			// Records may carry more than the descriptor (e.g. image + sampler); any dword stride works.
			if (table_stride < words * 4u || (table_stride & 3u) != 0u) { TraceIndirectReject(__LINE__); return false; }
			const auto* selector = key.Resolve().TryInstruction();
			const bool bitscan = selector != nullptr && selector->GetOpcode() == ValueOpcode::FindILsb32 &&
			    selector->NumArgs() == 1u && !m_shader_writes &&
			    NonzeroOnEntry(selector->Arg(0), handle.Parent());
			if (bitscan) {
				indirect.key_count = Value(32u);
			} else {
				// The loop bound is itself a snapshot-validated runtime value (BoundedLoop),
				// and the table words are immutable snapshot dependencies, so shader writes
				// elsewhere do not change which records the loop can select.
				if (const auto* bound = BoundedLoop(key, handle.Parent()))
					indirect.key_count = bound->Arg(1);
				else if (const auto bound = DoWhileImmediateBound(key); bound != 0u)
					indirect.key_count = Value(bound);
				else if (WorkaroundWaterfallPointerRecord0()) {
					if (const auto guarded = GuardedKeyBound(key, handle.Parent()); guarded != 0u)
						indirect.key_count = Value(guarded);
				}
			}
			// A low-bit mask bounds the key independently of memory: every masked value is
			// a candidate, and the table words themselves are snapshot dependencies.
			const auto* masked = key.Resolve().TryInstruction();
			uint32_t mask = 0;
			if (indirect.key_count.IsEmpty() && masked != nullptr &&
			    masked->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
			    (ImmediateU32(masked->Arg(1), mask) || ImmediateU32(masked->Arg(0), mask)) &&
			    mask != 0u && mask < 1024u && (mask & (mask + 1u)) == 0u)
				indirect.key_count = Value(mask + 1u);
			// So does an unsigned bit-field extract of constant width: every key is below 2^width.
			uint32_t width = 0;
			if (indirect.key_count.IsEmpty() && masked != nullptr &&
			    masked->GetOpcode() == ValueOpcode::BitFieldUExtract && masked->NumArgs() == 3u &&
			    ImmediateU32(masked->Arg(2), width) && width != 0u && width <= 10u)
				indirect.key_count = Value(1u << width);
			if (indirect.key_count.IsEmpty() && Common::EnvRead::IndirectImage())
				std::fprintf(stderr, "TRACE_INDIRECT_IMAGE key hash=0x%016llx opcode=%s writes=%u\n",
				     static_cast<unsigned long long>(m_program.shader_hash),
				     masked != nullptr ? ValueOpcodeName(masked->GetOpcode()).data() : "value",
				     m_shader_writes ? 1u : 0u);
			if (indirect.key_count.IsEmpty() &&
			    !MatchUniformizedMaterialKey(key, handle, indirect, material_source)) {
				{ TraceIndirectReject(__LINE__); return false; }
			}
			if ((table_offset & 3u) != 0u ||
			    (bitscan && table_offset > UINT32_MAX - (32u * 32u - 1u))) return false;
		} else if (sampler) {
			// Samplers have no host table translation (no heap like the image table below): the S# of every record of
			// the V# is a candidate (RecordTableSampler).
			if (!RecordTableSampler(handle, plan, table_offset, table_immediate, table_stride, indirect))
				{ TraceIndirectReject(__LINE__); return false; }
		} else if (!plan.table_indexed &&
		           MatchScalarMaterialKey(key, indirect, material_source)) {
			// The host enumerates the guarded scalar loop; its key and T# payload stay on the GPU.
			plan.retain_reads = true;
		} else {
			// A bounded V# supplies the complete image table. Leave every GPU selector
			// and descriptor read in the shader; the host only translates table bytes.
			if (plan.split_offsets) return false;
			// The bindless tables: the record field the key points at, before the offsets are cleared. Run
			// computes the key as index * (the V#'s stride) + immediate for a vector index (buffer_load idxen), and as
			// (soffset & ~3) + (immediate & ~3) for a scalar offset, where MatchTableOffset proved soffset =
			// index * table_stride + table_offset; a stride that is a multiple of 4 keeps index * stride clear of the
			// mask. A stride below 16 bytes is not a record of descriptors, one beyond 1 MiB (a negative multiplier)
			// not a table the host translates (no field list: the host falls back). The U32 additions MatchTableOffset
			// summed can wrap: a negative offset reaches the same bytes from a later record, so it becomes its place
			// in the record (records before it are translated in vain, never wrongly).
			if (plan.table_indexed) {
				plan.field_known  = true;
				plan.field_stride = 0u;
				plan.field_offset = table_immediate;
			} else if (table_stride >= 16u && table_stride <= (1u << 20u) && (table_stride & 3u) == 0u) {
				int64_t field = int64_t {static_cast<int32_t>(table_offset & ~3u)} + (table_immediate & ~3u);
				if (field < 0) field = (field % table_stride + table_stride) % table_stride;
				if (field <= INT32_MAX) {
					plan.field_known  = true;
					plan.field_stride = table_stride;
					plan.field_offset = static_cast<uint32_t>(field);
				}
			}
			indirect.table_stride = 0u;
			indirect.table_offset = 0u;
			indirect.table_immediate = 0u;
			indirect.workgroup_axis = UINT32_MAX;
			plan.table_read = plan.reads[0];
			if (plan.table_indexed) indirect.table_record_bytes = table_immediate + 32u;
			else indirect.table_scalar = true;
			plan.retain_reads = true;
		}

		if (plan.retain_reads) plan.reads.fill(nullptr);
		indirect.table_source = InternSource(table_source);
		if (sampler) {
			DescriptorSource sampler_source;
			sampler_source.dword_count = 4u;
			sampler_source.dwords.fill(Value(0u));
			sampler_source.indirect_descriptor = indirect;
			plan.handle = &handle;
			plan.source = InternSource(sampler_source);
			plan.key = key;
			return true;
		}
		DescriptorSource image_source;
		image_source.dword_count = 8u;
		image_source.dwords.fill(Value(0u));
		std::copy_n(material_source.dwords.begin(), material_source.dword_count,
		            image_source.dwords.begin());
		std::copy_n(table_source.dwords.begin(), table_source.dword_count,
		            image_source.dwords.begin() + 4u);
		image_source.indirect_descriptor = indirect;
		plan.handle = &handle;
		plan.source = InternSource(image_source);
		plan.key = key;
		plan.roots = image_source.dwords;
		return true;
	}

	bool TryMakeIndirectBuffer(Inst& handle, const DescriptorSource& descriptor,
	                           IndirectDescriptorPlan& plan) {
		if (handle.GetOpcode() != ValueOpcode::GetBufferResource || handle.NumArgs() != 4u ||
		    handle.Uses().empty()) return false;
		for (const auto& use: handle.Uses()) {
			const auto& inst = *use.user;
			if (inst.GetOpcode() != ValueOpcode::StoreBufferU32) return false;
			const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
			uint32_t scalar = 1u;
			if (memory.kind != ResourceKind::Buffer || memory.typed || memory.formatted ||
			    memory.idxen || !memory.offen || !ImmediateU32(inst.Arg(3), scalar) || scalar != 0u)
				return false;
		}
		DescriptorSource table_source;
		Value key;
		uint32_t table_offset = 0;
		uint32_t table_stride = 0;
		uint32_t table_immediate = 0;
		if (!MatchDescriptorTable(handle, descriptor, plan, table_source, key, table_offset,
		                          table_stride, table_immediate) || table_stride != 16u ||
		    table_source.dword_count != 4u || table_offset != 0u || table_immediate != 0u) return false;
		const auto* lane_read = key.Resolve().TryInstruction();
		if (lane_read == nullptr || lane_read->GetOpcode() != ValueOpcode::ReadFirstLane)
			return false;
		const Inst* selected = nullptr;
		for (const auto& use: handle.Uses()) {
			const auto* value = UniformizedMaterialValue(
			    key, *use.user, use.user->Arg(use.user->NumArgs() - 1u));
			if (value == nullptr || (selected != nullptr && selected != value)) return false;
			selected = value;
		}
		const auto* read = selected->Arg(1).Resolve().TryInstruction();
		uint32_t lane = 0;
		if (read != nullptr && read->GetOpcode() == ValueOpcode::CompositeExtractU32x2) {
			if (!ImmediateU32(read->Arg(1), lane) || lane >= 2u) return false;
			read = read->Arg(0).Resolve().TryInstruction();
			if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadBufferU32x2) return false;
		} else if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadBufferU32) {
			return false;
		}
		const auto& memory = m_program.memory_info[read->Flags<MemoryFlags>().index];
		uint32_t offset = 1u, scalar = 1u;
		if (memory.kind != ResourceKind::Buffer || memory.typed || memory.formatted ||
		    !memory.idxen || memory.offen || memory.offset > UINT32_MAX - lane * 4u ||
		    !ImmediateU32(read->Arg(2), offset) || offset != 0u ||
		    !ImmediateU32(read->Arg(3), scalar) || scalar != 0u ||
		    !EquivalentValue(m_program, selected->Arg(0), read->Arg(4))) return false;
		DescriptorSource material_source;
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_stride = table_stride;
		if (!MakeRuntimeTableSource(*read, material_source) ||
		    !BoundedSelectedIndex(read->Arg(1), read->Arg(4), indirect)) return false;
		indirect.selector.emplace(DescriptorSource::IndirectDescriptor::SelectorRead{.source = InternSource(material_source),
		                              .offset = memory.offset + lane * 4u});
		indirect.table_source = InternSource(table_source);
		DescriptorSource source;
		source.dword_count = 4u;
		source.dwords.fill(Value(0u));
		source.indirect_descriptor = indirect;
		plan.handle = &handle;
		plan.source = InternSource(source);
		plan.key = key;
		return true;
	}

	// Composed V#s: the largest vector offset bits RecordKeyedAccess admits (beyond, the access keeps
	// its path: a record-keyed range should cover the bytes the access can reach, not a wide guess).
	static constexpr uint32_t MaxComposedVectorOffsetBits = (1u << 20u) - 1u;

	// Composed V#s: the bits a vector offset can have on a lane that executes the access (`active`: the
	// access's EXEC). A select on exactly that mask, or on a condition true on every lane, gives its new value there (a
	// predicated VGPR write); otherwise both arms count. Then only fixed-width forms are bounded (and, or, shifts by
	// constants, bit-field extracts of constant width); UINT32_MAX otherwise. The offset is at most these bits.
	uint32_t VectorOffsetBits(Value value, Value active, uint32_t depth = 0) const {
		if (depth > 32u) return UINT32_MAX;
		value = value.Resolve();
		if (value.IsImmediate()) return value.GetType() == Type::U32 ? value.U32() : UINT32_MAX;
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return UINT32_MAX;
		const auto arg = [&](size_t index) { return VectorOffsetBits(inst->Arg(index), active, depth + 1u); };
		switch (inst->GetOpcode()) {
			case ValueOpcode::SelectU32: {
				if (inst->NumArgs() != 3u) return UINT32_MAX;
				const auto condition = inst->Arg(0).Resolve();
				if ((!active.IsEmpty() && condition == active.Resolve()) || TrueOnEveryLane(condition)) return arg(1);
				return arg(1) | arg(2);
			}
			case ValueOpcode::BitwiseAnd32: return inst->NumArgs() == 2u ? arg(0) & arg(1) : UINT32_MAX;
			case ValueOpcode::BitwiseOr32: return inst->NumArgs() == 2u ? arg(0) | arg(1) : UINT32_MAX;
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftRightLogical32: {
				const auto shift = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
				if (!shift.IsImmediate() || shift.GetType() != Type::U32) return UINT32_MAX;
				const auto bits = arg(0);
				return inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 ? bits << (shift.U32() & 31u)
				                                                            : bits >> (shift.U32() & 31u);
			}
			case ValueOpcode::BitFieldUExtract: {
				const auto count = inst->NumArgs() == 3u ? inst->Arg(2).Resolve() : Value {};
				return count.IsImmediate() && count.GetType() == Type::U32 && count.U32() < 32u
				           ? (1u << count.U32()) - 1u
				           : UINT32_MAX;
			}
			default: return UINT32_MAX;
		}
	}

	// Record-keyed V#s: the bytes one BDA access through a record-keyed V# can touch before its index term,
	// [first, end) from the V# base. Only accesses the BDA emitters implement exactly (raw loads, stores and 32-bit
	// atomics, formatted loads) with fixed vector and scalar offsets: CalculateBufferAddress adds those and the
	// instruction offset in 32 bits, and none of them may wrap. `indexed`: the access adds a lane index.
	// Composed V#s: the vector offset may instead be a value VectorOffsetBits bounds; the access
	// then covers [offsets, offsets + bits + its bytes).
	bool RecordKeyedAccess(const Inst& inst, uint32_t& first, uint64_t& end, bool& indexed,
	                       BufferAccess& access) const {
		const auto op = inst.GetOpcode();
		access        = BufferAccessOf(op);
		if (access == BufferAccess::None || op == ValueOpcode::ReadConstBuffer || inst.NumArgs() < 5u) return false;
		const auto index = inst.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) return false;
		const auto& memory  = m_program.memory_info[index];
		uint32_t    voffset = 0, soffset = 0, element = 1u, voffset_bits = 0;
		if (memory.kind != ResourceKind::Buffer ||
		    !(access == BufferAccess::Read    ? memory.SupportsIndirectBufferLoad(op)
		      : access == BufferAccess::Write ? memory.SupportsUntrackedIndirectBufferStore(op) ||
		                                            memory.SupportsRecordKeyedFormattedStore(op)
		                                      : memory.SupportsUntrackedIndirectBufferAtomic(op)) ||
		    !ImmediateU32(inst.Arg(3), soffset))
			return false;
		if (!ImmediateU32(inst.Arg(2), voffset)) {
			voffset      = 0u;
			voffset_bits = VectorOffsetBits(inst.Arg(2), inst.Arg(inst.NumArgs() - 1u));
			if (voffset_bits > MaxComposedVectorOffsetBits) return false;
		}
		// A formatted load reads one element of its runtime format (at most 16 bytes), a raw access its data; whole
		// dwords around them are covered where the ranges are aligned (MaterializeIndirectRanges).
		const uint64_t bytes = memory.formatted                                                          ? 16u
		                       : op == ValueOpcode::LoadBufferU8 || op == ValueOpcode::StoreBufferU8     ? 1u
		                       : op == ValueOpcode::LoadBufferU16 || op == ValueOpcode::StoreBufferU16   ? 2u
		                                                                                                  : 4u * std::max(BufferComponentCount(op), 1u);
		const uint64_t offset = uint64_t {voffset} + memory.offset + soffset;
		if (offset + voffset_bits + bytes > uint64_t {UINT32_MAX} + 1u) return false;
		first   = static_cast<uint32_t>(offset);
		end     = offset + voffset_bits + bytes;
		indexed = !ImmediateU32(inst.Arg(1), element) || element != 0u;
		return true;
	}

	// Path proofs: the U1 terms `guard` is a conjunction of (each holds on every lane it admits).
	void GuardTerms(Value guard, std::vector<const Inst*>& terms) const {
		guard            = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd && inst->NumArgs() == 2u) {
			GuardTerms(inst->Arg(0), terms);
			GuardTerms(inst->Arg(1), terms);
		} else {
			terms.push_back(inst);
		}
	}

	// Path proofs: the non-phi values a phi web resolves to (false: it reaches an immediate).
	static bool PhiLeaves(Value value, std::vector<const Inst*>& leaves, std::unordered_set<const Inst*>& seen) {
		const auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr) return false;
		if (!seen.insert(inst).second) return true;
		if (inst->GetOpcode() != ValueOpcode::Phi) {
			if (std::ranges::find(leaves, inst) == leaves.end()) leaves.push_back(inst);
			return true;
		}
		for (size_t arm = 0; arm < inst->NumArgs(); ++arm) {
			if (!PhiLeaves(inst->Arg(arm), leaves, seen)) return false;
		}
		return inst->NumArgs() != 0u;
	}

	struct PathProof {
		uint32_t                 wanted = 0;
		uint32_t                 budget = 4096u;
		std::vector<const Inst*> leaves;
		std::set<std::tuple<const Inst*, const Inst*, std::vector<const Inst*>>> seen;
	};

	// Path proofs: adds to proof.leaves the reads `word` can hold where `state` equals proof.wanted.
	// `state` and `word` hold together: a phi of `state` pairs each arm with the arm of the same edge of a `word` phi
	// in its block (one entry into the block sets both), else with `word` unchanged. A select on a mask in `masks` (each
	// true on the lane) gives its new value there. A phi arm on a loop back edge drops the masks the loop defines (the
	// arm is an earlier iteration's). False when `state` reaches a value that is no constant, select or phi.
	bool PairedLeaves(Value state, Value word, std::vector<const Inst*> masks, PathProof& proof) const {
		if (proof.budget == 0u) return false;
		--proof.budget;
		state = state.Resolve();
		word  = word.Resolve();
		if (state.IsImmediate()) {
			if (state.GetType() != Type::U32) return false;
			if (state.U32() != proof.wanted) return true;
			std::unordered_set<const Inst*> phis;
			return PhiLeaves(word, proof.leaves, phis);
		}
		const auto* inst = state.TryInstruction();
		if (inst == nullptr) return false;
		if (!proof.seen.emplace(inst, word.TryInstruction(), masks).second) return true;
		switch (inst->GetOpcode()) {
			case ValueOpcode::SelectU32: {
				// The mask holds on the lane when each of its conjunction terms is one of `masks`.
				std::vector<const Inst*> terms;
				GuardTerms(inst->Arg(0), terms);
				const bool held = !terms.empty() && std::ranges::all_of(terms, [&](const Inst* term) {
					return std::ranges::any_of(
					    masks, [&](const Inst* mask) { return mask == term; }); // the same SSA value, not a re-read
				});
				return PairedLeaves(inst->Arg(1), word, masks, proof) &&
				       (held || PairedLeaves(inst->Arg(2), word, masks, proof));
			}
			case ValueOpcode::Phi: {
				const auto* paired = word.TryInstruction();
				if (paired != nullptr && (paired->GetOpcode() != ValueOpcode::Phi || paired->Parent() != inst->Parent()))
					paired = nullptr;
				for (size_t arm = 0; arm < inst->NumArgs(); ++arm) {
					Value arm_word = word;
					if (paired != nullptr) {
						arm_word = {};
						for (size_t other = 0; other < paired->NumArgs(); ++other) {
							if (paired->PhiBlock(other) == inst->PhiBlock(arm)) arm_word = paired->Arg(other);
						}
						if (arm_word.IsEmpty()) return false;
					}
					auto kept = masks;
					if (inst->PhiBlock(arm) == nullptr || BlockDominates(inst->Parent(), inst->PhiBlock(arm))) {
						std::erase_if(kept, [&](const Inst* mask) { return BlockDominates(inst->Parent(), mask->Parent()); });
					}
					if (!PairedLeaves(inst->Arg(arm), arm_word, std::move(kept), proof)) return false;
				}
				return inst->NumArgs() != 0u;
			}
			default: return false;
		}
	}

	// Path proofs: the handle's V# words are phis of reads, and every access through it runs only where
	// some VGPR state equals a constant (a term of its EXEC); pairing that state with each word leaves one read, the same
	// for every access. `candidate` holds those reads and must be a valid direct source (e.g. fixed SRT reads).
	bool MatchProvenDescriptor(const Inst& handle, DescriptorSource& candidate) {
		candidate.dword_count = 4u;
		for (uint32_t word = 0; word < 4u; ++word) {
			const Inst* read = nullptr;
			for (const auto& use: handle.Uses()) {
				std::vector<const Inst*> masks;
				GuardTerms(use.user->Arg(use.user->NumArgs() - 1u), masks);
				bool proven = false;
				for (const auto* test: masks) {
					if (test->GetOpcode() != ValueOpcode::IEqual32 || test->NumArgs() != 2u) continue;
					for (uint32_t arg = 0; arg < 2u && !proven; ++arg) {
						PathProof proof;
						if (!ImmediateU32(test->Arg(arg), proof.wanted)) continue;
						if (!PairedLeaves(test->Arg(arg ^ 1u), handle.Arg(word), masks, proof)) continue;
						if (proof.leaves.size() != 1u) continue;
						if (read != nullptr && read != proof.leaves[0]) return false; // the accesses disagree
						read   = proof.leaves[0];
						proven = true;
					}
					if (proven) break;
				}
				if (!proven) return false;
			}
			candidate.dwords[word] = LowerDescriptorValue(Value(const_cast<Inst*>(read)));
		}
		uint32_t bad_dword = 0;
		return ValidateSource(candidate, bad_dword); // the proven reads must be a direct V#
	}

	// Composed V#s: `value` as the host evaluates a V# word the shader composes. A select whose condition is
	// true on every lane (TrueOnEveryLane: a VGPR written under a full EXEC) is its new value, a conjunction with such a
	// condition is its other term, an invariant phi is its value; any other instruction is copied with its operands
	// projected (host-only copies in value_storage, as LowerDescriptorValue makes: the GPU program is unchanged). Every
	// step is an equality on every lane, so the projected word equals the GPU's word on any lane it is read from.
	Value ProjectLaneUniform(Value value, uint32_t depth = 0) {
		value      = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr || depth > 256u) return value;
		if (const auto found = m_lane_projections.find(inst); found != m_lane_projections.end()) return found->second;
		Value      result = value;
		const auto op     = inst->GetOpcode();
		if ((op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectF32) &&
		    inst->NumArgs() == 3u && TrueOnEveryLane(inst->Arg(0))) {
			result = ProjectLaneUniform(inst->Arg(1), depth + 1u);
		} else if (op == ValueOpcode::LogicalAnd && inst->NumArgs() == 2u && TrueOnEveryLane(inst->Arg(0))) {
			result = ProjectLaneUniform(inst->Arg(1), depth + 1u);
		} else if (op == ValueOpcode::LogicalAnd && inst->NumArgs() == 2u && TrueOnEveryLane(inst->Arg(1))) {
			result = ProjectLaneUniform(inst->Arg(0), depth + 1u);
		} else if (op == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (!invariant.IsEmpty() && invariant.Resolve() != value)
				result = ProjectLaneUniform(invariant, depth + 1u);
		} else {
			Inst* projected = nullptr;
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
				const auto original = inst->Arg(arg).Resolve();
				if (original.TryInstruction() == nullptr) continue;
				const auto lowered = ProjectLaneUniform(original, depth + 1u);
				if (lowered == original) continue;
				if (projected == nullptr) {
					projected = &m_program.value_storage.emplace_back(op, inst->Flags<uint64_t>());
					for (size_t i = 0; i < inst->NumArgs(); ++i) projected->SetArg(i, inst->Arg(i));
				}
				projected->SetArg(arg, lowered);
			}
			if (projected != nullptr) result = Value(projected);
		}
		m_lane_projections.emplace(inst, result);
		return result;
	}

	// Composed V#s: the handle's V# (`descriptor`: its words as MakeSource lowered them) composed from
	// uniform values: every word, projected (ProjectLaneUniform), is a value the host evaluates (ValidateSource). The
	// words are ReadFirstLane results, so the V# is one for the whole wave and the candidate is exactly it.
	bool MatchComposedDescriptor(const DescriptorSource& descriptor, DescriptorSource& candidate) {
		if (descriptor.dword_count != 4u) return false;
		candidate.dword_count = 4u;
		for (uint32_t word = 0; word < 4u; ++word) candidate.dwords[word] = ProjectLaneUniform(descriptor.dwords[word]);
		uint32_t bad_dword = 0;
		return ValidateSource(candidate, bad_dword);
	}

	// Composed V#s: `value` as opcodes and immediates to `depth` levels, at most ~4 KiB.
	static void DescribeValueTree(Value value, uint32_t depth, std::string& out) {
		if (out.size() > 4096u) return;
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			out += value.IsImmediate() && value.GetType() == Type::U32 ? fmt::format("{:x}", value.U32())
			       : value.IsImmediate() && value.GetType() == Type::U1 ? (value.U1() ? "true" : "false")
			                                                             : "imm";
			return;
		}
		out += ValueOpcodeName(inst->GetOpcode());
		if (depth == 0u || inst->NumArgs() == 0u) return;
		out += "(";
		for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
			if (arg != 0u) out += ",";
			DescribeValueTree(inst->Arg(arg), depth - 1u, out);
		}
		out += ")";
	}

	// Record-keyed V#s: a V# that a bounded loop reads from the records of a table V# and that feeds raw BDA
	// stores or atomics (RecordKeyedAccess for every use, at least one write). Every V# dword is ReadConstBuffer(table,
	// key * stride + field) of one table at consecutive fields, and every read runs only while the loop key is below
	// its runtime count (BoundedLoop on the read's block), so the GPU can only select one of the count's candidate
	// records: the host lists their ranges. The reads and the handle stay (the BDA emitters decode the V# at run time).
	bool TryMakeRecordKeyedBuffer(Inst& handle, const DescriptorSource& descriptor, IndirectDescriptorPlan& plan) {
		if (handle.GetOpcode() != ValueOpcode::GetBufferResource || handle.NumArgs() != 4u || handle.Uses().empty())
			return false;
		bool written = false;
		for (const auto& use: handle.Uses()) {
			uint32_t     first   = 0;
			uint64_t     end     = 0;
			bool         indexed = false;
			BufferAccess access  = BufferAccess::None;
			if (use.operand != 0u || !RecordKeyedAccess(*use.user, first, end, indexed, access)) return false;
			written |= access != BufferAccess::Read;
		}
		// Loads alone keep their plain BDA path: only writes need their guest ranges known.
		if (!written) return false;
		DescriptorSource table_source;
		Value            key;
		uint32_t         table_offset = 0, table_stride = 0, table_immediate = 0;
		plan.range_only = true;
		if (!MatchDescriptorTable(handle, descriptor, plan, table_source, key, table_offset, table_stride,
		                          table_immediate)) {
			// Path proofs: a V# the path picks among fixed reads is the one read its accesses can
			// see: a direct candidate source, read at every materialization.
			DescriptorSource candidate;
			bool             composed = false;
			if (!MatchProvenDescriptor(handle, candidate)) {
				// Composed V#s: a V# the shader composes from uniform values is the one V# its accesses
				// use: a direct candidate source as well.
				candidate = {};
				if (!MatchComposedDescriptor(descriptor, candidate)) return false;
				composed = true;
			}
			DescriptorSource::IndirectDescriptor indirect;
			indirect.sources      = {InternSource(candidate)};
			indirect.record_keyed = true;
			indirect.composed     = composed;
			plan                  = {};
			plan.range_only       = true;
			plan.retain_reads     = true;
			DescriptorSource source;
			source.dword_count = 4u;
			source.dwords.fill(Value(0u));
			source.indirect_descriptor = indirect;
			plan.handle = &handle;
			plan.source = InternSource(source);
			plan.key    = Value(0u);
			return true;
		}
		// Record-keyed address tables: a 64-bit address table (two dwords) is taken with a bounding loop below.
		const bool address_table = table_source.dword_count == 2u;
		if (plan.table_indexed || plan.pointer_record || plan.split_offsets ||
		    (table_source.dword_count != 4u && !address_table) || table_stride < 16u || (table_stride & 3u) != 0u ||
		    (table_offset & 3u) != 0u)
			return false;
		const auto* bound = BoundedLoop(key, plan.reads[0]->Parent());
		DescriptorSource::IndirectDescriptor indirect;
		// Game counts: without a loop, the game's own count of the address table's filled records.
		const bool game_count = bound == nullptr && address_table &&
		                        MatchGameRecordCount(uint64_t {table_offset} + table_immediate, table_stride, indirect);
		// Value bounds: without a loop, the key's own form may bound it (e.g. a select of constants).
		constexpr uint32_t MaxValueBoundKey = 65535u;
		uint32_t           key_max          = UINT32_MAX;
		if (bound == nullptr && !game_count) {
			// A ReadFirstLane key reads a lane of its mask that is active whenever an access through the V# executes (its
			// EXEC implies the mask; an empty mask writes nothing), so selects on the mask bound only their new value.
			Value       bounded = key, active;
			const auto* lane    = key.Resolve().TryInstruction();
			if (lane != nullptr && lane->GetOpcode() == ValueOpcode::ReadFirstLane && lane->NumArgs() == 2u &&
			    std::ranges::all_of(handle.Uses(), [&](const auto& use) {
				    return Implies(use.user->Arg(use.user->NumArgs() - 1u), lane->Arg(1));
			    })) {
				bounded = lane->Arg(0);
				active  = lane->Arg(1).Resolve();
			}
			BoundMemo memo;
			key_max = BoundedU32Max(
			    bounded, 0u, active,
			    [&](const Inst& phi, size_t arm) {
				    // An unknown predecessor counts as a back edge (the mask is dropped).
				    return phi.PhiBlock(arm) == nullptr || BlockDominates(phi.Parent(), phi.PhiBlock(arm));
			    },
			    memo);
		}
		const bool value_bound = key_max <= MaxValueBoundKey;
		// All-records lists: otherwise every record of the table is a candidate. An address table has no
		// size: only a loop or the key's value bounds its records.
		const bool all_records =
		    bound == nullptr && !game_count && !value_bound && !address_table;
		if (bound == nullptr && !game_count && !value_bound && !all_records) return false;
		for (uint32_t word = 1; bound != nullptr && word < 4u; ++word) {
			// The same block gives the same proof (the words are usually read together).
			if (plan.reads[word]->Parent() != plan.reads[0]->Parent() &&
			    BoundedLoop(key, plan.reads[word]->Parent()) != bound)
				return false;
		}
		indirect.table_source    = InternSource(table_source);
		indirect.table_offset    = table_offset;
		indirect.table_immediate = table_immediate;
		indirect.table_stride    = table_stride;
		if (bound != nullptr) indirect.key_count = bound->Arg(1);
		if (value_bound) indirect.key_count = Value(key_max + 1u); // keys 0..key_max
		indirect.all_records     = all_records;
		indirect.record_keyed    = true;
		indirect.unsigned_bound  = value_bound || (bound != nullptr && bound->GetOpcode() == ValueOpcode::ULessThan32);
		indirect.widened         = value_bound || all_records;
		// The V# words stay GPU loads (PrepareIndirectBuffer reads them) and are no planning memory.
		plan.retain_reads = true;
		plan.reads.fill(nullptr);
		DescriptorSource source;
		source.dword_count = 4u;
		source.dwords.fill(Value(0u));
		source.indirect_descriptor = indirect;
		plan.handle = &handle;
		plan.source = InternSource(source);
		plan.key    = key;
		return true;
	}

	// Record-keyed V#s: stores and atomics through a V# that no other form resolves.
	void PlanRecordKeyedBuffers() {
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				const auto access = BufferAccessOf(inst.GetOpcode());
				if ((access != BufferAccess::Write && access != BufferAccess::Atomic) || inst.NumArgs() == 0u) continue;
				auto*      handle = inst.Arg(0).Resolve().TryInstruction();
				const auto flags  = inst.Flags<MemoryFlags>();
				if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
				    handle->NumArgs() != 4u || FindIndirectDescriptor(*handle) != nullptr ||
				    flags.index >= m_program.memory_info.size())
					continue;
				DescriptorSource descriptor;
				MakeSource(*handle, 4u, false, false, m_program.memory_info[flags.index].resource * 4u, descriptor,
				           flags.pc);
				uint32_t bad_dword = 0;
				if (ValidateSource(descriptor, bad_dword)) continue;
				IndirectDescriptorPlan plan;
				if (TryMakeRecordKeyedBuffer(*handle, descriptor, plan)) {
					m_indirect_descriptors.push_back(std::move(plan));
				}
			}
		}
	}

	using SelectorBounds = std::span<const std::pair<const Inst*, uint32_t>>;

	bool SelectorPredicateFalse(Value value, SelectorBounds bounds, uint32_t depth = 0) const {
		if (depth > 32u) return false;
		value = value.Resolve();
		if (value.IsImmediate()) return value.GetType() == Type::U1 && !value.U1();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd)
			return SelectorPredicateFalse(inst->Arg(0), bounds, depth + 1u) ||
			       SelectorPredicateFalse(inst->Arg(1), bounds, depth + 1u);
		if (inst->GetOpcode() == ValueOpcode::IEqual32) {
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				uint32_t target;
				if (ImmediateU32(inst->Arg(arg), target) &&
				    SelectorMaximum(inst->Arg(arg ^ 1u), bounds, depth + 1u) < target) return true;
			}
		}
		return false;
	}

	uint32_t SelectorMaximum(Value value, SelectorBounds bounds = {}, uint32_t depth = 0) const {
		if (depth > 32u) return UINT32_MAX;
		value = value.Resolve();
		if (value.IsImmediate() && value.GetType() == Type::U32) return value.U32();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return UINT32_MAX;
		for (const auto& [phi, maximum]: bounds)
			if (inst == phi) return maximum;
		const auto maximum = [&](uint32_t arg) {
			return SelectorMaximum(inst->Arg(arg), bounds, depth + 1u);
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::FindILsb32:
				return NonzeroOnEntry(inst->Arg(0), inst->Parent()) ? 31u : UINT32_MAX;
			case ValueOpcode::UMin32:
			case ValueOpcode::BitwiseAnd32: return std::min(maximum(0), maximum(1));
			case ValueOpcode::SelectU32:
				return SelectorPredicateFalse(inst->Arg(0), bounds, depth + 1u)
				           ? maximum(2) : std::max(maximum(1), maximum(2));
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftLeftLogical32: {
				uint32_t shift;
				if (!ImmediateU32(inst->Arg(1), shift) || shift >= 32u) break;
				if (inst->GetOpcode() == ValueOpcode::ShiftRightLogical32) return maximum(0) >> shift;
				const auto result = uint64_t {maximum(0)} << shift;
				if (result <= UINT32_MAX) return static_cast<uint32_t>(result);
				break;
			}
			default: break;
		}
		return UINT32_MAX;
	}

	bool ImpossibleSwitchEdge(const Block* from, const Block* to) const {
		const auto& term = from->terminator;
		if (term.kind != CFG::TerminatorKind::IndirectBranch || !term.indexed) return false;
		const auto maximum = SelectorMaximum(from->indirect_target);
		bool found = false;
		for (const auto& branch: term.cases) {
			if (branch.target != to) continue;
			if (branch.value <= maximum) return false;
			found = true;
		}
		return found;
	}

	// Lane-selected descriptors: an image descriptor a waterfall reads lane by lane
	// (every word a ReadFirstLane under one mask) whose lane values are each lane's pick from a finite set (lane selects
	// on one condition per pick, or branches) is planned like a branch-selected one: every pick is a candidate, the key
	// is the lane's pick index read through the waterfall's ReadFirstLane, so it names the candidate whose words the
	// iteration reads (terrain deformation CS 7ceb0f34, 4 tile T#s 440 bytes apart).
	bool TryMakeFiniteDescriptor(Inst& handle, IndirectDescriptorPlan& plan) {
		const bool sampler = handle.GetOpcode() == ValueOpcode::GetSamplerResource;
		if ((!sampler && handle.GetOpcode() != ValueOpcode::GetImageResource) ||
		    handle.NumArgs() != (sampler ? 4u : 8u))
			return false;
		if (std::ranges::any_of(handle.Uses(), [&](const Use& use) {
			const auto op = use.user->GetOpcode();
			if (op != ValueOpcode::ImageSampleRaw) return sampler || op != ValueOpcode::ImageGatherRaw;
			return sampler && (m_program.memory_info[use.user->Flags<MemoryFlags>().index]
			                       .image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0u;
		})) return false;
		// Lane-selected descriptors: the waterfall's ReadFirstLane of every word (one mask), unwrapped.
		DescriptorSource root;
		root.dword_count = handle.NumArgs();
		for (uint32_t word = 0; word < root.dword_count; ++word) root.dwords[word] = handle.Arg(word);
		const Inst* lane_read = nullptr;
		if (!sampler) {
			bool all = true;
			for (uint32_t word = 0; word < root.dword_count && all; ++word) {
				const auto* read = root.dwords[word].Resolve().TryInstruction();
				all = read != nullptr && read->GetOpcode() == ValueOpcode::ReadFirstLane && read->NumArgs() == 2u &&
				      (lane_read == nullptr || EquivalentValue(m_program, lane_read->Arg(1), read->Arg(1)));
				if (all && lane_read == nullptr) lane_read = read;
			}
			if (all) {
				for (uint32_t word = 0; word < root.dword_count; ++word)
					root.dwords[word] = root.dwords[word].Resolve().TryInstruction()->Arg(0);
			} else {
				lane_read = nullptr;
			}
		}
		const auto is_pick = [&](Value value) {
			const auto* inst = value.Resolve().TryInstruction();
			return inst != nullptr && (inst->GetOpcode() == ValueOpcode::Phi ||
			                           (lane_read != nullptr && inst->GetOpcode() == ValueOpcode::SelectU32));
		};
		bool has_phi = false;
		for (uint32_t word = 0; word < root.dword_count; ++word) has_phi |= is_pick(root.dwords[word]);
		if (!has_phi) { TraceIndirectReject(__LINE__); return false; }
		// Normalize synchronized descriptor words to one GPU ordinal.
		// Build the complete graph before changing IR so an unsupported leaf is transactional.
		struct Choice {
			DescriptorSource descriptor;
			const Inst* branch = nullptr;
			std::vector<uint32_t> children;
			Value key;
		};
		std::vector<Choice> choices;
		const auto visit = [&](const auto& self, DescriptorSource descriptor) -> uint32_t {
			for (auto& word: descriptor.dwords) word = word.Resolve();
			for (uint32_t i = 0; i < choices.size(); ++i)
				if (choices[i].descriptor.dwords == descriptor.dwords) return i;
			const auto index = static_cast<uint32_t>(choices.size());
			choices.push_back({descriptor});
			uint32_t bad = 0;
			if (ValidateSource(descriptor, bad)) {
				// Finite inline samplers have literal leaves; live descriptor tables use the
				// existing direct sampler planning path.
				if (sampler && !std::ranges::all_of(
				    std::span(descriptor.dwords).first(descriptor.dword_count),
				    [](Value word) { return word.IsImmediate() && word.GetType() == Type::U32; }))
					return UINT32_MAX;
				return index;
			}
			const auto* branch = descriptor.dwords[bad].TryInstruction();
			// Lane-selected descriptors: a lane select is a two-way pick (arm 0 its true value, arm 1 its false
			// value); every word that is a select must select on the same condition.
			if (lane_read != nullptr && branch != nullptr && branch->Parent() != nullptr &&
			    branch->GetOpcode() == ValueOpcode::SelectU32) {
				choices[index].branch = branch;
				for (uint32_t arm = 0; arm < 2u; ++arm) {
					auto child = descriptor;
					for (uint32_t word = 0; word < child.dword_count; ++word) {
						const auto* selected = descriptor.dwords[word].TryInstruction();
						if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32) {
							if (!ValidateRuntimeValue(m_program, descriptor.dwords[word])) return UINT32_MAX;
							continue;
						}
						if (!EquivalentValue(m_program, selected->Arg(0), branch->Arg(0))) return UINT32_MAX;
						child.dwords[word] = selected->Arg(1u + arm);
					}
					const auto next = self(self, child);
					if (next == UINT32_MAX) return UINT32_MAX;
					choices[index].children.push_back(next);
				}
				return index;
			}
			if (branch == nullptr || branch->Parent() == nullptr ||
			    branch->GetOpcode() != ValueOpcode::Phi ||
			    branch->NumPhiBlocks() != branch->NumArgs()) return UINT32_MAX;
			choices[index].branch = branch;
			const auto count = branch->NumArgs();
			for (uint32_t arm = 0; arm < count; ++arm) {
				// A nonzero bit scan cannot choose its zero-input switch sentinel.
				// Preserve the Phi edge; its ordinal is never observed at runtime.
				if (ImpossibleSwitchEdge(branch->PhiBlock(arm), branch->Parent())) {
					choices[index].children.push_back(UINT32_MAX);
					continue;
				}
				auto child = descriptor;
				for (uint32_t word = 0; word < child.dword_count; ++word) {
					const auto* selected = descriptor.dwords[word].TryInstruction();
					if (selected == nullptr || selected->GetOpcode() != branch->GetOpcode() ||
					    selected->Parent() != branch->Parent()) {
						if (!ValidateRuntimeValue(m_program, descriptor.dwords[word])) return UINT32_MAX;
						continue;
					}
					if (selected->NumArgs() != count || selected->NumPhiBlocks() != count)
						return UINT32_MAX;
					uint32_t incoming = 0;
					while (incoming < count && selected->PhiBlock(incoming) != branch->PhiBlock(arm))
						++incoming;
					if (incoming == count) return UINT32_MAX;
					child.dwords[word] = selected->Arg(incoming);
				}
				const auto next = self(self, child);
				if (next == UINT32_MAX) return UINT32_MAX;
				choices[index].children.push_back(next);
			}
			return index;
		};
		if (visit(visit, root) == UINT32_MAX || choices.front().branch == nullptr) { TraceIndirectReject(__LINE__); return false; }
		DescriptorSource finite_source;
		finite_source.dword_count = root.dword_count;
		finite_source.dwords.fill(Value(0u));
		finite_source.indirect_descriptor.emplace(DescriptorSource::IndirectDescriptor{});
		auto& sources = finite_source.indirect_descriptor->sources;
		for (auto& choice: choices) {
			if (choice.branch != nullptr) continue;
			const auto source = InternSource(choice.descriptor);
			auto found = std::ranges::find(sources, source);
			choice.key = Value(static_cast<uint32_t>(found - sources.begin()));
			if (found == sources.end()) sources.push_back(source);
		}
		if (sources.empty()) { TraceIndirectReject(__LINE__); return false; }
		for (auto& choice: choices) {
			if (choice.branch == nullptr) continue;
			auto* block = choice.branch->Parent();
			const auto where = std::ranges::find_if(block->Instructions(), [&](const Inst& inst) {
				return &inst == choice.branch;
			});
			if (choice.branch->GetOpcode() == ValueOpcode::SelectU32) {
				// Lane-selected descriptors: the pick's key, selected on the pick's condition (arms set below).
				auto& key = *block->PrependNewInst(where, ValueOpcode::SelectU32,
				                                   {choice.branch->Arg(0), Value(0u), Value(0u)});
				choice.key = Value(&key);
				continue;
			}
			auto& key = *block->PrependNewInst(where, choice.branch->GetOpcode());
			key.SetFlags(Type::U32);
			choice.key = Value(&key);
		}
		for (const auto& choice: choices) {
			if (choice.branch == nullptr) continue;
			auto* key = choice.key.Instruction();
			for (uint32_t arm = 0; arm < choice.children.size(); ++arm) {
				const auto child = choice.children[arm];
				const auto value = child == UINT32_MAX ? Value(0u) : choices[child].key;
				if (choice.branch->GetOpcode() == ValueOpcode::SelectU32) {
					key->SetArg(1u + arm, value);
				} else {
					key->AddPhiOperand(choice.branch->PhiBlock(arm), value);
				}
			}
		}
		plan.handle = &handle;
		plan.source = InternSource(finite_source);
		plan.key = choices.front().key;
		if (lane_read != nullptr) {
			// The key of the lane whose words this waterfall iteration reads: the same ReadFirstLane, on the key.
			auto*      block = lane_read->Parent();
			const auto where = std::ranges::find_if(block->Instructions(), [&](const Inst& inst) { return &inst == lane_read; });
			auto&      read  = *block->PrependNewInst(where, ValueOpcode::ReadFirstLane, {plan.key, lane_read->Arg(1)});
			plan.key         = Value(&read);
		}
		plan.roots = finite_source.dwords;
		plan.reads.fill(nullptr);
		return true;
	}

	const IndirectDescriptorPlan* FindIndirectDescriptor(const Inst& handle) const {
		const auto found =
		    std::find_if(m_indirect_descriptors.begin(), m_indirect_descriptors.end(),
		                 [&](const IndirectDescriptorPlan& plan) {
			    return plan.handle == &handle;
		    });
		return found == m_indirect_descriptors.end() ? nullptr : &*found;
	}

	bool IsIndirectPlanningMemory(uint32_t index) const {
		return std::any_of(m_indirect_descriptors.begin(), m_indirect_descriptors.end(),
		                   [&](const IndirectDescriptorPlan& plan) {
			if (plan.reads[0] == nullptr) return false;
			for (uint32_t word = 0; word < plan.handle->NumArgs(); ++word)
				if (plan.reads[word] != nullptr && plan.memory[word] == index) return true;
			return false;
		});
	}

	// The sampler of a sampled image operation may itself be GPU-selected from a keyed table.
	// An adjusted sampler (sample_adjust) rewrites dword3 in IR and keeps the existing refusal.
	void PlanIndirectSampler(Inst& inst, uint32_t pc) {
		if (!ImageOpcodeInfoOf(inst.GetOpcode()).needs_sampler || inst.NumArgs() < 2u) return;
		auto* handle = inst.Arg(1).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetSamplerResource ||
		    handle->NumArgs() != 4u || FindIndirectDescriptor(*handle) != nullptr) return;
		const auto& memory = m_program.memory_info[inst.Flags<MemoryFlags>().index];
		if ((memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0u) return;
		DescriptorSource descriptor;
		descriptor.dword_count = 4u;
		for (uint32_t word = 0; word < 4u; ++word)
			descriptor.dwords[word] = LowerDescriptorValue(NativeDescriptorSource(handle->Arg(word),
			                                                                    memory.sampler * 4u + word, pc));
		uint32_t bad_dword = 0;
		if (ValidateSource(descriptor, bad_dword)) return;
		IndirectDescriptorPlan plan;
		if (TryMakeIndirectImage(*handle, descriptor, plan)) m_indirect_descriptors.push_back(std::move(plan));
	}

	void PlanIndirectDescriptors() {
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if ((ImageOpcodeInfoOf(inst.GetOpcode()).access == ImageAccess::None &&
				     inst.GetOpcode() != ValueOpcode::StoreBufferU32) ||
				    inst.NumArgs() == 0u) {
					continue;
				}
				// GPU-selected samplers keyed by a table (our planner; the emitter maps the key to a slot).
				PlanIndirectSampler(inst, inst.Flags<MemoryFlags>().pc);
				const auto flags = inst.Flags<MemoryFlags>();
				const auto& memory = m_program.memory_info[flags.index];
				const auto plan_handle = [&](Value value, uint32_t width, uint32_t base, bool sampler) {
					auto* handle = value.Resolve().TryInstruction();
					if (handle == nullptr || FindIndirectDescriptor(*handle) != nullptr) return;
					DescriptorSource descriptor;
					MakeSource(*handle, width, sampler, false, base, descriptor, flags.pc);
					uint32_t bad_dword = 0;
					if (ValidateSource(descriptor, bad_dword)) return;
					IndirectDescriptorPlan plan;
					if (sampler ? TryMakeFiniteDescriptor(*handle, plan)
					             : (TryMakeIndirectImage(*handle, descriptor, plan) ||
					                TryMakeFiniteDescriptor(*handle, plan) ||
					                TryMakeIndirectBuffer(*handle, descriptor, plan))) {
						for (const auto& previous: m_indirect_descriptors) {
							if (plan.reads[0] != nullptr && previous.reads[0] == plan.reads[0] &&
							    !EquivalentValue(m_program, previous.key, plan.key))
								Fail(flags.pc, "shared descriptor read has incompatible keys");
						}
						m_indirect_descriptors.push_back(std::move(plan));
					}
				};
				// A sampler picked by branches over literal words: a finite list of candidates (upstream 8c813865).
				// An adjusted sampler rewrites dword3 in IR and keeps the existing refusal.
				if (inst.GetOpcode() == ValueOpcode::ImageSampleRaw &&
				    (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) == 0u) {
					plan_handle(inst.Arg(1), 4u, memory.sampler * 4u, true);
				}
				plan_handle(inst.Arg(0), inst.GetOpcode() == ValueOpcode::StoreBufferU32 ? 4u : 8u,
				            memory.resource * 4u, false);
			}
		}
		PlanRecordKeyedBuffers();
	}

	bool GetHandle(Value value, ValueOpcode expected, uint32_t width, uint32_t pc,
	               uint32_t base_reg, Inst*& handle, uint32_t& source, bool sampler = false,
	               bool sample_adjust = false) {
		handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != expected) {
			Fail(pc, fmt::format("memory operation requires {}", ValueOpcodeName(expected)));
		}
		DescriptorSource descriptor;
		MakeSource(*handle, width, sampler, sample_adjust, base_reg, descriptor, pc);
		uint32_t bad_dword = 0;
		if (!ValidateSource(descriptor, bad_dword)) {
			if (expected == ValueOpcode::GetBufferResource &&
			    std::all_of(descriptor.dwords.begin(), descriptor.dwords.begin() + width,
			                [](Value word) { return word.Resolve().GetType() == Type::U32; })) {
				return false;
			}
			// A descriptor whose words match no supported form stops translation.
			if (Common::EnvRead::IndirectImage()) { // the failing word's expression
				std::string tree;
				DescribeValueTree(descriptor.dwords[bad_dword], 12u, tree);
				std::fprintf(stderr, "TRACE_INDIRECT_IMAGE unresolved hash=0x%016llx pc=0x%08x dword=%u tree=%s\n",
				             static_cast<unsigned long long>(m_program.shader_hash), pc, bad_dword, tree.c_str());
			}
			Fail(pc, fmt::format("{} dword {} is not a valid runtime value", ValueOpcodeName(expected), bad_dword));
		}
		source = InternSource(descriptor);
		return true;
	}

	void ValidateAddressHandle(Value value, uint32_t pc) const {
		const auto* handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource) {
			Fail(pc, "address operation requires GetAddressResource");
		}
		if (handle->NumArgs() != 2) {
			Fail(pc, "GetAddressResource must have two address dwords");
		}
	}

	uint32_t AddBuffer(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc,
	                   std::optional<std::pair<uint32_t, uint32_t>> fixed_write) {
		for (uint32_t i = 0; i < m_info.buffers.size(); i++) {
			if (m_info.buffers[i].source == source) {
				Merge(m_info.buffers[i], memory, op, pc, fixed_write);
				return i;
			}
		}
		if (m_info.buffers.size() >= ShaderInfo::MaxBuffers) {
			return UINT32_MAX;
		}
		BufferResource resource;
		resource.source       = source;
		resource.first_use_pc = pc;
		Merge(resource, memory, op, pc, fixed_write);
		m_info.buffers.push_back(resource);
		return static_cast<uint32_t>(m_info.buffers.size() - 1);
	}

	static void Merge(BufferResource& resource, const MemoryInfo& memory, ValueOpcode op,
	                  uint32_t pc, std::optional<std::pair<uint32_t, uint32_t>> fixed_write) {
		const auto access        = BufferAccessOf(op);
		const bool atomic        = access == BufferAccess::Atomic;
		const bool write         = access == BufferAccess::Write || atomic;
		resource.first_use_pc    = std::min(resource.first_use_pc, pc);
		resource.max_byte_extent = std::max(resource.max_byte_extent, ByteExtent(memory));
		resource.read            = resource.read || !write || atomic;
		if (write) {
			// The first write starts the fixed list; a write without a fixed address ends it for good.
			if (!resource.written) resource.fixed_writes = true;
			auto& ranges = resource.fixed_write_ranges;
			if (resource.fixed_writes && fixed_write) {
				ranges.push_back(*fixed_write);
				std::ranges::sort(ranges);
				size_t kept = 0;
				for (const auto& range: ranges) {
					if (kept != 0 && range.first <= ranges[kept - 1].second) {
						ranges[kept - 1].second = std::max(ranges[kept - 1].second, range.second);
					} else {
						ranges[kept++] = range;
					}
				}
				ranges.resize(kept);
			}
			if (!fixed_write || ranges.size() > MaxFixedWriteRanges) {
				resource.fixed_writes = false;
				ranges.clear();
			}
		}
		resource.written         = resource.written || write;
		resource.atomic          = resource.atomic || atomic;
		resource.formatted       = resource.formatted || memory.formatted;
		resource.scalar          = resource.scalar || op == ValueOpcode::ReadConstBuffer ||
		                           memory.kind == ResourceKind::ScalarBuffer;
	}

	// split_field (KYTY_PERF_BINDLESS_CLASSES): the {stride, offset} of the bounded-table field the handle reads, when
	// that field gets a resource of its own; such a resource is shared only by handles of the same field.
	uint32_t AddImage(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc,
	                  std::optional<std::pair<uint32_t, uint32_t>> split_field = std::nullopt) {
		const auto resource_class = ImageOpcodeInfoOf(op).resource_class;
		const bool atomic64 = ImageOpcodeInfoOf(op).access == ImageAccess::Atomic &&
		                      memory.data_bits == 64u;
		const bool dynamic_mip =
		    (resource_class == ImageResourceClass::Storage && memory.image_has_mip) ||
		    (op == ValueOpcode::ImageGatherRaw &&
		     (memory.image_sample_flags & Decoder::ImageSampleFlagLod) != 0u);
		const auto mip = dynamic_mip ? ImageMipMode::Dynamic : ImageMipMode::None;
		const bool depth = (memory.image_sample_flags & Decoder::ImageSampleFlagCompare) != 0;
		const auto find  = [&](const std::optional<std::pair<uint32_t, uint32_t>>& field) {
			for (uint32_t i = 0; i < m_info.images.size(); i++) {
				auto& image = m_info.images[i];
				if (image.source == source && image.resource_class == resource_class &&
				    image.dimension == memory.image_dimension && image.mip_mode == mip &&
				    image.depth_compare == depth && image.r128 == memory.image_r128 &&
				    image.atomic64 == atomic64 && image.bindless_field_split == field.has_value() &&
				    (!field.has_value() ||
				     (image.bindless_fields.size() == 1u && image.bindless_fields.front() == *field))) {
					return i;
				}
			}
			return UINT32_MAX;
		};
		auto found = find(split_field);
		// A new field resource only while the shader is far from the resource limits (the shared resources it still
		// adds need room; MaxImages and MaxSampledPairs end the translation): past half of them the field shares.
		if (found == UINT32_MAX && split_field.has_value() &&
		    (m_info.images.size() >= ShaderInfo::MaxImages / 2u ||
		     m_info.sampled_pairs.size() >= ShaderInfo::MaxSampledPairs / 2u)) {
			GetBindlessClassCounters().split_refused.fetch_add(1, std::memory_order_relaxed);
			split_field.reset();
			found = find(split_field);
		}
		if (found != UINT32_MAX) {
			Merge(m_info.images[found], op, pc);
			return found;
		}
		if (m_info.images.size() >= ShaderInfo::MaxImages) {
			return UINT32_MAX;
		}
		ImageResource image;
		image.source         = source;
		image.first_use_pc   = pc;
		image.resource_class = resource_class;
		image.dimension      = memory.image_dimension;
		image.mip_mode       = mip;
		image.depth_compare  = depth;
		image.r128           = memory.image_r128;
		image.atomic64       = atomic64;
		if (split_field.has_value()) {
			image.bindless_field_split = true;
			image.bindless_fields.push_back(*split_field);
			m_splits_made++;
			GetBindlessClassCounters().split_resources.fetch_add(1, std::memory_order_relaxed);
			// BINDLESS_CLASS split: the first 64 field resources, with the shader and the instruction.
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 64u)
				std::fprintf(stderr,
				             "BINDLESS_CLASS split hash=0x%016llx pc=0x%08x image=%zu field(stride:offset)=%u:%u op=%s\n",
				             static_cast<unsigned long long>(m_program.shader_hash), pc, m_info.images.size(),
				             split_field->first, split_field->second, ValueOpcodeName(op).data());
		}
		Merge(image, op, pc);
		m_info.images.push_back(image);
		return static_cast<uint32_t>(m_info.images.size() - 1);
	}

	// KYTY_PERF_BINDLESS_CLASSES=1: whether an image load or a gather reads through the handle (directly or through
	// Identity): the instructions a shader reads integer texels with (an integer texture is never filtered, so a plain
	// sample of it is not expected).
	static bool HandleReadsTexels(const Inst& handle, uint32_t depth = 0) {
		for (const auto& use: handle.Uses()) {
			const auto op = use.user->GetOpcode();
			if (op == ValueOpcode::ImageRead || op == ValueOpcode::ImageGatherRaw) return true;
			if (op == ValueOpcode::Identity && depth < 4u && HandleReadsTexels(*use.user, depth + 1u)) return true;
		}
		return false;
	}

	static void Merge(ImageResource& image, ValueOpcode op, uint32_t pc) {
		const auto access  = ImageOpcodeInfoOf(op).access;
		const bool atomic  = access == ImageAccess::Atomic;
		const bool write   = access == ImageAccess::Write || atomic;
		image.first_use_pc = std::min(image.first_use_pc, pc);
		image.read         = image.read || !write || atomic;
		image.written      = image.written || write;
		image.atomic       = image.atomic || atomic;
	}

	uint32_t AddSampler(uint32_t source, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.samplers.size(); i++) {
			if (m_info.samplers[i].source == source) {
				m_info.samplers[i].first_use_pc = std::min(m_info.samplers[i].first_use_pc, pc);
				return i;
			}
		}
		if (m_info.samplers.size() >= ShaderInfo::MaxSamplers) {
			return UINT32_MAX;
		}
		m_info.samplers.push_back({source, pc});
		const auto root = static_cast<uint32_t>(m_info.samplers.size() - 1);
		const auto* descriptor = Source(source);
		// A finite sampler lists its candidates; a table-keyed one (PlanIndirectSampler) gets them from the
		// table's records at materialization.
		if (descriptor != nullptr && descriptor->indirect_descriptor &&
		    !descriptor->indirect_descriptor->sources.empty()) {
			const auto& sources = descriptor->indirect_descriptor->sources;
			std::vector<uint32_t> resources {root};
			for (size_t i = 1; i < sources.size(); ++i) {
				const auto candidate = AddSampler(sources[i], pc);
				if (candidate == UINT32_MAX) return UINT32_MAX;
				resources.push_back(candidate);
			}
			m_info.samplers[root].indirect_resources = std::move(resources);
		}
		return root;
	}

	void AddSampledPair(uint32_t image, uint32_t sampler, uint32_t pc) {
		for (auto& pair: m_info.sampled_pairs) {
			if (pair.image == image && pair.sampler == sampler) {
				pair.first_use_pc = std::min(pair.first_use_pc, pc);
				return;
			}
		}
		if (m_info.sampled_pairs.size() >= ShaderInfo::MaxSampledPairs) {
			if (m_splits_made != 0u) {
				m_split_overflow = true; // KYTY_PERF_BINDLESS_CLASSES: Run tracks the shader again without field resources
				return;
			}
			Fail(pc, "sampled image/sampler pair limit exceeded");
		}
		m_info.sampled_pairs.push_back({image, sampler, pc});
	}

	void AddHandlePatch(Inst* handle, uint32_t resource, uint32_t pc) {
		for (const auto& patch: m_handle_patches) {
			if (patch.handle == handle) {
				if (patch.resource != resource) {
					Fail(pc, fmt::format("{} is reused with incompatible resource classes",
					                     ValueOpcodeName(handle->GetOpcode())));
				}
				return;
			}
		}
		m_handle_patches.push_back({handle, resource});
	}

	void AddMemoryPatch(uint32_t index, uint32_t resource, uint32_t sampler, bool has_sampler,
	                    uint32_t pc) {
		for (auto& patch: m_memory_patches) {
			if (patch.index != index) {
				continue;
			}
			if (patch.resource != resource ||
			    (has_sampler && patch.has_sampler && patch.sampler != sampler)) {
				Fail(pc, "memory metadata is reused with incompatible resources");
			}
			if (has_sampler) {
				patch.sampler     = sampler;
				patch.has_sampler = true;
			}
			return;
		}
		m_memory_patches.push_back({index, resource, sampler, has_sampler});
	}

	[[noreturn]] void FailUnresolvedBuffer(uint32_t pc, ValueOpcode op, const MemoryInfo& memory) const {
		Fail(pc, fmt::format("buffer descriptor is not a valid runtime value; GPU-selected access "
		                     "requires a scalar, raw DWORD x1/x2/x3/x4, or formatted X load (op={} kind={} "
		                     "typed={} formatted={} bits={})", ValueOpcodeName(op), static_cast<int>(memory.kind),
		                     memory.typed ? 1 : 0, memory.formatted ? 1 : 0, memory.data_bits));
	}

	// Record-keyed V#s: one access through a record-keyed V# (TryMakeRecordKeyedBuffer), merged by source.
	void AddIndirectRange(uint32_t source, const Inst& inst, uint32_t pc) {
		uint32_t     first   = 0;
		uint64_t     end     = 0;
		bool         indexed = false;
		BufferAccess access  = BufferAccess::None;
		if (!RecordKeyedAccess(inst, first, end, indexed, access)) {
			Fail(pc, "record-keyed buffer access lost its fixed offsets");
		}
		auto range = std::ranges::find(m_info.indirect_ranges, source, &IndirectRangeResource::source);
		if (range == m_info.indirect_ranges.end()) {
			m_info.indirect_ranges.push_back({.source = source, .first_use_pc = pc});
			range = m_info.indirect_ranges.end() - 1;
		}
		range->first_use_pc = std::min(range->first_use_pc, pc);
		range->access_begin = std::min(range->access_begin, first);
		range->access_end   = std::max(range->access_end, end);
		range->indexed      = range->indexed || indexed;
		range->read         = range->read || access != BufferAccess::Write;
		range->written      = range->written || access != BufferAccess::Read;
		range->atomic       = range->atomic || access == BufferAccess::Atomic;
	}

	// Record-keyed V#s: record-keyed ranges only help a shader they leave without address writes. With other
	// writes through unknown addresses a plan that captures reads is refused anyway (ExtractResourcePlan), so the
	// stores fail as an untracked store through a GPU-selected V# does (D6: no store reaches memory unlisted).
	void DropRecordKeyedRanges() {
		if (!m_program.has_address_writes || m_info.indirect_ranges.empty()) return;
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				const auto access = BufferAccessOf(inst.GetOpcode());
				if (access == BufferAccess::None || access == BufferAccess::Read || inst.NumArgs() == 0u) continue;
				const auto* handle = inst.Arg(0).Resolve().TryInstruction();
				const auto* plan   = handle != nullptr ? FindIndirectDescriptor(*handle) : nullptr;
				if (plan == nullptr || !plan->range_only) continue;
				const auto flags  = inst.Flags<MemoryFlags>();
				auto       memory = m_program.memory_info[flags.index];
				memory.kind       = ResourceKind::Buffer;
				FailUnresolvedBuffer(flags.pc, inst.GetOpcode(), memory);
			}
		}
		for (const auto& range: m_info.indirect_ranges) {
			m_sources[range.source].indirect_descriptor.reset();
		}
		m_info.indirect_ranges.clear();
	}

	void Collect(Inst& inst) {
		if (m_split_overflow) return; // KYTY_PERF_BINDLESS_CLASSES: Run tracks the shader again
		const auto op           = inst.GetOpcode();
		if (op == ValueOpcode::BvhIntersect || op == ValueOpcode::Bvh8Intersect) {
			m_info.uses_dma = true;
			return;
		}
		const auto buffer       = BufferAccessOf(op);
		const auto address_info = AddressOpcodeInfoOf(op);
		const auto image_info   = ImageOpcodeInfoOf(op);
		if (buffer == BufferAccess::None && address_info.access == AddressAccess::None &&
		    image_info.access == ImageAccess::None) {
			return;
		}
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			Fail(flags.pc, fmt::format("memory metadata index {} is out of range", flags.index));
		}
		if (inst.NumArgs() == 0) {
			Fail(flags.pc, "memory operation has no resource handle");
		}
		const auto& memory = m_program.memory_info[flags.index];
		if (memory.planning_only || IsIndirectPlanningMemory(flags.index)) {
			return;
		}
		Inst*    handle   = nullptr;
		uint32_t source   = 0;
		uint32_t resource = 0;

		if (buffer != BufferAccess::None) {
			handle = inst.Arg(0).Resolve().TryInstruction();
			const auto* indirect = handle == nullptr ? nullptr : FindIndirectDescriptor(*handle);
			if (indirect != nullptr && indirect->range_only) {
				// Record-keyed V#s: a BDA access through a record-keyed V#. Every materialization lists the
				// guest ranges it can touch, so it is no access through an unknown address.
				AddIndirectRange(indirect->source, inst, flags.pc);
				m_program.memory_info[flags.index].kind         = ResourceKind::IndirectBuffer;
				m_program.memory_info[flags.index].record_keyed = true;
				m_info.uses_dma                                 = true;
				return;
			}
			if (indirect != nullptr) {
				source = indirect->source;
			} else if (!GetHandle(inst.Arg(0), ValueOpcode::GetBufferResource, 4, flags.pc,
			               memory.resource * 4u, handle, source)) {
				// Hardware behaviour (M48; RDNA2 MUBUF BUFFER_STORE_FORMAT_*: address, bounds and data/number format come
				// from the V# the GPU selected): a formatted 32-bit store through a GPU-selected descriptor is written
				// through BDA with the V# read at run time (still dropped and logged: D16 stores, channels wider than 32
				// bits). Any other formatted store through an unresolved descriptor stops translation below.
				const bool via_bda = memory.kind == ResourceKind::Buffer && memory.formatted && !memory.typed &&
				                     BufferAccessOf(op) == BufferAccess::Write && memory.data_bits == 32u &&
				                     (op == ValueOpcode::StoreBufferU32 || op == ValueOpcode::StoreBufferU32x2 ||
				                      op == ValueOpcode::StoreBufferU32x3 || op == ValueOpcode::StoreBufferU32x4);
				if (via_bda) {
					std::fprintf(stderr, "INDIRECT_FORMATTED_STORES hash=0x%016llx pc=0x%08x "
					             "formatted store through BDA (op=%s bits=%u components=%u)\n",
					             static_cast<unsigned long long>(m_program.shader_hash), flags.pc,
					             ValueOpcodeName(op).data(), memory.data_bits, memory.component_count);
					m_program.memory_info[flags.index].kind = ResourceKind::IndirectBuffer;
					m_info.uses_dma = true;
					// Not marked as an address write: the written buffer is a GPU-selected data buffer,
					// and the mark would invalidate captured specialization reads (e.g. two-level tables).
					return;
				}
				// D6: a store or atomic through a GPU-selected V# that is not record-keyed has no listing, so it stops
				// translation with its name.
				if (memory.kind != (op == ValueOpcode::ReadConstBuffer ? ResourceKind::ScalarBuffer
				                                                        : ResourceKind::Buffer) ||
				    !memory.SupportsIndirectBufferLoad(op)) {
					FailUnresolvedBuffer(flags.pc, op, memory);
				}
				m_program.memory_info[flags.index].kind = ResourceKind::IndirectBuffer;
				m_info.uses_dma                         = true;
				return;
			}
			resource = AddBuffer(source, memory, op, flags.pc,
			                     buffer == BufferAccess::Read ? std::nullopt : FixedWriteRange(inst, memory, op));
			if (resource == UINT32_MAX) {
				Fail(flags.pc, "buffer resource limit exceeded");
			}
			AddHandlePatch(handle, resource, flags.pc);
			AddMemoryPatch(flags.index, resource, 0, false, flags.pc);
			return;
		}
		if (address_info.access != AddressAccess::None) {
			if (!IsAddressResourceKind(memory.kind)) {
				Fail(flags.pc, "address operation has invalid resource kind");
			}
			if (memory.kind == ResourceKind::Flat && memory.address_is_full &&
			    IsNoncanonicalFlatAddress(inst.Arg(2), inst.Arg(inst.NumArgs() - 1))) {
				ValidateAddressHandle(inst.Arg(0), flags.pc);
				if (memory.data_bits != 32u || m_program.scratch_dwords == 0) {
					Fail(flags.pc, "local FLAT access requires DWORD data and per-thread scratch storage");
				}
				m_program.memory_info[flags.index].kind = ResourceKind::FlatLocal;
				return;
			}
			if (memory.kind == ResourceKind::Scratch) {
				handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetScratchResource ||
				    handle->NumArgs() != 0) {
					Fail(flags.pc, "scratch operation requires GetScratchResource");
				}
				if (m_program.scratch_dwords == 0) {
					Fail(flags.pc, "scratch operation requires a nonzero AGC per-thread size");
				}
				return;
			}
			ValidateAddressHandle(inst.Arg(0), flags.pc);
			if (address_info.access == AddressAccess::Write) {
				m_program.has_address_writes = true;
			}
			m_info.uses_dma = true;
			return;
		}

		if (memory.kind != ResourceKind::Image ||
		    image_info.resource_class == ImageResourceClass::None) {
			Fail(flags.pc, "image operation has invalid resource kind");
		}
		handle               = inst.Arg(0).Resolve().TryInstruction();
		const auto* indirect = handle != nullptr ? FindIndirectDescriptor(*handle) : nullptr;
		if (indirect != nullptr) {
			source = indirect->source;
		} else {
			GetHandle(inst.Arg(0), ValueOpcode::GetImageResource, 8, flags.pc,
			          memory.resource * 4u, handle, source);
		}
		// KYTY_PERF_BINDLESS_CLASSES: a field of a bounded heap table that will be bindless gets a resource of its own
		// (=1 when the shader loads or gathers through it, =2 always), so its numeric class can follow its records; the
		// handles of other fields share one resource as before (all of them float). A resource per field is one more
		// set-0 binding and heap use per draw, hence =1 by default.
		std::optional<std::pair<uint32_t, uint32_t>> split_field;
		if (m_split_fields && indirect != nullptr && indirect->table_read != nullptr && indirect->field_known &&
		    BindlessImageExpected(m_program.stage, image_info.resource_class, memory.image_r128) &&
		    (BindlessClassesMode() >= 2u || HandleReadsTexels(*handle)))
			split_field = std::pair {indirect->field_stride, indirect->field_offset};
		resource = AddImage(source, memory, op, flags.pc, split_field);
		if (resource == UINT32_MAX) {
			if (m_splits_made != 0u) {
				m_split_overflow = true; // KYTY_PERF_BINDLESS_CLASSES: Run tracks the shader again without field resources
				return;
			}
			Fail(flags.pc, "image resource limit exceeded");
		}
		// The bindless tables: the record field each handle of a bounded heap table reads (the handles of one
		// table with the same image properties share the resource, so it collects all of them).
		if (indirect != nullptr && indirect->table_read != nullptr && resource < m_info.images.size()) {
			constexpr size_t MaxFields = 64;
			auto&            image     = m_info.images[resource];
			const std::pair<uint32_t, uint32_t> field {indirect->field_stride, indirect->field_offset};
			if (!indirect->field_known || (image.bindless_fields.size() >= MaxFields &&
			                               std::ranges::find(image.bindless_fields, field) == image.bindless_fields.end())) {
				image.bindless_fields_complete = false;
			} else if (std::ranges::find(image.bindless_fields, field) == image.bindless_fields.end()) {
				image.bindless_fields.push_back(field);
			}
		}
		AddHandlePatch(handle, resource, flags.pc);
		uint32_t sampler = 0;
		if (image_info.needs_sampler) {
			if (inst.NumArgs() < 2) {
				Fail(flags.pc, "sampled image operation has no sampler handle");
			}
			Inst*      sampler_handle = nullptr;
			uint32_t   sampler_source = 0;
			const bool sample_adjust =
			    (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0;
			auto* planned = inst.Arg(1).Resolve().TryInstruction();
			const auto* indirect_sampler = planned != nullptr ? FindIndirectDescriptor(*planned) : nullptr;
			if (indirect_sampler != nullptr) {
				sampler_handle = planned;
				sampler_source = indirect_sampler->source;
			} else {
				GetHandle(inst.Arg(1), ValueOpcode::GetSamplerResource, 4, flags.pc,
				          memory.sampler * 4u, sampler_handle, sampler_source, true, sample_adjust);
			}
			sampler = AddSampler(sampler_source, flags.pc);
			if (sampler == UINT32_MAX) {
				Fail(flags.pc, "sampler resource limit exceeded");
			}
			m_info.samplers[sampler].gather_lod |=
			    op == ValueOpcode::ImageGatherRaw &&
			    (memory.image_sample_flags & Decoder::ImageSampleFlagLod) != 0u;
			AddHandlePatch(sampler_handle, sampler, flags.pc);
			AddSampledPair(resource, sampler, flags.pc);
		}
		AddMemoryPatch(flags.index, resource, sampler, image_info.needs_sampler, flags.pc);
	}

	const DescriptorSource* Source(uint32_t source) const {
		return source < m_sources.size() ? &m_sources[source] : nullptr;
	}

	void LinkImageAliases() {
		for (auto& buffer: m_info.buffers) {
			const auto* buffer_source = Source(buffer.source);
			if (buffer_source == nullptr || buffer_source->dword_count != 4) {
				continue;
			}
			for (uint32_t image = 0; image < m_info.images.size(); image++) {
				const auto* image_source = Source(m_info.images[image].source);
				if (image_source == nullptr || image_source->dword_count != 8 ||
				    image_source->indirect_descriptor.has_value()) {
					continue;
				}
				bool alias = true;
				for (uint32_t dword = 0; dword < 4; dword++) {
					alias = alias && EquivalentValue(m_program, buffer_source->dwords[dword],
					                                 image_source->dwords[dword]);
				}
				if (alias) {
					buffer.image_alias = image;
					break;
				}
			}
		}
	}

	Program&                                   m_program;
	const Decoder::Program&                    m_decoded;
	const CFG::Graph&                          m_native_cfg;
	std::vector<Program::ScalarWrite>          m_scalar_writes;
	std::vector<ResolvedHandle>                m_resolved_handles;
	std::vector<const Inst*>                   m_srt_visiting;
	std::unordered_set<const Inst*>            m_srt_visited;
	std::vector<MemoryIndexUse>                m_memory_index_uses;
	std::vector<Inst*>                         m_scalar_reads;
	ShaderInfo                                 m_info;
	std::vector<DescriptorSource>              m_sources;
	std::vector<HandlePatch>                   m_handle_patches;
	std::vector<MemoryPatch>                   m_memory_patches;
	bool                                       m_keep_table_reads = false;
	mutable std::vector<std::vector<uint8_t>>  m_dominators;
	mutable std::unordered_map<const Block*, size_t> m_dominator_index;
	std::vector<IndirectDescriptorPlan>             m_indirect_descriptors;
	std::map<const Inst*, Value>              m_descriptor_values;
	std::unordered_map<const Inst*, Value>    m_lane_projections; // composed V#s (ProjectLaneUniform)
	bool                                       m_shader_writes = false;
	// KYTY_PERF_BINDLESS_CLASSES: Collect may give table fields resources of their own; it made that many; they took
	// the shader past a resource limit (Collect stops, Run tracks the shader again without them).
	bool                                       m_split_fields   = false;
	uint32_t                                   m_splits_made    = 0;
	bool                                       m_split_overflow = false;
};

} // namespace

void TrackResources(Program& program, const Decoder::Program& decoded, const CFG::Graph& native_cfg) {
	Tracker(program, decoded, native_cfg).Run();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
