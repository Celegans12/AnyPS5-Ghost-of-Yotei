#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "common/shaderEnv.h"

#include <algorithm>
#include <optional>
#include <array>
#include <bit>
#include <utility>

#include <queue>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

struct ChainResult {
	Value value;
	Inst* write = nullptr;
};

ChainResult SearchChain(Value value, uint32_t lane, uint32_t wave_size) {
	for (;;) {
		value      = value.Resolve();
		auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::WriteLane) {
			return {value};
		}
		const auto selector = inst->Arg(2).Resolve();
		if (!selector.IsImmediate() || selector.GetType() != Type::U32) {
			return {value};
		}
		if (selector.U32() % wave_size == lane) {
			return {value, inst};
		}
		value = inst->Arg(0);
	}
}

bool IsPossibleToEliminate(Value source, uint32_t lane, uint32_t wave_size) {
	std::queue<Value>         queue;
	std::unordered_set<Inst*> visited;
	queue.push(source);

	while (!queue.empty()) {
		const auto chain = SearchChain(queue.front(), lane, wave_size);
		queue.pop();
		if (chain.write != nullptr) {
			continue;
		}
		auto* inst = chain.value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::Phi || inst->NumArgs() == 0) {
			return false;
		}
		if (!visited.insert(inst).second) {
			continue;
		}
		for (size_t index = inst->NumArgs(); index-- > 0;) {
			queue.push(inst->Arg(index));
		}
	}
	return true;
}

using PhiMap = std::unordered_map<Inst*, Inst*>;

Value GetRealValue(PhiMap& phi_map, Value source, uint32_t lane, uint32_t wave_size) {
	const auto chain = SearchChain(source, lane, wave_size);
	if (chain.write != nullptr) {
		return chain.write->Arg(1);
	}

	auto* inst = chain.value.ResolveInstruction();
	EXIT_IF(inst->GetOpcode() != ValueOpcode::Phi);
	const auto [entry, is_new] = phi_map.try_emplace(inst);
	if (!is_new) {
		return Value(entry->second);
	}

	auto* block           = inst->Parent();
	auto  insertion_point = std::find_if(block->begin(), block->end(),
	                                     [&](const Inst& candidate) { return &candidate == inst; });
	EXIT_IF(insertion_point == block->end());
	auto& phi = *block->PrependNewInst(insertion_point, ValueOpcode::Phi);
	phi.SetFlags(Type::U32);
	entry->second = &phi;

	std::vector<Value> arguments;
	arguments.reserve(inst->NumArgs());
	for (size_t index = 0; index < inst->NumArgs(); index++) {
		arguments.push_back(GetRealValue(phi_map, inst->Arg(index), lane, wave_size));
	}
	const auto first = arguments.front().Resolve();
	if (std::ranges::all_of(arguments,
	                        [&](Value argument) { return argument.Resolve() == first; })) {
		phi.ReplaceUsesWith(first);
	} else {
		for (size_t index = 0; index < arguments.size(); index++) {
			phi.AddPhiOperand(inst->PhiBlock(index), arguments[index]);
		}
	}
	return Value(&phi);
}

// A full-EXEC prefix scan with both half-wave endpoint reads is a reduction. Wave64 first joins
// the two rows of each half; a wave32 half is one row.
// Fragment subgroups may have no physical invocation at either endpoint.
class WaveReductionMatcher {
public:
	WaveReductionMatcher(const Block* block, ValueOpcode operation)
	    : block(block), operation(operation) {}

	Value Match(const Inst& reduction, uint32_t wave_size) const {
		const auto* low  = Get(reduction.Arg(0), ValueOpcode::ReadLane);
		const auto* high = Get(reduction.Arg(1), ValueOpcode::ReadLane);
		if (low == nullptr || high == nullptr || low->Arg(0).Resolve() != high->Arg(0).Resolve()) {
			return {};
		}
		const auto lane = [&](Value value, uint32_t expected) {
			value = value.Resolve();
			return value.IsImmediate() && value.GetType() == Type::U32 &&
			       value.U32() % wave_size == expected;
		};
		const auto half = wave_size / 2u;
		if (!((lane(low->Arg(1), half - 1u) && lane(high->Arg(1), wave_size - 1u)) ||
		      (lane(low->Arg(1), wave_size - 1u) && lane(high->Arg(1), half - 1u)))) {
			return {};
		}
		if (wave_size == 32u) return MatchScan(low->Arg(0));
		const auto* joined = Get(low->Arg(0), operation);
		if (joined == nullptr) return {};
		for (size_t index = 0; index < 2; index++) {
			const auto* cross = Get(joined->Arg(index), ValueOpcode::Permlane16U32);
			if (cross == nullptr) continue;
			const auto flags  = cross->Flags<PermlaneFlags>();
			const auto prefix = joined->Arg(1 - index).Resolve();
			if (!flags.x16 || flags.fetch_inactive || !flags.bound_control ||
			    cross->Arg(1).Resolve() != Value(UINT32_MAX) ||
			    cross->Arg(2).Resolve() != Value(UINT32_MAX) || !FullExec(cross->Arg(3)) ||
			    cross->Arg(0).Resolve() != prefix) {
				continue;
			}
			if (const auto seed = MatchScan(prefix); !seed.IsEmpty()) return seed;
		}
		return {};
	}

private:
	// The row scan's input.
	Value MatchScan(Value prefix) const {
		for (const uint32_t shift: {8u, 4u, 2u, 1u}) {
			prefix = MatchRow(prefix, shift);
		}
		return prefix;
	}

	Inst* Get(Value value, ValueOpcode opcode) const {
		auto* inst = value.Resolve().TryInstruction();
		return inst != nullptr && inst->Parent() == block && inst->GetOpcode() == opcode
		           ? inst : nullptr;
	}

	// Constant propagation folds the x | ~x EXEC of s_orn2_saveexec to true.
	static bool FullExec(Value predicate) { return predicate.Resolve() == Value(true); }

	static bool IsRowShift(const Inst& inst, uint32_t shift) {
		const auto flags = inst.Flags<DppMoveFlags>();
		return flags.control == 0x110u + shift && flags.row_mask == 0xf &&
		       flags.bank_mask == 0xf && !flags.fetch_inactive && !flags.bound_control && !flags.dpp8;
	}

	Value MatchRow(Value value, uint32_t shift) const {
		const auto* update = Get(value, ValueOpcode::DppUpdateU32);
		if (update == nullptr || !IsRowShift(*update, shift) || !FullExec(update->Arg(2))) return {};
		const auto* combined = Get(update->Arg(0), operation);
		if (combined == nullptr) return {};
		for (size_t index = 0; index < 2; index++) {
			const auto* move = Get(combined->Arg(index), ValueOpcode::DppMoveU32);
			if (move == nullptr || !IsRowShift(*move, shift) || !FullExec(move->Arg(1))) continue;
			const auto previous = combined->Arg(1 - index).Resolve();
			if (move->Arg(0).Resolve() == previous && update->Arg(1).Resolve() == previous) {
				return previous;
			}
		}
		return {};
	}

	const Block* block;
	ValueOpcode  operation;
};

// The subgroup reduction that replaces a proven whole-wave scan of operation.
std::optional<ValueOpcode> WaveReduction(ValueOpcode operation) {
	switch (operation) {
		case ValueOpcode::UMin32: return ValueOpcode::SubgroupUMinU32;
		case ValueOpcode::UMax32: return ValueOpcode::SubgroupUMaxU32;
		case ValueOpcode::BitwiseOr32: return ValueOpcode::SubgroupOrU32;
		default: return std::nullopt;
	}
}

} // namespace

ReadLaneStats EliminateReadLane(Program& program, uint32_t wave_size, bool rewrite_reductions) {
	ReadLaneStats stats;
	if (wave_size != 32u && wave_size != 64u) {
		return stats;
	}

	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			const auto opcode = inst.GetOpcode();
			// Paired wave64 shaders (rewrite_reductions false) keep their guest reduction loops: a pair computes a
			// whole-wave value from one half of the wave only.
			if (const auto reduce = WaveReduction(opcode); reduce && rewrite_reductions) {
				const auto seed = WaveReductionMatcher(block, opcode).Match(inst, wave_size);
				if (!seed.IsEmpty()) {
					const auto position = std::find_if(block->begin(), block->end(),
					                                  [&](const Inst& candidate) { return &candidate == &inst; });
					auto&      reduction = *block->PrependNewInst(position, *reduce, {seed});
					inst.ReplaceUsesWith(Value(&reduction));
					stats.rewritten_reductions++;
				}
				continue;
			}
			if (opcode != ValueOpcode::ReadLane) {
				continue;
			}
			const auto selector = inst.Arg(1).Resolve();
			if (!selector.IsImmediate() || selector.GetType() != Type::U32) {
				continue;
			}

			const auto lane  = selector.U32() % wave_size;
			const auto chain = SearchChain(inst.Arg(0), lane, wave_size);
			if (chain.write != nullptr) {
				inst.ReplaceUsesWith(chain.write->Arg(1));
				stats.rewritten_reads++;
				continue;
			}
			auto* producer = chain.value.TryInstruction();
			if (producer == nullptr || producer->GetOpcode() != ValueOpcode::Phi ||
			    !IsPossibleToEliminate(chain.value, lane, wave_size)) {
				continue;
			}

			PhiMap phi_map;
			inst.ReplaceUsesWith(GetRealValue(phi_map, chain.value, lane, wave_size));
			stats.rewritten_reads++;
		}
	}
	return stats;
}

// ---------------------------------------------------------------------------------------------
// Removal of exec selects whose "old" lanes are never observed.
//
// The frontend writes every VGPR as v' = Select(exec, new, old). A lane that is off in `exec`
// keeps `old`; the select is only needed when some later reader can observe such a lane. This pass
// proves, per select, that every observation of its result happens on lanes where the select
// condition C is true; then v' == new on all observed lanes and the select is replaced by `new`.
//
// Model. IR values are SSA; a value is a per-lane quantity. Lane masks are U1 SSA values (the exec
// chain: LogicalAnd / LogicalNot / LogicalOr / Phi of compares). A "demand" of a value is a set of
// terms; a term is the lane set  AND(P) AND NOT(OR(N))  for positive/negative mask atoms, evaluated
// at the time of the observation; the value is only observed on the union of its terms (or on
// every lane when the demand is TOP). Demand flows backwards along def-use edges:
//   * lane-wise operations pass the result demand to their operands;
//   * Select(C, a, b):  a gets  D AND C,  b gets  D AND NOT C  (a term already inside C adds
//     nothing to b);  LogicalAnd(x, y): x gets D AND y (y gets D AND x);
//   * predicated memory operations observe their operands on the lanes of their predicate;
//   * every other use (ballot, branch conditions, DPP, read-lane, images, ...) observes all lanes (TOP).
// Mask subset tests are structural (Sub): X <= Y when Y is X, a conjunct of X, ... An SSA value
// denotes its latest instance, and the relations used only follow operand edges of And/Or/Not and
// Phi whose operands are bounded by a value that strictly dominates the Phi block, so both sides
// always name the same dynamic instance (a def that dominates a use is not re-executed on any path
// from the latest execution of a def it dominates to that use). Demand that crosses a loop back
// edge keeps only atoms defined strictly before the loop header (the same instance in every
// iteration); everything else becomes TOP. Phi cycles are handled by the usual least fixpoint
// (demand only grows) and a coinductive hypothesis for Sub across Phi cycles.
// ---------------------------------------------------------------------------------------------
namespace {

using V = uint32_t;
constexpr V      kNoValue = 0xffffffffu;
constexpr V      kTrue    = 0xfffffffeu;
constexpr V      kFalse   = 0xfffffffdu;
constexpr size_t kMaxLit  = 4;
constexpr size_t kMaxTerms = 48;

// AND(p[0..np)) AND NOT(OR(n[0..nn))). p and n are kept sorted, unused entries are zero.
struct Term {
	std::array<V, kMaxLit> p {};
	std::array<V, kMaxLit> n {};
	uint8_t                np = 0;
	uint8_t                nn = 0;
	[[nodiscard]] bool     IsAllLanes() const { return np == 0 && nn == 0; }
	bool                   operator==(const Term&) const = default;
};

struct Demand {
	bool              top = false;
	std::vector<Term> terms;
};

// The Sub results (key (x << 32) | y, never ~0: x and y are instruction ids): an open-addressing table. The pass asks
// it ~140 million times in the largest shaders (~20 thousand distinct keys); std::unordered_map was most of the
// pass's time (700-800 ms per shader on the GPU thread).
class SubMemo {
public:
	// 0 or 1 when present, -1 when not.
	[[nodiscard]] int Find(uint64_t key) const {
		if (m_keys.empty()) return -1;
		for (size_t slot = Slot(key);; slot = (slot + 1u) & m_mask) {
			if (m_keys[slot] == key) return m_values[slot];
			if (m_keys[slot] == kEmpty) return -1;
		}
	}
	void Set(uint64_t key, uint8_t value) {
		if ((m_count + 1u) * 2u > m_keys.size()) Grow();
		for (size_t slot = Slot(key);; slot = (slot + 1u) & m_mask) {
			if (m_keys[slot] == key) {
				m_values[slot] = value;
				return;
			}
			if (m_keys[slot] == kEmpty) {
				m_keys[slot]   = key;
				m_values[slot] = value;
				m_count++;
				return;
			}
		}
	}
	[[nodiscard]] size_t size() const { return m_count; }

private:
	static constexpr uint64_t kEmpty = ~uint64_t {0};
	[[nodiscard]] size_t Slot(uint64_t key) const { return static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> m_shift); }
	void Grow() {
		const size_t capacity = m_keys.empty() ? 1024u : m_keys.size() * 2u;
		std::vector<uint64_t> keys(capacity, kEmpty);
		std::vector<uint8_t>  values(capacity, 0);
		std::swap(keys, m_keys);
		std::swap(values, m_values);
		m_mask  = capacity - 1u;
		m_shift = 64 - std::countr_zero(capacity);
		m_count = 0;
		for (size_t index = 0; index < keys.size(); index++)
			if (keys[index] != kEmpty) Set(keys[index], values[index]);
	}
	std::vector<uint64_t> m_keys;
	std::vector<uint8_t>  m_values;
	size_t                m_mask  = 0;
	int                   m_shift = 64;
	size_t                m_count = 0;
};

// TermSubTerm results by the two terms' contents: the demand sets of a chain of instructions hold the same terms, so
// the same pairs are compared again and again (~6 million comparisons, ~14 Sub queries each, in the largest shaders).
class TermPairMemo {
public:
	// 0 or 1 when present, -1 when not.
	[[nodiscard]] int Find(const Term& a, const Term& b) const {
		if (m_entries.empty()) return -1;
		for (size_t slot = Slot(a, b);; slot = (slot + 1u) & m_mask) {
			const auto& entry = m_entries[slot];
			if (!entry.used) return -1;
			if (entry.a == a && entry.b == b) return entry.value;
		}
	}
	void Set(const Term& a, const Term& b, bool value) {
		if ((m_count + 1u) * 2u > m_entries.size()) Grow();
		for (size_t slot = Slot(a, b);; slot = (slot + 1u) & m_mask) {
			auto& entry = m_entries[slot];
			if (!entry.used) {
				entry = {a, b, static_cast<uint8_t>(value ? 1 : 0), true};
				m_count++;
				return;
			}
			if (entry.a == a && entry.b == b) {
				entry.value = value ? 1 : 0;
				return;
			}
		}
	}
	[[nodiscard]] size_t size() const { return m_count; }

private:
	struct Entry {
		Term    a;
		Term    b;
		uint8_t value = 0;
		bool    used  = false;
	};
	static uint64_t Mix(uint64_t hash, uint64_t value) { return (hash ^ value) * 0x100000001b3ull; }
	[[nodiscard]] size_t Slot(const Term& a, const Term& b) const {
		uint64_t hash = 0xcbf29ce484222325ull;
		for (const Term* term: {&a, &b}) {
			hash = Mix(hash, (uint64_t {term->np} << 8u) | term->nn);
			for (size_t i = 0; i < kMaxLit; i++) hash = Mix(hash, (uint64_t {term->p[i]} << 32u) | term->n[i]);
		}
		return static_cast<size_t>((hash * 0x9E3779B97F4A7C15ull) >> m_shift);
	}
	void Grow() {
		const size_t       capacity = m_entries.empty() ? 4096u : m_entries.size() * 2u;
		std::vector<Entry> entries(capacity);
		std::swap(entries, m_entries);
		m_mask  = capacity - 1u;
		m_shift = 64 - std::countr_zero(capacity);
		m_count = 0;
		for (const auto& entry: entries)
			if (entry.used) Set(entry.a, entry.b, entry.value != 0);
	}
	std::vector<Entry> m_entries;
	size_t             m_mask  = 0;
	int                m_shift = 64;
	size_t             m_count = 0;
};

class ExecSelectElider {
public:
	explicit ExecSelectElider(Program& program): m_program(program) {}

	ExecSelectElideStats Run() {
		ExecSelectElideStats stats;
		if (m_program.dispatcher_fallback || m_program.blocks.empty()) {
			stats.skipped = true;
			return stats;
		}
		if (!BuildCfg() || !CollectInstructions()) {
			stats.skipped = true;
			return stats;
		}
		ComputePhiRepresentatives();
		stats.selects = static_cast<uint32_t>(m_selects.size());
		stats.rewritten = RewriteStackedSelects();
		if (!SolveDemand()) {
			stats.skipped = true;
			return stats;
		}
		std::vector<Inst*> victims;
		for (const auto id: m_selects) {
			Inst*        select = m_insts[id];
			const auto&  demand = m_dem[id];
			const V      cond   = Find(ValueId(select->Arg(0)));
			if (cond == kNoValue) continue;
			bool elide = false;
			if (select->Arg(1).Resolve() == select->Arg(2).Resolve()) {
				elide = true; // both arms are the same value
			} else if (cond == kTrue) {
				elide = true;
			} else if (cond == kFalse) {
				elide = false;
			} else if (demand.top) {
				stats.top++;
			} else if (std::ranges::all_of(demand.terms,
			                                [&](const Term& term) { return TermSubAtom(term, cond, 0); })) {
				elide = true;
				if (demand.terms.empty()) stats.unobserved++;
			} else {
				stats.wider++;
			}
			if (elide) victims.push_back(select);
		}
		for (auto* select: victims) {
			select->ReplaceUsesWith(select->Arg(1).Resolve());
			stats.elided++;
		}
		for (size_t index = m_stack_records.size(); index-- > 0;) {
			const auto& record = m_stack_records[index];
			auto*       inner  = record.inner.TryInstruction();
			if (record.outer->GetOpcode() != ValueOpcode::SelectU32 || inner == nullptr ||
			    inner->GetOpcode() != ValueOpcode::SelectU32 || inner->UseCount() == 0) {
				continue;
			}
			record.outer->SetArg(record.operand, record.inner);
			stats.restored++;
		}
		return stats;
	}

private:
	// ----- CFG, dominators -----
	bool BuildCfg() {
		const auto count = m_program.blocks.size();
		m_block_index.reserve(count);
		for (size_t index = 0; index < count; index++) {
			if (!m_block_index.emplace(m_program.blocks[index], static_cast<uint32_t>(index)).second) return false;
		}
		// Reverse postorder from the entry block.
		std::vector<uint32_t> postorder;
		std::vector<uint8_t>  seen(count, 0);
		std::vector<std::pair<uint32_t, size_t>> stack;
		stack.emplace_back(0u, 0u);
		seen[0] = 1;
		while (!stack.empty()) {
			auto& [block, next] = stack.back();
			const auto successors = m_program.blocks[block]->ImmSuccessors();
			if (next < successors.size()) {
				const auto found = m_block_index.find(successors[next++]);
				if (found == m_block_index.end()) return false;
				if (!seen[found->second]) {
					seen[found->second] = 1;
					stack.emplace_back(found->second, 0u);
				}
			} else {
				postorder.push_back(block);
				stack.pop_back();
			}
		}
		if (postorder.size() != count) return false; // unreachable blocks: do not guess
		m_rpo.assign(postorder.rbegin(), postorder.rend());
		std::vector<uint32_t> rpo_number(count, 0);
		for (size_t index = 0; index < count; index++) rpo_number[m_rpo[index]] = static_cast<uint32_t>(index);
		constexpr uint32_t kUndefined = UINT32_MAX;
		std::vector<uint32_t> idom(count, kUndefined);
		idom[0] = 0;
		const auto intersect = [&](uint32_t a, uint32_t b) {
			while (a != b) {
				while (rpo_number[a] > rpo_number[b]) a = idom[a];
				while (rpo_number[b] > rpo_number[a]) b = idom[b];
			}
			return a;
		};
		for (bool changed = true; changed;) {
			changed = false;
			for (size_t order = 1; order < count; order++) {
				const auto block = m_rpo[order];
				uint32_t   candidate = kUndefined;
				for (auto* predecessor: m_program.blocks[block]->ImmPredecessors()) {
					const auto found = m_block_index.find(predecessor);
					if (found == m_block_index.end()) return false;
					if (idom[found->second] == kUndefined) continue;
					candidate = candidate == kUndefined ? found->second : intersect(candidate, found->second);
				}
				if (candidate == kUndefined) return false;
				if (idom[block] != candidate) {
					idom[block] = candidate;
					changed     = true;
				}
			}
		}
		// Dominator tree numbering for O(1) dominance queries.
		std::vector<std::vector<uint32_t>> children(count);
		for (size_t block = 1; block < count; block++) children[idom[block]].push_back(static_cast<uint32_t>(block));
		m_tin.assign(count, 0);
		m_tout.assign(count, 0);
		uint32_t clock = 0;
		std::vector<std::pair<uint32_t, size_t>> walk;
		walk.emplace_back(0u, 0u);
		m_tin[0] = clock++;
		while (!walk.empty()) {
			auto& [block, next] = walk.back();
			if (next < children[block].size()) {
				const auto child = children[block][next++];
				m_tin[child]     = clock++;
				walk.emplace_back(child, 0u);
			} else {
				m_tout[block] = clock++;
				walk.pop_back();
			}
		}
		return true;
	}

	[[nodiscard]] bool Dominates(uint32_t a, uint32_t b) const { return m_tin[a] <= m_tin[b] && m_tout[b] <= m_tout[a]; }
	[[nodiscard]] bool StrictlyDominates(uint32_t a, uint32_t b) const { return a != b && Dominates(a, b); }

	bool CollectInstructions() {
		for (uint32_t block = 0; block < m_program.blocks.size(); block++) {
			for (auto& inst: *m_program.blocks[block]) {
				if (inst.GetOpcode() == ValueOpcode::Identity) return false; // run after RemoveIdentities
				m_inst_id.emplace(&inst, static_cast<uint32_t>(m_insts.size()));
				m_insts.push_back(&inst);
				m_inst_block.push_back(block);
				if (inst.GetOpcode() == ValueOpcode::SelectU32) m_selects.push_back(static_cast<uint32_t>(m_insts.size() - 1));
			}
		}
		m_dem.assign(m_insts.size(), Demand {});
		m_queued.assign(m_insts.size(), 0);
		return true;
	}

	// ----- values -----
	[[nodiscard]] V ValueId(Value value) const {
		value = value.Resolve();
		if (value.IsImmediate()) {
			if (value.GetType() == Type::U1) return value.U1() ? kTrue : kFalse;
			return kNoValue;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return kNoValue;
		const auto found = m_inst_id.find(inst);
		return found == m_inst_id.end() ? kNoValue : found->second;
	}
	[[nodiscard]] bool IsInst(V value) const { return value < m_insts.size(); }

	// A U1 Phi whose operands (other than itself) all denote one value is that value.
	void ComputePhiRepresentatives() {
		m_rep.resize(m_insts.size());
		for (uint32_t id = 0; id < m_insts.size(); id++) m_rep[id] = id;
		for (bool changed = true; changed;) {
			changed = false;
			for (uint32_t id = 0; id < m_insts.size(); id++) {
				const auto* phi = m_insts[id];
				if (phi->GetOpcode() != ValueOpcode::Phi || phi->GetType() != Type::U1 || m_rep[id] != id) continue;
				V    same     = kNoValue;
				bool trivial  = true;
				bool has_same = false;
				for (size_t k = 0; k < phi->NumArgs() && trivial; k++) {
					const V operand = Find(ValueId(phi->Arg(k)));
					if (operand == id) continue;
					if (!has_same) {
						same     = operand;
						has_same = true;
					} else if (same != operand) {
						trivial = false;
					}
				}
				if (trivial && has_same && same != kNoValue) {
					m_rep[id] = same;
					changed   = true;
				}
			}
		}
	}
	[[nodiscard]] V Find(V value) const {
		while (IsInst(value) && m_rep[value] != value) value = m_rep[value];
		return value;
	}
	[[nodiscard]] V ArgOf(V inst, size_t index) const {
		if (!m_arg_begin.empty()) return m_arg_ids[m_arg_begin[inst] + index];
		return Find(ValueId(m_insts[inst]->Arg(index)));
	}
	// ArgOf as a table for the demand analysis (SolveDemand and the decisions after it): the instructions and
	// the Phi representatives do not change until the elision itself. The lookups were a third of the pass's time
	// in the largest compute shaders.
	void BuildArgTable() {
		m_arg_begin.assign(m_insts.size() + 1u, 0u);
		m_arg_ids.clear();
		for (size_t id = 0; id < m_insts.size(); id++) {
			m_arg_begin[id] = static_cast<uint32_t>(m_arg_ids.size());
			const auto* inst = m_insts[id];
			for (size_t k = 0; k < inst->NumArgs(); k++) m_arg_ids.push_back(Find(ValueId(inst->Arg(k))));
		}
		m_arg_begin[m_insts.size()] = static_cast<uint32_t>(m_arg_ids.size());
	}
	[[nodiscard]] ValueOpcode OpOf(V inst) const { return m_insts[inst]->GetOpcode(); }

	// ----- L1 / L2: Select(E, n2, Select(E, n1, o)) = Select(E, n2, o); Select(E, Select(E, a, b), c) = Select(E, a, c) -----
	// Exact for any E. Each step is recorded so it can be undone when the inner select survives with
	// other users: then the rewrite would only keep `o` (or `a`) live longer for nothing.
	uint32_t RewriteStackedSelects() {
		uint32_t rewrites = 0;
		for (const auto id: m_selects) {
			Inst* select = m_insts[id];
			const V cond = Find(ValueId(select->Arg(0)));
			if (!IsInst(cond)) continue;
			for (;;) {
				auto* inner = select->Arg(2).Resolve().TryInstruction();
				if (inner == nullptr || inner->GetOpcode() != ValueOpcode::SelectU32 ||
				    Find(ValueId(inner->Arg(0))) != cond) break;
				m_stack_records.push_back({select, 2u, Value(inner)});
				select->SetArg(2, inner->Arg(2));
				rewrites++;
			}
			for (;;) {
				auto* inner = select->Arg(1).Resolve().TryInstruction();
				if (inner == nullptr || inner->GetOpcode() != ValueOpcode::SelectU32 ||
				    Find(ValueId(inner->Arg(0))) != cond) break;
				m_stack_records.push_back({select, 1u, Value(inner)});
				select->SetArg(1, inner->Arg(1));
				rewrites++;
			}
		}
		return rewrites;
	}

	// ----- structural lane-set relations on mask values -----
	// Sub(x, y): the lanes of x are a subset of the lanes of y (both canonical, latest instances).
	bool Sub(V x, V y, int depth = 0) {
		if (x == y) return true;
		if (y == kTrue || x == kFalse) return true;
		if (!IsInst(x) || !IsInst(y) || depth > 48) return false;
		const uint64_t key = (static_cast<uint64_t>(x) << 32u) | y;
		if (const int found = m_sub_memo.Find(key); found >= 0) return found != 0;
		if (const auto active = m_sub_active.find(key); active != m_sub_active.end()) {
			m_sub_assumed = std::min(m_sub_assumed, active->second);
			return true; // coinductive hypothesis across Phi cycles (induction on loop iterations)
		}
		// m_sub_assumed: the outermost active query (stack position) that the result below assumed.
		const auto position = static_cast<uint32_t>(m_sub_active.size());
		m_sub_active.emplace(key, position);
		const auto assumed_before = m_sub_assumed;
		m_sub_assumed             = UINT32_MAX;
		const bool result         = SubRules(x, y, depth);
		m_sub_active.erase(key);
		// A result that assumed no query but this one is a closed proof and is kept, as the outermost query's
		// always was; one that assumed an enclosing query holds only under that assumption and is computed again
		// when asked again. (Closed inner proofs used not to be kept either: nested loops recomputed them
		// exponentially often, seconds per shader.)
		const bool closed = position == 0u || m_sub_assumed == UINT32_MAX;
		if (!result || closed) m_sub_memo.Set(key, result ? 1 : 0);
		m_sub_assumed = std::min(assumed_before, closed ? UINT32_MAX : m_sub_assumed);
		return result;
	}

	bool SubRules(V x, V y, int depth) {
		const auto ox = OpOf(x);
		const auto oy = OpOf(y);
		if (ox == ValueOpcode::LogicalAnd) {
			if (Sub(ArgOf(x, 0), y, depth + 1) || Sub(ArgOf(x, 1), y, depth + 1)) return true;
		}
		if (oy == ValueOpcode::LogicalAnd) {
			if (Sub(x, ArgOf(y, 0), depth + 1) && Sub(x, ArgOf(y, 1), depth + 1)) return true;
		}
		if (oy == ValueOpcode::LogicalOr) {
			if (Sub(x, ArgOf(y, 0), depth + 1) || Sub(x, ArgOf(y, 1), depth + 1)) return true;
		}
		if (ox == ValueOpcode::LogicalOr) {
			if (Sub(ArgOf(x, 0), y, depth + 1) && Sub(ArgOf(x, 1), y, depth + 1)) return true;
		}
		if (ox == ValueOpcode::Phi && m_insts[x]->NumArgs() != 0 &&
		    StrictlyDominates(m_inst_block[y], m_inst_block[x])) {
			// Every incoming value is inside y; y's definition dominates the Phi block, so y names the
			// same instance on every edge (also on loop back edges).
			bool all     = true;
			bool covered = false;
			for (size_t k = 0; k < m_insts[x]->NumArgs() && all; k++) {
				const V operand = ArgOf(x, k);
				if (operand == x) continue;
				covered = true;
				all     = Sub(operand, y, depth + 1);
			}
			if (all && covered) return true;
		}
		if (ox == ValueOpcode::LogicalNot && oy == ValueOpcode::LogicalNot) {
			if (Sub(ArgOf(y, 0), ArgOf(x, 0), depth + 1)) return true;
		}
		return false;
	}

	// SubNot(x, z): the lanes of x are a subset of NOT z.
	bool SubNot(V x, V z, int depth) {
		if (x == z || !IsInst(x) || depth > 24) return false;
		if (IsInst(z) && OpOf(z) == ValueOpcode::LogicalNot && Sub(x, ArgOf(z, 0), depth + 1)) return true;
		switch (OpOf(x)) {
			case ValueOpcode::LogicalAnd:
				return SubNot(ArgOf(x, 0), z, depth + 1) || SubNot(ArgOf(x, 1), z, depth + 1);
			case ValueOpcode::LogicalNot: return Sub(z, ArgOf(x, 0), depth + 1);
			default: return false;
		}
	}

	[[nodiscard]] bool MaskLike(V value) const {
		if (!IsInst(value)) return false;
		switch (OpOf(value)) {
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalNot: return true;
			case ValueOpcode::Phi: return m_insts[value]->GetType() == Type::U1;
			default: return false;
		}
	}

	// ----- terms -----
	bool TermEmpty(const Term& term) {
		for (size_t i = 0; i < term.np; i++)
			for (size_t j = 0; j < term.nn; j++)
				if (Sub(term.p[i], term.n[j])) return true;
		return false;
	}

	// term <= y
	bool TermSubAtom(const Term& term, V y, int depth) {
		if (y == kTrue) return true;
		if (!IsInst(y) || depth > 16) return false;
		for (size_t i = 0; i < term.np; i++)
			if (Sub(term.p[i], y)) return true;
		switch (OpOf(y)) {
			case ValueOpcode::LogicalAnd:
				return TermSubAtom(term, ArgOf(y, 0), depth + 1) && TermSubAtom(term, ArgOf(y, 1), depth + 1);
			case ValueOpcode::LogicalOr:
				return TermSubAtom(term, ArgOf(y, 0), depth + 1) || TermSubAtom(term, ArgOf(y, 1), depth + 1);
			case ValueOpcode::LogicalNot: {
				const V z = ArgOf(y, 0);
				for (size_t j = 0; j < term.nn; j++)
					if (Sub(z, term.n[j])) return true;
				for (size_t i = 0; i < term.np; i++)
					if (SubNot(term.p[i], z, 0)) return true;
				return false;
			}
			default: return false;
		}
	}

	// t1 <= t2
	bool TermSubTerm(const Term& t1, const Term& t2) {
		if (const int found = m_term_pairs.Find(t1, t2); found >= 0) return found != 0;
		const bool result = TermSubTermRules(t1, t2);
		m_term_pairs.Set(t1, t2, result);
		return result;
	}
	bool TermSubTermRules(const Term& t1, const Term& t2) {
		for (size_t i = 0; i < t2.np; i++)
			if (!TermSubAtom(t1, t2.p[i], 0)) return false;
		for (size_t j = 0; j < t2.nn; j++) {
			bool ok = false;
			for (size_t k = 0; k < t1.nn && !ok; k++) ok = Sub(t2.n[j], t1.n[k]);
			for (size_t i = 0; i < t1.np && !ok; i++) ok = SubNot(t1.p[i], t2.n[j], 0);
			if (!ok) return false;
		}
		return true;
	}

	static void Insert(std::array<V, kMaxLit>& list, uint8_t& count, V value) {
		for (size_t i = 0; i < count; i++)
			if (list[i] == value) return;
		if (count == kMaxLit) {
			// Dropping a conjunct only enlarges the lane set (sound); drop the largest id.
			list[kMaxLit - 1] = value;
		} else {
			list[count++] = value;
		}
		std::sort(list.begin(), list.begin() + count);
	}

	// term AND m
	Term AndPositive(const Term& term, V m) {
		if (!MaskLike(m) || TermSubAtom(term, m, 0)) return term;
		Term out;
		for (size_t i = 0; i < term.np; i++)
			if (!Sub(m, term.p[i])) Insert(out.p, out.np, term.p[i]);
		for (size_t j = 0; j < term.nn; j++) Insert(out.n, out.nn, term.n[j]);
		Insert(out.p, out.np, m);
		return out;
	}

	// term AND NOT m; false when the result is empty
	bool AndNegative(const Term& term, V m, Term& out) {
		out = term;
		if (!MaskLike(m)) return true;
		if (TermSubAtom(term, m, 0)) return false;
		out = Term {};
		for (size_t i = 0; i < term.np; i++) Insert(out.p, out.np, term.p[i]);
		for (size_t j = 0; j < term.nn; j++)
			if (!Sub(term.n[j], m)) Insert(out.n, out.nn, term.n[j]);
		Insert(out.n, out.nn, m);
		return !TermEmpty(out);
	}

	// ----- demand sets -----
	bool AddTerm(Demand& demand, const Term& term) {
		if (demand.top) return false;
		if (term.IsAllLanes()) {
			demand.top = true;
			demand.terms.clear();
			return true;
		}
		// A term the demand already holds adds nothing (TermSubTerm(term, term) holds structurally); the worklist pushes
		// the same demand again and again.
		for (const auto& existing: demand.terms)
			if (existing == term) return false;
		for (const auto& existing: demand.terms)
			if (TermSubTerm(term, existing)) return false;
		std::erase_if(demand.terms, [&](const Term& existing) { return TermSubTerm(existing, term); });
		demand.terms.push_back(term);
		if (demand.terms.size() > kMaxTerms) {
			demand.top = true;
			demand.terms.clear();
		}
		return true;
	}

	bool Union(Demand& dst, const Demand& src) {
		if (dst.top) return false;
		if (src.top) {
			dst.top = true;
			dst.terms.clear();
			return true;
		}
		bool changed = false;
		for (const auto& term: src.terms) {
			changed |= AddTerm(dst, term);
			if (dst.top) return true;
		}
		return changed;
	}

	static Demand Top() {
		Demand demand;
		demand.top = true;
		return demand;
	}
	static Demand Single(const Term& term) {
		Demand demand;
		if (term.IsAllLanes()) demand.top = true;
		else demand.terms.push_back(term);
		return demand;
	}
	// The lanes of mask m (one term); an unusable mask means every lane.
	Demand LanesOf(V mask) {
		if (!MaskLike(mask)) return Top();
		Term term;
		term.p[0] = mask;
		term.np   = 1;
		return Single(term);
	}

	void Enqueue(V id) {
		if (!m_queued[id]) {
			m_queued[id] = 1;
			m_work.push_back(id);
		}
	}
	void Push(Value target, const Demand& demand) {
		const V id = ValueId(target);
		if (!IsInst(id)) return;
		if (Union(m_dem[id], demand)) Enqueue(id);
	}
	void PushAll(const Inst& inst, const Demand& demand) {
		for (size_t k = 0; k < inst.NumArgs(); k++) Push(inst.Arg(k), demand);
	}

	Demand AndMask(const Demand& demand, V mask) {
		if (!MaskLike(mask)) return demand;
		if (demand.top) return LanesOf(mask);
		Demand out;
		for (const auto& term: demand.terms) AddTerm(out, AndPositive(term, mask));
		return out;
	}
	Demand AndNotMask(const Demand& demand, V mask) {
		if (!MaskLike(mask)) return demand;
		if (demand.top) {
			Term term;
			term.n[0] = mask;
			term.nn   = 1;
			return Single(term);
		}
		Demand out;
		for (const auto& term: demand.terms) {
			Term narrowed;
			if (AndNegative(term, mask, narrowed)) AddTerm(out, narrowed);
		}
		return out;
	}

	// Demand carried over the loop back edge into header block `header`.
	Demand AcrossBackEdge(const Demand& demand, uint32_t header) {
		if (demand.top || demand.terms.empty()) return demand;
		Demand out;
		for (const auto& term: demand.terms) {
			Term kept;
			for (size_t i = 0; i < term.np; i++)
				if (StrictlyDominates(m_inst_block[term.p[i]], header)) Insert(kept.p, kept.np, term.p[i]);
			for (size_t j = 0; j < term.nn; j++)
				if (StrictlyDominates(m_inst_block[term.n[j]], header)) Insert(kept.n, kept.nn, term.n[j]);
			if (kept.IsAllLanes()) return Top();
			AddTerm(out, kept);
		}
		return out;
	}

	[[nodiscard]] bool PredicatedMemoryKindOk(const Inst& inst) const {
		const auto index = inst.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) return false;
		switch (m_program.memory_info[index].kind) {
			case ResourceKind::Buffer:
			case ResourceKind::IndirectBuffer:
			case ResourceKind::Flat:
			case ResourceKind::FlatLocal:
			case ResourceKind::Global:
			case ResourceKind::Scratch:
			case ResourceKind::Lds:
			case ResourceKind::Gds: return true;
			default: return false;
		}
	}

	void Process(V id) {
		const Inst& inst = *m_insts[id];
		const Demand demand = m_dem[id]; // copy: pushes may touch other entries
		switch (inst.GetOpcode()) {
			case ValueOpcode::Phi: {
				const uint32_t block = m_inst_block[id];
				for (size_t k = 0; k < inst.NumArgs(); k++) {
					const auto pred = m_block_index.find(inst.PhiBlock(k));
					const bool back = pred == m_block_index.end() || Dominates(block, pred->second);
					Push(inst.Arg(k), back ? AcrossBackEdge(demand, block) : demand);
				}
				return;
			}
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectF32:
			case ValueOpcode::SelectU1: {
				const V cond = Find(ValueId(inst.Arg(0)));
				Push(inst.Arg(0), demand);
				if (!IsInst(cond)) {
					Push(inst.Arg(1), demand);
					Push(inst.Arg(2), demand);
					return;
				}
				Push(inst.Arg(1), AndMask(demand, cond));
				Push(inst.Arg(2), AndNotMask(demand, cond));
				return;
			}
			case ValueOpcode::LogicalAnd: {
				const V lhs = Find(ValueId(inst.Arg(0)));
				const V rhs = Find(ValueId(inst.Arg(1)));
				Push(inst.Arg(0), IsInst(rhs) ? AndMask(demand, rhs) : demand);
				Push(inst.Arg(1), IsInst(lhs) ? AndMask(demand, lhs) : demand);
				return;
			}
			case ValueOpcode::LoadBufferU8:
			case ValueOpcode::LoadBufferU16:
			case ValueOpcode::LoadBufferU32:
			case ValueOpcode::LoadBufferU32x2:
			case ValueOpcode::LoadBufferU32x3:
			case ValueOpcode::LoadBufferU32x4:
			case ValueOpcode::StoreBufferU8:
			case ValueOpcode::StoreBufferU16:
			case ValueOpcode::StoreBufferU32:
			case ValueOpcode::StoreBufferU32x2:
			case ValueOpcode::StoreBufferU32x3:
			case ValueOpcode::StoreBufferU32x4:
			case ValueOpcode::LoadAddressU8:
			case ValueOpcode::LoadAddressU16:
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::StoreAddressU8:
			case ValueOpcode::StoreAddressU16:
			case ValueOpcode::StoreAddressU32:
			case ValueOpcode::LoadSharedU8:
			case ValueOpcode::LoadSharedU16:
			case ValueOpcode::LoadSharedU32:
			case ValueOpcode::LoadSharedU32x2:
			case ValueOpcode::LoadSharedU32x3:
			case ValueOpcode::LoadSharedU32x4:
			case ValueOpcode::WriteSharedU8:
			case ValueOpcode::WriteSharedU16:
			case ValueOpcode::WriteSharedU32:
			case ValueOpcode::WriteSharedU32x2:
			case ValueOpcode::WriteSharedU32x3:
			case ValueOpcode::WriteSharedU32x4: {
				// Every operand is only consumed on the lanes of the trailing predicate.
				const size_t last = inst.NumArgs() - 1;
				const V      pred = Find(ValueId(inst.Arg(last)));
				const bool   gated = PredicatedMemoryKindOk(inst) && IsInst(pred);
				const Demand lanes = gated ? LanesOf(pred) : Top();
				for (size_t k = 0; k < last; k++) Push(inst.Arg(k), lanes);
				Push(inst.Arg(last), Top());
				return;
			}
			// Handles whose operands are read lane-wise by the memory operations that use them.
			case ValueOpcode::GetBufferResource:
			case ValueOpcode::GetAddressResource:
			case ValueOpcode::BitCastU16F16:
			case ValueOpcode::BitCastF16U16:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU64F64:
			case ValueOpcode::BitCastF64U64:
			case ValueOpcode::ConvertU16U32:
			case ValueOpcode::ConvertU32U16:
			case ValueOpcode::ConvertU8U32:
			case ValueOpcode::ConvertU32U8:
			case ValueOpcode::ConvertF32F16:
			case ValueOpcode::ConvertF16F32:
			case ValueOpcode::HalfToF32Arith:
			case ValueOpcode::F32ToHalfArith:
			case ValueOpcode::ConvertS32F32:
			case ValueOpcode::ConvertU32F32:
			case ValueOpcode::ConvertF32S32:
			case ValueOpcode::ConvertF64S32:
			case ValueOpcode::ConvertF32F64:
			case ValueOpcode::ConvertF64F32:
			case ValueOpcode::ConvertF32U32:
			case ValueOpcode::ConvertF64U32:
			case ValueOpcode::CompositeConstructU64:
			case ValueOpcode::CompositeConstructU32x2:
			case ValueOpcode::CompositeConstructU32x3:
			case ValueOpcode::CompositeConstructF32x2:
			case ValueOpcode::CompositeConstructU32x4:
			case ValueOpcode::CompositeExtractU64:
			case ValueOpcode::CompositeExtractU32x2:
			case ValueOpcode::CompositeExtractU32x3:
			case ValueOpcode::CompositeExtractU32x4:
			case ValueOpcode::PackHalf2x16:
			case ValueOpcode::PackSnorm2x16:
			case ValueOpcode::PackUnorm2x16:
			case ValueOpcode::PackFloat2x16Rtz:
			case ValueOpcode::FPAbs32:
			case ValueOpcode::FPNeg32:
			case ValueOpcode::FPSaturate32:
			case ValueOpcode::BitFieldInsert:
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::IAdd32:
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub32:
			case ValueOpcode::ISub64:
			case ValueOpcode::IMul32:
			case ValueOpcode::UDiv32:
			case ValueOpcode::SMulHi:
			case ValueOpcode::UMulHi:
			case ValueOpcode::IAbs32:
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftLeftLogical64:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightLogical64:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::ShiftRightArithmetic64:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::BitwiseNot32:
			case ValueOpcode::BitReverse32:
			case ValueOpcode::BitCount32:
			case ValueOpcode::FindUMsb32:
			case ValueOpcode::FindILsb32:
			case ValueOpcode::SMin32:
			case ValueOpcode::UMin32:
			case ValueOpcode::SMax32:
			case ValueOpcode::UMax32:
			case ValueOpcode::SMinTri32:
			case ValueOpcode::UMinTri32:
			case ValueOpcode::SMaxTri32:
			case ValueOpcode::UMaxTri32:
			case ValueOpcode::SMedTri32:
			case ValueOpcode::UMedTri32:
			case ValueOpcode::SLessThan32:
			case ValueOpcode::SLessThan64:
			case ValueOpcode::SLessThanEqual64:
			case ValueOpcode::ULessThanEqual64:
			case ValueOpcode::UGreaterThanEqual64:
			case ValueOpcode::ULessThan32:
			case ValueOpcode::ULessThan64:
			case ValueOpcode::IEqual32:
			case ValueOpcode::IEqual64:
			case ValueOpcode::SLessThanEqual32:
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::SGreaterThan32:
			case ValueOpcode::UGreaterThan32:
			case ValueOpcode::UGreaterThan64:
			case ValueOpcode::INotEqual32:
			case ValueOpcode::INotEqual64:
			case ValueOpcode::SGreaterThanEqual32:
			case ValueOpcode::UGreaterThanEqual32:
			case ValueOpcode::LogicalOr:
			case ValueOpcode::LogicalXor:
			case ValueOpcode::LogicalNot:
			case ValueOpcode::FPOrdEqual32:
			case ValueOpcode::FPOrdEqual64:
			case ValueOpcode::FPOrdLessThanEqual64:
			case ValueOpcode::FPOrdGreaterThanEqual64:
			case ValueOpcode::FPUnordEqual32:
			case ValueOpcode::FPOrdNotEqual32:
			case ValueOpcode::FPUnordNotEqual32:
			case ValueOpcode::FPOrdLessThan32:
			case ValueOpcode::FPUnordLessThan32:
			case ValueOpcode::FPOrdGreaterThan32:
			case ValueOpcode::FPUnordGreaterThan32:
			case ValueOpcode::FPOrdLessThanEqual32:
			case ValueOpcode::FPUnordLessThanEqual32:
			case ValueOpcode::FPOrdGreaterThanEqual32:
			case ValueOpcode::FPUnordGreaterThanEqual32:
			case ValueOpcode::FPIsNan32:
			case ValueOpcode::FPCmpClass32:
			case ValueOpcode::FPCmpClass16:
			case ValueOpcode::HalfPkAdd:
			case ValueOpcode::HalfPkMul:
			case ValueOpcode::HalfPkFma:
			case ValueOpcode::HalfPkMin:
			case ValueOpcode::HalfPkMax:
			case ValueOpcode::HalfPkMinTri:
			case ValueOpcode::HalfPkMaxTri:
			case ValueOpcode::HalfPkMedTri:
			case ValueOpcode::HalfPkSaturate:
			case ValueOpcode::FPAdd32:
			case ValueOpcode::FPSub32:
			case ValueOpcode::FPFma32:
			case ValueOpcode::FPMad32:
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPAdd64:
			case ValueOpcode::FPMul64:
			case ValueOpcode::FPFma64:
			case ValueOpcode::FPRecip64:
			case ValueOpcode::FPMin32:
			case ValueOpcode::FPMax32:
			case ValueOpcode::FPMin64:
			case ValueOpcode::FPMax64:
			case ValueOpcode::FPMinTri32:
			case ValueOpcode::FPMaxTri32:
			case ValueOpcode::FPMedTri32:
			case ValueOpcode::FPRecip32:
			case ValueOpcode::FPRecipIFlag32:
			case ValueOpcode::FPRecipSqrt32:
			case ValueOpcode::FPSqrt:
			case ValueOpcode::FPSin:
			case ValueOpcode::FPCos:
			case ValueOpcode::FPExp2:
			case ValueOpcode::FPLog2:
			case ValueOpcode::FPLdexp:
			case ValueOpcode::FPRoundEven32:
			case ValueOpcode::FPFloor32:
			case ValueOpcode::FPCeil32:
			case ValueOpcode::FPTrunc32:
			case ValueOpcode::FPFloor64:
			case ValueOpcode::FPCeil64:
			case ValueOpcode::FPTrunc64:
			case ValueOpcode::FPFract32:
			case ValueOpcode::FPFract64:
				PushAll(inst, demand);
				return;
			default: PushAll(inst, Top()); return;
		}
	}

	bool SolveDemand() {
		BuildArgTable();
		// Branch conditions are consumed by every lane's control flow decision.
		for (const auto* block: m_program.blocks) {
			Push(block->condition, Top());
			Push(block->indirect_target, Top());
		}
		for (size_t index = m_rpo.size(); index-- > 0;) {
			const auto* block = m_program.blocks[m_rpo[index]];
			std::vector<V> ids;
			for (const auto& inst: *block) ids.push_back(m_inst_id.at(&inst));
			for (size_t k = ids.size(); k-- > 0;) Process(ids[k]);
		}
		const size_t limit = 64u * m_insts.size() + 1024u;
		size_t       steps = 0;
		while (!m_work.empty()) {
			if (++steps > limit) return false;
			const V id = m_work.back();
			m_work.pop_back();
			m_queued[id] = 0;
			Process(id);
		}
		return true;
	}

	struct StackRecord {
		Inst*  outer;
		size_t operand;
		Value  inner;
	};

	Program&                                   m_program;
	std::vector<StackRecord>                   m_stack_records;
	std::unordered_map<const Block*, uint32_t> m_block_index;
	std::vector<uint32_t>                      m_rpo;
	std::vector<uint32_t>                      m_tin;
	std::vector<uint32_t>                      m_tout;
	std::vector<Inst*>                         m_insts;
	std::unordered_map<const Inst*, uint32_t>  m_inst_id;
	std::vector<uint32_t>                      m_inst_block;
	std::vector<uint32_t>                      m_selects;
	std::vector<V>                             m_rep;
	std::vector<Demand>                        m_dem;
	std::vector<uint8_t>                       m_queued;
	std::vector<V>                             m_work;
	std::vector<uint32_t>                      m_arg_begin; // BuildArgTable
	std::vector<V>                             m_arg_ids;
	SubMemo                                    m_sub_memo;
	TermPairMemo                               m_term_pairs;
	std::unordered_map<uint64_t, uint32_t>     m_sub_active; // query -> stack position
	uint32_t                                   m_sub_assumed = UINT32_MAX;
};

} // namespace

ExecSelectElideStats ElideRedundantExecSelects(Program& program) {
	ExecSelectElider elider(program);
	return elider.Run();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
