#include "graphics/shader/recompiler/ir/passes/RayQuerySource.h"

#include "graphics/shader/recompiler/ir/RayQuery.h"

#include <algorithm>
#include <array>
#include <bit>
#include <compare>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

constexpr size_t   MaxChoices    = 16;
constexpr size_t   MaxHypotheses = 8;
constexpr uint64_t Literal       = 1ull << 63;
constexpr uint64_t Zext          = 0x10000;
constexpr uint64_t Base48        = 0x10001;
constexpr uint32_t None          = 0xffffffffu;
constexpr uint32_t One           = 0xfffffffeu;

using Key     = std::vector<uint64_t>;
using Choices = std::vector<uint32_t>;

enum class Kind : uint64_t { Opaque, Word, Pointer, Load, Op, Half, Join };

struct Atom {
	Kind        kind  = Kind::Opaque;
	uint32_t    width = 32;
	Key         args;
	const Inst* inst = nullptr;
};

struct Form {
	uint32_t                                   width    = 32;
	uint64_t                                   constant = 0;
	std::vector<std::pair<uint32_t, uint64_t>> terms;

	auto operator<=>(const Form&) const = default;
};

uint64_t WidthMask(uint32_t width) {
	return width >= 64 ? ~0ull : (1ull << width) - 1u;
}

class Store {
public:
	uint32_t Atomic(Kind kind, uint32_t width, Key args, const Inst* inst = nullptr) {
		Key key {static_cast<uint64_t>(kind), width, reinterpret_cast<uintptr_t>(inst)};
		key.insert(key.end(), args.begin(), args.end());
		const auto [at, fresh] = m_atom_ids.try_emplace(std::move(key), static_cast<uint32_t>(m_atoms.size()));
		if (fresh) m_atoms.push_back({kind, width, std::move(args), inst});
		return at->second;
	}

	uint32_t Make(Form form) {
		const auto mask = WidthMask(form.width);
		std::ranges::sort(form.terms);
		std::vector<std::pair<uint32_t, uint64_t>> merged;
		for (const auto& [atom, scale]: form.terms) {
			if (!merged.empty() && merged.back().first == atom) {
				merged.back().second += scale;
			} else {
				merged.emplace_back(atom, scale);
			}
		}
		form.terms.clear();
		for (auto [atom, scale]: merged) {
			if ((scale & mask) != 0) form.terms.emplace_back(atom, scale & mask);
		}
		form.constant &= mask;
		const auto [at, fresh] = m_form_ids.try_emplace(form, static_cast<uint32_t>(m_forms.size()));
		if (fresh) m_forms.push_back(std::move(form));
		return at->second;
	}

	[[nodiscard]] const Form& F(uint32_t id) const { return m_forms[id]; }
	[[nodiscard]] const Atom& A(uint32_t id) const { return m_atoms[id]; }

	uint32_t Constant(uint32_t width, uint64_t value) { return Make({width, value, {}}); }
	uint32_t Of(uint32_t atom, uint32_t width) { return Make({width, 0, {{atom, 1}}}); }
	uint32_t Op(uint32_t width, Key args) { return Of(Atomic(Kind::Op, width, std::move(args)), width); }

	uint32_t Add(uint32_t a, uint32_t b, uint64_t factor = 1) {
		auto       form  = m_forms[a];
		const auto other = m_forms[b];
		form.constant += other.constant * factor;
		for (const auto& [atom, scale]: other.terms) form.terms.emplace_back(atom, scale * factor);
		return Make(std::move(form));
	}

	uint32_t Scale(uint32_t a, uint64_t factor) {
		auto form = m_forms[a];
		form.constant *= factor;
		for (auto& term: form.terms) term.second *= factor;
		return Make(std::move(form));
	}

	uint32_t Strip(uint32_t a) {
		auto form     = m_forms[a];
		form.constant = 0;
		return Make(std::move(form));
	}

	[[nodiscard]] std::optional<uint64_t> ConstantOf(uint32_t id) const {
		const auto& form = m_forms[id];
		return form.terms.empty() ? std::optional(form.constant) : std::nullopt;
	}

	[[nodiscard]] std::optional<uint32_t> AtomOf(uint32_t id) const {
		const auto& form = m_forms[id];
		if (form.constant != 0 || form.terms.size() != 1 || form.terms[0].second != 1) return std::nullopt;
		return form.terms[0].first;
	}

	[[nodiscard]] const Atom* OpOf(uint32_t id, uint64_t opcode) const {
		const auto atom = AtomOf(id);
		if (!atom) return nullptr;
		const auto& a = m_atoms[*atom];
		return a.kind == Kind::Op && !a.args.empty() && a.args[0] == opcode ? &a : nullptr;
	}

	uint32_t Widen(uint32_t id) {
		const auto form = m_forms[id];
		if (form.width == 64) return id;
		if (form.terms.empty()) return Constant(64, form.constant);
		if (form.constant == 0 && form.terms.size() == 1 && form.terms[0].second < (1u << 16)) {
			return Make({64, 0, {form.terms[0]}});
		}
		return Op(64, {Zext, id});
	}

	uint32_t Lo(uint32_t id) {
		const auto form = m_forms[id];
		if (form.width == 32) return id;
		uint32_t result = Constant(32, form.constant);
		for (const auto& [atom, scale]: form.terms) result = Add(result, LoAtom(atom), scale);
		return result;
	}

	uint32_t Hi(uint32_t id) {
		const auto form = m_forms[id];
		if (form.width == 32) return Constant(32, 0);
		if (form.terms.empty()) return Constant(32, form.constant >> 32u);
		if (form.constant == 0 && form.terms.size() == 1 && form.terms[0].second == 1) {
			const auto atom = m_atoms[form.terms[0].first];
			if (atom.width == 32) return Constant(32, 0);
			if (atom.kind == Kind::Pointer) return Of(Atomic(Kind::Word, 32, {atom.args[0] + 1u}), 32);
			if (atom.kind == Kind::Load) {
				return Of(Atomic(Kind::Load, 32, {Add(static_cast<uint32_t>(atom.args[0]), Constant(64, 4)), 4}), 32);
			}
			if (atom.kind == Kind::Join) return static_cast<uint32_t>(atom.args[1]);
		}
		return Of(Atomic(Kind::Half, 32, {id, 1}), 32);
	}

	uint32_t Join(uint32_t lo, uint32_t hi) {
		const auto high = ConstantOf(hi);
		if (high && *high == 0) return Widen(lo);
		if (const auto low = ConstantOf(lo); low && high) return Constant(64, *low | *high << 32u);
		const auto la = AtomOf(lo), ha = AtomOf(hi);
		if (la && ha) {
			const auto l = m_atoms[*la], h = m_atoms[*ha];
			if (l.kind == Kind::Word && h.kind == Kind::Word && h.args[0] == l.args[0] + 1u) {
				return Of(Atomic(Kind::Pointer, 64, {l.args[0]}), 64);
			}
			if (l.kind == Kind::Load && h.kind == Kind::Load && l.args[1] == 4 && h.args[1] == 4 &&
			    h.args[0] == Add(static_cast<uint32_t>(l.args[0]), Constant(64, 4))) {
				return Of(Atomic(Kind::Load, 64, {l.args[0], 8}), 64);
			}
			if (l.kind == Kind::Half && h.kind == Kind::Half && l.args[0] == h.args[0] && l.args[1] == 0 && h.args[1] == 1) {
				return static_cast<uint32_t>(l.args[0]);
			}
			if (l.kind == Kind::Word && h.kind == Kind::Op && h.args.size() == 3 && h.args[0] == static_cast<uint64_t>(ValueOpcode::BitwiseAnd32) &&
			    h.args[2] == (Literal | 0xffffu)) {
				const auto inner = AtomOf(static_cast<uint32_t>(h.args[1]));
				if (inner && m_atoms[*inner].kind == Kind::Word && m_atoms[*inner].args[0] == l.args[0] + 1u) {
					return Of(Atomic(Kind::Pointer, 64, {l.args[0]}), 64);
				}
			}
		}
		return Of(Atomic(Kind::Join, 64, {lo, hi}), 64);
	}

private:
	uint32_t LoAtom(uint32_t id) {
		const auto atom = m_atoms[id];
		if (atom.width == 32) return Of(id, 32);
		switch (atom.kind) {
			case Kind::Pointer: return Of(Atomic(Kind::Word, 32, {atom.args[0]}), 32);
			case Kind::Load: return Of(Atomic(Kind::Load, 32, {atom.args[0], 4}), 32);
			case Kind::Join: return static_cast<uint32_t>(atom.args[0]);
			default: return Of(Atomic(Kind::Half, 32, {Of(id, 64), 0}), 32);
		}
	}

	std::vector<Atom>             m_atoms;
	std::map<Key, uint32_t>       m_atom_ids;
	std::vector<Form>             m_forms;
	std::map<Form, uint32_t>      m_form_ids;
};

void Cap(Choices& choices) {
	std::vector<uint32_t> unique;
	for (const auto choice: choices) {
		if (std::ranges::find(unique, choice) == unique.end()) unique.push_back(choice);
	}
	if (unique.size() > MaxChoices) unique.resize(MaxChoices);
	choices = std::move(unique);
}

void Append(Choices& to, const Choices& from) {
	to.insert(to.end(), from.begin(), from.end());
	Cap(to);
}

std::optional<uint32_t> ImmediateU32(Value value) {
	value = value.Resolve();
	if (!value.IsImmediate() || value.GetType() != Type::U32) return std::nullopt;
	return value.U32();
}

const Inst* ProducerOf(Value value, ValueOpcode opcode) {
	const auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
}

size_t BlockIndex(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

class Order {
public:
	explicit Order(const Program& program) {
		for (size_t index = 0; index < program.blocks.size(); ++index) m_index.emplace(program.blocks[index], index);
	}

	[[nodiscard]] size_t Of(const Block* block) const {
		const auto at = m_index.find(block);
		return at == m_index.end() ? SIZE_MAX : at->second;
	}

	[[nodiscard]] bool Carried(const Inst& inst) const {
		if (inst.GetOpcode() != ValueOpcode::Phi || inst.Parent() == nullptr) return false;
		if (inst.Parent()->terminator.loop_header) return true;
		const auto at = Of(inst.Parent());
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (Of(inst.PhiBlock(index)) >= at) return true;
		}
		return false;
	}

private:
	std::unordered_map<const Block*, size_t> m_index;
};

Value Incoming(const Inst& phi, const Block* from) {
	for (size_t index = 0; index < phi.NumArgs(); ++index) {
		if (phi.PhiBlock(index) == from) return phi.Arg(index);
	}
	return {};
}

using FForm = std::array<uint32_t, 7>;
using World = std::array<std::vector<Value>, 6>;

struct Blas {
	uint32_t address = 0;
	bool     packed  = false;
};

std::optional<Blas> BlasOf(Store& store, uint32_t form);

class Evaluator {
public:
	Evaluator(const Program& program, const Order& order, Store& store, Value active)
	    : m_program(program), m_order(order), m_store(store) {
		if (!active.IsEmpty()) AddFact(active);
	}

	std::optional<bool> Decide(Value condition, uint32_t depth = 0) const {
		condition = condition.Resolve();
		if (condition.IsEmpty() || depth > 12) return std::nullopt;
		if (condition.IsImmediate()) return condition.GetType() == Type::U1 ? std::optional(condition.U1()) : std::nullopt;
		if (std::ranges::find(m_facts, condition) != m_facts.end()) return true;
		const auto* inst = condition.TryInstruction();
		switch (inst->GetOpcode()) {
			case ValueOpcode::ConditionRef: return Decide(inst->Arg(0), depth + 1);
			case ValueOpcode::LogicalNot: {
				const auto inner = Decide(inst->Arg(0), depth + 1);
				return inner ? std::optional(!*inner) : std::nullopt;
			}
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr: {
				const bool any = inst->GetOpcode() == ValueOpcode::LogicalOr;
				const auto a = Decide(inst->Arg(0), depth + 1), b = Decide(inst->Arg(1), depth + 1);
				if (a == any || b == any) return any;
				if (a && b) return !any;
				return std::nullopt;
			}
			case ValueOpcode::SelectU1: {
				const auto pick = Decide(inst->Arg(0), depth + 1);
				return pick ? Decide(inst->Arg(*pick ? 1 : 2), depth + 1) : std::nullopt;
			}
			default: return std::nullopt;
		}
	}

	Value Strip(Value value) const {
		for (uint32_t depth = 0; depth < 32; ++depth) {
			value            = value.Resolve();
			const auto* inst = value.TryInstruction();
			if (inst == nullptr) break;
			const auto op = inst->GetOpcode();
			if (op == ValueOpcode::BitCastF32U32 || op == ValueOpcode::BitCastU32F32 || op == ValueOpcode::ReadFirstLane ||
			    op == ValueOpcode::ReadLane) {
				value = inst->Arg(0);
				continue;
			}
			if (op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectF32) {
				const auto pick = Decide(inst->Arg(0));
				if (!pick) break;
				value = inst->Arg(*pick ? 1 : 2);
				continue;
			}
			break;
		}
		return value;
	}

	Choices Word(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U32: return {m_store.Constant(32, value.U32())};
				case Type::F32: return {m_store.Constant(32, std::bit_cast<uint32_t>(value.F32Value()))};
				case Type::U1: return {m_store.Constant(32, value.U1() ? 1u : 0u)};
				default: return {};
			}
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		if (const auto at = m_words.find(inst); at != m_words.end()) return at->second;
		if (!m_busy.insert(inst).second) return {Opaque(*inst, 32)};
		auto result = WordOf(*inst);
		m_busy.erase(inst);
		Cap(result);
		m_words[inst] = result;
		return result;
	}

	Choices Wide(Value value) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return value.GetType() == Type::U64 ? Choices {m_store.Constant(64, value.U64())} : Choices {};
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		if (const auto at = m_wides.find(inst); at != m_wides.end()) return at->second;
		if (!m_busy.insert(inst).second) return {Opaque(*inst, 64)};
		auto result = WideOf(*inst);
		m_busy.erase(inst);
		Cap(result);
		m_wides[inst] = result;
		return result;
	}

	Choices Pair(Value lo, Value hi, uint32_t depth = 0) {
		lo = lo.Resolve();
		hi = hi.Resolve();
		if (depth > 48) return {};
		const auto* l = lo.TryInstruction();
		const auto* h = hi.TryInstruction();
		if (l != nullptr && l->GetOpcode() == ValueOpcode::SelectU32) {
			if (const auto pick = Decide(l->Arg(0))) return Pair(l->Arg(*pick ? 1 : 2), hi, depth + 1);
		}
		if (h != nullptr && h->GetOpcode() == ValueOpcode::SelectU32) {
			if (const auto pick = Decide(h->Arg(0))) return Pair(lo, h->Arg(*pick ? 1 : 2), depth + 1);
		}
		if (l != nullptr && h != nullptr && l->GetOpcode() == ValueOpcode::SelectU32 &&
		    h->GetOpcode() == ValueOpcode::SelectU32 && l->Arg(0).Resolve() == h->Arg(0).Resolve()) {
			auto result = Pair(l->Arg(1), h->Arg(1), depth + 1);
			Append(result, Pair(l->Arg(2), h->Arg(2), depth + 1));
			return result;
		}
		if (l != nullptr && h != nullptr && l->GetOpcode() == ValueOpcode::Phi && h->GetOpcode() == ValueOpcode::Phi &&
		    l->Parent() == h->Parent()) {
			if (m_order.Carried(*l)) return {Carried(*l, *h)};
			if (!m_busy.insert(l).second) return {Opaque(*l, 64)};
			Choices result;
			for (size_t index = 0; index < l->NumArgs(); ++index) {
				const auto from = l->PhiBlock(index);
				Append(result, Pair(l->Arg(index), Incoming(*h, from), depth + 1));
			}
			m_busy.erase(l);
			return result;
		}
		std::vector<Value> leaves;
		Flatten(hi, leaves, 0);
		std::vector<const Inst*> carries;
		std::vector<const Inst*> highs;
		uint32_t                 high = 0;
		Choices                  rest {m_store.Constant(32, 0)};
		for (const auto leaf: leaves) {
			if (const auto* carry = CarryOf(leaf)) {
				carries.push_back(carry);
			} else if (const auto* product = ProducerOf(leaf, ValueOpcode::UMulHi)) {
				highs.push_back(product);
			} else if (const auto constant = ImmediateU32(leaf)) {
				high += *constant;
			} else {
				Choices next;
				for (const auto sum: rest) {
					for (const auto word: Word(leaf)) next.push_back(m_store.Add(sum, word));
				}
				Cap(next);
				rest = std::move(next);
			}
		}
		Choices result;
		for (const auto& lift: Lift(lo, 0, leaves)) {
			if (lift.adds.size() != carries.size() ||
			    !std::ranges::all_of(carries, [&](const Inst* carry) { return std::ranges::find(lift.adds, carry) != lift.adds.end(); })) {
				continue;
			}
			if (lift.products.size() != highs.size() ||
			    !std::ranges::all_of(highs, [&](const Inst* product) {
				    return std::ranges::any_of(lift.products, [&](const Inst* low) {
					    const auto a = low->Arg(0).Resolve(), b = low->Arg(1).Resolve();
					    const auto c = product->Arg(0).Resolve(), d = product->Arg(1).Resolve();
					    return (a == c && b == d) || (a == d && b == c);
				    });
			    })) {
				continue;
			}
			if (std::ranges::find(rest, lift.expected) == rest.end()) continue;
			result.push_back(m_store.Add(lift.form, m_store.Constant(64, static_cast<uint64_t>(high) << 32u)));
		}
		if (result.empty()) {
			Choices loose;
			for (const auto low: Word(lo)) {
				for (const auto top: Word(hi)) (Adjacent(low, top) ? result : loose).push_back(m_store.Join(low, top));
			}
			Append(result, loose);
		}
		Cap(result);
		return result;
	}

	bool Adjacent(uint32_t low, uint32_t high) {
		const auto la = m_store.AtomOf(low);
		if (!la || m_store.A(*la).kind != Kind::Load || m_store.A(*la).args[1] != 4) return false;
		const auto next = m_store.Add(static_cast<uint32_t>(m_store.A(*la).args[0]), m_store.Constant(64, 4));
		auto       word = high;
		if (const auto* cut = m_store.OpOf(high, static_cast<uint64_t>(ValueOpcode::BitwiseAnd32)); cut != nullptr && cut->args.size() == 3) {
			if ((cut->args[2] & Literal) != 0 && (cut->args[1] & Literal) == 0) word = static_cast<uint32_t>(cut->args[1]);
			if ((cut->args[1] & Literal) != 0 && (cut->args[2] & Literal) == 0) word = static_cast<uint32_t>(cut->args[2]);
		}
		const auto ha = m_store.AtomOf(word);
		return ha && m_store.A(*ha).kind == Kind::Load && m_store.A(*ha).args[1] == 4 && m_store.A(*ha).args[0] == next;
	}

	Choices Expanded(Value value) {
		value            = Strip(value);
		const auto* inst = value.TryInstruction();
		Choices     result;
		if (inst != nullptr && inst->GetOpcode() == ValueOpcode::CompositeConstructU64) {
			const auto* l = Strip(inst->Arg(0)).TryInstruction();
			const auto* h = Strip(inst->Arg(1)).TryInstruction();
			if (l != nullptr && h != nullptr && m_order.Carried(*l) && m_order.Carried(*h) && l->Parent() == h->Parent()) {
				for (size_t index = 0; index < l->NumArgs(); ++index) {
					if (auto* edge = Along(*l, index)) Append(result, edge->Pair(l->Arg(index), Incoming(*h, l->PhiBlock(index))));
				}
			}
		} else if (inst != nullptr && m_order.Carried(*inst) && TypeOf(inst->GetOpcode()) != Type::U32) {
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				if (auto* edge = Along(*inst, index)) Append(result, edge->Wide(inst->Arg(index)));
			}
		}
		Append(result, Wide(value));
		return result;
	}

	Choices Unroll(uint32_t form) {
		const auto atom = m_store.AtomOf(form);
		if (!atom) return {form};
		const auto a = m_store.A(*atom);
		if (a.kind == Kind::Opaque && a.width == 64 && a.args.empty() && a.inst != nullptr && m_order.Carried(*a.inst)) {
			Choices result;
			for (size_t index = 0; index < a.inst->NumArgs(); ++index) {
				if (auto* edge = Along(*a.inst, index)) Append(result, edge->Wide(a.inst->Arg(index)));
			}
			return result;
		}
		if (a.kind != Kind::Opaque || a.width != 64 || a.args.size() != 1 || a.inst == nullptr) return {form};
		const auto* high = reinterpret_cast<const Inst*>(static_cast<uintptr_t>(a.args[0]));
		Choices     result;
		for (size_t index = 0; index < a.inst->NumArgs(); ++index) {
			if (auto* edge = Along(*a.inst, index)) Append(result, edge->Pair(a.inst->Arg(index), Incoming(*high, a.inst->PhiBlock(index))));
		}
		return result;
	}

	Choices Bases(uint32_t form, uint32_t depth = 0) {
		const auto atom = m_store.AtomOf(form);
		if (!atom || depth > 3) return {};
		const auto a = m_store.A(*atom);
		Choices    result;
		if (a.kind == Kind::Op && a.args.size() == 3 && a.args[0] == static_cast<uint64_t>(ValueOpcode::ShiftRightLogical64) &&
		    a.args[2] == (Literal | 3u)) {
			for (const auto base: Unroll(static_cast<uint32_t>(a.args[1]))) Append(result, {base});
		} else if (a.kind == Kind::Opaque && a.width == 64 && a.args.size() == 1) {
			for (const auto carried: Unroll(form)) {
				if (carried != form) Append(result, Bases(carried, depth + 1));
			}
		} else if (a.kind == Kind::Join && a.width == 64) {
			const auto address = m_store.Scale(form, 8);
			if (BlasOf(m_store, address)) Append(result, {address});
		}
		return result;
	}

	Choices Address(const Inst& load) {
		const auto& memory = m_program.memory_info[load.Flags<MemoryFlags>().index];
		auto        offset = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(memory.offset)));
		Choices     bases;
		auto&       ev = load.NumArgs() > 3 ? Given(load.Arg(3)) : *this;
		if (memory.address_is_full) {
			bases = ev.Pair(load.Arg(1), load.Arg(2));
		} else {
			const auto* handle = ProducerOf(load.Arg(0), ValueOpcode::GetAddressResource);
			if (handle == nullptr) return {};
			for (const auto base: ev.Pair(handle->Arg(0), handle->Arg(1))) {
				for (const auto index: ev.Word(load.Arg(1))) bases.push_back(m_store.Add(base, m_store.Widen(index)));
			}
		}
		Choices result;
		for (const auto base: bases) result.push_back(m_store.Add(base, m_store.Constant(64, offset)));
		Cap(result);
		return result;
	}

	Choices BufferAddress(const Inst& read) {
		const auto& memory = m_program.memory_info[read.Flags<MemoryFlags>().index];
		const auto* buffer = ProducerOf(read.Arg(0), ValueOpcode::GetBufferResource);
		if (buffer == nullptr) return {};
		Choices result;
		for (const auto base: Pair(buffer->Arg(0), buffer->Arg(1))) {
			const auto start = m_store.Op(64, {Base48, base});
			for (const auto index: Word(read.Arg(1))) {
				result.push_back(m_store.Add(m_store.Add(start, m_store.Widen(index)), m_store.Constant(64, memory.offset)));
			}
		}
		Cap(result);
		return result;
	}

	std::vector<FForm> Float(Value value, const World& world, uint32_t depth = 0) {
		value = value.Resolve();
		if (depth > 48) return {};
		for (uint32_t k = 0; k < 6; ++k) {
			if (std::ranges::find(world[k], value) != world[k].end()) return {Single(k, One)};
		}
		if (const auto* read = value.TryInstruction();
		    read != nullptr && (read->GetOpcode() == ValueOpcode::ReadConstBuffer || read->GetOpcode() == ValueOpcode::LoadAddressU32)) {
			const auto words = Word(value);
			for (uint32_t k = 0; k < 6 && words.size() == 1; ++k) {
				for (const auto other: world[k]) {
					const auto* same = other.TryInstruction();
					if (same == nullptr || same == read || same->GetOpcode() != read->GetOpcode()) continue;
					const auto theirs = Word(other);
					if (theirs.size() == 1 && theirs.front() == words.front()) return {Single(k, One)};
				}
			}
		}
		if (value.IsImmediate()) {
			const auto bits = value.GetType() == Type::F32 ? std::bit_cast<uint32_t>(value.F32Value())
			                  : value.GetType() == Type::U32 ? value.U32()
			                                                 : 1u;
			if ((bits & 0x7fffffffu) == 0) return {Empty()};
			return {Single(6, m_store.Constant(32, bits))};
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		if (const auto at = m_floats.find(inst); at != m_floats.end()) return at->second;
		auto result = FloatOf(*inst, value, world, depth);
		m_floats[inst] = result;
		return result;
	}

	std::vector<FForm> FloatOf(const Inst& node, Value value, const World& world, uint32_t depth) {
		const auto* inst = &node;
		switch (inst->GetOpcode()) {
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadLane:
			case ValueOpcode::FPNeg32: return Float(inst->Arg(0), world, depth + 1);
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectF32: {
				if (const auto pick = Decide(inst->Arg(0))) return Float(inst->Arg(*pick ? 1 : 2), world, depth + 1);
				auto result = Float(inst->Arg(1), world, depth + 1);
				for (const auto& other: Float(inst->Arg(2), world, depth + 1)) result.push_back(other);
				return Limit(std::move(result));
			}
			case ValueOpcode::Phi: {
				if (m_order.Carried(*inst) || !m_busy.insert(inst).second) break;
				std::vector<FForm> result;
				for (size_t index = 0; index < inst->NumArgs(); ++index) {
					for (const auto& other: Float(inst->Arg(index), world, depth + 1)) result.push_back(other);
				}
				m_busy.erase(inst);
				return Limit(std::move(result));
			}
			case ValueOpcode::FPAdd32:
			case ValueOpcode::FPSub32: return Merge(Float(inst->Arg(0), world, depth + 1), Float(inst->Arg(1), world, depth + 1));
			case ValueOpcode::FPMul32: return Product(Float(inst->Arg(0), world, depth + 1), Float(inst->Arg(1), world, depth + 1));
			case ValueOpcode::FPFma32:
			case ValueOpcode::FPMad32:
				return Merge(Product(Float(inst->Arg(0), world, depth + 1), Float(inst->Arg(1), world, depth + 1)),
				             Float(inst->Arg(2), world, depth + 1));
			default: break;
		}
		std::vector<FForm> result;
		for (const auto word: Word(value)) result.push_back(Single(6, word));
		return result;
	}

private:
	struct LiftResult {
		uint32_t                 form     = 0;
		uint32_t                 expected = 0;
		std::vector<const Inst*> adds;
		std::vector<const Inst*> products;
	};

	Evaluator& Given(Value fact) {
		fact = fact.Resolve();
		const auto* inst = fact.TryInstruction();
		if (inst == nullptr || fact.GetType() != Type::U1 || std::ranges::find(m_facts, fact) != m_facts.end()) return *this;
		auto& child = m_given[inst];
		if (!child) {
			child          = std::make_unique<Evaluator>(m_program, m_order, m_store, Value {});
			child->m_facts = m_facts;
			child->AddFact(fact);
		}
		return *child;
	}

	Evaluator* Along(const Inst& phi, size_t index) {
		const auto* header = phi.Parent();
		const auto* from   = phi.PhiBlock(index);
		std::vector<Value> extra;
		for (const auto fact: m_facts) {
			const auto* inst = fact.TryInstruction();
			if (inst == nullptr || inst->GetOpcode() != ValueOpcode::Phi || inst->Parent() != header) continue;
			const auto incoming = Incoming(*inst, from).Resolve();
			if (incoming.IsEmpty()) continue;
			if (incoming.IsImmediate()) {
				if (incoming.GetType() == Type::U1 && !incoming.U1()) return nullptr;
				continue;
			}
			if (std::ranges::find(m_facts, incoming) == m_facts.end()) extra.push_back(incoming);
		}
		if (extra.empty()) return this;
		auto& child = m_edges[{header, from}];
		if (!child) {
			child          = std::make_unique<Evaluator>(m_program, m_order, m_store, Value {});
			child->m_facts = m_facts;
			for (const auto fact: extra) child->AddFact(fact);
		}
		return child.get();
	}

	void AddFact(Value value) {
		value = value.Resolve();
		if (value.IsImmediate() || std::ranges::find(m_facts, value) != m_facts.end()) return;
		m_facts.push_back(value);
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			AddFact(inst->Arg(0));
			AddFact(inst->Arg(1));
		} else if (inst->GetOpcode() == ValueOpcode::ConditionRef) {
			AddFact(inst->Arg(0));
		} else if (inst->GetOpcode() == ValueOpcode::Phi && inst->Parent() != nullptr && inst->Parent()->terminator.loop_header) {
			const auto at = m_order.Of(inst->Parent());
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				if (m_order.Of(inst->PhiBlock(index)) < at) AddFact(inst->Arg(index));
			}
		}
	}

	uint32_t Carried(const Inst& lo, const Inst& hi) {
		return m_store.Of(m_store.Atomic(Kind::Opaque, 64, {reinterpret_cast<uintptr_t>(&hi)}, &lo), 64);
	}

	uint32_t Opaque(const Inst& inst, uint32_t width) { return m_store.Of(m_store.Atomic(Kind::Opaque, width, {}, &inst), width); }

	static FForm Empty() {
		FForm form;
		form.fill(None);
		return form;
	}

	static FForm Single(uint32_t slot, uint32_t value) {
		auto form  = Empty();
		form[slot] = value;
		return form;
	}

	static std::vector<FForm> Limit(std::vector<FForm> forms) {
		std::vector<FForm> unique;
		for (const auto& form: forms) {
			if (std::ranges::find(unique, form) == unique.end()) unique.push_back(form);
		}
		if (unique.size() > MaxChoices * 2) unique.resize(MaxChoices * 2);
		return unique;
	}

	static std::vector<FForm> Merge(const std::vector<FForm>& a, const std::vector<FForm>& b) {
		std::vector<FForm> result;
		for (const auto& x: a) {
			for (const auto& y: b) {
				FForm sum   = x;
				bool  clash = false;
				for (uint32_t slot = 0; slot < 7; ++slot) {
					if (y[slot] == None) continue;
					clash      = clash || sum[slot] != None;
					sum[slot] = y[slot];
				}
				if (!clash) result.push_back(sum);
			}
		}
		return Limit(std::move(result));
	}

	static std::vector<FForm> Product(const std::vector<FForm>& a, const std::vector<FForm>& b) {
		const auto world = [](const FForm& form) -> std::optional<uint32_t> {
			std::optional<uint32_t> found;
			for (uint32_t slot = 0; slot < 7; ++slot) {
				if (form[slot] == None) continue;
				if (slot == 6 || form[slot] != One || found) return std::nullopt;
				found = slot;
			}
			return found;
		};
		const auto scalar = [](const FForm& form) -> std::optional<uint32_t> {
			for (uint32_t slot = 0; slot < 6; ++slot) {
				if (form[slot] != None) return std::nullopt;
			}
			return form[6] != None ? std::optional(form[6]) : std::nullopt;
		};
		std::vector<FForm> result;
		for (const auto& x: a) {
			for (const auto& y: b) {
				if (const auto k = world(x), s = scalar(y); k && s) result.push_back(Single(*k, *s));
				if (const auto k = world(y), s = scalar(x); k && s) result.push_back(Single(*k, *s));
			}
		}
		return Limit(std::move(result));
	}

	void Flatten(Value value, std::vector<Value>& leaves, uint32_t depth) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst != nullptr && depth < 8) {
			if (inst->GetOpcode() == ValueOpcode::IAdd32) {
				Flatten(inst->Arg(0), leaves, depth + 1);
				Flatten(inst->Arg(1), leaves, depth + 1);
				return;
			}
			if (inst->GetOpcode() == ValueOpcode::SelectU32 && !CarryOf(value)) {
				if (const auto pick = Decide(inst->Arg(0))) {
					Flatten(inst->Arg(*pick ? 1 : 2), leaves, depth + 1);
					return;
				}
			}
		}
		leaves.push_back(value);
	}

	const Inst* CarryOf(Value value) const {
		const auto* select = ProducerOf(value, ValueOpcode::SelectU32);
		if (select == nullptr || ImmediateU32(select->Arg(1)) != 1u || ImmediateU32(select->Arg(2)) != 0u) return nullptr;
		Value condition = select->Arg(0);
		const auto compares = [](Value value) {
			const auto* inst = value.Resolve().TryInstruction();
			return inst != nullptr && (inst->GetOpcode() == ValueOpcode::ULessThan32 || inst->GetOpcode() == ValueOpcode::UGreaterThan32);
		};
		for (uint32_t depth = 0; depth < 4; ++depth) {
			const auto* both = ProducerOf(condition, ValueOpcode::LogicalAnd);
			if (both == nullptr) break;
			const bool left  = Decide(both->Arg(0)) == true || (compares(both->Arg(1)) && !compares(both->Arg(0)));
			const bool right = Decide(both->Arg(1)) == true || (compares(both->Arg(0)) && !compares(both->Arg(1)));
			if (left == right) break;
			condition = both->Arg(left ? 1 : 0);
		}
		if (const auto* less = ProducerOf(condition, ValueOpcode::ULessThan32)) {
			return less->Arg(0).Resolve().TryInstruction();
		}
		if (const auto* more = ProducerOf(condition, ValueOpcode::UGreaterThan32)) {
			return more->Arg(1).Resolve().TryInstruction();
		}
		return nullptr;
	}

	std::vector<LiftResult> Lift(Value lo, uint32_t depth, const std::vector<Value>& leaves) {
		lo = lo.Resolve();
		std::vector<LiftResult> result;
		const auto              zero = m_store.Constant(32, 0);
		if (const auto constant = ImmediateU32(lo)) return {{m_store.Constant(64, *constant), zero, {}}};
		const auto* inst = lo.TryInstruction();
		if (inst == nullptr || depth > 12) return {};
		switch (inst->GetOpcode()) {
			case ValueOpcode::IAdd32: {
				const auto left = Lift(inst->Arg(0), depth + 1, leaves), right = Lift(inst->Arg(1), depth + 1, leaves);
				for (size_t diagonal = 0; diagonal + 1 < left.size() + right.size(); ++diagonal) {
					for (size_t i = 0; i <= diagonal && i < left.size(); ++i) {
						if (diagonal - i >= right.size()) continue;
						const auto& a = left[i];
						const auto& b = right[diagonal - i];
						LiftResult sum {m_store.Add(a.form, b.form), m_store.Add(a.expected, b.expected), a.adds, a.products};
						sum.adds.insert(sum.adds.end(), b.adds.begin(), b.adds.end());
						sum.products.insert(sum.products.end(), b.products.begin(), b.products.end());
						sum.adds.push_back(inst);
						result.push_back(std::move(sum));
						if (result.size() >= MaxChoices) return result;
					}
				}
				return result;
			}
			case ValueOpcode::ShiftLeftLogical32: {
				const auto shift = ImmediateU32(inst->Arg(1));
				if (!shift || *shift == 0 || *shift >= 32) break;
				const auto right = static_cast<uint64_t>(ValueOpcode::ShiftRightLogical32);
				for (const auto word: Word(inst->Arg(0))) {
					auto        inner = static_cast<uint64_t>(word);
					uint32_t    total = 32u - *shift;
					const auto* cut   = m_store.OpOf(word, right);
					if (cut != nullptr && cut->args.size() == 3 && (cut->args[2] & Literal) != 0 && (cut->args[1] & Literal) == 0 &&
					    (cut->args[2] & ~Literal) + total < 32u) {
						inner = cut->args[1];
						total += static_cast<uint32_t>(cut->args[2] & ~Literal);
					}
					result.push_back({m_store.Scale(m_store.Widen(word), 1ull << *shift), m_store.Op(32, {right, inner, Literal | total}), {}});
				}
				break;
			}
			case ValueOpcode::IMul32:
				for (uint32_t side = 0; side < 2; ++side) {
					const auto factor = ImmediateU32(inst->Arg(side));
					if (!factor) continue;
					for (const auto word: Word(inst->Arg(1 - side))) {
						result.push_back({m_store.Scale(m_store.Widen(word), *factor), m_store.Constant(32, 0), {}, {inst}});
					}
				}
				break;
			case ValueOpcode::ReadConstBuffer:
				for (const auto address: BufferAddress(*inst)) {
					const auto wide = m_store.Of(m_store.Atomic(Kind::Load, 64, {address, 8}), 64);
					result.push_back({wide, m_store.Hi(wide), {}});
				}
				break;
			case ValueOpcode::LoadAddressU32:
				for (const auto address: Address(*inst)) {
					const auto wide = m_store.Of(m_store.Atomic(Kind::Load, 64, {address, 8}), 64);
					result.push_back({wide, m_store.Hi(wide), {}});
				}
				break;
			case ValueOpcode::GetUserData: {
				const auto reg  = static_cast<uint64_t>(inst->Arg(0).ScalarRegister());
				const auto wide = m_store.Of(m_store.Atomic(Kind::Pointer, 64, {reg}), 64);
				result.push_back({wide, m_store.Hi(wide), {}});
				result.push_back({wide,
				                  m_store.Op(32, {static_cast<uint64_t>(ValueOpcode::BitwiseAnd32), m_store.Hi(wide), Literal | 0xffffu}),
				                  {}});
				break;
			}
			case ValueOpcode::CompositeExtractU64:
				if (ImmediateU32(inst->Arg(1)) == 0u) {
					for (const auto wide: Wide(inst->Arg(0))) result.push_back({wide, m_store.Hi(wide), {}});
				}
				break;
			case ValueOpcode::SelectU32: {
				if (CarryOf(lo)) break;
				if (const auto pick = Decide(inst->Arg(0))) return Lift(inst->Arg(*pick ? 1 : 2), depth + 1, leaves);
				result = Lift(inst->Arg(1), depth + 1, leaves);
				for (auto& other: Lift(inst->Arg(2), depth + 1, leaves)) result.push_back(std::move(other));
				if (result.size() > MaxChoices) result.resize(MaxChoices);
				return result;
			}
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::ReadFirstLane: return Lift(inst->Arg(0), depth + 1, leaves);
			case ValueOpcode::Phi:
				if (!m_order.Carried(*inst)) break;
				for (const auto leaf: leaves) {
					const auto* high = leaf.Resolve().TryInstruction();
					if (high == nullptr || high == inst || !m_order.Carried(*high) || high->Parent() != inst->Parent()) continue;
					for (const auto word: Word(leaf)) result.push_back({Carried(*inst, *high), word, {}});
				}
				break;
			default: break;
		}
		for (const auto word: Word(lo)) result.push_back({m_store.Widen(word), zero, {}});
		if (result.size() > MaxChoices) result.resize(MaxChoices);
		return result;
	}

	Choices OpOf(const Inst& inst, uint32_t width) {
		std::vector<Key> keys {{static_cast<uint64_t>(inst.GetOpcode())}};
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			const auto arg = inst.Arg(index).Resolve();
			Choices    options;
			if (arg.IsImmediate()) {
				if (arg.GetType() == Type::U32) {
					for (auto& key: keys) key.push_back(Literal | arg.U32());
					continue;
				}
				if (arg.GetType() == Type::U64) {
					for (auto& key: keys) key.push_back(Literal | (arg.U64() & ~Literal));
					continue;
				}
				for (auto& key: keys) key.push_back(Literal);
				continue;
			}
			const auto* operand = arg.TryInstruction();
			options = operand != nullptr && TypeOf(operand->GetOpcode()) == Type::U64 ? Wide(arg)
			          : operand != nullptr && TypeOf(operand->GetOpcode()) == Type::U1 ? Choices {Opaque(*operand, 1)}
			                                                                           : Word(arg);
			if (options.empty()) return {Opaque(inst, width)};
			std::vector<Key> next;
			for (const auto& key: keys) {
				for (const auto option: options) {
					auto extended = key;
					extended.push_back(option);
					next.push_back(std::move(extended));
					if (next.size() >= MaxChoices) break;
				}
				if (next.size() >= MaxChoices) break;
			}
			keys = std::move(next);
		}
		Choices result;
		for (auto& key: keys) result.push_back(m_store.Op(width, std::move(key)));
		return result;
	}

	template <typename Fn>
	Choices Combine(Value a, Value b, Fn&& fn) {
		Choices result;
		for (const auto x: Word(a)) {
			for (const auto y: Word(b)) result.push_back(fn(x, y));
		}
		Cap(result);
		return result;
	}

	Choices WordOf(const Inst& inst) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::GetUserData:
				return {m_store.Of(m_store.Atomic(Kind::Word, 32, {static_cast<uint64_t>(inst.Arg(0).ScalarRegister())}), 32)};
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU32F32:
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadLane: return Word(inst.Arg(0));
			case ValueOpcode::SelectU32: {
				if (const auto pick = Decide(inst.Arg(0))) return Word(inst.Arg(*pick ? 1 : 2));
				if (ImmediateU32(inst.Arg(1)) && ImmediateU32(inst.Arg(2))) return OpOf(inst, 32);
				auto result = Word(inst.Arg(1));
				Append(result, Word(inst.Arg(2)));
				return result;
			}
			case ValueOpcode::Phi: {
				if (m_order.Carried(inst)) return {Opaque(inst, 32)};
				Choices result;
				for (size_t index = 0; index < inst.NumArgs(); ++index) Append(result, Word(inst.Arg(index)));
				return result;
			}
			case ValueOpcode::IAdd32: return Combine(inst.Arg(0), inst.Arg(1), [&](uint32_t x, uint32_t y) { return m_store.Add(x, y); });
			case ValueOpcode::ISub32:
				return Combine(inst.Arg(0), inst.Arg(1), [&](uint32_t x, uint32_t y) { return m_store.Add(x, y, ~0ull); });
			case ValueOpcode::IMul32:
				for (uint32_t side = 0; side < 2; ++side) {
					if (const auto factor = ImmediateU32(inst.Arg(side))) {
						Choices result;
						for (const auto x: Word(inst.Arg(1 - side))) result.push_back(m_store.Scale(x, *factor));
						return result;
					}
				}
				return OpOf(inst, 32);
			case ValueOpcode::ShiftLeftLogical32:
				if (const auto shift = ImmediateU32(inst.Arg(1)); shift && *shift < 32u) {
					Choices result;
					for (const auto x: Word(inst.Arg(0))) result.push_back(m_store.Scale(x, 1ull << *shift));
					return result;
				}
				return OpOf(inst, 32);
			case ValueOpcode::CompositeExtractU64: {
				const auto part = ImmediateU32(inst.Arg(1));
				if (!part || *part > 1u) return {Opaque(inst, 32)};
				Choices result;
				for (const auto wide: Wide(inst.Arg(0))) result.push_back(*part == 0 ? m_store.Lo(wide) : m_store.Hi(wide));
				return result;
			}
			case ValueOpcode::LoadAddressU32: {
				Choices result;
				for (const auto address: Address(inst)) result.push_back(m_store.Of(m_store.Atomic(Kind::Load, 32, {address, 4}), 32));
				return result.empty() ? Choices {Opaque(inst, 32)} : result;
			}
			case ValueOpcode::ReadConstBuffer: {
				Choices result;
				for (const auto address: BufferAddress(inst)) result.push_back(m_store.Of(m_store.Atomic(Kind::Load, 32, {address, 4}), 32));
				return result.empty() ? Choices {Opaque(inst, 32)} : result;
			}
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftRightArithmetic32:
			case ValueOpcode::BitFieldUExtract:
			case ValueOpcode::BitFieldSExtract:
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32: return OpOf(inst, 32);
			default: return {Opaque(inst, 32)};
		}
	}

	Choices WideOf(const Inst& inst) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::CompositeConstructU64: return Pair(inst.Arg(0), inst.Arg(1));
			case ValueOpcode::ShiftLeftLogical64:
				if (const auto shift = ImmediateU32(inst.Arg(1)); shift && *shift < 64u) {
					Choices result;
					for (const auto x: Wide(inst.Arg(0))) result.push_back(m_store.Scale(x, 1ull << *shift));
					return result;
				}
				return OpOf(inst, 64);
			case ValueOpcode::IAdd64:
			case ValueOpcode::ISub64: {
				Choices result;
				for (const auto x: Wide(inst.Arg(0))) {
					for (const auto y: Wide(inst.Arg(1))) {
						result.push_back(m_store.Add(x, y, inst.GetOpcode() == ValueOpcode::IAdd64 ? 1ull : ~0ull));
					}
				}
				return result;
			}
			case ValueOpcode::BitCastU64F64: return Wide(inst.Arg(0));
			case ValueOpcode::Phi: {
				if (m_order.Carried(inst)) return {Opaque(inst, 64)};
				Choices result;
				for (size_t index = 0; index < inst.NumArgs(); ++index) Append(result, Wide(inst.Arg(index)));
				return result;
			}
			case ValueOpcode::BitwiseAnd64:
			case ValueOpcode::ShiftRightLogical64: return OpOf(inst, 64);
			default: return {Opaque(inst, 64)};
		}
	}

	const Program&                                  m_program;
	const Order&                                    m_order;
	Store&                                          m_store;
	std::vector<Value>                              m_facts;
	std::unordered_set<const Inst*>                 m_busy;
	std::unordered_map<const Inst*, Choices>        m_words;
	std::unordered_map<const Inst*, Choices>        m_wides;
	std::unordered_map<const Inst*, std::vector<FForm>> m_floats;
	std::unordered_map<const Inst*, std::unique_ptr<Evaluator>> m_given;
	std::map<std::pair<const Block*, const Block*>, std::unique_ptr<Evaluator>> m_edges;
};

struct Source {
	uint32_t user_sgpr     = 0;
	uint32_t header_offset = 0;
	uint32_t header_vsharp = 0xffffffffu;

	bool operator==(const Source&) const = default;
};

std::optional<Source> SourceOf(const Store& store, uint32_t form) {
	const auto atom = store.AtomOf(form);
	if (!atom) return std::nullopt;
	const auto& a = store.A(*atom);
	if (a.kind == Kind::Pointer) return Source {static_cast<uint32_t>(a.args[0]), ShaderInfo::UserDataRayQueryHeader};
	if (a.kind != Kind::Load || a.args[1] != 8) return std::nullopt;
	const auto& address = store.F(static_cast<uint32_t>(a.args[0]));
	if (address.terms.size() != 1 || address.terms[0].second != 1 || address.constant > 0xffffu) return std::nullopt;
	const auto& base = store.A(address.terms[0].first);
	const auto  at   = static_cast<uint32_t>(address.constant);
	if (base.kind == Kind::Pointer) return Source {static_cast<uint32_t>(base.args[0]), at};
	if (base.kind != Kind::Op || base.args.size() != 2 || base.args[0] != Base48) return std::nullopt;
	const auto inner = store.AtomOf(static_cast<uint32_t>(base.args[1]));
	if (!inner) return std::nullopt;
	const auto& vsharp = store.A(*inner);
	if (vsharp.kind == Kind::Pointer) {
		return Source {static_cast<uint32_t>(vsharp.args[0]), at, ShaderInfo::UserDataRayQueryVsharp};
	}
	if (vsharp.kind != Kind::Load || vsharp.args[1] != 8) return std::nullopt;
	const auto& table = store.F(static_cast<uint32_t>(vsharp.args[0]));
	if (table.terms.size() != 1 || table.terms[0].second != 1 || table.constant > 0xffffu) return std::nullopt;
	const auto& user = store.A(table.terms[0].first);
	if (user.kind != Kind::Pointer) return std::nullopt;
	return Source {static_cast<uint32_t>(user.args[0]), at, static_cast<uint32_t>(table.constant)};
}

struct Record {
	bool     indexed = false;
	uint32_t key     = 0;
	uint64_t offset  = 0;
	uint32_t stride  = 0;
	uint32_t base    = 0;
	uint32_t node    = 0;
};

std::optional<Record> RecordOf(Store& store, uint32_t address, uint32_t header) {
	const auto form   = store.F(address);
	const auto top    = store.F(header);
	Record     record {.key = store.Strip(address), .offset = form.constant};
	auto       rest   = form.terms;
	for (const auto& term: top.terms) {
		const auto at = std::ranges::find(rest, term);
		if (at == rest.end()) return std::nullopt;
		rest.erase(at);
	}
	if (top.constant != 0) record.offset -= top.constant;
	if (rest.empty()) return record;
	if (rest.size() != 2) return std::nullopt;
	for (const auto& [atom, scale]: rest) {
		const auto& a = store.A(atom);
		if (a.kind == Kind::Load && a.args[1] == 8 && scale == 1) {
			const auto& at = store.F(static_cast<uint32_t>(a.args[0]));
			if (store.Strip(static_cast<uint32_t>(a.args[0])) != store.Strip(header) || at.constant < top.constant) return std::nullopt;
			record.base = static_cast<uint32_t>(at.constant - top.constant);
			continue;
		}
		if (a.kind == Kind::Op && a.args.size() == 3 && a.args[0] == static_cast<uint64_t>(ValueOpcode::ShiftRightLogical32) &&
		    a.args[2] == (Literal | 3u) && scale >= 128u && scale <= 224u && scale % 32u == 0) {
			record.stride = static_cast<uint32_t>(scale);
			record.node   = static_cast<uint32_t>(a.args[1]);
			continue;
		}
		return std::nullopt;
	}
	if (record.stride == 0) return std::nullopt;
	record.indexed = true;
	return record;
}

std::optional<Blas> BlasOf(Store& store, uint32_t form) {
	if (const auto atom = store.AtomOf(form)) {
		const auto& a = store.A(*atom);
		if (a.kind == Kind::Load && a.args[1] == 8) return Blas {static_cast<uint32_t>(a.args[0]), false};
		return std::nullopt;
	}
	const auto& f = store.F(form);
	if (f.constant != 0 || f.terms.size() != 1 || f.terms[0].second != 8) return std::nullopt;
	const auto& join = store.A(f.terms[0].first);
	if (join.kind != Kind::Join) return std::nullopt;
	const auto low  = store.AtomOf(static_cast<uint32_t>(join.args[0]));
	const auto* cut = store.OpOf(static_cast<uint32_t>(join.args[1]), static_cast<uint64_t>(ValueOpcode::BitwiseAnd32));
	if (!low || cut == nullptr || cut->args.size() != 3) return std::nullopt;
	const bool mask_last = cut->args[2] == (Literal | 0x3fffffu) && (cut->args[1] & Literal) == 0;
	const bool mask_first = cut->args[1] == (Literal | 0x3fffffu) && (cut->args[2] & Literal) == 0;
	if (!mask_last && !mask_first) return std::nullopt;
	const auto high = store.AtomOf(static_cast<uint32_t>(cut->args[mask_last ? 1 : 2]));
	const auto& l   = store.A(*low);
	if (!high || l.kind != Kind::Load || l.args[1] != 4) return std::nullopt;
	const auto& h = store.A(*high);
	if (h.kind != Kind::Load || h.args[1] != 4 || h.args[0] != store.Add(static_cast<uint32_t>(l.args[0]), store.Constant(64, 4))) {
		return std::nullopt;
	}
	return Blas {static_cast<uint32_t>(l.args[0]), true};
}

struct Site {
	Inst*              bvh        = nullptr;
	const Inst*        descriptor = nullptr;
	const Inst*        address    = nullptr;
	uint32_t           words      = 1;
	Choices            bases;
	std::vector<Value> nodes;
	bool               top        = false;
	std::vector<Blas>  blases;
};

using Assumed = std::vector<std::pair<Value, bool>>;

std::optional<bool> TypeHolds(Value condition, uint32_t type, const Assumed& assumed, uint32_t depth = 0);

std::optional<uint32_t> TypedWord(Value value, uint32_t type, const Assumed& assumed, uint32_t depth = 0) {
	value = value.Resolve();
	if (const auto constant = ImmediateU32(value)) return constant;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 24) return std::nullopt;
	const auto arg = [&](size_t index) { return TypedWord(inst->Arg(index), type, assumed, depth + 1); };
	switch (inst->GetOpcode()) {
		case ValueOpcode::BitwiseAnd32: {
			for (uint32_t side = 0; side < 2; ++side) {
				if (ImmediateU32(inst->Arg(side)) == 7u && !ImmediateU32(inst->Arg(1 - side))) return type;
			}
			const auto a = arg(0), b = arg(1);
			if (a == 0u || b == 0u) return 0u;
			return a && b ? std::optional(*a & *b) : std::nullopt;
		}
		case ValueOpcode::BitwiseOr32: {
			const auto a = arg(0), b = arg(1);
			return a && b ? std::optional(*a | *b) : std::nullopt;
		}
		case ValueOpcode::BitwiseNot32: {
			const auto a = arg(0);
			return a ? std::optional(~*a) : std::nullopt;
		}
		case ValueOpcode::IAdd32: {
			const auto a = arg(0), b = arg(1);
			return a && b ? std::optional(*a + *b) : std::nullopt;
		}
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftLeftLogical32: {
			const auto a = arg(0), b = arg(1);
			if (a == 0u) return 0u;
			if (!a || !b || *b >= 32u) return std::nullopt;
			return inst->GetOpcode() == ValueOpcode::ShiftRightLogical32 ? *a >> *b : *a << *b;
		}
		case ValueOpcode::SelectU32: {
			if (const auto pick = TypeHolds(inst->Arg(0), type, assumed, depth + 1)) return arg(*pick ? 1 : 2);
			const auto a = arg(1), b = arg(2);
			return a && a == b ? a : std::nullopt;
		}
		case ValueOpcode::CompositeExtractU32x4: {
			const auto* ballot = ProducerOf(inst->Arg(0), ValueOpcode::Ballot);
			if (ballot != nullptr && TypeHolds(ballot->Arg(0), type, assumed, depth + 1) == false) return 0u;
			return std::nullopt;
		}
		default: return std::nullopt;
	}
}

std::optional<bool> TypeHolds(Value condition, uint32_t type, const Assumed& assumed, uint32_t depth) {
	condition = condition.Resolve();
	if (condition.IsImmediate()) return condition.GetType() == Type::U1 ? std::optional(condition.U1()) : std::nullopt;
	for (const auto& [value, holds]: assumed) {
		if (value == condition) return holds;
	}
	for (const auto& [value, holds]: assumed) {
		if (!holds) continue;
		std::vector<Value> pending {value};
		for (size_t at = 0; at < pending.size() && at < 64; ++at) {
			const auto* both = pending[at].Resolve().TryInstruction();
			if (both == nullptr) continue;
			if (both->GetOpcode() == ValueOpcode::LogicalAnd || both->GetOpcode() == ValueOpcode::ConditionRef) {
				for (size_t index = 0; index < both->NumArgs(); ++index) {
					if (both->Arg(index).Resolve() == condition) return true;
					pending.push_back(both->Arg(index));
				}
			}
		}
	}
	const auto* inst = condition.TryInstruction();
	if (inst == nullptr || depth > 24) return std::nullopt;
	const auto op = inst->GetOpcode();
	const auto sub = [&](size_t index) { return TypeHolds(inst->Arg(index), type, assumed, depth + 1); };
	switch (op) {
		case ValueOpcode::ConditionRef: return sub(0);
		case ValueOpcode::LogicalNot: {
			const auto inner = sub(0);
			return inner ? std::optional(!*inner) : std::nullopt;
		}
		case ValueOpcode::LogicalAnd: {
			const auto a = sub(0), b = sub(1);
			if (a == false || b == false) return false;
			if (a && b) return true;
			return std::nullopt;
		}
		case ValueOpcode::LogicalOr: {
			const auto a = sub(0), b = sub(1);
			if (a == true || b == true) return true;
			if (a && b) return false;
			return std::nullopt;
		}
		case ValueOpcode::SelectU1: {
			if (const auto pick = sub(0)) return sub(*pick ? 1 : 2);
			const auto a = sub(1), b = sub(2);
			return a && a == b ? a : std::nullopt;
		}
		case ValueOpcode::IEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThanEqual32: break;
		default: return std::nullopt;
	}
	const auto a = TypedWord(inst->Arg(0), type, assumed, depth + 1), b = TypedWord(inst->Arg(1), type, assumed, depth + 1);
	if (!a || !b) return std::nullopt;
	switch (op) {
		case ValueOpcode::IEqual32: return *a == *b;
		case ValueOpcode::INotEqual32: return *a != *b;
		case ValueOpcode::ULessThan32: return *a < *b;
		case ValueOpcode::ULessThanEqual32: return *a <= *b;
		case ValueOpcode::UGreaterThan32: return *a > *b;
		default: return *a >= *b;
	}
}

std::optional<uint32_t> TypesWhere(Value condition, bool wanted, const Assumed& assumed = {}) {
	uint32_t types = 0;
	for (uint32_t type = 0; type < 8; ++type) {
		if (TypeHolds(condition, type, assumed) == wanted) types |= 1u << type;
	}
	return types != 0 && types != 0xffu ? std::optional(types) : std::nullopt;
}

std::optional<uint32_t> EdgeTypes(const Block* to, const Block* from) {
	for (uint32_t depth = 0; depth < 6 && from != nullptr; ++depth) {
		if (from->terminator.kind == CFG::TerminatorKind::ConditionalBranch && from->terminator.true_block != from->terminator.false_block) {
			if (const auto types = TypesWhere(from->condition, to == from->terminator.true_block)) return types;
		}
		const auto preds = from->ImmPredecessors();
		if (preds.size() != 1) return std::nullopt;
		to   = from;
		from = preds[0];
	}
	return std::nullopt;
}

Block* EnclosingLoop(const Program& program, const Block* block) {
	const auto at = BlockIndex(program, block);
	for (auto header = at; header-- > 0;) {
		const auto& term = program.blocks[header]->terminator;
		if (term.loop_header && term.continue_block != nullptr && BlockIndex(program, term.continue_block) >= at) {
			return program.blocks[header];
		}
	}
	return nullptr;
}

Block* OutermostLoop(const Program& program, const Block* block) {
	Block* loop = EnclosingLoop(program, block);
	while (loop != nullptr) {
		const auto at    = BlockIndex(program, loop);
		const auto end   = BlockIndex(program, loop->terminator.continue_block);
		Block*     outer = nullptr;
		for (auto header = at; header-- > 0 && outer == nullptr;) {
			const auto& term = program.blocks[header]->terminator;
			if (term.loop_header && term.continue_block != nullptr && BlockIndex(program, term.continue_block) >= end) {
				outer = program.blocks[header];
			}
		}
		if (outer == nullptr) break;
		loop = outer;
	}
	return loop;
}

bool InLoop(const Program& program, const Block* block, const Block* header, const Block* latch) {
	const auto at = BlockIndex(program, block);
	return at >= BlockIndex(program, header) && at <= BlockIndex(program, latch);
}

std::vector<std::pair<Value, uint32_t>> NodeSources(const Evaluator& ev, const std::vector<Value>& starts) {
	std::vector<std::pair<Value, uint32_t>> found;
	std::unordered_set<const Inst*>         seen;
	std::vector<std::pair<Value, uint32_t>> queue;
	for (const auto start: starts) queue.emplace_back(start, 0);
	for (size_t at = 0; at < queue.size() && at < 8192; ++at) {
		const auto [raw, distance] = queue[at];
		const auto value           = raw.Resolve();
		if (value.IsImmediate()) {
			found.emplace_back(value, distance);
			continue;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || !seen.insert(inst).second) continue;
		switch (inst->GetOpcode()) {
			case ValueOpcode::Phi:
				for (size_t index = 0; index < inst->NumArgs(); ++index) queue.emplace_back(inst->Arg(index), distance + 1);
				break;
			case ValueOpcode::SelectU32:
				if (const auto pick = ev.Decide(inst->Arg(0))) {
					queue.emplace_back(inst->Arg(*pick ? 1 : 2), distance + 1);
				} else {
					queue.emplace_back(inst->Arg(1), distance + 1);
					queue.emplace_back(inst->Arg(2), distance + 1);
				}
				break;
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadLane: queue.emplace_back(inst->Arg(0), distance + 1); break;
			case ValueOpcode::WriteLane:
				queue.emplace_back(inst->Arg(0), distance + 1);
				queue.emplace_back(inst->Arg(1), distance + 1);
				break;
			case ValueOpcode::LoadAddressU32:
			case ValueOpcode::ReadConstBuffer: found.emplace_back(value, distance); break;
			default: break;
		}
	}
	return found;
}

struct Fields {
	std::vector<uint32_t>   indexed_keys;
	uint32_t                inline_key    = None;
	uint64_t                inline_offset = 0;

	std::optional<uint32_t> Dword(Store& store, uint32_t address) const {
		const auto key    = store.Strip(address);
		const auto offset = store.F(address).constant;
		if (std::ranges::find(indexed_keys, key) != indexed_keys.end() && offset % 4u == 0 && offset < 256u) {
			return static_cast<uint32_t>(offset / 4u);
		}
		if (key == inline_key && offset >= inline_offset && (offset - inline_offset) % 4u == 0 && offset - inline_offset < 256u) {
			return static_cast<uint32_t>((offset - inline_offset) / 4u);
		}
		return std::nullopt;
	}

	std::optional<uint32_t> LoadDword(Store& store, uint32_t form) const {
		const auto atom = store.AtomOf(form);
		if (!atom) return std::nullopt;
		const auto& a = store.A(*atom);
		if (a.kind != Kind::Load || a.args[1] != 4) return std::nullopt;
		return Dword(store, static_cast<uint32_t>(a.args[0]));
	}
};

struct MaskField {
	uint32_t dword    = 0;
	uint32_t shift    = 0;
	uint32_t width    = 0;
	bool     variable = false;
	bool operator==(const MaskField&) const = default;
};

std::optional<MaskField> MaskOf(Store& store, const Fields& fields, uint32_t form, bool equal_one) {
	const auto* both = store.OpOf(form, static_cast<uint64_t>(ValueOpcode::BitwiseAnd32));
	const auto* bit  = store.OpOf(form, static_cast<uint64_t>(ValueOpcode::BitFieldUExtract));
	if (bit != nullptr && bit->args.size() == 4 && bit->args[3] == (Literal | 1u) && (bit->args[2] & Literal) != 0 && !(bit->args[1] & Literal)) {
		const auto dword = fields.LoadDword(store, static_cast<uint32_t>(bit->args[1]));
		const auto shift = static_cast<uint32_t>(bit->args[2] & 31u);
		if (dword) return MaskField {*dword, shift, 1};
	}
	if (both == nullptr || both->args.size() != 3 || equal_one) return std::nullopt;
	for (uint32_t side = 1; side <= 2; ++side) {
		const auto x = both->args[side], y = both->args[3 - side];
		if ((x & Literal) != 0) continue;
		const auto* shifted = store.OpOf(static_cast<uint32_t>(x), static_cast<uint64_t>(ValueOpcode::ShiftRightLogical32));
		const auto* field   = store.OpOf(static_cast<uint32_t>(x), static_cast<uint64_t>(ValueOpcode::BitFieldUExtract));
		std::optional<uint32_t> dword;
		uint32_t                shift = 0;
		if (shifted != nullptr && shifted->args.size() == 3 && (shifted->args[2] & Literal) != 0 && !(shifted->args[1] & Literal)) {
			dword = fields.LoadDword(store, static_cast<uint32_t>(shifted->args[1]));
			shift = static_cast<uint32_t>(shifted->args[2] & 31u);
		} else if (field != nullptr && field->args.size() == 4 && field->args[3] == (Literal | 8u) && (field->args[2] & Literal) != 0 &&
		           !(field->args[1] & Literal)) {
			dword = fields.LoadDword(store, static_cast<uint32_t>(field->args[1]));
			shift = static_cast<uint32_t>(field->args[2] & 31u);
		}
		if (dword && (y & Literal) == 0 && !fields.LoadDword(store, static_cast<uint32_t>(y)) && !store.ConstantOf(static_cast<uint32_t>(y))) {
			return MaskField {*dword, shift, 0, true};
		}
		if ((y & Literal) != 0 && (x & Literal) == 0) {
			const auto bits = static_cast<uint32_t>(y & 0xffffffffu);
			const auto at   = fields.LoadDword(store, static_cast<uint32_t>(x));
			if (!at || bits == 0) continue;
			const auto low  = static_cast<uint32_t>(std::countr_zero(bits));
			const auto mask = bits >> low;
			if ((mask & (mask + 1u)) != 0) continue;
			return MaskField {*at, low, static_cast<uint32_t>(std::popcount(mask))};
		}
	}
	return std::nullopt;
}

using Known = std::vector<std::pair<Value, bool>>;

std::optional<bool> Lookup(const Known& known, Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) return value.GetType() == Type::U1 ? std::optional(value.U1()) : std::nullopt;
	for (const auto& [fact, holds]: known) {
		if (fact == value) return holds;
	}
	return std::nullopt;
}

bool Learn(Known& known, Value value, bool holds) {
	value = value.Resolve();
	if (value.IsEmpty()) return true;
	if (const auto already = Lookup(known, value)) return *already == holds;
	if (known.size() > 512) return true;
	known.emplace_back(value, holds);
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return true;
	switch (inst->GetOpcode()) {
		case ValueOpcode::ConditionRef: return Learn(known, inst->Arg(0), holds);
		case ValueOpcode::LogicalNot: return Learn(known, inst->Arg(0), !holds);
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr:
			if (holds == (inst->GetOpcode() == ValueOpcode::LogicalAnd)) {
				const bool first = Learn(known, inst->Arg(0), holds);
				return Learn(known, inst->Arg(1), holds) && first;
			}
			return true;
		default: return true;
	}
}

bool Close(Known& known) {
	bool consistent = true;
	for (bool changed = true; changed && consistent;) {
		changed          = false;
		const auto count = known.size();
		for (size_t at = 0; at < count; ++at) {
			const auto [value, holds] = known[at];
			const auto* inst          = value.TryInstruction();
			if (inst == nullptr) continue;
			const auto op = inst->GetOpcode();
			if (!((op == ValueOpcode::LogicalAnd && !holds) || (op == ValueOpcode::LogicalOr && holds))) continue;
			const bool unit = op == ValueOpcode::LogicalAnd;
			for (uint32_t side = 0; side < 2; ++side) {
				const auto other = Lookup(known, inst->Arg(1 - side));
				if (Lookup(known, inst->Arg(side)) == unit && other != !unit) {
					consistent = consistent && !other && Learn(known, inst->Arg(1 - side), !unit);
					changed    = true;
				}
			}
		}
	}
	return consistent;
}

Known Executes(const Inst& load) {
	Known known;
	if (load.NumArgs() > 3) Learn(known, load.Arg(3), true);
	const Block* to = load.Parent();
	for (uint32_t depth = 0; depth < 8 && to != nullptr; ++depth) {
		const auto preds = to->ImmPredecessors();
		if (preds.size() != 1) break;
		const auto* from = preds[0];
		if (from->terminator.kind == CFG::TerminatorKind::ConditionalBranch && from->terminator.true_block != from->terminator.false_block) {
			Learn(known, from->condition, to == from->terminator.true_block);
		}
		to = from;
	}
	Close(known);
	return known;
}

std::unordered_set<const Inst*> Feeding(const Order& order, std::span<const Value> starts) {
	std::unordered_set<const Inst*> carried, seen;
	std::vector<Value>              pending(starts.begin(), starts.end());
	while (!pending.empty() && seen.size() < 4096) {
		const auto* inst = pending.back().Resolve().TryInstruction();
		pending.pop_back();
		if (inst == nullptr || !seen.insert(inst).second) continue;
		if (inst->GetOpcode() == ValueOpcode::Phi && order.Carried(*inst)) {
			carried.insert(inst);
			continue;
		}
		if (inst->GetOpcode() == ValueOpcode::LoadAddressU32 || inst->GetOpcode() == ValueOpcode::ReadConstBuffer) continue;
		for (size_t index = 0; index < inst->NumArgs(); ++index) pending.push_back(inst->Arg(index));
	}
	return carried;
}

std::optional<bool> Commits(const Order& order, const std::unordered_set<const Inst*>& bases, const Inst& load, Value test, bool want) {
	struct Step {
		const Inst* inst = nullptr;
		Known       known;
		uint32_t    depth = 0;
	};
	std::vector<Step> pending {{&load, Executes(load), 0}};
	bool              reached = false;
	for (size_t visited = 0; !pending.empty() && visited < 2048; ++visited) {
		auto step = std::move(pending.back());
		pending.pop_back();
		for (const auto& use: step.inst->Uses()) {
			const auto* user = use.user;
			const auto  op   = user->GetOpcode();
			if (op == ValueOpcode::MakeImageAddress || op == ValueOpcode::CompositeConstructU32x4 || bases.contains(user)) {
				reached = true;
				if (Lookup(step.known, test) != want) return false;
				continue;
			}
			if (step.depth >= 24 || user->GetType() == Type::U1 || user->GetType() == Type::Void || op == ValueOpcode::LoadAddressU32 ||
			    op == ValueOpcode::ReadConstBuffer || op == ValueOpcode::GetAddressResource) {
				continue;
			}
			Step next {user, step.known, step.depth + 1};
			if (op == ValueOpcode::Phi && order.Carried(*user)) next.known.clear();
			if (op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectF32) {
				if (use.operand == 0) continue;
				if (!Learn(next.known, user->Arg(0), use.operand == 1) || !Close(next.known)) continue;
			}
			pending.push_back(std::move(next));
		}
	}
	return reached ? std::optional(true) : std::nullopt;
}

bool Reach(const Program& program, Block* header, std::vector<std::pair<uint32_t, uint32_t>>& reach, Block*& latch) {
	if (header == nullptr || header->terminator.continue_block == nullptr || header->terminator.merge_block == nullptr) return false;
	latch              = header->terminator.continue_block;
	const auto entries = header->ImmPredecessors();
	if (entries.size() != 2 || std::ranges::find(entries, latch) == entries.end()) return false;
	uint32_t begin = UINT32_MAX;
	uint32_t last  = 0;
	for (auto at = BlockIndex(program, header); at <= BlockIndex(program, latch) && at < program.blocks.size(); ++at) {
		const auto* block = program.blocks[at];
		if (block->end_pc > block->start_pc) {
			begin = std::min(begin, block->start_pc);
			last  = std::max(last, block->end_pc);
		}
	}
	const auto* merge = header->terminator.merge_block;
	const auto  end   = merge->end_pc > merge->start_pc ? merge->start_pc : last;
	if (begin == UINT32_MAX || end <= begin) return false;
	reach.emplace_back(begin, end);
	return true;
}


OracleWalks Refuse(const char* reason, uint32_t detail = 0) {
	OracleWalks walks;
	walks.refusal = reason;
	walks.detail  = detail;
	return walks;
}

}

OracleWalks DeriveOracleWalks(const Program& program, std::span<Inst* const> intersections) {
	if (intersections.empty()) return Refuse("no intersection");
	const Order       order(program);
	Store             store;
	std::vector<Site> sites;
	for (auto* bvh: intersections) {
		Site site {.bvh = bvh};
		site.descriptor = ProducerOf(bvh->Arg(0), ValueOpcode::CompositeConstructU32x4);
		site.address    = ProducerOf(bvh->Arg(1), ValueOpcode::MakeImageAddress);
		site.words      = (bvh->Flags<uint32_t>() & 0xffu) == 2u ? 2u : 1u;
		if (site.descriptor == nullptr || site.address == nullptr) return Refuse("an intersection has no composed descriptor");
		sites.push_back(std::move(site));
	}
	std::vector<std::tuple<Source, uint32_t, uint32_t>> named_sources;
	for (auto& site: sites) {
		Evaluator   ev(program, order, store, site.bvh->Arg(2));
		const auto* low   = ProducerOf(site.descriptor->Arg(0), ValueOpcode::CompositeExtractU64);
		const auto* field = low != nullptr ? ProducerOf(low->Arg(0), ValueOpcode::BitwiseAnd64) : nullptr;
		const auto* shift = field != nullptr ? ProducerOf(field->Arg(0), ValueOpcode::ShiftRightLogical64) : nullptr;
		site.nodes.push_back(site.address->Arg(0));
		if (shift != nullptr && ImmediateU32(shift->Arg(1)) == 8u && ImmediateU32(low->Arg(1)) == 0u) {
			site.bases = ev.Expanded(shift->Arg(0));
		} else if (ImmediateU32(site.descriptor->Arg(0)) == 0u && ImmediateU32(site.descriptor->Arg(1)) == 0x80000000u &&
		           site.words == 2u) {
			site.nodes.clear();
			for (const auto node: ev.Pair(site.address->Arg(0), site.address->Arg(1))) {
				const auto terms = store.F(node).terms;
				for (const auto& [atom, scale]: terms) {
					const auto a = store.A(atom);
					if (scale != 1) continue;
					if (a.width == 32 && a.kind == Kind::Opaque && a.inst != nullptr) {
						const auto value = Value(const_cast<Inst*>(a.inst));
						if (std::ranges::find(site.nodes, value) == site.nodes.end()) site.nodes.push_back(value);
					} else if (a.width == 64) {
						Append(site.bases, ev.Bases(store.Of(atom, 64)));
					}
				}
			}
		}
		for (const auto base: site.bases) {
			if (const auto named = SourceOf(store, base)) {
				const auto at = std::ranges::find_if(named_sources, [&](const auto& entry) { return std::get<0>(entry) == *named; });
				if (at == named_sources.end()) {
					named_sources.emplace_back(*named, base, 1u);
				} else {
					++std::get<2>(*at);
				}
				site.top = true;
				continue;
			}
			if (const auto blas = BlasOf(store, base)) {
				for (const auto address: ev.Unroll(blas->address)) {
					site.blases.push_back({address, blas->packed});
				}
			}
		}
	}
	if (named_sources.empty()) return Refuse("no intersection reads a structure named by the user data");
	std::ranges::stable_sort(named_sources, [](const auto& a, const auto& b) { return std::get<2>(a) > std::get<2>(b); });
	const auto header = std::get<1>(named_sources.front());
	std::optional<Record> indexed;
	std::vector<uint32_t> keys;
	std::vector<std::pair<uint32_t, uint64_t>> inlined;
	std::optional<bool>   packed;
	for (const auto& site: sites) {
		for (const auto& blas: site.blases) {
			const auto record = RecordOf(store, blas.address, header);
			if (!record) continue;
			if (packed && *packed != blas.packed) return Refuse("the BLAS address is read two ways");
			packed = blas.packed;
			if (record->indexed) {
				if (indexed && (indexed->base != record->base || indexed->stride != record->stride || indexed->offset != record->offset)) {
					return Refuse("several instance record arrays");
				}
				if (!indexed) indexed = record;
				keys.push_back(record->key);
			} else {
				inlined.emplace_back(record->key, record->offset);
			}
		}
	}
	if (!indexed) return Refuse("no intersection reads an instance record");
	if (indexed->offset % 4u != 0) return Refuse("the BLAS address is not a record dword");
	Fields fields {.indexed_keys = keys};
	RayTracingLayout layout;
	layout.valid           = true;
	layout.instance_offset = indexed->base;
	layout.record_stride   = indexed->stride;
	layout.blas_dword      = static_cast<uint32_t>(indexed->offset / 4u);
	layout.blas_packed     = packed.value_or(false);
	layout.inline_type     = NoField;
	if (!inlined.empty()) {
		fields.inline_key    = inlined.front().first;
		fields.inline_offset = inlined.front().second - indexed->offset;
	}
	World world {};
	const Site*          top = nullptr;
	for (const auto& site: sites) {
		if (site.top && (top == nullptr || (top->blases.size() > 0 && site.blases.empty()))) top = &site;
	}
	{
		Evaluator ev(program, order, store, top->bvh->Arg(2));
		for (uint32_t k = 0; k < 6; ++k) {
			const auto current = ev.Strip(top->address->Arg(top->words + 1 + k));
			world[k].push_back(current);
			std::vector<Value> leaves {current};
			for (size_t at = 0; at < leaves.size() && at < 16; ++at) {
				const auto* inst = leaves[at].Resolve().TryInstruction();
				if (inst == nullptr) continue;
				const auto op = inst->GetOpcode();
				std::vector<Value> next;
				if (op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectF32) {
					next = {inst->Arg(1), inst->Arg(2)};
				} else if (op == ValueOpcode::BitCastF32U32 || op == ValueOpcode::BitCastU32F32 || op == ValueOpcode::FPNeg32 ||
				           op == ValueOpcode::ReadFirstLane) {
					next = {inst->Arg(0)};
				}
				for (auto value: next) {
					value = value.Resolve();
					if (value.IsImmediate() || std::ranges::find(world[k], value) != world[k].end()) continue;
					world[k].push_back(value);
					leaves.push_back(value);
				}
			}
			const auto* phi = current.TryInstruction();
			if (phi == nullptr || !order.Carried(*phi)) continue;
			for (size_t index = 0; index < phi->NumArgs(); ++index) {
				if (order.Of(phi->PhiBlock(index)) >= order.Of(phi->Parent())) continue;
				std::vector<Value> pending {phi->Arg(index)};
				for (size_t at = 0; at < pending.size() && at < 32; ++at) {
					const auto value = pending[at].Resolve();
					if (value.IsImmediate() || std::ranges::find(world[k], value) != world[k].end()) continue;
					world[k].push_back(value);
					const auto* inst = value.TryInstruction();
					const auto  op   = inst->GetOpcode();
					if (op == ValueOpcode::SelectU32 || op == ValueOpcode::SelectF32) {
						pending.push_back(inst->Arg(1));
						pending.push_back(inst->Arg(2));
					} else if (op == ValueOpcode::BitCastF32U32 || op == ValueOpcode::BitCastU32F32 || op == ValueOpcode::ReadFirstLane) {
						pending.push_back(inst->Arg(0));
					}
				}
			}
		}
	}
	std::optional<std::array<uint8_t, 12>> transform;
	std::vector<const Site*>               lower;
	for (const auto& site: sites) {
		if (site.blases.empty()) continue;
		lower.push_back(&site);
		Evaluator ev(program, order, store, site.bvh->Arg(2));
		std::array<std::vector<FForm>, 6> rays;
		for (uint32_t k = 0; k < 6; ++k) {
			const auto arg   = site.address->Arg(site.words + 1 + k);
			const auto inner = ev.Strip(arg);
			rays[k]          = ev.Float(arg, world);
			if (const auto* phi = inner.TryInstruction(); phi != nullptr && order.Carried(*phi)) {
				for (size_t index = 0; index < phi->NumArgs(); ++index) {
					for (const auto& other: ev.Float(phi->Arg(index), world)) rays[k].push_back(other);
				}
			}
		}
		std::array<uint8_t, 12> found {};
		bool                    complete = true;
		for (uint32_t row = 0; row < 3 && complete; ++row) {
			bool matched = false;
			for (const auto& origin: rays[row]) {
				if (matched) break;
				if (origin[3] != None || origin[4] != None || origin[5] != None || origin[0] == None || origin[1] == None ||
				    origin[2] == None || origin[6] == None) {
					continue;
				}
				std::array<std::optional<uint32_t>, 4> dwords;
				for (uint32_t c = 0; c < 3; ++c) dwords[c] = fields.LoadDword(store, origin[c]);
				dwords[3] = fields.LoadDword(store, origin[6]);
				if (!dwords[0] || !dwords[1] || !dwords[2] || !dwords[3]) continue;
				for (const auto& direction: rays[3 + row]) {
					if (direction[0] != None || direction[1] != None || direction[2] != None || direction[6] != None) continue;
					if (direction[3] != origin[0] || direction[4] != origin[1] || direction[5] != origin[2]) continue;
					for (uint32_t c = 0; c < 4; ++c) found[row * 4 + c] = static_cast<uint8_t>(*dwords[c]);
					matched = true;
					break;
				}
			}
			complete = matched;
		}
		if (!complete) continue;
		if (transform && *transform != found) return Refuse("two instance transforms");
		transform = found;
	}
	if (!transform) return Refuse("the instance transform is not affine in the ray");
	layout.transform = *transform;
	std::unordered_set<uint32_t> floats;
	{
		Evaluator ev(program, order, store, Value {});
		for (auto* block: program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer) continue;
				const bool number = std::ranges::any_of(inst.Uses(), [](const Use& use) {
					return use.user->GetOpcode() == ValueOpcode::BitCastF32U32;
				});
				if (!number) continue;
				for (const auto word: ev.Word(Value(&inst))) {
					if (const auto dword = fields.LoadDword(store, word)) floats.insert(*dword);
				}
			}
		}
	}
	const auto excluded = [&](uint32_t dword) {
		return floats.contains(dword) || std::ranges::find(layout.transform, static_cast<uint8_t>(dword)) != layout.transform.end() ||
		       dword == layout.blas_dword || dword == layout.blas_dword + 1u;
	};
	std::vector<std::pair<uint32_t, uint32_t>> roots;
	std::vector<uint32_t>                      inline_roots;
	for (const auto* site: lower) {
		Evaluator ev(program, order, store, site->bvh->Arg(2));
		for (const auto& [value, distance]: NodeSources(ev, site->nodes)) {
			if (value.IsImmediate()) continue;
			for (const auto word: ev.Word(value)) {
				const auto dword = fields.LoadDword(store, word);
				if (!dword || excluded(*dword)) continue;
				const auto at = std::ranges::find_if(roots, [&](const auto& root) { return root.first == *dword; });
				if (at == roots.end()) {
					roots.emplace_back(*dword, distance);
				} else {
					at->second = std::min(at->second, distance);
				}
				for (const auto& use: value.Resolve().TryInstruction()->Uses()) {
					const auto* user = use.user;
					Value       other;
					if (user->GetOpcode() == ValueOpcode::SelectU32 && use.operand != 0) other = user->Arg(3 - use.operand);
					if (user->GetOpcode() == ValueOpcode::Phi && user->NumArgs() == 2) other = user->Arg(1 - use.operand);
					const auto constant = ImmediateU32(other);
					if (constant && (*constant & 7u) >= 4u && (*constant & 7u) <= 5u &&
					    std::ranges::find(inline_roots, *constant) == inline_roots.end()) {
						inline_roots.push_back(*constant);
					}
				}
			}
		}
	}
	if (roots.empty()) return Refuse("no record dword reaches a node argument");
	std::ranges::stable_sort(roots, [](const auto& a, const auto& b) { return a.second < b.second; });
	std::vector<uint32_t> tops;
	{
		Evaluator ev(program, order, store, top->bvh->Arg(2));
		for (const auto& [value, distance]: NodeSources(ev, top->nodes)) {
			const auto constant = ImmediateU32(value);
			if (!constant || *constant == 0xffffffffu || (*constant & 7u) < 4u || (*constant & 7u) > 5u) continue;
			if (std::ranges::find(tops, *constant) == tops.end()) tops.push_back(*constant);
		}
	}
	if (tops.empty()) return Refuse("the top-level root is not a constant");
	if (!inlined.empty()) {
		std::optional<uint32_t> type;
		uint32_t                types = 0xffu;
		Evaluator               ev(program, order, store, Value {});
		const auto              start = store.Lo(store.Add(header, store.Constant(64, fields.inline_offset)));
		std::vector<uint32_t>   starts;
		for (const auto key: keys) starts.push_back(store.Lo(key));
		const auto in = [&](Value value, bool inline_record) {
			return std::ranges::any_of(ev.Word(value), [&](uint32_t word) {
				if (inline_record ? word == start : std::ranges::find(starts, word) != starts.end()) return true;
				const auto atom = store.AtomOf(word);
				if (!atom || store.A(*atom).kind != Kind::Load) return false;
				const auto key = store.Strip(static_cast<uint32_t>(store.A(*atom).args[0]));
				return inline_record ? key == fields.inline_key : std::ranges::find(keys, key) != keys.end();
			});
		};
		const auto here = [&](Value value) { return in(value, true) && !in(value, false); };
		for (auto* block: program.blocks) {
			for (auto& inst: *block) {
				if (inst.GetOpcode() == ValueOpcode::Phi && !order.Carried(inst)) {
					bool other = false;
					for (size_t index = 0; index < inst.NumArgs(); ++index) other = other || in(inst.Arg(index), false);
					for (size_t index = 0; index < inst.NumArgs() && other; ++index) {
						if (!here(inst.Arg(index))) continue;
						const auto edge = EdgeTypes(inst.Parent(), inst.PhiBlock(index));
						if (edge && *edge != 0) types &= *edge;
					}
				} else if (inst.GetOpcode() == ValueOpcode::SelectU32) {
					for (uint32_t arm = 1; arm <= 2; ++arm) {
						if (!here(inst.Arg(arm)) || !in(inst.Arg(3 - arm), false)) continue;
						Assumed assumed;
						auto    inner = inst.Arg(arm).Resolve();
						for (uint32_t depth = 0; depth < 8; ++depth) {
							const auto* select = ProducerOf(inner, ValueOpcode::SelectU32);
							if (select == nullptr) break;
							const bool first = here(select->Arg(1)), second = here(select->Arg(2));
							if (first == second) break;
							assumed.emplace_back(select->Arg(0).Resolve(), first);
							inner = select->Arg(first ? 1 : 2).Resolve();
						}
						const auto edge = TypesWhere(inst.Arg(0), arm == 1, assumed);
						if (edge && *edge != 0) types &= *edge;
					}
				}
			}
		}
		if (std::popcount(types) == 1) type = static_cast<uint32_t>(std::countr_zero(types));
		if (!type || fields.inline_offset % 4u != 0 || fields.inline_offset >= 2048u) return Refuse("the inline record type is not found");
		layout.inline_type   = *type;
		layout.inline_offset = static_cast<uint32_t>(fields.inline_offset);
		layout.inline_root   = inline_roots.empty() ? tops.front() : inline_roots.front();
	}
	std::vector<bool> node_counts;
	{
		std::array<std::optional<bool>, 2> counts;
		for (const auto* site: {top, lower.front()}) {
			auto&      count = counts[site == top ? 0 : 1];
			Evaluator  ev(program, order, store, site->bvh->Arg(2));
			Value      size = site->descriptor->Arg(2);
			const auto* part = ProducerOf(size, ValueOpcode::CompositeExtractU64);
			if (part != nullptr && ImmediateU32(part->Arg(1)) == 0u) {
				const auto* masked = ProducerOf(part->Arg(0), ValueOpcode::BitwiseAnd64);
				const auto* wide   = masked != nullptr ? ProducerOf(masked->Arg(0), ValueOpcode::CompositeConstructU64) : nullptr;
				if (wide != nullptr) size = wide->Arg(0);
			}
			for (const auto word: ev.Word(size)) {
				const auto& form = store.F(word);
				if (form.terms.size() != 1 || form.terms[0].second != 1) continue;
				const auto& load = store.A(form.terms[0].first);
				if (load.kind != Kind::Load || load.args[1] != 4) continue;
				const auto at = store.F(static_cast<uint32_t>(load.args[0])).constant;
				if (site == top) {
					count = at == 80u && form.constant == 0xffffffffu;
					break;
				}
				if (at == 80u) count = true;
				if (at == 88u) count = false;
			}
			if (site != top && count && *count && layout.inline_type != NoField) count = false;
			if (site == top && lower.front() == top) break;
		}
		const auto first     = counts[0] ? counts[0] : counts[1];
		layout.node_count_80 = first.value_or(false);
		if (counts[0] && counts[1] && *counts[0] != *counts[1]) node_counts.push_back(*counts[1]);
	}
	std::vector<OracleImprovement> improvements;
	{
		std::vector<MaskField>                                masks;
		std::vector<std::tuple<MaskField, const Inst*, bool>> tests;
		Evaluator                                             ev(program, order, store, Value {});
		const auto test = [&](const MaskField& mask, const Inst& inst, bool nonzero) {
			if (mask.width == 0 || mask.width > 31u || excluded(mask.dword)) return;
			if (std::ranges::none_of(tests, [&](const auto& other) { return std::get<0>(other) == mask && std::get<1>(other) == &inst; })) {
				tests.emplace_back(mask, &inst, nonzero);
			}
		};
		for (auto* block: program.blocks) {
			for (auto& inst: *block) {
				const auto op = inst.GetOpcode();
				const bool left  = op == ValueOpcode::UGreaterThan32 || op == ValueOpcode::ULessThanEqual32;
				const bool right = op == ValueOpcode::ULessThan32 || op == ValueOpcode::UGreaterThanEqual32;
				if (left || right) {
					const auto below = ImmediateU32(inst.Arg(left ? 1 : 0));
					if (!below || *below == 0u || *below > 0xffffffu || (*below & (*below + 1u)) != 0) continue;
					for (const auto word: ev.Word(inst.Arg(left ? 0 : 1))) {
						const auto dword = fields.LoadDword(store, word);
						if (!dword) continue;
						const auto low = static_cast<uint32_t>(std::popcount(*below));
						test({*dword, low, 32u - low, false}, inst, op == ValueOpcode::UGreaterThan32 || op == ValueOpcode::ULessThan32);
					}
					continue;
				}
				if (op != ValueOpcode::INotEqual32 && op != ValueOpcode::IEqual32) continue;
				for (uint32_t side = 0; side < 2; ++side) {
					const auto constant = ImmediateU32(inst.Arg(side));
					if (!constant || (*constant != 0u && !(op == ValueOpcode::IEqual32 && *constant == 1u))) continue;
					if (op == ValueOpcode::IEqual32 && *constant == 0u) continue;
					for (const auto word: ev.Word(inst.Arg(1 - side))) {
						const auto mask = MaskOf(store, fields, word, op == ValueOpcode::IEqual32);
						if (!mask) continue;
						if (!mask->variable) {
							test(*mask, inst, true);
						} else if (!excluded(mask->dword) && std::ranges::find(masks, *mask) == masks.end()) {
							masks.push_back(*mask);
						}
					}
				}
			}
		}
		layout.mask_dword = NoField;
		layout.mask_shift = 0;
		layout.mask_width = 0;
		if (masks.size() == 1) {
			layout.mask_dword = masks.front().dword;
			layout.mask_shift = masks.front().shift;
		} else if (masks.empty() && !tests.empty()) {
			std::vector<Value> starts;
			for (const auto& site: sites) {
				starts.push_back(site.descriptor->Arg(0));
				starts.push_back(site.descriptor->Arg(1));
				starts.push_back(site.address->Arg(0));
				if (site.words == 2u) starts.push_back(site.address->Arg(1));
			}
			const auto               bases = Feeding(order, starts);
			std::vector<const Inst*> entries;
			for (auto* block: program.blocks) {
				for (auto& inst: *block) {
					if (inst.GetOpcode() != ValueOpcode::LoadAddressU32) continue;
					const bool blas = std::ranges::any_of(ev.Address(inst), [&](uint32_t address) {
						const auto record = RecordOf(store, address, header);
						return record && (record->indexed ? record->offset == indexed->offset && record->stride == indexed->stride
						                                  : record->key == fields.inline_key && !inlined.empty() && record->offset == inlined.front().second);
					});
					if (blas) entries.push_back(&inst);
				}
			}
			for (const auto& [mask, inst, nonzero]: tests) {
				size_t     committed = 0;
				const bool proven    = std::ranges::all_of(entries, [&](const Inst* load) {
					const auto commits = Commits(order, bases, *load, Value(const_cast<Inst*>(inst)), nonzero);
					committed += commits.has_value() ? 1u : 0u;
					return commits.value_or(true);
				}) && committed != 0;
				const OracleImprovement improvement {OracleImprovementMask, mask.dword, mask.shift, mask.width};
				if (proven && std::ranges::find(improvements, improvement) == improvements.end()) improvements.push_back(improvement);
			}
		}
	}
	OracleWalks walks;
	std::vector<std::pair<uint32_t, uint32_t>> reach;
	for (const auto& site: sites) {
		if (!site.top) continue;
		Block* header_block = OutermostLoop(program, site.bvh->Parent());
		if (header_block == nullptr) return Refuse("a top-level intersection is not in a natural loop");
		if (std::ranges::any_of(walks.loops, [&](const auto& loop) { return loop.first == header_block; })) continue;
		Block* latch = nullptr;
		if (!Reach(program, header_block, reach, latch)) return Refuse("the walk has no code range");
		walks.loops.emplace_back(header_block, latch);
	}
	for (const auto& site: sites) {
		if (std::ranges::none_of(walks.loops, [&](const auto& loop) {
			    return InLoop(program, site.bvh->Parent(), loop.first, loop.second);
		    })) {
			return Refuse("an intersection is outside every top-level walk");
		}
	}
	node_counts.insert(node_counts.begin(), layout.node_count_80);
	std::vector<uint32_t> candidates;
	for (const auto& [root, distance]: roots) {
		if (layout.mask_dword == NoField || root != layout.mask_dword) candidates.push_back(root);
	}
	std::vector<std::array<size_t, 5>> picks;
	for (size_t source = 0; source < named_sources.size(); ++source) {
		for (size_t count = 0; count < node_counts.size(); ++count) {
			for (size_t tlas_root = 0; tlas_root < tops.size(); ++tlas_root) {
				for (size_t root = 0; root < candidates.size() && picks.size() < 4096; ++root) {
					const auto shifts = static_cast<size_t>(source != 0) + (count != 0) + (tlas_root != 0) + (root != 0);
					picks.push_back({shifts, source, count, tlas_root, root});
				}
			}
		}
	}
	std::ranges::stable_sort(picks, [](const auto& a, const auto& b) { return a[0] < b[0]; });
	for (const auto& pick: picks) {
		if (walks.sources.size() >= MaxHypotheses) break;
		const auto& source       = std::get<0>(named_sources[pick[1]]);
		auto        hypothesis   = layout;
		hypothesis.root_dword    = candidates[pick[4]];
		hypothesis.node_count_80 = node_counts[pick[2]];
		NativeTraversals traversals;
		traversals.source_valid  = true;
		traversals.user_sgpr     = source.user_sgpr;
		traversals.header_offset = source.header_offset;
		traversals.header_vsharp = source.header_vsharp;
		traversals.root          = tops[pick[3]];
		traversals.layout        = hypothesis.Pack();
		traversals.reach         = reach;
		walks.sources.push_back(std::move(traversals));
	}
	walks.improvements = std::move(improvements);
	if (walks.sources.empty()) return Refuse("no record dword reaches a node argument");
	return walks;
}

}
