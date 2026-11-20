#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <array>
#if defined(ANYPS5_KYTY)
#include <bit>
#include <memory>
#endif
#include <optional>
#include <unordered_map>
#if defined(ANYPS5_KYTY)
#include <unordered_set>
#include <vector>
#endif
#include <utility>

#include <cstdio>
#define TESS_LOG(...) do { LOGF(__VA_ARGS__); std::fprintf(stderr, "[tess] " __VA_ARGS__); } while (0)

namespace Libs::Graphics::ShaderRecompiler {
namespace {

struct TessellationAddress {
	enum class Kind { Unknown, Affine, PackedControlPoint };
	Kind     kind        = Kind::Unknown;
	uint32_t coefficient = 0;
	uint32_t constant    = 0;

	bool operator==(const TessellationAddress&) const = default;
};

void ReflectTessellationLayout(const Decoder::Program& program, bool local,
                               ShaderTessellationInputInfo& info) {
	constexpr uint32_t scalars = IR::NumScalarRegs + 2u;
	using namespace Decoder;
	using Address   = TessellationAddress;
	using Kind      = Address::Kind;
	using Registers = std::array<Address, IR::NumVectorRegs + scalars>;

	const auto control_points = local ? info.input_control_points : info.output_control_points;
	Registers  entry {}, registers;
	entry[local ? 3u : 1u] =
	    local ? Address {Kind::Affine, 1u, 0u} : Address {Kind::PackedControlPoint};
	const auto constant = [](uint32_t value) { return Address {Kind::Affine, 0u, value}; };
	const bool single = !local && control_points == 1u;
	const auto patch  = single ? Address {Kind::Affine, 1u, 0u} : constant(0u);
	const auto point  = single ? constant(0u) : Address {Kind::Affine, 1u, 0u};
	const auto read   = [&](const Operand& operand) -> Address {
		if (operand.absolute || operand.negate || operand.sdwa_sext || operand.op_sel ||
		    operand.op_sel_hi || operand.negate_hi || operand.dpp)
			return {};
		Address value;
		if (operand.kind == OperandKind::IntegerInlineConstant ||
		    operand.kind == OperandKind::LiteralConstant) {
			value = constant(operand.value);
		} else if (operand.kind == OperandKind::Vgpr) {
			value = registers.at(operand.reg);
		} else if (uint32_t code = 0; CFG::ScalarOperandCode(operand, code) && code < scalars) {
			value = registers[IR::NumVectorRegs + code];
		}
		if (value.kind == Kind::PackedControlPoint && operand.sdwa_sel == 0u) return patch;
		return operand.sdwa_sel == 6u ? value : Address {};
	};
	const auto affine = [](uint64_t coefficient, uint64_t offset) {
		return coefficient <= UINT32_MAX && offset <= UINT32_MAX
		           ? Address {Kind::Affine, static_cast<uint32_t>(coefficient),
		                      static_cast<uint32_t>(offset)}
		           : Address {};
	};
	const auto add = [&](Address lhs, Address rhs) {
		return lhs.kind == Kind::Affine && rhs.kind == Kind::Affine
		           ? affine(uint64_t {lhs.coefficient} + rhs.coefficient,
		                    uint64_t {lhs.constant} + rhs.constant)
		           : Address {};
	};
	const auto multiply = [&](Address lhs, Address rhs) {
		if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0u) ||
		    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0u))
			return constant(0u);
		if (lhs.kind != Kind::Affine || rhs.kind != Kind::Affine) return Address {};
		if (rhs.coefficient != 0u) std::swap(lhs, rhs);
		return rhs.coefficient == 0u ? affine(uint64_t {lhs.coefficient} * rhs.constant,
		                                      uint64_t {lhs.constant} * rhs.constant)
		                             : Address {};
	};
	const auto low24 = [&](Address value) {
		if (value.kind != Kind::Affine) return Address {};
		if (value.coefficient == 0u) return constant(value.constant & 0xffffffu);
		const uint64_t maximum = single ? 0xffu : control_points - 1u;
		return value.coefficient * maximum + value.constant <= 0xffffffu ? value : Address {};
	};
	std::unordered_map<uint32_t, Registers> joins, loops;
	const auto merge = [](Registers& into, const Registers& from) {
		bool changed = false;
		for (uint32_t index = 0; index < into.size(); index++) {
			if (into[index] == from[index]) continue;
			changed |= into[index] != Address {};
			into[index] = {};
		}
		return changed;
	};

	uint32_t stride = 0, patch_begin = UINT32_MAX, patch_end = 0;
	for (bool repeat = true; repeat;) {
		repeat    = false;
		registers = entry;
		for (const auto& inst: program.instructions) {
			if (const auto join = joins.find(inst.pc); join != joins.end()) {
				merge(registers, join->second);
				joins.erase(join);
			}
			if (const auto loop = loops.find(inst.pc); loop != loops.end())
				merge(registers, loop->second);
			if (IsDirectBranch(inst.opcode)) {
				const bool back = inst.branch_target <= inst.pc;
				const auto [join, inserted] =
				    (back ? loops : joins).try_emplace(inst.branch_target, registers);
				const bool changed = !inserted && merge(join->second, registers);
				repeat |= back && (inserted || changed);
			}
			const bool local_store  = IsDsWriteOpcode(inst.opcode);
			const bool buffer_store = inst.opcode == Opcode::BUFFER_STORE_DWORD ||
			                          inst.opcode == Opcode::BUFFER_STORE_DWORDX2 ||
			                          inst.opcode == Opcode::BUFFER_STORE_DWORDX3 ||
			                          inst.opcode == Opcode::BUFFER_STORE_DWORDX4;
			const bool control_store = !local && buffer_store &&
			                           inst.src2.kind == OperandKind::Sgpr && inst.src2.reg == 2u;
			if ((local && local_store) || control_store) {
				const auto address = read(inst.src0);
				if (address.kind != Kind::Affine) {
					EXIT("%s tessellation address is not affine at pc 0x%08x\n",
					     local ? "LS" : "HS", inst.pc);
				}
				if (static_cast<int32_t>(address.coefficient) > 0) {
					EXIT_NOT_IMPLEMENTED((address.coefficient & 3u) != 0u ||
					                     (stride != 0u && stride != address.coefficient));
					stride = address.coefficient;
				} else if (control_store) {
					const auto begin = address.constant + inst.offset;
					patch_begin      = std::min(patch_begin, begin);
					patch_end        = std::max(patch_end, begin + 4u * inst.data_dwords);
				}
			}
			for (uint32_t code = 0; code < scalars; code++) {
				auto& scalar = registers[IR::NumVectorRegs + code];
				if (scalar != Address {} && CFG::MayWriteScalarRegister(inst, code, 64u))
					scalar = {};
			}
			if (uint32_t code = 0; inst.opcode == Opcode::S_MOVK_I32 &&
			                       CFG::ScalarOperandCode(inst.dst, code) && code < scalars)
				registers[IR::NumVectorRegs + code] = constant(inst.src0.value);
			if (local_store || buffer_store || inst.dst.kind != OperandKind::Vgpr) continue;
			const auto lhs   = read(inst.src0);
			const auto rhs   = read(inst.src1);
			const auto third = read(inst.src2);
			Address    value;
			switch (inst.opcode) {
				case Opcode::V_BFE_U32:
					if (lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
					    rhs.coefficient == 0u && third.kind == Kind::Affine &&
					    third.coefficient == 0u) {
						if (rhs.constant == 8u && third.constant == 5u) value = point;
						if (rhs.constant == 0u && third.constant == 8u) value = patch;
					}
					break;
				case Opcode::V_AND_B32: {
					const auto mask = lhs.kind == Kind::PackedControlPoint ? rhs
					                : rhs.kind == Kind::PackedControlPoint ? lhs
					                                                       : Address {};
					const bool low_byte = mask.kind == Kind::Affine && mask.coefficient == 0u &&
					                      (mask.constant & ~0xffu) == 0u;
					if (mask == constant(0xffu)) value = patch;
					else if (low_byte && !single) value = constant(0u);
					if (mask == constant(0x1f00u)) value = multiply(point, constant(256u));
					break;
				}
				case Opcode::V_ADD_NC_U32: value = add(lhs, rhs); break;
				case Opcode::V_MUL_LO_U32: value = multiply(lhs, rhs); break;
				case Opcode::V_MUL_U32_U24: value = multiply(low24(lhs), low24(rhs)); break;
				case Opcode::V_MAD_U32_U24:
					value = add(multiply(low24(lhs), low24(rhs)), third);
					break;
				case Opcode::V_LSHL_ADD_U32:
					if (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant < 32u)
						value = add(multiply(lhs, constant(1u << rhs.constant)), third);
					break;
				case Opcode::V_LSHLREV_B32:
					if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant < 32u)
						value = multiply(rhs, constant(1u << lhs.constant));
					break;
				case Opcode::V_SUB_NC_U32:
					if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine &&
					    lhs.constant >= rhs.constant)
						value = {Kind::Affine, lhs.coefficient - rhs.coefficient,
						         lhs.constant - rhs.constant};
					break;
				default: break;
			}
			if (inst.dst.sdwa_sel != 6u || inst.dst.op_sel || inst.dst.omod || inst.dst.clamp ||
			    inst.dst.dpp)
				value = {};
			for (uint32_t index = 0;
			     index < std::max(inst.data_dwords, 1u) && inst.dst.reg + index < IR::NumVectorRegs;
			     index++) {
				registers[inst.dst.reg + index] = {};
			}
			registers.at(inst.dst.reg) = value;
		}
	}
	EXIT_NOT_IMPLEMENTED(!joins.empty() || stride == 0u);
	(local ? info.ls_stride : info.hs_stride) = stride;
	if (patch_end != 0u) {
		EXIT_NOT_IMPLEMENTED(patch_begin < stride * control_points);
		info.patch_base = patch_begin;
		info.patch_size = patch_end - patch_begin;
	}
}

const IR::Inst* TessellationBufferBase(const IR::Inst& inst) {
	if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::None || inst.NumArgs() <= 3u)
		return nullptr;
	const auto* base = inst.Arg(3).Resolve().TryInstruction();
	return base != nullptr && base->GetOpcode() == IR::ValueOpcode::TessellationBase ? base
	                                                                                 : nullptr;
}

bool Implies(IR::Value predicate, IR::Value condition) {
	predicate = predicate.Resolve();
	if (predicate == condition.Resolve()) return true;
	const auto* inst = predicate.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == IR::ValueOpcode::LogicalAnd &&
	       (Implies(inst->Arg(0), condition) || Implies(inst->Arg(1), condition));
}

IR::Value ActiveAddress(IR::Block& block, IR::Block::iterator before, IR::Value value,
                        IR::Value predicate, std::unordered_map<IR::Inst*, IR::Value>& resolved) {
	using namespace IR;
	value        = value.Resolve();
	auto* source = value.TryInstruction();
	if (source == nullptr) return value;
	if (const auto found = resolved.find(source); found != resolved.end()) return found->second;
	if (source->GetOpcode() == ValueOpcode::SelectU32) {
		return Implies(predicate, source->Arg(0))
		           ? ActiveAddress(block, before, source->Arg(1), predicate, resolved)
		           : value;
	}
	const auto opcode = source->GetOpcode();
	if (opcode != ValueOpcode::IAdd32 && opcode != ValueOpcode::ISub32 &&
	    opcode != ValueOpcode::IMul32 && opcode != ValueOpcode::BitFieldUExtract)
		return value;
	std::array<Value, 3> args;
	bool                 changed = false;
	for (size_t index = 0; index < source->NumArgs(); index++) {
		args[index] = ActiveAddress(block, before, source->Arg(index), predicate, resolved);
		changed |= args[index] != source->Arg(index).Resolve();
	}
	if (changed) {
		const auto flags = source->Flags<uint64_t>();
		const auto copy =
		    source->NumArgs() == 2u
		        ? block.PrependNewInst(before, opcode, {args[0], args[1]}, flags)
		        : block.PrependNewInst(before, opcode, {args[0], args[1], args[2]}, flags);
		value = Value(&*copy);
	}
	resolved.emplace(source, value);
	return value;
}

#if defined(ANYPS5_KYTY)
struct MergedHullAddress {
	bool     known       = false;
	uint64_t coefficient = 0;
	uint64_t constant    = 0;

	[[nodiscard]] uint64_t Highest(uint32_t control_points) const {
		return coefficient * (control_points - 1u) + constant;
	}
};

class MergedHullAddresses {
public:
	MergedHullAddresses(uint32_t control_points, IR::Value predicate)
	    : m_control_points(control_points), m_predicate(predicate) {}

	MergedHullAddress Evaluate(IR::Value value) {
		using Address = MergedHullAddress;
		value         = value.Resolve();
		if (value.IsImmediate()) {
			return value.GetType() == IR::Type::U32 ? Address {true, 0u, value.U32()} : Address {};
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		const auto [known, inserted] = m_known.try_emplace(inst);
		if (!inserted) return known->second;
		const auto result = Compute(*inst);
		m_known[inst]     = result;
		return result;
	}

private:
	using Address = MergedHullAddress;

	Address Bounded(uint64_t coefficient, uint64_t constant) const {
		const Address result {true, coefficient, constant};
		return coefficient <= UINT32_MAX && constant <= UINT32_MAX &&
		               result.Highest(m_control_points) <= UINT32_MAX
		           ? result
		           : Address {};
	}

	static bool Multiple(const Address& value, uint32_t bits) {
		const uint64_t low = (uint64_t {1u} << bits) - 1u;
		return (value.coefficient & low) == 0u && (value.constant & low) == 0u;
	}

	static bool Same(const Address& lhs, const Address& rhs) {
		return lhs.known && rhs.known && lhs.coefficient == rhs.coefficient &&
		       lhs.constant == rhs.constant;
	}

	Address Compute(const IR::Inst& inst) {
		using namespace IR;
		const auto arg   = [&](size_t index) { return Evaluate(inst.Arg(index)); };
		const auto shift = [&](size_t index, uint32_t& amount) {
			const auto count = arg(index);
			amount           = static_cast<uint32_t>(count.constant);
			return count.known && count.coefficient == 0u && count.constant < 32u;
		};
		if (inst.GetOpcode() == ValueOpcode::Phi) {
			if (inst.NumArgs() == 0u) return {};
			if (!m_predicate.IsEmpty() && m_carried == nullptr) {
				m_carried = std::make_unique<MergedHullAddresses>(m_control_points, Value {});
			}
			auto&      carried = m_predicate.IsEmpty() ? *this : *m_carried;
			const auto first   = carried.Evaluate(inst.Arg(0));
			for (size_t index = 1; index < inst.NumArgs(); index++) {
				if (!Same(first, carried.Evaluate(inst.Arg(index)))) return {};
			}
			return first;
		}
		switch (inst.GetOpcode()) {
			case ValueOpcode::LaneId: return {true, 1u, 0u};
			case ValueOpcode::GetBuiltin:
				return inst.Arg(0).U32() == static_cast<uint32_t>(StageInputKind::InvocationId)
				           ? Address {true, 1u, 0u}
				           : Address {};
			case ValueOpcode::SelectU32: {
				const auto condition = inst.Arg(0).Resolve();
				if ((condition.IsImmediate() && condition.U1()) ||
				    (!m_predicate.IsEmpty() && Implies(m_predicate, condition)))
					return arg(1);
				if (condition.IsImmediate()) return arg(2);
				const auto lhs = arg(1);
				return Same(lhs, arg(2)) ? lhs : Address {};
			}
			case ValueOpcode::IAdd32: {
				const auto lhs = arg(0), rhs = arg(1);
				return lhs.known && rhs.known
				           ? Bounded(lhs.coefficient + rhs.coefficient, lhs.constant + rhs.constant)
				           : Address {};
			}
			case ValueOpcode::ISub32: {
				const auto lhs = arg(0), rhs = arg(1);
				return lhs.known && rhs.known && lhs.coefficient >= rhs.coefficient &&
				               lhs.constant >= rhs.constant
				           ? Address {true, lhs.coefficient - rhs.coefficient,
				                      lhs.constant - rhs.constant}
				           : Address {};
			}
			case ValueOpcode::IMul32: {
				auto lhs = arg(0), rhs = arg(1);
				if (!lhs.known || !rhs.known) return {};
				if (lhs.coefficient == 0u) std::swap(lhs, rhs);
				return rhs.coefficient == 0u
				           ? Bounded(lhs.coefficient * rhs.constant, lhs.constant * rhs.constant)
				           : Address {};
			}
			case ValueOpcode::ShiftLeftLogical32: {
				const auto lhs = arg(0);
				uint32_t   amount;
				return lhs.known && shift(1, amount)
				           ? Bounded(lhs.coefficient << amount, lhs.constant << amount)
				           : Address {};
			}
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::BitFieldUExtract: {
				const auto lhs = arg(0);
				uint32_t   amount, bits = 32u;
				if (!lhs.known || !shift(1, amount)) return {};
				if (inst.GetOpcode() == ValueOpcode::BitFieldUExtract && !shift(2, bits)) return {};
				if (Multiple(lhs, std::min(amount + bits, 32u))) return {true, 0u, 0u};
				if (!Multiple(lhs, amount) || (lhs.Highest(m_control_points) >> amount) >> bits != 0u)
					return {};
				return {true, lhs.coefficient >> amount, lhs.constant >> amount};
			}
			case ValueOpcode::BitwiseAnd32: {
				auto lhs = arg(0), rhs = arg(1);
				if (!lhs.known || !rhs.known) return {};
				if (lhs.coefficient == 0u) std::swap(lhs, rhs);
				if (rhs.coefficient != 0u) return {};
				if (lhs.coefficient == 0u) return {true, 0u, lhs.constant & rhs.constant};
				const auto mask = static_cast<uint32_t>(rhs.constant);
				if (mask == 0u || Multiple(lhs, static_cast<uint32_t>(std::bit_width(mask)))) {
					return {true, 0u, 0u};
				}
				const auto low  = static_cast<uint32_t>(std::countr_zero(mask));
				const auto bits = mask >> low;
				return (bits & (bits + 1u)) == 0u && Multiple(lhs, low) &&
				               lhs.Highest(m_control_points) <= mask
				           ? lhs
				           : Address {};
			}
			case ValueOpcode::BitwiseOr32: {
				auto lhs = arg(0), rhs = arg(1);
				if (!lhs.known || !rhs.known) return {};
				if (lhs.coefficient == 0u) std::swap(lhs, rhs);
				if (rhs.coefficient != 0u) return {};
				if (lhs.coefficient == 0u) return {true, 0u, lhs.constant | rhs.constant};
				const uint64_t below = rhs.constant == 0u ? 0u : std::bit_ceil(rhs.constant + 1u) - 1u;
				return (lhs.coefficient & below) == 0u && (lhs.constant & below) == 0u
				           ? Bounded(lhs.coefficient, lhs.constant + rhs.constant)
				           : Address {};
			}
			default: return {};
		}
	}

	uint32_t                                             m_control_points;
	IR::Value                                            m_predicate;
	std::unordered_map<const IR::Inst*, MergedHullAddress> m_known;
	std::unique_ptr<MergedHullAddresses>                   m_carried;
};

IR::Value MergedHullAddressValue(IR::Block& block, IR::Block::iterator before,
                                 const MergedHullAddress& address) {
	using namespace IR;
	const auto constant = Value(static_cast<uint32_t>(address.constant));
	if (address.coefficient == 0u) return constant;
	auto scaled = Value(&*block.PrependNewInst(before, ValueOpcode::LaneId));
	if (address.coefficient != 1u) {
		scaled = Value(&*block.PrependNewInst(
		    before, ValueOpcode::IMul32,
		    {scaled, Value(static_cast<uint32_t>(address.coefficient))}));
	}
	return address.constant == 0u
	           ? scaled
	           : Value(&*block.PrependNewInst(before, ValueOpcode::IAdd32, {scaled, constant}));
}

void RequireMergedHullBarriers(const IR::Program& program) {
	using namespace IR;
	const auto walk = [](std::vector<const Block*> pending, const Block* barrier, auto&& visit) {
		std::unordered_set<const Block*> seen;
		while (!pending.empty()) {
			const auto* block = pending.back();
			pending.pop_back();
			visit(block);
			if (block == barrier || !seen.insert(block).second) continue;
			const auto successors = block->ImmSuccessors();
			pending.insert(pending.end(), successors.begin(), successors.end());
		}
	};
	for (const auto* block: program.blocks) {
		if (std::ranges::none_of(*block, [](const Inst& inst) {
			    return inst.GetOpcode() == ValueOpcode::Barrier;
		    }))
			continue;
		walk({program.blocks.front()}, block, [&](const Block* reached) {
			if (reached != block && reached->ImmSuccessors().empty()) {
				EXIT("the merged hull program has an s_barrier that some invocations do not "
				     "reach, at pc 0x%08x\n",
				     block->start_pc);
			}
		});
		const auto successors = block->ImmSuccessors();
		walk({successors.begin(), successors.end()}, block, [&](const Block* reached) {
			if (reached == block) {
				EXIT("the merged hull program has an s_barrier inside a loop, at pc 0x%08x\n",
				     block->start_pc);
			}
		});
	}
}

bool MergedHullOwnLaneMask(const IR::Inst& mask, std::unordered_set<const IR::Inst*>& seen) {
	using namespace IR;
	if (!seen.insert(&mask).second) return true;
	const auto zero = [](Value value) {
		value = value.Resolve();
		return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == 0u;
	};
	for (const auto& use: mask.Uses()) {
		const auto& user = *use.user;
		switch (user.GetOpcode()) {
			case ValueOpcode::Identity:
			case ValueOpcode::Phi:
			case ValueOpcode::CompositeExtractU32x4:
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32:
			case ValueOpcode::BitwiseNot32:
				if (!MergedHullOwnLaneMask(user, seen)) return false;
				break;
			case ValueOpcode::SelectU32:
				if (use.operand == 0u || !MergedHullOwnLaneMask(user, seen)) return false;
				break;
			case ValueOpcode::IEqual32:
			case ValueOpcode::INotEqual32:
				if (!zero(user.Arg(1u - use.operand))) return false;
				break;
			case ValueOpcode::ShiftRightLogical32: {
				const auto* index = user.Arg(1).Resolve().TryInstruction();
				if (use.operand != 0u || index == nullptr ||
				    index->GetOpcode() != ValueOpcode::BitwiseAnd32) {
					return false;
				}
				const auto* lane = index->Arg(0).Resolve().TryInstruction();
				const auto  low  = index->Arg(1).Resolve();
				if (lane == nullptr || lane->GetOpcode() != ValueOpcode::LaneId ||
				    !low.IsImmediate() || low.U32() != 31u) {
					return false;
				}
				for (const auto& bit: user.Uses()) {
					const auto one = bit.user->GetOpcode() == ValueOpcode::BitwiseAnd32
					                     ? bit.user->Arg(1u - bit.operand).Resolve()
					                     : Value {};
					if (!one.IsImmediate() || one.GetType() != Type::U32 || one.U32() != 1u) {
						return false;
					}
				}
				break;
			}
			default: return false;
		}
	}
	return true;
}

bool IsHullImmediate(IR::Value value, uint32_t expected) {
	value = value.Resolve();
	return value.IsImmediate() && value.GetType() == IR::Type::U32 && value.U32() == expected;
}

const IR::Inst* HullInstruction(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.Resolve().TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode ? inst : nullptr;
}

IR::Value HullOtherOperand(const IR::Inst& inst, uint32_t immediate) {
	if (IsHullImmediate(inst.Arg(1), immediate)) return inst.Arg(0).Resolve();
	if (IsHullImmediate(inst.Arg(0), immediate)) return inst.Arg(1).Resolve();
	return {};
}

const IR::Inst* HullMaskProducer(IR::Value word, IR::ValueOpcode extract, uint32_t half) {
	const auto* inst = HullInstruction(word, extract);
	return inst != nullptr && IsHullImmediate(inst->Arg(1), half)
	           ? inst->Arg(0).Resolve().TryInstruction()
	           : nullptr;
}

const IR::Inst* QuadMaskOwnBit(const IR::Inst& read, uint32_t wave_size,
                              std::unordered_set<const IR::Inst*>& parts) {
	using namespace IR;
	if (read.GetOpcode() != ValueOpcode::INotEqual32) return nullptr;
	const auto* bit = HullInstruction(HullOtherOperand(read, 0u), ValueOpcode::BitwiseAnd32);
	if (bit == nullptr) return nullptr;
	const auto* shift =
	    HullInstruction(HullOtherOperand(*bit, 1u), ValueOpcode::ShiftRightLogical32);
	if (shift == nullptr) return nullptr;
	const auto* index = HullInstruction(shift->Arg(1), ValueOpcode::BitwiseAnd32);
	if (index == nullptr ||
	    HullInstruction(HullOtherOperand(*index, 31u), ValueOpcode::LaneId) == nullptr) {
		return nullptr;
	}
	auto low = shift->Arg(0).Resolve();
	const Inst* select = nullptr;
	const Inst* high   = nullptr;
	if (wave_size == 64u) {
		select = HullInstruction(low, ValueOpcode::SelectU32);
		if (select == nullptr) return nullptr;
		const auto* low_half = HullInstruction(select->Arg(0), ValueOpcode::ULessThan32);
		if (low_half == nullptr || !IsHullImmediate(low_half->Arg(1), 32u) ||
		    HullInstruction(low_half->Arg(0), ValueOpcode::LaneId) == nullptr) {
			return nullptr;
		}
		low  = select->Arg(1).Resolve();
		high = HullInstruction(select->Arg(2), ValueOpcode::CompositeExtractU64);
	}
	const auto* wqm = HullMaskProducer(low, ValueOpcode::CompositeExtractU64, 0u);
	if (wqm == nullptr || wqm->GetOpcode() != ValueOpcode::WqmU64 ||
	    (select != nullptr &&
	     HullMaskProducer(select->Arg(2), ValueOpcode::CompositeExtractU64, 1u) != wqm)) {
		return nullptr;
	}
	parts.insert({wqm, low.TryInstruction(), shift, bit});
	if (select != nullptr) parts.insert({select, high});
	return wqm;
}

IR::Value QuadMaskPredicate(const IR::Inst& wqm, uint32_t wave_size,
                            const std::unordered_set<const IR::Inst*>& branch_inputs) {
	using namespace IR;
	const auto* words = HullInstruction(wqm.Arg(0), ValueOpcode::CompositeConstructU64);
	if (words == nullptr) return {};
	const auto exclusive = [&](const Inst& value, const Inst& user) {
		return value.UseCount() == 1u && value.Uses().front().user == &user &&
		       !branch_inputs.contains(&value);
	};
	if (!exclusive(*words, wqm)) return {};
	const auto low = words->Arg(0).Resolve();
	const auto high = words->Arg(1).Resolve();
	const auto* ballot = HullMaskProducer(low, ValueOpcode::CompositeExtractU32x4, 0u);
	if (ballot == nullptr || ballot->GetOpcode() != ValueOpcode::Ballot ||
	    branch_inputs.contains(ballot) || !exclusive(*low.TryInstruction(), *words)) return {};
	if (wave_size == 64u) {
		if (ballot->UseCount() != 2u ||
		    HullMaskProducer(high, ValueOpcode::CompositeExtractU32x4, 1u) != ballot ||
		    !exclusive(*high.TryInstruction(), *words)) return {};
	} else if (ballot->UseCount() != 1u || !IsHullImmediate(high, 0u)) return {};
	return ballot->Arg(0).Resolve();
}

struct QuadHelperPath {
	bool            local;
	const IR::Inst* guard = nullptr;
	bool            selected = false;

	bool operator==(const QuadHelperPath&) const = default;
};

std::optional<bool> QuadHelperCondition(IR::Value value, IR::Value predicate,
                                        QuadHelperPath path, uint32_t depth = 0u) {
	using namespace IR;
	value = value.Resolve();
	if (value.IsImmediate()) return value.GetType() == Type::U1 ? std::optional{value.U1()} : std::nullopt;
	if (value == predicate) return false;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth >= 8u) return {};
	if (inst == path.guard) return path.selected;
	const auto condition = [&](size_t index) {
		return QuadHelperCondition(inst->Arg(index), predicate, path, depth + 1u);
	};
	switch (inst->GetOpcode()) {
		case ValueOpcode::LogicalNot:
			if (const auto input = condition(0)) return !*input;
			break;
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalOr: {
			const bool short_circuit = inst->GetOpcode() == ValueOpcode::LogicalOr;
			const auto left = condition(0);
			if (left == short_circuit) return short_circuit;
			const auto right = condition(1);
			if (right == short_circuit) return short_circuit;
			if (left && right) return !short_circuit;
			break;
		}
		case ValueOpcode::SelectU1: {
			if (const auto select = condition(0)) return condition(*select ? 1u : 2u);
			const auto yes = condition(1);
			if (yes && yes == condition(2)) return yes;
			break;
		}
		default: break;
	}
	return {};
}

bool QuadHelperValueOnly(IR::ValueOpcode opcode) {
	using enum IR::ValueOpcode;
	switch (opcode) {
		case Phi:
		case Identity:
		case IEqual32:
		case INotEqual32:
		case ULessThan32:
		case BitCastF32U32:
		case BitCastU32F32:
		case CompositeConstructU32x2:
		case CompositeConstructU32x3:
		case CompositeConstructU32x4:
		case CompositeExtractU32x2:
		case CompositeExtractU32x3:
		case CompositeExtractU32x4:
		case BitwiseAnd32:
		case BitwiseOr32:
		case BitwiseXor32:
		case BitwiseNot32:
		case LogicalAnd:
		case LogicalOr:
		case LogicalXor:
		case LogicalNot:
		case IAdd32:
		case ISub32:
		case IMul32: return true;
		default: return false;
	}
}

bool QuadHelpersUnobserved(const IR::Inst& wqm, IR::Value predicate,
                           std::span<IR::Inst* const> reads,
                           const std::unordered_set<const IR::Inst*>& branch_inputs) {
	using namespace IR;
	std::unordered_set<const Inst*> affected;
	std::vector<const Inst*>         users(reads.begin(), reads.end());
	for (size_t index = 0; index < users.size(); ++index) {
		if (!affected.insert(users[index]).second) continue;
		for (const auto& use: users[index]->Uses()) users.push_back(use.user);
	}
	std::unordered_map<const Inst*, std::vector<QuadHelperPath>> seen;
	std::vector<std::pair<const Inst*, QuadHelperPath>>            pending;
	for (const auto* read: reads) pending.emplace_back(read, QuadHelperPath {read->Parent() == wqm.Parent()});
	while (!pending.empty()) {
		const auto [inst, path] = pending.back();
		pending.pop_back();
		auto& paths = seen[inst];
		if (std::ranges::find(paths, path) != paths.end()) continue;
		if (paths.size() >= 8u) return false;
		paths.push_back(path);
		if (branch_inputs.contains(inst)) return false;
		for (const auto& use: inst->Uses()) {
			const auto& user = *use.user;
			const auto  op   = user.GetOpcode();
			const bool local = path.local && user.Parent() == wqm.Parent() && op != ValueOpcode::Phi;
			auto next = local ? path : QuadHelperPath {false};
			if (op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
			    op == ValueOpcode::SelectF32) {
				if (use.operand != 0u && local) {
					const bool selected = use.operand == 1u;
					const auto condition = QuadHelperCondition(user.Arg(0), predicate, path);
					if (condition && *condition != selected) continue;
					const auto* guard = user.Arg(0).Resolve().TryInstruction();
					if (!condition && path.guard == nullptr && guard != nullptr &&
					    !affected.contains(guard)) next = {true, guard, selected};
				}
			} else if (!QuadHelperValueOnly(op)) {
				return false;
			}
			if (local && user.GetType() == Type::U1 &&
			    QuadHelperCondition(Value(use.user), predicate, next)) {
				continue;
			}
			pending.emplace_back(&user, next);
		}
	}
	return true;
}

void LowerMergedHullQuadMasks(IR::Program& program) {
	using namespace IR;
	struct MaskReads {
		std::unordered_set<const Inst*> parts;
		std::vector<Inst*>              reads;
	};
	std::unordered_map<const Inst*, MaskReads> masks;
	std::vector<const Inst*> order;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			std::unordered_set<const Inst*> parts;
			const auto* mask = QuadMaskOwnBit(inst, program.wave_size, parts);
			if (mask == nullptr) continue;
			auto& found = masks[mask];
			if (found.reads.empty()) order.push_back(mask);
			found.parts.insert(parts.begin(), parts.end());
			found.reads.push_back(&inst);
		}
	}
	if (order.empty()) return;
	std::unordered_set<const Inst*> branch_inputs;
	for (const auto* block: program.blocks) {
		branch_inputs.insert(block->condition.Resolve().TryInstruction());
		branch_inputs.insert(block->indirect_target.Resolve().TryInstruction());
	}
	std::vector<std::pair<Inst*, Value>> replacements;
	for (const auto* wqm: order) {
		const auto& found = masks.at(wqm);
		const auto predicate = QuadMaskPredicate(*wqm, program.wave_size, branch_inputs);
		if (predicate.IsEmpty()) continue;
		const auto own_uses = [&](const Inst* part) {
			if (branch_inputs.contains(part)) return false;
			return std::ranges::all_of(part->Uses(), [&](const Use& use) {
				return found.parts.contains(use.user) ||
				       std::ranges::find(found.reads, use.user) != found.reads.end();
			});
		};
		if (!std::ranges::all_of(found.parts, own_uses) ||
		    !QuadHelpersUnobserved(*wqm, predicate, found.reads, branch_inputs)) continue;
		for (auto* read: found.reads) replacements.emplace_back(read, predicate);
	}
	if (replacements.empty()) return;
	for (const auto& [read, predicate]: replacements) read->ReplaceUsesWith(predicate);
	RemoveIdentities(program);
	EliminateDeadCode(program.blocks);
}

void LowerMergedHullLanes(IR::Program& program) {
	using namespace IR;
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			switch (inst.GetOpcode()) {
				case ValueOpcode::ReadLane:
				case ValueOpcode::WriteLane:
				case ValueOpcode::DppMoveU32:
				case ValueOpcode::DppUpdateU32:
				case ValueOpcode::Permlane16U32:
				case ValueOpcode::SwizzleU32:
				case ValueOpcode::PermuteU32:
				case ValueOpcode::BpermuteU32:
				case ValueOpcode::SubgroupUMinU32:
				case ValueOpcode::SubgroupUMaxU32:
				case ValueOpcode::SubgroupOrU32:
					EXIT("the merged hull program reads or writes another lane's register, which "
					     "its hull stage does not hold at one lane per invocation (block pc "
					     "0x%08x)\n",
					     block->start_pc);
				case ValueOpcode::Ballot: {
					std::unordered_set<const Inst*> seen;
					if (!MergedHullOwnLaneMask(inst, seen)) {
						EXIT("the merged hull program uses a lane mask other than its own lane's "
						     "bit (block pc 0x%08x)\n",
						     block->start_pc);
					}
					const auto lane = Value(&*block->PrependNewInst(it, ValueOpcode::LaneId));
					const auto bit  = Value(&*block->PrependNewInst(
					    it, ValueOpcode::ShiftLeftLogical32, {Value(1u), lane}));
					const auto low  = Value(&*block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(0), bit, Value(0u)}));
					inst.ReplaceUsesWith(Value(&*block->PrependNewInst(
					    it, ValueOpcode::CompositeConstructU32x4,
					    {low, Value(0u), Value(0u), Value(0u)})));
					break;
				}
				case ValueOpcode::ConditionRef:
				case ValueOpcode::ReadFirstLane: inst.ReplaceUsesWith(inst.Arg(0)); break;
				default: break;
			}
		}
	}
}

void RequireMergedHullAligned(const MergedHullAddress& base, uint32_t offset, uint32_t pc) {
	if (((base.coefficient | (base.constant + offset)) & 3u) != 0u) {
		EXIT("the merged hull program uses an unaligned tessellation ring or LDS address at pc "
		     "0x%08x\n",
		     pc);
	}
}

void RequireMergedHullRecord(const ShaderTessellationInputInfo& tess, IR::Value address,
                             uint32_t offset, uint32_t components, bool write, uint32_t pc) {
	const auto base   = MergedHullAddresses(tess.output_control_points, {}).Evaluate(address);
	const auto stride = uint64_t {tess.ls_stride};
	const auto begin  = base.constant + offset;
	const auto end    = begin + 4u * uint64_t {components};
	if (!base.known) {
		EXIT("the merged hull program accesses LDS at pc 0x%08x through an address that is not a "
		     "constant offset in a control point record of its patch\n",
		     pc);
	}
	RequireMergedHullAligned(base, offset, pc);
	if (write) {
		if (base.coefficient != stride || end > stride) {
			EXIT("the merged hull program writes LDS outside its own control point's record at "
			     "pc 0x%08x\n",
			     pc);
		}
	} else if ((base.coefficient != 0u && base.coefficient != stride) ||
	           end + base.coefficient * (tess.output_control_points - 1u) >
	               stride * tess.input_control_points) {
		EXIT("the merged hull program reads LDS outside its patch's control point records at pc "
		     "0x%08x\n",
		     pc);
	}
}

void RequireMergedHullRing(const ShaderTessellationInputInfo& tess, IR::Value address,
                           uint32_t offset, uint32_t components, bool write, uint32_t pc) {
	const auto base = MergedHullAddresses(tess.output_control_points, {}).Evaluate(address);
	if (!base.known && write) {
		EXIT("the merged hull program stores to the off-chip ring at pc 0x%08x through an address "
		     "that is neither a constant nor an offset in its own control point's record\n",
		     pc);
	}
	if (!base.known) {
		EXIT("the merged hull program reads the off-chip ring at pc 0x%08x through an address "
		     "that is neither a constant nor an offset in a control point's record\n",
		     pc);
	}
	RequireMergedHullAligned(base, offset, pc);
	const auto stride = uint64_t {tess.hs_stride};
	const auto points = stride * tess.output_control_points;
	const auto end    = base.constant + offset + 4u * uint64_t {components};
	if (base.coefficient == 0u && base.constant >= points) {
		if (base.constant + offset < tess.patch_base ||
		    end > uint64_t {tess.patch_base} + tess.patch_size) {
			EXIT("the merged hull program %s off-chip memory outside its control points and "
			     "patch constants at pc 0x%08x\n",
			     write ? "writes" : "reads", pc);
		}
	} else if (!write) {
		if ((base.coefficient != 0u && base.coefficient != stride) ||
		    end + base.coefficient * (tess.output_control_points - 1u) > points) {
			EXIT("the merged hull program reads off-chip memory outside its patch's control "
			     "points at pc 0x%08x\n",
			     pc);
		}
	} else if (base.coefficient != 0u) {
		if (base.coefficient != stride || end > stride) {
			EXIT("the merged hull program stores to the off-chip ring outside its own control "
			     "point's record at pc 0x%08x\n",
			     pc);
		}
	} else if (tess.output_control_points != 1u || end > stride) {
		EXIT("the merged hull program stores a control point's record at a constant off-chip "
		     "address at pc 0x%08x, which only that control point's invocation can write\n",
		     pc);
	}
}

struct MergedHullTraffic {
	const IR::Block* block;
	uint32_t         order;
	uint32_t         pc;
	const IR::Inst*  inst = nullptr;
};

enum class MergedHullBarrierResult { Ordered, UnreachableRead, NoUniformPoint, LateWrite };

MergedHullBarrierResult InsertMergedHullBarrier(IR::Program& program,
                                                std::span<const MergedHullTraffic> writes,
                                                std::span<const MergedHullTraffic> reads,
                                                bool require_acyclic_prefix,
                                                uint32_t& late_write_pc) {
	using namespace IR;
	if (reads.empty()) return MergedHullBarrierResult::Ordered;
	const auto reached = [](std::span<Block* const> from, const Block* avoided) {
		std::unordered_set<const Block*> seen;
		std::vector<const Block*>        pending(from.begin(), from.end());
		while (!pending.empty()) {
			const auto* block = pending.back();
			pending.pop_back();
			if (block == avoided || !seen.insert(block).second) continue;
			const auto next = block->ImmSuccessors();
			pending.insert(pending.end(), next.begin(), next.end());
		}
		return seen;
	};
	const auto reads_in = [&](const std::unordered_set<const Block*>& blocks, bool inside) {
		return std::ranges::any_of(reads, [&](const MergedHullTraffic& read) {
			return blocks.contains(read.block) == inside;
		});
	};
	Block* const                   entry = program.blocks.front();
	const std::span<Block* const> start(&entry, 1u);
	const auto                     live = reached(start, nullptr);
	if (reads_in(live, false)) {
		return MergedHullBarrierResult::UnreachableRead;
	}
	Block* barrier = nullptr;
	for (auto* candidate: program.blocks) {
		if (!live.contains(candidate)) continue;
		const auto before = candidate == entry ? std::unordered_set<const Block*> {}
		                                       : reached(start, candidate);
		if (std::ranges::any_of(before,
		                        [](const Block* block) { return block->ImmSuccessors().empty(); }) ||
		    reads_in(before, true) ||
		    reached(candidate->ImmSuccessors(), nullptr).contains(candidate)) {
			continue;
		}
		if (barrier == nullptr ||
		    reached(barrier->ImmSuccessors(), nullptr).contains(candidate)) {
			barrier = candidate;
		}
	}
	if (barrier == nullptr) {
		return MergedHullBarrierResult::NoUniformPoint;
	}
	if (require_acyclic_prefix && barrier != entry) {
		const auto                                 prefix = reached(start, barrier);
		std::unordered_map<const Block*, uint32_t> incoming;
		for (const auto* block: prefix) {
			for (const auto* next: block->ImmSuccessors()) {
				if (next != barrier) incoming[next]++;
			}
		}
		if (incoming[entry] != 0u) return MergedHullBarrierResult::NoUniformPoint;
		std::vector<const Block*> ready {entry};
		size_t                   visited = 0;
		while (!ready.empty()) {
			const auto* block = ready.back();
			ready.pop_back();
			visited++;
			for (const auto* next: block->ImmSuccessors()) {
				if (next != barrier && --incoming.at(next) == 0u) ready.push_back(next);
			}
		}
		if (visited != prefix.size()) return MergedHullBarrierResult::NoUniformPoint;
	}
	const auto               after = reached(barrier->ImmSuccessors(), nullptr);
	const MergedHullTraffic* first = nullptr;
	for (const auto& read: reads) {
		if (read.block == barrier && (first == nullptr || read.order < first->order)) first = &read;
	}
	for (const auto& write: writes) {
		if (!live.contains(write.block)) continue;
		if (first != nullptr && write.block == barrier ? write.order > first->order
		                                               : after.contains(write.block)) {
			late_write_pc = write.pc;
			return MergedHullBarrierResult::LateWrite;
		}
	}
	auto position = barrier->end();
	if (first != nullptr) {
		position = std::ranges::find_if(*barrier,
		                                [&](const Inst& inst) { return &inst == first->inst; });
	}
	barrier->PrependNewInst(position, ValueOpcode::Barrier);
	return MergedHullBarrierResult::Ordered;
}

void RequireMergedHullBarrierOrder(IR::Program& program, const ShaderTessellationInputInfo& tess,
                                   std::span<const MergedHullTraffic> barriers,
                                   std::span<const MergedHullTraffic> writes,
                                   std::span<const MergedHullTraffic> reads) {
	if (reads.empty()) return;
	for (const auto& barrier: barriers) {
		std::unordered_set<const IR::Block*> after;
		const auto                           first = barrier.block->ImmSuccessors();
		std::vector<const IR::Block*>        pending(first.begin(), first.end());
		while (!pending.empty()) {
			const auto* block = pending.back();
			pending.pop_back();
			if (!after.insert(block).second) continue;
			const auto next = block->ImmSuccessors();
			pending.insert(pending.end(), next.begin(), next.end());
		}
		const auto later = [&](const MergedHullTraffic& access) {
			return after.contains(access.block) ||
			       (access.block == barrier.block && access.order > barrier.order);
		};
		if (std::ranges::none_of(writes, later) && std::ranges::all_of(reads, later)) return;
	}
	uint32_t late_write_pc = 0;
	if (barriers.empty() && !program.dispatcher_fallback && tess.input_control_points != 0u &&
	    tess.input_control_points == tess.output_control_points &&
	    (program.wave_size == 32u || program.wave_size == 64u) &&
	    program.wave_size % tess.input_control_points == 0u &&
	    InsertMergedHullBarrier(program, writes, reads, true, late_write_pc) ==
	        MergedHullBarrierResult::Ordered) {
		LOGF("Merged hull wave-local LDS ordering: control_points=%u wave_size=%u first_read=0x%08x\n",
		     tess.input_control_points, program.wave_size, reads.front().pc);
		return;
	}
	EXIT("the merged hull program reads another control point's LDS record at pc 0x%08x, but no "
	     "s_barrier has every LDS write before it and every such read after it\n",
	     reads.front().pc);
}

void OrderMergedHullRingReads(IR::Program& program, std::span<const MergedHullTraffic> writes,
                              std::span<const MergedHullTraffic> reads) {
	uint32_t late_write_pc = 0;
	switch (InsertMergedHullBarrier(program, writes, reads, false, late_write_pc)) {
		case MergedHullBarrierResult::Ordered: return;
		case MergedHullBarrierResult::UnreachableRead:
			EXIT("the merged hull program reads another control point's off-chip output in a block "
			     "no control point reaches, at pc 0x%08x\n",
			     reads.front().pc);
		case MergedHullBarrierResult::NoUniformPoint:
			EXIT("the merged hull program reads another control point's off-chip output at pc "
			     "0x%08x, but no block before the read is reached once by every invocation, so no "
			     "barrier can order it after the writes\n",
			     reads.front().pc);
		case MergedHullBarrierResult::LateWrite:
			EXIT("the merged hull program reads another control point's off-chip output, but its "
			     "control point store at pc 0x%08x comes after the last block every invocation "
			     "reaches before the read\n",
			     late_write_pc);
	}
}

bool MergedHullFirstInvocationOnly(IR::Value predicate, MergedHullAddresses& addresses,
                                   std::unordered_set<const IR::Inst*>& path) {
	using namespace IR;
	predicate = predicate.Resolve();
	if (predicate.IsImmediate()) return predicate.GetType() == Type::U1 && !predicate.U1();
	const auto* inst = predicate.TryInstruction();
	if (inst == nullptr) return false;
	if (!path.insert(inst).second) return true;
	const auto invocation = [&](size_t index) {
		const auto value = addresses.Evaluate(inst->Arg(index));
		return value.known && value.coefficient == 1u && value.constant == 0u;
	};
	const auto immediate = [&](size_t index, uint32_t expected) {
		const auto value = inst->Arg(index).Resolve();
		return value.IsImmediate() && value.GetType() == Type::U32 && value.U32() == expected;
	};
	bool result = false;
	switch (inst->GetOpcode()) {
		case ValueOpcode::LogicalAnd:
			result = MergedHullFirstInvocationOnly(inst->Arg(0), addresses, path) ||
			         MergedHullFirstInvocationOnly(inst->Arg(1), addresses, path);
			break;
		case ValueOpcode::ULessThan32: result = invocation(0) && immediate(1, 1u); break;
		case ValueOpcode::UGreaterThan32: result = immediate(0, 1u) && invocation(1); break;
		case ValueOpcode::ULessThanEqual32: result = invocation(0) && immediate(1, 0u); break;
		case ValueOpcode::UGreaterThanEqual32: result = immediate(0, 0u) && invocation(1); break;
		case ValueOpcode::IEqual32:
			result = (invocation(0) && immediate(1, 0u)) || (immediate(0, 0u) && invocation(1));
			break;
		case ValueOpcode::Phi:
			result = inst->NumArgs() != 0u;
			for (size_t index = 0; result && index < inst->NumArgs(); index++) {
				result = MergedHullFirstInvocationOnly(inst->Arg(index), addresses, path);
			}
			break;
		default: break;
	}
	path.erase(inst);
	return result;
}

void RequireMergedHullFirstInvocation(const ShaderTessellationInputInfo& tess, IR::Value predicate,
                                      uint32_t pc) {
	if (tess.output_control_points == 1u) return;
	MergedHullAddresses                 addresses(tess.output_control_points, predicate);
	std::unordered_set<const IR::Inst*> path;
	if (!MergedHullFirstInvocationOnly(predicate, addresses, path)) {
		EXIT("the merged hull program's per-patch tessellation output at pc 0x%08x is not "
		     "written by control point 0 alone\n",
		     pc);
	}
}
#endif

}

#if defined(ANYPS5_KYTY)
void RequireMergedHullControlPoints(const ShaderTessellationInputInfo& info) {
	if (info.input_control_points > info.output_control_points) {
		EXIT("a merged hull program with more input (%u) than output (%u) control points runs "
		     "its local part on lanes the hull stage does not have\n",
		     info.input_control_points, info.output_control_points);
	}
}

void RequireMergedHullLimits(const ShaderTessellationInputInfo& info,
                             const TessellationLimits&          limits) {
	const auto vectors    = [](uint32_t bytes) { return (bytes + 15u) / 16u; };
	const auto patch      = vectors(info.patch_size);
	const auto per_vertex = vectors(info.hs_stride) + vectors(info.ls_stride);
	const auto locations  = per_vertex + patch;
	const auto total      = 4u * (uint64_t {info.output_control_points} * per_vertex + patch);
	if (4u * uint64_t {locations} > limits.per_vertex_output_components ||
	    4u * uint64_t {patch} > limits.per_patch_output_components ||
	    total > limits.total_output_components ||
	    vectors(IR::MergedHullRecordBytes) * 4u > limits.per_vertex_input_components) {
		EXIT("the merged hull program's outputs exceed the device's tessellation limits (LDS "
		     "stride %u, control point stride %u, patch constants %u bytes: %u per-vertex "
		     "locations, %u patch vectors, %" PRIu64 " components)\n",
		     info.ls_stride, info.hs_stride, info.patch_size, locations, patch, total);
	}
}
#endif

bool AnalyzeTessellationPrograms(std::span<const uint32_t> local, std::span<const uint32_t> control,
                                 ShaderTessellationInputInfo& info) {
#if defined(ANYPS5_KYTY)
	if (info.merged_hull != 0) {
		RequireMergedHullControlPoints(info);
		Decoder::Program merged;
		Decoder::DecodeProgram(control, merged);
		ReflectTessellationLayout(merged, true, info);
		ReflectTessellationLayout(merged, false, info);
		LOGF("Merged hull tessellation interface: input_cp=%u output_cp=%u ls_stride=%u "
		     "hs_stride=%u patch_base=0x%x patch_size=%u\n",
		     info.input_control_points, info.output_control_points, info.ls_stride,
		     info.hs_stride, info.patch_base, info.patch_size);
		return true;
	}
#endif
	const auto       local_program = Decoder::DecodeFrontProgram(local);
	Decoder::Program control_program;
	Decoder::DecodeProgram(control, control_program);
	ReflectTessellationLayout(local_program, true, info);
	ReflectTessellationLayout(control_program, false, info);
	LOGF("Tessellation interface: input_cp=%u output_cp=%u ls_stride=%u hs_stride=%u "
	     "patch_base=0x%x patch_size=%u\n",
	     info.input_control_points, info.output_control_points, info.ls_stride, info.hs_stride,
	     info.patch_base, info.patch_size);
	return true;
}

bool LowerTessellationMemory(IR::Program& program, const CompileOptions& options) {
	using namespace IR;
	if (options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		return true;
	}
	const auto unsupported = [&](const char* what) {
		TESS_LOG("%s tessellation lowering does not model %s\n",
		     options.stage == ShaderType::Local                 ? "LS"
		     : options.stage == ShaderType::TessellationControl ? "HS"
		                                                       : "TES",
		     what);
		return false;
	};
	const auto& tess = options.input_info.vertex->tess;
#if defined(ANYPS5_KYTY)
	const bool merged = options.stage == ShaderType::TessellationControl && tess.merged_hull != 0;
	if (merged) {
		RequireMergedHullControlPoints(tess);
		RequireMergedHullBarriers(program);
		LowerMergedHullQuadMasks(program);
		LowerMergedHullLanes(program);
		ConstantPropagationPass(program.blocks, program.wave_size);
	}
#endif
	{
		const auto shift_amount = [](const Value& value, uint32_t& amount) {
			const auto* shl = value.Resolve().TryInstruction();
			if (shl == nullptr || shl->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
			    !shl->Arg(1).Resolve().IsImmediate())
				return false;
			amount = shl->Arg(1).Resolve().U32();
			return amount < 32u;
		};
		for (auto* block: program.blocks) {
			for (auto& inst: *block) {
				uint32_t amount = 0;
				if (inst.GetOpcode() == ValueOpcode::BitwiseAnd32) {
					for (uint32_t side = 0; side < 2u; side++) {
						const auto mask = inst.Arg(side).Resolve();
						if (mask.IsImmediate() && shift_amount(inst.Arg(1u - side), amount) &&
						    (mask.U32() >> amount) == 0u) {
							inst.ReplaceUsesWith(Value(0u));
							break;
						}
					}
				} else if (inst.GetOpcode() == ValueOpcode::BitFieldUExtract &&
				           inst.Arg(1).Resolve().IsImmediate() &&
				           inst.Arg(2).Resolve().IsImmediate() &&
				           shift_amount(inst.Arg(0), amount) &&
				           inst.Arg(1).Resolve().U32() + inst.Arg(2).Resolve().U32() <= amount) {
					inst.ReplaceUsesWith(Value(0u));
				}
			}
		}
	}
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
#if defined(ANYPS5_KYTY)
			if (merged) {
				const bool lds =
				    SharedAccessOf(it->GetOpcode()) != SharedAccess::None &&
				    program.memory_info.at(it->Flags<MemoryFlags>().index).kind == ResourceKind::Lds;
				if (lds || TessellationBufferBase(*it) != nullptr) {
					const size_t index   = lds ? 0u : 2u;
					const auto   address = MergedHullAddresses(tess.output_control_points,
					                                           it->Arg(it->NumArgs() - 1u))
					                           .Evaluate(it->Arg(index));
					if (address.known) {
						it->SetArg(index, MergedHullAddressValue(*block, it, address));
						continue;
					}
					if (lds) {
						std::unordered_map<Inst*, Value> active;
						it->SetArg(0, ActiveAddress(*block, it, it->Arg(0),
						                            it->Arg(it->NumArgs() - 1u), active));
						continue;
					}
				}
			}
#endif
			if (TessellationBufferBase(*it) == nullptr) continue;
			std::unordered_map<Inst*, Value> resolved;
			it->SetArg(
			    2, ActiveAddress(*block, it, it->Arg(2), it->Arg(it->NumArgs() - 1u), resolved));
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	uint32_t reads = 0, writes = 0, factors = 0, patches = 0;
#if defined(ANYPS5_KYTY)
	std::vector<MergedHullTraffic> barriers, record_writes, record_reads, ring_writes, ring_reads;
	uint32_t                       order = 0;
#endif
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto&      inst   = *it;
#if defined(ANYPS5_KYTY)
			order++;
			if (merged && inst.GetOpcode() == ValueOpcode::Barrier) {
				barriers.push_back({block, order, block->start_pc});
			}
#endif
			const auto shared = SharedAccessOf(inst.GetOpcode());
			const auto buffer = BufferAccessOf(inst.GetOpcode());
			if (shared == SharedAccess::None && buffer == BufferAccess::None) {
				continue;
			}
			const auto&           memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			bool                  write  = false;
#if defined(ANYPS5_KYTY)
			bool                  cross_record_read = false;
#endif
			uint32_t              components = 0;
			TessellationAttribute kind;
			Value                 address, predicate;
			if (shared != SharedAccess::None && memory.kind == ResourceKind::Lds) {
				write = shared == SharedAccess::Write;
#if defined(ANYPS5_KYTY)
				if (merged) {
					if (memory.data_bits != 32u ||
					    (shared != SharedAccess::Read && shared != SharedAccess::Write)) {
						return unsupported("this LDS access");
					}
					RequireMergedHullRecord(tess, inst.Arg(0), memory.offset,
					                        SharedComponentCount(inst.GetOpcode()), write,
					                        inst.Flags<MemoryFlags>().pc);
					const bool own = MergedHullAddresses(tess.output_control_points, {})
					                     .Evaluate(inst.Arg(0))
					                     .coefficient != 0u;
					cross_record_read = !write && !own;
					if (write) record_writes.push_back({block, order, inst.Flags<MemoryFlags>().pc});
				} else
#endif
				if (memory.data_bits != 32u ||
				    (options.stage == ShaderType::Local
				         ? !write
				         : options.stage != ShaderType::TessellationControl ||
				               shared != SharedAccess::Read)) {
					return unsupported("this LDS access");
				}
				kind       = write ? TessellationAttribute::LocalOutput
				                   : TessellationAttribute::ControlInput;
#if defined(ANYPS5_KYTY)
				if (merged) kind = TessellationAttribute::ControlShared;
#endif
				components = SharedComponentCount(inst.GetOpcode());
				address    = inst.Arg(0);
				predicate  = inst.Arg(inst.NumArgs() - 1u);
			} else if (buffer != BufferAccess::None && memory.kind == ResourceKind::Buffer) {
				const auto* base = TessellationBufferBase(inst);
				if (base == nullptr) {
					continue;
				}
				if (memory.data_bits != 32u || memory.formatted || memory.idxen || !memory.offen ||
				    buffer == BufferAccess::Atomic) {
					return unsupported("this ring buffer access");
				}
				write      = buffer == BufferAccess::Write;
				components = BufferComponentCount(inst.GetOpcode());
				address    = inst.Arg(2).Resolve();
				predicate  = inst.Arg(inst.NumArgs() - 1u);
				if (base->Arg(0).U32() == 1u) {
					if (!write || options.stage != ShaderType::TessellationControl) {
						return unsupported("a tessellation factor access outside the HS");
					}
					kind = TessellationAttribute::Factor;
					factors += components;
#if defined(ANYPS5_KYTY)
					if (merged) {
						RequireMergedHullFirstInvocation(tess, predicate,
						                                 inst.Flags<MemoryFlags>().pc);
					}
#endif
				} else {
					const bool output = options.stage == ShaderType::TessellationControl;
					EXIT_NOT_IMPLEMENTED(
					    !output && (write || options.stage != ShaderType::TessellationEvaluation));
					kind = output ? TessellationAttribute::ControlOutput
					              : TessellationAttribute::EvaluationInput;
#if defined(ANYPS5_KYTY)
					if (merged) {
						RequireMergedHullRing(tess, address, memory.offset, components, write,
						                      inst.Flags<MemoryFlags>().pc);
					}
#endif
					if (address.IsImmediate() &&
					    address.U32() >= tess.hs_stride * tess.output_control_points) {
						const auto begin = address.U32() + memory.offset;
						EXIT_NOT_IMPLEMENTED(begin < tess.patch_base ||
						                     begin + 4u * components >
						                         tess.patch_base + tess.patch_size);
						kind = output ? TessellationAttribute::PatchOutput
						              : TessellationAttribute::PatchInput;
#if defined(ANYPS5_KYTY)
						if (merged && write) {
							RequireMergedHullFirstInvocation(tess, predicate,
							                                 inst.Flags<MemoryFlags>().pc);
						}
#endif
						if (write || !output) patches += components;
					}
				}
			} else {
				continue;
			}
			const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
				return Value(&*block->PrependNewInst(it, opcode, args));
			};
			std::array<Value, 4> values;
			for (uint32_t component = 0; component < components; component++) {
				const auto offset = memory.offset + 4u * component;
				const auto byte_address =
				    offset == 0u ? address : emit(ValueOpcode::IAdd32, {address, Value(offset)});
				if (write) {
					Value data = shared != SharedAccess::None ? inst.Arg(component + 1u)
					             : components == 1u
					                 ? inst.Arg(4)
					                 : emit(components == 2u   ? ValueOpcode::CompositeExtractU32x2
					                        : components == 3u ? ValueOpcode::CompositeExtractU32x3
					                                           : ValueOpcode::CompositeExtractU32x4,
					                        {inst.Arg(4), Value(component)});
					emit(ValueOpcode::SetTessellationAttribute,
					     {Value(static_cast<uint32_t>(kind)), byte_address, data, predicate});
					writes++;
				} else {
					values[component] =
					    emit(ValueOpcode::GetTessellationAttribute,
					         {Value(static_cast<uint32_t>(kind)), byte_address, predicate});
					reads++;
				}
			}
			if (!write) {
				Value replacement = values[0];
				if (components == 2u)
					replacement =
					    emit(ValueOpcode::CompositeConstructU32x2, {values[0], values[1]});
				if (components == 3u)
					replacement = emit(ValueOpcode::CompositeConstructU32x3,
					                   {values[0], values[1], values[2]});
				if (components == 4u)
					replacement = emit(ValueOpcode::CompositeConstructU32x4,
					                   {values[0], values[1], values[2], values[3]});
				inst.ReplaceUsesWith(replacement);
			}
#if defined(ANYPS5_KYTY)
			if (cross_record_read) {
				record_reads.push_back(
				    {block, order, inst.Flags<MemoryFlags>().pc, values[0].TryInstruction()});
			}
			if (merged && kind == TessellationAttribute::ControlOutput &&
			    tess.output_control_points != 1u) {
				const auto pc = inst.Flags<MemoryFlags>().pc;
				if (write) {
					ring_writes.push_back({block, order, pc});
				} else if (MergedHullAddresses(tess.output_control_points, {})
				               .Evaluate(address)
				               .coefficient == 0u) {
					ring_reads.push_back({block, order, pc, values[0].TryInstruction()});
				}
			}
#endif
			inst.Invalidate();
		}
	}
#if defined(ANYPS5_KYTY)
	if (merged) RequireMergedHullBarrierOrder(program, tess, barriers, record_writes, record_reads);
	OrderMergedHullRingReads(program, ring_writes, ring_reads);
#endif
	ConstantPropagationPass(program.blocks, program.wave_size);
	RemoveIdentities(program);
	EliminateDeadCode(program.blocks);
	EXIT_NOT_IMPLEMENTED(options.stage == ShaderType::TessellationControl &&
	                     tess.patch_size != 0u && patches == 0u);
	LOGF("%s tessellation lowering: reads=%u writes=%u factors=%u patches=%u\n",
	     options.stage == ShaderType::Local                 ? "LS"
	     : options.stage == ShaderType::TessellationControl ? "HS"
	                                                       : "TES",
	     reads, writes, factors, patches);
	return true;
}

}
