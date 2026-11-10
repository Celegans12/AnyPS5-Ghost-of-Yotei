#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/envRead.h"
#include "common/shaderEnv.h"
#include "common/logging/log.h"
#include "common/workaroundHits.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

struct TessellationAddress {
	enum class Kind { Unknown, Affine, PackedControlPoint };
	Kind     kind        = Kind::Unknown;
	uint32_t coefficient = 0;
	uint32_t constant    = 0;
};

uint32_t ReflectTessellationStride(const Decoder::Program& program, bool local,
                                   uint32_t control_points, uint32_t input_stride) {
	using namespace Decoder;
	using Address = TessellationAddress;
	using Kind    = Address::Kind;
	std::array<Address, IR::NumVectorRegs> registers {};
	registers[local ? 3u : 1u] =
	    local ? Address {Kind::Affine, 1u, 0u} : Address {Kind::PackedControlPoint};
	const auto constant = [](uint32_t value) { return Address {Kind::Affine, 0u, value}; };
	const auto read     = [&](const Operand& operand) -> Address {
		if (operand.absolute || operand.negate || operand.sdwa_sext || operand.op_sel ||
		    operand.op_sel_hi || operand.negate_hi || operand.dpp)
			return {};
		Address value;
		if (operand.kind == OperandKind::IntegerInlineConstant ||
		    operand.kind == OperandKind::LiteralConstant) {
			value = constant(operand.value);
		} else if (operand.kind == OperandKind::Vgpr) {
			value = registers.at(operand.reg);
		}
		// The native packed HS ID's low byte is the patch ordinal. The logical
		// interface addresses one patch, as in the frontend's v1 initialization.
		if (value.kind == Kind::PackedControlPoint && operand.sdwa_sel == 0u) return constant(0u);
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
		if (value.kind == Kind::Affine && value.coefficient == 0u)
			return constant(value.constant & 0xffffffu);
		return value.kind == Kind::Affine &&
		               uint64_t {value.coefficient} * (control_points - 1u) + value.constant <=
		                   0xffffffu
		           ? value
		           : Address {};
	};
	uint32_t stride = 0;
	// Data flow over the instructions: the register state at a branch target is the merge of the states of every path
	// reaching it (equal definitions survive, others become unknown), iterated to a fixed point so forward branches and
	// loops are handled; code after an unconditional branch is reached only through such a join. Branches to the last
	// instruction (stage exits) preserve the active path's definitions, as before. The address checks run once over the
	// converged states.
	using Registers = std::array<Address, IR::NumVectorRegs>;
	const auto same = [](const Address& a, const Address& b) {
		return a.kind == b.kind && a.coefficient == b.coefficient && a.constant == b.constant;
	};
	const auto& insts = program.instructions;
	const auto  exit_pc = insts.back().pc;
	std::unordered_map<uint32_t, size_t> index_of;
	for (size_t i = 0; i < insts.size(); i++) index_of.emplace(insts[i].pc, i);
	std::vector<Registers> in(insts.size());
	std::vector<bool>      reached(insts.size(), false), queued(insts.size(), false);
	std::vector<size_t>    work;
	const auto join_into = [&](size_t target, const Registers& state) {
		if (!reached[target]) {
			reached[target] = true;
			in[target]      = state;
		} else {
			bool changed = false;
			for (size_t r = 0; r < state.size(); r++) {
				if (!same(in[target][r], state[r]) && in[target][r].kind != Kind::Unknown) {
					in[target][r] = {};
					changed       = true;
				}
			}
			if (!changed) return;
		}
		if (!queued[target]) {
			queued[target] = true;
			work.push_back(target);
		}
	};
	const auto step = [&](const Decoder::Instruction& inst, bool check) {
		const bool local_store =
		    inst.opcode == Opcode::DS_WRITE_B32 || inst.opcode == Opcode::DS_WRITE2_B32 ||
		    inst.opcode == Opcode::DS_WRITE_B64 || inst.opcode == Opcode::DS_WRITE2_B64 ||
		    inst.opcode == Opcode::DS_WRITE_B96 || inst.opcode == Opcode::DS_WRITE_B128;
		const bool buffer_store = inst.opcode == Opcode::BUFFER_STORE_DWORD ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX2 ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX3 ||
		                          inst.opcode == Opcode::BUFFER_STORE_DWORDX4;
		const bool control_store =
		    !local && buffer_store && inst.src2.kind == OperandKind::Sgpr && inst.src2.reg == 2u;
		const bool control_read =
		    !local && (inst.opcode == Opcode::DS_READ_B32 || inst.opcode == Opcode::DS_READ2_B32 ||
		               inst.opcode == Opcode::DS_READ_B64 || inst.opcode == Opcode::DS_READ2_B64 ||
		               inst.opcode == Opcode::DS_READ_B96 || inst.opcode == Opcode::DS_READ_B128);
		if (check && ((local && local_store) || control_store || control_read)) {
			const auto address = read(inst.src0);
			if (address.kind != Kind::Affine) {
				EXIT("%s tessellation address is not affine at pc 0x%08x\n", local ? "LS" : "HS",
				     inst.pc);
			}
			if (address.coefficient != 0u) {
				const auto expected = control_read ? input_stride : stride;
				EXIT_NOT_IMPLEMENTED((address.coefficient & 3u) != 0u ||
				                     (expected != 0u && expected != address.coefficient));
				if (!control_read) stride = address.coefficient;
			}
		}
		// Store vdata is a source, despite occupying the decoder's dst field.
		if (local_store || buffer_store || inst.dst.kind != OperandKind::Vgpr) return;
		const auto lhs   = read(inst.src0);
		const auto rhs   = read(inst.src1);
		const auto third = read(inst.src2);
		Address    value;
		switch (inst.opcode) {
			case Opcode::V_BFE_U32:
				if (lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
				    rhs.coefficient == 0u && third.kind == Kind::Affine &&
				    third.coefficient == 0u) {
					if (rhs.constant == 8u && third.constant == 5u) value = {Kind::Affine, 1u, 0u};
					if (rhs.constant == 0u && third.constant == 8u) value = constant(0u);
				}
				break;
			case Opcode::V_MUL_U32_U24: value = multiply(low24(lhs), low24(rhs)); break;
			case Opcode::V_MAD_U32_U24: value = add(multiply(low24(lhs), low24(rhs)), third); break;
			case Opcode::V_LSHL_ADD_U32:
				if (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant < 32u)
					value = add(multiply(lhs, constant(1u << rhs.constant)), third);
				break;
			case Opcode::V_LSHLREV_B32:
				if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant < 32u)
					value = multiply(rhs, constant(1u << lhs.constant));
				break;
			case Opcode::V_MOV_B32: value = lhs; break;
			case Opcode::V_ADD_NC_U32: value = add(lhs, rhs); break;
			case Opcode::V_AND_B32:
				// The packed HS ID's low byte (& 0xff) is the patch ordinal, 0 for the logical interface.
				if ((lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine && rhs.coefficient == 0u &&
				     rhs.constant == 0xffu) ||
				    (rhs.kind == Kind::PackedControlPoint && lhs.kind == Kind::Affine && lhs.coefficient == 0u &&
				     lhs.constant == 0xffu)) {
					value = constant(0u);
				} else if ((lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine && rhs.coefficient == 0u &&
				            rhs.constant == 0x1f00u) ||
				           (rhs.kind == Kind::PackedControlPoint && lhs.kind == Kind::Affine && lhs.coefficient == 0u &&
				            lhs.constant == 0x1f00u)) {
					// Upstream 77458260f: the masked control-point field (& 0x1f00) is the control point times 256.
					value = {Kind::Affine, 256u, 0u};
				} else if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && rhs.kind == Kind::Affine &&
				           rhs.coefficient == 0u) {
					value = constant(lhs.constant & rhs.constant);
				}
				break;
			case Opcode::V_OR_B32:
				if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && rhs.kind == Kind::Affine &&
				    rhs.coefficient == 0u)
					value = constant(lhs.constant | rhs.constant);
				break;
			case Opcode::V_SUB_NC_U32:
				if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine &&
				    lhs.coefficient >= rhs.coefficient && lhs.constant >= rhs.constant)
					value = affine(lhs.coefficient - rhs.coefficient, lhs.constant - rhs.constant);
				break;
			default: break;
		}
		if (inst.dst.sdwa_sel != 6u || inst.dst.op_sel || inst.dst.omod || inst.dst.clamp ||
		    inst.dst.dpp)
			value = {};
		for (uint32_t index = 0;
		     index < std::max(inst.data_dwords, 1u) && inst.dst.reg + index < registers.size(); index++) {
			registers[inst.dst.reg + index] = {};
		}
		registers.at(inst.dst.reg) = value;
	};
	join_into(0, registers);
	while (!work.empty()) {
		const auto i = work.back();
		work.pop_back();
		queued[i]          = false;
		registers          = in[i];
		const auto& inst   = insts[i];
		bool        falls  = true;
		if (IsDirectBranch(inst.opcode) && inst.branch_target != exit_pc) {
			const auto target = index_of.find(inst.branch_target);
			EXIT_NOT_IMPLEMENTED(target == index_of.end());
			join_into(target->second, registers);
			falls = inst.opcode != Opcode::S_BRANCH;
		} else if (!IsDirectBranch(inst.opcode)) {
			step(inst, false);
		}
		if (falls && i + 1 < insts.size()) join_into(i + 1, registers);
	}
	for (size_t i = 0; i < insts.size(); i++) {
		if (!reached[i] || IsDirectBranch(insts[i].opcode)) continue;
		registers = in[i];
		step(insts[i], true);
	}
	EXIT_NOT_IMPLEMENTED(stride == 0u);
	return stride;
}

const IR::Inst* TessellationBufferBase(const IR::Inst& inst) {
	if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::None || inst.NumArgs() <= 3u)
		return nullptr;
	const auto* base = inst.Arg(3).Resolve().TryInstruction();
	return base != nullptr && base->GetOpcode() == IR::ValueOpcode::TessellationBase ? base
	                                                                                 : nullptr;
}

IR::Value ActiveAddress(IR::Block& block, IR::Block::iterator before, IR::Value value,
                        IR::Value predicate, std::unordered_map<IR::Inst*, IR::Value>& resolved) {
	using namespace IR;
	value        = value.Resolve();
	auto* source = value.TryInstruction();
	if (source == nullptr) return value;
	if (const auto found = resolved.find(source); found != resolved.end()) return found->second;
	if (source->GetOpcode() == ValueOpcode::SelectU32) {
		return source->Arg(0).Resolve() == predicate.Resolve()
		           ? ActiveAddress(block, before, source->Arg(1), predicate, resolved)
		           : value;
	}
	if (source->GetOpcode() != ValueOpcode::IAdd32 && source->GetOpcode() != ValueOpcode::ISub32)
		return value;
	const auto lhs = ActiveAddress(block, before, source->Arg(0), predicate, resolved);
	const auto rhs = ActiveAddress(block, before, source->Arg(1), predicate, resolved);
	if (lhs != source->Arg(0).Resolve() || rhs != source->Arg(1).Resolve()) {
		auto copy = block.PrependNewInst(before, source->GetOpcode(), {lhs, rhs},
		                                 source->Flags<uint64_t>());
		value     = Value(&*copy);
	}
	resolved.emplace(source, value);
	return value;
}

// A ring address the constant propagation cannot fold is evaluated with the lane's predicated VGPR writes taken as
// performed (the access runs under the EXEC of the lanes that wrote its address registers) through additions, 24-bit
// multiplies, shifts and masks. The water hull shader c1a55c56c2e59361 stores its per-patch data at
// 0x8000 - 0x60 * (patch ordinal + 1); unfolded, that address was taken as control-point data of invocation 0
// (32672 % 80) and overwrote control point 0 (exploded water). Exact when every write taken as performed was made
// by every accessing lane or leaves the same address either way (all cases in the game are the first);
// an address that depends on the lane stops the translation.
// `access_predicate` / `other_exec` (KYTY_TRACE_PERF, PERF workaround_hits tess_ring): set when a predicated write taken
// as performed was made under another EXEC value than the access's (the case the evaluation cannot prove exact).
bool EvaluateActiveAddress(IR::Value value, uint32_t& result, uint32_t depth = 0, const IR::Value* access_predicate = nullptr,
                           bool* other_exec = nullptr) {
	using namespace IR;
	value = value.Resolve();
	if (value.IsImmediate()) {
		if (value.GetType() != Type::U32) return false;
		result = value.U32();
		return true;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 48u) return false;
	uint32_t a = 0, b = 0, c = 0;
	const auto arg = [&](size_t index, uint32_t& out) {
		return EvaluateActiveAddress(inst->Arg(index), out, depth + 1u, access_predicate, other_exec);
	};
	// (x << s) & m == 0 and BFE(x << s, offset, count) == 0 when the field lies below the shift (packed patch ordinal).
	const auto shifted_below = [](const Inst* source, uint32_t limit) {
		if (source == nullptr || source->GetOpcode() != ValueOpcode::ShiftLeftLogical32) return false;
		const auto shift = source->Arg(1).Resolve();
		return shift.IsImmediate() && shift.U32() < 32u && limit <= shift.U32();
	};
	switch (inst->GetOpcode()) {
		case ValueOpcode::SelectU32:
			// A predicated VGPR write: Select(exec, new value, old value).
			if (arg(1, a)) {
				if (other_exec != nullptr && access_predicate != nullptr &&
				    inst->Arg(0).Resolve() != access_predicate->Resolve()) {
					// Still exact when every accessing lane made the write (the access's EXEC is a LogicalAnd chain with
					// the write's) or when the old value is the same address; otherwise the address depends on the lane.
					const auto implies = [&](auto&& self, Value p, Value q, uint32_t d) -> bool {
						p = p.Resolve();
						if (p == q) return true;
						const auto* conj = p.TryInstruction();
						return d < 8u && conj != nullptr && conj->GetOpcode() == ValueOpcode::LogicalAnd &&
						       (self(self, conj->Arg(0), q, d + 1u) || self(self, conj->Arg(1), q, d + 1u));
					};
					uint32_t   old     = 0;
					const bool covered = implies(implies, *access_predicate, inst->Arg(0).Resolve(), 0u);
					const bool same    = !covered && arg(2, old) && old == a;
					static std::atomic_uint logged {0};
					if (logged.fetch_add(1) < 32u) {
						std::fprintf(stderr, "TESS_RING_PREDICATE %s new=%u%s\n",
						             covered ? "covered" : same ? "same_both" : "lane_dependent", a,
						             same ? "" : covered ? "" : " (address depends on the lane)");
					}
					if (!covered && !same) *other_exec = true;
				}
				result = a;
				return true;
			}
			return false;
		case ValueOpcode::Phi: {
			bool found = false;
			for (size_t i = 0; i < inst->NumArgs(); i++) {
				if (!arg(i, a)) return false;
				if (found && a != result) return false;
				result = a;
				found  = true;
			}
			return found;
		}
		case ValueOpcode::IAdd32:
			if (!arg(0, a) || !arg(1, b)) return false;
			result = a + b;
			return true;
		case ValueOpcode::ISub32:
			if (!arg(0, a) || !arg(1, b)) return false;
			result = a - b;
			return true;
		case ValueOpcode::IMul32:
			if (!arg(0, a) || !arg(1, b)) return false;
			result = a * b;
			return true;
		case ValueOpcode::ShiftLeftLogical32:
			if (!arg(0, a) || !arg(1, b) || b >= 32u) return false;
			result = a << b;
			return true;
		case ValueOpcode::ShiftRightLogical32:
			if (!arg(0, a) || !arg(1, b) || b >= 32u) return false;
			result = a >> b;
			return true;
		case ValueOpcode::BitwiseOr32:
			if (!arg(0, a) || !arg(1, b)) return false;
			result = a | b;
			return true;
		case ValueOpcode::BitwiseAnd32: {
			for (size_t side = 0; side < 2u; side++) {
				const auto mask = inst->Arg(side ^ 1u).Resolve();
				if (mask.IsImmediate()) {
					uint32_t width = 0;
					while (width < 32u && (mask.U32() >> width) != 0u) width++;
					if (shifted_below(inst->Arg(side).Resolve().TryInstruction(), width)) {
						result = 0;
						return true;
					}
				}
			}
			if (!arg(0, a) || !arg(1, b)) return false;
			result = a & b;
			return true;
		}
		case ValueOpcode::BitFieldUExtract: {
			if (!arg(1, b) || !arg(2, c) || b > 32u || c > 32u - b) return false;
			if (shifted_below(inst->Arg(0).Resolve().TryInstruction(), b + c)) {
				result = 0;
				return true;
			}
			if (!arg(0, a)) return false;
			result = c == 0u ? 0u : c == 32u ? a : (a >> b) & ((1u << c) - 1u);
			return true;
		}
		default: return false;
	}
}

// The VGPRs the HS part of a split merged LS+HS program reads before writing them (ShaderTessellationInputInfo::
// carried_vgprs). On the hardware one wave runs the LS part and then the HS part, so such a register still holds what
// the LS part left in the lane: the 2.000 translucent-particle HS takes the particle's index, flags and camera-relative
// position that way (v6-v9, v12, v13) instead of reading them back from LDS as the 1.x HS halves do.
// The set may only err on the large side: a read counts every register its operand could cover, a write counts only
// when it certainly replaces the whole register in every lane that later reads it, i.e. a full-width write in the
// straight-line code after the entry sequence (EXEC = all, V_CMPX lane < HS threads, S_CBRANCH_EXECZ) and before the
// first branch or other EXEC change (a write under a narrower EXEC keeps the LS value in the other lanes). An extra
// register only costs one carried u32.
std::array<uint32_t, 8> MergedHullCarriedVgprs(const Decoder::Program& control) {
	using namespace Decoder;
	std::bitset<IR::NumVectorRegs> defined, live;
	defined.set(0); // the back half's own V_MBCNT lane index (SplitMergedHullProgram)
	defined.set(1); // the packed HS IDs the frontend sets
	bool entry    = true;
	bool definite = true;
	for (const auto& inst: control.instructions) {
		const auto text  = InstructionToString(inst);
		const auto colon = text.find(": ");
		const auto start = colon == std::string::npos ? 0u : colon + 2u;
		const auto name  = text.substr(start, text.find(' ', start) - start);
		const auto has   = [&](const char* part) { return name.find(part) != std::string::npos; };
		const bool memory = inst.family == Family::MUBUF || inst.family == Family::MTBUF ||
		                    inst.family == Family::FLAT || inst.family == Family::DS || inst.family == Family::MIMG;
		const bool load   = memory && !has("D16") &&
		                  (has("LOAD") || name.starts_with("DS_READ") || has("SAMPLE") || has("GATHER"));
		const bool valu   = inst.family == Family::VOP1 || inst.family == Family::VOP2 ||
		                  inst.family == Family::VOP3 || inst.family == Family::VOP3P;
		// Destinations that keep part of the old value (or read it): accumulators, lane writes, sub-dword writes.
		const bool partial = has("MAC") || has("DOT2C") || has("DOT4C") || has("DOT8C") || has("WRITELANE") ||
		                     has("MOVREL") || has("D16") || inst.dst.sdwa_sel != 6u || inst.dst.op_sel || inst.dst.dpp;
		const auto read = [&](const Operand& operand, uint32_t width) {
			if (operand.kind != OperandKind::Vgpr) return;
			for (uint32_t index = 0; index < width && operand.reg + index < IR::NumVectorRegs; index++) {
				if (!defined[operand.reg + index]) live.set(operand.reg + index);
			}
		};
		const uint32_t source_width = memory ? std::max(inst.data_dwords, 4u)
		                              : has("64")  ? 2u
		                                           : 1u;
		for (const auto* operand: {&inst.src0, &inst.src1, &inst.src2, &inst.src3}) read(*operand, source_width);
		if (inst.family == Family::MIMG) {
			read(inst.src0, std::max(inst.image_address_components, 4u));
			for (uint32_t index = 0; index < inst.image_nsa_dwords && index < MaxImageNsaAddressComponents; index++) {
				const Operand address {.kind = OperandKind::Vgpr, .reg = inst.image_nsa_addr[index]};
				read(address, 1u);
			}
		}
		if ((memory && !load) || ((valu || memory) && partial)) {
			read(inst.dst, std::max(inst.data_dwords, 2u)); // store data, atomics, partial writes
		}
		if (entry) {
			const bool entry_inst = inst.opcode == Opcode::V_MBCNT_LO_U32_B32 || inst.opcode == Opcode::V_MBCNT_HI_U32_B32 ||
			                        inst.opcode == Opcode::S_MOV_B64 || inst.opcode == Opcode::V_CMPX_LT_U32 ||
			                        inst.opcode == Opcode::S_BARRIER || inst.opcode == Opcode::S_INST_PREFETCH ||
			                        inst.opcode == Opcode::S_WAITCNT || inst.opcode == Opcode::S_NOP;
			if (inst.opcode == Opcode::S_CBRANCH_EXECZ || !entry_inst) entry = false;
			if (entry || inst.opcode == Opcode::S_CBRANCH_EXECZ) continue;
		}
		if (IsDirectBranch(inst.opcode) || inst.opcode == Opcode::S_SETPC_B64 || inst.opcode == Opcode::S_SWAPPC_B64 ||
		    inst.opcode == Opcode::S_ENDPGM || text.find("exec") != std::string::npos ||
		    text.find("EXEC") != std::string::npos) {
			definite = false;
		}
		if (!definite || inst.dst.kind != OperandKind::Vgpr || partial) continue;
		if (valu) {
			defined.set(inst.dst.reg);
		} else if (load && (inst.family == Family::DS || inst.family == Family::MUBUF || inst.family == Family::MTBUF)) {
			for (uint32_t index = 0; index < std::max(inst.data_dwords, 1u) && inst.dst.reg + index < IR::NumVectorRegs;
			     index++) {
				defined.set(inst.dst.reg + index);
			}
		} else if (load) {
			defined.set(inst.dst.reg);
		}
	}
	live.reset(0);
	live.reset(1);
	std::array<uint32_t, 8> mask {};
	for (uint32_t reg = 0; reg < IR::NumVectorRegs; reg++) {
		if (live[reg]) mask[reg / 32u] |= 1u << (reg % 32u);
	}
	return mask;
}

} // namespace


void AnalyzeTessellationPrograms(std::span<const uint32_t> local, std::span<const uint32_t> control,
                                 ShaderTessellationInputInfo& info, bool merged) {
	const auto       local_program = Decoder::DecodeFrontProgram(local);
	Decoder::Program control_program;
	Decoder::DecodeProgram(control, control_program);
	// KYTY_TRACE_SHADER_CODE_DIR=<dir> (diagnostic): the LS and HS code of every analysed pair, raw and decoded, named
	// by an FNV-1a hash of the code (written before the analysis, so an unsupported pattern can be inspected).
	if (const char* dir = Common::EnvRead::ShaderCodeDir(); dir != nullptr) {
		const auto dump = [&](const char* stage, std::span<const uint32_t> code, const Decoder::Program& decoded) {
			uint64_t hash = 0xcbf29ce484222325ull;
			for (const auto word: code) hash = (hash ^ word) * 0x100000001b3ull;
			const auto base = fmt::format("{}/tess_{}_{:016x}", dir, stage, hash);
			if (FILE* f = std::fopen((base + ".bin").c_str(), "wb")) {
				std::fwrite(code.data(), 1, code.size_bytes(), f);
				std::fclose(f);
			}
			const auto text = Decoder::ProgramToString(decoded);
			if (FILE* f = std::fopen((base + ".asm").c_str(), "w")) {
				std::fwrite(text.data(), 1, text.size(), f);
				std::fclose(f);
			}
			std::fprintf(stderr, "TESS_DUMP %s %s.bin (%zu dwords)\n", stage, base.c_str(), code.size());
		};
		dump("ls", local, local_program);
		dump("hs", control, control_program);
	}
	info.ls_stride = ReflectTessellationStride(local_program, true, info.input_control_points, 0u);
	info.hs_stride = ReflectTessellationStride(control_program, false, info.output_control_points,
	                                           info.ls_stride);
	LOGF("Tessellation interface: input_cp=%u output_cp=%u ls_stride=%u hs_stride=%u\n",
	     info.input_control_points, info.output_control_points, info.ls_stride, info.hs_stride);
	info.carried_vgprs = {};
	if (!merged) return;
	info.carried_vgprs = MergedHullCarriedVgprs(control_program);
	std::string list;
	uint32_t    count = 0;
	for (uint32_t reg = 0; reg < IR::NumVectorRegs; reg++) {
		if (((info.carried_vgprs[reg / 32u] >> (reg % 32u)) & 1u) == 0u) continue;
		list += fmt::format(" v{}", reg);
		count++;
	}
	if (count == 0u) return;
	// HS thread i of a patch ran the LS part for the patch's vertex i: the same lane only when the counts agree.
	EXIT_NOT_IMPLEMENTED(info.input_control_points != info.output_control_points);
	// The carried registers follow the LS outputs in the LS -> HS interface (Vulkan guarantees 64 components per
	// vertex there, the GPUs this runs on offer 128).
	const auto components = (info.ls_stride + 15u) / 16u * 4u + (count + 3u) / 4u * 4u;
	if (components > 128u) {
		EXIT("merged HS program carries %u VGPRs from its LS part: %u interface components (max 128):%s\n", count,
		     components, list.c_str());
	}
	LOGF("Tessellation interface: the merged program's HS part carries %u VGPRs from its LS part:%s\n", count,
	     list.c_str());
	if (std::getenv("KYTY_TRACE_TESS_LOWERING") != nullptr) {
		std::fprintf(stderr, "TESS_CARRIED count=%u%s\n", count, list.c_str());
	}
}

void LowerTessellationMemory(IR::Program& program, const CompileOptions& options) {
	using namespace IR;
	if (options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		return;
	}
	const auto& tess = options.input_info.vertex->tess;
	// Ring addresses can reuse data VGPRs. Their inactive values are irrelevant to
	// a store guarded by the same EXEC predicate, but must remain intact elsewhere.
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			if (TessellationBufferBase(*it) == nullptr) continue;
			std::unordered_map<Inst*, Value> resolved;
			it->SetArg(
			    2, ActiveAddress(*block, it, it->Arg(2), it->Arg(it->NumArgs() - 1u), resolved));
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	uint32_t reads = 0, writes = 0, factors = 0;
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto&      inst   = *it;
			const auto shared = SharedAccessOf(inst.GetOpcode());
			const auto buffer = BufferAccessOf(inst.GetOpcode());
			if (shared == SharedAccess::None && buffer == BufferAccess::None) {
				continue;
			}
			const auto&           memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			bool                  write  = false;
			uint32_t              components = 0;
			TessellationAttribute kind;
			Value                 address, predicate;
			if (shared != SharedAccess::None && memory.kind == ResourceKind::Lds) {
				write = shared == SharedAccess::Write;
				EXIT_NOT_IMPLEMENTED(memory.data_bits != 32u ||
				                     (options.stage == ShaderType::Local
				                          ? !write
				                          : options.stage != ShaderType::TessellationControl ||
				                                shared != SharedAccess::Read));
				kind       = write ? TessellationAttribute::LocalOutput
				                   : TessellationAttribute::ControlInput;
				components = SharedComponentCount(inst.GetOpcode());
				address    = inst.Arg(0);
				predicate  = inst.Arg(inst.NumArgs() - 1u);
			} else if (buffer != BufferAccess::None && memory.kind == ResourceKind::Buffer) {
				const auto* base = TessellationBufferBase(inst);
				if (base == nullptr) {
					continue;
				}
				EXIT_NOT_IMPLEMENTED(memory.data_bits != 32u || memory.formatted || memory.idxen ||
				                     !memory.offen || buffer == BufferAccess::Atomic);
				write      = buffer == BufferAccess::Write;
				components = BufferComponentCount(inst.GetOpcode());
				address    = inst.Arg(2).Resolve();
				predicate  = inst.Arg(inst.NumArgs() - 1u);
				if (bool other_exec = false; !address.IsImmediate()) {
					if (uint32_t evaluated = 0; EvaluateActiveAddress(address, evaluated, 0, &predicate, &other_exec)) {
						static std::atomic_uint logged {0};
						if (other_exec || logged.fetch_add(1) < 16u) {
							std::fprintf(stderr, "TESS_RING_ADDRESS stage=%u write=%d address=%u%s\n",
							             static_cast<uint32_t>(options.stage), write ? 1 : 0, evaluated,
							             other_exec ? " other_exec" : "");
						}
						address = Value(evaluated);
						// KYTY_TRACE_PERF (PERF workaround_hits tess_ring): per translated program.
						Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::TessRingEvaluated);
						if (other_exec) {
							EXIT("tessellation ring address %u depends on the lane (stage %u): a predicated write the access does not "
							     "cover\n",
							     evaluated, static_cast<uint32_t>(options.stage));
						}
					}
				}
				if (base->Arg(0).U32() == 1u) {
					EXIT_NOT_IMPLEMENTED(!write ||
					                     options.stage != ShaderType::TessellationControl);
					kind = TessellationAttribute::Factor;
					factors += components;
				} else {
					kind = write ? TessellationAttribute::ControlOutput
					             : TessellationAttribute::EvaluationInput;
					EXIT_NOT_IMPLEMENTED(write
					                         ? options.stage != ShaderType::TessellationControl
					                         : options.stage != ShaderType::TessellationEvaluation);
					if (address.IsImmediate() &&
					    address.U32() >= tess.hs_stride * tess.output_control_points) {
						if (!write) {
							EXIT_NOT_IMPLEMENTED(options.stage != ShaderType::TessellationEvaluation);
							kind = TessellationAttribute::PatchInput;
						} else {
							EXIT_NOT_IMPLEMENTED(!write);
							kind = TessellationAttribute::PatchOutput;
						}
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
			inst.Invalidate();
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	RemoveIdentities(program);
	EliminateDeadCode(program.blocks);
	LOGF("%s tessellation lowering: reads=%u writes=%u factors=%u\n",
	     options.stage == ShaderType::Local                 ? "LS"
	     : options.stage == ShaderType::TessellationControl ? "HS"
	                                                       : "TES",
	     reads, writes, factors);
	// KYTY_TRACE_TESS_LOWERING (diagnostic): the same summary on stderr, with the program and its layout.
	if (std::getenv("KYTY_TRACE_TESS_LOWERING") != nullptr) {
		std::fprintf(stderr, "TESS_LOWERING %s hash=0x%016llx reads=%u writes=%u factors=%u ls_stride=%u hs_stride=%u cp=%u/%u\n",
		             options.stage == ShaderType::Local                 ? "LS"
		             : options.stage == ShaderType::TessellationControl ? "HS"
		                                                               : "TES",
		             static_cast<unsigned long long>(options.shader_hash), reads, writes, factors, tess.ls_stride,
		             tess.hs_stride, tess.input_control_points, tess.output_control_points);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler
