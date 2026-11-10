#include "graphics/shader/recompiler/frontend/translate/Translator.h"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

// f16 adds, multiplies, FMAs and min/max/min3/max3/med3 (V_PK_* and the scalar V_*_F16 forms) are HalfPk* operations on
// the packed bits, emitted as f16vec2 arithmetic instead of each lane through exact f16 -> f32 and f32 -> f16 conversions
// (FSR3 accumulate d4288915: ~71% of its SPIR-V operations were those conversions, 7.1 -> 4.7 ms on the GPU), and the
// mixed-precision FMA's f16 operands and result take plain conversions. Same results as the f32 path (all lane pairs,
// spirvEmitterAluHelpers.cpp), except that an FMA rounds once, as the guest's fused FMA does.
// The signaling-NaN override of the native V_PK_MIN/MAX_F16 path works on whole packed words instead of per lane (package
// FSR3 1.84 -> 1.62 ms, Volfog 1.02 -> 0.86 ms, open world +1.4%). For an operand's two raw lanes X (exactly the bits the
// per-lane form reads), s = ((X & 0x7dff7dff) + 0x03ff03ff) & 0x80008000 has bit 15 of a lane set iff the lane's exponent
// is all ones and its low 9 payload bits are not zero (the per-lane test; no carry crosses into the other lane, 0x7dff +
// 0x3ff < 0x10000); m = (s >> 15) * 0xffff is that lane's mask; the result's masked lanes take X | 0x02000200 (the quieted
// lanes). src1 first, then src0, gives src0 priority as the selects do. CPU check over every 32-bit word and 84 M random
// triples (verify_snan_swar.py): 0 differences. FSR3 accumulate: 978 -> 423 SPIR-V operations.

// The HalfPk* operation of an f32 opcode the f16 translations use, or None.
static IR::ValueOpcode HalfOpcode(IR::ValueOpcode opcode) {
	switch (opcode) {
		case IR::ValueOpcode::FPAdd32: return IR::ValueOpcode::HalfPkAdd;
		case IR::ValueOpcode::FPMul32: return IR::ValueOpcode::HalfPkMul;
		case IR::ValueOpcode::FPFma32: return IR::ValueOpcode::HalfPkFma;
		case IR::ValueOpcode::FPMin32: return IR::ValueOpcode::HalfPkMin;
		case IR::ValueOpcode::FPMax32: return IR::ValueOpcode::HalfPkMax;
		case IR::ValueOpcode::FPMinTri32: return IR::ValueOpcode::HalfPkMinTri;
		case IR::ValueOpcode::FPMaxTri32: return IR::ValueOpcode::HalfPkMaxTri;
		case IR::ValueOpcode::FPMedTri32: return IR::ValueOpcode::HalfPkMedTri;
		default: return IR::ValueOpcode::Void;
	}
}

void Translator::PackedFloat16(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool accumulator, bool quiet_snan) {
	if (const auto half_opcode = HalfOpcode(opcode); half_opcode != IR::ValueOpcode::Void && inst.dst.omod == 0u) {
		const auto packed = [&](const Decoder::Operand& operand) {
			// A register operand read whole (low lane from bits 0-15, high from 16-31, no SDWA, no inline float) is the
			// register with abs as AND 0x7fff7fff and neg/neg_hi as XOR of the sign bits: the same bits as extracting both
			// lanes, applying the modifiers to each and packing them again (FSR3 2.89 -> 2.50 ms on the GPU).
			if (operand.kind != Decoder::OperandKind::FloatInlineConstant && operand.sdwa_sel == 6u && !operand.op_sel &&
			    operand.op_sel_hi) {
				auto raw_operand      = operand;
				raw_operand.sdwa_sext = false;
				auto bits             = ApplyBitSourceModifiers(raw_operand, ReadRawU32(operand));
				if (operand.absolute) bits = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x7fff7fffu)));
				const uint32_t sign = (operand.negate ? 0x8000u : 0u) | (operand.negate_hi ? 0x80000000u : 0u);
				if (sign != 0u) bits = ir.BitwiseXor(bits, IR::U32(IR::Value(sign)));
				return bits;
			}
			return PackU16Lanes(Read16LaneBits(operand, false), Read16LaneBits(operand, true));
		};
		const auto lhs = packed(inst.src0);
		const auto rhs = packed(inst.src1);
		IR::U32    native;
		if (accumulator) {
			native = IR::U32(ir.Emit(half_opcode, {lhs, rhs, packed(inst.dst)}));
		} else if (inst.src_count == 3u) {
			native = IR::U32(ir.Emit(half_opcode, {lhs, rhs, packed(inst.src2)}));
		} else {
			native = IR::U32(ir.Emit(half_opcode, {lhs, rhs}));
		}
		if (inst.dst.clamp) {
			native = IR::U32(ir.Emit(IR::ValueOpcode::HalfPkSaturate, {native}));
		}
		auto raw  = DestinationOperand(inst);
		raw.omod  = 0u;
		raw.clamp = false;
		if (quiet_snan) {
			// As below, on the packed raw lanes of each operand: a signaling NaN operand gives itself quieted.
			const auto raw_lanes = [&](const Decoder::Operand& operand) {
				if (operand.sdwa_sel == 6u && !operand.op_sel && operand.op_sel_hi &&
				    operand.kind != Decoder::OperandKind::FloatInlineConstant) {
					auto raw_operand      = operand;
					raw_operand.sdwa_sext = false;
					return ApplyBitSourceModifiers(raw_operand, ReadRawU32(operand));
				}
				return PackU16Lanes(Read16LaneBits(operand, false), Read16LaneBits(operand, true));
			};
			const auto take_quiet = [&](IR::U32 result, const Decoder::Operand& operand) {
				if (operand.kind == Decoder::OperandKind::FloatInlineConstant) return result; // never a NaN
				const auto x = raw_lanes(operand);
				const auto s = ir.BitwiseAnd(ir.IAdd(ir.BitwiseAnd(x, IR::U32(IR::Value(0x7dff7dffu))),
				                                     IR::U32(IR::Value(0x03ff03ffu))),
				                             IR::U32(IR::Value(0x80008000u)));
				const auto m = ir.IMul(ir.ShiftRightLogical(s, IR::U32(IR::Value(15u))), IR::U32(IR::Value(0xffffu)));
				const auto q = ir.BitwiseOr(x, IR::U32(IR::Value(0x02000200u)));
				return ir.BitwiseXor(result, ir.BitwiseAnd(ir.BitwiseXor(result, q), m));
			};
			native = take_quiet(take_quiet(native, inst.src1), inst.src0);
		}
		WriteOperand(raw, native);
		return;
	}
	const auto translate_lane = [&](bool high) {
		const auto lhs = ReadF16LaneAsF32(inst.src0, high);
		const auto rhs = ReadF16LaneAsF32(inst.src1, high);
		IR::F32    result;
		if (accumulator) {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs, ReadF16LaneAsF32(inst.dst, high)}));
		} else if (inst.src_count == 3u) {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs, ReadF16LaneAsF32(inst.src2, high)}));
		} else {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs}));
		}
		return ApplyF32ResultModifiers(inst.dst, result);
	};
	auto raw    = DestinationOperand(inst);
	raw.omod    = 0u;
	raw.clamp   = false;
	auto result = PackHalf2x16(translate_lane(false), translate_lane(true));
	if (quiet_snan) {
		const auto quiet_snan_lane = [&](const Decoder::Operand& operand, bool high) {
			const auto bits     = ReadU16LaneAsU32(operand, high, false);
			const auto exponent = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x7e00u)));
			const auto payload  = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x01ffu)));
			const auto snan     = ir.LogicalAnd(ir.IEqual(exponent, IR::U32(IR::Value(0x7c00u))),
			                                    ir.INotEqual(payload, IR::U32(IR::Value(0u))));
			return std::pair {snan, ir.BitwiseOr(bits, IR::U32(IR::Value(0x0200u)))};
		};
		const auto override_lane = [&](bool high) {
			const auto [lhs_snan, lhs_quiet] = quiet_snan_lane(inst.src0, high);
			const auto [rhs_snan, rhs_quiet] = quiet_snan_lane(inst.src1, high);
			const auto normal = high ? ir.ShiftRightLogical(result, IR::U32(IR::Value(16u)))
			                         : ir.BitwiseAnd(result, IR::U32(IR::Value(0xffffu)));
			return ir.Select(lhs_snan, lhs_quiet, ir.Select(rhs_snan, rhs_quiet, normal));
		};
		result = PackU16Lanes(override_lane(false), override_lane(true));
	}
	WriteOperand(raw, result);
}

void Translator::Float16Unary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool invalid_negative) {
	const auto argument = ReadF16AsF32(inst.src0);
	auto       result   = IR::F32(ir.Emit(opcode, {argument}));
	if (!invalid_negative) {
		WriteF16(DestinationOperand(inst), result);
		return;
	}
	const auto negative =
	    IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {argument, IR::Value::F32(0.0f)}));
	result             = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(negative, invalid, bits));
}

void Translator::Float16Trig(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto argument = ReadF16AsF32(inst.src0);
	const auto magnitude =
	    ir.BitwiseAnd(ir.BitCastU32(argument), IR::U32(IR::Value(0x7fffffffu)));
	auto result = IR::F32(ir.Emit(opcode, {argument}));
	if (opcode == IR::ValueOpcode::FPSin) {
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto whole = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.0f)}));
		const auto half = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.5f)}));
		const auto nonzero = ir.INotEqual(magnitude, IR::U32(IR::Value(0u)));
		const auto cardinal = ir.LogicalOr(ir.LogicalAnd(whole, nonzero), half);
		result = IR::F32(ir.Emit(
		    IR::ValueOpcode::SelectF32, {cardinal, IR::Value::F32(0.0f), result}));
	} else {
		// The architectural quarter-cycle result is exactly zero. Evaluating cos(pi/2) with a
		// rounded F32 PI can otherwise produce a half-precision subnormal instead.
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto quarter = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.25f)}));
		const auto three_quarters = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.75f)}));
		result = IR::F32(
		    ir.Emit(IR::ValueOpcode::SelectF32,
		            {ir.LogicalOr(quarter, three_quarters), IR::Value::F32(0.0f), result}));
	}

	const auto infinite = ir.IEqual(magnitude, IR::U32(IR::Value(0x7f800000u)));
	result              = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(infinite, invalid, bits));
}

void Translator::Float16Binary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool reverse) {
	// The low lane of a HalfPk* operation; a - b as a + (-b) (IEEE subtraction).
	const bool subtract = opcode == IR::ValueOpcode::FPSub32;
	if (const auto half_opcode = HalfOpcode(subtract ? IR::ValueOpcode::FPAdd32 : opcode);
	    half_opcode != IR::ValueOpcode::Void && inst.dst.omod == 0u) {
		const auto lhs = Read16LaneBits(reverse ? inst.src1 : inst.src0, false);
		auto       rhs = Read16LaneBits(reverse ? inst.src0 : inst.src1, false);
		if (subtract) {
			rhs = ir.BitwiseXor(rhs, IR::U32(IR::Value(0x8000u)));
		}
		auto result = IR::U32(ir.Emit(half_opcode, {lhs, rhs}));
		if (inst.dst.clamp) {
			result = IR::U32(ir.Emit(IR::ValueOpcode::HalfPkSaturate, {result}));
		}
		Write16Bits(DestinationOperand(inst), result);
		return;
	}
	const auto lhs = ReadF16AsF32(reverse ? inst.src1 : inst.src0);
	const auto rhs = ReadF16AsF32(reverse ? inst.src0 : inst.src1);
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {lhs, rhs})));
}

void Translator::Float16Ternary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                bool accumulator, bool mix) {
	// The low lane of a HalfPk* operation (not the mixed-precision forms).
	if (const auto half_opcode = HalfOpcode(opcode);
	    !mix && half_opcode != IR::ValueOpcode::Void && inst.dst.omod == 0u) {
		std::array<IR::Value, 3> bits;
		for (uint32_t index = 0; index < bits.size(); index++) {
			bits[index] = Read16LaneBits(accumulator && index == 2u ? inst.dst : SourceAt(inst, index), false);
		}
		auto result = IR::U32(ir.Emit(half_opcode, {bits[0], bits[1], bits[2]}));
		if (inst.dst.clamp) {
			result = IR::U32(ir.Emit(IR::ValueOpcode::HalfPkSaturate, {result}));
		}
		Write16Bits(DestinationOperand(inst), result);
		return;
	}
	// Mixed precision: the f16 operands and the f16 result of the f32 FMA converted without
	// the NaN payload encodings (HalfToF32Arith / F32ToHalfArith), as WriteF16 does otherwise.
	const bool arith_conversions = mix && opcode == IR::ValueOpcode::FPFma32;
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		if (arith_conversions && operand.op_sel_hi) {
			args[index] = ir.Emit(IR::ValueOpcode::HalfToF32Arith, {Read16LaneBits(operand, false)});
			continue;
		}
		args[index] = mix ? IR::Value(ReadMixF32(operand)) : IR::Value(ReadF16AsF32(operand));
	}
	if (arith_conversions) {
		const auto destination = DestinationOperand(inst);
		const auto value =
		    ApplyF32ResultModifiers(destination, IR::F32(ir.Emit(opcode, {args[0], args[1], args[2]})));
		Write16Bits(destination, IR::U32(ir.Emit(IR::ValueOpcode::F32ToHalfArith, {value})));
		return;
	}
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {args[0], args[1], args[2]})));
}

void Translator::FloatUnary(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto type = IR::ArgTypeOf(opcode, 0);
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {ReadOperand(inst.src0, type)}));
}

void Translator::V_FRACT(const Decoder::Instruction& inst, IR::Type type) {
	// x - floor(x) rounds to 1.0 for a tiny negative x; v_fract returns the largest value below
	// one instead. The ordered compare lets NaN through.
	const auto destination = DestinationOperand(inst);
	if (type == IR::Type::F64) {
		const auto result    = ir.Emit(IR::ValueOpcode::FPFract64, {ReadOperand(inst.src0, type)});
		const auto below_one = ir.Emit(IR::ValueOpcode::BitCastF64U64,
		                               {IR::Value(uint64_t {0x3fefffffffffffff})});
		const auto above     = ir.Emit(IR::ValueOpcode::FPOrdGreaterThan64, {result, below_one});
		WriteOperand(destination, ir.Emit(IR::ValueOpcode::SelectF64, {above, below_one, result}));
		return;
	}
	const bool half   = type == IR::Type::F16;
	const auto source = half ? ReadF16AsF32(inst.src0) : IR::F32(ReadOperand(inst.src0, type));
	const auto result = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {source}));
	// Clamp in the destination precision: 0x3f7fe000 is the half 0x3bff.
	const auto below_one =
	    IR::F32(IR::Value::F32(std::bit_cast<float>(half ? 0x3f7fe000u : 0x3f7fffffu)));
	const auto above   = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThan32, {result, below_one}));
	const auto clamped = SelectF32(above, below_one, result);
	if (half) {
		WriteF16(destination, clamped);
	} else {
		WriteOperand(destination, clamped);
	}
}

void Translator::FloatBinary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                             bool reverse) {
	std::array<IR::Value, 2> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = SourceAt(inst, reverse ? 1u - index : index);
		args[index]        = ReadOperand(operand, IR::ArgTypeOf(opcode, index));
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1]}));
}

void Translator::FloatTernary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool accumulator, bool mix) {
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		const auto type    = IR::ArgTypeOf(opcode, index);
		args[index]        = type == IR::Type::F32 && mix ? IR::Value(ReadMixF32(operand))
		                                                  : ReadOperand(operand, type);
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1], args[2]}));
}

void Translator::V_FREXP_MANT_F32(const Decoder::Instruction& inst) {
	const auto bits = ReadU32(inst.src0);
	const auto exponent =
	    IR::U32(ir.Emit(IR::ValueOpcode::BitFieldUExtract, {bits, IR::Value(23u), IR::Value(8u)}));
	const auto mantissa = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x007fffffu)));
	const auto sign     = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x80000000u)));
	const auto base     = ir.BitwiseOr(sign, IR::U32(IR::Value(0x3f000000u)));
	const auto normal   = ir.BitwiseOr(base, mantissa);
	const auto msb      = IR::U32(ir.Emit(IR::ValueOpcode::FindUMsb32, {mantissa}));
	const auto shift    = ir.ISub(IR::U32(IR::Value(23u)), msb);
	const auto fraction =
	    ir.BitwiseAnd(ir.ShiftLeftLogical(mantissa, shift), IR::U32(IR::Value(0x007fffffu)));
	const auto subnormal = ir.BitwiseOr(base, fraction);
	const auto zero      = ir.IEqual(mantissa, IR::U32(IR::Value(0u)));
	const auto finite    = ir.Select(ir.INotEqual(exponent, IR::U32(IR::Value(0u))), normal,
	                                 ir.Select(zero, bits, subnormal));
	const auto result    = ir.Select(ir.IEqual(exponent, IR::U32(IR::Value(0xffu))), bits, finite);
	WriteOperand(DestinationOperand(inst), ir.BitCastF32(result));
}

void Translator::V_DOT2C_F32_F16(const Decoder::Instruction& inst) {
	auto a          = inst.src0;
	a.op_sel        = false;
	a.op_sel_hi     = true;
	auto b          = inst.src1;
	b.op_sel        = false;
	b.op_sel_hi     = true;
	const auto a_lo = ReadF16LaneAsF32(a, false);
	const auto a_hi = ReadF16LaneAsF32(a, true);
	const auto b_lo = ReadF16LaneAsF32(b, false);
	const auto b_hi = ReadF16LaneAsF32(b, true);
	const auto acc  = ReadMixF32(inst.dst);
	const auto lo   = IR::F32(ir.Emit(IR::ValueOpcode::FPFma32, {a_lo, b_lo, acc}));
	WriteOperand(DestinationOperand(inst), ir.Emit(IR::ValueOpcode::FPFma32, {a_hi, b_hi, lo}));
}

void Translator::V_CUBEID_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 0u);
}

void Translator::V_CUBESC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 1u);
}

void Translator::V_CUBETC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 2u);
}

void Translator::V_CUBEMA_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 3u);
}

void Translator::FloatCube(const Decoder::Instruction& inst, uint32_t result_kind) {
	const auto x  = ReadMixF32(inst.src0);
	const auto y  = ReadMixF32(inst.src1);
	const auto z  = ReadMixF32(inst.src2);
	const auto nx = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {x}));
	const auto ny = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {y}));
	const auto nz = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {z}));
	const auto ax = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {x}));
	const auto ay = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {y}));
	const auto az = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {z}));
	const auto z_face =
	    ir.LogicalAnd(IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ax})),
	                  IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ay})));
	const auto y_face = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {ay, ax}));
	const auto x_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {x, IR::Value::F32(0.0f)}));
	const auto y_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {y, IR::Value::F32(0.0f)}));
	const auto z_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {z, IR::Value::F32(0.0f)}));
	const auto select_face = [&](IR::F32 x_value, IR::F32 y_value, IR::F32 z_value) {
		return SelectF32(z_face, z_value, SelectF32(y_face, y_value, x_value));
	};
	IR::F32 result;
	switch (result_kind) {
		case 0u:
			result = select_face(
			    SelectF32(x_neg, IR::F32(IR::Value::F32(1.0f)), IR::F32(IR::Value::F32(0.0f))),
			    SelectF32(y_neg, IR::F32(IR::Value::F32(3.0f)), IR::F32(IR::Value::F32(2.0f))),
			    SelectF32(z_neg, IR::F32(IR::Value::F32(5.0f)), IR::F32(IR::Value::F32(4.0f))));
			break;
		case 1u: result = select_face(SelectF32(x_neg, z, nz), x, SelectF32(z_neg, nx, x)); break;
		case 2u: result = select_face(ny, SelectF32(y_neg, nz, z), ny); break;
		case 3u:
			result = IR::F32(
			    ir.Emit(IR::ValueOpcode::FPMul32, {select_face(x, y, z), IR::Value::F32(2.0f)}));
			break;
		default: EXIT("invalid cube result kind");
	}
	WriteOperand(DestinationOperand(inst), result);
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
