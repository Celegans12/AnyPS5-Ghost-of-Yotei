#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <array>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t EmitExt(EmitterState& state, uint32_t type, uint32_t opcode,
                 std::initializer_list<uint32_t> args) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, type, result, GlslStd450(state), opcode, args);
	return result;
}

uint32_t EmitMinMaxF64(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value) {
	if (state.profile.relaxed_float) {
		return EmitExt(state, TypeF64(state), max_value ? GLSLstd450NMax : GLSLstd450NMin,
		               {lhs, rhs});
	}
	const auto bits_type = TypeU32Vector(state, 2);
	const auto lhs_bits = Unary(state, spv::OpBitcast, bits_type, lhs);
	const auto rhs_bits = Unary(state, spv::OpBitcast, bits_type, rhs);
	const auto ordered = Binary(state, max_value ? spv::OpFOrdGreaterThan : spv::OpFOrdLessThan,
	                            TypeBool(state), lhs, rhs);
	auto result = Select(state, TypeF64(state), ordered, lhs, rhs);

	const auto equal = Binary(state, spv::OpFOrdEqual, TypeBool(state), lhs, rhs);
	const auto equal_bits = Binary(state, max_value ? spv::OpBitwiseAnd : spv::OpBitwiseOr,
	                               bits_type, lhs_bits, rhs_bits);
	result = Select(state, TypeF64(state), equal,
	                Unary(state, spv::OpBitcast, TypeF64(state), equal_bits), result);

	return Select(state, TypeF64(state), Unary(state, spv::OpIsNan, TypeBool(state), rhs),
	              lhs, result);
}

uint32_t EmitMulHigh(EmitterState& state, uint32_t lhs, uint32_t rhs, bool signed_value) {
	const auto operand_type = signed_value ? TypeI32(state) : TypeU32(state);
	const auto pair_type    = signed_value ? TypeI32Pair(state) : TypeU32Pair(state);
	uint32_t   lhs_operand  = lhs;
	uint32_t   rhs_operand  = rhs;
	if (signed_value) {
		lhs_operand = Unary(state, spv::OpBitcast, TypeI32(state), lhs);
		rhs_operand = Unary(state, spv::OpBitcast, TypeI32(state), rhs);
	}
	const auto extended = state.builder.AllocateId();
	state.builder.AddFunction(signed_value ? spv::OpSMulExtended : spv::OpUMulExtended, pair_type,
	                          extended, lhs_operand, rhs_operand);
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, operand_type, high, extended, 1);
	return signed_value ? Unary(state, spv::OpBitcast, TypeU32(state), high) : high;
}

uint32_t EmitFMinMax3(EmitterState& state, uint32_t a, uint32_t b, uint32_t c, bool max_value) {
	return EmitMinMaxF32Value(state, EmitMinMaxF32Value(state, a, b, max_value), c, max_value);
}

uint32_t EmitF32ToU32(EmitterState& state, uint32_t src, bool signed_value) {
	const auto clamped =
	    EmitExt(state, TypeF32(state), GLSLstd450NClamp,
	            {src, ConstantF32(state, signed_value ? 0xcf000000u : 0u),
	             ConstantF32(state, signed_value ? 0x4effffffu : 0x4f7fffffu)});
	const auto converted = signed_value
	                           ? Unary(state, spv::OpBitcast, TypeU32(state),
	                                   Unary(state, spv::OpConvertFToS, TypeI32(state), clamped))
	                           : Unary(state, spv::OpConvertFToU, TypeU32(state), clamped);
	const auto above     = Binary(state, spv::OpFOrdGreaterThanEqual, TypeBool(state), src,
	                              ConstantF32(state, signed_value ? 0x4f000000u : 0x4f800000u));
	const auto high =
	    Select(state, TypeU32(state), above,
	           ConstantU32(state, signed_value ? 0x7fffffffu : 0xffffffffu), converted);
	return Select(state, TypeU32(state), Unary(state, spv::OpIsNan, TypeBool(state), src),
	              ConstantU32(state, 0), high);
}

}

uint32_t EmitF32BitsToF16RtzBits(EmitterState& state, uint32_t bits) {
	const auto abs  = EmitAndConstant(state, bits, 0x7fffffffu);
	const auto sign = EmitAndConstant(
	    state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits, ConstantU32(state, 16)),
	    0x8000u);
	const auto exp = Binary(state, spv::OpShiftRightLogical, TypeU32(state), abs,
	                        ConstantU32(state, 23));
	const auto mant = EmitAndConstant(state, abs, 0x007fffffu);

	const auto shift = EmitUMin32(
	    state, Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 126), exp),
	    ConstantU32(state, 31));
	const auto sub = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                        EmitOrU32(state, mant, ConstantU32(state, 0x00800000u)), shift);

	const auto truncated = Binary(state, spv::OpShiftRightLogical, TypeU32(state), abs,
	                              ConstantU32(state, 13));
	const auto normal = EmitUMin32(
	    state, Binary(state, spv::OpISub, TypeU32(state), truncated, ConstantU32(state, 0x1c000u)),
	    ConstantU32(state, 0x7bffu));
	const auto finite = Select(state, TypeU32(state),
	                           EmitCompareU32Constant(state, spv::OpULessThan, abs, 0x38800000u),
	                           sub, normal);
	const auto nan = EmitOrU32(state, EmitAndConstant(state, truncated, 0x03ffu),
	                           ConstantU32(state, 0x7e00u));
	const auto special = Select(state, TypeU32(state),
	                            EmitCompareU32Constant(state, spv::OpIEqual, mant, 0),
	                            ConstantU32(state, 0x7c00u), nan);
	const auto magnitude = Select(
	    state, TypeU32(state),
	    EmitCompareU32Constant(state, spv::OpUGreaterThanEqual, abs, 0x7f800000u), special, finite);
	return EmitOrU32(state, sign, magnitude);
}

uint32_t EmitFPFma32(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	const auto result = EmitGlsl<GLSLstd450Fma, IR::Type::F32>(state, a, b, c);
	state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	return result;
}

uint32_t EmitFPMad32(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	const auto product = EmitFPMul32(state, a, b);
	state.builder.AddAnnotation(spv::OpDecorate, product, spv::DecorationNoContraction);
	const auto sum = EmitFPAdd32(state, product, c);
	state.builder.AddAnnotation(spv::OpDecorate, sum, spv::DecorationNoContraction);
	return sum;
}

uint32_t EmitConvertU16U32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpBitwiseAnd, IR::Type::U16>(state, arg0, ConstantU32(state, 0xffffu));
}

uint32_t EmitConvertU8U32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpBitwiseAnd, IR::Type::U8>(state, arg0, ConstantU32(state, 0xffu));
}

uint32_t EmitF32ToF16Bits(EmitterState& state, uint32_t arg0) {
	if (!state.profile.native_float16) {
		const auto bits = Unary(state, spv::OpBitcast, TypeU32(state), arg0);
		const auto magnitude = EmitAndConstant(state, bits, 0x7fffffffu);
		const auto sign = EmitAndConstant(
		    state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits, ConstantU32(state, 16)), 0x8000u);
		const auto exponent = Binary(state, spv::OpShiftRightLogical, TypeU32(state), magnitude, ConstantU32(state, 23));
		const auto significand = EmitOrU32(state, EmitAndConstant(state, magnitude, 0x7fffffu), ConstantU32(state, 0x800000u));
		const auto subnormal = EmitCompareU32Constant(state, spv::OpULessThan, exponent, 113);
		const auto shift = Select(
		    state, TypeU32(state), subnormal,
		    EmitUMin32(state, Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 126), exponent), ConstantU32(state, 24)),
		    ConstantU32(state, 13));
		const auto halfway = Binary(
		    state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
		    Binary(state, spv::OpISub, TypeU32(state), shift, ConstantU32(state, 1)));
		const auto truncated = Binary(state, spv::OpShiftRightLogical, TypeU32(state), significand, shift);
		const auto bias = Binary(
		    state, spv::OpIAdd, TypeU32(state),
		    Binary(state, spv::OpISub, TypeU32(state), halfway, ConstantU32(state, 1)), EmitAndConstant(state, truncated, 1));
		const auto rounded = Binary(
		    state, spv::OpShiftRightLogical, TypeU32(state),
		    Binary(state, spv::OpIAdd, TypeU32(state), significand, bias), shift);
		const auto normal = Binary(
		    state, spv::OpIAdd, TypeU32(state), rounded,
		    Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
		           Binary(state, spv::OpISub, TypeU32(state), exponent, ConstantU32(state, 113)), ConstantU32(state, 10)));
		auto result = Select(state, TypeU32(state), subnormal, rounded, normal);
		result = Select(state, TypeU32(state),
		                EmitCompareU32Constant(state, spv::OpULessThan, magnitude, 0x33000000u), ConstantU32(state, 0), result);
		result = Select(state, TypeU32(state),
		                EmitCompareU32Constant(state, spv::OpUGreaterThanEqual, magnitude, 0x477ff000u), ConstantU32(state, 0x7c00u), result);
		const auto payload = EmitAndConstant(
		    state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), magnitude, ConstantU32(state, 13)), 0x3ffu);
		result = Select(state, TypeU32(state),
		                EmitCompareU32Constant(state, spv::OpUGreaterThan, magnitude, 0x7f800000u),
		                EmitOrU32(state, payload, ConstantU32(state, 0x7e00u)), result);
		return EmitOrU32(state, sign, result);
	}
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, arg0,
	                          ConstantF32(state, 0));
	return EmitPackHalf2x16(state, pair);
}

uint32_t EmitPackHalf2x16(EmitterState& state, uint32_t arg0) {
	if (state.profile.native_float16) return EmitGlsl<GLSLstd450PackHalf2x16, IR::Type::U32>(state, arg0);
	const auto low = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), low, arg0, 0u);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), high, arg0, 1u);
	const auto low_bits = EmitF32ToF16Bits(state, low);
	const auto high_bits = EmitF32ToF16Bits(state, high);
	return EmitOrU32(state, low_bits,
	                 Binary(state, spv::OpShiftLeftLogical, TypeU32(state), high_bits, ConstantU32(state, 16)));
}

namespace {

uint32_t HalfOfU32(EmitterState& state, uint32_t bits, uint32_t lane) {
	if (!state.profile.native_float16) {
		const auto shifted = lane == 0u ? bits : Binary(
		    state, spv::OpShiftRightLogical, TypeU32(state), bits, ConstantU32(state, 16));
		return EmitAndConstant(state, shifted, 0xffffu);
	}
	const auto halves = Unary(state, spv::OpBitcast, TypeF16x2(state), bits);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF16(state), result, halves, lane);
	return result;
}

uint32_t LaneOf(IR::Value lane) {
	return lane.Resolve().U32() != 0u ? 1u : 0u;
}

uint32_t ExactFloatBinary(EmitterState& state, spv::Op opcode, uint32_t a, uint32_t b) {
	const auto result = Binary(state, opcode, TypeF32(state), a, b);
	state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	return result;
}

uint32_t SoftwareHalfToFloat(EmitterState& state, uint32_t bits) {
	const auto magnitude = EmitAndConstant(state, bits, 0x7fffu);
	const auto sign = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
	                         EmitAndConstant(state, bits, 0x8000u), ConstantU32(state, 16));
	const auto exponent = EmitAndConstant(state, magnitude, 0x7c00u);
	const auto mantissa = EmitAndConstant(state, magnitude, 0x3ffu);
	const auto normal = Binary(
	    state, spv::OpShiftLeftLogical, TypeU32(state),
	    Binary(state, spv::OpIAdd, TypeU32(state), magnitude, ConstantU32(state, 0x1c000u)), ConstantU32(state, 13));
	const auto special = EmitOrU32(
	    state, Binary(state, spv::OpShiftLeftLogical, TypeU32(state), mantissa, ConstantU32(state, 13)), ConstantU32(state, 0x7f800000u));
	const auto subnormal = ExactFloatBinary(state, spv::OpFMul,
	                                       Unary(state, spv::OpConvertUToF, TypeF32(state), mantissa), ConstantF32(state, 0x33800000u));
	const auto large = Select(state, TypeU32(state),
	                          EmitCompareU32Constant(state, spv::OpIEqual, exponent, 0x7c00u), special, normal);
	const auto result = Select(state, TypeU32(state),
	                           EmitCompareU32Constant(state, spv::OpIEqual, exponent, 0),
	                           Unary(state, spv::OpBitcast, TypeU32(state), subnormal), large);
	return Unary(state, spv::OpBitcast, TypeF32(state), EmitOrU32(state, sign, result));
}

uint32_t HalfFma(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	const auto product = ExactFloatBinary(state, spv::OpFMul, a, b);
	const auto sum = ExactFloatBinary(state, spv::OpFAdd, product, c);
	const auto b_virtual = ExactFloatBinary(state, spv::OpFSub, sum, product);
	const auto a_virtual = ExactFloatBinary(state, spv::OpFSub, sum, b_virtual);
	const auto b_roundoff = ExactFloatBinary(state, spv::OpFSub, c, b_virtual);
	const auto a_roundoff = ExactFloatBinary(state, spv::OpFSub, product, a_virtual);
	const auto error = ExactFloatBinary(state, spv::OpFAdd, a_roundoff, b_roundoff);
	const auto bits = Unary(state, spv::OpBitcast, TypeU32(state), sum);
	const auto exponent = EmitAndConstant(
	    state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits, ConstantU32(state, 23)), 0xffu);
	const auto subnormal_shift = Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 126), exponent);
	const auto shift = EmitUMin32(
	    state, Select(state, TypeU32(state),
	                  EmitCompareU32Constant(state, spv::OpULessThan, exponent, 113),
	                  subnormal_shift, ConstantU32(state, 13)), ConstantU32(state, 24));
	const auto halfway = Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ConstantU32(state, 1),
	                            Binary(state, spv::OpISub, TypeU32(state), shift, ConstantU32(state, 1)));
	const auto mask = Binary(state, spv::OpISub, TypeU32(state),
	                         Binary(state, spv::OpShiftLeftLogical, TypeU32(state), halfway, ConstantU32(state, 1)),
	                         ConstantU32(state, 1));
	const auto remainder = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                              EmitOrU32(state, bits, ConstantU32(state, 0x00800000u)), mask);
	const auto in_range = EmitLogicalAndBool(
	    state, EmitCompareU32Constant(state, spv::OpUGreaterThanEqual, exponent, 102),
	    EmitCompareU32Constant(state, spv::OpULessThanEqual, exponent, 142));
	const auto midpoint = EmitLogicalAndBool(
	    state, in_range, Binary(state, spv::OpIEqual, TypeBool(state), remainder, halfway));
	const auto correction = EmitLogicalAndBool(
	    state, midpoint, Binary(state, spv::OpFOrdNotEqual, TypeBool(state), error, ConstantF32(state, 0)));
	const auto error_bits = Unary(state, spv::OpBitcast, TypeU32(state), error);
	const auto same_sign = EmitCompareU32Constant(
	    state, spv::OpIEqual,
	    EmitAndConstant(state, Binary(state, spv::OpBitwiseXor, TypeU32(state), bits, error_bits), 0x80000000u), 0);
	const auto step = Select(state, TypeU32(state), same_sign, ConstantU32(state, 1), ConstantU32(state, 0xffffffffu));
	const auto adjusted = Binary(state, spv::OpIAdd, TypeU32(state), bits, step);
	return Unary(state, spv::OpBitcast, TypeF32(state), Select(state, TypeU32(state), correction, adjusted, bits));
}

uint32_t SoftwareHalfOperation(EmitterState& state, spv::Op opcode, uint32_t glsl,
                                std::initializer_list<uint32_t> args) {
	const auto* arg = args.begin();
	if (opcode == spv::OpFNegate) {
		return Binary(state, spv::OpBitwiseXor, TypeU32(state), arg[0], ConstantU32(state, 0x8000u));
	}
	if (glsl == GLSLstd450FAbs) return EmitAndConstant(state, arg[0], 0x7fffu);
	const auto a = SoftwareHalfToFloat(state, arg[0]);
	const auto b = SoftwareHalfToFloat(state, arg[1]);
	if (glsl == GLSLstd450NMin || glsl == GLSLstd450NMax) {
		const bool maximum = glsl == GLSLstd450NMax;
		const auto ordered = Binary(state, maximum ? spv::OpFOrdGreaterThan : spv::OpFOrdLessThan,
		                            TypeBool(state), a, b);
		auto result = Select(state, TypeU32(state), ordered, arg[0], arg[1]);
		const auto equal = Binary(state, spv::OpFOrdEqual, TypeBool(state), a, b);
		const auto equal_bits = Binary(state, maximum ? spv::OpBitwiseAnd : spv::OpBitwiseOr,
		                               TypeU32(state), arg[0], arg[1]);
		result = Select(state, TypeU32(state), equal, equal_bits, result);
		return Select(state, TypeU32(state), Unary(state, spv::OpIsNan, TypeBool(state), b), arg[0], result);
	}
	const auto value = glsl == GLSLstd450Fma
	                       ? HalfFma(state, a, b, SoftwareHalfToFloat(state, arg[2]))
	                       : ExactFloatBinary(state, opcode, a, b);
	return EmitF32ToF16Bits(state, value);
}

uint32_t HalfOperation(EmitterState& state, spv::Op opcode, uint32_t glsl,
                       std::initializer_list<uint32_t> args, bool exact) {
	if (!state.profile.native_float16) return SoftwareHalfOperation(state, opcode, glsl, args);
	const auto result = state.builder.AllocateId();
	const auto* arg   = args.begin();
	if (opcode != spv::OpExtInst) {
		if (args.size() == 1) {
			state.builder.AddFunction(opcode, TypeF16(state), result, arg[0]);
		} else {
			state.builder.AddFunction(opcode, TypeF16(state), result, arg[0], arg[1]);
		}
	} else if (args.size() == 1) {
		state.builder.AddFunction(spv::OpExtInst, TypeF16(state), result, GlslStd450(state), glsl,
		                          arg[0]);
	} else if (args.size() == 2) {
		state.builder.AddFunction(spv::OpExtInst, TypeF16(state), result, GlslStd450(state), glsl,
		                          arg[0], arg[1]);
	} else {
		state.builder.AddFunction(spv::OpExtInst, TypeF16(state), result, GlslStd450(state), glsl,
		                          arg[0], arg[1], arg[2]);
	}
	if (exact) {
		state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	}
	return result;
}

}

uint32_t EmitConvertF16F32(EmitterState& state, uint32_t arg0) {
	return HalfOfU32(state, EmitF32ToF16Bits(state, arg0), 0u);
}

uint32_t EmitConvertF32F16(EmitterState& state, uint32_t arg0) {
	if (!state.profile.native_float16) return SoftwareHalfToFloat(state, arg0);
	return Unary(state, spv::OpFConvert, TypeF32(state), arg0);
}

uint32_t EmitBitCastF16U16(EmitterState& state, uint32_t arg0) {
	return HalfOfU32(state, arg0, 0u);
}

uint32_t EmitBitCastU16F16(EmitterState& state, uint32_t arg0) {
	if (!state.profile.native_float16) return EmitAndConstant(state, arg0, 0xffffu);
	const auto halves = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF16x2(state), halves, arg0,
	                          state.builder.Constant(spv::OpConstant, TypeF16(state), 0u));
	return Unary(state, spv::OpBitcast, TypeU32(state), halves);
}

uint32_t EmitExtractF16Lane(EmitterState& state, uint32_t bits, IR::Value lane) {
	return HalfOfU32(state, bits, LaneOf(lane));
}

uint32_t EmitInsertF16Lane(EmitterState& state, uint32_t bits, uint32_t value, IR::Value lane) {
	if (!state.profile.native_float16) {
		return EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
		    state, bits, value, ConstantU32(state, LaneOf(lane) * 16u), ConstantU32(state, 16));
	}
	const auto halves   = Unary(state, spv::OpBitcast, TypeF16x2(state), bits);
	const auto inserted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeInsert, TypeF16x2(state), inserted, value, halves,
	                          LaneOf(lane));
	return Unary(state, spv::OpBitcast, TypeU32(state), inserted);
}

uint32_t EmitFPAdd16(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return HalfOperation(state, spv::OpFAdd, 0, {arg0, arg1}, true);
}

uint32_t EmitFPSub16(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return HalfOperation(state, spv::OpFSub, 0, {arg0, arg1}, true);
}

uint32_t EmitFPMul16(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return HalfOperation(state, spv::OpFMul, 0, {arg0, arg1}, true);
}

uint32_t EmitFPFma16(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return HalfOperation(state, spv::OpExtInst, GLSLstd450Fma, {arg0, arg1, arg2}, true);
}

uint32_t EmitFPMin16(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return HalfOperation(state, spv::OpExtInst, GLSLstd450NMin, {arg0, arg1}, false);
}

uint32_t EmitFPMax16(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return HalfOperation(state, spv::OpExtInst, GLSLstd450NMax, {arg0, arg1}, false);
}

uint32_t EmitFPAbs16(EmitterState& state, uint32_t arg0) {
	return HalfOperation(state, spv::OpExtInst, GLSLstd450FAbs, {arg0}, false);
}

uint32_t EmitFPNeg16(EmitterState& state, uint32_t arg0) {
	return HalfOperation(state, spv::OpFNegate, 0, {arg0}, false);
}

uint32_t EmitConvertS32F32(EmitterState& state, uint32_t arg0) {
	return EmitF32ToU32(state, arg0, true);
}

uint32_t EmitConvertU32F32(EmitterState& state, uint32_t arg0) {
	return EmitF32ToU32(state, arg0, false);
}

uint32_t EmitConvertF32F64(EmitterState& state, uint32_t arg0) {
	const auto converted = Unary(state, spv::OpFConvert, TypeF32(state), arg0);
	const auto source    = EmitNative<spv::OpCompositeExtract, IR::Type::U32>(
	    state, Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), arg0), 1u);
	const auto exponent = EmitAndConstant(state, source, 0x7ff00000u);
	const auto overflow =
	    Binary(state, spv::OpLogicalAnd, TypeBool(state),
	           EmitCompareU32Constant(state, spv::OpUGreaterThan, exponent, 0x47e00000u),
	           EmitCompareU32Constant(state, spv::OpULessThan, exponent, 0x7ff00000u));
	const auto clamped = EmitOrU32(state, EmitAndConstant(state, source, 0x80000000u),
	                               ConstantU32(state, 0x7f7fffffu));
	return EmitFlushF32DenormToSignedZero(
	    state, Select(state, TypeF32(state), overflow,
	                  Unary(state, spv::OpBitcast, TypeF32(state), clamped), converted));
}

uint32_t EmitConvertF64F32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpFConvert, IR::Type::F64>(state,
	                                                  EmitFlushF32DenormToSignedZero(state, arg0));
}

uint32_t EmitCompositeConstructU64(ValueEmitContext& ctx, uint32_t arg0, IR::Value arg1) {
	arg1 = arg1.Resolve();
	return arg1.IsImmediate() && arg1.U32() == 0u
	           ? Unary(ctx.state, spv::OpUConvert, TypeU64(ctx.state), arg0)
	           : PackU64(ctx.state, arg0, ctx.Def(arg1));
}

uint32_t EmitCompositeExtractU64(EmitterState& state, uint32_t arg0, IR::Value arg1) {
	const auto value = arg1.U32() == 0u ? arg0
	                                    : Binary(state, spv::OpShiftRightLogical, TypeU64(state),
	                                             arg0, ConstantU32(state, 32));
	return Unary(state, spv::OpUConvert, TypeU32(state), value);
}

uint32_t EmitCompositeExtractU32x2(EmitterState& state, uint32_t arg0, IR::Value arg1) {
	return EmitNative<spv::OpCompositeExtract, IR::Type::U32>(state, arg0, arg1.U32());
}

uint32_t EmitPackFloat2x16Rtz(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
	    state, EmitF32BitsToF16RtzBits(state, EmitBitcastF32ToU32(state, arg0)),
	    EmitF32BitsToF16RtzBits(state, EmitBitcastF32ToU32(state, arg1)),
	    ConstantU32(state, 16), ConstantU32(state, 16));
}

uint32_t EmitFPSaturate32(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450NClamp,
	               {arg0, ConstantF32(state, 0), ConstantF32(state, 0x3f800000u)});
}

uint32_t EmitSMulHi(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMulHigh(state, arg0, arg1, true);
}

uint32_t EmitUMulHi(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMulHigh(state, arg0, arg1, false);
}

uint32_t EmitSMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitSMin32(state, arg0, EmitSMin32(state, arg1, arg2));
}

uint32_t EmitSMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitSMax32(state, arg0, EmitSMax32(state, arg1, arg2));
}

uint32_t EmitUMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitUMin32(state, arg0, EmitUMin32(state, arg1, arg2));
}

uint32_t EmitUMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitUMax32(state, arg0, EmitUMax32(state, arg1, arg2));
}

uint32_t EmitSMedTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	const auto high_min = EmitSMin32(state, EmitSMax32(state, arg0, arg1), arg2);
	return EmitSMax32(state, EmitSMin32(state, arg0, arg1), high_min);
}

uint32_t EmitUMedTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	const auto high_min = EmitUMin32(state, EmitUMax32(state, arg0, arg1), arg2);
	return EmitUMax32(state, EmitUMin32(state, arg0, arg1), high_min);
}

uint32_t EmitFPMin32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF32Value(state, arg0, arg1, false);
}

uint32_t EmitFPMax32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF32Value(state, arg0, arg1, true);
}

namespace {

uint32_t PackedF16Operation(EmitterState& state, spv::Op opcode, uint32_t glsl,
                            std::initializer_list<uint32_t> args) {
	if (!state.profile.native_float16) {
		std::array<uint32_t, 3> low {};
		std::array<uint32_t, 3> high {};
		size_t count = 0;
		for (const auto arg: args) {
			low[count] = HalfOfU32(state, arg, 0);
			high[count++] = HalfOfU32(state, arg, 1);
		}
		const auto operation = [&](const auto& values) {
			return count == 2 ? SoftwareHalfOperation(state, opcode, glsl, {values[0], values[1]})
			                  : SoftwareHalfOperation(state, opcode, glsl, {values[0], values[1], values[2]});
		};
		const auto lo = operation(low);
		const auto hi = operation(high);
		return EmitOrU32(state, lo, Binary(state, spv::OpShiftLeftLogical, TypeU32(state), hi, ConstantU32(state, 16)));
	}
	std::array<uint32_t, 3> halves {};
	size_t                  count = 0;
	for (const auto arg: args) {
		halves[count++] = Unary(state, spv::OpBitcast, TypeF16x2(state), arg);
	}
	const auto result = state.builder.AllocateId();
	if (opcode == spv::OpExtInst) {
		if (count == 2) {
			state.builder.AddFunction(spv::OpExtInst, TypeF16x2(state), result, GlslStd450(state),
			                          glsl, halves[0], halves[1]);
		} else {
			state.builder.AddFunction(spv::OpExtInst, TypeF16x2(state), result, GlslStd450(state),
			                          glsl, halves[0], halves[1], halves[2]);
		}
	} else {
		state.builder.AddFunction(opcode, TypeF16x2(state), result, halves[0], halves[1]);
	}
	state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	return Unary(state, spv::OpBitcast, TypeU32(state), result);
}

}

uint32_t EmitPackedFPAdd16x2(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return PackedF16Operation(state, spv::OpFAdd, 0, {arg0, arg1});
}

uint32_t EmitPackedFPMul16x2(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return PackedF16Operation(state, spv::OpFMul, 0, {arg0, arg1});
}

uint32_t EmitPackedFPFma16x2(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return PackedF16Operation(state, spv::OpExtInst, GLSLstd450Fma, {arg0, arg1, arg2});
}

uint32_t EmitPackedFPMin16x2(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return PackedF16Operation(state, spv::OpExtInst, GLSLstd450NMin, {arg0, arg1});
}

uint32_t EmitPackedFPMax16x2(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return PackedF16Operation(state, spv::OpExtInst, GLSLstd450NMax, {arg0, arg1});
}

uint32_t EmitFPMin64(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF64(state, arg0, arg1, false);
}

uint32_t EmitFPMax64(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF64(state, arg0, arg1, true);
}

uint32_t EmitFPMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitFMinMax3(state, arg0, arg1, arg2, false);
}

uint32_t EmitFPMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitFMinMax3(state, arg0, arg1, arg2, true);
}

uint32_t EmitFPRecip32(EmitterState& state, uint32_t arg0) {
	const auto source = EmitFlushF32DenormToSignedZero(state, arg0);
	return Binary(state, spv::OpFDiv, TypeF32(state), ConstantF32(state, 0x3f800000u), source);
}

uint32_t EmitFPRecip64(EmitterState& state, uint32_t arg0) {
	const auto one = state.builder.Constant(spv::OpConstant, TypeF64(state), 0u, 0x3ff00000u);
	return EmitNative<spv::OpFDiv, IR::Type::F64>(state, one, arg0);
}

uint32_t EmitFPRecipIFlag32(EmitterState& state, uint32_t arg0) {
	return Binary(state, spv::OpFDiv, TypeF32(state), ConstantF32(state, 0x3f800000u), arg0);
}

uint32_t EmitFPRecipSqrt32(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450InverseSqrt,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPSqrt(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Sqrt,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPExp2(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Exp2, {arg0});
}

uint32_t EmitFPLog2(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Log2,
	               {EmitFlushF32DenormToSignedZero(state, arg0)});
}

uint32_t EmitFPLdexp(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	const auto exponent = Unary(state, spv::OpBitcast, TypeI32(state), arg1);
	return EmitExt(state, TypeF32(state), GLSLstd450Ldexp, {arg0, exponent});
}

uint32_t EmitFPSin(EmitterState& state, uint32_t arg0) {
	auto source = EmitTrigCycleF32(state, arg0, !state.profile.relaxed_float);
	source = Binary(state, spv::OpFMul, TypeF32(state), source, ConstantF32(state, 0x40c90fdbu));
	return EmitExt(state, TypeF32(state), GLSLstd450Sin, {source});
}

uint32_t EmitFPCos(EmitterState& state, uint32_t arg0) {
	auto source = EmitTrigCycleF32(state, arg0, false);
	source = Binary(state, spv::OpFMul, TypeF32(state), source, ConstantF32(state, 0x40c90fdbu));
	return EmitExt(state, TypeF32(state), GLSLstd450Cos, {source});
}

}
