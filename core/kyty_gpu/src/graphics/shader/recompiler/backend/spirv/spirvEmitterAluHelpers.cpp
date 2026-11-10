#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "common/shaderEnv.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t EmitAndConstant(EmitterState& state, uint32_t value, uint32_t mask) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), ret, value,
	                          ConstantU32(state, mask));
	return ret;
}

uint32_t EmitCompareU32Constant(EmitterState& state, spv::Op opcode, uint32_t value,
                                uint32_t constant) {
	const auto ret = state.builder.AllocateId();
	state.builder.AddFunction(opcode, TypeBool(state), ret, value, ConstantU32(state, constant));
	return ret;
}

uint32_t EmitF32ToF16RtzBits(EmitterState& state, uint32_t value) {
	// Compact integer RTZ conversion (upstream 9f88043eb); bit-identical to the former one (same cases: subnormal shift
	// capped at 31, RTZ overflow to max half, NaN keeps the top 10 payload bits with the quiet bit).
	const auto bits = EmitBitcastF32ToU32(state, value);
	const auto abs  = EmitAndConstant(state, bits, 0x7fffffffu);
	const auto sign = EmitAndConstant(
	    state, Binary(state, spv::OpShiftRightLogical, TypeU32(state), bits, ConstantU32(state, 16)),
	    0x8000u);
	const auto exp = Binary(state, spv::OpShiftRightLogical, TypeU32(state), abs,
	                        ConstantU32(state, 23));
	const auto mant = EmitAndConstant(state, abs, 0x007fffffu);

	// Align the hidden bit to the half subnormal mantissa. Even unused shifts must be < 32.
	const auto shift = EmitUMin32(
	    state, Binary(state, spv::OpISub, TypeU32(state), ConstantU32(state, 126), exp),
	    ConstantU32(state, 31));
	const auto sub = Binary(state, spv::OpShiftRightLogical, TypeU32(state),
	                        EmitOrU32(state, mant, ConstantU32(state, 0x00800000u)), shift);

	// Drop 13 mantissa bits and rebias the exponent; finite overflow saturates to max half.
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

F32Class EmitClassifyF32Bits(EmitterState& state, uint32_t bits) {
	const auto abs_bits = EmitAndConstant(state, bits, 0x7fffffffu);
	return {EmitCompareU32Constant(state, spv::OpUGreaterThan, abs_bits, 0x7f800000u),
	        EmitCompareU32Constant(state, spv::OpIEqual, abs_bits, 0)};
}

uint32_t EmitClassMaskBitMatch(EmitterState& state, uint32_t mask, uint32_t bit,
                               uint32_t class_match) {
	// A constant mask (the common literal V_CMP_CLASS form) selects classes at compile time.
	if (uint32_t constant = 0; state.builder.TryScalarConstant(mask, constant))
		return (constant & (1u << bit)) != 0u ? class_match : ConstantBool(state, false);
	const auto selected =
	    EmitCompareU32Constant(state, spv::OpINotEqual, EmitAndConstant(state, mask, 1u << bit), 0);
	return EmitLogicalAndBool(state, selected, class_match);
}

static uint32_t EmitClassMaskBits(EmitterState& state, uint32_t bits, uint32_t mask, bool half) {
	const uint32_t sign_mask     = half ? 0x8000u : 0x80000000u;
	const uint32_t exponent_mask = half ? 0x7c00u : 0x7f800000u;
	const uint32_t mantissa_mask = half ? 0x03ffu : 0x007fffffu;
	const uint32_t quiet_mask    = half ? 0x0200u : 0x00400000u;
	const auto     sign_bits     = EmitAndConstant(state, bits, sign_mask);
	const auto     abs_bits      = EmitAndConstant(state, bits, sign_mask - 1u);
	const auto     exponent_bits = EmitAndConstant(state, abs_bits, exponent_mask);
	const auto     mantissa_bits = EmitAndConstant(state, abs_bits, mantissa_mask);
	const auto     quiet_bits    = EmitAndConstant(state, mantissa_bits, quiet_mask);

	const auto sign             = EmitCompareU32Constant(state, spv::OpINotEqual, sign_bits, 0);
	const auto positive         = EmitLogicalNotBool(state, sign);
	const auto exponent_zero    = EmitCompareU32Constant(state, spv::OpIEqual, exponent_bits, 0);
	const auto exponent_nonzero = EmitLogicalNotBool(state, exponent_zero);
	const auto exponent_inf =
	    EmitCompareU32Constant(state, spv::OpIEqual, exponent_bits, exponent_mask);
	const auto finite_exponent  = EmitLogicalNotBool(state, exponent_inf);
	const auto mantissa_zero    = EmitCompareU32Constant(state, spv::OpIEqual, mantissa_bits, 0);
	const auto mantissa_nonzero = EmitLogicalNotBool(state, mantissa_zero);
	const auto quiet            = EmitCompareU32Constant(state, spv::OpINotEqual, quiet_bits, 0);

	const auto nan_common = EmitLogicalAndBool(state, exponent_inf, mantissa_nonzero);
	const auto snan       = EmitLogicalAndBool(state, nan_common, EmitLogicalNotBool(state, quiet));
	const auto qnan       = EmitLogicalAndBool(state, nan_common, quiet);
	const auto inf        = EmitLogicalAndBool(state, exponent_inf, mantissa_zero);
	const auto normal     = EmitLogicalAndBool(state, exponent_nonzero, finite_exponent);
	const auto denorm     = EmitLogicalAndBool(state, exponent_zero, mantissa_nonzero);
	const auto zero       = EmitCompareU32Constant(state, spv::OpIEqual, abs_bits, 0);

	// Terms whose mask bit is a constant zero are dropped instead of OR-ing a false.
	const std::array<uint32_t, 10> classes {
	    snan, qnan, EmitLogicalAndBool(state, inf, sign), EmitLogicalAndBool(state, normal, sign),
	    EmitLogicalAndBool(state, denorm, sign), EmitLogicalAndBool(state, zero, sign),
	    EmitLogicalAndBool(state, zero, positive), EmitLogicalAndBool(state, denorm, positive),
	    EmitLogicalAndBool(state, normal, positive), EmitLogicalAndBool(state, inf, positive)};
	uint32_t constant_mask = 0;
	const bool constant = state.builder.TryScalarConstant(mask, constant_mask);
	uint32_t match = 0;
	for (uint32_t bit = 0; bit < classes.size(); ++bit) {
		if (constant && (constant_mask & (1u << bit)) == 0u) continue;
		const auto term = EmitClassMaskBitMatch(state, mask, bit, classes[bit]);
		match = match == 0u ? term : EmitLogicalOrBool(state, match, term);
	}
	return match == 0u ? ConstantBool(state, false) : match;
}

uint32_t EmitClassMaskF32(EmitterState& state, uint32_t value, uint32_t mask) {
	return EmitClassMaskBits(state, EmitBitcastF32ToU32(state, value), mask, false);
}

uint32_t EmitClassMaskF16(EmitterState& state, uint32_t bits, uint32_t mask) {
	return EmitClassMaskBits(state, bits, mask, true);
}

// V_MIN/MAX(3)/MED3_F32 as GLSL.std.450 NMin/NMax under the modules' f32 controls (none declared: the host's default
// denormal flush). Checked on the RTX 5070 Ti against the emulation it replaced (ordered compare, two zeros by bits, a NaN
// source gives the other source; test shader f32_nmin.comp, every pair of top-16-bit patterns with 4 low-bit
// variants, 4 x 2^32 pairs): 0 differences on numbers, with one NaN operand and with signed zeros; when both are NaN only
// the payload differs; a denormal operand is treated as a zero of its sign (the emulation passed its bits through), which
// is the console's result under the flush mode the game runs with.
uint32_t EmitMinMaxF32Value(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), result, GlslStd450(state),
	                          max_value ? GLSLstd450NMax : GLSLstd450NMin, lhs, rhs);
	return result;
}

uint32_t EmitFPMedTri32(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	// RDNA2 MED3 (upstream 3c4124daf, defb2f11d). The MIN3/MAX3 steps are host NMin/NMax too, as every V_MIN/MAX_F32 (only
	// the payload of two NaN sources can differ, see EmitMinMaxF32Value).
	// The NaN path is RDNA2 MIN3, which keeps the later source when two are NaN.
	const auto min_ba = EmitMinMaxF32Value(state, b, a, false);
	const auto min3   = EmitMinMaxF32Value(state, c, min_ba, false);
	const auto max_ab = EmitMinMaxF32Value(state, a, b, true);
	const auto max3   = EmitMinMaxF32Value(state, max_ab, c, true);
	// RDNA2 compares numerically against MAX3, including ties between signed zeros, then takes
	// V_MAX_F32 of the other two sources in their original order.
	const auto a_is_max = Binary(state, spv::OpFOrdEqual, TypeBool(state), max3, a);
	const auto b_is_max = Binary(state, spv::OpFOrdEqual, TypeBool(state), max3, b);
	const auto x        = Select(state, TypeF32(state), a_is_max, b, a);
	const auto y =
	    Select(state, TypeF32(state), EmitLogicalOrBool(state, a_is_max, b_is_max), c, b);
	const auto median = EmitMinMaxF32Value(state, x, y, true);
	const auto is_nan = [&](uint32_t value) {
		return Unary(state, spv::OpIsNan, TypeBool(state), value);
	};
	const auto any_nan =
	    EmitLogicalOrBool(state, EmitLogicalOrBool(state, is_nan(a), is_nan(b)), is_nan(c));
	return Select(state, TypeF32(state), any_nan, min3, median);
}

// GPU time cloth 4.62 -> 4.47, FSR3 2.50 -> 2.31, volfog 1.33 -> 1.21 ms: the emulator flushes a denormal f32 input of
// 1/x (FDiv), InverseSqrt, Sqrt, Exp2, Log2 and of every f32 compare to a zero of its sign (4 ops each; cloth ~1,000
// flushes, ~6% of its ops). The host's default f32 mode already does it: checked on the RTX 5070 Ti over all 2^32 f32
// patterns (test shader denorm.comp), op(x) against op(flush(x)) for the five operations and the
// six compares against 16 values (zeros, denormals, FLT_MIN, +-1, +-inf, NaN), each as an immediate and as a dynamic
// operand, with and without the 16-bit float controls the native-half modules declare: 0 differences, 16,777,214
// denormal inputs. Limited to the device gate of those tests (HostFloatFeatures::exact_half, NVIDIA) and to modules
// without f64 (whose float controls were not tested).
static bool F32FlushHost(const EmitterState& state) {
	return GetHostFloatFeatures().exact_half && !state.program.info.float64;
}

uint32_t EmitFlushF32InputUnlessHost(EmitterState& state, uint32_t value) {
	return F32FlushHost(state) ? value : EmitFlushF32DenormToSignedZero(state, value);
}

uint32_t EmitFlushF32DenormToSignedZero(EmitterState& state, uint32_t value) {
	// x * 0 keeps the sign of a denormal; NaN and infinity fail the ordered compare.
	const auto magnitude = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), magnitude, GlslStd450(state),
	                          GLSLstd450FAbs, value);
	const auto denormal = Binary(state, spv::OpFOrdLessThan, TypeBool(state), magnitude,
	                             ConstantF32(state, 0x00800000u));
	return Select(state, TypeF32(state), denormal,
	              Binary(state, spv::OpFMul, TypeF32(state), value, ConstantF32(state, 0)), value);
}

uint32_t EmitTrigCycleF32(EmitterState& state, uint32_t src, bool preserve_signed_zero) {
	// Finite values of magnitude 2^23 and above are integers, so fract already gives them +0.
	const auto fract = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), fract, GlslStd450(state),
	                          GLSLstd450Fract, src);
	if (!preserve_signed_zero) {
		return fract;
	}
	const auto abs_bits = Binary(state, spv::OpBitwiseAnd, TypeU32(state),
	                             Unary(state, spv::OpBitcast, TypeU32(state), src),
	                             ConstantU32(state, 0x7fffffffu));
	return Select(state, TypeF32(state),
	              Binary(state, spv::OpIEqual, TypeBool(state), abs_bits, ConstantU32(state, 0)),
	              src, fract);
}

// Half <-> float conversions with integer arithmetic, keeping f16 subnormals as RDNA does (GLSL Pack/UnpackHalf2x16 may
// flush them to zero on a host driver).
static uint32_t U32Const(EmitterState& state, uint32_t value) {
	return ConstantU32(state, value);
}

// Low 16 bits of `bits` as an IEEE half, converted exactly to float.
static uint32_t ExactHalfToF32(EmitterState& state, uint32_t bits) {
	const auto u32  = TypeU32(state);
	const auto h    = Binary(state, spv::OpBitwiseAnd, u32, bits, U32Const(state, 0xffffu));
	const auto sign = Binary(state, spv::OpShiftLeftLogical, u32,
	                         Binary(state, spv::OpBitwiseAnd, u32, h, U32Const(state, 0x8000u)), U32Const(state, 16u));
	const auto exponent = Binary(state, spv::OpBitwiseAnd, u32,
	                             Binary(state, spv::OpShiftRightLogical, u32, h, U32Const(state, 10u)), U32Const(state, 0x1fu));
	const auto mantissa = Binary(state, spv::OpBitwiseAnd, u32, h, U32Const(state, 0x3ffu));
	const auto mantissa_shifted = Binary(state, spv::OpShiftLeftLogical, u32, mantissa, U32Const(state, 13u));
	const auto normal = Binary(
	    state, spv::OpBitwiseOr, u32, sign,
	    Binary(state, spv::OpBitwiseOr, u32,
	           Binary(state, spv::OpShiftLeftLogical, u32,
	                  Binary(state, spv::OpIAdd, u32, exponent, U32Const(state, 112u)), U32Const(state, 23u)),
	           mantissa_shifted));
	const auto special = Binary(state, spv::OpBitwiseOr, u32, sign,
	                            Binary(state, spv::OpBitwiseOr, u32, U32Const(state, 0x7f800000u), mantissa_shifted));
	// Subnormal: mantissa * 2^-24, exact in float.
	const auto scaled = Binary(state, spv::OpFMul, TypeF32(state), Unary(state, spv::OpConvertUToF, TypeF32(state), mantissa),
	                           ConstantF32Value(state, 5.9604644775390625e-8f));
	const auto subnormal =
	    Binary(state, spv::OpBitwiseOr, u32, sign, Unary(state, spv::OpBitcast, u32, scaled));
	const auto is_zero_exp = Binary(state, spv::OpIEqual, TypeBool(state), exponent, U32Const(state, 0u));
	const auto is_max_exp  = Binary(state, spv::OpIEqual, TypeBool(state), exponent, U32Const(state, 31u));
	const auto result =
	    Select(state, u32, is_zero_exp, subnormal, Select(state, u32, is_max_exp, special, normal));
	return Unary(state, spv::OpBitcast, TypeF32(state), result);
}

// A float converted exactly to an IEEE half (round to nearest even), in the low 16 bits.
static uint32_t ExactF32ToHalf(EmitterState& state, uint32_t value) {
	const auto u32  = TypeU32(state);
	const auto bln  = TypeBool(state);
	const auto bits = Unary(state, spv::OpBitcast, u32, value);
	const auto sign = Binary(state, spv::OpBitwiseAnd, u32,
	                         Binary(state, spv::OpShiftRightLogical, u32, bits, U32Const(state, 16u)), U32Const(state, 0x8000u));
	const auto magnitude = Binary(state, spv::OpBitwiseAnd, u32, bits, U32Const(state, 0x7fffffffu));
	const auto nan_half  = Binary(state, spv::OpBitwiseOr, u32, U32Const(state, 0x7e00u),
	                              Binary(state, spv::OpBitwiseAnd, u32,
	                                     Binary(state, spv::OpShiftRightLogical, u32, magnitude, U32Const(state, 13u)),
	                                     U32Const(state, 0x3ffu)));
	const auto lsb = Binary(state, spv::OpBitwiseAnd, u32,
	                        Binary(state, spv::OpShiftRightLogical, u32, magnitude, U32Const(state, 13u)), U32Const(state, 1u));
	const auto normal = Binary(
	    state, spv::OpShiftRightLogical, u32,
	    Binary(state, spv::OpIAdd, u32,
	           Binary(state, spv::OpISub, u32, magnitude, U32Const(state, 0x38000000u)),
	           Binary(state, spv::OpIAdd, u32, U32Const(state, 0xfffu), lsb)),
	    U32Const(state, 13u));
	const auto absolute = Unary(state, spv::OpBitcast, TypeF32(state), magnitude);
	const auto scaled   = Binary(state, spv::OpFMul, TypeF32(state), absolute, ConstantF32Value(state, 16777216.0f));
	const auto rounded  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), rounded, GlslStd450(state), GLSLstd450RoundEven, scaled);
	const auto subnormal = Unary(state, spv::OpConvertFToU, u32, rounded);
	const auto is_nan    = Binary(state, spv::OpUGreaterThan, bln, magnitude, U32Const(state, 0x7f800000u));
	const auto is_inf    = Binary(state, spv::OpUGreaterThanEqual, bln, magnitude, U32Const(state, 0x477ff000u));
	const auto is_normal = Binary(state, spv::OpUGreaterThanEqual, bln, magnitude, U32Const(state, 0x38800000u));
	const auto half      = Select(state, u32, is_nan, nan_half,
	                              Select(state, u32, is_inf, U32Const(state, 0x7c00u),
	                                     Select(state, u32, is_normal, normal, subnormal)));
	return Binary(state, spv::OpBitwiseOr, u32, sign, half);
}

// Exact half conversions in hardware (HostFloatFeatures): OpFConvert to and from 16-bit
// floats, the integer forms above in about 12 instructions instead of about 25 (the integer forms doubled the SPIR-V
// of the largest compute shaders: 40395313 380k -> 963k words). The module declares RoundingModeRTE and
// SignedZeroInfNanPreserve for 16 bits (DeclareNativeHalf). The host conversion returns a canonical NaN, while RDNA
// (and the integer forms) keep a NaN's sign and payload, so NaNs take the integer forms' encoding (NanHalfBits,
// NanFloatBits).
// No DenormPreserve for 16 bits: with it (8b50efca) the NVIDIA driver (RTX 5070 Ti, 0x9a438000) compiles the whole
// module's 32-bit arithmetic without flushing denormals (f32 FMul/FAdd/compares/division keep them, OpFNegate
// canonicalizes NaNs) - the mode alone does it, RoundingModeRTE 16, SignedZeroInfNanPreserve 16 and 16-bit floats do
// not. Every Ghost of Yotei pixel and vertex shader runs with FLOAT_MODE 0xc0 (f32 denormals flushed, shader.cpp),
// which the emulator gets from the host's default f32 flush (the device cannot declare DenormFlushToZero 32:
// shaderDenormFlushToZeroFloat32 = 0). With the mode Atsu's kimono, an NPC's hat and saddle parts turned black with a
// gold pattern and her hair grey, although every conversion was exact (half_native_study tests). Without it the
// driver still keeps f16 subnormals in these conversions (all 2^32 floats, 2^16 halves and 2^32 lane pairs checked,
// half_check and half_native_study tests), which no mode guarantees then; the device gate
// (HostFloatFeatures::exact_half) asks for shaderDenormPreserveFloat16 as the sign that it can.
static bool NativeExactHalf(EmitterState& state) {
	const auto features = GetHostFloatFeatures();
	if (!(state.program.info.float64 ? features.exact_half_float64 : features.exact_half)) return false;
	state.native_half = true;
	return true;
}

void DeclareNativeHalf(EmitterState& state) {
	state.builder.RequireCapability(spv::CapabilityFloat16);
	state.builder.RequireCapability(spv::CapabilityRoundingModeRTE);
	state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeRoundingModeRTE, 16u);
	state.builder.AddExecutionMode(state.main_func, spv::ExecutionModeSignedZeroInfNanPreserve, 16u);
	// No DenormPreserve 16: it turns off the host's f32 denormal flush for the whole module (see NativeExactHalf).
}

// V_CVT_PKRTZ_F16_F32 through the hardware conversion of both lanes (FSR3 3.03 -> 2.89 ms on the GPU; round to nearest
// even under the NativeExactHalf modules' RoundingModeRTE 16), converted back exactly; a lane whose converted-back value
// is above the input's rounded away from zero and steps one f16 magnitude toward zero (keeps the sign, crosses exponent
// boundaries, turns an overflow to inf into 0x7bff; such a lane is never 0, so the step cannot borrow from the other
// lane); NaN lanes take the integer form's encoding (sign, 0x7e00, top 10 payload bits). Checked on the RTX 5070 Ti against
// EmitF32ToF16RtzBits over all 2^32 f32 patterns under the same float controls (test shader rtz.comp): 0
// differences, 167,772,160 subnormal results.
// "Rounded away" is back_bits > in_bits (a non-NaN lane keeps its sign through the conversion, so this equals the
// magnitude compare; NaN lanes are replaced anyway) and both lanes' NaN encodings are built once on the packed word and
// merged with one masked XOR (package FSR3 1.84 -> 1.62 ms, Volfog 1.02 -> 0.86 ms, open world +1.4%). CPU checks
// (verify_rtz_trim.py): away over all 2^32 inputs 0 differences, NaN merge 0 differences over 67 M random cases.
uint32_t EmitPackF16RtzNative(EmitterState& state, uint32_t a, uint32_t b) {
	if (!NativeExactHalf(state)) return 0;
	const auto u32    = TypeU32(state);
	const auto f32    = TypeF32(state);
	const auto pair   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, a, b);
	const auto packed = Unary(state, spv::OpBitcast, u32, Unary(state, spv::OpFConvert, TypeF16Pair(state), pair));
	const auto back   = Unary(state, spv::OpFConvert, TypeF32Vector(state, 2), Unary(state, spv::OpBitcast, TypeF16Pair(state), packed));
	const auto lane   = [&](uint32_t input, uint32_t index, uint32_t step, uint32_t nan_mask) {
		const auto back_lane = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, f32, back_lane, back, index);
		const auto in_bits = Unary(state, spv::OpBitcast, u32, input);
		const auto away    = Binary(state, spv::OpUGreaterThan, TypeBool(state), Unary(state, spv::OpBitcast, u32, back_lane), in_bits);
		const auto is_nan  = Binary(state, spv::OpUGreaterThan, TypeBool(state),
		                            Binary(state, spv::OpBitwiseAnd, u32, in_bits, U32Const(state, 0x7fffffffu)),
		                            U32Const(state, 0x7f800000u));
		return std::tuple {Select(state, u32, away, U32Const(state, step), U32Const(state, 0u)),
		                   Select(state, u32, is_nan, U32Const(state, nan_mask), U32Const(state, 0u)), in_bits};
	};
	const auto [step_lo, mask_lo, bits_lo] = lane(a, 0u, 1u, 0xffffu);
	const auto [step_hi, mask_hi, bits_hi] = lane(b, 1u, 0x10000u, 0xffff0000u);
	const auto stepped = Binary(state, spv::OpISub, u32, Binary(state, spv::OpISub, u32, packed, step_lo), step_hi);
	// Both NaN encodings at once: sign, 0x7e00, the top 10 payload bits of each input.
	const auto nan_pair = Binary(
	    state, spv::OpBitwiseOr, u32,
	    Binary(state, spv::OpBitwiseOr, u32,
	           Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftRightLogical, u32, bits_lo, U32Const(state, 13u)),
	                  U32Const(state, 0x3ffu)),
	           Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftRightLogical, u32, bits_lo, U32Const(state, 16u)),
	                  U32Const(state, 0x8000u))),
	    Binary(state, spv::OpBitwiseOr, u32,
	           Binary(state, spv::OpBitwiseOr, u32,
	                  Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftLeftLogical, u32, bits_hi, U32Const(state, 3u)),
	                         U32Const(state, 0x03ff0000u)),
	                  Binary(state, spv::OpBitwiseAnd, u32, bits_hi, U32Const(state, 0x80000000u))),
	           U32Const(state, 0x7e007e00u)));
	const auto mask = Binary(state, spv::OpBitwiseOr, u32, mask_lo, mask_hi);
	return Binary(state, spv::OpBitwiseXor, u32, stepped,
	              Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpBitwiseXor, u32, stepped, nan_pair), mask));
}

uint32_t TypeF16(EmitterState& state) {
	if (state.f16_type == 0) {
		state.f16_type = state.builder.Type(spv::OpTypeFloat, 16);
	}
	return state.f16_type;
}

uint32_t TypeF16Pair(EmitterState& state) {
	if (state.f16_pair_type == 0) {
		state.f16_pair_type = state.builder.Type(spv::OpTypeVector, TypeF16(state), 2);
	}
	return state.f16_pair_type;
}

// KYTY_PERF_F16_CVT_PLAIN (default: FSR3 accumulate d42889155b0a6b9f only; "0" off, "1"/"all"
// every shader, or a list of shader hashes): the guest f16 <-> f32 conversions (V_CVT_F16_F32, V_CVT_F32_F16,
// V_CVT_PKRTZ / pack) take the hardware conversion as it is, without the NaN payload re-encoding below (~10 instructions per conversion; FSR3's accumulate pass has ~390 of them). Only a
// NaN's payload differs (the host returns its canonical NaN); every other value converts bit-exactly
// (half_native_study tests). The NaN re-encoding was added while the black kimono was blamed on payloads; the cause
// was DenormPreserve 16 (memory nan-payloads-matter).
// FSR3: 1.5376 -> 1.4050 ms per dispatch (KYTY_TRACE_GPU_TIME); refinterp of the frame-1600 capture WORSE 0 (baseline
// captures 0-3), capture set ref_cvt1.
static bool F16CvtPlain(const EmitterState& state) {
	static const std::string list = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_PERF_F16_CVT_PLAIN");
		std::string v     = value != nullptr ? std::string(value) : std::string("d42889155b0a6b9f");
		if (!v.empty() && v != "0") std::fprintf(stderr, "PERF_F16_CVT_PLAIN %s\n", v.c_str());
		return v;
	}();
	if (list.empty() || list == "0") return false;
	if (list == "1" || list == "all") return true;
	char text[17];
	std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(state.program.shader_hash));
	return list.find(text) != std::string::npos;
}

// `converted` (a hardware f32 -> f16 conversion of `value` in the low 16 bits, the rest zero) with a NaN encoded as
// ExactF32ToHalf does: the sign, the quiet bit and the top ten payload bits.
static uint32_t NanHalfBits(EmitterState& state, uint32_t value, uint32_t converted) {
	if (F16CvtPlain(state)) return converted;
	const auto u32  = TypeU32(state);
	const auto bits = Unary(state, spv::OpBitcast, u32, value);
	const auto nan  = Binary(
	    state, spv::OpBitwiseOr, u32,
	    Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftRightLogical, u32, bits, U32Const(state, 16u)),
	           U32Const(state, 0x8000u)),
	    Binary(state, spv::OpBitwiseOr, u32,
	           Binary(state, spv::OpBitwiseAnd, u32,
	                  Binary(state, spv::OpShiftRightLogical, u32, bits, U32Const(state, 13u)), U32Const(state, 0x3ffu)),
	           U32Const(state, 0x7e00u)));
	return Select(state, u32, Unary(state, spv::OpIsNan, TypeBool(state), value), nan, converted);
}

// `converted` (a hardware f16 -> f32 conversion of the low 16 bits of `bits`) with a NaN encoded as ExactHalfToF32
// does: the sign and the payload widened, not quieted.
static uint32_t NanFloatBits(EmitterState& state, uint32_t bits, uint32_t converted) {
	if (F16CvtPlain(state)) return converted;
	const auto u32 = TypeU32(state);
	// For an all-ones f16 exponent, (h & 0x7fff) << 13 | 0x70000000 is the all-ones f32 exponent and the payload.
	const auto nan = Binary(
	    state, spv::OpBitwiseOr, u32,
	    Binary(state, spv::OpBitwiseAnd, u32, Binary(state, spv::OpShiftLeftLogical, u32, bits, U32Const(state, 16u)),
	           U32Const(state, 0x80000000u)),
	    Binary(state, spv::OpBitwiseOr, u32,
	           Binary(state, spv::OpShiftLeftLogical, u32,
	                  Binary(state, spv::OpBitwiseAnd, u32, bits, U32Const(state, 0x7fffu)), U32Const(state, 13u)),
	           U32Const(state, 0x70000000u)));
	return Select(state, TypeF32(state), Unary(state, spv::OpIsNan, TypeBool(state), converted),
	              Unary(state, spv::OpBitcast, TypeF32(state), nan), converted);
}

uint32_t EmitPackHalf2x16(EmitterState& state, uint32_t pair) {
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), low, pair, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), high, pair, 1);
	if (NativeExactHalf(state)) {
		const auto u32    = TypeU32(state);
		const auto packed = Unary(state, spv::OpBitcast, u32, Unary(state, spv::OpFConvert, TypeF16Pair(state), pair));
		if (F16CvtPlain(state)) return packed; // KYTY_PERF_F16_CVT_PLAIN: both lanes as converted
		return Binary(
		    state, spv::OpBitwiseOr, u32,
		    NanHalfBits(state, low, Binary(state, spv::OpBitwiseAnd, u32, packed, U32Const(state, 0xffffu))),
		    Binary(state, spv::OpShiftLeftLogical, u32,
		           NanHalfBits(state, high, Binary(state, spv::OpShiftRightLogical, u32, packed, U32Const(state, 16u))),
		           U32Const(state, 16u)));
	}
	return Binary(state, spv::OpBitwiseOr, TypeU32(state), ExactF32ToHalf(state, low),
	              Binary(state, spv::OpShiftLeftLogical, TypeU32(state), ExactF32ToHalf(state, high),
	                     U32Const(state, 16u)));
}

uint32_t EmitF32ToF16Bits(EmitterState& state, uint32_t f32) {
	if (!NativeExactHalf(state)) {
		return ExactF32ToHalf(state, f32); // the high lane of a pair with zero is zero: one conversion, not two
	}
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, f32, ConstantF32(state, 0));
	const auto packed =
	    Unary(state, spv::OpBitcast, TypeU32(state), Unary(state, spv::OpFConvert, TypeF16Pair(state), pair));
	return NanHalfBits(state, f32, packed); // the high half converts 0.0: zero
}

uint32_t EmitF16BitsToF32(EmitterState& state, uint32_t bits) {
	if (NativeExactHalf(state)) {
		const auto halves = Unary(state, spv::OpBitcast, TypeF16Pair(state), bits);
		const auto low    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF16(state), low, halves, 0);
		return NanFloatBits(state, bits, Unary(state, spv::OpFConvert, TypeF32(state), low));
	}
	return ExactHalfToF32(state, bits);
}

// HalfPk* operate on two IEEE halves packed in a U32 (low lane bits 0-15,
// high lane bits 16-31), as V_PK_*_F16 and, on the low lane, the scalar V_*_F16 use them. Natively (NativeExactHalf:
// the module declares Float16, RoundingModeRTE 16 and SignedZeroInfNanPreserve 16, no DenormPreserve 16) one f16vec2
// instruction does both lanes. Checked on the RTX 5070 Ti over all 2^32 lane pairs against the f32 path
// (half_math tests): add, mul, min, max and saturate 0 differences, f16 subnormals kept; fma differs in
// 0.0035% of hashed triples by 1 ulp, every sampled one the single rounding of the fused f16 FMA (the f32 path rounds
// twice). Adds and multiplies are NoContraction: a guest product is rounded to f16 before its sum. Without native halves
// each lane goes through the f32 path these ops replace.
namespace {

enum class HalfOp { Add, Mul, Fma, Min, Max, MinTri, MaxTri, MedTri, Saturate };

uint32_t HalvesOf(EmitterState& state, uint32_t bits) {
	return Unary(state, spv::OpBitcast, TypeF16Pair(state), bits);
}

uint32_t BitsOf(EmitterState& state, uint32_t halves) {
	return Unary(state, spv::OpBitcast, TypeU32(state), halves);
}

// Lane mask of a bvec2: 0x0000ffff for the low lane, 0xffff0000 for the high lane.
uint32_t LaneMask(EmitterState& state, uint32_t condition) {
	const auto low  = state.builder.AllocateId();
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeBool(state), low, condition, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeBool(state), high, condition, 1);
	const auto u32 = TypeU32(state);
	return Binary(state, spv::OpBitwiseOr, u32, Select(state, u32, low, U32Const(state, 0x0000ffffu), U32Const(state, 0u)),
	              Select(state, u32, high, U32Const(state, 0xffff0000u), U32Const(state, 0u)));
}

// Per lane: `when_set` where `mask` is set, else `otherwise` (bit selection, NaN payloads kept).
uint32_t SelectLanes(EmitterState& state, uint32_t mask, uint32_t when_set, uint32_t otherwise) {
	const auto u32 = TypeU32(state);
	return Binary(state, spv::OpBitwiseOr, u32, Binary(state, spv::OpBitwiseAnd, u32, when_set, mask),
	              Binary(state, spv::OpBitwiseAnd, u32, otherwise, Unary(state, spv::OpNot, u32, mask)));
}

// The packed f16 min/max as one GLSL.std.450 NMin/NMax (FSR3 4.75 -> 3.30 ms on the GPU; open world 2+2 alternating runs
// 12.15/11.97 against 11.82/11.74 fps, +2.4%; refinterp of FSR3: no new differences, captures refcap_nmm, refcap_nmm0).
// Checked on the RTX 5070 Ti over all 2^32 lane pairs against the emulation it replaced (per lane: ordered compare, two
// zeros by bits, a NaN operand gives the other one; test shader half_nmin.comp): 0 differences with one NaN
// operand, with signed zeros and on numbers; only when BOTH operands are NaN the NaN payload differs (4 + 4 pairs: 0x7fff
// against the rhs quieted). The median built from it differs only when all three operands are NaN (a NaN either way). The
// emulation cost ~40 instructions per min/max; FSR3's accumulate pass (d42889155b0a6b9f) spent ~2/3 of its samples on such
// integer logic (Nsight real-time shader profiler).
uint32_t NativeHalfMinMax(EmitterState& state, uint32_t a, uint32_t b, bool max_value) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF16Pair(state), result, GlslStd450(state),
	                          max_value ? GLSLstd450NMax : GLSLstd450NMin, HalvesOf(state, a), HalvesOf(state, b));
	return BitsOf(state, result);
}

uint32_t NativeHalfMedTri(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	// EmitFPMedTri32 per lane: the median, or the minimum of the three when any is NaN.
	const auto min_ab   = NativeHalfMinMax(state, a, b, false);
	const auto min3     = NativeHalfMinMax(state, min_ab, c, false);
	const auto max_ab   = NativeHalfMinMax(state, a, b, true);
	const auto high_min = NativeHalfMinMax(state, max_ab, c, false);
	const auto median   = NativeHalfMinMax(state, min_ab, high_min, true);
	const auto bvec     = TypeBoolVector(state, 2);
	const auto nan_ab   = Binary(state, spv::OpLogicalOr, bvec, Unary(state, spv::OpIsNan, bvec, HalvesOf(state, a)),
	                             Unary(state, spv::OpIsNan, bvec, HalvesOf(state, b)));
	const auto any_nan = Binary(state, spv::OpLogicalOr, bvec, nan_ab, Unary(state, spv::OpIsNan, bvec, HalvesOf(state, c)));
	return SelectLanes(state, LaneMask(state, any_nan), min3, median);
}

uint32_t NativeHalfOp(EmitterState& state, HalfOp op, uint32_t a, uint32_t b, uint32_t c) {
	const auto pair = TypeF16Pair(state);
	switch (op) {
		case HalfOp::Add:
		case HalfOp::Mul: {
			const auto result =
			    Binary(state, op == HalfOp::Add ? spv::OpFAdd : spv::OpFMul, pair, HalvesOf(state, a), HalvesOf(state, b));
			state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
			return BitsOf(state, result);
		}
		case HalfOp::Fma: {
			const auto result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpExtInst, pair, result, GlslStd450(state), GLSLstd450Fma,
			                          HalvesOf(state, a), HalvesOf(state, b), HalvesOf(state, c));
			return BitsOf(state, result);
		}
		case HalfOp::Min: return NativeHalfMinMax(state, a, b, false);
		case HalfOp::Max: return NativeHalfMinMax(state, a, b, true);
		case HalfOp::MinTri: return NativeHalfMinMax(state, NativeHalfMinMax(state, a, b, false), c, false);
		case HalfOp::MaxTri: return NativeHalfMinMax(state, NativeHalfMinMax(state, a, b, true), c, true);
		case HalfOp::MedTri: return NativeHalfMedTri(state, a, b, c);
		case HalfOp::Saturate: {
			// KYTY_FIX_CLAMP_NAN: as EmitFPSaturate32, per half.
			const int  mode   = IR::ClampNanModeFor(state.program.shader_hash);
			const bool dx10   = mode != 0 && ProgramDx10Clamp(state);
			const auto result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpExtInst, pair, result, GlslStd450(state),
			                          mode == 1 && dx10 ? GLSLstd450NClamp : GLSLstd450FClamp,
			                          HalvesOf(state, a), Unary(state, spv::OpBitcast, pair, U32Const(state, 0u)),
			                          Unary(state, spv::OpBitcast, pair, U32Const(state, 0x3c003c00u)));
			if (mode == 2 && dx10) {
				const auto lanes = Unary(state, spv::OpIsNan, TypeBoolVector(state, 2), result);
				ClampNanCheck(state, Unary(state, spv::OpAny, TypeBool(state), lanes), 1);
			}
			return BitsOf(state, result);
		}
	}
	return a;
}

// The f32 path of the translation these ops replace: each lane converted exactly, the f32 operation, both lanes packed.
uint32_t FallbackHalfOp(EmitterState& state, HalfOp op, uint32_t a, uint32_t b, uint32_t c) {
	const auto u32  = TypeU32(state);
	const auto lane = [&](uint32_t bits, bool high) {
		return EmitF16BitsToF32(state, high ? Binary(state, spv::OpShiftRightLogical, u32, bits, U32Const(state, 16u))
		                                    : Binary(state, spv::OpBitwiseAnd, u32, bits, U32Const(state, 0xffffu)));
	};
	uint32_t results[2] = {};
	for (uint32_t index = 0; index < 2; index++) {
		const bool high = index != 0;
		const auto x    = lane(a, high);
		const auto y    = op == HalfOp::Saturate ? 0u : lane(b, high);
		const auto z    = op == HalfOp::Fma || op == HalfOp::MinTri || op == HalfOp::MaxTri || op == HalfOp::MedTri
		                      ? lane(c, high)
		                      : 0u;
		switch (op) {
			case HalfOp::Add: results[index] = EmitFPAdd32(state, x, y); break;
			case HalfOp::Mul: results[index] = EmitFPMul32(state, x, y); break;
			case HalfOp::Fma: results[index] = EmitFPFma32(state, x, y, z); break;
			case HalfOp::Min: results[index] = EmitFPMin32(state, x, y); break;
			case HalfOp::Max: results[index] = EmitFPMax32(state, x, y); break;
			case HalfOp::MinTri: results[index] = EmitFPMinTri32(state, x, y, z); break;
			case HalfOp::MaxTri: results[index] = EmitFPMaxTri32(state, x, y, z); break;
			case HalfOp::MedTri: results[index] = EmitFPMedTri32(state, x, y, z); break;
			case HalfOp::Saturate: results[index] = EmitFPSaturate32(state, x); break;
		}
	}
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, results[0], results[1]);
	return EmitPackHalf2x16(state, pair);
}

uint32_t EmitHalfOp(EmitterState& state, HalfOp op, uint32_t a, uint32_t b = 0, uint32_t c = 0) {
	return NativeExactHalf(state) ? NativeHalfOp(state, op, a, b, c) : FallbackHalfOp(state, op, a, b, c);
}

} // namespace

uint32_t EmitHalfPkAdd(EmitterState& state, uint32_t a, uint32_t b) { return EmitHalfOp(state, HalfOp::Add, a, b); }
uint32_t EmitHalfPkMul(EmitterState& state, uint32_t a, uint32_t b) { return EmitHalfOp(state, HalfOp::Mul, a, b); }
uint32_t EmitHalfPkFma(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	return EmitHalfOp(state, HalfOp::Fma, a, b, c);
}
uint32_t EmitHalfPkMin(EmitterState& state, uint32_t a, uint32_t b) { return EmitHalfOp(state, HalfOp::Min, a, b); }
uint32_t EmitHalfPkMax(EmitterState& state, uint32_t a, uint32_t b) { return EmitHalfOp(state, HalfOp::Max, a, b); }
uint32_t EmitHalfPkMinTri(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	return EmitHalfOp(state, HalfOp::MinTri, a, b, c);
}
uint32_t EmitHalfPkMaxTri(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	return EmitHalfOp(state, HalfOp::MaxTri, a, b, c);
}
uint32_t EmitHalfPkMedTri(EmitterState& state, uint32_t a, uint32_t b, uint32_t c) {
	return EmitHalfOp(state, HalfOp::MedTri, a, b, c);
}
uint32_t EmitHalfPkSaturate(EmitterState& state, uint32_t a) { return EmitHalfOp(state, HalfOp::Saturate, a); }

// Mixed-precision FMA (V_MAD_MIX*): the f16 operands of an f32 FMA and its f16 result
// converted by hardware alone, without NanFloatBits / NanHalfBits: an f32 FMA (and its clamp/omod) returns the host's
// canonical NaN whatever NaN came in, so the payload those encodings keep never reaches the result. Checked end to end
// on the RTX 5070 Ti (test shader half_mix.comp: (f16, f32, f16), (f32, f16, f16) and a clamped (f16, f16, f32)
// FMA, both f16 operands exhaustive in pairs, the f32 one hashed over all bit patterns): 0 differences.
uint32_t EmitHalfToF32Arith(EmitterState& state, uint32_t bits) {
	if (!NativeExactHalf(state)) return EmitF16BitsToF32(state, bits);
	const auto low = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF16(state), low, HalvesOf(state, bits), 0);
	return Unary(state, spv::OpFConvert, TypeF32(state), low);
}

uint32_t EmitF32ToHalfArith(EmitterState& state, uint32_t f32) {
	if (!NativeExactHalf(state)) return EmitF32ToF16Bits(state, f32);
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, f32, ConstantF32(state, 0));
	return Unary(state, spv::OpBitcast, TypeU32(state), Unary(state, spv::OpFConvert, TypeF16Pair(state), pair)); // high half 0
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
