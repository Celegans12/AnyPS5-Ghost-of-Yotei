#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/frontend/decode/ImageOps.h"
#include "common/shaderEnv.h"

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t DmaskComponentIndex(uint32_t dmask, uint32_t component) {
	uint32_t index = 0;
	for (uint32_t i = 0; i < component; i++) {
		index += (dmask >> i) & 1u;
	}
	return index;
}

uint32_t DmaskComponent(uint32_t dmask, uint32_t index) {
	for (uint32_t component = 0; component < 4u; component++) {
		if (((dmask >> component) & 1u) != 0u && index-- == 0u) return component;
	}
	return 0u;
}

uint32_t ImageGatherComponent(uint32_t dmask) {
	switch (dmask) {
		case 0x2u: return 1;
		case 0x4u: return 2;
		case 0x8u: return 3;
		default: return 0;
	}
}

bool HasFlag(const IR::MemoryInfo& mem, uint32_t flag) {
	return (mem.image_sample_flags & flag) != 0u;
}

uint32_t AddressU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto layout = Decoder::ImageAddressComponentLayout(mem.image_sample_flags, mem.image_dimension, component);
	const auto packed = layout.bit_offset / 32u;
	if (packed >= address.NumArgs()) return ConstantU32(ctx.state, 0);
	auto value = ctx.Def(address.Arg(packed));
	if (layout.bit_width == 16u) {
		if ((layout.bit_offset & 31u) != 0u) {
			value = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), value,
			               ConstantU32(ctx.state, 16));
		}
		value = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), value,
		               ConstantU32(ctx.state, 0xffffu));
	}
	return value;
}

uint32_t AddressF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                    uint32_t component) {
	const auto value = AddressU32(ctx, mem, address, component);
	return Decoder::ImageAddressComponentLayout(mem.image_sample_flags, mem.image_dimension, component).bit_width == 16u
	           ? EmitF16BitsToF32(ctx.state, value)
	           : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), value);
}

ImageSampleLayout Layout(const IR::MemoryInfo& mem) {
	ImageSampleLayout layout;
	uint32_t          cursor = 0;
	const auto&       info   = ImageDimensionInfoFor(mem.image_dimension);
	if (HasFlag(mem, Decoder::ImageSampleFlagOffset)) layout.offset = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagBias)) layout.bias = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagCompare)) layout.dref = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagDerivative)) {
		layout.grad_x = cursor;
		cursor += info.spatial_components;
		layout.grad_y = cursor;
		cursor += info.spatial_components;
	}
	layout.coord = cursor;
	cursor += info.coordinate_components;
	if (HasFlag(mem, Decoder::ImageSampleFlagLod)) layout.lod = cursor++;
	if (HasFlag(mem, Decoder::ImageSampleFlagLodClamp)) layout.clamp = cursor++;
	return layout;
}

uint32_t ZeroF32(EmitterState& state) {
	return ConstantF32(state, 0);
}

uint32_t GatherMip(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                   const IR::Inst& address, const ImageSampleLayout& layout, uint32_t mip_count) {
	auto& state = ctx.state;
	const auto Extract = [&](uint32_t word, uint32_t first, uint32_t count) {
		return EmitBitFieldUExtract(state, word, ConstantU32(state, first),
		                           ConstantU32(state, count));
	};
	const auto LodFixed = [&](uint32_t word, uint32_t first) {
		return Binary(state, spv::OpFMul, TypeF32(state),
		              Unary(state, spv::OpConvertUToF, TypeF32(state), Extract(word, first, 12)),
		              ConstantF32Value(state, 1.0f / 256.0f));
	};
	const auto sampler_control = ctx.Arg(inst, 6);
	const auto mip_filter = Extract(sampler_control, 26, 2);
	const auto preclamp = Binary(state, spv::OpINotEqual, TypeBool(state),
	                            Extract(sampler_control, 28, 1), ConstantU32(state, 0));
	const auto zero = ZeroF32(state);
	// Point selects the nearest mip; linear gathers return the lower, primary mip only (upstream ac3711ce7).
	const auto rounding = Select(state, TypeF32(state),
	    Binary(state, spv::OpIEqual, TypeBool(state), mip_filter, ConstantU32(state, 1)),
	    ConstantF32Value(state, 0.5f), zero);
	const auto base = Unary(state, spv::OpConvertUToF, TypeF32(state),
	                        Extract(ctx.Arg(inst, 4), 12, 4));
	const auto minimum = Binary(state, spv::OpFSub, TypeF32(state),
	                            LodFixed(ctx.Arg(inst, 3), 8), base);
	const auto last = ConstantF32Value(state, static_cast<float>(mip_count - 1u));
	auto lod = Binary(state, spv::OpFAdd, TypeF32(state),
	                  AddressF32(ctx, mem, address, layout.lod),
	                  Select(state, TypeF32(state), preclamp, rounding, zero));
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(
	    state, lod, LodFixed(ctx.Arg(inst, 5), 0), LodFixed(ctx.Arg(inst, 5), 12));
	// T# minimum LOD is in physical mip space and applies after the S# view-relative clamp.
	lod = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(state, lod, minimum);
	lod = EmitGlsl<GLSLstd450FClamp, IR::Type::F32>(state, lod, zero, last);
	lod = Binary(state, spv::OpFAdd, TypeF32(state), lod,
	             Select(state, TypeF32(state), preclamp, zero, rounding));
	// The clamped value is nonnegative, so conversion performs the integer mip rounding.
	const auto mip = EmitUMin32(state, Unary(state, spv::OpConvertFToU, TypeU32(state), lod),
	                           ConstantU32(state, mip_count - 1u));
	const auto mip_none = Binary(state, spv::OpIEqual, TypeBool(state),
	                            mip_filter, ConstantU32(state, 0));
	return Select(state, TypeU32(state), mip_none, ConstantU32(state, 0), mip);
}

uint32_t CubeAxis(EmitterState& state, uint32_t value) {
	return Binary(state, spv::OpFSub, TypeF32(state), value, ConstantF32(state, 0x3f800000u));
}

uint32_t CubeLayer(EmitterState& state, uint32_t value) {
	const auto guest = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), guest, value);
	const auto padding = Binary(
	    state, spv::OpShiftLeftLogical, TypeU32(state),
	    Binary(state, spv::OpShiftRightLogical, TypeU32(state), guest, ConstantU32(state, 3)),
	    ConstantU32(state, 1));
	const auto host   = Binary(state, spv::OpISub, TypeU32(state), guest, padding);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), result, host);
	return result;
}

uint32_t CoordF32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  uint32_t first, uint32_t components, bool cube = false) {
	auto x = AddressF32(ctx, mem, address, first);
	if (components == 1u) return x;
	auto y = mem.image_address_components > first + 1u ? AddressF32(ctx, mem, address, first + 1u)
	                                                   : ZeroF32(ctx.state);
	if (cube) {
		x = CubeAxis(ctx.state, x);
		y = CubeAxis(ctx.state, y);
	}
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		auto z = mem.image_address_components > first + 2u
		             ? AddressF32(ctx, mem, address, first + 2u)
		             : ZeroF32(ctx.state);
		if (cube) z = CubeLayer(ctx.state, z);
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 3),
		                              result, x, y, z);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 2),
		                              result, x, y);
	}
	return result;
}

uint32_t CoordU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                  ImageDimension dimension) {
	const auto components = ImageDimensionInfoFor(dimension).coordinate_components;
	const auto x          = AddressU32(ctx, mem, address, 0);
	if (components == 1u) return x;
	const auto y      = mem.image_address_components > 1u ? AddressU32(ctx, mem, address, 1)
	                                                      : ConstantU32(ctx.state, 0);
	const auto result = ctx.state.builder.AllocateId();
	if (components == 3u) {
		const auto z = mem.image_address_components > 2u ? AddressU32(ctx, mem, address, 2)
		                                                 : ConstantU32(ctx.state, 0);
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 3),
		                              result, x, y, z);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 2),
		                              result, x, y);
	}
	return result;
}

uint32_t LodU32(ValueEmitContext& ctx, const IR::MemoryInfo& mem, const IR::Inst& address,
                ImageDimension dimension) {
	const auto component = ImageDimensionInfoFor(dimension).coordinate_components;
	return mem.image_has_mip && mem.image_address_components > component
	           ? AddressU32(ctx, mem, address, component)
	           : ConstantU32(ctx.state, 0);
}

uint32_t FloatBits(ValueEmitContext& ctx, uint32_t value) {
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentBits(ValueEmitContext& ctx, uint32_t value,
                              Prospero::TextureNumericClass numeric_class) {
	if (numeric_class == Prospero::TextureNumericClass::Uint) {
		return value;
	}
	return Unary(ctx.state, spv::OpBitcast, TypeU32(ctx.state), value);
}

uint32_t SampledComponentZero(EmitterState& state, Prospero::TextureNumericClass numeric_class) {
	switch (numeric_class) {
		case Prospero::TextureNumericClass::Float: return ZeroF32(state);
		case Prospero::TextureNumericClass::Uint: return ConstantU32(state, 0);
		case Prospero::TextureNumericClass::Sint: return ConstantI32(state, 0);
		case Prospero::TextureNumericClass::Unsupported: break;
	}
	EXIT("invalid sampled image numeric class");
}

uint32_t ResultVector(ValueEmitContext& ctx, uint32_t value,
                      Prospero::TextureNumericClass numeric_class, bool dref,
                      const IR::MemoryInfo& mem, bool gather = false) {
	auto value_class = numeric_class;
	// A converted format returns the class UnpackImageTexel produces (upstream aafe014d0, 1905b28b3); gathers keep the
	// caller's class (UnpackImageGather returns the packed fields as before).
	const auto conversion = ctx.state.program.info.images[mem.resource].conversion_format;
	if (conversion != Prospero::BufferFormat::kInvalid && !gather) {
		value_class = Prospero::SampledTextureNumericClass(conversion);
	}
	if (dref) {
		value_class = Prospero::TextureNumericClass::Float;
	}
	const bool integer = value_class == Prospero::TextureNumericClass::Uint ||
	                     value_class == Prospero::TextureNumericClass::Sint;
	if (mem.data_bits == 16u) {
		uint32_t   packed[4] = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
		                        ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
		const auto scalar    = [&](uint32_t index) {
			if (dref) return value;
			const auto component = gather ? index : DmaskComponent(mem.dmask, index);
			const auto result    = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract,
			                              ImageScalarType(ctx.state, value_class), result, value,
			                              component);
			return result;
		};
		for (uint32_t word = 0; word < mem.data_dwords; word++) {
			const auto low_index  = word * 2u;
			const auto high_index = low_index + 1u;
			const auto low        = scalar(low_index);
			const auto high       = high_index < mem.component_count
			                            ? scalar(high_index)
			                            : SampledComponentZero(ctx.state, value_class);
			if (integer) {
				const auto low_bits  = SampledComponentBits(ctx, low, value_class);
				const auto high_bits = SampledComponentBits(ctx, high, value_class);
				const auto mask      = ConstantU32(ctx.state, 0xffffu);
				packed[word] =
				    Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state),
				           Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), low_bits, mask),
				           Binary(ctx.state, spv::OpShiftLeftLogical, TypeU32(ctx.state),
				                  Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state),
				                         high_bits, mask),
				                  ConstantU32(ctx.state, 16u)));
			} else {
				const auto pair = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
				                              TypeF32Vector(ctx.state, 2), pair, low, high);
				packed[word] = ctx.state.builder.AllocateId();
				ctx.state.builder.AddFunction(spv::OpExtInst, TypeU32(ctx.state), packed[word],
				                              GlslStd450(ctx.state), GLSLstd450PackHalf2x16, pair);
			}
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4),
		                              result, packed[0], packed[1], packed[2], packed[3]);
		return result;
	}
	uint32_t component[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		if (dref) {
			component[index] = index == 0u ? FloatBits(ctx, value) : ConstantU32(ctx.state, 0);
			continue;
		}
		const auto scalar = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(
		    spv::OpCompositeExtract, ImageScalarType(ctx.state, value_class), scalar, value, index);
		component[index] = SampledComponentBits(ctx, scalar, value_class);
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), result,
	                              component[0], component[1], component[2], component[3]);
	return result;
}

uint32_t QueryDimensions(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                         const IR::Inst& address) {
	const auto  dimension = ctx.state.program.info.images.at(mem.resource).dimension;
	const auto& info      = ImageDimensionInfoFor(dimension);
	const auto  image     = LoadImageDescriptor(ctx.state, mem.resource);
	const auto  size      = ctx.state.builder.AllocateId();
	if (info.multisampled != 0u) {
		ctx.state.builder.AddFunction(spv::OpImageQuerySize,
		                              ImageViewSizeType(ctx.state, dimension), size, image);
	} else {
		ctx.state.builder.AddFunction(spv::OpImageQuerySizeLod,
		                              ImageViewSizeType(ctx.state, dimension), size, image,
		                              AddressU32(ctx, mem, address, 0));
	}
	const auto components = info.coordinate_components;
	uint32_t   result[4]  = {ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0),
	                         ConstantU32(ctx.state, 0), ConstantU32(ctx.state, 0)};
	if (components == 1u) {
		result[0] = size;
	} else {
		for (uint32_t index = 0; index < components; index++) {
			result[index] = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state),
			                              result[index], size, index);
		}
	}
	if (info.multisampled == 0u) {
		result[3] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(ctx.state), result[3],
		                              image);
	}
	const auto vector = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(ctx.state, 4), vector,
	                              result[0], result[1], result[2], result[3]);
	return vector;
}

uint32_t PackedOffset(ValueEmitContext& ctx, IR::Value value, ImageDimension dimension,
                      uint32_t& mask) {
	const auto components = ImageDimensionInfoFor(dimension).spatial_components;
	uint32_t values[3] {};
	if (value.IsImmediate()) {
		// RDNA2 packs signed six-bit texel offsets into the low bits of each byte.
		for (uint32_t index = 0; index < components; index++) {
			const auto bits = (value.U32() >> (index * 8u)) & 0x3fu;
			values[index] = ConstantI32(ctx.state, static_cast<int32_t>(bits ^ 0x20u) - 0x20);
		}
		mask |= spv::ImageOperandsConstOffsetMask;
		return components == 1u
		           ? values[0]
		           : ctx.state.builder.Constant(spv::OpConstantComposite,
		                                        TypeI32Vector(ctx.state, components),
		                                        std::span(values).first(components));
	}
	// A non-constant Offset operand needs ImageGatherExtended (upstream).
	ctx.state.builder.RequireCapability(spv::CapabilityImageGatherExtended);
	mask |= spv::ImageOperandsOffsetMask;
	const auto packed = Unary(ctx.state, spv::OpBitcast, TypeI32(ctx.state), ctx.Def(value));
	for (uint32_t index = 0; index < components; index++) {
		values[index] = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(ctx.state), values[index],
		                              packed, ConstantU32(ctx.state, index * 8u),
		                              ConstantU32(ctx.state, 6));
	}
	if (components == 1u) return values[0];
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeI32Vector(ctx.state, components),
	                              result, std::span(values).first(components));
	return result;
}

// IMAGE_SAMPLE*_O (RDNA 2 ISA 8.2.5): one dword of six-bit signed texel offsets, x in bits 5:0,
// y in 13:8 and z in 21:16, added to the texel coordinates of the level that is read. Vulkan
// accepts only constant offsets on sample instructions (ConstOffset), within
// [minTexelOffset, maxTexelOffset]; every implementation supports -8..7. Other offsets move the
// normalized coordinate by offset / level size instead (ApplyDynamicSampleOffset).
struct SampleOffset {
	uint32_t const_offset = 0;     // ConstOffset operand id; 0 when there is none
	uint32_t packed       = 0;     // the packed offsets when they need ApplyDynamicSampleOffset
	bool     dynamic      = false;
};

int32_t SignExtend6(uint32_t value) {
	return static_cast<int32_t>((value & 0x3fu) ^ 0x20u) - 0x20;
}

// The lanes of `exec` lie within those of `condition`: it is that value, or a conjunction holding it (a mask narrowed
// after the write: S_AND_SAVEEXEC is LogicalAnd(old EXEC, source)).
bool ExecWithin(const IR::Value& exec, const IR::Value& condition, uint32_t depth = 0u) {
	const auto value = exec.Resolve();
	if (value == condition) return true;
	const auto* conjunction = value.TryInstruction();
	if (conjunction == nullptr || conjunction->GetOpcode() != IR::ValueOpcode::LogicalAnd || depth >= 8u) return false;
	return ExecWithin(conjunction->Arg(0), condition, depth + 1u) ||
	       ExecWithin(conjunction->Arg(1), condition, depth + 1u);
}

SampleOffset SampleOffsetFor(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem,
                             const IR::Inst& address, const ImageSampleLayout& layout,
                             ImageDimension dimension, bool cube, bool bindless) {
	SampleOffset result;
	if (layout.offset == NoImageComponent || mem.image_address_components <= layout.offset) {
		return result;
	}
	{
		// Shaders that sample with offsets (first 64 instructions), for bisecting.
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 64u)
			std::fprintf(stderr, "SAMPLE_OFFSET hash=0x%016llx stage=%d\n",
			             static_cast<unsigned long long>(ctx.state.program.shader_hash),
			             static_cast<int>(ctx.state.program.stage));
	}
	// KYTY_SAMPLE_OFFSETS_ONLY=<hex,...>: offsets only in the listed shaders (bisecting a hang).
	static const std::string offsets_only = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_SAMPLE_OFFSETS_ONLY");
		return value != nullptr ? std::string(value) : std::string();
	}();
	bool listed = true;
	if (!offsets_only.empty()) {
		char text[17];
		std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(ctx.state.program.shader_hash));
		listed = offsets_only.find(text) != std::string::npos;
	}
	if (!listed) {
		return result;
	}
	if (cube) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			std::fputs("Warning: IMAGE_SAMPLE*_O offsets on a cube map are ignored.\n", stderr);
		}
		{
			static std::atomic_uint ignored {0}; // workaround C7: which shaders lose their offsets
			if (ignored.fetch_add(1) < 32u)
				std::fprintf(stderr, "SAMPLE_OFFSET_IGNORED case=%s hash=0x%016llx stage=%d\n","cube",
				             static_cast<unsigned long long>(ctx.state.program.shader_hash),
				             static_cast<int>(ctx.state.program.stage));
		}
		return result;
	}
	auto&      state      = ctx.state;
	const auto components = ImageDimensionInfoFor(dimension).spatial_components;
	const auto packed_index =
	    Decoder::ImageAddressComponentLayout(mem.image_sample_flags, mem.image_dimension, layout.offset).bit_offset / 32u;
	auto value =
	    packed_index < address.NumArgs() ? address.Arg(packed_index).Resolve() : IR::Value(0u);
	// Offsets are literals in guest code; under a non-uniform EXEC the register holds Select(condition, literal, old).
	// The hardware samples only the lanes of the instruction's EXEC, so where those lie within `condition` every sampled
	// lane reads the literal and `old` is never seen. EXEC is the instruction's own operand: the writes of the result do
	// not show it, their selects are gone once a store or the exec select pass proves them redundant.
	const auto exec = inst.Arg(inst.NumArgs() - 1u);
	while (!value.IsImmediate()) {
		const auto* select = value.TryInstruction();
		if (select == nullptr || select->GetOpcode() != IR::ValueOpcode::SelectU32 ||
		    !ExecWithin(exec, select->Arg(0).Resolve())) {
			break;
		}
		value = select->Arg(1).Resolve();
	}
	if (value.IsImmediate() &&
	    (value.GetType() == IR::Type::U32 || value.GetType() == IR::Type::F32)) {
		const auto bits = value.GetType() == IR::Type::U32 ? value.U32()
		                                                   : std::bit_cast<uint32_t>(value.F32Value());
		int32_t offsets[3] = {};
		bool    in_range   = true;
		bool    any        = false;
		for (uint32_t index = 0; index < components; index++) {
			offsets[index] = SignExtend6(bits >> (index * 8u));
			in_range       = in_range && offsets[index] >= -8 && offsets[index] <= 7;
			any            = any || offsets[index] != 0;
		}
		if (!any) {
			return result;
		}
		if (in_range) {
			uint32_t ids[3] = {};
			for (uint32_t index = 0; index < components; index++) {
				ids[index] = ConstantI32(state, offsets[index]);
			}
			if (components == 1u) {
				result.const_offset = ids[0];
			} else if (components == 2u) {
				result.const_offset = state.builder.Constant(
				    spv::OpConstantComposite, TypeI32Vector(state, 2), ids[0], ids[1]);
			} else {
				result.const_offset = state.builder.Constant(
				    spv::OpConstantComposite, TypeI32Vector(state, 3), ids[0], ids[1], ids[2]);
			}
			return result;
		}
	}
	// A bindless image takes the dynamic path below as well (workaround C7, fixed): its size and level queries
	// go through the set-1 array element of the instruction's slot (LoadImageDescriptor -> ImageDescriptorPointer, the
	// same descriptor the sample reads).
	// KYTY_SAMPLE_DYNAMIC_OFFSETS=1: non-constant or out-of-range offsets move the coordinate.
	// Off by default: a pixel shader (0x7de43c82ec18f609) hung the GPU with this build.
	static const bool dynamic_enabled = [] {
		const char* value = Common::ShaderEnv::Read("KYTY_SAMPLE_DYNAMIC_OFFSETS");
		return value != nullptr && value[0] == '1';
	}();
	// The level-zero dynamic offset: level-zero samples (IMAGE_SAMPLE*_LZ_O;
	// Tiled Decals CS e519fa9713f7b8e3 and b7c3bc6fcd140d60, PS 1cc2604335dc5dc2 and 1c26edd99e9c93cb) move the
	// coordinate without KYTY_SAMPLE_DYNAMIC_OFFSETS too. They read the view's base level (relative level 0, or the
	// _CL clamp), whose size ApplyDynamicSampleOffset queries directly, without the OpImageQueryLod of implicit-LOD
	// pixel-shader samples that the opt-in switch also guards: coordinate + offset / size is the hardware's texel
	// offset up to float rounding, for normalized coordinates (not for a FORCE_UNNORMALIZED S#, whose host sampler
	// takes texel coordinates).
	const bool applied = dynamic_enabled || HasFlag(mem, Decoder::ImageSampleFlagLevelZero);
	{
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 64u)
			std::fprintf(stderr, "SAMPLE_DYNAMIC_OFFSET hash=0x%016llx stage=%d bindless=%d %s\n",
			             static_cast<unsigned long long>(state.program.shader_hash),
			             static_cast<int>(state.program.stage), bindless ? 1 : 0, applied ? "applied" : "ignored");
	}
	if (!applied) return result;
	result.dynamic = true;
	result.packed  = AddressU32(ctx, mem, address, layout.offset);
	return result;
}

// Moves `coord` by the packed texel offsets divided by the size of the level `level_f32`
// (relative to the view's base, clamped to the view's levels). Exact for single-level reads up
// to float rounding; with linear mip filtering the coarser level moves by half the offset.
uint32_t ApplyDynamicSampleOffset(EmitterState& state, uint32_t resource, uint32_t coord,
                                  ImageDimension dimension, uint32_t packed, uint32_t level_f32) {
	static std::atomic_flag warned = ATOMIC_FLAG_INIT;
	if (!warned.test_and_set(std::memory_order_relaxed)) {
		std::fputs("Warning: IMAGE_SAMPLE*_O offsets that are not constant or lie outside -8..7 "
		           "move the coordinate by offset / level size.\n",
		           stderr);
	}
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto& info       = ImageDimensionInfoFor(dimension);
	const auto  components = info.spatial_components;
	const auto  image      = LoadImageDescriptor(state, resource);
	if (state.program.info.images[resource].bindless) { // a bindless image: a per-lane slot
		state.builder.AddAnnotation(spv::OpDecorate, image, spv::DecorationNonUniform);
	}
	const auto  levels     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQueryLevels, TypeU32(state), levels, image);
	const auto floored = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), floored, GlslStd450(state),
	                          GLSLstd450FMax, level_f32, ConstantF32Value(state, 0.0f));
	const auto level_float = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), level_float, GlslStd450(state),
	                          GLSLstd450Floor, floored);
	const auto level_u32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), level_u32, level_float);
	const auto last  = Binary(state, spv::OpISub, TypeU32(state), levels, ConstantU32(state, 1));
	const auto level = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), level, GlslStd450(state),
	                          GLSLstd450UMin, level_u32, last);
	const auto size = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, ImageViewSizeType(state, dimension), size,
	                          image, level);
	const auto packed_i32 = Unary(state, spv::OpBitcast, TypeI32(state), packed);
	uint32_t   moved[4]   = {};
	for (uint32_t index = 0; index < info.coordinate_components; index++) {
		auto component = coord;
		if (info.coordinate_components > 1u) {
			component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, coord,
			                          index);
		}
		if (index < components) {
			const auto offset = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldSExtract, TypeI32(state), offset, packed_i32,
			                          ConstantU32(state, index * 8u), ConstantU32(state, 6));
			const auto offset_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertSToF, TypeF32(state), offset_f32, offset);
			auto extent = size;
			if (info.coordinate_components > 1u) {
				extent = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), extent, size,
				                          index);
			}
			const auto extent_f32 = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), extent_f32, extent);
			component = Binary(state, spv::OpFAdd, TypeF32(state), component,
			                   Binary(state, spv::OpFDiv, TypeF32(state), offset_f32, extent_f32));
		}
		moved[index] = component;
	}
	if (info.coordinate_components == 1u) {
		return moved[0];
	}
	const auto result = state.builder.AllocateId();
	if (info.coordinate_components == 2u) {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), result,
		                          moved[0], moved[1]);
	} else {
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 3), result,
		                          moved[0], moved[1], moved[2]);
	}
	return result;
}

uint32_t HorizontalOffsets(EmitterState& state, ImageDimension dimension) {
	const auto components   = ImageDimensionInfoFor(dimension).spatial_components;
	const auto count        = ConstantU32(state, 4);
	const auto element_type = components == 1u ? TypeI32(state) : TypeI32Vector(state, 2);
	const auto array_type   = state.builder.Type(spv::OpTypeArray, element_type, count);
	uint32_t   offsets[4] {};
	for (uint32_t index = 0; index < 4u; index++) {
		const auto x = ConstantI32(state, static_cast<int32_t>(index) - 1);
		if (components == 1u) {
			offsets[index] = x;
		} else {
			offsets[index] = state.builder.Constant(
			    spv::OpConstantComposite, TypeI32Vector(state, 2), x, ConstantI32(state, 0));
		}
	}
	return state.builder.Constant(spv::OpConstantComposite, array_type, offsets[0], offsets[1],
	                              offsets[2], offsets[3]);
}

uint32_t InverseSwizzle(uint32_t swizzle, uint32_t component) {
	for (uint32_t source = 0; source < 4u; source++) {
		if (((swizzle >> (source * 3u)) & 7u) == 4u + component) return source;
	}
	return UINT32_MAX;
}

Format::BufferFormatInfo ImageConversionFormat(const EmitterState&   state,
                                               const IR::MemoryInfo& mem) {
	const auto& image = state.program.info.images[mem.resource];
	const auto format = image.conversion_format;
	if (format == Prospero::BufferFormat::kInvalid) return {};
	const auto info = Format::GetFormatInfo(format);
	if (IR::ReadsDepthBits(image)) return info;
	// The precompiler rejects a journal record whose image fails this (ImageConversionEmittable) before it gets here.
	EXIT_IF(!ImageConversionEmittable(image, format));
	return info;
}

uint32_t ExtractPackedImageComponent(EmitterState& state, const Format::BufferFormatInfo& info,
                                     uint32_t packed, uint32_t type, uint32_t component) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitFieldUExtract, type, result, packed,
	                          ConstantU32(state, info.component_bit_offset[component]),
	                          ConstantU32(state, info.component_bits[component]));
	return result;
}

uint32_t ImageGatherSource(const EmitterState& state, const IR::MemoryInfo& mem) {
	const auto component = ImageGatherComponent(mem.dmask);
	const auto info      = ImageConversionFormat(state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return component;
	if (info.packed_bitfield) return 0u;
	const auto selector =
	    (state.program.info.images[mem.resource].shader_swizzle >> (component * 3u)) & 7u;
	return selector >= 4u ? (selector - 4u) % info.component_count : 0u;
}

// The sRGB EOTF of the PS5 sampler for one UNorm-read value (8 / 8_8 sRGB textures, Format::IsSrgbFormat).
uint32_t SrgbToLinear(EmitterState& state, uint32_t value) {
	const auto type   = TypeF32(state);
	const auto linear = Binary(state, spv::OpFMul, type, value, ConstantF32Value(state, 1.0f / 12.92f));
	const auto base   = Binary(state, spv::OpFMul, type,
	                           Binary(state, spv::OpFAdd, type, value, ConstantF32Value(state, 0.055f)),
	                           ConstantF32Value(state, 1.0f / 1.055f));
	const auto curve  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, type, curve, GlslStd450(state), GLSLstd450Pow, base,
	                          ConstantF32Value(state, 2.4f));
	const auto low    = Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), value,
	                           ConstantF32Value(state, 0.04045f));
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, type, result, low, linear, curve);
	return result;
}

uint32_t UnpackImageTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t texel) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return texel;
	if (Format::IsSrgbFormat(info.format)) {
		// UNorm backing read as float: decode the colour channels, then the guest swizzle (constants 0 / 1 as is).
		uint32_t components[4] {};
		for (uint32_t component = 0; component < info.component_count; component++) {
			const auto value = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(ctx.state), value, texel, component);
			components[component] = SrgbToLinear(ctx.state, value);
		}
		const auto swizzle = ctx.state.program.info.images[mem.resource].shader_swizzle;
		uint32_t   selected[4] {};
		for (uint32_t component = 0; component < 4u; component++) {
			const auto selector = (swizzle >> (component * 3u)) & 7u;
			selected[component] = selector == 1u ? ConstantF32Value(ctx.state, 1.0f)
			                      : selector >= 4u && selector - 4u < info.component_count
			                          ? components[selector - 4u]
			                          : ZeroF32(ctx.state);
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(ctx.state, 4), result, selected[0],
		                              selected[1], selected[2], selected[3]);
		return result;
	}

	const auto numeric_class = ctx.state.program.info.images[mem.resource].numeric_class;
	const auto scalar_type   = ImageScalarType(ctx.state, numeric_class);
	const bool integer       = info.packed_bitfield || info.type == Format::ComponentType::Uint;
	uint32_t   packed        = 0;
	if (info.packed_bitfield) {
		packed = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, packed, texel, 0u);
	}
	uint32_t components[4] {};
	for (uint32_t component = 0; component < info.component_count; component++) {
		if (info.packed_bitfield) {
			components[component] =
			    ExtractPackedImageComponent(ctx.state, info, packed, scalar_type, component);
			components[component] =
			    NormalizeFormatComponent(ctx.state, info, component, components[component]);
		} else {
			components[component] = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, scalar_type,
			                              components[component], texel, component);
			// UNorm backing preserves the guest's filtering; scaling precedes swizzle constants.
			const auto unorm_max = static_cast<float>((1u << info.component_bits[component]) - 1u);
			components[component] = Binary(ctx.state, spv::OpFMul, scalar_type, components[component],
			                               ConstantF32Value(ctx.state, unorm_max));
			if (integer) {
				// Depth bits: round back to the stored UNorm code.
				components[component] =
				    Unary(ctx.state, spv::OpConvertFToU, TypeU32(ctx.state),
				          Binary(ctx.state, spv::OpFAdd, scalar_type, components[component],
				                 ConstantF32Value(ctx.state, 0.5f)));
			}
		}
	}
	for (uint32_t component = info.component_count; component < 4u; component++) {
		components[component] = components[component % info.component_count];
	}

	const auto swizzle = ctx.state.program.info.images[mem.resource].shader_swizzle;
	uint32_t   selected[4] {};
	for (uint32_t component = 0; component < 4u; component++) {
		const auto selector = (swizzle >> (component * 3u)) & 7u;
		if (selector == 1u) {
			selected[component] = integer
			    ? ConstantU32(ctx.state, Format::FormattedConstantBits(info, Format::FormattedSourceKind::One))
			    : ConstantF32Value(ctx.state, 1.0f);
		} else if (selector >= 4u) {
			selected[component] = components[selector - 4u];
		} else {
			selected[component] = integer ? ConstantU32(ctx.state, 0u) : ZeroF32(ctx.state);
		}
	}
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              integer ? TypeU32Vector(ctx.state, 4) : TypeF32Vector(ctx.state, 4),
	                              result, selected[0], selected[1], selected[2], selected[3]);
	return info.type == Format::ComponentType::Unorm
	           ? Unary(ctx.state, spv::OpBitcast, TypeF32Vector(ctx.state, 4), result) : result;
}

uint32_t UnpackImageGather(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t gathered) {
	const auto info = ImageConversionFormat(ctx.state, mem);
	if (info.format == Prospero::BufferFormat::kInvalid) return gathered;

	const auto component = ImageGatherComponent(mem.dmask);
	const auto selector =
	    (ctx.state.program.info.images[mem.resource].shader_swizzle >> (component * 3u)) & 7u;
	const auto numeric_class = ctx.state.program.info.images[mem.resource].numeric_class;
	const auto vector_type   = ImageVectorType(ctx.state, numeric_class, 4);
	if (selector < 4u) {
		uint32_t value;
		switch (info.type) {
			case Format::ComponentType::Uscaled:
				value = ConstantF32Value(ctx.state, selector == 1u ? 1.0f : 0.0f);
				break;
			default: value = ConstantU32(ctx.state, selector == 1u ? 1u : 0u); break;
		}
		return ctx.state.builder.Constant(spv::OpConstantComposite, vector_type, value, value,
		                                  value, value);
	}
	if (Format::IsSrgbFormat(info.format)) {
		// One stored channel of four texels: each decoded from sRGB (every channel of 8 / 8_8 is colour).
		uint32_t texels[4] {};
		for (uint32_t index = 0; index < 4u; index++) {
			const auto value = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(ctx.state), value, gathered, index);
			texels[index] = SrgbToLinear(ctx.state, value);
		}
		const auto result = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, result, texels[0], texels[1],
		                              texels[2], texels[3]);
		return result;
	}
	if (!info.packed_bitfield) {
		const auto scale  = ConstantF32Value(ctx.state, 255.0f);
		const auto scales = ctx.state.builder.Constant(spv::OpConstantComposite, vector_type,
		                                                scale, scale, scale, scale);
		return Binary(ctx.state, spv::OpFMul, vector_type, gathered, scales);
	}

	return ExtractPackedImageComponent(ctx.state, info, gathered, TypeU32Vector(ctx.state, 4),
	                                   (selector - 4u) % info.component_count);
}

uint32_t SamplePackedUnormLz(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t coord) {
	auto& state = ctx.state;
	const auto& image = state.program.info.images[mem.resource];
	const auto info = ImageConversionFormat(state, mem);
	const auto sampled = MakeSampledImage(state, mem.resource,
	                                     LoadSamplerDescriptor(state, mem.sampler));
	const auto gathered = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageGather, TypeU32Vector(state, 4), gathered, sampled,
	                          coord, ConstantU32(state, 0u));
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto size = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, TypeU32Vector(state, 2), size,
	                          LoadImageDescriptor(state, mem.resource), ConstantU32(state, 0u));
	const auto size_f32 = Unary(state, spv::OpConvertUToF, TypeF32Vector(state, 2), size);
	const auto half = ConstantF32Value(state, 0.5f);
	const auto halves = state.builder.Constant(spv::OpConstantComposite,
	                                           TypeF32Vector(state, 2), half, half);
	const auto fractions = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpExtInst, TypeF32Vector(state, 2), fractions, GlslStd450(state), GLSLstd450Fract,
	    Binary(state, spv::OpFSub, TypeF32Vector(state, 2),
	           Binary(state, spv::OpFMul, TypeF32Vector(state, 2), coord, size_f32), halves));
	uint32_t fraction[2];
	uint32_t complement[2];
	for (uint32_t axis = 0; axis < 2u; axis++) {
		fraction[axis] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), fraction[axis],
		                          fractions, axis);
		complement[axis] = Binary(state, spv::OpFSub, TypeF32(state),
		                          ConstantF32Value(state, 1.0f), fraction[axis]);
	}
	// Gather order is (x0, y1), (x1, y1), (x1, y0), (x0, y0).
	const auto weights = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpCompositeConstruct, TypeF32Vector(state, 4), weights,
	    Binary(state, spv::OpFMul, TypeF32(state), complement[0], fraction[1]),
	    Binary(state, spv::OpFMul, TypeF32(state), fraction[0], fraction[1]),
	    Binary(state, spv::OpFMul, TypeF32(state), fraction[0], complement[1]),
	    Binary(state, spv::OpFMul, TypeF32(state), complement[0], complement[1]));
	uint32_t decoded[4] {};
	uint32_t selected[4] {ZeroF32(state), ZeroF32(state), ZeroF32(state), ZeroF32(state)};
	for (uint32_t component = 0; component < 4u; component++) {
		if ((mem.dmask & (1u << component)) == 0u) continue;
		const auto selector = (image.shader_swizzle >> (component * 3u)) & 7u;
		if (selector == 1u) {
			selected[component] = ConstantF32Value(state, 1.0f);
		} else if (selector >= 4u) {
			const auto physical = (selector - 4u) % info.component_count;
			if (decoded[physical] == 0u) {
				const auto texels = Unary(
				    state, spv::OpConvertUToF, TypeF32Vector(state, 4),
				    ExtractPackedImageComponent(state, info, gathered,
				                                TypeU32Vector(state, 4), physical));
				decoded[physical] = Binary(
				    state, spv::OpFMul, TypeF32(state),
				    Binary(state, spv::OpDot, TypeF32(state), texels, weights),
				    ConstantF32Value(state, 1.0f / ((1u << info.component_bits[physical]) - 1u)));
			}
			selected[component] = decoded[physical];
		}
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), result,
	                          selected[0], selected[1], selected[2], selected[3]);
	return result;
}

// A finite sampler: the handle's key already is the candidate's ordinal, with no table behind it.
bool FiniteSampler(const IR::SamplerResource& sampler) {
	return sampler.indirect_resources.size() >= 2u && sampler.indirect_root == IR::SamplerResource::NoIndirectSampler;
}

// A GPU-selected sampler resolves its key through the flattened key/ordinal mapping; its
// children follow the root contiguously in the sampler binding.
uint32_t LoadInstructionSampler(ValueEmitContext& ctx, const IR::Inst& inst, const IR::MemoryInfo& mem) {
	auto& state = ctx.state;
	const auto& sampler = state.program.info.samplers.at(mem.sampler);
	if (sampler.indirect_resources.size() < 2u) return LoadSamplerDescriptor(state, mem.sampler);
	const bool finite = FiniteSampler(sampler);
	const auto* handle = inst.NumArgs() > 1u ? inst.Arg(1).ResolveInstruction() : nullptr;
	if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetSamplerResource ||
	    handle->NumArgs() == 0u || (!finite && state.flattened_srt_variable == 0)) {
		ctx.Fail(inst, "has invalid GPU-selected sampler key provenance");
		return 0u;
	}
	const auto& resources = sampler.indirect_resources;
	if (finite) {
		// Literal candidates sit wherever AddSampler and the class split put them, often not side by side.
		const auto slot = EmitIndexSwitch(
		    state, ctx.Def(handle->Arg(0)), static_cast<uint32_t>(resources.size()), TypeU32(state), [&](uint32_t ordinal) {
			    return ConstantU32(state, ResourceForDescriptor(state, IR::DescriptorBindingKind::Samplers, resources[ordinal]));
		    });
		return LoadSamplerDescriptorAt(state, mem.sampler, slot);
	}
	const auto root_slot = ResourceForDescriptor(state, IR::DescriptorBindingKind::Samplers, resources[0]);
	const auto first_child = ResourceForDescriptor(state, IR::DescriptorBindingKind::Samplers, resources[1]);
	for (uint32_t ordinal = 2; ordinal < resources.size(); ++ordinal) {
		if (ResourceForDescriptor(state, IR::DescriptorBindingKind::Samplers, resources[ordinal]) !=
		    first_child + ordinal - 1u) {
			ctx.Fail(inst, "has GPU-selected sampler children outside one binding range");
			return 0u;
		}
	}
	const auto selected = EmitIndirectResourceIndex(state, ctx.Def(handle->Arg(0)),
	                                                sampler.indirect_mapping_offset,
	                                                sampler.indirect_search_iterations, 0u);
	const auto is_root = Binary(state, spv::OpIEqual, TypeBool(state), selected, ConstantU32(state, 0u));
	const auto slot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, is_root, ConstantU32(state, root_slot),
	                          Binary(state, spv::OpIAdd, TypeU32(state), selected,
	                                 ConstantU32(state, first_child - 1u)));
	return LoadSamplerDescriptorAt(state, mem.sampler, slot);
}

// KYTY_BINDLESS: the set-1 slot of the image handle's key. The key is the byte offset of the
// descriptor in its heap; entry = key / 4 indexes the heap's translation region (base and entry
// count from the flattened SRT, patched by the host). A pending entry is flagged in the feedback
// buffer for the host to resolve and samples slot 2 (placeholder) meanwhile; keys outside the
// heap use translation[0].
static uint32_t BindlessSlot(ValueEmitContext& ctx, const IR::ImageResource& image, IR::Value image_arg) {
	auto&       state  = ctx.state;
	const auto* handle = image_arg.ResolveInstruction();
	EXIT_IF(handle == nullptr || handle->NumArgs() == 0u || state.flattened_srt_variable == 0 ||
	        state.bindless_translation_variable == 0);
	const auto Load = [&](uint32_t variable, uint32_t index) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
		                          variable, ConstantU32(state, 0), index);
		const auto value = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
		return value;
	};
	// Heap tables key by byte offset; record tables key by record index.
	const auto entry_key =
	    image.bindless_stride_dwords == 0u
	        ? Binary(state, spv::OpShiftRightLogical, TypeU32(state), ctx.Def(handle->Arg(0)),
	                 ConstantU32(state, 2u))
	        : Binary(state, spv::OpIMul, TypeU32(state), ctx.Def(handle->Arg(0)),
	                 ConstantU32(state, image.bindless_stride_dwords));
	const auto base      = Load(state.flattened_srt_variable, ConstantU32(state, image.indirect_mapping_offset));
	const auto count     = Load(state.flattened_srt_variable, ConstantU32(state, image.indirect_mapping_offset + 1u));
	const auto in_range  = Binary(state, spv::OpULessThan, TypeBool(state), entry_key, count);
	const auto index     = Select(state, TypeU32(state), in_range,
	                              Binary(state, spv::OpIAdd, TypeU32(state), base, entry_key),
	                              ConstantU32(state, 0u));
	const auto raw       = Load(state.bindless_translation_variable, index);
	const auto pending   = Binary(state, spv::OpIEqual, TypeBool(state), raw,
	                              ConstantU32(state, IR::BindlessPending));
	if (state.bindless_feedback_variable != 0) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
		                          state.bindless_feedback_variable, ConstantU32(state, 0), index);
		state.builder.AddFunction(spv::OpStore, pointer,
		                          Select(state, TypeU32(state), pending, ConstantU32(state, 1u),
		                                 ConstantU32(state, 0u)));
	}
	const auto slot = state.builder.AllocateId();
	if (image.numeric_class == Prospero::TextureNumericClass::Uint ||
	    image.numeric_class == Prospero::TextureNumericClass::Sint) {
		// KYTY_PERF_BINDLESS_CLASSES: an integer array access never reaches the float placeholders (slots 0..2, and
		// translation[0] = slot 1): a pending or out-of-heap key reads the zero placeholder of its class.
		const auto zero  = ConstantU32(state, image.numeric_class == Prospero::TextureNumericClass::Uint
		                                          ? IR::BindlessUintPlaceholder
		                                          : IR::BindlessSintPlaceholder);
		const auto known = Select(state, TypeU32(state), pending, zero, raw);
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, in_range, known, zero);
	} else {
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), slot, pending, ConstantU32(state, 2u), raw);
	}
	if (IR::BindlessPlaceholderCensus() && state.fault_buffer_variable != 0u) { // KYTY_DIAG_BINDLESS_PLACEHOLDER_CENSUS
		const auto u32      = TypeU32(state);
		const auto below    = [&](uint32_t n) { return Binary(state, spv::OpULessThan, TypeBool(state), slot, ConstantU32(state, n)); };
		const auto category = Select(state, u32, below(3u), slot,
		                             Select(state, u32, below(5u), ConstantU32(state, 3u), ConstantU32(state, 4u)));
		state.builder.RequireCapability(spv::CapabilityGroupNonUniformArithmetic);
		std::array<uint32_t, IR::BindlessCensusWords> lanes {};
		for (uint32_t k = 0; k < IR::BindlessCensusWords; k++) {
			lanes[k] = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpGroupNonUniformIAdd, u32, lanes[k], ConstantU32(state, spv::ScopeSubgroup),
			                          spv::GroupOperationReduce,
			                          Select(state, u32, Binary(state, spv::OpIEqual, TypeBool(state), category, ConstantU32(state, k)),
			                                 ConstantU32(state, 1u), ConstantU32(state, 0u)));
		}
		const auto elected = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformElect, TypeBool(state), elected, ConstantU32(state, spv::ScopeSubgroup));
		EmitIfCondition(state, elected, [&]() {
			for (uint32_t k = 0; k < IR::BindlessCensusWords; k++) FaultDiagAdd(state, IR::BindlessCensusWord + k, lanes[k]);
		});
	}
	state.builder.AddAnnotation(spv::OpDecorate, slot, spv::DecorationNonUniform);
	return slot;
}

uint32_t EmitOneDimensionalGatherLz(ValueEmitContext& ctx, const IR::MemoryInfo& mem,
                                    uint32_t coord, Prospero::TextureNumericClass numeric_class) {
	auto& state = ctx.state;
	state.builder.RequireCapability(spv::CapabilityImageQuery);
	const auto image = LoadImageDescriptor(state, mem.resource);
	const auto width = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpImageQuerySizeLod, TypeU32(state), width, image,
	                          ConstantU32(state, 0));
	const auto width_f32 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToF, TypeF32(state), width_f32, width);
	const auto left = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32(state), left, GlslStd450(state),
	                          GLSLstd450Floor,
	                          Binary(state, spv::OpFSub, TypeF32(state),
	                                 Binary(state, spv::OpFMul, TypeF32(state), coord, width_f32),
	                                 ConstantF32(state, 0x3f000000u)));

	const auto sampled = MakeSampledImage(state, mem.resource,
	                                     LoadSamplerDescriptor(state, mem.sampler));
	const auto vector_type = ImageVectorType(state, numeric_class, 4);
	const auto scalar_type = ImageScalarType(state, numeric_class);
	const auto component = ImageGatherSource(state, mem);
	uint32_t values[2] {};
	for (uint32_t index = 0; index < 2u; index++) {
		const auto sample_coord =
		    Binary(state, spv::OpFDiv, TypeF32(state),
		           Binary(state, spv::OpFAdd, TypeF32(state), left,
		                  ConstantF32(state, index == 0u ? 0x3f000000u : 0x3fc00000u)),
		           width_f32);
		const auto texel = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpImageSampleExplicitLod, vector_type, texel, sampled,
		                          sample_coord, spv::ImageOperandsLodMask, ZeroF32(state));
		values[index] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, scalar_type, values[index], texel,
		                          component);
	}
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, vector_type, result, values[0], values[1],
	                          values[1], values[0]);
	return result;
}

// `resource`: the image whose store swizzle applies (a GPU-selected candidate under indirect image writes;
// UINT32_MAX = mem.resource). A converted format (mem.resource's conversion_format) is encoded here, as upstream
// c11254a99 / fb213e8ae / c88f05f7b: packed UInt/UNorm channels into one dword, raw signed storage per channel.
uint32_t StoreTexel(ValueEmitContext& ctx, const IR::MemoryInfo& mem, uint32_t data, bool integer,
                    uint32_t resource = UINT32_MAX) {
	const auto swizzle =
	    ctx.state.program.info.images[resource == UINT32_MAX ? mem.resource : resource].shader_swizzle;
	const auto info      = ImageConversionFormat(ctx.state, mem);
	const bool converted = info.format != Prospero::BufferFormat::kInvalid;
	const bool packed    = info.packed_bitfield;
	EXIT_IF(converted && info.type != Format::ComponentType::Sint &&
	        (!packed || (info.type != Format::ComponentType::Uint &&
	                     info.type != Format::ComponentType::Unorm)));
	const auto zero = ConstantU32(ctx.state, 0);
	uint32_t   values[4] {zero, zero, zero, zero};
	auto       packed_value = zero;
	const auto dmask = mem.dmask != 0u ? mem.dmask : 1u;
	for (uint32_t component = 0; component < (converted ? info.component_count : 4u); component++) {
		const auto source   = InverseSwizzle(swizzle, component);
		const bool selected = source < 4u && ((dmask >> source) & 1u) != 0u;
		uint32_t   raw      = zero;
		if (selected) {
			const auto packed_index = DmaskComponentIndex(dmask, source);
			raw                     = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), raw, data,
			                              mem.data_bits == 16u ? packed_index / 2u : packed_index);
			if (mem.data_bits == 16u) {
				if ((packed_index & 1u) != 0u) {
					raw = Binary(ctx.state, spv::OpShiftRightLogical, TypeU32(ctx.state), raw,
					             ConstantU32(ctx.state, 16u));
				}
				if (!packed) {
					raw = Binary(ctx.state, spv::OpBitwiseAnd, TypeU32(ctx.state), raw,
					             ConstantU32(ctx.state, 0xffffu));
				}
			}
		}
		if (converted) {
			if (!selected) continue;
			if (!packed) {
				values[component] =
				    EncodeFormattedStoreComponent(ctx.state, info, component, raw, mem.data_bits);
				continue;
			}
			const auto value =
			    EncodePackedColorComponent(ctx.state, info, component, raw, mem.data_bits);
			packed_value = packed_value == zero ? value
			    : Binary(ctx.state, spv::OpBitwiseOr, TypeU32(ctx.state), packed_value, value);
			continue;
		}
		values[component] = integer ? raw
		                    : mem.data_bits == 16u
		                        ? EmitF16BitsToF32(ctx.state, raw)
		                        : Unary(ctx.state, spv::OpBitcast, TypeF32(ctx.state), raw);
	}
	if (packed) values[0] = packed_value;
	const auto texel = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeConstruct,
	                              integer ? TypeU32Vector(ctx.state, 4)
	                                      : TypeF32Vector(ctx.state, 4),
	                              texel, values[0], values[1], values[2], values[3]);
	return texel;
}

// `emit` returns the access's value id, or nothing for a write (indirect image writes; result_type unused).
template <typename Emit>
auto EmitImageAccess(ValueEmitContext& ctx, const IR::Inst& inst, uint32_t result_type, const Emit& emit)
    -> std::invoke_result_t<const Emit&, uint32_t, uint32_t> {
	auto& state = ctx.state;
	const auto& mem = ctx.Memory(inst);
	const auto& image = state.program.info.images.at(mem.resource);
	// A bindless image selects its heap slot itself (state.bindless_slot, see BindlessSlot).
	if (image.indirect_root != mem.resource || image.bindless) return emit(mem.resource, 0u);
	// One candidate: every key selects the root.
	if (image.indirect_resources.size() < 2u) return emit(mem.resource, 0u);
	const auto* handle = inst.Arg(0).ResolveInstruction();
	const auto* source = image.source < state.program.descriptor_sources.size()
	                         ? &state.program.descriptor_sources[image.source]
	                         : nullptr;
	if (handle == nullptr || source == nullptr || !source->indirect_descriptor.has_value() ||
	    handle->NumArgs() == 0u) {
		ctx.Fail(inst, "has invalid indirect image key provenance");
	}
	const auto key = ctx.Def(handle->Arg(0));
	if (state.flattened_srt_variable == 0 ||
	    image.indirect_resources.size() < 2u) {
		ctx.Fail(inst, "has no indirect image runtime mapping");
	}
	const auto selected = EmitIndirectResourceIndex(
	    state, key, image.indirect_mapping_offset, image.indirect_search_iterations, 0u);
	struct ImageRun {
		uint32_t first;
		uint32_t count;
		uint32_t resource;
		uint32_t slot_bias;
	};
	std::vector<ImageRun> runs;
	for (uint32_t ordinal = 0; ordinal < image.indirect_resources.size(); ++ordinal) {
		const auto resource = image.indirect_resources[ordinal];
		const auto& candidate = state.program.info.images[resource];
		const auto kind = *IR::DescriptorBindingForImage(candidate);
		if (!runs.empty()) {
			auto& run = runs.back();
			const auto& first = state.program.info.images[run.resource];
			const auto next_slot = run.slot_bias + ordinal;
			if (candidate.dimension == first.dimension && candidate.cube == first.cube &&
			    IR::DescriptorBindingForImage(first) == kind &&
			    candidate.mip_count == 1u && first.mip_count == 1u &&
			    (ordinal == 1u || candidate.descriptor_index == next_slot)) {
				// Native roots precede appended children; only the root may have a slot gap.
				if (ordinal == 1u)
					run.slot_bias = candidate.descriptor_index - ordinal;
				++run.count;
				continue;
			}
		}
		runs.push_back({ordinal, 1u, resource,
		                candidate.descriptor_index - ordinal});
	}
	const auto EmitRun = [&](uint32_t index) {
		const auto& run = runs[index];
		if (run.count == 1u) return emit(run.resource, 0u);
		auto slot = Binary(state, spv::OpIAdd, TypeU32(state), selected,
		                   ConstantU32(state, run.slot_bias));
		if (run.first == 0u) {
			const auto root_slot = image.descriptor_index;
			if (root_slot != run.slot_bias) {
				const auto is_root = Binary(state, spv::OpIEqual, TypeBool(state), selected,
				                            ConstantU32(state, 0u));
				const auto root_or_child = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpSelect, TypeU32(state), root_or_child,
				                          is_root, ConstantU32(state, root_slot), slot);
				slot = root_or_child;
			}
		}
		return emit(run.resource, slot);
	};
	auto run_index = ConstantU32(state, 0u);
	for (uint32_t index = 1; index < runs.size(); ++index) {
		const auto at_or_after = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state),
		                               selected, ConstantU32(state, runs[index].first));
		const auto next = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), next, at_or_after,
		                          ConstantU32(state, index), run_index);
		run_index = next;
	}
	if (runs.size() == 1u) return EmitRun(0u);
	return EmitIndexSwitch(state, run_index, static_cast<uint32_t>(runs.size()), result_type, EmitRun);
}

} // namespace

void EmitImage(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto op         = inst.GetOpcode();
	const auto image_info = IR::ImageOpcodeInfoOf(op);
	auto&       state     = ctx.state;
	const auto& mem       = ctx.Memory(inst);
	const auto  image_arg = inst.Arg(0);
	ctx.ResourceIndex(image_arg, IR::ValueOpcode::GetImageResource);
	const auto& image   = state.program.info.images.at(mem.resource);
	state.bindless_slot = image.bindless ? BindlessSlot(ctx, image, image_arg) : 0u;
	const auto* address = ctx.ImageAddress(inst.Arg(image_info.needs_sampler ? 2 : 1));
	if (op == IR::ValueOpcode::ImageQueryDimensions) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		ctx.Define(inst, QueryDimensions(ctx, mem, *address));
		return;
	}
	// Packed 11-11-10 UNorm sampled through its R32_UINT backing (upstream 56738aa4e): one gather and the bilinear
	// weights in the shader.
	if (image_info.needs_sampler &&
	    image.conversion_format == Prospero::BufferFormat::k11_11_10UNorm) {
		if (op != IR::ValueOpcode::ImageSampleRaw ||
		    image.dimension != ImageDimension::Dim2D ||
		    mem.image_dimension != ImageDimension::Dim2D || mem.data_bits != 32u ||
		    mem.image_sample_flags != Decoder::ImageSampleFlagLevelZero ||
		    image.indirect_root != IR::ImageResource::NoIndirectImage ||
		    !state.program.info.samplers.at(mem.sampler).indirect_resources.empty()) {
			ctx.Fail(inst, "packed UNorm sampling requires a direct 2D sample_lz");
			return;
		}
		const auto coord = CoordF32(ctx, mem, *address, Layout(mem).coord, 2u);
		ctx.Define(inst, ResultVector(ctx, SamplePackedUnormLz(ctx, mem, coord),
		                              Prospero::TextureNumericClass::Float, false, mem));
		return;
	}
	if (op == IR::ValueOpcode::ImageQueryLod) {
		state.builder.RequireCapability(spv::CapabilityImageQuery);
		const auto dimension = image.dimension;
		const auto sampled = MakeSampledImage(state, mem.resource,
		                                     LoadInstructionSampler(ctx, inst, mem));
		const auto lod       = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpImageQueryLod, TypeF32Vector(state, 2), lod, sampled,
		    CoordF32(ctx, mem, *address, 0, ImageDimensionInfoFor(dimension).spatial_components,
		             image.cube));
		uint32_t values[4] = {ConstantU32(state, 0), ConstantU32(state, 0), ConstantU32(state, 0),
		                      ConstantU32(state, 0)};
		for (uint32_t index = 0; index < 2u; index++) {
			const auto component = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), component, lod,
			                          index);
			values[index] = FloatBits(ctx, component);
		}
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result,
		                          values[0], values[1], values[2], values[3]);
		ctx.Define(inst, result);
		return;
	}
	if (op == IR::ValueOpcode::ImageRead) {
		const auto numeric_class = image.numeric_class;
		const auto condition = ctx.Arg(inst, 2);
		ctx.Define(
		    inst,
		    EmitValueOrDefaultIfCondition(
		        state, condition, TypeU32Vector(state, 4), ConstantU32CompositeZero(state, 4),
		        [&]() {
			        const auto color = EmitImageAccess(
			            ctx, inst, ImageVectorType(state, numeric_class, 4),
			            [&](uint32_t resource, uint32_t array_index) {
				        const auto& candidate = state.program.info.images[resource];
				        const auto dimension = candidate.dimension;
				        const auto& dimension_info = ImageDimensionInfoFor(dimension);
				        const auto descriptor = LoadImageDescriptor(state, resource, 0u, array_index);
				        const auto fetched = state.builder.AllocateId();
				        const auto coord = CoordU32(ctx, mem, *address, dimension);
				        if (dimension_info.multisampled != 0u) {
					        state.builder.AddFunction(
					            spv::OpImageFetch, ImageVectorType(state, candidate.numeric_class, 4), fetched,
					            descriptor, coord, spv::ImageOperandsSampleMask,
					            AddressU32(ctx, mem, *address, dimension_info.coordinate_components));
				        } else {
					        state.builder.AddFunction(
					            spv::OpImageFetch, ImageVectorType(state, candidate.numeric_class, 4), fetched,
					            descriptor, coord, spv::ImageOperandsLodMask,
					            LodU32(ctx, mem, *address, mem.image_dimension));
				        }
				        return fetched;
			            });
			        return ResultVector(ctx, UnpackImageTexel(ctx, mem, color), numeric_class,
			                            false, mem);
		        }));
		return;
	}
	if (op == IR::ValueOpcode::ImageWrite) {
		const bool uint_image = image.numeric_class == Prospero::TextureNumericClass::Uint;
		const auto dimension  = image.dimension;
		// A write to a GPU-selected table goes to the candidate its key selects (as ImageRead reads it), with that
		// candidate's coordinates, mip and store swizzle.
		if (image.indirect_root == mem.resource && !image.bindless &&
		    image.indirect_resources.size() >= 2u) {
			const auto data = ctx.Arg(inst, 2);
			EmitIfCondition(state, ctx.Arg(inst, 3), [&]() {
				EmitImageAccess(ctx, inst, 0u, [&](uint32_t resource, uint32_t array_index) {
					const auto& candidate           = state.program.info.images[resource];
					const auto  candidate_dimension = candidate.dimension;
					EXIT_IF(candidate.resource_class != IR::ImageResourceClass::Storage ||
					        candidate.numeric_class != image.numeric_class);
					const auto mip_lod = candidate.mip_mode == IR::ImageMipMode::Dynamic
					                         ? LodU32(ctx, mem, *address, candidate_dimension)
					                         : 0u;
					const auto coord   = CoordU32(ctx, mem, *address, candidate_dimension);
					const auto texel   = StoreTexel(ctx, mem, data, uint_image, resource);
					// A 2D MSAA storage image takes the sample index (upstream 79837b423).
					const auto& candidate_info = ImageDimensionInfoFor(candidate_dimension);
					const auto  sample = candidate_info.multisampled != 0u
					                         ? AddressU32(ctx, mem, *address, candidate_info.coordinate_components)
					                         : 0u;
					if (array_index == 0u) {
						EmitStorageImageWrite(state, resource, mip_lod, coord, texel, 0u, sample);
						return;
					}
					// A run of single-mip candidates in one binding (EmitImageAccess): the slot is a runtime index.
					state.builder.RequireCapability(spv::CapabilityStorageImageWriteWithoutFormat);
					const auto descriptor = LoadImageDescriptor(state, resource, 0u, array_index);
					if (sample == 0u) {
						state.builder.AddFunction(spv::OpImageWrite, descriptor, coord, texel);
					} else {
						state.builder.AddFunction(spv::OpImageWrite, descriptor, coord, texel,
						                          spv::ImageOperandsSampleMask, sample);
					}
				});
			});
			return;
		}
		EmitIfCondition(state, ctx.Arg(inst, 3), [&]() {
			const auto mip_lod =
			    state.program.info.images[mem.resource].mip_mode == IR::ImageMipMode::Dynamic
			        ? LodU32(ctx, mem, *address, dimension)
			        : 0u;
			const auto coord = CoordU32(ctx, mem, *address, dimension);
			const auto texel = StoreTexel(ctx, mem, ctx.Arg(inst, 2), uint_image);
			// A 2D MSAA storage image takes the sample index (upstream 79837b423).
			const auto& dimension_info = ImageDimensionInfoFor(dimension);
			const auto  sample         = dimension_info.multisampled != 0u
			                                 ? AddressU32(ctx, mem, *address, dimension_info.coordinate_components)
			                                 : 0u;
			EmitStorageImageWrite(state, mem.resource, mip_lod, coord, texel, 0u, sample);
		});
		return;
	}
	if (op == IR::ValueOpcode::ImageSampleRaw || op == IR::ValueOpcode::ImageGatherRaw) {
		const auto  dimension      = image.dimension;
		const auto& dimension_info = ImageDimensionInfoFor(dimension);
		const auto  layout         = Layout(mem);
		const auto  numeric_class  = image.numeric_class;
		const bool  dref           = HasFlag(mem, Decoder::ImageSampleFlagCompare);
		IR::Value offset_value;
		if (layout.offset != NoImageComponent) {
			const auto value = address->Arg(layout.offset).Resolve();
			offset_value = IR::ResolveActiveU32(value, inst.Arg(inst.NumArgs() - 1u).Resolve());
			if (offset_value.IsEmpty()) offset_value = value;
		}
		if (dref && state.program.info.images[mem.resource].conversion_format !=
		                Prospero::BufferFormat::kInvalid) {
			ctx.Fail(inst, "uses depth comparison with a converted image");
			return;
		}
		if (op == IR::ValueOpcode::ImageGatherRaw) {
			if (dimension == ImageDimension::Dim1D) {
				if (image.indirect_root == mem.resource || dref ||
				    !HasFlag(mem, Decoder::ImageSampleFlagLevelZero) ||
				    HasFlag(mem, Decoder::ImageSampleFlagOffset) ||
				    HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
					ctx.Fail(inst, "has an unsupported 1D gather variant");
					return;
				}
				const auto coord = CoordF32(ctx, mem, *address, layout.coord,
				                            dimension_info.coordinate_components, image.cube);
				const auto sample = EmitOneDimensionalGatherLz(ctx, mem, coord, numeric_class);
				ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
				                              numeric_class, false, mem, true));
				return;
			}
			if (dimension == ImageDimension::Dim1DArray) {
				ctx.Fail(inst, "has an unsupported 1D-array gather");
				return;
			}
			const auto result_numeric_class = dref ? Prospero::TextureNumericClass::Float
			                                       : numeric_class;
			const auto result_type = ImageVectorType(state, result_numeric_class, 4);
			uint32_t component_or_dref;
			if (dref) {
				component_or_dref = layout.dref != NoImageComponent
				                        ? AddressF32(ctx, mem, *address, layout.dref)
				                        : ZeroF32(state);
			} else {
				component_or_dref = ConstantU32(state, ImageGatherSource(state, mem));
			}
			const auto sampler_id = LoadInstructionSampler(ctx, inst, mem);
			const auto EmitGather = [&](uint32_t resource, uint32_t array_index) {
				const auto& candidate = state.program.info.images[resource];
				const auto dimension = candidate.dimension;
				const auto coord = CoordF32(
				    ctx, mem, *address, layout.coord,
				    ImageDimensionInfoFor(dimension).coordinate_components, candidate.cube);
				uint32_t operand_mask = 0;
				uint32_t offset = 0;
				if (HasFlag(mem, Decoder::ImageSampleFlagGatherHorizontal)) {
					operand_mask = spv::ImageOperandsConstOffsetsMask;
					offset = HorizontalOffsets(state, dimension);
				} else if (layout.offset != NoImageComponent) {
					offset = PackedOffset(ctx, offset_value, dimension, operand_mask);
				}
				const auto EmitMip = [&](uint32_t mip) {
					const auto sampled = MakeSampledImage(state, resource, sampler_id, mip, array_index);
					const auto sample = state.builder.AllocateId();
					const std::array operands {operand_mask, offset};
					state.builder.AddFunction(
					    dref ? spv::OpImageDrefGather : spv::OpImageGather, result_type,
					    sample, sampled, coord, component_or_dref,
					    std::span(operands).first(operand_mask != 0u ? 2u : 0u));
					return sample;
				};
				return candidate.mip_mode == IR::ImageMipMode::Dynamic && candidate.mip_count > 1u
				           ? EmitIndexSwitch(
				                 state, GatherMip(ctx, inst, mem, *address, layout, candidate.mip_count),
				                 candidate.mip_count, result_type, EmitMip)
				           : EmitMip(0u);
			};
			const auto sample = EmitImageAccess(ctx, inst, result_type, EmitGather);
			ctx.Define(inst, ResultVector(ctx, UnpackImageGather(ctx, mem, sample),
			                              result_numeric_class, false, mem, true));
			return;
		}
		const bool explicit_lod = HasFlag(mem, Decoder::ImageSampleFlagDerivative) ||
		                          HasFlag(mem, Decoder::ImageSampleFlagLod) ||
		                          HasFlag(mem, Decoder::ImageSampleFlagLevelZero) ||
		                          state.program.stage != ShaderType::Pixel;
		auto       opcode       = spv::OpImageSampleImplicitLod;
		if (explicit_lod) {
			opcode = dref ? spv::OpImageSampleDrefExplicitLod : spv::OpImageSampleExplicitLod;
		} else if (dref) {
			opcode = spv::OpImageSampleDrefImplicitLod;
		}
		uint32_t result_type = ImageVectorType(state, numeric_class, 4);
		uint32_t dref_value  = 0;
		if (dref) {
			result_type = TypeF32(state);
			dref_value  = ZeroF32(state);
			if (layout.dref != NoImageComponent) {
				dref_value = AddressF32(ctx, mem, *address, layout.dref);
			}
		}
		// IMAGE_SAMPLE*_CL: the last address component is a minimum LOD. Implicit and gradient
		// samples take it as the MinLod operand; explicit-LOD forms (every sample outside pixel
		// shaders) raise their LOD to it.
		uint32_t lod_clamp = 0;
		if (layout.clamp != NoImageComponent && mem.image_address_components > layout.clamp) {
			lod_clamp = AddressF32(ctx, mem, *address, layout.clamp);
		}
		const bool gradient   = HasFlag(mem, Decoder::ImageSampleFlagDerivative);
		const bool min_lod_op = lod_clamp != 0u && (gradient || !explicit_lod) &&
		                        GetHostImageFeatures().min_lod;
		uint32_t              explicit_lod_value = 0;
		uint32_t              operand_mask       = 0;
		std::vector<uint32_t> operands;
		if (gradient) {
			operand_mask |= spv::ImageOperandsGradMask;
			operands.push_back(
			    CoordF32(ctx, mem, *address, layout.grad_x, dimension_info.spatial_components));
			operands.push_back(
			    CoordF32(ctx, mem, *address, layout.grad_y, dimension_info.spatial_components));
		} else if (explicit_lod) {
			operand_mask |= spv::ImageOperandsLodMask;
			auto lod = ZeroF32(state);
			if (HasFlag(mem, Decoder::ImageSampleFlagLod) && layout.lod != NoImageComponent) {
				lod = AddressF32(ctx, mem, *address, layout.lod);
			}
			if (lod_clamp != 0u) {
				const auto clamped = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpExtInst, TypeF32(state), clamped, GlslStd450(state),
				                          GLSLstd450FMax, lod, lod_clamp);
				lod = clamped;
			}
			explicit_lod_value = lod;
			operands.push_back(lod);
		} else if (layout.bias != NoImageComponent) {
			operand_mask |= spv::ImageOperandsBiasMask;
			operands.push_back(AddressF32(ctx, mem, *address, layout.bias));
		}
		const auto sampler_id = LoadInstructionSampler(ctx, inst, mem);
		const bool finite_sampler = FiniteSampler(state.program.info.samplers.at(mem.sampler));
		const auto sample_offset = SampleOffsetFor(ctx, inst, mem, *address, layout, dimension,
		                                           image.cube, image.bindless);
		if (sample_offset.const_offset != 0u) {
			operand_mask |= spv::ImageOperandsConstOffsetMask;
			operands.push_back(sample_offset.const_offset);
		}
		if (min_lod_op) {
			state.builder.RequireCapability(spv::CapabilityMinLod);
			operand_mask |= spv::ImageOperandsMinLodMask;
			operands.push_back(lod_clamp);
		} else if (lod_clamp != 0u && !explicit_lod) {
			static std::atomic_flag warned = ATOMIC_FLAG_INIT;
			if (!warned.test_and_set(std::memory_order_relaxed)) {
				std::fputs("Warning: IMAGE_SAMPLE*_CL clamp ignored: the device lacks "
				           "shaderResourceMinLod.\n",
				           stderr);
			}
		}
		const auto EmitSample = [&](uint32_t resource, uint32_t array_index = 0u) {
			const auto& candidate = state.program.info.images[resource];
			auto        coord =
			    CoordF32(ctx, mem, *address, layout.coord,
			             ImageDimensionInfoFor(candidate.dimension).coordinate_components,
			             candidate.cube);
			if (sample_offset.dynamic) {
				uint32_t level = ZeroF32(state);
				if (explicit_lod_value != 0u) {
					level = explicit_lod_value;
				} else if (!HasFlag(mem, Decoder::ImageSampleFlagLevelZero) &&
				           state.program.stage == ShaderType::Pixel) {
					// The level an implicit or gradient sample reads, relative to the view base.
					state.builder.RequireCapability(spv::CapabilityImageQuery);
					const auto query = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpImageQueryLod, TypeF32Vector(state, 2), query,
					    MakeSampledImage(state, resource, sampler_id, 0u, array_index, finite_sampler),
					    CoordF32(ctx, mem, *address, layout.coord,
					             ImageDimensionInfoFor(candidate.dimension).spatial_components,
					             candidate.cube));
					level = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), level, query,
					                          0u);
				}
				coord = ApplyDynamicSampleOffset(state, resource, coord, candidate.dimension,
				                                 sample_offset.packed, level);
			}
			const auto sampled = MakeSampledImage(state, resource, sampler_id, 0u, array_index, finite_sampler);
			const auto            sample  = state.builder.AllocateId();
			std::vector<uint32_t> sample_operands {result_type, sample, sampled, coord};
			if (dref) {
				sample_operands.push_back(dref_value);
			}
			if (operand_mask != 0u) {
				sample_operands.push_back(operand_mask);
				sample_operands.insert(sample_operands.end(), operands.begin(), operands.end());
			}
			state.builder.AddFunction(opcode, sample_operands);
			return sample;
		};
		auto result = EmitImageAccess(ctx, inst, result_type, EmitSample);
		if (!dref) {
			result = UnpackImageTexel(ctx, mem, result);
		}
		ctx.Define(inst, ResultVector(ctx, result, numeric_class, dref, mem));
		return;
	}
	if (image_info.access == IR::ImageAccess::Atomic) {
		const auto dimension = image.dimension;
		const auto result_type = TypeId(state, inst.GetType());
		const auto zero = image.atomic64 ? ConstantU64(state, 0) : ConstantU32(state, 0);
		ctx.Define(inst, EmitValueOrDefaultIfCondition(
		                     state, ctx.Arg(inst, inst.NumArgs() - 1), result_type, zero, [&]() {
			           const auto pointer      = state.builder.AllocateId();
			           const auto pointer_type =
			               state.builder.Type(spv::OpTypePointer, spv::StorageClassImage,
			                                  image.atomic64 ? TypeU64(state) : TypeU32(state));
			           state.builder.AddFunction(spv::OpImageTexelPointer, pointer_type, pointer,
			                                     ImageDescriptorPointer(state, mem.resource),
			                                     CoordU32(ctx, mem, *address, dimension),
			                                     ConstantU32(state, 0));
			           if (op == IR::ValueOpcode::ImageAtomicFMin32 ||
			               op == IR::ValueOpcode::ImageAtomicFMax32) {
				           return AtomicUpdate(state, pointer, IR::ResourceKind::Image,
				                               [&](uint32_t old) {
					                               return EmitFloatAtomicReplacement(
					                                   state, old, ctx.Arg(inst, 2),
					                                   op == IR::ValueOpcode::ImageAtomicFMax32);
				                               });
			           }
			           const auto old = EmitAtomicOperation(ctx, inst, pointer, spv::ScopeDevice);
			           EmitAtomicMemoryBarrier(state, IR::ResourceKind::Image);
			           return old;
		           }));
		return;
	}
	ctx.Fail(inst, "has no image SPIR-V emitter");
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

namespace Libs::Graphics::ShaderRecompiler::Spirv {

bool ImageConversionEmittable(const IR::ImageResource& image, Prospero::BufferFormat format) {
	if (format == Prospero::BufferFormat::kInvalid || format == Prospero::BufferFormat::k16UInt) {
		return true; // no conversion, or a D16 surface's codes (IR::ReadsDepthBits)
	}
	const auto info       = Format::GetFormatInfo(format);
	const bool raw_signed = IR::RawSignedStorageFormat(image, format) != Prospero::BufferFormat::kInvalid;
	if ((Prospero::RemapTextureFormat(format) == format && !raw_signed) || info.component_count == 0u ||
	    info.component_count > 4u) {
		return false;
	}
	return raw_signed || Format::IsSrgbFormat(format) || info.type == Format::ComponentType::Uscaled ||
	       ((info.type == Format::ComponentType::Uint || info.type == Format::ComponentType::Unorm) &&
	        info.packed_bitfield && info.byte_size == sizeof(uint32_t));
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
