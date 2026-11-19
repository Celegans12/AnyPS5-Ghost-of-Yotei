#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterRayQueryShared.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/RayQuery.h"
#undef near
#undef far

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

using namespace RayQueryEmit;

std::array<uint32_t, 2> GuestBarycentrics(EmitterState& s, uint32_t query, uint32_t intersection,
                                          uint32_t remap) {
	const auto u = TypeU32(s), f = TypeF32(s);
	const auto bary = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpRayQueryGetIntersectionBarycentricsKHR, TypeF32Vector(s, 2), bary, query,
	                      U32(s, intersection));
	std::array<uint32_t, 3> weights {};
	for (uint32_t i = 0; i < 2; i++) {
		weights[i + 1] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeExtract, f, weights[i + 1], bary, i);
	}
	weights[0] = Binary(s, spv::OpFSub, f, Binary(s, spv::OpFSub, f, ConstantF32Value(s, 1.0f), weights[1]),
	                    weights[2]);
	const auto weight = [&](uint32_t shift) {
		const auto index = And(s, Binary(s, spv::OpShiftRightLogical, u, remap, U32(s, shift)), 3u);
		return Select(s, f, Equal(s, index, U32(s, 1)), weights[1],
		              Select(s, f, Equal(s, index, U32(s, 2)), weights[2], weights[0]));
	};
	return {weight(0), weight(2)};
}

uint32_t AlphaPointer(EmitterState& s, const LaneVariables& lane, uint32_t index) {
	const auto pointer = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypePointer(s, spv::StorageClassFunction, TypeU32(s)), pointer,
	                      lane.alpha, U32(s, index));
	return pointer;
}

uint32_t AlphaU32(EmitterState& s, const LaneVariables& lane, AlphaArgument argument, uint32_t offset = 0) {
	return Load(s, TypeU32(s), AlphaPointer(s, lane, static_cast<uint32_t>(argument) + offset));
}

uint32_t AlphaF32(EmitterState& s, const LaneVariables& lane, AlphaArgument argument) {
	return Unary(s, spv::OpBitcast, TypeF32(s), AlphaU32(s, lane, argument));
}

enum class AlphaCache : uint32_t {
	Instance,
	BlasLow,
	BlasHigh,
	MaterialBase,
	LocalBase,
	TableOffset,
	RecordOffset,
	Material,
	Indices = 10,
	Vertices = 14,
	Coordinates = 18,
	Count = 24,
};

uint32_t AlphaCachePointer(EmitterState& s, const LaneVariables& lane, AlphaCache field, uint32_t offset = 0) {
	const auto pointer = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypePointer(s, spv::StorageClassFunction, TypeU32(s)), pointer,
	                      lane.alpha_cache, U32(s, static_cast<uint32_t>(field) + offset));
	return pointer;
}

uint32_t AlphaCacheU32(EmitterState& s, const LaneVariables& lane, AlphaCache field, uint32_t offset = 0) {
	return Load(s, TypeU32(s), AlphaCachePointer(s, lane, field, offset));
}

void ClearAlphaCache(EmitterState& s, const LaneVariables& lane) {
	Store(s, AlphaCachePointer(s, lane, AlphaCache::Instance), U32(s, None));
	Store(s, AlphaCachePointer(s, lane, AlphaCache::RecordOffset), U32(s, None));
}

uint32_t VsharpBase(EmitterState& s, const std::array<uint32_t, 4>& vsharp) {
	return PackU64(s, vsharp[0], And(s, vsharp[1], 0xffffu));
}

uint32_t VsharpStride(EmitterState& s, const std::array<uint32_t, 4>& vsharp) {
	return EmitBitFieldUExtract(s, vsharp[1], U32(s, 16), U32(s, 14));
}

std::array<uint32_t, 3> StructuredDwords(ValueEmitContext& ctx, const std::array<uint32_t, 4>& vsharp,
                                         uint32_t index) {
	auto&      s = ctx.state;
	const auto u = TypeU32(s), wide = TypeU64(s);
	const auto stride  = VsharpStride(s, vsharp);
	const auto address = [&](uint32_t element) {
		return Binary(s, spv::OpIAdd, wide, VsharpBase(s, vsharp),
		              Unary(s, spv::OpUConvert, wide, Binary(s, spv::OpIMul, u, element, stride)));
	};
	const auto packed = Equal(s, stride, U32(s, 4));
	const auto object = EmitGuestObject(ctx, address(index), 12);
	std::array<uint32_t, 3> words {};
	for (uint32_t k = 0; k < 3; k++) {
		const auto element = Add(s, index, U32(s, k));
		words[k]           = EmitValueOrZeroIfCondition(s, Less(s, element, vsharp[2]), [&] {
			const auto fast = EmitValueOrZeroIfCondition(s, packed, [&] { return EmitGuestObjectDword(ctx, object, k * 4); });
			const auto slow = EmitValueOrZeroIfCondition(s, LogicalNot(s, packed),
			                                             [&] { return EmitGuestDwordLoad(ctx, address(element)); });
			return Select(s, u, packed, fast, slow);
		});
	}
	return words;
}

uint32_t AlphaRecordCheck(EmitterState& s, std::span<const uint32_t> words) {
	const auto u    = TypeU32(s);
	uint32_t   hash = U32(s, 0x811c9dc5u);
	for (const auto word: words) {
		hash = Binary(s, spv::OpIMul, u, Binary(s, spv::OpBitwiseXor, u, hash, word), U32(s, 0x01000193u));
		hash = Binary(s, spv::OpBitwiseXor, u, hash, Binary(s, spv::OpShiftRightLogical, u, hash, U32(s, 15)));
	}
	return Or(s, And(s, hash, ~3u), U32(s, 1));
}

uint32_t AlphaTest(ValueEmitContext& ctx, const LaneVariables& lane, const GuestEntry& candidate, bool lane_layout,
                   bool factor_reference, uint32_t format) {
	auto&      s = ctx.state;
	const auto u = TypeU32(s), f = TypeF32(s), wide = TypeU64(s), b = TypeBool(s);
	const auto shr = [&](uint32_t value, uint32_t bits) {
		return Binary(s, spv::OpShiftRightLogical, u, value, U32(s, bits));
	};
	const auto shl = [&](uint32_t value, uint32_t bits) {
		return Binary(s, spv::OpShiftLeftLogical, u, value, U32(s, bits));
	};
	const auto mul    = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpIMul, u, x, y); };
	const auto fmul   = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpFMul, f, x, y); };
	const auto fadd   = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpFAdd, f, x, y); };
	const auto as_f32 = [&](uint32_t value) { return Unary(s, spv::OpBitcast, f, value); };
	const auto cache  = [&](AlphaCache field, uint32_t offset = 0) { return AlphaCacheU32(s, lane, field, offset); };
	const auto keep   = [&](AlphaCache field, uint32_t value, uint32_t offset = 0) {
		Store(s, AlphaCachePointer(s, lane, field, offset), value);
	};
	const uint32_t material_dword = format & 63u, table_dword = (format >> 6u) & 63u, first_dword = (format >> 12u) & 63u;
	const bool     vertex_align = ((format >> 18u) & 1u) != 0;
	const uint32_t vertex_mask  = (format >> 19u) & 0xffu;
	const uint32_t blas_dword   = (format >> 27u) & 15u;
	const bool     blas_packed  = (format >> 31u) != 0;
	std::array<uint32_t, 7> record {};
	for (uint32_t i = 0; i < 7; i++) {
		record[i] = LoadAddressU32(s, AddressPlus(s, candidate.record, i * 4));
	}
	const auto node = InstanceNode(s, lane, CandidateIntersection);
	EmitIfCondition(s, Binary(s, spv::OpINotEqual, b, node, cache(AlphaCache::Instance)), [&] {
		const auto record = EmitGuestObject(ctx, InstanceRecordAddress(s, lane.base, node), 0x80);
		const auto blas_low  = EmitGuestObjectDword(ctx, record, blas_dword * 4);
		const auto blas_high = EmitGuestObjectDword(ctx, record, blas_dword * 4 + 4);
		const auto blas      = blas_packed ? Binary(s, spv::OpShiftLeftLogical, wide,
		                                            PackU64(s, blas_low, And(s, blas_high, 0x3fffffu)), ConstantU64(s, 3))
		                                   : PackU64(s, blas_low, blas_high);
		keep(AlphaCache::MaterialBase, EmitGuestObjectDword(ctx, record, material_dword * 4));
		const auto header = EmitGuestObject(ctx, blas, std::max(0x80u, (std::max(table_dword, first_dword) + 1) * 4));
		keep(AlphaCache::LocalBase, EmitGuestObjectDword(ctx, header, first_dword * 4));
		keep(AlphaCache::TableOffset, EmitGuestObjectDword(ctx, header, table_dword * 4));
		const auto blas_words = Unary(s, spv::OpBitcast, TypeU32Vector(s, 2), blas);
		std::array<uint32_t, 2> halves {};
		for (uint32_t i = 0; i < 2; i++) {
			halves[i] = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeExtract, u, halves[i], blas_words, i);
		}
		keep(AlphaCache::BlasLow, halves[0]);
		keep(AlphaCache::BlasHigh, halves[1]);
		keep(AlphaCache::Instance, node);
	});
	const auto blas = PackU64(s, cache(AlphaCache::BlasLow), cache(AlphaCache::BlasHigh));
	const auto local = Binary(s, spv::OpISub, u, shr(candidate.node, 3), cache(AlphaCache::LocalBase));
	const auto at    = Add(s, shl(local, 2), cache(AlphaCache::TableOffset));
	const auto geometry =
	    EmitGuestDwordLoad(ctx, Binary(s, spv::OpIAdd, wide, blas, Unary(s, spv::OpUConvert, wide, at)));
	const auto material_index = Add(s, geometry, cache(AlphaCache::MaterialBase));
	const auto record_offset  = mul(material_index, U32(s, 12));
	EmitIfCondition(s, Binary(s, spv::OpINotEqual, b, record_offset, cache(AlphaCache::RecordOffset)), [&] {
		std::array<uint32_t, 4> material {}, buffers {};
		for (uint32_t i = 0; i < 4; i++) {
			material[i] = AlphaU32(s, lane, AlphaArgument::Material, i);
			buffers[i]  = AlphaU32(s, lane, AlphaArgument::Buffers, i);
		}
		std::array<uint32_t, 3> words {};
		if (lane_layout) {
			words = StructuredDwords(ctx, material, mul(material_index, U32(s, 3)));
		} else {
			ScalarBufferObject record;
			EmitScalarBufferObject(ctx, material, record_offset, 12, record);
			for (uint32_t i = 0; i < 3; i++) {
				words[i] = EmitScalarBufferObjectDword(ctx, record, i * 4);
			}
		}
		for (uint32_t i = 0; i < 3; i++) {
			keep(AlphaCache::Material, words[i], i);
		}
		const auto table_vsharp = [&](uint32_t word, AlphaCache field) {
			ScalarBufferObject entry;
			EmitScalarBufferObject(ctx, buffers, shl(shr(word, 8), 4), 16, entry);
			for (uint32_t i = 0; i < 4; i++) {
				keep(field, EmitScalarBufferObjectDword(ctx, entry, i * 4), i);
			}
		};
		table_vsharp(words[1], AlphaCache::Indices);
		if (lane_layout) {
			const auto element = shr(words[0], 8);
			const auto address = Binary(s, spv::OpIAdd, wide, VsharpBase(s, buffers),
			                            Unary(s, spv::OpUConvert, wide, mul(element, VsharpStride(s, buffers))));
			const auto entry   = EmitGuestObject(ctx, address, 8);
			const auto in_range = Less(s, element, buffers[2]);
			for (uint32_t i = 0; i < 2; i++) {
				keep(AlphaCache::Vertices,
				     EmitValueOrZeroIfCondition(s, in_range, [&] { return EmitGuestObjectDword(ctx, entry, i * 4); }), i);
			}
		} else {
			table_vsharp(words[0], AlphaCache::Vertices);
		}
		keep(AlphaCache::RecordOffset, record_offset);
	});
	const auto m0 = cache(AlphaCache::Material, 0), m1 = cache(AlphaCache::Material, 1),
	           m2 = cache(AlphaCache::Material, 2);
	std::array<uint32_t, 4> indices {}, vertices {};
	for (uint32_t i = 0; i < 4; i++) {
		indices[i]  = cache(AlphaCache::Indices, i);
		vertices[i] = cache(AlphaCache::Vertices, i);
	}
	const auto first  = mul(And(s, candidate.word, 0x3fffffffu), U32(s, 3));
	const auto stride = And(s, m0, vertex_mask), uv_offset = And(s, m1, vertex_mask);
	std::vector<uint32_t> selectors {candidate.word, U32(s, lane_layout ? 1u : 0u), stride, uv_offset,
	                                 indices[0], indices[1], indices[2], indices[3], vertices[0], vertices[1]};
	if (!lane_layout) {
		selectors.push_back(vertices[2]);
		selectors.push_back(vertices[3]);
	}
	const auto check = [&](std::span<const uint32_t> coordinates) {
		std::vector<uint32_t> words = selectors;
		words.insert(words.end(), coordinates.begin(), coordinates.end());
		return AlphaRecordCheck(s, words);
	};
	const auto recorded = Equal(s, record[0], check(std::span(record).subspan(1)));
	for (uint32_t c = 0; c < 6; c++) {
		keep(AlphaCache::Coordinates, record[1 + c], c);
	}
	EmitIfCondition(s, LogicalNot(s, recorded), [&] {
		std::array<uint32_t, 6> words {};
		const auto triangle = EmitTriangleIndices(ctx, indices, first);
		for (uint32_t k = 0; k < 3; k++) {
			auto offset = Add(s, uv_offset, mul(stride, triangle[k]));
			if (vertex_align) {
				offset = And(s, offset, ~3u);
			}
			if (lane_layout) {
				const auto vertex = EmitGuestObject(
				    ctx, Binary(s, spv::OpIAdd, wide, VsharpBase(s, vertices), Unary(s, spv::OpUConvert, wide, offset)),
				    8);
				words[k * 2]     = EmitGuestObjectDword(ctx, vertex, 0);
				words[k * 2 + 1] = EmitGuestObjectDword(ctx, vertex, 4);
				continue;
			}
			ScalarBufferObject vertex;
			EmitScalarBufferObject(ctx, vertices, offset, 8, vertex);
			words[k * 2]     = EmitScalarBufferObjectDword(ctx, vertex, 0);
			words[k * 2 + 1] = EmitScalarBufferObjectDword(ctx, vertex, 4);
		}
		for (uint32_t c = 0; c < 6; c++) {
			keep(AlphaCache::Coordinates, words[c], c);
			StoreAddressU32(s, AddressPlus(s, candidate.record, 4 + c * 4), words[c]);
		}
		StoreAddressU32(s, candidate.record, check(words));
	});
	std::array<std::array<uint32_t, 2>, 3> uv {};
	for (uint32_t k = 0; k < 3; k++) {
		uv[k][0] = as_f32(cache(AlphaCache::Coordinates, k * 2));
		uv[k][1] = as_f32(cache(AlphaCache::Coordinates, k * 2 + 1));
	}
	const auto [i, j] = GuestBarycentrics(s, lane.query, CandidateIntersection, candidate.remap);
	const auto w0     = Binary(s, spv::OpFSub, f, Binary(s, spv::OpFSub, f, ConstantF32Value(s, 1.0f), i), j);
	const auto coord  = [&](uint32_t c) {
		return fadd(fadd(fmul(uv[0][c], w0), fmul(uv[1][c], i)), fmul(uv[2][c], j));
	};
	auto t = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpRayQueryGetIntersectionTKHR, f, t, lane.query, U32(s, CandidateIntersection));
	if (lane.alpha_distance != 0) {
		t = fadd(t, as_f32(Load(s, u, lane.alpha_distance)));
	}
	const auto cone =
	    fmul(fadd(fmul(AlphaF32(s, lane, AlphaArgument::Spread), t), AlphaF32(s, lane, AlphaArgument::Lod)),
	         ConstantF32Value(s, 2.0f));
	const auto log = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpExtInst, f, log, GlslStd450(s), GLSLstd450Log2, cone);
	const auto level = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpExtInst, f, level, GlslStd450(s), GLSLstd450NMin,
	                      Unary(s, spv::OpConvertUToF, f, And(s, m2, 15u)),
	                      fadd(log, AlphaF32(s, lane, AlphaArgument::LodOffset)));
	const auto texel =
	    EmitImageTableSampleLod(ctx, *s.ray_query_alpha_sample, shr(m2, 4), coord(0), coord(1), level);
	const auto alpha = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeExtract, u, alpha, texel, 3u);
	if (lane_layout && !factor_reference) {
		return Binary(s, spv::OpFOrdGreaterThan, b, as_f32(alpha), AlphaF32(s, lane, AlphaArgument::Threshold));
	}
	const auto factor = Select(s, f, BitSet(s, m0, 1u), AlphaF32(s, lane, AlphaArgument::Factor),
	                           ConstantF32Value(s, 1.0f));
	return Binary(s, spv::OpFOrdGreaterThan, b, as_f32(alpha),
	              fmul(AlphaF32(s, lane, AlphaArgument::Threshold), factor));
}

enum class EntryCache : uint32_t {
	Instance,
	IndexBase,
	VertexBase,
	FirstTexture,
	ScaleU,
	ScaleV,
	OffsetU,
	Pair0,
	Pair1,
	OffsetV,
	SecondTexture,
	FirstX,
	FirstY,
	Alpha,
	Count,
};
static_assert(static_cast<uint32_t>(EntryCache::Count) <= static_cast<uint32_t>(AlphaCache::Count));

uint32_t EntryCachePointer(EmitterState& s, const LaneVariables& lane, EntryCache field) {
	const auto pointer = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypePointer(s, spv::StorageClassFunction, TypeU32(s)), pointer,
	                      lane.alpha_cache, U32(s, static_cast<uint32_t>(field)));
	return pointer;
}

uint32_t UnalignedGuestDword(ValueEmitContext& ctx, uint32_t address) {
	auto&      s     = ctx.state;
	const auto u     = TypeU32(s);
	const auto shift = Binary(s, spv::OpShiftLeftLogical, u, And(s, Unary(s, spv::OpUConvert, u, address), 3u), U32(s, 3));
	const auto low   = EmitGuestDwordLoad(ctx, address);
	const auto high  = EmitValueOrZeroIfCondition(s, Binary(s, spv::OpINotEqual, TypeBool(s), shift, U32(s, 0)),
	                                              [&] { return EmitGuestDwordLoad(ctx, AddressPlus(s, address, 4)); });
	return Or(s, Binary(s, spv::OpShiftRightLogical, u, low, shift),
	          Binary(s, spv::OpShiftLeftLogical, u, high, And(s, Binary(s, spv::OpISub, u, U32(s, 32), shift), 31u)));
}

uint32_t AlphaTestEntry(ValueEmitContext& ctx, const LaneVariables& lane, const GuestEntry& candidate,
                        const IR::Inst& first_sample, const IR::Inst& second_sample, uint32_t format) {
	auto&      s = ctx.state;
	const auto u = TypeU32(s), f = TypeF32(s), b = TypeBool(s);
	const auto shr    = [&](uint32_t value, uint32_t bits) { return Binary(s, spv::OpShiftRightLogical, u, value, U32(s, bits)); };
	const auto shl    = [&](uint32_t value, uint32_t bits) { return Binary(s, spv::OpShiftLeftLogical, u, value, U32(s, bits)); };
	const auto mul    = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpIMul, u, x, y); };
	const auto add    = [&](uint32_t x, uint32_t y) { return Add(s, x, y); };
	const auto fmul   = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpFMul, f, x, y); };
	const auto fadd   = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpFAdd, f, x, y); };
	const auto as_f32 = [&](uint32_t value) { return Unary(s, spv::OpBitcast, f, value); };
	const auto cache  = [&](EntryCache field) { return Load(s, u, EntryCachePointer(s, lane, field)); };
	const auto keep   = [&](EntryCache field, uint32_t value) { Store(s, EntryCachePointer(s, lane, field), value); };
	const auto not_none = [&](uint32_t value) { return Binary(s, spv::OpINotEqual, b, value, U32(s, None)); };
	std::array<uint32_t, 4> entries {}, buffers {};
	for (uint32_t i = 0; i < 4; i++) {
		entries[i] = AlphaU32(s, lane, AlphaArgument::Material, i);
		buffers[i] = AlphaU32(s, lane, AlphaArgument::Buffers, i);
	}
	const auto from_entries = [&](uint32_t offset) { return EmitScalarBufferDwordLoad(ctx, entries, offset); };
	const auto from_buffers = [&](uint32_t offset) { return EmitScalarBufferDwordLoad(ctx, buffers, offset); };
	const auto node = InstanceNode(s, lane, CandidateIntersection);
	EmitIfCondition(s, Binary(s, spv::OpINotEqual, b, node, cache(EntryCache::Instance)), [&] {
		const auto record = EmitGuestObject(ctx, InstanceRecordAddress(s, lane.base, node), 0x80);
		const auto entry  = shl(EmitGuestObjectDword(ctx, record, (format & 63u) * 4u), 4);
		const auto e0 = from_entries(entry), e1 = from_entries(add(entry, U32(s, 4))),
		           e2 = from_entries(add(entry, U32(s, 8))), e3 = from_entries(add(entry, U32(s, 12)));
		const auto strides = from_buffers(shl(add(shr(e1, 4), U32(s, 4)), 4));
		const auto signed24 = [&](uint32_t value) {
			const auto result = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpBitFieldSExtract, u, result, value, U32(s, 0), U32(s, 24));
			return result;
		};
		const auto buffer_record = And(s, e1, ~12u);
		keep(EntryCache::IndexBase, add(from_buffers(buffer_record), mul(And(s, strides, 0xffu), signed24(e2))));
		keep(EntryCache::VertexBase, add(from_buffers(add(buffer_record, U32(s, 12))), mul(shr(strides, 24), signed24(e3))));
		const auto material = And(s, e0, ~14u);
		const auto at       = [&](uint32_t offset) { return from_buffers(add(material, U32(s, offset))); };
		keep(EntryCache::FirstTexture, at(16));
		keep(EntryCache::ScaleU, at(16 + 5 * 4));
		keep(EntryCache::ScaleV, at(16 + 6 * 4));
		keep(EntryCache::OffsetU, at(16 + 7 * 4));
		keep(EntryCache::Pair0, at(0x48));
		keep(EntryCache::Pair1, at(0x4c));
		keep(EntryCache::OffsetV, from_buffers(shl(add(shr(e0, 4), U32(s, 3)), 4)));
		keep(EntryCache::SecondTexture, from_buffers(add(And(s, e0, ~12u), U32(s, 12))));
		keep(EntryCache::Instance, node);
	});
	const auto bytes   = mul(And(s, candidate.word, 0x3fffffffu), U32(s, 6));
	const auto aligned = Equal(s, bytes, And(s, bytes, ~2u));
	const auto index_address = PackU64(s, add(cache(EntryCache::IndexBase), And(s, bytes, ~2u)), U32(s, 31));
	const auto first_word    = UnalignedGuestDword(ctx, index_address);
	const auto second_word   = UnalignedGuestDword(ctx, AddressPlus(s, index_address, 4));
	const auto low16  = [&](uint32_t value) { return And(s, value, 0xffffu); };
	const auto high16 = [&](uint32_t value) { return shr(value, 16); };
	const std::array<uint32_t, 3> indices {Select(s, u, aligned, low16(first_word), high16(first_word)),
	                                       Select(s, u, aligned, high16(first_word), low16(second_word)),
	                                       Select(s, u, aligned, low16(second_word), high16(second_word))};
	std::array<std::array<uint32_t, 2>, 3> uv {};
	for (uint32_t k = 0; k < 3; k++) {
		const auto address = PackU64(s, add(cache(EntryCache::VertexBase), shl(indices[k], 2)), U32(s, 31));
		const auto halves  = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpExtInst, TypeF32Vector(s, 2), halves, GlslStd450(s), GLSLstd450UnpackHalf2x16,
		                      UnalignedGuestDword(ctx, address));
		for (uint32_t c = 0; c < 2; c++) {
			uv[k][c] = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeExtract, f, uv[k][c], halves, c);
		}
	}
	const auto [i, j] = GuestBarycentrics(s, lane.query, CandidateIntersection, candidate.remap);
	const auto w0     = Binary(s, spv::OpFSub, f, Binary(s, spv::OpFSub, f, ConstantF32Value(s, 1.0f), i), j);
	const auto coord  = [&](uint32_t c) { return fadd(fadd(fmul(uv[0][c], w0), fmul(uv[1][c], i)), fmul(uv[2][c], j)); };
	const auto x = coord(0), y = coord(1);
	const auto threshold = AlphaF32(s, lane, AlphaArgument::Threshold);
	const auto level     = AlphaF32(s, lane, AlphaArgument::LodOffset);
	const auto key       = [&](uint32_t texture) { return And(s, add(mul(texture, U32(s, 48)), U32(s, 16)), ~3u); };
	keep(EntryCache::FirstX, U32(s, 0));
	keep(EntryCache::FirstY, U32(s, 0));
	EmitIfCondition(s, not_none(cache(EntryCache::FirstTexture)), [&] {
		const auto texel = EmitImageSampleLodAtKey(ctx, first_sample, key(cache(EntryCache::FirstTexture)), x, y, level);
		for (uint32_t c = 0; c < 2; c++) {
			const auto value = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeExtract, u, value, texel, c);
			keep(c == 0 ? EntryCache::FirstX : EntryCache::FirstY, value);
		}
	});
	const auto above = [&](EntryCache field) {
		return Binary(s, spv::OpFOrdGreaterThan, b, as_f32(cache(field)), threshold);
	};
	const auto pair0 = as_f32(cache(EntryCache::Pair0)), pair1 = as_f32(cache(EntryCache::Pair1));
	const auto select_f32 = [&](uint32_t condition, uint32_t yes, uint32_t no) { return Select(s, f, condition, yes, no); };
	const auto x_hit = above(EntryCache::FirstX), y_hit = above(EntryCache::FirstY);
	const auto scale_u = select_f32(y_hit, pair1, select_f32(x_hit, pair0, as_f32(cache(EntryCache::ScaleU))));
	const auto scale_v = select_f32(y_hit, pair1, select_f32(x_hit, pair0, as_f32(cache(EntryCache::ScaleV))));
	const auto u2 = fadd(fmul(x, scale_u), as_f32(cache(EntryCache::OffsetU)));
	const auto v2 = fadd(fmul(y, scale_v), as_f32(cache(EntryCache::OffsetV)));
	keep(EntryCache::Alpha, U32(s, std::bit_cast<uint32_t>(1.0f)));
	EmitIfCondition(s, not_none(cache(EntryCache::SecondTexture)), [&] {
		const auto texel = EmitImageSampleLodAtKey(ctx, second_sample, key(cache(EntryCache::SecondTexture)), u2, v2, level);
		const auto value = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeExtract, u, value, texel, 0u);
		keep(EntryCache::Alpha, value);
	});
	return above(EntryCache::Alpha);
}

void Refill(ValueEmitContext& ctx, uint32_t site) {
	auto&      s = ctx.state;
	const auto u = TypeU32(s), b = TypeBool(s), wide = TypeU64(s), f = TypeF32(s);
	const bool alpha_test = s.ray_query_alpha_sites[site];
	const bool entry_alpha = alpha_test && s.ray_query_alpha_entry[site];
	const bool lane_layout      = s.ray_query_alpha_lane[site];
	const bool factor_reference = s.ray_query_alpha_factor[site];
	const bool geometry_opacity = s.ray_query_geometry_opacity[site] && alpha_test;
	std::array<LaneVariables, 2> lanes {};
	std::array<uint32_t, 2>      refills {};
	for (uint32_t half = 0; half < s.lane_count; half++) {
		lanes[half]      = Lane(s, site, half);
		const auto state = Load(s, u, lanes[half].state);
		const auto empty = Equal(s, Load(s, u, lanes[half].current), Load(s, u, lanes[half].count));
		refills[half]    = LogicalAnd(s, empty, LogicalAnd(s, BitSet(s, state, TraversalBit),
		                                                   LogicalNot(s, BitSet(s, state, DoneBit))));
	}
	const auto handle = [&](const LaneVariables& lane, uint32_t count, uint32_t step, uint32_t proceed) {
		const auto confirm = [&] { s.builder.AddFunction(spv::OpRayQueryConfirmIntersectionKHR, lane.query); };
		EmitIfCondition(s, LogicalAnd(s, step, LogicalNot(s, proceed)), [&] {
			const auto committed = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpRayQueryGetIntersectionTypeKHR, u, committed, lane.query,
			                      U32(s, CommittedIntersection));
			const auto hit = Equal(
			    s, committed,
			    U32(s, spv::RayQueryCommittedIntersectionTypeRayQueryCommittedIntersectionTriangleKHR));
			EmitIfCondition(s, hit, [&] {
				const auto t = s.builder.AllocateId();
				s.builder.AddFunction(spv::OpRayQueryGetIntersectionTKHR, f, t, lane.query,
				                      U32(s, CommittedIntersection));
				const auto hit_entry = TriangleEntry(s, lane, CommittedIntersection);
				const auto node = InstanceNode(s, lane, CommittedIntersection);
				Store(s, lane.result_t, Unary(s, spv::OpBitcast, u, t));
				Store(s, lane.result_word, hit_entry.word);
				Store(s, lane.result_node,
				      entry_alpha ? Add(s, Binary(s, spv::OpShiftRightLogical, u, hit_entry.entry, U32(s, 3)), U32(s, 1))
				                  : hit_entry.node);
				const auto [bary_u, bary_v] =
				    GuestBarycentrics(s, lane.query, CommittedIntersection, hit_entry.remap);
				Store(s, lane.result_u, Unary(s, spv::OpBitcast, u, bary_u));
				Store(s, lane.result_v, Unary(s, spv::OpBitcast, u, bary_v));
				auto instance_record = InstanceRecordAddress(s, lane.base, node);
				if (const auto inline_layout = ProgramInlineLayout(s); inline_layout.valid) {
					instance_record = Select(s, wide, Equal(s, And(s, node, 7u), U32(s, inline_layout.type)),
					                         Load(s, wide, lane.inline_record), instance_record);
				}
				Store(s, lane.result_instance, instance_record);
				ClearKeys(s, lane);
				Store(s, lane.state, Or(s, Load(s, u, lane.state), U32(s, CommittedBit)));
			});
			Store(s, lane.state, Or(s, Load(s, u, lane.state), U32(s, DoneBit)));
		});
		EmitIfCondition(s, proceed, [&] {
			const auto type = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpRayQueryGetIntersectionTypeKHR, u, type, lane.query,
			                      U32(s, CandidateIntersection));
			const auto triangle = Equal(
			    s, type,
			    U32(s, spv::RayQueryCandidateIntersectionTypeRayQueryCandidateIntersectionTriangleKHR));
			EmitIfCondition(s, triangle, [&] {
				if (geometry_opacity) {
					const auto geometry = s.builder.AllocateId();
					s.builder.AddFunction(spv::OpRayQueryGetIntersectionGeometryIndexKHR, u, geometry, lane.query,
					                      U32(s, CandidateIntersection));
					const auto opaque = Equal(s, geometry, U32(s, 0));
					EmitIfCondition(s, opaque, confirm);
					EmitIfCondition(s, LogicalNot(s, opaque), [&] {
						const auto candidate = TriangleEntry(s, lane, CandidateIntersection);
						EmitIfCondition(s, candidate.valid, [&] {
							EmitIfCondition(s,
							                AlphaTest(ctx, lane, candidate, lane_layout, factor_reference,
							                          s.ray_query_alpha_format[site]),
							                confirm);
						});
					});
					return;
				}
				const auto candidate = TriangleEntry(s, lane, CandidateIntersection);
				EmitIfCondition(s, candidate.valid, [&] {
					if (alpha_test) {
						EmitIfCondition(s,
						                entry_alpha ? AlphaTestEntry(ctx, lane, candidate, *s.ray_query_entry_samples[0],
						                                             *s.ray_query_entry_samples[1],
						                                             s.ray_query_alpha_format[site])
						                            : AlphaTest(ctx, lane, candidate, lane_layout, factor_reference,
						                                        s.ray_query_alpha_format[site]),
						                confirm);
						return;
					}
					InsertKey(s, lane, count, InstanceNode(s, lane, CandidateIntersection),
					          candidate.entry);
					Store(s, lane.count, Add(s, count, U32(s, 1)));
				});
			});
		});
	};
	const auto refill = [&](std::span<const uint32_t> halves) {
		auto any = ConstantBool(s, false);
		for (const auto half: halves) {
			any = LogicalOr(s, any, refills[half]);
		}
		EmitIfCondition(s, any, [&] {
			for (const auto half: halves) {
				EmitIfCondition(s, refills[half], [&] { ClearKeys(s, lanes[half]); });
			}
			EmitLoop(s, [&](uint32_t merge, uint32_t cont) {
				std::array<uint32_t, 2> counts {}, steps {}, proceeds {};
				for (const auto half: halves) {
					const auto& lane = lanes[half];
					counts[half]     = Load(s, u, lane.count);
					const auto done  = BitSet(s, Load(s, u, lane.state), DoneBit);
					steps[half]      = LogicalAnd(s, LogicalAnd(s, refills[half], LogicalNot(s, done)),
					                              Less(s, counts[half], U32(s, ListSize)));
					proceeds[half] = EmitValueOrDefaultIfCondition(s, steps[half], b, ConstantBool(s, false), [&] {
						const auto result = s.builder.AllocateId();
						s.builder.AddFunction(spv::OpRayQueryProceedKHR, b, result, lane.query);
						return result;
					});
				}
				auto more = ConstantBool(s, false);
				for (const auto half: halves) {
					handle(lanes[half], counts[half], steps[half], proceeds[half]);
					more = LogicalOr(s, more, proceeds[half]);
				}
				s.builder.AddFunction(spv::OpBranchConditional, more, cont, merge);
			});
		});
	};
	constexpr std::array<uint32_t, 2> Halves {0, 1};
	refill(std::span(Halves).first(s.lane_count));
}

std::pair<uint32_t, uint32_t> LaneMinimum(EmitterState& s, const LaneVariables& lane) {
	const auto u       = TypeU32(s), b = TypeBool(s);
	const auto current = Load(s, u, lane.current);
	const auto has     = Less(s, current, Load(s, u, lane.count));
	const auto at      = Binary(s, spv::OpShiftLeftLogical, u, current, U32(s, 1));
	const auto instance = EmitValueOrDefaultIfCondition(s, has, u, U32(s, None),
	                                                    [&] { return LoadWord(s, lane, at); });
	const auto leaf = EmitValueOrDefaultIfCondition(s, has, u, U32(s, None),
	                                                [&] { return LoadWord(s, lane, Add(s, at, U32(s, 1))); });
	(void)b;
	return {instance, leaf};
}

}

void DeclareRayQueryVariables(EmitterState& s) {
	if (!s.program.info.ray_query) {
		return;
	}
	const auto u          = TypeU32(s);
	const auto query_type = s.builder.Type(spv::OpTypeRayQueryKHR);
	const auto list_type = s.builder.Type(spv::OpTypeArray, u, ConstantU32(s, 2 * ListSize + 1));
	const auto declare   = [&](uint32_t pointee, uint32_t& id) {
		id = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpVariable, TypePointer(s, spv::StorageClassFunction, pointee), id,
		                      spv::StorageClassFunction);
	};
	DeclareOracleVariables(s);
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		if ((s.program.info.ray_query_sites & (1u << site)) == 0) {
			continue;
		}
		for (uint32_t half = 0; half < s.lane_count; half++) {
			declare(query_type, s.ray_query_variables[site][half]);
			declare(u, s.ray_query_states[site][half]);
			declare(TypeU64(s), s.ray_query_nodes[site][half]);
			declare(TypeU64(s), s.ray_query_slots[site][half]);
			declare(list_type, s.ray_query_lists[site][half]);
			declare(u, s.ray_query_counts[site][half]);
			declare(u, s.ray_query_currents[site][half]);
			declare(TypeU64(s), s.ray_query_bases[site][half]);
			declare(TypeU64(s), s.ray_query_inline_records[site][half]);
			declare(u, s.ray_query_result_t[site][half]);
			declare(u, s.ray_query_result_word[site][half]);
			declare(TypeU64(s), s.ray_query_result_instance[site][half]);
			declare(u, s.ray_query_result_u[site][half]);
			declare(u, s.ray_query_result_v[site][half]);
			declare(u, s.ray_query_result_node[site][half]);
		}
		declare(u, s.ray_query_selected_instance[site]);
		declare(u, s.ray_query_selected_leaf[site]);
	}
	std::array<bool, IR::MaxRayQuerySites> alpha_sites {};
	std::array<bool, IR::MaxRayQuerySites> distance_sites {};
	std::vector<const IR::Inst*>           samples;
	std::array<const IR::Inst*, 2>         entry_samples {};
	bool                                   entry_ambiguous = false;
	for (const auto* block: s.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::RayQueryAlpha) {
				const auto site   = IR::RayQuerySite(inst.Flags<uint32_t>()) % IR::MaxRayQuerySites;
				alpha_sites[site] = true;
				const auto variant             = IR::RayQueryVariant(inst.Flags<uint32_t>());
				s.ray_query_alpha_lane[site]   = (variant & IR::RayQueryAlphaLane) != 0;
				s.ray_query_alpha_factor[site] = (variant & IR::RayQueryAlphaFactor) != 0;
				s.ray_query_alpha_entry[site]  = (variant & IR::RayQueryAlphaEntry) != 0;
				s.ray_query_alpha_format[site] = inst.Arg(static_cast<size_t>(AlphaArgument::Count)).U32();
			} else if (inst.GetOpcode() == IR::ValueOpcode::RayQueryAlphaDistance) {
				distance_sites[IR::RayQuerySite(inst.Flags<uint32_t>()) % IR::MaxRayQuerySites] = true;
			} else if (inst.GetOpcode() == IR::ValueOpcode::RayQueryInit) {
				const auto flags = inst.Flags<uint32_t>();
				if ((IR::RayQueryVariant(flags) & IR::RayQueryInitGeometryOpacity) != 0) {
					s.ray_query_geometry_opacity[IR::RayQuerySite(flags) % IR::MaxRayQuerySites] = true;
				}
			} else if (IsImageTableSampleLod(s, inst)) {
				samples.push_back(&inst);
			}
			if (IsImageKeySampleLod(s, inst)) {
				const auto dmask = s.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index).dmask;
				if (dmask == 3u || dmask == 1u) {
					auto& slot = entry_samples[dmask == 3u ? 0 : 1];
					entry_ambiguous |= slot != nullptr;
					slot = &inst;
				}
			}
		}
	}
	const bool entry_samples_found = entry_samples[0] != nullptr && entry_samples[1] != nullptr && !entry_ambiguous;
	if (entry_samples_found) {
		s.ray_query_entry_samples = entry_samples;
	}
	const auto memory = [&](const IR::Inst* inst) -> const IR::MemoryInfo& {
		return s.program.memory_info.at(inst->Flags<IR::MemoryFlags>().index);
	};
	const bool one_table = !samples.empty() && std::ranges::all_of(samples, [&](const IR::Inst* inst) {
		const auto& first = s.program.info.images[memory(samples[0]).resource];
		const auto& image = s.program.info.images[memory(inst).resource];
		return memory(inst).sampler == memory(samples[0]).sampler &&
		       image.table_descriptor_base == first.table_descriptor_base &&
		       image.table_mapping_offset == first.table_mapping_offset &&
		       image.table_entry_mask == first.table_entry_mask;
	});
	const auto alpha_type = s.builder.Type(spv::OpTypeArray, u, ConstantU32(s, static_cast<uint32_t>(AlphaArgument::Count)));
	const auto cache_type = s.builder.Type(spv::OpTypeArray, u, ConstantU32(s, static_cast<uint32_t>(AlphaCache::Count)));
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		s.ray_query_alpha_sites[site] = alpha_sites[site] && (s.ray_query_alpha_entry[site] ? entry_samples_found : one_table);
		if (!alpha_sites[site]) {
			continue;
		}
		for (uint32_t half = 0; half < s.lane_count; half++) {
			declare(alpha_type, s.ray_query_alpha[site][half]);
			declare(cache_type, s.ray_query_alpha_cache[site][half]);
			if (distance_sites[site]) declare(u, s.ray_query_alpha_distance[site][half]);
		}
	}
	s.ray_query_alpha_sample = one_table ? samples[0] : nullptr;
	if (std::ranges::any_of(alpha_sites, [](bool site) { return site; })) {
		const bool per_lane = std::ranges::any_of(alpha_sites, [&](bool site) { return site; }) &&
		                      (one_table || entry_samples_found);
		LOGF("ShaderRecompiler hash=0x%016" PRIx64 " native ray tracing alpha test: %s\n", s.program.shader_hash,
		     per_lane ? "per lane" : "guest leaf code (no single bindless table sample)");
	}
}

void InitializeRayQueryVariables(EmitterState& s) {
	InitializeOracleVariables(s);
	for (uint32_t site = 0; site < IR::MaxRayQuerySites; site++) {
		for (uint32_t half = 0; half < s.lane_count; half++) {
			if (s.ray_query_states[site][half] != 0) {
				Store(s, s.ray_query_states[site][half], U32(s, DoneBit));
				ClearKeys(s, Lane(s, site, half));
			}
			if (s.ray_query_alpha_cache[site][half] != 0) {
				ClearAlphaCache(s, Lane(s, site, half));
			}
		}
		if (s.ray_query_selected_instance[site] != 0) {
			Store(s, s.ray_query_selected_instance[site], U32(s, None));
			Store(s, s.ray_query_selected_leaf[site], U32(s, None));
		}
	}
}

uint32_t EmitRayQueryInit(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = Lane(s, site, ctx.half);
	const auto u = TypeU32(s), f = TypeF32(s), b = TypeBool(s), wide = TypeU64(s);
	std::array<uint32_t, 8> ray;
	for (uint32_t i = 0; i < 8; i++) {
		ray[i] = i < 7 ? Unary(s, spv::OpBitcast, f, ctx.Arg(inst, i)) : ctx.Arg(inst, i);
	}
	const auto header    = PackU64(s, ctx.Arg(inst, 8), ctx.Arg(inst, 9));
	const auto alpha_off = Binary(s, spv::OpINotEqual, b, ctx.Arg(inst, 10), U32(s, 0));
	const auto exec      = ctx.Arg(inst, 11);
	const auto entry = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(s, 64), entry,
	                      s.bda_pagetable_variable, U32(s, 0),
	                      U32(s, static_cast<uint32_t>(BufferCache::NATIVE_RT_CONTEXT_PAGE)));
	const auto context     = Load(s, wide, entry);
	const auto variant     = IR::RayQueryVariant(inst.Flags<uint32_t>());
	const bool geometry_opacity = (variant & IR::RayQueryInitGeometryOpacity) != 0;
	const bool no_leaf      = (variant & IR::RayQueryInitNoLeaf) != 0;
	const bool opaque_only  = (variant & IR::RayQueryInitOpaqueOnly) != 0;
	const bool always_guest = (geometry_opacity || no_leaf) && !s.ray_query_alpha_sites[site] && !opaque_only;
	const auto has_context = always_guest ? ConstantBool(s, false)
	                                      : Binary(s, spv::OpINotEqual, b, context, ConstantU64(s, 0));
	const auto host = EmitValueOrDefaultIfCondition(s, has_context, b, ConstantBool(s, false), [&] {
		const auto valid = Binary(s, spv::OpINotEqual, b,
		                          LoadAddressU32(s, AddressPlus(s, context, ContextValid)), U32(s, 0));
		const auto same = Equal(s, LoadAddressU64(s, AddressPlus(s, context, ContextHeader)), header);
		return LogicalAnd(s, valid, same);
	});
	const auto& allow   = inst.Arg(14);
	const bool  allow_all = allow.IsImmediate() && allow.U1();
	const auto  native  = allow_all ? host : LogicalAnd(s, host, ctx.Arg(inst, 14));
	const auto  served  = always_guest ? native : allow_all ? ConstantBool(s, true) : ctx.Arg(inst, 14);
	if ((always_guest || !allow_all) && NoteRayTracingProgram(RayTracingProgramKind::AlwaysGuest, s.program.shader_hash)) {
		::Log::WriteToConsoleAndLog(fmt::format("[RT] cs 0x{:016x}: site {} leaves {} lanes to the guest walk, for a leaf test "
		                                        "the host does not run; they stay exact and emulated\n",
		                                        s.program.shader_hash, site, always_guest ? "all its" : "some of its"));
	}
	if (s.ray_stats) {
		const auto guest  = LogicalAnd(s, exec, LogicalNot(s, native));
		const auto ballot = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(s, 4), ballot,
		                      U32(s, spv::ScopeSubgroup), guest);
		auto count = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpGroupNonUniformBallotBitCount, u, count, U32(s, spv::ScopeSubgroup),
		                      spv::GroupOperationReduce, ballot);
		if (ctx.half != 0) {
			count = Add(s, count, s.ray_stats_first_half[site]);
		}
		if (ctx.half + 1 < s.lane_count) {
			s.ray_stats_first_half[site] = count;
		} else {
			const auto elected = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpGroupNonUniformElect, b, elected, U32(s, spv::ScopeSubgroup));
			EmitIfCondition(s, LogicalAnd(s, elected, Binary(s, spv::OpINotEqual, b, count, U32(s, 0))),
			                [&] { EmitGuestRayCount(s, count); });
		}
	}
	auto lane_ok = LogicalAnd(s, native, exec);
	for (uint32_t i = 0; i < 6; i++) {
		lane_ok = LogicalAnd(s, lane_ok, Finite(s, ray[i]));
	}
	lane_ok = LogicalAnd(s, lane_ok,
	                     Binary(s, spv::OpFOrdGreaterThanEqual, b, ray[6], ConstantF32Value(s, 0.0f)));
	const auto traversal = Select(s, u, LogicalAnd(s, native, exec), U32(s, TraversalBit), U32(s, 0));
	Store(s, lane.state, Or(s, traversal, Select(s, u, lane_ok, U32(s, 0), U32(s, DoneBit))));
	ClearKeys(s, lane);
	EmitIfCondition(s, lane_ok, [&] {
		const auto tlas = LoadAddressU64(s, AddressPlus(s, context, ContextTlas));
		Store(s, lane.nodes, LoadAddressU64(s, AddressPlus(s, context, ContextInstanceNodes)));
		Store(s, lane.slots, LoadAddressU64(s, AddressPlus(s, context, ContextSlots)));
		Store(s, lane.base, LoadAddressU64(s, AddressPlus(s, context, ContextInstanceBase)));
		if (const auto inline_layout = ProgramInlineLayout(s); inline_layout.valid) {
			Store(s, lane.inline_record, AddressPlus(s, header, inline_layout.offset));
		}
		const auto structure = Unary(s, spv::OpConvertUToAccelerationStructureKHR,
		                             s.builder.Type(spv::OpTypeAccelerationStructureKHR), tlas);
		const auto origin = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), origin, ray[0], ray[1],
		                      ray[2]);
		const auto direction = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), direction, ray[3],
		                      ray[4], ray[5]);
		const uint32_t facing  = (variant & IR::RayQueryInitBothFacings) != 0 ? 0u : RayFlagCullBackFacing;
		const uint32_t opacity = geometry_opacity ? RayFlagNoOpaque : opaque_only ? RayFlagCullNoOpaque : 0u;
		const auto&    cull    = inst.Arg(12);
		const auto&    flip    = inst.Arg(13);
		const bool     always  = facing == 0 || (cull.IsImmediate() && cull.U1());
		const bool     fixed   = facing == 0 || (flip.IsImmediate() && !flip.U1());
		const auto     culled  = fixed ? U32(s, facing)
		                               : Select(s, u, ctx.Arg(inst, 13), U32(s, RayFlagCullFrontFacing), U32(s, facing));
		uint32_t       flags   = 0;
		if ((variant & IR::RayQueryInitAllCandidates) != 0) {
			flags = U32(s, RayFlagNoOpaque);
		} else if (always && fixed) {
			flags = Select(s, u, alpha_off, U32(s, facing | RayFlagOpaque), U32(s, facing | opacity));
		} else {
			const auto lane_facing = always ? culled : Select(s, u, ctx.Arg(inst, 12), culled, U32(s, 0));
			flags = Or(s, lane_facing, Select(s, u, alpha_off, U32(s, RayFlagOpaque), U32(s, opacity)));
		}
		s.builder.AddFunction(spv::OpRayQueryInitializeKHR, lane.query, structure, flags,
		                      And(s, ray[7], 0xffu), origin, ConstantF32Value(s, 0.0f), direction,
		                      ray[6]);
	});
	return served;
}

uint32_t EmitRayQueryNext(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto u = TypeU32(s), b = TypeBool(s);
	uint32_t   local_instance = 0, local_leaf = 0;
	Refill(ctx, site);
	for (uint32_t half = 0; half < s.lane_count; half++) {
		const auto lane = Lane(s, site, half);
		const auto [instance, leaf] = LaneMinimum(s, lane);
		if (half == 0) {
			local_instance = instance;
			local_leaf     = leaf;
		} else {
			const auto smaller = KeyLess(s, instance, leaf, local_instance, local_leaf);
			local_instance     = Select(s, u, smaller, instance, local_instance);
			local_leaf         = Select(s, u, smaller, leaf, local_leaf);
		}
	}
	const bool lane_leaf = (IR::RayQueryVariant(inst.Flags<uint32_t>()) & IR::RayQueryKeyLaneLeaf) != 0;
	const auto instance  = GroupMin(s, local_instance);
	const auto leaf      = lane_leaf ? U32(s, None)
	                                 : GroupMin(s, Select(s, u, Equal(s, local_instance, instance), local_leaf, U32(s, None)));
	const auto has   = Binary(s, spv::OpINotEqual, b, instance, U32(s, None));
	Store(s, s.ray_query_selected_instance[site], instance);
	Store(s, s.ray_query_selected_leaf[site], leaf);
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(s, 4), result, instance, leaf,
	                      Select(s, u, has, U32(s, 1), U32(s, 0)), U32(s, 0));
	return result;
}

uint32_t EmitRayQueryMatch(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = Lane(s, site, ctx.half);
	const auto u    = TypeU32(s);
	const auto [instance, leaf] = LaneMinimum(s, lane);
	const bool lane_leaf = (IR::RayQueryVariant(inst.Flags<uint32_t>()) & IR::RayQueryKeyLaneLeaf) != 0;
	const auto same_instance = Equal(s, instance, Load(s, u, s.ray_query_selected_instance[site]));
	const auto matched = LogicalAnd(
	    s, lane_leaf ? same_instance
	                 : LogicalAnd(s, same_instance, Equal(s, leaf, Load(s, u, s.ray_query_selected_leaf[site]))),
	    Binary(s, spv::OpINotEqual, TypeBool(s), instance, U32(s, None)));
	const auto state = And(s, Load(s, u, lane.state), ~MatchedBit);
	Store(s, lane.state, Or(s, state, Select(s, u, matched, U32(s, MatchedBit), U32(s, 0))));
	return matched;
}

uint32_t EmitRayQuerySelectedLeaf(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	if ((IR::RayQueryVariant(inst.Flags<uint32_t>()) & IR::RayQueryKeyLaneLeaf) != 0) {
		return LaneMinimum(s, Lane(s, site, ctx.half)).second;
	}
	return Load(s, TypeU32(s), s.ray_query_selected_leaf[site]);
}

uint32_t EmitRayQueryResult(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s     = ctx.state;
	const auto site  = SiteOf(ctx, inst);
	const auto lane  = Lane(s, site, ctx.half);
	const auto u     = TypeU32(s), wide = TypeU64(s);
	const auto state = Load(s, u, lane.state);
	const auto has   = BitSet(s, state, CommittedBit);
	const auto instance = Load(s, wide, lane.result_instance);
	switch (static_cast<IR::RayQueryResultKind>(IR::RayQueryVariant(inst.Flags<uint32_t>()))) {
		case IR::RayQueryResultKind::Has: return Select(s, u, has, U32(s, 1), U32(s, 0));
		case IR::RayQueryResultKind::T: return Load(s, u, lane.result_t);
		case IR::RayQueryResultKind::Word: return Load(s, u, lane.result_word);
		case IR::RayQueryResultKind::InstanceLo: return Unary(s, spv::OpUConvert, u, instance);
		case IR::RayQueryResultKind::InstanceHi:
			return Unary(s, spv::OpUConvert, u, Binary(s, spv::OpShiftRightLogical, wide, instance, ConstantU64(s, 32)));
		case IR::RayQueryResultKind::U: return Load(s, u, lane.result_u);
		case IR::RayQueryResultKind::V: return Load(s, u, lane.result_v);
		case IR::RayQueryResultKind::Node: return Load(s, u, lane.result_node);
	}
	ctx.Fail(inst, "unknown ray query result");
}

void EmitRayQueryAlpha(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = Lane(s, site, ctx.half);
	for (uint32_t i = 0; i < static_cast<uint32_t>(AlphaArgument::Count); i++) {
		Store(s, AlphaPointer(s, lane, i), ctx.Arg(inst, i));
	}
}

void EmitRayQueryAlphaDistance(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s    = ctx.state;
	const auto site = SiteOf(ctx, inst);
	const auto lane = Lane(s, site, ctx.half);
	if (lane.alpha_distance == 0) {
		ctx.Fail(inst, "ray query alpha distance without alpha test");
	}
	Store(s, lane.alpha_distance, ctx.Arg(inst, 0));
}

void EmitRayQueryMark(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s       = ctx.state;
	const auto site    = SiteOf(ctx, inst);
	const auto lane    = Lane(s, site, ctx.half);
	const auto u       = TypeU32(s);
	const auto state   = Load(s, u, lane.state);
	const auto matched = BitSet(s, state, MatchedBit);
	switch (static_cast<IR::RayQueryMarkKind>(IR::RayQueryVariant(inst.Flags<uint32_t>()))) {
		case IR::RayQueryMarkKind::Resolve:
			EmitIfCondition(s, matched, [&] { PopKey(s, lane); });
			Store(s, lane.state, And(s, state, ~MatchedBit));
			return;
		case IR::RayQueryMarkKind::Exit:
			Store(s, lane.state, U32(s, DoneBit));
			ClearKeys(s, lane);
			return;
	}
	ctx.Fail(inst, "unknown ray query mark");
}

}

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr uint32_t Candidate = spv::RayQueryIntersectionRayQueryCandidateIntersectionKHR;

uint32_t RayQuery(EmitterState& s) {
	auto& variable = s.ray_query_mirror_variables[s.lane_half];
	if (variable == 0) {
		variable = s.builder.DefineGlobalVariable(
		    TypePointer(s, spv::StorageClassPrivate, s.builder.Type(spv::OpTypeRayQueryKHR)),
		    spv::StorageClassPrivate);
	}
	return variable;
}

uint32_t ContextAddress(EmitterState& s) {
	const auto dword = s.program.bindings.ray_query_dword;
	return PackU64(s, EmitShaderDataDwordLoad(s, dword), EmitShaderDataDwordLoad(s, dword + 1));
}

uint32_t LoadContext(EmitterState& s, uint32_t address, uint32_t offset, uint32_t type,
                     uint32_t alignment) {
	const auto at      = Binary(s, spv::OpIAdd, TypeU64(s), address, ConstantU64(s, offset));
	const auto pointer = Unary(s, spv::OpConvertUToPtr,
	                           TypePointer(s, spv::StorageClassPhysicalStorageBuffer, type), at);
	const auto value   = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, type, value, pointer, spv::MemoryAccessAlignedMask,
	                      alignment);
	return value;
}

}

uint32_t EmitRayQueryValid(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      s       = ctx.state;
	const auto address = ContextAddress(s);
	const auto present = Binary(s, spv::OpINotEqual, TypeBool(s), address, ConstantU64(s, 0));
	return EmitValueOrDefaultIfCondition(s, present, TypeBool(s), ConstantBool(s, false), [&] {
		return Binary(s, spv::OpINotEqual, TypeBool(s), LoadContext(s, address, ContextValid, TypeU32(s), 4),
		              ConstantU32(s, 0));
	});
}

void EmitRayQueryInitialize(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	EmitIfCondition(s, ctx.Arg(inst, 8), [&] {
		const auto              f = TypeF32(s), b = TypeBool(s), zero = ConstantF32Value(s, 0.0f);
		std::array<uint32_t, 7> ray {};
		for (uint32_t i = 0; i < ray.size(); ++i) {
			ray[i] = Unary(s, spv::OpBitcast, f, ctx.Arg(inst, i));
		}
		auto traced = Binary(s, spv::OpFOrdGreaterThanEqual, b, ray[6], zero);
		for (uint32_t i = 0; i < 6; ++i) {
			const auto finite = Binary(s, spv::OpFOrdLessThan, b, EmitFAbsValue(s, ray[i]),
			                           ConstantF32(s, 0x7f800000u));
			traced            = Binary(s, spv::OpLogicalAnd, b, traced, finite);
		}
		for (auto& value: ray) {
			value = Select(s, f, traced, value, zero);
		}
		const auto vector = [&](uint32_t first) {
			const auto result = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(s, 3), result,
			                      ray[first], ray[first + 1], ray[first + 2]);
			return result;
		};
		const auto structure = Unary(s, spv::OpConvertUToAccelerationStructureKHR,
		                             s.builder.Type(spv::OpTypeAccelerationStructureKHR),
		                             LoadContext(s, ContextAddress(s), ContextTlas, TypeU64(s), 8));
		s.builder.AddFunction(spv::OpRayQueryInitializeKHR, RayQuery(s), structure,
		                      ConstantU32(s, 0),
		                      Select(s, TypeU32(s), traced, ctx.Arg(inst, 7), ConstantU32(s, 0)),
		                      vector(0), zero, vector(3), ray[6]);
	});
}

uint32_t EmitRayQueryProceed(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    s, ctx.Arg(inst, 0), TypeBool(s), ConstantBool(s, false),
	    [&] { return Unary(s, spv::OpRayQueryProceedKHR, TypeBool(s), RayQuery(s)); });
}

uint32_t EmitRayQueryCandidate(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    s, ctx.Arg(inst, 0), TypeU32Vector(s, 4), ConstantU32CompositeZero(s, 4), [&] {
		    const auto query = RayQuery(s);
		    const auto entry = TriangleEntry(s, query, Candidate, [&] {
			    return LoadContext(s, ContextAddress(s), ContextSlots, TypeU64(s), 8);
		    });
		    const auto [weight_u, weight_v] = GuestBarycentrics(s, query, Candidate, entry.remap);
		    const auto u      = Unary(s, spv::OpBitcast, TypeU32(s), weight_u);
		    const auto v      = Unary(s, spv::OpBitcast, TypeU32(s), weight_v);
		    const auto record = Binary(s, spv::OpShiftRightLogical, TypeU32(s), entry.entry, ConstantU32(s, 3));
		    const auto t      = Unary(s, spv::OpBitcast, TypeU32(s),
			                          Binary(s, spv::OpRayQueryGetIntersectionTKHR, TypeF32(s), query,
			                                 ConstantU32(s, Candidate)));
		    const auto result = s.builder.AllocateId();
		    s.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(s, 4), result, t, u, v,
			                      record);
		    return result;
	    });
}

uint32_t EmitRayQueryCandidateInstance(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	return EmitValueOrDefaultIfCondition(s, ctx.Arg(inst, 0), TypeU64(s), ConstantU64(s, 0), [&] {
		const auto index  = Binary(s, spv::OpRayQueryGetIntersectionInstanceIdKHR, TypeU32(s),
		                           RayQuery(s), ConstantU32(s, Candidate));
		const auto offset = Binary(s, spv::OpShiftLeftLogical, TypeU64(s),
		                           Unary(s, spv::OpUConvert, TypeU64(s), index), ConstantU32(s, 7));
		return Binary(s, spv::OpIAdd, TypeU64(s),
		              LoadContext(s, ContextAddress(s), ContextInstanceBase, TypeU64(s), 8), offset);
	});
}

uint32_t EmitRayQueryCandidateFrontFace(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	return EmitValueOrDefaultIfCondition(
	    s, ctx.Arg(inst, 0), TypeBool(s), ConstantBool(s, false), [&] {
		    return Binary(s, spv::OpRayQueryGetIntersectionFrontFaceKHR, TypeBool(s), RayQuery(s),
			              ConstantU32(s, Candidate));
	    });
}

void EmitRayQueryConfirm(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	EmitIfCondition(s, ctx.Arg(inst, 0), [&] {
		s.builder.AddFunction(spv::OpRayQueryConfirmIntersectionKHR, RayQuery(s));
	});
}

}
