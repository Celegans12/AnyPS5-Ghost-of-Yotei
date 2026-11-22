#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SERIALIZE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SERIALIZE_H_

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Writer {
	std::vector<uint8_t>                      bytes;
	std::unordered_map<const Inst*, uint32_t> index;
	EncodeFailure                             failure;
	const Inst*                               owner = nullptr;

	void Raw(const void* data, size_t size) {
		const auto* begin = static_cast<const uint8_t*>(data);
		bytes.insert(bytes.end(), begin, begin + size);
	}

	void Fail(EncodeFailure::Reason reason, uint32_t type) {
		if (failure.Failed()) return;
		failure.reason         = reason;
		failure.type           = type;
		failure.in_instruction = owner != nullptr;
		failure.opcode         = owner != nullptr ? owner->GetOpcode() : ValueOpcode::Void;
	}
};

struct Reader {
	std::span<const uint8_t> bytes;
	std::vector<Inst*>       insts;
	bool                     ok = true;
	size_t                   allocation_budget = std::numeric_limits<size_t>::max();
	bool                     shader_info_has_relative_vgprs = true;

	void Raw(void* data, size_t size) {
		ok = ok && size <= bytes.size();
		if (!ok) return;
		std::copy_n(bytes.data(), size, static_cast<uint8_t*>(data));
		bytes = bytes.subspan(size);
	}
	uint32_t Count(uint32_t count, size_t element_size = 1) {
		ok = ok && count <= bytes.size() && count <= allocation_budget / element_size;
		if (ok) allocation_budget -= count * element_size;
		return ok ? count : 0u;
	}
};

template <class S> constexpr bool Reading = std::is_same_v<S, Reader>;
template <class T, template <class...> class Of> constexpr bool IsA = false;
template <template <class...> class Of, class... A> constexpr bool IsA<Of<A...>, Of> = true;
template <class T, class U> concept Is = std::is_same_v<std::remove_const_t<T>, U>;

template <class S, class T> void Io(S& io, T& value);

template <class S, class... T> void IoAll(S& io, T&&... values) {
	(Io(io, values), ...);
}

inline void IoValue(Writer& writer, const Value& value) {
	auto     type = Type::Opaque;
	uint64_t bits = 0;
	if (const auto* inst = value.TryInstruction()) {
		if (const auto found = writer.index.find(inst); found != writer.index.end()) {
			bits = found->second;
		} else {
			writer.Fail(EncodeFailure::Reason::UnknownInstruction, static_cast<uint32_t>(Type::Opaque));
			type = Type::Void;
		}
	} else {
		type = value.GetType();
		switch (type) {
			case Type::Void: break;
			case Type::ScalarReg: bits = RegIndex(value.ScalarRegister()); break;
			case Type::VectorReg: bits = RegIndex(value.VectorRegister()); break;
			case Type::U1: bits = value.U1(); break;
			case Type::U8: bits = value.U8(); break;
			case Type::U16: bits = value.U16(); break;
			case Type::U32: bits = value.U32(); break;
			case Type::U64: bits = value.U64(); break;
			case Type::F16: bits = value.F16Bits(); break;
			case Type::F32: bits = std::bit_cast<uint32_t>(value.F32Value()); break;
			default:
				writer.Fail(EncodeFailure::Reason::ValueType, static_cast<uint32_t>(type));
				type = Type::Void;
				break;
		}
	}
	IoAll(writer, type, bits);
}

inline void IoValue(Reader& reader, Value& value) {
	auto     type = Type::Void;
	uint64_t bits = 0;
	IoAll(reader, type, bits);
	switch (type) {
		case Type::Void: value = {}; break;
		case Type::Opaque:
			reader.ok = reader.ok && bits < reader.insts.size();
			if (reader.ok) value = Value(reader.insts[bits]);
			break;
		case Type::ScalarReg: value = Value(static_cast<ScalarReg>(bits)); break;
		case Type::VectorReg: value = Value(static_cast<VectorReg>(bits)); break;
		case Type::U1: value = Value(bits != 0u); break;
		case Type::U8: value = Value(static_cast<uint8_t>(bits)); break;
		case Type::U16: value = Value(static_cast<uint16_t>(bits)); break;
		case Type::U32: value = Value(static_cast<uint32_t>(bits)); break;
		case Type::U64: value = Value(bits); break;
		case Type::F16: value = Value::F16(static_cast<uint16_t>(bits)); break;
		case Type::F32: value = Value::F32(std::bit_cast<float>(static_cast<uint32_t>(bits))); break;
		default: reader.ok = false; break;
	}
}

template <class S, class T> void Io(S& io, T& value) {
	using U = std::remove_const_t<T>;
	static_assert(!std::is_pointer_v<U>, "pointers do not persist");
	if constexpr (Reading<S> && std::is_same_v<U, bool>) {
		uint8_t encoded = 0;
		io.Raw(&encoded, sizeof(encoded));
		io.ok = io.ok && encoded <= 1;
		value = encoded == 1;
	} else if constexpr (std::is_arithmetic_v<U> || std::is_enum_v<U>) {
		io.Raw(&value, sizeof(U));
	} else if constexpr (std::is_same_v<U, Value>) {
		IoValue(io, value);
	} else if constexpr (IsA<U, std::vector> || IsA<U, std::basic_string>) {
		auto count = static_cast<uint32_t>(value.size());
		Io(io, count);
		using E = typename U::value_type;
		if constexpr (Reading<S>) value.resize(io.Count(count, sizeof(E)));
		if constexpr (std::is_arithmetic_v<E> || std::is_enum_v<E>) {
			io.Raw(value.data(), value.size() * sizeof(E));
		} else {
			for (auto& element: value) Io(io, element);
		}
	} else if constexpr (IsA<U, std::optional>) {
		bool has = value.has_value();
		Io(io, has);
		if constexpr (Reading<S>) {
			if (has) value = typename U::value_type {};
			else value.reset();
		}
		if (has) Io(io, *value);
	} else {
		static_assert(requires { std::tuple_size<U>::value; }, "struct without an Io overload");
		std::apply([&](auto&... elements) { IoAll(io, elements...); }, value);
	}
}

#define KYTY_IO_FIELDS(Struct, ...)                                                                \
	template <class Stream, Is<Struct> Object> void Io(Stream& io, Object& object) {               \
		auto& [__VA_ARGS__] = object;                                                              \
		IoAll(io, __VA_ARGS__);                                                                    \
	}
KYTY_IO_FIELDS(MemoryInfo, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x,
               y, z)
KYTY_IO_FIELDS(BufferWriteSpan, a, b)
KYTY_IO_FIELDS(BufferResource, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w)
KYTY_IO_FIELDS(ImageResource, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w,
               x, y, z, A, B, C, D, E)
KYTY_IO_FIELDS(SamplerResource, a, b, c, d, e, f, g, h, i, j, k)
KYTY_IO_FIELDS(SampledResourcePair, a, b, c)
KYTY_IO_FIELDS(StageInput, a, b, c, d, e)
KYTY_IO_FIELDS(StageOutput, a, b, c, d)
KYTY_IO_FIELDS(ShaderInfo::DmaWindow, a, b, c)
KYTY_IO_FIELDS(ShaderInfo::DmaDescriptorTable, a, b, c)
KYTY_IO_FIELDS(ShaderInfo::DmaRecordTable, a, b, c)
KYTY_IO_FIELDS(ShaderInfo::SelfSizedBuffer, a, b, c)
template <class Stream, Is<ShaderInfo> Object>
void Io(Stream& io, Object& object) {
	auto& [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z,
	       relative_vgprs, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X,
	       Y, Z, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9] = object;
	IoAll(io, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z);
	if constexpr (requires { io.shader_info_has_relative_vgprs; }) {
		if (io.shader_info_has_relative_vgprs) IoAll(io, relative_vgprs);
	} else {
		IoAll(io, relative_vgprs);
	}
	IoAll(io, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, a0,
	      a1, a2, a3, a4, a5, a6, a7, a8, a9);
}
KYTY_IO_FIELDS(DescriptorSource::IndirectDescriptor::SelectorRead::CountedRows, a, b)
KYTY_IO_FIELDS(DescriptorSource::IndirectDescriptor::SelectorRead::MaskRanks, a, b, c)
KYTY_IO_FIELDS(DescriptorSource::IndirectDescriptor::SelectorRead, a, b, c, d, e, f, g, h, i, j, k,
               l, m, n, o)
KYTY_IO_FIELDS(DescriptorSource::IndirectDescriptor::PointerTable, a, b, c, d, e, f, g, h, i)
KYTY_IO_FIELDS(DescriptorSource::IndirectDescriptor, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p,
               q, r, s, t)
KYTY_IO_FIELDS(DescriptorSource, a, b, c)
KYTY_IO_FIELDS(SrtRead, a, b)
KYTY_IO_FIELDS(ResourceBlock, a, b, c, d, e)
KYTY_IO_FIELDS(UniformFill, a, b, c, d, e, f, g, h)
KYTY_IO_FIELDS(UniformFillPlan, a, b, c, d)
KYTY_IO_FIELDS(ResourceSpecialization::Buffer, a, b, c, d, e, f, g, h)
KYTY_IO_FIELDS(ResourceSpecialization::Image, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q)
KYTY_IO_FIELDS(ResourceSpecialization::Sampler, a, b, c)
KYTY_IO_FIELDS(ResourceSpecialization, a, b, c)
KYTY_IO_FIELDS(Bindings, a, b, c, d, e, f, g, h, i, j, k, l, m)
KYTY_IO_FIELDS(TablePlan::Operand, a, b)
KYTY_IO_FIELDS(TablePlan::Slot, a, b, c, d, e, f)
KYTY_IO_FIELDS(TablePlan::Descriptor, a, b)
KYTY_IO_FIELDS(TablePlan, a, b, c, d, e, f, g, h)
KYTY_IO_FIELDS(BdaReadWord, a, b)
KYTY_IO_FIELDS(BdaReadSpan, a, b, c, d, e)
KYTY_IO_FIELDS(BdaReadPlan, a, b)
KYTY_IO_FIELDS(CompiledShaderInfo, a, b, c, d, e, f, g, h, i, j, k)
KYTY_IO_FIELDS(Spirv::Profile, a, b, c, d, e, f, g, h, i, j)
#if defined(ANYPS5_KYTY)
KYTY_IO_FIELDS(ShaderTessellationInputInfo, a, b, c, d, e, f, g, h, i, j)
#else
KYTY_IO_FIELDS(ShaderTessellationInputInfo, a, b, c, d, e, f, g, h, i)
#endif
KYTY_IO_FIELDS(RectListParameter, a, b, c)
#undef KYTY_IO_FIELDS

}

#endif
