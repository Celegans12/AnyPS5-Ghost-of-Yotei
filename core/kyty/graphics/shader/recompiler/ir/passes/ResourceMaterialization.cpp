#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/runtimeSwitches.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <optional>
#include <array>
#include <atomic>
#include <utility>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <numeric>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr uint64_t AddressMask            = 0x0000ffffffffffffull;

struct DescriptorValueHash {
	size_t operator()(const DescriptorValue& value) const noexcept {
		uint64_t hash = 0xcbf29ce484222325ull ^ value.dword_count;
		for (const auto dword: value.dwords) {
			hash = (hash ^ dword) * 0x100000001b3ull;
			hash ^= hash >> 29u;
		}
		return static_cast<size_t>(hash);
	}
};

constexpr size_t StableSrtWords = 4096u;
struct RecordBlocks {
	size_t                mapping_offset = 0;
	size_t                entries        = 0;
	std::vector<uint32_t> blocks;
};
thread_local std::vector<RecordBlocks> t_record_blocks;

void FlushRecordBlocks(std::vector<uint32_t>& flattened_srt) {
	for (auto& record: t_record_blocks) {
		const auto base = static_cast<uint32_t>(flattened_srt.size());
		for (size_t entry = 0; entry < record.entries; ++entry) {
			flattened_srt[record.mapping_offset + 2u + entry * 2u] += base;
		}
		flattened_srt.insert(flattened_srt.end(), record.blocks.begin(), record.blocks.end());
	}
	t_record_blocks.clear();
}

constexpr size_t MinIndirectMappingCapacity = 8u;
constexpr uint64_t MaxRecordTableElements = uint64_t {1} << 21u;
std::atomic<size_t> g_compute_indirect_keys {MinIndirectMappingCapacity};
std::atomic<size_t> g_compute_indirect_images {1u};
std::atomic<bool> g_stable_indirect_layouts {false};

void ReportUnknownTileMode(const char* path, const ResourcePlan& program, uint32_t resource,
                           uint64_t address, const DescriptorValue& value) {
	static const bool enabled = [] {
		const char* setting = std::getenv("KYTY_TILED_DIAG");
		return setting != nullptr && setting[0] != 0 && setting[0] != '0';
	}();
	if (!enabled || (value.dwords[0] == 0u && (value.dwords[1] & 0xffu) == 0u)) return;
	switch (static_cast<Prospero::TileMode>((value.dwords[3] >> 20u) & 0x1fu)) {
		case Prospero::TileMode::kLinear:
		case Prospero::TileMode::kStandard256B:
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
		case Prospero::TileMode::kPrt:
		case Prospero::TileMode::kDepth:
		case Prospero::TileMode::kRenderTarget: return;
		default: break;
	}
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1u, std::memory_order_relaxed) >= 256u) return;
	std::fprintf(stderr,
	             "[kyty-tiled-diag] %s: shader 0x%016llx resource %u plan %p descriptor at 0x%llx "
	             "words %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	             path, static_cast<unsigned long long>(program.shader_hash), resource,
	             static_cast<const void*>(&program), static_cast<unsigned long long>(address),
	             value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
	             value.dwords[4], value.dwords[5], value.dwords[6], value.dwords[7]);
}

bool SpecializationFail(std::string_view message) {
	std::fprintf(stderr, "shader resource specialization failed: %.*s\n",
	             static_cast<int>(message.size()), message.data());
	return false;
}

Decoder::ImageDimension DescriptorDimension(const DescriptorValue&  descriptor,
                                            Decoder::ImageDimension requested) {
	const bool is_array = requested == Decoder::ImageDimension::Dim1DArray ||
	                      requested == Decoder::ImageDimension::Dim2DArray ||
	                      requested == Decoder::ImageDimension::Dim2DMsaaArray;
	switch (static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu)) {
		case Prospero::ImageType::kColor1D: return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor1DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim1DArray;
			}
			return Decoder::ImageDimension::Dim1D;
		case Prospero::ImageType::kColor3D: return Decoder::ImageDimension::Dim3D;
		case Prospero::ImageType::kCube: return Decoder::ImageDimension::Dim2DArray;
		case Prospero::ImageType::kColor2DArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DArray;
			}
			return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaaArray:
			if (is_array) {
				return Decoder::ImageDimension::Dim2DMsaaArray;
			}
			return Decoder::ImageDimension::Dim2DMsaa;
		case Prospero::ImageType::kColor2D: return Decoder::ImageDimension::Dim2D;
		case Prospero::ImageType::kColor2DMsaa: return Decoder::ImageDimension::Dim2DMsaa;
		default: return Decoder::ImageDimension::Unknown;
	}
}

bool NullImageDescriptor(const DescriptorValue& descriptor) {
	return descriptor.dwords[0] == 0 && (descriptor.dwords[1] & 0xffu) == 0;
}

bool NullBufferDescriptor(Value value) {
	const auto* handle = value.Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
	    handle->NumArgs() != 4u) return false;
	for (size_t index = 0; index < 4u; ++index)
		if (handle->Arg(index).Resolve() != Value(0u)) return false;
	return true;
}

bool ValidImageDescriptor(const DescriptorValue& descriptor, bool r128 = false) {
	const auto& words = descriptor.dwords;
	if ((words[1] & 0x20000000u) != 0u || (words[2] & 0x70003000u) != 0u ||
	    (!r128 && ((words[4] & 0xe000e000u) != 0u || (words[5] & 0xf9000000u) != 0u ||
	               (words[6] & 0x00007b00u) != 0u))) {
		return false;
	}
	const auto type   = static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu);
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	if (type < Prospero::ImageType::kColor1D || format == Prospero::BufferFormat::kInvalid ||
	    format > Prospero::BufferFormat::kBc7Srgb) {
		return false;
	}
	if (r128 && type != Prospero::ImageType::kColor1D && type != Prospero::ImageType::kColor2D &&
	    type != Prospero::ImageType::kColor2DMsaa) {
		return false;
	}
	if (!IsValidImageSwizzle(words[3] & 0xfffu)) return false;
	const bool array = type == Prospero::ImageType::kColor1DArray ||
	                   type == Prospero::ImageType::kColor2DArray ||
	                   type == Prospero::ImageType::kColor2DMsaaArray ||
	                   type == Prospero::ImageType::kCube;
	if (array && ((words[4] >> 16u) & 0x1fffu) > (words[4] & 0x1fffu)) {
		return false;
	}
	const auto base_level = (words[3] >> 12u) & 0xfu;
	const auto last_level = (words[3] >> 16u) & 0xfu;
	if (type == Prospero::ImageType::kColor2DMsaa ||
	    type == Prospero::ImageType::kColor2DMsaaArray) {
		const auto max_mip    = (descriptor.dwords[5] >> 4u) & 0xfu;
		return base_level == 0 && last_level >= 1 && last_level <= 3 &&
		       (r128 || max_mip == last_level);
	}
	return base_level <= last_level;
}

uint32_t DescriptorImageSwizzle(const DescriptorValue& descriptor) {
	return descriptor.dwords[3] & 0xfffu;
}

uint32_t CanonicalStorageSwizzle(uint32_t swizzle, Prospero::BufferFormat format) {
	const auto info = Format::GetFormatInfo(format);
	if (info.type == Format::ComponentType::Unknown || info.component_count == 0u ||
	    info.component_count > 4u) {
		return swizzle;
	}
	for (uint32_t channel = 0; channel < info.component_count; channel++) {
		uint32_t source = 0;
		while (source < 4u && ((swizzle >> (source * 3u)) & 7u) != 4u + channel) source++;
		if (source != channel) return swizzle;
	}
	return ShaderImageIdentitySwizzle;
}

Prospero::BufferFormat ImageConversionFormat(Prospero::BufferFormat format) {
	return Prospero::RemapTextureFormat(format) != format ? format
	                                                      : Prospero::BufferFormat::kInvalid;
}

bool ValidatePackedUnormSampling(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                                 uint32_t image_index) {
	ShaderTextureResource texture;
	std::copy_n(snapshot.images[image_index].dwords.begin(), std::size(texture.fields),
	            texture.fields);
	if (texture.Type() != Prospero::ImageType::kColor2D ||
	    texture.BaseLevel() != texture.LastLevel() || texture.MinLod() != 0u ||
	    texture.CornerSample()) {
		return SpecializationFail("packed UNorm sampling requires a centered, single-mip 2D image");
	}
	for (const auto& pair: program.info.sampled_pairs) {
		if (pair.image != image_index) continue;
		if (pair.sampler >= snapshot.samplers.size() ||
		    program.descriptor_sources[program.info.samplers[pair.sampler].source].indirect_descriptor) {
			return SpecializationFail("packed UNorm sampling requires a direct sampler");
		}
		ShaderSamplerResource sampler;
		std::copy_n(snapshot.samplers[pair.sampler].dwords.begin(), std::size(sampler.fields),
		            sampler.fields);
		const auto linear = static_cast<uint32_t>(Prospero::SamplerFilter::kBilinear);
		const auto clamp = static_cast<uint32_t>(Prospero::SamplerClampMode::kClampLastTexel);
		if (sampler.XyMagFilter() != linear || sampler.XyMinFilter() != linear ||
		    sampler.ClampX() != clamp || sampler.ClampY() != clamp ||
		    sampler.ForceUnormCoords() || sampler.MaxAnisoRatio() != 0u ||
		    sampler.FilterMode() != 0u || sampler.MinLod() != 0u || sampler.ForceSrgb() ||
		    sampler.TruncCoord() || (sampler.fields[0] & (1u << 19u)) != 0u) {
			return SpecializationFail(fmt::format(
			    "packed UNorm image {} requires normalized bilinear clamp sampling (sampler {})",
			    image_index, pair.sampler));
		}
	}
	return true;
}

enum class SamplerClass : uint8_t { Float, Integer, Compare };

SamplerClass ClassifySampler(const ImageResource& image) {
	if (image.depth_compare) {
		return SamplerClass::Compare;
	}
	return image.numeric_class == Prospero::TextureNumericClass::Uint ||
	               image.numeric_class == Prospero::TextureNumericClass::Sint
	           ? SamplerClass::Integer : SamplerClass::Float;
}

bool DescriptorIsCube(const DescriptorValue& descriptor) {
	return static_cast<Prospero::ImageType>((descriptor.dwords[3] >> 28u) & 0xfu) ==
	       Prospero::ImageType::kCube;
}

uint32_t ImageMipCount(const ImageResource& image, const DescriptorValue& descriptor) {
	if (image.mip_mode != ImageMipMode::Dynamic || NullImageDescriptor(descriptor)) {
		return 1;
	}
	const auto base = (descriptor.dwords[3] >> 12u) & 0xfu;
	const auto last = (descriptor.dwords[3] >> 16u) & 0xfu;
	return last - base + 1u;
}

bool DecodeBufferDescriptor(const DescriptorValue& descriptor, ShaderBufferResource& result) {
	if (descriptor.dword_count != std::size(result.fields)) {
		return false;
	}
	std::copy_n(descriptor.dwords.begin(), std::size(result.fields), result.fields);
	return true;
}

const DescriptorSource* Source(const ResourcePlan& program, uint32_t source) {
	if (source >= program.descriptor_sources.size()) {
		return nullptr;
	}
	return &program.descriptor_sources[source];
}

bool ReadTable(uint64_t base, uint64_t size, uint64_t offset,
               const SrtRuntime& runtime, std::span<uint32_t> words) {
	const auto count = std::min<uint64_t>(words.size(), offset < size ? (size - offset) / 4u : 0u);
	std::ranges::fill(words.subspan(count), 0u);
	if (count == 0u) {
		return true;
	}
	if (base > AddressMask || offset > AddressMask - base) {
		return false;
	}
	const auto address = base + offset;
	const auto prefix = words.first(count);
	return prefix.size_bytes() - 1u <= AddressMask - address &&
	       runtime.read_specialization_memory != nullptr &&
	       runtime.read_specialization_memory(runtime.userdata, address, prefix);
}

bool ReadScalarTable(uint64_t base, uint64_t size, uint64_t offset,
                     const SrtRuntime& runtime, std::span<uint32_t> words) {
	return ReadTable(base & AddressMask & ~uint64_t {3}, size, offset & ~uint64_t {3}, runtime, words);
}

bool TableInGuestMemory(uint64_t base, uint64_t size, const SrtRuntime& runtime) {
	std::array<uint32_t, 1> word {};
	return base != 0u && size >= 4u && ReadScalarTable(base, size, 0u, runtime, word) &&
	       ReadScalarTable(base, size, size - 4u, runtime, word);
}

bool NormalizeIndirectBuffer(DescriptorValue& value, bool indexed, const BufferResource& resource) {
	ShaderBufferResource descriptor;
	if (!DecodeBufferDescriptor(value, descriptor)) return false;
	const bool formatted_read = resource.formatted && resource.read && !resource.written;
	if (descriptor.Type() != 0u || descriptor.Format() == Prospero::BufferFormat::kInvalid ||
	    (!formatted_read && (descriptor.Base48() == 0u || descriptor.GetSize() == 0u))) {
		value.dwords.fill(0u);
		return true;
	}
	if (descriptor.SwizzleEnabled() || descriptor.AddTid() ||
	    descriptor.OutOfBounds() != (indexed ? 0u : 3u)) {
		return SpecializationFail(fmt::format(
		    "indirect buffers require exact linear index or byte bounds: "
		    "descriptor={:08x},{:08x},{:08x},{:08x} swizzle={} add_tid={} oob={} "
		    "indexed={} formatted={} read={} written={} max_extent={}",
		    value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
		    descriptor.SwizzleEnabled(), descriptor.AddTid(), descriptor.OutOfBounds(), indexed,
		    resource.formatted, resource.read, resource.written, resource.max_byte_extent));
	}
	if (resource.formatted) {
		const auto format = Format::GetFormatInfo(descriptor.Format());
		if (!indexed || format.type == Format::ComponentType::Unknown)
			return SpecializationFail("indirect formatted buffers require a supported indexed format");
		if (!formatted_read) {
			const bool scalar = resource.max_byte_extent == 4u &&
			    (descriptor.Format() == Prospero::BufferFormat::k32UInt ||
			     descriptor.Format() == Prospero::BufferFormat::k8SInt);
			const bool rgba16 = resource.max_byte_extent == 16u &&
			    descriptor.Format() == Prospero::BufferFormat::k16_16_16_16Float;
			if ((!scalar && !rgba16) || descriptor.Stride() != format.byte_size ||
			    descriptor.DstSelXYZW() != (rgba16 ? DstSel(4, 5, 6, 7) : DstSel(4, 0, 0, 0)))
				return SpecializationFail("indirect formatted stores require tight identity SInt8, UInt32 or RGBA16Float records");
		}
		if (formatted_read &&
		    (descriptor.Base48() == 0u || descriptor.Stride() == 0u || descriptor.NumRecords() == 0u)) {
			value.dwords[1] &= ~(0x3fffu << 16u);
			value.dwords[2] = 0u;
			return true;
		}
		const auto alignment = std::min(4u, format.byte_size);
		if (descriptor.Stride() < format.byte_size ||
		    (descriptor.Stride() & (alignment - 1u)) != 0u ||
		    (descriptor.Base48() & (alignment - 1u)) != 0u)
			return SpecializationFail("indirect formatted buffers require aligned full-format records");
	} else if ((!indexed && descriptor.Stride() != 1u) ||
	           (indexed && resource.max_byte_extent != 1u && descriptor.Stride() != 4u)) {
		return SpecializationFail("indirect buffer stores require exact linear DWORD or byte bounds");
	}
	return true;
}

bool IsBoundedDescriptorTable(const ResourcePlan& program,
                              const DescriptorSource::IndirectDescriptor& indirect) {
	const auto* table = Source(program, indirect.table_source);
	return indirect.sources.empty() && (!indirect.selector || indirect.table_scalar) &&
	       (indirect.key_count.IsEmpty() || indirect.table_maximum) && indirect.workgroup_axis == UINT32_MAX &&
	       table != nullptr && (table->dword_count == 4u || indirect.table_scalar);
}

bool CanReadIndexedSelector(const ShaderBufferResource& material, uint32_t offset) {
	if (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
	    material.OutOfBounds() != 0u) return false;
	if (material.Format() == Prospero::BufferFormat::kInvalid || material.NumRecords() == 0u ||
	    offset >= material.Stride()) return true;
	return material.Base48() != 0u &&
	       ((material.Base48() | material.Stride() | offset) & 3u) == 0u &&
	       material.NumRecords() <= MaxIndirectDescriptorProbes &&
	       uint64_t {material.NumRecords() - 1u} * material.Stride() + offset + 4u <=
	           uint64_t {UINT32_MAX} + 1u;
}

bool EvaluateIndirectCount(const ResourcePlan& program, const DescriptorSource::IndirectDescriptor& indirect,
                           const SrtRuntime& runtime, uint32_t& count) {
	if (indirect.workgroup_axis != UINT32_MAX) {
		if (runtime.workgroup_counts.size() != 3u || indirect.workgroup_axis >= 3u) return false;
		count = std::ranges::find(runtime.workgroup_counts, 0u) != runtime.workgroup_counts.end()
		    ? 0u : runtime.workgroup_counts[indirect.workgroup_axis];
		return true;
	}
	if (!WalkedValue(program, indirect.key_count, count)) return false;
	if (indirect.signed_count && std::bit_cast<int32_t>(count) < 0) count = 0;
	return true;
}

bool ResolveSelectorKeys(const ResourcePlan& program, const DescriptorSource::IndirectDescriptor& indirect,
                         const DescriptorValue& material_value, const SrtRuntime& runtime,
                         std::vector<uint32_t>& keys) {
	const auto* selector = &*indirect.selector;
	keys.clear();
	const auto read_selector = [&](uint64_t base, uint64_t size, uint64_t offset,
	                               std::span<uint32_t> words) {
		return selector->global ? ReadTable(base, size, offset, runtime, words)
		                        : ReadScalarTable(base, size, offset, runtime, words);
	};
	const auto read_keys = [&](uint64_t base, uint64_t size, uint64_t first,
	                           uint64_t step, uint64_t count) {
		keys.resize(count);
		if (count == 0u) return true;
		if (step == 4u) {
			return read_selector(base, size, first, keys);
		}
		if (step < 4u) return false;
		for (uint64_t copied = 0; copied < count;) {
			const auto offset = first + copied * step;
			const auto bias = selector->global ? 0u : offset & 3u;
			const auto length = std::min(count - copied,
			    1u + (uint64_t {MaxIndirectDescriptorProbes - 1u} * 4u - bias) / step);
			keys.resize(count + (bias + (length - 1u) * step) / 4u + 1u);
			const auto batch = std::span(keys).subspan(count);
			if (!read_selector(base, size, offset, batch)) {
				keys.resize(count);
				return false;
			}
			for (uint64_t index = 0; index < length; ++index)
				keys[copied + index] = batch[(bias + index * step) / 4u];
			copied += length;
		}
		keys.resize(count);
		return true;
	};
	const auto read_ranks = [&](uint64_t base, uint64_t size, uint64_t first, uint64_t step,
	                            uint32_t ranks) {
		keys.reserve(std::popcount(ranks));
		for (; ranks != 0u; ranks &= ranks - 1u) {
			const auto offset = first + std::countr_zero(ranks) * step;
			uint32_t key = 0;
			if (offset > UINT32_MAX || !read_selector(base, size, offset, {&key, 1})) return false;
			keys.push_back(key);
		}
		return true;
	};
	if (selector->rows) {
		const auto& rows = *selector->rows;
		DescriptorValue count_value;
		ShaderBufferResource counts, material;
		if (!WalkedDescriptor(program, rows.source, count_value) ||
		    !DecodeBufferDescriptor(count_value, counts) ||
		    !DecodeBufferDescriptor(material_value, material) || selector->stride != 4u ||
		    selector->offset != 0u || rows.stride == 0u || (rows.stride & 3u) != 0u ||
		    counts.GetSize() / 4u > MaxIndirectDescriptorProbes) return false;
		std::vector<uint32_t> lengths(counts.GetSize() / 4u);
		if (!ReadScalarTable(counts.Base48(), counts.GetSize(), 0u, runtime, lengths)) return false;
		uint32_t total = 0;
		for (size_t row = 0; row < lengths.size(); ++row) {
			const auto length = lengths[row];
			if (length > MaxIndirectDescriptorProbes - total ||
			    (length != 0u && uint64_t {row} * rows.stride + uint64_t {length} * 4u >
			                         UINT32_MAX + 1ull)) return false;
			total += length;
		}
		keys.reserve(total);
		std::vector<uint32_t> batch;
		for (size_t row = 0; row < lengths.size();) {
			if (lengths[row] == 0u) { ++row; continue; }
			const auto first = uint64_t {row} * rows.stride / 4u;
			auto end = first + lengths[row];
			size_t next = row + 1u;
			for (; next < lengths.size(); ++next) {
				if (lengths[next] == 0u) continue;
				const auto candidate_end =
				    std::max(end, uint64_t {next} * rows.stride / 4u + lengths[next]);
				if (candidate_end - first > MaxIndirectDescriptorProbes) break;
				end = candidate_end;
			}
			batch.resize(end - first);
			if (!ReadScalarTable(material.Base48(), material.GetSize(), first * 4u, runtime, batch))
				return false;
			for (; row < next; ++row) {
				if (lengths[row] == 0u) continue;
				const auto begin = batch.begin() + (uint64_t {row} * rows.stride / 4u - first);
				keys.insert(keys.end(), begin, begin + lengths[row]);
			}
		}
	} else if (selector->bitset) {
		ShaderBufferResource material;
		std::array<uint32_t, 8> storage {};
		const auto record_words = std::min<size_t>(selector->stride / 4u, storage.size());
		const auto masks = std::span(storage).first(record_words);
		uint32_t first = 0, count = 0;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !WalkedValue(program, indirect.selector_first, first) ||
		    !EvaluateIndirectCount(program, indirect, runtime, count) || count != masks.size() * 32u ||
		    selector->stride != masks.size_bytes() || selector->offset != 0u ||
		    material.Type() != 0u || material.Stride() != selector->stride ||
		    material.SwizzleEnabled() || material.AddTid() || material.OutOfBounds() != 0u ||
		    (material.Base48() & 3u) != 0u) return false;
		if (material.Format() != Prospero::BufferFormat::kInvalid) {
			const auto words = material.GetSize() / sizeof(uint32_t);
			for (uint64_t first_word = 0; first_word < words; first_word += keys.size()) {
				keys.resize(std::min<uint64_t>(words - first_word, MaxIndirectDescriptorProbes));
				if (!ReadScalarTable(material.Base48(), material.GetSize(), first_word * 4u,
				                     runtime, keys)) return false;
				for (size_t word = 0; word < keys.size(); ++word) masks[word % masks.size()] |= keys[word];
				if (std::ranges::all_of(masks, [](uint32_t mask) { return mask == UINT32_MAX; })) break;
			}
			keys.clear();
			for (uint32_t word = 0; word < masks.size(); ++word) {
				auto mask = masks[word];
				while (mask != 0u) {
					const auto rank = word * 32u + std::countr_zero(mask);
					keys.push_back(first + count - 1u - rank);
					mask &= mask - 1u;
				}
			}
		}
	} else if (!selector->byte_ranges.empty()) {
		if (material_value.dword_count != 2u) return false;
		const auto base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		std::array<bool, 256> selected {};
		for (const auto [offset, size]: selector->byte_ranges) {
			if (size > MaxIndirectDescriptorProbes || ((offset | size) & 3u) != 0u) return false;
			keys.resize(size / 4u);
			if (!ReadScalarTable(base, UINT64_MAX, offset, runtime, keys)) return false;
			for (const auto word: keys)
				for (uint32_t shift = 0; shift < 32u; shift += 8u)
					selected[(word >> shift) & 255u] = true;
		}
		keys.clear();
		for (uint32_t key = 0; key < selected.size(); ++key)
			if (selected[key]) keys.push_back(key);
	} else if (!indirect.selector_first.IsEmpty()) {
		ShaderBufferResource material;
		uint64_t             base = 0, size = UINT64_MAX;
		uint32_t             first = 0, count = 0;
		if (material_value.dword_count == 2u) {
			if (selector->stride == 0u) return false;
			base = (uint64_t {material_value.dwords[1]} << 32u) | material_value.dwords[0];
		} else {
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			base = material.Base48();
			size = material.GetSize();
		}
		if (!WalkedValue(program, indirect.selector_first, first) ||
		    !EvaluateIndirectCount(program, indirect, runtime, count) || count > MaxIndirectDescriptorProbes)
			return false;
		const auto step = selector->stride != 0u ? selector->stride : material.Stride();
		if (selector->stride == 0u &&
		    (material.Type() != 0u || material.SwizzleEnabled() || material.AddTid() ||
		     material.OutOfBounds() != 0u || uint64_t {selector->offset} + 4u > step ||
		     uint64_t {first} + count > material.NumRecords())) return false;
		const auto last = count != 0u ? uint64_t {first} + count - 1u : 0u;
		if (last > UINT32_MAX) return false;
		const auto end = count != 0u ? last * step + selector->offset + 4u : 0u;
		if (end > uint64_t {UINT32_MAX} + 1u) return false;
		const auto start = uint64_t {first} * step + selector->offset;
		if (selector->ranks) {
			const auto& mask = *selector->ranks;
			DescriptorValue mask_value;
			ShaderBufferResource masks;
			if (count > 32u || !WalkedDescriptor(program, mask.source, mask_value) ||
			    !DecodeBufferDescriptor(mask_value, masks)) return false;
			const auto words = mask.stride / 4u;
			const auto chunk =
			    words <= MaxIndirectDescriptorProbes ? MaxIndirectDescriptorProbes : 1u;
			uint32_t ranks = 0;
			for (auto at = uint64_t {mask.offset}; at + 4u <= masks.GetSize();
			     at += uint64_t {std::max(chunk, words)} * 4u) {
				keys.resize(std::min<uint64_t>((masks.GetSize() - at) / 4u, chunk));
				if (!ReadScalarTable(masks.Base48(), masks.GetSize(), at, runtime, keys))
					return false;
				for (size_t word = 0; word < keys.size(); word += words) ranks |= keys[word];
			}
			keys.clear();
			if (count < 32u) ranks &= (1u << count) - 1u;
			if (!read_ranks(base, size, start, step, ranks)) return false;
		} else if (!read_keys(base, size, start, step, count)) {
			return false;
		}
	} else if (!indirect.selector_mask.IsEmpty()) {
		uint32_t mask = 0;
		uint32_t count = 0;
		if (material_value.dword_count != 2u ||
		    !WalkedValue(program, indirect.selector_mask, mask) ||
		    !EvaluateIndirectCount(program, indirect, runtime, count) || count == 0u || count > 32u) {
			return false;
		}
		if (count < 32u) mask &= (1u << count) - 1u;
		const auto material_base =
		    (static_cast<uint64_t>(material_value.dwords[1]) << 32u) | material_value.dwords[0];
		if (!read_ranks(material_base, UINT64_MAX, selector->offset, selector->stride, mask))
			return false;
	} else if (selector->stride == 0u) {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) ||
		    !CanReadIndexedSelector(material, selector->offset)) return false;
		const auto records = material.NumRecords();
		if (material.Format() != Prospero::BufferFormat::kInvalid && records > selector->first_record &&
		    selector->offset < material.Stride() &&
		    !read_keys(material.Base48(), material.GetSize(),
		               uint64_t {selector->first_record} * material.Stride() + selector->offset,
		               uint64_t {selector->record_step} * material.Stride(),
		               (records - selector->first_record - 1u) / selector->record_step + 1u)) return false;
		keys.push_back(0u);
	} else {
		ShaderBufferResource material;
		if (!DecodeBufferDescriptor(material_value, material) || selector->stride < 4u ||
		    !std::has_single_bit(selector->stride) ||
		    ((selector->offset & 3u) != 0u && indirect.selector_byte_bias.IsEmpty()))
			return false;
		const auto domain_count = (uint64_t {1} << 32u) / selector->stride;
		uint32_t bias = 0;
		if (!indirect.selector_byte_bias.IsEmpty() &&
		    !WalkedValue(program, indirect.selector_byte_bias, bias)) return false;
		const auto first = (uint64_t {selector->offset} + (bias % selector->stride)) & ~uint64_t {3};
		const auto available = material.GetSize() > first ? material.GetSize() - first : 0u;
		const auto count = std::min(domain_count,
		    available < 4u ? 0u : (available - 4u) / selector->stride + 1u);
		if (count > MaxIndirectDescriptorProbes ||
		    !read_keys(material.Base48(), material.GetSize(), first, selector->stride, count)) return false;
		if (count < domain_count) keys.push_back(0u);
	}
	if (selector->byte_ranges.empty()) {
		for (auto& key: keys) {
			auto field = (key >> selector->bit_offset) & selector->bit_mask;
			if (selector->non_negative && static_cast<int32_t>(field) < 0) field = 0u;
			key = std::min(field, selector->maximum);
		}
		if (selector->excluded) std::erase(keys, *selector->excluded);
		std::ranges::sort(keys);
		keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
	}
	return true;
}

template <typename Specialization, typename Normalize>
bool MaterializeIndirectDescriptor(const ResourcePlan&                         program,
                                   uint32_t source_index,
                                   uint32_t resource_index, uint32_t dword_count, uint32_t read_dwords,
                                   const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                                   std::vector<DescriptorValue>& descriptors,
                                   std::vector<Specialization>&  specializations,
                                   Normalize&& normalize) {
	const auto& indirect = *program.descriptor_sources[source_index].indirect_descriptor;
	const auto  descriptor_bytes = read_dwords * sizeof(uint32_t);
	const auto& sources = indirect.sources;
	const auto* selector = indirect.selector ? &*indirect.selector : nullptr;
	const bool byte_keys = IsBoundedDescriptorTable(program, indirect);
	const auto scratch = PlanScratch(program);
	auto* selector_keys = selector != nullptr
	    ? &scratch.selector_keys[program.selector_groups[source_index]] : nullptr;
	auto& keys = selector_keys != nullptr ? selector_keys->values : scratch.material_keys;
	auto& byte_offsets = scratch.material_keys;
	const auto children_begin = descriptors.size();
	const auto root_resource = specializations[resource_index];
	std::unordered_map<DescriptorValue, uint32_t, DescriptorValueHash> ordinals;
	const auto intern_candidate = [&](const DescriptorValue& candidate) {
		if (candidate == descriptors[resource_index]) return 0u;
		const auto [found, inserted] = ordinals.try_emplace(
		    candidate, static_cast<uint32_t>(descriptors.size() - children_begin + 1u));
		if (inserted) {
			descriptors.push_back(candidate);
			auto child = root_resource;
			child.indirect_root = resource_index;
			specializations.push_back(child);
		}
		return found->second;
	};
	uint64_t table_base = 0;
	uint64_t table_size = UINT64_MAX;
	DescriptorValue table_value;
	ShaderBufferResource table;
	const auto failed = [&](std::string_view reason, std::string_view details = {}) {
		const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
		return SpecializationFail(fmt::format(
		    "indirect {} resource={} source={} table_source={} words={:08x},{:08x},{:08x},{:08x} "
		    "base={:#x} size={} type={} runtime_stride={} records={} swizzle={} addtid={} oob={} "
		    "maximum={} offset={} immediate={} record_bytes={} child_count={} child_stride={} "
		    "child_offset={} indexed_pointer={} candidates={} {}",
		    reason, resource_index, source_index, indirect.table_source,
		    table_value.dwords[0], table_value.dwords[1], table_value.dwords[2], table_value.dwords[3],
		    table_base, table_size, table.Type(), table.Stride(), table.NumRecords(),
		    table.SwizzleEnabled(), table.AddTid(), table.OutOfBounds(),
		    indirect.table_maximum ? fmt::format("{}", *indirect.table_maximum) : "none",
		    indirect.table_offset, indirect.table_immediate, indirect.table_record_bytes,
		    pointer != nullptr ? pointer->count : 1u, pointer != nullptr ? pointer->stride : 0u,
		    pointer != nullptr ? pointer->offset : 0u, pointer != nullptr && pointer->indexed,
		    descriptors.size(), details));
	};
	const auto read_failed = [&](std::string_view reason, uint32_t row, uint32_t child,
	                             uint64_t base, uint64_t offset, size_t bytes) {
		return failed(reason, fmt::format("row={} child={} read_base={:#x} read_offset={:#x} "
		                                 "read_address={:#x} requested_bytes={}", row, child, base, offset,
		    (base & AddressMask & ~uint64_t {3}) + (offset & ~uint64_t {3}), bytes));
	};
	const auto scalar_buffer_offset = [&](uint64_t offset) {
		return (offset + indirect.table_immediate) & ~uint64_t {3};
	};
	const auto normalization_failed = [&](uint64_t offset, uint32_t key, uint64_t step,
	                                      std::string_view key_kind) {
		return failed("candidate", fmt::format("candidate_offset={:#x} {}={} stride={} step={}",
		    offset & ~uint64_t {3}, key_kind, key, indirect.table_stride, step));
	};
	if (indirect.pointer_table && indirect.pointer_table->buffer) {
		const auto& pointer = *indirect.pointer_table;
		if (!WalkedDescriptor(program, indirect.table_source, table_value) ||
		    !DecodeBufferDescriptor(table_value, table)) return false;
		table_base = table.Base48();
		table_size = table.GetSize();
		descriptors[resource_index] = {.dword_count = dword_count};
		const uint64_t first = scalar_buffer_offset(indirect.table_offset);
		const uint64_t rows = table_size >= first + 8u
		    ? (table_size - first - 8u) / indirect.table_stride + 1u : 0u;
		std::vector<uint32_t> words(table_size / 4u);
		if (rows != 0u && !ReadScalarTable(table_base, table_size, 0u, runtime, words))
			return read_failed("pointer table read", 0u, 0u, table_base, 0u, table_size);
		const auto row_word = [&](uint64_t row, uint64_t byte) {
			const auto word = (row * indirect.table_stride + byte) / 4u;
			return (uint64_t {words[word + 1u]} << 32u) | words[word];
		};
		std::optional<uint64_t> function_slot;
		if (pointer.function != 0u) {
			const auto pointer_slot = first - (indirect.table_offset & ~3u);
			for (uint64_t slot = 0; slot + 8u <= indirect.table_stride && !function_slot; slot += 8u) {
				if (slot < pointer_slot + 8u && pointer_slot < slot + 8u) continue;
				for (uint64_t row = 0; row < rows; ++row) {
					if (((indirect.table_offset & ~3u) + row * indirect.table_stride + slot + 8u) > table_size) break;
					if ((row_word(row, (indirect.table_offset & ~3u) + slot) & AddressMask) ==
					    (pointer.function & AddressMask)) {
						function_slot = slot;
						break;
					}
				}
			}
			if (!function_slot) return failed("record function", fmt::format("function={:#x}", pointer.function));
		}
		std::unordered_set<uint64_t> functions {pointer.function & AddressMask};
		for (const auto alias: pointer.functions) functions.insert(alias & AddressMask);
		std::vector<uint64_t> records;
		for (uint64_t row = 0; row < rows; ++row) {
			if (function_slot &&
			    ((indirect.table_offset & ~3u) + row * indirect.table_stride + *function_slot + 8u > table_size ||
			     !functions.contains(row_word(row, (indirect.table_offset & ~3u) + *function_slot) & AddressMask))) continue;
			const auto word = (first + row * indirect.table_stride) / 4u;
			const auto record = ((uint64_t {words[word + 1u]} << 32u) | words[word]) & AddressMask;
			if (record != 0u) records.push_back(record);
		}
		std::ranges::sort(records);
		records.erase(std::unique(records.begin(), records.end()), records.end());
		std::vector<std::pair<uint32_t, uint32_t>> entries;
		struct KeyedTable {
			uint64_t base  = 0;
			uint64_t count = 0;
			size_t   entry = 0;
		};
		std::unordered_map<uint32_t, KeyedTable> keyed_tables;
		std::unordered_set<DescriptorValue, DescriptorValueHash> tables;
		std::vector<uint32_t> blocks;
		bool heap_list = false;
		if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
			const auto& image = program.info.images[resource_index];
			heap_list = runtime.image_heap != nullptr && dword_count == 8u &&
			            image.resource_class == ImageResourceClass::Sampled &&
			            image.mip_mode != ImageMipMode::Dynamic;
		}
		std::vector<uint32_t>        element_words;
		std::vector<DescriptorValue> heap_values;
		std::vector<size_t>          heap_blocks;
		for (const auto record: records) {
			DescriptorValue record_table {.dword_count = 4u};
			if (!ReadScalarTable(record, UINT64_MAX, pointer.offset, runtime,
			                     std::span(record_table.dwords).first(4u)))
				return read_failed("record table read", 0u, 0u, record, pointer.offset, 16u);
			const std::array<uint32_t, 4> table_words {record_table.dwords[0], record_table.dwords[1],
			                                           record_table.dwords[2], record_table.dwords[3]};
			if (!tables.insert(record_table).second) continue;
			ShaderBufferResource elements;
			if (!DecodeBufferDescriptor(record_table, elements)) return false;
			const uint64_t element_base = elements.Base48();
			const uint64_t element_size = elements.GetSize();
			const uint64_t count = pointer.element_stride == 0u ? 1u
			    : element_size >= pointer.element_offset + descriptor_bytes
			        ? (element_size - pointer.element_offset - descriptor_bytes) / pointer.element_stride + 1u
			        : 0u;
			if (count != 0u && !TableInGuestMemory(element_base, element_size, runtime)) continue;
			if (count > MaxRecordTableElements)
				return failed("record table elements", fmt::format("record={:#x} count={}", record, count));
			const auto block = static_cast<uint32_t>(blocks.size());
			blocks.push_back(static_cast<uint32_t>(count));
			if (element_base != 0u && count > 1u) {
				element_words.resize((element_size + 3u) / 4u);
				if (!ReadScalarTable(element_base, element_size, 0u, runtime, element_words))
					return read_failed("record table read", 0u, 0u, element_base, 0u, element_size);
			}
			for (uint32_t element = 0; element < count; ++element) {
				const auto offset = pointer.element_offset + uint64_t {element} * pointer.element_stride;
				DescriptorValue candidate {.dword_count = dword_count};
				if (element_base != 0u && offset + descriptor_bytes <= element_size && count > 1u) {
					std::copy_n(element_words.begin() + static_cast<std::ptrdiff_t>(offset / 4u),
					            read_dwords, candidate.dwords.begin());
				} else if (element_base != 0u && offset + descriptor_bytes <= element_size &&
				    !ReadScalarTable(element_base, element_size, offset, runtime,
				                     std::span(candidate.dwords).first(read_dwords)))
					return read_failed("record descriptor read", 0u, element, element_base, offset, descriptor_bytes);
				if (!normalize(candidate)) return normalization_failed(offset, element, pointer.element_stride, "element");
				if (dword_count == 8u)
					ReportUnknownTileMode("record table candidate", program, resource_index, element_base + offset, candidate);
				if (heap_list) {
					heap_blocks.push_back(blocks.size());
					heap_values.push_back(candidate);
					blocks.push_back(0u);
				} else {
					blocks.push_back(intern_candidate(candidate));
				}
			}
			const auto [same_key, inserted] =
			    keyed_tables.try_emplace(table_words[0], KeyedTable {element_base, count, entries.size()});
			if (inserted) {
				entries.emplace_back(table_words[0], block);
			} else if (same_key->second.base != element_base) {
				return failed("record tables", "two record tables share the first word of their V#");
			} else if (count > same_key->second.count) {
				same_key->second.count                 = count;
				entries[same_key->second.entry].second = block;
			}
		}
		ImageHeapView heap_view;
		if (heap_list) {
			heap_list = runtime.image_heap->AcquireList(
			    {.owner = &program, .resource = resource_index,
			     .root = &program.info.images[resource_index], .values = heap_values},
			    heap_view);
			const auto heap_kinds = heap_view.kinds;
			for (size_t i = 0; i < heap_blocks.size(); ++i) {
				blocks[heap_blocks[i]] = heap_list ? static_cast<uint32_t>(i + 1u)
				                                   : intern_candidate(heap_values[i]);
			}
			if (heap_list) {
				for (const auto& kind: heap_kinds) {
					descriptors.push_back(kind);
					auto child          = root_resource;
					child.indirect_root = resource_index;
					specializations.push_back(child);
				}
			}
		}
		if (descriptors.size() == children_begin) return true;
		std::ranges::sort(entries);
		for (size_t i = 1; i < entries.size(); ++i) {
			if (entries[i].first == entries[i - 1u].first)
				return failed("record tables", "two record tables share the first word of their V#");
		}
		if (entries.empty()) return true;
		const bool stable_layout = g_stable_indirect_layouts.load(std::memory_order_relaxed);
		const auto key_capacity  = !stable_layout ? entries.size() : std::max<size_t>(
		    program.stage == ShaderType::Compute ? g_compute_indirect_keys.load(std::memory_order_relaxed)
		                                         : MinIndirectMappingCapacity,
		    std::bit_ceil(entries.size()));
		const auto list_offset    = snapshot.flattened_srt.size();
		const auto mapping_offset = list_offset + (heap_list ? 3u : 0u);
		snapshot.flattened_srt.resize(mapping_offset + 1u + key_capacity * 2u, 0u);
		if (heap_list) {
			snapshot.flattened_srt[list_offset]      = heap_view.slot_count;
			snapshot.flattened_srt[list_offset + 1u] = static_cast<uint32_t>(heap_view.slots);
			snapshot.flattened_srt[list_offset + 2u] = static_cast<uint32_t>(heap_view.slots >> 32u);
		}
		snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(entries.size());
		for (size_t entry = 0; entry < entries.size(); ++entry) {
			snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = entries[entry].first;
			snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = entries[entry].second;
		}
		t_record_blocks.push_back({mapping_offset, entries.size(), std::move(blocks)});
#if defined(ANYPS5_KYTY)
		if (((std::is_same_v<Specialization, ResourceSpecialization::Image> && !heap_list) ||
		     std::is_same_v<Specialization, ResourceSpecialization::Sampler> ||
		     std::is_same_v<Specialization, ResourceSpecialization::Buffer>) && stable_layout) {
#else
		if (((std::is_same_v<Specialization, ResourceSpecialization::Image> && !heap_list) ||
		     std::is_same_v<Specialization, ResourceSpecialization::Sampler>) && stable_layout) {
#endif
			const auto children = descriptors.size() - children_begin;
			const auto total    = std::max<size_t>(
                program.stage == ShaderType::Compute
                    ? g_compute_indirect_images.load(std::memory_order_relaxed)
                    : 1u,
                std::bit_ceil(children + 1u));
			for (auto padded = children; padded + 1u < total; ++padded) {
				descriptors.push_back(descriptors.back());
				specializations.push_back(specializations.back());
			}
		}
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset    = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_capacity);
		if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
			root.heap = heap_list;
			if (heap_list) {
				for (auto child = children_begin; child < specializations.size(); ++child) {
					specializations[child].heap = true;
				}
			}
		}
		return true;
	}
	if (sources.empty()) {
		if (selector == nullptr) keys.clear();
		DescriptorValue material_value;
		if ((selector != nullptr && (byte_keys || !selector_keys->resolved) &&
		     !WalkedDescriptor(program, selector->source, material_value)) ||
		    !WalkedDescriptor(program, indirect.table_source, table_value)) {
			return false;
		}
		if (table_value.dword_count == 2u) {
			table_base = (static_cast<uint64_t>(table_value.dwords[1]) << 32u) | table_value.dwords[0];
			if (byte_keys) {
				table_base += uint64_t {indirect.table_offset} + (indirect.table_immediate & ~3u);
				table_size = (uint64_t {*indirect.table_maximum} + 1u) * indirect.table_stride;
			}
		} else if (DecodeBufferDescriptor(table_value, table)) {
			table_base = table.Base48();
			table_size = table.GetSize();
		} else {
			return false;
		}
		if (byte_keys && table_size > UINT32_MAX) return failed("table geometry");
		bool use_selector = selector != nullptr;
		if (byte_keys && selector != nullptr) {
			ShaderBufferResource material;
			if (!DecodeBufferDescriptor(material_value, material)) return false;
			use_selector = CanReadIndexedSelector(material, selector->offset);
			if (use_selector && material.Format() != Prospero::BufferFormat::kInvalid &&
			    material.NumRecords() != 0u && selector->offset < material.Stride()) {
				const auto begin = material.Base48() + selector->offset;
				const auto end = begin + uint64_t {material.NumRecords() - 1u} * material.Stride() + 4u;
				for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
					const auto& buffer = program.info.buffers[i];
					if (!buffer.written) continue;
					ShaderBufferResource written;
					if (program.descriptor_sources[buffer.source].indirect_descriptor ||
					    !DecodeBufferDescriptor(snapshot.buffers[i], written) ||
					    (written.Base48() != 0u && written.GetSize() != 0u &&
					     begin < written.Base48() + written.GetSize() &&
					     written.Base48() < end)) {
						use_selector = false;
						break;
					}
				}
			}
		}
		if (byte_keys && !use_selector) {
			if (!indirect.table_scalar && table.Format() == Prospero::BufferFormat::kInvalid) {
				descriptors[resource_index] = {.dword_count = dword_count};
				return true;
			}
			if (indirect.table_record_bytes != 0u &&
			     (table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			      table.OutOfBounds() != 0u || (table_base & 3u) != 0u ||
			      (table.Stride() & 3u) != 0u || table.Stride() < indirect.table_record_bytes))
				return failed("table geometry");
			const auto* pointer = indirect.pointer_table ? &*indirect.pointer_table : nullptr;
			const bool indexed_pointer = pointer != nullptr && pointer->indexed;
			const auto child_count = pointer != nullptr ? pointer->count : 1u;
			uint64_t domain = indirect.table_maximum ? uint64_t {*indirect.table_maximum} + 1u
			                                       : uint64_t {1} << 32u;
			if (!indirect.key_count.IsEmpty()) {
				uint32_t key_count = 0u;
				if (!EvaluateIndirectCount(program, indirect, runtime, key_count)) return false;
				domain = std::min(domain, uint64_t {key_count});
			}
			if (indexed_pointer &&
			    (!indirect.table_maximum || table.Type() != 0u || table.SwizzleEnabled() || table.AddTid() ||
			     table.OutOfBounds() != 0u || (table_base & 3u) != 0u || (table.Stride() & 3u) != 0u ||
			     uint64_t {indirect.table_immediate} + 8u > table.Stride() ||
			     domain > table.NumRecords()))
				return failed("indexed pointer geometry", fmt::format("rows={}", domain));
			const bool affine = indirect.table_stride != 0u || indexed_pointer;
			const bool image_heap = dword_count == 8u && pointer == nullptr;
			const uint64_t step = indexed_pointer ? table.Stride()
			    : indirect.table_maximum ? indirect.table_stride
			    : affine ? std::max(uint64_t {1} << std::countr_zero(indirect.table_stride), uint64_t {4})
			             : 4u;
			uint64_t first = 0u;
			if (table_value.dword_count != 2u)
				first = scalar_buffer_offset(indirect.table_maximum ? indirect.table_offset
				                                                : indirect.table_offset % step);
			if (!indirect.table_maximum) domain /= step;
			const auto count = table_size >= first + 4u
			    ? std::min((table_size - first - 4u) / step + 1u, domain)
			    : 0u;
			descriptors[resource_index] = {.dword_count = dword_count};
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				const auto& image = program.info.images[resource_index];
				ImageHeapView view;
				if (image_heap && runtime.image_heap != nullptr &&
				    image.resource_class == ImageResourceClass::Sampled &&
				    image.mip_mode != ImageMipMode::Dynamic &&
				    runtime.image_heap->Acquire({table_base, table_size, step, first, count, &image},
				                                view)) {
					snapshot.specialization_reads.emplace_back(table_base, table_size);
					if (view.kinds.empty()) return true;
					auto root = root_resource;
					root.indirect_root = resource_index;
					root.indirect_mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
					root.heap = true;
					specializations[resource_index] = root;
					snapshot.flattened_srt.insert(snapshot.flattened_srt.end(),
					    {view.slot_count, static_cast<uint32_t>(view.slots), static_cast<uint32_t>(view.slots >> 32u)});
					for (const auto& kind: view.kinds) {
						descriptors.push_back(kind);
						specializations.push_back(root);
					}
					return true;
				}
			}
#if defined(ANYPS5_KYTY)
			if constexpr (std::is_same_v<Specialization, ResourceSpecialization::Image>) {
				const auto& image = program.info.images[resource_index];
				if (pointer != nullptr && dword_count == 8u && runtime.image_heap != nullptr && child_count != 0u &&
				    image.resource_class == ImageResourceClass::Sampled && image.mip_mode != ImageMipMode::Dynamic &&
				    count * child_count <= MaxIndirectDescriptorProbes) {
					std::vector<DescriptorValue> values(count * child_count, DescriptorValue {.dword_count = dword_count});
					bool read = true;
					for (uint32_t word = 0; read && word < count; ++word) {
						const auto table_offset = affine ? first + word * step : uint64_t {word} * 4u;
						std::array<uint32_t, 2> address {};
						if (!ReadScalarTable(table_base, table_size, table_offset, runtime, address)) {
							read = false;
							break;
						}
						const auto pointer_base = (uint64_t {address[1]} << 32u) | address[0];
						for (uint32_t child = 0; child < child_count; ++child) {
							auto& candidate = values[word * child_count + child];
							if (!ReadScalarTable(pointer_base, UINT64_MAX, uint64_t {child} * pointer->stride + pointer->offset,
							                     runtime, std::span(candidate.dwords).first(read_dwords)) ||
							    !normalize(candidate)) {
								read = false;
								break;
							}
							ReportUnknownTileMode("pointer table candidate", program, resource_index,
							                      pointer_base + uint64_t {child} * pointer->stride + pointer->offset,
							                      candidate);
						}
					}
					ImageHeapView view;
					if (read && !values.empty() &&
					    runtime.image_heap->AcquireList({.owner = &program, .resource = resource_index, .root = &image, .values = values},
					                                    view)) {
						if (view.kinds.empty()) return true;
						auto root = root_resource;
						root.indirect_root           = resource_index;
						root.indirect_mapping_offset = static_cast<uint32_t>(snapshot.flattened_srt.size());
						root.heap                    = true;
						specializations[resource_index] = root;
						const auto slots = view.slots + sizeof(uint32_t);
						snapshot.flattened_srt.insert(snapshot.flattened_srt.end(),
						    {view.slot_count - 1u, static_cast<uint32_t>(slots), static_cast<uint32_t>(slots >> 32u)});
						for (const auto& kind: view.kinds) {
							descriptors.push_back(kind);
							specializations.push_back(root);
						}
						return true;
					}
				}
			}
#endif
			if (child_count == 0u || (!image_heap && count * child_count > MaxIndirectDescriptorProbes))
				return failed("candidate count", fmt::format("rows={} limit={}", count, MaxIndirectDescriptorProbes));
			auto& heap_words = PlanScratch(program).material_keys;
			if (image_heap) {
				heap_words.resize(table_size / 4u);
				if (!ReadScalarTable(table_base, table_size, 0u, runtime, heap_words))
					return read_failed("image heap read", 0u, 0u, table_base, 0u, heap_words.size() * sizeof(uint32_t));
			}
			const auto mapping_offset = snapshot.flattened_srt.size();
			snapshot.flattened_srt.resize(mapping_offset + 2u, 0u);
			for (uint32_t word = 0; word < count; ++word) {
				const auto table_offset = affine ? first + word * step : uint64_t {word} * 4u;
				uint64_t pointer_base = 0u;
				if (pointer != nullptr) {
					std::array<uint32_t, 2> address {};
					if (!ReadScalarTable(table_base, table_size, table_offset, runtime, address))
						return read_failed("parent pointer read", word, 0u, table_base, table_offset, sizeof(address));
					pointer_base = (uint64_t {address[1]} << 32u) | address[0];
				}
				for (uint32_t child = 0; child < child_count; ++child) {
					DescriptorValue candidate {.dword_count = dword_count};
					if (pointer != nullptr) {
						if (!ReadScalarTable(pointer_base, UINT64_MAX,
						        uint64_t {child} * pointer->stride + pointer->offset, runtime,
						        std::span(candidate.dwords).first(read_dwords)))
							return read_failed("child descriptor read", word, child, pointer_base,
							    uint64_t {child} * pointer->stride + pointer->offset, descriptor_bytes);
					} else if (!image_heap) {
						if (!ReadScalarTable(table_base, table_size, table_offset, runtime,
						                     std::span(candidate.dwords).first(read_dwords))) return false;
					} else {
						const auto table_word = table_offset / 4u;
						std::copy_n(heap_words.begin() + table_word,
						            std::min<size_t>(read_dwords, heap_words.size() - table_word),
						            candidate.dwords.begin());
					}
					if (!normalize(candidate))
						return normalization_failed(table_offset, word, step,
						                            indirect.table_maximum ? "key" : "probe");
					const auto ordinal = intern_candidate(candidate);
					if (ordinal == 0u) continue;
					const auto mapping_word = image_heap ? table_offset / 4u : word * child_count + child;
					snapshot.flattened_srt.resize(mapping_offset + 2u + mapping_word, 0u);
					snapshot.flattened_srt[mapping_offset + 1u + mapping_word] = ordinal;
				}
			}

			if (descriptors.size() == children_begin) {
				snapshot.flattened_srt.resize(mapping_offset);
			} else {
				snapshot.flattened_srt[mapping_offset] =
				    static_cast<uint32_t>(snapshot.flattened_srt.size() - mapping_offset - 1u);
#if defined(ANYPS5_KYTY)
				if (g_stable_indirect_layouts.load(std::memory_order_relaxed)) {
					const auto used = snapshot.flattened_srt.size() - mapping_offset - 1u;
					snapshot.flattened_srt.resize(mapping_offset + 1u + std::bit_ceil(std::max<size_t>(used, 1u)), 0u);
				}
				if (g_stable_indirect_layouts.load(std::memory_order_relaxed)) {
					const auto children = descriptors.size() - children_begin;
					const auto total    = std::max<size_t>(
					    program.stage == ShaderType::Compute ? g_compute_indirect_images.load(std::memory_order_relaxed) : 1u,
					    std::bit_ceil(children + 1u));
					for (auto padded = children; padded + 1u < total; ++padded) {
						descriptors.push_back(descriptors.back());
						specializations.push_back(specializations.back());
					}
				}
#endif
				auto& root = specializations[resource_index];
				root.indirect_root = resource_index;
				root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
				root.indirect_search_iterations = 0u;
			}
			return true;
		}
		if (selector == nullptr) {
			uint32_t key_count = 0;
			uint32_t first_key = 0;
			if (!EvaluateIndirectCount(program, indirect, runtime, key_count)) return false;
			if (!indirect.selector_first.IsEmpty() && !WalkedValue(program, indirect.selector_first, first_key))
				return false;
			if (indirect.table_stride == 0u || key_count > MaxIndirectDescriptorProbes ||
			    (indirect.selector_first.IsEmpty() && key_count != 0u &&
			         uint64_t {indirect.table_offset} + indirect.table_immediate +
			         uint64_t {key_count - 1u} * indirect.table_stride + descriptor_bytes >
			             UINT32_MAX + 1ull)) {
				return false;
			}
			keys.resize(key_count);
			std::iota(keys.begin(), keys.end(), first_key);
			if (!keys.empty() && keys.back() < keys.front())
				std::rotate(keys.begin(), keys.begin() + (UINT32_MAX - first_key + 1ull), keys.end());
			if (indirect.workgroup_shift != 0u) {
				for (auto& key: keys)
					key = static_cast<uint32_t>(uint64_t {key} * indirect.workgroup_multiplier >>
					                            indirect.workgroup_shift);
				keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
			}
		} else {
			if (!indirect.selector_mask.IsEmpty() && table_value.dword_count != 2u) return false;
			if (!selector_keys->resolved) {
				if (!ResolveSelectorKeys(program, indirect, material_value, runtime, keys)) return false;
				selector_keys->resolved = true;
			}
		}
		if (byte_keys) {
			byte_offsets.clear();
			byte_offsets.reserve(keys.size());
			for (const auto key: keys) {
				const auto indexed_offset = uint64_t {key * indirect.table_stride + indirect.table_offset};
				const auto offset = table_value.dword_count == 4u
				    ? scalar_buffer_offset(indexed_offset)
				    : (indexed_offset & ~uint64_t {3}) + (indirect.table_immediate & ~3u);
				byte_offsets.push_back(offset <= UINT32_MAX ? static_cast<uint32_t>(offset) : UINT32_MAX);
			}
			std::ranges::sort(byte_offsets);
			byte_offsets.erase(std::unique(byte_offsets.begin(), byte_offsets.end()), byte_offsets.end());
		}
	}

	const auto mapping_offset = snapshot.flattened_srt.size();
	const auto selected_keys = byte_keys ? std::span<const uint32_t>(byte_offsets)
	                                    : std::span<const uint32_t>(keys);
	const auto key_count = sources.empty() ? selected_keys.size() : sources.size();
	const bool stable_layout = g_stable_indirect_layouts.load(std::memory_order_relaxed);
	const auto key_capacity  = !stable_layout ? key_count : std::max<size_t>(
	    program.stage == ShaderType::Compute ? g_compute_indirect_keys.load(std::memory_order_relaxed)
	                                         : MinIndirectMappingCapacity,
	    std::bit_ceil(key_count));
	snapshot.flattened_srt.resize(mapping_offset + 1u + key_capacity * 2u);
	snapshot.flattened_srt[mapping_offset] = static_cast<uint32_t>(key_count);
	for (uint32_t entry = 0; entry < key_count; ++entry) {
		const auto key = sources.empty() ? selected_keys[entry] : entry;
		uint64_t table_offset = 0;
		DescriptorValue candidate;
		candidate.dword_count = dword_count;
		if (sources.empty()) {
			const auto indexed_offset = uint64_t {key * indirect.table_stride + indirect.table_offset};
			table_offset = byte_keys ? key
			    : table_value.dword_count == 4u ? scalar_buffer_offset(indexed_offset)
			                                  : (indexed_offset & ~uint64_t {3}) +
			                                        (indirect.table_immediate & ~3u);
			if (!(byte_keys && key == UINT32_MAX) && table_offset <= UINT32_MAX &&
			    !ReadScalarTable(table_base, table_size, table_offset, runtime,
			                     std::span(candidate.dwords).first(read_dwords))) {
				return read_failed("table descriptor read", key, 0u, table_base, table_offset, descriptor_bytes);
			}
		} else if (!WalkedDescriptor(program, sources[entry], candidate)) {
			return false;
		}
		if (!normalize(candidate))
			return normalization_failed(table_offset, key, indirect.table_stride, "key");
		uint32_t ordinal = 0;
		if (entry == 0 && !byte_keys) {
			descriptors[resource_index] = candidate;
		} else {
			ordinal = intern_candidate(candidate);
		}
		snapshot.flattened_srt[mapping_offset + 1u + entry * 2u] = key;
		snapshot.flattened_srt[mapping_offset + 2u + entry * 2u] = ordinal;
	}
	if (descriptors.size() == children_begin) {
		snapshot.flattened_srt.resize(mapping_offset);
	} else {
#if defined(ANYPS5_KYTY)
		if ((std::is_same_v<Specialization, ResourceSpecialization::Image> ||
		     std::is_same_v<Specialization, ResourceSpecialization::Buffer>) && stable_layout) {
#else
		if (std::is_same_v<Specialization, ResourceSpecialization::Image> && stable_layout) {
#endif
			const auto children = descriptors.size() - children_begin;
			const auto total    = std::max<size_t>(
                program.stage == ShaderType::Compute
                    ? g_compute_indirect_images.load(std::memory_order_relaxed)
                    : 1u,
                std::bit_ceil(children + 1u));
			for (auto padded = children;
			     padded + 1u < total; ++padded) {
				descriptors.push_back(descriptors.back());
				specializations.push_back(specializations.back());
			}
		}
		auto& root                      = specializations[resource_index];
		root.indirect_root              = resource_index;
		root.indirect_mapping_offset = static_cast<uint32_t>(mapping_offset);
		root.indirect_search_iterations = std::bit_width(key_capacity);
	}
	return true;
}

bool TableExemplar(const ImageResource& root, const DescriptorValue& descriptor, bool float_only) {
	if (NullImageDescriptor(descriptor) || DescriptorIsCube(descriptor) != root.cube) {
		return false;
	}
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto numeric_class = Prospero::SampledTextureNumericClass(format);
	return DescriptorDimension(descriptor, root.dimension) == root.dimension &&
	       numeric_class != Prospero::TextureNumericClass::Unsupported &&
	       (!float_only || numeric_class == Prospero::TextureNumericClass::Float) &&
	       ImageConversionFormat(format) == Prospero::BufferFormat::kInvalid &&
	       !Prospero::IsFmaskTextureFormat(format);
}

class TableSlotIndex {
public:
	void Reset(const uint32_t* words, size_t entries) {
		const auto capacity = std::bit_ceil(std::max<size_t>(entries * 2u, 16u));
		if (m_cells.size() < capacity) {
			m_cells.resize(capacity);
		}
		std::fill_n(m_cells.begin(), capacity, Cell {});
		m_mask  = capacity - 1u;
		m_words = words;
	}
	std::pair<uint32_t, bool> TryEmplace(uint32_t entry, uint32_t slot) {
		const auto* key = m_words + static_cast<size_t>(entry) * 8u;
		uint64_t    h   = 0x9e3779b97f4a7c15ull;
		for (uint32_t i = 0; i < 8u; i += 2u) {
			uint64_t lane = (uint64_t {key[i + 1u]} << 32u) | key[i];
			h             = (h ^ lane) * 0xff51afd7ed558ccdull;
			h ^= h >> 32u;
		}
		for (auto at = static_cast<size_t>(h) & m_mask;; at = (at + 1u) & m_mask) {
			auto& cell = m_cells[at];
			if (cell.entry_plus_one == 0u) {
				cell = {.entry_plus_one = entry + 1u, .slot = slot};
				return {slot, true};
			}
			if (std::memcmp(m_words + static_cast<size_t>(cell.entry_plus_one - 1u) * 8u, key,
			                8u * sizeof(uint32_t)) == 0) {
				return {cell.slot, false};
			}
		}
	}

private:
	struct Cell {
		uint32_t entry_plus_one = 0;
		uint32_t slot           = 0;
	};
	std::vector<Cell> m_cells;
	size_t            m_mask  = 0;
	const uint32_t*   m_words = nullptr;
};

uint32_t ImageTableCapacity(size_t slots) {
	return std::bit_ceil(std::max<uint32_t>(static_cast<uint32_t>(slots), 256u));
}

bool MaterializeImageTable(const ResourcePlan& program,
                           const DescriptorSource::IndirectDescriptor& indirect,
                           const DescriptorValue& table_value, uint32_t image_index,
                           const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                           ResourceSpecialization& specialization,
                           std::vector<ImageTableSnapshot>& previous,
                           std::vector<uint32_t>& mapping_offsets) {
	ShaderBufferResource table;
	if (table_value.dword_count != 4u || !DecodeBufferDescriptor(table_value, table) ||
	    (indirect.table_offset & 31u) != 0u) {
		return false;
	}
	const auto base       = table.Base48();
	const auto size       = table.GetSize();
	const auto entry_mask = indirect.table_mask >> 5u;
	const auto entries    = std::min<uint64_t>(
	    {size > indirect.table_offset ? (size - indirect.table_offset) / 32u : 0u,
	     uint64_t {entry_mask} + 1u, MaxIndirectDescriptorProbes});
	const auto& root = program.info.images[image_index];
	const auto same_table = [&](const ImageTableSnapshot& candidate) {
		return candidate.base == base && candidate.size == size &&
		       candidate.offset == indirect.table_offset && candidate.entry_mask == entry_mask;
	};
	auto found = std::ranges::find_if(snapshot.image_tables, same_table);
	if (found == snapshot.image_tables.end()) {
		ImageTableSnapshot next {.base = base, .size = size, .offset = indirect.table_offset,
		                         .entry_mask = entry_mask};
		static thread_local std::vector<uint32_t> words;
		words.resize(static_cast<size_t>(entries) * 8u);
		if (!words.empty() && !ReadScalarTable(base, size, indirect.table_offset, runtime, words)) {
			return false;
		}
		const auto reusable = std::ranges::find_if(previous, [&](const ImageTableSnapshot& old) {
			return same_table(old) && old.raw == words && !old.slots.empty();
		});
		static thread_local std::array<ImageTableSnapshot, 4> built;
		static thread_local uint32_t                          built_next = 0;
		const auto cached = reusable != previous.end()
		                        ? built.end()
		                        : std::ranges::find_if(built, [&](const ImageTableSnapshot& old) {
			                          return same_table(old) && !old.slots.empty() &&
			                                 old.raw == words;
		                          });
		if (reusable != previous.end()) {
			next.raw      = std::move(reusable->raw);
			next.slots    = std::move(reusable->slots);
			next.mapping  = std::move(reusable->mapping);
			next.slots_id = std::exchange(reusable->slots_id, 0);
		} else if (cached != built.end()) {
			next.raw.swap(words);
			next.slots    = cached->slots;
			next.mapping  = cached->mapping;
			next.slots_id = cached->slots_id;
		} else {
			static std::atomic<uint64_t> next_slots_id {0};
			next.slots_id = ++next_slots_id;
			next.raw.swap(words);
			next.slots.push_back(DescriptorValue {.dword_count = 8u});
			next.mapping.assign(static_cast<size_t>(entries) + 1u, 0u);
			next.mapping[0] = static_cast<uint32_t>(entries);
			static thread_local TableSlotIndex flat_slot_of;
			flat_slot_of.Reset(next.raw.data(), static_cast<size_t>(entries));
			for (uint32_t entry = 0; entry < entries; entry++) {
				DescriptorValue candidate {.dword_count = 8u};
				std::copy_n(next.raw.begin() + entry * 8u, 8u, candidate.dwords.begin());
				if (NullImageDescriptor(candidate) || !ValidImageDescriptor(candidate, root.r128)) {
					continue;
				}
				ReportUnknownTileMode("image table entry", program, image_index,
				                      base + indirect.table_offset + uint64_t {entry} * 32u, candidate);
				const auto new_slot = static_cast<uint32_t>(next.slots.size());
				uint32_t   slot     = new_slot;
				bool       inserted = false;
				std::tie(slot, inserted) = flat_slot_of.TryEmplace(entry, new_slot);
				if (inserted) {
					next.slots.push_back(candidate);
				}
				next.mapping[entry + 1u] = slot;
			}
			built[built_next++ % built.size()] = next;
		}
		snapshot.image_tables.push_back(std::move(next));
		mapping_offsets.push_back(static_cast<uint32_t>(snapshot.flattened_srt.size()));
		const auto& mapping = snapshot.image_tables.back().mapping;
		snapshot.flattened_srt.insert(snapshot.flattened_srt.end(), mapping.begin(), mapping.end());
		if (g_stable_indirect_layouts.load(std::memory_order_relaxed)) {
			snapshot.flattened_srt.resize(
			    mapping_offsets.back() + Common::AlignUp(mapping.size(), StableSrtWords), 0u);
		}
		found = snapshot.image_tables.end() - 1;
	}
	const auto table_index     = static_cast<uint32_t>(found - snapshot.image_tables.begin());
	auto&      image           = specialization.images[image_index];
	image.table                = table_index;
	image.table_capacity       = ImageTableCapacity(found->slots.size());
	image.table_mapping_offset = mapping_offsets[table_index];
	image.table_entry_mask     = entry_mask;
	snapshot.images[image_index] = {.dword_count = 8u};
	for (const bool float_only: {true, false}) {
		const auto exemplar = std::ranges::find_if(found->slots, [&](const DescriptorValue& slot) {
			return TableExemplar(root, slot, float_only);
		});
		if (exemplar != found->slots.end()) {
			snapshot.images[image_index] = *exemplar;
			break;
		}
	}
	return true;
}

bool SamplerSkipsDegamma(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                         uint32_t image_index) {
	return std::ranges::any_of(program.info.sampled_pairs, [&](const auto& pair) {
		return pair.image == image_index && pair.sampler < snapshot.samplers.size() &&
		       (snapshot.samplers[pair.sampler].dwords[0] >> 31u) != 0u;
	});
}

#if defined(ANYPS5_KYTY)
bool MaterializeImageTableHeap(const ResourcePlan& program,
                               const DescriptorSource::IndirectDescriptor& indirect,
                               const DescriptorValue& table_value, uint32_t image_index,
                               const SrtRuntime& runtime, ResourceSnapshot& snapshot,
                               ResourceSpecialization& specialization) {
	const auto&          root = program.info.images[image_index];
	ShaderBufferResource table;
	if (root.resource_class != ImageResourceClass::Sampled ||
	    root.mip_mode == ImageMipMode::Dynamic ||
	    table_value.dword_count != 4u ||
	    !DecodeBufferDescriptor(table_value, table) || (indirect.table_offset & 31u) != 0u) {
		return false;
	}
	const auto base    = table.Base48();
	const auto size    = table.GetSize();
	const auto entries = std::min<uint64_t>(
	    {size > indirect.table_offset ? (size - indirect.table_offset) / 32u : 0u,
	     uint64_t {indirect.table_mask >> 5u} + 1u, MaxIndirectDescriptorProbes});
	if (entries == 0u) {
		return false;
	}
	const auto profile = ImageTableHeapProfile(root, SamplerSkipsDegamma(program, snapshot, image_index));
	ImageHeapView   view;
	DescriptorValue exemplar {.dword_count = 8u};
	if (runtime.image_heap != nullptr) {
		if (!runtime.image_heap->Acquire({.base  = base,
		                                  .size  = size,
		                                  .step  = 32u,
		                                  .first = indirect.table_offset,
		                                  .count = entries,
		                                  .root  = &profile,
		                                  .table = true},
		                                 view) ||
		    view.kinds.size() != 1u || !ImageTableHeapExemplar(profile, view.kinds[0])) {
			return false;
		}
		snapshot.specialization_reads.emplace_back(base, size);
		exemplar = view.kinds[0];
	} else {
		static thread_local std::vector<uint32_t> words;
		words.resize(static_cast<size_t>(entries) * 8u);
		if (!ReadScalarTable(base, size, indirect.table_offset, runtime, words)) {
			return false;
		}
		bool found = false;
		for (size_t entry = 0; !found && entry < entries; ++entry) {
			std::copy_n(words.begin() + static_cast<std::ptrdiff_t>(entry * 8u), root.r128 ? 4u : 8u,
			            exemplar.dwords.begin());
			found = ImageTableSlotCompatible(profile, exemplar) && ImageTableHeapExemplar(profile, exemplar);
		}
		if (!found) {
			return false;
		}
		view.slot_count = static_cast<uint32_t>((size & ~uint64_t {3}) / 4u);
	}
	auto heap_root                      = specialization.images[image_index];
	heap_root.indirect_root             = image_index;
	heap_root.indirect_mapping_offset   = static_cast<uint32_t>(snapshot.flattened_srt.size());
	heap_root.heap                      = true;
	specialization.images[image_index]  = heap_root;
	snapshot.images[image_index]        = {.dword_count = 8u};
	snapshot.flattened_srt.insert(snapshot.flattened_srt.end(),
	                              {view.slot_count, static_cast<uint32_t>(view.slots),
	                               static_cast<uint32_t>(view.slots >> 32u)});
	snapshot.images.push_back(exemplar);
	specialization.images.push_back(heap_root);
	return true;
}
#endif

}

bool ImageTableSlotCompatible(const ImageResource& root, const DescriptorValue& descriptor) {
	if (NullImageDescriptor(descriptor) || !ValidImageDescriptor(descriptor, root.r128) ||
	    DescriptorIsCube(descriptor) != root.cube) {
		return false;
	}
	const auto format = static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
	const auto dimension = DescriptorDimension(descriptor, root.dimension);
	const auto is_2d     = [](Decoder::ImageDimension value) {
		return value == Decoder::ImageDimension::Dim2D ||
		       value == Decoder::ImageDimension::Dim2DArray;
	};
	return (dimension == root.dimension || (is_2d(dimension) && is_2d(root.dimension))) &&
	       Prospero::SampledTextureNumericClass(format) == root.numeric_class &&
	       ImageConversionFormat(format) == root.conversion_format &&
	       !Prospero::IsFmaskTextureFormat(format);
}

struct SamplerPlan {
	struct Binding {
		uint32_t     source;
		SamplerClass type;
	};
	std::vector<std::array<uint32_t, 3>> mapping;
	std::vector<Binding>                bindings;
};

template <typename T, typename Keep>
void CompactImages(std::vector<T>& images, Keep&& keep) {
	size_t count = 0;
	for (size_t index = 0; index < images.size(); ++index) {
		if (!keep(index)) continue;
		if (count != index) images[count] = std::move(images[index]);
		++count;
	}
	images.resize(count);
}

struct ImageRemap {
	explicit ImageRemap(const ResourceSpecialization& specialization)
	    : indices(specialization.images.size()),
	      source_count(static_cast<uint32_t>(specialization.images.size())) {
		for (uint32_t index = 0; index < source_count; index++) {
			indices[index] = specialization.images[index].fmask ? UINT32_MAX : count++;
		}
	}

	uint32_t operator[](uint32_t index) const {
		EXIT_IF(index >= source_count);
		return indices[index];
	}

	template <typename T>
	void Apply(std::vector<T>& images) const {
		EXIT_IF(images.size() != source_count);
		if (count == source_count) {
			return;
		}
		CompactImages(images, [&](size_t index) { return indices[index] != UINT32_MAX; });
	}

private:
	std::vector<uint32_t> indices;
	uint32_t                                    source_count;
	uint32_t                                    count = 0;
};

static bool MemoryImageCandidates(const ResourcePlan& program, const ImageResource& base) {
	const auto* source = Source(program, base.source);
	return source != nullptr && source->indirect_descriptor.has_value() &&
	       source->indirect_descriptor->sources.empty();
}

static bool BuildResourceSpecialization(const ResourcePlan& program, ResourceSnapshot& snapshot,
                                        ResourceSpecialization& specialization) {
	for (uint32_t i = 0; i < specialization.images.size(); i++) {
		const auto& descriptor = snapshot.images[i];
		auto&       image      = specialization.images[i];
		const auto  base_index = i < program.info.images.size() ? i : image.indirect_root;
		if (base_index >= program.info.images.size()) {
			return SpecializationFail(fmt::format("image resource {} has an invalid root", i));
		}
		const auto& base = program.info.images[base_index];
		if (base.resource_class == ImageResourceClass::None ||
		    (base.atomic && base.resource_class != ImageResourceClass::Storage)) {
			return SpecializationFail(fmt::format("image resource {} has an invalid class", i));
		}
		image.mip_count = ImageMipCount(base, descriptor);
		if (NullImageDescriptor(descriptor)) {
			image.numeric_class = base.atomic ? Prospero::TextureNumericClass::Uint
			                                  : Prospero::TextureNumericClass::Float;
			if (image.dimension == Decoder::ImageDimension::Unknown) {
				image.dimension = Decoder::ImageDimension::Dim2D;
				image.cube      = false;
			}
			continue;
		}
		const auto descriptor_dimension = DescriptorDimension(descriptor, base.dimension);
		if (descriptor_dimension == Decoder::ImageDimension::Unknown) {
			return SpecializationFail(fmt::format(
			    "image descriptor {} has unsupported type {}: {:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, (descriptor.dwords[3] >> 28u) & 0xfu, descriptor.dwords[0], descriptor.dwords[1],
			    descriptor.dwords[2], descriptor.dwords[3], descriptor.dwords[4],
			    descriptor.dwords[5], descriptor.dwords[6], descriptor.dwords[7]));
		}
		image.dimension = descriptor_dimension;
		image.cube      = DescriptorIsCube(descriptor);
		const auto format =
		    static_cast<Prospero::BufferFormat>((descriptor.dwords[1] >> 20u) & 0x1ffu);
		if (base.atomic &&
		    (base.atomic64 ? format != Prospero::BufferFormat::k32_32UInt
		                   : format != Prospero::BufferFormat::k32UInt &&
		                         format != Prospero::BufferFormat::k32SInt &&
		                         format != Prospero::BufferFormat::k32Float)) {
			return SpecializationFail(
			    fmt::format("atomic image descriptor {} uses unsupported format {}", i,
			                static_cast<uint32_t>(format)));
		}
		const bool storage      = base.resource_class == ImageResourceClass::Storage;
		image.fmask             = Prospero::IsFmaskTextureFormat(format);
		if (image.fmask) {
			if (storage || base.depth_compare ||
			    image.indirect_root != ImageResource::NoIndirectImage ||
			    std::ranges::any_of(program.info.sampled_pairs,
			                        [&](const auto& pair) { return pair.image == i; })) {
				return SpecializationFail("FMASK requires a direct image load");
			}
		}
		const bool depth_bits =
		    !storage && base.unfiltered_access_only && format == Prospero::BufferFormat::k16UInt &&
		    static_cast<Prospero::TileMode>((descriptor.dwords[3] >> 20u) & 0x1fu) ==
		        Prospero::TileMode::kDepth;
		image.conversion_format = depth_bits ? format : ImageConversionFormat(format);
		image.skip_degamma = !image.heap && Prospero::IsSrgbFormat(format) &&
		                     SamplerSkipsDegamma(program, snapshot, base_index);
		if (!storage && !base.depth_compare && !image.skip_degamma &&
		    image.indirect_root == ImageResource::NoIndirectImage &&
		    (format == Prospero::BufferFormat::k8Srgb || format == Prospero::BufferFormat::k8_8Srgb)) {
			image.srgb_decode = format;
		}
		if (storage || image.conversion_format != Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = DescriptorImageSwizzle(descriptor);
		}
		const auto raw_sint_format = RawSignedStorageFormat(base, format);
		const bool raw_sint_storage = raw_sint_format != Prospero::BufferFormat::kInvalid;
		if (raw_sint_storage && raw_sint_format != Prospero::BufferFormat::k32UInt)
			image.conversion_format = format;
		if (storage && image.conversion_format == Prospero::BufferFormat::kInvalid) {
			image.shader_swizzle = CanonicalStorageSwizzle(image.shader_swizzle, format);
		}
		image.numeric_class = Prospero::SampledTextureNumericClass(
		    depth_bits ? Prospero::BufferFormat::k16UNorm : Prospero::RemapTextureFormat(format));
		if (!storage && format == Prospero::BufferFormat::k11_11_10UNorm) {
			if (image.indirect_root != ImageResource::NoIndirectImage) {
				return SpecializationFail("packed UNorm sampling requires a direct image");
			}
			if (std::ranges::any_of(program.info.sampled_pairs,
			                       [&](const auto& pair) { return pair.image == i; }) &&
			    !ValidatePackedUnormSampling(program, snapshot, i)) return false;
		}
		if (storage) {
			if ((!raw_sint_storage && image.numeric_class == Prospero::TextureNumericClass::Sint) ||
			    image.numeric_class == Prospero::TextureNumericClass::Unsupported) {
				return SpecializationFail(
				    fmt::format("storage image descriptor {} uses unsupported format {}", i,
				                static_cast<uint32_t>(format)));
			}
			if (raw_sint_storage || base.atomic) {
				image.numeric_class = Prospero::TextureNumericClass::Uint;
			}
		} else if (image.numeric_class == Prospero::TextureNumericClass::Unsupported ||
		           (base.depth_compare &&
		            image.numeric_class != Prospero::TextureNumericClass::Float)) {
			auto message = fmt::format(
			    "sampled image descriptor {} uses unsupported format {}: root={} source={} "
			    "first_use_pc={:#x} r128={} words={:08x},{:08x},{:08x},{:08x},"
			    "{:08x},{:08x},{:08x},{:08x}",
			    i, static_cast<uint32_t>(format), base_index, base.source, base.first_use_pc,
			    base.r128, descriptor.dwords[0], descriptor.dwords[1], descriptor.dwords[2],
			    descriptor.dwords[3], descriptor.dwords[4], descriptor.dwords[5],
			    descriptor.dwords[6], descriptor.dwords[7]);
			const auto* source = Source(program, base.source);
			if (i >= program.info.images.size() && source != nullptr &&
			    source->indirect_descriptor &&
			    IsBoundedDescriptorTable(program, *source->indirect_descriptor) &&
			    specialization.images[base_index].indirect_search_iterations == 0u &&
			    !source->indirect_descriptor->pointer_table) {
				const auto& indirect = *source->indirect_descriptor;
				const size_t mapping = specialization.images[base_index].indirect_mapping_offset;
				const auto& flat = snapshot.flattened_srt;
				if (mapping < flat.size() && flat[mapping] <= flat.size() - mapping - 1u) {
					uint32_t ordinal = 0;
					for (size_t child = program.info.images.size(); child <= i; ++child)
						ordinal += specialization.images[child].indirect_root == base_index;
					uint32_t matches = 0;
					uint64_t first_offset = 0;
					for (uint32_t probe = 0; probe < flat[mapping]; ++probe) {
						if (flat[mapping + 1u + probe] != ordinal) continue;
						if (matches++ == 0u) first_offset = uint64_t {probe} * 4u;
					}
					message += fmt::format(" table_source={} ordinal={} dense_matches={} "
					                       "first_candidate_offset={:#x}",
					                       indirect.table_source, ordinal, matches, first_offset);
				}
			}
			return SpecializationFail(message);
		}
	}
	for (uint32_t root_index = 0; root_index < specialization.images.size(); root_index++) {
		auto& root = specialization.images[root_index];
		if (root.indirect_root != root_index) {
			continue;
		}
		const auto key_count = root.indirect_mapping_offset < snapshot.flattened_srt.size()
		                           ? snapshot.flattened_srt[root.indirect_mapping_offset]
		                           : 0u;
		size_t mapping_words = 3u;
		if (!root.heap)
			mapping_words = 1u + size_t {key_count} * (root.indirect_search_iterations == 0u ? 1u : 2u);
		if (key_count == 0u ||
		    root.indirect_mapping_offset + mapping_words > snapshot.flattened_srt.size()) {
			return SpecializationFail("indirect image specialization has an invalid key mapping");
		}
		uint32_t exemplar       = ImageResource::NoIndirectImage;
		uint32_t resource_count = 0;
		for (uint32_t resource = 0; resource < specialization.images.size(); resource++) {
			if (specialization.images[resource].indirect_root != root_index) {
				continue;
			}
			resource_count++;
			if (exemplar == ImageResource::NoIndirectImage &&
			    !NullImageDescriptor(snapshot.images[resource])) {
				exemplar = resource;
			}
		}
		if (resource_count < 2u || exemplar == ImageResource::NoIndirectImage) {
			return SpecializationFail("indirect image specialization has no typed candidate");
		}
		const auto& image_class = specialization.images[exemplar];
		const auto& base = program.info.images[root_index];
		const bool null_strays = MemoryImageCandidates(program, base) && base.sample_access_only &&
		                         image_class.dimension == base.dimension;
		const auto coordinates = [](Decoder::ImageDimension dimension) {
			switch (dimension) {
				case Decoder::ImageDimension::Dim1D: return 1u;
				case Decoder::ImageDimension::Dim1DArray:
				case Decoder::ImageDimension::Dim2D:
				case Decoder::ImageDimension::Dim2DMsaa: return 2u;
				case Decoder::ImageDimension::Dim3D:
				case Decoder::ImageDimension::Dim2DArray:
				case Decoder::ImageDimension::Dim2DMsaaArray: return 3u;
				default: return 0u;
			}
		};
		const bool storage_swizzle = base.unfiltered_access_only &&
		    base.resource_class == ImageResourceClass::Storage;
		const auto is_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim2D ||
			       dimension == Decoder::ImageDimension::Dim2DArray;
		};
		const auto is_1d_or_2d = [](Decoder::ImageDimension dimension) {
			return dimension == Decoder::ImageDimension::Dim1D ||
			       dimension == Decoder::ImageDimension::Dim2D;
		};
		const auto leading = [&](Decoder::ImageDimension dimension, const DescriptorValue& value) {
			const bool sliced =
			    static_cast<Prospero::ImageType>((value.dwords[3] >> 28u) & 0xfu) ==
			        Prospero::ImageType::kColor2DArray &&
			    (value.dwords[4] & 0x1fffu) != ((value.dwords[4] >> 16u) & 0x1fffu);
			return !sliced &&
			       (dimension == Decoder::ImageDimension::Dim2D || dimension == base.dimension);
		};
		const bool leading_coordinates = (base.dimension == Decoder::ImageDimension::Dim2D ||
		                                  base.dimension == Decoder::ImageDimension::Dim3D) &&
		    leading(image_class.dimension, snapshot.images[exemplar]) &&
		    std::ranges::none_of(program.memory_info, [&](const MemoryInfo& memory) {
			    return memory.kind == ResourceKind::Image && memory.resource == root_index &&
			           (memory.image_has_mip || (memory.image_sample_flags &
			               (Decoder::ImageSampleFlagLod | Decoder::ImageSampleFlagDerivative |
			                Decoder::ImageSampleFlagLodClamp)) != 0u);
		    });
		for (uint32_t candidate = 0; candidate < specialization.images.size(); candidate++) {
			auto& image = specialization.images[candidate];
			if (image.indirect_root != root_index) {
				continue;
			}
			const auto null_candidate = [&] {
				image.numeric_class     = image_class.numeric_class;
				image.dimension         = image_class.dimension;
				image.mip_count         = image_class.mip_count;
				image.conversion_format = image_class.conversion_format;
				image.shader_swizzle    = image_class.shader_swizzle;
				image.cube              = image_class.cube;
			};
			if (NullImageDescriptor(snapshot.images[candidate])) {
				null_candidate();
			}
			const bool same_coordinates = image.dimension == image_class.dimension &&
			                              image.cube == image_class.cube;
			const bool unfiltered_coordinates = base.unfiltered_access_only &&
			    base.dimension == Decoder::ImageDimension::Dim2D &&
			    !image.cube && !image_class.cube && is_1d_or_2d(image.dimension) &&
			    is_1d_or_2d(image_class.dimension);
			const bool prefix_coordinates =
			    leading_coordinates && leading(image.dimension, snapshot.images[candidate]);
			const bool other_coordinates =
			    !same_coordinates && !(is_2d(image.dimension) && is_2d(image_class.dimension)) &&
			    !unfiltered_coordinates && !prefix_coordinates;
			if (other_coordinates && null_strays && !image.cube && coordinates(image.dimension) != 0u &&
			    coordinates(image.dimension) < coordinates(base.dimension)) {
				snapshot.images[candidate].dwords.fill(0u);
				null_candidate();
			} else if (other_coordinates ||
			           image.conversion_format != image_class.conversion_format ||
			           (!storage_swizzle && image.shader_swizzle != image_class.shader_swizzle)) {
				return SpecializationFail(
				    fmt::format("indirect image table at pc 0x{:08x} has incompatible candidates",
				                program.info.images[root_index].first_use_pc));
			}
		}
	}
	CompactImages(snapshot.images, [&](size_t index) { return !specialization.images[index].fmask; });
	return true;
}

bool BuildSamplerPlan(const ShaderInfo& base, SamplerPlan& plan) {
	plan.mapping.resize(base.samplers.size());
	plan.bindings.resize(base.samplers.size());
	std::vector<uint8_t> usage(base.samplers.size());
	for (const auto& pair: base.sampled_pairs) {
		if (pair.image >= base.images.size() || pair.sampler >= base.samplers.size()) {
			return false;
		}
		usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[pair.image]));
		for (const auto candidate: base.images[pair.image].indirect_resources) {
			if (candidate >= base.images.size()) return false;
			usage[pair.sampler] |= 1u << static_cast<uint32_t>(ClassifySampler(base.images[candidate]));
		}
		for (const auto candidate: base.samplers[pair.sampler].indirect_resources) {
			if (candidate >= base.samplers.size()) return false;
			usage[candidate] |= usage[pair.sampler];
		}
	}
	for (uint32_t index = 0; index < base.samplers.size(); index++) {
		auto& mapping = plan.mapping[index];
		mapping.fill(UINT32_MAX);
		const auto classes = usage[index] == 0u ? 1u : usage[index];
		bool       first   = true;
		for (uint32_t type = 0; type < mapping.size(); type++) {
			if ((classes & (1u << type)) == 0u) continue;
			const auto target = first ? index : static_cast<uint32_t>(plan.bindings.size());
			if (!first) plan.bindings.emplace_back();
			mapping[type]         = target;
			plan.bindings[target] = {index, static_cast<SamplerClass>(type)};
			first                = false;
		}
	}
	return true;
}

template <typename Predicate>
static std::vector<ResourceBlock> ResourceControlFlow(const Program& program, const Predicate& predicate) {
	if (program.has_address_writes) {
		return {};
	}
	std::unordered_map<const Block*, uint32_t> indices;
	for (uint32_t i = 0; i < program.blocks.size(); i++) {
		if (!indices.emplace(program.blocks[i], i).second) {
			return {};
		}
	}
	std::vector<ResourceBlock> blocks(program.blocks.size());
	for (uint32_t i = 0; i < blocks.size(); i++) {
		auto&                 block      = blocks[i];
		const auto&             info       = *program.blocks[i];
		const auto&           terminator = info.terminator;
		const std::array        targets {terminator.true_block, terminator.false_block};
		std::span<Block* const> successors;
		switch (terminator.kind) {
			case CFG::TerminatorKind::Branch: successors = std::span(targets).first(1); break;
			case CFG::TerminatorKind::ConditionalBranch:
				successors      = targets;
				block.condition = info.condition;
				break;
			case CFG::TerminatorKind::IndirectBranch: successors = info.ImmSuccessors(); break;
			case CFG::TerminatorKind::Return: break;
			default: return {};
		}
		for (const auto successor: successors) {
			const auto found = indices.find(successor);
			if (found == indices.end()) {
				return {};
			}
			block.successors.push_back(found->second);
		}
		for (const auto& inst: *program.blocks[i]) {
			const auto op     = inst.GetOpcode();
			if (op == ValueOpcode::ReadConst) {
				block.srt_reads.push_back(inst.Arg(1).Resolve().U32());
				continue;
			}
			const auto buffer = BufferAccessOf(op);
			const auto image  = ImageOpcodeInfoOf(op);
			if (buffer == BufferAccess::None && image.access == ImageAccess::None) {
				continue;
			}
			const auto& memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			if (memory.planning_only || memory.kind == ResourceKind::IndirectBuffer) {
				continue;
			}
			if (buffer != BufferAccess::None) {
				block.sources.push_back(program.info.buffers.at(memory.resource).source);
			} else {
				block.sources.push_back(program.info.images.at(memory.resource).source);
				block.images.push_back(memory.resource);
				if (image.needs_sampler) {
					block.sources.push_back(program.info.samplers.at(memory.sampler).source);
				}
			}
		}
		for (auto* list: {&block.sources, &block.images}) {
			std::ranges::sort(*list);
			list->erase(std::unique(list->begin(), list->end()), list->end());
		}
	}
	for (auto& block: blocks) block.condition = predicate(block.condition);
	if (std::ranges::none_of(
	        blocks, [](const ResourceBlock& block) { return !block.condition.IsEmpty(); })) {
		return {};
	}
	return blocks;
}

static bool IsFullDispatchPredicate(Value value, uint32_t depth = 0) {
	value = value.Resolve();
	if (value == Value(true)) return true;
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || depth > 32) return false;
	if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
		return IsFullDispatchPredicate(inst->Arg(0), depth + 1) &&
		       IsFullDispatchPredicate(inst->Arg(1), depth + 1);
	}
	if (inst->GetOpcode() != ValueOpcode::ULessThan32) return false;
	const auto* global = inst->Arg(0).Resolve().TryInstruction();
	const auto* extent = inst->Arg(1).Resolve().TryInstruction();
	if (global == nullptr || global->GetOpcode() != ValueOpcode::GetBuiltin ||
	    global->Arg(0).Resolve() != Value(static_cast<uint32_t>(StageInputKind::GlobalInvocationId)) ||
	    extent == nullptr || extent->GetOpcode() != ValueOpcode::GetDispatchThreadExtent)
		return false;
	const auto axis = global->Arg(1).Resolve();
	return axis.IsImmediate() && axis.U32() < 3 && extent->Arg(0).Resolve() == axis;
}

static std::optional<std::array<uint64_t, 3>> FillIndex(Value value, uint32_t axis, Value guard) {
	return AffineIndex<3>(value, [&](Value at) { return ResolveActiveU32(at, guard); },
	    [&](Value at) -> std::optional<size_t> {
		    const auto* inst = at.TryInstruction();
		    if (inst == nullptr || inst->GetOpcode() != ValueOpcode::GetBuiltin ||
		        inst->Arg(1) != Value(axis)) return {};
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId))) return 1u;
		    if (inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId))) return 2u;
		    return {};
	    });
}

static UniformFillPlan AnalyzeUniformFill(const Program& program) {
	if (program.stage != ShaderType::Compute || program.blocks.empty() ||
	    !program.info.samplers.empty()) {
		return {};
	}
	std::unordered_set<const Block*> visited;
	const auto*                      block = program.blocks.front();
	const Inst*                      store = nullptr;
	for (;;) {
		if (block == nullptr || !visited.insert(block).second) return {};
		for (const auto& inst: *block) {
			if (AddressOpcodeInfoOf(inst.GetOpcode()).access != AddressAccess::None) return {};
			if (!inst.MayHaveSideEffects()) continue;
			if (store != nullptr || (BufferAccessOf(inst.GetOpcode()) != BufferAccess::Write &&
			                         inst.GetOpcode() != ValueOpcode::ImageWrite))
				return {};
			store = &inst;
		}
		const auto& term = block->terminator;
		if (term.kind == CFG::TerminatorKind::Return) break;
		if (term.kind != CFG::TerminatorKind::Branch) return {};
		block = term.true_block;
	}
	if (store == nullptr || visited.size() != program.blocks.size()) return {};
	for (const auto& buffer: program.info.buffers) {
		if (buffer.read && (!buffer.scalar || buffer.written)) return {};
	}
	const auto& memory = program.memory_info.at(store->Flags<MemoryFlags>().index);
	UniformFillPlan result;
	result.fill.resource = memory.resource;
	Value data;
	Value guard(true);
	if (store->GetOpcode() == ValueOpcode::ImageWrite) {
		if (program.info.images.size() != 1 || memory.dmask != 1 || memory.data_bits != 32 ||
		    memory.image_has_mip || memory.image_sample_flags != 0 || memory.image_r128 ||
		    memory.image_dimension != Decoder::ImageDimension::Dim2DArray ||
		    store->Arg(3).Resolve() != Value(true)) return {};
		const auto& image = program.info.images[memory.resource];
		if (image.read || image.atomic || image.mip_mode != ImageMipMode::None) return {};
		const auto* address = store->Arg(1).ResolveInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress) return {};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const auto index = FillIndex(address->Arg(axis), axis, guard);
			if (!index || (*index)[0] != 0) return {};
			if (axis < 2) {
				if ((*index)[1] != 1 || (*index)[2] == 0) return {};
			} else if ((*index)[1] != 0 || (*index)[2] != 1) {
				return {};
			}
			result.fill.group_stride[axis] = static_cast<uint32_t>((*index)[2]);
		}
		const auto* values = store->Arg(2).ResolveInstruction();
		if (values == nullptr || values->GetOpcode() != ValueOpcode::CompositeConstructU32x4)
			return {};
		result.fill.kind  = UniformFillKind::Image;
		result.fill.words = 1;
		data = values->Arg(0);
	} else {
		if (!program.info.images.empty()) return {};
		const auto           op = store->GetOpcode();
		constexpr std::array stores {ValueOpcode::StoreBufferU32, ValueOpcode::StoreBufferU32x2,
		                             ValueOpcode::StoreBufferU32x3, ValueOpcode::StoreBufferU32x4};
		const auto           store_op = std::ranges::find(stores, op);
		if (store_op == stores.end() || store->Arg(2).Resolve() != Value(0u) ||
		    store->Arg(3).Resolve() != Value(0u))
			return {};
		guard = store->Arg(5).Resolve();
		if (!IsFullDispatchPredicate(guard)) return {};
		if (!memory.formatted || memory.typed || !memory.idxen || memory.offen || memory.offset != 0 ||
		    memory.data_bits != 32 ||
		    memory.data_dwords != static_cast<uint32_t>(store_op - stores.begin() + 1))
			return {};
		const auto address = FillIndex(store->Arg(1), 0, guard);
		if (!address || (*address)[0] != 0 || (*address)[1] != 1 || (*address)[2] == 0) return {};
		result.fill.kind = UniformFillKind::Buffer;
		result.fill.group_stride[0] = static_cast<uint32_t>((*address)[2]);
		result.fill.words = memory.data_dwords;
		data = store->Arg(4);
	}
	data = ResolveActiveU32(data, guard);
	const auto*          vector = data.TryInstruction();
	constexpr std::array composites {ValueOpcode::CompositeConstructU32x2,
	                                 ValueOpcode::CompositeConstructU32x3,
	                                 ValueOpcode::CompositeConstructU32x4};
	if (result.fill.words > 1 &&
	    (vector == nullptr || vector->GetOpcode() != composites[result.fill.words - 2]))
		return {};
	for (uint32_t i = 0; i < result.fill.words; ++i) {
		const auto word = ResolveActiveU32(result.fill.words == 1 ? data : vector->Arg(i), guard);
		if (word.GetType() != Type::U32 ||
		    !ValidateRuntimeValue(program, word, RuntimeValueType::Integer))
			return {};
		result.values[i] = word;
	}
	return result;
}

static bool DerivePlanState(ResourcePlan& plan) {
	for (const auto group: plan.selector_groups) {
		if (group == plan.selector_keys.size()) plan.selector_keys.emplace_back();
		if (group != UINT32_MAX && group >= plan.selector_keys.size()) return false;
	}
	if (plan.srt_plan_complete && plan.resource_tracking_complete) CompileSrtWalker(plan);
	return true;
}

static void FindBufferSpecializationUses(const Program& program, ResourcePlan& plan) {
	const auto& buffers = program.info.buffers;
	plan.buffer_vector_access.assign(buffers.size(), 0u);
	plan.buffer_feeds_descriptors.assign(buffers.size(), 0u);
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		if (buffers[index].indirect_root != BufferResource::NoIndirectBuffer ||
		    !buffers[index].indirect_resources.empty()) {
			plan.buffer_vector_access[index] = plan.buffer_feeds_descriptors[index] = 1u;
		}
	}
	for (const auto& memory: program.memory_info) {
		if (!memory.planning_only && memory.resource < buffers.size() &&
		    (memory.kind == ResourceKind::Buffer || memory.kind == ResourceKind::IndirectBuffer)) {
			plan.buffer_vector_access[memory.resource] = 1u;
		}
	}
	std::unordered_set<const Inst*> visited;
	std::vector<const Inst*>        pending;
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto opcode = inst.GetOpcode();
			if (opcode == ValueOpcode::GetBufferResource || opcode == ValueOpcode::GetImageResource ||
			    opcode == ValueOpcode::GetSamplerResource) {
				pending.push_back(&inst);
			}
		}
	}
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		if (!visited.insert(inst).second) continue;
		if (inst->GetOpcode() == ValueOpcode::ReadConstBuffer) {
			const auto& memory = program.memory_info.at(inst->Flags<MemoryFlags>().index);
			if (memory.kind == ResourceKind::ScalarBuffer && !memory.planning_only &&
			    memory.resource < buffers.size()) {
				plan.buffer_feeds_descriptors[memory.resource] = 1u;
			}
		}
		for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
			if (const auto* source = inst->Arg(arg).Resolve().TryInstruction()) pending.push_back(source);
		}
	}
}

ResourcePlan ExtractResourcePlan(const Program& program) {
	ResourcePlan plan;
	plan.stage                      = program.stage;
	plan.shader_hash                = program.shader_hash;
	plan.user_data_base             = program.user_data_base;
	plan.user_data_count            = program.user_data_count;
	plan.info                       = program.info;
	plan.memory_info                = program.memory_info;
	plan.source_reads               = program.source_reads;
	plan.srt_plan_complete          = program.srt_plan_complete;
	plan.resource_tracking_complete = program.resource_tracking_complete;
	FindBufferSpecializationUses(program, plan);

	std::unordered_map<const Inst*, Inst*> cloned;
	std::function<Value(Value)>            Clone = [&](Value value) -> Value {
		value              = value.Resolve();
		const auto* source = value.TryInstruction();
		if (source == nullptr) {
			return value;
		}
		if (source->GetOpcode() == ValueOpcode::ReadConst) {
			const auto slot = source->Arg(1).Resolve();
			EXIT_IF(!slot.IsImmediate() || slot.GetType() != Type::U32 ||
			        slot.U32() >= program.srt_reads.size());
			return Clone(program.srt_reads[slot.U32()].value);
		}
		if (source->GetOpcode() == ValueOpcode::ReadFirstLane &&
		    ValidateRuntimeValue(program, source->Arg(0))) {
			return Clone(source->Arg(0));
		}
		if (source->GetOpcode() == ValueOpcode::Phi) {
			const auto invariant = ResolveInvariantPhi(program, value);
			if (!invariant.IsEmpty() && invariant != value) {
				return Clone(invariant);
			}
		}
		if (const auto found = cloned.find(source); found != cloned.end()) {
			return Value(found->second);
		}
		if (source->GetOpcode() == ValueOpcode::LoadBufferU32) plan.capture_specialization_reads = true;
		auto& target =
		    plan.value_storage.emplace_back(source->GetOpcode(), source->Flags<uint64_t>());
		if (source->GetOpcode() == ValueOpcode::LoadAddressU32 ||
		    source->GetOpcode() == ValueOpcode::ReadConstBuffer ||
		    source->GetOpcode() == ValueOpcode::LoadBufferU32) {
			target.SetFlags(MemoryFlags {.index = source->Flags<MemoryFlags>().index});
		}
		cloned.emplace(source, &target);
		if (source->GetOpcode() == ValueOpcode::Phi) {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.AddPhiOperand(nullptr, Clone(source->Arg(index)));
			}
		} else {
			for (size_t index = 0; index < source->NumArgs(); index++) {
				target.SetArg(index, Clone(source->Arg(index)));
			}
		}
		return Value(&target);
	};

	bool capture_indirect_reads = false;
	plan.descriptor_sources.reserve(program.descriptor_sources.size());
	for (const auto& source: program.descriptor_sources) {
		auto& target          = plan.descriptor_sources.emplace_back();
		target.dword_count    = source.dword_count;
		target.indirect_descriptor = source.indirect_descriptor;
		if (target.indirect_descriptor.has_value()) {
			capture_indirect_reads = true;
			target.indirect_descriptor->key_count = Clone(target.indirect_descriptor->key_count);
			target.indirect_descriptor->selector_first =
			    Clone(target.indirect_descriptor->selector_first);
			target.indirect_descriptor->selector_mask =
			    Clone(target.indirect_descriptor->selector_mask);
			target.indirect_descriptor->selector_byte_bias = Clone(target.indirect_descriptor->selector_byte_bias);
		}
		for (uint32_t dword = 0; dword < source.dword_count; dword++) {
			target.dwords[dword] = Clone(source.dwords[dword]);
		}
	}
	std::vector<uint32_t> group_sources;
	plan.selector_groups.resize(plan.descriptor_sources.size(), UINT32_MAX);
	for (uint32_t source = 0; source < plan.descriptor_sources.size(); ++source) {
		const auto& descriptor = program.descriptor_sources[source].indirect_descriptor;
		if (!descriptor || !descriptor->selector) continue;
		const auto& a = *descriptor;
		uint32_t group = 0;
		for (; group < group_sources.size(); ++group) {
			const auto& b = *program.descriptor_sources[group_sources[group]].indirect_descriptor;
			if (EquivalentSelectorDomain(program, a, b)) break;
		}
		if (group == group_sources.size()) group_sources.push_back(source);
		plan.selector_groups[source] = group;
	}
	plan.srt_reads.reserve(program.srt_reads.size());
	for (const auto& read: program.srt_reads) {
		plan.srt_reads.push_back({Clone(read.value), read.flat_offset});
	}
	bool capture_control_reads = false;
	bool uniform_control = false;
	const auto control_value_count = plan.value_storage.size();
	struct ControlValue {
		Value value;
		bool complete = false;
	};
	std::unordered_map<const Inst*, ControlValue> predicates;
	std::function<Value(Value)> ClonePredicate = [&](Value value) -> Value {
		value = value.Resolve();
		if (value.IsEmpty()) return {};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			uniform_control |= value.GetType() == Type::U1;
			return ValidateRuntimeValue(program, value, RuntimeValueType::Integer) ? value : Value {};
		}
		const auto [found, inserted] = predicates.try_emplace(inst);
		auto& entry = found->second;
		auto& result = entry.value;
		const auto opaque = [&]() -> Value {
			if (value.GetType() != Type::U1 && value.GetType() != Type::U32) return {};
			return result = Value(&plan.value_storage.emplace_back(
			    value.GetType() == Type::U1 ? ValueOpcode::UndefU1 : ValueOpcode::UndefU32));
		};
		if (!inserted) return entry.complete || !result.IsEmpty() ? result : opaque();
		const auto finish = [&](Value projected) {
			entry.complete = true;
			return result = projected;
		};
		if (ValidateRuntimeValue(program, value, RuntimeValueType::Integer)) {
			uniform_control = true;
			capture_control_reads |=
			    !ValidateRuntimeValue(program, value, RuntimeValueType::ImmutableInteger);
			return finish(Clone(value));
		}
		const auto op = inst->GetOpcode();
		if (op == ValueOpcode::ConditionRef) return finish(ClonePredicate(inst->Arg(0)));
		if (op == ValueOpcode::Phi) {
			if (inst->Parent() == nullptr || inst->Parent()->terminator.loop_header ||
			    inst->NumPhiBlocks() != inst->NumArgs() || inst->NumArgs() == 0u ||
			    (value.GetType() != Type::U1 && value.GetType() != Type::U32)) return finish(opaque());
			std::vector<uint32_t> predecessors;
			for (size_t index = 0; index < inst->NumArgs(); ++index) {
				const auto predecessor = std::ranges::find(program.blocks, inst->PhiBlock(index));
				if (predecessor == program.blocks.end()) return finish(opaque());
				predecessors.push_back(static_cast<uint32_t>(predecessor - program.blocks.begin()));
			}
			auto& node = plan.value_storage.emplace_back(op, inst->Flags<uint64_t>());
			plan.control_phi_predecessors.emplace(&node, std::move(predecessors));
			for (size_t index = 0; index < inst->NumArgs(); ++index)
				node.AddPhiOperand(nullptr, ClonePredicate(inst->Arg(index)));
			return finish(Value(&node));
		}
		if (op != ValueOpcode::LogicalAnd && op != ValueOpcode::LogicalOr &&
		    op != ValueOpcode::LogicalNot && op != ValueOpcode::IEqual32 &&
		    op != ValueOpcode::INotEqual32 && op != ValueOpcode::ULessThan32 &&
		    op != ValueOpcode::ULessThanEqual32 &&
		    op != ValueOpcode::SelectU1 && op != ValueOpcode::SelectU32) return finish(opaque());
		auto& node = plan.value_storage.emplace_back(op);
		for (size_t index = 0; index < inst->NumArgs(); ++index)
			node.SetArg(index, ClonePredicate(inst->Arg(index)));
		return finish(Value(&node));
	};
	plan.control_flow = ResourceControlFlow(program, ClonePredicate);
	if (!uniform_control) {
		plan.control_flow.clear();
		plan.control_phi_predecessors.clear();
		auto first = std::next(plan.value_storage.begin(), control_value_count);
		for (auto it = first; it != plan.value_storage.end(); ++it) it->Invalidate();
		plan.value_storage.erase(first, plan.value_storage.end());
	}
	plan.uniform_fill = program.uniform_fill.fill.kind != UniformFillKind::None
	    ? program.uniform_fill : AnalyzeUniformFill(program);
	for (auto& value: plan.uniform_fill.values) value = Clone(value);
	plan.uniform_fill.pattern_size = Clone(plan.uniform_fill.pattern_size);
	plan.uniform_fill.element_count = Clone(plan.uniform_fill.element_count);
	plan.capture_specialization_reads |=
	    capture_indirect_reads || capture_control_reads || !plan.source_reads.empty();
	if (capture_indirect_reads || !plan.source_reads.empty())
		plan.resource_tracking_complete &= !program.has_address_writes;
	DerivePlanState(plan);
	return plan;
}

bool ReadsStrideAtRunTime(const ResourcePlan& program, uint32_t index,
                          const ResourceSpecialization::Buffer& buffer) {
	if (index >= program.info.buffers.size() || index >= program.buffer_vector_access.size() ||
	    !program.buffer_vector_access[index] || buffer.zero_stride_oob ||
	    (buffer.packed_stride & (1u << 14u)) != 0u) {
		return false;
	}
	const auto source = program.info.buffers[index].source;
	return source < program.descriptor_sources.size() &&
	       !program.descriptor_sources[source].indirect_descriptor.has_value();
}

void ReadStridesAtRunTime(const ResourcePlan& program, ResourceSpecialization& specialization) {
	for (uint32_t i = 0; i < specialization.buffers.size(); i++) {
		if (ReadsStrideAtRunTime(program, i, specialization.buffers[i])) {
			specialization.buffers[i].packed_stride |= RuntimeBufferStride;
		}
	}
}

#if defined(ANYPS5_KYTY)
bool SharesEmptyBuffers() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SHARE_EMPTY_BUFFERS");
		return value == nullptr || std::string_view(value) != "0";
	}();
	return on;
}

static bool DirectBufferSlot(const ResourcePlan& program, uint32_t index,
                             const ResourceSpecialization::Buffer& buffer) {
	if (index >= program.info.buffers.size() || buffer.indirect_root != BufferResource::NoIndirectBuffer) {
		return false;
	}
	const auto& base = program.info.buffers[index];
	return base.source < program.descriptor_sources.size() &&
	       !program.descriptor_sources[base.source].indirect_descriptor.has_value() &&
	       base.indirect_root == BufferResource::NoIndirectBuffer && base.indirect_resources.empty();
}

static void ShareEmptyBuffer(const ResourcePlan& program, uint32_t index, bool runtime_stride,
                             ResourceSpecialization::Buffer& buffer) {
	if (!DirectBufferSlot(program, index, buffer)) return;
	const bool vector = index < program.buffer_vector_access.size() && program.buffer_vector_access[index];
	if (!vector) {
		buffer.scalar_empty = false;
	} else if (runtime_stride && !program.info.buffers[index].formatted) {
		buffer.packed_stride   = RuntimeBufferStride;
		buffer.zero_stride_oob = false;
		buffer.scalar_empty    = false;
	} else {
		buffer.scalar_empty = true;
	}
}

void ShareNullBufferVariants(const ResourcePlan& program, ResourceSpecialization& specialization,
                             bool runtime_stride) {
	if (!SharesEmptyBuffers()) return;
	for (uint32_t i = 0; i < specialization.buffers.size(); i++) {
		auto& buffer = specialization.buffers[i];
		if (buffer.scalar_empty || (buffer.zero_stride_oob && buffer.packed_stride == 0u)) {
			ShareEmptyBuffer(program, i, runtime_stride, buffer);
		}
	}
}

uint32_t FormattedReadComponents(const ResourcePlan& program, uint32_t index) {
	uint32_t mask = 0;
	for (const auto& memory: program.memory_info) {
		if (memory.kind != ResourceKind::Buffer || memory.resource != index || !memory.formatted ||
		    memory.typed) {
			continue;
		}
		const auto count = std::min(memory.data_bits == 16u ? memory.component_count : memory.data_dwords, 4u);
		mask |= (1u << count) - 1u;
	}
	return mask;
}

std::optional<uint32_t> EmptyBufferReadBits(const ResourceSpecialization::Buffer& buffer, uint32_t component) {
	const auto selector = GetDstSel(buffer.descriptor_swizzle, component);
	const auto format   = Format::GetFormatInfo(buffer.descriptor_format);
	if (buffer.zero_stride_oob) {
		return format.type != Format::ComponentType::Unknown && selector == 1u
		           ? Format::FormattedConstantBits(format, Format::FormattedSourceKind::One)
		           : 0u;
	}
	const auto source = Format::ResolveFormattedSource(format, selector);
	if (source.kind == Format::FormattedSourceKind::Invalid) return std::nullopt;
	return Format::FormattedConstantBits(format, source.kind);
}

bool SpecializedBuffersValid(const ResourceSpecialization& specialization, bool compute) {
	return std::ranges::none_of(specialization.buffers, [&](const ResourceSpecialization::Buffer& buffer) {
		if (!compute && (buffer.packed_stride & (1u << 20u)) != 0u) return true;
		for (uint32_t component = 0; component < 4u; component++) {
			const auto selector = GetDstSel(buffer.descriptor_swizzle, component);
			if (selector == 2u || selector == 3u) return true;
		}
		return false;
	});
}

bool EmptyBufferServes(const ResourcePlan& program, const ResourceSnapshot& snapshot, uint32_t index,
                       const ResourceSpecialization::Buffer& requested,
                       const ResourceSpecialization::Buffer& candidate) {
	if (!SharesEmptyBuffers() || index >= snapshot.buffers.size() || snapshot.buffers[index].dword_count != 4u ||
	    snapshot.buffers[index].dwords[2] != 0u ||
	    !DirectBufferSlot(program, index, requested) || !DirectBufferSlot(program, index, candidate) ||
	    requested.indirect_mapping_offset != candidate.indirect_mapping_offset ||
	    requested.indirect_search_iterations != candidate.indirect_search_iterations) {
		return false;
	}
	const auto components = FormattedReadComponents(program, index);
	for (uint32_t component = 0; component < 4u; component++) {
		if (((components >> component) & 1u) == 0u) continue;
		const auto wanted = EmptyBufferReadBits(requested, component);
		const auto served = EmptyBufferReadBits(candidate, component);
		if (!wanted || !served || *wanted != *served) return false;
	}
	return true;
}

bool SameModuleCode(const ResourcePlan& program, const ResourceSpecialization& requested,
                    const ResourceSpecialization& candidate) {
	if (requested.buffers.size() != candidate.buffers.size() || requested.images != candidate.images ||
	    requested.samplers != candidate.samplers) {
		return false;
	}
	for (uint32_t i = 0; i < requested.buffers.size(); i++) {
		const auto& wanted = requested.buffers[i];
		const auto& served = candidate.buffers[i];
		if (wanted == served) continue;
		if (i >= program.info.buffers.size() || program.info.buffers[i].scalar || wanted.scalar_empty == served.scalar_empty) {
			return false;
		}
		auto unmarked         = wanted;
		unmarked.scalar_empty = served.scalar_empty;
		if (!(unmarked == served)) return false;
	}
	return true;
}

bool EmptyBuffersServe(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                       const ResourceSpecialization& requested, const ResourceSpecialization& candidate) {
	if (requested.buffers.size() != candidate.buffers.size() || requested.images != candidate.images ||
	    requested.samplers != candidate.samplers) {
		return false;
	}
	bool served = false;
	for (uint32_t i = 0; i < requested.buffers.size(); i++) {
		if (requested.buffers[i] == candidate.buffers[i]) continue;
		if (!EmptyBufferServes(program, snapshot, i, requested.buffers[i], candidate.buffers[i])) return false;
		served = true;
	}
	return served;
}
#endif

bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization) {
	t_record_blocks.clear();
	for (auto& keys: PlanScratch(program).selector_keys) keys.resolved = false;
	if (!program.resource_tracking_complete) return false;
	const auto uses_workgroup_count = [](const DescriptorSource& source) {
		return source.indirect_descriptor && source.indirect_descriptor->workgroup_axis != UINT32_MAX;
	};
	const bool capture_reads = program.capture_specialization_reads || !program.source_reads.empty() ||
	    (runtime.indirect_args_address != 0u &&
	     std::ranges::any_of(program.descriptor_sources, uses_workgroup_count));
	auto& reads = snapshot.specialization_reads;
	reads = program.source_reads;
	if (!WalkSrt(program, runtime, snapshot.flattened_srt, capture_reads ? &reads : nullptr)) {
		if (runtime.record_load != nullptr) return false;
		return SpecializationFail(fmt::format("SRT refresh complete={} reads={} user_data={}",
		    program.srt_plan_complete, program.srt_reads.size(), runtime.user_data.size()));
	}
	snapshot.flattened_srt.insert(snapshot.flattened_srt.end(), runtime.call_words.begin(),
	                              runtime.call_words.end());
	SrtReadCapture capture {runtime, reads};
	auto observed = capture_reads ? capture.ObservedRuntime() : runtime;
	std::array<uint32_t, 3> workgroup_counts {};
	if (runtime.indirect_args_address != 0u) {
		for (uint32_t source = 0; source < program.descriptor_sources.size(); ++source) {
			if (!WalkedActive(program, source) ||
			    !uses_workgroup_count(program.descriptor_sources[source])) continue;
			if ((runtime.indirect_args_address & 3u) != 0u ||
			    !ReadTable(runtime.indirect_args_address, sizeof(workgroup_counts), 0u,
			               observed, workgroup_counts)) return false;
			observed.workgroup_counts = workgroup_counts;
			break;
		}
	}
	snapshot.uniform_fill = {};
	const auto& fill = program.uniform_fill;
	if (fill.fill.kind == UniformFillKind::Replicate) {
		uint32_t count = 0;
		uint32_t size  = 0;
		if (WalkedValue(program, fill.element_count, count) &&
		    WalkedValue(program, fill.pattern_size, size) && count != 0 && size != 0) {
			snapshot.uniform_fill                  = fill.fill;
			snapshot.uniform_fill.bounded_elements = count;
			snapshot.uniform_fill.pattern_elements = size;
		}
	}
	auto words = fill.fill.kind == UniformFillKind::Replicate ? 0u : fill.fill.words;
	std::array<uint32_t, 4> stored {};
	bool uniform_fill = words != 0;
	if (uniform_fill && !fill.pattern_size.IsEmpty())
		uniform_fill = WalkedValue(program, fill.pattern_size, words) && words >= 1 && words <= 4;
	uint32_t elements = 0;
	if (uniform_fill && !fill.element_count.IsEmpty())
		uniform_fill = WalkedValue(program, fill.element_count, elements) && elements != 0;
	for (uint32_t i = 0; i < words && uniform_fill; ++i) {
		uniform_fill = WalkedValue(program, fill.values[i], stored[i]) && stored[i] == stored[0];
	}
	if (uniform_fill) {
		snapshot.uniform_fill = fill.fill;
		snapshot.uniform_fill.value = stored[0];
		snapshot.uniform_fill.bounded_elements = elements;
	}
	const auto program_failed = [&](std::string_view message) {
		return SpecializationFail(fmt::format("{} hash=0x{:016x} stage={} shader_base=0x{:x}", message,
		    program.shader_hash, static_cast<uint32_t>(program.stage), runtime.shader_base));
	};
	const auto evaluate = [&](uint32_t source, DescriptorValue& value) {
		if (source >= program.descriptor_sources.size()) {
			return false;
		}
		if (WalkedActive(program, source)) {
			return WalkedDescriptor(program, source, value) ||
			    program_failed(fmt::format("descriptor evaluation source={} dwords={}", source,
			        program.descriptor_sources[source].dword_count));
		}
		value = {};
		value.dword_count = program.descriptor_sources[source].dword_count;
		return true;
	};
	const auto resource_failed = [&](std::string_view kind, uint32_t resource, uint32_t source,
	                                 size_t candidates) {
		return program_failed(fmt::format("indirect {} resource={} source={} candidates={}",
		    kind, resource, source, candidates));
	};
	snapshot.buffers.resize(program.info.buffers.size());
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto base_index =
		    i < program.info.buffers.size() ? i : specialization.buffers[i].indirect_root;
		if (base_index >= program.info.buffers.size()) return false;
		const auto& base = program.info.buffers[base_index];
		if (i < program.info.buffers.size()) {
			const auto* source = Source(program, base.source);
			if (source == nullptr) return false;
			if (source->indirect_descriptor.has_value()) {
				snapshot.buffers[i] = {.dword_count = 4u};
				if (WalkedActive(program, base.source)) {
					if (!MaterializeIndirectDescriptor(
					        program, base.source, i, 4u, 4u, observed, snapshot,
					        snapshot.buffers, specialization.buffers,
					        [&](DescriptorValue& value) {
						        return NormalizeIndirectBuffer(
						            value, source->indirect_descriptor->indexed, base);
					        }))
						return resource_failed("buffer", i, base.source, snapshot.buffers.size());
				}
			} else if (!evaluate(base.source, snapshot.buffers[i])) {
				return false;
			}
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) {
			return SpecializationFail(fmt::format("buffer descriptor {} has invalid width", i));
		}
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		auto& buffer         = specialization.buffers[i];
		buffer.packed_stride = packed_stride;
		buffer.descriptor_format =
		    base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		buffer.scalar_empty       = base.scalar && descriptor.GetSize() == 0u;
		if (i < program.buffer_vector_access.size() && !program.buffer_vector_access[i]) {
			buffer.packed_stride   = 0u;
			buffer.zero_stride_oob = false;
		}
		if (i < program.buffer_feeds_descriptors.size() && !program.buffer_feeds_descriptors[i]) {
			buffer.scalar_empty = false;
		}
		if (runtime.runtime_stride && ReadsStrideAtRunTime(program, i, buffer)) {
			buffer.packed_stride |= RuntimeBufferStride;
		}
#if defined(ANYPS5_KYTY)
		if (descriptor.NumRecords() == 0u && SharesEmptyBuffers()) {
			ShareEmptyBuffer(program, i, runtime.runtime_stride, buffer);
		}
#endif
	}
	snapshot.images.resize(program.info.images.size());
	specialization.images.resize(program.info.images.size());
	auto previous_tables = std::move(snapshot.image_tables);
	snapshot.image_tables.clear();
	std::vector<uint32_t> table_mappings;
#if defined(ANYPS5_KYTY)
	std::vector<uint32_t> deferred_tables;
#endif
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image = program.info.images[i];
		specialization.images[i] = {
		    .numeric_class = image.numeric_class,
		    .dimension = image.dimension,
		    .mip_count = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle = image.shader_swizzle,
		    .indirect_root = image.indirect_root,
		    .indirect_mapping_offset = image.indirect_mapping_offset,
		    .indirect_search_iterations = image.indirect_search_iterations,
		    .table = image.table,
		    .table_capacity = image.table_capacity,
		    .table_mapping_offset = image.table_mapping_offset,
		    .table_entry_mask = image.table_entry_mask,
		    .cube = image.cube,
		};
		const auto* source = Source(program, image.source);
		if (source == nullptr) {
			return false;
		}
		if (source->indirect_descriptor.has_value()) {
			snapshot.images[i] = {.dword_count = 8u};
			if (!WalkedActive(program, image.source) || !WalkedImageActive(program, i)) {
				continue;
			}
			const auto& indirect = *source->indirect_descriptor;
			if (indirect.table_array) {
#if defined(ANYPS5_KYTY)
				deferred_tables.push_back(i);
#else
				DescriptorValue table;
				if (!WalkedDescriptor(program, indirect.table_source, table) ||
				    !MaterializeImageTable(program, indirect, table, i, observed, snapshot,
				                           specialization, previous_tables, table_mappings)) {
					return false;
				}
#endif
				continue;
			}
			const bool bounded_table = IsBoundedDescriptorTable(program, indirect);
			if (!MaterializeIndirectDescriptor(
			        program, image.source, i, 8u, image.r128 ? 4u : 8u, observed, snapshot, snapshot.images,
			        specialization.images, [&](DescriptorValue& value) {
				        if (NullImageDescriptor(value) || !ValidImageDescriptor(value, image.r128) ||
				            (image.sample_access_only && !image.depth_compare &&
				             DescriptorImageSwizzle(value) == 0u) ||
				            (bounded_table && DescriptorDimension(value, image.dimension) != image.dimension))
					        value.dwords.fill(0u);
				        return true;
			        })) {
				return resource_failed("image", i, image.source, snapshot.images.size());
			}
		} else {
			if (!evaluate(image.source, snapshot.images[i])) {
				return false;
			}
			if (!WalkedImageActive(program, i) ||
			    !ValidImageDescriptor(snapshot.images[i], image.r128)) {
				snapshot.images[i].dwords.fill(0);
			}
		}
	}
	snapshot.samplers.resize(program.info.samplers.size());
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < snapshot.samplers.size(); ++i) {
		const auto base_index =
		    i < program.info.samplers.size() ? i : specialization.samplers[i].indirect_root;
		if (base_index >= program.info.samplers.size()) return false;
		if (i < program.info.samplers.size()) {
			auto source = program.info.samplers[i].source;
			const auto* descriptor = Source(program, source);
			if (descriptor == nullptr) return false;
			const auto* indirect = descriptor->indirect_descriptor ? &*descriptor->indirect_descriptor : nullptr;
			if (indirect != nullptr && indirect->sources.empty()) {
				snapshot.samplers[i] = {.dword_count = 4u};
				if (WalkedActive(program, source) && !MaterializeIndirectDescriptor(
				        program, source, i, 4u, 4u, observed, snapshot, snapshot.samplers,
				        specialization.samplers,
				        [](DescriptorValue&) { return true; }))
					return resource_failed("sampler", i, source, snapshot.samplers.size());
			} else {
				if (indirect != nullptr) source = indirect->sources[0];
				if (!evaluate(source, snapshot.samplers[i])) return false;
			}
		}
		if (program.info.samplers[base_index].gather_lod) {
			const auto control = snapshot.samplers[i].dwords[2];
			const auto filter = (control >> 26u) & 3u;
			const auto perf_mip = (snapshot.samplers[i].dwords[1] >> 24u) & 15u;
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u) ||
			    (filter == 2u && perf_mip != 0u)) {
				return SpecializationFail(
				    "explicit-LOD gather requires None, unbiased Point, or unbiased Linear without mip snapping");
			}
		}
	}
#if defined(ANYPS5_KYTY)
	for (const auto i: deferred_tables) {
		const auto&     indirect = *Source(program, program.info.images[i].source)->indirect_descriptor;
		DescriptorValue table;
		if (!WalkedDescriptor(program, indirect.table_source, table)) {
			return false;
		}
		if (!MaterializeImageTableHeap(program, indirect, table, i, observed, snapshot, specialization) &&
		    !MaterializeImageTable(program, indirect, table, i, observed, snapshot, specialization,
		                           previous_tables, table_mappings)) {
			return false;
		}
	}
#endif
	FlushRecordBlocks(snapshot.flattened_srt);
	snapshot.user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	return BuildResourceSpecialization(program, snapshot, specialization);
}

bool SpecializeDirectResources(const ResourcePlan& program, ResourceSnapshot& snapshot,
                               ResourceSpecialization& specialization) {
	if (snapshot.buffers.size() != program.info.buffers.size() ||
	    snapshot.images.size() != program.info.images.size() ||
	    snapshot.samplers.size() != program.info.samplers.size()) {
		return false;
	}
	specialization.buffers.assign(program.info.buffers.size(), {});
	for (uint32_t i = 0; i < snapshot.buffers.size(); ++i) {
		const auto& base   = program.info.buffers[i];
		const auto* source = Source(program, base.source);
		if (source == nullptr || source->indirect_descriptor.has_value() ||
		    base.indirect_root != BufferResource::NoIndirectBuffer) {
			return false;
		}
		auto&                descriptor_value = snapshot.buffers[i];
		ShaderBufferResource descriptor;
		if (!DecodeBufferDescriptor(descriptor_value, descriptor)) return false;
		if (descriptor.Type() != 0) {
			descriptor_value.dwords.fill(0);
			descriptor = {};
		}
		auto       packed_stride = descriptor.PackedStride();
		const auto stride        = packed_stride & 0x3fffu;
		const bool swizzle       = stride != 0u && ((packed_stride >> 14u) & 1u) != 0u;
		if (stride == 0u) {
			packed_stride &= ~((1u << 14u) | (3u << 16u));
		} else if (!swizzle) {
			packed_stride &= ~(3u << 16u);
		}
		auto& buffer              = specialization.buffers[i];
		buffer.packed_stride      = packed_stride;
		buffer.descriptor_format  = base.formatted ? descriptor.Format() : Prospero::BufferFormat::kInvalid;
		buffer.descriptor_swizzle = base.formatted ? descriptor.DstSelXYZW() : DstSel(4, 5, 6, 7);
		buffer.zero_stride_oob    = descriptor.OutOfBounds() == 0u && stride == 0u;
		buffer.scalar_empty       = base.scalar && descriptor.GetSize() == 0u;
		if (i < program.buffer_vector_access.size() && !program.buffer_vector_access[i]) {
			buffer.packed_stride   = 0u;
			buffer.zero_stride_oob = false;
		}
		if (i < program.buffer_feeds_descriptors.size() && !program.buffer_feeds_descriptors[i]) {
			buffer.scalar_empty = false;
		}
	}
	specialization.images.resize(program.info.images.size());
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& image  = program.info.images[i];
		const auto* source = Source(program, image.source);
		if (source == nullptr || source->indirect_descriptor.has_value() ||
		    image.indirect_root != ImageResource::NoIndirectImage || image.table_capacity != 0) {
			return false;
		}
		specialization.images[i] = {
		    .numeric_class     = image.numeric_class,
		    .dimension         = image.dimension,
		    .mip_count         = image.mip_count,
		    .conversion_format = image.conversion_format,
		    .shader_swizzle    = image.shader_swizzle,
		    .cube              = image.cube,
		};
		if (!ValidImageDescriptor(snapshot.images[i], image.r128)) snapshot.images[i].dwords.fill(0);
	}
	specialization.samplers.assign(program.info.samplers.size(), {});
	for (uint32_t i = 0; i < snapshot.samplers.size(); ++i) {
		const auto& sampler = program.info.samplers[i];
		const auto* source  = Source(program, sampler.source);
		if (source == nullptr || source->indirect_descriptor.has_value()) return false;
		if (sampler.gather_lod) {
			const auto control  = snapshot.samplers[i].dwords[2];
			const auto filter   = (control >> 26u) & 3u;
			const auto perf_mip = (snapshot.samplers[i].dwords[1] >> 24u) & 15u;
			if (filter > 2u || (filter != 0u && (control & 0xfffffu) != 0u) ||
			    (filter == 2u && perf_mip != 0u)) {
				return false;
			}
		}
	}
	return BuildResourceSpecialization(program, snapshot, specialization);
}

uint32_t ImageHeapKind(const ImageResource& root, DescriptorValue& value) {
	const auto format = static_cast<Prospero::BufferFormat>((value.dwords[1] >> 20u) & 0x1ffu);
	const auto tile = static_cast<Prospero::TileMode>((value.dwords[3] >> 20u) & 0x1fu);
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (NullImageDescriptor(value) || !ValidImageDescriptor(value, root.r128) ||
	    DescriptorDimension(value, root.dimension) != root.dimension ||
	    Prospero::IsFmaskTextureFormat(format) || format == Prospero::BufferFormat::k11_11_10UNorm ||
	    ImageConversionFormat(format) != Prospero::BufferFormat::kInvalid ||
	    (root.unfiltered_access_only && format == Prospero::BufferFormat::k16UInt &&
	     tile == Prospero::TileMode::kDepth) ||
	    numeric_class == Prospero::TextureNumericClass::Unsupported ||
	    (root.depth_compare && numeric_class != Prospero::TextureNumericClass::Float)) {
		value.dwords.fill(0u);
		return 0u;
	}
	return 1u + static_cast<uint32_t>(numeric_class) * 2u + (DescriptorIsCube(value) ? 1u : 0u);
}

ImageResource ImageTableHeapProfile(const ImageResource& root, bool skip_degamma) {
	ImageResource profile;
	profile.resource_class         = ImageResourceClass::Sampled;
	profile.read                   = true;
	profile.heap                   = true;
	profile.numeric_class          = Prospero::TextureNumericClass::Float;
	profile.dimension              = root.dimension;
	profile.mip_mode               = root.mip_mode;
	profile.unfiltered_access_only = root.unfiltered_access_only;
	profile.sample_access_only     = root.sample_access_only;
	profile.depth_compare          = root.depth_compare;
	profile.skip_degamma           = skip_degamma;
	profile.cube                   = root.cube;
	profile.r128                   = root.r128;
	return profile;
}

uint32_t ImageTableHeapKind(const ImageResource& profile, DescriptorValue& value) {
	if (!ImageTableSlotCompatible(profile, value)) {
		value.dwords.fill(0u);
		return 0u;
	}
	return 1u + static_cast<uint32_t>(profile.numeric_class) * 2u + (profile.cube ? 1u : 0u);
}

bool ImageTableHeapExemplar(const ImageResource& profile, const DescriptorValue& value) {
	return TableExemplar(profile, value, true);
}

ImageResource ImageTableHeapResource(const ImageResource& profile, const DescriptorValue& value) {
	auto       resource = profile;
	const auto format = static_cast<Prospero::BufferFormat>((value.dwords[1] >> 20u) & 0x1ffu);
	resource.skip_degamma = profile.skip_degamma && Prospero::IsSrgbFormat(format);
	return resource;
}

ImageResource ImageHeapResource(const ImageResource& root, uint32_t kind) {
	ImageResource resource;
	resource.resource_class         = ImageResourceClass::Sampled;
	resource.read                   = true;
	resource.heap                   = true;
	resource.dimension              = root.dimension;
	resource.unfiltered_access_only = root.unfiltered_access_only;
	resource.depth_compare          = root.depth_compare;
	resource.r128                   = root.r128;
	if (kind != 0u) {
		resource.numeric_class = static_cast<Prospero::TextureNumericClass>((kind - 1u) / 2u);
		resource.cube          = ((kind - 1u) & 1u) != 0u;
	}
	return resource;
}

void SetComputeIndirectFloors(size_t keys, size_t images) {
	g_compute_indirect_keys.store(std::max(std::bit_ceil(keys), MinIndirectMappingCapacity),
	                              std::memory_order_relaxed);
	g_compute_indirect_images.store(std::max<size_t>(std::bit_ceil(images), 1u),
	                                std::memory_order_relaxed);
	g_stable_indirect_layouts.store(true, std::memory_order_relaxed);
}

bool ResourceSpecializationFits(const Program& program, const ResourceSpecialization& specialization) {
	const auto fits = [](size_t count, const auto& elements, uint32_t none) {
		if (count > elements.size()) return false;
		for (size_t index = 0; index < elements.size(); index++) {
			const auto root = elements[index].indirect_root;
			if (index >= count ? root >= count : root != none && root >= elements.size()) return false;
		}
		return true;
	};
	return fits(program.info.buffers.size(), specialization.buffers, BufferResource::NoIndirectBuffer) &&
	       fits(program.info.images.size(), specialization.images, ImageResource::NoIndirectImage) &&
	       fits(program.info.samplers.size(), specialization.samplers,
	            SamplerResource::NoIndirectSampler);
}

void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization) {
	EXIT_IF(!program.resource_tracking_complete || program.shader_info_complete);
	EXIT_IF(program.info.buffers.size() > specialization.buffers.size() ||
	        program.info.images.size() > specialization.images.size() ||
	        program.info.samplers.size() > specialization.samplers.size());

	auto& buffers = program.info.buffers;
	const auto original_buffer_count = buffers.size();
	buffers.reserve(specialization.buffers.size());
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& source = specialization.buffers[index];
		if (index >= buffers.size()) {
			EXIT_IF(source.indirect_root >= original_buffer_count);
			buffers.push_back(buffers[source.indirect_root]);
		}
		auto& buffer                      = buffers[index];
		buffer.packed_stride              = source.packed_stride;
		buffer.descriptor_format          = source.descriptor_format;
		buffer.descriptor_swizzle         = source.descriptor_swizzle;
		buffer.indirect_root              = source.indirect_root;
		buffer.indirect_mapping_offset    = source.indirect_mapping_offset;
		buffer.indirect_search_iterations = source.indirect_search_iterations;
		buffer.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < buffers.size(); ++index) {
		const auto root = buffers[index].indirect_root;
		if (root != BufferResource::NoIndirectBuffer) {
			EXIT_IF(root >= buffers.size());
			buffers[root].indirect_resources.push_back(index);
		}
	}
	auto& images = program.info.images;
	const auto original_image_count = images.size();
	images.reserve(specialization.images.size());
	for (uint32_t index = 0; index < specialization.images.size(); index++) {
		const auto& source = specialization.images[index];
		if (index >= images.size()) {
			EXIT_IF(source.indirect_root >= original_image_count);
			images.push_back(images[source.indirect_root]);
		}
		auto& image                      = images[index];
		image.numeric_class              = source.numeric_class;
		image.dimension                  = source.dimension;
		image.mip_count                  = source.mip_count;
		image.conversion_format          = source.conversion_format;
		image.shader_swizzle             = source.shader_swizzle;
		image.srgb_decode                = source.srgb_decode;
		image.indirect_root              = source.indirect_root;
		image.indirect_mapping_offset    = source.indirect_mapping_offset;
		image.indirect_search_iterations = source.indirect_search_iterations;
		image.table                      = source.table;
		image.table_capacity             = source.table_capacity;
		image.table_mapping_offset       = source.table_mapping_offset;
		image.table_entry_mask           = source.table_entry_mask;
		image.cube                       = source.cube;
		image.skip_degamma               = source.skip_degamma;
		image.heap                       = source.heap;
		image.indirect_resources.clear();
	}
	for (uint32_t index = 0; index < images.size(); index++) {
		const auto root = images[index].indirect_root;
		if (root != ImageResource::NoIndirectImage) {
			EXIT_IF(root >= images.size());
			images[root].indirect_resources.push_back(index);
		}
	}

	auto& samplers      = program.info.samplers;
	auto& sampled_pairs = program.info.sampled_pairs;
	const auto base_sampler_count = samplers.size();
	samplers.reserve(specialization.samplers.size());
	for (uint32_t index = 0; index < specialization.samplers.size(); ++index) {
		const auto& source = specialization.samplers[index];
		if (index >= samplers.size()) {
			EXIT_IF(source.indirect_root >= base_sampler_count);
			samplers.push_back(samplers[source.indirect_root]);
			samplers.back().indirect_resources.clear();
		}
		samplers[index].indirect_root = source.indirect_root;
		samplers[index].indirect_mapping_offset = source.indirect_mapping_offset;
		samplers[index].indirect_search_iterations = source.indirect_search_iterations;
	}
	for (uint32_t index = 0; index < samplers.size(); ++index) {
		const auto root = samplers[index].indirect_root;
		if (root != SamplerResource::NoIndirectSampler) {
			EXIT_IF(root >= samplers.size());
			samplers[root].indirect_resources.push_back(index);
		}
	}
	SamplerPlan sampler_plan;
	EXIT_IF(!BuildSamplerPlan(program.info, sampler_plan));
	const auto original_sampler_count = samplers.size();
	samplers.reserve(sampler_plan.bindings.size());
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		if (index >= original_sampler_count) {
			samplers.push_back(samplers[binding.source]);
		}
		samplers[index].snapshot_index = binding.source;
		samplers[index].force_point_filtering = binding.type == SamplerClass::Integer;
		samplers[index].integer_border        = binding.type == SamplerClass::Integer;
		samplers[index].depth_compare         = binding.type == SamplerClass::Compare;
	}
	for (uint32_t index = 0; index < sampler_plan.bindings.size(); index++) {
		const auto& binding = sampler_plan.bindings[index];
		for (auto& candidate: samplers[index].indirect_resources) {
			candidate = sampler_plan.mapping[candidate][static_cast<uint32_t>(binding.type)];
			EXIT_IF(candidate == UINT32_MAX);
		}
	}
	for (auto& pair: sampled_pairs) {
		const auto type = static_cast<uint32_t>(ClassifySampler(images[pair.image]));
		pair.sampler = sampler_plan.mapping[pair.sampler][type];
		EXIT_IF(pair.sampler == UINT32_MAX);
		samplers[pair.sampler].depth_compare |= images[pair.image].depth_compare;
		for (const auto candidate: samplers[pair.sampler].indirect_resources)
			samplers[candidate].depth_compare |= images[pair.image].depth_compare;
	}

	auto& memory_info = program.memory_info;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::ReadConstBuffer) continue;
			const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
			if (memory.kind == ResourceKind::ScalarBuffer && !memory.planning_only &&
			    specialization.buffers[memory.resource].scalar_empty &&
			    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer)
				inst.ReplaceUsesWith(Value(0u));
		}
	}
	const ImageRemap image_remap(specialization);
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto& inst = *it;
			if (inst.GetOpcode() == ValueOpcode::GetImageResource) {
				inst.SetFlags(image_remap[inst.Flags<uint32_t>()]);
				continue;
			}
			if (BufferAccessOf(inst.GetOpcode()) == BufferAccess::Read) {
				const auto& memory = memory_info[inst.Flags<MemoryFlags>().index];
				const bool native_oob = memory.kind == ResourceKind::Buffer &&
				    specialization.buffers[memory.resource].zero_stride_oob &&
				    buffers[memory.resource].indirect_root == BufferResource::NoIndirectBuffer;
				const bool null_indirect = memory.kind == ResourceKind::IndirectBuffer &&
				    NullBufferDescriptor(inst.Arg(0));
				if (native_oob || null_indirect) {
					const auto count = BufferComponentCount(inst.GetOpcode());
					std::array<Value, 4> values {Value(0u), Value(0u), Value(0u), Value(0u)};
					if (native_oob && memory.formatted && !memory.typed) {
						const auto& buffer = buffers[memory.resource];
						const auto format = Format::GetFormatInfo(buffer.descriptor_format);
						const bool d16 = memory.data_bits == 16u;
						const auto components = d16 ? memory.component_count : count;
						std::array<uint32_t, 4> constants {};
						for (uint32_t component = 0; component < components; component++) {
							if (format.type == Format::ComponentType::Unknown ||
							    GetDstSel(buffer.descriptor_swizzle, component) != 1u) continue;
							auto one = Format::FormattedConstantBits(
							    format, Format::FormattedSourceKind::One);
							if (d16 && one == 0x3f800000u) one = 0x3c00u;
							constants[d16 ? component / 2u : component] |=
							    one << (d16 ? (component % 2u) * 16u : 0u);
						}
						for (uint32_t component = 0; component < count; component++) {
							if (constants[component] == 0u) continue;
							values[component] = Value(&*block->PrependNewInst(
							    it, ValueOpcode::SelectU32,
							    {inst.Arg(inst.NumArgs() - 1), Value(constants[component]), Value(0u)}));
						}
					}
					Value result = values[0];
					switch (inst.GetType()) {
						case Type::U8: result = Value(uint8_t {0}); break;
						case Type::U16: result = Value(uint16_t {0}); break;
						case Type::U32x2:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x2,
							                                      {values[0], values[1]}));
							break;
						case Type::U32x3:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x3,
							                                      {values[0], values[1], values[2]}));
							break;
						case Type::U32x4:
							result = Value(&*block->PrependNewInst(it, ValueOpcode::CompositeConstructU32x4,
							                                      {values[0], values[1], values[2], values[3]}));
							break;
						default: break;
					}
					inst.ReplaceUsesWith(result);
				}
				continue;
			}
			const auto image_opcode = ImageOpcodeInfoOf(inst.GetOpcode());
			if (image_opcode.access == ImageAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			EXIT_IF(index >= memory_info.size());
			auto& memory = memory_info[index];
			EXIT_IF(memory.resource >= images.size());
			const auto& image = images[memory.resource];
			if (specialization.images[memory.resource].fmask) {
				EXIT_IF(inst.GetOpcode() != ValueOpcode::ImageRead || memory.data_bits != 32u);
				constexpr uint32_t indices[] = {0x76543210u, 0xfedcba98u};
				std::array<Value, 2> fragments;
				for (uint32_t component = 0; component < fragments.size(); component++) {
					const auto selected = block->PrependNewInst(
					    it, ValueOpcode::SelectU32, {inst.Arg(2), Value(indices[component]), Value(0u)});
					fragments[component] = Value(&*selected);
				}
				const auto result = block->PrependNewInst(
				    it, ValueOpcode::CompositeConstructU32x4,
				    {fragments[0], fragments[1], Value(0u), Value(0u)});
				inst.ReplaceUsesWith(Value(&*result));
				continue;
			}
			if (image_opcode.needs_sampler &&
			    memory.sampler < original_sampler_count) {
				const auto type = static_cast<uint32_t>(ClassifySampler(image));
				memory.sampler = sampler_plan.mapping[memory.sampler][type];
				EXIT_IF(memory.sampler == UINT32_MAX);
			}
			EXIT_IF(image.indirect_root == memory.resource &&
			        inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageGatherRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageQueryDimensions &&
			        inst.GetOpcode() != ValueOpcode::ImageRead &&
			        inst.GetOpcode() != ValueOpcode::ImageWrite);
			EXIT_IF(image.table_capacity != 0u && inst.GetOpcode() != ValueOpcode::ImageSampleRaw &&
			        inst.GetOpcode() != ValueOpcode::ImageQueryDimensions);
			memory.resource = image_remap[memory.resource];
		}
	}
	for (auto& pair: sampled_pairs) {
		pair.image = image_remap[pair.image];
	}
	for (auto& image: images) {
		if (image.indirect_root != ImageResource::NoIndirectImage) {
			image.indirect_root = image_remap[image.indirect_root];
		}
		for (auto& resource: image.indirect_resources) {
			resource = image_remap[resource];
		}
	}
	image_remap.Apply(images);
}

bool LoadResourcePlan(std::span<const uint8_t> bytes, ResourcePlan& plan) {
	return DecodeResourcePlan(bytes, plan) && DerivePlanState(plan);
}

#if defined(ANYPS5_KYTY)
bool GenericComputeVariants() {
	return Common::RuntimeSwitches::On(Common::RuntimeSwitches::Switch::GenericComputeVariant);
}

namespace {

enum class SlotLayout : uint8_t { Plain, Search, Heap };

struct RootShape {
	SlotLayout            layout     = SlotLayout::Plain;
	uint32_t              offset     = 0;
	uint32_t              words      = 0;
	uint32_t              iterations = 0;
	std::vector<uint32_t> children;
};

struct SrtRegion {
	uint32_t offset = 0;
	uint32_t words  = 0;
};

constexpr uint32_t MaxShapeIterations = 24u;

ResourceSpecialization::Image SlotForm(ResourceSpecialization::Image image) {
	image.indirect_root              = 0;
	image.indirect_mapping_offset    = 0;
	image.indirect_search_iterations = 0;
	return image;
}

ResourceSpecialization::Sampler SlotForm(ResourceSpecialization::Sampler sampler) {
	sampler.indirect_root              = 0;
	sampler.indirect_mapping_offset    = 0;
	sampler.indirect_search_iterations = 0;
	return sampler;
}

uint32_t SearchWords(uint32_t iterations) {
	return 1u + 2u * (1u << (iterations - 1u));
}

uint32_t RoomFor(uint32_t count) {
	return static_cast<uint32_t>(std::bit_ceil(uint64_t {count} * 2u + 1u) - 1u);
}

bool SourceRelocates(const ResourcePlan& program, uint32_t source, bool& deferred) {
	const auto* descriptor = Source(program, source);
	deferred               = false;
	if (descriptor == nullptr) return false;
	if (!descriptor->indirect_descriptor) return true;
	const auto& indirect = *descriptor->indirect_descriptor;
	if (indirect.pointer_table && indirect.pointer_table->buffer) return false;
	deferred = indirect.table_array;
	return true;
}

bool ShapeRoot(RootShape& shape, uint32_t offset, uint32_t iterations, bool heap) {
	shape.offset     = offset;
	shape.iterations = iterations;
	if (heap) {
		if (iterations != 0u) return false;
		shape.layout = SlotLayout::Heap;
		shape.words  = 3u;
		return true;
	}
	if (iterations == 0u || iterations > MaxShapeIterations) return false;
	shape.layout = SlotLayout::Search;
	shape.words  = SearchWords(iterations);
	return true;
}

bool ImageShapes(const ResourcePlan& program, const std::vector<ResourceSpecialization::Image>& images,
                 std::vector<RootShape>& shapes, std::vector<uint8_t>& deferred) {
	const auto base = program.info.images.size();
	if (images.size() < base) return false;
	shapes.assign(base, {});
	deferred.assign(base, 0u);
	for (uint32_t index = 0; index < images.size(); ++index) {
		const auto& image = images[index];
		if (image.fmask || image.table != ImageResource::NoImageTable || image.table_capacity != 0u) return false;
		if (index >= base) {
			if (image.indirect_root >= base || shapes[image.indirect_root].layout == SlotLayout::Plain) return false;
			shapes[image.indirect_root].children.push_back(index);
			continue;
		}
		if (image.indirect_root == ImageResource::NoIndirectImage) {
			if (image.heap) return false;
			continue;
		}
		bool table = false;
		if (image.indirect_root != index || !SourceRelocates(program, program.info.images[index].source, table) ||
		    !ShapeRoot(shapes[index], image.indirect_mapping_offset, image.indirect_search_iterations, image.heap) ||
		    (table && !image.heap)) {
			return false;
		}
		deferred[index] = table ? 1u : 0u;
	}
	return std::ranges::none_of(shapes, [](const RootShape& shape) {
		return shape.layout != SlotLayout::Plain && shape.children.empty();
	});
}

bool SamplerShapes(const ResourcePlan& program, const std::vector<ResourceSpecialization::Sampler>& samplers,
                   std::vector<RootShape>& shapes) {
	const auto base = program.info.samplers.size();
	if (samplers.size() < base) return false;
	shapes.assign(base, {});
	for (uint32_t index = 0; index < samplers.size(); ++index) {
		const auto& sampler = samplers[index];
		if (index >= base) {
			if (sampler.indirect_root >= base || shapes[sampler.indirect_root].layout == SlotLayout::Plain) return false;
			shapes[sampler.indirect_root].children.push_back(index);
			continue;
		}
		if (sampler.indirect_root == SamplerResource::NoIndirectSampler) continue;
		bool table = false;
		if (sampler.indirect_root != index || !SourceRelocates(program, program.info.samplers[index].source, table) ||
		    table || !ShapeRoot(shapes[index], sampler.indirect_mapping_offset, sampler.indirect_search_iterations, false)) {
			return false;
		}
	}
	return std::ranges::none_of(shapes, [](const RootShape& shape) {
		return shape.layout != SlotLayout::Plain && shape.children.empty();
	});
}

void AddRegions(const std::vector<RootShape>& shapes, std::vector<SrtRegion>& regions) {
	for (const auto& shape: shapes) {
		if (shape.layout != SlotLayout::Plain) regions.push_back({shape.offset, shape.words});
	}
}

bool RegionsTile(std::vector<SrtRegion> regions, size_t end, size_t& prefix) {
	std::ranges::sort(regions, {}, &SrtRegion::offset);
	size_t total = 0;
	for (const auto& region: regions) total += region.words;
	if (total > end) return false;
	prefix      = end - total;
	auto cursor = prefix;
	for (const auto& region: regions) {
		if (region.offset != cursor) return false;
		cursor += region.words;
	}
	return true;
}

bool RegionsFollow(std::vector<SrtRegion> regions, size_t prefix, size_t& end) {
	std::ranges::sort(regions, {}, &SrtRegion::offset);
	end = prefix;
	if (!regions.empty() && regions.front().offset != prefix) return false;
	for (const auto& region: regions) {
		if (region.offset < end) return false;
		end = size_t {region.offset} + region.words;
	}
	return true;
}

template <typename Slot>
bool PlaceSearch(const RootShape& from, const RootShape& to, const std::vector<Slot>& wanted,
                 const std::vector<Slot>& served, std::span<const DescriptorValue> descriptors,
                 std::span<const uint32_t> source, std::vector<DescriptorValue>* placed,
                 std::vector<uint32_t>* flat) {
	if (from.layout == SlotLayout::Plain) return true;
	const auto count = source[from.offset];
	if (count > (from.words - 1u) / 2u || count > (to.words - 1u) / 2u) return false;
	std::vector<uint32_t> remap(from.children.size() + 1u, 0u);
	std::vector<uint8_t>  taken(to.children.size(), 0u);
	for (uint32_t entry = 0; entry < count; ++entry) {
		const auto ordinal = source[from.offset + 2u + entry * 2u];
		if (ordinal > from.children.size()) return false;
		if (ordinal == 0u || remap[ordinal] != 0u) continue;
		const auto form     = SlotForm(wanted[from.children[ordinal - 1u]]);
		size_t     position = 0;
		while (position < to.children.size() &&
		       (taken[position] != 0u || !(SlotForm(served[to.children[position]]) == form))) {
			++position;
		}
		if (position == to.children.size()) return false;
		taken[position] = 1u;
		remap[ordinal]  = static_cast<uint32_t>(position + 1u);
		if (placed != nullptr) (*placed)[to.children[position]] = descriptors[from.children[ordinal - 1u]];
	}
	if (flat != nullptr) {
		(*flat)[to.offset] = count;
		for (uint32_t entry = 0; entry < count; ++entry) {
			(*flat)[to.offset + 1u + entry * 2u] = source[from.offset + 1u + entry * 2u];
			(*flat)[to.offset + 2u + entry * 2u] = remap[source[from.offset + 2u + entry * 2u]];
		}
	}
	return true;
}

template <typename Slot>
bool PlaceHeap(const RootShape& from, const RootShape& to, const std::vector<Slot>& wanted,
               const std::vector<Slot>& served, std::span<const DescriptorValue> descriptors,
               std::span<const uint32_t> source, std::vector<DescriptorValue>* placed,
               std::vector<uint32_t>* flat) {
	if (from.children.size() > to.children.size()) return false;
	for (size_t arm = 0; arm < from.children.size(); ++arm) {
		if (!(SlotForm(wanted[from.children[arm]]) == SlotForm(served[to.children[arm]]))) return false;
		if (placed != nullptr) (*placed)[to.children[arm]] = descriptors[from.children[arm]];
	}
	if (flat != nullptr) {
		std::copy_n(source.begin() + from.offset, 3u, flat->begin() + to.offset);
	}
	return true;
}

template <typename Slot>
bool PlaceRoots(const std::vector<RootShape>& from, const std::vector<RootShape>& to,
                const std::vector<Slot>& wanted, const std::vector<Slot>& served,
                std::span<const DescriptorValue> descriptors, std::span<const uint32_t> source,
                std::vector<DescriptorValue>* placed, std::vector<uint32_t>* flat) {
	for (uint32_t root = 0; root < from.size(); ++root) {
		if (!(SlotForm(wanted[root]) == SlotForm(served[root]))) return false;
		if (placed != nullptr) {
			(*placed)[root] = descriptors[root];
			for (const auto child: to[root].children) (*placed)[child] = {.dword_count = descriptors[root].dword_count};
		}
		const auto& origin = from[root];
		const auto& target = to[root];
		if (target.layout == SlotLayout::Plain) {
			if (origin.layout != SlotLayout::Plain) return false;
		} else if (target.layout == SlotLayout::Search) {
			if (origin.layout == SlotLayout::Heap ||
			    !PlaceSearch(origin, target, wanted, served, descriptors, source, placed, flat)) {
				return false;
			}
		} else if (origin.layout != SlotLayout::Heap ||
		           !PlaceHeap(origin, target, wanted, served, descriptors, source, placed, flat)) {
			return false;
		}
	}
	return true;
}

template <typename Slot>
struct RootPlan {
	SlotLayout                                layout     = SlotLayout::Plain;
	uint32_t                                  iterations = 0;
	std::vector<std::pair<Slot, uint32_t>>    groups;
};

template <typename Slot>
RootPlan<Slot> PlanOf(const RootShape& shape, const std::vector<Slot>& slots) {
	RootPlan<Slot> plan {.layout = shape.layout, .iterations = shape.iterations};
	for (const auto child: shape.children) {
		const auto form = SlotForm(slots[child]);
		if (shape.layout == SlotLayout::Search) {
			const auto found = std::ranges::find_if(plan.groups, [&](const auto& group) {
				return SlotForm(group.first) == form;
			});
			if (found != plan.groups.end()) {
				++found->second;
				continue;
			}
		}
		plan.groups.emplace_back(slots[child], 1u);
	}
	return plan;
}

template <typename Slot>
void MergePlan(RootPlan<Slot>& plan, const RootPlan<Slot>& other) {
	if (other.layout == SlotLayout::Search &&
	    (plan.layout == SlotLayout::Plain || plan.layout == SlotLayout::Search)) {
		plan.layout     = SlotLayout::Search;
		plan.iterations = std::max(plan.iterations, other.iterations);
		for (const auto& [slot, count]: other.groups) {
			const auto found = std::ranges::find_if(plan.groups, [&](const auto& group) {
				return SlotForm(group.first) == SlotForm(slot);
			});
			if (found == plan.groups.end()) {
				plan.groups.emplace_back(slot, count);
			} else {
				found->second = std::max(found->second, count);
			}
		}
		return;
	}
	if (other.layout != SlotLayout::Heap || plan.layout != SlotLayout::Heap ||
	    other.groups.size() <= plan.groups.size()) {
		return;
	}
	for (size_t arm = 0; arm < plan.groups.size(); ++arm) {
		if (!(SlotForm(plan.groups[arm].first) == SlotForm(other.groups[arm].first))) return;
	}
	plan.groups = other.groups;
}

template <typename Slot>
uint32_t KnownCount(const std::vector<RootPlan<Slot>>& known, const Slot& form) {
	uint32_t count = 0;
	for (const auto& plan: known) {
		if (plan.layout != SlotLayout::Search) continue;
		for (const auto& [slot, slots]: plan.groups) {
			if (SlotForm(slot) == SlotForm(form)) count = std::max(count, slots);
		}
	}
	return count;
}

template <typename Slot>
void PlaceRoot(uint32_t root, const RootPlan<Slot>& plan, const Slot& form, std::vector<Slot>& slots,
               uint32_t& cursor) {
	auto& base = slots[root];
	base       = form;
	if (plan.layout == SlotLayout::Plain) return;
	base.indirect_root           = root;
	base.indirect_mapping_offset = cursor;
	if (plan.layout == SlotLayout::Search) {
		base.indirect_search_iterations = plan.iterations;
		cursor += SearchWords(plan.iterations);
	} else {
		base.indirect_search_iterations = 0u;
		cursor += 3u;
	}
	const auto root_slot = base;
	for (const auto& [slot, count]: plan.groups) {
		auto child          = slot;
		child.indirect_root = root;
		if (plan.layout == SlotLayout::Heap) {
			child.indirect_mapping_offset    = root_slot.indirect_mapping_offset;
			child.indirect_search_iterations = 0u;
		}
		for (uint32_t copy = 0; copy < count; ++copy) slots.push_back(child);
	}
}

}

bool AlignResources(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                    const ResourceSpecialization& specialization, const ResourceSpecialization& target,
                    ResourceSnapshot* aligned) {
	if (!GenericComputeVariants() || program.stage != ShaderType::Compute ||
	    !g_stable_indirect_layouts.load(std::memory_order_relaxed) || !snapshot.image_tables.empty() ||
	    specialization.buffers.size() != target.buffers.size() ||
	    snapshot.buffers.size() != specialization.buffers.size() ||
	    snapshot.images.size() != specialization.images.size() ||
	    snapshot.samplers.size() != specialization.samplers.size()) {
		return false;
	}
	for (uint32_t index = 0; index < specialization.buffers.size(); ++index) {
		const auto& wanted = specialization.buffers[index];
		const auto& served = target.buffers[index];
		if (wanted.indirect_root != BufferResource::NoIndirectBuffer ||
		    served.indirect_root != BufferResource::NoIndirectBuffer ||
		    (!(wanted == served) && !EmptyBufferServes(program, snapshot, index, wanted, served))) {
			return false;
		}
	}
	std::vector<RootShape> images_from, images_to, samplers_from, samplers_to;
	std::vector<uint8_t>   deferred_from, deferred_to;
	if (!ImageShapes(program, specialization.images, images_from, deferred_from) ||
	    !ImageShapes(program, target.images, images_to, deferred_to) || deferred_from != deferred_to ||
	    !SamplerShapes(program, specialization.samplers, samplers_from) ||
	    !SamplerShapes(program, target.samplers, samplers_to)) {
		return false;
	}
	std::vector<SrtRegion> regions_from, regions_to;
	AddRegions(images_from, regions_from);
	AddRegions(samplers_from, regions_from);
	AddRegions(images_to, regions_to);
	AddRegions(samplers_to, regions_to);
	size_t prefix = 0;
	size_t end    = 0;
	if (!RegionsTile(regions_from, snapshot.flattened_srt.size(), prefix) ||
	    !RegionsFollow(regions_to, prefix, end) ||
	    std::ranges::any_of(snapshot.gpu_flat_loads, [&](const GpuFlatLoad& load) {
		    return size_t {load.slot} + load.count > prefix;
	    })) {
		return false;
	}
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flat;
	if (aligned != nullptr) {
		images.assign(target.images.size(), {});
		samplers.assign(target.samplers.size(), {});
		flat.assign(end, 0u);
		std::copy_n(snapshot.flattened_srt.begin(), prefix, flat.begin());
	}
	const std::span<const uint32_t> source(snapshot.flattened_srt);
	if (!PlaceRoots(images_from, images_to, specialization.images, target.images,
	                std::span<const DescriptorValue>(snapshot.images), source,
	                aligned != nullptr ? &images : nullptr, aligned != nullptr ? &flat : nullptr) ||
	    !PlaceRoots(samplers_from, samplers_to, specialization.samplers, target.samplers,
	                std::span<const DescriptorValue>(snapshot.samplers), source,
	                aligned != nullptr ? &samplers : nullptr, aligned != nullptr ? &flat : nullptr)) {
		return false;
	}
	if (aligned != nullptr) {
		aligned->buffers              = snapshot.buffers;
		aligned->images               = std::move(images);
		aligned->samplers             = std::move(samplers);
		aligned->flattened_srt        = std::move(flat);
		aligned->image_tables         = {};
		aligned->user_data            = snapshot.user_data;
		aligned->specialization_reads = snapshot.specialization_reads;
		aligned->uniform_fill         = snapshot.uniform_fill;
		aligned->gpu_flat_loads       = snapshot.gpu_flat_loads;
	}
	return true;
}

bool WidenSpecialization(const ResourcePlan& program, const ResourceSnapshot& snapshot,
                         const ResourceSpecialization&                  specialization,
                         std::span<const ResourceSpecialization* const> known, ResourceSpecialization& widened) {
	if (!GenericComputeVariants() || program.stage != ShaderType::Compute ||
	    !g_stable_indirect_layouts.load(std::memory_order_relaxed)) {
		return false;
	}
	std::vector<RootShape> images, samplers;
	std::vector<uint8_t>   deferred;
	if (!ImageShapes(program, specialization.images, images, deferred) ||
	    !SamplerShapes(program, specialization.samplers, samplers)) {
		return false;
	}
	std::vector<SrtRegion> regions;
	AddRegions(images, regions);
	AddRegions(samplers, regions);
	size_t prefix = 0;
	if (!RegionsTile(regions, snapshot.flattened_srt.size(), prefix)) return false;
	using Image   = ResourceSpecialization::Image;
	using Sampler = ResourceSpecialization::Sampler;
	std::vector<RootPlan<Image>>   image_plans;
	std::vector<RootPlan<Sampler>> sampler_plans;
	for (uint32_t root = 0; root < images.size(); ++root) image_plans.push_back(PlanOf(images[root], specialization.images));
	for (uint32_t root = 0; root < samplers.size(); ++root) {
		sampler_plans.push_back(PlanOf(samplers[root], specialization.samplers));
	}
	std::vector<Image>   image_forms(specialization.images.begin(),
	                                 specialization.images.begin() + static_cast<std::ptrdiff_t>(images.size()));
	std::vector<Sampler> sampler_forms(specialization.samplers.begin(),
	                                   specialization.samplers.begin() + static_cast<std::ptrdiff_t>(samplers.size()));
	std::vector<std::vector<RootPlan<Image>>>   known_images(images.size());
	std::vector<std::vector<RootPlan<Sampler>>> known_samplers(samplers.size());
	for (const auto* other: known) {
		std::vector<RootShape> other_images, other_samplers;
		std::vector<uint8_t>   other_deferred;
		if (other == nullptr || !ImageShapes(program, other->images, other_images, other_deferred) ||
		    other_deferred != deferred || !SamplerShapes(program, other->samplers, other_samplers)) {
			continue;
		}
		for (uint32_t root = 0; root < images.size(); ++root) {
			if (!(SlotForm(other->images[root]) == SlotForm(specialization.images[root]))) continue;
			auto plan = PlanOf(other_images[root], other->images);
			if (image_plans[root].layout == SlotLayout::Plain && plan.layout == SlotLayout::Search) {
				image_forms[root] = other->images[root];
			}
			MergePlan(image_plans[root], plan);
			known_images[root].push_back(std::move(plan));
		}
		for (uint32_t root = 0; root < samplers.size(); ++root) {
			if (!(SlotForm(other->samplers[root]) == SlotForm(specialization.samplers[root]))) continue;
			auto plan = PlanOf(other_samplers[root], other->samplers);
			if (sampler_plans[root].layout == SlotLayout::Plain && plan.layout == SlotLayout::Search) {
				sampler_forms[root] = other->samplers[root];
			}
			MergePlan(sampler_plans[root], plan);
			known_samplers[root].push_back(std::move(plan));
		}
	}
	const auto make_room = [](auto& plans, const auto& shapes, const auto& slots, const auto& known_plans) {
		for (uint32_t root = 0; root < plans.size(); ++root) {
			if (plans[root].layout != SlotLayout::Search || shapes[root].layout != SlotLayout::Search) continue;
			const auto natural = PlanOf(shapes[root], slots);
			for (const auto& [slot, count]: natural.groups) {
				if (count <= KnownCount(known_plans[root], slot)) continue;
				for (auto& group: plans[root].groups) {
					if (SlotForm(group.first) == SlotForm(slot)) group.second = std::max(group.second, RoomFor(count));
				}
			}
		}
	};
	make_room(image_plans, images, specialization.images, known_images);
	make_room(sampler_plans, samplers, specialization.samplers, known_samplers);
	widened = specialization;
	widened.images.resize(images.size());
	widened.samplers.resize(samplers.size());
	auto cursor = static_cast<uint32_t>(prefix);
	for (uint32_t root = 0; root < images.size(); ++root) {
		if (deferred[root] == 0u) PlaceRoot(root, image_plans[root], image_forms[root], widened.images, cursor);
	}
	for (uint32_t root = 0; root < samplers.size(); ++root) {
		PlaceRoot(root, sampler_plans[root], sampler_forms[root], widened.samplers, cursor);
	}
	for (uint32_t root = 0; root < images.size(); ++root) {
		if (deferred[root] != 0u) PlaceRoot(root, image_plans[root], image_forms[root], widened.images, cursor);
	}
	return true;
}
#endif

}
