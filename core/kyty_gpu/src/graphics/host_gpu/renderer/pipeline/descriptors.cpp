#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorsTrace.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envDefaults.h"
#include "common/envRead.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/recordPackets.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <optional>
#include <mutex>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <set>
#include <vulkan/vulkan_format_traits.hpp>
#include <span>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <xxhash.h>
#include <array>
#include <map>
#include <set>
#include <string>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		// Uniform const buffers: constant-index reads only (ShaderIR.h ReadConstUniform).
		case BindingKind::FlattenedSrtUniform:
		case BindingKind::ShaderData: return vk::DescriptorType::eUniformBuffer;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	// Every binding has an image: unbound textures bind our null images (TextureCache::GetNullImage); upstream
	// 892e70c8a binds native null descriptors instead (not taken).
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(!texture.image_id || view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

// The descriptor range of `size` bytes at `offset` in `buffer` (NativeStorageBuffer's, after ObtainBuffer): from the
// storage-aligned offset, `buffer_offset` the bytes before the requested ones (below 256).
static vk::DescriptorBufferInfo StorageBufferRange(RenderContext& context, Buffer& buffer, uint64_t offset, uint64_t size,
                                                   uint32_t& buffer_offset) {
	const auto& graphics       = context.GetGraphics();
	const auto  alignment      = graphics.StorageMinAlignment();
	const auto  aligned_offset = Common::AlignDown(offset, alignment);
	const auto  adjustment     = offset - aligned_offset;
	const auto  max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	return {buffer.Handle(), aligned_offset, size + adjustment};
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		const auto& null_buffer = context.GetBufferCache().GetBuffer(NULL_BUFFER_ID);
		return {null_buffer.Handle(), 0, 16};
	}
	const auto& graphics = context.GetGraphics();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	std::optional<DispatchSubTimer> sub_timer(std::in_place, resource.written ? DispatchSub::ObtainWritten
	                                                                          : DispatchSub::ObtainRead); // KYTY_TRACE_DISPATCH_CENSUS
	// D4b step 1 (append windows): an append pool's cycle starts before its whole-binding mark.
	const bool append_pool = resource.written && size >= BufferCache::AppendMinPoolBytes &&
	                         resource.append_counter != ShaderRecompiler::IR::BufferResource::NoAppendCounter;
	if (append_pool) {
		context.GetBufferCache().AppendBeforePoolMark(address, size);
		context.GetBufferCache().AppendSetTaggedObtain(true);
	}
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	if (append_pool) context.GetBufferCache().AppendSetTaggedObtain(false);
	sub_timer.reset();
	const auto result = StorageBufferRange(context, *buffer, offset, size, buffer_offset);
	if (resource.written) {
		const DispatchSubTimer invalidate_timer(DispatchSub::TcInvalidate); // KYTY_TRACE_DISPATCH_CENSUS
		if (TextureCache::OwnerMode() != 0u) {
			// D4c (KYTY_FIX_IMAGE_MEMORY_OWNER write-back): the binding's write-range facts.
			TextureCache::OwnerCensusBinding facts;
			facts.kind            = 1;
			facts.fixed_ranges    = resource.fixed_writes ? &resource.fixed_write_ranges : nullptr;
			facts.append          = resource.append_counter != ShaderRecompiler::IR::BufferResource::NoAppendCounter;
			facts.append_tracked  = append_pool;
			facts.write_extent    = resource.write_extent;
			facts.max_byte_extent = resource.max_byte_extent;
			facts.read            = resource.read;
			facts.atomic          = resource.atomic;
			facts.formatted       = resource.formatted;
			TextureCache::OwnerCensusSetBinding(&facts);
			context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
			TextureCache::OwnerCensusSetBinding(nullptr);
		} else {
			context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
		}
	}
	return result;
}

// Reserved and unsupported T# fields must be zero; perf_mod only scales sampler state.
static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if ((descriptor.fields[1] & field1_reserved_mask) != 0 ||
	    (descriptor.fields[2] & field2_reserved_mask) != 0 || descriptor.fields[3] != field3_expected) {
		return false;
	}
	if (r128) {
		return descriptor.fields[4] == 0 && descriptor.fields[5] == 0 && descriptor.fields[6] == 0 &&
		       descriptor.fields[7] == 0;
	}
	return descriptor.fields[4] == field4_expected && descriptor.fields[5] == field5_expected;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	if (!IsCanonicalTextureEncoding(descriptor, r128)) {
		return false;
	}
	if (r128) {
		return true;
	}
	if (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type())) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

// KYTY_WORKAROUND_DEPTH_ENCODING_LENIENT (stage 2 pack 2, #15): IsSupportedDepthTextureEncoding compares words 4-7 whole. RDNA2
// ISA table 45 (8.2.6 Image Resource) defines them per field, and for a non-array 1D/2D T# (types 8 and 9, the 128-bit
// resource kinds) the texture unit does not use: in word 4, DEPTH [12:0] and PITCH[13] [13] (depth-1 of a 3D map, the
// last slice of an array; for 1D/2D the low bits of pitch-1, which only a linear surface with a custom pitch uses - Mesa
// ac_descriptors.c sets it for GFX10.3 linear surfaces only - and a depth T# is never linear) and the reserved bits
// [15:14], [31:29]; in word 5, ARRAY_PITCH [3:0] (arrays and 3D), MIN_LOD_WARN [19:8] (an LOD feedback trigger) and
// [31:24], which the table does not list; in words 6-7 with COMPRESSION_EN (word 6 [21]) = 0, everything: no metadata is
// read, so META_ADDR, write compression, ALPHA_IS_ON_MSB, the color transform, the block sizes, ITERATE_256 (compressed
// MSAA depth only), the LLC policy [9:8] and the MIP stats id [7:0] change nothing a sample returns. Still used:
// BASE_ARRAY (word 4 [28:16]; the view takes it, see KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL), MAX_MIP and PERF_MOD (word 5,
// accepted by the strict check as they are), CORNER_SAMPLES (word 5 [23], where texels sit: not emulated) and, with
// COMPRESSION_EN = 1, the HTILE encoding the strict check knows. The verdict: the strict check on a copy with the unused
// fields cleared. Seen in the game's logs (9 runs, 2D depth T#s at 0x5002870000 and 0x508f360000): five
// kinds differ in unused fields only (e.g. words 4-7 00000318,504ef100,0000031b,00000000); one has CORNER_SAMPLES = 1
// (00000080,00fff000,0a503f00,00000000).
DepthEncodingFields DepthEncodingFieldsVerdict(const ShaderTextureResource& descriptor, bool r128) {
	if (r128 || (descriptor.Type() != Prospero::ImageType::kColor1D && descriptor.Type() != Prospero::ImageType::kColor2D) ||
	    descriptor.BaseArray5() != 0) {
		return DepthEncodingFields::Other;
	}
	auto       used       = descriptor;
	const bool compressed = descriptor.MetaCompress();
	used.fields[4]        = 0;
	used.fields[5]        = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                 (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	if (!compressed) {
		used.fields[6] = 0;
		used.fields[7] = 0;
	}
	if (!IsSupportedDepthTextureEncoding(used, false)) return DepthEncodingFields::Other;
	return descriptor.CornerSample() ? DepthEncodingFields::Corner : DepthEncodingFields::Ignored;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	// KYTY_WORKAROUND_DEPTH_ENCODING_LENIENT=1 (logged APPROXIMATION): a sampled depth descriptor whose
	// unused words carry other data (seen: word 4 = 1.0f after a 2D depth T#) is bound anyway; the
	// view clamps an out-of-range base layer (KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL).
	// =0 (stage 2 pack 2, #15; before: every such descriptor stopped the emulator): a descriptor that differs only in
	// fields the texture unit does not use for it (DepthEncodingFieldsVerdict, RDNA2 ISA table 45) is the hardware's T#
	// and is bound as such (logged DEPTH_ENCODING_FIELDS, no approximation); CORNER_SAMPLES = 1 and anything else still
	// stop it. KYTY_TRACE_PERF counts the three kinds whatever the value (PERF workaround_hits depth_encoding).
	static const bool lenient = [] {
		// Default from Common::EnvDefaults; workaround: D8 (stencil / HTILE), to
		// be replaced by the proper implementation; =0 turns it off.
		const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_DEPTH_ENCODING_LENIENT");
		return value != nullptr && value[0] == '1';
	}();
	const auto fields = !encoding_ok && resource_ok && view_ok ? DepthEncodingFieldsVerdict(descriptor, resource.r128)
	                                                            : DepthEncodingFields::Other;
	if (!encoding_ok && resource_ok && view_ok) {
		using Common::WorkaroundHits::Hit;
		Common::WorkaroundHits::Note(fields == DepthEncodingFields::Ignored  ? Hit::DepthEncodingIgnored
		                             : fields == DepthEncodingFields::Corner ? Hit::DepthEncodingCorner
		                                                                     : Hit::DepthEncodingOther);
	}
	const char* fields_name = fields == DepthEncodingFields::Ignored  ? "ignored"
	                          : fields == DepthEncodingFields::Corner ? "corner"
	                                                                  : "other";
	if (lenient && resource_ok && view_ok) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u) {
			std::fprintf(stderr,
			             "WORKAROUND_DEPTH_ENCODING_LENIENT APPROXIMATION addr=0x%llx r128=%d dwords=%08x,%08x,%08x,%08x,"
			             "%08x,%08x,%08x,%08x fields=%s\n",
			             static_cast<unsigned long long>(descriptor.Base40()), resource.r128 ? 1 : 0,
			             descriptor.fields[0], descriptor.fields[1], descriptor.fields[2], descriptor.fields[3],
			             descriptor.fields[4], descriptor.fields[5], descriptor.fields[6], descriptor.fields[7], fields_name);
		}
		return;
	}
	if (resource_ok && view_ok && fields == DepthEncodingFields::Ignored) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u) {
			std::fprintf(stderr,
			             "DEPTH_ENCODING_FIELDS unused by the texture unit addr=0x%llx dwords=%08x,%08x,%08x,%08x,%08x,"
			             "%08x,%08x,%08x\n",
			             static_cast<unsigned long long>(descriptor.Base40()), descriptor.fields[0], descriptor.fields[1],
			             descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
			             descriptor.fields[6], descriptor.fields[7]);
		}
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	const bool is_2d_msaa =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa &&
	    descriptor.Type() == Prospero::ImageType::kColor2DMsaa && descriptor.Depth() == 0 &&
	    descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d || is_2d_msaa) &&
	       supported_tile && descriptor.BaseLevel() <= descriptor.LastLevel() &&
	       descriptor.MinLod() == 0 && supported_swizzle && descriptor.BCSwizzle() == 0 &&
	       !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = (format == Prospero::BufferFormat::k32SInt ||
	                               format == Prospero::BufferFormat::k16SInt ||
	                               format == Prospero::BufferFormat::k8SInt) && uint_resource &&
	                              resource.written && !resource.read && !resource.atomic;
	// A format the shader decodes from its backing (11_11_10 UNorm through R32_UINT) is typed by the backing's class.
	const auto numeric_class = Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float: {
			// A sampled null float texture is the RGBA32F null image, whose texel TextureCache::GetNullImage writes:
			// (0.5, 0.5, 0.5, 1) with KYTY_DIAG_NULL_TEXTURE_GRAY=1 (workaround D1, logged APPROXIMATION there: unresolved
			// material textures read neutral), else (0, 0, 0, 0), what the hardware returns for a null T#. (Before
			// =0 bound an R32F image whose texel was never written, read as (x, 0, 0, 1).) Storage: R32F.
			const bool sampled = binding == TextureCache::BindingType::Texture;
			if (sampled) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::NullFloatSampled); // KYTY_TRACE_PERF
			desc.info.guest_format = sampled ? Prospero::BufferFormat::k32_32_32_32Float : Prospero::BufferFormat::k32Float;
			break;
		}
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

// A null image on a depth-compare binding is a 1x1 D16_UNORM image instead of the R32F/RGBA32F null texture, whose
// compare sample Vulkan leaves undefined (VUID-vkCmdDispatch-None-06479; CS 6cc64dee32dc7094, GPU-selected compare
// table entry 2). The hardware returns 0 for a null T# (Mesa RADV writes all-zero descriptors for null views and relies
// on that), so the image holds the depth that fails the S#'s comparison (reference OP texel; D16 clamps the reference
// to [0, 1]): 0.0 for NEVER and LESS, and for EQUAL and LEQUAL unless the reference is <= 0; 1.0 for GREATER, and for
// GEQUAL unless the reference is >= 1. NOTEQUAL, ALWAYS or a mix of both families pass for some reference whatever the
// depth, so no image gives the hardware's 0: such a binding stops the emulator with its compare functions (none in
// the game's data: 0 of 8964 on the gate and village route, 28.10).

// DEPTH_COMPARE_FUNC values, as 1 << func, of the S#s that compare-sample image `index` (a GPU-selected candidate is
// sampled with the S#s of its table root); 0 when the image has no compare sample.
static uint32_t ImageCompareFuncs(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                  const ShaderRecompiler::IR::ResourceSnapshot& snapshot, uint32_t index) {
	const auto& images = program.info.images;
	if (index >= images.size() || !images[index].depth_compare) {
		return 0u;
	}
	const auto root = images[index].indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage
	                      ? images[index].indirect_root
	                      : index;
	const auto& samplers = program.info.samplers;
	uint32_t    funcs    = 0;
	const auto  add      = [&](uint32_t sampler) {
		if (sampler < samplers.size() && samplers[sampler].snapshot_index < snapshot.samplers.size()) {
			funcs |= 1u << ((snapshot.samplers[samplers[sampler].snapshot_index].dwords[0] >> 12u) & 7u);
		}
	};
	for (const auto& pair: program.info.sampled_pairs) {
		if ((pair.image != index && pair.image != root) || pair.sampler >= samplers.size()) {
			continue;
		}
		add(pair.sampler);
		for (const auto candidate: samplers[pair.sampler].indirect_resources) {
			add(candidate);
		}
	}
	return funcs;
}

static TextureBinding NullCompareBinding(TextureCache& texture_cache, uint32_t compare_funcs) {
	constexpr uint32_t FailAtZero = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u); // never less equal lequal
	constexpr uint32_t FailAtOne  = (1u << 0u) | (1u << 4u) | (1u << 6u);              // never greater gequal
	const bool         one        = compare_funcs != 0u && (compare_funcs & ~FailAtOne) == 0u;
	Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::NullCompareUse); // KYTY_TRACE_PERF
	if (!one && (compare_funcs & ~FailAtZero) != 0u) {
		EXIT("null image on a depth-compare binding with compare functions 0x%02x (NOTEQUAL/ALWAYS or both families): no "
		     "depth gives the hardware's 0 for a null T#\n",
		     compare_funcs);
	}
	TextureCache::ImageDesc desc {};
	desc.info.pixel_format    = vk::Format::eD16Unorm;
	desc.info.guest_format    = Prospero::BufferFormat::k16UNorm;
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = 2;
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = vk::Format::eD16Unorm;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eDepth;
	desc.view_info.usage      = vk::ImageUsageFlagBits::eSampled;
	desc.type                 = TextureCache::BindingType::Texture;
	return {texture_cache.GetNullDepthImage(one), nullptr, std::move(desc)};
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	// The sampler clamps a minimum LOD beyond the last mip level to that level (streaming
	// textures publish min_lod=4095 before their mips are resident).
	const auto min_lod  = std::min<uint32_t>(descriptor.MinLod(), descriptor.LastLevel() * 256u);
	const auto base_lod = view.base_level * 256u;
	if (min_lod > base_lod) {
		view.min_lod = min_lod - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				// RDNA2 ISA table 45 lists BASE_ARRAY among the fields of array, 3D, cube and MSAA resources only, so a
				// 1D/2D view's BASE_ARRAY naming a layer the image does not have (junk: 7967, 7710, 514, 122 logged)
				// selects no slice: an image of one layer is read at that layer (exact; every case in the game's data,
				// 28.10: clamp_layer=10/0). An image of more layers stops the emulator: which slice the texture unit
				// reads then is not established. (The layer part of KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL, M48, until 28.10.)
				if (image_layers > 1u) {
					EXIT("texture base layer %u is out of bounds of an image of %u layers\n", view.base_layer, image_layers);
				}
				view.base_layer = 0;
				Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::ClampLayer); // KYTY_TRACE_PERF
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds (base_layer=%u image_layers=%u)\n", view.base_layer,
				     image_layers);
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				// RDNA2 ISA table 45 lists BASE_ARRAY among the fields of array, 3D, cube and MSAA resources only, so a
				// 1D/2D view's BASE_ARRAY naming a layer the image does not have (junk: 7967, 7710, 514, 122 logged)
				// selects no slice: an image of one layer is read at that layer (exact; every case in the game's data,
				// 28.10: clamp_layer=10/0). An image of more layers stops the emulator: which slice the texture unit
				// reads then is not established. (The layer part of KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL, M48, until 28.10.)
				if (image_layers > 1u) {
					EXIT("texture base layer %u is out of bounds of an image of %u layers\n", view.base_layer, image_layers);
				}
				view.base_layer = 0;
				Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::ClampLayer); // KYTY_TRACE_PERF
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical        = info;
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

namespace {

// The description ResolveTexture derives for a texture (decode, format policy, layout through UpdateSize, mip view, DCC
// size, view info, validation; port of Senaxx 50a04054 and BryanKAdams 151d7916 part 2) is a pure function of the eight
// T# dwords and of the ImageResource fields it reads, so each thread keeps it per such key. The texture cache lookup
// (FindImage) and every check against the found image still run every time, on a copy. Not cached: soft resolves
// (m_soft_texture_resolve: bindless keys, GPU-selected candidates), which also read the guest's reserved memory map
// (TryClampReservedRangeSize) and turn unsupported encodings into null bindings, and every outcome other than a
// description (null bindings, exits; compare_funcs only shapes null bindings). The environment switches the description
// reads are fixed for the process; the rate-limited logs of the description phase are not repeated on hits.

// The T# and every ImageResource field the description phase reads (validation included); no padding.
struct TextureDescKey {
	std::array<uint32_t, 8> fields {};
	uint32_t                resource_class = 0;
	uint32_t                numeric_class  = 0;
	uint32_t                dimension      = 0;
	uint32_t                mip_mode       = 0;
	uint32_t                mip_count      = 0;
	uint32_t                conversion     = 0;
	uint32_t                swizzle        = 0;
	uint32_t                flags          = 0;

	bool operator==(const TextureDescKey& other) const = default;
};
static_assert(sizeof(TextureDescKey) == 64);
static_assert(sizeof(ShaderTextureResource) == sizeof(TextureDescKey::fields));

struct TextureDescKeyHash {
	size_t operator()(const TextureDescKey& key) const noexcept {
		return static_cast<size_t>(XXH3_64bits(&key, sizeof(key)));
	}
};

// Everything the lookup and the checks after it take from the description phase.
struct TextureDescEntry {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	bool                    shader_conversion = false;
};

constexpr size_t TextureDescCacheCapacity = 8192; // ~10 MB per thread; cleared when full

// Hits, misses and clears of the description memo; a PERF line about every 10 s (checked every 4096 lookups).
void TextureDescCacheCount(bool hit, bool cleared) {
	static std::atomic<uint64_t> hits {0};
	static std::atomic<uint64_t> misses {0};
	static std::atomic<uint64_t> clears {0};
	static std::atomic<int64_t>  last_log {0};
	const auto lookups = (hit ? hits : misses).fetch_add(1, std::memory_order_relaxed) + 1;
	if (cleared) clears.fetch_add(1, std::memory_order_relaxed);
	if ((lookups & 4095u) != 0 || !Common::EnvRead::PerfLines()) return;
	const auto now =
	    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	auto last = last_log.load(std::memory_order_relaxed);
	if (now - last >= 10 && last_log.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		std::fprintf(stderr, "PERF texture_desc_cache hits=%llu misses=%llu clears=%llu\n",
		             static_cast<unsigned long long>(hits.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(misses.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(clears.load(std::memory_order_relaxed)));
	}
}

TextureDescKey MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource& resource,
                                  const ShaderTextureResource&               descriptor) {
	TextureDescKey key {};
	std::copy_n(std::begin(descriptor.fields), key.fields.size(), key.fields.begin());
	key.resource_class = static_cast<uint32_t>(resource.resource_class);
	key.numeric_class  = static_cast<uint32_t>(resource.numeric_class);
	key.dimension      = static_cast<uint32_t>(resource.dimension);
	key.mip_mode       = static_cast<uint32_t>(resource.mip_mode);
	key.mip_count      = resource.mip_count;
	key.conversion     = static_cast<uint32_t>(resource.conversion_format);
	key.swizzle        = resource.shader_swizzle;
	key.flags          = (resource.read ? 1u : 0u) | (resource.written ? 2u : 0u) | (resource.atomic ? 4u : 0u) |
	            (resource.atomic64 ? 8u : 0u) | (resource.depth_compare ? 16u : 0u) | (resource.cube ? 32u : 0u) |
	            (resource.r128 ? 64u : 0u) | (resource.bindless ? 128u : 0u) |
	            (resource.indirect_root == ShaderRecompiler::IR::ImageResource::NoIndirectImage ? 256u : 0u);
	return key;
}

} // namespace

// ResolveTexture's description phase (everything between the T# decode and the texture cache lookup: decode, format
// policy, layout, mip view, DCC size, view info, validation). Described: `desc` (value-initialized by the caller) and the
// three formats/flags hold the description. SoftNull: the caller binds the null image (a soft resolve's rejection, a
// GPU-selected candidate whose layout is unusable).
enum class TextureDescribe : uint8_t { Described, SoftNull };

// M14 S2: replaces `info` (a tiled single-layer 2D chain) by the chain of the levels
// [view_base, view_base + view_levels) alone when every one of them is stored exactly as in the full chain (offset,
// size, pitch, height, tail position, extent, tail membership) - the bytes and their meaning are then the same, and the
// largest levels, which a streamed texture does not have in memory, get no host storage.
static bool StreamedViewSubChain(ImageInfo& info, uint32_t view_base, uint32_t view_levels) {
	if (!info.IsTiled() || view_base == 0u || view_levels == 0u || view_base + view_levels > info.resources.levels) {
		return false;
	}
	ImageInfo sub        = info;
	sub.extent           = {std::max(info.extent.width >> view_base, 1u), std::max(info.extent.height >> view_base, 1u), 1u};
	sub.resources.levels = view_levels;
	sub.pitch            = 0;
	sub.data.size        = 0;
	sub.UpdateSize();
	for (uint32_t i = 0; i < view_levels; ++i) {
		const auto& left  = sub.mip_layout[i];
		const auto& right = info.mip_layout[view_base + i];
		if (left.offset != right.offset || left.size != right.size || left.pitch != right.pitch ||
		    left.height != right.height || left.tail_x != right.tail_x || left.tail_y != right.tail_y ||
		    left.surface_z != right.surface_z || sub.MipExtent(i) != info.MipExtent(view_base + i) ||
		    (i >= sub.first_tail_level) != (view_base + i >= info.first_tail_level)) {
			return false;
		}
	}
	info = std::move(sub);
	return true;
}

static TextureDescribe DescribeTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                       const ShaderTextureResource& descriptor, bool soft_resolve,
                                       TextureCache::ImageDesc& desc, bool& out_shader_conversion,
                                       vk::Format& out_pixel_format, vk::Format& out_view_format) {
	const bool storage = resource.written;
	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	auto view_levels = multisampled || single_storage_mip
	                       ? 1u
	                       : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		if (soft_resolve) return TextureDescribe::SoftNull;
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	// Upstream aafe014d0: a raw load of a kDepth k16UInt T# binds the D16 surface as UNorm (IR::ReadsDepthBits; the
	// shader converts back).
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource) ? Prospero::BufferFormat::k16UNorm
	                                                                           : descriptor.Format();
	// KYTY_TRACE_PERF (PERF workaround_hits fmt30, KYTY_FIX_RT_11_11_10_FLOAT): a T# of format 30, or of format 36
	// (10_11_11_FLOAT) over the memory of a target programmed as format 30.
	Common::WorkaroundHits::NoteFmt30Texture(static_cast<uint32_t>(format), address);
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class == Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	// (As upstream 56738aa4e: the materialization types a format the shader decodes from its backing - 11_11_10 UNorm
	// through R32_UINT when KYTY_FIX_RT_11_11_10_FLOAT=0 - by the backing's class; every other format remaps to itself.)
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		if (soft_resolve) return TextureDescribe::SoftNull;
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);

	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	// Our raw signed storage views (k32/k16/k8 SInt as UInt; upstream c88f05f7b's IR::RawSignedStorageFormat covers
	// k32/k16 written-only stores, which this keeps a subset of).
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && (resource.atomic ||
	                                               format == Prospero::BufferFormat::k32SInt)
	                                     ? vk::Format::eR32Uint
	                                 : storage && format == Prospero::BufferFormat::k16SInt ? vk::Format::eR16Uint
	                                 : storage && format == Prospero::BufferFormat::k8SInt  ? vk::Format::eR8Uint
	                                     : SrgbStorageViewFormat(pixel_format);
	// A read of an 8_SRGB T#: the image keeps its R8_UNORM backing (shared with the game's UNORM writes), the view
	// decodes sRGB like the PS5 sampler (fetch, gather and filtered samples alike) where the device samples R8_SRGB.
	const bool r8_srgb_read = !storage && format == Prospero::BufferFormat::k8Srgb &&
	                          pixel_format == vk::Format::eR8Unorm && SampledR8SrgbSupported().load();
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                 : r8_srgb_read ? vk::Format::eR8Srgb
	                                                : pixel_format;
	// Guard: a read of an sRGB T# whose view does not decode and whose
	// shader does not decode either (an 8_SRGB T# on a device without sampled R8_SRGB). Expected never; the first
	// distinct textures are logged.
	if (!storage && format == Prospero::BufferFormat::k8Srgb && view_format != vk::Format::eR8Srgb) {
		static std::atomic<uint32_t> undecoded {0};
		if (undecoded.fetch_add(1) < 8u) {
			std::fprintf(stderr, "SRGB_READ_UNDECODED addr=0x%llx fmt=%u view=%s\n",
			             static_cast<unsigned long long>(address), static_cast<uint32_t>(format),
			             vk::to_string(view_format).c_str());
		}
	}
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	// (Upstream 8c7fe539b marks DCC here, before the mip view, for its compact mip-tail sizes; ours marks it with the
	// DCC range below.)
	if (levels > physical_levels) {
		// KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL=1: logged APPROXIMATION. A view whose LAST_LEVEL
		// exceeds the allocated MAX_MIP (partially resident streaming textures) keeps only the
		// allocated mips.
		static constinit Common::OnceValue<bool> clamp_last_level_once;
		const bool clamp_last_level = clamp_last_level_once.Get([] {
			// Default from Common::EnvDefaults: probably hardware behaviour (M48),
			// exactness check open: RDNA2 ISA table 45 says MAX_MIP describes the resource and BASE/LAST_LEVEL the view,
			// but not that the texture unit clamps the view's LOD to MAX_MIP; =0 turns it off.
			const char* value = Common::EnvOrDefault("KYTY_FIX_TEXTURE_CLAMP_LAST_LEVEL");
			return value != nullptr && value[0] == '1';
		});
		const bool resolved = ResolveTextureMipView(desc.info, physical_levels,
		                                            !resource.r128 && descriptor.MetaCompress(), view_levels,
		                                            view_base);
		if (!resolved && clamp_last_level) {
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 8u) {
				std::fprintf(stderr,
				             "FIX_TEXTURE_CLAMP_LAST_LEVEL APPROXIMATION base=%u last=%u max=%u extent=%ux%u\n",
				             base_level, last_level, max_mip, width, height);
			}
			// KYTY_TRACE_PERF (PERF workaround_hits clamp_mip): views cut at MAX_MIP, and those whose BASE_LEVEL is past it.
			Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::ClampMip);
			if (base_level > max_mip) Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::ClampMipBasePast);
			// A BASE_LEVEL beyond MAX_MIP views the last allocated mip.
			const auto first           = std::min<uint32_t>(base_level, max_mip);
			view_levels                = static_cast<uint32_t>(max_mip) - first + 1u;
			levels                     = physical_levels;
			view_base                  = first;
			desc.info.resources.levels = physical_levels;
			desc.info.UpdateSize();
		} else if (!resolved) {
			if (soft_resolve) return TextureDescribe::SoftNull;
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	} else {
		KYTY_PROFILER_BLOCK("ResolveTexture::UpdateSize");
		desc.info.UpdateSize();
	}
	// A soft resolve (bindless key, GPU-selected candidate) whose image does not even fit in the guest's reserved
	// virtual memory read words that are not a live descriptor (a 3840x2160 R8 view once claimed 1537 layers,
	// ~17.6 GB): null, like the other descriptors such a key can land on. Reserved, not committed: partially
	// resident textures pass.
	// Streamed texture views: what must be reserved is the levels the view reads. Guest
	// chains run backwards (the largest level last), so a streamed texture whose largest levels are not loaded (BASE_LEVEL
	// above them, their pages not even reserved) still has every viewed level inside reserved memory; the whole-chain
	// test dropped it (the distant terrain's snow mask: transparent black, no snow on Mount Yotei). Single-layer 2D only.
	// The image is then described as those levels alone (StreamedViewSubChain) when their layout allows it.
	uint64_t reserved_needed = desc.info.data.size;
	if (view_base != 0u && desc.info.resources.layers == 1u && !volume) {
		uint64_t end = 0;
		for (uint32_t level = view_base; level < view_base + view_levels && level < desc.info.mip_layout.size(); level++)
			end = std::max(end, desc.info.mip_layout[level].offset + desc.info.mip_layout[level].size);
		if (end != 0 && end < reserved_needed) {
			reserved_needed = end;
			// Not for storage or DCC-compressed textures (their metadata and validation describe the whole chain).
			if (!storage && !(!resource.r128 && descriptor.MetaCompress())) {
				static std::atomic_uint sub_chains {0};
				static std::atomic_uint whole_chains {0};
				const bool sub = StreamedViewSubChain(desc.info, view_base, view_levels);
				const auto n   = (sub ? sub_chains : whole_chains).fetch_add(1) + 1u;
				if (n <= 4u || (n & (n - 1u)) == 0u) {
					std::fprintf(stderr,
					             "FIX_STREAMED_TEXTURE_VIEW %s%s n=%u addr=0x%llx extent=%ux%u base=%u levels=%u bytes=0x%llx\n",
					             sub ? "sub_chain" : "whole_chain", soft_resolve ? "" : "_direct", n, static_cast<unsigned long long>(address), width,
					             height, view_base, view_levels, static_cast<unsigned long long>(desc.info.data.size));
				}
				if (sub) {
					view_base       = 0;
					reserved_needed = desc.info.data.size;
				}
			}
		}
	}
	// An array view that starts past layer 0 gets an image of its layers alone (ImageInfo::ViewLayersAlone); not for
	// storage or DCC-compressed textures (their metadata and validation describe the whole array).
	uint32_t array_shift = 0;
	if (layered && image_layers > 1u && descriptor.BaseArray5() != 0u && descriptor.BaseArray5() < image_layers &&
	    !storage && !(!resource.r128 && descriptor.MetaCompress())) {
		const uint32_t first = descriptor.BaseArray5();
		if (desc.info.ViewLayersAlone(first, image_layers - first)) {
			array_shift     = first;
			reserved_needed = desc.info.data.size;
			static std::atomic_uint logged {0};
			const auto n = logged.fetch_add(1) + 1u;
			if (n <= 4u || (n & (n - 1u)) == 0u)
				std::fprintf(stderr, "ARRAY_VIEW_SUBRANGE n=%u addr=0x%llx extent=%ux%u layers=%u..%u bytes=0x%llx\n", n,
				             static_cast<unsigned long long>(address), width, height, first, image_layers - 1u,
				             static_cast<unsigned long long>(desc.info.data.size));
		}
	}
	// BIG_TEXTURE_VIEW: a texture of 512 MiB or more, with its T# view fields (first 32 distinct T#s; Ghost of Yotei's
	// one-layer views of 2 GiB arrays).
	if (desc.info.data.size >= (uint64_t {512} << 20u)) {
		static std::mutex                        mutex;
		static std::set<std::array<uint32_t, 8>> seen;
		std::array<uint32_t, 8>                  words {};
		std::copy_n(descriptor.fields, 8u, words.begin());
		std::lock_guard lock(mutex);
		if (seen.size() < 32u && seen.insert(words).second)
			std::fprintf(stderr,
			             "BIG_TEXTURE_VIEW addr=0x%llx %ux%u type=%u dim=%u base_array=%u depth=%u base_level=%u last_level=%u "
			             "max_mip=%u image=L%u x%u view_levels=%u array_shift=%u soft=%d storage=%d MiB=%llu "
			             "words=%08x %08x %08x %08x %08x %08x %08x %08x\n",
			             static_cast<unsigned long long>(address), width, height, static_cast<uint32_t>(type),
			             static_cast<uint32_t>(resource.dimension), descriptor.BaseArray5(), descriptor.Depth(),
			             static_cast<uint32_t>(base_level), static_cast<uint32_t>(last_level), max_mip,
			             desc.info.resources.levels, desc.info.resources.layers, view_levels, array_shift,
			             soft_resolve ? 1 : 0, storage ? 1 : 0,
			             static_cast<unsigned long long>(desc.info.data.size >> 20u), words[0], words[1], words[2],
			             words[3], words[4], words[5], words[6], words[7]);
	}
	if (soft_resolve && desc.info.data.size != 0 &&
	    Libs::LibKernel::Memory::TryClampReservedRangeSize(desc.info.data.address, reserved_needed) < reserved_needed) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 16u) {
			std::fprintf(stderr,
			             "TEXTURE_BEYOND_GUEST_MEMORY soft null: address=0x%016llx size=0x%llx extent=%ux%u layers=%u "
			             "levels=%u\n",
			             static_cast<unsigned long long>(desc.info.data.address),
			             static_cast<unsigned long long>(desc.info.data.size), desc.info.extent.width,
			             desc.info.extent.height, desc.info.resources.layers, desc.info.resources.levels);
		}
		return TextureDescribe::SoftNull;
	}
	const auto alignment =
	    tile == Prospero::TileMode::kLinear ? 256u : desc.info.tiling.block.block_size;
	// A texture whose base is less aligned than its swizzle block (but 256-byte aligned) is bound at its address
	// (swizzle math is base-relative). Hardware behaviour, not a workaround (RDNA2 ISA table 45: the T# base address is
	// "256-byte aligned", nothing more; texel addresses are the base plus the swizzle offset of the coordinates, as
	// here). Every logged case (16,640 on 12-13.10) was SW_4KB_S (no XOR) at a mip-tail position.
	const bool misaligned = desc.info.data.size != 0 && alignment != 0 &&
	                        (address & (static_cast<uint64_t>(alignment) - 1u)) != 0 && (address & 0xffu) == 0u;
	if (misaligned) {
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 64)
			std::fprintf(stderr, "TEXTURE_MISALIGNED_BASE addr=0x%016llx align=0x%x format=%u tile=%u "
			             "extent=%ux%ux%u levels=%u size=0x%llx storage=%u\n",
			             static_cast<unsigned long long>(address), static_cast<uint32_t>(alignment),
			             static_cast<uint32_t>(format), static_cast<uint32_t>(tile), width, height, depth,
			             physical_levels, static_cast<unsigned long long>(desc.info.data.size), storage ? 1u : 0u);
	} else if (desc.info.data.size == 0 || alignment == 0 ||
	           (address & (static_cast<uint64_t>(alignment) - 1u)) != 0) {
		// A GPU-selected candidate enumerated from table records may be a record that only
		// resembles a texture; it binds as null. Directly bound textures stay strict.
		if (resource.indirect_root == ShaderRecompiler::IR::ImageResource::NoIndirectImage) {
			EXIT("unsupported texture layout: format=%u tile=%u extent=%ux%ux%u levels=%u size=0x%" PRIx64
			     " align=0x%x addr=0x%016" PRIx64 "\n",
			     static_cast<uint32_t>(format), static_cast<uint32_t>(tile), width, height, depth,
			     physical_levels, static_cast<uint64_t>(desc.info.data.size), static_cast<uint32_t>(alignment),
			     static_cast<uint64_t>(address));
		}
		return TextureDescribe::SoftNull;
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	// A compressed texture without a metadata address cannot be decompressed: a stale or garbage
	// descriptor (seen through bindless heaps). Sample its raw data instead of exiting.
	if (!resource.r128 && descriptor.MetaCompress() && descriptor.MetaAddr() == 0) {
		static std::atomic<uint32_t> warned {0};
		if (warned.fetch_add(1) < 8) {
			std::fprintf(stderr, "DIAG_TEXTURE_DCC_WITHOUT_METADATA addr=0x%016llx format=%u extent=%ux%u\n",
			             static_cast<unsigned long long>(address), static_cast<uint32_t>(format), width,
			             height);
		}
	}
	if (!resource.r128 && descriptor.MetaCompress() && descriptor.MetaAddr() != 0 &&
	    tile != Prospero::TileMode::kDepth) {
		KYTY_PROFILER_BLOCK("ResolveTexture::DccSize");
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.kind          = ImageMetadataKind::Dcc;
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers + array_shift);
	desc.view_info.base_level = view_base;
	desc.view_info.base_layer -= array_shift; // the view's layers counted from the image's first
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	out_shader_conversion = shader_conversion;
	out_pixel_format      = pixel_format;
	out_view_format       = view_format;
	return TextureDescribe::Described;
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              uint32_t                                     compare_funcs) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	// The null image of this binding; KYTY_BINDLESS also binds a heap word that only resembles a texture as null
	// instead of exiting.
	const auto soft_null = [&]() -> TextureBinding {
		if (!storage && resource.depth_compare) {
			return NullCompareBinding(texture_cache, compare_funcs);
		}
		auto       desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                    : TextureCache::BindingType::Texture);
		const auto id   = texture_cache.FindImage(desc);
		return {id, nullptr, std::move(desc)};
	};
	if (descriptor.IsNull()) {
		return soft_null();
	}
	// The thread's description memo; soft resolves are never cached.
	thread_local std::unordered_map<TextureDescKey, TextureDescEntry, TextureDescKeyHash> desc_cache;
	const bool     use_desc_cache = !m_soft_texture_resolve;
	TextureDescKey desc_key {};
	if (use_desc_cache) {
		desc_key = MakeTextureDescKey(resource, descriptor);
		if (const auto found = desc_cache.find(desc_key); found != desc_cache.end()) {
			const TextureDescEntry& cached = found->second;
			TextureDescCacheCount(true, false);
			auto desc = cached.desc;
			return FindResolvedTexture(resource, descriptor, desc, cached.shader_conversion, cached.pixel_format,
			                           cached.view_format);
		}
	}
	std::optional<tracy::ScopedZone> desc_zone;
	{
		static constexpr tracy::SourceLocationData location {"ResolveTexture::Desc", TracyFunction, TracyFile, __LINE__, 0};
		desc_zone.emplace(&location, TRACY_CALLSTACK, tracy::ProfilerAvailable());
	}

	TextureCache::ImageDesc desc {};
	bool                    shader_conversion = false;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	if (DescribeTexture(resource, descriptor, m_soft_texture_resolve, desc, shader_conversion, pixel_format, view_format) ==
	    TextureDescribe::SoftNull) {
		return soft_null();
	}

	desc_zone.reset();
	if (use_desc_cache) {
		const bool full = desc_cache.size() >= TextureDescCacheCapacity;
		if (full) {
			desc_cache.clear();
		}
		desc_cache.emplace(desc_key, TextureDescEntry {desc, pixel_format, view_format, shader_conversion});
		TextureDescCacheCount(false, full);
	}
	return FindResolvedTexture(resource, descriptor, desc, shader_conversion, pixel_format, view_format);
}

// ---- the resolve thread's lookups (stageFLookup.h) ----

namespace StageF {

// What the resolve thread found for one ring slot's snapshot: FindBuffers' source of snapshot buffer i (buffers) and of
// record-keyed range k (ranges). The vectors keep their capacity; only the first buffer_count / range_count entries
// belong to the current snapshot.
struct ResolveRecord {
	struct BufferSource {
		BufferOutcome           outcome  = BufferOutcome::None;
		bool                    sentinel = false; // buffers: NumRecords == UINT32_MAX (holds for a read-only binding only)
		bool                    written  = false; // ranges: the range's own flag (part of its key)
		std::array<uint32_t, 4> words {};         // buffers: the V#
		uint64_t                address    = 0;
		uint64_t                range_size = 0; // ranges: the range's own size (part of its key)
		uint64_t                size       = 0; // the source size (buffers) or resident size (ranges)
		BufferId                id {};
		uint64_t                buffer_epoch      = 0; // DrawRepeat::g_buffer_registry_epoch at the lookup
		uint64_t                ranges_generation = 0; // Memory::RangesGeneration() at the clamp
	};
	std::vector<BufferSource> buffers;
	std::vector<BufferSource> ranges;
	uint32_t                  buffer_count = 0;
	uint32_t                  range_count  = 0;
};

void ResolveRecordDeleter::operator()(ResolveRecord* record) const noexcept {
	delete record;
}

void ClearResolveRecord(ResolveRecord* record) noexcept {
	if (record != nullptr) {
		record->buffer_count = 0;
		record->range_count  = 0;
	}
}

namespace {

// The resolve thread's lookups, resolve thread: FindBuffers' sources for the snapshot's storage-buffer V#s and record-keyed
// ranges (stageFLookup.h). The decode, size rules and clamp first (ClampRangeSizeAhead takes the virtual ranges' own
// lock), then the buffer cache lookups under its lookup lock, taken only when it is free.
void BuildBufferSources(RenderContext& renderer, const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                        ResolveRecord& r) {
	const auto buffer_count = static_cast<uint32_t>(snapshot.buffers.size());
	const auto range_count  = static_cast<uint32_t>(snapshot.indirect_ranges.size());
	if (r.buffers.size() < buffer_count) {
		r.buffers.resize(buffer_count);
	}
	if (r.ranges.size() < range_count) {
		r.ranges.resize(range_count);
	}
	r.buffer_count = buffer_count;
	r.range_count  = range_count;
	// FindBuffers' bound for a read-only sentinel V# (NumRecords == UINT32_MAX).
	const auto&    graphics = renderer.GetGraphics();
	const uint64_t sentinel_limit =
	    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} - (graphics.StorageMinAlignment() - 1);
	uint32_t pending = 0; // entries whose clamp held: Found until the lookup decides
	for (uint32_t i = 0; i < buffer_count; i++) {
		auto&       b     = r.buffers[i];
		const auto& value = snapshot.buffers[i];
		b                 = {};
		if (value.dword_count < 4u) {
			b.outcome = BufferOutcome::Short;
			continue;
		}
		std::copy_n(value.dwords.begin(), b.words.size(), b.words.begin());
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(value);
		const auto address    = descriptor.Base48();
		auto       size       = descriptor.GetSize();
		if (address == 0 || size == 0) {
			b.outcome = BufferOutcome::Null;
			continue;
		}
		if (descriptor.NumRecords() == UINT32_MAX) {
			b.sentinel = true;
			size       = std::min({size, uint64_t {256} * 1024 * 1024, sentinel_limit});
		}
		uint64_t generation = 0;
		if (Libs::LibKernel::Memory::ClampRangeSizeAhead(address, size, &generation) != size) {
			b.outcome = BufferOutcome::Clamped;
			continue;
		}
		b.address           = address;
		b.size              = size;
		b.ranges_generation = generation;
		b.outcome           = BufferOutcome::Found;
		pending++;
	}
	for (uint32_t k = 0; k < range_count; k++) {
		auto&       g     = r.ranges[k];
		const auto& range = snapshot.indirect_ranges[k];
		g                 = {};
		g.address         = range.address;
		g.range_size      = range.size;
		g.written         = range.written;
		uint64_t generation = 0;
		if (Libs::LibKernel::Memory::ClampRangeSizeAhead(range.address, range.size, &generation) != range.size) {
			g.outcome = BufferOutcome::Clamped;
			continue;
		}
		g.size              = range.size;
		g.ranges_generation = generation;
		g.outcome           = BufferOutcome::Found;
		pending++;
	}
	if (pending == 0) {
		return;
	}
	auto& cache  = renderer.GetBufferCache();
	bool  locked = false;
	for (uint32_t attempt = 0; attempt < 4u && !locked; attempt++) {
		locked = cache.TryLockLookup();
		for (uint32_t spin = 0; !locked && spin < 64u; spin++) {
			SpinPause();
		}
	}
	// Read under the lock: the GPU thread bumps it inside the same lock (BufferCache::ChangeRegister).
	const uint64_t epoch   = locked ? DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_relaxed) : 0;
	uint64_t       lookups = 0, found = 0;
	const auto     look    = [&](ResolveRecord::BufferSource& source) {
		if (source.outcome != BufferOutcome::Found) {
			return;
		}
		if (!locked) {
			source.outcome = BufferOutcome::Busy;
			return;
		}
		lookups++;
		source.id = cache.LookupBuffer(source.address, source.size);
		if (!source.id) {
			source.outcome = BufferOutcome::NoBuffer;
			return;
		}
		source.buffer_epoch = epoch;
		found++;
	};
	for (uint32_t i = 0; i < buffer_count; i++) {
		look(r.buffers[i]);
	}
	for (uint32_t k = 0; k < range_count; k++) {
		look(r.ranges[k]);
	}
	if (locked) {
		cache.UnlockLookup();
	}
	NoteResolveBufferBatch(!locked, lookups, found);
}

// The resolve thread's lookups, GPU thread (FindBuffers): whether a resolved source found by the lookup is still what
// FindBuffers would compute: the buffer registry and the virtual ranges are unchanged since. Counted.
bool BufferSourceHolds(const ResolveRecord::BufferSource& source) {
	if (source.buffer_epoch != DrawRepeat::g_buffer_registry_epoch.load(std::memory_order_relaxed)) {
		NoteBufferUse(BufferUse::Stale);
		return false;
	}
	if (source.ranges_generation != Libs::LibKernel::Memory::RangesGeneration()) {
		NoteBufferUse(BufferUse::StaleRanges);
		return false;
	}
	NoteBufferUse(BufferUse::Used);
	return true;
}

// The resolve thread's source of snapshot buffer `resource` (V# `value`, bound read-only unless `written`) when it still holds;
// null (counted by reason) otherwise.
const ResolveRecord::BufferSource* BufferSourceAhead(const ResolveRecord* record, uint32_t resource,
                                                     const ShaderRecompiler::IR::DescriptorValue& value, bool written) {
	if (record == nullptr) {
		NoteBufferUse(BufferUse::NoRecord);
		return nullptr;
	}
	if (resource >= record->buffer_count) {
		NoteBufferUse(BufferUse::NotAhead);
		return nullptr;
	}
	const auto& source = record->buffers[resource];
	if (source.outcome != BufferOutcome::Found) {
		NoteBufferMiss(source.outcome);
		return nullptr;
	}
	if (value.dword_count < source.words.size() ||
	    !std::equal(source.words.begin(), source.words.end(), value.dwords.begin())) {
		NoteBufferUse(BufferUse::Words);
		return nullptr;
	}
	if (source.sentinel && written) {
		NoteBufferUse(BufferUse::Written);
		return nullptr;
	}
	return BufferSourceHolds(source) ? &source : nullptr;
}

// The resolve thread's source of record-keyed range `index` when it still holds; null (counted by reason) otherwise.
const ResolveRecord::BufferSource* RangeSourceAhead(const ResolveRecord* record, uint32_t index,
                                                    const ShaderRecompiler::IR::IndirectRange& range) {
	if (record == nullptr) {
		NoteBufferUse(BufferUse::NoRecord);
		return nullptr;
	}
	if (index >= record->range_count) {
		NoteBufferUse(BufferUse::NotAhead);
		return nullptr;
	}
	const auto& source = record->ranges[index];
	if (source.outcome != BufferOutcome::Found) {
		NoteBufferMiss(source.outcome);
		return nullptr;
	}
	if (source.address != range.address || source.range_size != range.size || source.written != range.written) {
		NoteBufferUse(BufferUse::Words);
		return nullptr;
	}
	return BufferSourceHolds(source) ? &source : nullptr;
}

} // namespace

void BuildResolveRecord(RenderContext& renderer, const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                        ResolveRecordPtr& record) {
	if (!record) {
		record.reset(new ResolveRecord {});
	}
	BuildBufferSources(renderer, snapshot, *record);
}

} // namespace StageF

// ResolveTexture after the description: the texture cache lookup (FindImage may adjust desc) and the checks against
// the found image (also for a cached description).
TextureBinding RenderExecutor::FindResolvedTexture(const ShaderRecompiler::IR::ImageResource& resource,
                                                   const ShaderTextureResource& descriptor, TextureCache::ImageDesc& desc,
                                                   bool shader_conversion, vk::Format pixel_format,
                                                   vk::Format view_format) {
	auto&      texture_cache = m_context.GetTextureCache();
	const bool storage       = resource.written;
	const auto address       = descriptor.Base40();
	const auto format        = descriptor.Format();
	ImageId    id;
	{
		KYTY_PROFILER_BLOCK("ResolveTexture::FindImage");
		id = texture_cache.FindImage(desc, shader_conversion);
	}
	KYTY_PROFILER_BLOCK("ResolveTexture::Tail");
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	// KYTY_WORKAROUND_STENCIL_ALIAS_COPY=1: logged APPROXIMATION for dynamic resolution. A stencil plane
	// sampled at an extent other than its depth target's is cached as a separate R8 image nothing
	// writes; copy the overlapping top-left region of the depth image's stencil into it first.
	static constinit Common::OnceValue<bool> stencil_alias_copy_once;
	const bool stencil_alias_copy = stencil_alias_copy_once.Get([] {
		// Default from Common::EnvDefaults; workaround: D8 (stencil / HTILE), to
		// be replaced by the proper implementation; =0 turns it off.
		const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_STENCIL_ALIAS_COPY");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) std::fprintf(stderr, "WORKAROUND_STENCIL_ALIAS_COPY APPROXIMATION enabled\n");
		return on;
	});
	if (stencil_alias_copy && address == 0x5004850000ull) {
		static std::atomic_uint traced {0};
		if (traced.fetch_add(1) < 6u) {
			const auto association = texture_cache.FindStencilAssociationAt(address);
			std::fprintf(stderr,
			             "DIAG_STENCIL_ALIAS_TRACE assoc=%d storage=%d bpb=%u depth=%d image=%ux%u size=0x%llx found=%d\n",
			             stencil_association ? 1 : 0, storage ? 1 : 0, image->info.bytes_per_block,
			             image->info.IsDepth() ? 1 : 0, image->backing.extent.width, image->backing.extent.height,
			             static_cast<unsigned long long>(image->info.data.size), association ? 1 : 0);
		}
	}
	// KYTY_WORKAROUND_STENCIL_PLANE_SYNC=1: a read of a guest stencil plane through a plain 1-byte image gets the plane's
	// current contents (owning depth image's stencil aspect, or its saved copy when the depth image holds the other
	// plane); see TextureCache::SyncStencilPlaneReader.
	bool stencil_plane_synced = false;
	if (TextureCache::StencilPlaneSyncEnabled() && !stencil_association && !storage &&
	    image->info.bytes_per_block == 1u && !image->info.IsDepth()) {
		stencil_plane_synced = texture_cache.SyncStencilPlaneReader(address, id);
		image                = &texture_cache.GetImage(id);
	}
	if (stencil_alias_copy && !stencil_plane_synced && !stencil_association && !storage &&
	    image->info.bytes_per_block == 1u && !image->info.IsDepth()) {
		if (const auto association = texture_cache.FindStencilAssociationAt(address)) {
			const auto depth_id = texture_cache.GetImage(association).depth_id;
			if (depth_id && depth_id != id) {
				texture_cache.EnsureStencilPlane(depth_id, address);
				CopyStencilToAlias(depth_id, id);
				image = &texture_cache.GetImage(id);
			}
		}
	}
	if (stencil_association && storage) {
		// A storage view of the associated depth/stencil image is invalid Vulkan (no storage usage or format support):
		// the stencil shadow takes the write (KYTY_WORKAROUND_STENCIL_STORAGE_SHADOW), else a null storage image is bound
		// (logged APPROXIMATION below).
		static const bool shadow_stencil = [] {
			// Default from Common::EnvDefaults; workaround: D8 (stencil /
			// HTILE), to be replaced by the proper implementation; =0 turns it off.
			const char* value = Common::EnvOrDefault("KYTY_WORKAROUND_STENCIL_STORAGE_SHADOW");
			const bool  on    = value != nullptr && value[0] == '1';
			if (on) std::fprintf(stderr, "WORKAROUND_STENCIL_STORAGE_SHADOW APPROXIMATION enabled\n");
			return on;
		}();
		if (shadow_stencil) {
			auto&      depth  = texture_cache.GetImage(image->depth_id);
			const auto width  = depth.backing.extent.width;
			const auto height = depth.backing.extent.height;
			const StencilShadow pair {texture_cache.GetStencilShadowImage(width, height), image->depth_id};
			if (std::ranges::none_of(m_stencil_shadows, [&](const StencilShadow& known) {
				    return known.shadow == pair.shadow && known.depth == pair.depth;
			    })) {
				m_stencil_shadows.push_back(pair);
				CopyStencilShadow(m_context.GetCommandScheduler().Current().Handle(), pair, false);
			}
			auto shadow_desc                 = NullTextureDesc(resource, TextureCache::BindingType::Storage);
			shadow_desc.info.guest_format    = Prospero::BufferFormat::k8UInt;
			shadow_desc.info.pixel_format    = vk::Format::eR8Uint;
			shadow_desc.info.extent          = {width, height, 1};
			shadow_desc.info.bytes_per_block = 1;
			shadow_desc.info.mip_layout[0]   = {0, uint64_t {width} * height, width, height};
			shadow_desc.view_info.format     = vk::Format::eR8Uint;
			return {pair.shadow, nullptr, std::move(shadow_desc)};
		}
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 8u) {
			std::fprintf(stderr,
			             "DIAG_DEPTH_STORAGE_NULL APPROXIMATION storage write to depth-associated image "
			             "addr=0x%llx fmt=%u bound as null\n",
			             static_cast<unsigned long long>(desc.info.data.address), static_cast<uint32_t>(format));
		}
		DebugDepthStorageNulled() = true;
		auto       null_desc = NullTextureDesc(resource, TextureCache::BindingType::Storage);
		const auto null_id   = texture_cache.FindImage(null_desc);
		return {null_id, nullptr, std::move(null_desc)};
	}
	if (stencil_association) {
		const auto plane_address = image->info.data.address;
		id    = image->depth_id;
		texture_cache.EnsureStencilPlane(id, plane_address);
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, pixel_format,
		                             descriptor.DstSelXYZW());
	}
	return {id, nullptr, std::move(desc)};
}

void RenderExecutor::CopyStencilToAlias(ImageId depth_id, ImageId target_id) {
	auto&      texture_cache = m_context.GetTextureCache();
	auto&      depth         = texture_cache.GetImage(depth_id);
	auto&      target        = texture_cache.GetImage(target_id);
	if (!depth.info.HasStencil() || depth.backing.samples != 1u || target.backing.samples != 1u) return;
	const auto width  = std::min(depth.backing.extent.width, target.backing.extent.width);
	const auto height = std::min(depth.backing.extent.height, target.backing.extent.height);
	const auto bytes  = uint64_t {width} * height;
	if (width == 0u || height == 0u) return;
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 8u) {
		std::fprintf(stderr, "WORKAROUND_STENCIL_ALIAS_COPY addr=0x%llx depth=%ux%u alias=%ux%u\n",
		             static_cast<unsigned long long>(target.info.data.address), depth.backing.extent.width,
		             depth.backing.extent.height, target.backing.extent.width, target.backing.extent.height);
	}
	m_context.GetCommandScheduler().EndRendering();
	const auto command = m_context.GetCommandScheduler().Current().Handle();
	if (!m_stencil_copy_buffer || m_stencil_copy_buffer->Size() < bytes) {
		m_stencil_copy_buffer = std::make_unique<Buffer>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(bytes, uint64_t {3840} * 2160));
	}
	const auto          staging = m_stencil_copy_buffer->Handle();
	vk::BufferImageCopy stencil_copy {};
	stencil_copy.bufferRowLength  = width;
	stencil_copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
	stencil_copy.imageExtent      = vk::Extent3D {width, height, 1};
	vk::BufferImageCopy color_copy         = stencil_copy;
	color_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	depth.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	command.copyImageToBuffer(depth.backing.image, vk::ImageLayout::eTransferSrcOptimal, staging, 1, &stencil_copy);
	command.pipelineBarrier2(dependency);
	target.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyBufferToImage(staging, target.backing.image, vk::ImageLayout::eTransferDstOptimal, 1, &color_copy);
	command.pipelineBarrier2(dependency);
	target.MarkGpuModified();
}

void RenderExecutor::CopyStencilShadow(vk::CommandBuffer command, const StencilShadow& pair, bool to_stencil) {
	auto&      texture_cache = m_context.GetTextureCache();
	auto&      shadow        = texture_cache.GetImage(pair.shadow);
	auto&      depth         = texture_cache.GetImage(pair.depth);
	const auto width         = shadow.backing.extent.width;
	const auto height        = shadow.backing.extent.height;
	const auto bytes         = uint64_t {width} * height;
	if (!m_stencil_copy_buffer || m_stencil_copy_buffer->Size() < bytes) {
		m_stencil_copy_buffer = std::make_unique<Buffer>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
	}
	const auto          staging = m_stencil_copy_buffer->Handle();
	vk::BufferImageCopy stencil_copy {};
	stencil_copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
	stencil_copy.imageExtent      = vk::Extent3D {width, height, 1};
	vk::BufferImageCopy color_copy         = stencil_copy;
	color_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	if (to_stencil) {
		shadow.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
		command.copyImageToBuffer(shadow.backing.image, vk::ImageLayout::eTransferSrcOptimal, staging, 1,
		                          &color_copy);
		command.pipelineBarrier2(dependency);
		depth.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyBufferToImage(staging, depth.backing.image, vk::ImageLayout::eTransferDstOptimal, 1,
		                          &stencil_copy);
	} else {
		depth.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
		command.copyImageToBuffer(depth.backing.image, vk::ImageLayout::eTransferSrcOptimal, staging, 1,
		                          &stencil_copy);
		command.pipelineBarrier2(dependency);
		shadow.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyBufferToImage(staging, shadow.backing.image, vk::ImageLayout::eTransferDstOptimal, 1,
		                          &color_copy);
	}
	command.pipelineBarrier2(dependency);
}

void RenderExecutor::FlushStencilShadows(vk::CommandBuffer command) {
	static std::atomic_uint sampled {0};
	auto& texture_cache = m_context.GetTextureCache();
	for (const auto& pair: m_stencil_shadows) {
		// A depth image deleted since the pair was made has no
		// stencil left to take the shadow: there is nothing to copy back.
		if (const auto* depth = texture_cache.m_slot_images.try_get(pair.depth); depth == nullptr || !depth->registered) {
			static std::atomic_uint skipped {0};
			if (skipped.fetch_add(1) < 32u)
				std::fprintf(stderr, "STENCIL_SHADOW_SKIPPED depth image deleted before the copy back (alive=%d)\n",
				             depth != nullptr ? 1 : 0);
			continue;
		}
		if (DebugPresentCount().load() > 300u && sampled.fetch_add(1) < 6u) {
			auto& depth = m_context.GetTextureCache().GetImage(pair.depth);
			DebugQueueSample(m_context.GetGraphics(), command, m_context.GetTextureCache().GetImage(pair.shadow),
			                 "stencil shadow for depth 0x" + std::to_string(depth.info.data.address) + " " +
			                     std::to_string(depth.backing.extent.width) + "x" +
			                     std::to_string(depth.backing.extent.height));
		}
		CopyStencilShadow(command, pair, true);
		static std::atomic_uint dumped {0};
		if (static const char* dir = Common::EnvRead::FrameDir(); // read once: getenv is slow
		    dir != nullptr && DebugPresentCount().load() > 300u && dumped.fetch_add(1) < 4u) {
			DebugQueueFullDump(m_context.GetGraphics(), command, m_context.GetTextureCache().GetImage(pair.depth),
			                   std::string(dir) + "/after_shadow_flush" + std::to_string(dumped.load()));
		}
	}
	static std::atomic_uint logged {0};
	if (!m_stencil_shadows.empty() && logged.fetch_add(1) < 4u) {
		std::fprintf(stderr, "WORKAROUND_STENCIL_STORAGE_SHADOW copied %zu shadow(s) to stencil\n",
		             m_stencil_shadows.size());
	}
	m_stencil_shadows.clear();
}

static vk::Sampler NativeSampler(RenderContext&                       context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		// An integer image under a min/max FILTER_MODE: RDNA returns the minimum or maximum of the whole filter
		// footprint, which point sampling one texel of it does not give (ResourceMaterialization.cpp
		// ClassifySamplerFor): stopped by name.
		if (descriptor.FilterMode() != 0u &&
		    (descriptor.XyMagFilter() != 0u || descriptor.XyMinFilter() != 0u || descriptor.MipFilter() == 2u)) {
			EXIT("an integer image sampled with a min/max filter mode (mode=%u mag=%u min=%u mip=%u): the hardware "
			     "reduces the footprint, point sampling does not\n",
			     static_cast<uint32_t>(descriptor.FilterMode()), static_cast<uint32_t>(descriptor.XyMagFilter()),
			     static_cast<uint32_t>(descriptor.XyMinFilter()), static_cast<uint32_t>(descriptor.MipFilter()));
		}
		Common::WorkaroundHits::SetPointFilteringCounted(descriptor); // KYTY_TRACE_PERF: workaround_hits
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

RenderExecutor::PlanImageFlags* RenderExecutor::PrepFlags() const {
	return m_prep_flags;
}

void RenderExecutor::SetPrepFlags(PlanImageFlags* flags) {
	m_prep_flags = flags;
}

namespace {
// Image::plan_uses: counted with atomics.
void PlanUsesAdd(uint16_t& uses) {
	std::atomic_ref<uint16_t>(uses).fetch_add(1, std::memory_order_relaxed);
}
void PlanUsesSub(uint16_t& uses) {
	std::atomic_ref<uint16_t> ref(uses);
	uint16_t                  current = ref.load(std::memory_order_relaxed);
	while (current != 0 && !ref.compare_exchange_weak(current, static_cast<uint16_t>(current - 1u), std::memory_order_relaxed)) {
	}
}
} // namespace

ImageBinding& RenderExecutor::PlanFlags(PlanImageFlags& plan, ImageId id) {
	for (auto& [flag_id, binding]: plan.flags) {
		if (flag_id == id) return binding;
	}
	if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) PlanUsesAdd(image->plan_uses);
	return plan.flags.emplace_back(id, ImageBinding {}).second;
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	const auto apply = [&](ImageBinding& binding) {
		if (binding.is_bound) {
			binding.force_general |= binding.shader_write != storage;
		}
		binding.is_bound = true;
		binding.shader_write |= storage;
	};
	const uint64_t mix = (static_cast<uint64_t>(id.index) << 2u) | (storage ? 1u : 0u);
	if (auto* prep = PrepFlags(); prep != nullptr) { // the plan flags
		apply(PlanFlags(*prep, id));
		prep->bound.push_back(id);
		prep->hash = (prep->hash ^ mix) * 0x100000001b3ull;
		return;
	}
	apply(image.binding);
	m_bound_images.push_back(id);
	m_bound_hash = (m_bound_hash ^ mix) * 0x100000001b3ull;
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto&          image = m_context.GetTextureCache().GetImage(id);
	const uint64_t mix   = (static_cast<uint64_t>(id.index) << 2u) | 2u;
	if (auto* prep = PrepFlags(); prep != nullptr) { // the plan flags
		PlanFlags(*prep, id).is_target = true;
		prep->bound.push_back(id);
		prep->hash = (prep->hash ^ mix) * 0x100000001b3ull;
		return;
	}
	image.binding.is_target = true;
	m_bound_images.push_back(id);
	m_bound_hash = (m_bound_hash ^ mix) * 0x100000001b3ull;
}

// `image->binding = {}` of a bound image whose binding is replaced during the preparation.
void RenderExecutor::ClearImageBinding(Image& image, ImageId id) {
	image.binding = {};
	if (auto* prep = PrepFlags(); prep != nullptr) { // the plan flags: the plan forgets its flags too
		auto& flags = prep->flags;
		for (size_t i = 0; i < flags.size(); i++) {
			if (flags[i].first != id) continue;
			PlanUsesSub(image.plan_uses);
			flags.erase(flags.begin() + static_cast<std::ptrdiff_t>(i));
			break;
		}
	}
}

// A draw entry, before its render state is resolved: its flags go to the next plan.
void RenderExecutor::BeginDrawPlan() {
	if (m_prep_flags != nullptr) return; // an unfinished plan keeps collecting, as m_bound_images
	m_prep_flags = &SlotFlags(*m_current_slot);
	if (!m_texture_plan_transfer_set) {
		m_texture_plan_transfer_set = true;
		m_context.GetTextureCache().on_plan_target_transfer = [this](ImageId from, ImageId to) {
			auto* prep = PrepFlags();
			if (prep == nullptr) return;
			bool target = false;
			for (const auto& [image_id, flags]: prep->flags) {
				if (image_id == from) target |= flags.is_target;
			}
			if (target) PlanFlags(*prep, to).is_target = true;
		};
	}
}

// The commit of the draw being prepared: its flags are applied to the images.
void RenderExecutor::ApplyPlanFlags() {
	auto* const prep = PrepFlags(); // the committing thread's plan
	if (prep == nullptr) return;
	auto& plan  = *prep;
	auto& cache = m_context.GetTextureCache();
	for (const auto& [id, flags]: plan.flags) {
		if (auto* image = cache.m_slot_images.try_get(id); image != nullptr) {
			auto applied         = flags;
			applied.needs_rebind = image->binding.needs_rebind;
			// The cache may have set attachment fields of a target outside a draw: kept as the image had them.
			applied.attachment_layout = image->binding.attachment_layout;
			applied.attachment_access = image->binding.attachment_access;
			image->binding            = applied;
		}
	}
	m_bound_images.insert(m_bound_images.end(), plan.bound.begin(), plan.bound.end());
	m_bound_hash = plan.hash;
	m_applied_flags = prep;
	SetPrepFlags(nullptr);
}

void RenderExecutor::ReleasePlanFlags(PlanImageFlags& plan) {
	auto& cache = m_context.GetTextureCache();
	for (const auto& [id, flags]: plan.flags) {
		(void)flags;
		if (auto* image = cache.m_slot_images.try_get(id); image != nullptr) PlanUsesSub(image->plan_uses);
	}
	plan.flags.clear();
	plan.bound.clear();
	plan.hash = 0;
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
	m_bound_hash = 0;
	// The plan flags: the committed plan, and a plan that ended before its commit (a dropped draw).
	if (m_applied_flags != nullptr) {
		ReleasePlanFlags(*m_applied_flags);
		m_applied_flags = nullptr;
	}
	if (auto* const prep = PrepFlags(); prep != nullptr) {
		ReleasePlanFlags(*prep);
		SetPrepFlags(nullptr);
	}
}

static uint32_t BindlessBindingFor(const ShaderRecompiler::IR::ImageResource& resource) {
	using ShaderRecompiler::Decoder::ImageDimension;
	if (resource.cube) return ShaderRecompiler::IR::BindlessImagesCube;
	if (resource.dimension == ImageDimension::Dim3D) return ShaderRecompiler::IR::BindlessImages3D;
	if (resource.dimension == ImageDimension::Dim2DArray ||
	    resource.dimension == ImageDimension::Dim2DMsaaArray)
		return ShaderRecompiler::IR::BindlessImages2DArray;
	return ShaderRecompiler::IR::BindlessImages2D;
}

static bool BindlessShapeCompatible(const ShaderTextureResource& descriptor, uint32_t binding);

// A heap entry the bindless array of this binding can hold: a float-sampled, single-sample
// texture of a matching shape. Anything else stays the placeholder (KytyPS5 PR #937).
static bool BindlessCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	if (descriptor.IsNull() || Prospero::SampledTextureNumericClass(descriptor.Format()) !=
	                               Prospero::TextureNumericClass::Float) {
		return false;
	}
	return BindlessShapeCompatible(descriptor, binding);
}

// KYTY_PERF_BINDLESS_CLASSES: what an array view of a table's class can hold. Float tables: BindlessCompatible. An
// integer table: a T# of exactly its class, in a format the host views as is (11_11_10_UINT, which the host reads as
// 32_UINT and unpacks in the shader of a bound texture, has no bindless view), of a shape the array holds.
static bool BindlessCompatibleClass(const ShaderTextureResource& descriptor, uint32_t binding, uint32_t numeric_class) {
	if (!BindlessTable::IntegerClass(numeric_class)) return BindlessCompatible(descriptor, binding);
	const auto format = descriptor.Format();
	return !descriptor.IsNull() &&
	       static_cast<uint32_t>(Prospero::SampledTextureNumericClass(format)) == numeric_class &&
	       Prospero::RemapTextureFormat(format) == format && BindlessShapeCompatible(descriptor, binding);
}

static bool BindlessShapeCompatible(const ShaderTextureResource& descriptor, uint32_t binding) {
	switch (descriptor.Type()) {
		case Prospero::ImageType::kColor2D:
		case Prospero::ImageType::kColor2DArray:
			return binding == ShaderRecompiler::IR::BindlessImages2D ||
			       binding == ShaderRecompiler::IR::BindlessImages2DArray;
		case Prospero::ImageType::kCube: return binding == ShaderRecompiler::IR::BindlessImagesCube;
		case Prospero::ImageType::kColor3D: return binding == ShaderRecompiler::IR::BindlessImages3D;
		default: return false;
	}
}

// KYTY_BINDLESS statistics (stderr every 300 presented frames): why heap entries stay unresolved.
namespace {
enum BindlessStat : uint32_t {
	StatResolved, StatReadFail, StatIncompatible, StatBadAddress, StatSwizzle, StatSoftNull, StatNoView,
	StatNoSlot, StatCached, StatStale, StatRemapped, StatRequests, StatRequestsSettled, StatDeferred, StatPrefilled, StatReleased,
	StatCount
};
std::array<uint64_t, StatCount>          g_bindless_stats {};
std::map<std::pair<uint32_t, uint32_t>, uint64_t> g_bindless_incompatible; // {type, format} -> count
} // namespace

// Heap or record words of the bindless table.
static bool ReadBindlessWords(uint64_t address, void* data, uint64_t size) {
	if (size == 0) return false;
	return Libs::LibKernel::Memory::TryReadBacking(address, data, size);
}

// Words that start an image descriptor (T# type 8..15) with sane array and mip ranges: a key can
// land on words that only look like one (a false match once made a 2641-layer image and ended
// the run).
static bool LooksLikeImageDescriptor(const uint32_t* words, bool r128) {
	if (words[0] == 0 || (words[3] >> 28u) < 8u) return false;
	ShaderTextureResource probe {};
	std::memcpy(probe.fields, words, (r128 ? 4u : 8u) * sizeof(uint32_t)); // R128: words 4..7 read as zero
	return !(probe.Depth() >= 2048u || probe.BaseArray5() > probe.Depth() || probe.LastLevel() < probe.BaseLevel() ||
	         probe.LastLevel() > 14u);
}

// KYTY_TRACE_BINDLESS_KEYS: why ResolveBindlessKey runs when a draw already sampled the key
// ("feedback"); nullptr for keys resolved ahead of use.
static const char* g_bindless_origin = nullptr;

static const char* BindlessStatName(BindlessStat stat) {
	switch (stat) {
		case StatReadFail: return "read_fail";
		case StatIncompatible: return "incompatible";
		case StatBadAddress: return "bad_address";
		case StatSwizzle: return "swizzle";
		case StatSoftNull: return "soft_null";
		case StatNoView: return "no_view";
		case StatNoSlot: return "no_slot";
		default: return "other";
	}
}

static bool BindlessReject(BindlessTable& table, BindlessTable::Heap& heap, uint32_t entry, BindlessStat stat,
                           const std::array<uint32_t, 8>* words = nullptr) {
	g_bindless_stats[stat]++;
	table.SetTranslation(heap, entry, 0u);
	if (words != nullptr) heap.rejected_entries.push_back({entry, 0u, {}, *words});
	if (g_bindless_origin != nullptr && BindlessTable::KeyTraceEnabled())
		table.TraceKeyRejected(heap, entry, words != nullptr ? *words : std::array<uint32_t, 8> {},
		                       BindlessStatName(stat), g_bindless_origin);
	return false;
}

// A ring heap is rewritten by the guest every frame at the same address: entries whose words
// changed since they were resolved (or rejected) go back to pending, once per frame.
bool RenderExecutor::ValidateBindlessHeap(BindlessTable::Heap& heap) {
	auto&                   table = m_context.GetBindlessTable();
	std::array<uint32_t, 8> words {};
	const auto              unchanged = [&](const BindlessTable::Heap::ResolvedEntry& item) {
		if (!ReadBindlessWords(heap.base + uint64_t {item.entry} * 4u, words.data(), sizeof(words)))
			return false;
		if (heap.resource.r128) std::fill(words.begin() + 4, words.end(), 0u);
		return words == item.descriptor;
	};
	const auto make_pending = [&](uint32_t entry) {
		heap.slots[entry]   = 0;
		heap.settled[entry] = 0;
		table.SetTranslation(heap, entry, ShaderRecompiler::IR::BindlessPending);
	};
	std::vector<uint32_t> changed;
	size_t kept = 0;
	for (size_t i = 0; i < heap.resolved_entries.size(); i++) {
		const auto item = heap.resolved_entries[i];
		if (unchanged(item)) {
			heap.resolved_entries[kept++] = item;
			continue;
		}
		g_bindless_stats[StatStale]++;
		table.RemoveImageReference(item.image, heap, item.entry);
		if (const auto it = std::find(heap.resolved.begin(), heap.resolved.end(), item.image);
		    it != heap.resolved.end())
			heap.resolved.erase(it);
		if (item.entry < heap.settled.size()) {
			table.TraceKeyDropped(heap, item.entry, item.descriptor, item.slot, "stale");
			make_pending(item.entry);
			changed.push_back(item.entry);
		}
	}
	heap.resolved_entries.resize(kept);
	kept = 0;
	for (size_t i = 0; i < heap.rejected_entries.size(); i++) {
		const auto item = heap.rejected_entries[i];
		if (unchanged(item)) {
			heap.rejected_entries[kept++] = item;
			continue;
		}
		g_bindless_stats[StatStale]++;
		if (item.entry < heap.settled.size()) {
			make_pending(item.entry);
			changed.push_back(item.entry);
		}
	}
	heap.rejected_entries.resize(kept);
	// Re-resolve now instead of waiting for feedback: a moving camera rewrites ring heaps every
	// frame, and a key that waits for feedback samples the placeholder for two or three frames.
	for (const auto entry: changed)
		if (heap.settled[entry] == 0) (void)ResolveBindlessKey(heap, entry, true);
	return !changed.empty();
}

// A new or rewritten heap: every entry holding a descriptor some heap already resolved gets its
// slot now, before the first draw samples it (a pending key samples the placeholder, and in a
// vertex shader that misplaces geometry, e.g. terrain heights).
void RenderExecutor::PrefillBindlessHeap(BindlessTable::Heap& heap, uint32_t words) {
	auto& table = m_context.GetBindlessTable();
	if (table.descriptor_slots.empty() || words < 8u) return;
	words = std::min<uint32_t>(words, static_cast<uint32_t>(heap.settled.size()));
	m_bindless_words.resize(words);
	if (!ReadBindlessWords(heap.base, m_bindless_words.data(), words * sizeof(uint32_t))) return;
	auto& texture_cache = m_context.GetTextureCache();
	std::pair<uint32_t, std::array<uint32_t, 8>> key {heap.binding, {}};
	for (uint32_t entry = 0; entry + 8u <= words; entry++) {
		if (heap.settled[entry] != 0 || m_bindless_words[entry] == 0) continue;
		std::copy_n(m_bindless_words.begin() + entry, 8u, key.second.begin());
		if (heap.resource.r128) std::fill(key.second.begin() + 4, key.second.end(), 0u);
		const auto found = table.descriptor_slots.find(key);
		if (found == table.descriptor_slots.end()) continue;
		const auto* image = texture_cache.m_slot_images.try_get(found->second.image);
		if (image == nullptr || !image->registered) continue;
		heap.settled[entry] = 1;
		heap.slots[entry]   = found->second.slot;
		heap.resolved.push_back(found->second.image);
		heap.resolved_entries.push_back({entry, found->second.slot, found->second.image, key.second});
		table.AddImageReference(found->second.image, heap, entry);
		table.SetTranslation(heap, entry, found->second.slot);
		table.TraceKeyResolved(heap, entry, key.second, found->second.slot, "prefill");
		g_bindless_stats[StatPrefilled]++;
	}
}

void RenderExecutor::RefreshBindlessDescriptor(BindlessTable::Heap& heap, const std::array<uint32_t, 8>& key_words,
                                               BindlessTable::DescriptorSlot& cached_entry) {
	auto&                                 table         = m_context.GetBindlessTable();
	auto&                                 texture_cache = m_context.GetTextureCache();
	// key_words and cached_entry live in the descriptor_slots node, which the lookups below can
	// erase: they may drop the cached image (an expanded or grown texture), and the table then
	// erases its entries (OnImageUnregistered). Keep a copy and look the entry up again.
	const auto                            words = key_words;
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count = 8u;
	std::copy_n(words.begin(), 8u, value.dwords.begin());
	m_soft_texture_resolve = true;
	auto binding           = ResolveTexture(heap.resource, value);
	m_soft_texture_resolve = false;
	auto* image            = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty()) return;
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || view == nullptr) return;
	const auto found = table.descriptor_slots.find({heap.binding, words});
	if (found == table.descriptor_slots.end() || &found->second != &cached_entry) return;
	auto& cached = found->second;
	if (image->IsGpuModified()) cached.gpu_written = true;
	if (cached.gpu_written) BindImage(binding.image_id, false);
	if (binding.image_id == cached.image) return;
	g_bindless_stats[StatRemapped]++;
	const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
	table.WriteSlot(heap.binding, cached.slot, view, layout);
	image->bindless_pinned = true;
	const auto old_image   = cached.image;
	cached.image           = binding.image_id;
	for (auto& other: table.Heaps()) {
		for (auto& entry: other.resolved_entries) {
			if (entry.slot != cached.slot || entry.image != old_image || other.binding != heap.binding) continue;
			entry.image = binding.image_id;
			table.RemoveImageReference(old_image, other, entry.entry);
			table.AddImageReference(binding.image_id, other, entry.entry);
			if (const auto it = std::find(other.resolved.begin(), other.resolved.end(), old_image);
			    it != other.resolved.end())
				*it = binding.image_id;
		}
	}
}

// KYTY_BINDLESS: entry = DWORD index of a descriptor in its heap (the shader key / 4).
bool RenderExecutor::ResolveBindlessKey(BindlessTable::Heap& heap, uint32_t entry, bool immediate) {
	auto&                                 table = m_context.GetBindlessTable();
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count   = 8u;
	const auto address  = heap.base + static_cast<uint64_t>(entry) * 4u;
	heap.settled[entry] = 1;
	m_bindless_cached   = false;
	if (!ReadBindlessWords(address, value.dwords.data(), 32u)) {
		return BindlessReject(table, heap, entry, StatReadFail);
	}
	// A 128-bit (R128) image instruction reads only the first four descriptor words; the rest
	// are zero, as in the bound-descriptor path.
	if (heap.resource.r128) std::fill(value.dwords.begin() + 4, value.dwords.begin() + 8, 0u);
	std::array<uint32_t, 8> words {};
	std::copy_n(value.dwords.begin(), 8u, words.begin());
	auto& texture_cache = m_context.GetTextureCache();
	const auto finish = [&](uint32_t slot, ImageId id) {
		heap.slots[entry] = slot;
		heap.resolved.push_back(id);
		heap.resolved_entries.push_back({entry, slot, id, words});
		table.AddImageReference(id, heap, entry);
		table.SetTranslation(heap, entry, slot);
		if (BindlessTable::KeyTraceEnabled())
			table.TraceKeyResolved(heap, entry, words, slot,
			                       g_bindless_origin != nullptr ? g_bindless_origin
			                       : m_bindless_cached        ? "cached"
			                                                  : "immediate");
		return true;
	};
	// The same descriptor seen through any heap entry already has a slot.
	if (const auto cached = table.descriptor_slots.find({heap.binding, words});
	    cached != table.descriptor_slots.end()) {
		const auto* image = texture_cache.m_slot_images.try_get(cached->second.image);
		if (image != nullptr && image->registered) {
			g_bindless_stats[StatCached]++;
			m_bindless_cached = true;
			return finish(cached->second.slot, cached->second.image);
		}
		table.descriptor_slots.erase(cached);
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	// A key can land on heap words that are not a live descriptor: reject texture or metadata
	// addresses outside guest memory instead of creating an image there.
	uint32_t probe = 0;
	if (!BindlessCompatible(descriptor, heap.binding)) {
		if (!descriptor.IsNull())
			g_bindless_incompatible[{static_cast<uint32_t>(descriptor.Type()),
			                         static_cast<uint32_t>(descriptor.Format())}]++;
		return BindlessReject(table, heap, entry, StatIncompatible, &words);
	}
	if (!GuestRange {descriptor.Base40(), 1u}.Valid() ||
	    !Libs::LibKernel::Memory::TryReadBacking(descriptor.Base40(), &probe, sizeof(probe)) ||
	    (descriptor.MetaCompress() && descriptor.MetaAddr() != 0 &&
	     !GuestRange {descriptor.MetaAddr() << 8u, 1u}.Valid())) {
		return BindlessReject(table, heap, entry, StatBadAddress, &words);
	}
	for (uint32_t channel = 0; channel < 4u; channel++) {
		const auto selector = GetDstSel(descriptor.DstSelXYZW(), channel);
		if (selector == 2u || selector == 3u) return BindlessReject(table, heap, entry, StatSwizzle, &words);
	}
	m_soft_texture_resolve = true;
	auto binding           = ResolveTexture(heap.resource, value);
	m_soft_texture_resolve = false;
	auto* image            = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || image->info.data.Empty())
		return BindlessReject(table, heap, entry, StatSoftNull, &words);
	BindImage(binding.image_id, false);
	const auto view = texture_cache.FindTexture(binding.image_id, binding.desc);
	image           = texture_cache.m_slot_images.try_get(binding.image_id);
	if (image == nullptr || view == nullptr) return BindlessReject(table, heap, entry, StatNoView, &words);
	const auto slot = table.AllocateSlot(heap.binding);
	if (slot == 0) {
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::BindlessNoSlot); // KYTY_TRACE_PERF
		return BindlessReject(table, heap, entry, StatNoSlot, &words);
	}
	const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
	table.WriteSlot(heap.binding, slot, view, layout);
	image->bindless_pinned = true;
	table.descriptor_slots[{heap.binding, words}] = {slot, binding.image_id, 0u, 0u, image->IsGpuModified()};
	g_bindless_stats[StatResolved]++;
	return finish(slot, binding.image_id);
}

// Textures no draw sampled for a while give their slot back and are unpinned, so the texture
// cache can free them (a later draw that needs one resolves it again).
void RenderExecutor::ReleaseUnusedBindlessSlots(uint64_t frame) {
	constexpr uint64_t UnusedFrames = 300;
	auto&              table        = m_context.GetBindlessTable();
	if (frame < UnusedFrames) return;
	std::unordered_set<uint64_t> freed; // binding << 32 | slot
	std::vector<ImageId>         released;
	for (auto it = table.descriptor_slots.begin(); it != table.descriptor_slots.end();) {
		if (it->second.refreshed_frame + UnusedFrames > frame) {
			++it;
			continue;
		}
		freed.insert((uint64_t {it->first.first} << 32u) | it->second.slot);
		released.push_back(it->second.image);
		table.FreeSlot(it->first.first, it->second.slot);
		it = table.descriptor_slots.erase(it);
	}
	if (freed.empty()) return;
	g_bindless_stats[StatReleased] += freed.size();
	for (auto& heap: table.Heaps()) {
		if (heap.entries == 0) continue;
		size_t kept = 0;
		for (size_t i = 0; i < heap.resolved_entries.size(); i++) {
			const auto item = heap.resolved_entries[i];
			if (!freed.contains((uint64_t {heap.binding} << 32u) | item.slot)) {
				heap.resolved_entries[kept++] = item;
				continue;
			}
			table.RemoveImageReference(item.image, heap, item.entry);
			if (const auto found = std::find(heap.resolved.begin(), heap.resolved.end(), item.image);
			    found != heap.resolved.end())
				heap.resolved.erase(found);
			if (item.entry < heap.settled.size()) {
				table.TraceKeyDropped(heap, item.entry, item.descriptor, item.slot, "released");
				heap.slots[item.entry]   = 0;
				heap.settled[item.entry] = 0;
				table.SetTranslation(heap, item.entry, ShaderRecompiler::IR::BindlessPending);
			}
		}
		heap.resolved_entries.resize(kept);
	}
	// An image stays pinned while another descriptor still maps to it.
	std::unordered_set<uint64_t> in_use;
	for (const auto& [key, cached]: table.descriptor_slots)
		in_use.insert((uint64_t {cached.image.generation} << 32u) | cached.image.index);
	for (const auto id: released) {
		if (in_use.contains((uint64_t {id.generation} << 32u) | id.index)) continue;
		if (table.TableUsesImage(id)) continue; // a table slot still shows it
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr)
			image->bindless_pinned = false;
	}
}

void RenderExecutor::ResolveBindlessRequests() {
	const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
	if (frame == m_bindless_frame) return;
	auto& table     = m_context.GetBindlessTable();
	auto& scheduler = m_context.GetCommandScheduler();
	// The flags come from a snapshot recorded in an earlier frame. Until the GPU has executed
	// it, check again at the next bindless draw instead of waiting.
	if (table.SnapshotRecorded() && !table.SnapshotReady(scheduler)) return;
	m_bindless_frame = frame;
	table.ReportDroppedKeys(); // KYTY_TRACE_BINDLESS_KEYS
	if (frame / 60u != m_bindless_stats_frame / 60u) {
		ReleaseUnusedBindlessSlots(frame);
		m_bindless_stats_frame = frame;
		uint64_t pending = 0, settled = 0;
		for (const auto& heap: table.Heaps())
			for (const auto value: heap.settled) settled += value;
		std::string incompatible;
		for (const auto& [key, count]: g_bindless_incompatible)
			incompatible += fmt::format(" t{}f{}:{}", key.first, key.second, count);
		if (Common::EnvRead::PerfLines()) std::fprintf(stderr,
		             "BINDLESS_STATS frame=%llu heaps=%zu settled=%llu resolved=%llu read_fail=%llu "
		             "incompatible=%llu bad_address=%llu swizzle=%llu soft_null=%llu no_view=%llu no_slot=%llu cached=%llu stale=%llu remapped=%llu requests=%llu requests_settled=%llu deferred=%llu prefilled=%llu released=%llu slots=%zu "
		             "unregistered=%llu unregistered_keys=%llu heap_fail=%llu heap_evict=%llu%s\n",
		             static_cast<unsigned long long>(frame), table.Heaps().size(),
		             static_cast<unsigned long long>(settled),
		             static_cast<unsigned long long>(g_bindless_stats[StatResolved]),
		             static_cast<unsigned long long>(g_bindless_stats[StatReadFail]),
		             static_cast<unsigned long long>(g_bindless_stats[StatIncompatible]),
		             static_cast<unsigned long long>(g_bindless_stats[StatBadAddress]),
		             static_cast<unsigned long long>(g_bindless_stats[StatSwizzle]),
		             static_cast<unsigned long long>(g_bindless_stats[StatSoftNull]),
		             static_cast<unsigned long long>(g_bindless_stats[StatNoView]),
		             static_cast<unsigned long long>(g_bindless_stats[StatNoSlot]),
		             static_cast<unsigned long long>(g_bindless_stats[StatCached]),
		             static_cast<unsigned long long>(g_bindless_stats[StatStale]),
		             static_cast<unsigned long long>(g_bindless_stats[StatRemapped]),
		             static_cast<unsigned long long>(g_bindless_stats[StatRequests]),
		             static_cast<unsigned long long>(g_bindless_stats[StatRequestsSettled]),
		             static_cast<unsigned long long>(g_bindless_stats[StatDeferred]),
		             static_cast<unsigned long long>(g_bindless_stats[StatPrefilled]),
		             static_cast<unsigned long long>(g_bindless_stats[StatReleased]), table.descriptor_slots.size(),
		             static_cast<unsigned long long>(table.UnregisteredImages()),
		             static_cast<unsigned long long>(table.UnregisteredKeys()),
		             static_cast<unsigned long long>(table.HeapFailures()),
		             static_cast<unsigned long long>(table.HeapEvictions()), incompatible.c_str());
		(void)pending;
	}
	if (!table.SnapshotRecorded()) {
		table.RecordFeedbackSnapshot(scheduler);
		return;
	}
	table.ConsumeSnapshot();
	for (auto& heap: table.Heaps()) {
		m_bindless_requests.clear();
		table.TakeRequests(heap, m_bindless_requests);
		for (const auto entry: m_bindless_requests) {
			g_bindless_stats[StatRequests]++;
			if (entry < heap.settled.size() && heap.settled[entry] != 0) g_bindless_stats[StatRequestsSettled]++;
			if (entry >= heap.settled.size() || heap.settled[entry] != 0) continue;
			g_bindless_origin = "feedback";
			(void)ResolveBindlessKey(heap, entry);
			g_bindless_origin = nullptr;
		}
	}
	table.RecordFeedbackSnapshot(scheduler);
}

// KYTY_TRACE_LONG_PREP: PrepareBindlessHeaps' parts (t_long_prep_parts 13-20): 0 head, 1 requests, 2 re-resolve and table
// frame, 3 table path per use, 4 heap find/validate/prefill/inherit, 5 refresh loop, 6 image list, 7 tail.
namespace {
struct BindlessSplit {
	bool                                  on = LongPrepPartsOn();
	std::chrono::steady_clock::time_point last {};
	BindlessSplit() {
		if (on) last = std::chrono::steady_clock::now();
	}
	void Mark(int part) {
		if (!on) return;
		const auto now = std::chrono::steady_clock::now();
		const auto ns  = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count());
		t_long_prep_parts[13u + static_cast<size_t>(part)] += ns; // KYTY_TRACE_LONG_PREP
		last = now;
	}
};
} // namespace

void RenderExecutor::PrepareBindlessHeaps(const ShaderStageRuntime& runtime,
                                          PreparedBindings& prepared) {
	BindlessSplit      bsplit; // KYTY_TRACE_LONG_PREP
	prepared.bindless_patches.clear();
	prepared.bindless_images.clear();
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       table    = m_context.GetBindlessTable();
	if (snapshot.bindless_heaps.empty() || !table.Enabled()) return;
	m_bindless_draw++;
	bsplit.Mark(0);
	{
		KYTY_PROFILER_BLOCK("Bindless::Requests");
		ResolveBindlessRequests();
	}
	bsplit.Mark(1);
	m_bindless_draw_heaps.clear();
	// Tables are translated before the draw; only an empty use stays a heap (bindless E2c:
	// any other use the table path cannot take stops the emulator, PrepareBindlessTable).
	const bool tables = table.TablesEnabled();
	if (tables) PrepareTablesFrame(m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed) + 1u);
	bsplit.Mark(2);
	for (const auto& use: snapshot.bindless_heaps) {
		if (use.image >= program.info.images.size()) continue;
		const auto& resource = program.info.images[use.image];
		if (tables) {
			const bool taken = PrepareBindlessTable(program.shader_hash, use, resource, prepared);
			bsplit.Mark(3);
			if (taken) continue;
		}
		// KYTY_PERF_BINDLESS_CLASSES: an integer resource has no heap path (heaps resolve float views only). Its mapping
		// words stay zero, so every key is outside its heap and reads the zero placeholder of its class (BindlessSlot).
		if (BindlessTable::IntegerClass(static_cast<uint32_t>(resource.numeric_class))) {
			table.GetClassStats().heap_skipped++;
			continue;
		}
		const auto  entries  = static_cast<uint32_t>(std::min<uint64_t>(use.size / 4u, 1u << 18u));
		// An empty table has no region (FindOrCreateHeap returns null); not a full translation buffer.
		if (entries == 0) continue;
		const auto frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed);
		auto* heap = table.FindOrCreateHeap(use.base, 0u, BindlessBindingFor(resource), entries, resource,
		                                    frame + 1u);
		if (heap == nullptr) {
			static std::atomic_uint logged {0};
			if (logged.fetch_add(1) < 8u)
				std::fprintf(stderr, "BINDLESS_TRANSLATION_FULL heaps=%zu base=0x%llx entries=%u\n", table.Heaps().size(),
				             static_cast<unsigned long long>(use.base), entries);
			continue; // region 0, count 0: every key samples the placeholder
		}
		m_bindless_draw_heaps.push_back(heap);
		const bool first_check   = heap->validated_frame != frame + 1u;
		if (first_check) {
			heap->validated_frame = frame + 1u;
			const bool fresh   = heap->created_frame == frame + 1u;
			const bool changed = !fresh && ValidateBindlessHeap(*heap);
			if (fresh || changed) PrefillBindlessHeap(*heap, std::min(entries, heap->entries));
		}
		// A prefilled heap never inherits.
		const bool inherit = heap->created_frame == frame + 1u && heap->resolved_entries.empty();
		if (inherit) {
			// The guest rewrites this heap at a new address (per-frame ring): inherit the slots of
			// the most recent heap of the same shape whose descriptors are still the same.
			const BindlessTable::Heap* source = nullptr;
			for (const auto& other: table.Heaps()) {
				if (&other == heap || other.binding != heap->binding || other.entries != heap->entries ||
				    other.resolved_entries.empty())
					continue;
				if (source == nullptr || other.last_used > source->last_used) source = &other;
			}
			if (source != nullptr) {
				// The keys the previous heap was asked for are likely asked for again: resolve them
				// now from this heap's own words (known descriptors reuse their slots).
				std::vector<uint32_t> keys;
				for (const auto& item: source->resolved_entries) keys.push_back(item.entry);
				for (const auto& item: source->rejected_entries) keys.push_back(item.entry);
				for (const auto entry: keys)
					if (entry < heap->settled.size() && heap->settled[entry] == 0)
						(void)ResolveBindlessKey(*heap, entry, true);
			}
		}
		bsplit.Mark(4);
		// A bound texture is looked up again by every draw: the texture cache then refreshes CPU
		// writes and follows GPU-rendered data to the image that holds it now. Bindless textures
		// are looked up again once a frame; a descriptor that now maps to another image gets
		// that image's view in its slot.
		// Textures the GPU renders (depth pyramids, render targets) are looked up again by every
		// draw, so a draw samples what earlier draws of this frame wrote, not last frame's copy.
		KYTY_PROFILER_BLOCK("Bindless::RefreshAndCollect"); // to the end of this heap use
		for (size_t i = 0; i < heap->resolved_entries.size(); i++) {
			const auto item = heap->resolved_entries[i];
			auto found = table.descriptor_slots.find({heap->binding, item.descriptor});
			if (found == table.descriptor_slots.end()) continue;
			auto& cached = found->second;
			if (cached.gpu_written ? cached.refreshed_draw == m_bindless_draw
			                       : cached.refreshed_frame == frame + 1u)
				continue;
			cached.refreshed_frame = frame + 1u;
			cached.refreshed_draw  = m_bindless_draw;
			RefreshBindlessDescriptor(*heap, found->first.second, cached);
		}
		bsplit.Mark(5);
		prepared.bindless_patches.push_back({use.mapping_offset, heap->region, std::min(entries, heap->entries)});
		prepared.bindless_images.insert(prepared.bindless_images.end(), heap->resolved.begin(),
		                                heap->resolved.end());
		bsplit.Mark(6);
	}
	bsplit.Mark(7);
}

// ---- The bindless tables (bindlessTable.h, ShaderIR.h) ----
//
// What the guest does (Ghost of Yotei, read in its code): every run-time indexed texture lives in a table of records
// the CPU rebuilds each frame in a per-frame ring (or in a persistent, double-buffered buffer), written before the
// frame's submit, the shader picking record * stride + field. A table never holds a hole: a texture that is not ready
// is replaced by a default texture's T# before the record is written (records past a capacity-sized table's live count
// can hold old descriptors, which the shader never selects). So everything a draw can select is in guest memory when
// the draw is processed, and nothing on the console ever samples a missing texture. Here the draw's tables are read
// when it is prepared (never earlier: the guest writes them up to its submit), every field of every record is
// translated, and the draw's heap entries point into that translation.

namespace {

// The entry of a table byte that no shader field names: the red placeholder (KYTY_BINDLESS_DEBUG_COLORS=1, "key outside
// its heap"); a key lands there only if a shader's field list missed a field.
constexpr uint32_t TableOutsideEntry = 1u;
// Tables over 1 MiB (the particle and mist tables) take the list path of RefreshTableSlots.
constexpr uint64_t TableMaxBytes = uint64_t {1} << 20u;

// Everything of the shader's image resource that changes the view ResolveTexture makes of a descriptor.
BindlessTable::TableView MakeTableView(const ShaderRecompiler::IR::ImageResource& resource) {
	BindlessTable::TableView view;
	view.binding       = BindlessBindingFor(resource);
	view.dimension     = static_cast<uint32_t>(resource.dimension);
	view.numeric_class = static_cast<uint32_t>(resource.numeric_class);
	view.mip_count     = resource.mip_count;
	view.conversion    = static_cast<uint32_t>(resource.conversion_format);
	view.swizzle       = resource.shader_swizzle;
	view.flags         = (resource.r128 ? 1u : 0u) | (resource.depth_compare ? 2u : 0u) | (resource.cube ? 4u : 0u) |
	             (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic ? 8u : 0u);
	return view;
}

// A use the table path does not take (bindless E2c: the heap path no longer serves these; 0 such uses in the
// open world with the large tables on the table path, run 20261029-122125): the emulator stops with the reason, the
// reader and the table instead of drawing with a heap whose keys read placeholders - no field list for a heap table
// (no_fields), a record table without its table or records (no_layout), more bytes than the translation space
// (too_large), table bytes that cannot be read (read_fail), no arena room once idle tables went (arena_full). An empty
// use (fewer bytes than one descriptor: nothing to translate) goes to the heap path as before
// (BINDLESS_TABLE_FALLBACK, the first 8).
void NoteTableFallback(BindlessTable::TableFallback reason, uint64_t shader_hash,
                       const ShaderRecompiler::IR::BindlessHeapUse& use,
                       const ShaderRecompiler::IR::ImageResource&   resource) {
	static constexpr const char* names[] = {"no_fields", "no_layout", "too_large", "read_fail", "arena_full", "empty"};
	static_assert(std::size(names) == static_cast<size_t>(BindlessTable::TableFallback::Count));
	const auto index = static_cast<size_t>(reason);
	if (reason != BindlessTable::TableFallback::Empty) {
		EXIT("BINDLESS_TABLE_REFUSED reason=%s hash=0x%016llx image=%u base=0x%llx size=0x%llx table=0x%llx "
		     "record=%u stride_dwords=%u fields=%llu complete=%d r128=%d\n",
		     names[index], static_cast<unsigned long long>(shader_hash), use.image,
		     static_cast<unsigned long long>(use.base), static_cast<unsigned long long>(use.size),
		     static_cast<unsigned long long>(use.table_base), use.record_bytes, resource.bindless_stride_dwords,
		     static_cast<unsigned long long>(resource.bindless_fields.size()), resource.bindless_fields_complete ? 1 : 0,
		     resource.r128 ? 1 : 0);
	}
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) >= 8u) return;
	std::fprintf(stderr,
	             "BINDLESS_TABLE_FALLBACK reason=%s hash=0x%016llx image=%u base=0x%llx size=0x%llx table=0x%llx "
	             "record=%u stride_dwords=%u fields=%zu complete=%d r128=%d\n",
	             names[index], static_cast<unsigned long long>(shader_hash), use.image,
	             static_cast<unsigned long long>(use.base), static_cast<unsigned long long>(use.size),
	             static_cast<unsigned long long>(use.table_base), use.record_bytes, resource.bindless_stride_dwords,
	             resource.bindless_fields.size(), resource.bindless_fields_complete ? 1 : 0, resource.r128 ? 1 : 0);
}

} // namespace

bool RenderExecutor::PrepareBindlessTable(uint64_t shader_hash, const ShaderRecompiler::IR::BindlessHeapUse& use,
                                          const ShaderRecompiler::IR::ImageResource& resource,
                                          PreparedBindings&                          prepared) {
	using Fallback      = BindlessTable::TableFallback;
	auto&      bindless = m_context.GetBindlessTable();
	auto&      stats    = bindless.GetTableStats();
	const auto fallback = [&](Fallback reason) {
		stats.fallback[static_cast<size_t>(reason)]++;
		NoteTableFallback(reason, shader_hash, use, resource);
		return false;
	};
	stats.uses++;
	// A table may be as large as the translation space (bindless E1: the large tables too).
	const uint64_t max_bytes        = uint64_t {ShaderRecompiler::IR::WatchdogReportBase} * 4u;
	const uint32_t descriptor_bytes = resource.r128 ? 16u : 32u;
	// The table the heap lies in, and the fields this use reads at byte offsets from the table's first byte.
	uint64_t table_base = 0;
	uint64_t needed     = 0;
	m_table_fields.clear();
	if (resource.bindless_stride_dwords != 0u) {
		// A record table (KYTY_BINDLESS_RECORD_TABLES): the heap is one field, `field` bytes into each of key_count
		// records (ResourceMaterialization: size = (key_count - 1) * stride + 32). The table is the whole records (up
		// to the V#'s end; a record pointer has none), so every field of one table asks for the same bytes.
		const uint32_t stride = resource.bindless_stride_dwords * 4u;
		if (use.table_base == 0u || use.base < use.table_base || use.base - use.table_base >= max_bytes ||
		    use.size < 32u)
			return fallback(Fallback::NoLayout);
		const auto     field   = static_cast<uint32_t>(use.base - use.table_base);
		const uint64_t records = (use.size - 32u) / stride + 1u;
		const uint64_t whole   = use.table_bytes != 0u ? std::min<uint64_t>(records * stride, use.table_bytes)
		                                               : records * stride;
		const uint64_t end     = std::min<uint64_t>(uint64_t {field} + use.size, max_bytes);
		table_base             = use.table_base;
		needed                 = std::max<uint64_t>(whole, uint64_t {field} + use.size);
		m_table_fields.push_back({stride, field, static_cast<uint32_t>(end)});
	} else {
		// A heap table: the whole V#, keyed by byte offset; the fields are the shader's (ResourceTracking). Stride 0:
		// the V#'s record stride (a vector-indexed load multiplies the index by it).
		if (!resource.bindless_fields_complete || resource.bindless_fields.empty()) return fallback(Fallback::NoFields);
		table_base     = use.base;
		needed         = use.size;
		const auto end = static_cast<uint32_t>(std::min<uint64_t>(use.size, max_bytes));
		for (const auto& [stride, offset]: resource.bindless_fields) {
			const uint32_t record = stride != 0u ? stride : use.record_bytes;
			if (record < 16u) return fallback(Fallback::NoFields);
			m_table_fields.push_back({record, offset, end});
		}
	}
	needed &= ~uint64_t {3};
	if (needed < descriptor_bytes) return fallback(Fallback::Empty);
	if (needed > max_bytes) return fallback(Fallback::TooLarge);
	const bool large = needed > TableMaxBytes;
	if (large) stats.large_uses++;

	const uint64_t frame = m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed) + 1u;
	auto&          table = bindless.FindOrCreateTable({table_base, MakeTableView(resource)}, resource);
	table.last_frame     = frame;
	bool fields_changed  = false;
	// KYTY_PERF_BINDLESS_CLASSES: every field remembers the frame a draw last asked for it (TranslateBindlessTable lets
	// unasked ones go) and what its records held (float tables: LearnBindlessClass).
	const bool classes = bindless.ClassesEnabled();
	for (const auto& field: m_table_fields) {
		const auto known = std::ranges::find_if(table.fields, [&](const BindlessTable::TableField& other) {
			return other.stride == field.stride && other.offset == field.offset;
		});
		if (known == table.fields.end()) {
			table.fields.push_back(field);
			if (classes) {
				table.field_frames.resize(table.fields.size() - 1u, frame);
				table.field_frames.push_back(frame);
				table.field_tallies.resize(table.fields.size());
			}
			fields_changed = true;
			continue;
		}
		if (classes && static_cast<size_t>(known - table.fields.begin()) < table.field_frames.size())
			table.field_frames[known - table.fields.begin()] = frame;
		if (known->end < field.end) {
			known->end     = field.end;
			fields_changed = true;
		}
	}
	// Within a presented frame a table only grows (draws reading other fields or more records of it); a new frame
	// starts from what this draw reads (a ring table changes its record count from frame to frame).
	const auto covered =
	    static_cast<uint32_t>(table.covered_frame != frame ? needed : std::max<uint64_t>(table.covered, needed));
	// Guest memory is compared once per command-buffer tick: a slice of a submission ends with a submission, and the
	// guest cannot rewrite a table between two draws of one slice without racing the GPU that reads it.
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	// A slot of the table died (its image was dropped, or the slot released): translate again, at most once per tick
	// (two descriptors whose lookups keep replacing each other's image would otherwise translate at every draw).
	// Bindless E1 v3 cost split: a large table's TableSlotsAlive calls, counted and timed (PERF on).
	const auto slots_alive = [&](const BindlessTable::Table& checked) {
		// The slot journal: the owners' count of its dead slots instead of the walk.
		if (!large || !PerfEnabled()) return bindless.TableSlotsAliveFast(checked);
		const auto alive_start = std::chrono::steady_clock::now();
		const bool alive       = bindless.TableSlotsAliveFast(checked);
		stats.large_alive_calls++;
		stats.large_alive_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - alive_start).count());
		return alive;
	};
	const bool stale      = table.generation != bindless.TableGeneration();
	const bool slots_dead = stale && !slots_alive(table);
	if (stale && !slots_dead) table.generation = bindless.TableGeneration();
	const bool retranslate = slots_dead && table.translated_tick != tick;
	// KYTY_DIAG_TABLES_ASSUME_UNCHANGED=1 (ceiling test only, map 5.11 "ruch przez miasto": the bindless tables were ~20%
	// of the GPU thread in the town): a table with a translation of the same size and fields is taken as unchanged - no
	// read, hash, translation or slot refresh (stale textures may show; the draws per frame must stay). Read once.
	static const bool assume_unchanged = [] {
		const char* value = std::getenv("KYTY_DIAG_TABLES_ASSUME_UNCHANGED");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "DIAG_TABLES_ASSUME_UNCHANGED ceiling: table checks skipped""\n");
		return set;
	}();
	const bool skip_check = assume_unchanged && table.region != 0 && covered == table.covered && !fields_changed &&
	                        !retranslate;
	const bool check_due = !skip_check && (table.checked_tick != tick || covered != table.covered || fields_changed ||
	                                       retranslate || table.region == 0);
	// Tables over 1 MiB: guest bytes equal in place to those of the last check (whose hash is the translation's) need no
	// copy and hash: the hash would be the same.
	const bool in_place_check = check_due && large;
	bool       in_place       = false;
	if (in_place_check && table.region != 0 && covered == table.covered && !fields_changed &&
	    !retranslate && table.hash != 0u && table.check_hash == table.hash && table.check_words.size() == covered / 4u) {
		const auto compare_start = PerfEnabled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		in_place = Libs::LibKernel::Memory::CompareBacking(table_base, table.check_words.data(), covered) == 0;
		if (PerfEnabled()) {
			const auto ns = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - compare_start).count());
			stats.check_ns += ns;
			stats.large_check_ns += ns;
		}
	}
	if (check_due && !in_place) {
		const bool timed = PerfEnabled() || LongPrepPartsOn();
		const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		m_table_words.resize(covered / 4u);
		if (!ReadBindlessWords(table_base, m_table_words.data(), covered)) {
			table.checked_tick = 0;
			return fallback(Fallback::ReadFail);
		}
		const auto hash = XXH3_64bits(m_table_words.data(), covered);
		const auto read_end = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		if (timed) {
			const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(read_end - start).count());
			if (PerfEnabled()) stats.check_ns += ns;
			if (PerfEnabled() && large) stats.large_check_ns += ns; // Bindless E1 v3 cost split
			t_long_prep_parts[30] += ns; // KYTY_TRACE_LONG_PREP
			t_long_prep_parts[33]++;
		}
		if (table.region != 0 && hash == table.hash && covered == table.covered && !fields_changed && !retranslate) {
			stats.unchanged++;
		} else {
			if (timed) {
				t_long_prep_parts[34]++;
				// KYTY_TRACE_LONG_PREP: the first reason of this translation (36 new table, 37 covered size changed,
				// 38 fields changed, 39 slots dead, 40 content hash changed), and the sizes (41 covered, 42 before).
				t_long_prep_parts[table.region == 0       ? 36u
				                  : covered != table.covered ? 37u
				                  : fields_changed           ? 38u
				                  : retranslate              ? 39u
				                                             : 40u]++;
				t_long_prep_parts[41] = covered;
				t_long_prep_parts[42] = table.covered;
			}
			const auto large_start = large ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
			if (!TranslateBindlessTable(table, covered, hash, frame, shader_hash)) return fallback(Fallback::ArenaFull);
			if (large) {
				stats.large_translate_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                                                      std::chrono::steady_clock::now() - large_start)
				                                                      .count());
			}
		}
		if (timed)
			t_long_prep_parts[31] += static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - read_end).count());
		// Tables over 1 MiB: the bytes the next check compares in place (with the hash they have).
		if (in_place_check) {
			table.check_words.assign(m_table_words.begin(), m_table_words.begin() + covered / 4u);
			table.check_hash = hash;
		}
		table.checked_tick  = tick;
		table.covered_frame = frame;
	} else if (in_place) {
		stats.unchanged++;
		table.checked_tick  = tick;
		table.covered_frame = frame;
	} else {
		stats.epoch_hits++;
	}
	// The textures are looked up again once per presented frame, like the heap path's resolved keys (CPU writes,
	// GPU-rendered data). A lookup that dropped an image one of the entries holds makes the table translate once
	// more, with the replacement in the cache (once per tick besides the translation above).
	{
		const bool parts = LongPrepPartsOn(); // KYTY_TRACE_LONG_PREP
		const auto start = parts || large ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		if (!skip_check) RefreshTableSlots(table, frame);
		if (parts || large) {
			const auto ns = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
			if (parts) t_long_prep_parts[32] += ns;
			if (large) {
				stats.large_refresh_ns += ns;
				stats.large_images += table.images.size();
				// Sampled once per PERF line: the last large table's slots and distinct images.
				if (stats.large_slots == 0) {
					std::vector<ImageId> unique(table.images.begin(), table.images.end());
					std::ranges::sort(unique, [](ImageId a, ImageId b) {
						return (uint64_t {a.generation} << 32u | a.index) < (uint64_t {b.generation} << 32u | b.index);
					});
					stats.large_slots         = table.slots.size();
					stats.large_unique_images = static_cast<uint64_t>(std::ranges::distance(
					    unique.begin(), std::unique(unique.begin(), unique.end())));
				}
			}
		}
	}
	if (table.generation != bindless.TableGeneration()) {
		if (slots_alive(table)) {
			table.generation = bindless.TableGeneration();
		} else if (table.retranslated_tick != tick) {
			table.retranslated_tick = tick;
			stats.retranslated++;
			m_table_words.resize(table.covered / 4u);
			if (!ReadBindlessWords(table_base, m_table_words.data(), table.covered)) {
				table.checked_tick = 0;
				return fallback(Fallback::ReadFail);
			}
			if (!TranslateBindlessTable(table, table.covered, XXH3_64bits(m_table_words.data(), table.covered), frame,
			                            shader_hash))
				return fallback(Fallback::ArenaFull);
			RefreshTableSlots(table, frame);
		}
	}
	// This draw reads a translation that still names a dead slot (dropped in this
	// tick after the table's one translation of the tick).
	if (table.generation != bindless.TableGeneration()) stats.dead_slot_reads++;
	const auto first = static_cast<uint32_t>((use.base - table_base) / 4u);
	const auto total = table.covered / 4u;
	if (first >= total) return fallback(Fallback::NoLayout);
	const auto count = static_cast<uint32_t>(std::min<uint64_t>(use.size / 4u, total - first));
	if (classes && !BindlessTable::IntegerClass(table.view.numeric_class)) LearnBindlessClass(shader_hash, use, resource, table);
	prepared.bindless_patches.push_back({use.mapping_offset, table.region + first, count});
	// KYTY_PERF_BINDLESS_CLASSES: the field resources of one table share it; its images join the draw's layout list
	// once (again only when the list changed in between). Refresh events: only the images whose layout may have left the
	// read layout since the table's last listing (ListTableImagesByEvents).
	const size_t listed_before = prepared.bindless_images.size(); // Bindless E1 v3 cost split
	if (!classes || table.listed_draw != m_bindless_draw || table.listed_version != table.images_version) {
		ListTableImagesByEvents(table, prepared.bindless_images);
		table.listed_draw    = m_bindless_draw;
		table.listed_version = table.images_version;
	} else {
		bindless.GetClassStats().images_deduped++;
	}
	if (large) stats.large_listed += prepared.bindless_images.size() - listed_before;
	bindless.MarkTableInDraw(table);
	return true;
}

// KYTY_PERF_BINDLESS_CLASSES: a float resource of one field whose records hold integer T#s (the tally of the table's last
// translation, decided as at the field's first materialization) is registered with that class; its next
// materialization compiles it as an integer field. The registry keeps it (fields only ever become integer).
void RenderExecutor::LearnBindlessClass(uint64_t shader_hash, const ShaderRecompiler::IR::BindlessHeapUse& use,
                                        const ShaderRecompiler::IR::ImageResource& resource,
                                        const BindlessTable::Table&                table) {
	using Class = Prospero::TextureNumericClass;
	if (resource.depth_compare) return;
	// The field as the shader names it (the registry key) and as the table holds it (record stride, offset).
	uint32_t key_stride = 0, key_offset = 0, record = 0;
	if (resource.bindless_stride_dwords != 0u) {
		key_stride = record = resource.bindless_stride_dwords * 4u;
		key_offset          = static_cast<uint32_t>(use.base - use.table_base);
	} else if (resource.bindless_field_split && resource.bindless_fields.size() == 1u) {
		key_stride = resource.bindless_fields.front().first;
		key_offset = resource.bindless_fields.front().second;
		record     = key_stride != 0u ? key_stride : use.record_bytes;
	} else {
		return;
	}
	const auto field = std::ranges::find_if(table.fields, [&](const BindlessTable::TableField& other) {
		return other.stride == record && other.offset == key_offset;
	});
	const auto index = static_cast<size_t>(field - table.fields.begin());
	if (field == table.fields.end() || index >= table.field_tallies.size()) return;
	const auto& tally   = table.field_tallies[index];
	const auto  decided = ShaderRecompiler::IR::DecideBindlessFieldClass(tally);
	if (decided == Class::Float) return;
	const auto key   = ShaderRecompiler::IR::BindlessClassKey(shader_hash, use.image, key_stride, key_offset);
	const auto known = ShaderRecompiler::IR::FindBindlessClass(key);
	// Only a float (or unknown) field is raised: an integer one keeps its class.
	if (known == Class::Uint || known == Class::Sint ||
	    (known == Class::Unsupported && ShaderRecompiler::IR::BindlessClassRegistryFull()))
		return;
	auto& counters = ShaderRecompiler::IR::GetBindlessClassCounters();
	if (!ShaderRecompiler::IR::StoreBindlessClass(key, decided)) {
		counters.registry_full.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	counters.learned[decided == Class::Uint ? 0 : 1].fetch_add(1, std::memory_order_relaxed);
	m_context.GetBindlessTable().GetClassStats().learned++;
	static std::atomic_uint logged {0};
	if (logged.fetch_add(1) < 64u)
		std::fprintf(stderr,
		             "BINDLESS_CLASS learned hash=0x%016llx image=%u field(stride:offset)=%u:%u table=0x%llx uint=%u "
		             "sint=%u float=%u tiny=%u other=%u -> %s\n",
		             static_cast<unsigned long long>(shader_hash), use.image, key_stride, key_offset,
		             static_cast<unsigned long long>(table.base), tally.uint_count, tally.sint_count,
		             tally.float_count, tally.float_tiny, tally.other, decided == Class::Uint ? "uint" : "sint");
}

// A translation takes, for an entry whose words equal those of the table's previous translation (Table::src_*), the slot
// that translation gave, while that slot is live, shows those words and its image is registered: ResolveTableDescriptor
// would find exactly that slot (FindTableSlot) and return it with no other effect. Everything else resolves as before.
// Gate 22.10: one 93,840-byte table alternates with 23,184 bytes every frame (two full translations per frame).
static std::atomic<uint64_t> g_ttd_translations {0}, g_ttd_entries {0}, g_ttd_reused {0};

bool RenderExecutor::TranslateBindlessTable(BindlessTable::Table& table, uint32_t covered, uint64_t hash,
                                            uint64_t frame, uint64_t shader_hash) {
	auto&          bindless = m_context.GetBindlessTable();
	auto&          stats    = bindless.GetTableStats();
	const bool     timed    = PerfEnabled();
	const auto     start    = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const uint32_t descriptor_dwords = (table.view.flags & 1u) != 0u ? 4u : 8u; // R128: words 4..7 read as zero
	const uint32_t entries           = covered / 4u;
	const auto     new_before        = stats.new_textures;
	auto&          delta_cache    = m_context.GetTextureCache();
	uint64_t       delta_reused   = 0;
	uint32_t       descriptors = 0, placeholders = 0;
	// KYTY_PERF_BINDLESS_CLASSES: bytes no field names read the zero placeholder of an integer table's class (an integer
	// access never reaches a float view); a float table counts what its fields hold (LearnBindlessClass); fields no draw
	// asked this table for in 64 presented frames go (a field whose resource became integer reads its own table now).
	const bool     classes = bindless.ClassesEnabled();
	const bool     integer = BindlessTable::IntegerClass(table.view.numeric_class);
	const uint32_t outside = integer ? BindlessTable::PlaceholderSlot(table.view.numeric_class) : TableOutsideEntry;
	// The mark of entries no field named yet: the outside value for a float table (no lookup ever gives slot 1); an
	// integer table's lookups give its placeholder (= outside), so it marks them with a value no slot has.
	constexpr uint32_t UnnamedEntry = 0xfffffffeu;
	const uint32_t     unnamed      = integer ? UnnamedEntry : outside;
	const size_t   class_index =
	    !integer ? 0u : table.view.numeric_class == static_cast<uint32_t>(Prospero::TextureNumericClass::Uint) ? 1u : 2u;
	if (classes && table.field_frames.size() == table.fields.size()) {
		constexpr uint64_t UnusedFrames = 64;
		size_t             kept         = 0;
		for (size_t index = 0; index < table.fields.size(); index++) {
			if (table.field_frames[index] + UnusedFrames <= frame) continue;
			table.fields[kept]       = table.fields[index];
			table.field_frames[kept] = table.field_frames[index];
			kept++;
		}
		bindless.GetClassStats().pruned_fields += table.fields.size() - kept;
		table.fields.resize(kept);
		table.field_frames.resize(kept);
	}
	// The slot journal: the slots before (their owners are updated after the rounds).
	if (bindless.SlotOwnersOn()) m_table_old_slots.assign(table.slots.begin(), table.slots.end());
	// A lookup can drop the image an earlier entry holds (the cache replaced it, e.g. with more mips): once more then,
	// now that the cache holds the replacement.
	for (uint32_t round = 0; round < 2u; round++) {
		const auto generation = bindless.TableGeneration();
		if (++m_table_stamp == 0u) m_table_stamp = 1u;
		table.values.assign(entries, unnamed);
		table.slots.clear();
		if (classes) table.field_tallies.assign(table.fields.size(), {});
		descriptors  = 0;
		placeholders = 0;
		for (size_t index = 0; index < table.fields.size(); index++) {
			const auto& field = table.fields[index];
			if (field.stride == 0u) continue;
			const uint64_t end = std::min(covered, field.end);
			for (uint64_t byte = field.offset; byte + uint64_t {descriptor_dwords} * 4u <= end; byte += field.stride) {
				const auto entry = static_cast<uint32_t>(byte / 4u);
				if (table.values[entry] != unnamed) continue; // another field named the same bytes
				std::array<uint32_t, 8> words {};
				std::copy_n(m_table_words.begin() + entry, descriptor_dwords, words.begin());
				if (classes && !integer)
					ShaderRecompiler::IR::TallyBindlessFieldDescriptor(words.data(), descriptor_dwords == 4u,
					                                                   table.view.binding, table.field_tallies[index]);
				// The previous translation's slot for unchanged words (see g_ttd_translations).
				uint32_t taken = 0;
				if (round == 0u && entry + descriptor_dwords <= table.src_words.size() &&
				    std::equal(words.begin(), words.begin() + descriptor_dwords, table.src_words.begin() + entry)) {
					const uint32_t source = table.src_values[entry];
					if (source != outside && source != unnamed && !bindless.IsPlaceholderSlot(source)) {
						const auto& source_info = bindless.GetTableSlot(table.view.binding, source);
						const auto* image       = delta_cache.m_slot_images.try_get(source_info.image);
						if (source_info.live && source_info.key == BindlessTable::TableSlotKey {table.view, words} &&
						    image != nullptr && image->registered) {
							taken = source;
						}
					}
				}
				uint32_t slot = taken;
				if (taken != 0u) {
					delta_reused++;
				} else {
					slot = ResolveTableDescriptor(table, words, frame);
				}
				table.values[entry] = slot;
				stats.fields++;
				descriptors++;
				if (bindless.IsPlaceholderSlot(slot)) {
					placeholders++;
					if (classes) bindless.GetClassStats().placeholders[class_index]++;
					continue;
				}
				auto& info = bindless.GetTableSlot(table.view.binding, slot);
				if (info.stamp == m_table_stamp) continue;
				info.stamp = m_table_stamp;
				table.slots.emplace_back(slot, info.generation);
			}
		}
		if (bindless.TableGeneration() == generation || bindless.TableSlotsAlive(table)) break;
	}
	if (unnamed != outside) std::replace(table.values.begin(), table.values.end(), unnamed, outside);
	// The slot journal: the slots that left and joined change owners (the last round stamped the new ones).
	if (bindless.SlotOwnersOn()) bindless.UpdateTableOwners(table, m_table_old_slots, m_table_stamp);
	// The source for the next translation (the larger one wins within two frames, so a table that alternates between two
	// sizes keeps the large one).
	if (covered >= table.src_covered || table.src_frame + 2u <= frame) {
		table.src_words.assign(m_table_words.begin(), m_table_words.begin() + entries);
		table.src_values  = table.values;
		table.src_covered = covered;
		table.src_frame   = frame;
	}
	const uint64_t translations = g_ttd_translations.fetch_add(1, std::memory_order_relaxed) + 1u;
	g_ttd_entries.fetch_add(entries, std::memory_order_relaxed);
	g_ttd_reused.fetch_add(delta_reused, std::memory_order_relaxed);
	if (PerfEnabled() && (translations & 4095u) == 0u)
		std::fprintf(stderr, "PERF table_translate_delta translations=%llu entries=%llu reused=%llu\n",
		             static_cast<unsigned long long>(translations),
		             static_cast<unsigned long long>(g_ttd_entries.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(g_ttd_reused.load(std::memory_order_relaxed)));
	// BINDLESS_TABLE: the first translation of a table of each distinct layout (view and fields; at most 128 lines),
	// with its reader, its fields and how much of it resolved, to check the field lists against the guest's records.
	{
		uint64_t fields_hash = 0; // order-independent: tables collect their fields in draw order
		for (const auto& field: table.fields)
			fields_hash += ((uint64_t {field.stride} << 32u) | field.offset) * 0x9E3779B97F4A7C15ull;
		static std::set<uint64_t> layouts; // GPU thread
		if (layouts.size() < 128u && layouts.insert(BindlessTable::HashTableView(fields_hash, table.view)).second) {
			std::string fields;
			for (const auto& field: table.fields)
				fields += fmt::format(" {}:{}<{}", field.stride, field.offset, field.end);
			std::fprintf(stderr,
			             "BINDLESS_TABLE layout frame=%llu hash=0x%016llx table=0x%llx bytes=%u binding=%u view_flags=%u "
			             "fields(stride:offset<end)=%s descriptors=%u placeholders=%u distinct_slots=%zu "
			             "new_textures=%llu%s\n",
			             static_cast<unsigned long long>(frame > 0u ? frame - 1u : 0u),
			             static_cast<unsigned long long>(shader_hash), static_cast<unsigned long long>(table.base),
			             covered, table.view.binding, table.view.flags, fields.c_str(), descriptors, placeholders,
			             table.slots.size(), static_cast<unsigned long long>(stats.new_textures - new_before),
			             !integer ? "" : class_index == 1u ? " class=uint" : " class=sint"); // KYTY_PERF_BINDLESS_CLASSES
		}
	}
	table.covered         = covered;
	table.hash            = hash;
	table.translated_tick = m_context.GetCommandScheduler().CurrentTick();
	if (bindless.TableSlotsAlive(table)) table.generation = bindless.TableGeneration();
	table.refreshed_frame = 0; // its slots are looked up and its images listed before the draw
	stats.translated++;
	const bool stored = bindless.StoreTableRegion(table);
	if (!stored) {
		// The region (if any) still holds the previous translation for the draws that read it; translate again later.
		table.hash         = 0;
		table.checked_tick = 0;
	}
	if (timed)
		stats.translate_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
	return stored;
}

uint32_t RenderExecutor::ResolveTableDescriptor(BindlessTable::Table& table, const std::array<uint32_t, 8>& words,
                                                uint64_t frame) {
	using Placeholder        = BindlessTable::TablePlaceholder;
	auto&      bindless      = m_context.GetBindlessTable();
	auto&      stats         = bindless.GetTableStats();
	auto&      texture_cache = m_context.GetTextureCache();
	const auto binding       = table.view.binding;
	// 0 for a float table (KYTY_PERF_BINDLESS_CLASSES: an integer table's entries read its class's zero placeholder).
	const auto placeholder_slot = BindlessTable::PlaceholderSlot(table.view.numeric_class);
	const auto placeholder      = [&](Placeholder why) {
		stats.placeholder[static_cast<size_t>(why)]++;
		return placeholder_slot;
	};
	// The same descriptor seen through any table already has a slot.
	const BindlessTable::TableSlotKey key {table.view, words};
	if (const auto slot = bindless.FindTableSlot(key); slot != 0u) {
		const auto* image = texture_cache.m_slot_images.try_get(bindless.GetTableSlot(binding, slot).image);
		if (image != nullptr && image->registered) return slot;
		bindless.DropTableSlot(binding, slot); // its image went without a notice: resolve the descriptor anew
	}
	// Words that only resemble a descriptor (a record past a table's live count, a field that holds data in some
	// records) stay the placeholder; so does what the float-typed arrays cannot hold (integer formats, other shapes),
	// as on the heap path.
	if (!LooksLikeImageDescriptor(words.data(), (table.view.flags & 1u) != 0u))
		return placeholder(Placeholder::NotDescriptor);
	ShaderTextureResource descriptor {};
	std::memcpy(descriptor.fields, words.data(), sizeof(descriptor.fields));
	if (!BindlessCompatibleClass(descriptor, binding, table.view.numeric_class)) {
		// KYTY_PERF_BINDLESS_CLASSES: an integer table's T#s of another class (the guest's float defaults) read zero, as
		// the console's gather of a 1x1 black default does; they are not "formats the float arrays cannot hold".
		const auto format_class = Prospero::SampledTextureNumericClass(descriptor.Format());
		if (placeholder_slot != 0u) {
			if (static_cast<uint32_t>(format_class) != table.view.numeric_class) bindless.GetClassStats().class_mismatch++;
		} else {
			if (bindless.ClassesEnabled() && (format_class == Prospero::TextureNumericClass::Uint ||
			                                  format_class == Prospero::TextureNumericClass::Sint))
				bindless.GetClassStats().int_in_float++;
			stats.incompatible[{static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(descriptor.Format())}]++;
		}
		return placeholder(Placeholder::Incompatible);
	}
	uint32_t probe = 0;
	if (!GuestRange {descriptor.Base40(), 1u}.Valid() ||
	    !Libs::LibKernel::Memory::TryReadBacking(descriptor.Base40(), &probe, sizeof(probe)) ||
	    (descriptor.MetaCompress() && descriptor.MetaAddr() != 0 &&
	     !GuestRange {descriptor.MetaAddr() << 8u, 1u}.Valid()))
		return placeholder(Placeholder::BadAddress);
	for (uint32_t channel = 0; channel < 4u; channel++) {
		const auto selector = GetDstSel(descriptor.DstSelXYZW(), channel);
		if (selector == 2u || selector == 3u) return placeholder(Placeholder::Swizzle);
	}
	// The texture cache lookup a bound texture gets, as a soft resolve (a null image instead of an exit).
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count = 8u;
	std::copy_n(words.begin(), 8u, value.dwords.begin());
	m_soft_texture_resolve = true;
	auto resolved          = ResolveTexture(table.resource, value);
	m_soft_texture_resolve = false;
	auto* image            = texture_cache.m_slot_images.try_get(resolved.image_id);
	if (image == nullptr || image->info.data.Empty()) {
		return placeholder(Placeholder::SoftNull);
	}
	BindImage(resolved.image_id, false);
	const auto view = texture_cache.FindTexture(resolved.image_id, resolved.desc);
	image           = texture_cache.m_slot_images.try_get(resolved.image_id);
	// Registered: an image the lookups dropped before it was pinned would never tell the table it died.
	if (image == nullptr || view == nullptr || !image->registered) return placeholder(Placeholder::NoView);
	const auto slot = bindless.AllocateSlot(binding);
	if (slot == 0) {
		Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::BindlessNoSlot); // KYTY_TRACE_PERF
		return placeholder(Placeholder::NoSlot);
	}
	const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
	bindless.WriteSlot(binding, slot, view, layout);
	image->bindless_pinned = true;
	bindless.AddTableSlot(key, slot, resolved.image_id, image->IsGpuModified(), frame, m_bindless_draw);
	stats.new_textures++;
	return slot;
}

// A table's slot refreshes take the texture cache's lock once for a run of replayed slots instead of twice per slot
// (ReplayImage, then TableTexturesReady): the same work in the same order. The lock is released before anything that may
// take it or wait (a full lookup, BindImage), after 64 slots, and when the GPU thread waits for it.
struct TableRefreshBatch {
	TextureCache& cache;
	bool          held = false;
	uint32_t      run  = 0;
	explicit TableRefreshBatch(TextureCache& c): cache(c) {}
	TableRefreshBatch(const TableRefreshBatch&)            = delete;
	TableRefreshBatch& operator=(const TableRefreshBatch&) = delete;
	~TableRefreshBatch() { Release(); }
	void Release() {
		if (!held) return;
		cache.UnlockLookup();
		held = false;
	}
	bool Replay(ImageId id) {
		if (held && (++run >= 64u || cache.LookupYieldRequested())) Release();
		if (!held) {
			cache.LockTableRefresh();
			held = true;
			run  = 0;
		}
		return cache.ReplayTableSlotLocked(id);
	}
};

// Refresh events (bindless E1 fast version, the only path): a slot
// whose last refresh replayed - its description unchanged, so it only touched the image - is not refreshed again while
// nothing the replay's checks read has changed since: the image's change serial (Image::HeapChanged: CPU-dirty marks,
// buffer writes, untracking, leaving the read layout) and the image-page epochs of its range (registry changes, pending
// rebinds, alias epochs), for a table of the same view; its image gets the replay's access tick and LRU touch in one
// batch per table. A table of 15,948 textures (the particle table) refreshed every slot every frame (~64 ns each). The
// slot journal (bindlessTable.h) names the slots whose checks a change reaches; the others pass without the check.

void RenderExecutor::ListTableImagesByEvents(BindlessTable::Table& table, std::vector<ImageId>& list) {
	auto&        bindless = m_context.GetBindlessTable();
	auto&        stats    = bindless.GetTableStats();
	auto&        log      = g_bindless_layout_log;
	const auto   key      = [](ImageId id) { return (uint64_t {id.generation} << 32u) | id.index; };
	const size_t first    = list.size();
	{
		std::scoped_lock lock(log.mutex);
		const uint64_t end = log.base + log.ids.size();
		if (!table.walked || table.walked_hash != table.images_hash || table.layout_log_pos < log.base) {
			// The set changed (or the log was trimmed past this table): every image once.
			list.insert(list.end(), table.images.begin(), table.images.end());
			table.image_set.clear();
			for (const auto id: table.images) table.image_set.insert(key(id));
			table.unwatched.assign(table.images.begin(), table.images.end());
			table.walked         = true;
			table.walked_hash    = table.images_hash;
			table.layout_log_pos = end;
			stats.layout_full_lists++;
			return;
		}
		for (uint64_t pos = table.layout_log_pos; pos < end; pos++) {
			const auto id = log.ids[pos - log.base];
			if (table.image_set.contains(key(id))) list.push_back(id);
		}
		table.layout_log_pos = end;
	}
	// Images no layout walk has seen yet (a draw that listed them was dropped before its commit): listed until one does
	// (Image::bindless_watched, set by the walk), then their layout changes are logged.
	if (!table.unwatched.empty()) {
		auto&  texture_cache = m_context.GetTextureCache();
		size_t kept          = 0;
		for (const auto id: table.unwatched) {
			const auto* image = texture_cache.m_slot_images.try_get(id);
			if (image == nullptr || image->bindless_watched) continue;
			list.push_back(id);
			table.unwatched[kept++] = id;
		}
		table.unwatched.resize(kept);
	}
	// The GPU-written images: their layout follows the draws that render them within the frame.
	for (const auto& [slot, generation]: table.gpu_slots) {
		const auto& info = bindless.GetTableSlot(table.view.binding, slot);
		if (info.live && info.generation == generation) list.push_back(info.image);
	}
	stats.layout_listed += list.size() - first;
}


void RenderExecutor::RefreshTableSlots(BindlessTable::Table& table, uint64_t frame) {
	auto&      bindless = m_context.GetBindlessTable();
	const bool full     = table.refreshed_frame != frame;
	// The journal names its slots first.
	bindless.DrainSlotJournal();
	// The list path (bindless C9 variant A, points 1-3; bindlessTable.h): a table over
	// 1 MiB lists the slots that may need more than the trusted skip (journal-named, refreshed without a replay) and visits
	// only those and its gpu_slots - the rest would only be skipped and LRU-touched, now answered by the owners' last full
	// refresh (table.used_frame). After a translation (refreshed_frame 0) every slot once, then the list is built.
	const bool large = table.covered > TableMaxBytes;
	if (full) table.used_frame = frame; // the use of every slot it owns (ReleaseUnusedTableSlots)
	if (!large && table.fast_list) {    // no longer large: no list kept (a large one starts again with a full refresh)
		table.fast_list = false;
		table.fast_slots.clear();
	}
	const bool on_list = large && table.fast_list && table.refreshed_frame != 0u;
	// The run's lock, released when this scope ends.
	TableRefreshBatch batch {m_context.GetTextureCache()};
	// A slot refresh of any table: one it leaves without a replay goes on its owners' lists (NoteFastSlot).
	const auto refresh_slot = [&](uint32_t slot) {
		const bool remap = RefreshTableSlot(table, slot, frame, batch);
		bindless.NoteFastSlot(table.view.binding, slot);
		return remap;
	};
	// Slots are shared: another table's lookup may have given one of this table's slots another image, or found its
	// image GPU-written, since this table listed them.
	const bool list_gpu    = full || table.gpu_epoch != bindless.TableGpuEpoch();
	// The slot journal: the images again only when one of its own slots changed image or died.
	const bool list_images = table.images_dirty;
	if (!list_gpu && !list_images && table.gpu_slots.empty()) return;
	const bool timed    = PerfEnabled();
	const auto start    = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	bool       remapped = false;
	// Bindless E1 v3 cost split: a large table's slot loop, LRU touches and image-list rebuild (PERF on).
	auto&      stats       = bindless.GetTableStats();
	const bool large_timed = timed && large;
	const auto large_since = [](std::chrono::steady_clock::time_point since) {
		return static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
	};
	auto&          texture_cache = m_context.GetTextureCache();
	const uint64_t view_tag      = XXH3_64bits(&table.view, sizeof(table.view)); // the view the replay memos were taken for
	// The trusted skip: the journal did not name the slot since its replay check last ran (or it was last refreshed) and
	// its last refresh replayed, so nothing that check reads changed since.
	const auto trusted = [&](const BindlessTable::TableSlotInfo& info, uint32_t generation) {
		return !info.journal_named && info.event_valid && !info.gpu_written && info.event_view == view_tag &&
		       info.memo_generation == generation;
	};
	// The replay check: what the slot's last replay read is as then (the next refresh would replay again).
	const auto unchanged = [&](const BindlessTable::TableSlotInfo& info, uint32_t generation) {
		const auto* image = texture_cache.m_slot_images.try_get(info.image);
		return info.event_valid && !info.gpu_written && info.event_view == view_tag && info.memo_generation == generation &&
		       image != nullptr && image->registered && image->change_serial == info.event_serial &&
		       DrawRepeat::ImagePageEpochSumExact(image->info.data.address, image->info.data.size) == info.event_image_sum &&
		       DrawRepeat::ImagePageEpochSum(image->info.data.address, image->info.data.size) == info.event_image_alias_sum &&
		       DrawRepeat::ImagePageEpochSumExact(info.memo_address, info.memo_size) == info.memo_page_sum;
	};
	// Whether the slot is due: once per presented frame, and at every draw for an image the GPU writes.
	const auto due = [&](const BindlessTable::TableSlotInfo& info) {
		return (full && info.refreshed_frame != frame) || (info.gpu_written && info.refreshed_draw != m_bindless_draw);
	};
	// One due slot: the trusted skip (at the frame's first refresh), else the check, else a refresh. True: skipped or
	// passed (it only needs the replay's touch).
	const auto refresh_due = [&](BindlessTable::TableSlotInfo& info, uint32_t slot, uint32_t generation) {
		// (A change journaled by this refresh's own lookups so far names its slots first.)
		if (full && g_table_slot_journal_pending.load(std::memory_order_acquire)) bindless.DrainSlotJournal();
		if (!full || !trusted(info, generation)) {
			if (full) stats.journal_checked++;
			if (!unchanged(info, generation)) {
				remapped = refresh_slot(slot) || remapped;
				return false;
			}
			info.journal_named = false; // the check ran and passed
		}
		info.refreshed_frame = frame;
		info.refreshed_draw  = m_bindless_draw;
		stats.refresh_event_skips++;
		return true;
	};
	if (list_gpu && on_list) {
		// The list and the GPU-written slots; the slots the list path leaves alone get no touch (FastTouchKeeps).
		const auto     loop_start = large_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		const uint64_t visit      = bindless.NextFastStamp();
		const auto     visit_slot = [&](uint32_t slot, uint32_t generation) {
			auto& info = bindless.GetTableSlot(table.view.binding, slot);
			if (!info.live || info.generation != generation || info.fast_stamp == visit) return;
			info.fast_stamp = visit;
			stats.fast_visited++;
			if (due(info)) (void)refresh_due(info, slot, generation);
		};
		// By index: the visits append to both lists (journal names, refreshes without a replay, GPU-written slots).
		size_t listed = 0, gpu = 0;
		while (listed < table.fast_slots.size() || gpu < table.gpu_slots.size()) {
			for (; listed < table.fast_slots.size(); listed++)
				visit_slot(table.fast_slots[listed].first, table.fast_slots[listed].second);
			for (; gpu < table.gpu_slots.size(); gpu++) visit_slot(table.gpu_slots[gpu].first, table.gpu_slots[gpu].second);
		}
		// The list for the next refresh: the slots still named or not trusted, once each.
		const uint64_t keep = bindless.NextFastStamp();
		size_t         kept = 0;
		for (size_t i = 0; i < table.fast_slots.size(); i++) {
			const auto entry = table.fast_slots[i];
			auto&      info  = bindless.GetTableSlot(table.view.binding, entry.first);
			if (!info.live || info.generation != entry.second || info.fast_stamp == keep || trusted(info, entry.second))
				continue;
			info.fast_stamp          = keep;
			table.fast_slots[kept++] = entry;
		}
		table.fast_slots.resize(kept);
		stats.fast_list_peak = std::max<uint64_t>(stats.fast_list_peak, kept);
		// The GPU-written slots still live, once each (NoteTableSlotGpuWritten adds new ones; the full loop below lists
		// them from every slot instead).
		const uint64_t gpu_keep = bindless.NextFastStamp();
		size_t         gpu_kept = 0;
		for (size_t i = 0; i < table.gpu_slots.size(); i++) {
			const auto entry = table.gpu_slots[i];
			auto&      info  = bindless.GetTableSlot(table.view.binding, entry.first);
			if (!info.live || info.generation != entry.second || !info.gpu_written || info.fast_stamp == gpu_keep) continue;
			info.fast_stamp              = gpu_keep;
			table.gpu_slots[gpu_kept++] = entry;
		}
		table.gpu_slots.resize(gpu_kept);
		if (full) {
			table.refreshed_frame = frame;
			stats.journal_refreshes++; // PERF bindless_journal
		}
		table.gpu_epoch = bindless.TableGpuEpoch();
		stats.fast_uses++;
		if (large_timed) stats.large_loop_ns += large_since(loop_start);
	} else if (list_gpu) {
		m_table_touch.clear();
		const auto loop_start = large_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		// Once per presented frame: every slot (a texture the GPU renders: every draw, so a draw samples what earlier
		// draws of this frame wrote).
		table.gpu_slots.clear();
		for (const auto& [slot, generation]: table.slots) {
			auto& info = bindless.GetTableSlot(table.view.binding, slot);
			if (!info.live || info.generation != generation) continue;
			if (due(info) && refresh_due(info, slot, generation)) m_table_touch.push_back(info.image);
			if (info.live && info.generation == generation && info.gpu_written)
				table.gpu_slots.emplace_back(slot, generation);
		}
		if (full) {
			table.refreshed_frame = frame;
			stats.journal_refreshes++; // PERF bindless_journal
		}
		table.gpu_epoch = bindless.TableGpuEpoch();
		// A table over 1 MiB: its list, built once every slot has been refreshed (a translation, a table new to the size).
		if (large && full) {
			table.fast_slots.clear();
			for (const auto& [slot, generation]: table.slots) {
				const auto& info = bindless.GetTableSlot(table.view.binding, slot);
				if (info.live && info.generation == generation && !trusted(info, generation))
					table.fast_slots.emplace_back(slot, generation);
			}
			table.fast_list = true;
			stats.fast_builds++;
			stats.fast_list_peak = std::max<uint64_t>(stats.fast_list_peak, table.fast_slots.size());
		}
		if (large_timed) stats.large_loop_ns += large_since(loop_start);
		if (!m_table_touch.empty()) {
			batch.Release(); // it holds the cache's lock; the touches take it themselves
			const auto touch_start = large_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
			texture_cache.TouchTableImages(m_table_touch);
			if (large_timed) stats.large_touch_ns += large_since(touch_start);
		}
	} else {
		for (const auto& [slot, generation]: table.gpu_slots) {
			const auto& info = bindless.GetTableSlot(table.view.binding, slot);
			if (!info.live || info.generation != generation || info.refreshed_draw == m_bindless_draw) continue;
			remapped = refresh_slot(slot) || remapped;
		}
	}
	if (list_images || remapped) {
		const auto list_start = large_timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		table.images.clear();
		for (const auto& [slot, generation]: table.slots) {
			const auto& info = bindless.GetTableSlot(table.view.binding, slot);
			if (info.live && info.generation == generation) table.images.push_back(info.image);
		}
		table.images_epoch = bindless.TableSlotEpoch();
		table.images_version++; // KYTY_PERF_BINDLESS_CLASSES: a draw that appended the list before appends it again
		// Refresh events: a list rebuilt with the same images needs no full listing again.
		table.images_hash  = XXH3_64bits(table.images.data(), table.images.size() * sizeof(ImageId));
		table.images_dirty = false;
		if (large_timed) {
			stats.large_list_ns += large_since(list_start);
			stats.large_lists++;
		}
	}
	if (timed)
		bindless.GetTableStats().refresh_ns += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

// RefreshBindlessDescriptor for a table slot: the texture cache's current image of the slot's descriptor goes into the
// slot when it is another one.
bool RenderExecutor::RefreshTableSlot(BindlessTable::Table& table, uint32_t slot, uint64_t frame, TableRefreshBatch& batch) {
	auto&      bindless      = m_context.GetBindlessTable();
	auto&      texture_cache = m_context.GetTextureCache();
	auto&      info          = bindless.GetTableSlot(table.view.binding, slot); // a deque that only grows: stays valid
	const auto generation    = info.generation;
	const auto words         = info.key.words; // the lookups below can drop the slot
	info.refreshed_frame     = frame;
	info.refreshed_draw      = m_bindless_draw;
	info.journal_named       = false; // the slot journal: what this refresh records is current
	bindless.GetTableStats().refreshed++;
	t_long_prep_parts[35]++; // KYTY_TRACE_LONG_PREP
	// The replay: the slot's descriptor, resource and soft resolve are the same as at its last full refresh; while the
	// image-page epochs of the description's range sum to the same value, FindImage finds the same image (its lookup then
	// was plain and the image replayable), so ReplayImage does its work; TableTexturesReady proves FindTexture would only
	// touch it (the slot keeps its view: no remap). The GPU-written check and the layout-list entry follow as below.
	// The memo's pages unchanged, by the epochs without aliasing (DrawRepeat::ImagePageEpochSumExact): a registry change in
	// another 64 GiB lane does not force the full lookup.
	const auto     pages_same    = [&] {
		return DrawRepeat::ImagePageEpochSumExact(info.memo_address, info.memo_size) == info.memo_page_sum;
	};
	const ImageId  memo_image    = info.image;
	ShaderRecompiler::IR::DescriptorValue value {};
	value.dword_count = 8u;
	std::copy_n(words.begin(), 8u, value.dwords.begin());
	const auto key_full = MakeTextureDescKey(table.resource, DecodeNativeDescriptor<ShaderTextureResource>(value));
	const std::array<uint32_t, 8> resource_key {key_full.resource_class, key_full.numeric_class, key_full.dimension,
	                                            key_full.mip_mode,       key_full.mip_count,     key_full.conversion,
	                                            key_full.swizzle,        key_full.flags};
	if (info.memo_valid && info.memo_resource == resource_key && info.memo_generation == generation && memo_image &&
	    pages_same() &&
	    batch.Replay(memo_image)) {
		auto* image = texture_cache.m_slot_images.try_get(memo_image);
		if (image == nullptr) return false;
		if (image->IsGpuModified() && !info.gpu_written) {
			info.gpu_written = true;
			bindless.NoteTableSlotGpuWritten(table.view.binding, slot);
		}
		if (info.gpu_written) {
			batch.Release(); // BindImage may wait for a commit
			BindImage(memo_image, false);
		}
		// Refresh events: a replay only touched the image; the next one may be skipped until it changes.
		info.event_valid      = !info.gpu_written;
		info.event_serial     = image->change_serial;
		info.event_view       = XXH3_64bits(&table.view, sizeof(table.view));
		info.event_image_sum  = DrawRepeat::ImagePageEpochSumExact(image->info.data.address, image->info.data.size);
		info.event_image_alias_sum = DrawRepeat::ImagePageEpochSum(image->info.data.address, image->info.data.size);
		// The slot journal: the page index lists the slot under the pages whose epochs this check reads.
		if (info.event_valid)
			bindless.IndexSlotPages(table.view.binding, slot, image->info.data.address, image->info.data.size,
			                        info.memo_address, info.memo_size);
		return false;
	}
	batch.Release(); // the full lookup below takes the cache's lock
	info.event_valid      = false; // refresh events: a full refresh is no replay
	const auto memo_epoch = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire);
	m_soft_texture_resolve = true;
	(void)TextureCache::TakeLastFindImagePlain();
	auto resolved          = ResolveTexture(table.resource, value);
	const bool plain       = TextureCache::TakeLastFindImagePlain();
	m_soft_texture_resolve = false;
	auto* image            = texture_cache.m_slot_images.try_get(resolved.image_id);
	info.memo_valid = false;
	if (image == nullptr || image->info.data.Empty()) return false;
	const auto view = texture_cache.FindTextureReplay(resolved.image_id, resolved.desc);
	if (plain && resolved.desc.type == TextureCache::BindingType::Texture &&
	    texture_cache.ImageReplayable(resolved.image_id) &&
	    DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire) == memo_epoch) {
		// Kept only if the slot shows this image afterwards (checked below, where the slot is remapped or not).
		info.memo_valid      = true;
		info.memo_resource   = resource_key;
		info.memo_generation = generation;
		info.memo_address    = resolved.desc.info.data.address;
		info.memo_size       = resolved.desc.info.data.size;
		info.memo_page_sum   = DrawRepeat::ImagePageEpochSumExact(info.memo_address, info.memo_size);
	}
	image           = texture_cache.m_slot_images.try_get(resolved.image_id);
	if (image == nullptr || view == nullptr || !image->registered || !info.live || info.generation != generation)
		return false;
	if (image->IsGpuModified() && !info.gpu_written) {
		info.gpu_written = true;
		// every table that holds the slot refreshes it at every draw from now on
		bindless.NoteTableSlotGpuWritten(table.view.binding, slot);
	}
	if (info.gpu_written) BindImage(resolved.image_id, false);
	if (resolved.image_id == info.image) return false;
	const auto old_image = info.image;
	const auto layout    = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                             : vk::ImageLayout::eShaderReadOnlyOptimal;
	bindless.WriteSlot(table.view.binding, slot, view, layout);
	image->bindless_pinned = true;
	bindless.RemapTableSlot(table.view.binding, slot, resolved.image_id);
	// The image the slot showed before stays pinned only while a heap descriptor or another table slot maps to it.
	if (!bindless.TableUsesImage(old_image) &&
	    std::ranges::none_of(bindless.descriptor_slots, [&](const auto& entry) { return entry.second.image == old_image; })) {
		if (auto* old = texture_cache.m_slot_images.try_get(old_image); old != nullptr) old->bindless_pinned = false;
	}
	return true;
}

void RenderExecutor::PrepareTablesFrame(uint64_t frame) {
	if (m_table_frame == frame) return;
	m_table_frame  = frame;
	auto& bindless = m_context.GetBindlessTable();
	{
		// PERF bindless_tables max_new_frame: the new textures of the frame that just ended (the counters restart at
		// every PERF line, so a start above the count is a restart).
		static uint64_t frame_start = 0;
		auto&           stats       = bindless.GetTableStats();
		if (frame_start > stats.new_textures) frame_start = 0;
		stats.max_new_frame = std::max(stats.max_new_frame, stats.new_textures - frame_start);
		frame_start         = stats.new_textures;
	}
	if (g_bindless_layout_log_on.load(std::memory_order_relaxed)) {
		// Refresh events: the layout log keeps at most 64 Ki entries (a table read past the cut lists
		// all its images once).
		std::scoped_lock lock(g_bindless_layout_log.mutex);
		auto&            ids = g_bindless_layout_log.ids;
		if (ids.size() > 65536u) {
			const size_t drop = ids.size() - 32768u;
			ids.erase(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(drop));
			g_bindless_layout_log.base += drop;
		}
	}
	bindless.EvictStaleTables(frame);
	if (frame / 60u == m_table_release_frame / 60u) return;
	m_table_release_frame = frame;
	std::vector<ImageId> released;
	bindless.ReleaseUnusedTableSlots(frame, released);
	if (released.empty()) return;
	// An image stays pinned while a heap descriptor or another table slot still maps to it.
	std::unordered_set<uint64_t> in_use;
	for (const auto& [key, cached]: bindless.descriptor_slots)
		in_use.insert((uint64_t {cached.image.generation} << 32u) | cached.image.index);
	for (const auto id: released) {
		if (in_use.contains((uint64_t {id.generation} << 32u) | id.index) || bindless.TableUsesImage(id)) continue;
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr)
			image->bindless_pinned = false;
	}
}

// The image replay: per program and image slot, the last texture binding is kept when its lookup was replayable: sampled,
// FindImage found an existing image by the plain search and did nothing else (TextureCache::TakeLastFindImagePlain) and
// the image is one ImageReplayable accepts (so the resolve's tail only validates). A later binding of the same slot with
// the same T# words, compare functions and soft-resolve flag gets the same description (a pure function of them and the
// program's resource), and, while the image-page epochs of the description's range sum to the same value (every
// registration change of an image over a page bumps it), FindImage's search sees the same candidates and picks the same
// image: only its LRU touch and access tick are replayed (ReplayImage). A store is skipped when the global registry epoch
// moved during the resolve.

namespace {
struct ImageReplaySlot {
	bool                    valid         = false;
	bool                    soft          = false;
	uint32_t                compare_funcs = 0;
	uint32_t                dword_count   = 0;
	uint64_t                shader_hash   = 0;
	uint64_t                page_sum      = 0;
	std::array<uint32_t, 8> words {};
	ImageId                 id;
	TextureCache::ImageDesc desc;
};
} // namespace

// The table sets: whether the last image binding of this thread is one a stage set may stand for (an image
// replay hit, or a plain lookup of a replayable sampled image).
static thread_local bool g_slot_exact = false;

template <typename Resolve>
static TextureBinding ResolveTextureReplay(TextureCache& cache, bool soft, const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                           uint32_t slot, const ShaderRecompiler::IR::DescriptorValue& value,
                                           uint32_t compare_funcs, Resolve&& resolve) {
	static thread_local std::unordered_map<uint64_t, ImageReplaySlot> slots;
	auto&      memo = slots[reinterpret_cast<uint64_t>(&program) * 64u + slot];
	const bool same = memo.valid && memo.shader_hash == program.shader_hash && memo.dword_count == value.dword_count &&
	                  memo.words == value.dwords && memo.compare_funcs == compare_funcs && memo.soft == soft &&
	                  memo.page_sum == DrawRepeat::ImagePageEpochSum(memo.desc.info.data.address, memo.desc.info.data.size);
	if (same) {
		if (const auto id = cache.ReplayImage(memo.id)) {
			g_slot_exact = true;
			return {id, nullptr, memo.desc};
		}
	}
	const auto epoch = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire);
	(void)TextureCache::TakeLastFindImagePlain();
	auto       binding = resolve();
	const bool plain   = TextureCache::TakeLastFindImagePlain();
	memo.valid         = false;
	g_slot_exact       = plain && binding.desc.type == TextureCache::BindingType::Texture && cache.ImageReplayable(binding.image_id);
	if (plain && binding.desc.type == TextureCache::BindingType::Texture && cache.ImageReplayable(binding.image_id)) {
		const auto page_sum = DrawRepeat::ImagePageEpochSum(binding.desc.info.data.address, binding.desc.info.data.size);
		if (DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire) == epoch) {
			memo.valid         = true;
			memo.soft          = soft;
			memo.compare_funcs = compare_funcs;
			memo.dword_count   = value.dword_count;
			memo.shader_hash   = program.shader_hash;
			memo.page_sum      = page_sum;
			memo.words         = value.dwords;
			memo.id            = binding.image_id;
			memo.desc          = binding.desc;
		}
	}
	return binding;
}

// The table sets (TABLE-MODE-DESIGN.md T1): a stage's image set is keyed by its program and the words its image and sampler
// bindings read (every T# with its compare functions, every S#, the soft-resolve flag). The set holds the bindings today's
// path produced (image id, description, views) and the samplers. A later stage with the same key reuses them after a
// per-use proof:
// - PrepareBindings: the description is the same pure function of the key; while the image-page epochs over each
//   description's range sum to the stored values (every registration change of an image over a page bumps it), the
//   search finds the same image; TableReplayStage then does FindImage's work (exists, registered, no pending rebind;
//   LRU touch and access tick) for all sampled images under one lock. Only sampled textures and null images the image
//   replay accepts (TextureCache::ImageReplayable: no depth, stencil, metadata, 1-byte texels) are set members; other
//   slots run today's code in their place.
// - RebindImages: when no registration changed since the proof (global registry epoch), TableTexturesReady proves
//   FindTexture would do only its touch (FindTextureReplay's predicate) for all members under one lock, and the
//   stored views are used (a view is FindView's cached handle of that image and view info).
// Any failed proof takes today's path for the whole stage, and the set is stored again. Per-binding traces of the image
// loop would miss the stages a set serves, so they turn the sets off.
static bool TableSetsUsable() {
	static constinit Common::OnceValue<bool> once;
	return once.Get([] {
		for (const char* trace: {"KYTY_TRACE_TEX_ADDRS", "KYTY_TRACE_TEX_HASH", "KYTY_TRACE_TEX_WRITES_EXTENT"}) {
			if (std::getenv(trace) != nullptr) {
				std::fprintf(stderr, "Table sets off: %s is set\n", trace);
				return false;
			}
		}
		return true;
	});
}

// How a stage finds its set in the program's list: the first set in list order that TableSetMatches accepts (moved to the
// front), reached through two cheaper steps: a per-thread direct-mapped cache from the program to its list (the map's nodes
// are never erased or moved, so the list pointer stays valid) instead of a map lookup per stage, and a 64-bit hash of the
// key words kept in each set (TableSet::key_hash, set where the words are stored), compared before the full word compare.
// Equal words always have equal hashes (the hash is a function of the words in key order), so a set TableSetMatches accepts
// is never skipped; an equal hash only lets the full compare decide.

namespace {
struct TableSet {
	std::vector<uint32_t>       words;
	uint64_t                    key_hash = 0; // TableSetWordsHash of words (set with them)
	uint32_t                    frame = 0;
	uint64_t                    version = 0; // unique per store: PreparedBindings::image_source tags
	bool                        views_ready = false;
	std::vector<TextureBinding> images;
	std::vector<uint8_t>        memo;
	std::vector<uint64_t>       page_sums;
	std::vector<ImageId>        member_ids;
	std::vector<vk::Sampler>    samplers;
};
struct TableSetStats {
	uint64_t stages = 0, hits = 0, prep_fail = 0, rebind_hits = 0, rebind_fail = 0, stores = 0;
};
std::unordered_map<const void*, std::vector<std::unique_ptr<TableSet>>>& TableSets() {
	static thread_local std::unordered_map<const void*, std::vector<std::unique_ptr<TableSet>>> sets;
	return sets;
}
using TableSetList = std::vector<std::unique_ptr<TableSet>>;
// TableSets()[program] through a direct-mapped per-thread cache. The map never erases and an
// unordered_map keeps its nodes where they are, so a cached list pointer stays valid; a conflicting program replaces the
// entry and finds its list in the map again.
TableSetList& TableSetListCached(const void* program) {
	struct Entry {
		const void*   program;
		TableSetList* list;
	};
	static thread_local Entry cache[256] = {};
	Entry& entry = cache[((reinterpret_cast<uintptr_t>(program) >> 4) * 0x9E3779B97F4A7C15ull) >> 56];
	if (entry.program != program) {
		entry.list    = &TableSets()[program];
		entry.program = program;
	}
	return *entry.list;
}
TableSetStats g_table_stats_shared;

} // namespace

static TableSetStats& TableStats() {
	return g_table_stats_shared;
}

static void TableSetKey(const ShaderRecompiler::IR::CompiledShaderInfo& program, const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                        bool soft, std::vector<uint32_t>& words) {
	words.clear();
	words.reserve(program.info.images.size() * 9u + program.info.samplers.size() * 4u + 2u);
	// The compare functions (ImageCompareFuncs) come from the S# words, which are in the key.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		words.insert(words.end(), snapshot.images[i].dwords.begin(), snapshot.images[i].dwords.end());
		words.push_back(snapshot.images[i].dword_count);
	}
	for (const auto& sampler: program.info.samplers) {
		const auto& value = snapshot.samplers[sampler.snapshot_index];
		words.insert(words.end(), value.dwords.begin(), value.dwords.begin() + 4);
		words.push_back(value.dword_count);
	}
	words.push_back(soft ? 1u : 0u);
}

// TableSetKey(program, snapshot, soft) == words, without building the key.
static bool TableSetMatches(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                            const ShaderRecompiler::IR::ResourceSnapshot& snapshot, bool soft, const std::vector<uint32_t>& words) {
	if (words.size() != program.info.images.size() * 9u + program.info.samplers.size() * 5u + 1u) return false;
	const uint32_t* w = words.data();
	for (uint32_t i = 0; i < program.info.images.size(); i++, w += 9) {
		const auto& value = snapshot.images[i];
		if (!std::equal(value.dwords.begin(), value.dwords.end(), w) || w[8] != value.dword_count) return false;
	}
	for (const auto& sampler: program.info.samplers) {
		const auto& value = snapshot.samplers[sampler.snapshot_index];
		if (!std::equal(value.dwords.begin(), value.dwords.begin() + 4, w) || w[4] != value.dword_count) return false;
		w += 5;
	}
	return *w == (soft ? 1u : 0u);
}

// The key's hash, a function of the key words in TableSetKey's order: per image its 8 dwords and
// its dword_count (9 words), per sampler its first 4 dwords and its dword_count (5 words), then the soft flag (1 word).
// Each group is read as 64-bit pairs (independent multiplies) and folded into the running value; the stored key
// (TableSetWordsHash, from the words vector) and the stage's current key (TableSetSnapshotHash, from the snapshot) use the
// same group functions, so equal words give equal hashes.
static_assert(sizeof(ShaderRecompiler::IR::DescriptorValue::dwords) == 8u * sizeof(uint32_t));
constexpr uint64_t kSetHashA = 0x9E3779B97F4A7C15ull, kSetHashB = 0xC2B2AE3D27D4EB4Full, kSetHashC = 0x165667B19E3779F9ull,
                   kSetHashD = 0x27D4EB2F165667C5ull, kSetHashE = 0xFF51AFD7ED558CCDull, kSetHashF = 0xC4CEB9FE1A85EC53ull;

static inline uint64_t SetHashPair(const uint32_t* p) {
	uint64_t pair;
	std::memcpy(&pair, p, sizeof(pair));
	return pair;
}

static inline uint64_t SetHashImage(const uint32_t* d, uint32_t count) {
	return (SetHashPair(d) * kSetHashA) ^ (SetHashPair(d + 2) * kSetHashB) ^ (SetHashPair(d + 4) * kSetHashC) ^
	       (SetHashPair(d + 6) * kSetHashD) ^ (static_cast<uint64_t>(count) * kSetHashE);
}

static inline uint64_t SetHashSampler(const uint32_t* d, uint32_t count) {
	return (SetHashPair(d) * kSetHashC) ^ (SetHashPair(d + 2) * kSetHashD) ^ (static_cast<uint64_t>(count) * kSetHashA);
}

static inline uint64_t SetHashStep(uint64_t h, uint64_t group) {
	return std::rotl(h ^ group, 23) * kSetHashB;
}

static inline uint64_t SetHashEnd(uint64_t h, bool soft) {
	h = (h ^ (soft ? kSetHashF : 0ull)) * kSetHashA;
	return h ^ (h >> 32);
}

// The hash of a stored key (words as TableSetKey made them for a program with these image and sampler counts).
static uint64_t TableSetWordsHash(size_t image_count, size_t sampler_count, const std::vector<uint32_t>& words) {
	if (words.size() != image_count * 9u + sampler_count * 5u + 1u) return 0u; // no valid key (TableSetMatches rejects it too)
	uint64_t        h = kSetHashE;
	const uint32_t* w = words.data();
	for (size_t i = 0; i < image_count; i++, w += 9) h = SetHashStep(h, SetHashImage(w, w[8]));
	for (size_t i = 0; i < sampler_count; i++, w += 5) h = SetHashStep(h, SetHashSampler(w, w[4]));
	return SetHashEnd(h, *w != 0u);
}

// The hash TableSetKey(program, snapshot, soft) would have, without building the key.
static uint64_t TableSetSnapshotHash(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                     const ShaderRecompiler::IR::ResourceSnapshot& snapshot, bool soft) {
	uint64_t h = kSetHashE;
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto& value = snapshot.images[i];
		h                 = SetHashStep(h, SetHashImage(value.dwords.data(), value.dword_count));
	}
	for (const auto& sampler: program.info.samplers) {
		const auto& value = snapshot.samplers[sampler.snapshot_index];
		h                 = SetHashStep(h, SetHashSampler(value.dwords.data(), value.dword_count));
	}
	return SetHashEnd(h, soft);
}

// The index of the first set in list order that TableSetMatches accepts (list.size() when none), calling TableSetMatches
// only for the sets whose stored hash equals the stage's.
static size_t TableSetFindHashed(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 const ShaderRecompiler::IR::ResourceSnapshot& snapshot, bool soft, const TableSetList& list) {
	if (list.empty()) return list.size();
	const uint64_t hash = TableSetSnapshotHash(program, snapshot, soft);
	for (size_t k = 0; k < list.size(); k++) {
		if (list[k]->key_hash == hash && TableSetMatches(program, snapshot, soft, list[k]->words)) return k;
	}
	return list.size();
}

static void TableSetsReport() {
	static uint32_t              calls_shared = 0; // the clock is read on every 256th call (once per stage preparation before)
	uint32_t&                    calls        = calls_shared;
	if ((++calls & 255u) != 0u || !Common::EnvRead::PerfLines()) return;
	static auto window = std::chrono::steady_clock::now();
	const auto  now    = std::chrono::steady_clock::now();
	if (now - window < std::chrono::seconds(5)) return;
	window   = now;
	auto& s  = TableStats();
	std::fprintf(stderr,
	             "PERF table_sets stages=%llu hits=%llu prep_fail=%llu rebind_hits=%llu rebind_fail=%llu stores=%llu\n",
	             static_cast<unsigned long long>(s.stages), static_cast<unsigned long long>(s.hits),
	             static_cast<unsigned long long>(s.prep_fail), static_cast<unsigned long long>(s.rebind_hits),
	             static_cast<unsigned long long>(s.rebind_fail), static_cast<unsigned long long>(s.stores));
	s = {};
}

// KYTY_TRACE_LONG_PREP: PrepareBindings' parts (t_long_prep_parts 1-7): 1 head (resets, trace switches, decode modes),
// 2 table-set key and proof, 3 image slots, 4 samplers, 5 set store and census, 6 bindless heaps, 7 shader data and GDS.
namespace {
struct BindSplit {
	bool                                  on = LongPrepPartsOn();
	std::chrono::steady_clock::time_point last {};
	BindSplit() {
		if (on) last = std::chrono::steady_clock::now();
	}
	void Mark(int part) {
		if (!on) return;
		const auto now = std::chrono::steady_clock::now();
		const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count());
		t_long_prep_parts[static_cast<size_t>(part)] += ns; // KYTY_TRACE_LONG_PREP
		last = now;
	}
};
} // namespace

// KYTY_DIAG_BIND_REPLAY=1 (ceiling test only, map 5.11 "ruch przez miasto", option A: the short draw path). A stage whose
// program and resolved descriptors (the snapshot's buffers, images, samplers, user data and bindless heap uses) equal those
// of an earlier stage takes that stage's PrepareBindings result instead of resolving its textures and buffers again; only
// its bindless heaps and image binding flags are prepared as today. The draw is still recorded (the image and the game's
// decisions stay), RebindImages/RebindBuffers still check and upload. A cached image that is gone or whose view is gone
// is a miss. Read once; BIND_REPLAY lines every ~5 s (stages, hits, stores).
namespace BindReplay {
struct Entry {
	PreparedBindings bindings;
};
struct Stats {
	uint64_t stages = 0, hits = 0, stores = 0, refused = 0, skipped = 0;
	std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_BIND_REPLAY");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "DIAG_BIND_REPLAY ceiling: stage bindings replayed by descriptor key" "\n");
		return set;
	}();
	return on;
}
std::unordered_map<uint64_t, Entry>& Map() { // GPU thread
	static std::unordered_map<uint64_t, Entry> map;
	return map;
}
Stats& GetStats() {
	static Stats stats;
	return stats;
}
template <typename T>
uint64_t Mix(uint64_t hash, const std::vector<T>& values) {
	const uint64_t words[] = {hash, values.empty() ? 0u : XXH3_64bits(values.data(), values.size() * sizeof(T)),
	                          static_cast<uint64_t>(values.size())};
	return XXH3_64bits(words, sizeof(words));
}
uint64_t Key(const ShaderStageRuntime& runtime) {
	const auto& snapshot = *runtime.resources;
	uint64_t    hash     = reinterpret_cast<uint64_t>(runtime.program);
	hash = Mix(hash, snapshot.buffers);
	hash = Mix(hash, snapshot.images);
	hash = Mix(hash, snapshot.samplers);
	hash = Mix(hash, snapshot.user_data);
	hash = Mix(hash, snapshot.bindless_heaps);
	return hash;
}
void Report() {
	auto& s = GetStats();
	if (std::chrono::steady_clock::now() - s.window < std::chrono::seconds(5)) return;
	s.window = std::chrono::steady_clock::now();
	std::fprintf(stderr, "BIND_REPLAY stages=%llu hits=%llu stores=%llu refused=%llu skipped=%llu entries=%zu" "\n",
	             static_cast<unsigned long long>(s.stages), static_cast<unsigned long long>(s.hits),
	             static_cast<unsigned long long>(s.stores), static_cast<unsigned long long>(s.refused),
	             static_cast<unsigned long long>(s.skipped), Map().size());
	s.stages = s.hits = s.stores = s.refused = s.skipped = 0;
}
} // namespace BindReplay

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	if (!BindReplay::On() || !GuestGpu::IsGpuThread() || !runtime.resources->indirect_ranges.empty()) {
		if (BindReplay::On()) BindReplay::GetStats().skipped++;
		PrepareBindingsToday(runtime, prepared);
		return;
	}
	auto&      stats = BindReplay::GetStats();
	const auto key   = BindReplay::Key(runtime);
	stats.stages++;
	auto& map   = BindReplay::Map();
	auto  found = map.find(key);
	if (found != map.end()) {
		auto& cache = m_context.GetTextureCache();
		bool  alive = true;
		for (const auto& binding: found->second.bindings.images) {
			if (!binding.image_id) continue;
			const auto* image = cache.m_slot_images.try_get(binding.image_id);
			if (image == nullptr || !image->registered || !binding.mip_views.empty() ||
			    (binding.image_view &&
			     std::ranges::none_of(image->views, [&](const CachedImageView& v) { return v.view == binding.image_view; }))) {
				alive = false;
				break;
			}
		}
		if (alive) {
			stats.hits++;
			prepared              = found->second.bindings;
			prepared.runtime      = &runtime;
			prepared.table_hit    = nullptr;
			prepared.table_store  = nullptr;
			std::fill(prepared.image_source.begin(), prepared.image_source.end(), 0u);
			for (const auto& binding: prepared.images) {
				if (binding.image_id) BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
			}
			const auto& snapshot = *runtime.resources;
			if (snapshot.bindless_heaps.empty() || !m_context.GetBindlessTable().Enabled()) {
				PrepareBindlessHeaps(runtime, prepared);
			} else {
				auto&           bindless_table = m_context.GetBindlessTable();
				std::lock_guard binder_lock(bindless_table.BinderMutex());
				PrepareBindlessHeaps(runtime, prepared);
			}
			BindReplay::Report();
			return;
		}
		stats.refused++;
		map.erase(found);
	}
	PrepareBindingsToday(runtime, prepared);
	if (map.size() >= 200000u) map.clear();
	BindReplay::Entry entry;
	entry.bindings = prepared;
	map.emplace(key, std::move(entry));
	stats.stores++;
	BindReplay::Report();
}

void RenderExecutor::PrepareBindingsToday(const ShaderStageRuntime& runtime,
                                          PreparedBindings& prepared) {
	BindSplit bind_split; // KYTY_TRACE_LONG_PREP
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.flattened_srt_uniform = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	prepared.image_source.resize(program.info.images.size(), 0u); // the table sets
	// KYTY_TRACE_CS_WRITER_ADDR: any stage (graphics too) whose written buffer covers the address.
	static constinit Common::OnceValue<uint64_t> writer_addr_once;
	const uint64_t writer_addr = writer_addr_once.Get([] { return Common::EnvRead::CsWriterAddr(); });
	if (writer_addr != 0) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (!program.info.buffers[i].written) continue;
			const auto r      = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
			const auto stride = std::max<uint64_t>(r.Stride(), 1u);
			if (writer_addr < r.Base48() || writer_addr >= r.Base48() + uint64_t {r.NumRecords()} * stride) continue;
			static std::mutex            mutex;
			static std::vector<uint64_t> seen;
			std::scoped_lock             lock(mutex);
			static uint32_t logged = 0;
			if (std::ranges::find(seen, program.shader_hash) == seen.end() || (logged < 40u && DebugStencilWindow())) {
				seen.push_back(program.shader_hash);
				logged++;
				std::string ud;
				for (size_t k = 0; k < std::min<size_t>(12u, snapshot.user_data.size()); k++) ud += fmt::format(" {:08x}", snapshot.user_data[k]);
				std::fprintf(stderr, "STAGE_WRITER frame=%u addr=0x%llx stage=%u hash=0x%016llx wbuf=%u base=0x%llx stride=%u records=%u bytes=0x%llx user_data=%s\n",
				             DebugPresentCount().load(), static_cast<unsigned long long>(writer_addr), static_cast<uint32_t>(program.stage),
				             static_cast<unsigned long long>(program.shader_hash), i,
				             static_cast<unsigned long long>(r.Base48()), r.Stride(), r.NumRecords(),
				             static_cast<unsigned long long>(uint64_t {r.NumRecords()} * stride), ud.c_str());
			}
		}
	}
	prepared.samplers.clear();
	prepared.shader_data.clear();
	bind_split.Mark(1);
	// The table sets: per slot, whether the binding may become a set member (g_slot_exact).
	static thread_local std::vector<uint8_t> table_exact;
	table_exact.assign(program.info.images.size(), 0u);
	const auto table_loop_epoch = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire);
	// One image binding of today's path (for the slots a table set does not serve).
	const auto prepare_image_slot = [&](uint32_t i) {
		DebugDepthStorageNulled() = false;
		KYTY_PROFILER_BLOCK("PrepareBindings::Image");
		const uint32_t compare_funcs = ImageCompareFuncs(program, snapshot, i);
		const auto     resolve       = [&] {
			return ResolveTexture(program.info.images[i], snapshot.images[i], compare_funcs);
		};
		TextureBinding binding = ResolveTextureReplay(m_context.GetTextureCache(), m_soft_texture_resolve, program, i,
		                                              snapshot.images[i], compare_funcs, resolve);
		if (i < table_exact.size()) table_exact[i] = g_slot_exact ? 1u : 0u;
		if (DescriptorTrace::ImageBindingsTraced()) DescriptorTrace::ImageBinding(m_context, program, i, binding);
		if (DebugDepthStorageNulled()) {
			static std::mutex            mutex;
			static std::vector<uint64_t> seen;
			std::scoped_lock             lock(mutex);
			if (std::ranges::find(seen, program.shader_hash) == seen.end() && seen.size() < 32u) {
				seen.push_back(program.shader_hash);
				std::fprintf(stderr, "DIAG_DEPTH_STORAGE_NULL shader stage=%u hash=0x%016llx image=%u\n",
				             static_cast<uint32_t>(program.stage), static_cast<unsigned long long>(program.shader_hash), i);
			}
		}
		KYTY_PROFILER_BLOCK("PrepareBindings::Store");
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	};
	// The stage's table set and its PrepareBindings proof.
	const bool table_sets = (!program.info.images.empty() || !program.info.samplers.empty()) && TableSetsUsable();
	TableSet* table_set = nullptr; // the set stored under this stage's key
	TableSet* table_use = nullptr; // proven: its members stand for today's lookups
	// This program's list once the cache found it (the store below does not look it up again).
	TableSetList*  table_list = nullptr;
	prepared.table_hit    = nullptr;
	prepared.table_store  = nullptr;
	if (table_sets) {
		TableStats().stages++;
		table_list = &TableSetListCached(&program);
		auto&        list  = *table_list;
		const size_t found = TableSetFindHashed(program, snapshot, m_soft_texture_resolve, list);
		if (found < list.size()) {
			if (found != 0u) std::swap(list[0], list[found]); // the last found first (sets stay where they are)
			table_set = list[0].get();
		}
		if (table_set != nullptr && table_set->views_ready) {
			bool   pages  = true;
			size_t member = 0;
			for (size_t i = 0; i < table_set->images.size() && pages; i++) {
				if (table_set->memo[i] == 0u) continue;
				const auto& data = table_set->images[i].desc.info.data;
				pages = DrawRepeat::ImagePageEpochSum(data.address, data.size) == table_set->page_sums[member++];
			}
			const bool ok = pages && m_context.GetTextureCache().TableReplayStage(table_set->member_ids);
			if (ok) {
				table_use             = table_set;
				prepared.table_epoch  = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire);
				TableStats().hits++;
			} else {
				TableStats().prep_fail++;
			}
		}
	}
	bind_split.Mark(2);
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (table_use != nullptr && table_use->memo[i] != 0u) {
			const auto& stored = table_use->images[i];
			BindImage(stored.image_id, false);
			auto& slot      = prepared.images[i];
			slot.image_view = nullptr;
			slot.mip_views.clear();
			// The slot still holds this set slot's binding when it was last filled from it (the layout is set again
			// by CommitBindings before any use): the ~1 KB description copy is skipped.
			if (prepared.image_source[i] != table_use->version || slot.image_id != stored.image_id) {
				slot.image_id             = stored.image_id;
				slot.desc                 = stored.desc;
				slot.layout               = stored.layout;
				prepared.image_source[i]  = table_use->version;
			}
		} else {
			prepared.image_source[i] = 0u;
			prepare_image_slot(i);
		}
	}
	bind_split.Mark(3);
	if (table_use != nullptr) {
		prepared.samplers = table_use->samplers;
	} else {
	prepared.samplers.reserve(program.info.samplers.size());
	for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
		KYTY_PROFILER_BLOCK("PrepareBindings::Sampler");
		prepared.samplers.push_back(NativeSampler(
		    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
	}
	}
	bind_split.Mark(4);
	if (table_sets) {
		// Hand the hit to RebindImages or store the set (views come in RebindImages).
		const uint32_t frame = DebugPresentCount().load();
		auto&          cache = m_context.GetTextureCache();
		if (table_use != nullptr) {
			prepared.table_hit = table_use;
			table_use->frame   = frame;
		} else {
			auto& list = *table_list; // found above
			if (table_set == nullptr) {
				if (list.size() >= 16u) {
					// The least recently used set not used in this frame (a stage of this event may hold one).
					auto oldest = list.end();
					for (auto it = list.begin(); it != list.end(); ++it) {
						if ((*it)->frame != frame && (oldest == list.end() || (*it)->frame < (*oldest)->frame)) oldest = it;
					}
					if (oldest != list.end()) {
						list.erase(oldest);
					}
				}
				if (list.size() < 16u) {
					list.push_back(std::make_unique<TableSet>());
					table_set = list.back().get();
				}
			}
			if (table_set != nullptr && table_set->frame != frame &&
			    DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire) == table_loop_epoch) {
				TableSetKey(program, snapshot, m_soft_texture_resolve, table_set->words);
				table_set->key_hash = TableSetWordsHash(program.info.images.size(), program.info.samplers.size(), table_set->words);
				table_set->frame       = frame;
				table_set->views_ready = false;
				table_set->images      = prepared.images;
				table_set->samplers    = prepared.samplers;
				table_set->memo.assign(prepared.images.size(), 0u);
				table_set->page_sums.clear();
				table_set->member_ids.clear();
				for (size_t i = 0; i < prepared.images.size(); i++) {
					const auto& binding = prepared.images[i];
					const bool  memo    = table_exact[i] != 0u && binding.desc.type == TextureCache::BindingType::Texture &&
					                  (binding.desc.info.data.Empty() || cache.ImageReplayable(binding.image_id));
					if (!memo) continue;
					table_set->memo[i] = 1u;
					table_set->page_sums.push_back(
					    DrawRepeat::ImagePageEpochSum(binding.desc.info.data.address, binding.desc.info.data.size));
					table_set->member_ids.push_back(binding.image_id);
				}
				static std::atomic<uint64_t> version {0};
				table_set->version   = version.fetch_add(1, std::memory_order_relaxed) + 1u;
				for (size_t i = 0; i < prepared.images.size(); i++) {
					if (table_set->memo[i] != 0u) prepared.image_source[i] = table_set->version;
				}
				prepared.table_store = table_set;
				TableStats().stores++;
			}
		}
		TableSetsReport();
	}
	bind_split.Mark(5);
	// A stage without bindless heaps (or with the table off) skips the binder lock block below: its PrepareBindlessHeaps only
	// clears the stage's bindless lists.
	if (snapshot.bindless_heaps.empty() || !m_context.GetBindlessTable().Enabled()) {
		PrepareBindlessHeaps(runtime, prepared);
	} else {
		KYTY_PROFILER_BLOCK("PrepareBindings::Bindless");
		auto&           bindless_table = m_context.GetBindlessTable();
		std::lock_guard binder_lock(bindless_table.BinderMutex());
		const DispatchSubTimer heaps_timer(DispatchSub::BindlessHeaps); // KYTY_TRACE_DISPATCH_CENSUS
		PrepareBindlessHeaps(runtime, prepared);
	}
	bind_split.Mark(6);
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
	bind_split.Mark(7);
}

// The guest range of a buffer V# (FindBuffers): nothing for a null V# (base or size 0). `for_each_read` calls its
// argument with every scalar read (address, size) of the draw's stages.
template <typename ForEachRead>
static std::optional<std::pair<uint64_t, uint64_t>> VSharpRange(RenderContext&                                  renderer,
                                                                const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                                                uint32_t                                        resource,
                                                                const ShaderRecompiler::IR::DescriptorValue&    value,
                                                                ForEachRead&& for_each_read) {
	const auto& metadata   = program.info.buffers[resource];
	const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(value);
	const auto  address    = descriptor.Base48();
	auto        size       = descriptor.GetSize();
	if (address == 0 || size == 0) return std::nullopt;
	if (descriptor.NumRecords() == UINT32_MAX) {
		if (metadata.written) {
			// Sentinel write ranges end before tables captured by any bound stage.
			for_each_read([&](uint64_t read_address, uint64_t read_size) {
				if (read_size == 0) return;
				if (read_address > address) {
					size = std::min(size, read_address - address);
				} else if (address - read_address < read_size) {
					EXIT("scalar resource reads overlap a shader buffer write\n");
				}
			});
		} else {
			const auto& graphics = renderer.GetGraphics();
			const auto limit = uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
			                   (graphics.StorageMinAlignment() - 1);
			size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
		}
	}
	{
		// M109: an invalid V# address exits with the shader and resource named.
		const Libs::LibKernel::Memory::ClampContext context {program.shader_hash, resource, metadata.written,
		                                                    value.dwords.data()};
		const Libs::LibKernel::Memory::ClampContextScope clamp_context(context);
		size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
	}
	return std::pair {address, size};
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	// Each storage-buffer binding's memo entry is fetched now, for the ObtainBuffer of this stage's RebindBuffers (once per call).
	const auto prefetch_memo = [&](const PreparedBindings::BufferSource& source,
	                               const ShaderRecompiler::IR::BufferResource& metadata) {
		if (source.address == 0 || source.size == 0) return;
		BufferCache::PrefetchObtainMemo(source.address, source.size, metadata.written,
		                                metadata.formatted || (metadata.read && Config::SyncRawImageBuffersEnabled()));
	};
	for (auto* stage: stages) {
		auto& prepared = *stage;
		EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		// Each binding first asks the resolve thread's record of its stage's snapshot.
		const StageF::ResolveRecord* const lookup_record = prepared.runtime->stagef_record;
		// Record-keyed V#s: the record-keyed ranges get their buffers here, before any binding is obtained (a
		// buffer created later could merge one that an earlier binding already refers to). Only the committed part of
		// a range can be resident: the BDA page table maps nothing else.
		prepared.indirect_ranges.clear();
		for (uint32_t index = 0; index < snapshot.indirect_ranges.size(); index++) {
			const auto& range = snapshot.indirect_ranges[index];
			auto& target    = prepared.indirect_ranges.emplace_back();
			target.address  = range.address;
			target.size     = range.size;
			target.written  = range.written;
			if (const auto* ahead = StageF::RangeSourceAhead(lookup_record, index, range); ahead != nullptr) {
				target.resident = ahead->size;
				target.id       = ahead->id;
				continue;
			}
			// A candidate V# whose range starts outside mapped memory (all-records lists / _VALUE_BOUND list
			// records that need not hold a real V#) has nothing resident: the BDA page table maps nothing there either,
			// so no store of the shader can land in it.
			uint64_t generation = 0;
			target.resident     = Libs::LibKernel::Memory::ClampRangeSizeAhead(range.address, range.size, &generation);
			// Memory without a guest backing (e.g. the executable's data, where such a record's "V#" can point) is no
			// memory the GPU writes either.
			if (uint8_t probe = 0; target.resident != 0u &&
			                       (!Libs::LibKernel::Memory::TryReadBacking(target.address, &probe, 1u) ||
			                        !Libs::LibKernel::Memory::TryReadBacking(target.address + target.resident - 1u, &probe, 1u)))
				target.resident = 0u;
			if (target.resident != 0u) target.id = cache.FindBuffer(target.address, target.resident);
		}
		// KYTY_RECORD_KEYED_VERIFY: calls whose listing is incomplete keep unlisted BDA stores (their pages count as
		// uncovered); the summary shows how many ran.
		if ((BufferCache::RecordKeyedVerifyEnabled() || BufferCache::D6C0Enabled()) && // and KYTY_D6_C0
		    !program.info.indirect_ranges.empty() && !snapshot.indirect_ranges_complete) {
			cache.NoteRecordKeyedIncomplete();
			// D6 step K0 (KYTY_TRACE_PERF: PERF workaround_hits bda_unlisted, third field).
			Common::WorkaroundHits::Note(Common::WorkaroundHits::Hit::BdaIncompleteCalls);
		}
		// D6 step K2: (without KYTY_RECORD_KEYED_VERIFY, which only counts it) a store
		// through a written record-keyed source the listing left out would change bytes nobody knows the GPU owns (CPU
		// reads and uploads would take the old ones): no exact answer, so the emulator stops with the shader's name. The
		// RECORD_KEYED_PROVEN_BAD / RECORD_KEYED_GAME_COUNT_BAD lines before it say which source and why.
		if (!program.info.indirect_ranges.empty() && !snapshot.indirect_writes_complete &&
		    !BufferCache::RecordKeyedVerifyEnabled()) {
			EXIT("D6: shader 0x%016llx stores through a GPU-selected V# that its record-time listing does not cover "
			     "(KYTY_RECORD_KEYED_VERIFY=1 counts such stores instead)\n",
			     static_cast<unsigned long long>(program.shader_hash));
		}
		const auto& layout = program.bindings;
		prepared.buffer_sources.resize(layout.BufferBindings());
		if (layout.BufferBindings() == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			// The resolve thread's source when it still holds.
			if (const auto* ahead = StageF::BufferSourceAhead(lookup_record, resource, snapshot.buffers[resource],
			                                                  metadata.written);
			    ahead != nullptr) {
				source = {ahead->address, ahead->size, ahead->id};
				prefetch_memo(source, metadata);
				continue;
			}
			const auto range = VSharpRange(m_context, program, resource, snapshot.buffers[resource], [&](auto&& read) {
				for (const auto* reader: stages) {
					for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
						read(read_address, read_size);
					}
				}
			});
			if (!range.has_value()) {
				source = {};
				continue;
			}
			const auto [address, size] = *range;
			source = {address, size, cache.FindBuffer(address, size)};
			prefetch_memo(source, metadata);
		}
	}
}

// D4b step 1 (append windows): the append pools of a stage (BufferResource::append_counter) - qualified ones
// with their counter address, record stride and bytes, the others by their bound range (defined after
// HasFixedBufferWrites).
static void CollectAppendBindings(const PreparedBindings& prepared, std::vector<uint64_t>* counters,
                                  std::vector<BufferCache::AppendBinding>*             bindings,
                                  std::vector<std::pair<uint64_t, uint64_t>>*           unqualified);

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	const auto  buffer_count = layout.BufferBindings();
	EXIT_IF(prepared.buffer_sources.size() != buffer_count);

	prepared.buffers.resize(buffer_count);
	DebugWriterHash().store(program.shader_hash, std::memory_order_relaxed);
	// D4b step 1 (append windows): the stage's own counters are no foreign writes while its bindings are obtained;
	// its windows open (counter copied) once they are, before the draw or dispatch is recorded (any return below).
	struct AppendScope {
		BufferCache&            cache;
		const PreparedBindings& stage;
		bool                    on = false;
		~AppendScope() {
			if (!on) return;
			std::vector<BufferCache::AppendBinding>      bindings;
			std::vector<std::pair<uint64_t, uint64_t>> unqualified;
			CollectAppendBindings(stage, nullptr, &bindings, &unqualified);
			cache.AppendDispatch(bindings, unqualified);
		}
	} append_scope {m_context.GetBufferCache(), prepared};
	if (std::ranges::any_of(program.info.buffers, [](const auto& b) {
		    return b.append_counter != ShaderRecompiler::IR::BufferResource::NoAppendCounter;
	    })) {
		std::vector<uint64_t> counters;
		CollectAppendBindings(prepared, &counters, nullptr, nullptr);
		m_context.GetBufferCache().AppendSetDispatchCounters(std::move(counters));
		append_scope.on = true;
	}
	// The upload batch: the read-only synchronizations below (ObtainBdaRange, ObtainBuffer) only claim their pages;
	// the claimed uploads are recorded when the scope ends, before the draw or dispatch that uses these bindings. Nothing
	// below reads a cached buffer's contents on the GPU before then (the descriptors only name the buffers), and every
	// buffer-cache operation that must see the uploads commits the batch itself.
	const BufferCache::UploadBatchScope upload_batch(m_context.GetBufferCache());
	std::optional<DispatchSubTimer>     sub_timer(std::in_place, DispatchSub::BdaRanges); // KYTY_TRACE_DISPATCH_CENSUS
	// Record-keyed V#s: resident and synchronized before the shader reaches them through BDA, and written
	// ranges GPU-modified, like the written storage buffers below (NativeStorageBuffer). First, so that a read-only
	// binding of the same bytes cannot take a stream copy of what the shader writes through BDA.
	for (const auto& range: prepared.indirect_ranges) {
		if (range.resident == 0u) continue;
		(void)m_context.GetBufferCache().ObtainBdaRange(range.address, range.resident, range.written, range.id);
		if (range.written) {
			if (TextureCache::OwnerMode() != 0u) {
				// D4c (KYTY_FIX_IMAGE_MEMORY_OWNER write-back): a written device-address range.
				TextureCache::OwnerCensusBinding facts;
				facts.kind = 2;
				TextureCache::OwnerCensusSetBinding(&facts);
				m_context.GetTextureCache().InvalidateMemoryFromGPU(range.address, range.resident);
				TextureCache::OwnerCensusSetBinding(nullptr);
			} else {
				m_context.GetTextureCache().InvalidateMemoryFromGPU(range.address, range.resident);
			}
			if (BufferCache::RecordKeyedVerifyEnabled() || BufferCache::D6C0Enabled()) // KYTY_RECORD_KEYED_VERIFY, KYTY_D6_C0
				m_context.GetBufferCache().NoteRecordKeyedWrite(range.address, range.resident);
		}
	}
	sub_timer.reset();
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (uint32_t resource = 0; resource < program.info.buffers.size(); resource++) {
		const auto& buffer = program.info.buffers[resource];
		const auto  index  = buffer.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		KYTY_PROFILER_BLOCK("RebindBuffers::Storage");
		prepared.buffers[index] = NativeStorageBuffer(m_context, prepared.buffer_sources[index], buffer, buffer_offset);
		pack_memory_offset(index, buffer_offset);
		if (layout.memory_stride_dword != ShaderRecompiler::IR::PushData::NoStart) {
			// The record stride the shader reads at run time.
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			prepared.shader_data[layout.memory_stride_dword + index] = descriptor.Type() == 0u ? descriptor.Stride() : 0u;
		}
	}
	const bool srt_uniform = layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrtUniform)] != 0;
	// Uniform const buffers (ShaderIR.h): words uploaded once, padded with zeros to whole uvec4 elements; the uniform view
	// covers the padded words (the uvec4 elements the shader declares lie inside: every ReadConst slot is below the
	// plan's SRT read count, which the flat buffer always holds), the storage view the words themselves.
	const auto upload_uniform = [&](std::span<const uint32_t> words, vk::DescriptorBufferInfo* storage_view,
	                                vk::DescriptorBufferInfo* uniform_view) {
		m_uniform_words.assign(words.begin(), words.end());
		m_uniform_words.resize((words.size() + 3u) & ~size_t {3}, 0u);
		const auto view = NativeUpload(m_context, m_uniform_words);
		if (storage_view != nullptr) *storage_view = {view.buffer, view.offset, words.size_bytes()};
		*uniform_view = {view.buffer, view.offset,
		                 std::min<vk::DeviceSize>(view.range, ShaderRecompiler::IR::UniformBufferMaxBytes)};
	};
	const auto upload_srt = [&](std::span<const uint32_t> words) {
		if (srt_uniform) {
			upload_uniform(words, &prepared.flattened_srt, &prepared.flattened_srt_uniform);
		} else {
			prepared.flattened_srt = NativeUpload(m_context, words);
		}
	};
	sub_timer.emplace(DispatchSub::SrtUpload);
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0 || srt_uniform) {
		if (DescriptorTrace::SrtDumpOn()) DescriptorTrace::SrtDump(program.shader_hash, snapshot.flattened_srt);
		if (prepared.bindless_patches.empty()) {
			upload_srt(snapshot.flattened_srt);
		} else {
			m_bindless_srt.assign(snapshot.flattened_srt.begin(), snapshot.flattened_srt.end());
			for (const auto& patch: prepared.bindless_patches) {
				if (static_cast<size_t>(patch.offset) + 1u < m_bindless_srt.size()) {
					m_bindless_srt[patch.offset]      = patch.region;
					m_bindless_srt[patch.offset + 1u] = patch.entries;
				}
			}
			upload_srt(m_bindless_srt);
		}
	}
	sub_timer.emplace(DispatchSub::DataUpload);
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		upload_uniform(prepared.shader_data, nullptr, &prepared.shader_data_buffer);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	const auto rebind_image_slot = [&](uint32_t i) {
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				ClearImageBinding(*old_image, images[i].image_id);
			}
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i],
			                           ImageCompareFuncs(program, snapshot, i));
			if (i < prepared.image_source.size()) prepared.image_source[i] = 0u;
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTextureReplay(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTextureReplay(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
		image.usage.texture |= !storage;
		// KYTY_WORKAROUND_DCC_FAST_CLEAR: after the view's contents are final for this use.
		if (TextureCache::DccFastClearEnabled()) {
			texture_cache.ApplyDccFastClear(m_context.GetCommandScheduler().Current(), binding.image_id, binding.desc);
		}
	};
	// The table sets: a hit stage takes its set's views when no image registration changed since its proof and
	// FindTexture would only touch every member (TableTexturesReady); members have no metadata, so the DCC fast-clear
	// step and the storage flag do nothing for them.
	auto* table_hit = static_cast<TableSet*>(prepared.table_hit);
	bool  table_ok  = false;
	if (table_hit != nullptr) {
		table_ok = DrawRepeat::g_image_registry_epoch.load(std::memory_order_acquire) == prepared.table_epoch &&
		           texture_cache.TableTexturesReady(table_hit->member_ids);
		(table_ok ? TableStats().rebind_hits : TableStats().rebind_fail)++;
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (table_ok && table_hit->memo[i] != 0u) {
			auto&       binding = images[i];
			const auto& stored  = table_hit->images[i];
			binding.image_view  = stored.image_view;
			binding.mip_views   = stored.mip_views;
			texture_cache.GetImage(binding.image_id).usage.texture = true;
		} else {
			rebind_image_slot(i);
		}
	}
	if (auto* store = static_cast<TableSet*>(prepared.table_store); store != nullptr) {
		// The set's views: today's, for the images the set was stored with (a slot resolved again here keeps it unready).
		bool same = true;
		for (size_t i = 0; i < store->images.size() && i < images.size(); i++) {
			if (store->memo[i] != 0u && store->images[i].image_id != images[i].image_id) same = false;
		}
		if (same) {
			for (size_t i = 0; i < store->images.size() && i < images.size(); i++) {
				store->images[i].image_view = images[i].image_view;
				store->images[i].mip_views  = images[i].mip_views;
			}
			store->views_ready = true;
		}
		prepared.table_store = nullptr;
	}
	prepared.table_hit = nullptr;
}

// PrepareGraphicsBindings: a color target whose image was replaced (unregistered with data, or to be rebound) is found
// again there.
static bool ColorTargetStale(const Image* image) {
	return image == nullptr || (!image->registered && !image->info.data.Empty()) || image->binding.needs_rebind;
}

namespace {
// KYTY_TRACE_LONG_PREP: the times of PrepareGraphicsBindings' parts (t_long_prep_parts 8-12), each from the previous mark.
struct GraphicsPartMarks {
	bool                                  on   = LongPrepPartsOn();
	std::chrono::steady_clock::time_point last = on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	void Mark(size_t part) {
		if (!on) return;
		const auto now = std::chrono::steady_clock::now();
		t_long_prep_parts[part] +=
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - last).count());
		last = now;
	}
};
} // namespace

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	PrepareGraphicsBuffersHead(stages);
	PrepareGraphicsImages(stages, colors);
	PrepareGraphicsBuffersTail(stages);
}

void RenderExecutor::PrepareGraphicsBuffersHead(std::span<PreparedBindings* const> stages) {
	GraphicsPartMarks marks;
	bool uses_dma = false;
	FindBuffers(stages);
	marks.Mark(8);
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		m_context.PrepareBda();
	}
	marks.Mark(9);
}

void RenderExecutor::PrepareGraphicsImages(std::span<PreparedBindings* const> stages, std::span<RenderColorInfo> colors) {
	GraphicsPartMarks marks;
	for (auto* stage: stages) {
		RebindImages(*stage);
	}
	marks.Mark(10);
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (ColorTargetStale(old_image)) {
			if (old_image != nullptr) {
				ClearImageBinding(*old_image, target.image_id);
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	marks.Mark(11);
}

void RenderExecutor::PrepareGraphicsBuffersTail(std::span<PreparedBindings* const> stages) {
	GraphicsPartMarks marks;
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	// The upload batch: the read-only buffer synchronizations of all stages write-protect their pages with one
	// update per tracker region when the scope ends (each stage's RebindBuffers scope is then an inner one).
	{
		const BufferCache::UploadBatchScope upload_batch(m_context.GetBufferCache());
		for (auto* stage: stages) {
			RebindBuffers(*stage);
		}
	}
	marks.Mark(12);
}

// Which bytes of an image a draw writes, for the scalar-read check (D6 step K5): the bytes the draw's own descriptions name
// (ImageWrites): a color target's bound level and layers plus its metadata, a depth target's data, stencil and HTILE, a
// written storage image's whole description - the registered image can be a larger image the texture cache matched to the
// description (its level 0 then lies elsewhere: guest mip chains run backwards), so its ranges are no answer; an id no
// description names keeps them. A read the registered ranges hit but the draw does not write is no overlap.
// SCALAR_OVERLAP_EXACT lines (first 32 per class), SCALAR_OVERLAP_EXACT_STATS every 64 classed hits, PERF workaround_hits
// scalar_exact.
namespace ScalarExact {

enum class Verdict { Inside, Outside, Unknown };

struct Totals {
	std::atomic<uint64_t> both {0}, fallback {0}, narrowed {0}, events {0};
	std::atomic<uint32_t> logged[3] {};
};

Totals& Get() {
	static Totals totals;
	return totals;
}

void Report() {
	auto&      t = Get();
	const auto n = [](const std::atomic<uint64_t>& v) { return static_cast<unsigned long long>(v.load(std::memory_order_relaxed)); };
	std::fprintf(stderr, "SCALAR_OVERLAP_EXACT_STATS both=%llu fallback=%llu narrowed=%llu\n", n(t.both), n(t.fallback),
	             n(t.narrowed));
}

void Count(std::atomic<uint64_t>& counter, Common::WorkaroundHits::Hit hit) {
	counter.fetch_add(1, std::memory_order_relaxed);
	Common::WorkaroundHits::Note(hit); // KYTY_TRACE_PERF: PERF workaround_hits scalar_exact
	if (Get().events.fetch_add(1, std::memory_order_relaxed) % 64u == 0u) Report();
}

// The guest ranges `id`'s descriptions in this commit name; false when none names it.
bool ImageWrites(const CommitTargets* targets, std::span<PreparedBindings* const> stages, ImageId id,
                 std::vector<GuestRange>& out) {
	out.clear();
	bool named = false;
	const auto add = [&](uint64_t address, uint64_t size) {
		if (size != 0u) out.push_back({address, size});
	};
	if (targets != nullptr) {
		for (uint32_t i = 0; i < targets->color_count && targets->colors != nullptr; ++i) {
			const auto& color = targets->colors[i];
			if (!color.image_id || color.image_id != id) continue;
			named = true;
			color.desc.info.ForEachLevelLayerBytes(color.guest_mip_level, color.guest_array_layer,
			                                       color.desc.view_info.layer_count, add);
			add(color.desc.info.metadata.range.address, color.desc.info.metadata.range.size);
		}
		if (targets->depth != nullptr && targets->depth->image_id && targets->depth->image_id == id) {
			named = true;
			const auto& info = targets->depth->desc.info;
			add(info.data.address, info.data.size);
			add(info.stencil.address, info.stencil.size);
			add(info.metadata.range.address, info.metadata.range.size);
		}
	}
	for (const auto* stage: stages) {
		for (const auto& binding: stage->images) {
			if (!binding.image_id || binding.image_id != id ||
			    binding.desc.type != TextureCache::BindingType::Storage) continue;
			named = true;
			add(binding.desc.info.data.address, binding.desc.info.data.size);
			add(binding.desc.info.stencil.address, binding.desc.info.stencil.size);
			add(binding.desc.info.metadata.range.address, binding.desc.info.metadata.range.size);
		}
	}
	return named;
}

Verdict Classify(const std::vector<GuestRange>& writes, bool named, uint64_t address, uint64_t size) {
	if (!named) return Verdict::Unknown;
	for (const auto& range: writes) {
		if (ImageRangeOverlaps(address, size, range.address, range.size)) return Verdict::Inside;
	}
	return Verdict::Outside;
}

void Log(uint32_t kind, const char* name, uint64_t reader, uint64_t address, uint64_t size, ImageId id,
         const Image* image, const std::vector<GuestRange>& writes) {
	if (Get().logged[kind].fetch_add(1, std::memory_order_relaxed) >= 32u) return;
	std::string ranges;
	for (const auto& range: writes) ranges += fmt::format("{}{:x}+{:x}", ranges.empty() ? "" : ",", range.address, range.size);
	std::fprintf(stderr,
	             "SCALAR_OVERLAP_EXACT class=%s frame=%u reader=%016llx read=0x%llx+0x%llx image=%u reg=0x%llx+0x%llx "
	             "%ux%u L%u x%u written=[%s] marker=[%s]\n",
	             name, DebugPresentCount().load(), static_cast<unsigned long long>(reader),
	             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
	             static_cast<unsigned>(id.index), static_cast<unsigned long long>(image != nullptr ? image->info.data.address : 0),
	             static_cast<unsigned long long>(image != nullptr ? image->info.data.size : 0),
	             image != nullptr ? image->info.extent.width : 0u, image != nullptr ? image->info.extent.height : 0u,
	             image != nullptr ? image->info.resources.levels : 0u, image != nullptr ? image->info.resources.layers : 0u,
	             ranges.c_str(), DebugMarkerPath().c_str());
}

} // namespace ScalarExact

namespace {

struct ScalarWriteRange {
	uint64_t begin = 0;
	uint64_t end   = 0;
};

} // namespace

// The scalar-read scan first asks whether a read intersects any written range of the commit (ranges sorted by begin
// with a running maximum of their ends: the ranges beginning before the read's end reach past its begin iff that maximum
// does). A read that intersects none would report no hit, so it skips the per-range loops; the others run them unchanged,
// in the same order.

// A written range as [begin, end), checked once the way ImageRangeOverlaps checks it per call.
static ScalarWriteRange MakeScalarWriteRange(uint64_t address, uint64_t size) {
	if (size == 0 || address > UINT64_MAX - size) EXIT("invalid image overlap range\n");
	return {address, address + size};
}


// Whether buffer resource `resource` of a stage writes only the fixed ranges the recompiler recorded: every write to
// it has an address fixed in the shader, and its descriptor is direct, unswizzled and adds no thread id to the index.
static bool HasFixedBufferWrites(const PreparedBindings& writer, uint32_t resource) {
	using ShaderRecompiler::IR::BufferResource;
	const auto& buffer = writer.runtime->program->info.buffers[resource];
	if (buffer.descriptor_index == UINT32_MAX || !buffer.written || !buffer.fixed_writes ||
	    buffer.fixed_write_ranges.empty() || buffer.indirect_root != BufferResource::NoIndirectBuffer ||
	    !buffer.indirect_resources.empty() || buffer.image_alias != BufferResource::NoImageAlias) {
		return false;
	}
	const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(writer.runtime->resources->buffers[resource]);
	return !descriptor.SwizzleEnabled() && !descriptor.AddTid() &&
	       descriptor.Base48() == writer.buffer_sources[buffer.descriptor_index].address;
}

static void CollectAppendBindings(const PreparedBindings& prepared, std::vector<uint64_t>* counters,
                                  std::vector<BufferCache::AppendBinding>*             bindings,
                                  std::vector<std::pair<uint64_t, uint64_t>>*           unqualified) {
	using ShaderRecompiler::IR::BufferResource;
	const auto& info = prepared.runtime->program->info;
	for (uint32_t r = 0; r < info.buffers.size(); r++) {
		const auto& pool = info.buffers[r];
		if (pool.append_counter == BufferResource::NoAppendCounter || pool.descriptor_index == UINT32_MAX || !pool.written) {
			continue;
		}
		const auto& source = prepared.buffer_sources[pool.descriptor_index];
		if (source.address == 0 || source.size < BufferCache::AppendMinPoolBytes) continue; // small pools: as today
		const auto c       = pool.append_counter;
		bool       ok      = c < info.buffers.size() && HasFixedBufferWrites(prepared, c) &&
		                     info.buffers[c].fixed_write_ranges.size() == 1u;
		uint64_t   counter = 0;
		if (ok) {
			const auto& counter_source = prepared.buffer_sources[info.buffers[c].descriptor_index];
			const auto  range          = info.buffers[c].fixed_write_ranges[0];
			counter                    = counter_source.address + range.first;
			ok = range.second - range.first == 4u && counter_source.size >= range.second;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(prepared.runtime->resources->buffers[r]);
		ok = ok && !descriptor.SwizzleEnabled() && !descriptor.AddTid() && descriptor.Type() == 0u &&
		     descriptor.Stride() != 0u && descriptor.Base48() == source.address &&
		     pool.indirect_root == BufferResource::NoIndirectBuffer && pool.indirect_resources.empty() &&
		     pool.image_alias == BufferResource::NoImageAlias &&
		     (counter + 4u <= source.address || source.address + source.size <= counter);
		if (ok && counters != nullptr) counters->push_back(counter);
		if (bindings == nullptr) continue;
		if (!ok) {
			unqualified->emplace_back(source.address, source.size);
			continue;
		}
		BufferCache::AppendBinding binding;
		binding.pool      = source.address;
		binding.pool_size = source.size;
		binding.counter   = counter;
		binding.threads   = uint64_t {1} << 40u; // no bound from here: foreign counter writes are caught directly
		binding.stride    = descriptor.Stride();
		binding.min_off   = pool.append_min_offset;
		binding.max_end   = pool.append_max_end;
		bindings->push_back(binding);
	}
}

// The guest bytes buffer resource `resource` of a stage can write, as emit(address, size) calls: its fixed write
// ranges within the bound range when HasFixedBufferWrites, otherwise the whole bound range; nothing when the resource
// is not bound or not written.
template <typename Emit>
static void ForEachWrittenBufferRange(const PreparedBindings& writer, uint32_t resource, Emit&& emit) {
	const auto& buffer = writer.runtime->program->info.buffers[resource];
	if (!buffer.written || buffer.descriptor_index == UINT32_MAX) return;
	const auto& written = writer.buffer_sources[buffer.descriptor_index];
	if (written.size == 0) return;
	if (!HasFixedBufferWrites(writer, resource)) {
		emit(written.address, written.size);
		return;
	}
	for (const auto& [begin, end]: buffer.fixed_write_ranges) {
		// Bytes past the bound range are out of bounds: those writes are dropped.
		if (begin >= written.size) continue;
		emit(written.address + begin, std::min<uint64_t>(end, written.size) - begin);
	}
}

// Every guest range a stage can write that the scalar-read check tests, in one order for all its variants: the written
// buffer bindings (ForEachWrittenBufferRange), then the written ranges of its record-keyed V#s (record-keyed V#s,
// whole: a part past committed memory is no guest memory to read either).
template <typename Emit>
static void ForEachWrittenRange(const PreparedBindings& writer, Emit&& emit) {
	for (uint32_t resource = 0; resource < writer.runtime->program->info.buffers.size(); ++resource) {
		ForEachWrittenBufferRange(writer, resource, emit);
	}
	for (const auto& range: writer.indirect_ranges) {
		if (range.written && range.size != 0u) emit(range.address, range.size);
	}
}

// A stage writes through addresses unknown at record time: untracked address writes, or record-keyed V#s whose ranges
// the materialization did not list completely (record-keyed V#s).
static bool HasUnknownAddressWrites(const PreparedBindings& writer) {
	const auto& program = *writer.runtime->program;
	return program.has_address_writes ||
	       (!program.info.indirect_ranges.empty() && !writer.runtime->resources->indirect_ranges_complete);
}

// PERF bindless_layout_walk (bindless E1 cost): commits that walked their bindless image list, images walked, time.
namespace {
std::atomic<uint64_t> g_layout_walks {0}, g_layout_walk_images {0}, g_layout_walk_ns {0};
class LayoutWalkTimer {
public:
	explicit LayoutWalkTimer(size_t images): m_on(PerfEnabled()) {
		if (!m_on) return;
		m_start = std::chrono::steady_clock::now();
		g_layout_walks.fetch_add(1, std::memory_order_relaxed);
		g_layout_walk_images.fetch_add(images, std::memory_order_relaxed);
	}
	~LayoutWalkTimer() {
		if (m_on)
			g_layout_walk_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                                     std::chrono::steady_clock::now() - m_start)
			                                                     .count()),
			                           std::memory_order_relaxed);
	}
	LayoutWalkTimer(const LayoutWalkTimer&)            = delete;
	LayoutWalkTimer& operator=(const LayoutWalkTimer&) = delete;

private:
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
};
} // namespace

void ReportBindlessLayoutWalk() {
	std::fprintf(stderr, "PERF bindless_layout_walk walks=%llu images=%llu ms=%llu\n",
	             static_cast<unsigned long long>(g_layout_walks.exchange(0)),
	             static_cast<unsigned long long>(g_layout_walk_images.exchange(0)),
	             static_cast<unsigned long long>(g_layout_walk_ns.exchange(0) / 1000000u));
}

// The bindless layout memo: CommitBindings walked every draw's bindless image list (~20 images) to move images out of a
// non-read-only layout (profile: ~36% of CommitBindings, ~1 ms per frame). The last 8 lists walked are remembered
// with the image registry epoch and the bindless layout epoch (bumped by GetBarriers when a walked image's layout changes);
// an identical list under the same epochs is not walked again: every image in it is still registered as it was and in the
// layout the walk left it in.

// The commit packet (step 1 of option O1): CommitBindings
// writes the push constants and the descriptor contents (the buffer and image infos, back to back in the order of a
// descriptor update template of the pipeline) straight into the record ring as one packet, and the record worker makes
// the calls the commit made before: push constants, then push descriptors (now with the template) or, for a pipeline
// with more descriptors than push descriptors allow, the update of the set taken from the descriptor heap (now with
// the template; vkUpdateDescriptorSets ran on the GPU thread, ~1 us per call, ~1100 calls per frame) and its bind,
// then the bindless set. Before, the GPU thread built VkWriteDescriptorSet arrays and the hooked calls copied them and
// their infos into the ring once more.
// The device functions the packet's calls need (push descriptors with templates, set updates with templates).
static bool CommitPacketCallsLoaded() {
	static constinit Common::OnceValue<bool> once;
	return once.Get([] {
		return VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPushDescriptorSetWithTemplateKHR != nullptr &&
		       VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorUpdateTemplate != nullptr &&
		       VULKAN_HPP_DEFAULT_DISPATCHER.vkUpdateDescriptorSetWithTemplate != nullptr;
	});
}


// The record packets (recordPackets.h): the commit is the packet above; the set of a pipeline without push descriptors
// comes from the recording thread's heap, taken there (the GPU thread no longer allocates or updates descriptor sets).
static bool RecordPacketsCommit() {
	if (!RecordPackets::On()) return false;
	static constinit Common::OnceValue<bool> loaded_once;
	return loaded_once.Get([] {
		const bool ok = CommitPacketCallsLoaded();
		if (!ok) std::fprintf(stderr, "Record packets: no push descriptors with update templates, commits as before\n");
		return ok;
	});
}

namespace {

constexpr uint32_t CommitPacketStride = 24; // One VkDescriptorBufferInfo or VkDescriptorImageInfo.
static_assert(sizeof(vk::DescriptorBufferInfo) == CommitPacketStride &&
              sizeof(vk::DescriptorImageInfo) == CommitPacketStride);

// The packet: this header, the push constants (push_bytes), then the descriptor infos (descriptor_bytes).
struct CommitPacketHeader {
	vk::CommandBuffer            buffer;
	vk::PipelineLayout           layout;
	vk::DescriptorUpdateTemplate push_template;
	vk::DescriptorSet            bindless_set; // Null: the pipeline has no bindless set.
	vk::Device                   device;
	vk::DescriptorSet            set;          // Null: push descriptors, or heap_layout's set; else the set to update and bind.
	// The record packets: with no set, a set of this layout from the recording thread's heap for the command
	// buffer of heap_tick (RecordPackets::WorkerHeapCommit); null: push descriptors.
	vk::DescriptorSetLayout      heap_layout;
	RenderContext*               heap_context = nullptr;
	uint64_t                     heap_tick    = 0;
	vk::PipelineBindPoint        bind_point       = vk::PipelineBindPoint::eGraphics;
	vk::ShaderStageFlags         push_stages;
	uint32_t                     push_bytes       = 0;
	uint32_t                     descriptor_bytes = 0;
};
static_assert(std::is_trivially_copyable_v<CommitPacketHeader> && sizeof(CommitPacketHeader) % 8u == 0u);

// On the record worker (or on the GPU thread when its commands go straight to the driver): the calls of the old
// commit, in the same order.
void RunCommitPacket(const void* blob) {
	CommitPacketHeader header;
	std::memcpy(&header, blob, sizeof(header));
	const auto* data = static_cast<const uint8_t*>(blob) + sizeof(header);
	if (header.push_bytes != 0) {
		header.buffer.pushConstants(header.layout, header.push_stages, 0, header.push_bytes, data);
	}
	if (header.descriptor_bytes != 0) {
		// The C entry points: Vulkan-Hpp's templated overloads would take a pointer argument by reference and pass the
		// address of the pointer.
		const auto* descriptors = static_cast<const void*>(data + header.push_bytes);
		if (!header.set && header.heap_layout) { // the record packets: the recording thread's set
			EXIT_IF(header.heap_context == nullptr);
			header.set = RecordPackets::WorkerHeapCommit(*header.heap_context, header.heap_layout, header.heap_tick);
		}
		if (header.set) {
			VULKAN_HPP_DEFAULT_DISPATCHER.vkUpdateDescriptorSetWithTemplate(
			    static_cast<VkDevice>(header.device), static_cast<VkDescriptorSet>(header.set),
			    static_cast<VkDescriptorUpdateTemplate>(header.push_template), descriptors);
			header.buffer.bindDescriptorSets(header.bind_point, header.layout, 0, 1, &header.set, 0, nullptr);
		} else {
			VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPushDescriptorSetWithTemplateKHR(
			    static_cast<VkCommandBuffer>(header.buffer), static_cast<VkDescriptorUpdateTemplate>(header.push_template),
			    static_cast<VkPipelineLayout>(header.layout), 0, descriptors);
		}
	}
	if (header.bindless_set) {
		header.buffer.bindDescriptorSets(header.bind_point, header.layout, 1, 1, &header.bindless_set, 0, nullptr);
	}
}

// The template entries of the stages' descriptor bindings, in commit order (the order of the old writes).
void BuildCommitPacketEntries(std::span<PreparedBindings* const>              stages,
                              std::vector<vk::DescriptorUpdateTemplateEntry>& entries) {
	entries.clear();
	uint32_t offset = 0;
	for (const auto* prepared: stages) {
		const auto& program = *prepared->runtime->program;
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind  = static_cast<BindingKind>(index);
			const auto count = program.bindings.descriptor_counts[index];
			entries.push_back({ShaderRecompiler::IR::NativeBinding(program.stage, kind), 0, count,
			                   NativeDescriptorType(kind), offset, CommitPacketStride});
			offset += count * CommitPacketStride;
		}
	}
}

// Writes the descriptor infos of `stages` at `out` in template order and returns their bytes. The same values and
// checks as the writes CommitBindings builds otherwise; image slots no image resource fills are zero.
uint32_t FillCommitPacket(RenderContext& context, uint8_t* out, std::span<PreparedBindings* const> stages) {
	uint32_t   slot       = 0;
	const auto put_buffer = [&](uint32_t at, const vk::DescriptorBufferInfo& info) {
		std::memcpy(out + static_cast<size_t>(at) * CommitPacketStride, &info, CommitPacketStride);
	};
	const auto put_image = [&](uint32_t at, const vk::DescriptorImageInfo& info) {
		std::memcpy(out + static_cast<size_t>(at) * CommitPacketStride, &info, CommitPacketStride);
	};
	for (const auto* prepared: stages) {
		const auto& program     = *prepared->runtime->program;
		const auto& descriptors = *prepared;
		std::array<uint32_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind  = static_cast<BindingKind>(index);
			const auto count = program.bindings.descriptor_counts[index];
			const auto start = slot;
			slot += count;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) != ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = start;
				std::memset(out + static_cast<size_t>(start) * CommitPacketStride, 0,
				            static_cast<size_t>(count) * CommitPacketStride);
				continue;
			}
			switch (kind) {
				case BindingKind::Buffers: {
					EXIT_IF(descriptors.buffers.size() != count);
					uint32_t at = start;
					for (const auto& view: descriptors.buffers) {
						EXIT_IF(view.buffer == nullptr);
						put_buffer(at++, view);
					}
					break;
				}
				case BindingKind::BdaPagetable:
				case BindingKind::FaultBuffer: {
					auto&       cache      = context.GetBufferCache();
					const auto* bda_buffer = kind == BindingKind::BdaPagetable ? cache.GetBdaPageTableBuffer()
					                                                           : cache.GetFaultBuffer();
					put_buffer(start, {bda_buffer->Handle(), 0, bda_buffer->Size()});
					break;
				}
				case BindingKind::FlattenedSrt:
				case BindingKind::FlattenedSrtUniform:
				case BindingKind::ShaderData:
				case BindingKind::SharedMemory:
				case BindingKind::Gds: {
					const vk::DescriptorBufferInfo* view = &descriptors.gds;
					if (kind == BindingKind::FlattenedSrt) {
						view = &descriptors.flattened_srt;
					} else if (kind == BindingKind::FlattenedSrtUniform) {
						view = &descriptors.flattened_srt_uniform;
					} else if (kind == BindingKind::ShaderData) {
						view = &descriptors.shader_data_buffer;
					} else if (kind == BindingKind::SharedMemory) {
						view = &descriptors.shared_memory;
					}
					EXIT_IF(view->buffer == nullptr);
					put_buffer(start, *view);
					break;
				}
				case BindingKind::Samplers: {
					uint32_t at = start;
					for (const auto sampler: descriptors.samplers) {
						EXIT_IF(sampler == nullptr);
						put_image(at++, {sampler, nullptr, vk::ImageLayout::eUndefined});
					}
					break;
				}
				case BindingKind::Count: EXIT("invalid descriptor binding kind");
			}
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture  = descriptors.images[i];
			EXIT_IF(resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()));
			const auto kind  = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] + resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				put_image(start + mip, MakeImageInfo(texture, mip));
			}
		}
	}
	return slot * CommitPacketStride;
}

} // namespace

// Binds what a stage list was prepared with: barriers of the bound resources, push constants, descriptors.
//
//   Called from                        Thread       Bind point  `prepared_bindings`
//   CommitPreparedDraw (see there)     GPU thread   graphics    the draw's stages, prepared by its job or on this thread
//   DispatchDirect, DispatchIndirect   GPU thread   compute     the dispatch's one stage, prepared on this thread
//
// The calls themselves: with RecordPacketsCommit() the push constants and descriptors are one packet in the record ring,
// and RunCommitPacket makes the calls on the recording thread when it records this buffer; with no ring open it runs
// here, on the GPU thread (CommandRecorder::ReserveCall returned null).
void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	std::optional<DispatchSubTimer> census_part(std::in_place, DispatchSub::BPre); // KYTY_TRACE_DRAW_CENSUS
	auto   vk_buffer        = buffer.Handle();
	size_t image_descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	uint32_t                       push_used     = 0; // the push dwords the stages' data took
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	bool has_scalar_reads = false;
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		has_scalar_reads |= !prepared->runtime->resources->specialization_reads.empty();
		write_count += std::popcount(program.bindings.descriptor_mask);
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		const auto shader_stage = NativeShaderStage(program.stage);
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// Scalar resource reads vs. image / buffer writes of this commit: the written ranges are resolved and validated once
	// per commit, each read is tested against that short list, a pair with two compares.
	struct ScalarImageWrite {
		ImageId      id;
		const Image* image = nullptr;
		GuestRange   range;
	};
	static thread_local std::vector<ScalarImageWrite>  scalar_image_writes;
	static thread_local std::vector<ScalarWriteRange>  scalar_image_ranges;  // [begin, end) of scalar_image_writes[i]
	static thread_local std::vector<ScalarWriteRange>  scalar_buffer_ranges; // buffer writes
	static thread_local std::vector<ScalarWriteRange>  scalar_sorted_ranges; // both, by begin
	static thread_local std::vector<uint64_t>          scalar_sorted_max_end; // max end of sorted ranges [0, i]
	auto scan_scalar_reads = [&](auto&& on_image, auto&& on_buffer) {
		const auto intersects_any = [&](uint64_t begin, uint64_t end) {
            const auto first_after = std::partition_point(scalar_sorted_ranges.begin(), scalar_sorted_ranges.end(),
			                                                      [&](const ScalarWriteRange& range) { return range.begin < end; });
            const auto count = static_cast<size_t>(first_after - scalar_sorted_ranges.begin());
            return count != 0u && scalar_sorted_max_end[count - 1u] > begin;
		};
		bool listed = false;
		for (const auto* reader: prepared_bindings) {
			const auto& reads = reader->runtime->resources->specialization_reads;
			if (reads.empty()) continue;
			if (!listed) {
				listed = true;
				for (const auto* writer: prepared_bindings) {
					if (HasUnknownAddressWrites(*writer)) {
						EXIT("scalar resource reads (shader 0x%016llx stage %u) cannot be proven disjoint from shader address writes (shader 0x%016llx stage %u)\n",
				     static_cast<unsigned long long>(reader->runtime->program->shader_hash),
				     static_cast<uint32_t>(reader->runtime->program->stage),
				     static_cast<unsigned long long>(writer->runtime->program->shader_hash),
				     static_cast<uint32_t>(writer->runtime->program->stage));
					}
				}
				scalar_image_writes.clear();
				scalar_image_ranges.clear();
				scalar_buffer_ranges.clear();
				// Every overlap stops the commit, so the image side is always built.
				for (const auto id: m_bound_images) {
					const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
					if (image == nullptr ||
					    (!image->binding.shader_write && !image->binding.is_target)) continue;
					for (const auto* written: {&image->info.data, &image->info.stencil,
					                           &image->info.metadata.range}) {
						if (written->size == 0) continue;
						scalar_image_ranges.push_back(MakeScalarWriteRange(written->address, written->size));
						scalar_image_writes.push_back({id, image, *written});
					}
				}
				for (const auto* writer: prepared_bindings) {
					ForEachWrittenRange(*writer, [&](uint64_t address, uint64_t size) {
						scalar_buffer_ranges.push_back(MakeScalarWriteRange(address, size));
					});
				}
				scalar_sorted_ranges.assign(scalar_image_ranges.begin(), scalar_image_ranges.end());
				scalar_sorted_ranges.insert(scalar_sorted_ranges.end(), scalar_buffer_ranges.begin(),
				                            scalar_buffer_ranges.end());
				std::sort(scalar_sorted_ranges.begin(), scalar_sorted_ranges.end(),
				          [](const ScalarWriteRange& a, const ScalarWriteRange& b) { return a.begin < b.begin; });
				scalar_sorted_max_end.resize(scalar_sorted_ranges.size());
				uint64_t max_end = 0;
				for (size_t i = 0; i < scalar_sorted_ranges.size(); ++i) {
					max_end                  = std::max(max_end, scalar_sorted_ranges[i].end);
					scalar_sorted_max_end[i] = max_end;
				}
			}
			if (scalar_image_ranges.empty() && scalar_buffer_ranges.empty()) continue;
			for (const auto [address, size]: reads) {
				if (size == 0 || address > UINT64_MAX - size) EXIT("invalid image overlap range\n");
				const auto end = address + size;
				if (!intersects_any(address, end)) continue;
				for (size_t i = 0; i < scalar_image_ranges.size(); ++i) {
					const auto& range = scalar_image_ranges[i];
					if (address < range.end && range.begin < end) {
						const auto& written = scalar_image_writes[i];
						on_image(reader, address, size, written.id, *written.image, written.range);
					}
				}
				for (const auto& range: scalar_buffer_ranges) {
					if (address < range.end && range.begin < end) {
						on_buffer(address, size, range.begin, range.end - range.begin);
					}
				}
			}
		}
	};
	const auto report_image_overlap =[&](const PreparedBindings* reader, uint64_t address,
	                                      uint64_t size, ImageId id, const Image& image,
	                                      GuestRange written) {
		// The hit under the bytes the draw's own descriptions name.
		static thread_local std::vector<GuestRange> exact_writes;
		const bool named   = ScalarExact::ImageWrites(m_commit_targets, prepared_bindings, id, exact_writes);
		const auto verdict = ScalarExact::Classify(exact_writes, named, address, size);
		const auto reader_hash = reader->runtime->program->shader_hash;
		auto&      totals      = ScalarExact::Get();
		if (verdict == ScalarExact::Verdict::Unknown) {
			ScalarExact::Count(totals.fallback, Common::WorkaroundHits::Hit::ScalarExactFallback);
			ScalarExact::Log(2u, "fallback", reader_hash, address, size, id, &image, exact_writes);
		} else if (verdict == ScalarExact::Verdict::Inside) {
			ScalarExact::Count(totals.both, Common::WorkaroundHits::Hit::ScalarExactBoth);
			ScalarExact::Log(0u, "both", reader_hash, address, size, id, &image, exact_writes);
		} else {
			ScalarExact::Count(totals.narrowed, Common::WorkaroundHits::Hit::ScalarExactNarrowed);
			ScalarExact::Log(1u, "narrowed", reader_hash, address, size, id, &image, exact_writes);
			// A byte this draw does not write: the record-time value is the value the draw reads.
			return;
		}
		// A scalar read of bytes this draw writes: the record-time value is not the value the draw reads (0 hits in the
		// user's 34-minute session and every test place since D6 K5).
		{
			std::fprintf(stderr,
			             "SCALAR_OVERLAP frame=%u reader=%016llx read=0x%llx+0x%llx image=%u "
			             "range=0x%llx+0x%llx data=0x%llx+0x%llx extent=%ux%ux%u fmt=%d guest_fmt=%d "
			             "tile=%d levels=%u layers=%u write=%d target=%d\n",
			             DebugPresentCount().load(),
			             static_cast<unsigned long long>(reader->runtime->program->shader_hash),
			             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
			             static_cast<unsigned>(id.index), static_cast<unsigned long long>(written.address),
			             static_cast<unsigned long long>(written.size),
			             static_cast<unsigned long long>(image.info.data.address),
			             static_cast<unsigned long long>(image.info.data.size), image.info.extent.width,
			             image.info.extent.height, image.info.extent.depth,
			             static_cast<int>(image.info.pixel_format), static_cast<int>(image.info.guest_format),
			             static_cast<int>(image.info.tile_mode), image.info.resources.levels,
			             image.info.resources.layers, image.binding.shader_write ? 1 : 0,
			             image.binding.is_target ? 1 : 0);
		}
		EXIT("scalar resource reads overlap an image or attachment write\n");
	};
	const auto report_buffer_overlap = [&](uint64_t read_address, uint64_t read_size, uint64_t written_address,
	                                       uint64_t written_size) {
		// Which stages take part: every stage with its written buffers, so the writer of the overlapping range shows.
		std::fprintf(stderr,
		             "SCALAR_BUFFER_OVERLAP frame=%u read=0x%llx+0x%llx written=0x%llx+0x%llx marker=[%s]\n",
		             DebugPresentCount().load(), static_cast<unsigned long long>(read_address),
		             static_cast<unsigned long long>(read_size), static_cast<unsigned long long>(written_address),
		             static_cast<unsigned long long>(written_size), DebugMarkerPath().c_str());
		for (const auto* stage: prepared_bindings) {
			const auto& program = *stage->runtime->program;
			std::string writes;
			for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
				const auto index = program.info.buffers[resource].descriptor_index;
				if (index == UINT32_MAX) continue;
				const auto& source = stage->buffer_sources[index];
				writes += fmt::format(" b{}{}={:x}+{:x}", index, program.info.buffers[resource].written ? "W" : "r",
				                      source.address, source.size);
				if (HasFixedBufferWrites(*stage, resource)) {
					writes += " fixed:";
					for (const auto& [begin, end]: program.info.buffers[resource].fixed_write_ranges) {
						writes += fmt::format("[{:x},{:x})", begin, end);
					}
				}
			}
			std::fprintf(stderr, "  stage %u shader=%016llx address_writes=%d buffers:%s\n",
			             static_cast<uint32_t>(program.stage), static_cast<unsigned long long>(program.shader_hash),
			             program.has_address_writes ? 1 : 0, writes.c_str());
			// Record-keyed V#s: the ranges of the stage's record-keyed V#s (resident part after the slash).
			if (!stage->indirect_ranges.empty()) {
				std::string ranges;
				for (const auto& range: stage->indirect_ranges) {
					ranges += fmt::format(" {}{:x}+{:x}/{:x}", range.written ? "W" : "r", range.address, range.size,
					                      range.resident);
				}
				std::fprintf(stderr, "  stage %u INDIRECT_RANGES complete=%d:%s\n", static_cast<uint32_t>(program.stage),
				             stage->runtime->resources->indirect_ranges_complete ? 1 : 0, ranges.c_str());
			}
		}
		std::fflush(stderr);
		EXIT("scalar resource reads overlap a shader buffer write\n");
	};
	// The record packets, with the recording thread's heap sets.
	const bool packet = RecordPacketsCommit();
	census_part.emplace(DispatchSub::BScan);
	// Nothing to scan unless a stage reads scalar resources (the scan skips the others anyway).
	if (has_scalar_reads) {
		KYTY_PROFILER_BLOCK("Commit::OverlapScan");
		scan_scalar_reads(report_image_overlap, report_buffer_overlap);
	}
		census_part.emplace(DispatchSub::BDesc);
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	m_descriptor_writes.reserve(write_count);
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

	for (auto* prepared: prepared_bindings) {
		const auto& program       = *prepared->runtime->program;
		auto&       descriptors   = *prepared;
		const auto  shader_stage  = NativeShaderStage(program.stage);
		const auto  shader_stages = ShaderPipelineStages(shader_stage);
		census_part.emplace(DispatchSub::BTransit);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		// KYTY_BINDLESS: resolved bindless textures are sampled through set 1, read-only.
		// The layout memo: a list walked before, with no image (un)registered and no walked image's layout changed since,
		// still has every image in its read-only layout: the walk would do nothing.
		bool layout_known = false;
		if (!descriptors.bindless_images.empty()) {
			const uint64_t registry = DrawRepeat::g_image_registry_epoch.load(std::memory_order_relaxed);
			const uint64_t layouts  = g_bindless_layout_epoch.load(std::memory_order_relaxed);
			for (const auto& memo: m_bindless_layout_memos) {
				if (memo.registry == registry && memo.layouts == layouts && memo.ids == descriptors.bindless_images) {
					layout_known = true;
					break;
				}
			}
		}
		if (!layout_known) {
			const LayoutWalkTimer layout_walk_timer(descriptors.bindless_images.size()); // PERF bindless_layout_walk
			for (const auto id: descriptors.bindless_images) {
				auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr || !image->registered || image->info.data.Empty()) continue;
				const auto layout = image->info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                          : vk::ImageLayout::eShaderReadOnlyOptimal;
				image->bindless_watched = true;
				if (image->backing.state.layout != layout) {
					buffer.EndRendering();
					image->Transit(layout, vk::AccessFlagBits2::eShaderRead, {}, vk_buffer);
				}
			}
			if (!descriptors.bindless_images.empty()) {
				auto& memo    = m_bindless_layout_memos[m_bindless_layout_memo_next++ % m_bindless_layout_memos.size()];
				memo.registry = DrawRepeat::g_image_registry_epoch.load(std::memory_order_relaxed);
				memo.layouts  = g_bindless_layout_epoch.load(std::memory_order_relaxed);
				memo.ids.assign(descriptors.bindless_images.begin(), descriptors.bindless_images.end());
			}
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& image   = m_context.GetTextureCache().GetImage(descriptors.images[i].image_id);
			auto& binding = descriptors.images[i];
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.ReadLayout(), vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}
		census_part.emplace(DispatchSub::BDesc);

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = packet ? decltype(program.bindings.descriptor_mask) {} : program.bindings.descriptor_mask;
		     mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != write.descriptorCount);
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
						}
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::FlattenedSrtUniform:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::FlattenedSrtUniform) {
							view = &descriptors.flattened_srt_uniform;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							EXIT_IF(sampler == nullptr);
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; !packet && i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			EXIT_IF(resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()));
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		// The stages' push dwords never overlap (each lookup places its data after or before the earlier stages',
		// PushData::StartFor; the pixel stage's data is at the end): an overlap is an error.
		const auto claim = [&](uint32_t start, uint32_t count) {
			EXIT_IF(start + count > ShaderRecompiler::IR::PushData::DwordCount);
			const uint32_t bits = count == 0u ? 0u : (count >= 32u ? ~0u : ((1u << count) - 1u)) << start;
			EXIT_IF((push_used & bits) != 0u);
			push_used |= bits;
		};
		if (program.bindings.UsesPushData()) {
			claim(program.bindings.push_data_start_dword, shader_data_dwords);
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	census_part.emplace(DispatchSub::BEmit);
	if (packet) {
		// The template is made once per pipeline, from the same stage order as the writes.
		if (write_count != 0 && pipeline.push_template == nullptr) {
			std::vector<vk::DescriptorUpdateTemplateEntry> entries;
			BuildCommitPacketEntries(prepared_bindings, entries);
			EXIT_IF(pipeline.descriptor_set_layout == nullptr || entries.empty());
			vk::DescriptorUpdateTemplateCreateInfo info {};
			info.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
			info.pDescriptorUpdateEntries   = entries.data();
			info.templateType               = pipeline.uses_push_descriptors
			                                      ? vk::DescriptorUpdateTemplateType::ePushDescriptors
			                                      : vk::DescriptorUpdateTemplateType::eDescriptorSet;
			info.descriptorSetLayout        = pipeline.descriptor_set_layout;
			info.pipelineBindPoint          = pipeline_bind_point;
			info.pipelineLayout             = pipeline.pipeline_layout;
			info.set                        = 0;
			EXIT_IF(m_context.GetGraphics().device.createDescriptorUpdateTemplate(&info, nullptr, &pipeline.push_template) !=
			        vk::Result::eSuccess);
			pipeline.push_template_bytes = entries.back().offset + entries.back().descriptorCount * CommitPacketStride;
		}
		CommitPacketHeader header;
		header.buffer           = vk_buffer;
		header.layout           = pipeline.pipeline_layout;
		header.push_template    = pipeline.push_template;
		header.bind_point       = pipeline_bind_point;
		header.push_stages      = push_stages;
		header.push_bytes       = has_push_data ? static_cast<uint32_t>(sizeof(push_data)) : 0u;
		header.descriptor_bytes = write_count != 0 ? pipeline.push_template_bytes : 0u;
		const bool heap_set     = header.descriptor_bytes != 0 && !pipeline.uses_push_descriptors;
		// The record packets: the recording thread takes the set from its own heap for this command buffer's tick
		// (its pools are touched by that thread only). Otherwise the heap set is taken here, where the old commit took it
		// (the heap belongs to the committing thread), before the ring room is reserved: a pool reset drains the ring.
		const bool worker_heap = heap_set && packet && CommandRecorder::RecordedByWorker(vk_buffer);
		if (heap_set) {
			header.device = m_context.GetGraphics().device;
			if (!worker_heap) header.set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
		}
		if (pipeline.uses_bindless) {
			header.bindless_set = m_context.GetGraphics().bindless_set;
			EXIT_IF(!header.bindless_set);
		}
		if (header.push_bytes != 0 || header.descriptor_bytes != 0 || header.bindless_set) {
			const size_t total = sizeof(header) + header.push_bytes + header.descriptor_bytes;
			static thread_local std::vector<uint8_t> direct_packet;
			auto*      out    = static_cast<uint8_t*>(CommandRecorder::ReserveCall(vk_buffer, total));
			const bool direct = out == nullptr;
			if (direct) {
				direct_packet.resize(total);
				out = direct_packet.data();
			}
			if (worker_heap && !direct) {
				header.heap_layout  = pipeline.descriptor_set_layout;
				header.heap_context = &m_context;
				header.heap_tick    = m_context.GetCommandScheduler().CurrentTick();
			} else if (worker_heap) {
				// Not routed after all (no ring room is open): this thread's heap, as without the switch.
				header.set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			}
			if (packet && !direct) RecordPackets::Note(RecordPackets::Stat::CommitPackets);
			std::memcpy(out, &header, sizeof(header));
			if (header.push_bytes != 0) std::memcpy(out + sizeof(header), push_data.dwords.data(), header.push_bytes);
			if (header.descriptor_bytes != 0) {
				const uint32_t filled =
				    FillCommitPacket(m_context, out + sizeof(header) + header.push_bytes, prepared_bindings);
				EXIT_IF(filled != header.descriptor_bytes);
			}
			if (direct) {
				RunCommitPacket(out);
			} else {
				CommandRecorder::CommitCall(RunCommitPacket);
			}
		}
	} else {
	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			const auto set = m_context.GetDescriptorHeap().Commit(pipeline.descriptor_set_layout);
			for (auto& write: m_descriptor_writes) {
				write.dstSet = set;
			}
			m_context.GetGraphics().device.updateDescriptorSets(
			    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(), 0,
			    nullptr);
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
	if (pipeline.uses_bindless) {
		const auto set = m_context.GetGraphics().bindless_set;
		EXIT_IF(set == nullptr);
		vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 1, 1, &set, 0, nullptr);
	}
	}
	census_part.reset();
	// The bindless heaps prepared for this draw are read by the commands recorded at this tick.
	{
		// The worker may be preparing the next draw's bindless heaps meanwhile.
		auto&                        bindless_table = m_context.GetBindlessTable();
		std::unique_lock<std::mutex> binder_lock(bindless_table.BinderMutex(), std::defer_lock);
		const auto                   tick = m_context.GetCommandScheduler().CurrentTick();
		binder_lock.lock();
		bindless_table.MarkDrawHeapsCommitted(tick, m_commit_bindless_heaps, m_commit_bindless_tables, m_commit_bindless_plan,
		                                      true);
	}
}

} // namespace Libs::Graphics
