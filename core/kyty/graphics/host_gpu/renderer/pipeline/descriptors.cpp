#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hackNotice.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/execHelpers.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/lodStats.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/occlusionQueries.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

constexpr size_t MaxTableElements   = size_t {1} << 16u;
constexpr size_t TableElementBudget = LegacyTableResolutions * MaxTableElements;

constexpr auto TableOversized = [](const auto& table) { return table.elements.size() > MaxTableElements; };

}

bool TableDescriptorWrites(vk::DriverId driver) {
	enum class Forced { None, Writes, Copies };
	static const Forced forced = [] {
		const char* value = std::getenv("KYTY_TABLE_DESCRIPTOR_WRITES");
		if (value == nullptr || value[0] == '\0') {
			return Forced::None;
		}
		if (std::string_view(value) == "1") {
			return Forced::Writes;
		}
		if (std::string_view(value) != "0") {
			EXIT("invalid KYTY_TABLE_DESCRIPTOR_WRITES: %s (0 or 1)\n", value);
		}
		return Forced::Copies;
	}();
	if (forced == Forced::None) {
		return driver == vk::DriverId::eMesaRadv;
	}
	return forced == Forced::Writes;
}

TableResolutionSettings ReadTableResolutionSettings() {
	TableResolutionSettings settings;
	if (const char* value = std::getenv("KYTY_TABLE_RESOLUTION_LEGACY"); value != nullptr && value[0] != '\0') {
		const std::string_view text(value);
		if (text != "0" && text != "1") {
			EXIT("invalid KYTY_TABLE_RESOLUTION_LEGACY: %s (0 or 1)\n", value);
		}
		settings.legacy = text == "1";
	}
	if (settings.legacy) {
		settings.capacity = LegacyTableResolutions;
		return settings;
	}
	if (const char* value = std::getenv("KYTY_TABLE_RESOLUTIONS"); value != nullptr && value[0] != '\0') {
		const std::string_view text(value);
		size_t                 capacity = 0;
		bool                   valid    = text.size() <= 4;
		for (const char digit: text) {
			valid    = valid && digit >= '0' && digit <= '9';
			capacity = capacity * 10 + static_cast<size_t>(digit - '0');
		}
		if (!valid || capacity == 0 || capacity > MaxTableResolutions) {
			EXIT("invalid KYTY_TABLE_RESOLUTIONS: %s (1 to %zu)\n", value, MaxTableResolutions);
		}
		settings.capacity = capacity;
	}
	return settings;
}

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
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory:
		case BindingKind::MipStats: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::UniformBuffers: return vk::DescriptorType::eUniformBuffer;
#if defined(ANYPS5_KYTY)
		case BindingKind::PackedTargets: return vk::DescriptorType::eStorageImage;
#endif
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorType NativeDescriptorType(const ShaderRecompiler::IR::Bindings& bindings, BindingKind kind) {
	if (kind == BindingKind::FlattenedSrt && bindings.flattened_srt_uniform) {
		return vk::DescriptorType::eUniformBuffer;
	}
	return NativeDescriptorType(kind);
}

template <typename Fn>
void ForEachImageTable(const std::vector<ShaderRecompiler::IR::ImageResource>& images, Fn&& fn) {
	for (uint32_t i = 0; i < images.size(); i++) {
		const auto& image = images[i];
		if (image.table_capacity == 0u) {
			continue;
		}
		const auto kind  = *ShaderRecompiler::IR::DescriptorBindingForImage(image);
		bool       first = true;
		for (uint32_t j = 0; j < i && first; j++) {
			first = images[j].table_capacity == 0u ||
			        images[j].table_descriptor_base != image.table_descriptor_base ||
			        ShaderRecompiler::IR::DescriptorBindingForImage(images[j]) != kind;
		}
		if (first) {
			fn(i, kind);
		}
	}
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

void AppendImageTableGaps(const vk::WriteDescriptorSet& whole, std::span<const ImageTableFill> tables,
                          std::vector<vk::WriteDescriptorSet>& gap_writes) {
	EXIT_IF(whole.dstArrayElement != 0 || whole.pImageInfo == nullptr);
	uint32_t next = 0;
	for (const auto& table: tables) {
		EXIT_IF(table.base < next || table.base + table.capacity > whole.descriptorCount);
		if (table.base > next) {
			auto gap            = whole;
			gap.dstArrayElement = next;
			gap.descriptorCount = table.base - next;
			gap.pImageInfo      = whole.pImageInfo + next;
			gap_writes.push_back(gap);
		}
		next = table.base + table.capacity;
	}
	if (next < whole.descriptorCount) {
		auto tail            = whole;
		tail.dstArrayElement = next;
		tail.descriptorCount = whole.descriptorCount - next;
		tail.pImageInfo      = whole.pImageInfo + next;
		gap_writes.push_back(tail);
	}
}

void AppendImageTableFills(const vk::WriteDescriptorSet& whole, std::span<const ImageTableFill> tables, bool writes,
                           vk::DescriptorSet persistent, std::span<const vk::DescriptorImageInfo> slot_infos,
                           std::span<const uint8_t> slot_live, std::vector<vk::WriteDescriptorSet>& table_writes,
                           std::vector<vk::CopyDescriptorSet>& table_copies) {
	EXIT_IF(!slot_live.empty() && slot_live.size() != slot_infos.size());
	const auto copy_range = [&](uint32_t element, uint32_t count) {
		vk::CopyDescriptorSet copy {};
		copy.srcSet          = persistent;
		copy.srcBinding      = whole.dstBinding;
		copy.srcArrayElement = element;
		copy.dstBinding      = whole.dstBinding;
		copy.dstArrayElement = element;
		copy.descriptorCount = count;
		table_copies.push_back(copy);
	};
	const auto live = [&](uint32_t slot) { return slot_live.empty() || slot_live[slot] != 0; };
	for (const auto& table: tables) {
		EXIT_IF(table.base + table.capacity > whole.descriptorCount);
		if (!writes) {
			copy_range(table.base, table.capacity);
			continue;
		}
		EXIT_IF(size_t {table.first_slot} + table.capacity > slot_infos.size());
		for (uint32_t begin = 0; begin < table.capacity;) {
			const bool run_live = live(table.first_slot + begin);
			uint32_t   end      = begin + 1;
			while (end < table.capacity && live(table.first_slot + end) == run_live) {
				end++;
			}
			if (run_live) {
				auto write            = whole;
				write.dstArrayElement = table.base + begin;
				write.descriptorCount = end - begin;
				write.pImageInfo      = slot_infos.data() + table.first_slot + begin;
				table_writes.push_back(write);
			} else {
				copy_range(table.base + begin, end - begin);
			}
			begin = end;
		}
	}
}

void AppendImageTableSeed(uint32_t binding, vk::DescriptorType type, const ImageTableFill& table, bool writes,
                          vk::DescriptorSet set, vk::DescriptorSet persistent,
                          std::span<const vk::DescriptorImageInfo> slot_infos,
                          std::vector<vk::WriteDescriptorSet>&     seed_writes,
                          std::vector<vk::CopyDescriptorSet>&      seed_copies) {
	if (writes) {
		EXIT_IF(size_t {table.first_slot} + table.capacity > slot_infos.size());
		vk::WriteDescriptorSet seed {};
		seed.dstSet          = persistent;
		seed.dstBinding      = binding;
		seed.dstArrayElement = table.base;
		seed.descriptorCount = table.capacity;
		seed.descriptorType  = type;
		seed.pImageInfo      = slot_infos.data() + table.first_slot;
		seed_writes.push_back(seed);
		return;
	}
	vk::CopyDescriptorSet copy {};
	copy.srcSet          = set;
	copy.srcBinding      = binding;
	copy.srcArrayElement = table.base;
	copy.dstSet          = persistent;
	copy.dstBinding      = binding;
	copy.dstArrayElement = table.base;
	copy.descriptorCount = table.capacity;
	seed_copies.push_back(copy);
}

void PatchTableSetUpdate(vk::DescriptorSet set, std::vector<vk::WriteDescriptorSet>& writes,
                         std::span<const vk::WriteDescriptorSet> table_writes,
                         std::span<vk::CopyDescriptorSet>         copies,
                         std::span<const vk::WriteDescriptorSet> slot_set_writes) {
	writes.insert(writes.end(), table_writes.begin(), table_writes.end());
	for (auto& write: writes) {
		write.dstSet = set;
	}
	for (auto& copy: copies) {
		copy.dstSet = set;
	}
	writes.insert(writes.end(), slot_set_writes.begin(), slot_set_writes.end());
}

bool MarkLiveTableSlots(std::span<const uint32_t> slots, std::span<const vk::DescriptorImageInfo> slot_infos,
                        std::span<const vk::DescriptorImageInfo> infos, std::span<const uint8_t> element_live,
                        std::vector<uint8_t>& slot_live) {
	EXIT_IF(slots.size() != slot_infos.size() || element_live.size() != infos.size());
	slot_live.resize(slots.size());
	bool all_live = true;
	for (size_t slot = 0; slot < slots.size(); slot++) {
		const auto element = slots[slot];
		const auto view    = slot_infos[slot].imageView;
		const bool live    = view == nullptr || (element < infos.size() && element_live[element] != 0 &&
		                                      view == infos[element].imageView);
		slot_live[slot] = live ? 1 : 0;
		all_live        = all_live && live;
	}
	return all_live;
}

static bool DepthTableElement(TextureCache& texture_cache, const TextureBinding& texture) {
	if (!texture.image_id) return false;
	return texture_cache.GetImage(texture.image_id).info.IsDepth();
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D:
		case Prospero::ImageType::kColor1DArray:
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

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address == 0 || size == 0) {
		return {nullptr, 0, vk::WholeSize};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment =
	    resource.uniform_index == UINT32_MAX
	        ? graphics.StorageMinAlignment()
	        : std::max<vk::DeviceSize>(
	              graphics.StorageMinAlignment(),
	              graphics.GetPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment);
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	auto&      queries  = context.GetOcclusionQueries();
	const bool counters = queries.PrepareRead(address, size);
	auto [buffer, offset] = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written,
	    resource.formatted, id);
	if (counters) {
		queries.NoteReadTarget(address, size, *buffer, offset);
	}
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

static vk::DescriptorBufferInfo StreamedStorageBuffer(RenderContext& context, uint64_t offset, uint64_t size,
                                                      const ShaderRecompiler::IR::BufferResource& resource,
                                                      uint32_t&                                   buffer_offset) {
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = resource.uniform_index == UINT32_MAX
	                            ? graphics.StorageMinAlignment()
	                            : std::max<vk::DeviceSize>(
	                                  graphics.StorageMinAlignment(),
	                                  graphics.GetPhysicalDeviceProperties().limits.minUniformBufferOffsetAlignment);
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	buffer_offset             = static_cast<uint32_t>(offset - aligned_offset);
	const auto& stream        = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	return {stream.Handle(), aligned_offset, size + buffer_offset};
}

static bool IsCanonicalTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 1u << 29u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (uint32_t {descriptor.MipStatsCntEn()} << 25u) |
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
	const auto         metadata_control = descriptor.fields[6] & 0x00ffff00u;
	constexpr uint32_t pipe_aligned     = 1u << 19u;
	if (!descriptor.MetaCompress()) {
		return true;
	}
	constexpr uint32_t htile_control = pipe_aligned | 1u << 21u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

struct SampledDepthBindingSupport {
	bool resource_ok = false;
	bool encoding_ok = false;
	bool view_ok     = false;

	[[nodiscard]] bool Supported() const { return resource_ok && encoding_ok && view_ok; }
};

static SampledDepthBindingSupport
CheckSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                         const ShaderTextureResource& descriptor, const Image& image,
                         vk::Format view_format) {
	return {
	    .resource_ok = IsSupportedSampledDepthResource(resource),
	    .encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                   IsSupportedDepthTextureEncoding(descriptor, resource.r128),
	    .view_ok =
	        IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW()),
	};
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const auto [resource_ok, encoding_ok, view_ok] =
	    CheckSampledDepthBinding(resource, descriptor, image, view_format);
	if (resource_ok && encoding_ok && view_ok) {
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
	       descriptor.MinLod() <= descriptor.BaseLevel() * 256u && supported_swizzle &&
	       descriptor.BCSwizzle() == 0 && !descriptor.MsaaDepth();
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format        = descriptor.Format();
	const bool resource_ok   = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok   = IsCanonicalTextureEncoding(descriptor, resource.r128);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
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
	EXIT("unsupported storage texture: shader=0x%016" PRIx64 " resource=%d descriptor=%d encoding=%d "
	     "format=%d class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     Common::HackNotice::CurrentShaderSlot(), resource_ok, descriptor_ok,
	     encoding_ok, format_ok, static_cast<uint32_t>(resource.resource_class),
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

static vk::Format SampledViewFormat(const GraphicContext& graphics, Prospero::BufferFormat format,
                                    vk::Format pixel_format, bool skip_degamma) {
	if (skip_degamma) return SrgbUnormViewFormat(pixel_format);
	switch (format) {
		case Prospero::BufferFormat::k8Srgb:
			return graphics.r8_srgb_views ? vk::Format::eR8Srgb : pixel_format;
		case Prospero::BufferFormat::k8_8Srgb:
			return graphics.r8g8_srgb_views ? vk::Format::eR8G8Srgb : pixel_format;
		default: return pixel_format;
	}
}

static vk::Format StorageViewFormat(const ShaderRecompiler::IR::ImageResource& resource,
                                    Prospero::BufferFormat format, vk::Format pixel_format) {
	if (resource.atomic64) return vk::Format::eR64Uint;
	if (resource.atomic) return vk::Format::eR32Uint;
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	if (raw_sint_format != Prospero::BufferFormat::kInvalid) return VulkanFormat(raw_sint_format);
	return SrgbUnormViewFormat(pixel_format);
}

static bool TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, vk::Format format,
                            const SurfaceFormatInfo& surface_format, bool storage,
                            uint32_t view_levels, uint32_t image_layers, ImageViewInfo& view,
                            std::string& unsupported) {
	view             = {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		unsupported =
		    fmt::format("texture minimum LOD exceeds last mip level: min_lod={} last_level={}",
		                descriptor.MinLod(), descriptor.LastLevel());
		return false;
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || resource.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	const auto base_layer = descriptor.Type() == Prospero::ImageType::kColor1D ||
	                                descriptor.Type() == Prospero::ImageType::kColor2D
	                            ? 0u
	                            : descriptor.BaseArray5();
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				unsupported = "texture array base layer is out of bounds";
				return false;
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = base_layer;
			if (view.base_layer >= image_layers) {
				unsupported = "texture base layer is out of bounds";
				return false;
			}
			view.layer_count = 1;
			break;
		default: unsupported = "unsupported texture view dimension"; return false;
	}
	return true;
}

static bool MatchTextureMipView(ImageInfo& info, ImageInfo physical, uint32_t physical_levels,
                                bool metadata, uint32_t view_levels, uint32_t& base_level);

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	if (base_level >= physical_levels) {
		ImageInfo  physical = info;
		const auto mip      = info.MipExtent(base_level);
		physical.extent     = {mip.width, mip.height, info.IsVolume() ? mip.depth : 1u};
		if (MatchTextureMipView(info, std::move(physical), view_levels, metadata, view_levels,
		                        base_level)) {
			return true;
		}
	}
	return MatchTextureMipView(info, info, physical_levels, metadata, view_levels, base_level);
}

static bool MatchTextureMipView(ImageInfo& info, ImageInfo physical, uint32_t physical_levels,
                                bool metadata, uint32_t view_levels, uint32_t& base_level) {
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

static TextureDescDerivation DeriveTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               const ShaderTextureResource& descriptor) {
	const auto unsupported = [](std::string reason) {
		TextureDescDerivation derived;
		derived.unsupported = std::move(reason);
		return derived;
	};
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
		return unsupported(fmt::format(
		    "unsupported texture mip view: base={} last={} levels={} max={} type={} tile={} "
		    "class={} numeric={} dimension={} mip_mode={} read={} written={} "
		    "dwords={:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}",
		    base_level, last_level, levels, descriptor.MaxMip(),
		    static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		    static_cast<uint32_t>(resource.resource_class),
		    static_cast<uint32_t>(resource.numeric_class),
		    static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		    resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		    descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		    descriptor.fields[6], descriptor.fields[7]));
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = ShaderRecompiler::IR::ReadsDepthBits(resource)
	                                ? Prospero::BufferFormat::k16UNorm
	                                : descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		return unsupported(fmt::format("sampled image numeric class mismatch: numeric={} format={} "
		                               "addr={:#018x}",
		                               static_cast<uint32_t>(resource.numeric_class),
		                               static_cast<uint32_t>(format), address));
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	auto          image_layers = layered ? depth : 1u;
	if (descriptor.Type() == Prospero::ImageType::kCube) {
		image_layers = std::max(image_layers, 6u);
	}
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto view_format =
	    storage ? StorageViewFormat(resource, format, pixel_format) : pixel_format;
	const auto block_bytes   = Prospero::BlockCompressedBytesPerBlock(format);
	const bool meta_compress = block_bytes == 0 && !resource.r128 && descriptor.MetaCompress();
	TextureCache::ImageDesc desc {};
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
	if (meta_compress && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (!desc.info.HasTiledLayout()) {
		auto derived = unsupported(fmt::format(
		    "unsupported tiled texture layout: format={} tile={} type={} extent={}x{}x{} "
		    "levels={} layers={} addr={:#018x} "
		    "dwords={:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}",
		    static_cast<uint32_t>(format), static_cast<uint32_t>(tile),
		    static_cast<uint32_t>(descriptor.Type()), width, height, depth, levels, image_layers,
		    address, descriptor.fields[0], descriptor.fields[1], descriptor.fields[2],
		    descriptor.fields[3], descriptor.fields[4], descriptor.fields[5], descriptor.fields[6],
		    descriptor.fields[7]));
		derived.unknown_tiling = true;
		derived.desc.info      = std::move(desc.info);
		return derived;
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           meta_compress, view_levels,
		                           view_base)) {
			if (base_level >= physical_levels ||
			    resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
				return unsupported(fmt::format(
				    "unsupported texture mip view changes physical layout: base={} last={} max={} "
				    "extent={}x{}x{} tile={}",
				    base_level, last_level, max_mip, width, height, depth,
				    static_cast<uint32_t>(tile)));
			}
			view_levels                = physical_levels - base_level;
			desc.info.resources.levels = physical_levels;
			desc.info.data.size        = 0;
			desc.info.UpdateSize();
		}
	} else {
		desc.info.UpdateSize();
	}
	const auto alignment = desc.info.DataAlignment();
	if (desc.info.data.size == 0 || alignment == 0 ||
	    (address & (static_cast<uint64_t>(alignment) - 1u)) != 0) {
		return unsupported(fmt::format(
		    "unsupported texture data range: addr={:#018x} size={:#x} alignment={:#x} "
		    "dwords={:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x},{:08x}",
		    address, desc.info.data.size, alignment, descriptor.fields[0], descriptor.fields[1],
		    descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		    descriptor.fields[6], descriptor.fields[7]));
	}
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (meta_compress && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	std::string view_unsupported;
	if (!TextureViewInfo(resource, descriptor, view_format, surface_format, storage, view_levels,
	                     desc.info.resources.layers, desc.view_info, view_unsupported)) {
		return unsupported(std::move(view_unsupported));
	}
	desc.view_info.base_level = view_base;
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	desc.heap = resource.heap;
	const auto data_size = desc.info.data.size;
	return {std::move(desc), shader_conversion, pixel_format, view_format, data_size};
}

size_t RenderExecutor::TextureDescKeyHash::operator()(const TextureDescKey& key) const noexcept {
	std::array<uint64_t, 7> lanes {};
	static_assert(sizeof(key.dwords) + sizeof(key.view) <= sizeof(lanes));
	std::memcpy(lanes.data(), key.dwords.data(), sizeof(key.dwords));
	std::memcpy(reinterpret_cast<uint8_t*>(lanes.data()) + sizeof(key.dwords), key.view.data(),
	            sizeof(key.view));
	constexpr std::array<uint64_t, 7> Multipliers {
	    0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0xd6e8feb86659fd93ull,
	    0xff51afd7ed558ccdull, 0xc4ceb9fe1a85ec53ull, 0x94d049bb133111ebull};
	uint64_t hash = 0;
	for (size_t i = 0; i < lanes.size(); i++) {
		const auto mixed = (lanes[i] ^ (lanes[i] >> 31u)) * Multipliers[i];
		hash ^= (mixed << (i * 9u % 64u)) | (mixed >> ((64u - i * 9u % 64u) % 64u));
	}
	hash ^= hash >> 32u;
	hash *= 0x9e3779b97f4a7c15ull;
	return static_cast<size_t>(hash ^ (hash >> 29u));
}

RenderExecutor::TextureDescKey
RenderExecutor::MakeTextureDescKey(const ShaderRecompiler::IR::ImageResource&   resource,
                                   const ShaderRecompiler::IR::DescriptorValue& value) {
	TextureDescKey key {};
	key.dwords  = value.dwords;
	key.view[0] = static_cast<uint32_t>(resource.resource_class) |
	              (static_cast<uint32_t>(resource.numeric_class) << 8u) |
	              (static_cast<uint32_t>(resource.dimension) << 16u) |
	              (static_cast<uint32_t>(resource.mip_mode) << 24u);
	key.view[1] = resource.mip_count;
	key.view[2] = static_cast<uint32_t>(resource.conversion_format);
	key.view[3] = resource.shader_swizzle;
	key.view[4] =
	    static_cast<uint32_t>(resource.read) | (static_cast<uint32_t>(resource.written) << 1u) |
	    (static_cast<uint32_t>(resource.atomic) << 2u) |
	    (static_cast<uint32_t>(resource.depth_compare) << 3u) |
	    (static_cast<uint32_t>(resource.cube) << 4u) | (static_cast<uint32_t>(resource.r128) << 5u) |
	    (static_cast<uint32_t>(resource.heap) << 6u);
	return key;
}

bool RenderExecutor::ReuseTextureLookup(TextureDescDerivation& derived, TextureBinding& binding) {
	auto&       texture_cache = m_context.GetTextureCache();
	auto&       found         = derived.found;
	auto*       image         = texture_cache.m_slot_images.try_get(found.id);
	const auto& range         = found.desc.info.data;
	if (image == nullptr || !image->registered || image->binding.needs_rebind ||
	    ImageLookupState::Of(found.id, *image) != found.state) {
		found.valid = false;
		return false;
	}
	if (const auto set_epoch = texture_cache.ImageSetEpoch(); found.image_set_epoch != set_epoch) {
		if (texture_cache.ImageEpochInRegion(range.address, range.size) > found.image_set_epoch) {
			found.valid = false;
			return false;
		}
		found.image_set_epoch = set_epoch;
	}
	image->tick_accessed_last = texture_cache.m_gc_tick;
	texture_cache.TouchImage(*image);
	binding.image_id   = found.id;
	binding.image_view = found.view;
	binding.desc       = found.desc;
	binding.layout     = vk::ImageLayout::eUndefined;
	binding.mip_views.clear();
	binding.lookup = &derived;
	binding.reused = true;
	return true;
}

static void ReportDeclinedTiling(const ShaderRecompiler::IR::ImageResource&   resource,
                                 const ShaderRecompiler::IR::DescriptorValue& value,
                                 bool table_candidate, const ImageInfo& info) {
	const bool indirect =
	    resource.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage;
	if (!table_candidate && !resource.heap && !indirect) {
		return;
	}
	static std::mutex                           mutex;
	static std::vector<std::array<uint32_t, 8>> seen;
	{
		std::scoped_lock lock {mutex};
		if (seen.size() >= 64u || std::ranges::find(seen, value.dwords) != seen.end()) {
			return;
		}
		seen.push_back(value.dwords);
	}
	std::fprintf(stderr,
	             "[kyty-tiled] declined %s candidate: shader 0x%016llx, format %u, tile mode %u, "
	             "%ux%ux%u, %u levels, words %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	             table_candidate ? "table" : resource.heap ? "image heap" : "indirect image",
	             static_cast<unsigned long long>(Common::HackNotice::CurrentShaderSlot()),
	             static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
	             info.extent.width, info.extent.height, info.extent.depth, info.resources.levels,
	             value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3],
	             value.dwords[4], value.dwords[5], value.dwords[6], value.dwords[7]);
	if (ImageLayoutDiagEnabled()) {
		std::fprintf(stderr,
		             "[kyty-tiled-diag] declined candidate: source %u, descriptor index %u, first "
		             "use pc 0x%x, indirect root %u, descriptor at 0x%llx, base 0x%llx\n",
		             resource.source, resource.descriptor_index, resource.first_use_pc,
		             resource.indirect_root,
		             static_cast<unsigned long long>(CurrentImageLayoutOrigin().address),
		             static_cast<unsigned long long>(info.data.address));
	}
}

static TextureBinding DeclineTableView(TextureCache::ImageDesc& desc, vk::Format image_format,
                                       vk::Format view_format) {
	Common::HackNotice::Report(
	    "bindless table texture bound as null image",
	    "RenderExecutor::ResolveTexture (incompatible image view)", 0, 0,
	    fmt::format("texture 0x{:016x} image format {} view format {}", desc.info.data.address,
	                static_cast<int>(image_format), static_cast<int>(view_format)));
	return {ImageId {}, nullptr, std::move(desc)};
}

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              bool table_candidate, GuestRange* examined) {
	if (resource.atomic64 && !m_context.GetGraphics().shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto       descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage    = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	auto& texture_cache = m_context.GetTextureCache();
	if (descriptor.IsNull()) {
		return {.layout = vk::ImageLayout::eGeneral};
	}

	const auto key    = MakeTextureDescKey(resource, value);
	auto       cached = m_texture_descs.find(key);
	if (cached == m_texture_descs.end()) {
		auto& origin = CurrentImageLayoutOrigin();
		if (ImageLayoutDiagEnabled()) {
			origin.path = table_candidate ? "table candidate"
			              : resource.heap ? "image heap"
			              : resource.indirect_root !=
			                        ShaderRecompiler::IR::ImageResource::NoIndirectImage
			                    ? "indirect image"
			                    : "direct";
			origin.shader           = Common::HackNotice::CurrentShaderSlot();
			origin.source           = resource.source;
			origin.descriptor_index = resource.descriptor_index;
			origin.first_use_pc     = resource.first_use_pc;
			origin.indirect_root    = resource.indirect_root;
			origin.words            = value.dwords;
		}
		auto derived = DeriveTextureDesc(resource, descriptor);
		if (derived.unknown_tiling) {
			ReportDeclinedTiling(resource, value, table_candidate, derived.desc.info);
			derived.desc = {};
		}
		origin.path = nullptr;
		if (m_texture_descs.size() >= MaxTextureDescs) {
			m_clear_texture_descs = true;
		}
		cached = m_texture_descs.emplace(key, std::move(derived)).first;
	}
	auto& derived = cached->second;
	const bool candidate =
	    table_candidate || resource.heap ||
	    resource.indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage;
	if (!candidate && !derived.unsupported.empty()) {
		EXIT("%s\n", derived.unsupported.c_str());
	}
	if (candidate) {
		const auto& info     = derived.desc.info;
		const char* declined = !derived.unsupported.empty() ? derived.unsupported.c_str()
		                       : !info.data.Valid()         ? "data range outside guest memory"
		                       : info.metadata.kind != ImageMetadataKind::None &&
		                               !GuestRange {info.metadata.range.address,
		                                            std::max<uint64_t>(info.metadata.range.size, 1)}
		                                    .Valid()
		                           ? "metadata range outside guest memory"
		                           : nullptr;
		if (declined != nullptr) {
			static std::atomic_bool reported {false};
			if (!reported.exchange(true, std::memory_order_relaxed)) {
				LOGF("Texture: a T# candidate the host cannot view is bound as a null image "
				     "(shader 0x%016" PRIx64 "): %s\n",
				     Common::HackNotice::CurrentShaderSlot(), declined);
			}
			return {.layout = vk::ImageLayout::eGeneral};
		}
	}
	if (resource.heap &&
	    !m_context.IsMapped(derived.desc.info.data.address, derived.desc.info.data.size)) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	const bool reusable = !table_candidate && !storage && examined == nullptr &&
	                      resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic &&
	                      derived.desc.info.metadata.kind == ImageMetadataKind::None &&
	                      !derived.desc.info.HasStencil();
	if (reusable && derived.found.valid) {
		TextureBinding binding;
		if (ReuseTextureLookup(derived, binding)) {
			return binding;
		}
	}
	auto        desc              = derived.desc;
	const bool  shader_conversion = derived.shader_conversion;
	const auto  pixel_format      = derived.pixel_format;
	const auto  view_format       = derived.view_format;

	if (examined != nullptr) {
		*examined = desc.info.data;
	}
	if (table_candidate &&
	    ((!texture_cache.IsRegionGpuModified(desc.info.data.address, desc.info.data.size) &&
	      !Libs::LibKernel::Memory::IsBackingReadable(desc.info.data.address,
	                                                  desc.info.data.size)) ||
	     (desc.info.metadata.kind != ImageMetadataKind::None &&
	      !Libs::LibKernel::Memory::IsBackingReadable(desc.info.metadata.range.address,
	                                                  desc.info.metadata.range.size)))) {
		return {.layout = vk::ImageLayout::eGeneral};
	}
	desc.candidate = candidate;
	ImageId depth_source;
	auto    id = table_candidate
	                 ? texture_cache.FindImageSpeculative(desc, shader_conversion, &depth_source)
	                 : texture_cache.FindImage(desc, shader_conversion);
	if (!id && depth_source) {
		id = texture_cache.FindDepthCopy(depth_source, desc.info);
	}
	if (!id) {
		if (table_candidate) {
			Common::HackNotice::Report("bindless table texture bound as null image",
			                           "RenderExecutor::ResolveTexture (conversion declined)", 0, 0);
			return {id, nullptr, std::move(desc)};
		}
		return {.desc = std::move(desc), .layout = vk::ImageLayout::eGeneral};
	}
	auto*      image               = &texture_cache.GetImage(id);
	const bool stencil_association = static_cast<bool>(image->depth_id);
	const ImageId live_stencil_copy =
	    stencil_association && storage ? texture_cache.FindLiveStencilStorageCopy(id, desc) : ImageId {};
	if (live_stencil_copy) {
		id = live_stencil_copy;
		ValidateStorageColorView(texture_cache.GetImage(id).info.pixel_format, view_format,
		                         descriptor.DstSelXYZW());
	} else if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		if (candidate &&
		    !CheckSampledDepthBinding(resource, descriptor, *image, pixel_format).Supported()) {
			if (table_candidate) {
				return DeclineTableView(desc, image->info.pixel_format, pixel_format);
			}
			static std::atomic_bool reported {false};
			if (!reported.exchange(true, std::memory_order_relaxed)) {
				LOGF("Texture: a T# candidate the host cannot sample as depth is bound as a null "
				     "image (shader 0x%016" PRIx64 ", texture 0x%016" PRIx64
				     ", T# words 6-7 %08x,%08x)\n",
				     Common::HackNotice::CurrentShaderSlot(), desc.info.data.address,
				     descriptor.fields[6], descriptor.fields[7]);
			}
			return {.layout = vk::ImageLayout::eGeneral};
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, view_format, descriptor.DstSelXYZW());
	} else {
		const auto format = ShaderRecompiler::IR::ReadsDepthBits(resource)
		                        ? Prospero::BufferFormat::k16UNorm
		                        : descriptor.Format();
		desc.view_info.format = SampledViewFormat(m_context.GetGraphics(), format, pixel_format,
		                                          resource.skip_degamma);
		if (table_candidate && !IsSupportedSampledColorView(image->info.pixel_format,
		                                                    desc.view_info.format,
		                                                    descriptor.DstSelXYZW())) {
			return DeclineTableView(desc, image->info.pixel_format, desc.view_info.format);
		}
		(void)SelectSampledColorView(image->info.pixel_format, desc.view_info.format,
		                             descriptor.DstSelXYZW());
	}
	TextureBinding binding {id, nullptr, std::move(desc)};
	if (reusable && !stencil_association && !image->info.HasStencil()) {
		binding.lookup = &derived;
	}
	return binding;
}

TextureBinding RenderExecutor::ResolveTableTexture(
    const ShaderRecompiler::IR::ImageResource&   root,
    const ShaderRecompiler::IR::DescriptorValue& value, GuestRange* examined) {
	if (ShaderRecompiler::IR::ImageTableSlotCompatible(root, value)) {
		auto texture = ResolveTexture(root, value, true, examined);
		if (texture.image_id) {
			return texture;
		}
	}
	return ResolveTexture(root, ShaderRecompiler::IR::DescriptorValue {.dword_count = 8u});
}

size_t RenderExecutor::TableDescriptorHash::operator()(
    const std::array<uint32_t, 8>& dwords) const noexcept {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto dword: dwords) {
		hash = (hash ^ dword) * 0x100000001b3ull;
		hash ^= hash >> 29u;
	}
	return static_cast<size_t>(hash);
}

size_t RenderExecutor::TableResolutionKeyHash::operator()(
    const ShaderRecompiler::IR::ImageResource& root) const noexcept {
	const std::array<uint32_t, 16> fields {root.descriptor_index,
	                                       static_cast<uint32_t>(root.resource_class),
	                                       static_cast<uint32_t>(root.numeric_class),
	                                       static_cast<uint32_t>(root.dimension),
	                                       static_cast<uint32_t>(root.mip_mode),
	                                       root.mip_count,
	                                       static_cast<uint32_t>(root.conversion_format),
	                                       root.shader_swizzle,
	                                       static_cast<uint32_t>(root.srgb_decode),
	                                       (root.read ? 1u : 0u) | (root.written ? 2u : 0u) |
	                                           (root.atomic ? 4u : 0u) | (root.atomic64 ? 8u : 0u) |
	                                           (root.unfiltered_access_only ? 16u : 0u) |
	                                           (root.sample_access_only ? 32u : 0u) |
	                                           (root.depth_compare ? 64u : 0u) |
	                                           (root.skip_degamma ? 128u : 0u) | (root.cube ? 256u : 0u) |
	                                           (root.r128 ? 512u : 0u) | (root.heap ? 1024u : 0u),
	                                       root.indirect_root,
	                                       root.indirect_mapping_offset,
	                                       root.indirect_search_iterations,
	                                       static_cast<uint32_t>(root.indirect_resources.size()),
	                                       root.table_descriptor_base,
	                                       root.source};
	uint64_t hash = 0xcbf29ce484222325ull;
	const auto mix = [&hash](uint32_t value) {
		hash = (hash ^ value) * 0x100000001b3ull;
		hash ^= hash >> 29u;
	};
	for (const auto field: fields) {
		mix(field);
	}
	for (const auto resource: root.indirect_resources) {
		mix(resource);
	}
	return static_cast<size_t>(hash);
}

void RenderExecutor::ConfigureTableResolutions() {
	const auto settings = ReadTableResolutionSettings();
	m_table_resolutions.Configure(settings.capacity, settings.legacy);
}

void RenderExecutor::RetireTableResolution(TableResolution& table) {
	m_table_element_count -= std::min(m_table_element_count, table.elements.size());
	if (!table.elements.empty()) {
		m_retired_table_elements.push_back(std::move(table.elements));
	}
	table.elements.clear();
	table.slots.clear();
}

void RenderExecutor::TrimTableResolutions() {
	const auto weight = [](const TableResolution& table) { return table.elements.size(); };
	const auto retire = [this](TableResolution& table) { RetireTableResolution(table); };
	while (m_table_element_count > TableElementBudget / 4 * 3 &&
	       m_table_resolutions.ClearLeastRecent(weight, retire)) {
	}
}

void RenderExecutor::LogTableSummary() {
	auto&      stats = m_table_stats;
	const auto now   = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now().time_since_epoch())
                                                 .count());
	if (stats.window_ns == 0) {
		stats.window_ns = now;
		return;
	}
	if (now - stats.window_ns < 10'000'000'000ull) {
		return;
	}
	const auto seconds = static_cast<double>(now - stats.window_ns) / 1e9;
	const auto cache   = m_table_resolutions.TakeCounters();
	const auto taken   = stats;
	stats              = {.resets = taken.resets, .window_ns = now, .idle_logged = taken.idle_logged};
	if (taken.bindings == 0 || (taken.prepared == 0 && taken.idle_logged)) {
		return;
	}
	stats.idle_logged = taken.prepared == 0;
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-tables] summary of the last {:.0f} s: {} of {} binding preparations had image tables; {} reused "
	    "their table state, {} rebuilt it ({} without a finished state, {} after a table entry it used was "
	    "renewed, {} with other tables); {} finished states left invalid ({} for a renewed table entry, {} for "
	    "a size mismatch); table cache of {} entries, {}: {} lookups, {} hits, {} new keys, {} keys made again "
	    "after an eviction, {} evictions, {} cleared past {} elements, {} cleared for the element budget, {} "
	    "generation bumps; {} entries in use, {} keys seen, {} elements\n",
	    seconds, taken.prepared, taken.bindings, taken.reused, taken.prepared - taken.reused, taken.unfinished,
	    taken.renewed, taken.layout, taken.left_renewed + taken.left_size, taken.left_renewed, taken.left_size,
	    m_table_resolutions.Capacity(),
	    m_table_resolutions.SharedGeneration() ? "one generation for every state (KYTY_TABLE_RESOLUTION_LEGACY=1)"
	                                           : "one generation per entry",
	    cache.lookups, cache.hits, cache.created, cache.recreated, cache.evicted, cache.overflowed,
	    MaxTableElements, cache.trimmed, cache.generations, m_table_resolutions.Size(),
	    m_table_resolutions.KnownKeys(), m_table_element_count));
}

RenderExecutor::TableResolution&
RenderExecutor::FindTableResolution(const ShaderRecompiler::IR::ImageResource& root) {
	auto view                 = root;
	view.source               = 0;
	view.first_use_pc         = 0;
	view.table                = ShaderRecompiler::IR::ImageResource::NoImageTable;
	view.table_capacity       = 0;
	view.table_mapping_offset = 0;
	view.table_entry_mask     = 0;
	return m_table_resolutions.Find(
	    view, [this](TableResolution& table) { RetireTableResolution(table); }, TableOversized);
}

bool RenderExecutor::TableElementBackingReadable(const TableElementResolution& element) {
	const auto& data = element.range;
	const bool  data_readable =
	    m_context.GetTextureCache().IsRegionGpuModified(data.address, data.size) ||
	    Libs::LibKernel::Memory::IsBackingReadable(data.address, data.size);
	return data_readable &&
	       (element.metadata_range.Empty() ||
	        Libs::LibKernel::Memory::IsBackingReadable(element.metadata_range.address,
	                                                   element.metadata_range.size));
}

RenderExecutor::TableElementResolution&
RenderExecutor::ResolveTableElement(TableResolution&                             table,
                                    const ShaderRecompiler::IR::DescriptorValue& value,
                                    uint64_t stamp, TableElementResolution* known) {
	auto&      texture_cache = m_context.GetTextureCache();
	const auto backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
	bool       inserted      = false;
	if (known == nullptr) {
		const auto [entry, emplaced] = table.elements.try_emplace(value.dwords);
		known                        = &entry->second;
		inserted                     = emplaced;
		m_table_element_count += emplaced ? 1 : 0;
	}
	auto& element = *known;
	if (inserted) {
		if (m_free_table_element_versions.empty()) {
			element.version_slot = static_cast<uint32_t>(m_table_element_versions.size());
			m_table_element_versions.push_back(0);
		} else {
			element.version_slot = m_free_table_element_versions.back();
			m_free_table_element_versions.pop_back();
		}
	}
	if (!inserted && element.prepared_stamp == stamp) {
		return element;
	}
	const auto set_epoch = texture_cache.ImageSetEpoch();
	if (!inserted && !element.volatile_dcc) {
		if (element.permanent ||
		    (element.checked_epoch == set_epoch && element.backing_epoch == backing_epoch)) {
			return element;
		}
		if (element.backing_epoch != backing_epoch && element.backing_checkable &&
		    TableElementBackingReadable(element) == element.backing_readable) {
			element.backing_epoch = backing_epoch;
		}
		const auto image_epoch =
		    element.range.Valid()
		        ? texture_cache.ImageEpochInRegion(element.range.address, element.range.size)
		        : 0;
		if (image_epoch <= element.image_epoch && element.backing_epoch == backing_epoch) {
			element.checked_epoch = set_epoch;
			return element;
		}
	}
	const auto prepared_stamp = element.prepared_stamp;
	const auto prepared_index = element.prepared_index;
	const auto version_slot   = element.version_slot;
	element                   = {};
	element.prepared_stamp    = prepared_stamp;
	element.prepared_index    = prepared_index;
	element.version_slot      = version_slot;
	TableElementChanged(element);
	element.image_epoch       = set_epoch;
	element.checked_epoch = set_epoch;
	element.backing_epoch = backing_epoch;
	GuestRange examined {};
	element.texture  = ResolveTableTexture(table.root, value, &examined);
	const auto& data = element.texture.desc.info.data;
	if (!data.Empty()) {
		element.range        = data;
		element.volatile_dcc = element.texture.desc.info.metadata.kind == ImageMetadataKind::Dcc;
		if (element.texture.desc.info.metadata.kind != ImageMetadataKind::None) {
			element.metadata_range = element.texture.desc.info.metadata.range;
		}
		element.backing_checkable = data.Valid();
		element.backing_readable  = element.backing_checkable && TableElementBackingReadable(element);
	} else if (!ShaderRecompiler::IR::ImageTableSlotCompatible(table.root, value)) {
		element.permanent = true;
	} else {
		element.range = examined;
	}
	return element;
}

vk::Sampler RenderExecutor::NativeSampler(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                          uint32_t                                        index,
                                          const ShaderRecompiler::IR::DescriptorValue&    value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler    = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	uint64_t hash = sampler.integer_border ? 0x9e3779b97f4a7c15ull : 0;
	for (const auto field: descriptor.fields) {
		hash = (hash ^ field) * 0x100000001b3ull;
	}
	auto& entry = m_sampler_memo[(hash ^ (hash >> 32u)) % m_sampler_memo.size()];
	if (entry.sampler != nullptr && entry.integer_border == sampler.integer_border &&
	    std::memcmp(entry.fields.data(), descriptor.fields, sizeof(entry.fields)) == 0) {
		return entry.sampler;
	}
	const auto native = m_context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
	std::memcpy(entry.fields.data(), descriptor.fields, sizeof(entry.fields));
	entry.integer_border = sampler.integer_border;
	entry.sampler        = native;
	return native;
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto&      cache  = context.GetBufferCache();
	const auto offset = cache.StreamUpload(data.data(), data.size_bytes(), 256);
	auto&      buffer = cache.GetUtilityBuffer(MemoryUsage::Stream);
	return {buffer.Handle(), offset, data.size_bytes()};
}

static void CopyGpuFlatLoads(RenderContext& context, const vk::DescriptorBufferInfo& target,
                             std::span<const ShaderRecompiler::IR::GpuFlatLoad> loads) {
	auto& cache = context.GetBufferCache();
	std::vector<std::pair<vk::Buffer, vk::BufferCopy>> copies;
	copies.reserve(loads.size());
	for (const auto& load: loads) {
		const auto bytes            = uint64_t {load.count} * sizeof(uint32_t);
		const auto [buffer, offset] = cache.ObtainBuffer(load.address, bytes, false);
		copies.push_back({buffer->Handle(),
		                  vk::BufferCopy {offset, target.offset + uint64_t {load.slot} * sizeof(uint32_t),
		                                  bytes}});
	}
	auto& command = context.GetCommandScheduler().Current();
	command.EndRendering();
	const auto        native = command.Handle();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                       nullptr);
	for (const auto& [source, copy]: copies) {
		native.copyBuffer(source, target.buffer, 1, &copy);
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
	                       nullptr);
}

static bool TableViewKeepable(const ImageLookupState& state, const Image& image) {
	return !state.cpu_dirty && !state.buffer_modified &&
	       (!image.info.data.Empty() || image.is_depth_copy);
}

static bool TableViewCurrent(const TextureCache& texture_cache, const Image& image,
                             const TextureBinding& texture, const TableViewMemo& view) {
	return view.valid && texture.image_view &&
	       ImageLookupState::Of(texture.image_id, image) == view.state &&
	       (!image.is_depth_copy || texture_cache.DepthCopyCurrent(image));
}

static bool ReadsStencilPlane(const Image& image, const TextureCache::ImageDesc& desc) {
	return image.info.HasStencil() && desc.info.data.address >= image.info.stencil.address &&
	       desc.info.data.End() <= image.info.stencil.End();
}

static bool ImageReadReady(const Image& image, const TextureBinding& texture) {
	const auto layout = image.info.data.Empty() && !image.is_depth_copy ? vk::ImageLayout::eGeneral
	                    : image.info.IsDepth()  ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
	                                            : vk::ImageLayout::eShaderReadOnlyOptimal;
	if (texture.layout != layout) {
		return false;
	}
	const auto ready = [layout](const VulkanImageState& state) {
		return state.layout == layout && state.access_mask == vk::AccessFlagBits2::eShaderRead;
	};
	const auto& subresources = image.backing.subresource_states;
	const auto& view         = texture.desc.view_info;
	ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
	                             view.layer_count};
	if (image.info.IsVolume()) {
		range.base_layer  = 0;
		range.layer_count = 1;
	}
	const auto& resources = image.info.resources;
	const bool  partial   = range.base_level != 0 || range.level_count != resources.levels ||
	                     range.base_layer != 0 || range.layer_count != resources.layers;
	if (!partial && subresources.empty()) {
		return ready(image.backing.state);
	}
	if (subresources.empty()) {
		return ready(image.backing.state);
	}
	if (subresources.size() != size_t {resources.levels} * resources.layers) {
		return false;
	}
	for (uint32_t level = range.base_level; level < range.base_level + range.level_count; level++) {
		for (uint32_t layer = range.base_layer; layer < range.base_layer + range.layer_count;
		     layer++) {
			const auto index = size_t {level} * resources.layers + layer;
			if (index >= subresources.size() || !ready(subresources[index])) {
				return false;
			}
		}
	}
	return true;
}

static bool TableImageReady(const Image& image, const TextureBinding& texture) {
	return !image.binding.is_bound && !image.binding.is_target && !image.binding.force_general &&
	       texture.desc.type != TextureCache::BindingType::Storage && ImageReadReady(image, texture);
}

void RenderExecutor::MarkTableElement(TableBindingState& state, uint32_t element) {
	if (state.marks[element] != state.mark_stamp) {
		state.marks[element] = state.mark_stamp;
		m_table_marked.push_back(element);
	}
}

void RenderExecutor::MarkTablePages(TableBindingState& state, uint64_t address, uint64_t size) {
	if (size == 0) {
		return;
	}
	constexpr uint32_t PageBits = TextureCache::ImagePageBits;
	const uint64_t     first    = address >> PageBits;
	const uint64_t     last =
	    (size > UINT64_MAX - address ? UINT64_MAX : address + size - 1) >> PageBits;
	const uint64_t from = first > state.max_span ? first - state.max_span : 0;
	auto span = std::ranges::lower_bound(state.spans, from, {}, &TableBindingState::PageSpan::first);
	for (; span != state.spans.end() && span->first <= last; ++span) {
		if (span->last >= first) {
			MarkTableElement(state, span->element);
		}
	}
}

void RenderExecutor::MarkTableChanges(TableBindingState& state) {
	const auto backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
	if (backing_epoch != state.backing_epoch) {
		for (const auto element: state.unchecked_backing) {
			MarkTableElement(state, element);
		}
		m_table_changes.clear();
		if (Libs::LibKernel::Memory::BackingChangesSince(state.backing_epoch, m_table_changes)) {
			for (const auto& [address, size]: m_table_changes) {
				MarkTablePages(state, address, size);
			}
		} else {
			for (uint32_t element = 0; element < state.elements.size(); element++) {
				MarkTableElement(state, element);
			}
		}
		state.backing_epoch = backing_epoch;
	}
	auto&      texture_cache = m_context.GetTextureCache();
	const auto set_epoch     = texture_cache.ImageSetEpoch();
	if (set_epoch != state.set_epoch) {
		m_table_changes.clear();
		const bool known = texture_cache.ForEachImageSetChangeSince(
		    state.set_epoch,
		    [&](uint64_t address, uint64_t size) { m_table_changes.emplace_back(address, size); });
		if (known) {
			for (const auto& [address, size]: m_table_changes) {
				MarkTablePages(state, address, size);
			}
		} else {
			for (uint32_t element = 0; element < state.elements.size(); element++) {
				MarkTableElement(state, element);
			}
		}
		state.set_epoch = set_epoch;
	}
}

void RenderExecutor::RebuildTableSpans(TableBindingState& state) {
	constexpr uint32_t PageBits = TextureCache::ImagePageBits;
	state.spans.clear();
	state.max_span = 0;
	for (uint32_t element = 0; element < state.elements.size(); element++) {
		auto&       entry      = state.elements[element];
		const auto& resolution = *entry.resolution;
		entry.span_range       = resolution.range;
		entry.span_metadata    = resolution.metadata_range;
		for (const auto& range: {resolution.range, resolution.metadata_range}) {
			if (range.size == 0) {
				continue;
			}
			const uint64_t first = range.address >> PageBits;
			const uint64_t last  = (range.size > UINT64_MAX - range.address
			                            ? UINT64_MAX
			                            : range.address + range.size - 1) >>
			                      PageBits;
			state.spans.push_back({first, last, element});
			state.max_span = std::max(state.max_span, last - first);
		}
	}
	std::ranges::sort(state.spans, {}, &TableBindingState::PageSpan::first);
}

void RenderExecutor::AddTableSpans(TableBindingState& state, uint32_t element) {
	constexpr uint32_t PageBits   = TextureCache::ImagePageBits;
	auto&              entry      = state.elements[element];
	const auto&        resolution = *entry.resolution;
	entry.span_range              = resolution.range;
	entry.span_metadata           = resolution.metadata_range;
	for (const auto& range: {resolution.range, resolution.metadata_range}) {
		if (range.size == 0) {
			continue;
		}
		const uint64_t first = range.address >> PageBits;
		const uint64_t last  = (range.size > UINT64_MAX - range.address
		                            ? UINT64_MAX
		                            : range.address + range.size - 1) >>
		                      PageBits;
		state.spans.push_back({first, last, element});
		state.max_span = std::max(state.max_span, last - first);
	}
}

void RenderExecutor::MergeTableSpans(TableBindingState& state, size_t sorted) {
	auto& spans = state.spans;
	if (sorted == spans.size()) {
		return;
	}
	const auto middle = spans.begin() + static_cast<std::ptrdiff_t>(sorted);
	std::stable_sort(middle, spans.end(), [](const auto& a, const auto& b) {
		return a.first < b.first;
	});
	m_table_spans.clear();
	m_table_spans.reserve(spans.size());
	std::ranges::merge(std::ranges::subrange(spans.begin(), middle),
	                   std::ranges::subrange(middle, spans.end()),
	                   std::back_inserter(m_table_spans), {}, &TableBindingState::PageSpan::first,
	                   &TableBindingState::PageSpan::first);
	spans.swap(m_table_spans);
}

void RenderExecutor::UnpinTableBindings(TableBindingState& state) {
	auto& texture_cache = m_context.GetTextureCache();
	for (auto& element: state.elements) {
		if (element.pinned) {
			texture_cache.UnpinImage(element.pinned);
			element.pinned = {};
		}
	}
	state.valid = false;
}

void RenderExecutor::IndexTableSlots(TableBindingState& state) {
	state.element_slot_begin.assign(state.elements.size() + 1, 0);
	for (const auto element: state.slots) {
		state.element_slot_begin[element + 1]++;
	}
	for (size_t element = 0; element < state.elements.size(); element++) {
		state.element_slot_begin[element + 1] += state.element_slot_begin[element];
	}
	state.element_slots.resize(state.slots.size());
	std::vector<uint32_t> next(state.element_slot_begin.begin(), state.element_slot_begin.end() - 1);
	for (size_t slot = 0; slot < state.slots.size(); slot++) {
		state.element_slots[next[state.slots[slot]]++] = static_cast<uint32_t>(slot);
	}
}

void RenderExecutor::PatchTableSlots(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                     const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                                     TableBindingState& state, uint64_t stamp) {
	auto& texture_cache = m_context.GetTextureCache();
	const ShaderRecompiler::IR::DescriptorValue null_value {.dword_count = 8u};
	std::vector<uint32_t> roots;
	ForEachImageTable(program.info.images,
	                  [&](uint32_t root_index, auto ) { roots.push_back(root_index); });
	auto&        element_of   = state.element_of_version;
	const size_t sorted_spans = state.spans.size();
	for (const auto& changed: m_table_changed_slots) {
		auto&       range = state.ranges[changed.range];
		const auto& table = snapshot.image_tables.at(range.table);
		const auto& value = changed.index != 0u && changed.index < table.slots.size()
		                        ? table.slots[changed.index]
		                        : null_value;
		state.values[changed.slot] = value;
		auto&                   resolution = *range.resolution;
		auto&                   memo       = resolution.slots[changed.index];
		TableElementResolution* element    = nullptr;
		bool                    resolved   = false;
		if (memo.element != nullptr && memo.dwords == value.dwords) {
			element = memo.element;
		} else {
			if (const auto found = resolution.elements.find(value.dwords);
			    found != resolution.elements.end()) {
				element = &found->second;
			} else {
				element  = &ResolveTableElement(resolution, value, stamp);
				resolved = true;
			}
			memo = {value.dwords, element, element->version_slot};
		}
		const auto version_slot = memo.version_slot;
		if (version_slot < element_of.size() && element_of[version_slot] != UINT32_MAX) {
			state.slots[changed.slot] = element_of[version_slot];
			m_table_moved_slots.push_back(changed.slot);
			continue;
		}
		if (!resolved) {
			(void)ResolveTableElement(resolution, value, stamp, element);
		}
		const auto index = static_cast<uint32_t>(state.elements.size());
		state.elements.push_back({.resolution = element, .table = &resolution, .source = value,
		                          .root = roots.at(changed.range)});
		auto& entry = state.elements.back();
		if (element->volatile_dcc || (!state.compute && DepthTableElement(texture_cache, element->texture))) {
			state.always.push_back(index);
		} else if (!element->permanent && !element->backing_checkable) {
			state.unchecked_backing.push_back(index);
		}
		entry.pinned = element->texture.image_id;
		texture_cache.PinImage(entry.pinned);
		state.infos.emplace_back();
		state.marks.push_back(0);
		state.checked.emplace_back();
		AddTableSpans(state, index);
		EXIT_IF(version_slot == UINT32_MAX);
		if (version_slot >= element_of.size()) {
			element_of.resize(size_t {version_slot} + 1, UINT32_MAX);
		}
		element_of[version_slot]  = index;
		state.slots[changed.slot] = index;
		MarkTableElement(state, index);
	}
	MergeTableSpans(state, sorted_spans);
	IndexTableSlots(state);
	size_t unused = 0;
	for (size_t element = 0; element < state.elements.size(); element++) {
		unused += state.element_slot_begin[element + 1] == state.element_slot_begin[element] ? 1 : 0;
	}
	if (unused > std::max<size_t>(64, state.elements.size() / 16)) {
		DropUnusedTableElements(state);
	}
}

void RenderExecutor::DropUnusedTableElements(TableBindingState& state) {
	auto&      texture_cache = m_context.GetTextureCache();
	const auto count         = static_cast<uint32_t>(state.elements.size());
	auto&      remap         = m_table_remap;
	remap.assign(count, UINT32_MAX);
	uint32_t kept = 0;
	for (uint32_t element = 0; element < count; element++) {
		auto& entry = state.elements[element];
		if (state.element_slot_begin[element + 1] == state.element_slot_begin[element]) {
			if (entry.pinned) {
				texture_cache.UnpinImage(entry.pinned);
			}
			state.element_of_version[entry.resolution->version_slot] = UINT32_MAX;
			continue;
		}
		remap[element] = kept;
		if (kept != element) {
			state.element_of_version[entry.resolution->version_slot] = kept;
			state.elements[kept]                                     = std::move(entry);
			state.infos[kept]                                        = state.infos[element];
			state.marks[kept]                                        = state.marks[element];
			state.checked[kept]                                      = state.checked[element];
		}
		kept++;
	}
	state.elements.resize(kept);
	state.infos.resize(kept);
	state.marks.resize(kept);
	state.checked.resize(kept);
	for (auto& slot: state.slots) {
		slot = remap[slot];
	}
	const auto renumber = [&](std::vector<uint32_t>& elements) {
		std::erase_if(elements, [&](uint32_t element) { return remap[element] == UINT32_MAX; });
		for (auto& element: elements) {
			element = remap[element];
		}
	};
	renumber(state.always);
	renumber(state.unchecked_backing);
	renumber(m_table_marked);
	std::erase_if(state.spans, [&](const auto& span) { return remap[span.element] == UINT32_MAX; });
	for (auto& span: state.spans) {
		span.element = remap[span.element];
	}
	IndexTableSlots(state);
}

void RenderExecutor::FinishTableBindings(TableBindingState& state,
                                         const PreparedBindings& prepared) {
	if (!m_table_resolutions.Current(state.generation, state.ranges)) {
		m_table_stats.left_renewed++;
		UnpinTableBindings(state);
		return;
	}
	if (state.elements.size() != m_table_image_infos.size() ||
	    state.values.size() != prepared.table_slots.size()) {
		m_table_stats.left_size++;
		UnpinTableBindings(state);
		return;
	}
	auto& texture_cache = m_context.GetTextureCache();
	state.infos         = m_table_image_infos;
	state.slots         = prepared.table_slots;
	state.slot_infos.resize(state.slots.size());
	for (size_t slot = 0; slot < state.slots.size(); slot++) {
		state.slot_infos[slot] = state.infos[state.slots[slot]];
	}
	IndexTableSlots(state);
	auto& element_of = state.element_of_version;
	std::ranges::fill(element_of, UINT32_MAX);
	for (uint32_t index = 0; index < state.elements.size(); index++) {
		const auto version_slot = state.elements[index].resolution->version_slot;
		EXIT_IF(version_slot == UINT32_MAX);
		if (version_slot >= element_of.size()) {
			element_of.resize(size_t {version_slot} + 1, UINT32_MAX);
		}
		element_of[version_slot] = index;
	}
	state.always.clear();
	state.unchecked_backing.clear();
	for (uint32_t index = 0; index < state.elements.size(); index++) {
		auto&       element    = state.elements[index];
		const auto& resolution = *element.resolution;
		if (resolution.volatile_dcc || (!state.compute && DepthTableElement(texture_cache, resolution.texture))) {
			state.always.push_back(index);
		} else if (!resolution.permanent && !resolution.backing_checkable) {
			state.unchecked_backing.push_back(index);
		}
		element.pinned = resolution.texture.image_id;
		texture_cache.PinImage(element.pinned);
	}
	RebuildTableSpans(state);
	state.marks.assign(state.elements.size(), 0);
	state.checked.assign(state.elements.size(), {});
	state.mark_stamp = 0;
	state.valid      = true;
}

bool RenderExecutor::ReuseTableBindings(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                        const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                                        TableBindingState& state, uint64_t stamp,
                                        PreparedBindings& prepared) {
	if (!state.valid) {
		m_table_stats.unfinished++;
		return false;
	}
	if (!m_table_resolutions.SharedCurrent(state.generation)) {
		m_table_stats.renewed++;
		return false;
	}
	const ShaderRecompiler::IR::DescriptorValue null_value {.dword_count = 8u};
	size_t                                      range_index = 0;
	size_t                                      slot_index  = 0;
	bool                                        same        = true;
	bool                                        renewed     = false;
	m_table_changed_slots.clear();
	ForEachImageTable(program.info.images, [&](uint32_t root_index, auto ) {
		if (!same) {
			return;
		}
		const auto& root = program.info.images.at(root_index);
		if (range_index == state.ranges.size()) {
			same = false;
			return;
		}
		auto& cached = state.ranges[range_index++];
		if (cached.table != root.table || cached.capacity != root.table_capacity ||
		    !(cached.root == root)) {
			same = false;
			return;
		}
		if (!m_table_resolutions.Reusable(
		        *cached.resolution, cached.generation, state.generation,
		        [&]() -> TableResolution& { return FindTableResolution(root); }, TableOversized)) {
			renewed = true;
			same    = false;
			return;
		}
		const auto& table = snapshot.image_tables.at(root.table);
		if (table.slots_id != 0 && table.slots_id == cached.slots_id) {
			slot_index += root.table_capacity;
			return;
		}
		cached.slots_id = table.slots_id;
		for (uint32_t slot = 0; slot < root.table_capacity; slot++) {
			const auto& value =
			    slot != 0u && slot < table.slots.size() ? table.slots[slot] : null_value;
			if (!(state.values[slot_index] == value)) {
				m_table_changed_slots.push_back(
				    {static_cast<uint32_t>(slot_index), static_cast<uint32_t>(range_index - 1), slot});
			}
			slot_index++;
		}
	});
	if (!same) {
		(renewed ? m_table_stats.renewed : m_table_stats.layout)++;
		return false;
	}
	if (range_index != state.ranges.size()) {
		m_table_stats.layout++;
		return false;
	}
	state.mark_stamp++;
	m_table_marked.clear();
	m_table_moved_slots.clear();
	if (!m_table_changed_slots.empty()) {
		PatchTableSlots(program, snapshot, state, stamp);
	}
	auto& texture_cache = m_context.GetTextureCache();
	for (const auto element: state.always) {
		auto& resolution = *state.elements[element].resolution;
		if (resolution.volatile_dcc &&
		    (state.compute || !DepthTableElement(texture_cache, resolution.texture)) &&
		    texture_cache.ColorClearSettled(resolution.texture.desc, resolution.clear_memo)) {
			continue;
		}
		MarkTableElement(state, element);
	}
	MarkTableChanges(state);
	for (const auto element: m_table_marked) {
		auto& entry = state.elements[element];
		(void)ResolveTableElement(*entry.table, entry.source, stamp, entry.resolution);
	}
	prepared.table_visits      = m_table_marked;
	prepared.table_moved_slots = m_table_moved_slots;
	return true;
}

void RenderExecutor::RebindReusedTables(PreparedBindings& prepared) {
	auto&       state         = *static_cast<TableBindingState*>(prepared.table_state);
	const auto& program       = prepared.runtime->module->info;
	auto&       texture_cache = m_context.GetTextureCache();
	const auto  tick          = texture_cache.m_gc_tick;
	const auto  stale         = [&](const TextureBinding& binding) {
        if (!binding.image_id) {
            return false;
        }
        const auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
        return image == nullptr || (!image->registered && !image->info.data.Empty()) ||
               image->binding.needs_rebind;
	};
	const auto unchanged = [&](uint32_t index) {
		const auto& element = state.elements[index];
		const auto& texture = element.resolution->texture;
		const auto& view    = element.resolution->view;
		if (!texture.image_id) {
			return element.pinned == texture.image_id && texture.image_view == nullptr &&
			       state.infos[index].imageView == nullptr &&
			       texture.layout == state.infos[index].imageLayout;
		}
		const auto* image = texture_cache.m_slot_images.try_get(texture.image_id);
		return image != nullptr && !image->binding.needs_rebind &&
		       texture.image_id == element.pinned && texture.image_view != nullptr &&
		       texture.image_view == state.infos[index].imageView &&
		       texture.layout == state.infos[index].imageLayout &&
		       (image->info.data.Empty() ||
		        (image->registered && view.valid &&
		         ImageLookupState::Of(texture.image_id, *image) == view.state)) &&
		       (!image->is_depth_copy || texture_cache.DepthCopyCurrent(*image)) && TableImageReady(*image, texture);
	};
	const auto bound_stamp = ++m_bound_image_stamp;
	for (const auto id: m_bound_images) {
		if (!id) {
			continue;
		}
		if (id.index >= m_bound_image_marks.size()) {
			m_bound_image_marks.resize(size_t {id.index} + 1, 0);
		}
		m_bound_image_marks[id.index] = bound_stamp;
	}
	using Checked              = TableBindingState::Checked;
	const auto still_unchanged = [&](const Checked& checked) {
		if (checked.image == Checked::None ||
		    m_table_element_versions[checked.element] != checked.element_version) {
			return false;
		}
		return checked.image == Checked::NullImage ||
		       ((checked.image >= m_bound_image_marks.size() ||
		         m_bound_image_marks[checked.image] != bound_stamp) &&
		        texture_cache.ImageVersion(ImageId {checked.image}) == checked.image_version);
	};
	const auto element_count = static_cast<uint32_t>(state.elements.size());
	EXIT_IF(state.checked.size() != element_count);
	m_table_checks.clear();
	m_table_element_flags.assign((size_t {element_count} + 7) & ~size_t {7}, 0);
	GetExecHelpers().ParallelFor(element_count, 2048, [&](uint32_t begin, uint32_t end) {
		for (uint32_t index = begin; index < end; index++) {
			m_table_element_flags[index] =
			    state.marks[index] != state.mark_stamp && !still_unchanged(state.checked[index]) ? 1 : 0;
		}
	});
	for (uint32_t first = 0; first < element_count; first += 8) {
		uint64_t word = 0;
		std::memcpy(&word, m_table_element_flags.data() + first, sizeof(word));
		for (; word != 0; word &= word - 1) {
			m_table_checks.push_back(first + static_cast<uint32_t>(std::countr_zero(word)) / 8u);
		}
	}
	const auto check_count = static_cast<uint32_t>(m_table_checks.size());
	m_table_flags.assign(check_count, 0);
	GetExecHelpers().ParallelFor(check_count, 256, [&](uint32_t begin, uint32_t end) {
		for (uint32_t check = begin; check < end; check++) {
			const auto index   = m_table_checks[check];
			auto&      checked = state.checked[index];
			const auto& resolution = *state.elements[index].resolution;
			const auto  image_id   = resolution.texture.image_id;
			const auto  slot       = resolution.version_slot;
			const Checked seen {
			    .image           = image_id ? image_id.index : Checked::NullImage,
			    .image_version   = image_id ? texture_cache.ImageVersion(image_id) : 0u,
			    .element         = slot,
			    .element_version = slot < m_table_element_versions.size()
			                           ? m_table_element_versions[slot]
			                           : 0u};
			if (!unchanged(index)) {
				m_table_flags[check] = 1;
				checked              = {};
				continue;
			}
			const auto* image = image_id ? texture_cache.m_slot_images.try_get(image_id) : nullptr;
			if (slot >= m_table_element_versions.size() ||
			    (image != nullptr && (image->info.IsDepth() || image->is_depth_copy))) {
				checked = {};
				continue;
			}
			checked = seen;
		}
	});
	for (uint32_t check = 0; check < check_count; check++) {
		if (m_table_flags[check] != 0) {
			MarkTableElement(state, m_table_checks[check]);
		}
	}
	size_t visited = 0;
	for (;;) {
		for (; visited < m_table_marked.size(); visited++) {
			auto& element = state.elements[m_table_marked[visited]];
			auto& texture = element.resolution->texture;
			auto& view    = element.resolution->view;
			if (stale(texture)) {
				if (auto* old_image = texture_cache.m_slot_images.try_get(texture.image_id)) {
					old_image->binding = {};
				}
				texture = ResolveTableTexture(program.info.images.at(element.root), element.source);
				view.valid = false;
			}
			BindImage(texture.image_id, false);
			if (!texture.image_id) {
				view.valid = false;
			} else if (auto* image = texture_cache.m_slot_images.try_get(texture.image_id);
			    image != nullptr && TableViewCurrent(texture_cache, *image, texture, view)) {
				image->tick_accessed_last = tick;
				texture_cache.TouchImage(*image);
			} else {
				texture.image_view  = texture_cache.FindTexture(texture.image_id, texture.desc);
				auto& acquired         = texture_cache.GetImage(texture.image_id);
				view.state             = ImageLookupState::Of(texture.image_id, acquired);
				view.valid             = TableViewKeepable(view.state, acquired);
			}
			if (texture.image_id != element.pinned) {
				texture_cache.UnpinImage(element.pinned);
				element.pinned = texture.image_id;
				texture_cache.PinImage(element.pinned);
			}
			TableElementChanged(*element.resolution);
			state.checked[m_table_marked[visited]] = {};
		}
		MarkTableChanges(state);
		if (visited == m_table_marked.size()) {
			break;
		}
		for (auto index = visited; index < m_table_marked.size(); index++) {
			auto& element = state.elements[m_table_marked[index]];
			(void)ResolveTableElement(*element.table, element.source, m_table_prepare_stamp,
			                          element.resolution);
		}
	}
	for (const auto index: m_table_marked) {
		const auto& element    = state.elements[index];
		const auto& resolution = *element.resolution;
		if (resolution.range.address != element.span_range.address ||
		    resolution.range.size != element.span_range.size ||
		    resolution.metadata_range.address != element.span_metadata.address ||
		    resolution.metadata_range.size != element.span_metadata.size) {
			RebuildTableSpans(state);
			break;
		}
	}
	prepared.table_visits = m_table_marked;
}

void RenderExecutor::FlushDeferredDispatchBarrier() {
	m_dispatch_chained = false;
	if (!m_deferred_barrier) {
		return;
	}
	m_deferred_barrier = false;
	ShaderAccessBarrier(m_context.GetCommandScheduler().Current().Handle(),
	                    vk::PipelineStageFlagBits::eComputeShader);
}

void RenderExecutor::BeginGuestOperation(uint32_t queue, bool dispatch, bool draw,
                                         bool guest_sync) {
	if (!draw || guest_sync || queue != m_guest_queue) {
		if (auto* native_rt = m_context.GetRayTracing(); native_rt != nullptr) {
			native_rt->NoteGuestOperation();
		}
	}
	m_guest_queue = queue;
	m_context.GetOcclusionQueries().DropReadTargets();
	if (m_shader_write_deferred && (!draw || guest_sync || queue != m_shader_write_queue)) {
		m_shader_write_deferred = false;
		m_context.GetCommandScheduler().EndRendering();
	}
	if (m_deferred_barrier && dispatch && !guest_sync && queue == m_deferred_barrier_queue) {
		m_dispatch_chained = true;
		return;
	}
	FlushDeferredDispatchBarrier();
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
#if defined(ANYPS5_KYTY)
	const auto frame                 = m_context.FlipNumber();
	image.target_frames[frame & 3u] = frame + 1u;
#endif
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (auto& elements: m_retired_table_elements) {
		for (auto& [dwords, element]: elements) {
			if (element.version_slot < m_table_element_versions.size()) {
				m_table_element_versions[element.version_slot]++;
				m_free_table_element_versions.push_back(element.version_slot);
			}
		}
	}
	m_retired_table_elements.clear();
	if (!m_table_resolutions.SharedGeneration() && m_table_element_count > TableElementBudget) {
		TrimTableResolutions();
	}
	if ((++m_table_stats.resets & 255u) == 0) {
		LogTableSummary();
	}
	auto& texture_cache = m_context.GetTextureCache();
	for (const auto id: m_bound_images) {
		auto& image = texture_cache.m_slot_images[id];
		if (image.is_stencil_copy && image.binding.shader_write) {
			texture_cache.FinishStencilStorageWrite(id);
		}
		image.binding = {};
	}
	m_bound_images.clear();
	if (m_clear_texture_descs) {
		m_clear_texture_descs = false;
		m_texture_descs.clear();
		m_texture_descs_generation++;
	}
}

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = runtime.module->info;
	const auto& snapshot = *runtime.resources;
	const Common::HackNotice::ScopedShader shader_scope(program.shader_hash);
	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.flattened_srt_known = false;
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.template_stage  = nullptr;
	prepared.template_images = 0;
	prepared.images.resize(program.info.images.size());
	prepared.shader_data.clear();
	const bool same_program = prepared.last_program == &program &&
	                          prepared.last_descs_generation == m_texture_descs_generation;
	const bool same_images  = same_program && prepared.last_images == snapshot.images;
	const bool same_slots = same_program && prepared.last_images.size() == snapshot.images.size() &&
	                        prepared.image_descs.size() == program.info.images.size();
	if (!same_slots) {
		prepared.image_descs.assign(program.info.images.size(), nullptr);
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (program.info.images[i].heap) {
			prepared.images[i] = {};
			continue;
		}
		const auto& resource = program.info.images[i];
		if (!resource.written &&
		    resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			auto* derived = prepared.image_descs[i];
			if (!same_images &&
			    (derived == nullptr || !same_slots || prepared.last_images[i] != snapshot.images[i])) {
				const auto cached =
				    m_texture_descs.find(MakeTextureDescKey(resource, snapshot.images[i]));
				derived = cached != m_texture_descs.end() ? &cached->second : nullptr;
			}
			if (derived != nullptr && derived->found.valid &&
			    ReuseTextureLookup(*derived, prepared.images[i])) {
				prepared.image_descs[i] = derived;
				BindImage(prepared.images[i].image_id, false);
				continue;
			}
		}
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i]);
		prepared.image_descs[i] = binding.lookup;
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	if (!same_images) {
		prepared.last_images = snapshot.images;
	}
	prepared.table_images.clear();
	prepared.table_views.clear();
	prepared.table_sources.clear();
	prepared.table_roots.clear();
	prepared.table_versions.clear();
	prepared.table_slots.clear();
	prepared.table_state   = nullptr;
	prepared.tables_reused = false;
	prepared.table_visits.clear();
	prepared.table_moved_slots.clear();
	m_table_stats.bindings++;
	const auto stamp = ++m_table_prepare_stamp;
	TableBindingState* table_state = nullptr;
	if (std::ranges::any_of(program.info.images, [](const ShaderRecompiler::IR::ImageResource& image) {
		    return image.table_capacity != 0u;
	    })) {
		table_state           = &m_table_states[&program];
		table_state->compute  = program.stage == ShaderType::Compute;
		prepared.table_state  = table_state;
		prepared.tables_reused =
		    ReuseTableBindings(program, snapshot, *table_state, stamp, prepared);
		m_table_stats.prepared++;
		m_table_stats.reused += prepared.tables_reused ? 1 : 0;
		if (!prepared.tables_reused) {
			UnpinTableBindings(*table_state);
			table_state->valid         = false;
			table_state->generation    = m_table_resolutions.Generation();
			table_state->set_epoch     = m_context.GetTextureCache().ImageSetEpoch();
			table_state->backing_epoch = Libs::LibKernel::Memory::BackingEpoch();
			table_state->ranges.clear();
			table_state->values.clear();
			table_state->elements.clear();
		}
	}
	ForEachImageTable(program.info.images, [&](uint32_t root_index, auto ) {
		if (prepared.tables_reused) {
			return;
		}
		const auto& root  = program.info.images.at(root_index);
		const auto& table = snapshot.image_tables.at(root.table);
		const ShaderRecompiler::IR::DescriptorValue null_value {.dword_count = 8u};
		auto& resolution = FindTableResolution(root);
		table_state->ranges.push_back(
		    {root, &resolution, root.table, root.table_capacity, table.slots_id, resolution.generation});
		const auto add = [&](TableElementResolution&                      element,
		                     const ShaderRecompiler::IR::DescriptorValue& value) {
			if (element.prepared_stamp != stamp) {
				element.prepared_stamp = stamp;
				element.prepared_index = static_cast<uint32_t>(prepared.table_images.size());
				BindImage(element.texture.image_id, false);
				prepared.table_images.push_back(&element.texture);
				prepared.table_views.push_back(&element.view);
				prepared.table_sources.push_back(value);
				prepared.table_roots.push_back(root_index);
				prepared.table_versions.push_back(element.version_slot);
				table_state->elements.push_back(
				    {.resolution = &element, .table = &resolution, .source = value,
				     .root = root_index});
			}
			prepared.table_slots.push_back(element.prepared_index);
		};
		if (resolution.slots.size() < root.table_capacity) {
			resolution.slots.resize(root.table_capacity);
		}
		for (uint32_t slot = 0; slot < root.table_capacity; slot++) {
			const auto& value =
			    slot != 0u && slot < table.slots.size() ? table.slots[slot] : null_value;
			table_state->values.push_back(value);
			auto& memo = resolution.slots[slot];
			if (memo.element != nullptr && memo.dwords == value.dwords) {
				add(ResolveTableElement(resolution, value, stamp, memo.element), value);
			} else {
				auto& element = ResolveTableElement(resolution, value, stamp);
				memo          = {value.dwords, &element, element.version_slot};
				add(element, value);
			}
		}
	});
	if (!same_program || prepared.samplers.size() != program.info.samplers.size() ||
	    prepared.last_samplers != snapshot.samplers) {
		prepared.samplers.clear();
		prepared.samplers.reserve(program.info.samplers.size());
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			prepared.samplers.push_back(NativeSampler(
			    program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
		}
		prepared.last_samplers = snapshot.samplers;
	}
	prepared.last_program          = &program;
	prepared.last_descs_generation = m_texture_descs_generation;
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

bool RenderExecutor::DrawTemplateEligible(std::span<const ShaderStageRuntime* const> stages) {
	if (stages.size() > std::tuple_size_v<decltype(DrawTemplate::stages)>) {
		return false;
	}
	for (const auto* stage: stages) {
		const auto& program = stage->module->info;
		if (program.info.uses_image_heap || program.info.images.size() > 64u) {
			return false;
		}
		for (const auto& image: program.info.images) {
			if (image.heap || image.table_capacity != 0u || image.written ||
			    image.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
				return false;
			}
		}
	}
	return true;
}

DrawTemplate* RenderExecutor::FindDrawTemplate(uint64_t key,
                                               std::span<const ShaderStageRuntime* const> stages) {
	if (m_draw_templates == nullptr) {
		return nullptr;
	}
	auto* found = m_draw_templates->Find(key);
	return found != nullptr && DrawTemplateStagesMatch(*found, stages) ? found : nullptr;
}

bool RenderExecutor::DrawTemplateStagesMatch(const DrawTemplate&                        found,
                                             std::span<const ShaderStageRuntime* const> stages) {
	if (found.stage_count != stages.size()) {
		return false;
	}
	for (uint32_t s = 0; s < found.stage_count; s++) {
		const auto& stage    = found.stages[s];
		const auto& snapshot = *stages[s]->resources;
		if (stage.program != &stages[s]->module->info || stage.image_values != snapshot.images ||
		    stage.sampler_values != snapshot.samplers) {
			return false;
		}
	}
	return true;
}

bool RenderExecutor::DrawTemplateRefused(uint64_t key) const {
	const auto* found = m_draw_templates != nullptr ? m_draw_templates->Find(key) : nullptr;
	return found != nullptr && found->stage_count == 0 &&
	       found->refused_tick == m_context.GetTextureCache().m_gc_tick;
}

void RenderExecutor::PrepareBindingsFromTemplate(const ShaderStageRuntime& runtime,
                                                 const DrawTemplateStage&  stage,
                                                 PreparedBindings&         prepared) {
	const auto& program  = runtime.module->info;
	const auto& snapshot = *runtime.resources;
	prepared.runtime            = &runtime;
	prepared.gds                = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt      = {};
	prepared.flattened_srt_known = false;
	prepared.shader_data_buffer = {};
	prepared.shared_memory      = {};
	prepared.template_stage  = &stage;
	prepared.template_images = 0;
	prepared.images.resize(stage.images.size());
	for (size_t i = 0; i < stage.images.size(); i++) {
		const auto& source  = stage.images[i];
		auto&       binding = prepared.images[i];
		binding.image_id    = source.image_id;
		binding.image_view  = source.image_view;
		binding.layout      = source.layout;
		if (!binding.mip_views.empty()) {
			binding.mip_views.clear();
		}
		binding.lookup = nullptr;
		binding.reused = true;
		if (source.image_id) {
			prepared.template_images |= uint64_t {1} << i;
			binding.desc.type = source.desc.type;
		} else {
			binding.desc = source.desc;
		}
	}
	prepared.table_images.clear();
	prepared.table_views.clear();
	prepared.table_sources.clear();
	prepared.table_roots.clear();
	prepared.table_versions.clear();
	prepared.table_slots.clear();
	prepared.table_state   = nullptr;
	prepared.tables_reused = false;
	prepared.table_visits.clear();
	prepared.table_moved_slots.clear();
	prepared.samplers.assign(stage.samplers.begin(), stage.samplers.end());
	prepared.last_program = nullptr;
	prepared.shader_data.clear();
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
}

uint64_t RenderExecutor::TemplateRangeEpoch(uint64_t address, uint64_t size) {
	auto&      texture_cache = m_context.GetTextureCache();
	const auto set_epoch     = texture_cache.ImageSetEpoch();
	if (set_epoch != m_template_pages_epoch) {
		if (m_template_page_epochs.empty()) {
			m_template_page_epochs.assign(TemplatePageBuckets, 0);
		}
		auto       change = m_template_pages_epoch;
		const bool known  = texture_cache.ForEachImageSetChangeSince(
            m_template_pages_epoch, [&](uint64_t change_address, uint64_t change_size) {
                change++;
                if (change_size == 0) {
                    return;
                }
                const auto last = (change_address + change_size - 1) >> TextureCache::ImagePageBits;
                for (auto page = change_address >> TextureCache::ImagePageBits; page <= last; page++) {
                    m_template_page_epochs[page & (TemplatePageBuckets - 1)] = change;
                }
            });
		if (known) {
			m_template_pages_epoch = change;
		} else {
			m_template_pages_epoch = set_epoch;
			m_template_pages_floor = set_epoch;
		}
	}
	uint64_t   epoch = m_template_pages_floor;
	const auto last  = (address + std::max<uint64_t>(size, 1) - 1) >> TextureCache::ImagePageBits;
	for (auto page = address >> TextureCache::ImagePageBits; page <= last; page++) {
		epoch = std::max(epoch, m_template_page_epochs[page & (TemplatePageBuckets - 1)]);
	}
	return epoch;
}

bool RenderExecutor::CheckTemplateImages(PreparedBindings& prepared) {
	auto&      stage         = *const_cast<DrawTemplateStage*>(prepared.template_stage);
	auto&      texture_cache = m_context.GetTextureCache();
	const auto set_epoch     = texture_cache.ImageSetEpoch();
	const auto tick          = texture_cache.m_gc_tick;
	const bool touch         = stage.touched_tick != tick;
	const bool set_moved     = stage.image_set_epoch != set_epoch;
	bool       changed       = false;
	for (auto mask = prepared.template_images; mask != 0; mask &= mask - 1) {
		const auto  i      = static_cast<uint32_t>(std::countr_zero(mask));
		const auto& source = stage.images[i];
		bool current = texture_cache.ImageVersion(source.image_id) == stage.image_versions[i];
		if (current && set_moved) {
			const auto& range = source.desc.info.data;
			current           = TemplateRangeEpoch(range.address, range.size) <= stage.image_set_epoch;
		}
		if (!current) {
			const auto& program  = prepared.runtime->module->info;
			const auto& snapshot = *prepared.runtime->resources;
			const Common::HackNotice::ScopedShader shader_scope(program.shader_hash);
			prepared.template_images &= ~(uint64_t {1} << i);
			prepared.images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(prepared.images[i].image_id, false);
			changed = true;
			continue;
		}
		if (touch) {
			auto& image              = texture_cache.GetImage(source.image_id);
			image.tick_accessed_last = tick;
			texture_cache.TouchImage(image);
		}
	}
	if (changed) {
		return false;
	}
	stage.image_set_epoch = set_epoch;
	stage.touched_tick    = tick;
	return true;
}

void RenderExecutor::StoreDrawTemplate(uint64_t key, std::span<PreparedBindings* const> stages) {
	if (stages.size() > std::tuple_size_v<decltype(DrawTemplate::stages)>) {
		return;
	}
	if (m_draw_templates == nullptr) {
		m_draw_templates = std::make_unique<DrawTemplateTable>();
	}
	auto& texture_cache = m_context.GetTextureCache();
	auto& stored        = m_draw_templates->Insert(key);
	stored.stage_count  = 0;
	stored.refused_tick = 0;
	stored.commit_valid = false;
	for (const auto* prepared: stages) {
		const auto& snapshot  = *prepared->runtime->resources;
		auto&       stage     = stored.stages[stored.stage_count++];
		stage.program         = &prepared->runtime->module->info;
		stage.image_values    = snapshot.images;
		stage.sampler_values  = snapshot.samplers;
		stage.samplers        = prepared->samplers;
		stage.image_set_epoch = texture_cache.ImageSetEpoch();
		stage.touched_tick    = texture_cache.m_gc_tick;
		stage.images.resize(prepared->images.size());
		stage.image_versions.resize(prepared->images.size());
		for (size_t i = 0; i < prepared->images.size(); i++) {
			const auto& binding = prepared->images[i];
			auto&       copy    = stage.images[i];
			copy.image_id       = binding.image_id;
			copy.image_view     = binding.image_view;
			const bool held = prepared->template_stage != nullptr && i < 64u &&
			                  ((prepared->template_images >> i) & 1u) != 0;
			if (!held) {
				copy.desc = binding.desc;
			} else if (const auto& source = prepared->template_stage->images[i]; &source != &copy) {
				copy.desc = source.desc;
			}
			copy.layout         = binding.layout;
			copy.mip_views.clear();
			copy.lookup             = nullptr;
			copy.reused             = false;
			stage.image_versions[i] = 0;
			if (!binding.image_id) {
				continue;
			}
			const auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
			if (image == nullptr || !image->registered || image->version == nullptr ||
			    image->binding.needs_rebind || image->binding.is_target ||
			    image->binding.force_general || image->binding.shader_write ||
			    (image->info.IsDepth() && ReadsStencilPlane(*image, copy.desc)) ||
			    image->is_depth_copy || image->is_stencil_copy || image->depth_id ||
			    binding.image_view == nullptr ||
			    binding.desc.type != TextureCache::BindingType::Texture ||
			    !ImageReadReady(*image, copy)) {
				stored.stage_count  = 0;
				stored.refused_tick = texture_cache.m_gc_tick;
				return;
			}
			stage.image_versions[i] = texture_cache.ImageVersion(binding.image_id);
		}
	}
}

bool RenderExecutor::TemplateCommitsSupported() {
	static const bool supported = VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdPushDescriptorSetWithTemplateKHR != nullptr &&
	                              VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateDescriptorUpdateTemplate != nullptr;
	return supported;
}

void RenderExecutor::StoreTemplateCommit(DrawTemplate& stored, const PipelineCache::Pipeline& pipeline) {
	stored.commit_valid = false;
	stored.commit_data.clear();
	stored.commit_slots.clear();
	if (m_descriptor_writes.empty()) {
		return;
	}
	std::vector<vk::DescriptorUpdateTemplateEntry> entries;
	entries.reserve(m_descriptor_writes.size());
	for (size_t i = 0; i < m_descriptor_writes.size(); i++) {
		const auto& write = m_descriptor_writes[i];
		const auto  stage = m_commit_capture_kinds[i].first;
		const auto  kind  = m_commit_capture_kinds[i].second;
		const bool  image = write.pImageInfo != nullptr;
		if (write.pTexelBufferView != nullptr || image == (write.pBufferInfo != nullptr) ||
		    write.descriptorCount == 0 || write.descriptorCount > UINT16_MAX) {
			return;
		}
		const size_t stride = image ? sizeof(vk::DescriptorImageInfo) : sizeof(vk::DescriptorBufferInfo);
		const size_t offset = (stored.commit_data.size() + 7u) & ~size_t {7};
		const size_t bytes  = stride * write.descriptorCount;
		stored.commit_data.resize(offset + bytes);
		std::memcpy(stored.commit_data.data() + offset,
		            image ? static_cast<const void*>(write.pImageInfo) : static_cast<const void*>(write.pBufferInfo),
		            bytes);
		entries.push_back({write.dstBinding, write.dstArrayElement, write.descriptorCount, write.descriptorType,
		                   offset, stride});
		if (!image) {
			stored.commit_slots.push_back({stage, kind, static_cast<uint16_t>(write.descriptorCount),
			                               static_cast<uint32_t>(offset)});
		}
	}
	uint64_t hash = XXH3_64bits(entries.data(), entries.size() * sizeof(vk::DescriptorUpdateTemplateEntry));
	hash          = XXH3_64bits_withSeed(&pipeline.pipeline_layout, sizeof(pipeline.pipeline_layout), hash);
	auto& update_template = m_commit_templates[hash];
	if (!update_template) {
		vk::DescriptorUpdateTemplateCreateInfo create {};
		create.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
		create.pDescriptorUpdateEntries   = entries.data();
		create.templateType               = vk::DescriptorUpdateTemplateType::ePushDescriptors;
		create.descriptorSetLayout        = pipeline.descriptor_set_layout;
		create.pipelineBindPoint          = vk::PipelineBindPoint::eGraphics;
		create.pipelineLayout             = pipeline.pipeline_layout;
		create.set                        = 0;
		if (m_context.GetGraphics().device.createDescriptorUpdateTemplate(&create, nullptr, &update_template) !=
		    vk::Result::eSuccess) {
			update_template = nullptr;
			return;
		}
	}
	stored.commit_template = update_template;
	stored.commit_layout   = pipeline.pipeline_layout;
	stored.commit_valid    = true;
}

void RenderExecutor::CommitTemplateBindings(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
                                            std::span<PreparedBindings* const> stages, const DrawTemplate& found) {
	auto                           vk_buffer = buffer.Handle();
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	for (const auto* prepared: stages) {
		const auto& program = prepared->runtime->module->info;
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data, push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}
	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}
	m_commit_data.resize(found.commit_data.size());
	std::memcpy(m_commit_data.data(), found.commit_data.data(), found.commit_data.size());
	for (const auto& slot: found.commit_slots) {
		const auto& descriptors = *stages[slot.stage];
		auto*       views       = m_commit_data.data() + slot.offset;
		const auto  put         = [&](const vk::DescriptorBufferInfo* source, size_t count) {
			EXIT_IF(count != slot.count);
			std::memcpy(views, source, sizeof(vk::DescriptorBufferInfo) * count);
		};
		switch (static_cast<BindingKind>(slot.kind)) {
			case BindingKind::Buffers: put(descriptors.buffers.data(), descriptors.buffers.size()); break;
			case BindingKind::UniformBuffers:
				put(descriptors.uniform_buffers.data(), descriptors.uniform_buffers.size());
				break;
			case BindingKind::BdaPagetable:
			case BindingKind::FaultBuffer: {
				auto&       cache      = m_context.GetBufferCache();
				const auto* bda_buffer = static_cast<BindingKind>(slot.kind) == BindingKind::BdaPagetable
				                             ? cache.GetBdaPageTableBuffer()
				                             : cache.GetFaultBuffer();
				const vk::DescriptorBufferInfo view {bda_buffer->Handle(), 0, bda_buffer->Size()};
				put(&view, 1);
				break;
			}
			case BindingKind::MipStats: {
				const auto&                    counters = m_context.GetLodStats().CounterBuffer();
				const vk::DescriptorBufferInfo view {counters.Handle(), 0, counters.Size()};
				put(&view, 1);
				break;
			}
			case BindingKind::FlattenedSrt: put(&descriptors.flattened_srt, 1); break;
			case BindingKind::ShaderData: put(&descriptors.shader_data_buffer, 1); break;
			case BindingKind::SharedMemory: put(&descriptors.shared_memory, 1); break;
			case BindingKind::Gds: put(&descriptors.gds, 1); break;
			default: EXIT("invalid template commit binding kind %u\n", static_cast<uint32_t>(slot.kind));
		}
	}
	vk_buffer.pushDescriptorSetWithTemplateKHR(found.commit_template, pipeline.pipeline_layout, 0,
	                                           m_commit_data.data(), m_commit_data.size());
}

[[gnu::noinline]] static uint64_t BufferMapCheckFloor(uint64_t address, uint64_t size) noexcept {
	return GuestRange {address, size}.Valid() ? uint64_t {1} << 30u : 0;
}

[[gnu::noinline]] static bool NullBufferNoticed(uint64_t shader_hash) noexcept {
	return !Common::HackNotice::Report("V# outside guest memory bound as a null buffer",
	                                   "RenderExecutor::FindBuffers", shader_hash, 0);
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages, bool find_buffers) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto&       prepared = *stage;
		const auto& program  = prepared.runtime->module->info;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout   = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (size != 0 && descriptor.Stride() == 0 && descriptor.OutOfBounds() == 2 &&
			    !metadata.scalar) {
				size = std::max(size, uint64_t {metadata.max_byte_extent});
			}
			if (address == 0 || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					for (const auto* reader: stages) {
						const bool own_compute_reads =
						    reader == stage && program.stage == ShaderType::Compute;
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size && !own_compute_reads) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				}
				const auto& graphics = m_context.GetGraphics();
				const auto  limit =
				    uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
				    (graphics.StorageMinAlignment() - 1);
				size = std::min(std::min(size, uint64_t {256} * 1024 * 1024), limit);
			}
			if (size >= BufferMapCheckFloor(address, size) || descriptor.NumRecords() == UINT32_MAX) {
#if defined(ANYPS5_KYTY)
				if (!Libs::LibKernel::Memory::IsFullyMapped(address, 1)) {
					if (!NullBufferNoticed(program.shader_hash)) {
						::Log::WriteToConsoleAndLog(fmt::format(
						    "Buffer: a V# whose base is outside guest memory is bound as a null buffer "
						    "(shader 0x{:016x}, resource {}, base 0x{:016x}, size 0x{:x}, "
						    "V# {:08x} {:08x} {:08x} {:08x})\n",
						    program.shader_hash, resource, address, size, descriptor.fields[0],
						    descriptor.fields[1], descriptor.fields[2], descriptor.fields[3]));
					}
					source = {};
					continue;
				}
#endif
				size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			}
			source = {address, size, find_buffers ? cache.FindBuffer(address, size) : NULL_BUFFER_ID};
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program  = prepared.runtime->module->info;
	const auto& snapshot = *prepared.runtime->resources;
	const auto& layout   = program.bindings;

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	prepared.uniform_buffers.resize(
	    layout.descriptor_counts[static_cast<size_t>(BindingKind::UniformBuffers)]);
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword,
	          prepared.shader_data.end(), 0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		const auto& source = prepared.buffer_sources[index];
		const uint32_t flags = (resource.uniform_index != UINT32_MAX ? 1u : 0u) |
		                       (resource.formatted ? 2u : 0u);
		const bool     share = m_draw_buffer_views_on && !resource.written && source.address != 0 && source.size != 0;
		const DrawStreamRead* streamed = nullptr;
		if (m_draw_stream_read_count != 0 && !resource.written && (flags & 2u) == 0) {
			for (uint32_t i = 0; i < m_draw_stream_read_count; i++) {
				const auto& read = m_draw_stream_reads[i];
				if (read.address == source.address && read.size == source.size && read.offset != UINT64_MAX) {
					streamed = &read;
					break;
				}
			}
		}
		const DrawBufferView* shared = nullptr;
		if (share && streamed == nullptr) {
			for (uint32_t i = 0; i < m_draw_buffer_view_count; i++) {
				const auto& view = m_draw_buffer_views[i];
				if (view.address == source.address && view.size == source.size && view.flags == flags) {
					shared = &view;
					break;
				}
			}
		}
		if (streamed != nullptr) {
			prepared.buffers[index] = StreamedStorageBuffer(m_context, streamed->offset, source.size, resource, buffer_offset);
		} else if (shared != nullptr) {
			prepared.buffers[index] = shared->view;
			buffer_offset           = shared->offset;
		} else {
			prepared.buffers[index] = NativeStorageBuffer(m_context, source, resource, buffer_offset);
			if (share && m_draw_buffer_view_count < m_draw_buffer_views.size()) {
				m_draw_buffer_views[m_draw_buffer_view_count++] = {source.address, source.size, flags, buffer_offset,
				                                                   prepared.buffers[index]};
			}
		}
		pack_memory_offset(index, buffer_offset);
		prepared.shader_data[layout.MemorySizeDword() + index] =
		    static_cast<uint32_t>(prepared.buffer_sources[index].size + buffer_offset);
		if (resource.uniform_index != UINT32_MAX) {
			auto view = prepared.buffers[index];
			if (view.buffer != nullptr) {
				view.range = std::min(
				    std::min<vk::DeviceSize>(view.range, ShaderRecompiler::IR::UniformBufferBytes),
				    vk::DeviceSize {m_context.GetGraphics().GetPhysicalDeviceProperties().limits.maxUniformBufferRange});
			}
			prepared.uniform_buffers[resource.uniform_index] = view;
		}
	}
	if (layout.stride_count != 0u) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto index = program.info.buffers[i].descriptor_index;
			if (index >= layout.stride_count || i >= snapshot.buffers.size() ||
			    !ShaderRecompiler::IR::HasRuntimeStride(program.info.buffers[i].packed_stride)) {
				continue;
			}
			ShaderBufferResource descriptor;
			std::copy_n(snapshot.buffers[i].dwords.begin(), std::size(descriptor.fields), descriptor.fields);
			prepared.shader_data[layout.StrideDword() + index] =
			    descriptor.PackedStride() & ShaderRecompiler::IR::BufferStrideMask;
		}
	}
	for (uint32_t i = 0; i < layout.mip_stats_count; i++) {
		const auto& resource = program.info.images[i];
		const auto& value    = snapshot.images[i];
		prepared.shader_data[layout.MipStatsDword() + i] =
		    !resource.heap && resource.table_capacity == 0u && value.dword_count >= 8u
		        ? LodStatsReport::ImageField(value.dwords.data())
		        : LodStatsReport::NoCounterFlag;
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt       = NativeUpload(m_context, snapshot.flattened_srt);
		if (layout.flattened_srt_uniform) {
			prepared.flattened_srt.range = std::min<vk::DeviceSize>(
			    prepared.flattened_srt.range,
			    m_context.GetGraphics().GetPhysicalDeviceProperties().limits.maxUniformBufferRange);
		}
		prepared.flattened_srt_known = snapshot.gpu_flat_loads.empty();
		if (!snapshot.gpu_flat_loads.empty()) {
			CopyGpuFlatLoads(m_context, prepared.flattened_srt, snapshot.gpu_flat_loads);
			snapshot.gpu_flat_loads.clear();
		}
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	const auto& program       = prepared.runtime->module->info;
	const auto& snapshot      = *prepared.runtime->resources;
	const Common::HackNotice::ScopedShader shader_scope(program.shader_hash);
	auto&       images        = prepared.images;
	auto&       texture_cache = m_context.GetTextureCache();
	const auto stale = [&](const TextureBinding& binding) {
		if (!binding.image_id) {
			return false;
		}
		const auto* image = texture_cache.m_slot_images.try_get(binding.image_id);
		return image == nullptr || (!image->registered && !image->info.data.Empty()) ||
		       image->binding.needs_rebind;
	};
	std::vector<bool> stale_tables(prepared.table_images.size());
	for (size_t k = 0; k < prepared.table_images.size(); k++) {
		stale_tables[k] = stale(*prepared.table_images[k]);
	}
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		auto& old_image = texture_cache.m_slot_images[images[i].image_id];
		if (!old_image.registered || old_image.binding.needs_rebind) {
			old_image.binding = {};
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i]);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
			if (!images[i].image_id) continue;
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
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else if (!binding.reused) {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		if (binding.lookup != nullptr && !binding.reused) {
			auto& found           = binding.lookup->found;
			found.id              = binding.image_id;
			found.desc            = binding.desc;
			found.view            = binding.image_view;
			found.image_set_epoch = texture_cache.ImageSetEpoch();
			found.state           = ImageLookupState::Of(binding.image_id, image);
			found.valid           = true;
		}
		binding.lookup = nullptr;
		binding.reused = false;
	}
	const auto tick = m_context.GetTextureCache().m_gc_tick;
	EXIT_IF(prepared.table_versions.size() != prepared.table_images.size());
	const auto element_changed = [&](size_t k) {
		if (const auto slot = prepared.table_versions[k]; slot < m_table_element_versions.size()) {
			m_table_element_versions[slot]++;
		}
	};
	for (size_t k = 0; k < prepared.table_images.size(); k++) {
		auto& texture = *prepared.table_images[k];
		auto& view    = *prepared.table_views[k];
		if (stale_tables[k] || stale(texture)) {
			if (auto* old_image = texture_cache.m_slot_images.try_get(texture.image_id)) {
				old_image->binding = {};
			}
			texture = ResolveTableTexture(program.info.images.at(prepared.table_roots[k]),
			                              prepared.table_sources[k]);
			BindImage(texture.image_id, false);
			view.valid = false;
			element_changed(k);
		}
		if (!texture.image_id) {
			view.valid = false;
			continue;
		}
		if (auto* image = texture_cache.m_slot_images.try_get(texture.image_id);
		    image != nullptr && TableViewCurrent(texture_cache, *image, texture, view)) {
			image->tick_accessed_last = tick;
			texture_cache.TouchImage(*image);
			continue;
		}
		texture.image_view = texture_cache.FindTexture(texture.image_id, texture.desc);
		auto& image        = texture_cache.GetImage(texture.image_id);
		view.state = ImageLookupState::Of(texture.image_id, image);
		view.valid = TableViewKeepable(view.state, image);
		element_changed(k);
	}
	if (prepared.tables_reused) {
		RebindReusedTables(prepared);
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo> colors) {
	bool                                     dma_write = false;
	std::array<const ShaderStageRuntime*, 4> dma_stages {};
	uint32_t                                 dma_stage_count = 0;
	bool                                     uses_image_heap = false;
	FindBuffers(stages, std::ranges::any_of(stages, [](const auto* stage) {
		            return stage->runtime->module->info.info.uses_dma;
	            }));
	for (auto* stage: stages) {
		if (stage->runtime->module->info.info.uses_dma) {
			m_context.CacheDmaBases(*stage->runtime);
			dma_write |= stage->runtime->module->info.has_address_writes;
			EXIT_IF(dma_stage_count == dma_stages.size());
			dma_stages[dma_stage_count++] = stage->runtime;
		}
		uses_image_heap |= stage->runtime->module->info.info.uses_image_heap;
	}
	if (dma_stage_count != 0) {
		const bool reach_synchronized =
		    m_context.SynchronizeDmaFootprint(std::span {dma_stages.data(), dma_stage_count});
		m_context.PrepareBda(dma_write, !reach_synchronized);
	}
	if (uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	std::array<bool, 4> rebind {};
	EXIT_IF(stages.size() > rebind.size());
	for (bool settled = false; !settled;) {
		settled = true;
		for (size_t i = 0; i < stages.size(); i++) {
			if (stages[i]->template_stage != nullptr && !CheckTemplateImages(*stages[i])) {
				rebind[i] = true;
				settled   = false;
			}
		}
	}
	for (size_t i = 0; i < stages.size(); i++) {
		if (stages[i]->template_stage == nullptr || rebind[i]) {
			stages[i]->template_stage = nullptr;
			RebindImages(*stages[i]);
		}
	}
	auto& cache = m_context.GetTextureCache();
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || !old_image->registered ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			InvalidateRenderTargetMemo();
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	if (DrawBuffersEnabled()) {
		StreamDrawReads(stages);
	}
	m_draw_buffer_view_count = 0;
	m_draw_buffer_views_on   = DrawStateTemplatesEnabled();
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
	m_draw_buffer_views_on   = false;
	m_draw_stream_read_count = 0;
}

void RenderExecutor::StreamDrawReads(std::span<PreparedBindings* const> stages) {
	constexpr uint64_t MaxRange = 16384;
	constexpr uint64_t MaxGap   = 4096;
	constexpr uint64_t MaxSpan  = 65536;
	m_draw_stream_read_count    = 0;
	std::array<std::pair<uint64_t, uint64_t>, 8> written {};
	uint32_t                                     written_count = 0;
	auto&                                        queries       = m_context.GetOcclusionQueries();
	auto&                                        cache         = m_context.GetBufferCache();
	for (const auto* stage: stages) {
		const auto& program = stage->runtime->module->info;
		for (const auto& resource: program.info.buffers) {
			if (resource.descriptor_index == UINT32_MAX) continue;
			const auto& source = stage->buffer_sources[resource.descriptor_index];
			if (source.address == 0 || source.size == 0) continue;
			if (resource.written) {
				if (written_count == written.size()) {
					m_draw_stream_read_count = 0;
					return;
				}
				written[written_count++] = {source.address, source.address + source.size};
				continue;
			}
			if (source.size > MaxRange || m_draw_stream_read_count == m_draw_stream_reads.size() ||
			    resource.formatted ||
			    queries.MayPrepareRead(source.address, source.size) ||
			    !cache.IsRegionCpuModified(source.address, source.size)) {
				continue;
			}
			const auto same = [&](const DrawStreamRead& read) {
				return read.address == source.address && read.size == source.size;
			};
			if (std::none_of(m_draw_stream_reads.begin(), m_draw_stream_reads.begin() + m_draw_stream_read_count, same)) {
				m_draw_stream_reads[m_draw_stream_read_count++] = {source.address, source.size, UINT64_MAX};
			}
		}
	}
	uint32_t copied = 0;
	uint32_t count  = 0;
	for (uint32_t i = 0; i < m_draw_stream_read_count; i++) {
		auto       read     = m_draw_stream_reads[i];
		const bool overlaps = std::any_of(written.begin(), written.begin() + written_count, [&](const auto& range) {
			return read.address < range.second && range.first < read.address + read.size;
		});
		if (overlaps) continue;
		if (const auto offset = cache.FindStreamCopy(read.address, read.size)) {
			read.offset                   = *offset;
			m_draw_stream_reads[count++]  = m_draw_stream_reads[copied];
			m_draw_stream_reads[copied++] = read;
		} else {
			m_draw_stream_reads[count++] = read;
		}
	}
	std::sort(m_draw_stream_reads.begin() + copied, m_draw_stream_reads.begin() + count,
	          [](const DrawStreamRead& a, const DrawStreamRead& b) { return a.address < b.address; });
	m_draw_stream_read_count = count;
	for (uint32_t first = copied; first < count;) {
		uint64_t begin = m_draw_stream_reads[first].address;
		uint64_t end   = begin + m_draw_stream_reads[first].size;
		uint32_t last  = first + 1;
		for (; last < count; last++) {
			const auto& read = m_draw_stream_reads[last];
			const auto  next = std::max(end, read.address + read.size);
			if (read.address > end + MaxGap || next - begin > MaxSpan) break;
			end = next;
		}
		if (const auto offset = cache.StreamGuestRange(begin, end - begin)) {
			for (uint32_t i = first; i < last; i++) {
				m_draw_stream_reads[i].offset = *offset + (m_draw_stream_reads[i].address - begin);
				cache.NoteStreamCopy(m_draw_stream_reads[i].address, m_draw_stream_reads[i].size,
				                     m_draw_stream_reads[i].offset);
			}
		}
		first = last;
	}
}

RenderExecutor::TableBindingState::RingSet* RenderExecutor::AcquireTableRingSet(TableBindingState&      state,
                                                                               vk::DescriptorSetLayout layout) {
	constexpr size_t MaxRingSets = 6;
	auto&            semaphore   = m_context.GetCommandScheduler().GetMasterSemaphore();
	for (int pass = 0; pass < 2; pass++) {
		for (auto& entry: state.ring) {
			if (entry.layout == layout && semaphore.IsFree(entry.tick)) {
				return &entry;
			}
		}
		if (pass == 0) {
			semaphore.Refresh();
		}
	}
	if (state.ring.size() >= MaxRingSets) {
		return nullptr;
	}
	auto& entry  = state.ring.emplace_back();
	entry.set    = m_context.GetDescriptorHeap().CommitPersistent(layout);
	entry.layout = layout;
	entry.version = 0;
	return &entry;
}

bool RenderExecutor::ProveTableSlotsLive(const TableBindingState& state) {
	auto& texture_cache = m_context.GetTextureCache();
	EXIT_IF(state.infos.size() != state.elements.size());
	MarkLiveTableElements(
	    state.infos,
	    [&](size_t index) -> const std::vector<CachedImageView>* {
		    const auto* image = texture_cache.m_slot_images.try_get(state.elements[index].pinned);
		    return image != nullptr ? &image->views : nullptr;
	    },
	    m_table_element_live);
	return MarkLiveTableSlots(state.slots, state.slot_infos, state.infos, m_table_element_live,
	                          m_table_slot_live);
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	auto                           vk_buffer              = buffer.Handle();
	size_t                         image_descriptor_count = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	for (const auto* prepared: prepared_bindings) {
		const auto& program = prepared->runtime->module->info;
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		EXIT_IF(!(pipeline.push_stages & NativeShaderStage(program.stage)));
	}
	if (pipeline.uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, pipeline_bind_point, pipeline.pipeline_layout);
	}
	if (m_descriptor_images.size() < image_descriptor_count) {
		m_descriptor_images.resize(image_descriptor_count);
	}
	m_descriptor_writes.clear();
	m_descriptor_copies.clear();
	m_slot_set_writes.clear();
	m_table_fills.clear();
	m_table_writes.clear();
	const bool fill_writes = TableDescriptorWrites(m_context.GetGraphics().driver_id);
	m_commit_capture_ok = m_commit_capture != nullptr && pipeline.uses_push_descriptors;
	m_commit_capture_kinds.clear();
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;
	vk::DescriptorBufferInfo                mip_stats_view;
	struct TableRange {
		TableBindingState* state;
		BindingKind        kind;
		uint32_t           binding;
		uint32_t           base;
		uint32_t           capacity;
		uint32_t           first_slot;
		bool               copied;
	};
	std::vector<TableRange> table_ranges;

	for (auto* prepared: prepared_bindings) {
		const auto& program     = prepared->runtime->module->info;
		auto&       descriptors = *prepared;
		const auto  stage_index = static_cast<uint8_t>(
		    std::ranges::find(prepared_bindings, prepared) - prepared_bindings.begin());
		if (descriptors.gds.buffer != nullptr || descriptors.table_state != nullptr) {
			m_commit_capture_ok = false;
		}
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier       = MakeGdsDependency(descriptors.gds.buffer);
			const auto shader_stages = ShaderPipelineStages(NativeShaderStage(program.stage));
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		const auto transit = [&](TextureBinding& binding) {
			if (!binding.image_id) return;
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.binding.is_target) {
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
		};
		for (size_t i = 0; i < descriptors.images.size(); i++) {
			if (i >= 64u || ((descriptors.template_images >> i) & 1u) == 0) {
				transit(descriptors.images[i]);
			}
		}
		for (auto* binding: descriptors.table_images) {
			transit(*binding);
		}
		auto* table_state = static_cast<TableBindingState*>(descriptors.table_state);
		if (descriptors.tables_reused) {
			for (const auto index: descriptors.table_visits) {
				transit(table_state->elements[index].resolution->texture);
			}
		}

		const bool copy_tables = !pipeline.uses_push_descriptors && table_state != nullptr &&
		                         descriptors.tables_reused && table_state->slot_set_current;
		const auto stage_ranges = table_ranges.size();
		if (table_state != nullptr) {
			uint32_t first_slot = 0;
			ForEachImageTable(program.info.images, [&](uint32_t root_index, BindingKind kind) {
				const auto& root = program.info.images[root_index];
				const auto  binding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
				table_ranges.push_back({.state      = table_state,
				                        .kind       = kind,
				                        .binding    = binding,
				                        .base       = root.table_descriptor_base,
				                        .capacity   = root.table_capacity,
				                        .first_slot = first_slot,
				                        .copied     = copy_tables});
				first_slot += root.table_capacity;
			});
			if (!copy_tables) {
				table_state->slot_set_current = false;
				table_state->slot_log.clear();
				table_state->slot_log_floor = ++table_state->slot_version;
			}
		}
		const std::span<const TableRange> stage_tables =
		    std::span {table_ranges}.subspan(stage_ranges);

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(program.bindings, kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
				ImageTableBinding fill {.state = table_state, .whole = write};
				for (const auto& table: stage_tables) {
					if (copy_tables && table.kind == kind) {
						EXIT_IF(fill.count == fill.tables.size());
						fill.tables[fill.count++] = {.base       = table.base,
						                             .capacity   = table.capacity,
						                             .first_slot = table.first_slot};
					}
				}
				if (fill.count != 0) {
					const auto end = fill.tables.begin() + static_cast<std::ptrdiff_t>(fill.count);
					std::sort(fill.tables.begin(), end,
					          [](const auto& a, const auto& b) { return a.base < b.base; });
					fill.persistent       = table_state->slot_set;
					fill.slot_infos       = table_state->slot_infos;
					fill.whole.pImageInfo = m_descriptor_images.data() + image_start;
					AppendImageTableGaps(fill.whole, std::span {fill.tables.data(), fill.count},
					                     m_descriptor_writes);
					m_table_fills.push_back(fill);
					continue;
				}
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::UniformBuffers:
						EXIT_IF(descriptors.uniform_buffers.size() != write.descriptorCount);
						for (const auto& view: descriptors.uniform_buffers) {
							EXIT_IF(view.range == 0);
						}
						write.pBufferInfo = descriptors.uniform_buffers.data();
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
					case BindingKind::MipStats: {
						const auto& counters = m_context.GetLodStats().CounterBuffer();
						mip_stats_view       = {counters.Handle(), 0, counters.Size()};
						write.pBufferInfo    = &mip_stats_view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
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
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
#if defined(ANYPS5_KYTY)
					case BindingKind::PackedTargets:
						EXIT_IF(descriptors.packed_targets.size() != write.descriptorCount);
						write.pImageInfo    = descriptors.packed_targets.data();
						m_commit_capture_ok = false;
						break;
#endif
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
			if (m_commit_capture_ok) {
				m_commit_capture_kinds.emplace_back(stage_index, static_cast<uint8_t>(index));
			}
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			if (resource.heap) continue;
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}
		size_t table_element = 0;
		m_table_image_infos.clear();
		if (descriptors.tables_reused) {
			for (const auto index: descriptors.table_visits) {
				table_state->infos[index] =
				    MakeImageInfo(table_state->elements[index].resolution->texture, 0u);
			}
			const auto write_slot = [&](uint32_t slot, const vk::DescriptorImageInfo& info) {
				auto&      slot_info = table_state->slot_infos[slot];
				const bool changed   = slot_info.sampler != info.sampler ||
				                     slot_info.imageView != info.imageView ||
				                     slot_info.imageLayout != info.imageLayout;
				if (copy_tables && changed) {
					table_state->slot_log.emplace_back(++table_state->slot_version, slot);
					const auto table =
					    std::ranges::find_if(stage_tables, [&](const TableRange& range) {
						    return slot >= range.first_slot && slot - range.first_slot < range.capacity;
					    });
					EXIT_IF(table == stage_tables.end());
					vk::WriteDescriptorSet write {};
					write.dstSet          = table_state->slot_set;
					write.dstBinding      = table->binding;
					write.dstArrayElement = table->base + (slot - table->first_slot);
					write.descriptorCount = 1;
					write.descriptorType  = NativeDescriptorType(table->kind);
					write.pImageInfo      = &slot_info;
					m_slot_set_writes.push_back(write);
				}
				slot_info = info;
			};
			for (const auto index: descriptors.table_visits) {
				const auto& info = table_state->infos[index];
				for (auto k = table_state->element_slot_begin[index];
				     k < table_state->element_slot_begin[index + 1]; k++) {
					write_slot(table_state->element_slots[k], info);
				}
			}
			for (const auto slot: descriptors.table_moved_slots) {
				write_slot(slot, table_state->infos[table_state->slots[slot]]);
			}
		}
		ForEachImageTable(program.info.images, [&](uint32_t root_index, BindingKind kind) {
			const auto& root  = program.info.images[root_index];
			const auto  start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   root.table_descriptor_base;
			if (copy_tables) {
				table_element += root.table_capacity;
				return;
			}
			if (descriptors.tables_reused) {
				EXIT_IF(table_element + root.table_capacity > table_state->slot_infos.size());
				std::copy_n(table_state->slot_infos.begin() +
				                static_cast<std::ptrdiff_t>(table_element),
				            root.table_capacity,
				            m_descriptor_images.begin() + static_cast<std::ptrdiff_t>(start));
				table_element += root.table_capacity;
				return;
			}
			if (m_table_image_infos.empty()) {
				m_table_image_infos.reserve(descriptors.table_images.size());
				for (const auto* texture: descriptors.table_images) {
					m_table_image_infos.push_back(MakeImageInfo(*texture, 0u));
				}
			}
			EXIT_IF(table_element + root.table_capacity > descriptors.table_slots.size());
			for (uint32_t slot = 0; slot < root.table_capacity; slot++) {
				m_descriptor_images[start + slot] =
				    m_table_image_infos[descriptors.table_slots[table_element++]];
			}
		});
		if (table_state != nullptr && !descriptors.tables_reused) {
			FinishTableBindings(*table_state, descriptors);
		}

		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        sizeof(push_data), push_data.dwords.data());
	}

	if (m_commit_capture_ok && m_commit_capture_kinds.size() == m_descriptor_writes.size() &&
	    m_table_fills.empty() && m_slot_set_writes.empty()) {
		StoreTemplateCommit(*m_commit_capture, pipeline);
	}
	if (!m_descriptor_writes.empty()) {
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			auto&      heap   = m_context.GetDescriptorHeap();
			const auto device = m_context.GetGraphics().device;
			TableBindingState* ring_state = nullptr;
			bool               ring_ok    = !table_ranges.empty();
			for (const auto& table: table_ranges) {
				ring_ok    = ring_ok && table.copied && (ring_state == nullptr || ring_state == table.state);
				ring_state = table.state;
			}
			auto* ring = ring_ok ? AcquireTableRingSet(*ring_state, pipeline.descriptor_set_layout) : nullptr;
			const auto set = ring != nullptr ? ring->set : heap.Commit(pipeline.descriptor_set_layout);
			if (ring != nullptr && ring->version >= ring_state->slot_log_floor) {
				for (auto change = std::ranges::upper_bound(ring_state->slot_log, ring->version, {},
				                                            [](const auto& entry) { return entry.first; });
				     change != ring_state->slot_log.end(); ++change) {
					const auto slot  = change->second;
					const auto table = std::ranges::find_if(table_ranges, [&](const TableRange& range) {
						return slot >= range.first_slot && slot - range.first_slot < range.capacity;
					});
					EXIT_IF(table == table_ranges.end());
					vk::WriteDescriptorSet write {};
					write.dstBinding      = table->binding;
					write.dstArrayElement = table->base + (slot - table->first_slot);
					write.descriptorCount = 1;
					write.descriptorType  = NativeDescriptorType(table->kind);
					write.pImageInfo      = &ring_state->slot_infos[slot];
					m_descriptor_writes.push_back(write);
				}
			} else {
				AppendTableBindingFills(
				    m_table_fills, fill_writes,
				    [&](const void* state) {
					    return ProveTableSlotsLive(*static_cast<const TableBindingState*>(state))
					               ? std::span<const uint8_t> {}
					               : std::span<const uint8_t> {m_table_slot_live};
				    },
				    m_table_writes, m_descriptor_copies);
			}
			if (ring != nullptr) {
				ring->tick    = m_context.GetCommandScheduler().CurrentTick();
				ring->version = ring_state->slot_version;
			}
			for (const auto& table: table_ranges) {
				auto& log = table.state->slot_log;
				if (log.size() > 4096) {
					const auto keep             = log.end() - 2048;
					table.state->slot_log_floor = (keep - 1)->first;
					log.erase(log.begin(), keep);
				}
			}
			PatchTableSetUpdate(set, m_descriptor_writes, m_table_writes, m_descriptor_copies, m_slot_set_writes);
			m_table_writes.clear();
			device.updateDescriptorSets(static_cast<uint32_t>(m_descriptor_writes.size()),
			                            m_descriptor_writes.data(),
			                            static_cast<uint32_t>(m_descriptor_copies.size()),
			                            m_descriptor_copies.data());
			m_descriptor_copies.clear();
			for (const auto& table: table_ranges) {
				auto& state = *table.state;
				if (table.copied || !state.valid) {
					continue;
				}
				if (!state.slot_set) {
					state.slot_set = heap.CommitPersistent(pipeline.descriptor_set_layout);
				}
				AppendImageTableSeed(table.binding, NativeDescriptorType(table.kind),
				                     {.base = table.base, .capacity = table.capacity, .first_slot = table.first_slot},
				                     fill_writes, set, state.slot_set, state.slot_infos, m_table_writes,
				                     m_descriptor_copies);
				state.slot_set_current = true;
			}
			if (!m_table_writes.empty() || !m_descriptor_copies.empty()) {
				device.updateDescriptorSets(static_cast<uint32_t>(m_table_writes.size()),
				                            m_table_writes.data(),
				                            static_cast<uint32_t>(m_descriptor_copies.size()),
				                            m_descriptor_copies.data());
			}
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
}

}
