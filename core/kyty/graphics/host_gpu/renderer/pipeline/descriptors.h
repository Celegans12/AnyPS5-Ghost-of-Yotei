#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureDescDerivation;
struct DrawTemplateStage;

struct ImageLookupState {
	ImageId  id;
	ImageId  depth_id;
	uint64_t track_addr      = 0;
	uint64_t track_addr_end  = 0;
	bool     cpu_dirty       = false;
	bool     buffer_modified = false;
	bool     gpu_modified    = false;
	bool     render_target   = false;

	bool operator==(const ImageLookupState&) const = default;

	[[nodiscard]] static ImageLookupState Of(ImageId id, const Image& image) {
		return {.id              = id,
		        .depth_id        = image.depth_id,
		        .track_addr      = image.track_addr,
		        .track_addr_end  = image.track_addr_end,
		        .cpu_dirty       = image.IsCpuDirty(),
		        .buffer_modified = image.IsBufferModified(),
		        .gpu_modified    = image.IsGpuModified(),
		        .render_target   = image.usage.render_target};
	}
};

struct TableViewMemo {
	ImageLookupState state;
	bool             valid = false;
};

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
	TextureDescDerivation* lookup = nullptr;
	bool                   reused = false;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	const ShaderStageRuntime* runtime = nullptr;
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<vk::DescriptorBufferInfo> uniform_buffers;
	std::vector<TextureBinding>           images;
	std::vector<TextureBinding*>                        table_images;
	std::vector<TableViewMemo*>                         table_views;
	std::vector<ShaderRecompiler::IR::DescriptorValue> table_sources;
	std::vector<uint32_t>                               table_roots;
	std::vector<uint32_t>                               table_versions;
	std::vector<uint32_t>                               table_slots;
	void*                                               table_state   = nullptr;
	bool                                                tables_reused = false;
	std::vector<uint32_t>                               table_visits;
	std::vector<uint32_t>                               table_moved_slots;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	bool                                  flattened_srt_known = false;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
#if defined(ANYPS5_KYTY)
	std::vector<vk::DescriptorImageInfo>  packed_targets;
#endif
	const void*                                        last_program          = nullptr;
	uint64_t                                           last_descs_generation = 0;
	std::vector<ShaderRecompiler::IR::DescriptorValue> last_images;
	std::vector<ShaderRecompiler::IR::DescriptorValue> last_samplers;
	std::vector<TextureDescDerivation*>                image_descs;
	const DrawTemplateStage* template_stage  = nullptr;
	uint64_t                 template_images = 0;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

struct ImageTableFill {
	uint32_t base       = 0;
	uint32_t capacity   = 0;
	uint32_t first_slot = 0;
};

[[nodiscard]] bool TableDescriptorWrites(vk::DriverId driver);

inline constexpr size_t DefaultTableResolutions = 256;
inline constexpr size_t MaxTableResolutions     = 4096;

struct TableResolutionSettings {
	size_t capacity = DefaultTableResolutions;
	bool   legacy   = false;
};

[[nodiscard]] TableResolutionSettings ReadTableResolutionSettings();

void AppendImageTableGaps(const vk::WriteDescriptorSet& whole, std::span<const ImageTableFill> tables,
                          std::vector<vk::WriteDescriptorSet>& gap_writes);
void AppendImageTableFills(const vk::WriteDescriptorSet& whole, std::span<const ImageTableFill> tables, bool writes,
                           vk::DescriptorSet persistent, std::span<const vk::DescriptorImageInfo> slot_infos,
                           std::span<const uint8_t> slot_live, std::vector<vk::WriteDescriptorSet>& table_writes,
                           std::vector<vk::CopyDescriptorSet>& table_copies);
void AppendImageTableSeed(uint32_t binding, vk::DescriptorType type, const ImageTableFill& table, bool writes,
                          vk::DescriptorSet set, vk::DescriptorSet persistent,
                          std::span<const vk::DescriptorImageInfo> slot_infos,
                          std::vector<vk::WriteDescriptorSet>&     seed_writes,
                          std::vector<vk::CopyDescriptorSet>&      seed_copies);
void PatchTableSetUpdate(vk::DescriptorSet set, std::vector<vk::WriteDescriptorSet>& writes,
                         std::span<const vk::WriteDescriptorSet> table_writes,
                         std::span<vk::CopyDescriptorSet>         copies,
                         std::span<const vk::WriteDescriptorSet> slot_set_writes);

struct ImageTableBinding {
	const void*                              state = nullptr;
	vk::DescriptorSet                        persistent;
	std::span<const vk::DescriptorImageInfo> slot_infos;
	vk::WriteDescriptorSet                   whole;
	std::array<ImageTableFill, 16>           tables {};
	uint32_t                                 count = 0;
};

template <typename TProve>
void AppendTableBindingFills(std::span<const ImageTableBinding> bindings, bool writes, TProve&& prove,
                             std::vector<vk::WriteDescriptorSet>& table_writes,
                             std::vector<vk::CopyDescriptorSet>&  table_copies) {
	const void*              proven = nullptr;
	std::span<const uint8_t> slot_live;
	for (const auto& binding: bindings) {
		if (writes && binding.state != proven) {
			proven    = binding.state;
			slot_live = prove(binding.state);
		}
		AppendImageTableFills(binding.whole, std::span {binding.tables.data(), binding.count}, writes,
		                      binding.persistent, binding.slot_infos, slot_live, table_writes, table_copies);
	}
}

template <typename TViewsOf>
void MarkLiveTableElements(std::span<const vk::DescriptorImageInfo> infos, TViewsOf&& views_of,
                           std::vector<uint8_t>& element_live) {
	element_live.resize(infos.size());
	for (size_t index = 0; index < infos.size(); index++) {
		const auto view = infos[index].imageView;
		if (view == nullptr) {
			element_live[index] = 1;
			continue;
		}
		const std::vector<CachedImageView>* views = views_of(index);
		element_live[index] =
		    views != nullptr && std::ranges::find(*views, view, &CachedImageView::view) != views->end() ? 1 : 0;
	}
}

[[nodiscard]] bool MarkLiveTableSlots(std::span<const uint32_t>                slots,
                                      std::span<const vk::DescriptorImageInfo> slot_infos,
                                      std::span<const vk::DescriptorImageInfo> infos,
                                      std::span<const uint8_t> element_live, std::vector<uint8_t>& slot_live);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

}

#endif
