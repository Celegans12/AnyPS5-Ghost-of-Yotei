#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<TextureBinding>           images;
	std::vector<vk::Sampler>              samplers;
	// The table sets (descriptors.cpp): the stage set used by this call (hit) or to complete with its views (store);
	// the image registry epoch after the hit proof.
	void*                                 table_hit    = nullptr;
	void*                                 table_store  = nullptr;
	uint64_t                              table_epoch  = 0;
	std::vector<uint64_t>                 image_source; // the table sets: per image slot, the set version it was filled from (0: none)
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	// Uniform const buffers (ShaderIR.h): the same uploaded SRT words as a uniform buffer view (ReadConst).
	vk::DescriptorBufferInfo              flattened_srt_uniform;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
	// KYTY_BINDLESS: flattened-SRT words to patch (offset, region, entries) and the resolved
	// bindless textures the draw may sample.
	struct BindlessPatch {
		uint32_t offset  = 0;
		uint32_t region  = 0;
		uint32_t entries = 0;
	};
	std::vector<BindlessPatch>            bindless_patches;
	std::vector<ImageId>                  bindless_images;
	// Record-keyed V#s: guest ranges the stage's record-keyed V#s can touch through BDA (from the snapshot).
	// The scalar-read check tests [address, address + size); its first `resident` bytes are committed memory, made
	// resident (and GPU-modified when written) before the draw or dispatch.
	struct IndirectRange {
		uint64_t address  = 0;
		uint64_t size     = 0;
		uint64_t resident = 0;
		bool     written  = false;
		BufferId id;
	};
	std::vector<IndirectRange>            indirect_ranges;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

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
// KYTY_WORKAROUND_DEPTH_ENCODING_LENIENT (stage 2 pack 2, #15): for a depth T# IsSupportedDepthTextureEncoding refused, what
// differs from the encoding it accepts (descriptors.cpp has the rule, from RDNA2 ISA table 45).
enum class DepthEncodingFields : uint32_t {
	Ignored, // only fields the texture unit does not use for this T#: sampling reads what the accepted encoding reads
	Corner,  // as Ignored, but CORNER_SAMPLES = 1 (texel placement; not emulated)
	Other,   // anything else (other types, R128, BASE_ARRAY, compression with other metadata, reserved bits of words 1-3)
};
[[nodiscard]] DepthEncodingFields DepthEncodingFieldsVerdict(const ShaderTextureResource& descriptor, bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
