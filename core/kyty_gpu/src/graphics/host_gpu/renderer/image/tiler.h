#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TILER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TILER_H_

#include "common/common.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <span>

namespace Libs::Graphics {

class CommandScheduler;
class Buffer;
class Image;
class StreamBuffer;
struct GraphicContext;
struct TileManagerTestAccess;

class TileManager final {
public:
	enum class D16Direction { Promote, Demote };

	struct Result {
		vk::Buffer buffer = nullptr;
		uint64_t   offset = 0;
		uint64_t   size   = 0;
	};
	struct D16Layout {
		uint32_t width               = 0;
		uint32_t height              = 0;
		uint32_t layers              = 0;
		uint64_t source_row_stride   = 0;
		uint64_t target_row_stride   = 0;
		uint64_t source_slice_stride = 0;
		uint64_t target_slice_stride = 0;
	};

	TileManager(GraphicContext& graphics, CommandScheduler& scheduler, StreamBuffer& stream_buffer);
	~TileManager();
	KYTY_CLASS_NO_COPY(TileManager);

	// Consume scratch results before the next acquisition, or pass their buffer as input.
	[[nodiscard]] Result Detile(vk::Buffer tiled, uint64_t tiled_offset, const ImageInfo& info);
	void Tile(vk::Buffer linear, uint64_t linear_offset, uint64_t linear_capacity, vk::Buffer tiled,
	          uint64_t tiled_offset, uint64_t tiled_capacity, const ImageInfo& info,
	          uint32_t levels = 0);
	void TileImage(Image& image, std::span<const vk::BufferImageCopy> regions, vk::Buffer tiled,
	               uint64_t tiled_offset, uint64_t tiled_capacity, uint32_t levels = 0);
	[[nodiscard]] Result GetScratchBuffer(uint64_t size, vk::Buffer input = nullptr);
	void                 ConvertD16(Result source, Result target, D16Direction direction, bool d32,
	                                const D16Layout& layout);
	[[nodiscard]] Result TransformColor(Result input, ColorTransform transform, bool to_host);
	void TransformColor(Result input, Result output, ColorTransform transform, bool to_host);
	// Color-clear keys are multiples of 0x20, so key >> 5 selects one of these slots.
	static constexpr uint32_t ColorClearSlots = 8;
	// metadata spans `slices` equal runs `stride` bytes apart. Returns the buffer whose dword
	// slice * ColorClearSlots + (key >> 5) is set when every byte of that run equals a key whose
	// bit `keys` selects; a hit expands to 0xff when `expand` is set.
	[[nodiscard]] vk::Buffer CheckColorClear(Result metadata, uint64_t stride, uint32_t slices,
	                                         uint32_t keys, bool expand);
	// KYTY_FIX_SHARED_IMAGE_MEMORY's direct path: the linear host copy of `target`'s level 0 takes, texel by texel, the
	// guest bytes the linear host copy of `source`'s level 0 holds at the same guest addresses (gpu_tiler_alias_copy).
	// Both images: tiled, 2D, one level and layer, level 0 outside the mip tail (AliasCopyFits).
	// KYTY_FIX_IMAGE_MEMORY_OWNER: one rectangle of level 0 of a tiled 2D image with one level and layer (level 0 outside
	// the mip tail, 1x1 texel elements): with `tile`, the rectangle's texels in `linear` (rows of width * bytes per
	// element) go to their guest offsets in `tiled`, whose byte at `tiled_offset` stands for guest address
	// `tiled_guest_address`; otherwise the reverse. x0 and y0 are multiples of the block's texel width and height.
	void TileRect(bool tile, const ImageInfo& info, uint32_t x0, uint32_t y0, uint32_t width, uint32_t height,
	              vk::Buffer linear, uint64_t linear_offset, uint64_t linear_capacity, vk::Buffer tiled,
	              uint64_t tiled_offset, uint64_t tiled_capacity, uint64_t tiled_guest_address);
	// KYTY_FIX_IMAGE_MEMORY_OWNER: whole guest-memory blocks of `source` into `target` in one pass (gpu_tiler_retile),
	// texel by texel through both in-block tables, on raw texel bits copied out of and back into the images. A job is
	// one block: the target's texel rectangle (origin and the size inside its extent) and position XOR value, the
	// source's likewise (t_base/s_base and the row pitches: set by Retile). Both images: tiled 2D, one level and layer,
	// the same block size, no colour transform (RetileFits); jobs at most 65535.
	struct RetileJob {
		uint32_t t_x0, t_y0, w, h, s_x0, s_y0, s_w, s_h, t_xor, s_xor, t_base, s_base, t_pitch, s_pitch;
	};
	[[nodiscard]] static bool RetileFits(const Image& target, const Image& source) noexcept;
	void Retile(Image& target, Image& source, std::span<const RetileJob> jobs);
	[[nodiscard]] static bool AliasCopyFits(const ImageInfo& info) noexcept;
	void AliasCopy(vk::Buffer target_linear, uint64_t target_size, const ImageInfo& target, vk::Buffer source_linear,
	               uint64_t source_size, const ImageInfo& source);

private:
	friend struct TileManagerTestAccess;

	static constexpr uint32_t FamilyCount          = static_cast<uint32_t>(TileBlockFamily::Count);
	static constexpr uint32_t BytesPerElementCount = 5;
	static constexpr uint32_t DirectionCount       = 2;
	static constexpr uint32_t PipelineCount = FamilyCount * BytesPerElementCount * DirectionCount;
	static constexpr uint32_t MaxMipLevels         = 16;

	struct alignas(16) Mip {
		uint32_t src_base;
		uint32_t dst_base;
		uint32_t width;
		uint32_t height;
		uint32_t pitch_bytes;
		uint32_t slice_bytes;
		uint32_t tiled_slice_bytes;
		uint32_t blocks_per_row;
		uint32_t surface_z;
		uint32_t tail_x;
		uint32_t tail_y;
		uint32_t num_texels;
	};
	struct TilingParams {
		uint32_t                      num_texels      = 0;
		uint32_t                      color_transform = 0;
		std::array<uint32_t, 2>        padding {};
		std::array<Mip, MaxMipLevels> mips;
	};
	struct Tiling {
		uint64_t params_offset;
		uint32_t num_texels;
		uint16_t params_size;
		uint16_t pipeline_slot;
	};
	struct Conversion {
		uint32_t src_base;
		uint32_t dst_base;
		uint32_t width;
		uint32_t height;
		uint32_t source_pitch;
		uint32_t target_pitch;
		uint32_t color_transform;
		uint32_t to_host;
	};
	struct StorageBinding {
		vk::DescriptorBufferInfo info;
		uint32_t                 base = 0;
	};

	[[nodiscard]] StorageBinding  BindStorage(Result buffer, uint64_t size) const;
	[[nodiscard]] static uint32_t ConversionRows(uint64_t offset, uint64_t row_stride,
	                                             uint64_t active, uint32_t remaining,
	                                             uint64_t alignment, uint64_t max_range,
	                                             uint32_t max_groups) noexcept;
	[[nodiscard]] Tiling Prepare(bool tile, uint64_t tiled_capacity, uint64_t linear_capacity,
	                             const ImageInfo& info, uint32_t levels, uint64_t source_base,
	                             uint64_t target_base);
	void                 Record(vk::Buffer source, uint64_t source_offset, uint64_t source_capacity,
	                            vk::Buffer target, uint64_t target_offset, uint64_t target_capacity,
	                            const Tiling& tiling, bool clear_target);
	[[nodiscard]] vk::Pipeline GetPipeline(uint32_t slot);
	[[nodiscard]] vk::Pipeline CreatePipeline(std::span<const uint32_t>     code,
	                                          const vk::SpecializationInfo* specialization) const;

	GraphicContext&                         m_graphics;
	CommandScheduler&                       m_scheduler;
	StreamBuffer&                           m_stream_buffer;
	std::array<std::unique_ptr<Buffer>, 2>   m_scratch;
	vk::DescriptorSetLayout                 m_descriptor_layout = nullptr;
	vk::PipelineLayout                      m_pipeline_layout   = nullptr;
	std::array<vk::Pipeline, PipelineCount> m_pipelines {};
	vk::Pipeline                            m_d16_to_d24      = nullptr;
	vk::Pipeline                            m_d16_to_d32      = nullptr;
	vk::Pipeline                            m_d24_to_d16      = nullptr;
	vk::Pipeline                            m_d32_to_d16      = nullptr;
	vk::Pipeline                            m_color_transform = nullptr;
	vk::Pipeline                            m_color_clear     = nullptr;
	// AliasCopy: its own layout (three storage buffers, 76 bytes of push constants) and the in-block tables, one pair per
	// (block family, element size) written once into a host-visible buffer.
	vk::DescriptorSetLayout                 m_alias_descriptor_layout = nullptr;
	vk::PipelineLayout                      m_alias_pipeline_layout   = nullptr;
	vk::Pipeline                            m_alias_copy              = nullptr;
	std::unique_ptr<Buffer>                 m_alias_tables;
	uint32_t                                m_alias_tables_used = 0; // words
	std::array<std::array<uint32_t, 2>, FamilyCount * BytesPerElementCount> m_alias_table_at {}; // +1; 0: none
	[[nodiscard]] std::array<uint32_t, 2> AliasTables(const TileBlockLayout& block);
	std::unique_ptr<Buffer>                 m_clear_predicates;
	// Retile: its own layout (four storage buffers: source and target rectangles, jobs, tables) and its linear buffers.
	vk::DescriptorSetLayout                 m_retile_descriptor_layout = nullptr;
	vk::PipelineLayout                      m_retile_pipeline_layout   = nullptr;
	vk::Pipeline                            m_retile                   = nullptr;
	std::array<std::unique_ptr<Buffer>, 2>   m_retile_linear;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TILER_H_
