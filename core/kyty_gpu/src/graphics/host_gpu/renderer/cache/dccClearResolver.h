#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCCLEARRESOLVER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCCLEARRESOLVER_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <atomic>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

// KYTY_FIX_DCC_GPU_CLEAR_CHECK (port of BryanKAdams 8db16e32 + c41cebee).
// Decides on the GPU whether GPU-written DCC metadata slices hold a fast clear. The check writes
// one predicate per (slice, clear code); callers wrap each candidate clear in conditional
// rendering, so the CPU never waits for the metadata bytes.
class DccClearResolver {
public:
	static constexpr uint32_t CodeCount = 5;
	static constexpr uint32_t MaxSlices = 64;

	// verify: KYTY_PERF_DCC_GPU_ALL=2, the checks can be copied aside for a comparison (BeginVerify / EndVerify).
	DccClearResolver(GraphicContext& graphics, CommandScheduler& scheduler, bool verify = false);
	~DccClearResolver();
	KYTY_CLASS_NO_COPY(DccClearResolver);

	[[nodiscard]] bool Available() const noexcept { return m_pipeline != nullptr; }

	// Code index used by the predicate layout, or CodeCount for a code that never clears.
	[[nodiscard]] static uint32_t CodeIndex(uint8_t code) noexcept;
	[[nodiscard]] static uint8_t  Code(uint32_t index) noexcept;

	// Records the check of `slices` consecutive slices of `slice_size` bytes starting at
	// `metadata_offset` in `metadata`. `code_mask` selects the decodable codes; `consume`
	// rewrites cleared slices to 0xff. Must be recorded outside a rendering instance. Returns false
	// (nothing recorded) when the slices cannot be bound as one storage buffer range.
	[[nodiscard]] bool Record(vk::CommandBuffer command, const Buffer& metadata, uint64_t metadata_offset,
	            uint64_t slice_size, uint32_t slices, uint32_t code_mask, bool consume);
	// Whether Record would record these slices (it records nothing and returns false otherwise).
	[[nodiscard]] bool CanRecord(const Buffer& metadata, uint64_t metadata_offset, uint64_t slice_size,
	                             uint32_t slices) const;

	// KYTY_PERF_DCC_GPU_ALL=2 (verify). BeginVerify, recorded right before Record, copies the keys the check is about
	// to read (before it consumes them) into a host-visible ring; EndVerify, recorded right after Record, copies the
	// predicates it wrote next to them. Once the GPU finished, VerifyKeys / VerifyPredicates read them and
	// ReleaseVerify(slot.end) frees the ring up to that point (in allocation order). slot.ok false: the ring is full,
	// nothing was recorded.
	struct VerifySlot {
		bool     ok                = false;
		uint64_t keys_offset       = 0;
		uint64_t predicates_offset = 0;
		uint64_t end               = 0;
	};
	[[nodiscard]] VerifySlot BeginVerify(vk::CommandBuffer command, const Buffer& metadata, uint64_t metadata_offset,
	                                     uint64_t slice_size, uint32_t slices);
	void                     EndVerify(vk::CommandBuffer command, const VerifySlot& slot, uint32_t slices);
	[[nodiscard]] const uint8_t*  VerifyKeys(const VerifySlot& slot, uint64_t size);
	[[nodiscard]] const uint32_t* VerifyPredicates(const VerifySlot& slot, uint32_t slices);
	void                          ReleaseVerify(uint64_t end);

	[[nodiscard]] vk::Buffer PredicateBuffer() const noexcept;
	[[nodiscard]] static uint64_t PredicateOffset(uint32_t slice, uint32_t code_index) noexcept {
		return (static_cast<uint64_t>(slice) * CodeCount + code_index) * sizeof(uint32_t);
	}

private:
	GraphicContext&         m_graphics;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
	std::unique_ptr<Buffer> m_predicates;
	// KYTY_PERF_DCC_GPU_ALL=2: the verify ring (GPU thread allocates at m_verify_head; the comparisons release
	// m_verify_tail in the same order). Positions grow without wrapping; offsets are positions modulo its size.
	std::unique_ptr<Buffer> m_verify_ring;
	uint64_t                m_verify_head = 0;
	std::atomic<uint64_t>   m_verify_tail {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCCLEARRESOLVER_H_
