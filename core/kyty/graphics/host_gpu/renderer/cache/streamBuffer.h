#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;
struct StreamBufferTestAccess;
struct GraphicContext;

enum class MemoryUsage : uint8_t {
	DeviceLocal,
	Upload,
	Download,
	Stream,
};

inline constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

inline constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct MirrorWindow {
	vk::Buffer        buffer  = nullptr;
	vk::DeviceAddress address = 0;
};

class Buffer {
public:
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	       uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size);
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t cpu_address, uint64_t size,
	       vk::Buffer imported, vk::DeviceAddress device_address);
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t cpu_address, uint64_t size,
	       MirrorWindow mirror);
	~Buffer();
	KYTY_CLASS_NO_COPY(Buffer);

	[[nodiscard]] vk::Buffer         Handle() const noexcept { return m_buffer; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	[[nodiscard]] std::span<uint8_t> Mapped() const noexcept { return m_mapped; }
	[[nodiscard]] bool               IsCoherent() const noexcept { return m_coherent; }
#if defined(ANYPS5_KYTY)
	[[nodiscard]] bool               IsDeviceLocal() const noexcept { return m_device_local; }
#endif
	[[nodiscard]] MemoryUsage        Usage() const noexcept { return m_usage; }
	[[nodiscard]] bool               InPlace() const noexcept { return m_in_place; }
	[[nodiscard]] bool               IsWindow() const noexcept { return m_window; }
	[[nodiscard]] uint64_t           CpuAddress() const noexcept { return m_cpu_address; }
	[[nodiscard]] uint64_t           HandleOffset() const noexcept { return m_handle_offset; }
	[[nodiscard]] uint64_t           HandleOrigin() const noexcept { return m_cpu_address - m_handle_offset; }
	[[nodiscard]] vk::DeviceAddress  BufferDeviceAddress() const noexcept {
		EXIT_IF(m_device_address == 0);
		return m_device_address;
	}
	[[nodiscard]] uint64_t           Offset(uint64_t address) const noexcept {
		return address - m_cpu_address + m_handle_offset;
	}
	[[nodiscard]] bool HoldsOffsets(uint64_t offset, uint64_t size) const noexcept {
		return offset >= m_handle_offset && size <= Size() && offset - m_handle_offset <= Size() - size;
	}
	[[nodiscard]] uint64_t Capacity(uint64_t offset) const noexcept {
		return HoldsOffsets(offset, 0) ? Size() - (offset - m_handle_offset) : 0;
	}
	[[nodiscard]] bool IsInBounds(uint64_t address, uint64_t size) const noexcept {
		return address >= m_cpu_address && size <= Size() &&
		       address - m_cpu_address <= Size() - size;
	}
	void               IncreaseStreamScore(int score) noexcept { stream_score += score; }
	[[nodiscard]] int  StreamScore() const noexcept { return stream_score; }
	void               Flush(uint64_t offset, uint64_t size);
	void               Invalidate(uint64_t offset, uint64_t size);
	void CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
	              uint64_t destination_offset, uint64_t size);
	void Fill(uint64_t offset, uint64_t size, uint32_t value);

	bool   is_deleted   = false;
	int    stream_score = 0;
	size_t lru_id       = 0;
	bool device_address_pinned = false;
	bool in_place_refused      = false;

protected:
	[[nodiscard]] GraphicContext&   Graphics() const noexcept { return *m_graphics; }
	[[nodiscard]] CommandScheduler& Scheduler() const noexcept { return *m_scheduler; }

private:
	[[nodiscard]] vk::BufferMemoryBarrier Barrier(uint64_t offset, uint64_t size,
	                                              vk::AccessFlags source,
	                                              vk::AccessFlags destination) const;

	GraphicContext*               m_graphics    = nullptr;
	CommandScheduler*             m_scheduler   = nullptr;
	MemoryUsage                   m_usage       = MemoryUsage::DeviceLocal;
	uint64_t                      m_cpu_address = 0;
	uint64_t                      m_handle_offset = 0;
	vk::DeviceAddress             m_device_address = 0;
	vk::Buffer                    m_buffer     = nullptr;
	VmaAllocation                 m_allocation = nullptr;
	uint64_t                      m_size;
	bool                          m_coherent = false;
	bool                          m_in_place = false;
	bool                          m_window   = false;
#if defined(ANYPS5_KYTY)
	bool                          m_device_local = false;
#endif
	std::span<uint8_t>            m_mapped;
};

class StreamBuffer final: public Buffer {
public:
	struct Download {
		Buffer*                 buffer;
		uint8_t*                mapped;
		uint64_t                offset;
		std::unique_ptr<Buffer> owner;
	};

	StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	             uint64_t size);

	[[nodiscard]] std::pair<uint8_t*, uint64_t> Map(uint64_t size, uint64_t alignment = 0,
	                                                bool allow_wait = true);
	void                                        Commit();
	[[nodiscard]] Download                      MapDownload(uint64_t size, uint64_t alignment);
	[[nodiscard]] uint64_t Copy(const void* source, uint64_t size, uint64_t alignment = 0);
	[[nodiscard]] std::optional<uint64_t> TryCopy(const void* source, uint64_t size, uint64_t alignment = 0) {
		return Copy(source, size, alignment);
	}
	[[nodiscard]] uint64_t Lap() const noexcept { return m_lap; }

private:
	friend struct StreamBufferTestAccess;

	struct Watch {
		uint64_t tick        = 0;
		uint64_t upper_bound = 0;
	};

	[[nodiscard]] static bool NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
	                                               uint64_t& alignment);
	[[nodiscard]] bool        WaitPendingOperations(const std::vector<Watch>& watches,
	                                                std::optional<size_t>     invalidation_mark,
	                                                uint64_t requested_upper_bound, bool allow_wait,
	                                                size_t& wait_cursor, uint64_t& wait_bound);

	uint64_t              m_offset      = 0;
	uint64_t              m_mapped_size = 0;
	uint64_t              m_lap         = 0;
	std::vector<Watch>    m_current_watches;
	size_t                m_current_watch_cursor = 0;
	std::optional<size_t> m_invalidation_mark;
	std::vector<Watch>    m_previous_watches;
	size_t                m_wait_cursor = 0;
	uint64_t              m_wait_bound  = 0;
};

}

#endif
