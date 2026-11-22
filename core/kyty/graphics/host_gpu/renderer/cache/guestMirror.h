#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTMIRROR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GUESTMIRROR_H_

#include "common/common.h"
#include "common/threadSafety.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

class GuestMirror {
public:
	static constexpr uint64_t GiB             = uint64_t {1} << 30u;
	static constexpr uint64_t LargestSize     = 864 * GiB;
	static constexpr uint64_t Headroom        = 96 * GiB;
	static constexpr uint32_t MinBlockBits    = 16;
	static constexpr uint64_t ChunkBytes      = uint64_t {32} << 20u;
	static constexpr size_t   MaxBindsPerCall = 256;
	static constexpr uint64_t FlushBlocks     = 4096;
	static constexpr uint64_t FlushNs         = 1'000'000'000;

	static constexpr std::array<uint64_t, 10> DefaultSizes {864 * GiB, 832 * GiB, 800 * GiB, 768 * GiB, 704 * GiB,
	                                                        640 * GiB, 512 * GiB, 256 * GiB, 128 * GiB, 64 * GiB};

	struct ProbeResult {
		uint64_t    size = 0;
		double      ms   = 0;
		std::string trials;
		std::string refusal;
	};
	[[nodiscard]] static ProbeResult Probe(const GraphicContext& graphics, std::span<const uint64_t> sizes);
	[[nodiscard]] static std::string Unsupported(const GraphicContext& graphics);

	struct Stats {
		uint64_t bound_blocks   = 0;
		uint64_t chunks         = 0;
		uint64_t calls          = 0;
		uint64_t blocks         = 0;
		uint64_t unbound_blocks = 0;
		uint64_t call_ns        = 0;
		uint64_t done_ns        = 0;
		uint64_t longest_ns     = 0;
	};

	GuestMirror(GraphicContext& graphics, std::span<const uint64_t> sizes = DefaultSizes);
	~GuestMirror();
	KYTY_CLASS_NO_COPY(GuestMirror);

	[[nodiscard]] bool               Available() const noexcept { return m_size != 0; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	[[nodiscard]] uint64_t           BlockSize() const noexcept { return uint64_t {1} << m_block_bits; }
	[[nodiscard]] uint64_t           DeviceAddress() const noexcept { return m_device_address; }
	[[nodiscard]] const std::string& Description() const noexcept { return m_description; }
	[[nodiscard]] vk::Buffer         Handle();
	[[nodiscard]] bool               IsBound(uint64_t vaddr) const noexcept;
	[[nodiscard]] uint64_t           OutsideBlocks() const noexcept { return m_outside_blocks; }

	void NoteRange(uint64_t vaddr, uint64_t size);
	void Submit();
	void Flush();
	void Trim(std::span<const std::pair<uint64_t, uint64_t>> live);
	void WaitIdle();
	[[nodiscard]] Stats TakeStats();
	[[nodiscard]] Stats Totals() const;

private:
	struct Run {
		uint64_t first = 0;
		uint64_t count = 0;
	};
	enum class Kind : uint8_t { Bind, Unbind };
	struct Command {
		Kind             kind = Kind::Bind;
		std::vector<Run> runs;
	};
	struct Chunk {
		vk::DeviceMemory memory = nullptr;
		uint32_t         used   = 0;
		uint32_t         next   = 0;
	};
	struct Slot {
		uint32_t chunk = 0;
		uint32_t index = 0;
	};
	struct Grant {
		uint32_t chunk = 0;
		uint32_t index = 0;
		uint32_t count = 0;
	};

	[[nodiscard]] bool CreateBuffer();
	void               DestroyBuffer();
	void               BinderThread(std::stop_token stop);
	void               Bind(const std::vector<Run>& runs);
	void               Unbind(const std::vector<Run>& runs);
	void               Execute(std::vector<vk::SparseMemoryBind>& binds, const char* what);
	[[nodiscard]] Grant Allocate(uint64_t wanted);
	void               Free(Slot slot);
	void               SetBound(uint64_t first, uint64_t count, bool bound) noexcept;
	void               Post(Kind kind, std::vector<Run>&& runs);

	GraphicContext& m_graphics;
	uint64_t        m_size        = 0;
	uint32_t        m_block_bits  = MinBlockBits;
	uint32_t        m_memory_type = 0;
	uint64_t        m_device_address = 0;
	std::string     m_description;
	bool            m_shared_queue = false;

	std::vector<uint64_t> m_bound;
	std::vector<Run>      m_pending;
	uint64_t              m_pending_blocks = 0;
	uint64_t              m_last_post      = 0;
	uint64_t              m_outside_blocks = 0;

	vk::Buffer                      m_buffer = nullptr;
	vk::Fence                       m_fence  = nullptr;
	std::vector<Chunk>                     m_chunks;
	uint32_t                               m_current_chunk = UINT32_MAX;
	std::vector<Slot>                      m_free_slots;
	std::vector<uint32_t>                  m_free_chunks;
	std::unordered_map<uint64_t, Slot>     m_block_slots;
	std::vector<vk::SparseMemoryBind>      m_binds;

	Common::AnnotatedMutex      m_mutex;
	std::condition_variable_any m_changed;
	std::deque<Command>         m_commands KYTY_GUARDED_BY(m_mutex);
	bool                        m_busy KYTY_GUARDED_BY(m_mutex) = false;

	std::atomic<uint64_t> m_bound_blocks {0};
	std::atomic<uint64_t> m_chunk_count {0};
	std::atomic<uint64_t> m_calls {0};
	std::atomic<uint64_t> m_blocks {0};
	std::atomic<uint64_t> m_unbound_blocks {0};
	std::atomic<uint64_t> m_call_ns {0};
	std::atomic<uint64_t> m_done_ns {0};
	std::atomic<uint64_t> m_longest_ns {0};
	Stats                 m_taken;

	std::jthread m_binder;
};

}

#endif
