#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEMEMORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEMEMORY_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

class InPlaceMemory final {
public:
	struct Imported {
		vk::Buffer        buffer;
		vk::DeviceAddress address = 0;
	};

	InPlaceMemory() = default;
	KYTY_CLASS_NO_COPY(InPlaceMemory);

	[[nodiscard]] static bool Requested() noexcept;
	[[nodiscard]] static bool Live() noexcept { return s_live_buffers.load(std::memory_order_acquire) != 0; }
	static void               NoteLabelAtCompletion() noexcept { s_labels.fetch_add(1, std::memory_order_relaxed); }
	static void               NoteBoundedFlip(uint64_t waited_ns) noexcept {
		s_bounded_flips.fetch_add(1, std::memory_order_relaxed);
		s_bounded_ns.fetch_add(waited_ns, std::memory_order_relaxed);
	}
	static void NoteConversion() noexcept { s_conversions.fetch_add(1, std::memory_order_relaxed); }
	static void NoteGpuWrittenRefusal() noexcept { s_refused_gpu.fetch_add(1, std::memory_order_relaxed); }
	static void LogSummary(double seconds);

	[[nodiscard]] bool                    CanImport(GraphicContext& graphics, uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::optional<Imported> Import(GraphicContext& graphics, uint64_t vaddr, uint64_t size,
	                                             vk::BufferUsageFlags usage);
	void                                  Register(uint64_t bytes, uint64_t pages) noexcept;
	void                                  Unregister(uint64_t bytes, uint64_t pages) noexcept;

	enum class WriteKind : uint8_t { Binding, WriteData, DmaData, TexelBuffer, Count };

	struct WriteCover {
		uint64_t tick  = 0;
		uint32_t kinds = 0;
		bool     exact = false;
	};

	void                   NoteGpuWrite(uint64_t vaddr, uint64_t size, uint64_t tick, uint64_t known = 0,
	                                    WriteKind kind = WriteKind::Binding);
	[[nodiscard]] WriteCover PendingGpuWriteCover(uint64_t vaddr, uint64_t size, uint64_t known) const;
	[[nodiscard]] uint64_t PendingGpuWrite(uint64_t vaddr, uint64_t size, uint64_t known) const;
	[[nodiscard]] uint64_t PendingGpuWriteBytes(uint64_t vaddr, uint64_t size, uint64_t known) const;
	static void            NoteGpuWriteOverPending() noexcept { s_writes_over_pending.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] static uint64_t TakeGpuWritesOverPending() noexcept {
		return s_writes_over_pending.exchange(0, std::memory_order_relaxed);
	}
	static void            NoteReadWait() noexcept { s_read_waits.fetch_add(1, std::memory_order_relaxed); }

	enum class LabelMeet : uint8_t { None, Pending, WouldBePending };

	[[nodiscard]] static uint64_t  NoteDeferredLabel(uint64_t vaddr, uint64_t size, uint64_t expires = 0);
	static void                    ForgetDeferredLabel(uint64_t id);
	[[nodiscard]] static LabelMeet MeetsDeferredLabel(uint64_t vaddr, uint64_t size, uint64_t natural_tick,
	                                                  bool settle = false);
	static void                    NoteLabelVerifyDrain() noexcept { s_label_verify_drains.fetch_add(1, std::memory_order_relaxed); }
	[[nodiscard]] static uint64_t  LabelVerifyDrains() noexcept { return s_label_verify_drains.load(std::memory_order_relaxed); }
	static void                    NoteGpuWriteOverLabel(uint64_t vaddr, uint64_t size);
	[[nodiscard]] static uint64_t  TakeGpuWritesOverLabels() noexcept {
		return s_writes_over_labels.exchange(0, std::memory_order_relaxed);
	}

	void                   RequestConversion(uint64_t page) noexcept;
	template <typename Func>
	void TakeConversions(Func&& func) {
		if (!m_requested.exchange(false, std::memory_order_acquire)) {
			return;
		}
		for (auto& slot: m_requests) {
			if (const auto page = slot.exchange(0, std::memory_order_relaxed); page != 0) {
				func(page);
			}
		}
	}

private:
	struct Chunk {
		VkDeviceMemory memory = VK_NULL_HANDLE;
		uint32_t       type   = 0;
		bool           failed = false;
	};

	enum class Refusal : uint8_t { None, Unbacked, GuestFile, Straddle, Chunk };

	struct DeferredLabel {
		uint64_t id      = 0;
		uint64_t begin   = 0;
		uint64_t end     = 0;
		uint64_t expires = 0;
	};

	struct WrittenRange {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
		WriteKind kind = WriteKind::Binding;
	};

	bool    Initialize(GraphicContext& graphics);
	Refusal Locate(GraphicContext& graphics, uint64_t vaddr, uint64_t size, uint64_t* offset, Chunk** chunk,
	               uint64_t* chunk_begin, uint64_t* chunk_end);
	void    MakeChunk(GraphicContext& graphics, Chunk& chunk, uint64_t begin, uint64_t bytes);
	void    Count(Refusal refusal) noexcept;

	static constexpr unsigned BlockBits = 16;
	static constexpr size_t   Requests  = 64;

	bool                                 m_initialized = false;
	bool                                 m_ready       = false;
	const uint8_t*                       m_view        = nullptr;
	uint64_t                             m_guest_bytes = 0;
	uint64_t                             m_view_bytes  = 0;
	uint64_t                             m_chunk_bytes = 0;
	uint32_t                             m_type_bits   = 0;
	std::unordered_map<uint64_t, Chunk>  m_chunks;

	mutable std::mutex                       m_writes_mutex;
	std::atomic<uint64_t>                    m_writes_max {0};
	mutable std::unordered_map<uint64_t, uint64_t> m_writes;
	mutable std::vector<WrittenRange>              m_written_ranges;

	std::array<std::atomic<uint64_t>, Requests> m_requests {};
	std::atomic<uint32_t>                       m_request_head {0};
	std::atomic<bool>                           m_requested {false};

	inline static std::atomic<uint64_t> s_live_buffers {0};
	inline static std::atomic<uint64_t> s_live_bytes {0};
	inline static std::atomic<uint64_t> s_imported {0};
	inline static std::atomic<uint64_t> s_imported_bytes {0};
	inline static std::atomic<uint64_t> s_refused_unbacked {0};
	inline static std::atomic<uint64_t> s_refused_guest {0};
	inline static std::atomic<uint64_t> s_refused_straddle {0};
	inline static std::atomic<uint64_t> s_refused_chunk {0};
	inline static std::atomic<uint64_t> s_refused_bind {0};
	inline static std::atomic<uint64_t> s_refused_gpu {0};
	inline static std::atomic<uint64_t> s_chunks {0};
	inline static std::atomic<uint64_t> s_pinned_bytes {0};
	inline static std::atomic<uint64_t> s_pages_in {0};
	inline static std::atomic<uint64_t> s_pages_out {0};
	inline static std::atomic<uint64_t> s_gpu_writes {0};
	inline static std::atomic<uint64_t> s_read_waits {0};
	inline static std::atomic<uint64_t> s_writes_over_pending {0};
	inline static std::atomic<uint64_t> s_writes_over_labels {0};
	inline static std::atomic<uint64_t> s_label_verify_drains {0};
	inline static std::atomic<uint64_t> s_deferred_label_count {0};
	inline static std::mutex            s_deferred_labels_mutex;
	inline static std::vector<DeferredLabel> s_deferred_labels;
	inline static uint64_t              s_next_deferred_label = 0;
	inline static std::atomic<uint64_t> s_labels {0};
	inline static std::atomic<uint64_t> s_bounded_flips {0};
	inline static std::atomic<uint64_t> s_bounded_ns {0};
	inline static std::atomic<uint64_t> s_conversions {0};
};

}

#endif
