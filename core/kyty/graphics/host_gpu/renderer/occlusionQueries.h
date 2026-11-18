#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_OCCLUSIONQUERIES_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

class Buffer;
class CommandRecorder;
class CommandScheduler;
class RenderContext;

namespace OcclusionCounters {

constexpr uint32_t DbCount  = 16;
constexpr uint64_t DbStride = 16;
constexpr uint64_t DumpSize  = (DbCount - 1u) * DbStride + sizeof(uint64_t);
constexpr uint64_t ReadyBit  = uint64_t {1} << 63u;
constexpr uint64_t ValueMask = ReadyBit - 1u;
constexpr uint64_t UncountedSamples = uint64_t {1} << 24u;

constexpr float ProxyBiasUnits = 64.0f;
constexpr float ProxyBiasSlope = 1.0f;

[[nodiscard]] constexpr uint64_t DbValue(uint32_t db, uint64_t total) {
	return ReadyBit | (db == 0 ? (total & ValueMask) : 0u);
}

class Segments {
public:
	struct Dump {
		uint64_t address    = 0;
		uint64_t first_slot = 0;
		uint64_t end_slot   = 0;
		bool     uncounted  = false;
	};

	explicit Segments(uint32_t capacity, uint32_t max_open_frames = 8)
	    : m_capacity(capacity), m_max_open_frames(max_open_frames) {}

	[[nodiscard]] Dump OnDump(uint64_t address) {
		Dump dump {.address    = address,
		           .first_slot = m_segment_first,
		           .end_slot   = m_next_slot,
		           .uncounted  = m_uncounted};
		m_segment_first  = m_next_slot;
		m_uncounted      = false;
		const auto begin = std::find_if(m_open.begin(), m_open.end(), [&](const Open& open) {
			return open.address + sizeof(uint64_t) == address;
		});
		if (begin != m_open.end()) {
			m_open.erase(begin);
		} else {
			m_open.push_back({address, m_frame});
		}
		return dump;
	}

	[[nodiscard]] bool Counting() const noexcept { return !m_open.empty(); }

	[[nodiscard]] bool AllocateSlot(uint32_t& slot) {
		if (m_next_slot - m_released >= m_capacity) {
			m_uncounted = true;
			return false;
		}
		slot = static_cast<uint32_t>(m_next_slot % m_capacity);
		m_next_slot++;
		return true;
	}

	void Release(uint64_t end_slot) { m_released = std::max(m_released, end_slot); }

	void OnFrame() {
		m_frame++;
		std::erase_if(
		    m_open, [this](const Open& open) { return open.frame + m_max_open_frames < m_frame; });
	}

	[[nodiscard]] uint32_t Capacity() const noexcept { return m_capacity; }
	[[nodiscard]] size_t   OpenQueries() const noexcept { return m_open.size(); }

private:
	struct Open {
		uint64_t address = 0;
		uint64_t frame   = 0;
	};

	uint32_t          m_capacity        = 0;
	uint32_t          m_max_open_frames = 0;
	uint64_t          m_next_slot       = 0;
	uint64_t          m_released        = 0;
	uint64_t          m_segment_first   = 0;
	uint64_t          m_frame           = 0;
	bool              m_uncounted       = false;
	std::vector<Open> m_open;
};

}

class OcclusionQueries {
public:
	explicit OcclusionQueries(RenderContext& context);
	~OcclusionQueries();
	KYTY_CLASS_NO_COPY(OcclusionQueries);

	[[nodiscard]] bool Enabled() const noexcept { return m_pool != nullptr; }

	void Dump(CommandScheduler& scheduler, uint64_t address);
	void OnBeginRendering(const CommandRecorder& recorder);
	void OnEndRendering(const CommandRecorder& recorder);
	void OnFrame();
	void PublishCompleted(bool refresh);
	void OnUnmap(uint64_t address, uint64_t size);
	[[nodiscard]] bool HasPendingDumps() const {
		std::lock_guard lock(m_mutex);
		return !m_pending.empty();
	}
#if defined(ANYPS5_KYTY)
	[[nodiscard]] uint64_t LastPendingDumpTick() const {
		std::lock_guard lock(m_mutex);
		return m_pending.empty() ? 0 : m_pending.back().tick;
	}
#endif
	[[nodiscard]] bool Counting() const noexcept { return Enabled() && m_segments.Counting(); }

	[[nodiscard]] bool PrepareRead(uint64_t address, uint64_t size);
	[[nodiscard]] bool MayPrepareRead(uint64_t address, uint64_t size) const;
	void               NoteReadTarget(uint64_t address, uint64_t size, const Buffer& buffer,
	                                  uint64_t offset);
	void               WriteForReaders();
	void DropReadTargets() noexcept {
		m_read_targets.clear();
		m_read_dumps.clear();
	}
	void WaitForCounters(uint64_t address, uint64_t size);

private:
	void BeginQuery(const CommandRecorder& recorder);
	void EndQuery(const CommandRecorder& recorder);
	struct PendingDump {
		OcclusionCounters::Segments::Dump dump;
		uint64_t                          tick  = 0;
		uint64_t                          frame = 0;
		uint64_t                          seq   = 0;
		bool                              write = true;
	};
	struct PendingPage {
		uint64_t page     = UINT64_MAX;
		uint64_t last_seq = 0;
		uint64_t chunks   = 0;
	};
	static constexpr size_t PendingPageCount = 4096;
	struct ReadDump {
		uint64_t address = 0;
		uint64_t seq     = 0;
	};
	struct ReadTarget {
		uint64_t address        = 0;
		uint64_t size           = 0;
		uint64_t device_address = 0;
		size_t   first_dump     = 0;
		size_t   end_dump       = 0;
	};

	void                   PublishFront(size_t count);
	[[nodiscard]] bool     ReadResults(uint64_t first_slot, uint64_t end_slot);
	void                   WriteCounters(uint64_t address);
	void                   ResetSlots(uint64_t first_slot, uint64_t end_slot);
	void                   ReleaseSlots(uint64_t end_slot);
	void                   NotePendingPages(uint64_t address, uint64_t seq);
	[[nodiscard]] bool     MayReadPendingDump(uint64_t address, uint64_t size) const;
	void                   WaitAndPublish(uint64_t tick);
	void                   CreateCounterPipeline();

	RenderContext&              m_context;
	vk::Device                  m_device = nullptr;
	vk::QueryPool               m_pool   = nullptr;
	vk::QueryControlFlags       m_control;
	bool                        m_rendering    = false;
	bool                        m_query_active = false;
	uint32_t                    m_active_slot  = 0;
	uint64_t                    m_frame        = 0;
	mutable std::mutex          m_mutex;
	OcclusionCounters::Segments m_segments;
	uint64_t                    m_total = 0;
	uint64_t                    m_completion_tick = 0;
	std::deque<PendingDump>     m_pending;
	std::vector<uint64_t>       m_results;
	struct SlotHold {
		uint64_t tick = 0;
		uint64_t slot = 0;
	};
	uint64_t                    m_published_slot = 0;
	uint64_t                    m_reset_slot     = 0;
	std::deque<SlotHold>        m_holds;
	uint64_t                    m_dump_seq = 0;
	std::atomic<uint64_t>       m_published_seq {0};
	std::vector<PendingPage>    m_pending_pages;
	uint64_t                    m_evicted_seq = 0;
	uint64_t                    m_pending_low  = 0;
	uint64_t                    m_pending_high = 0;
	std::vector<ReadTarget>     m_read_targets;
	std::vector<ReadDump>       m_read_dumps;
	std::unique_ptr<Buffer>     m_slot_results;
	std::unique_ptr<Buffer>     m_totals;
	vk::PipelineLayout          m_counter_layout   = nullptr;
	vk::Pipeline                m_counter_pipeline = nullptr;
};

}

#endif
