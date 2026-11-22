#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_VIEWLOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_VIEWLOG_H_

#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct ViewLogEntry {
	uint64_t tagged_base = 0;
	uint64_t bytes       = 0;

	static constexpr uint64_t BaseMask    = (uint64_t {1} << 48u) - 1u;
	static constexpr uint32_t Read        = 1u;
	static constexpr uint32_t Written     = 2u;
	static constexpr uint32_t FormMiss    = 4u;
	static constexpr uint32_t SlotShift   = 8u;
	static constexpr uint32_t AccessMask  = Read | Written;

	[[nodiscard]] uint64_t Base() const noexcept { return tagged_base & BaseMask; }
	[[nodiscard]] uint32_t Flags() const noexcept { return static_cast<uint32_t>(tagged_base >> 48u); }
	[[nodiscard]] uint32_t Slot() const noexcept { return Flags() >> SlotShift; }
	[[nodiscard]] static ViewLogEntry Make(uint64_t base, uint64_t bytes, uint32_t flags) noexcept {
		return {(base & BaseMask) | (uint64_t {flags} << 48u), bytes};
	}
	bool operator==(const ViewLogEntry&) const = default;
};

struct ViewLogSegment {
	uint64_t device_address = 0;
	uint32_t first          = 0;
	uint32_t capacity       = 0;
};

class ViewLog {
public:
	static constexpr uint32_t EntryBytes = 16;

	using Segment = ViewLogSegment;

	struct Stats {
		uint64_t segments  = 0;
		uint64_t reserved  = 0;
		uint64_t checked   = 0;
		uint64_t logged    = 0;
		uint64_t distinct  = 0;
		uint64_t expected  = 0;
		uint64_t matched   = 0;
		uint64_t missing   = 0;
		uint64_t extra     = 0;
		uint64_t overflow  = 0;
		uint64_t misses    = 0;
		uint64_t full      = 0;
	};

	ViewLog(std::span<uint8_t> memory, uint64_t device_address);

	[[nodiscard]] std::optional<Segment> Reserve(uint32_t entries, uint64_t tick, uint64_t& wait_tick);
	void Attach(const Segment& segment, uint64_t program, std::vector<ViewLogEntry> expected);
	void Consume(uint64_t completed_tick);
	[[nodiscard]] uint64_t Pending() const noexcept;
	void Inject(const Segment& segment, std::span<const ViewLogEntry> entries);
	[[nodiscard]] Stats TakeStats() noexcept;
	[[nodiscard]] std::vector<std::string> TakeReports();
	[[nodiscard]] uint32_t Units() const noexcept { return m_units; }
	void                   Trace(uint32_t segments) noexcept { m_trace = segments; }

private:
	struct Live {
		uint32_t                  first    = 0;
		uint32_t                  units    = 0;
		uint32_t                  capacity = 0;
		uint64_t                  tick     = 0;
		uint64_t                  program  = 0;
		bool                      attached = false;
		std::vector<ViewLogEntry> expected;
	};

	void Check(const Live& live);
	[[nodiscard]] uint32_t& Count(uint32_t first) noexcept;
	[[nodiscard]] ViewLogEntry* Entries(uint32_t first) noexcept;

	std::span<uint8_t>       m_memory;
	uint64_t                 m_device_address = 0;
	uint32_t                 m_units          = 0;
	uint32_t                 m_head           = 0;
	uint32_t                 m_trace          = 0;
	std::deque<Live>         m_live;
	Stats                    m_stats;
	std::vector<std::string> m_reports;
};

}

#endif
