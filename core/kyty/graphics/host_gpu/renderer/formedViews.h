#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FORMEDVIEWS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_FORMEDVIEWS_H_

#include "graphics/host_gpu/renderer/viewLog.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

enum class FormedState : uint8_t { Warming, Active, Demoted };

class FormedPrograms {
public:
	static constexpr uint32_t BlockBits = 16;
	static constexpr uint64_t BlockSize = uint64_t {1} << BlockBits;

	struct Ranges {
		uint64_t                                   version = 0;
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
	};

	FormedPrograms(int64_t warmup_ns, uint64_t warmup_uses): m_warmup_ns(warmup_ns), m_warmup_uses(warmup_uses) {}

	[[nodiscard]] static FormedPrograms& Instance();
	[[nodiscard]] static bool            StrictGate();
	[[nodiscard]] static int64_t         WarmupNs();
	[[nodiscard]] static uint64_t        WarmupUses();

	void               Seed(uint64_t program, uint64_t base, uint64_t bytes, int64_t now_ns);
	void               NoteUse(uint64_t program);
	[[nodiscard]] bool Allowed(uint64_t program, int64_t now_ns);
	void               Demote(uint64_t program);
	[[nodiscard]] FormedState State(uint64_t program) const;
	[[nodiscard]] bool        Covers(uint64_t program, uint64_t base, uint64_t bytes,
	                                 uint64_t version = UINT64_MAX) const;
	[[nodiscard]] bool        MayWrite(uint64_t address) const;
	[[nodiscard]] uint64_t    Version(uint64_t program) const;
	void                      CopyRanges(uint64_t program, Ranges& out) const;

	void               SetPending(size_t segments) noexcept { m_pending.store(segments, std::memory_order_seq_cst); }
	[[nodiscard]] bool Pending() const noexcept { return m_pending.load(std::memory_order_seq_cst) != 0; }
	[[nodiscard]] bool PendingMayWrite(uint64_t address) const { return StrictGate() && Pending() && MayWrite(address); }

	struct Counts {
		uint64_t new_blocks = 0;
		uint64_t activated  = 0;
		uint64_t demoted    = 0;
	};
	[[nodiscard]] Counts TakeCounts();

private:
	struct Program {
		FormedState                  state    = FormedState::Warming;
		std::map<uint64_t, uint64_t> blocks;
		uint64_t                     version  = 0;
		int64_t                      last_new = 0;
		std::atomic<uint64_t>        quiet_uses {0};
	};

	[[nodiscard]] static bool Holds(const Program& program, uint64_t base, uint64_t bytes, uint64_t version);

	int64_t                                m_warmup_ns   = 0;
	uint64_t                               m_warmup_uses = 0;
	mutable std::shared_mutex              m_mutex;
	std::unordered_map<uint64_t, Program>  m_programs;
	std::unordered_set<uint64_t>           m_blocks;
	std::atomic<size_t>                    m_pending {0};
	std::atomic<uint64_t>                  m_new_blocks {0};
	std::atomic<uint64_t>                  m_activated {0};
	std::atomic<uint64_t>                  m_demoted {0};
};

class FormedSink {
public:
	virtual ~FormedSink() = default;

	[[nodiscard]] virtual uint64_t Clamp(uint64_t base, uint64_t bytes)    = 0;
	[[nodiscard]] virtual bool     Windowed(uint64_t base, uint64_t bytes) = 0;
	[[nodiscard]] virtual bool     CpuDirty(uint64_t base, uint64_t bytes) = 0;
	virtual void                   Written(uint64_t base, uint64_t bytes, uint64_t tick) = 0;
};

struct FormedConsumeStats {
	uint64_t segments     = 0;
	uint64_t views        = 0;
	uint64_t written      = 0;
	uint64_t read_misses  = 0;
	uint64_t write_misses = 0;
	uint64_t form_misses  = 0;
	uint64_t conflicts    = 0;
	uint64_t overflows    = 0;
};

struct FormedConsumeResult {
	std::vector<std::string>                   reports;
	std::vector<std::pair<uint64_t, uint64_t>> binds;
	bool                                       stop = false;
};

void ConsumeFormedSegment(const ViewLogConsumed& segment, FormedPrograms& programs, FormedSink& sink,
                          FormedConsumeStats& stats, FormedConsumeResult& result, int64_t now_ns,
                          bool history_marked = false);

}

#endif
