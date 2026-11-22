#include "graphics/guest_gpu/scout.h"

#include "common/guardedCopy.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/lodStatsReport.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shaderBindings.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>
#include <x86intrin.h>
#include <xxhash.h>

namespace Libs::Graphics::Scout {

namespace {

namespace LiveCounters {
enum Id : uint32_t {
	ScoutAbandoned, ScoutStoodDown, ScoutWriteBacks, ScoutWaitsStopped, ScoutWaitsPassed, ScoutWaitOther, ScoutWaitChecks,
	ScoutWaitChanged, ScoutTooBig, ScoutTicketWords, ScoutTickets, ScoutSubmissions, ScoutShort, ScoutRewinds, ScoutRefused,
	ScoutNoViewOther, ScoutNoView, ScoutNotTable, ScoutLeadStalls, ScoutFenced, ScoutFeedFull, ScoutDrawPackets, ScoutConstant,
	ScoutBehind, ScoutBackingTickets, ScoutBackingRanges, Count
};
constexpr const char* Names[Count] = {
	"abandoned", "stood down", "write-backs", "waits stopped", "waits passed", "waits on other words", "waits checked",
	"waits changed", "too big", "ticket words", "tickets", "submissions", "short reads", "rewinds", "refused", "no view (other pair)",
	"no view", "not table", "lead stalls", "fenced", "feed full", "draw packets", "constant", "behind", "backing tickets",
	"backing ranges"};
std::atomic<uint64_t> g_counts[Count];
inline void           Add(Id id, uint64_t n = 1) noexcept { g_counts[id].fetch_add(n, std::memory_order_relaxed); }
}

struct alignas(128) ScoutWritten {
	std::atomic<uint64_t> quiescent {0};
};
struct alignas(128) ReadMostly {
	std::atomic<bool>     running {false};
	std::atomic<uint64_t> retire_clock {0};
};
struct alignas(128) RenderOwn {
	uint64_t render_sequence = 0;
	uint64_t version         = 0;
};
struct alignas(128) AnyWriter {
	std::atomic<uint64_t> write_clock {0};
};
ScoutWritten   g_scout_written;
ReadMostly     g_read_mostly;
RenderOwn      g_render_own;
AnyWriter      g_any_writer;

constexpr auto& g_render_sequence = g_render_own.render_sequence;

struct HostWrite {
	std::atomic<uint64_t>     clock {0};
	std::atomic<uint64_t>     first {0}, last {0};
	static constexpr uint64_t Busy = 1ull << 63u, WriteBack = 1ull << 63u;
};
constexpr uint64_t                  WriteEntries = 4096;
std::array<HostWrite, WriteEntries> g_writes;
constexpr auto&                     g_write_clock = g_any_writer.write_clock;

constexpr size_t                                 DirectoryEntries = 65536;
std::array<std::atomic<View*>, DirectoryEntries> g_directory {};
constexpr auto&                                  g_retire_clock = g_read_mostly.retire_clock;
constexpr auto&                                  g_quiescent    = g_scout_written.quiescent;
std::vector<View*>                               g_retired;
constexpr auto&                                  g_version = g_render_own.version;

constexpr auto&     g_running = g_read_mostly.running;
RenderContext* g_resources = nullptr;

constexpr std::array<uint32_t, 8> LodStatsMask {0, 0, 0, 0, 0, (0xfffu << 8u) | (1u << 25u), 0xffu, 0};

bool Read(void* out, uint64_t address, uint64_t bytes) {
	if (address == 0) return false;
#if !defined(_WIN32)
	if (Libs::LibKernel::Memory::TryReadBackingToHost(address, out, bytes)) return true;
#endif
	return Common::GuardedCopy(out, reinterpret_cast<const void*>(address), bytes) == 0;
}

void Quiesce() {
	const auto clock = g_retire_clock.load(std::memory_order_seq_cst);
	if (g_quiescent.load(std::memory_order_relaxed) != clock) g_quiescent.store(clock, std::memory_order_seq_cst);
}

void FreeRetired() {
	const auto quiescent = g_quiescent.load(std::memory_order_seq_cst);
	std::erase_if(g_retired, [&](View* view) {
		if (view->retired > quiescent) return false;
		delete view;
		return true;
	});
}

void RetireView(View* view) {
	view->retired = g_retire_clock.fetch_add(1, std::memory_order_seq_cst) + 1;
	g_retired.push_back(view);
}

struct Registers {
	uint64_t                 es = 0, ls = 0, hs = 0, ps = 0;
	std::array<uint32_t, 32> gs_user {}, ps_user {};
	int32_t                  start_instance = -1;
};

struct State {
	Registers                 registers;
	std::array<uint64_t, 256> mapped {};
	uint64_t                  mapping_epoch = UINT64_MAX;
	uint64_t                  constant      = 0;
	uint64_t                  fence         = 0;
	std::array<std::pair<uint64_t, uint32_t>, 256> draws {};
	std::array<std::vector<uint32_t>, 2>           blocks;
	std::vector<uint32_t>                          words, lod_stats;
	std::vector<std::array<uint64_t, 3>>           ranges;
	std::vector<uint64_t>                          pages;
	std::array<std::vector<std::pair<std::array<uint32_t, 4>, uint32_t>>, 2> entries;
};

enum class Loss { None, Refused, Short, TooBig };

Loss Evaluate(State& state, const View& view, Ticket& ticket) {
	auto& resources = *g_resources;
	const auto epoch = resources.MappingEpoch();
	if (epoch != state.mapping_epoch) {
		state.mapped.fill(UINT64_MAX);
		state.mapping_epoch = epoch;
	}
	ticket.mapping_epoch = epoch;
	ticket.written = g_write_clock.load(std::memory_order_acquire);
	ticket.version       = view.version;
	ticket.key           = view.key;
	ticket.stages        = static_cast<uint16_t>(view.stages);
	ticket.user_data     = {state.registers.gs_user, state.registers.ps_user};
	ticket.user_words    = {view.stage[0].user_words, view.stage[1].user_words};
	ticket.slot_words    = {};
	ticket.buffers       = {};
	state.words.clear();
	state.lod_stats.clear();
	state.ranges.clear();
	const auto* granules = resources.ReadGranules();
	uint32_t    backed   = 0;
	const auto  locate   = [&](uint64_t low, uint64_t high, uint32_t offset, int32_t immediate, uint32_t count,
                            uint64_t& out, bool bounded, uint32_t records, bool& watched) {
        if (bounded) {
            const auto stride = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
            const auto size   = stride == 0 ? uint64_t {records} : uint64_t {stride} * records;
            if (immediate < 0 || static_cast<uint64_t>(immediate) + count * 4ull > size) return Loss::Refused;
        }
        const auto address = ((low | high << 32u) & 0x0000fffffffffffcull) + (offset & ~3u) +
                             static_cast<uint64_t>(int64_t {immediate});
        const auto end = address + count * 4ull;
        for (auto page = address >> 14u; page <= (end - 1) >> 14u; ++page) {
            auto& seen = state.mapped[page & 255u];
            if (seen == page) continue;
            if (!Libs::LibKernel::Memory::IsFullyMapped(page << 14u, 16384)) return Loss::Refused;
            seen = page;
        }
        watched = false;
        for (auto page = address >> 12u; page <= (end - 1) >> 12u; ++page) {
            const auto granule = page >> (PageManager::READ_GRANULE_BITS - 12u);
            if ((page << 12u) < TRACKER_ADDRESS_SIZE &&
                ((granules[granule / 64u].load(std::memory_order_acquire) >> (granule % 64u)) & 1u) == 0)
                continue;
            if (resources.HasReadWatchers(page << 12u, 4096)) {
                watched = true;
                break;
            }
        }
        out = address;
        return Loss::None;
	};
	for (uint32_t st = 0; st < view.stages; ++st) {
		const auto& stage = view.stage[st];
		auto&       slots = state.blocks[st];
		slots.resize(stage.block_dwords);
		const std::array<const uint32_t*, 3> bases {st == 0 ? state.registers.gs_user.data() : state.registers.ps_user.data(),
		                                            slots.data(), stage.constants.data()};
		const auto get = [&](uint16_t operand) { return bases[operand >> 14u][operand & 0x3fffu]; };
		uint32_t   begin = 0;
		for (const auto end: stage.level_ends) {
			const auto first_range = state.ranges.size();
			for (auto i = begin; i < end; ++i) {
				const auto& run     = stage.runs[i];
				const bool  bounded = run.records != Run::NotBounded;
				uint64_t    address = 0;
				bool        watched = false;
				if (const auto loss = locate(get(run.address[0]), get(run.address[1]), get(run.address[2]), run.immediate,
				                             run.count, address, bounded, bounded ? get(run.records) : 0, watched);
				    loss != Loss::None)
					return loss;
				state.ranges.push_back({address, run.count * 4ull, watched ? 1ull : 0ull});
				backed += watched ? 1u : 0u;
			}
			for (auto i = begin; i < end; ++i) {
				const auto& run   = stage.runs[i];
				const auto& range = state.ranges[first_range + (i - begin)];
				if (range[2] != 0 ? !Libs::LibKernel::Memory::TryReadBackingToHost(range[0], slots.data() + run.first, range[1])
				                  : !Read(slots.data() + run.first, range[0], range[1]))
					return Loss::Short;
			}
			begin = end;
		}
		const auto first = state.words.size();
		state.words.resize(first + stage.image_words.size());
		auto*  out = state.words.data() + first;
		size_t at  = 0;
		for (size_t i = 0; i < stage.image_counts.size(); ++i) {
			std::array<uint32_t, 8> image {};
			const uint32_t          count = stage.image_counts[i];
			for (uint32_t w = 0; w < count; ++w) {
				image[w]    = get(stage.image_words[at + w]);
				out[at + w] = image[w] & ~LodStatsMask[w];
			}
			at += count;
			if (i < stage.lod_stats_count) {
				state.lod_stats.push_back(count >= 8u ? LodStatsReport::ImageField(image.data()) : LodStatsReport::NoCounterFlag);
			}
		}
		for (; at < stage.image_words.size(); ++at) out[at] = get(stage.image_words[at]);
		auto& entries = state.entries[st];
		entries.resize(stage.buffers);
		for (uint32_t i = 0; i < stage.buffers; ++i) {
			auto& [words4, index] = entries[i];
			for (uint32_t w = 0; w < 4; ++w) words4[w] = get(stage.buffer_words[4 * i + w]);
			index = static_cast<uint32_t>(XXH3_64bits(words4.data(), sizeof(words4)) & (ResolvedEntries - 1));
		}
	}
	state.pages.clear();
	for (const auto& [address, bytes, watched]: state.ranges)
		for (auto page = address >> 12u; page <= (address + bytes - 1) >> 12u; ++page)
			state.pages.push_back(page << 1u | (watched != 0 ? 0u : 1u));
	std::sort(state.pages.begin(), state.pages.end());
	size_t kept = 0;
	for (const auto entry: state.pages) {
		if (kept != 0 && (state.pages[kept - 1] >> 1u) == (entry >> 1u)) state.pages[kept - 1] |= entry & 1u;
		else state.pages[kept++] = entry;
	}
	state.pages.resize(kept);
	uint32_t need = static_cast<uint32_t>(state.words.size() + state.lod_stats.size() + 2 * state.pages.size() + 3 * backed);
	for (uint32_t st = 0; st < view.stages; ++st)
		need += view.stage[st].slot_words + 5 * view.stage[st].buffers + static_cast<uint32_t>(view.stage[st].segments.size());
	if (need > Ticket::Capacity || state.pages.empty()) return Loss::TooBig;
	auto* out = ticket.data;
	for (uint32_t st = 0; st < view.stages; ++st) {
		for (const auto& segment: view.stage[st].segments) {
			std::memcpy(out, state.blocks[st].data() + segment.first, segment.count * 4ull);
			out += segment.count;
		}
		ticket.slot_words[st] = static_cast<uint16_t>(view.stage[st].slot_words);
	}
	ticket.segments = {};
	for (uint32_t st = 0; st < view.stages; ++st) {
		for (const auto& segment: view.stage[st].segments) *out++ = segment.first | uint32_t {segment.count} << 16u;
		ticket.segments[st] = static_cast<uint16_t>(view.stage[st].segments.size());
	}
	out = std::copy(state.words.begin(), state.words.end(), out);
	out = std::copy(state.lod_stats.begin(), state.lod_stats.end(), out);
	for (uint32_t st = 0; st < view.stages; ++st) {
		for (const auto& [words4, index]: state.entries[st]) {
			out    = std::copy(words4.begin(), words4.end(), out);
			*out++ = index;
		}
		ticket.buffers[st] = static_cast<uint16_t>(view.stage[st].buffers);
	}
	for (const auto entry: state.pages) {
		const auto page = entry >> 1u;
		*out++          = static_cast<uint32_t>(page);
		*out++          = static_cast<uint32_t>(page >> 32u) | ((entry & 1u) != 0 ? 0u : Ticket::PageBacking);
	}
	uint32_t base  = 0;
	size_t   range = 0;
	for (uint32_t st = 0; st < view.stages; ++st) {
		for (const auto& run: view.stage[st].runs) {
			const auto& [address, bytes, watched] = state.ranges[range++];
			if (watched != 0) {
				*out++ = static_cast<uint32_t>(address);
				*out++ = static_cast<uint32_t>(address >> 32u);
				*out++ = (base + run.at) | static_cast<uint32_t>(bytes / 4u) << 16u;
			}
		}
		base += view.stage[st].slot_words;
	}
	if (backed != 0) {
		LiveCounters::Add(LiveCounters::ScoutBackingRanges, backed);
		LiveCounters::Add(LiveCounters::ScoutBackingTickets);
	}
	ticket.backing     = static_cast<uint16_t>(backed);
	ticket.first_page  = state.pages.front() >> 1u;
	ticket.last_page   = state.pages.back() >> 1u;
	ticket.image_words = static_cast<uint16_t>(state.words.size());
	ticket.lod_stats   = static_cast<uint16_t>(state.lod_stats.size());
	ticket.pages       = static_cast<uint16_t>(state.pages.size());
	return Loss::None;
}

}

bool Running() noexcept {
	return g_running.load(std::memory_order_relaxed);
}

void Start(RenderContext& resources) {
	if (Running()) return;
	g_resources = &resources;
	g_running.store(true);
}

void Stop() {
	if (!Running()) return;
	g_running.store(false);
	for (uint32_t i = 0; i < LiveCounters::Count; ++i)
		if (const auto n = LiveCounters::g_counts[i].load(std::memory_order_relaxed); n != 0)
			std::printf("Scout: %s %llu\n", LiveCounters::Names[i], static_cast<unsigned long long>(n));
	for (auto& entry: g_directory) delete entry.exchange(nullptr);
	for (auto* view: g_retired) delete view;
	g_retired.clear();
}

bool EvaluateAhead(const std::array<uint64_t, 3>& key, uint64_t key_hash, std::span<const uint32_t> gs_user,
                   std::span<const uint32_t> ps_user, Ticket& ticket) {
	static auto* const state = new State {};
	bool               made  = false;
	const auto*        view  = g_directory[key_hash & (DirectoryEntries - 1)].load(std::memory_order_seq_cst);
	if (view != nullptr && view->key == key) {
		auto& registers   = state->registers;
		registers.gs_user = {};
		registers.ps_user = {};
		std::copy_n(gs_user.begin(), std::min<size_t>(gs_user.size(), registers.gs_user.size()), registers.gs_user.begin());
		std::copy_n(ps_user.begin(), std::min<size_t>(ps_user.size(), registers.ps_user.size()), registers.ps_user.begin());
		made = Evaluate(*state, *view, ticket) == Loss::None;
	}
	Quiesce();
	return made;
}

void EnterSubmission(uint64_t sequence) {
	if (sequence == g_render_sequence) return;
	g_render_sequence = sequence;
	FreeRetired();
}

void NoteWrite(const void* address, uint32_t size) {
	NoteHostWrite(reinterpret_cast<uint64_t>(address), size);
}

void NoteHostWrite(uint64_t address, uint64_t size, bool write_back) {
	if (size == 0 || !Running()) return;
	if (write_back) LiveCounters::Add(LiveCounters::ScoutWriteBacks);
	const auto clock = g_write_clock.fetch_add(1, std::memory_order_acq_rel) + 1;
	auto&      entry = g_writes[clock % WriteEntries];
	for (auto old = entry.clock.load(std::memory_order_relaxed);;) {
		if ((old & HostWrite::Busy) != 0 || old >= clock) return;
		if (entry.clock.compare_exchange_weak(old, clock | HostWrite::Busy, std::memory_order_acquire,
		                                      std::memory_order_relaxed))
			break;
	}
	std::atomic_thread_fence(std::memory_order_release);
	entry.first.store(address >> 12u, std::memory_order_relaxed);
	entry.last.store((address + size - 1) >> 12u | (write_back ? HostWrite::WriteBack : 0), std::memory_order_relaxed);
	entry.clock.store(clock, std::memory_order_release);
}

bool WrittenSince(const Ticket& ticket, bool& write_back) {
	const auto now = g_write_clock.load(std::memory_order_acquire);
	if (now == ticket.written) return false;
	if (now - ticket.written >= WriteEntries) return true;
	const auto* pages = Pages(ticket);
	for (auto clock = ticket.written + 1; clock <= now; ++clock) {
		const auto& entry = g_writes[clock % WriteEntries];
		if (entry.clock.load(std::memory_order_acquire) != clock) return true;
		const auto first = entry.first.load(std::memory_order_relaxed);
		const auto last  = entry.last.load(std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_acquire);
		if (entry.clock.load(std::memory_order_relaxed) != clock) return true;
		const auto last_page = last & ~HostWrite::WriteBack;
		if (last_page < ticket.first_page || first > ticket.last_page) continue;
		for (uint32_t i = 0; i < ticket.pages; ++i) {
			const auto page = PageAt(pages, i);
			if (page > last_page) break;
			if (page >= first) {
				write_back = (last & HostWrite::WriteBack) != 0;
				return true;
			}
		}
	}
	return false;
}

uint64_t NextVersion() {
	return ++g_version;
}

void Publish(View* view, uint64_t key_hash) {
	if (auto* old = g_directory[key_hash & (DirectoryEntries - 1)].exchange(view, std::memory_order_seq_cst)) RetireView(old);
	FreeRetired();
}

void Retire(uint64_t key_hash, uint64_t version) {
	auto& entry = g_directory[key_hash & (DirectoryEntries - 1)];
	auto* view  = entry.load(std::memory_order_acquire);
	if (view != nullptr && view->version == version && entry.compare_exchange_strong(view, nullptr)) RetireView(view);
}

}
