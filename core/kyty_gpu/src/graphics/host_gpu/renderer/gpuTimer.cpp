#include "graphics/host_gpu/renderer/gpuTimer.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <atomic>
#include <chrono>

namespace Libs::Graphics::GpuTimer {

namespace {

// Each command buffer owns one chunk of the query pool: its start, one before/after pair per timed
// command and its end. Commands past the chunk are not timed (counted as overflow).
constexpr uint32_t ChunkQueries = 4096;
constexpr uint32_t Chunks       = 64;
constexpr uint32_t NoSlot       = ~0u;

struct TimedCommand {
	char     kind   = 0;
	uint64_t a      = 0;
	uint64_t b      = 0;
	uint32_t before = 0;
	uint32_t after  = 0;
	uint32_t marker = 0; // KYTY_TRACE_GPU_TIME_MARKERS: interned game marker path (0: none)
};

// KYTY_TRACE_GPU_TIME_MARKERS=1 (with KYTY_TRACE_GPU_TIME=1): the per-command GPU time is also summed by the game's
// marker path at the command (GPUTIME_MARKERS line, top 40 per window).
bool MarkersEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_GPU_TIME_MARKERS");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

struct Chunk {
	vk::CommandBuffer         buffer   = nullptr;
	uint32_t                  used     = 0;
	uint32_t                  reset    = ChunkQueries; // queries to reset before the next use
	bool                      busy     = false;        // recorded, results not read yet
	uint64_t                  tick     = 0;
	uint32_t                  overflow = 0;
	uint32_t                  frame    = 0; // KYTY_TRACE_GPU_TIMELINE: the presented-frame count when it was begun
	// KYTY_TRACE_GPU_TIMELINE (CPU side, steady clock ns): GPU thread began recording, handed it over for submission,
	// recording worker called vkQueueSubmit.
	int64_t                   cpu_begin   = 0;
	int64_t                   cpu_handoff = 0;
	int64_t                   cpu_submit  = 0;
	std::vector<TimedCommand> commands;
	std::vector<TimedCommand> marks; // emulator operations (Mark); 'after' unused
};

struct Key {
	char     kind = 0;
	uint64_t a    = 0;
	uint64_t b    = 0;
	bool     operator==(const Key& other) const { return kind == other.kind && a == other.a && b == other.b; }
};

struct KeyHash {
	size_t operator()(const Key& key) const {
		return std::hash<uint64_t>()((key.a * 0x9e3779b97f4a7c15ull) ^ key.b ^ (static_cast<uint64_t>(key.kind) << 56u));
	}
};

struct Cost {
	uint64_t ticks = 0;
	uint64_t count = 0;
};

struct State {
	std::mutex            mutex;
	bool                  init_done   = false;
	vk::Device            device      = nullptr;
	vk::QueryPool         pool        = nullptr;
	double                ns_per_tick = 1.0;
	uint64_t              valid_mask  = ~0ull;
	Chunk                 chunks[Chunks];
	uint32_t              next_chunk = 0;
	int32_t               current    = -1;
	std::deque<uint32_t>  pending; // submitted chunks in submission order
	uint64_t              last_end = 0;
	std::vector<uint64_t> results;
	// Mode 3 alternates the per-command (1) and command-buffer-only (2) measurement every window.
	int                   window_mode = 1;

	std::chrono::steady_clock::time_point  window_start = std::chrono::steady_clock::now();
	uint32_t                               window_frame = 0;
	uint64_t                               cbs          = 0;
	uint64_t                               untimed_cbs  = 0;
	uint64_t                               busy         = 0;
	uint64_t                               gap          = 0;
	uint64_t                               cmd          = 0;
	uint64_t                               cmd_count    = 0;
	uint64_t                               other        = 0;
	uint64_t                               marked       = 0;
	uint64_t                               overflow     = 0;
	std::unordered_map<Key, Cost, KeyHash> costs;
	// KYTY_TRACE_GPU_TIME_MARKERS
	std::unordered_map<std::string, uint32_t> marker_ids;
	std::vector<std::string>                  marker_names {std::string("-")};
	std::vector<Cost>                         marker_costs {Cost {}};
};

// KYTY_TRACE_GPU_TIMELINE=<frame> (diagnostic, with KYTY_TRACE_GPU_TIME=1 and KYTY_TRACE_GPU_TIME_MARKERS=1): for the
// command buffers begun in frames <frame>..<frame>+2, one GPUTL line each when its timestamps are read: its tick, its GPU
// start and end relative to the first such buffer, the idle gap before it, and its four costliest marker paths. Shows
// which GPU work runs while the GPU thread waits.
uint32_t TimelineFirstFrame() {
	static const uint32_t first = [] {
		const char* value = std::getenv("KYTY_TRACE_GPU_TIMELINE");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return first;
}

int64_t SteadyNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// KYTY_TRACE_GPU_TIMELINE CPU side: the GPU thread's waits during the timeline's frames, printed once they are over.
struct CpuWait {
	char    kind  = 0;
	int64_t start = 0;
	int64_t end   = 0;
};
std::mutex             g_cpu_waits_mutex;
std::vector<CpuWait>   g_cpu_waits;
std::atomic<int64_t>   g_cpu_base {0}; // steady ns of the first timeline buffer's recording start
std::atomic<bool>      g_cpu_waits_printed {false};

uint32_t InternMarker(State& s) {
	auto       path = DebugMarkerPath();
	const auto it   = s.marker_ids.find(path);
	if (it != s.marker_ids.end()) return it->second;
	const auto id = static_cast<uint32_t>(s.marker_names.size());
	s.marker_ids.emplace(path, id);
	s.marker_names.push_back(std::move(path));
	s.marker_costs.push_back({});
	return id;
}

State& GetState() {
	static State state;
	return state;
}

uint32_t ChunkBase(uint32_t index) {
	return index * ChunkQueries;
}

bool Init(GraphicContext& graphics) {
	auto& s = GetState();
	if (s.init_done) {
		return s.pool != nullptr;
	}
	s.init_done = true;
	s.device    = graphics.device;
	const auto families = graphics.physical_device.getQueueFamilyProperties();
	const uint32_t bits = graphics.queue_family < families.size() ? families[graphics.queue_family].timestampValidBits : 0u;
	if (bits == 0) {
		std::fprintf(stderr, "GPUTIME disabled: the queue family has no timestamps\n");
		return false;
	}
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = Chunks * ChunkQueries;
	const auto created = graphics.device.createQueryPool(info);
	if (created.result != vk::Result::eSuccess) {
		std::fprintf(stderr, "GPUTIME disabled: query pool creation failed (%d)\n", static_cast<int>(created.result));
		return false;
	}
	s.pool        = created.value;
	s.ns_per_tick = graphics.physical_device_properties.limits.timestampPeriod;
	s.window_mode = Mode() == 2 ? 2 : 1;
	s.valid_mask  = bits >= 64u ? ~0ull : ((1ull << bits) - 1u);
	s.results.resize(ChunkQueries);
	std::fprintf(stderr, "GPUTIME mode=%d queries=%u ns_per_tick=%.3f valid_bits=%u\n", Mode(), info.queryCount, s.ns_per_tick,
	             bits);
	return true;
}

void Accumulate(State& s, const Chunk& chunk) {
	const auto ts = [&](uint32_t query) { return s.results[query] & s.valid_mask; };
	if (chunk.used < 2u) {
		s.untimed_cbs++;
		return;
	}
	const auto begin = ts(0);
	const auto end   = ts(chunk.used - 1u);
	if (end < begin) {
		s.untimed_cbs++;
		return;
	}
	s.cbs++;
	s.busy += end - begin;
	if (s.last_end != 0 && begin > s.last_end) {
		s.gap += begin - s.last_end;
	}
	s.last_end = end;
	// Each interval between consecutive timestamps belongs to what started it: a timed draw or
	// dispatch, a marked emulator operation, or nothing known ("other").
	std::vector<int32_t> owner(chunk.used, -1);
	for (size_t i = 0; i < chunk.commands.size(); i++) {
		owner[chunk.commands[i].before] = static_cast<int32_t>(i);
	}
	for (size_t i = 0; i < chunk.marks.size(); i++) {
		owner[chunk.marks[i].before] = static_cast<int32_t>(chunk.commands.size() + i);
	}
	for (uint32_t query = 0; query + 1u < chunk.used; query++) {
		const auto from = ts(query);
		const auto to   = ts(query + 1u);
		if (to < from) {
			continue;
		}
		const auto ticks = to - from;
		const auto index = owner[query];
		if (index < 0) {
			s.other += ticks;
			continue;
		}
		const bool  mark    = static_cast<size_t>(index) >= chunk.commands.size();
		const auto& command = mark ? chunk.marks[index - chunk.commands.size()] : chunk.commands[index];
		auto&       cost    = s.costs[Key {command.kind, command.a, command.b}];
		cost.ticks += ticks;
		if (command.marker < s.marker_costs.size()) s.marker_costs[command.marker].ticks += ticks;
		if (mark) {
			s.marked += ticks;
		} else {
			s.cmd += ticks;
		}
	}
	for (const auto& command: chunk.commands) {
		s.costs[Key {command.kind, command.a, command.b}].count++;
		if (command.marker < s.marker_costs.size()) s.marker_costs[command.marker].count++;
	}
	for (const auto& command: chunk.marks) {
		s.costs[Key {command.kind, command.a, command.b}].count++;
	}
	s.cmd_count += chunk.commands.size();
	s.overflow += chunk.overflow;
	// KYTY_TRACE_GPU_TIMELINE
	const uint32_t first = TimelineFirstFrame();
	if (first != 0 && chunk.frame >= first + 3u && !g_cpu_waits_printed.exchange(true)) {
		// The GPU thread's waits of the timeline's frames (CPU side), on the GPUTL lines' CPU base.
		const int64_t    cpu_base = g_cpu_base.load();
		std::scoped_lock lock(g_cpu_waits_mutex);
		for (const auto& w: g_cpu_waits) {
			std::fprintf(stderr, "GPUTL_WAIT kind=%c start_us=%.1f dur_us=%.1f\n", w.kind,
			             static_cast<double>(w.start - cpu_base) / 1.0e3, static_cast<double>(w.end - w.start) / 1.0e3);
		}
	}
	if (first != 0 && chunk.frame >= first && chunk.frame < first + 3u) {
		static uint64_t base = 0, previous_end = 0;
		if (base == 0) base = begin;
		std::vector<std::pair<uint64_t, uint32_t>> by_marker; // (ticks, marker)
		std::vector<std::pair<uint64_t, uint64_t>> by_mark;   // (ticks, emulator operation id): marked operations
		uint64_t                                   other_ticks = 0;
		for (uint32_t query = 0; query + 1u < chunk.used; query++) {
			const auto from = ts(query), to = ts(query + 1u);
			const auto index = owner[query];
			if (to < from) continue;
			if (index < 0) {
				other_ticks += to - from;
				continue;
			}
			if (static_cast<size_t>(index) >= chunk.commands.size()) {
				const uint64_t id = chunk.marks[index - chunk.commands.size()].a;
				auto it = std::find_if(by_mark.begin(), by_mark.end(), [&](const auto& e) { return e.second == id; });
				if (it == by_mark.end()) by_mark.push_back({to - from, id});
				else it->first += to - from;
				continue;
			}
			const uint32_t marker = chunk.commands[index].marker;
			auto it = std::find_if(by_marker.begin(), by_marker.end(), [&](const auto& e) { return e.second == marker; });
			if (it == by_marker.end()) by_marker.push_back({to - from, marker});
			else it->first += to - from;
		}
		std::sort(by_marker.begin(), by_marker.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
		const auto us = [&](uint64_t ticks) { return static_cast<double>(ticks) * s.ns_per_tick / 1.0e3; };
		// Marker paths per line.
		constexpr size_t top_n = 4;
		std::string      top;
		for (size_t i = 0; i < by_marker.size() && i < top_n; i++) {
			const uint32_t m = by_marker[i].second;
			top += " [" + std::to_string(static_cast<int>(us(by_marker[i].first))) + "us " +
			       (m < s.marker_names.size() ? s.marker_names[m] : std::string("?")) + "]";
		}
		std::sort(by_mark.begin(), by_mark.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
		top += " | other=" + std::to_string(static_cast<int>(us(other_ticks))) + "us emu:";
		for (size_t i = 0; i < by_mark.size() && i < 3u; i++) {
			char item[48];
			std::snprintf(item, sizeof(item), " %llx=%dus", static_cast<unsigned long long>(by_mark[i].second),
			              static_cast<int>(us(by_mark[i].first)));
			top += item;
		}
		const int64_t cpu_base = g_cpu_base.load();
		const auto    cpu_us   = [&](int64_t ns) { return ns != 0 && cpu_base != 0 ? static_cast<double>(ns - cpu_base) / 1.0e3 : -1.0; };
		std::fprintf(stderr,
		             "GPUTL frame=%u tick=%llu start_us=%.1f dur_us=%.1f gap_us=%.1f cmds=%zu marks=%zu rec_us=%.1f handoff_us=%.1f "
		             "submit_us=%.1f%s\n",
		             chunk.frame, static_cast<unsigned long long>(chunk.tick), us(begin - base), us(end - begin),
		             previous_end != 0 && begin > previous_end ? us(begin - previous_end) : 0.0, chunk.commands.size(),
		             chunk.marks.size(), cpu_us(chunk.cpu_begin), cpu_us(chunk.cpu_handoff), cpu_us(chunk.cpu_submit),
		             top.c_str());
		previous_end = end;
	}
}

void Report(State& s) {
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window_start < std::chrono::seconds(5)) {
		return;
	}
	const auto window_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - s.window_start).count();
	s.window_start       = now;
	const auto frame     = DebugPresentCount().load();
	const auto frames    = frame - s.window_frame;
	s.window_frame       = frame;
	const auto ms        = [&](uint64_t ticks) { return static_cast<double>(ticks) * s.ns_per_tick / 1.0e6; };
	std::fprintf(stderr,
	             "GPUTIME mode=%d window_ms=%lld frame=%u frames=%u cbs=%llu untimed_cbs=%llu busy=%.0fms gap=%.0fms "
	             "cmds=%.0fms/%llu marked=%.0fms other=%.0fms overflow=%llu\n",
	             s.window_mode, static_cast<long long>(window_ms), frame, frames, static_cast<unsigned long long>(s.cbs),
	             static_cast<unsigned long long>(s.untimed_cbs), ms(s.busy), ms(s.gap), ms(s.cmd),
	             static_cast<unsigned long long>(s.cmd_count), ms(s.marked), ms(s.other), static_cast<unsigned long long>(s.overflow));
	if (!s.costs.empty()) {
		std::vector<std::pair<Key, Cost>> top(s.costs.begin(), s.costs.end());
		std::sort(top.begin(), top.end(), [](const auto& x, const auto& y) { return x.second.ticks > y.second.ticks; });
		std::fprintf(stderr, "GPUTIME_TOP");
		// KYTY_TRACE_GPU_TIME_TOP=<n>: list length (default 30).
		static const size_t top_count = [] {
			const char* value = std::getenv("KYTY_TRACE_GPU_TIME_TOP");
			return value != nullptr ? static_cast<size_t>(std::strtoul(value, nullptr, 10)) : size_t {30};
		}();
		for (size_t i = 0; i < top.size() && i < top_count; i++) {
			std::fprintf(stderr, " [%c %016llx %016llx %.1fms n=%llu]", top[i].first.kind,
			             static_cast<unsigned long long>(top[i].first.a), static_cast<unsigned long long>(top[i].first.b),
			             ms(top[i].second.ticks), static_cast<unsigned long long>(top[i].second.count));
		}
		std::fprintf(stderr, "\n");
	}
	if (MarkersEnabled()) {
		std::vector<uint32_t> order;
		for (uint32_t i = 0; i < s.marker_costs.size(); i++) {
			if (s.marker_costs[i].ticks != 0) order.push_back(i);
		}
		std::sort(order.begin(), order.end(),
		          [&](uint32_t x, uint32_t y) { return s.marker_costs[x].ticks > s.marker_costs[y].ticks; });
		std::fprintf(stderr, "GPUTIME_MARKERS frames=%u", frames);
		for (size_t i = 0; i < order.size() && i < 40u; i++) {
			std::fprintf(stderr, " [%.1fms n=%llu %s]", ms(s.marker_costs[order[i]].ticks),
			             static_cast<unsigned long long>(s.marker_costs[order[i]].count), s.marker_names[order[i]].c_str());
		}
		std::fprintf(stderr, "\n");
		for (auto& cost: s.marker_costs) cost = {};
	}
	s.cbs = s.untimed_cbs = s.busy = s.gap = s.cmd = s.cmd_count = s.other = s.marked = s.overflow = 0;
	s.costs.clear();
	if (Mode() == 3) {
		s.window_mode = s.window_mode == 1 ? 2 : 1;
	}
}

// Reads the results of command buffers the GPU finished, in submission order.
void Poll(State& s, CommandScheduler& scheduler) {
	while (!s.pending.empty()) {
		const auto index = s.pending.front();
		auto&      chunk = s.chunks[index];
		if (!scheduler.IsFree(chunk.tick)) {
			break;
		}
		s.pending.pop_front();
		const auto result = chunk.used == 0 ? vk::Result::eNotReady
		                                    : s.device.getQueryPoolResults(s.pool, ChunkBase(index), chunk.used,
		                                                                   chunk.used * sizeof(uint64_t), s.results.data(),
		                                                                   sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eSuccess) {
			Accumulate(s, chunk);
		} else {
			s.untimed_cbs++;
		}
		chunk.busy  = false;
		chunk.reset = std::max(chunk.used, 1u);
	}
	Report(s);
}

} // namespace

int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_TRACE_GPU_TIME");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	return mode;
}

void BeginCommandBuffer(CommandScheduler& scheduler, vk::CommandBuffer buffer) {
	// Only the renderer's scheduler is timed: the presentation scheduler records on its own thread
	// with its own timeline, so its buffers would interleave with the renderer's chunks.
	if (&scheduler.Context().GetCommandScheduler() != &scheduler) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	s.current = -1;
	if (!Init(scheduler.Graphics())) {
		return;
	}
	Poll(s, scheduler);
	const auto index = s.next_chunk;
	auto&      chunk = s.chunks[index];
	if (chunk.busy) {
		s.untimed_cbs++;
		return;
	}
	s.next_chunk = (index + 1u) % Chunks;
	buffer.resetQueryPool(s.pool, ChunkBase(index), chunk.reset);
	chunk.buffer   = buffer;
	chunk.used     = 0;
	chunk.busy     = true;
	chunk.tick     = 0;
	chunk.overflow = 0;
	chunk.frame    = DebugPresentCount().load();
	chunk.cpu_begin   = SteadyNs();
	chunk.cpu_handoff = 0;
	chunk.cpu_submit  = 0;
	if (const uint32_t first = TimelineFirstFrame(); first != 0 && chunk.frame >= first && g_cpu_base.load() == 0) {
		g_cpu_base.store(chunk.cpu_begin);
	}
	chunk.commands.clear();
	chunk.marks.clear();
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, ChunkBase(index) + chunk.used++);
	s.current = static_cast<int32_t>(index);
}

void EndCommandBuffer(vk::CommandBuffer buffer) {
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (s.current < 0 || s.chunks[s.current].buffer != buffer) {
		return;
	}
	auto& chunk = s.chunks[s.current];
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, ChunkBase(s.current) + chunk.used++);
}

void Submitted(vk::CommandBuffer buffer, uint64_t tick) {
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (s.current < 0 || s.chunks[s.current].buffer != buffer) {
		return;
	}
	s.chunks[s.current].tick        = tick;
	s.chunks[s.current].cpu_handoff = SteadyNs();
	s.pending.push_back(static_cast<uint32_t>(s.current));
	s.current = -1;
}

void NoteQueueSubmit(uint64_t tick) {
	if (TimelineFirstFrame() == 0) return;
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	for (const uint32_t index: s.pending) {
		if (s.chunks[index].tick == tick) {
			s.chunks[index].cpu_submit = SteadyNs();
			return;
		}
	}
}

bool TimelineCpuActive() {
	const uint32_t first = TimelineFirstFrame();
	if (first == 0) return false;
	const uint32_t frame = DebugPresentCount().load(std::memory_order_relaxed);
	return frame >= first && frame < first + 3u;
}

void NoteCpuWait(char kind, std::chrono::steady_clock::time_point start) {
	const int64_t end   = SteadyNs();
	const int64_t begin = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
	std::scoped_lock lock(g_cpu_waits_mutex);
	if (g_cpu_waits.size() < 200000u) g_cpu_waits.push_back({kind, begin, end});
}

uint32_t BeginCommand(vk::CommandBuffer buffer) {
	if (Mode() == 0 || Mode() == 2) {
		return NoSlot;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (s.window_mode != 1 || s.current < 0 || s.chunks[s.current].buffer != buffer) {
		return NoSlot;
	}
	auto& chunk = s.chunks[s.current];
	// Room for this command's two timestamps and the command buffer's end.
	if (chunk.used + 3u > ChunkQueries) {
		chunk.overflow++;
		return NoSlot;
	}
	const auto query = chunk.used++;
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, ChunkBase(s.current) + query);
	return (static_cast<uint32_t>(s.current) << 16u) | query;
}

void EndCommand(vk::CommandBuffer buffer, uint32_t slot, char kind, uint64_t a, uint64_t b) {
	if (slot == NoSlot) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (s.current < 0 || static_cast<uint32_t>(s.current) != (slot >> 16u) || s.chunks[s.current].buffer != buffer) {
		return;
	}
	auto&      chunk = s.chunks[s.current];
	const auto query = chunk.used++;
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, ChunkBase(s.current) + query);
	chunk.commands.push_back({kind, a, b, slot & 0xffffu, query, MarkersEnabled() ? InternMarker(s) : 0u});
}

} // namespace Libs::Graphics::GpuTimer

namespace Libs::Graphics::GpuTimer {

void Mark(vk::CommandBuffer buffer, char kind, uint64_t a, uint64_t b) {
	if (Mode() == 0 || Mode() == 2) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (s.window_mode != 1 || s.current < 0 || s.chunks[s.current].buffer != buffer) {
		return;
	}
	auto& chunk = s.chunks[s.current];
	if (chunk.used + 2u > ChunkQueries) {
		chunk.overflow++;
		return;
	}
	const auto query = chunk.used++;
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, ChunkBase(s.current) + query);
	chunk.marks.push_back({kind, a, b, query, 0});
}

} // namespace Libs::Graphics::GpuTimer

namespace Libs::Graphics::GpuTimer {

bool LabelsEnabled() {
	static const bool enabled = std::getenv("KYTY_TRACE_VK_LABELS") != nullptr;
	return enabled;
}

void LabelBegin(vk::CommandBuffer buffer, char kind, uint64_t a, uint64_t b) {
	// The innermost two game debug markers follow the shaders, e.g. "draw <vs> <ps> | Water>Water Deferred".
	char       name[320];
	const auto marker = DebugMarkerPath();
	if (kind == 'c') {
		std::snprintf(name, sizeof(name), "dispatch %016llx | %s", static_cast<unsigned long long>(a), marker.c_str());
		static std::atomic_uint64_t compute_labels {0};
		if ((compute_labels.fetch_add(1) & 0xfffu) == 0) std::fprintf(stderr, "VK_LABELS compute=%llu\n", static_cast<unsigned long long>(compute_labels.load()));
	} else {
		std::snprintf(name, sizeof(name), "draw %016llx %016llx | %s", static_cast<unsigned long long>(a),
		              static_cast<unsigned long long>(b), marker.c_str());
	}
	vk::DebugUtilsLabelEXT label {};
	label.pLabelName = name;
	buffer.beginDebugUtilsLabelEXT(label);
}

void LabelEnd(vk::CommandBuffer buffer) {
	buffer.endDebugUtilsLabelEXT();
}

void LabelEmu(vk::CommandBuffer buffer, const char* what) {
	char name[320];
	std::snprintf(name, sizeof(name), "emu %s | %s", what, DebugMarkerPath().c_str());
	vk::DebugUtilsLabelEXT label {};
	label.pLabelName = name;
	buffer.beginDebugUtilsLabelEXT(label);
}

} // namespace Libs::Graphics::GpuTimer
