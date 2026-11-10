#include "graphics/host_gpu/renderer/virtualGpuClock.h"

#include "common/onceValue.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <algorithm>
#include <mutex>
#include <vector>

namespace Libs::Graphics::VirtualGpuClock {

namespace {

// Each command buffer owns one chunk of the query pool: query 0 is its start, the last used query its end, the ones in
// between its clock events. More events than fit are written at once (counted as "full").
constexpr uint32_t ChunkQueries = 16;
constexpr uint32_t Chunks       = 4096;

struct Event {
	uint32_t query = 0;
	uint64_t dst   = 0;
};

struct Chunk {
	vk::CommandBuffer  buffer = nullptr;
	uint32_t           used   = 0;
	uint32_t           reset  = ChunkQueries; // queries to reset before the next use
	bool               busy   = false;        // recorded, results not read yet
	bool               ended  = false;        // the end timestamp is recorded
	uint64_t           tick   = 0;
	std::vector<Event> events;
};

struct Write {
	uint64_t dst     = 0;
	uint64_t value   = 0;
	uint32_t retries = 0;
};

// Attempts before a value that keeps waiting is given up (Retry never ends only for a bad address shape).
constexpr uint32_t MaxRetries = 4096;

struct Stats {
	uint64_t events = 0, deferred = 0, immediate = 0, full = 0;
	uint64_t stored = 0, plain = 0, retried = 0, dropped = 0;
	uint64_t cbs = 0, untimed_cbs = 0;
	double   busy_ns = 0.0;
};

struct State {
	// The GPU thread is the only user; recursive because a write can run deferred operations (ours included).
	std::recursive_mutex  mutex;
	bool                  flushing    = false; // Flush runs (a nested call returns at once)
	bool                  init_done   = false;
	vk::Device            device      = nullptr;
	vk::QueryPool         pool        = nullptr;
	double                ns_per_tick = 1.0;
	uint64_t              valid_mask  = ~0ull;
	Chunk                 chunks[Chunks];
	uint32_t              next_chunk = 0;
	int32_t               current    = -1;
	std::deque<uint32_t>  pending; // submitted chunks in submission order
	std::vector<uint64_t> results;
	// The clock: base (reference clock when the renderer started) + busy / 10 ns.
	uint64_t base_ticks = 0;
	double   busy_ns    = 0.0; // union of the finished command buffers' busy intervals
	uint64_t last_end   = 0;   // newest end timestamp seen (host GPU ticks)
	// Values resolved but not written yet, in stream order.
	std::deque<Write> ready;
	// KYTY_TRACE_PERF window.
	Stats                                 stats;
	std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
};

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
	const auto     families = graphics.physical_device.getQueueFamilyProperties();
	const uint32_t bits = graphics.queue_family < families.size() ? families[graphics.queue_family].timestampValidBits : 0u;
	if (bits == 0) {
		std::fprintf(stderr, "FIX_VIRTUAL_GPU_CLOCK unavailable: the queue family has no timestamps\n");
		return false;
	}
	vk::QueryPoolCreateInfo info {};
	info.queryType     = vk::QueryType::eTimestamp;
	info.queryCount    = Chunks * ChunkQueries;
	const auto created = graphics.device.createQueryPool(info);
	if (created.result != vk::Result::eSuccess) {
		std::fprintf(stderr, "FIX_VIRTUAL_GPU_CLOCK unavailable: query pool creation failed (%d)\n", static_cast<int>(created.result));
		return false;
	}
	s.pool        = created.value;
	s.ns_per_tick = graphics.physical_device_properties.limits.timestampPeriod;
	s.valid_mask  = bits >= 64u ? ~0ull : ((1ull << bits) - 1u);
	s.results.resize(ChunkQueries);
	s.base_ticks = Sync::ReadReferenceClock();
	std::fprintf(stderr, "FIX_VIRTUAL_GPU_CLOCK ready: ns_per_tick=%.3f valid_bits=%u\n", s.ns_per_tick, bits);
	return true;
}

// The clock value (100 MHz ticks) after `busy_ns` of GPU work.
uint64_t ClockAt(const State& s, double busy_ns) {
	return s.base_ticks + static_cast<uint64_t>(busy_ns / 10.0);
}

// Reads the results of the command buffers the GPU finished, in submission order: the busy time grows by each one's
// interval minus what overlaps the previous ones, and each clock event of it becomes a value ready to be written.
void Resolve(State& s, CommandScheduler& scheduler) {
	while (!s.pending.empty()) {
		const auto index = s.pending.front();
		auto&      chunk = s.chunks[index];
		if (!scheduler.IsFree(chunk.tick)) {
			break;
		}
		s.pending.pop_front();
		const auto result = chunk.used < 2u ? vk::Result::eNotReady
		                                    : s.device.getQueryPoolResults(s.pool, ChunkBase(index), chunk.used,
		                                                                   chunk.used * sizeof(uint64_t), s.results.data(),
		                                                                   sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		const auto ts      = [&](uint32_t query) { return s.results[query] & s.valid_mask; };
		bool       timed   = result == vk::Result::eSuccess && chunk.ended;
		uint64_t   begin   = 0;
		uint64_t   end     = 0;
		if (timed) {
			begin = ts(0);
			end   = ts(chunk.used - 1u);
			timed = end >= begin;
		}
		if (!timed) {
			// Its events get the clock as it stands: its own busy time is unknown.
			s.stats.untimed_cbs++;
			for (const auto& event: chunk.events) {
				s.ready.push_back({event.dst, ClockAt(s, s.busy_ns)});
			}
		} else {
			s.stats.cbs++;
			const uint64_t from = std::max(begin, s.last_end);
			for (const auto& event: chunk.events) {
				const uint64_t at = ts(event.query);
				const double   in = at > from ? static_cast<double>(at - from) * s.ns_per_tick : 0.0;
				s.ready.push_back({event.dst, ClockAt(s, s.busy_ns + in)});
			}
			if (end > from) {
				const double add = static_cast<double>(end - from) * s.ns_per_tick;
				s.busy_ns += add;
				s.stats.busy_ns += add;
			}
			s.last_end = std::max(s.last_end, end);
		}
		chunk.busy  = false;
		chunk.reset = std::max(chunk.used, 1u);
		chunk.events.clear();
	}
}

// Writes the resolved values, in order. A value the GPU-owned page cannot take right now (no command buffer to record
// the GPU copy's update into, or a pending copy over the bytes) stays queued and is tried again at the next event or
// deferred operation; a value whose bytes the GPU itself owns is dropped (writing it would need a GPU drain).
void Flush(State& s, CommandScheduler& scheduler) {
	if (s.flushing) {
		return;
	}
	s.flushing = true;
	auto& buffers = scheduler.Context().GetBufferCache();
	while (!s.ready.empty()) {
		const auto write  = s.ready.front();
		const auto result = buffers.WriteCompletedValue(write.dst, &write.value, sizeof(write.value));
		if (result == BufferCache::CompletedWrite::Retry && ++s.ready.front().retries < MaxRetries) {
			s.stats.retried++;
			break;
		}
		s.ready.pop_front();
		if (result == BufferCache::CompletedWrite::Plain) {
			std::memcpy(reinterpret_cast<void*>(write.dst), &write.value, sizeof(write.value));
			s.stats.plain++;
		} else if (result == BufferCache::CompletedWrite::Stored) {
			s.stats.stored++;
		} else {
			s.stats.dropped++;
			static uint32_t logged = 0;
			if (logged++ < 16u) {
				std::fprintf(stderr, "FIX_VIRTUAL_GPU_CLOCK dropped a value at 0x%llx (GPU-written bytes, an image or a merge "
				                     "page over it, or %u retries)\n",
				             static_cast<unsigned long long>(write.dst), MaxRetries);
			}
			continue;
		}
		PublishCpuDirtyRange(write.dst, sizeof(write.value)); // BDA sync epoch, as other GPU-thread writes
	}
	s.flushing = false;
}

void Report(State& s) {
	if (!PerfEnabled()) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now - s.window_start < std::chrono::seconds(5)) {
		return;
	}
	s.window_start = now;
	const auto& t  = s.stats;
	// events: clock events seen; deferred / immediate (interrupt) / full (no room); values stored through the GPU-owned
	// page, plain stores, retries, drops (must stay 0); command buffers timed / untimed; GPU busy time in the window.
	std::fprintf(stderr,
	             "PERF vgpu_clock events=%llu deferred=%llu immediate=%llu full=%llu stored=%llu plain=%llu retried=%llu "
	             "dropped=%llu cbs=%llu untimed_cbs=%llu busy=%.1fms queued=%zu\n",
	             static_cast<unsigned long long>(t.events), static_cast<unsigned long long>(t.deferred),
	             static_cast<unsigned long long>(t.immediate), static_cast<unsigned long long>(t.full),
	             static_cast<unsigned long long>(t.stored), static_cast<unsigned long long>(t.plain),
	             static_cast<unsigned long long>(t.retried), static_cast<unsigned long long>(t.dropped),
	             static_cast<unsigned long long>(t.cbs), static_cast<unsigned long long>(t.untimed_cbs), t.busy_ns / 1.0e6,
	             s.ready.size());
	s.stats = Stats {};
}

bool IsRenderer(CommandScheduler& scheduler) {
	// The presentation scheduler records on its own thread with its own timeline.
	return &scheduler.Context().GetCommandScheduler() == &scheduler;
}

} // namespace

bool Enabled() {
	static constinit Common::OnceValue<bool> enabled_once;
	return enabled_once.Get([] {
		const char* value   = std::getenv("KYTY_FIX_VIRTUAL_GPU_CLOCK");
		const bool  enabled = value == nullptr || value[0] != '0'; // default on (=0 turns it off)
		if (enabled) std::fprintf(stderr, "FIX_VIRTUAL_GPU_CLOCK enabled\n");
		return enabled;
	});
}

void BeginCommandBuffer(CommandScheduler& scheduler, vk::CommandBuffer buffer) {
	if (!IsRenderer(scheduler)) {
		return;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	s.current = -1;
	if (!Init(scheduler.Graphics())) {
		return;
	}
	Resolve(s, scheduler);
	Report(s);
	const auto index = s.next_chunk;
	auto&      chunk = s.chunks[index];
	if (chunk.busy) {
		// More command buffers in flight than chunks: this one is not timed (its events get the clock at once).
		s.stats.untimed_cbs++;
		return;
	}
	s.next_chunk = (index + 1u) % Chunks;
	buffer.resetQueryPool(s.pool, ChunkBase(index), chunk.reset);
	chunk.buffer = buffer;
	chunk.used   = 0;
	chunk.busy   = true;
	chunk.ended  = false;
	chunk.tick   = 0;
	chunk.events.clear();
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, s.pool, ChunkBase(index) + chunk.used++);
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
	chunk.ended = true;
}

void Submitted(vk::CommandBuffer buffer, uint64_t tick) {
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (s.current < 0 || s.chunks[s.current].buffer != buffer) {
		return;
	}
	s.chunks[s.current].tick = tick;
	s.pending.push_back(static_cast<uint32_t>(s.current));
	s.current = -1;
}

bool Stamp(CommandScheduler& scheduler, vk::CommandBuffer buffer, uint64_t dst, bool immediate) {
	if (!Enabled() || !IsRenderer(scheduler)) {
		return false;
	}
	auto&            s = GetState();
	std::scoped_lock lock(s.mutex);
	if (!Init(scheduler.Graphics())) {
		return false;
	}
	s.stats.events++;
	Resolve(s, scheduler);
	Flush(s, scheduler);
	const bool timed = s.current >= 0 && s.chunks[s.current].buffer == buffer;
	// Room for this event and the command buffer's end.
	if (immediate || !timed || s.chunks[s.current].used + 2u > ChunkQueries) {
		if (immediate) {
			s.stats.immediate++;
		} else {
			s.stats.full++;
		}
		s.ready.push_back({dst, ClockAt(s, s.busy_ns)});
		Flush(s, scheduler);
		return true;
	}
	auto&      chunk = s.chunks[s.current];
	const auto query = chunk.used++;
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, s.pool, ChunkBase(s.current) + query);
	chunk.events.push_back({query, dst});
	s.stats.deferred++;
	// Runs once this command buffer finished (the scheduler runs deferred operations in tick order).
	scheduler.DeferOperation([&scheduler] {
		auto&            state = GetState();
		std::scoped_lock deferred_lock(state.mutex);
		Resolve(state, scheduler);
		Flush(state, scheduler);
	});
	return true;
}

} // namespace Libs::Graphics::VirtualGpuClock
