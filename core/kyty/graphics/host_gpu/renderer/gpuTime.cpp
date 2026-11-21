#include "graphics/host_gpu/renderer/gpuTime.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::GpuTime {

namespace {

constexpr uint32_t PoolQueries = 65536;
constexpr size_t   SwitchAfter = PoolQueries / 2;
constexpr uint32_t Pools       = 6;
constexpr uint32_t MaxMarkers  = 4096;
constexpr size_t   TopRows     = 25;
constexpr auto     Window      = std::chrono::seconds(10);

enum class Kind : uint8_t { Start, Before, Draw, Dispatch, End };

struct Record {
	uint64_t a      = 0;
	uint64_t b      = 0;
	uint32_t marker = 0;
	Kind     kind   = Kind::Start;
};

struct Pool {
	vk::QueryPool       pool = nullptr;
	std::vector<Record> records;
	uint64_t            last_tick = 0;
	uint64_t            flips     = 0;
	uint64_t            dropped   = 0;
};

struct Key {
	uint64_t a    = 0;
	uint64_t b    = 0;
	Kind     kind = Kind::Draw;

	bool operator==(const Key& other) const noexcept {
		return a == other.a && b == other.b && kind == other.kind;
	}
};

struct KeyHash {
	size_t operator()(const Key& key) const noexcept {
		return static_cast<size_t>(key.a * 0x9e3779b97f4a7c15ull ^ (key.b + 0x632be59bd9b4e019ull) * 0xbf58476d1ce4e5b9ull ^
		                           static_cast<uint64_t>(key.kind));
	}
};

struct Row {
	uint64_t count = 0;
	uint64_t ticks = 0;
};

struct Totals {
	uint64_t draw_ticks     = 0;
	uint64_t dispatch_ticks = 0;
	uint64_t outside_ticks  = 0;
	uint64_t gap_ticks      = 0;
	uint64_t draws          = 0;
	uint64_t dispatches     = 0;
	uint64_t buffers        = 0;
	uint64_t timestamps     = 0;
	uint64_t flips          = 0;
	uint64_t dropped        = 0;
	uint64_t unread         = 0;
	uint64_t backwards      = 0;
	uint64_t pools          = 0;
	uint64_t readback_ns    = 0;
	std::unordered_map<Key, Row, KeyHash> shaders;
	std::unordered_map<uint32_t, Row>     markers;
};

struct State {
	bool          init_done   = false;
	bool          usable      = false;
	vk::Device    device      = nullptr;
	vk::Semaphore semaphore   = nullptr;
	double        ns_per_tick = 1.0;
	uint64_t      valid_mask  = ~0ull;

	std::vector<Pool> pools;
	Pool*             current       = nullptr;
	uint32_t          marker        = 0;
	bool              lost          = false;
	uint64_t          pending_flips = 0;
	uint64_t          dropped       = 0;

	std::mutex                  mutex;
	std::condition_variable_any changed;
	std::deque<Pool*>           free;
	std::deque<Pool*>           closed;
	std::jthread                reader;

	std::mutex                                marker_mutex;
	std::unordered_map<std::string, uint32_t> marker_ids;
	std::vector<std::string>                  marker_names {"(none)", "(more names)"};
};

State& GetState() {
	static auto* state = new State;
	return *state;
}

std::string_view MarkerName(State& s, uint32_t id, std::string& storage) {
	std::scoped_lock lock(s.marker_mutex);
	storage = id < s.marker_names.size() ? s.marker_names[id] : std::string("(unknown)");
	return storage;
}

double Ms(const State& s, uint64_t ticks) {
	return static_cast<double>(ticks) * s.ns_per_tick / 1e6;
}

void Print(State& s, Totals& t, double seconds) {
	const uint64_t measured = t.draw_ticks + t.dispatch_ticks + t.outside_ticks;
	const double   flips    = static_cast<double>(std::max<uint64_t>(t.flips, 1));
	const double   total_ms = Ms(s, measured);
	const auto     share    = [&](uint64_t ticks) {
        return measured == 0 ? 0.0 : 100.0 * static_cast<double>(ticks) / static_cast<double>(measured);
	};
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-gpu-time] summary of the last {:.0f} s: {} flips; measured GPU time {:.1f} ms ({:.2f} ms per flip) in {} "
	    "command buffers: draws {:.2f} ms per flip ({:.0f} per flip), dispatches {:.2f} ms per flip ({:.0f} per flip), "
	    "outside draws and dispatches {:.2f} ms per flip ({:.1f} %: barriers, copies, clears, uploads, helper passes, "
	    "buffer ends); between command buffers {:.2f} ms per flip (not measured GPU time); {:.0f} timestamps per flip, "
	    "{} dropped, {} unread, {} backwards; {} pools read in {:.1f} ms\n",
	    seconds, t.flips, total_ms, total_ms / flips, t.buffers, Ms(s, t.draw_ticks) / flips,
	    static_cast<double>(t.draws) / flips, Ms(s, t.dispatch_ticks) / flips, static_cast<double>(t.dispatches) / flips,
	    Ms(s, t.outside_ticks) / flips, share(t.outside_ticks), Ms(s, t.gap_ticks) / flips,
	    static_cast<double>(t.timestamps) / flips, t.dropped, t.unread, t.backwards, t.pools,
	    static_cast<double>(t.readback_ns) / 1e6));

	std::vector<std::pair<Key, Row>> shaders(t.shaders.begin(), t.shaders.end());
	const auto by_time = [](const auto& x, const auto& y) { return x.second.ticks > y.second.ticks; };
	const auto top     = std::min(shaders.size(), TopRows);
	std::partial_sort(shaders.begin(), shaders.begin() + static_cast<ptrdiff_t>(top), shaders.end(), by_time);
	for (size_t i = 0; i < top; i++) {
		const auto& [key, row] = shaders[i];
		const auto  hashes = key.kind == Kind::Draw ? fmt::format("vs 0x{:016x} ps 0x{:016x}", key.a, key.b)
		                                            : fmt::format("cs 0x{:016x}", key.a);
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-gpu-time] shader {:2} {} {}: count {} ({:.1f} per flip), total {:.1f} ms ({:.2f} ms per flip), mean "
		    "{:.1f} us, share {:.1f} %\n",
		    i + 1, key.kind == Kind::Draw ? "draw" : "dispatch", hashes, row.count, static_cast<double>(row.count) / flips,
		    Ms(s, row.ticks), Ms(s, row.ticks) / flips, Ms(s, row.ticks) * 1000.0 / static_cast<double>(row.count),
		    share(row.ticks)));
	}

	const bool named = std::any_of(t.markers.begin(), t.markers.end(), [](const auto& row) { return row.first != 0; });
	if (!named) {
		::Log::WriteToConsoleAndLog("[kyty-gpu-time] markers: none of the timed work ran under a game debug marker\n");
		return;
	}
	std::vector<std::pair<uint32_t, Row>> markers(t.markers.begin(), t.markers.end());
	const auto marker_top = std::min(markers.size(), TopRows);
	std::partial_sort(markers.begin(), markers.begin() + static_cast<ptrdiff_t>(marker_top), markers.end(), by_time);
	std::string storage;
	for (size_t i = 0; i < marker_top; i++) {
		const auto& [id, row] = markers[i];
		::Log::WriteToConsoleAndLog(fmt::format(
		    "[kyty-gpu-time] marker {:2} \"{}\": {} draws and dispatches ({:.1f} per flip), total {:.1f} ms ({:.2f} ms "
		    "per flip, with the work before each), share {:.1f} %\n",
		    i + 1, MarkerName(s, id, storage), row.count, static_cast<double>(row.count) / flips, Ms(s, row.ticks),
		    Ms(s, row.ticks) / flips, share(row.ticks)));
	}
}

struct Reader {
	Totals                                totals;
	std::vector<uint64_t>                 stamps;
	uint64_t                              last       = 0;
	bool                                  in_buffer  = false;
	bool                                  have_end   = false;
	std::chrono::steady_clock::time_point window_start = std::chrono::steady_clock::now();
};

void Process(State& s, Reader& r, const Pool& pool) {
	auto& t = r.totals;
	t.pools++;
	t.flips += pool.flips;
	t.dropped += pool.dropped;
	const auto count = static_cast<uint32_t>(pool.records.size());
	if (count == 0) {
		return;
	}
	const auto started = std::chrono::steady_clock::now();
	r.stamps.resize(count);
	const auto result = s.device.getQueryPoolResults(pool.pool, 0, count, count * sizeof(uint64_t), r.stamps.data(),
	                                                 sizeof(uint64_t), vk::QueryResultFlagBits::e64);
	if (result != vk::Result::eSuccess) {
		t.unread += count;
		r.in_buffer = false;
		r.have_end  = false;
		return;
	}
	t.timestamps += count;
	for (uint32_t i = 0; i < count; i++) {
		const auto& record = pool.records[i];
		const auto  stamp  = r.stamps[i] & s.valid_mask;
		if (record.kind == Kind::Start) {
			if (r.have_end && record.a == 0 && stamp >= r.last) {
				t.gap_ticks += stamp - r.last;
			}
			t.buffers++;
			r.in_buffer = true;
			r.have_end  = false;
			r.last      = stamp;
			continue;
		}
		if (!r.in_buffer) {
			continue;
		}
		uint64_t delta = 0;
		if (stamp >= r.last) {
			delta = stamp - r.last;
		} else {
			t.backwards++;
		}
		r.last = stamp;
		switch (record.kind) {
			case Kind::Before: {
				t.outside_ticks += delta;
				t.markers[record.marker].ticks += delta;
				break;
			}
			case Kind::Draw:
			case Kind::Dispatch: {
				auto& row = t.shaders[Key {record.a, record.b, record.kind}];
				row.count++;
				row.ticks += delta;
				auto& marker = t.markers[record.marker];
				marker.count++;
				marker.ticks += delta;
				if (record.kind == Kind::Draw) {
					t.draws++;
					t.draw_ticks += delta;
				} else {
					t.dispatches++;
					t.dispatch_ticks += delta;
				}
				break;
			}
			case Kind::End: {
				t.outside_ticks += delta;
				r.in_buffer = false;
				r.have_end  = true;
				break;
			}
			case Kind::Start: break;
		}
	}
	t.readback_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started).count());
}

void ReaderThread(std::stop_token stop) {
	auto&  s = GetState();
	Reader r;
	for (;;) {
		Pool* pool = nullptr;
		{
			std::unique_lock lock(s.mutex);
			if (!s.changed.wait(lock, stop, [&] { return !s.closed.empty(); })) {
				return;
			}
			pool = s.closed.front();
		}
		for (;;) {
			uint64_t counter = 0;
			EXIT_NOT_IMPLEMENTED(s.device.getSemaphoreCounterValue(s.semaphore, &counter) != vk::Result::eSuccess);
			if (counter >= pool->last_tick) {
				break;
			}
			if (stop.stop_requested()) {
				return;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
		}
		Process(s, r, *pool);
		{
			std::scoped_lock lock(s.mutex);
			s.closed.pop_front();
			s.free.push_back(pool);
		}
		const auto now = std::chrono::steady_clock::now();
		if (now - r.window_start >= Window) {
			Print(s, r.totals, std::chrono::duration<double>(now - r.window_start).count());
			r.totals       = {};
			r.window_start = now;
		}
	}
}

bool Init(State& s, CommandScheduler& scheduler) {
	if (s.init_done) {
		return s.usable;
	}
	s.init_done         = true;
	auto&      graphics = scheduler.Graphics();
	const auto families = graphics.physical_device.getQueueFamilyProperties();
	const uint32_t bits =
	    graphics.queue_family < families.size() ? families[graphics.queue_family].timestampValidBits : 0u;
	if (bits == 0) {
		::Log::WriteToConsoleAndLog("[kyty-gpu-time] off: the graphics queue family has no timestamps\n");
		return false;
	}
	s.pools.resize(Pools);
	for (auto& pool: s.pools) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = PoolQueries;
		if (graphics.device.createQueryPool(&info, nullptr, &pool.pool) != vk::Result::eSuccess) {
			::Log::WriteToConsoleAndLog("[kyty-gpu-time] off: a timestamp query pool cannot be created\n");
			return false;
		}
		pool.records.reserve(PoolQueries);
		s.free.push_back(&pool);
	}
	s.device      = graphics.device;
	s.semaphore   = scheduler.GetMasterSemaphore().Handle();
	s.ns_per_tick = graphics.physical_device_properties.limits.timestampPeriod;
	s.valid_mask  = bits >= 64u ? ~0ull : ((1ull << bits) - 1u);
	s.usable      = true;
	s.reader      = std::jthread([](std::stop_token stop) { ReaderThread(stop); });
	::Log::WriteToConsoleAndLog(fmt::format(
	    "[kyty-gpu-time] on: a bottom-of-pipe timestamp at the start and end of each command buffer and before and after "
	    "each guest draw and dispatch, {} pools of {} queries, {:.3f} ns per tick, {} valid bits; a summary every {} s\n",
	    Pools, PoolQueries, s.ns_per_tick, bits, std::chrono::duration_cast<std::chrono::seconds>(Window).count()));
	return true;
}

void Mark(State& s, CommandBuffer& buffer, Kind kind, uint64_t a, uint64_t b) {
	auto* pool = s.current;
	if (pool == nullptr || pool->records.size() >= PoolQueries) {
		s.dropped++;
		s.lost = true;
		return;
	}
	const auto query  = static_cast<uint32_t>(pool->records.size());
	const auto handle = pool->pool;
	buffer.Handle().Custom([handle, query](vk::CommandBuffer command) {
		command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, handle, query);
	});
	pool->records.push_back({a, b, s.marker, kind});
}

void Close(State& s) {
	if (s.current == nullptr) {
		return;
	}
	s.current->dropped = std::exchange(s.dropped, 0);
	{
		std::scoped_lock lock(s.mutex);
		s.closed.push_back(s.current);
	}
	s.changed.notify_one();
	s.current = nullptr;
}

}

bool ReadSwitch() {
	const char*            value = std::getenv("KYTY_GPU_TIME");
	const std::string_view text {value == nullptr ? "" : value};
	if (!text.empty() && text != "0" && text != "1") {
		EXIT("invalid KYTY_GPU_TIME: %s (0 or 1)\n", value);
	}
	return text == "1";
}

void BeginBuffer(CommandScheduler& scheduler, CommandBuffer& buffer) {
	if (&scheduler.Context().GetCommandScheduler() != &scheduler) {
		return;
	}
	auto& s = GetState();
	if (!Init(s, scheduler)) {
		return;
	}
	if (s.current != nullptr && s.current->records.size() >= SwitchAfter) {
		Close(s);
	}
	if (s.current == nullptr) {
		{
			std::scoped_lock lock(s.mutex);
			if (!s.free.empty()) {
				s.current = s.free.front();
				s.free.pop_front();
			}
		}
		if (s.current == nullptr) {
			s.dropped++;
			s.lost = true;
			return;
		}
		s.current->records.clear();
		s.current->flips     = std::exchange(s.pending_flips, 0);
		s.current->last_tick = 0;
		s.current->dropped   = 0;
		const auto handle    = s.current->pool;
		buffer.Handle().Custom(
		    [handle](vk::CommandBuffer command) { command.resetQueryPool(handle, 0, PoolQueries); });
	}
	Mark(s, buffer, Kind::Start, std::exchange(s.lost, false) ? 1u : 0u, 0);
}

void EndBuffer(CommandScheduler& scheduler, CommandBuffer& buffer) {
	auto& s = GetState();
	if (s.current == nullptr || &scheduler.Context().GetCommandScheduler() != &scheduler) {
		return;
	}
	buffer.EndRendering();
	Mark(s, buffer, Kind::End, 0, 0);
	s.current->last_tick = scheduler.CurrentTick();
}

void BeforeWork(CommandBuffer& buffer) {
	auto& s = GetState();
	if (s.current != nullptr) {
		Mark(s, buffer, Kind::Before, 0, 0);
	}
}

void AfterDraw(CommandBuffer& buffer, uint64_t vertex_hash, uint64_t pixel_hash) {
	auto& s = GetState();
	if (s.current != nullptr) {
		Mark(s, buffer, Kind::Draw, vertex_hash, pixel_hash);
	}
}

void AfterDispatch(CommandBuffer& buffer, uint64_t compute_hash) {
	auto& s = GetState();
	if (s.current != nullptr) {
		Mark(s, buffer, Kind::Dispatch, compute_hash, 0);
	}
}

void NoteFlip() {
	auto& s = GetState();
	if (s.current != nullptr) {
		s.current->flips++;
	} else {
		s.pending_flips++;
	}
}

uint32_t InternMarker(const char* text, size_t max_bytes) {
	const std::string name(text, strnlen(text, max_bytes));
	auto&             s = GetState();
	std::scoped_lock  lock(s.marker_mutex);
	if (const auto found = s.marker_ids.find(name); found != s.marker_ids.end()) {
		return found->second;
	}
	if (s.marker_names.size() >= MaxMarkers) {
		return 1;
	}
	const auto id = static_cast<uint32_t>(s.marker_names.size());
	s.marker_names.push_back(name.empty() ? std::string("(empty)") : name);
	s.marker_ids.emplace(name, id);
	return id;
}

void SetMarker(uint32_t marker) {
	GetState().marker = marker;
}

}
