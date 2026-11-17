#include "graphics/host_gpu/renderer/pipeline/programLock.h"

#if defined(ANYPS5_KYTY)

#include "common/profiler.h"
#include "common/spscQueue.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <functional>
#include <mutex>
#include <pthread.h>
#include <ranges>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics::ProgramLock {

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint32_t MaxThreads    = 64;
constexpr uint32_t MaxSites      = 48;
constexpr uint32_t SpinRounds    = 256;
constexpr uint64_t SlowReadNs    = 20'000;
constexpr size_t   ReportedSites = 24;

struct Counters {
	std::atomic<uint64_t> taken {0};
	std::atomic<uint64_t> contended {0};
	std::atomic<uint64_t> wait_ns {0};
	std::atomic<uint64_t> hold_ns {0};
	std::atomic<uint64_t> max_wait_ns {0};
	std::atomic<uint64_t> max_hold_ns {0};
	std::atomic<uint64_t> blocked_ns {0};
};

struct ReadCounters {
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> slow {0};
	std::atomic<uint64_t> max_ns {0};
};

struct Site {
	const char* file     = nullptr;
	const char* function = nullptr;
	uint32_t    line     = 0;
};

struct Stats {
	std::array<std::array<Counters, MaxSites>, MaxThreads>       counters;
	std::array<std::array<ReadCounters, 2>, MaxThreads>          reads;
	std::array<std::array<std::atomic<uint64_t>, 2>, MaxThreads> names {};
	std::atomic<uint32_t>                                        thread_count {0};
	std::array<Site, MaxSites>                                   sites {};
	std::atomic<uint32_t>                                        site_count {0};
	std::mutex                                                   site_mutex;
	std::atomic<uint32_t>                                        holder {0};
};

std::chrono::seconds Period() {
	static const auto period = [] {
		const char* value   = std::getenv("KYTY_LOCK_STATS_PERIOD");
		const auto  seconds = value != nullptr ? std::strtoul(value, nullptr, 10) : 0ul;
		return std::chrono::seconds(seconds != 0 ? seconds : 5);
	}();
	return period;
}

Stats& GetStats() {
	static auto* stats = new Stats;
	return *stats;
}

thread_local uint32_t          t_row       = UINT32_MAX;
thread_local uint32_t          t_held_site = 0;
thread_local Clock::time_point t_held_since {};
thread_local bool              t_holding   = false;
thread_local std::string       t_shader_line;

uint64_t Ns(Clock::duration duration) {
	return static_cast<uint64_t>(std::max<int64_t>(
	    0, std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()));
}

void NoteMax(std::atomic<uint64_t>& max, uint64_t value) {
	if (value > max.load(std::memory_order_relaxed)) {
		max.store(value, std::memory_order_relaxed);
	}
}

uint32_t Row() {
	if (t_row != UINT32_MAX) {
		return t_row;
	}
	auto&      stats = GetStats();
	const auto index = stats.thread_count.fetch_add(1, std::memory_order_relaxed);
	t_row            = std::min(index, MaxThreads - 1);
	std::array<char, 16> name {};
	if (index >= MaxThreads - 1) {
		std::snprintf(name.data(), name.size(), "others");
	} else if (pthread_getname_np(pthread_self(), name.data(), name.size()) != 0 || name[0] == '\0') {
		std::snprintf(name.data(), name.size(), "thread%u", index);
	}
	std::array<uint64_t, 2> words {};
	std::memcpy(words.data(), name.data(), sizeof(words));
	stats.names[t_row][0].store(words[0], std::memory_order_relaxed);
	stats.names[t_row][1].store(words[1], std::memory_order_relaxed);
	return t_row;
}

uint32_t SiteIndex(const std::source_location& where) {
	auto&      stats = GetStats();
	const auto known = stats.site_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < known; i++) {
		if (stats.sites[i].line == where.line() && stats.sites[i].file == where.file_name() &&
		    stats.sites[i].function == where.function_name()) {
			return i;
		}
	}
	std::scoped_lock lock(stats.site_mutex);
	const auto       count = stats.site_count.load(std::memory_order_relaxed);
	for (uint32_t i = known; i < count; i++) {
		if (stats.sites[i].line == where.line() && stats.sites[i].file == where.file_name() &&
		    stats.sites[i].function == where.function_name()) {
			return i;
		}
	}
	if (count >= MaxSites) {
		return MaxSites - 1;
	}
	stats.sites[count] = {.file = where.file_name(), .function = where.function_name(), .line = where.line()};
	stats.site_count.store(count + 1, std::memory_order_release);
	return count;
}

std::string SiteName(const Site& site) {
	std::string_view file     = site.file != nullptr ? site.file : "?";
	std::string_view function = site.function != nullptr ? site.function : "?";
	file                      = file.substr(file.find_last_of("/\\") + 1);
	auto name                 = function.substr(0, function.find('('));
	name                      = name.substr(name.find_last_of(": ") + 1);
	std::string result(name);
	if (const auto open = function.find(" [", function.find('(')); open != std::string_view::npos) {
		auto argument = function.substr(open + 2);
		argument      = argument.substr(0, argument.find_first_of(",]"));
		argument      = argument.substr(argument.find_last_of(": ") + 1);
		result += "<" + std::string(argument) + ">";
	}
	return fmt::format("{} {}:{}", result, file, site.line);
}

std::string ThreadName(const Stats& stats, uint32_t row) {
	std::array<uint64_t, 2> words {stats.names[row][0].load(std::memory_order_relaxed),
	                               stats.names[row][1].load(std::memory_order_relaxed)};
	std::array<char, 17>    name {};
	std::memcpy(name.data(), words.data(), sizeof(words));
	return name[0] != '\0' ? std::string(name.data()) : fmt::format("thread{}", row);
}

void Report(std::chrono::nanoseconds interval) {
	struct Line {
		std::string thread;
		uint32_t    site = 0;
		uint64_t    taken = 0, contended = 0, wait = 0, hold = 0, max_wait = 0, max_hold = 0, blocked = 0;
	};
	struct Reads {
		std::string thread;
		std::array<uint64_t, 2> count {}, ns {}, slow {}, max {};
	};
	static auto* previous = new std::array<std::array<std::array<uint64_t, 5>, MaxSites>, MaxThreads> {};
	static auto* previous_reads = new std::array<std::array<std::array<uint64_t, 3>, 2>, MaxThreads> {};
	auto&        stats          = GetStats();
	const auto   threads        = std::min(stats.thread_count.load(std::memory_order_relaxed), MaxThreads);
	const auto   sites          = std::min(stats.site_count.load(std::memory_order_acquire), MaxSites);
	std::vector<Line>  lines;
	std::vector<Reads> reads;
	for (uint32_t row = 0; row < threads; row++) {
		const auto name = ThreadName(stats, row);
		for (uint32_t site = 0; site < sites; site++) {
			auto&       counters = stats.counters[row][site];
			auto&       before   = (*previous)[row][site];
			const std::array<uint64_t, 5> now {counters.taken.load(std::memory_order_relaxed),
			                                   counters.contended.load(std::memory_order_relaxed),
			                                   counters.wait_ns.load(std::memory_order_relaxed),
			                                   counters.hold_ns.load(std::memory_order_relaxed),
			                                   counters.blocked_ns.load(std::memory_order_relaxed)};
			Line line {.thread    = name,
			           .site      = site,
			           .taken     = now[0] - before[0],
			           .contended = now[1] - before[1],
			           .wait      = now[2] - before[2],
			           .hold      = now[3] - before[3],
			           .max_wait  = counters.max_wait_ns.exchange(0, std::memory_order_relaxed),
			           .max_hold  = counters.max_hold_ns.exchange(0, std::memory_order_relaxed),
			           .blocked   = now[4] - before[4]};
			before = now;
			if (line.taken == 0 && line.wait == 0 && line.blocked == 0) {
				continue;
			}
			const auto same = std::ranges::find_if(lines, [&](const Line& other) {
				return other.thread == line.thread && other.site == line.site;
			});
			if (same == lines.end()) {
				lines.push_back(std::move(line));
				continue;
			}
			same->taken += line.taken;
			same->contended += line.contended;
			same->wait += line.wait;
			same->hold += line.hold;
			same->blocked += line.blocked;
			same->max_wait = std::max(same->max_wait, line.max_wait);
			same->max_hold = std::max(same->max_hold, line.max_hold);
		}
		for (uint32_t locked = 0; locked < 2; locked++) {
			auto&       counters = stats.reads[row][locked];
			auto&       before   = (*previous_reads)[row][locked];
			const std::array<uint64_t, 3> now {counters.count.load(std::memory_order_relaxed),
			                                   counters.ns.load(std::memory_order_relaxed),
			                                   counters.slow.load(std::memory_order_relaxed)};
			const auto max = counters.max_ns.exchange(0, std::memory_order_relaxed);
			if (now[0] == before[0]) {
				continue;
			}
			auto same = std::ranges::find_if(reads, [&](const Reads& other) { return other.thread == name; });
			if (same == reads.end()) {
				reads.push_back({.thread = name});
				same = reads.end() - 1;
			}
			same->count[locked] += now[0] - before[0];
			same->ns[locked] += now[1] - before[1];
			same->slow[locked] += now[2] - before[2];
			same->max[locked] = std::max(same->max[locked], max);
			before            = now;
		}
	}
	const double interval_ms = static_cast<double>(interval.count()) / 1e6;
	const auto   ms          = [](uint64_t ns) { return static_cast<double>(ns) / 1e6; };
	const auto   percent     = [&](uint64_t ns) { return interval_ms > 0.0 ? 100.0 * ms(ns) / interval_ms : 0.0; };
	uint64_t     taken = 0, contended = 0, wait = 0, hold = 0;
	for (const auto& line: lines) {
		taken += line.taken;
		contended += line.contended;
		wait += line.wait;
		hold += line.hold;
	}
	std::string text = fmt::format(
	    "[lock-stats] {:.1f} s, m_program_lock ({} mode): taken {}, contended {:.2f} %, held {:.1f} % of the time, "
	    "waits {:.1f} ms in total\n",
	    interval_ms / 1000.0, Narrow() ? "narrow" : "wide", taken,
	    taken != 0 ? 100.0 * static_cast<double>(contended) / static_cast<double>(taken) : 0.0, percent(hold), ms(wait));
	std::vector<std::pair<std::string, std::array<uint64_t, 3>>> per_thread;
	for (const auto& line: lines) {
		auto same = std::ranges::find_if(per_thread, [&](const auto& other) { return other.first == line.thread; });
		if (same == per_thread.end()) {
			per_thread.push_back({line.thread, {}});
			same = per_thread.end() - 1;
		}
		same->second[0] += line.wait;
		same->second[1] += line.hold;
		same->second[2] += line.taken;
	}
	std::ranges::sort(per_thread, std::ranges::greater {}, [](const auto& entry) { return entry.second[0] + entry.second[1]; });
	for (const auto& [thread, values]: per_thread) {
		text += fmt::format("[lock-stats]   thread {:<16} waited {:9.2f} ms ({:5.1f} %)  held {:9.2f} ms ({:5.1f} %)  taken {}\n",
		                    thread, ms(values[0]), percent(values[0]), ms(values[1]), percent(values[1]), values[2]);
	}
	std::ranges::sort(lines, std::ranges::greater {}, [](const Line& line) { return line.wait + line.hold + line.blocked; });
	for (size_t i = 0; i < lines.size() && i < ReportedSites; i++) {
		const auto& line = lines[i];
		text += fmt::format(
		    "[lock-stats]   site {:<16} {:<52} taken {:>8} contended {:5.1f} %  wait {:9.2f} ms (max {:8.3f})  "
		    "held {:9.2f} ms (max {:8.3f})  others waited {:9.2f} ms\n",
		    line.thread, SiteName(stats.sites[line.site]), line.taken,
		    line.taken != 0 ? 100.0 * static_cast<double>(line.contended) / static_cast<double>(line.taken) : 0.0,
		    ms(line.wait), ms(line.max_wait), ms(line.hold), ms(line.max_hold), ms(line.blocked));
	}
	for (const auto& read: reads) {
		text += fmt::format(
		    "[lock-stats]   guest reads {:<16} under the lock: {} reads {:.2f} ms, {} over {} us, max {:.3f} ms | "
		    "outside: {} reads {:.2f} ms, {} over {} us, max {:.3f} ms\n",
		    read.thread, read.count[1], ms(read.ns[1]), read.slow[1], SlowReadNs / 1000, ms(read.max[1]), read.count[0],
		    ms(read.ns[0]), read.slow[0], SlowReadNs / 1000, ms(read.max[0]));
	}
	std::fputs(text.c_str(), stderr);
	std::fflush(stderr);
}

void StartReporter() {
	static std::once_flag once;
	std::call_once(once, [] {
		std::thread([] {
			::Profiler::SetThreadName("Thread_LockStats");
			auto last = Clock::now();
			for (;;) {
				std::this_thread::sleep_for(Period());
				const auto now = Clock::now();
				Report(now - last);
				last = now;
			}
		}).detach();
	});
}

void Backoff(uint32_t& rounds) {
	if (rounds < SpinRounds) {
		rounds++;
		for (uint32_t i = 0; i < 4; i++) {
			Common::SpinPause();
		}
		return;
	}
	std::this_thread::yield();
}

void SpinAcquire(std::atomic_flag& flag) {
	uint32_t rounds = 0;
	while (flag.test_and_set(std::memory_order_acquire)) {
		while (flag.test(std::memory_order_relaxed)) {
			Backoff(rounds);
		}
	}
}

void Blame(Stats& stats, uint32_t holder, uint64_t ns) {
	if (holder == 0 || ns == 0) {
		return;
	}
	stats.counters[(holder >> 8u) - 1u][holder & 0xffu].blocked_ns.fetch_add(ns, std::memory_order_relaxed);
}

}

bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LOCK_STATS");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

bool Narrow() {
	static const bool narrow = [] {
		const char* value = std::getenv("KYTY_PROGRAM_LOCK");
		return value == nullptr || std::string_view(value) != "wide";
	}();
	return narrow;
}

void DeferLine(std::string&& line) {
	t_shader_line += line;
}

void Acquire(std::atomic_flag& flag, const std::source_location& where) {
	if (!Enabled()) {
		SpinAcquire(flag);
		return;
	}
	StartReporter();
	auto&      stats    = GetStats();
	const auto row      = Row();
	const auto site     = SiteIndex(where);
	auto&      counters = stats.counters[row][site];
	auto       start    = Clock::now();
	if (flag.test_and_set(std::memory_order_acquire)) {
		auto     holder  = stats.holder.load(std::memory_order_relaxed);
		auto     segment = start;
		uint32_t rounds  = 0;
		do {
			while (flag.test(std::memory_order_relaxed)) {
				Backoff(rounds);
				if (const auto current = stats.holder.load(std::memory_order_relaxed); current != holder) {
					const auto now = Clock::now();
					Blame(stats, holder, Ns(now - segment));
					holder  = current;
					segment = now;
				}
			}
		} while (flag.test_and_set(std::memory_order_acquire));
		const auto acquired = Clock::now();
		Blame(stats, holder, Ns(acquired - segment));
		const auto waited = Ns(acquired - start);
		counters.contended.fetch_add(1, std::memory_order_relaxed);
		counters.wait_ns.fetch_add(waited, std::memory_order_relaxed);
		NoteMax(counters.max_wait_ns, waited);
		start = acquired;
	}
	counters.taken.fetch_add(1, std::memory_order_relaxed);
	stats.holder.store(((row + 1u) << 8u) | site, std::memory_order_relaxed);
	t_held_site  = site;
	t_held_since = start;
	t_holding    = true;
}

void Release(std::atomic_flag& flag) {
	if (t_holding) {
		auto&      stats    = GetStats();
		const auto held     = Ns(Clock::now() - t_held_since);
		auto&      counters = stats.counters[t_row][t_held_site];
		counters.hold_ns.fetch_add(held, std::memory_order_relaxed);
		NoteMax(counters.max_hold_ns, held);
		stats.holder.store(0, std::memory_order_relaxed);
		t_holding = false;
	}
	flag.clear(std::memory_order_release);
	if (!t_shader_line.empty()) {
		std::fputs(t_shader_line.c_str(), stdout);
		t_shader_line.clear();
	}
}

bool TimedGuestRead(uint64_t address, std::span<uint32_t> values) {
	const auto start  = Clock::now();
	const bool result = Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
	const auto ns     = Ns(Clock::now() - start);
	auto&      reads  = GetStats().reads[Row()][t_holding ? 1u : 0u];
	reads.count.fetch_add(1, std::memory_order_relaxed);
	reads.ns.fetch_add(ns, std::memory_order_relaxed);
	if (ns >= SlowReadNs) {
		reads.slow.fetch_add(1, std::memory_order_relaxed);
	}
	NoteMax(reads.max_ns, ns);
	return result;
}

}

#endif
