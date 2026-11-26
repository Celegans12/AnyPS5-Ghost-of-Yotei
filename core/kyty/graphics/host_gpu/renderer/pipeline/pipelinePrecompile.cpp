#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#if defined(ANYPS5_KYTY)
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#endif
#include "graphics/presentation/precompileNotice.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <utility>

extern "C" void KernelPauseGuestClock_nid_no_patch();
extern "C" void KernelResumeGuestClock_nid_no_patch();

namespace Libs::Graphics {

namespace {

constexpr double SlowSeconds   = 0.5;
constexpr auto   NoticeLinger  = std::chrono::seconds(3);
constexpr auto     WatchPeriod = std::chrono::milliseconds(100);
constexpr auto     LogPeriod   = std::chrono::seconds(10);
constexpr auto     StallShown  = std::chrono::seconds(3);
constexpr auto     StallGap    = std::chrono::seconds(1);
constexpr auto     ManifestBarDelay  = std::chrono::seconds(1);
constexpr auto     ManifestBarLinger = std::chrono::seconds(2);

template <typename... Args>
void PrecompileLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

struct Estimate {
	uint32_t pending  = 0;
	uint64_t finished = 0;
	uint32_t shaders  = 0;
	uint64_t shaders_finished = 0;
	uint32_t running  = 0;
};

}

PipelineCache::GuestClockHold::GuestClockHold() {
	KernelPauseGuestClock_nid_no_patch();
}

PipelineCache::GuestClockHold::~GuestClockHold() {
	KernelResumeGuestClock_nid_no_patch();
}

PipelineCache::StallScope::StallScope(PipelineCache& cache): StallScope(cache, true) {}

PipelineCache::StallScope::StallScope(PipelineCache& cache, bool hold_clock): m_cache(cache) {
	if (hold_clock) {
		m_clock_hold.emplace();
	}
	std::scoped_lock lock(cache.m_stall_mutex);
	if (cache.m_stall.threads++ == 0) {
		cache.m_stall.since = std::chrono::steady_clock::now();
	}
}

#if defined(ANYPS5_KYTY)
PipelineCache::StallScope::StallScope(PipelineCache& cache, const char* what, uint64_t hash, const char* why,
                                      bool hold_clock)
    : StallScope(cache, hold_clock) {
	PipelineRouteNote() = nullptr;
	m_started = std::chrono::steady_clock::now();
	m_what    = what;
	m_hash    = hash;
	m_why     = why;
}

static const char* ComputeWrites(const ShaderModule& module) {
	const auto& program = module.info;
	if (program.has_address_writes ||
	    std::ranges::any_of(program.info.buffers, [](const auto& buffer) { return buffer.written; })) {
		return "writes-buffers";
	}
	if (std::ranges::any_of(program.info.images, [](const auto& image) { return image.written; })) {
		return "writes-images";
	}
	return "writes-nothing";
}

PipelineCache::StallScope::StallScope(PipelineCache& cache, const char* what, const ShaderModule& module)
    : StallScope(cache, what, module.info.shader_hash, ComputeWrites(module)) {}
#endif

PipelineCache::StallScope::~StallScope() {
#if defined(ANYPS5_KYTY)
	const auto now = std::chrono::steady_clock::now();
	uint64_t   ns  = 0;
	{
		std::scoped_lock lock(m_cache.m_stall_mutex);
		if (--m_cache.m_stall.threads == 0) {
			m_cache.m_stall.ended = now;
		}
		if (m_what != nullptr) {
			ns           = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - m_started).count());
			auto& stats = m_cache.m_stall_stats;
			stats.count++;
			stats.total_ns += ns;
			stats.max_ns = std::max(stats.max_ns, ns);
			stats.over_100ms += ns >= 100'000'000u ? 1u : 0u;
			stats.over_500ms += ns >= 500'000'000u ? 1u : 0u;
		}
	}
	static const uint64_t log_ns = [] {
		const char* value = std::getenv("KYTY_PIPELINE_STALL_LOG_MS");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 100u) * 1'000'000u;
	}();
	if (m_what != nullptr && ns >= log_ns) {
		const char* route = PipelineRouteNote();
		PrecompileLog("[kyty-gpl] stall {:.3f} s {} 0x{:016x} {}{}{} @{}", static_cast<double>(ns) / 1e9, m_what,
		              m_hash, route != nullptr ? route : "", m_why != nullptr ? " why=" : "",
		              m_why != nullptr ? m_why : "", now.time_since_epoch().count());
	}
#else
	std::scoped_lock lock(m_cache.m_stall_mutex);
	if (--m_cache.m_stall.threads == 0) {
		m_cache.m_stall.ended = std::chrono::steady_clock::now();
	}
#endif
}

#if defined(ANYPS5_KYTY)
void PipelineCache::LogStallStats(bool force) {
	const std::array<uint64_t, 3> speculative {m_speculative_failed.load(std::memory_order_relaxed),
	                                           m_speculative_refused.load(std::memory_order_relaxed),
	                                           m_speculative_skipped.load(std::memory_order_relaxed)};
	if (force || speculative != m_speculative_logged) {
		m_speculative_logged = speculative;
		PrecompileLog("[kyty-spec-compile] compiles ahead failed {} and refused before compiling {}, each kept for its "
		              "variant; {} later compiles ahead skipped as kept",
		              speculative[0], speculative[1], speculative[2]);
	}
	StallStats stats;
	{
		std::scoped_lock lock(m_stall_mutex);
		m_stall_stats.pipelines = m_pipelines_created.load(std::memory_order_relaxed);
		stats                   = m_stall_stats;
		if (!force && stats.count == m_stall_logged.count && stats.pipelines == m_stall_logged.pipelines) {
			return;
		}
		m_stall_logged = stats;
	}
	PrecompileLog("[kyty-gpl] stalls {} total {:.3f} s max {:.3f} s >=0.1s {} >=0.5s {}; driver pipelines {}",
	              stats.count, static_cast<double>(stats.total_ns) / 1e9, static_cast<double>(stats.max_ns) / 1e9,
	              stats.over_100ms, stats.over_500ms, stats.pipelines);
	if (!m_async_threads.empty()) {
		const auto waits = m_async_first_waits.load(std::memory_order_relaxed);
		PrecompileLog("[kyty-async] mode {}: draws skipped kyty {} occlusion {} small-target {} small-program {} "
		              "layer {}; first draws waited {} times for {:.3f} s, {} ready in time; probes {} for {:.3f} s, "
		              "{} created at once; flips {}",
		              Config::AsyncPipelinesSkipMore() ? 2 : 1, m_async_skipped[0].load(std::memory_order_relaxed),
		              m_async_skipped[1].load(std::memory_order_relaxed),
		              m_async_skipped[2].load(std::memory_order_relaxed),
		              m_async_skipped[3].load(std::memory_order_relaxed),
		              m_async_skipped[4].load(std::memory_order_relaxed), waits,
		              static_cast<double>(m_async_first_wait_ns.load(std::memory_order_relaxed)) / 1e9,
		              m_async_first_ready.load(std::memory_order_relaxed),
		              m_async_probes.load(std::memory_order_relaxed),
		              static_cast<double>(m_async_probe_ns.load(std::memory_order_relaxed)) / 1e9,
		              m_async_created_now.load(std::memory_order_relaxed),
		              m_render != nullptr ? m_render->FlipNumber() : 0u);
	}
	if (m_libraries != nullptr) {
		m_libraries->LogStats();
	}
	if (!m_chain_threads.empty()) {
		const auto prepared = m_chain_prepared.load(std::memory_order_relaxed);
		const auto failed   = m_chain_failed.load(std::memory_order_relaxed);
		const auto runs     = prepared + failed;
		PrecompileLog("[kyty-gpl] chains queued {} dropped {} prepared {} failed {} modules compiled {} ({:.1f} ms avg)",
		              m_chain_queued.load(std::memory_order_relaxed), m_chain_dropped.load(std::memory_order_relaxed),
		              prepared, failed, m_chain_compiled.load(std::memory_order_relaxed),
		              runs != 0 ? static_cast<double>(m_chain_ns.load(std::memory_order_relaxed)) / 1e6 /
		                              static_cast<double>(runs)
		                        : 0.0);
	}
}
#endif

PipelineCache::PipelineStall PipelineCache::GetPipelineStall() const {
	std::scoped_lock lock(m_stall_mutex);
	return m_stall;
}

void PipelineCache::StartPrecompile() {
	m_precompile = std::jthread([this](const std::stop_token& stop) { Precompile(stop); });
}

void PipelineCache::Precompile(const std::stop_token& stop) {
	KYTY_PROFILER_THREAD("Thread_PipelinePrecompile");
	using Clock = std::chrono::steady_clock;

	const auto estimate = [this] {
		const auto backlog = GetManifestBacklog();
		Estimate   e;
		e.pending          = backlog.pipelines;
		e.finished         = backlog.pipelines_done;
		e.shaders          = backlog.modules;
		e.shaders_finished = backlog.modules_done;
		e.running          = backlog.running;
		return e;
	};

	Clock::time_point                      slow_until;
	bool                                   stalled    = false;
	Clock::time_point                      stall_since;
	Clock::time_point                      stall_logged;
#if defined(ANYPS5_KYTY)
	uint64_t                               stall_created = 0;
	uint64_t                               stall_modules = 0;
	uint64_t                               stall_waits   = 0;
#else
	uint64_t                               stall_base = 0;
#endif
	bool                                   shown       = false;
	uint64_t                               notice_base = 0;
	Clock::time_point                      notice_shown_at;
	bool                                   bar_shown = false;
	std::optional<Clock::time_point>       bar_work_since;
	Clock::time_point                      bar_last_work;
	uint64_t                               bar_base = 0;
	const auto seconds = [](Clock::duration duration) { return std::chrono::duration<double>(duration).count(); };

	std::mutex                  sleep_mutex;
	std::condition_variable_any sleeper;
	const auto                  sleep = [&](Clock::duration period) {
		std::unique_lock lock(sleep_mutex);
		sleeper.wait_for(lock, stop, period, [] { return false; });
	};
#if defined(ANYPS5_KYTY)
	Clock::time_point stats_logged = Clock::now();
#endif
	while (!stop.stop_requested()) {
		sleep(WatchPeriod);
#if defined(ANYPS5_KYTY)
		if (m_libraries != nullptr) {
			m_libraries->Tick();
		}
		if (Clock::now() - stats_logged >= LogPeriod) {
			stats_logged = Clock::now();
			LogStallStats(false);
		}
#endif
		const auto e     = estimate();
		const auto now   = Clock::now();
		const auto stall = GetPipelineStall();
		const auto left  = std::max(e.pending, stall.threads);
		const auto creations = GetDriverCreations(SlowSeconds);
		const bool waits_slow = stall.threads != 0 && now - stall.since >= std::chrono::duration<double>(SlowSeconds);
		const uint32_t slow = creations.slow;
		if (slow != 0 || waits_slow) {
			slow_until = now + NoticeLinger;
		}
		if (!stalled && stall.threads != 0 && now - stall.since >= StallShown) {
			stalled      = true;
			stall_since  = stall.since;
			stall_logged = now;
#if defined(ANYPS5_KYTY)
			stall_created = m_pipelines_created.load(std::memory_order_relaxed);
			stall_modules = m_chain_compiled.load(std::memory_order_relaxed);
			{
				std::scoped_lock lock(m_stall_mutex);
				stall_waits = m_stall_stats.count;
			}
#else
			stall_base = e.finished;
#endif
			PrecompileLog("Pipeline stall: the game has waited {:.1f} s for the pipelines the driver compiles "
			              "({} left)",
			              seconds(now - stall_since), left);
		} else if (stalled && stall.threads == 0 && now - stall.ended >= StallGap) {
			stalled = false;
#if defined(ANYPS5_KYTY)
			const uint64_t created = m_pipelines_created.load(std::memory_order_relaxed) - stall_created;
			const uint64_t modules = m_chain_compiled.load(std::memory_order_relaxed) - stall_modules;
			uint64_t       waits   = 0;
			{
				std::scoped_lock lock(m_stall_mutex);
				waits = m_stall_stats.count - stall_waits;
			}
			PrecompileLog("Pipeline stall: the game goes on after {:.1f} s ({} compiles finished meanwhile: {} "
			              "pipelines created by the driver, {} shader modules built by the chains; {} waits ended)",
			              seconds(stall.ended - stall_since), created + modules, created, modules, waits);
#else
			PrecompileLog("Pipeline stall: the game goes on after {:.1f} s ({} pipelines compiled meanwhile)",
			              seconds(stall.ended - stall_since), e.finished - stall_base);
#endif
		} else if (stalled && now - stall_logged >= LogPeriod) {
			stall_logged = now;
			PrecompileLog("Pipeline stall: {} pipelines left at {:.0f} s", left, seconds(now - stall_since));
		}
		if (m_manifest_warm) {
			const uint64_t finished = e.finished + e.shaders_finished;
			const uint32_t left     = e.pending + e.running + e.shaders;
			if (left != 0) {
				bar_last_work = now;
				if (!bar_shown && !bar_work_since) {
					bar_work_since = now;
					bar_base       = finished;
				}
			} else if (!bar_shown) {
				bar_work_since.reset();
			}
			if (!bar_shown && bar_work_since && now - *bar_work_since >= ManifestBarDelay) {
				bar_shown = true;
				PrecompileLog("Pipeline manifest bar: shown ({} pipelines left, {} shaders to translate)",
				              e.pending + e.running, e.shaders);
			} else if (bar_shown && now - bar_last_work >= ManifestBarLinger) {
				bar_shown = false;
				PrecompileLog("Pipeline manifest bar: hidden after {:.1f} s ({} shaders and pipelines done)",
				              seconds(now - *bar_work_since), finished - std::min(bar_base, finished));
				bar_work_since.reset();
			}
			if (bar_shown) {
				const auto done = static_cast<uint32_t>(finished - std::min(bar_base, finished));
				ShowManifestProgress({.done      = done,
				                      .total     = done + left,
				                      .pipelines = e.pending + e.running,
				                      .shaders   = e.shaders});
			} else {
				HideManifestProgress();
			}
		}

		std::optional<PrecompileNoticeState> state;
		if (stalled || waits_slow) {
			state = PrecompileNoticeState::Waiting;
		} else if (now < slow_until) {
			state = PrecompileNoticeState::Compiling;
		}
		if (!state) {
			if (shown) {
				shown = false;
				HidePrecompileNotice();
				PrecompileLog("Pipeline notice: hidden after {:.1f} s, {:.1f} s after the last compilation over {:.1f} s",
				              seconds(now - notice_shown_at), seconds(now - (slow_until - NoticeLinger)), SlowSeconds);
			}
			continue;
		}
		if (!shown) {
			shown           = true;
			notice_base     = e.finished;
			notice_shown_at = now;
			PrecompileLog("Pipeline notice: shown, state {} ({} driver creations for {:.1f} s, {} pipelines pending, "
			              "{} game threads waiting)",
			              *state == PrecompileNoticeState::Waiting ? "waiting" : "compiling", creations.running,
			              creations.longest, e.pending, stall.threads);
		}
		const auto done = static_cast<uint32_t>(e.finished - notice_base);
		ShowPrecompileNotice({.visible          = true,
		                      .state            = *state,
		                      .done             = done,
		                      .total            = done + left,
		                      .shaders          = *state == PrecompileNoticeState::Waiting ? 0u : e.shaders,
		                      .waited_pipelines = *state == PrecompileNoticeState::Compiling ? slow : stall.threads,
		                      .seconds_waited   = *state == PrecompileNoticeState::Compiling
		                                              ? static_cast<uint32_t>(creations.longest)
		                                              : static_cast<uint32_t>(seconds(now - (stall.threads != 0 ? stall.since : stall_since)))});
	}
	HidePrecompileNotice();
	HideManifestProgress();
#if defined(ANYPS5_KYTY)
	LogStallStats(true);
#endif
}

}
