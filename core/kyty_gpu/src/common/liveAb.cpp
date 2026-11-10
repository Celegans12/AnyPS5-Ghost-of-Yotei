#include "common/liveAb.h"

#include "common/onceValue.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

namespace Common::LiveAb {
namespace {

// The switches that may toggle at a frame boundary, and why each one is exact there (the call that reads the gate
// decides everything with the one value it read, and nothing it leaves behind depends on that value):
//  - KYTY_PERF_DIRECT_RANGES: read at every buffer lookup; a flip to off ends the active direct ranges on the GPU
//    thread (Finish, the buffer deleted, the pages tracked again from guest memory), a flip to on starts them at their
//    next use. The value only switches on/off; the ranges come from the environment.
struct SwitchInfo {
	Switch      id;
	const char* name;
	uint32_t    max_value; // the largest value that is still a per-call choice
};

constexpr SwitchInfo Switches[] = {
    {Switch::DirectRanges, "KYTY_PERF_DIRECT_RANGES", 1},
    // Measurement only (APPROXIMATION): whether the compute shaders listed in KYTY_DIAG_SKIP_CS_LIVE are dispatched;
    // per dispatch, nothing outlives it (their outputs simply stay as they were).
    {Switch::SkipCsLive, "KYTY_DIAG_SKIP_CS_LIVE", 1},
    // Measurement only: extra GPU-thread work at the place KYTY_DIAG_GPU_BURN_AT names (ns).
    {Switch::GpuBurn, "KYTY_DIAG_GPU_BURN_NS", 100000},
    // Only which of two exact paths the listed shaders take per load (KYTY_PERF_INDIRECT_LOAD_FAST=3 compiles both and
    // reads the live shader flags word, rewritten at frame boundaries; indirectLoadFast.h).
    {Switch::IndirectLoadFast, "KYTY_PERF_INDIRECT_LOAD_FAST", 1},
    // D4 step K1: only how an image bound for sampling, storage or rendering takes the bytes that other images over its
    // memory wrote after it (phase 1: through guest memory, exactly; phase 0: KYTY_WORKAROUND_ALIAS_SYNC's rectangle copy).
    // Both keep the content serials, so either phase continues from what the other left; read once per bind.
    {Switch::SharedImageMemory, "KYTY_FIX_SHARED_IMAGE_MEMORY", 1},
    // Measurement only (APPROXIMATION): whether draw and dispatch packets inside the listed marker passes return before the
    // renderer (guest_gpu/command_processor/passCensus.cpp; the list in KYTY_DIAG_CEILING_FREE_PASSES).
    {Switch::DiagCeilingFreePasses, "KYTY_DIAG_CEILING_FREE_PASSES", 1},
};
static_assert(std::size(Switches) == SwitchCount);

// Switches known to change persistent state or shader translation: refused with the reason.
struct RefusedInfo {
	const char* name;
	const char* reason;
};

constexpr RefusedInfo Refused[] = {
    {"KYTY_PERF_DCC_GPU_ALL", "device features and GPU-write observers are set up at start"},
};

struct Entry {
	Switch      id;
	std::string label; // phase name
	uint32_t    value; // its value in its own phase (and in "all")
};

struct Config {
	std::vector<Entry> entries;
	uint64_t           listed_mask[3] = {}; // bit i of word i / 64: switch i is listed
	uint32_t           period_ms      = 5000;
	uint32_t           start_frame    = 1100;
	uint32_t           warmup_frames  = 1;
	uint32_t           summary_cycles = 2;
	bool               all_phase      = false;
	bool               only_all       = false; // KYTY_LIVE_AB_ONLY_ALL: phases off and all only
	// KYTY_LIVE_AB_OFF_START=1: in the off phase a listed switch has its start value (environment/default), not 0: compares
	// the default with the listed value for switches whose 0 is not their default (thresholds, times).
	bool               off_start      = false;
	bool               alternate      = true;
	uint32_t           phase_count    = 0;
};

std::string Trim(const std::string& text) {
	const auto first = text.find_first_not_of(" \t");
	if (first == std::string::npos) return {};
	const auto last = text.find_last_not_of(" \t");
	return text.substr(first, last - first + 1);
}

uint32_t ReadUint(const char* name, uint32_t fallback, uint32_t min, uint32_t max) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return fallback;
	const auto parsed = std::strtoull(value, nullptr, 10);
	return static_cast<uint32_t>(std::clamp<unsigned long long>(parsed, min, max));
}

Config Parse() {
	Config      config;
	const char* list = std::getenv("KYTY_LIVE_AB");
	if (list == nullptr || list[0] == '\0') return config;
	const std::string text(list);
	size_t            start = 0;
	while (start <= text.size()) {
		size_t end = text.find(',', start);
		if (end == std::string::npos) end = text.size();
		const std::string item = Trim(text.substr(start, end - start));
		start                  = end + 1;
		if (item.empty()) continue;
		std::string name  = item;
		uint32_t    value = 1;
		if (const auto equals = item.find('='); equals != std::string::npos) {
			name  = Trim(item.substr(0, equals));
			value = static_cast<uint32_t>(std::strtoul(item.c_str() + equals + 1, nullptr, 10));
		}
		for (auto& c: name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
		if (name.rfind("KYTY_", 0) != 0) name = "KYTY_" + name;
		const auto* info = std::ranges::find_if(Switches, [&](const SwitchInfo& s) { return name == s.name; });
		const char* reason = nullptr;
		if (info == std::end(Switches)) {
			const auto* refused = std::ranges::find_if(Refused, [&](const RefusedInfo& r) { return name == r.name; });
			reason              = refused != std::end(Refused)     ? refused->reason
			                      : name.rfind("KYTY_DIAG_", 0) == 0 ? "DIAG switches are approximations or keep state"
			                      : name.rfind("KYTY_WORKAROUND_", 0) == 0
			                          ? "WORKAROUND switches are approximations or keep state"
			                          : "not audited for live toggling";
		} else if (value == 0) {
			reason = "=0 is the off phase already";
		} else if (value > info->max_value) {
			reason = "this value is not a per-call mode of the switch";
		}
		const uint32_t word = info != std::end(Switches) ? static_cast<uint32_t>(info->id) / 64u : 0u;
		const uint64_t bit  = info != std::end(Switches) ? uint64_t {1} << (static_cast<uint32_t>(info->id) % 64u) : 0u;
		if (reason == nullptr && (config.listed_mask[word] & bit) != 0) reason = "listed twice";
		if (reason != nullptr) {
			std::fprintf(stderr, "LIVE_AB refused %s: %s; it keeps its start value\n", item.c_str(), reason);
			continue;
		}
		config.listed_mask[word] |= bit;
		config.entries.push_back({info->id, value == 1 ? name : name + "=" + std::to_string(value), value});
	}
	if (config.entries.empty()) {
		std::fprintf(stderr, "LIVE_AB disabled: KYTY_LIVE_AB names no switch that can toggle live\n");
		return config;
	}
	config.period_ms      = ReadUint("KYTY_LIVE_AB_PERIOD_MS", 5000, 100, 600000);
	config.start_frame    = ReadUint("KYTY_LIVE_AB_START_FRAME", 1100, 1, 0x7fffffff);
	config.warmup_frames  = ReadUint("KYTY_LIVE_AB_WARMUP_FRAMES", 1, 0, 1000);
	config.summary_cycles = ReadUint("KYTY_LIVE_AB_SUMMARY_CYCLES", 2, 1, 1000);
	config.all_phase      = ReadUint("KYTY_LIVE_AB_ALL", 0, 0, 1) != 0 && config.entries.size() > 1;
	config.only_all       = ReadUint("KYTY_LIVE_AB_ONLY_ALL", 0, 0, 1) != 0 && config.entries.size() > 1;
	config.off_start      = ReadUint("KYTY_LIVE_AB_OFF_START", 0, 0, 1) != 0;
	const char* order     = std::getenv("KYTY_LIVE_AB_ORDER");
	config.alternate      = order == nullptr || std::string(order) != "rotate";
	config.phase_count    = config.only_all ? 2u : 1u + static_cast<uint32_t>(config.entries.size()) + (config.all_phase ? 1u : 0u);
	std::string phases    = "off";
	if (!config.only_all) {
		for (const auto& entry: config.entries) phases += "," + entry.label;
	}
	if (config.all_phase || config.only_all) phases += ",all";
	std::fprintf(stderr,
	             "LIVE_AB enabled phases=%s period_ms=%u start_frame=%u warmup_frames=%u summary_cycles=%u order=%s\n",
	             phases.c_str(), config.period_ms, config.start_frame, config.warmup_frames, config.summary_cycles,
	             config.alternate ? "alternate" : "rotate");
	return config;
}

const Config& GetConfig() {
	static constinit Common::OnceObject<Config> config_once;
	return config_once.Get([] { return Parse(); });
}

uint64_t ListedMask(uint32_t word) {
	static constinit Common::OnceValue<uint64_t> mask0_once;
	static constinit Common::OnceValue<uint64_t> mask1_once;
	static constinit Common::OnceValue<uint64_t> mask2_once;
	return word == 0   ? mask0_once.Get([] { return GetConfig().listed_mask[0]; })
	       : word == 1 ? mask1_once.Get([] { return GetConfig().listed_mask[1]; })
	                   : mask2_once.Get([] { return GetConfig().listed_mask[2]; });
}

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// One phase of one cycle.
struct PhaseResult {
	bool               valid     = false;
	uint32_t           frames    = 0; // presented frames measured
	double             ms        = 0; // their wall time
	double             fps       = 0;
	double             busy_ms   = 0;
	double             wait_ms   = 0;
	std::vector<float> frame_ms; // one entry per measured frame boundary
};

// One phase over all counted cycles.
struct PhaseTotals {
	std::vector<double> cycle_fps;
	std::vector<double> cycle_vs_off; // (fps / fps of "off" in the same cycle - 1) * 100
	std::vector<float>  frame_ms;
	uint64_t            frames  = 0;
	double              ms      = 0;
	double              busy_ms = 0;
	double              wait_ms = 0;
};

// GPU thread only.
struct State {
	bool     started       = false;
	uint32_t last_frame    = 0;
	uint32_t cycle         = 0;
	uint32_t position      = 0; // place of the running phase in its cycle
	uint32_t phase         = 0; // the running phase
	int64_t  phase_start   = 0;
	int64_t  last_boundary = 0;
	uint32_t boundaries    = 0; // frame boundaries since the running phase started
	// The measured part of the running phase.
	uint32_t           frames = 0;
	double             ms     = 0;
	std::vector<float> frame_ms;
	// GPU-thread time since the measurement started (BusyScope, AddFenceWait).
	bool    in_busy    = false;
	int64_t busy_begin = 0;
	int64_t busy_ns    = 0;
	int64_t wait_ns    = 0;

	std::vector<PhaseResult> cycle_results; // per phase, the running cycle
	std::vector<PhaseTotals> totals;        // per phase, all counted cycles
	uint32_t                 counted_cycles = 0;
	uint32_t                 skipped_cycles = 0; // a phase measured fewer than 2 frame boundaries
};

State& GetState() {
	static State* state = new State; // never destroyed: the GPU thread may still run during exit
	return *state;
}

std::string PhaseName(const Config& config, uint32_t phase) {
	if (phase == 0) return "off";
	if (config.only_all) return "all";
	if (phase <= config.entries.size()) return config.entries[phase - 1].label;
	return "all";
}

uint32_t PhaseAt(const Config& config, uint32_t cycle, uint32_t position) {
	return config.alternate && (cycle & 1u) != 0 ? config.phase_count - 1u - position : position;
}

void SetLiveValues(const Config& config, uint32_t phase) {
	const auto all = (config.all_phase || config.only_all) && phase == config.phase_count - 1u;
	for (size_t i = 0; i < config.entries.size(); ++i) {
		const auto& entry = config.entries[i];
		const bool  on    = phase != 0 && (all || phase == i + 1u);
		// 0 stored: the gate's start value (KYTY_LIVE_AB_OFF_START); otherwise the live value plus one.
		const uint32_t stored = on ? entry.value + 1u : config.off_start ? 0u : 1u;
		g_live_values[static_cast<uint32_t>(entry.id)].store(stored, std::memory_order_relaxed);
	}
}

void ResetMeasure(State& state) {
	state.frames = 0;
	state.ms     = 0;
	state.frame_ms.clear();
	state.busy_ns = 0;
	state.wait_ns = 0;
}

void StartPhase(State& state, const Config& config, int64_t now) {
	state.phase = PhaseAt(config, state.cycle, state.position);
	SetLiveValues(config, state.phase);
	state.phase_start   = now;
	state.last_boundary = now;
	state.boundaries    = 0;
	ResetMeasure(state);
}

double Median(std::vector<float> values) {
	if (values.empty()) return 0;
	const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
	std::nth_element(values.begin(), middle, values.end());
	if ((values.size() & 1u) != 0) return *middle;
	const auto lower = *std::max_element(values.begin(), middle);
	return (static_cast<double>(lower) + *middle) / 2.0;
}

void MeanSd(const std::vector<double>& values, double& mean, double& sd) {
	mean = 0;
	sd   = 0;
	if (values.empty()) return;
	for (const auto v: values) mean += v;
	mean /= static_cast<double>(values.size());
	if (values.size() < 2) return;
	double sum = 0;
	for (const auto v: values) sum += (v - mean) * (v - mean);
	sd = std::sqrt(sum / static_cast<double>(values.size() - 1));
}

void EndPhase(State& state, const Config& config, uint32_t frame) {
	PhaseResult result;
	result.frames  = state.frames;
	result.ms      = state.ms;
	result.fps     = state.ms > 0 ? state.frames * 1000.0 / state.ms : 0;
	result.busy_ms = static_cast<double>(state.busy_ns) / 1e6;
	result.wait_ms = static_cast<double>(state.wait_ns) / 1e6;
	result.valid   = state.frame_ms.size() >= 2;
	const auto   intervals = state.frame_ms.size();
	const double ft_mean   = intervals != 0 ? result.ms / static_cast<double>(intervals) : 0.0;
	std::fprintf(stderr,
	             "PERF live_ab phase=%s cycle=%u frames=%u ms=%.0f fps=%.3f gpu_thread_busy_ms=%.0f fence_wait_ms=%.0f "
	             "ft_mean_ms=%.1f ft_median_ms=%.1f frame=%u\n",
	             PhaseName(config, state.phase).c_str(), state.cycle, result.frames, result.ms, result.fps,
	             result.busy_ms, result.wait_ms, ft_mean, Median(state.frame_ms), frame);
	result.frame_ms = std::move(state.frame_ms);
	state.frame_ms  = {};
	state.cycle_results[state.phase] = std::move(result);
}

void PrintSummary(const State& state, const Config& config) {
	std::fprintf(stderr, "PERF live_ab summary cycles=%u skipped=%u period_ms=%u warmup_frames=%u order=%s\n",
	             state.counted_cycles, state.skipped_cycles, config.period_ms, config.warmup_frames,
	             config.alternate ? "alternate" : "rotate");
	if (state.counted_cycles == 0) return;
	for (uint32_t phase = 0; phase < config.phase_count; ++phase) {
		const auto& totals = state.totals[phase];
		double      fps_mean = 0, fps_sd = 0, vs_mean = 0, vs_sd = 0;
		MeanSd(totals.cycle_fps, fps_mean, fps_sd);
		MeanSd(totals.cycle_vs_off, vs_mean, vs_sd);
		const auto n      = static_cast<double>(totals.cycle_vs_off.size());
		const auto frames = static_cast<double>(totals.frames);
		std::fprintf(stderr,
		             "PERF live_ab summary phase=%s cycles=%zu fps_mean=%.3f fps_sd=%.3f fps_pooled=%.3f "
		             "ft_mean_ms=%.1f ft_median_ms=%.1f busy_ms_per_frame=%.1f fence_wait_ms_per_frame=%.1f "
		             "vs_off_pct=%+.2f vs_off_se_pct=%.2f\n",
		             PhaseName(config, phase).c_str(), totals.cycle_fps.size(), fps_mean, fps_sd,
		             totals.ms > 0 ? frames * 1000.0 / totals.ms : 0.0,
		             totals.frame_ms.empty() ? 0.0 : totals.ms / static_cast<double>(totals.frame_ms.size()),
		             Median(totals.frame_ms), frames > 0 ? totals.busy_ms / frames : 0.0,
		             frames > 0 ? totals.wait_ms / frames : 0.0, vs_mean, n > 0 ? vs_sd / std::sqrt(n) : 0.0);
	}
}

void EndCycle(State& state, const Config& config) {
	const bool valid = std::ranges::all_of(state.cycle_results, [](const PhaseResult& r) { return r.valid; });
	if (valid) {
		const double off = state.cycle_results[0].fps;
		for (uint32_t phase = 0; phase < config.phase_count; ++phase) {
			auto&       totals = state.totals[phase];
			const auto& result = state.cycle_results[phase];
			totals.cycle_fps.push_back(result.fps);
			totals.cycle_vs_off.push_back(off > 0 ? (result.fps / off - 1.0) * 100.0 : 0.0);
			totals.frame_ms.insert(totals.frame_ms.end(), result.frame_ms.begin(), result.frame_ms.end());
			totals.frames += result.frames;
			totals.ms += result.ms;
			totals.busy_ms += result.busy_ms;
			totals.wait_ms += result.wait_ms;
		}
		state.counted_cycles++;
	} else {
		state.skipped_cycles++;
	}
	state.cycle_results.assign(config.phase_count, {});
	if ((state.counted_cycles + state.skipped_cycles) % config.summary_cycles == 0) {
		PrintSummary(state, config);
	}
}

} // namespace

bool Enabled() {
	return ListedMask(0) != 0 || ListedMask(1) != 0 || ListedMask(2) != 0;
}

bool Listed(Switch id) {
	const auto index = static_cast<uint32_t>(id);
	return ((ListedMask(index / 64u) >> (index % 64u)) & 1u) != 0u;
}

void OnFrameBoundary(uint32_t frame) {
	if (!Enabled()) return;
	const auto& config = GetConfig();
	auto&       state  = GetState();
	// A flip whose request was cancelled presented nothing.
	if (frame == state.last_frame) return;
	const uint32_t presented = frame - state.last_frame;
	state.last_frame         = frame;
	const int64_t now        = NowNs();
	if (state.in_busy) {
		// The slice running now: its part up to here belongs to the frame that ends here.
		state.busy_ns += now - state.busy_begin;
		state.busy_begin = now;
	}
	if (!state.started) {
		if (frame < config.start_frame) {
			state.busy_ns = 0;
			state.wait_ns = 0;
			return;
		}
		state.started = true;
		state.cycle_results.assign(config.phase_count, {});
		state.totals.assign(config.phase_count, {});
		StartPhase(state, config, now);
		return;
	}
	const double interval_ms = static_cast<double>(now - state.last_boundary) / 1e6;
	state.last_boundary      = now;
	state.boundaries++;
	if (state.boundaries > config.warmup_frames) {
		state.frames += presented;
		state.ms += interval_ms;
		state.frame_ms.push_back(static_cast<float>(interval_ms));
	} else if (state.boundaries == config.warmup_frames) {
		ResetMeasure(state); // the warm-up frames end here
	}
	if (now - state.phase_start < static_cast<int64_t>(config.period_ms) * 1000000) return;
	EndPhase(state, config, frame);
	if (++state.position == config.phase_count) {
		EndCycle(state, config);
		state.position = 0;
		state.cycle++;
	}
	StartPhase(state, config, now);
}

void AddFenceWait(uint64_t nanoseconds) {
	if (!Enabled()) return;
	GetState().wait_ns += static_cast<int64_t>(nanoseconds);
}

BusyScope::BusyScope() {
	if (!Enabled()) return;
	m_on        = true;
	auto& state = GetState();
	state.in_busy    = true;
	state.busy_begin = NowNs();
}

BusyScope::~BusyScope() {
	if (!m_on) return;
	auto& state = GetState();
	state.busy_ns += NowNs() - state.busy_begin;
	state.in_busy = false;
}

} // namespace Common::LiveAb
