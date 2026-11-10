#include "graphics/guest_gpu/command_processor/passCensus.h"

#include "common/liveAb.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/debugSample.h"

#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::PassCensus {

namespace {

struct Totals {
	uint64_t ns         = 0; // GPU-thread time, the card waits taken out
	uint64_t wait_ns    = 0;
	uint64_t draws      = 0;
	uint64_t dispatches = 0;
	uint64_t skipped    = 0; // packets the ceiling left out
	std::map<uint64_t, uint64_t> shaders; // compute shader hash -> dispatches (NoteShader)
};

struct ProcessorState {
	std::vector<std::string> stack;
	std::string              key = "-";
	bool                     free = false; // the pass is listed by the ceiling
};

struct Census {
	std::mutex                              mutex;
	std::unordered_map<std::string, Totals> passes;
	std::chrono::steady_clock::time_point   window_start = std::chrono::steady_clock::now();
	uint32_t                                window_frame = 0;
};

Census& Global() {
	static Census c;
	return c;
}

// The GPU thread's state: per command processor its marker stack; the running processor's pass and since when.
thread_local std::unordered_map<const void*, ProcessorState> t_processors;
thread_local const void*                                   t_active      = nullptr;
thread_local std::string                                   t_active_key  = "-";
thread_local std::chrono::steady_clock::time_point         t_since       = {};
thread_local uint64_t                                      t_since_wait  = 0;

const std::vector<std::string>& FreeList() {
	static const std::vector<std::string> list = [] {
		std::vector<std::string> out;
		const char*              value = std::getenv("KYTY_DIAG_CEILING_FREE_PASSES");
		if (value == nullptr) return out;
		const std::string text(value);
		size_t            start = 0;
		while (start <= text.size()) {
			const auto end = std::min(text.find(',', start), text.size());
			if (end > start) out.push_back(text.substr(start, end - start));
			start = end + 1;
		}
		if (!out.empty()) std::fprintf(stderr, "DIAG_CEILING_FREE_PASSES APPROXIMATION list=%s\n", value);
		return out;
	}();
	return list;
}

bool CeilingOn() {
	static const bool listed = Common::LiveAb::Listed(Common::LiveAb::Switch::DiagCeilingFreePasses);
	static const uint32_t fixed = [] {
		const char* value = std::getenv("KYTY_DIAG_CEILING_FREE_PASSES_ON");
		return value != nullptr && value[0] == '1' ? 1u : 0u;
	}();
	if (FreeList().empty() || (fixed == 0u && !listed)) return false;
	return Common::LiveAb::Gate(Common::LiveAb::Switch::DiagCeilingFreePasses, fixed) != 0u;
}

void Report(Census& c) {
	const uint32_t frame  = DebugPresentCount().load(std::memory_order_relaxed);
	const auto     now    = std::chrono::steady_clock::now();
	if (now - c.window_start < std::chrono::seconds(5)) return;
	const uint32_t frames = frame > c.window_frame ? frame - c.window_frame : 1u;
	std::vector<std::pair<std::string, Totals>> rows(c.passes.begin(), c.passes.end());
	uint64_t total = 0;
	for (const auto& [k, t]: rows) total += t.ns;
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	std::fprintf(stderr, "PASS_CENSUS frames=%u gpu_thread_ms/frame=%.1f passes=%zu\n", frames,
	             static_cast<double>(total) / 1e6 / frames, rows.size());
	for (size_t i = 0; i < rows.size(); i++) { // every pass (3.11: comparing the passes of two game versions)
		const auto& [k, t] = rows[i];
		std::fprintf(stderr, "PASS_CENSUS pass=\"%s\" ms/frame=%.2f share=%.1f%% wait_ms/frame=%.2f draws/frame=%.0f "
		             "dispatches/frame=%.0f skipped/frame=%.0f\n",
		             k.c_str(), static_cast<double>(t.ns) / 1e6 / frames, total != 0 ? 100.0 * t.ns / total : 0.0,
		             static_cast<double>(t.wait_ns) / 1e6 / frames, static_cast<double>(t.draws) / frames,
		             static_cast<double>(t.dispatches) / frames, static_cast<double>(t.skipped) / frames);
		if (!t.shaders.empty()) {
			std::string list;
			for (const auto& [hash, n]: t.shaders) list += fmt::format("{}{:016x}:{}", list.empty() ? "" : ",", hash, n);
			std::fprintf(stderr, "PASS_CENSUS shaders pass=\"%s\" cs=%s\n", k.c_str(), list.c_str());
		}
	}
	c.passes.clear();
	c.window_start = now;
	c.window_frame = frame;
}

// The time since the last event goes to the pass that was running; `cp` (with `key`) runs from now.
void Switch(const void* cp, const std::string& key) {
	const auto now  = std::chrono::steady_clock::now();
	const auto wait = t_wait_ns;
	if (t_since != std::chrono::steady_clock::time_point {} && Enabled()) {
		const auto     ns      = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_since).count());
		const uint64_t waited  = wait - t_since_wait;
		auto&          c       = Global();
		std::scoped_lock lock(c.mutex);
		auto&          t       = c.passes[t_active_key];
		t.ns += ns > waited ? ns - waited : 0u;
		t.wait_ns += waited;
		Report(c);
	}
	t_since      = now;
	t_since_wait = wait;
	t_active     = cp;
	t_active_key = key;
}

} // namespace

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_TRACE_PASS_CENSUS");
		const bool  o     = value != nullptr && value[0] == '1';
		if (o) std::fprintf(stderr, "TRACE_PASS_CENSUS enabled\n");
		return o;
	}();
	return on;
}

static bool Active() {
	static const bool active = Enabled() || !FreeList().empty();
	return active;
}

void NoteShader(uint64_t hash) {
	if (!Enabled()) return;
	auto&            c = Global();
	std::scoped_lock lock(c.mutex);
	c.passes[t_active_key].shaders[hash]++;
}

void Marker(const void* cp, const char* push_name) {
	if (!Active()) return; // any thread that runs PM4 (the command processor thread since the GPU pipeline project)
	auto& state = t_processors[cp];
	if (push_name != nullptr) {
		state.stack.emplace_back(push_name);
	} else if (!state.stack.empty()) {
		state.stack.pop_back();
	}
	const auto n = state.stack.size();
	state.key    = n == 0 ? std::string("-") : n == 1 ? state.stack.back() : state.stack[n - 2] + ">" + state.stack.back();
	state.free   = false;
	for (const auto& s: FreeList()) {
		if (state.key.find(s) != std::string::npos) state.free = true;
	}
	Switch(cp, state.key);
}

bool Packet(const void* cp, bool dispatch) {
	if (!Active()) return false;
	auto& state = t_processors[cp];
	if (t_active != cp) Switch(cp, state.key);
	const bool skip = state.free && CeilingOn();
	// KYTY_DIAG_CEILING_FREE_BURN_NS=<ns> (the ceiling with the remainder):
	// a left-out packet still costs the GPU thread that much (busy wait) - the in-order work a parallel translation
	// keeps per draw.
	static const uint64_t burn_ns = [] {
		const char* value = std::getenv("KYTY_DIAG_CEILING_FREE_BURN_NS");
		const auto  ns    = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		if (ns != 0u) std::fprintf(stderr, "DIAG_CEILING_FREE_BURN_NS %llu\n", ns);
		return static_cast<uint64_t>(ns);
	}();
	if (skip && burn_ns != 0u) {
		const auto until = std::chrono::steady_clock::now() + std::chrono::nanoseconds(burn_ns);
		while (std::chrono::steady_clock::now() < until) {}
	}
	if (Enabled()) {
		auto&            c = Global();
		std::scoped_lock lock(c.mutex);
		auto&            t = c.passes[state.key];
		(dispatch ? t.dispatches : t.draws)++;
		if (skip) t.skipped++;
	}
	return skip;
}

} // namespace Libs::Graphics::PassCensus
