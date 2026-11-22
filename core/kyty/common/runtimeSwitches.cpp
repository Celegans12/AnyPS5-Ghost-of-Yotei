#include "common/runtimeSwitches.h"

#include "common/assert.h"

#include <fmt/format.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <utility>

namespace Common::RuntimeSwitches {

namespace {

struct Entry {
	const char* name;
	const char* off;
	const char* on;
};

constexpr Entry Entries[] = {
    {"KYTY_GENERIC_COMPUTE_VARIANT", "0", "1"},
    {"KYTY_E75_T1", "0", "1"},
};

static_assert(std::size(Entries) == SwitchCount);

bool Initial(size_t index) {
	const char* value = std::getenv(Entries[index].name);
	return value == nullptr || std::string_view {value} != Entries[index].off;
}

}

namespace Detail {
constinit std::atomic<bool> g_on[SwitchCount] {true};
}

namespace {

void Apply() {
	for (size_t index = 0; index < SwitchCount; ++index) {
		Detail::g_on[index].store(Initial(index), std::memory_order_relaxed);
	}
}

struct Environment {
	Environment() { Apply(); }
};

const Environment g_environment;

}

void Assign(Switch id, bool on) noexcept {
	Detail::g_on[static_cast<size_t>(id)].store(on, std::memory_order_relaxed);
}

const char* Name(Switch id) noexcept {
	return Entries[static_cast<size_t>(id)].name;
}

std::optional<Switch> Find(std::string_view name) noexcept {
	for (size_t index = 0; index < SwitchCount; ++index) {
		if (name == Entries[index].name) return static_cast<Switch>(index);
	}
	return std::nullopt;
}

std::string Value(Switch id, bool on) {
	const auto& entry = Entries[static_cast<size_t>(id)];
	return fmt::format("{}={}", entry.name, on ? entry.on : entry.off);
}

std::string FlippableList() {
	std::string list;
	for (const auto& entry: Entries) list += fmt::format("{}{}", list.empty() ? "" : ", ", entry.name);
	return list;
}

std::string Refusal(std::string_view name) {
	return fmt::format("KYTY_AB: {} cannot change while the title runs; settings KYTY_AB can change: {}", name,
	                   FlippableList());
}

void Reload() {
	Apply();
}

}

namespace Common::Ab {

namespace {

std::string_view Trim(std::string_view text) {
	while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) text.remove_prefix(1);
	while (!text.empty() &&
	       (text.back() == ' ' || text.back() == '\t' || text.back() == '\r' || text.back() == '\n')) {
		text.remove_suffix(1);
	}
	return text;
}

bool ParseSeconds(std::string_view text, std::string_view what, double& out, std::string& error) {
	const std::string copy(Trim(text));
	char*             end   = nullptr;
	const double      value = std::strtod(copy.c_str(), &end);
	if (copy.empty() || end != copy.c_str() + copy.size() || !std::isfinite(value) || value < 0) {
		error = fmt::format("{}: '{}' is not a number of seconds", what, copy);
		return false;
	}
	out = value;
	return true;
}

}

std::string Parse(std::string_view ab, std::string_view start, std::string_view settle, Config& config) {
	config = {};
	ab     = Trim(ab);
	std::string error;
	const auto  colon = ab.find(':');
	if (colon == std::string_view::npos) {
		return fmt::format("KYTY_AB: expected <seconds>:<setting>[+<setting>...], got '{}'", ab);
	}
	if (!ParseSeconds(ab.substr(0, colon), "KYTY_AB phase length", config.phase_seconds, error)) return error;
	if (config.phase_seconds < 1) return "KYTY_AB: a phase must last at least 1 s";
	auto list = ab.substr(colon + 1);
	if (Trim(list).empty()) return "KYTY_AB: no setting listed after the phase length";
	for (;;) {
		const auto plus = list.find('+');
		const auto name = Trim(list.substr(0, plus));
		if (name.empty()) return fmt::format("KYTY_AB: an empty setting name in '{}'", ab);
		const auto found = RuntimeSwitches::Find(name);
		if (!found.has_value()) return RuntimeSwitches::Refusal(name);
		if (std::ranges::find(config.switches, *found) != config.switches.end()) {
			return fmt::format("KYTY_AB: {} is listed twice", name);
		}
		config.switches.push_back(*found);
		if (plus == std::string_view::npos) break;
		list = list.substr(plus + 1);
	}
	start = Trim(start);
	if (!start.empty()) {
		if (start.back() == 's') {
			double seconds = 0;
			if (!ParseSeconds(start.substr(0, start.size() - 1), "KYTY_AB_START", seconds, error)) return error;
			config.start_seconds = seconds;
		} else {
			const std::string copy(start);
			char*             end = nullptr;
			errno                 = 0;
			const auto value      = std::strtoull(copy.c_str(), &end, 10);
			if (end != copy.c_str() + copy.size() || errno == ERANGE || copy.front() < '0' || copy.front() > '9') {
				return fmt::format("KYTY_AB_START: '{}' is neither a present index nor <n>s", copy);
			}
			config.start_present = value;
		}
	}
	if (!Trim(settle).empty() && !ParseSeconds(settle, "KYTY_AB_SETTLE", config.settle_seconds, error)) return error;
	if (config.settle_seconds >= config.phase_seconds) return "KYTY_AB_SETTLE must be shorter than the phase";
	return {};
}

Schedule::Schedule(Config config): m_config(std::move(config)) {}

Schedule::Step Schedule::AtPresent(uint64_t present, double seconds) {
	if (!m_first_seconds.has_value()) m_first_seconds = seconds;
	if (!m_started) {
		const bool due = m_config.start_seconds.has_value() ? seconds - *m_first_seconds >= *m_config.start_seconds
		                                                    : present >= m_config.start_present;
		if (!due) return {};
		m_started     = true;
		m_settled     = false;
		m_phase       = 0;
		m_phase_start = seconds;
		return {Event::Begin, 0, false};
	}
	if (seconds - m_phase_start >= m_config.phase_seconds) {
		++m_phase;
		m_settled     = false;
		m_phase_start = seconds;
		return {Event::Begin, m_phase, (m_phase & 1u) != 0};
	}
	if (!m_settled && seconds - m_phase_start >= m_config.settle_seconds) {
		m_settled = true;
		return {Event::Settled, m_phase, (m_phase & 1u) != 0};
	}
	return {};
}

Run::Run(Config config): m_schedule(std::move(config)) {
	for (const auto id: m_schedule.Settings().switches) m_environment.push_back(RuntimeSwitches::On(id));
}

std::string Run::Settings(bool b) const {
	std::string text;
	const auto& switches = m_schedule.Settings().switches;
	for (size_t index = 0; index < switches.size(); ++index) {
		text += fmt::format("{}{}", text.empty() ? "" : ", ",
		                    RuntimeSwitches::Value(switches[index], m_environment[index] != b));
	}
	return text;
}

std::vector<std::string> Run::AtPresent(uint64_t present, double seconds, double wall) {
	std::vector<std::string> lines;
	if (m_measure.open) {
		const double frame     = seconds - m_measure.last_seconds;
		m_measure.longest      = std::max(m_measure.longest, frame);
		m_measure.long_frames += frame >= 0.05 ? 1u : 0u;
		m_measure.last_seconds = seconds;
	}
	const auto step = m_schedule.AtPresent(present, seconds);
	if (step.event == Schedule::Event::None) return lines;
	const char arm = step.b ? 'B' : 'A';
	if (step.event == Schedule::Event::Settled) {
		m_measure = {.open          = true,
		             .phase         = step.phase,
		             .b             = step.b,
		             .first_present = present,
		             .first_seconds = seconds,
		             .first_wall    = wall,
		             .last_seconds  = seconds};
		lines.push_back(fmt::format("[kyty-ab] phase {} {} settled at present {} (t {:.3f} s, wall {:.3f})\n", arm,
		                            step.phase, present, seconds, wall));
		return lines;
	}
	if (m_measure.open) {
		const double length   = seconds - m_measure.first_seconds;
		const auto   presents = present - m_measure.first_present;
		lines.push_back(fmt::format(
		    "[kyty-ab] phase {} {} settled for {:.3f} s: {} presents, {:.2f} img/s, longest frame {:.1f} ms, {} frames of "
		    "50 ms or more; presents {} to {}, wall {:.3f} to {:.3f}\n",
		    m_measure.b ? 'B' : 'A', m_measure.phase, length, presents,
		    length > 0 ? static_cast<double>(presents) / length : 0.0, m_measure.longest * 1000.0, m_measure.long_frames,
		    m_measure.first_present, present, m_measure.first_wall, wall));
		m_measure.open = false;
	}
	const auto& config = m_schedule.Settings();
	if (step.phase == 0) {
		const auto from = config.start_seconds.has_value()
		                      ? fmt::format("{:.1f} s after the first present", *config.start_seconds)
		                      : fmt::format("present {}", config.start_present);
		lines.push_back(fmt::format("[kyty-ab] in-run A/B: phases of {:.1f} s from {}, settle {:.1f} s; A: {}; B: {}\n",
		                            config.phase_seconds, from, config.settle_seconds, Settings(false), Settings(true)));
	}
	for (size_t index = 0; index < config.switches.size(); ++index) {
		RuntimeSwitches::Assign(config.switches[index], m_environment[index] != step.b);
	}
	lines.push_back(fmt::format("[kyty-ab] phase {} {} begins at present {} (t {:.3f} s, wall {:.3f}): {}\n", arm,
	                            step.phase, present, seconds, wall, Settings(step.b)));
	return lines;
}

namespace {

const char* Variable(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr ? "" : value;
}

const Config& Parsed() {
	static const Config config = [] {
		Config     parsed;
		const auto error = Parse(Variable("KYTY_AB"), Variable("KYTY_AB_START"), Variable("KYTY_AB_SETTLE"), parsed);
		if (!error.empty()) EXIT("%s\n", error.c_str());
		return parsed;
	}();
	return config;
}

struct State {
	std::mutex                            mutex;
	std::optional<Run>                    run;
	uint64_t                              presents = 0;
	std::chrono::steady_clock::time_point first;
};

State& Current() {
	static State state;
	return state;
}

}

bool Configured() {
	static const bool configured = Variable("KYTY_AB")[0] != '\0';
	return configured;
}

bool Listed(RuntimeSwitches::Switch id) {
	return Configured() && std::ranges::find(Parsed().switches, id) != Parsed().switches.end();
}

void Start() {
	RuntimeSwitches::Reload();
	if (Configured()) static_cast<void>(Parsed());
}

std::vector<std::string> AtFlip() {
	if (!Configured()) return {};
	auto&           state = Current();
	std::lock_guard lock(state.mutex);
	const auto      now = std::chrono::steady_clock::now();
	if (!state.run.has_value()) {
		state.run.emplace(Parsed());
		state.first = now;
	}
	const double wall = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
	return state.run->AtPresent(++state.presents, std::chrono::duration<double>(now - state.first).count(), wall);
}

}
