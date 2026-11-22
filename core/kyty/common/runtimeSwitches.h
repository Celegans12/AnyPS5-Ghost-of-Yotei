#ifndef KYTY_COMMON_RUNTIME_SWITCHES_H_
#define KYTY_COMMON_RUNTIME_SWITCHES_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Common::RuntimeSwitches {

enum class Switch : uint32_t {
	GenericComputeVariant,
	Count
};

inline constexpr size_t SwitchCount = static_cast<size_t>(Switch::Count);

namespace Detail {
extern std::atomic<bool> g_on[SwitchCount];
}

[[nodiscard]] inline bool On(Switch id) noexcept {
	return Detail::g_on[static_cast<size_t>(id)].load(std::memory_order_relaxed);
}

void                                Assign(Switch id, bool on) noexcept;
[[nodiscard]] const char*           Name(Switch id) noexcept;
[[nodiscard]] std::optional<Switch> Find(std::string_view name) noexcept;
[[nodiscard]] std::string           Value(Switch id, bool on);
[[nodiscard]] std::string           FlippableList();
[[nodiscard]] std::string           Refusal(std::string_view name);
void                                Reload();

}

namespace Common::Ab {

struct Config {
	double                               phase_seconds  = 0;
	double                               settle_seconds = 2;
	uint64_t                             start_present  = 0;
	std::optional<double>                start_seconds;
	std::vector<RuntimeSwitches::Switch> switches;
};

[[nodiscard]] std::string Parse(std::string_view ab, std::string_view start, std::string_view settle, Config& config);

class Schedule {
public:
	enum class Event { None, Begin, Settled };

	struct Step {
		Event    event = Event::None;
		uint64_t phase = 0;
		bool     b     = false;
	};

	explicit Schedule(Config config);

	Step AtPresent(uint64_t present, double seconds);

	[[nodiscard]] const Config& Settings() const { return m_config; }

private:
	Config                m_config;
	bool                  m_started     = false;
	bool                  m_settled     = false;
	uint64_t              m_phase       = 0;
	double                m_phase_start = 0;
	std::optional<double> m_first_seconds;
};

class Run {
public:
	explicit Run(Config config);

	std::vector<std::string> AtPresent(uint64_t present, double seconds, double wall);

	[[nodiscard]] const Config& Settings() const { return m_schedule.Settings(); }

private:
	struct Measure {
		bool     open          = false;
		uint64_t phase         = 0;
		bool     b             = false;
		uint64_t first_present = 0;
		double   first_seconds = 0;
		double   first_wall    = 0;
		double   last_seconds  = 0;
		double   longest       = 0;
		uint64_t long_frames   = 0;
	};

	[[nodiscard]] std::string Settings(bool b) const;

	Schedule          m_schedule;
	std::vector<bool> m_environment;
	Measure           m_measure;
};

[[nodiscard]] bool                     Configured();
[[nodiscard]] bool                     Listed(RuntimeSwitches::Switch id);
void                                   Start();
[[nodiscard]] std::vector<std::string> AtFlip();

}

#endif
