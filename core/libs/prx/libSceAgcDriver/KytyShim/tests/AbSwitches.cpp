#include "common/runtimeSwitches.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace Ab       = Common::Ab;
namespace Switches = Common::RuntimeSwitches;
namespace IR       = Libs::Graphics::ShaderRecompiler::IR;
using Switches::Switch;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		++g_failures;
	}
}

void SetVariable(const char* name, const char* value) {
#if defined(_WIN32)
	_putenv_s(name, value == nullptr ? "" : value);
#else
	if (value == nullptr) {
		unsetenv(name);
	} else {
		setenv(name, value, 1);
	}
#endif
}

bool Contains(const std::string& text, std::string_view part) {
	return text.find(part) != std::string::npos;
}

bool AnyContains(const std::vector<std::string>& lines, std::string_view part) {
	for (const auto& line: lines) {
		if (Contains(line, part)) return true;
	}
	return false;
}

std::string Refused(const char* ab, const char* start = "", const char* settle = "") {
	Ab::Config config;
	return Ab::Parse(ab, start, settle, config);
}

void Refusals() {
	const std::string listed =
	    "settings KYTY_AB can change: KYTY_GENERIC_COMPUTE_VARIANT, KYTY_E2_VSHARP_MIRROR";
	Check(Contains(Refused("KYTY_GENERIC_COMPUTE_VARIANT"), "expected <seconds>:<setting>"),
	      "a value without a phase length is refused");
	Check(Contains(Refused("0.5:KYTY_GENERIC_COMPUTE_VARIANT"), "at least 1 s"), "a phase under 1 s is refused");
	Check(Contains(Refused("x:KYTY_GENERIC_COMPUTE_VARIANT"), "KYTY_AB phase length: 'x' is not a number of seconds"),
	      "a phase length that is not a number is refused");
	Check(Contains(Refused("30: "), "no setting listed"), "a list without a setting is refused");
	Check(Contains(Refused("30:KYTY_GENERIC_COMPUTE_VARIANT+"), "an empty setting name"), "an empty name is refused");
	Check(Contains(Refused("30:KYTY_GENERIC_COMPUTE_VARIANT+KYTY_GENERIC_COMPUTE_VARIANT"),
	               "KYTY_GENERIC_COMPUTE_VARIANT is listed twice"),
	      "a setting listed twice is refused");
	const auto relaxed = Refused("30:KYTY_RELAXED_SHADER_FLOAT");
	Check(Contains(relaxed, "KYTY_AB: KYTY_RELAXED_SHADER_FLOAT cannot change while the title runs; " + listed),
	      "a setting that changes how shaders are compiled is refused with the list");
	const auto labels = Refused("30:KYTY_GENERIC_COMPUTE_VARIANT+KYTY_PROTO_TRUE_LABELS");
	Check(Contains(labels, "KYTY_AB: KYTY_PROTO_TRUE_LABELS cannot change while the title runs; " + listed),
	      "a setting that is not a runtime switch is refused with the list");
	const auto removed = Refused("30:KYTY_FLAT_DEFER");
	Check(Contains(removed, "KYTY_AB: KYTY_FLAT_DEFER cannot change while the title runs; " + listed),
	      "a setting that no longer exists is refused with the list");
	Check(Contains(Refused("30:KYTY_GENERIC_COMPUTE_VARIANT", "", "30"), "KYTY_AB_SETTLE must be shorter than the phase"),
	      "a settle as long as the phase is refused");
	Check(Contains(Refused("30:KYTY_GENERIC_COMPUTE_VARIANT", "", "-1"), "KYTY_AB_SETTLE: '-1' is not a number of seconds"),
	      "a negative settle is refused");
	Check(Contains(Refused("30:KYTY_GENERIC_COMPUTE_VARIANT", "x"),
	               "KYTY_AB_START: 'x' is neither a present index nor <n>s"),
	      "a start that is neither a present nor seconds is refused");
	Check(Refused(" 30 : KYTY_GENERIC_COMPUTE_VARIANT ", "12s", "4").empty(), "spaces around the name are accepted");
	Ab::Config config;
	Check(Ab::Parse("45:KYTY_GENERIC_COMPUTE_VARIANT", "12s", "", config).empty() && config.start_seconds == 12.0 &&
	          config.settle_seconds == 2.0 && config.phase_seconds == 45.0,
	      "a start in seconds and the default settle");
	Check(Ab::Parse("45:KYTY_GENERIC_COMPUTE_VARIANT", "300", "5", config).empty() && config.start_present == 300 &&
	          !config.start_seconds.has_value() && config.settle_seconds == 5.0,
	      "a start at a present index");
}

void Defaults() {
	Check(Switches::On(Switch::GenericComputeVariant), "KYTY_GENERIC_COMPUTE_VARIANT is on without the variable");
	Check(IR::GenericComputeVariants(), "the compute variants see KYTY_GENERIC_COMPUTE_VARIANT on");
	Check(Switches::Find("KYTY_GENERIC_COMPUTE_VARIANT") == Switch::GenericComputeVariant,
	      "KYTY_GENERIC_COMPUTE_VARIANT is a runtime switch");
	Check(!Switches::Find("KYTY_RELAXED_SHADER_FLOAT").has_value(), "KYTY_RELAXED_SHADER_FLOAT is not a runtime switch");
	Check(!Ab::Configured() && Ab::AtFlip().empty() && !Ab::Listed(Switch::GenericComputeVariant),
	      "without KYTY_AB a flip prints nothing and changes nothing");
}

void Phases() {
	Ab::Config config;
	Check(Ab::Parse("2:KYTY_GENERIC_COMPUTE_VARIANT", "3", "0.5", config).empty(), "the A/B parses");
	Ab::Run                  run(config);
	std::vector<std::string> printed;
	std::vector<bool>        arms;
	double                   seconds = 0;
	for (uint64_t present = 1; present <= 50; ++present) {
		auto lines = run.AtPresent(present, seconds, 1000.0 + seconds);
		for (auto& line: lines) printed.push_back(std::move(line));
		arms.push_back(IR::GenericComputeVariants());
		seconds += present == 30 ? 0.5 : 0.125;
	}
	Check(printed.size() == 11, "two settled phases and the third begun print 11 lines");
	if (printed.size() != 11) {
		for (const auto& line: printed) std::fprintf(stderr, "%s", line.c_str());
		return;
	}
	Check(printed[0] == "[kyty-ab] in-run A/B: phases of 2.0 s from present 3, settle 0.5 s; A: "
	                    "KYTY_GENERIC_COMPUTE_VARIANT=1; B: KYTY_GENERIC_COMPUTE_VARIANT=0\n",
	      "the first line states both arms");
	Check(printed[1] == "[kyty-ab] phase A 0 begins at present 3 (t 0.250 s, wall 1000.250): "
	                    "KYTY_GENERIC_COMPUTE_VARIANT=1\n",
	      "phase A begins at the start present");
	Check(printed[2] == "[kyty-ab] phase A 0 settled at present 7 (t 0.750 s, wall 1000.750)\n",
	      "phase A settles after the settle time");
	Check(printed[3] == "[kyty-ab] phase A 0 settled for 1.500 s: 12 presents, 8.00 img/s, longest frame 125.0 ms, 12 "
	                    "frames of 50 ms or more; presents 7 to 19, wall 1000.750 to 1002.250\n",
	      "the end of phase A gives its presents and rate");
	Check(printed[4] == "[kyty-ab] phase B 1 begins at present 19 (t 2.250 s, wall 1002.250): "
	                    "KYTY_GENERIC_COMPUTE_VARIANT=0\n",
	      "phase B begins with the setting turned over");
	Check(printed[5] == "[kyty-ab] phase B 1 settled at present 23 (t 2.750 s, wall 1002.750)\n", "phase B settles");
	Check(printed[6] == "[kyty-ab] phase B 1 settled for 1.500 s: 9 presents, 6.00 img/s, longest frame 500.0 ms, 9 "
	                    "frames of 50 ms or more; presents 23 to 32, wall 1002.750 to 1004.250\n",
	      "a long frame of phase B is reported");
	Check(Contains(printed[7], "[kyty-ab] phase A 2 begins at present 32 "), "phase A comes back");
	Check(Contains(printed[9], "[kyty-ab] phase A 2 settled for 1.500 s: 12 presents, 8.00 img/s"), "phase A 2 ends");
	Check(Contains(printed[10], "[kyty-ab] phase B 3 begins at present 48 ") &&
	          Contains(printed[10], "KYTY_GENERIC_COMPUTE_VARIANT=0"),
	      "phase B comes back");
	bool before = true, a0 = true, b1 = true, a2 = true, b3 = true;
	for (size_t index = 0; index < arms.size(); ++index) {
		const auto present = index + 1;
		const bool a       = arms[index];
		if (present < 3) before = before && a;
		else if (present < 19) a0 = a0 && a;
		else if (present < 32) b1 = b1 && !a;
		else if (present < 48) a2 = a2 && a;
		else b3 = b3 && !a;
	}
	Check(before, "the setting keeps the environment's value before the start");
	Check(a0 && a2, "the points of use see the environment's value in phase A");
	Check(b1 && b3, "the points of use see the other value in phase B, from the flip that begins it");
	Switches::Assign(Switch::GenericComputeVariant, true);
}

void EnvironmentArms() {
	SetVariable("KYTY_GENERIC_COMPUTE_VARIANT", "0");
	Switches::Reload();
	Check(!Switches::On(Switch::GenericComputeVariant), "the runtime start applies the environment");
	Ab::Config config;
	Check(Ab::Parse("1:KYTY_GENERIC_COMPUTE_VARIANT", "", "0", config).empty(), "a one-setting A/B parses");
	Ab::Run run(config);
	const auto first = run.AtPresent(1, 0.0, 0.0);
	Check(AnyContains(first, "A: KYTY_GENERIC_COMPUTE_VARIANT=0; B: KYTY_GENERIC_COMPUTE_VARIANT=1"),
	      "arm A is the environment's value");
	static_cast<void>(run.AtPresent(2, 1.0, 1.0));
	Check(IR::GenericComputeVariants(), "phase B turns the listed setting over");
	static_cast<void>(run.AtPresent(3, 2.0, 2.0));
	Check(!IR::GenericComputeVariants(), "phase A restores the environment's value");
	SetVariable("KYTY_GENERIC_COMPUTE_VARIANT", nullptr);
	Switches::Reload();
}

void MirrorArms() {
	SetVariable("KYTY_E2_VSHARP_MIRROR", nullptr);
	Switches::Reload();
	Check(!Switches::On(Switch::E2VsharpMirror) && !IR::MirrorsVsharpBuffers(),
	      "KYTY_E2_VSHARP_MIRROR is off without the variable");
	Check(Switches::Find("KYTY_E2_VSHARP_MIRROR") == Switch::E2VsharpMirror, "KYTY_E2_VSHARP_MIRROR is a runtime switch");
	Ab::Config config;
	Check(Ab::Parse("1:KYTY_E2_VSHARP_MIRROR", "", "0", config).empty(), "an A/B of the mirror parses");
	Ab::Run run(config);
	const auto first = run.AtPresent(1, 0.0, 0.0);
	Check(AnyContains(first, "A: KYTY_E2_VSHARP_MIRROR=0; B: KYTY_E2_VSHARP_MIRROR=1"),
	      "without the variable arm A reads buffers bound and arm B by mirror address");
	Check(!IR::MirrorsVsharpBuffers(), "phase A keeps the mirror off");
	static_cast<void>(run.AtPresent(2, 1.0, 1.0));
	Check(IR::MirrorsVsharpBuffers(), "phase B turns the mirror on");
	static_cast<void>(run.AtPresent(3, 2.0, 2.0));
	Check(!IR::MirrorsVsharpBuffers(), "phase A turns it off again");
	SetVariable("KYTY_E2_VSHARP_MIRROR", "1");
	Switches::Reload();
	Check(IR::MirrorsVsharpBuffers(), "KYTY_E2_VSHARP_MIRROR=1 turns the mirror on");
	SetVariable("KYTY_E2_VSHARP_MIRROR", nullptr);
	Switches::Reload();
}

int Flips() {
	Ab::Start();
	Check(Ab::Configured() && Ab::Listed(Switch::GenericComputeVariant), "KYTY_AB lists KYTY_GENERIC_COMPUTE_VARIANT");
	const auto first = Ab::AtFlip();
	Check(AnyContains(first, "[kyty-ab] in-run A/B: phases of 1.0 s from present 0, settle 0.5 s; A: "
	                         "KYTY_GENERIC_COMPUTE_VARIANT=1; B: KYTY_GENERIC_COMPUTE_VARIANT=0"),
	      "the first flip states both arms");
	Check(AnyContains(first, "[kyty-ab] phase A 0 begins at present 1 (t 0.000 s, wall "), "the first flip begins phase A");
	Check(IR::GenericComputeVariants(), "phase A keeps KYTY_GENERIC_COMPUTE_VARIANT on");
	std::this_thread::sleep_for(std::chrono::milliseconds(1100));
	const auto second = Ab::AtFlip();
	Check(AnyContains(second, "[kyty-ab] phase B 1 begins at present 2 "), "a flip after the phase length begins phase B");
	Check(!IR::GenericComputeVariants(), "phase B turns KYTY_GENERIC_COMPUTE_VARIANT off");
	if (g_failures != 0) {
		for (const auto& line: first) std::fprintf(stderr, "%s", line.c_str());
		for (const auto& line: second) std::fprintf(stderr, "%s", line.c_str());
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_ab_switches_flip: ok\n");
	return 0;
}

}

int main() {
	if (const char* ab = std::getenv("KYTY_AB"); ab != nullptr && *ab != '\0') return Flips();
	SetVariable("KYTY_GENERIC_COMPUTE_VARIANT", nullptr);
	Switches::Reload();
	Defaults();
	Refusals();
	Phases();
	EnvironmentArms();
	MirrorArms();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("kyty_shim_ab_switches: ok\n");
	return 0;
}
