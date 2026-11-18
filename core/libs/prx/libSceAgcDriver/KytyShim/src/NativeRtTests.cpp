#include "prx/libSceAgcDriver/KytyShim/src/NativeRt.hpp"

#include <cstdio>
#include <string>
#include <string_view>

namespace {

using KytyShim::NativeRt::Decide;
using KytyShim::NativeRt::DeviceSupport;
using KytyShim::NativeRt::MissingSupport;
using KytyShim::NativeRt::Mode;
using KytyShim::NativeRt::OracleLine;
using KytyShim::NativeRt::ParseMode;
using KytyShim::NativeRt::ParseOracle;
using KytyShim::NativeRt::Verdict;

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
}

bool Has(const std::string& text, std::string_view fragment) {
    return text.find(fragment) != std::string::npos;
}

DeviceSupport Full() {
    return {true, true, true, true, true};
}

void Modes() {
    Expect(ParseMode(nullptr) == Mode::Auto, "an unset KYTY_NATIVE_RT means auto");
    Expect(ParseMode("") == Mode::Auto, "an empty KYTY_NATIVE_RT means auto");
    Expect(ParseMode("0") == Mode::Off, "KYTY_NATIVE_RT=0 forces emulation");
    Expect(ParseMode("1") == Mode::Required, "KYTY_NATIVE_RT=1 requires native ray tracing");
    Expect(!ParseMode("2").has_value(), "KYTY_NATIVE_RT=2 is refused");
    Expect(!ParseMode("01").has_value(), "KYTY_NATIVE_RT=01 is refused");
    Expect(!ParseMode("auto").has_value(), "KYTY_NATIVE_RT=auto is refused");
    Expect(!ParseMode("1 ").has_value(), "a trailing space is refused");
}

void Missing() {
    Expect(MissingSupport(Full()).empty(), "a full device lacks nothing");
    Expect(MissingSupport({}) == "VK_KHR_deferred_host_operations, VK_KHR_acceleration_structure, VK_KHR_ray_query", "a device without the extensions lacks all three");
    DeviceSupport noRayQuery = Full();
    noRayQuery.rayQueryExtension = false;
    noRayQuery.rayQueryFeature = false;
    Expect(MissingSupport(noRayQuery) == "VK_KHR_ray_query", "only the missing extension is named");
    DeviceSupport noFeature = Full();
    noFeature.rayQueryFeature = false;
    Expect(MissingSupport(noFeature) == "the rayQuery feature", "a missing feature is named when the extensions are there");
    DeviceSupport noBoth = Full();
    noBoth.accelerationStructureFeature = false;
    noBoth.rayQueryFeature = false;
    Expect(MissingSupport(noBoth) == "the accelerationStructure feature, the rayQuery feature", "both missing features are named");
    DeviceSupport noExtensionNoFeature = Full();
    noExtensionNoFeature.deferredHostOperations = false;
    noExtensionNoFeature.rayQueryFeature = false;
    Expect(MissingSupport(noExtensionNoFeature) == "VK_KHR_deferred_host_operations", "features are not named while an extension is missing");
}

void AutoMode() {
    const auto native = Decide(Mode::Auto, true, "NVIDIA GeForce RTX 4080 SUPER", Full());
    Expect(native.verdict == Verdict::Native, "auto with a capable device traces natively");
    Expect(Has(native.message, "native ray tracing on NVIDIA GeForce RTX 4080 SUPER"), "the native line names the GPU");
    Expect(Has(native.message, "KYTY_NATIVE_RT=0"), "the native line says how to turn it off");
    const auto emulated = Decide(Mode::Auto, false, "Old GPU", {});
    Expect(emulated.verdict == Verdict::Emulated, "auto without the extensions falls back to emulation");
    Expect(Has(emulated.message, "unavailable on Old GPU"), "the fallback line names the GPU");
    Expect(Has(emulated.message, "VK_KHR_ray_query") && Has(emulated.message, "VK_KHR_acceleration_structure") && Has(emulated.message, "VK_KHR_deferred_host_operations"), "the fallback line names the missing extensions");
    Expect(Has(emulated.message, "emulated"), "the fallback line says ray tracing is emulated");
    const auto unexplained = Decide(Mode::Auto, false, "Odd GPU", Full());
    Expect(unexplained.verdict == Verdict::Emulated, "auto with a device that did not enable ray queries falls back");
    Expect(Has(unexplained.message, "ray queries on the device Kyty created"), "the fallback line still says what is missing");
    const auto unnamed = Decide(Mode::Auto, false, "", {});
    Expect(Has(unnamed.message, "the Vulkan device"), "a nameless GPU is called the Vulkan device");
}

void RequiredMode() {
    const auto native = Decide(Mode::Required, true, "GPU", Full());
    Expect(native.verdict == Verdict::Native, "KYTY_NATIVE_RT=1 with a capable device traces natively");
    DeviceSupport noFeature = Full();
    noFeature.accelerationStructureFeature = false;
    const auto refused = Decide(Mode::Required, false, "Old GPU", noFeature);
    Expect(refused.verdict == Verdict::Refused, "KYTY_NATIVE_RT=1 without support stops");
    Expect(Has(refused.message, "KYTY_NATIVE_RT=1") && Has(refused.message, "Old GPU lacks the accelerationStructure feature"), "the refusal names the setting, the GPU and what it lacks");
    Expect(Has(refused.message, "set it to 0"), "the refusal says how to go on");
    const auto noExtensions = Decide(Mode::Required, false, "Old GPU", {});
    Expect(noExtensions.verdict == Verdict::Refused && Has(noExtensions.message, "VK_KHR_ray_query"), "the refusal names a missing extension");
}

void OffMode() {
    const auto off = Decide(Mode::Off, false, "GPU", Full());
    Expect(off.verdict == Verdict::Emulated, "KYTY_NATIVE_RT=0 emulates on a capable device");
    Expect(Has(off.message, "KYTY_NATIVE_RT=0") && Has(off.message, "emulated"), "the off line names the setting");
    const auto inconsistent = Decide(Mode::Off, true, "GPU", Full());
    Expect(inconsistent.verdict == Verdict::Refused, "ray queries on a device KYTY_NATIVE_RT=0 created stop the run");
}

void Oracle() {
    Expect(ParseOracle(nullptr) == KytyShim::NativeRt::DefaultOracle, "an unset KYTY_RT_ORACLE takes the default");
    Expect(ParseOracle("") == KytyShim::NativeRt::DefaultOracle, "an empty KYTY_RT_ORACLE takes the default");
    Expect(KytyShim::NativeRt::DefaultOracle == KytyShim::NativeRt::OracleRemaining, "the oracle serves the walks no form recognizes by default");
    Expect(ParseOracle("0") == 0u, "KYTY_RT_ORACLE=0 turns the oracle off");
    Expect(ParseOracle("1") == KytyShim::NativeRt::OracleRemaining, "KYTY_RT_ORACLE=1 serves the walks no form recognizes");
    Expect(ParseOracle("5") == (KytyShim::NativeRt::OracleRemaining | KytyShim::NativeRt::OracleOnly), "KYTY_RT_ORACLE=5 skips the forms");
    Expect(ParseOracle("7") == 7u, "KYTY_RT_ORACLE=7 is accepted");
    Expect(!ParseOracle("8").has_value(), "KYTY_RT_ORACLE=8 is refused");
    Expect(!ParseOracle("01").has_value(), "KYTY_RT_ORACLE=01 is refused");
    Expect(!ParseOracle("-1").has_value(), "KYTY_RT_ORACLE=-1 is refused");
    Expect(!ParseOracle("on").has_value(), "KYTY_RT_ORACLE=on is refused");
    Expect(OracleLine(Verdict::Native, 0).empty(), "no oracle line when the oracle is off");
    Expect(OracleLine(Verdict::Emulated, 1).empty(), "no oracle line when ray tracing is emulated");
    Expect(OracleLine(Verdict::Refused, 1).empty(), "no oracle line when native ray tracing is refused");
    const auto remaining = OracleLine(Verdict::Native, 1);
    Expect(Has(remaining, "the walks no form recognizes") && Has(remaining, "KYTY_RT_ORACLE=1"), "the oracle line names what it serves and the setting");
    Expect(Has(OracleLine(Verdict::Native, 3), "every recognized walk"), "KYTY_RT_ORACLE=3 serves every recognized walk");
    Expect(Has(OracleLine(Verdict::Native, 2), "the walks a form recognizes"), "KYTY_RT_ORACLE=2 serves the walks of the forms");
    Expect(Has(OracleLine(Verdict::Native, 5), "without trying the walk forms"), "KYTY_RT_ORACLE=5 says the forms are skipped");
    Expect(Has(OracleLine(Verdict::Native, 4), "every BVH walk runs emulated"), "KYTY_RT_ORACLE=4 says nothing is served");
    Expect(Has(OracleLine(Verdict::Native, 6), "every BVH walk runs emulated"), "KYTY_RT_ORACLE=6 says nothing is served");
}

}

int main() {
    Modes();
    Missing();
    AutoMode();
    RequiredMode();
    OffMode();
    Oracle();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("kyty shim native ray tracing: all checks passed\n");
    return 0;
}
