#include "prx/libSceAgcDriver/KytyShim/src/NativeRt.hpp"

namespace KytyShim::NativeRt {

std::optional<Mode> ParseMode(const char* value) {
    if (value == nullptr || *value == '\0') return Mode::Auto;
    const std::string_view text(value);
    if (text == "0") return Mode::Off;
    if (text == "1") return Mode::Required;
    return std::nullopt;
}

std::optional<std::uint32_t> ParseOracle(const char* value) {
    if (value == nullptr || *value == '\0') return DefaultOracle;
    const std::string_view text(value);
    if (text.size() == 1 && text[0] >= '0' && text[0] <= '9') return static_cast<std::uint32_t>(text[0] - '0');
    if (text.size() == 2 && text[0] == '1' && text[1] >= '0' && text[1] <= '5') return static_cast<std::uint32_t>(10 + text[1] - '0');
    return std::nullopt;
}

std::string OracleLine(Verdict verdict, std::uint32_t oracle) {
    if (verdict != Verdict::Native || (oracle & ~OraclePixel) == 0) return {};
    const std::string setting = "(KYTY_RT_ORACLE=" + std::to_string(oracle) + ")";
    const bool remaining = (oracle & OracleRemaining) != 0;
    if ((oracle & OracleOnly) != 0 && !remaining) {
        return "[kyty] no BVH walk form is tried and no walk answers from host ray queries " + setting + ": every BVH walk runs emulated\n";
    }
    std::string walks;
    if ((oracle & OracleOnly) != 0) {
        walks = "every walk whose TLAS and instance records are derived, without trying the walk forms,";
    } else if ((oracle & OracleForms) != 0) {
        walks = remaining ? "every recognized walk" : "the walks a form recognizes";
    } else {
        walks = "the walks no form recognizes";
    }
    if (remaining && (oracle & OraclePixel) != 0) walks += " in compute and pixel shaders";
    return "[kyty] BVH intersections of " + walks + " answer from host ray queries " + setting + "; the guest walk code runs as written\n";
}

std::string MissingSupport(const DeviceSupport& support) {
    std::string missing;
    const auto add = [&missing](bool present, const char* name) {
        if (present) return;
        missing += missing.empty() ? "" : ", ";
        missing += name;
    };
    add(support.deferredHostOperations, "VK_KHR_deferred_host_operations");
    add(support.accelerationStructureExtension, "VK_KHR_acceleration_structure");
    add(support.rayQueryExtension, "VK_KHR_ray_query");
    if (!missing.empty()) return missing;
    add(support.accelerationStructureFeature, "the accelerationStructure feature");
    add(support.rayQueryFeature, "the rayQuery feature");
    return missing;
}

Outcome Decide(Mode mode, bool deviceEnabled, std::string_view gpu, const DeviceSupport& support) {
    const std::string device(gpu.empty() ? std::string_view("the Vulkan device") : gpu);
    if (mode == Mode::Off) {
        if (deviceEnabled) return {Verdict::Refused, "KytyShim: KYTY_NATIVE_RT=0 asks for emulated ray tracing, but " + device + " was created with ray queries\n"};
        return {Verdict::Emulated, "[kyty] native ray tracing off (KYTY_NATIVE_RT=0): ray tracing runs emulated in shaders\n"};
    }
    if (deviceEnabled) {
        return {Verdict::Native, "[kyty] native ray tracing on " + device + ": the ray tracing passes whose BVH walk Kyty recognizes trace with ray queries, the others stay emulated; KYTY_NATIVE_RT=0 emulates them all\n"};
    }
    auto missing = MissingSupport(support);
    if (missing.empty()) missing = "ray queries on the device Kyty created";
    if (mode == Mode::Required) {
        return {Verdict::Refused, "KytyShim: KYTY_NATIVE_RT=1 asks for native ray tracing, but " + device + " lacks " + missing + "; unset KYTY_NATIVE_RT or set it to 0 to emulate ray tracing\n"};
    }
    return {Verdict::Emulated, "[kyty] native ray tracing unavailable on " + device + " (it lacks " + missing + "): ray tracing runs emulated in shaders\n"};
}

}
