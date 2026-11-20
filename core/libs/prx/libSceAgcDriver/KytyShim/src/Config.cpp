#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/NativeRt.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/ShaderCachePolicy.hpp"
#include "prx/libSceVideoOut/include/VblankSchedule.hpp"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/runtimeSwitches.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <magic_enum.hpp>

#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace KytyShim {
namespace {

using Options = Config::ConfigOptions;

std::atomic<NativeRt::Mode> nativeRtMode{NativeRt::Mode::Auto};

struct Flag {
    const char* variable;
    bool Options::* field;
};

constexpr Flag Flags[] = {
    {"KYTY_HDR", &Options::hdr_enabled},
    {"KYTY_STRETCH", &Options::stretch_enabled},
    {"KYTY_TRINITY", &Options::pro_enabled},
    {"KYTY_VULKAN_VALIDATION", &Options::vulkan_validation_enabled},
    {"KYTY_GPU_ASSISTED_VALIDATION", &Options::gpu_assisted_validation_enabled},
    {"KYTY_SHADER_VALIDATION", &Options::shader_validation_enabled},
    {"KYTY_SHADER_CACHE", &Options::shader_cache_enabled},
    {"KYTY_PIPELINE_MANIFEST", &Options::pipeline_manifest_enabled},
    {"KYTY_RECORD_PIPELINE_MANIFEST", &Options::pipeline_manifest_record},
    {"KYTY_STRICT_PIPELINE_MANIFEST", &Options::pipeline_manifest_strict},
    {"KYTY_TESSELLATION", &Options::tessellation_enabled},
    {"KYTY_OCCLUSION_QUERIES", &Options::occlusion_queries_enabled},
    {"KYTY_RELAXED_SHADER_FLOAT", &Options::relaxed_shader_float},
    {"KYTY_WAVE_SPLIT", &Options::wave_split},
    {"KYTY_IMAGE_TABLE_HEAP", &Options::image_table_heap},
    {"KYTY_RT_STATS", &Options::rt_stats_enabled},
    {"KYTY_COMMAND_BUFFER_DUMP", &Options::command_buffer_dump_enabled},
    {"KYTY_GRAPHICS_DEBUG_DUMP", &Options::graphics_debug_dump_enabled},
    {"KYTY_PROFILE", &Options::profiler_enabled},
    {"KYTY_SPIRV_DEBUG_PRINTF", &Options::spirv_debug_printf_enabled},
    {"KYTY_PERF_OVERLAY", &Options::perf_overlay_enabled},
};

struct Number {
    const char* variable;
    std::uint32_t Options::* field;
    std::uint32_t minimum;
    std::uint32_t maximum;
};

constexpr Number Numbers[] = {
    {"KYTY_VBLANK_FREQUENCY", &Options::vblank_frequency, 0, 0xffffffffu},
    {"KYTY_RT_CHECK", &Options::rt_check_period, 0, 0xffffffffu},
    {"KYTY_RT_CHECK_PPM", &Options::rt_check_ppm, 0, 1000000u},
    {"KYTY_RT_CHECK_STRIKES", &Options::rt_check_strikes, 1, 1000u},
};

struct Path {
    const char* variable;
    std::filesystem::path Options::* field;
};

constexpr Path Paths[] = {
    {"KYTY_DLSS_FOLDER", &Options::dlss_folder},
    {"KYTY_FSR_FOLDER", &Options::fsr_folder},
    {"KYTY_SHADER_LOG_FOLDER", &Options::shader_log_folder},
    {"KYTY_COMMAND_BUFFER_DUMP_FOLDER", &Options::command_buffer_dump_folder},
    {"KYTY_PRINTF_OUTPUT_FILE", &Options::printf_output_file},
    {"KYTY_SESSION_LOG", &Options::session_log_file},
};

const char* Read(const char* variable, std::string& applied) {
    const char* value = std::getenv(variable);
    if (value == nullptr || *value == '\0') return nullptr;
    applied += applied.empty() ? "" : " ";
    applied += variable;
    applied += '=';
    applied += value;
    return value;
}

template <typename Integer>
bool ParseInteger(std::string_view text, Integer& out) {
    Integer value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) return false;
    out = value;
    return true;
}

template <typename Enum>
void ReadEnum(const char* variable, Enum& out, std::string& applied) {
    const char* value = Read(variable, applied);
    if (value == nullptr) return;
    const auto parsed = magic_enum::enum_cast<Enum>(std::string_view(value), magic_enum::case_insensitive);
    if (!parsed.has_value()) {
        std::string expected;
        for (const auto name : magic_enum::enum_names<Enum>()) {
            expected += expected.empty() ? "" : ", ";
            expected += name;
        }
        EXIT("KytyShim: %s=%s is not one of %s\n", variable, value, expected.c_str());
    }
    out = *parsed;
}

Options OptionsFromEnvironment(std::string& applied) {
    Options options;
    options.shader_cache_enabled = true;
    for (const auto& flag : Flags) {
        const char* value = Read(flag.variable, applied);
        if (value == nullptr) continue;
        const std::string_view text(value);
        if (text != "0" && text != "1") EXIT("KytyShim: %s=%s is neither 0 nor 1\n", flag.variable, value);
        options.*flag.field = text == "1";
    }
    options.shader_cache_enabled = ResolveShaderCacheEnabled(options.shader_cache_enabled, Read("ANYPS5_NO_SHADER_CACHE", applied));
    if (const char* value = Read("KYTY_ASYNC_PIPELINES", applied)) {
        const std::string_view text(value);
        if (text != "0" && text != "1" && text != "2") EXIT("KytyShim: KYTY_ASYNC_PIPELINES=%s is not 0, 1 or 2\n", value);
        options.async_pipelines_enabled = text != "0";
        options.async_pipelines_skip_more = text == "2";
    }
    const char* nativeRt = Read("KYTY_NATIVE_RT", applied);
    const auto mode = NativeRt::ParseMode(nativeRt);
    if (!mode.has_value()) EXIT("KytyShim: KYTY_NATIVE_RT=%s is neither 0 nor 1\n", nativeRt);
    options.native_rt = *mode != NativeRt::Mode::Off;
    nativeRtMode.store(*mode, std::memory_order_relaxed);
    const char* oracle = Read("KYTY_RT_ORACLE", applied);
    const auto oracleMode = NativeRt::ParseOracle(oracle);
    if (!oracleMode.has_value()) EXIT("KytyShim: KYTY_RT_ORACLE=%s is not a whole number from 0 to 15\n", oracle);
    options.rt_oracle = *oracleMode;
    if (const char* results = Read("KYTY_RT_RESULTS", applied)) {
        const std::string_view text(results);
        if (text != "0" && text != "1") EXIT("KytyShim: KYTY_RT_RESULTS=%s is neither 0 nor 1\n", results);
        if (text == "1" && (options.rt_oracle & 7u) != 0) options.rt_oracle |= 32u;
    }
    for (const auto& number : Numbers) {
        const char* value = Read(number.variable, applied);
        if (value == nullptr) continue;
        std::uint32_t parsed = 0;
        if (!ParseInteger(value, parsed) || parsed < number.minimum || parsed > number.maximum) EXIT("KytyShim: %s=%s is not a whole number from %u to %u\n", number.variable, value, number.minimum, number.maximum);
        options.*number.field = parsed;
    }
    try {
        if (const auto hertz = VideoOutVblank::ConfiguredVblankHz()) options.vblank_frequency = *hertz;
    } catch (const std::runtime_error& error) {
        EXIT("KytyShim: %s\n", error.what());
    }
    for (const auto& path : Paths) {
        if (const char* value = Read(path.variable, applied)) options.*path.field = std::filesystem::path(value);
    }
    if (const char* value = Read("KYTY_GPU", applied); value != nullptr && !ParseInteger(value, options.gpu_index)) EXIT("KytyShim: KYTY_GPU=%s is not a GPU index\n", value);
    if (const char* value = Read("KYTY_DLSS_PRESET", applied); value != nullptr) {
        const std::string_view text(value);
        if (text.size() != 1 || text[0] < 'A' || text[0] > 'O') EXIT("KytyShim: KYTY_DLSS_PRESET=%s is not a letter from A to O\n", value);
        options.dlss_preset = static_cast<std::uint32_t>(text[0] - 'A' + 1);
    }
    ReadEnum("KYTY_PRESENT_MODE", options.present_mode, applied);
    ReadEnum("KYTY_PSSR_UPSCALER", options.pssr_upscaler, applied);
    ReadEnum("KYTY_SHADER_LOG_DIRECTION", options.shader_log_direction, applied);
    ReadEnum("KYTY_PRINTF_DIRECTION", options.printf_direction, applied);
    if (options.gpu_assisted_validation_enabled) options.vulkan_validation_enabled = true;
    return options;
}

bool HasExtension(const std::vector<vk::ExtensionProperties>& extensions, std::string_view name) {
    for (const auto& extension : extensions) {
        if (name == extension.extensionName.data()) return true;
    }
    return false;
}

NativeRt::DeviceSupport QueryDeviceSupport(vk::PhysicalDevice device) {
    std::uint32_t count = 0;
    auto result = device.enumerateDeviceExtensionProperties(nullptr, &count, nullptr);
    std::vector<vk::ExtensionProperties> extensions(count);
    if (result == vk::Result::eSuccess) result = device.enumerateDeviceExtensionProperties(nullptr, &count, extensions.data());
    extensions.resize(count);
    if (result != vk::Result::eSuccess) EXIT("KytyShim: vkEnumerateDeviceExtensionProperties failed (%s) while checking native ray tracing\n", vk::to_string(result).c_str());
    NativeRt::DeviceSupport support;
    support.deferredHostOperations = HasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    support.accelerationStructureExtension = HasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    support.rayQueryExtension = HasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    if (support.deferredHostOperations && support.accelerationStructureExtension && support.rayQueryExtension) {
        vk::PhysicalDeviceRayQueryFeaturesKHR rayQuery{};
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{};
        accelerationStructure.pNext = &rayQuery;
        vk::PhysicalDeviceFeatures2 features{};
        features.pNext = &accelerationStructure;
        device.getFeatures2(&features);
        support.accelerationStructureFeature = accelerationStructure.accelerationStructure == VK_TRUE;
        support.rayQueryFeature = rayQuery.rayQuery == VK_TRUE;
    }
    return support;
}

}

void StartRuntime() {
    static std::once_flag started;
    std::call_once(started, [] {
        std::string applied;
        const auto options = OptionsFromEnvironment(applied);
        Common::Ab::Start();
        Config::Initialize();
        Config::Load(options);
        Log::Initialize();
        std::fprintf(stderr, "[kyty] runtime started, settings: %s\n", applied.empty() ? "Kyty's defaults" : applied.c_str());
    });
}

void ReportNativeRayTracing(const Libs::Graphics::GraphicContext& graphics) {
    const auto mode = nativeRtMode.load(std::memory_order_relaxed);
    const bool enabled = graphics.native_ray_tracing_enabled;
    const auto support = mode != NativeRt::Mode::Off && !enabled ? QueryDeviceSupport(graphics.physical_device) : NativeRt::DeviceSupport{};
    const auto outcome = NativeRt::Decide(mode, enabled, graphics.physical_device_properties.deviceName.data(), support);
    if (outcome.verdict == NativeRt::Verdict::Refused) EXIT("%s", outcome.message.c_str());
    std::fputs(outcome.message.c_str(), stderr);
    std::fputs(NativeRt::OracleLine(outcome.verdict, Config::RayTracingOracle()).c_str(), stderr);
}

void RouteInterrupts(Libs::Graphics::RenderContext& renderer) {
    renderer.AddInterruptEq(InterruptRoute, GraphicsInterruptEvent);
    for (int event = FirstComputeInterruptEvent; event < FirstComputeInterruptEvent + ComputeInterruptEvents; ++event) renderer.AddInterruptEq(InterruptRoute, event);
}

}
