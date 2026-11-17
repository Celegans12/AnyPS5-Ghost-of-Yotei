#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <magic_enum.hpp>

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>

namespace KytyShim {
namespace {

using Options = Config::ConfigOptions;

struct Flag {
    const char* variable;
    bool Options::* field;
};

constexpr Flag Flags[] = {
    {"KYTY_FULLSCREEN", &Options::fullscreen_enabled},
    {"KYTY_HDR", &Options::hdr_enabled},
    {"KYTY_HIDE_CURSOR", &Options::hide_cursor_enabled},
    {"KYTY_STRETCH", &Options::stretch_enabled},
    {"KYTY_VR", &Options::vr_enabled},
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
    {"KYTY_NATIVE_RT", &Options::native_rt},
    {"KYTY_ASYNC_PIPELINES", &Options::async_pipelines_enabled},
    {"KYTY_RT_STATS", &Options::rt_stats_enabled},
    {"KYTY_COMMAND_BUFFER_DUMP", &Options::command_buffer_dump_enabled},
    {"KYTY_GRAPHICS_DEBUG_DUMP", &Options::graphics_debug_dump_enabled},
    {"KYTY_PROFILE", &Options::profiler_enabled},
    {"KYTY_SPIRV_DEBUG_PRINTF", &Options::spirv_debug_printf_enabled},
    {"KYTY_RENDERDOC", &Options::renderdoc_enabled},
    {"KYTY_READBACK_LINEAR_IMAGES", &Options::readback_linear_images},
    {"KYTY_READBACK_TILED_IMAGES", &Options::readback_tiled_images},
    {"KYTY_READBACK_STORAGE_IMAGES", &Options::readback_storage_images},
    {"KYTY_SYNC_RAW_IMAGE_BUFFERS", &Options::sync_raw_image_buffers},
    {"KYTY_PERF_OVERLAY", &Options::perf_overlay_enabled},
    {"KYTY_TROPHY_NOTIFICATIONS", &Options::trophy_enabled},
};

struct Number {
    const char* variable;
    std::uint32_t Options::* field;
    std::uint32_t minimum;
    std::uint32_t maximum;
};

constexpr Number Numbers[] = {
    {"KYTY_SCREEN_WIDTH", &Options::screen_width, 1, 0xffffffffu},
    {"KYTY_SCREEN_HEIGHT", &Options::screen_height, 1, 0xffffffffu},
    {"KYTY_VBLANK_FREQUENCY", &Options::vblank_frequency, 0, 0xffffffffu},
    {"KYTY_FLUSH_DRAWS", &Options::flush_draws, 0, 0xffffffffu},
    {"KYTY_TROPHY_DURATION", &Options::trophy_duration_seconds, 0, 0xffffffffu},
    {"KYTY_CONSOLE_LANGUAGE", &Options::console_language, 0, Config::MAX_CONSOLE_LANGUAGE},
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
    for (const auto& flag : Flags) {
        const char* value = Read(flag.variable, applied);
        if (value == nullptr) continue;
        const std::string_view text(value);
        if (text != "0" && text != "1") EXIT("KytyShim: %s=%s is neither 0 nor 1\n", flag.variable, value);
        options.*flag.field = text == "1";
    }
    for (const auto& number : Numbers) {
        const char* value = Read(number.variable, applied);
        if (value == nullptr) continue;
        std::uint32_t parsed = 0;
        if (!ParseInteger(value, parsed) || parsed < number.minimum || parsed > number.maximum) EXIT("KytyShim: %s=%s is not a whole number from %u to %u\n", number.variable, value, number.minimum, number.maximum);
        options.*number.field = parsed;
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
    ReadEnum("KYTY_PIPELINE_PRECOMPILE", options.pipeline_precompile, applied);
    ReadEnum("KYTY_SHADER_LOG_DIRECTION", options.shader_log_direction, applied);
    ReadEnum("KYTY_PRINTF_DIRECTION", options.printf_direction, applied);
    if (options.gpu_assisted_validation_enabled) options.vulkan_validation_enabled = true;
    return options;
}

}

void StartRuntime() {
    static std::once_flag started;
    std::call_once(started, [] {
        Common::InitializeThreads();
        std::string applied;
        const auto options = OptionsFromEnvironment(applied);
        Config::Initialize();
        Config::Load(options);
        Log::Initialize();
        std::fprintf(stderr, "[kyty] runtime started, settings: %s\n", applied.empty() ? "Kyty's defaults" : applied.c_str());
    });
}

void RouteInterrupts(Libs::Graphics::RenderContext& renderer) {
    renderer.AddInterruptEq(InterruptRoute, GraphicsInterruptEvent);
    for (int event = FirstComputeInterruptEvent; event < FirstComputeInterruptEvent + ComputeInterruptEvents; ++event) renderer.AddInterruptEq(InterruptRoute, event);
}

}
