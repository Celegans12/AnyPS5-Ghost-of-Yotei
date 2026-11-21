#include "prx/libSceAgcDriver/KytyShim/src/Settings.hpp"

#include <algorithm>
#include <atomic>
#include <iterator>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef _WIN32
extern char** environ;
#endif

namespace KytyShim::Settings {

namespace {

constexpr std::string_view Prefix = "KYTY_";

constexpr std::string_view Names[] = {
    "KYTY_AB",
    "KYTY_AB_SETTLE",
    "KYTY_AB_START",
    "KYTY_ASYNC_PIPELINES",
    "KYTY_COMMAND_BUFFER_DUMP",
    "KYTY_COMMAND_BUFFER_DUMP_FOLDER",
    "KYTY_COMMAND_MEMORY_IN_PLACE",
    "KYTY_COMPARE_WRITE_BACK",
    "KYTY_CX_OBSERVER",
    "KYTY_DIRECT_MEMORY_PAGES",
    "KYTY_DLSS_FOLDER",
    "KYTY_DLSS_PRESET",
    "KYTY_DRAW_BUFFERS",
    "KYTY_DRAW_TEMPLATES",
    "KYTY_EARLY_FLIP",
    "KYTY_F32_PRODUCT_NO_CONTRACTION",
    "KYTY_FALLBACKS",
    "KYTY_FAST_TLS",
    "KYTY_FSR_FOLDER",
    "KYTY_FUZZ_PAD",
    "KYTY_FUZZ_SPIRV",
    "KYTY_FUZZ_SPIRV_DIR",
    "KYTY_GENERIC_COMPUTE_SPECIALIZE",
    "KYTY_GENERIC_COMPUTE_VARIANT",
    "KYTY_GPU_ASSISTED_VALIDATION",
    "KYTY_GPU_PIPELINE",
    "KYTY_GPU_SPIN_US",
    "KYTY_GPU_THREAD_CPUS",
    "KYTY_GPU_TIME",
    "KYTY_GRAPHICS_DEBUG_DUMP",
    "KYTY_HDR",
    "KYTY_HOT_PAGE_WRITE_WATCH",
    "KYTY_LABELS_AFTER_DUMPS",
    "KYTY_LABELS_AROUND_GPU_WRITES",
    "KYTY_LOCK_STATS",
    "KYTY_LOCK_STATS_PERIOD",
    "KYTY_NATIVE_RT",
    "KYTY_OCCLUSION_QUERIES",
    "KYTY_PERF_OVERLAY",
    "KYTY_PGO_DIR",
    "KYTY_PGO_DUMP_SECONDS",
    "KYTY_PGO_WRITE_SECONDS",
    "KYTY_PIPELINE_CHAIN_THREADS",
    "KYTY_PIPELINE_LIBRARIES",
    "KYTY_PIPELINE_MANIFEST",
    "KYTY_PIPELINE_STALL_LOG_MS",
    "KYTY_PRESENT_MODE",
    "KYTY_PRINTF_DIRECTION",
    "KYTY_PRINTF_OUTPUT_FILE",
    "KYTY_PROFILE",
    "KYTY_PROGRAM_LOCK",
    "KYTY_PSSR_UPSCALER",
    "KYTY_RECORD_PIPELINE_MANIFEST",
    "KYTY_RELAXED_SHADER_FLOAT",
    "KYTY_REPLAY_AFTER_READBACK",
    "KYTY_REPLAY_GPU_PAGE_READS",
    "KYTY_RT_ARENA_MB",
    "KYTY_RT_CHECK",
    "KYTY_RT_CHECK_PPM",
    "KYTY_RT_CHECK_STRIKES",
    "KYTY_RT_ORACLE",
    "KYTY_RT_RESULTS",
    "KYTY_RT_STATS",
    "KYTY_SESSION_LOG",
    "KYTY_SHADER_CACHE",
    "KYTY_SHADER_CACHE_SHARED",
    "KYTY_SHADER_CACHE_STAMP",
    "KYTY_SHADER_LOG_DIRECTION",
    "KYTY_SHADER_LOG_FOLDER",
    "KYTY_SHADER_VALIDATION",
    "KYTY_SHADER_VARIANT_LOG",
    "KYTY_SHARE_EMPTY_BUFFERS",
    "KYTY_SPIRV_DEBUG_PRINTF",
    "KYTY_STRETCH",
    "KYTY_STRICT_PIPELINE_MANIFEST",
    "KYTY_SUSPEND_POINT_DEPTH",
    "KYTY_TABLE_DESCRIPTOR_WRITES",
    "KYTY_TABLE_MEMO_VERIFY",
    "KYTY_TABLE_RESOLUTIONS",
    "KYTY_TABLE_XPR",
    "KYTY_TESSELLATION",
    "KYTY_TILED_DIAG",
    "KYTY_TILE_VIA_VRAM",
    "KYTY_TRINITY",
    "KYTY_URGENT_WRITEBACK_WAIT",
    "KYTY_VBLANK_FREQUENCY",
    "KYTY_VULKAN_VALIDATION",
    "KYTY_WAVE_SPLIT_CHECK",
    "KYTY_WRITE_BACK_IMAGES",
};

static_assert(std::ranges::is_sorted(Names));
static_assert(std::ranges::adjacent_find(Names) == std::end(Names));

}

bool Read(std::string_view name) {
    return std::ranges::binary_search(Names, name);
}

std::vector<std::string> UnreadLines(const std::vector<std::string>& environment) {
    std::vector<std::string> unread;
    for (const auto& entry : environment) {
        const std::string_view name = std::string_view(entry).substr(0, entry.find('='));
        if (name.starts_with(Prefix) && !Read(name)) unread.emplace_back(name);
    }
    std::ranges::sort(unread);
    unread.erase(std::ranges::unique(unread).begin(), unread.end());
    std::vector<std::string> lines;
    lines.reserve(unread.size());
    for (const auto& name : unread) lines.push_back("[kyty] " + name + " is set but not read by this build\n");
    return lines;
}

std::vector<std::string> EnvironmentEntries() {
    std::vector<std::string> entries;
#ifdef _WIN32
    char* block = GetEnvironmentStringsA();
    if (block == nullptr) return entries;
    for (const char* entry = block; *entry != '\0'; entry += entries.back().size() + 1) entries.emplace_back(entry);
    FreeEnvironmentStringsA(block);
#else
    for (char** entry = environ; entry != nullptr && *entry != nullptr; ++entry) entries.emplace_back(*entry);
#endif
    return entries;
}

void ReportUnread(std::FILE* out) {
    static std::atomic<bool> reported{false};
    if (reported.exchange(true)) return;
    for (const auto& line : UnreadLines(EnvironmentEntries())) std::fputs(line.c_str(), out);
    std::fflush(out);
}

}
