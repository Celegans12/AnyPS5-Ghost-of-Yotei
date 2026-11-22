#include "common/emulatorConfig.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/threads.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	Common::DebugCounters::SetEnabled(false);
	g_config.reset();
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);

	*g_config = cfg;
	Common::DebugCounters::SetEnabled(cfg.perf_overlay_enabled);
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

const std::string& GetGpuName() {
	return g_config->gpu_name;
}

bool HdrEnabled() {
	return g_config->hdr_enabled;
}

bool StretchEnabled() {
	return g_config->stretch_enabled;
}

bool Ps5ProEnabled() {
	return g_config->pro_enabled;
}

PssrUpscaler GetPssrUpscaler() {
	return g_config->pssr_upscaler;
}

const std::filesystem::path& GetDlssFolder() {
	return g_config->dlss_folder;
}

uint32_t GetDlssPreset() {
	return g_config->dlss_preset;
}

const std::filesystem::path& GetFsrFolder() {
	return g_config->fsr_folder;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

bool ShaderCacheEnabled() {
	return g_config->shader_cache_enabled;
}

bool PipelineManifestEnabled() {
	return g_config->pipeline_manifest_enabled;
}

bool PipelineManifestRecord() {
	return g_config->pipeline_manifest_record;
}

bool PipelineManifestStrict() {
	return g_config->pipeline_manifest_strict;
}

LogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

bool GraphicsDebugDumpEnabled() {
	return g_config->graphics_debug_dump_enabled;
}

LogDirection GetPrintfDirection() {
	return g_config->printf_direction;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

std::filesystem::path GetSessionLogFile() {
	return g_config->session_log_file;
}

bool ProfilerEnabled() {
	return g_config->profiler_enabled;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool TessellationEnabled() {
	return g_config->tessellation_enabled;
}

bool OcclusionQueriesEnabled() {
	return g_config->occlusion_queries_enabled;
}

bool RelaxedShaderFloatEnabled() {
	return g_config->relaxed_shader_float;
}

uint32_t WaveSplitCheckPeriod() {
	return g_config->wave_split_check_period;
}

bool NativeRtEnabled() {
	return g_config->native_rt;
}

bool AsyncPipelinesEnabled() {
	return g_config->async_pipelines_enabled;
}

bool AsyncPipelinesSkipMore() {
	return g_config->async_pipelines_skip_more;
}

bool RayTracingStatsEnabled() {
	return g_config->native_rt && g_config->rt_stats_enabled;
}

uint32_t RayTracingOracle() {
	return g_config->native_rt ? g_config->rt_oracle : 0u;
}

uint32_t RayTracingCheckPeriod() {
	return g_config->native_rt ? g_config->rt_check_period : 0u;
}

uint32_t RayTracingCheckPpm() {
	return g_config->rt_check_ppm;
}

uint32_t RayTracingCheckStrikes() {
	return std::max(g_config->rt_check_strikes, 1u);
}

uint32_t ParseGpuPipelineStages(const char* value) {
	char*      end    = nullptr;
	const auto parsed = std::strtoul(value, &end, 0);
	if (end == value || *end != '\0') {
		EXIT("invalid GPU pipeline stages: %s\n", value);
	}
	return parsed == 1 ? GPU_PIPELINE_ALL : static_cast<uint32_t>(parsed) & GPU_PIPELINE_ALL;
}

void ConfigureGpuStageThread(GpuStageThread stage) {
	static const std::array<int, 5> cpus = [] {
		std::array<int, 5> result {4, 6, 8, 10, 2};
		const char*        value = std::getenv("KYTY_GPU_THREAD_CPUS");
		if (value == nullptr) {
			return result;
		}
		if (std::string_view(value) == "off") {
			return std::array<int, 5> {-1, -1, -1, -1, -1};
		}
		for (auto& cpu: result) {
			if (*value == '\0') {
				break;
			}
			char* end = nullptr;
			cpu       = static_cast<int>(std::strtol(value, &end, 10));
			if (end == value) {
				EXIT("invalid KYTY_GPU_THREAD_CPUS: %s\n", std::getenv("KYTY_GPU_THREAD_CPUS"));
			}
			value = *end == ',' ? end + 1 : end;
		}
		return result;
	}();
	Common::Thread::RaiseCurrentPriority();
#if defined(ANYPS5_KYTY) && KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	static const bool listed = [] {
		const char* value = std::getenv("KYTY_GPU_THREAD_CPUS");
		return value != nullptr && value[0] != '\0';
	}();
	if (!listed) {
		Common::Thread::KeepCurrentOnLargestCache();
		return;
	}
#endif
	static std::once_flag reserved;
	std::call_once(reserved, [] {
		Common::Thread::ReserveCores(std::span {cpus});
	});
	Common::Thread::PinCurrent(cpus[static_cast<uint32_t>(stage)]);
}

uint32_t GpuPipelineStages() {
	if (const char* value = std::getenv("KYTY_GPU_PIPELINE"); value != nullptr) {
		return ParseGpuPipelineStages(value);
	}
	return g_config != nullptr ? g_config->gpu_pipeline_stages : 0;
}

}
