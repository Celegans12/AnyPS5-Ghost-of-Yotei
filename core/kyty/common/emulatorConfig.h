#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

enum class PssrUpscaler { Linear, DLSS, FSR };

enum GpuPipelineStage : uint32_t {
	GPU_PIPELINE_RECORDING = 1u << 0u,
	GPU_PIPELINE_COMMAND_PROCESSOR = 1u << 1u,
	GPU_PIPELINE_RESOLVE = 1u << 2u,
	GPU_PIPELINE_ALL =
	    GPU_PIPELINE_RECORDING | GPU_PIPELINE_COMMAND_PROCESSOR | GPU_PIPELINE_RESOLVE,
};

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	PresentMode            present_mode                = PresentMode::Mailbox;
	std::string            gpu_name;
	bool                   hdr_enabled                 = false;
	bool                   stretch_enabled             = false;
	bool                   pro_enabled                 = false;
	PssrUpscaler           pssr_upscaler               = PssrUpscaler::Linear;
	std::filesystem::path  dlss_folder;
	uint32_t               dlss_preset                 = 0;
	std::filesystem::path  fsr_folder;
	uint32_t               vblank_frequency            = 60;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	bool                   shader_cache_enabled        = false;
	bool                   pipeline_manifest_enabled   = true;
	bool                   pipeline_manifest_record    = false;
	bool                   pipeline_manifest_strict    = KYTY_STRICT_PIPELINE_MANIFEST_DEFAULT != 0;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	std::filesystem::path  session_log_file;
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   tessellation_enabled        = true;
	bool                   occlusion_queries_enabled   = true;
	bool                   relaxed_shader_float        = false;
	uint32_t               wave_split_check_period     = 0;
	bool                   native_rt                   = false;
	bool                   async_pipelines_enabled     = true;
	bool                   async_pipelines_skip_more   = false;
	bool                   rt_stats_enabled            = false;
	uint32_t               rt_oracle                   = 0;
	uint32_t               rt_check_period             = 0;
	uint32_t               rt_check_ppm                = 1000;
	uint32_t               rt_check_strikes            = 3;
	bool                   perf_overlay_enabled            = false;
	uint32_t               gpu_pipeline_stages         = GPU_PIPELINE_ALL;
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
PresentMode GetPresentMode();
const std::string& GetGpuName();
bool     HdrEnabled();
bool     StretchEnabled();
bool     Ps5ProEnabled();
PssrUpscaler GetPssrUpscaler();
const std::filesystem::path& GetDlssFolder();
uint32_t GetDlssPreset();
const std::filesystem::path& GetFsrFolder();
uint32_t GetVblankFrequency();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
bool                   ShaderCacheEnabled();
bool                   PipelineManifestEnabled();
bool                   PipelineManifestRecord();
bool                   PipelineManifestStrict();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

bool GraphicsDebugDumpEnabled();

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();
std::filesystem::path GetSessionLogFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool TessellationEnabled();
bool OcclusionQueriesEnabled();
bool RelaxedShaderFloatEnabled();
uint32_t WaveSplitCheckPeriod();
bool NativeRtEnabled();
bool AsyncPipelinesEnabled();
bool AsyncPipelinesSkipMore();
bool RayTracingStatsEnabled();
uint32_t RayTracingOracle();
uint32_t RayTracingCheckPeriod();
uint32_t RayTracingCheckPpm();
uint32_t RayTracingCheckStrikes();
uint32_t GpuPipelineStages();
uint32_t ParseGpuPipelineStages(const char* value);

enum class GpuStageThread : uint32_t {
	CommandProcessor = 0,
	Execution        = 1,
	Recording        = 2,
	Resolve          = 3,
	HostCopy         = 4
};
void ConfigureGpuStageThread(GpuStageThread stage);

}

#endif
