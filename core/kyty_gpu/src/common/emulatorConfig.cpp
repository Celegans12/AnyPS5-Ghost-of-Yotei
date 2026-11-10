#include "common/emulatorConfig.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/threads.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <span>
#include <string>
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
	EXIT_IF(cfg.user_name.empty() || cfg.user_name.size() > MAX_USER_NAME_LENGTH);
	EXIT_IF(!IsConfiguredUserIdValid(cfg.user_id));
	EXIT_IF(cfg.controller_speaker_volume > 100 || cfg.controller_vibration_intensity > 100);
	EXIT_IF(cfg.keyboard_player > static_cast<uint32_t>(MAX_PLAYERS));

	*g_config = cfg;
	Common::DebugCounters::SetEnabled(cfg.perf_overlay_enabled);
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

const std::string& GetUserName() {
	return g_config->user_name;
}

int32_t GetUserId() {
	return g_config->user_id;
}

const std::string& GetAudioInputDevice() {
	return g_config->audio_input_device;
}

const std::optional<ControllerColor>& GetControllerColor() {
	return g_config->controller_color;
}

uint32_t GetControllerSpeakerVolume() {
	return g_config->controller_speaker_volume;
}

uint32_t GetControllerVibrationIntensity() {
	return g_config->controller_vibration_intensity;
}

uint32_t GetKeyboardPlayer() {
	return g_config->keyboard_player;
}

int32_t GetPlayerUserId(int player) {
	return g_config->user_id + player * PLAYER_USER_ID_STEP;
}

int GetPlayerFromUserId(int32_t user_id) {
	for (int player = 0; player < MAX_PLAYERS; player++) {
		if (user_id == GetPlayerUserId(player)) {
			return player;
		}
	}
	return -1;
}

bool IsGuestUserId(int32_t user_id) {
	return GetPlayerFromUserId(user_id) > 0;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

int32_t GetGpuIndex() {
	return g_config->gpu_index;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

bool HdrEnabled() {
	return g_config->hdr_enabled;
}

bool HideCursorEnabled() {
	return g_config->hide_cursor_enabled;
}

bool StretchEnabled() {
	return g_config->stretch_enabled;
}

bool VrEnabled() {
	return g_config->vr_enabled;
}

bool Ps5ProEnabled() {
	return g_config->pro_enabled;
}

bool AmdCpuEnabled() {
	return g_config->amd_cpu_enabled;
}

bool GuestTscEnabled() {
	return g_config->guest_tsc_enabled;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool IsNetworkConnected() {
	return g_config->network_connected;
}

bool PsnEmulationEnabled() {
	return g_config->psn_emulation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

bool ShaderCacheEnabled() {
	return g_config->shader_cache_enabled;
}

ShaderOptimizationType GetShaderOptimizationType() {
	return g_config->shader_optimization_type;
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

bool ProfilerEnabled() {
	return g_config->profiler_enabled;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	// Always on (M118, whatever the launcher's setting says): GPU-written linear images are written back to guest memory,
	// as on PS5 where the CPU sees what the GPU wrote. Ghost of Yotei's CPU occlusion test reads the hi-Z pyramid from
	// memory; without it the test used a pyramid frozen around a cutscene and distant objects vanished every other frame.
	return true;
}

bool ReadbackTiledImagesEnabled() {
	return g_config->readback_tiled_images;
}

bool SyncRawImageBuffersEnabled() {
	return g_config->sync_raw_image_buffers;
}

bool TessellationEnabled() {
	return g_config->tessellation_enabled;
}

bool OcclusionQueriesEnabled() {
	return g_config->occlusion_queries_enabled;
}

bool TrophyEnabled() {
	return g_config->trophy_enabled;
}

uint32_t GetTrophyDurationSeconds() {
	return std::clamp(g_config->trophy_duration_seconds, 5u, 10u);
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

bool SkipNoticeScreen() {
	return g_config->skip_notice_screen;
}

bool PrecompileOnly() {
	return g_config->precompile_only;
}

uint32_t GetPrecompileThreads() {
	return g_config->precompile_threads;
}

const std::filesystem::path& GetPrecompileExportHints() {
	return g_config->precompile_export_hints;
}

const std::filesystem::path& GetPrecompileImportHints() {
	return g_config->precompile_import_hints;
}

const std::filesystem::path& GetGameDir() {
	return g_config->game_dir;
}

// Based on TheCruZ/KytyPS5-GTA 12ec19890 and dd7a6e008 (emulatorConfig.cpp), GPL-2.0. The choice of cores and the
// priority rule are ours: four whole physical cores of the processor (Common::ChooseGpuStageCpus; on a 9800X3D physical
// cores 1, 3, 4 and 2), away from the interrupt-heavy core 0; only the threads that own the stream get the higher
// priority (the recording and resolve threads spin). KYTY_GPU_THREAD_CPUS=<e>,<r>,<v>,<c> or off: the user's choice.
void ConfigureGpuStageThread(GpuStageThread stage) {
	static const std::array<int, 4> cpus = [] {
		std::array<int, 4> result = Common::ChooseGpuStageCpus(Common::Thread::SystemLogicalCpus());
		const char*        value  = std::getenv("KYTY_GPU_THREAD_CPUS");
		if (value == nullptr) {
			return result;
		}
		if (std::string_view(value) == "off") {
			return std::array<int, 4> {-1, -1, -1, -1};
		}
		for (auto& cpu: result) {
			char* end = nullptr;
			cpu       = static_cast<int>(std::strtol(value, &end, 10));
			if (end == value) {
				EXIT("invalid KYTY_GPU_THREAD_CPUS: %s\n", std::getenv("KYTY_GPU_THREAD_CPUS"));
			}
			value = *end == ',' ? end + 1 : end;
		}
		return result;
	}();
	static const bool high_priority = [] {
		const char* value = std::getenv("KYTY_GPU_THREAD_PRIORITY");
		return value == nullptr || std::string_view(value) != "0";
	}();
	if (high_priority && (stage == GpuStageThread::Execution || stage == GpuStageThread::CommandProcessor)) {
		Common::Thread::RaiseCurrentPriority();
	}
	// Other threads otherwise run on the SMT siblings of the pinned stages and slow them down: give the stages whole
	// physical cores. KYTY_GPU_THREAD_MIN_CORES: the physical cores the rest of the process must keep (default 4: an
	// 8-core processor reserves, a 6-core one does not).
	static std::once_flag reserved;
	std::call_once(reserved, [] {
		const char*  value     = std::getenv("KYTY_GPU_THREAD_MIN_CORES");
		const size_t min_cores = value != nullptr ? std::strtoul(value, nullptr, 10) : 4u;
		const auto   result    = Common::Thread::ReserveCores(cpus, min_cores);
		std::fprintf(stderr, "GPU_THREAD_CPUS execution=%d recording=%d resolve=%d command_processor=%d: %s\n", cpus[0],
		             cpus[1], cpus[2], cpus[3], result.c_str());
	});
	Common::Thread::PinCurrent(cpus[static_cast<uint32_t>(stage)]);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

const std::vector<std::string>& GetGuestArguments() {
	return g_config->guest_arguments;
}

} // namespace Config
