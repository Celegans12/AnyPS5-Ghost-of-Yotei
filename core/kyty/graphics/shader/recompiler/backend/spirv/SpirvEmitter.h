#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

struct Profile {
	bool maintenance8         = false;
	bool native_float_min_max = false;
	uint32_t device_clock_period_ps = 0;
	bool robust_buffer_access = false;
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
	bool relaxed_float = false;
	bool ray_query = false;
	bool lds_lane_sync = true;
};

inline constexpr Profile DefaultProfile {};

struct SplitHints;

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool           emit_debug_info       = false,
                                  bool           mip_stats_records     = true,
                                  bool           ray_stats             = false,
                                  SplitHints*    split_hints           = nullptr);

[[nodiscard]] bool UsesMipStats(const IR::Program& program);
}

#endif
