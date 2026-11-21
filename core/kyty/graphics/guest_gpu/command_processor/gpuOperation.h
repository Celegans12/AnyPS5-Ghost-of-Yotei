#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H

#include "common/inlineFunction.h"
#include "graphics/host_gpu/renderer/render.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandProcessor;

namespace HW {
class Context;
class UserConfig;
class Shader;
}

struct GpuRegisterState {
	const HW::Context*    context     = nullptr;
	const HW::UserConfig* user_config = nullptr;
	const HW::Shader*     shaders     = nullptr;
	uint64_t context_id = 0;
	uint64_t user_config_id = 0;
};

struct GpuNumInstances {
	uint32_t value    = 1;
	uint64_t sequence = 0;
};

struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
	uint32_t draw_index;
};

struct GpuDrawIndirectArgs {
	uint64_t                 args_base         = 0;
	uint32_t                 data_offset       = 0;
	uint32_t                 max_count         = 1;
	const volatile uint32_t* count_addr        = nullptr;
	uint32_t                 stride            = 0;
	uint32_t                 draw_initiator    = 0;
	uint32_t                 index_buffer_size = 0;
	uint8_t                  index_type        = 0;
	bool                     indexed           = false;
	bool                     multi             = false;
	uint64_t                 index_base        = 0;
	IndirectDrawRegisters    registers {};
	const uint32_t*          run_offsets       = nullptr;
};
static_assert(sizeof(GpuDrawIndirectArgs) <= sizeof(DrawIndexArgs));

struct GpuDispatchArgs {
	uint32_t thread_group_x = 0;
	uint32_t thread_group_y = 0;
	uint32_t thread_group_z = 0;
	uint32_t mode           = 0;
	uint64_t args_addr      = 0;
};

struct GpuGuestWrite {
	uint64_t        address     = 0;
	const uint32_t* data        = nullptr;
	uint32_t        dwords      = 0;
	bool            one_address = false;
};

using GpuCallback = Common::InlineFunction<96>;

enum class GpuOperationKind : uint8_t {
	DrawIndex,
	DrawAuto,
	DrawIndirect,
	DispatchDirect,
	DispatchIndirect,
	Callback,
};

struct GpuOperation {
	GpuOperationKind  kind      = GpuOperationKind::Callback;
	CommandProcessor* processor = nullptr;
	uint64_t          submit_id = 0;
	uint64_t          sequence  = 0;
	GpuRegisterState  state;
	GpuNumInstances   num_instances;
	RenderExecutor::ResolvedDraw* resolved = nullptr;
	RenderExecutor::AheadNote     ahead_note = RenderExecutor::AheadNote::NotAttempted;
	E2Stats::Refusal              e2;
	bool after_wait = false;
	bool guest_sync = true;
	bool clean = false;
	bool stream_in_place = false;
	uint32_t gpu_time_marker = 0;
	const void*     table_ticket   = nullptr;
	const uint32_t* scout_packet   = nullptr;
	uint64_t        scout_sequence = 0;
	uint32_t        scout_ordinal  = 0;
	union {
		DrawIndexArgs       draw_index;
		DrawAutoArgs        draw_auto;
		GpuDrawIndirectArgs draw_indirect;
		GpuDispatchArgs     dispatch;
	};
	GpuCallback callback;
	GpuGuestWrite guest_write;

	GpuOperation(): draw_index {} {}
};

}

#endif
