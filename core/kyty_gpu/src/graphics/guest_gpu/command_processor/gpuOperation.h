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
} // namespace HW

// The register state a draw or a dispatch reads when it executes.
struct GpuRegisterState {
	HW::Context*    context     = nullptr;
	HW::UserConfig* user_config = nullptr;
	HW::Shader*     shaders     = nullptr;
};

// NUM_INSTANCES as the command processor last set it. Indirect draws also set the persistent
// instance count, from arguments read when they execute; `sequence` counts NUM_INSTANCES
// packets so an execution can tell which of the two was written last.
struct GpuNumInstances {
	uint32_t value    = 1;
	uint64_t sequence = 0;
};

// The SH registers (user SGPRs) a draw-indirect packet writes its offsets to, or Pm4::SH_NOP.
struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
	// Multi-draw packets with DRAW_INDEX_ENABLE: the SGPR receiving the draw index. Only the
	// GPU indirect path consumes it; 0x280 is Pm4::SH_NOP.
	uint32_t draw_index = 0x280;
};

struct GpuDrawIndirectArgs {
	uint64_t                 args_base         = 0;
	uint32_t                 data_offset       = 0;
	uint32_t                 max_count         = 1;
	const volatile uint32_t* count_addr        = nullptr;
	uint32_t                 stride            = 0;
	uint32_t                 draw_initiator    = 0;
	uint32_t                 index_type        = 0;
	uint32_t                 index_buffer_size = 0;
	uint32_t                 user_data_marker  = 0; // HW::UserSgprType of the offsets the CPU expansion writes
	uint64_t                 index_base        = 0;
	bool                     indexed           = false;
	bool                     multi             = false;
	IndirectDrawRegisters    registers {};
};

struct GpuDispatchArgs {
	uint32_t thread_group_x = 0;
	uint32_t thread_group_y = 0;
	uint32_t thread_group_z = 0;
	uint32_t mode           = 0;
	uint64_t args_addr      = 0; // Indirect dispatches.
};

// Callbacks of queued operations live in the operation itself.
using GpuCallback = Common::InlineFunction<128>;

enum class GpuOperationKind : uint8_t {
	DrawIndex,
	DrawAuto,
	DrawIndirect,
	DispatchDirect,
	DispatchIndirect,
	// Any other guest-visible effect or renderer work, in command order.
	Callback,
	// A callback that reads the registers of its point of the stream.
	StateCallback,
};

// Why the command processor waits for the execution thread to catch up (PERF gpu_pipeline_waits).
enum class GpuDrain : uint32_t { Fault, Predication, CondGuard, DirectWrites, Journal, Idle, Count };

// One unit of emulated GPU work produced by a command processor. Operations execute in the
// order the command processors produce them; every guest-visible effect of a command buffer
// (draws, dispatches, labels, interrupts, flips, DMA) is an operation.
struct GpuOperation {
	GpuOperationKind  kind      = GpuOperationKind::Callback;
	CommandProcessor* processor = nullptr;
	uint64_t          submit_id = 0;
	// Position in the stream; see GuestGpu::NextSequence().
	uint64_t          sequence  = 0;
	GpuRegisterState  state;
	GpuNumInstances   num_instances;
	// Draws and dispatches: what the resolve thread resolved ahead, if anything.
	RenderExecutor::ResolvedDraw* resolved = nullptr;
	union {
		DrawIndexArgs       draw_index;
		DrawAutoArgs        draw_auto;
		GpuDrawIndirectArgs draw_indirect;
		GpuDispatchArgs     dispatch;
	};
	GpuCallback callback;

	GpuOperation(): draw_index {} {}
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GPU_OPERATION_H
