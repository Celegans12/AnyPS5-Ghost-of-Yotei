#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RECORDPACKETS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RECORDPACKETS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

// The record packets: the work split of chenxiao07/KytyPS5 v20261020b (src/local/vulkan-recording.h, KYTY_VULKAN_RECORDING: draws and
// descriptor encoding as whole packets, KYTY_DEFERRED_SUBMIT), GPL-2.0, on top of our recording thread
// (commandRecorder.h): the GPU thread prepares a draw and appends ready-made
// packets to the record ring in program order; the recording thread is the only thread that calls the driver for the
// render scheduler's command buffers: their commands (already, one packet per vkCmd*), their begin, end and submission
// (already), and now also
//  - the descriptors of every commit (CommitBindings): push constants, push descriptors or the update of a descriptor
//    set taken from a descriptor heap the recording thread owns, and the binds, as one packet whose descriptor contents
//    the committing thread writes straight into the ring (descriptor update templates, the commit packet);
//    vkAllocateDescriptorSets and vkUpdateDescriptorSets of the heap no longer run on the GPU thread;
//  - the descriptor writes of the bindless set (BindlessTable), made by any thread : they
//    are copied into a queue in call order and moved into the ring by the recording side before its next packet
//    (CommandRecorder::QueueDescriptorWrites), i.e. before the commit that binds the set for the draw that needed them
//    and before the submission of its command buffer. Only the recording thread updates the set from then on: the "two
//    threads inside the driver on one descriptor set" of the teal-veil hypothesis (stage F
//    design, section 2.5) can no longer happen;
//  - a draw's dynamic state (viewports, scissors, depth/stencil, blend constants, color write enables, feedback loop) and
//    its graphics pipeline bind: the GPU thread computes the values from the draw's registers as before, the packet
//    carries them, and the recording thread decides with its own copy of what the command buffer holds
//    (KYTY_FIX_DRAW_STATE_FILTER's GraphicsState rules) which commands to make, at the same place of the stream.
// Lifetimes: everything a packet names is copied into it by value, or is a Vulkan object that lives until the GPU
// finished the tick of its use (image views, samplers, buffers, pipelines, templates, set layouts: the existing tick and
// drain rules; a queued descriptor write is in the ring before the submission of the tick its image could be freed after).
// Waits for GPU results are unchanged: a submission is a packet published at once, the fence waits wait for its tick.
// They need the recording thread with its submissions.
namespace Libs::Graphics {
class RenderContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::RecordPackets {

// Whether the packets are in use: the recording thread runs with its submissions (fixed for the run: writers on several
// threads must agree on the bindless set's queue).
[[nodiscard]] bool On();

// The recording thread: a descriptor set of `layout` for the command buffer of `tick`, from the heap that thread owns
// (created at its first use). The heap retires a pool at `tick` and reuses it once that tick completed, as the GPU
// thread's heap does with its current tick.
[[nodiscard]] vk::DescriptorSet WorkerHeapCommit(RenderContext& context, vk::DescriptorSetLayout layout, uint64_t tick);
// After the recording thread stopped (CommandScheduler::Shutdown): its heap is destroyed.
void Shutdown();

// Statistics (RECORD_PACKETS line, every ~5 s with the RECORD_THREAD line).
enum class Stat : uint32_t {
	CommitPackets,      // commits recorded as a packet
	HeapSets,           // descriptor sets the recording thread took from its heap
	DynPackets,         // draws whose dynamic state the recording thread made
	PipelinePackets,    // draws whose pipeline bind the recording thread made
	Invalidations,      // the recording thread's dynamic-state copy forgotten (new command buffer, blit, keeper switch)
	Count
};
void Note(Stat stat, uint64_t count = 1);
// The line (called by the recorder's report with the descriptor-write queue's counters of the interval).
void Report(double seconds, uint64_t queued_writes, uint64_t write_flushes);

} // namespace Libs::Graphics::RecordPackets

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RECORDPACKETS_H_
