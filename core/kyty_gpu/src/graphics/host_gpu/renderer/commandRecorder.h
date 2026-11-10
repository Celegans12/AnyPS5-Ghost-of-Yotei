#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "graphics/host_gpu/vulkanCommon.h"

// The driver cost of recording the render scheduler's Vulkan commands (the vkCmd* calls) moves off the GPU thread (stage 1
// of the multithreading plan: the GPU thread ~8 ms per frame less).
//
// - Install() replaces the vkCmd* members of the default Vulkan-Hpp dispatcher.
// - Route(buffer) is called once the render scheduler has begun a command buffer. From then until
//   that buffer is ended, the hooked commands recorded into it are copied (arguments, the arrays
//   and structs they point to) into a ring and a worker thread calls the driver with the copies, in
//   the same order. Commands of any other command buffer (the presentation scheduler's, the
//   readback and upload helper buffers) are not routed: they go straight to the driver.
// - vkEndCommandBuffer on the routed buffer first waits until the worker has recorded everything
//   appended; so do all other vkCmd* commands that are not serialized and everything that could
//   observe or destroy what the ring still references (see commandRecorder.cpp).
// - A RECORD_THREAD line with counters is printed to stderr about every 5 s.
namespace Libs::Graphics::CommandRecorder {

namespace detail {
extern bool g_active; // Set once by Install() when the flag is on.
void        DrainBeforeDestroySlow();
} // namespace detail

// Whether Install() put the hooks in place (the device dispatcher was ready).
[[nodiscard]] inline bool Enabled() noexcept {
	return detail::g_active;
}

// Installs the hooks. Call right after VULKAN_HPP_DEFAULT_DISPATCHER.init(device), before any other thread records
// commands.
void Install();

// The render scheduler's command buffer was begun: commands recorded into it are routed from now
// on, until vkEndCommandBuffer. The calling thread is the one that records (Vulkan requires
// recording into one command buffer to be externally synchronized).
void Route(vk::CommandBuffer buffer);

// After the last command buffer was submitted and waited for: stops the worker thread.
void Shutdown();

// The routed buffer is begun, ended and
// submitted by the worker, so the recording thread never waits for it at a command-buffer boundary. The buffer is
// routed before it is begun; queue calls of other threads first call SyncQueue.
[[nodiscard]] bool AsyncSubmit() noexcept;

// Every queue call outside the render scheduler (present, overlay, bindless init,
// readback submits) calls this BEFORE it takes the queue lock: it waits until the worker made every render
// submission enqueued so far (the worker takes that lock to submit; a later batch waiting for a timeline value
// of an earlier, not yet submitted batch on the same queue would never run). A no-op otherwise.
void SyncQueue();

// The worker calls `call` with a copy of `bytes` of `blob`, after everything appended before (the recording thread
// only). Made visible to the worker at once.
void EnqueueCall(void (*call)(const void* blob), const void* blob, size_t bytes);

// The record packets (step 1 of option O1): room for `bytes` of data of
// one call the worker makes on `buffer` after everything appended before it (the recording thread only). Null when
// commands of `buffer` go straight to the driver (no worker, or another buffer): the caller then makes the call itself.
// Otherwise the caller fills the bytes and calls CommitCall; `call` gets them and makes its Vulkan calls through the
// dispatcher, which on the worker go straight to the driver. No other command may be recorded in between.
[[nodiscard]] void* ReserveCall(vk::CommandBuffer buffer, size_t bytes);
void                CommitCall(void (*call)(const void* blob));

// The record packets (recordPackets.h): whether the commands of `buffer` recorded by the calling thread now are
// made by the recording thread (the worker runs, `buffer` is routed, the caller is not the worker). No side effects.
[[nodiscard]] bool RecordedByWorker(vk::CommandBuffer buffer) noexcept;

// The record packets: descriptor writes into a set the recording thread binds (the bindless set), from any thread.
// They are copied (the writes and the infos their type reads) into a queue in call order; the recording side moves the
// queue into the ring before its next packet, so the recording thread makes them after every command appended before
// that point and before every command appended after it (in particular the commit that binds the set and the submission
// of the command buffer). False when the queue is not used (no recording thread with submissions, or it stopped): the
// caller makes the call itself. A write with a pNext chain or a descriptor type the ring does not copy ends the emulator.
[[nodiscard]] bool QueueDescriptorWrites(VkDevice device, uint32_t count, const VkWriteDescriptorSet* writes);

// Before a resource is destroyed that is not owned by the dispatcher's functions (VMA's buffers and
// images): waits until the worker has recorded everything appended so far, since an appended
// command may still name the resource. Cheap when the flag is off.
inline void DrainBeforeDestroy() {
	if (detail::g_active) {
		detail::DrainBeforeDestroySlow();
	}
}

} // namespace Libs::Graphics::CommandRecorder

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
