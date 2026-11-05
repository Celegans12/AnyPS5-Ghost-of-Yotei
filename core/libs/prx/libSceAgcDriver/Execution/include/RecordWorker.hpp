#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECORDWORKER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_RECORDWORKER_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>

// v3 C1-R, the recording worker (APS5_V3E_RECORD_WORKER=1, start-time, default off): the driver's command-buffer
// recording calls (vkBeginCommandBuffer, vkCmd*, vkEndCommandBuffer, vkResetCommandBuffer) are copied into one FIFO
// on the calling thread and made by a worker thread, in the order they were made, so the threads that record under
// the GPU mutex pay a copy instead of the driver call. Every other call that can observe or end a recording (queue
// submits and waits, command pool and command buffer allocation, the destruction of any object a command can name,
// descriptor pool resets) first waits until the worker made every call queued before it, and is then made on the
// calling thread while no new call can be queued. A recording call whose arguments carry a structure chain or an
// inheritance info is made the same way (drained, on the calling thread). The worker's own call failures surface at
// the next drained call: a queue submit returns the first failure instead of submitting.
namespace AgcDriver::RecordWorker {

bool Enabled();
PFN_vkGetDeviceProcAddr Wrap(PFN_vkGetDeviceProcAddr resolver);
void Drain();
// The committer calls this once per item: with APS5_V3E_WORKER_PIN (RuntimeSwitch) set, the worker runs on the
// process's two highest logical processors and the committer on the others; unset, both may run anywhere.
// APS5_V3E_WORKER_NO_SPIN (RuntimeSwitch): the worker sleeps at once when the FIFO is empty instead of spinning.
void FollowCommitterAffinity();

struct Counters {
    std::uint64_t queued = 0;
    std::uint64_t queuedBytes = 0;
    std::uint64_t direct = 0;
    std::uint64_t drains = 0;
    std::uint64_t drainWaits = 0;
    std::uint64_t drainWaitNs = 0;
    std::uint64_t workerNs = 0;
    std::uint64_t workerWakes = 0;
};
Counters Totals();
void Close(const char* label);
void ForTestsReset(PFN_vkGetDeviceProcAddr resolver);

}

#endif
