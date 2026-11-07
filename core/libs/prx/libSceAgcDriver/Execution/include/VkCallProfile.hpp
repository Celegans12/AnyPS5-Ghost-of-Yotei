#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_VKCALLPROFILE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_VKCALLPROFILE_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

// v3 C1 measurement base (APS5_V3E_VK_PROFILE=1, start-time, default off): every Vulkan device entry point the
// driver resolves through vkGetDeviceProcAddr is wrapped in a timing trampoline, so each thread's time inside the
// driver's own calls (recording, submits, descriptor writes, object creation, waits) is measured per call and split
// by whether the thread held the GPU mutex. A [vkcalls] line reports it per window: every 10 s in the game, at each
// replayed frame's flip with frame windows (ProfileWindows::Close). With the switch off nothing is wrapped.
namespace AgcDriver::VkCallProfile {

bool Enabled();
PFN_vkGetDeviceProcAddr Wrap(PFN_vkGetDeviceProcAddr resolver);
void NameThread(const char* role);
void Close(const char* label);

}

#endif
