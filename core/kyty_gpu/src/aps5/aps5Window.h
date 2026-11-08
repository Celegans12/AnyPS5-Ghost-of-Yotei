#ifndef KYTY_APS5_APS5WINDOW_H_
#define KYTY_APS5_APS5WINDOW_H_

// AnyPS5's window (Aps5GpuWindow) in place of Kyty's SDL3 window: vulkanWindow.cpp takes the Vulkan loader, the instance
// extensions and the surface from here when built into kyty_gpu.dll.

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include <vector>

namespace Aps5 {

[[nodiscard]] PFN_vkGetInstanceProcAddr WindowVulkanLoader();
[[nodiscard]] std::vector<const char*>  WindowInstanceExtensions();
[[nodiscard]] VkSurfaceKHR              WindowCreateSurface(VkInstance instance);
// Before a present: the window's drawable size now (AnyPS5 resizes it to the flip's size) becomes the size the swapchain
// follows (Swapchain::NeedsResize), as Kyty's window loop does on a resize event.
void WindowRefreshSize();

} // namespace Aps5

#endif // KYTY_APS5_APS5WINDOW_H_
