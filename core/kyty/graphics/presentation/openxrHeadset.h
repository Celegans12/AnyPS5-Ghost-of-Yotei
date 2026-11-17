#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_OPENXRHEADSET_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_OPENXRHEADSET_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <vector>

namespace Libs::Graphics {
struct GraphicContext;
struct VulkanImage;
class CommandBuffer;
}

namespace Libs::Vr {

void OpenXrInit();

void OpenXrAppendExtensions(std::vector<const char*>&                   out,
                            const std::vector<vk::ExtensionProperties>& available, bool device);

bool OpenXrPhysicalDevice(vk::Instance instance, vk::PhysicalDevice& out);

void OpenXrStartSession(Graphics::GraphicContext& graphics);

void OpenXrShutdown();

void OpenXrQueueEyeSource(Graphics::CommandBuffer& command, const Graphics::VulkanImage& source);

}

#endif
