// Headset output without the OpenXR loader. Upstream's openxrHeadset.cpp needs the OpenXR SDK (3rdparty FetchContent),
// which this build does not fetch (offline build, FETCHCONTENT_FULLY_DISCONNECTED); every entry point reports that no
// headset session exists, so --vr runs the flat image as without a headset runtime.
#include "graphics/presentation/openxrHeadset.h"
#include "graphics/presentation/vrInput.h"

namespace Libs::Vr {

void OpenXrInit() {}

void OpenXrAppendExtensions(std::vector<const char*>& /*out*/, const std::vector<vk::ExtensionProperties>& /*available*/,
                            bool /*device*/) {}

bool OpenXrPhysicalDevice(vk::Instance /*instance*/, vk::PhysicalDevice& /*out*/) {
	return false;
}

void OpenXrStartSession(Graphics::GraphicContext& /*graphics*/) {}

void OpenXrShutdown() {}

void OpenXrQueueEyeSource(Graphics::CommandBuffer& /*command*/, const Graphics::VulkanImage& /*source*/) {}

VrPose OpenXrGetHeadPose() {
	return {};
}

VrControllerInput OpenXrGetController(int /*hand*/) {
	return {};
}

} // namespace Libs::Vr
