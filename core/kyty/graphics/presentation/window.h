#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

#include <array>
#include <string>

namespace Libs::Graphics {

class Presenter;
struct GraphicContext;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
void                     WindowShutdown();

struct DisplayLuminance {
	float max_full_frame = 80.0f;
	float max            = 1000.0f;
	float min            = 0.0f;
};

[[nodiscard]] DisplayLuminance WindowGetDisplayLuminance();

struct VulkanDeviceIdentity {
	uint32_t                vendor_id      = 0;
	uint32_t                device_id      = 0;
	uint32_t                driver_version = 0;
	std::array<uint8_t, 16> cache_uuid {};
	uint32_t                queue_family = 0;
};

[[nodiscard]] VulkanDeviceIdentity GetVulkanDeviceIdentity(const GraphicContext& graphics);

bool CreateHeadlessVulkan(GraphicContext& graphics, const VulkanDeviceIdentity* identity,
                          std::string& error);

}

#endif
