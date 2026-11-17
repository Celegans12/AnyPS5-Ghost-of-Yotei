#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFORMANCEOVERLAY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PERFORMANCEOVERLAY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

void PerformanceOverlayRecordFlip(uint32_t guest_width, uint32_t guest_height) noexcept;

[[nodiscard]] uint64_t PerformanceOverlayGuestFlips() noexcept;

[[nodiscard]] bool PerformanceOverlayEnabled() noexcept;
void               TogglePerformanceOverlay() noexcept;
void TogglePerformanceOverlayDetails() noexcept;

[[nodiscard]] uint64_t PerformanceOverlayRevision() noexcept;

float DrawPerformanceOverlay(const GraphicContext& graphics, vk::Extent2D extent);

}

#endif
