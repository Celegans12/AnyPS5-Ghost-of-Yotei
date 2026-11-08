// Kyty's system overlay, performance panel, RenderDoc hook and perf sampler are not part of kyty_gpu.dll (AnyPS5 owns the
// window, its title and its input). What the presenter and the renderer call of them answers "off" here: no overlay is
// ever active, so the presenter never creates one.

#include "aps5/aps5Host.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/renderDoc.h"
#include "graphics/presentation/systemOverlay.h"

namespace Libs::Graphics {

SystemOverlayVisualState GetSystemOverlayVisualState() noexcept {
	return {false, 0};
}

void ShutdownSystemOverlayInput() {}

struct SystemOverlay::Impl {};

SystemOverlay::SystemOverlay(GraphicContext& /*graphics*/) {
	Aps5::Fatal("Kyty's system overlay is not part of kyty_gpu.dll");
}

SystemOverlay::~SystemOverlay() = default;

bool SystemOverlay::PrepareFrame(vk::Extent2D /*extent*/, vk::Format /*format*/, uint32_t /*image_count*/) {
	return false;
}

void SystemOverlay::Record(vk::CommandBuffer /*command*/, vk::ImageView /*target*/) {}

void SystemOverlay::ReleaseVulkan() {}

bool PerformanceOverlayEnabled() noexcept {
	return false;
}

void PerformanceOverlayRecordFlip(uint32_t /*guest_width*/, uint32_t /*guest_height*/) noexcept {}

void RenderDocInit() {}

void RenderDocRequestCapture() {}

void RenderDocOnGuestFlip(RenderContext& /*renderer*/) {}

// As perfSampler.cpp's own build without the sampler.
void PerfSamplerStartForCurrentThread() {}
void PerfReportTcLockPcs() {}
void CpuProbeReport() {}

} // namespace Libs::Graphics
