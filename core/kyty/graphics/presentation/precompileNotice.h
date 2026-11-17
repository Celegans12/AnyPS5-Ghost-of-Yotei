#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRECOMPILENOTICE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRECOMPILENOTICE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

union SDL_Event;

namespace Libs::Graphics {

enum class PrecompileNoticeState : uint32_t {
	Question,
	Paused,
	Waiting,
	Ready,
	Compiling,
};

struct PrecompileNotice {
	bool                  visible = false;
	PrecompileNoticeState state   = PrecompileNoticeState::Paused;
	uint32_t done     = 0;
	uint32_t total    = 0;
	uint32_t shaders  = 0;
	uint32_t waited_pipelines = 0;
	uint32_t seconds_waited   = 0;
};

enum class PrecompileAnswer : uint32_t { None, Wait, Play };

void ShowPrecompileNotice(const PrecompileNotice& notice);
void HidePrecompileNotice();
[[nodiscard]] PrecompileAnswer TakePrecompileAnswer() noexcept;

[[nodiscard]] bool     PrecompileNoticeVisible() noexcept;
[[nodiscard]] uint64_t PrecompileNoticeRevision() noexcept;

bool ProcessPrecompileNoticeInput(const SDL_Event& event);

struct ManifestProgress {
	uint32_t done      = 0;
	uint32_t total     = 0;
	uint32_t pipelines = 0;
	uint32_t shaders   = 0;
};

void ShowManifestProgress(const ManifestProgress& progress);
void HideManifestProgress();
[[nodiscard]] bool ManifestProgressVisible() noexcept;

void DrawPrecompileNotice(vk::Extent2D extent);

}

#endif
