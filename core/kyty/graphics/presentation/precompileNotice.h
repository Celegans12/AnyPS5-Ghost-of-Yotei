#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRECOMPILENOTICE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRECOMPILENOTICE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

enum class PrecompileNoticeState : uint32_t {
	Waiting,
	Compiling,
};

struct PrecompileNotice {
	bool                  visible = false;
	PrecompileNoticeState state   = PrecompileNoticeState::Waiting;
	uint32_t done     = 0;
	uint32_t total    = 0;
	uint32_t shaders  = 0;
	uint32_t waited_pipelines = 0;
	uint32_t seconds_waited   = 0;
};

void ShowPrecompileNotice(const PrecompileNotice& notice);
void HidePrecompileNotice();

struct ManifestProgress {
	uint32_t done      = 0;
	uint32_t total     = 0;
	uint32_t pipelines = 0;
	uint32_t shaders   = 0;
};

void ShowManifestProgress(const ManifestProgress& progress);
void HideManifestProgress();
[[nodiscard]] bool ManifestProgressVisible() noexcept;

}

#endif
