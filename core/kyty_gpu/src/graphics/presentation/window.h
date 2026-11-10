#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

#include <atomic>

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
void                     WindowShutdown();
// --precompile-only (instead of WindowRun; the window was made hidden): PipelineCache::RunPrecompileOnly of the
// renderer, its saves under the renderer's lock. Returns the process exit code.
[[nodiscard]] int WindowPrecompileOnly(uint32_t threads, const std::atomic<bool>& stop);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
