#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMESTEPHACK_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_FRAMESTEPHACK_H_

#include <cstdint>
#include <span>

namespace Libs::VideoOut::FrameStepHack {

struct CodeRange {
	uint64_t begin = 0;
	uint64_t end   = 0;
};

void PatchMainProgram(uint64_t base_vaddr, uint64_t size, uint64_t mapped_size,
                      std::span<const CodeRange> code, bool cheat_file) noexcept;

void OnPresentedFrame() noexcept;

}

#endif
