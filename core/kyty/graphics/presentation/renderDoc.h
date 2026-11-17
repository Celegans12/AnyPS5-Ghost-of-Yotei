#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_RENDERDOC_H_

namespace Libs::Graphics {

class RenderContext;

void RenderDocInit();
void RenderDocRequestCapture();
void RenderDocOnGuestFlip(RenderContext& renderer);

}

#endif
