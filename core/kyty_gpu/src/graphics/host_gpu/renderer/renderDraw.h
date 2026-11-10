#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include <cstdint>
#include <utility>

namespace Libs::Graphics {

struct ShaderVertexInputInfo;

[[nodiscard]] std::pair<int32_t, uint32_t>
ResolveDrawOffsets(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info);

// Which offsets of a CPU-expanded indirect draw (base
// vertex, start instance) stay out of the host draw because the vertex shader adds the patched SGPRs
// itself. A shader without embedded-fetch metadata: both. With draw base once (a vertex
// stage's draw_base_once): each one its embedded-fetch plan does not name; a named one feeds the
// vertex inputs that replace its fetches, and the vertex entry takes it off v5/v8 again.
[[nodiscard]] std::pair<bool, bool>
IndirectDrawNativeOffsets(const ShaderVertexInputInfo& vs_input_info);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
