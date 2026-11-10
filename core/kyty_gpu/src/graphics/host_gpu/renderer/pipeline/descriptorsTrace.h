#pragma once

#include "common/onceValue.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

class RenderContext;
struct TextureBinding;

namespace DescriptorTrace {

// KYTY_TRACE_TEX_ADDRS, KYTY_TRACE_TEX_HASH, KYTY_TRACE_TEX_WRITES_EXTENT: a binding of PrepareBindings' image loop.
[[nodiscard]] inline bool ImageBindingsTraced() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] {
		return std::getenv("KYTY_TRACE_TEX_ADDRS") != nullptr || std::getenv("KYTY_TRACE_TEX_HASH") != nullptr ||
		       std::getenv("KYTY_TRACE_TEX_WRITES_EXTENT") != nullptr;
	});
}
void ImageBinding(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t i,
                  const TextureBinding& binding);

// KYTY_TRACE_SRT_DUMP.
[[nodiscard]] inline bool SrtDumpOn() {
	static constinit Common::OnceValue<bool> on_once;
	return on_once.Get([] { return std::getenv("KYTY_TRACE_SRT_DUMP") != nullptr; });
}
void SrtDump(uint64_t hash, const std::vector<uint32_t>& words);

} // namespace DescriptorTrace

} // namespace Libs::Graphics
