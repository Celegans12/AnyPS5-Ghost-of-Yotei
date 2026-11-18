#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RAY_TRACING_REPLAY_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RAY_TRACING_REPLAY_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

namespace Libs::Graphics {

template <typename Compiled>
[[nodiscard]] const char* RayTracingCheckUnreplayable(const Compiled& program) {
	const auto& info = program.info;
	if (program.has_address_writes) {
		return "writes through device addresses";
	}
	if (info.uses_gds) {
		return "uses GDS";
	}
	for (const auto& buffer: info.self_sized_buffers) {
		if (buffer.written) {
			return "writes a self-sized buffer";
		}
	}
	for (const auto& buffer: info.buffers) {
		if (buffer.written && buffer.indirect_write_table) {
			return "writes through an indirect write table";
		}
		if (buffer.written && buffer.descriptor_index == UINT32_MAX) {
			return "writes a buffer without a binding";
		}
	}
	for (const auto& image: info.images) {
		if (image.written && (image.heap || image.table_capacity != 0u)) {
			return "writes an image through a table";
		}
	}
	return nullptr;
}

}

#endif
