#pragma once

#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shader.h"

#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics::ShaderStoreFormat {

enum class Record : uint8_t { Module, GraphicsPipeline, TessellationLayout };

struct Frame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	uint64_t checksum        = 0;
};

struct Key {
	ShaderType stage = ShaderType::Unknown;
	uint64_t hash = 0;
	uint32_t user_data_count = 0;
	std::vector<uint32_t> static_state;
};

template <class Stream, class Stamp, class Profile, class Limit>
void IoStoreHeader(Stream& io, Stamp& stamp, Profile& profile, Limit& max_shared_memory) {
	ShaderRecompiler::IR::IoAll(io, stamp, profile, max_shared_memory);
}

template <class Stream, class Module>
void IoStoreModuleInfo(Stream& io, Module& module) {
	ShaderRecompiler::IR::IoAll(io, module.id, module.info, module.specialization, module.vertex_formats,
	                            module.rect_parameters, module.mesh_task, module.guest_traversal);
}

template <class Stream, class Program, class Module, class Plan, class Spirv>
void IoStoreModule(Stream& io, Program& key, Module& module, Plan& plan, Spirv& spirv, Spirv& plain_spirv) {
	using ShaderRecompiler::IR::IoAll;
	IoAll(io, key.stage, key.hash, key.user_data_count, key.static_state, plan);
	IoStoreModuleInfo(io, module);
	IoAll(io, spirv, plain_spirv);
}

inline bool DecodeShaderStoreFrame(const Frame& frame, std::span<const uint8_t> compressed,
                                   std::vector<uint8_t>& raw) {
	if (compressed.size() != frame.compressed_size ||
	    XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return false;
	}
	raw.resize(frame.raw_size);
	return ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) == raw.size();
}

}
