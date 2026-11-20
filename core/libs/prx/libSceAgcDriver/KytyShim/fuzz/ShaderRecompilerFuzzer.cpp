#include "common/assert.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineManifest.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include "spirv-tools/libspirv.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <span>
#include <variant>
#include <vector>

extern "C" size_t LLVMFuzzerMutate(uint8_t* data, size_t size, size_t max_size);

namespace {

namespace Gfx        = Libs::Graphics;
namespace Recompiler = Libs::Graphics::ShaderRecompiler;
namespace IR         = Libs::Graphics::ShaderRecompiler::IR;
using Gfx::ShaderType;

constexpr uint32_t SEndPgm = 0xbf810000u;
constexpr size_t   Header  = 4;
constexpr size_t   MaxWords = 8192;

constexpr std::array<uint32_t, 24> KnownWords = {
    0xbf810000u, 0xbf800000u, 0xbe800381u, 0x7e000301u, 0x7e020300u, 0x06000101u, 0x06020300u, 0x50000100u,
    0xd5190000u, 0xd7010000u, 0xbf8c0000u, 0xbf8c007fu, 0x7e000280u, 0x7e000281u, 0xbe820300u, 0x80000100u,
    0x81000001u, 0xbe80037eu, 0x7c000101u, 0xbf860001u, 0xbf850001u, 0xbefe0480u, 0xc0020000u, 0xe0301000u};

Recompiler::Spirv::Profile Profile() {
	return {.maintenance8           = true,
	        .native_float_min_max   = false,
	        .device_clock_period_ps = 1000,
	        .robust_buffer_access   = true,
	        .r8_srgb_views          = true,
	        .r8g8_srgb_views        = true,
	        .relaxed_float          = false,
	        .ray_query              = true,
	        .lds_lane_sync          = true};
}

bool Valid(const std::vector<uint32_t>& spirv) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	tools.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t&, const char*) {});
	spvtools::ValidatorOptions options;
	options.SetAllowOffsetTextureOperand(true);
	options.SetUniformBufferStandardLayout(true);
	return tools.Validate(spirv.data(), spirv.size(), options);
}

void Run(std::span<const uint32_t> code, uint8_t mode, uint8_t shape) {
	static const auto profile = Profile();
	Recompiler::CompileOptions options;
	options.shader_hash        = 0x1234567812345678ull;
	options.dump_ir            = false;
	options.emit_debug_info    = false;
	options.native_ray_tracing = true;
	options.ray_stats          = false;
	options.profile            = &profile;
	const std::vector<uint32_t> user_data(16u + (shape & 15u), 0u);
	options.user_data = user_data;

	Gfx::ShaderComputeInputInfo compute {};
	Gfx::ShaderVertexInputInfo  vertex {};
	Gfx::ShaderPixelInputInfo   pixel {};
	const uint32_t              wave = (mode & 1u) != 0 ? 32u : 64u;
	switch (mode % 6u) {
		case 3:
			options.stage             = ShaderType::Vertex;
			options.wave_size         = wave;
			options.user_data_base    = 8;
			options.input_info.vertex = &vertex;
			break;
		case 4:
			options.stage            = ShaderType::Pixel;
			options.wave_size        = wave;
			options.input_info.pixel = &pixel;
			break;
		default:
			compute.wave_size          = wave;
			compute.host_subgroup_size = 32;
			compute.threads_num[0]     = 1u << (shape & 3u);
			compute.threads_num[1]     = 1u << ((shape >> 2) & 3u);
			compute.threads_num[2]     = 1;
			compute.thread_ids_num     = 2;
			compute.group_id[0] = compute.group_id[1] = compute.group_id[2] = true;
			compute.workgroup_register = 16;
			compute.scratch_size_dwords = (mode % 6u) == 2 ? 1024u : 0u;
			compute.lds_size_dwords     = (mode % 6u) == 2 ? 2048u : 0u;
			options.stage              = ShaderType::Compute;
			options.wave_size          = wave;
			options.input_info.compute = &compute;
			break;
	}

	try {
		Common::SoftExitScope soft;
		auto                  source = Recompiler::PrepareShaderSource(code, options);
		if (!source.calls.empty() || source.unsupported_call_pc != UINT32_MAX) return;
		auto translated = Recompiler::TranslateProgram(source.decoded, options);
		IR::ResourceSpecialization specialization;
		specialization.buffers.resize(translated.program.info.buffers.size());
		specialization.images.resize(translated.program.info.images.size());
		specialization.samplers.resize(translated.program.info.samplers.size());
		for (size_t k = 0; k < specialization.images.size(); ++k) {
			specialization.images[k].numeric_class = Gfx::Prospero::TextureNumericClass::Float;
			specialization.images[k].dimension     = translated.program.info.images[k].dimension;
		}
		if (!IR::ResourceSpecializationFits(translated.program, specialization)) return;
		auto compiled = Recompiler::CompileProgram(std::move(translated), options, specialization, 0);
		if (!compiled.spirv.empty() && !Valid(compiled.spirv)) {
			std::fputs("kyty_shader_recompiler_fuzzer: the recompiler produced SPIR-V that spirv-val refuses\n", stderr);
			std::abort();
		}
	} catch (const std::exception&) {
	}
}

}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
	if (size < Header + 4) return 0;
	const size_t words = std::min((size - Header) / 4, MaxWords);
	std::vector<uint32_t> code(words);
	std::memcpy(code.data(), data + Header, words * 4);
	Run(code, data[0], data[1]);
	return 0;
}

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t* data, size_t size, size_t max_size, unsigned int seed) {
	std::mt19937 rng(seed);
	if (max_size < Header + 8) return size;
	if (size < Header + 8) {
		size = Header + 8;
		std::memset(data, 0, size);
		data[0] = static_cast<uint8_t>(rng());
		data[1] = static_cast<uint8_t>(rng());
		const uint32_t first = KnownWords[rng() % KnownWords.size()];
		std::memcpy(data + Header, &first, 4);
		std::memcpy(data + Header + 4, &SEndPgm, 4);
	}
	size                = Header + (size - Header) / 4 * 4;
	const size_t words  = (size - Header) / 4;
	const size_t rounds = 1 + rng() % 3;
	for (size_t round = 0; round < rounds; ++round) {
		const size_t at = Header + 4 * (rng() % words);
		uint32_t     word = 0;
		std::memcpy(&word, data + at, 4);
		switch (rng() % 7) {
			case 0: word ^= 1u << (rng() % 32); break;
			case 1: word = KnownWords[rng() % KnownWords.size()]; break;
			case 2: word = static_cast<uint32_t>(rng()); break;
			case 3: word = (word & 0xffff0000u) | (rng() & 0xffffu); break;
			case 4: word = (word & 0x0000ffffu) | (rng() << 16); break;
			case 5:
				if (size + 4 <= max_size) {
					std::memmove(data + at + 4, data + at, size - at);
					word = KnownWords[rng() % KnownWords.size()];
					size += 4;
				}
				break;
			default: {
				const size_t other = Header + 4 * (rng() % words);
				uint32_t     swap  = 0;
				std::memcpy(&swap, data + other, 4);
				std::memcpy(data + other, &word, 4);
				word = swap;
				break;
			}
		}
		std::memcpy(data + at, &word, 4);
	}
	if (rng() % 16 == 0) data[rng() % 2] = static_cast<uint8_t>(rng());
	if (rng() % 8 == 0) size = LLVMFuzzerMutate(data, size, max_size);
	uint32_t last = 0;
	if (size >= Header + 4) std::memcpy(&last, data + size - 4, 4);
	if (last != SEndPgm && size + 4 <= max_size) {
		std::memcpy(data + size, &SEndPgm, 4);
		size += 4;
	}
	return size;
}
