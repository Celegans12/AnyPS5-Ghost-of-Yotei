#include "graphics/host_gpu/renderer/pipeline/descriptorsTrace.h"

#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/envDefaults.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/workaroundHits.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/debugSample.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawRepeat.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/perfCounters.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/recordPackets.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/stageFLookup.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

// The binding paths' logs and censuses (descriptors.cpp calls them; each is off unless its name is set).
namespace Libs::Graphics {

namespace DescriptorTrace {

namespace {
const std::string& SrtDumpList() {
	static constinit Common::OnceObject<std::string> list_once;
	return list_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_SRT_DUMP");
		return value != nullptr ? std::string(value) : std::string();
	});
}
} // namespace

void ImageBinding(RenderContext& context, const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t i,
                  const TextureBinding& binding) {
	// KYTY_TRACE_TEX_ADDRS=<hex,hex,...>: every distinct (shader, binding) view of those guest
	// addresses, with the extent/format the descriptor asked for.
	static constinit Common::OnceObject<std::string> tex_addrs_once;
	const std::string& tex_addrs = tex_addrs_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_TEX_ADDRS");
		return value != nullptr ? std::string(value) : std::string();
	});
	// KYTY_TRACE_TEX_HASH=<hex hash>: the same for every binding of that shader.
	static constinit Common::OnceValue<uint64_t> tex_hash_once;
	const uint64_t tex_hash = tex_hash_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_TEX_HASH");
		return value != nullptr ? std::strtoull(value, nullptr, 16) : 0ull;
	});
	if ((!tex_addrs.empty() || tex_hash != 0) && binding.desc.info.data.address != 0) {
		char text[24];
		std::snprintf(text, sizeof(text), "%llx", static_cast<unsigned long long>(binding.desc.info.data.address));
		if ((!tex_addrs.empty() && tex_addrs.find(text) != std::string::npos) ||
		    program.shader_hash == tex_hash) {
			static std::mutex                                            mutex;
			static std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> seen;
			std::scoped_lock                                             lock(mutex);
			uint32_t image_width = 0, image_height = 0, image_tile = 0, image_depth = 0;
			std::string image_format;
			if (binding.image_id) {
				const auto& bound = context.GetTextureCache().GetImage(binding.image_id);
				image_width       = bound.backing.extent.width;
				image_height      = bound.backing.extent.height;
				image_tile        = static_cast<uint32_t>(bound.info.tile_mode);
				image_depth       = bound.info.IsDepth() ? 1u : 0u;
				image_format      = vk::to_string(bound.backing.format);
			}
			const auto key = std::make_tuple(program.shader_hash, i, binding.desc.info.extent.width * 65536u + image_width);
			if (std::ranges::find(seen, key) == seen.end() && seen.size() < 400u) {
				seen.push_back(key);
				std::fprintf(stderr, "TEX_VIEW addr=0x%s stage=%u hash=0x%016llx binding=%u %s extent=%ux%u image=%ux%u fmt=%s tile=0x%x guest_fmt=%u | bound_fmt=%s bound_tile=0x%x bound_depth=%u marker=[%s]\n",
				             text, static_cast<uint32_t>(program.stage),
				             static_cast<unsigned long long>(program.shader_hash), i,
				             program.info.images[i].written ? "write" : "read", binding.desc.info.extent.width,
				             binding.desc.info.extent.height, image_width, image_height,
				             vk::to_string(binding.desc.info.pixel_format).c_str(),
				             static_cast<uint32_t>(binding.desc.info.tile_mode),
				             static_cast<uint32_t>(binding.desc.info.guest_format), image_format.c_str(),
				             image_tile, image_depth, DebugMarkerPath().c_str());
			}
		}
	}
	// KYTY_TRACE_TEX_WRITES_EXTENT=<W>x<H>: every distinct (shader, binding) storage write to an
	// image of that extent, with its address (finds the pass that writes a small intermediate image).
	static constinit Common::OnceObject<std::pair<uint32_t, uint32_t>> write_extent_once;
	const std::pair<uint32_t, uint32_t>& write_extent = write_extent_once.Get([] {
		const char* value = std::getenv("KYTY_TRACE_TEX_WRITES_EXTENT");
		uint32_t    w = 0, h = 0;
		if (value != nullptr && std::sscanf(value, "%ux%u", &w, &h) != 2) w = h = 0;
		return std::pair<uint32_t, uint32_t> {w, h};
	});
	if (write_extent.first != 0 && program.info.images[i].written &&
	    binding.desc.info.extent.width == write_extent.first && binding.desc.info.extent.height == write_extent.second) {
		static std::mutex                                  mutex;
		static std::vector<std::pair<uint64_t, uint64_t>>  seen;
		std::scoped_lock                                   lock(mutex);
		const auto key = std::make_pair(program.shader_hash, binding.desc.info.data.address);
		if (std::ranges::find(seen, key) == seen.end() && seen.size() < 200u) {
			seen.push_back(key);
			uint32_t bw = 0, bh = 0;
			if (binding.image_id) {
				const auto& bound = context.GetTextureCache().GetImage(binding.image_id);
				bw = bound.backing.extent.width;
				bh = bound.backing.extent.height;
			}
			std::fprintf(stderr, "TEX_WRITE frame=%u addr=0x%llx stage=%u hash=0x%016llx binding=%u extent=%ux%u image=%ux%u fmt=%s marker=[%s]\n",
			             DebugPresentCount().load(), static_cast<unsigned long long>(binding.desc.info.data.address),
			             static_cast<uint32_t>(program.stage), static_cast<unsigned long long>(program.shader_hash), i,
			             binding.desc.info.extent.width, binding.desc.info.extent.height, bw, bh,
			             vk::to_string(binding.desc.info.pixel_format).c_str(), DebugMarkerPath().c_str());
		}
	}
}

void SrtDump(uint64_t hash, const std::vector<uint32_t>& words) {
	// KYTY_TRACE_SRT_DUMP=<hex hash,...>: the listed shaders' flattened SRT words (as floats), every
	// 150th use, first 12 dumps; for reading the constants a pass computes with.
	if (SrtDumpList().find(fmt::format("{:016x}", hash)) != std::string::npos) {
		static std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> counters;
		auto& [uses, dumps] = counters[hash];
		if (uses++ % 150u == 0u && dumps++ < 12u) {
			std::string line = fmt::format("SRT_DUMP hash=0x{:016x} words={}", hash, words.size());
			for (size_t w = 0; w < words.size(); w++) {
				const auto word = words[w];
				line += fmt::format(" {}:{:08x}/{:g}", w, word, std::bit_cast<float>(word));
			}
			std::fprintf(stderr, "%s\n", line.c_str());
		}
	}
}

} // namespace DescriptorTrace

} // namespace Libs::Graphics
