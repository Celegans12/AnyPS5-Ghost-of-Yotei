#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstddef>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;

class SamplerCache {
public:
	explicit SamplerCache(GraphicContext& graphics): m_graphics(graphics) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	}
	~SamplerCache();
	KYTY_CLASS_NO_COPY(SamplerCache);

	// Samplers are created once per key and destroyed only by ~SamplerCache, never replaced: a
	// per-thread memo (samplerCache.cpp) answers repeated keys without taking m_mutex.
	vk::Sampler GetSampler(const ShaderSamplerResource& r, bool integer_border);

private:
	using SamplerKey = std::array<uint32_t, 5>;

	static uint64_t NextInstanceId();

	// m_mutex must be held; finds or creates the sampler of `key`.
	vk::Sampler GetSamplerLocked(const ShaderSamplerResource& r, bool integer_border,
	                             const SamplerKey& key);

	struct SamplerKeyHash {
		std::size_t operator()(const SamplerKey& key) const {
			std::size_t hash = 0;
			for (auto value: key) {
				hash ^= static_cast<std::size_t>(value) +
				        static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
				        (hash >> 2u);
			}
			return hash;
		}
	};

	GraphicContext&                                             m_graphics;
	const uint64_t                                              m_instance_id = NextInstanceId(); // never reused, names this cache in the per-thread memo
	Common::Mutex                                               m_mutex;
	std::unordered_map<SamplerKey, vk::Sampler, SamplerKeyHash> m_samplers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SAMPLERCACHE_H_
