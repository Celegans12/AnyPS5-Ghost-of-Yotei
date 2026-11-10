#include "graphics/host_gpu/renderer/cache/samplerCache.h"

#include "common/assert.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/renderContext.h"

namespace Libs::Graphics {

// A sampler is created once per key and destroyed only by ~SamplerCache (m_samplers is never erased from or overwritten),
// so each thread keeps a direct-mapped memo key -> handle in front of the locked lookup. An entry names its SamplerCache
// by a never reused id: the entries of a destroyed cache can not match another one.

namespace {

constexpr size_t SamplerMemoSize = 256; // power of two

struct SamplerMemoEntry {
	uint64_t                owner = 0; // SamplerCache::m_instance_id, 0 = empty
	std::array<uint32_t, 5> key {};
	vk::Sampler             sampler;
};

thread_local std::array<SamplerMemoEntry, SamplerMemoSize> t_sampler_memo;

size_t SamplerMemoSlot(const std::array<uint32_t, 5>& key) {
	uint64_t hash = 0x9e3779b97f4a7c15ull;
	for (const auto value: key) {
		hash = (hash ^ value) * 0xff51afd7ed558ccdull;
		hash ^= hash >> 32u;
	}
	return static_cast<size_t>(hash) & (SamplerMemoSize - 1u);
}

} // namespace

uint64_t SamplerCache::NextInstanceId() {
	static std::atomic<uint64_t> next {1};
	return next.fetch_add(1, std::memory_order_relaxed);
}

SamplerCache::~SamplerCache() {
	for (const auto& [key, sampler]: m_samplers) {
		(void)key;
		m_graphics.device.destroySampler(sampler, nullptr);
	}
}

vk::Sampler SamplerCache::GetSampler(const ShaderSamplerResource& r, bool integer_border) {
	const SamplerKey key {r.fields[0], r.fields[1], r.fields[2], r.fields[3], integer_border};
	auto& entry = t_sampler_memo[SamplerMemoSlot(key)];
	if (entry.owner == m_instance_id && entry.key == key) return entry.sampler;

	Common::LockGuard lock(m_mutex);
	const auto sampler = GetSamplerLocked(r, integer_border, key);
	entry.owner   = m_instance_id;
	entry.key     = key;
	entry.sampler = sampler;
	return sampler;
}

vk::Sampler SamplerCache::GetSamplerLocked(const ShaderSamplerResource& r, bool integer_border,
                                           const SamplerKey& key) {
	if (auto iter = m_samplers.find(key); iter != m_samplers.end()) {
		return iter->second;
	}

	float      aniso_ratio = 1.0f;
	const auto mag_filter  = r.XyMagFilter();
	const auto min_filter  = r.XyMinFilter();

	auto is_aniso_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kAnisoPoint:
			case Prospero::SamplerFilter::kAnisoLinear: return true;
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kBilinear: return false;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return false;
	};

	auto to_vk_filter = [](uint8_t filter) {
		switch (static_cast<Prospero::SamplerFilter>(filter)) {
			case Prospero::SamplerFilter::kPoint:
			case Prospero::SamplerFilter::kAnisoPoint: return vk::Filter::eNearest;
			case Prospero::SamplerFilter::kBilinear:
			case Prospero::SamplerFilter::kAnisoLinear: return vk::Filter::eLinear;
			default: EXIT("unknown sampler filter: %u\n", filter);
		}
		return vk::Filter::eNearest;
	};

	const bool aniso = is_aniso_filter(mag_filter) || is_aniso_filter(min_filter);
	if (aniso) {
		switch (static_cast<Prospero::SamplerAnisoRatio>(r.MaxAnisoRatio())) {
			case Prospero::SamplerAnisoRatio::kOne: aniso_ratio = 1.0f; break;
			case Prospero::SamplerAnisoRatio::kTwo: aniso_ratio = 2.0f; break;
			case Prospero::SamplerAnisoRatio::kFour: aniso_ratio = 4.0f; break;
			case Prospero::SamplerAnisoRatio::kEight: aniso_ratio = 8.0f; break;
			case Prospero::SamplerAnisoRatio::kSixteen:
			// Reserved encodings 5-7 fall back to the maximum ratio.
			default: aniso_ratio = 16.0f; break;
		}
	}

	const auto mip_filter = r.MipFilter();
	float      min_lod    = 0.0f;
	float      max_lod    = 0.0f;
	if (static_cast<Prospero::SamplerMipFilter>(mip_filter) != Prospero::SamplerMipFilter::kNone) {
		min_lod = static_cast<float>(r.MinLod()) / 256.0f;
		max_lod = static_cast<float>(r.MaxLod()) / 256.0f;
	}

	vk::SamplerCreateInfo sampler_info {};

	auto to_vk_address_mode = [](uint8_t clamp) {
		switch (static_cast<Prospero::SamplerClampMode>(clamp)) {
			case Prospero::SamplerClampMode::kWrap: return vk::SamplerAddressMode::eRepeat;
			case Prospero::SamplerClampMode::kMirror:
				return vk::SamplerAddressMode::eMirroredRepeat;
			case Prospero::SamplerClampMode::kClampLastTexel:
				return vk::SamplerAddressMode::eClampToEdge;
			case Prospero::SamplerClampMode::kMirrorOnceLastTexel:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampHalfBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceHalfBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			case Prospero::SamplerClampMode::kClampBorder:
				return vk::SamplerAddressMode::eClampToBorder;
			case Prospero::SamplerClampMode::kMirrorOnceBorder:
				return vk::SamplerAddressMode::eMirrorClampToEdge;
			default: EXIT("unknown clamp: %u\n", clamp);
		}
		return vk::SamplerAddressMode::eClampToBorder;
	};

	vk::BorderColor border = integer_border ? vk::BorderColor::eIntTransparentBlack
	                                       : vk::BorderColor::eFloatTransparentBlack;
	switch (static_cast<Prospero::SamplerBorderColor>(r.BorderColorType())) {
		case Prospero::SamplerBorderColor::kTransBlack: break;
		case Prospero::SamplerBorderColor::kOpaqueBlack:
			border = integer_border ? vk::BorderColor::eIntOpaqueBlack
			                        : vk::BorderColor::eFloatOpaqueBlack;
			break;
		case Prospero::SamplerBorderColor::kOpaqueWhite:
			border = integer_border ? vk::BorderColor::eIntOpaqueWhite
			                        : vk::BorderColor::eFloatOpaqueWhite;
			break;
		case Prospero::SamplerBorderColor::kFromTable:
			LOGF(
			    "temporary: approximating table border color as transparent black, index = %" PRIu16
			    "\n",
			    r.BorderColorPtr());
			break;
		default: EXIT("unknown border color: %d", static_cast<int>(r.BorderColorType()));
	}

	sampler_info.magFilter = to_vk_filter(mag_filter);
	sampler_info.minFilter = to_vk_filter(min_filter);
	sampler_info.mipmapMode =
	    (static_cast<Prospero::SamplerMipFilter>(mip_filter) == Prospero::SamplerMipFilter::kLinear
	         ? vk::SamplerMipmapMode::eLinear
	         : vk::SamplerMipmapMode::eNearest);
	sampler_info.addressModeU = to_vk_address_mode(r.ClampX());
	sampler_info.addressModeV = to_vk_address_mode(r.ClampY());
	sampler_info.addressModeW = to_vk_address_mode(r.ClampZ());
    const auto max_lod_bias = m_graphics.GetPhysicalDeviceProperties().limits.maxSamplerLodBias;
    sampler_info.mipLodBias = std::clamp(
        static_cast<float>(static_cast<int16_t>((r.LodBias() ^ 0x2000u) - 0x2000u)) / 256.0f,
        -max_lod_bias, max_lod_bias
    );
	sampler_info.anisotropyEnable        = (aniso ? VK_TRUE : VK_FALSE);
	sampler_info.maxAnisotropy           = aniso_ratio;
	sampler_info.compareEnable           = (r.DepthCompareFunc() != 0 ? VK_TRUE : VK_FALSE);
	sampler_info.compareOp               = static_cast<vk::CompareOp>(r.DepthCompareFunc());
	if (r.DepthCompareFunc() != 0) {
		static std::atomic_uint logged_funcs {0};
		const auto bit = 1u << r.DepthCompareFunc();
		if ((logged_funcs.fetch_or(bit) & bit) == 0) {
			std::fprintf(stderr, "SAMPLER_COMPARE func=%u\n", static_cast<uint32_t>(r.DepthCompareFunc()));
		}
	}
	sampler_info.minLod                  = min_lod;
	sampler_info.maxLod                  = max_lod;
	sampler_info.borderColor             = border;
	sampler_info.unnormalizedCoordinates = (r.ForceUnormCoords() ? VK_TRUE : VK_FALSE);

	if (r.ForceUnormCoords()) {
		sampler_info.addressModeU     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeV     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.addressModeW     = vk::SamplerAddressMode::eClampToEdge;
		sampler_info.mipmapMode       = vk::SamplerMipmapMode::eNearest;
		sampler_info.minLod           = 0.0f;
		sampler_info.maxLod           = 0.0f;
		sampler_info.anisotropyEnable = VK_FALSE;
		sampler_info.maxAnisotropy    = 1.0f;
		sampler_info.compareEnable    = VK_FALSE;
		sampler_info.mipLodBias       = 0.0f;
	}

	// FILTER_MODE (S# word 0 bits 29-30): 0 blends the filter footprint, 1 returns its minimum and
	// 2 its maximum (used e.g. for min/max depth pyramids).
	vk::SamplerReductionModeCreateInfo reduction {};
	if (r.FilterMode() == 1u || r.FilterMode() == 2u) {
		reduction.reductionMode = r.FilterMode() == 1u ? vk::SamplerReductionMode::eMin : vk::SamplerReductionMode::eMax;
		sampler_info.pNext      = &reduction;
		static std::atomic_uint logged {0};
		if (logged.fetch_add(1) < 4u) {
			std::fprintf(stderr, "SAMPLER_REDUCTION mode=%s mag=%u min=%u mip=%u\n",
			             r.FilterMode() == 1u ? "min" : "max", static_cast<uint32_t>(r.XyMagFilter()),
			             static_cast<uint32_t>(r.XyMinFilter()), static_cast<uint32_t>(r.MipFilter()));
		}
	}

	vk::Sampler vk_sampler = nullptr;
	const auto  result     = m_graphics.device.createSampler(&sampler_info, nullptr, &vk_sampler);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || vk_sampler == nullptr);

	m_samplers.emplace(key, vk_sampler);
	return vk_sampler;
}

} // namespace Libs::Graphics
