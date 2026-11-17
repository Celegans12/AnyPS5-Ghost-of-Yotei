#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_RAYQUERY_H_

#include <array>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

constexpr uint32_t MaxRayQuerySites = 4;
constexpr uint32_t BvhIntersectNative = 1u << 8;
constexpr uint32_t RayQueryListSize = 32;

enum class RayQueryResultKind : uint32_t {
	Has        = 0,
	T          = 1,
	Word       = 2,
	InstanceLo = 3,
	InstanceHi = 4,
	U          = 5,
	V          = 6,
	Node       = 7,
};

constexpr uint32_t RayQueryInitBothFacings = 1u;
constexpr uint32_t RayQueryInitGeometryOpacity = 2u;
constexpr uint32_t RayQueryInitNoLeaf = 4u;
constexpr uint32_t RayQueryInitAllCandidates = 8u;
constexpr uint32_t RayQueryInitOpaqueOnly = 16u;

constexpr uint32_t RayQueryKeyLaneLeaf = 1u;

constexpr uint32_t RayQueryAlphaScalar = 0u;
constexpr uint32_t RayQueryAlphaLane   = 1u;
constexpr uint32_t RayQueryAlphaFactor = 2u;
constexpr uint32_t RayQueryAlphaEntry  = 4u;

enum class RayQueryMarkKind : uint32_t {
	Resolve = 1,
	Exit    = 4,
};

constexpr uint32_t NoField = 0xffffffffu;

struct RayTracingLayout {
	bool                    valid           = false;
	uint32_t                instance_offset = 0;
	uint32_t                record_stride   = 128;
	std::array<uint8_t, 12> transform {};
	uint32_t                blas_dword      = 0;
	bool                    blas_packed     = false;
	uint32_t                root_dword      = 0;
	uint32_t                mask_dword      = 0;
	uint32_t                mask_shift      = 0;
	uint32_t                mask_width      = 0;
	uint32_t                flags_dword     = NoField;
	uint32_t                flags_shift     = 0;
	bool                    flip_facing     = true;
	uint32_t                inline_type     = NoField;
	uint32_t                inline_offset   = 0;
	uint32_t                inline_root     = 0;
	bool                    node_count_80   = false;

	bool operator==(const RayTracingLayout&) const = default;
	[[nodiscard]] std::array<uint32_t, 4> Pack() const;
	[[nodiscard]] uint32_t                TransformBase() const;
};

struct RayTracingAlphaFields {
	uint32_t material_dword = 0;
	uint32_t table_dword    = 0;
	uint32_t first_dword    = 0;
	uint32_t vertex_mask    = 0xfc;
	bool     vertex_align   = false;
	[[nodiscard]] uint32_t Pack(const RayTracingLayout& layout) const;
};

[[nodiscard]] constexpr uint32_t RayQueryFlags(uint32_t site, uint32_t variant) noexcept {
	return site | (variant << 8u);
}
[[nodiscard]] constexpr uint32_t RayQuerySite(uint32_t flags) noexcept {
	return flags & 0xffu;
}
[[nodiscard]] constexpr uint32_t RayQueryVariant(uint32_t flags) noexcept {
	return flags >> 8u;
}

}

#endif
