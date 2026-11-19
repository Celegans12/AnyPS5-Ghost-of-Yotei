#pragma once

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct Program;

inline constexpr uint64_t RelativeVgprMark  = uint64_t {1} << 63u;
inline constexpr uint64_t RelativeVgprWrite = uint64_t {1} << 62u;

constexpr uint64_t RelativeVgprReadFlags(uint32_t base, uint32_t steps) {
	return RelativeVgprMark | (uint64_t {steps & 0xffffu} << 32u) | (uint64_t {base & 0xffu} << 24u);
}

constexpr uint64_t RelativeVgprWriteFlags(uint32_t site, uint32_t base, uint32_t offset, uint32_t count) {
	return RelativeVgprMark | RelativeVgprWrite | (uint64_t {count & 0x3fffu} << 48u) |
	       (uint64_t {offset & 0xffffu} << 32u) | (uint64_t {base & 0xffu} << 24u) | (site & 0xffffffu);
}

constexpr uint64_t RelativeVgprPhiFlags(uint32_t type, uint32_t reg) {
	return RelativeVgprMark | (uint64_t {reg & 0xffu} << 40u) | type;
}

uint32_t LowerRelativeVgprs(Program& program);

}
