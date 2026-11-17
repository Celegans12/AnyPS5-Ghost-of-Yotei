#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LODSTATSREPORT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_LODSTATSREPORT_H_

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace Libs::Graphics::LodStatsReport {

constexpr uint32_t Counters   = 256;
constexpr uint32_t Entries    = Counters + 1;
constexpr uint32_t ReportSize = 64 + Counters * 8;
constexpr uint32_t Unsampled  = 0xffffffffu;
constexpr uint32_t NoCounterFlag = 0x8000u;

[[nodiscard]] inline bool HasCounter(const uint32_t* tsharp) {
	return ((tsharp[5] >> 25u) & 1u) != 0u;
}

[[nodiscard]] inline uint32_t ImageField(const uint32_t* tsharp) {
	if (!HasCounter(tsharp)) {
		return NoCounterFlag;
	}
	const uint32_t id         = tsharp[6] & 0xffu;
	const uint32_t base_level = (tsharp[3] >> 12u) & 0xfu;
	const uint32_t min_lod    = (tsharp[1] >> 8u) & 0xfffu;
	return id | (base_level << 8u) | (min_lod << 16u);
}

inline void RecordSample(uint32_t* words, uint32_t field, float lod) {
	if ((field & NoCounterFlag) != 0u) {
		return;
	}
	const uint32_t counter   = field & 0xffu;
	const float    absolute  = lod + static_cast<float>((field >> 8u) & 0xfu);
	const float    clamped   = std::clamp(absolute, 0.0f, 14.0f);
	const auto     level     = static_cast<uint32_t>(clamped);
	const float    threshold = static_cast<float>((field >> 16u) & 0xfffu) / 256.0f;
	words[counter]           = std::min(words[counter], level);
	if (clamped < threshold) {
		words[Entries + counter]++;
	}
}

[[nodiscard]] inline uint64_t PackEntry(uint32_t finest, uint32_t count, uint32_t counter) {
	const uint64_t level = finest == Unsampled ? 0xfu : std::min<uint32_t>(finest, 14u);
	return (level << 56u) | (uint64_t {counter & 0xffu} << 24u) |
	       std::min<uint32_t>(count, 0xffffffu);
}

inline void PackReport(const uint32_t* words, uint8_t* report) {
	std::memset(report, 0, ReportSize);
	const uint32_t valid = 1;
	std::memcpy(report, &valid, sizeof(valid));
	for (uint32_t counter = 0; counter < Counters; counter++) {
		const auto entry = PackEntry(words[counter], words[Entries + counter], counter);
		std::memcpy(report + 64 + counter * sizeof(uint64_t), &entry, sizeof(entry));
	}
}

}

#endif
