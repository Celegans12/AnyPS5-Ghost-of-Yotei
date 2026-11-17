#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_BDADIRTYREGIONS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_BDADIRTYREGIONS_H_

#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <bit>
#include <cstdint>

namespace Libs::Graphics::BdaDirtyRegions {

inline constexpr uint64_t Regions = TRACKER_ADDRESS_SIZE / TRACKER_REGION_SIZE;
inline constexpr uint64_t Words   = Regions / 64;

inline std::atomic<uint64_t> g_bits[Words] = {};

inline std::atomic<uint64_t> g_pending[Words] = {};
inline std::atomic<uint64_t> g_pending_words[Words / 64] = {};
inline std::atomic<uint64_t> g_pending_groups {0};
inline thread_local bool     g_translating = false;

inline void Mark(uint64_t region) {
	const auto word = region / 64;
	const auto bit  = uint64_t {1} << (region % 64);
	if (g_translating) {
		g_bits[word].fetch_or(bit, std::memory_order_release);
		return;
	}
	g_pending[word].fetch_or(bit, std::memory_order_acq_rel);
	g_pending_words[word / 64].fetch_or(uint64_t {1} << (word % 64), std::memory_order_acq_rel);
	g_pending_groups.fetch_or(uint64_t {1} << (word / 64), std::memory_order_release);
}

inline void Publish() {
	if (g_pending_groups.load(std::memory_order_acquire) == 0) return;
	for (auto groups = g_pending_groups.exchange(0, std::memory_order_acq_rel); groups != 0; groups &= groups - 1) {
		const auto group = static_cast<uint64_t>(std::countr_zero(groups));
		for (auto words = g_pending_words[group].exchange(0, std::memory_order_acq_rel); words != 0; words &= words - 1) {
			const auto word = group * 64 + static_cast<uint64_t>(std::countr_zero(words));
			if (const auto bits = g_pending[word].exchange(0, std::memory_order_acq_rel); bits != 0)
				g_bits[word].fetch_or(bits, std::memory_order_release);
		}
	}
}

}

#endif
