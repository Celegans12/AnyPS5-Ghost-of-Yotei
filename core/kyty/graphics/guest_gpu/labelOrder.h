#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_LABELORDER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_LABELORDER_H_

#include <cstddef>
#include <cstdint>

namespace Libs::Graphics::LabelOrder {

struct Facts {
	bool deferable      = false;
	bool tracked_page   = false;
	bool tracked_bytes  = false;
	bool in_place_block = false;
	bool in_place_bytes = false;
	bool around_enabled = false;
	bool backing        = false;
	bool label_pending  = false;
};

enum class Route : uint8_t { Defer, DeferAround, Drain };

enum class Reason : uint8_t {
	NotDeferable,
	InPlaceBytes,
	CacheBytes,
	AroundOff,
	NoBacking,
	InPlaceBlock,
	NoLabelPending,
	Other,
	Count
};

inline constexpr size_t ReasonCount = static_cast<size_t>(Reason::Count);

[[nodiscard]] constexpr Route Current(const Facts& facts) noexcept {
	if (!facts.deferable) {
		return Route::Drain;
	}
	if (!facts.tracked_page && !facts.in_place_block) {
		return Route::Defer;
	}
	const bool around = facts.around_enabled && facts.label_pending && !facts.in_place_block && !facts.tracked_bytes &&
	                    facts.backing;
	return around ? Route::DeferAround : Route::Drain;
}

[[nodiscard]] constexpr Route AtCompletion(const Facts& facts) noexcept {
	if (!facts.deferable || facts.in_place_bytes) {
		return Route::Drain;
	}
	if (!facts.tracked_page) {
		return Route::Defer;
	}
	return facts.around_enabled && !facts.tracked_bytes && facts.backing ? Route::DeferAround : Route::Drain;
}

[[nodiscard]] constexpr Reason DrainReason(const Facts& facts) noexcept {
	if (!facts.deferable) {
		return Reason::NotDeferable;
	}
	if (facts.in_place_bytes) {
		return Reason::InPlaceBytes;
	}
	if (!facts.tracked_page) {
		return facts.in_place_block ? Reason::InPlaceBlock : Reason::Other;
	}
	if (facts.tracked_bytes) {
		return Reason::CacheBytes;
	}
	if (!facts.around_enabled) {
		return Reason::AroundOff;
	}
	if (!facts.backing) {
		return Reason::NoBacking;
	}
	if (facts.in_place_block) {
		return Reason::InPlaceBlock;
	}
	return facts.label_pending ? Reason::Other : Reason::NoLabelPending;
}

}

#endif
