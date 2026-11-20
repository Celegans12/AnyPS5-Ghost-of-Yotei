#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COMPLETIONINVALIDATION_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_COMPLETIONINVALIDATION_H_

#include "common/threadSafety.h"

#include <cstdint>
#include <mutex>

namespace Libs::Graphics {

enum class CompletionInvalidation : uint8_t { InPlace, Deferred };

template <typename Lock, typename NeedsFlush, typename Invalidate, typename Defer>
CompletionInvalidation InvalidateWithoutWaiting(Lock& lock, NeedsFlush&& needs_flush,
                                                Invalidate&& invalidate, Defer&& defer) KYTY_NO_THREAD_SAFETY_ANALYSIS {
	if (lock.TryLockSpinning()) {
		std::unique_lock guard {lock, std::adopt_lock};
		if (!needs_flush()) {
			invalidate();
			return CompletionInvalidation::InPlace;
		}
	}
	defer();
	return CompletionInvalidation::Deferred;
}

}

#endif
