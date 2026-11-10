#pragma once

#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

// TextureCache's logs, reports and counters (textureCacheTrace.cpp, with the members that only log or count): they
// change nothing the cache does. What the cache itself still keeps nearby is on its hot paths (TcOptNote,
// NoteFindTextureLazy) or the gate of a code path (TraceImageSyncAddr, EvictReuseTrace).
namespace Libs::Graphics {

// A colour-metadata slice that could not be read from the backing store (MaterializeColorClear).
void MetadataReadFailure(uint64_t address, uint64_t size);
void LogDccKey10Apply(const char* path, const TextureCache::ImageDesc& desc, const vk::ClearColorValue& clear);

// KYTY_TRACE_PERF: PERF image_unregister (TextureCache::UnregisterImage counts, TakeImageUnregisterStats reports).
extern std::mutex                                g_unregister_mutex;
extern std::unordered_map<std::string, uint64_t> g_unregister_keys;
extern std::map<std::string, uint64_t>           g_unregister_reasons;

// Image read coherence (textureCache.cpp): its counters and the IMAGE_READ_SERVE line.
struct ImageReadStats {
	std::atomic<uint64_t> serves {0};    // guest read faults that reached the GPU thread
	std::atomic<uint64_t> images {0};    // images copied
	std::atomic<uint64_t> bytes {0};     // bytes copied
	std::atomic<uint64_t> record_ns {0}; // GPU thread: recording and submitting them
	std::atomic<uint64_t> wait_ns {0};   // guest threads: waiting for the GPU and publishing
	std::atomic<uint64_t> late {0};      // copies not published: the image was written again meanwhile
	std::atomic<uint64_t> orphan {0};    // watched pages or images with nothing to serve
	std::atomic<uint64_t> emu_lift {0};  // emulator threads' accesses that faulted
	std::atomic<uint64_t> writes {0};    // guest threads' writes and HLE writes that lifted a page
	// Armed GPU-written images freed (FreeImage): taken over by an armed image written later over their bytes (transferred),
	// or copied to guest memory on their way out (detached), or - the guard of this class of error: expected 0, the first
	// ones logged as IMAGE_READ_FREED - freed with their bytes reaching guest memory never.
	std::atomic<uint64_t> transferred {0};
	std::atomic<uint64_t> detached {0};
	std::atomic<uint64_t> freed {0};
	std::atomic<uint64_t> plain {0};     // bytes of watched pages the emulator read from the backing mapping (ReadGuestPlain)
	// KYTY_TRACE_READBACK, under m_lock: since the last IMAGE_READ_SERVE line, by image address.
	struct Served {
		uint64_t bytes = 0;
		uint32_t count = 0, width = 0, height = 0, format = 0;
	};
	std::unordered_map<uint64_t, Served> served;
};
inline ImageReadStats& ReadStats() {
	static ImageReadStats stats;
	return stats;
}
inline bool ImageReadTrace() {
	static const bool trace = [] {
		const char* value = std::getenv("KYTY_TRACE_READBACK");
		return value != nullptr && value[0] == '1';
	}();
	return trace;
}

// KYTY_FIX_HTILE_SLICE_CLEAR=2: what the new lookup did and how its decisions compare with the old ones.
struct HtileSliceStats {
	std::atomic<uint64_t> clears {0};       // clears of layers inside the metadata (the old lookup missed them)
	std::atomic<uint64_t> base_partial {0}; // clears at the first byte covering only some layers (old: every layer)
	std::atomic<uint64_t> layers {0};       // layers marked cleared by these two
	std::atomic<uint64_t> partial {0};      // writes inside the metadata not made of whole layers (no change)
	std::atomic<uint64_t> consumed {0};     // draws that cleared a layer only the new lookup knew as cleared
	std::atomic<uint64_t> checked {0};      // decisions on layers 0..63 compared with the old mask
	std::atomic<uint64_t> diffs {0};        // of them, different without a new-only event behind it
	std::atomic<uint64_t> guard_inside {0}; // dispatches that read inside metadata, not taken as a clear
	std::atomic<uint32_t> frame {UINT32_MAX};
	std::mutex            report_lock;
	uint64_t              last[6]     = {};
	uint32_t              frame_lines = 0;
};
inline HtileSliceStats& SliceStats() {
	static HtileSliceStats stats;
	return stats;
}
void SliceStatsTick();

} // namespace Libs::Graphics
