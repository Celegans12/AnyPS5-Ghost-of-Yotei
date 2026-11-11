#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PENDINGCLEARNOTE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PENDINGCLEARNOTE_H_

// KYTY_DIAG_PENDING_CLEAR_CHECK=1 (AnyPS5 lane v4x2; see pendingClearCheck.h): the switch, and the note of how a depth
// surface's fast-clear flag was set (TextureCache::ClearMetaWrite / ClearMetaRmw returned true), kept apart from the
// check itself so that the buffer cache can note a fill without the renderer's headers:
//  - path: an in-stream or shader-recognised fill of the HTILE (the fill word is known), a compute shader that only
//    writes it, or the read-modify-write shader;
//  - the depth clear register of the context at that dispatch, where the caller has it (it is the register's value
//    at that moment, not necessarily this surface's);
//  - how often. The first 6 notes per HTILE address and path are logged (PENDING_CLEAR_CHECK clear_set).
// An HTILE word without stencil: bits 3..0 the Z mask (0: the tile is cleared), 17..4 the minimum Z, 31..18 the maximum
// Z, each of 14 bits (0x3fff = 1.0).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics::PendingClearCheck {

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_PENDING_CLEAR_CHECK");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) std::fprintf(stderr, "DIAG_PENDING_CLEAR_CHECK enabled\n");
		return set;
	}();
	return on;
}

enum class ClearPath : uint32_t { Fill, ComputeWrite, ComputeRmw, Count };
inline constexpr const char* ClearPathNames[] = {"fill", "cs_write", "cs_rmw"};

struct ClearSet {
	uint64_t  count[static_cast<uint32_t>(ClearPath::Count)] {};
	ClearPath last_path = ClearPath::Fill;
	uint64_t  size      = 0, shader_hash = 0;
	bool      has_word = false, has_register = false;
	uint32_t  word           = 0;
	float     depth_register = 0.0f;
	uint32_t  frame          = 0;
};

struct ClearSets {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, ClearSet> by_address; // the address the clear was given (a surface's HTILE base, or
	                                                   // an address inside a layered HTILE)
};
// Never destroyed (as the check's state).
inline ClearSets& Sets() {
	static ClearSets* sets = new ClearSets;
	return *sets;
}

inline void NoteClearSet(uint64_t address, uint64_t size, ClearPath path, bool has_word, uint32_t word, bool has_register,
                         float depth_register, uint64_t shader_hash, uint32_t frame) {
	auto&            sets = Sets();
	std::scoped_lock lock(sets.mutex);
	auto&            set = sets.by_address[address];
	const auto       n   = ++set.count[static_cast<uint32_t>(path)];
	set.last_path        = path;
	set.size             = size;
	set.shader_hash      = shader_hash;
	set.has_word         = has_word;
	set.word             = word;
	set.has_register     = has_register;
	set.depth_register   = depth_register;
	set.frame            = frame;
	if (n <= 6u) {
		std::fprintf(stderr,
		             "PENDING_CLEAR_CHECK clear_set frame=%u htile=0x%llx size=0x%llx path=%s has_word=%d word=0x%08x "
		             "zmask=%u minz=%u maxz=%u has_register=%d depth_register=%g hash=%016llx\n",
		             frame, static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
		             ClearPathNames[static_cast<uint32_t>(path)], has_word ? 1 : 0, word, word & 0xfu, (word >> 4u) & 0x3fffu,
		             (word >> 18u) & 0x3fffu, has_register ? 1 : 0, depth_register,
		             static_cast<unsigned long long>(shader_hash));
	}
}

} // namespace Libs::Graphics::PendingClearCheck

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PENDINGCLEARNOTE_H_ */
