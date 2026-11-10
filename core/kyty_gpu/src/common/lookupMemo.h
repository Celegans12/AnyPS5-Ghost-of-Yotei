#ifndef KYTY_COMMON_LOOKUPMEMO_H_
#define KYTY_COMMON_LOOKUPMEMO_H_

#include "common/envRead.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>

// Lookups of the GPU thread (and the resolve thread) answered from per-thread memos of facts that stay true until a counted change:
//  - Shader: the shader map entry of an address (ShaderGetMappedData / ShaderLookupMapped) read at map generation G is
//    the map's entry while the generation is still G (every map write bumps it under the map's mutex);
//  - Permutation: a program source's last matching permutation is tried first (at most one permutation matches, so
//    the full scan would return the same one);
//  - Format: Vulkan image format properties (GraphicContext::GetImageFormatProperties), device facts that never change;
//  - ImageStart: "no registered image starts at this address" (TextureCache::FindImageFromRange): per 64 KiB granule
//    (hashed) a count of the registered images whose first byte lies there, changed with the registration under the
//    texture cache's lock; a count of 0 means none starts there, so the search would find no match.
// Their hits: the PERF lookup_memo line (~5 s, KYTY_TRACE_PERF).
namespace Common::LookupMemo {

enum class Kind : uint32_t { Shader, Permutation, Format, ImageStart, Count };

struct Counters {
	std::atomic<uint64_t> hits[4] {};
	std::atomic<int64_t>  window_start_ns {0};
};

inline Counters& Stats() {
	static Counters counters;
	return counters;
}

inline void Report() {
	auto&         c   = Stats();
	const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	int64_t start = c.window_start_ns.load(std::memory_order_relaxed);
	if (start == 0) {
		c.window_start_ns.compare_exchange_strong(start, now);
		return;
	}
	if (now - start < 5'000'000'000 || !c.window_start_ns.compare_exchange_strong(start, now)) return;
	if (!EnvRead::PerfLines()) return;
	uint64_t hits[4] {};
	for (uint32_t k = 0; k < 4; k++) hits[k] = c.hits[k].exchange(0, std::memory_order_relaxed);
	const auto u = [](uint64_t a) { return static_cast<unsigned long long>(a); };
	std::fprintf(stderr, "PERF lookup_memo hits shader=%llu perm=%llu format=%llu image_start=%llu\n", u(hits[0]), u(hits[1]),
	             u(hits[2]), u(hits[3]));
}

// A memo answer that was used.
inline void NoteHit(Kind kind) {
	const uint32_t k = static_cast<uint32_t>(kind);
	auto&          c = Stats();
	const uint64_t n = c.hits[k].fetch_add(1, std::memory_order_relaxed);
	if ((n & 4095u) == 0u) Report();
}

} // namespace Common::LookupMemo

#endif // KYTY_COMMON_LOOKUPMEMO_H_
