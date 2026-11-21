#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include <cstdint>
#include <cstdio>
#include <random>

namespace Libs::Graphics {

struct WriteTicksTestAccess {
	using Ticks = BufferCache::WriteTicks;
};

}

namespace {

using Ticks = Libs::Graphics::WriteTicksTestAccess::Ticks;

int g_failures = 0;

void Check(bool condition, const char* what, int round, int step) {
	if (!condition && g_failures++ < 8) {
		std::fprintf(stderr, "FAILED: %s (round %d, step %d)\n", what, round, step);
	}
}

}

extern "C" int RunKytyWriteTicksTest() {
	std::mt19937_64 random {0x77c4a9e1u};
	uint64_t        reused = 0;
	for (int round = 0; round < 48 && g_failures == 0; ++round) {
		Ticks          reference;
		Ticks          recycled;
		uint64_t       tick  = 1;
		const uint64_t span  = round % 3 == 0 ? 4096u : round % 3 == 1 ? 65536u : 1u << 22u;
		const uint64_t sizes = round % 2 == 0 ? 64u : 8192u;
		const uint64_t align = round % 4 < 2 ? 16u : 1u;
		uint64_t       last_begin = 0;
		uint64_t       last_end   = 0;
		for (int step = 0; step < 20000 && g_failures == 0; ++step) {
			if (random() % 16u == 0u) ++tick;
			uint64_t begin = random() % span / align * align;
			uint64_t end   = begin + random() % sizes;
			if (random() % 4u == 0u && last_end != 0u) {
				begin = last_begin;
				end   = last_end;
			}
			last_begin = begin;
			last_end   = end;
			reference.Assign(begin, end, tick);
			recycled.AssignRecycled(begin, end, tick);
			Check(recycled.SameRanges(reference), "the recycled ranges differ from the reference", round, step);
			Check(recycled.RangeCount() == reference.RangeCount(), "the range counts differ", round, step);
			const uint64_t from = random() % span;
			const uint64_t to   = from + random() % (sizes * 2u);
			Check(recycled.Latest(from, to) == reference.Latest(from, to), "Latest differs", round, step);
		}
		reused += recycled.reused;
	}
	Check(reused != 0u, "no node was reused", -1, -1);
	if (g_failures != 0) {
		return 1;
	}
	std::puts("kyty_shim_write_ticks: ok");
	return 0;
}
