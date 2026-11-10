#include "graphics/host_gpu/renderer/stageFLookup.h"

#include <array>
#include <atomic>
#include <cstdio>

// The resolve thread's lookups (stageFLookup.h): their counters.
namespace Libs::Graphics::StageF {

namespace {

constexpr size_t kBufferOutcomes = static_cast<size_t>(BufferOutcome::Count);
constexpr size_t kBufferUses     = static_cast<size_t>(BufferUse::Count);

// GPU thread only (FindBuffers, the PERF report). Per window.
struct LookupCounters {
	std::array<uint64_t, kBufferOutcomes> buffer_miss {}; // the resolve thread's outcome when it found nothing usable
	std::array<uint64_t, kBufferUses>     buffer_use {};
};
LookupCounters g_lookup;

// Resolve thread (written), PERF report (read).
struct ResolveLookupCounters {
	std::atomic<uint64_t> buffer_batches {0};
	std::atomic<uint64_t> buffer_busy {0}; // batches whose lock was held by the GPU thread
	std::atomic<uint64_t> buffer_lookups {0};
	std::atomic<uint64_t> buffer_found {0};
};
ResolveLookupCounters g_resolve_lookup;

uint64_t TakeCount(std::atomic<uint64_t>& counter) {
	return counter.exchange(0, std::memory_order_relaxed);
}

} // namespace

void NoteBufferMiss(BufferOutcome outcome) {
	g_lookup.buffer_miss[static_cast<size_t>(outcome)]++;
}

void NoteBufferUse(BufferUse use, uint64_t count) {
	g_lookup.buffer_use[static_cast<size_t>(use)] += count;
}

void NoteResolveBufferBatch(bool busy, uint64_t lookups, uint64_t found) {
	auto& s = g_resolve_lookup;
	s.buffer_batches.fetch_add(1, std::memory_order_relaxed);
	if (busy) s.buffer_busy.fetch_add(1, std::memory_order_relaxed);
	if (lookups != 0) s.buffer_lookups.fetch_add(lookups, std::memory_order_relaxed);
	if (found != 0) s.buffer_found.fetch_add(found, std::memory_order_relaxed);
}

void ReportLookupPerf() {
	auto&      c = g_lookup;
	auto&      s = g_resolve_lookup;
	const auto u = [](uint64_t value) { return static_cast<unsigned long long>(value); };
	const auto percent = [](uint64_t part, uint64_t whole) {
		return whole == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
	};
	const auto bmiss = [&](BufferOutcome outcome) { return c.buffer_miss[static_cast<size_t>(outcome)]; };
	const auto buse  = [&](BufferUse what) { return c.buffer_use[static_cast<size_t>(what)]; };
	uint64_t   buffers = 0;
	for (const auto value: c.buffer_miss) buffers += value;
	for (const auto value: c.buffer_use) buffers += value;
	std::fprintf(stderr,
	             "PERF stagef_lookup buf=%llu used=%llu(%.1f%%) miss[no_record=%llu not_ahead=%llu "
	             "none=%llu busy=%llu short=%llu null=%llu clamped=%llu no_buffer=%llu words=%llu written=%llu stale=%llu "
	             "stale_ranges=%llu] resolve[buf_batches=%llu buf_busy=%llu buf_lookups=%llu "
	             "buf_found=%llu]\n",
	             u(buffers), u(buse(BufferUse::Used)), percent(buse(BufferUse::Used), buffers),
	             u(buse(BufferUse::NoRecord)), u(buse(BufferUse::NotAhead)),
	             u(bmiss(BufferOutcome::None)), u(bmiss(BufferOutcome::Busy)), u(bmiss(BufferOutcome::Short)),
	             u(bmiss(BufferOutcome::Null)), u(bmiss(BufferOutcome::Clamped)), u(bmiss(BufferOutcome::NoBuffer)),
	             u(buse(BufferUse::Words)), u(buse(BufferUse::Written)), u(buse(BufferUse::Stale)),
	             u(buse(BufferUse::StaleRanges)), u(TakeCount(s.buffer_batches)),
	             u(TakeCount(s.buffer_busy)), u(TakeCount(s.buffer_lookups)), u(TakeCount(s.buffer_found)));
	c.buffer_miss = {};
	c.buffer_use  = {};
}

} // namespace Libs::Graphics::StageF
