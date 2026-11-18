#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::OcclusionCounters::Segments;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_occlusion_segments: %s\n", what);
    std::abort();
}

void SkippedDrawReadsVisible() {
    Segments segments(16);
    constexpr std::uint64_t address = 0x1000;
    (void)segments.OnDump(address);
    Require(segments.Counting(), "a begin dump opens a query");
    std::uint32_t slot = 0;
    Require(segments.AllocateSlot(slot), "a slot is free");
    segments.MarkUncounted();
    const auto end = segments.OnDump(address + sizeof(std::uint64_t));
    Require(end.uncounted, "a segment with a skipped draw is uncounted");
    Require(end.end_slot - end.first_slot == 1u, "the counted slot stays in the segment");
    Require(!segments.Counting(), "the end dump closes the query");
    const auto next = segments.OnDump(address + 0x100);
    Require(!next.uncounted, "the next segment starts counted");
}

void CountedSegmentStaysCounted() {
    Segments segments(16);
    (void)segments.OnDump(0x2000);
    std::uint32_t slot = 0;
    Require(segments.AllocateSlot(slot), "a slot is free");
    const auto end = segments.OnDump(0x2008);
    Require(!end.uncounted, "a segment without skipped draws is counted");
}

void SkippedDrawInNestedQueries() {
    Segments segments(16);
    (void)segments.OnDump(0x3000);
    (void)segments.OnDump(0x4000);
    segments.MarkUncounted();
    const auto inner = segments.OnDump(0x4008);
    Require(inner.uncounted, "the inner query sees the skipped draw");
    Require(segments.Counting(), "the outer query stays open");
    const auto outer = segments.OnDump(0x3008);
    Require(!outer.uncounted, "the segment after the inner query is counted");
}

}

int main() {
    SkippedDrawReadsVisible();
    CountedSegmentStaysCounted();
    SkippedDrawInNestedQueries();
    std::puts("kyty_shim_occlusion_segments: ok");
    return 0;
}
