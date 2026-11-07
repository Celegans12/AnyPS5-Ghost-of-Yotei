#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELCOMPARE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PARALLELCOMPARE_HPP

#include <cstddef>
#include <cstdint>
#include <span>

namespace AgcDriver::Graphics {

struct CompareSpan {
    const void* first;
    const void* second;
    std::size_t bytes;
};

// A job the compare helpers run once per index, from any of them or from the calling thread.
using HelperJob = void (*)(void* context, std::size_t index);

// Runs job(context, index) for every index below count on the compare helpers and the calling
// thread, and returns once all have finished; a job that throws is rethrown here after the others.
// Returns false without running anything when count is below 2, the helpers are stopped, or another
// thread is using them, so the caller runs the indices itself.
bool RunOnCompareHelpers(std::size_t count, HelperJob job, void* context);

void CompareSpans(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal);
void CompareSpansFrom(std::span<const CompareSpan> spans, std::span<std::uint8_t> equal, std::size_t parallelBytes);
unsigned CompareHelpers();
void StopCompareHelpers();
unsigned CompareHelpersRunning();

}

#endif
