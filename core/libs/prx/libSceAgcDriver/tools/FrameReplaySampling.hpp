#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TOOLS_FRAMEREPLAYSAMPLING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TOOLS_FRAMEREPLAYSAMPLING_HPP

#include <chrono>

namespace AgcDriver {

struct FrameReplaySampleTiming {
    using TimePoint = std::chrono::steady_clock::time_point;

    TimePoint start;
    TimePoint end;
};

[[nodiscard]] inline bool FrameReplaySamplesCoverInterval(FrameReplaySampleTiming before, FrameReplaySampleTiming after,
                                                          FrameReplaySampleTiming::TimePoint firstFlip,
                                                          FrameReplaySampleTiming::TimePoint secondFlip,
                                                          FrameReplaySampleTiming::TimePoint deadline) {
    return firstFlip <= before.start && before.start <= before.end && before.end <= secondFlip &&
           secondFlip <= after.start && after.start <= after.end && after.end <= deadline;
}

} // namespace AgcDriver

#endif
