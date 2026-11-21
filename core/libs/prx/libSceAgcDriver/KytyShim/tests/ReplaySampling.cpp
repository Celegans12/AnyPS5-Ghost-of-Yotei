#include "FrameReplaySampling.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

using AgcDriver::FrameReplaySampleTiming;
using AgcDriver::FrameReplaySamplesCoverInterval;

FrameReplaySampleTiming::TimePoint Tick(int milliseconds) {
    return FrameReplaySampleTiming::TimePoint{std::chrono::milliseconds{milliseconds}};
}

FrameReplaySampleTiming Timing(int start, int end) {
    return {Tick(start), Tick(end)};
}

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "kyty_shim_replay_sampling: %s\n", message);
    std::abort();
}

}

int main() {
    Require(FrameReplaySamplesCoverInterval(Timing(1, 3), Timing(11, 13), Tick(0), Tick(10), Tick(20)),
            "on-time adjacent samples did not qualify");
    Require(!FrameReplaySamplesCoverInterval(Timing(1, 11), Timing(12, 14), Tick(0), Tick(10), Tick(20)),
            "a sample that started on time but crossed the next flip qualified");
    Require(!FrameReplaySamplesCoverInterval(Timing(1, 3), Timing(9, 14), Tick(0), Tick(10), Tick(20)),
            "a sample that began before its own flip qualified");
    Require(FrameReplaySamplesCoverInterval(Timing(1, 3), Timing(11, 13), Tick(0), Tick(10), Tick(13)),
            "the final sample finished by the pass deadline but did not qualify");
    Require(!FrameReplaySamplesCoverInterval(Timing(1, 3), Timing(11, 14), Tick(0), Tick(10), Tick(13)),
            "the final sample crossed the pass deadline and qualified");

    const std::array flips{Tick(0), Tick(10), Tick(20), Tick(30), Tick(40)};
    const std::array samples{Timing(1, 21), Timing(22, 23), Timing(24, 25), Timing(31, 32)};
    Require(!FrameReplaySamplesCoverInterval(samples[0], samples[1], flips[0], flips[1], flips[2]),
            "the first backlogged interval qualified");
    Require(!FrameReplaySamplesCoverInterval(samples[1], samples[2], flips[1], flips[2], flips[3]),
            "the second backlogged interval qualified");
    Require(FrameReplaySamplesCoverInterval(samples[2], samples[3], flips[2], flips[3], flips[4]),
            "the first on-time interval after the backlog did not qualify");
    return 0;
}
