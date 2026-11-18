#include "prx/libSceVideoOut/include/VblankSchedule.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <ratio>
#include <stdexcept>
#include <string>

namespace {

using VideoOutVblank::VblankSchedule;

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "vblank_schedule: %s\n", message);
    std::exit(1);
}

template <typename TAction>
std::string Refusal(TAction action) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

void DefaultKeepsTheNtscCadence() {
    using Frame = std::chrono::duration<std::int64_t, std::ratio<1001, 120000>>;
    VblankSchedule schedule(std::nullopt);
    for (std::int64_t frame = 1; frame <= 2'000'000; ++frame) {
        const auto expected = std::chrono::duration_cast<std::chrono::nanoseconds>(Frame(frame)).count();
        Require(schedule.NextNanos() == expected, "without a rate the vblanks are not at 119.88 Hz as before");
        const auto due = schedule.Advance();
        Require(due.high, "without a rate a 119.88 Hz output missed a vblank");
        Require(due.standard == (frame % 2 == 0), "without a rate a 59.94 Hz output did not get every second vblank");
    }
}

struct Counts {
    std::uint64_t standard = 0;
    std::uint64_t high = 0;
    bool standardWithinHigh = true;
};

Counts CountSecond(std::uint32_t hertz) {
    VblankSchedule schedule(hertz);
    Counts counts;
    std::int64_t previous = 0;
    while (schedule.NextNanos() <= 1'000'000'000) {
        const auto at = schedule.NextNanos();
        Require(at > previous, "the vblanks are not strictly later than the one before");
        previous = at;
        const auto due = schedule.Advance();
        Require(due.standard || due.high, "a vblank was due on no output");
        counts.standard += due.standard ? 1 : 0;
        counts.high += due.high ? 1 : 0;
        if (due.standard && !due.high) counts.standardWithinHigh = false;
    }
    return counts;
}

void ConfiguredRateDrivesBothOutputs() {
    const auto at240 = CountSecond(240);
    Require(at240.standard == 240, "ANYPS5_VBLANK_HZ=240 did not give 59.94 Hz outputs 240 vblanks a second");
    Require(at240.high == 360, "ANYPS5_VBLANK_HZ=240 did not cap 119.88 Hz outputs at 360 vblanks a second");
    const auto at120 = CountSecond(120);
    Require(at120.standard == 120 && at120.high == 240 && at120.standardWithinHigh, "ANYPS5_VBLANK_HZ=120 did not give 120 and 240 vblanks a second on the same beat");
    const auto at30 = CountSecond(30);
    Require(at30.standard == 30 && at30.high == 60 && at30.standardWithinHigh, "ANYPS5_VBLANK_HZ=30 did not give 30 and 60 vblanks a second");
    const auto at360 = CountSecond(360);
    Require(at360.standard == 360 && at360.high == 360 && at360.standardWithinHigh, "ANYPS5_VBLANK_HZ=360 did not give both outputs 360 vblanks a second");
    VblankSchedule schedule(240);
    Require(schedule.NextNanos() == 1'000'000'000 / 360, "the first vblank at 240 Hz is not the first 360 Hz one");
}

void ParsesTheRate() {
    using VideoOutVblank::ParseVblankHz;
    Require(!ParseVblankHz(nullptr) && !ParseVblankHz(""), "an unset or empty ANYPS5_VBLANK_HZ was not left alone");
    Require(ParseVblankHz("30") == 30u && ParseVblankHz("240") == 240u && ParseVblankHz("360") == 360u, "a rate from 30 to 360 was not taken");
    for (const char* value : {"29", "361", "0", "-1", "240x", " 240", "60.5", "abc", "99999999999"}) {
        Require(Refusal([&] { ParseVblankHz(value); }).find("ANYPS5_VBLANK_HZ") != std::string::npos, "a rate outside 30 to 360 was not refused with the variable's name");
    }
}

void KytyFrequencyFeedsTheRate() {
    using VideoOutVblank::VblankHzFromEnvironment;
    Require(!VblankHzFromEnvironment(nullptr, nullptr) && !VblankHzFromEnvironment("", ""), "without either variable a rate was set");
    Require(VblankHzFromEnvironment(nullptr, "240") == 240u, "KYTY_VBLANK_FREQUENCY was not used when ANYPS5_VBLANK_HZ is unset");
    Require(VblankHzFromEnvironment("", "90") == 90u, "KYTY_VBLANK_FREQUENCY was not used when ANYPS5_VBLANK_HZ is empty");
    Require(VblankHzFromEnvironment("120", "240") == 120u, "ANYPS5_VBLANK_HZ did not win over KYTY_VBLANK_FREQUENCY");
    Require(VblankHzFromEnvironment(nullptr, "1000") == 360u && VblankHzFromEnvironment(nullptr, "0") == 30u, "KYTY_VBLANK_FREQUENCY was not clamped to 30 to 360 as Kyty does");
    Require(Refusal([] { VblankHzFromEnvironment(nullptr, "fast"); }).find("KYTY_VBLANK_FREQUENCY") != std::string::npos, "a KYTY_VBLANK_FREQUENCY that is not a number was not refused with its name");
    Require(Refusal([] { VblankHzFromEnvironment("1000", "240"); }).find("ANYPS5_VBLANK_HZ") != std::string::npos, "an ANYPS5_VBLANK_HZ out of range was not refused when KYTY_VBLANK_FREQUENCY is set");
}

void ParsesUnpacedFlips() {
    using VideoOutVblank::ParseUnpacedFlips;
    Require(ParseUnpacedFlips("1"), "APS5_UNPACED_FLIPS=1 leaves the flips paced");
    for (const char* text : {static_cast<const char*>(nullptr), "", "0"}) Require(!ParseUnpacedFlips(text), "an unset, empty or 0 APS5_UNPACED_FLIPS unpaces the flips");
    for (const char* text : {"2", "true", "yes", "on", "01", " 1", "1 ", "-1", "1.0"}) {
        const auto message = Refusal([&] { ParseUnpacedFlips(text); });
        Require(message.find("APS5_UNPACED_FLIPS") != std::string::npos && message.find(std::string("'") + text + "'") != std::string::npos, "an APS5_UNPACED_FLIPS other than 1, 0 or empty was not refused with the variable's name and value");
    }
}

}

int main() {
    DefaultKeepsTheNtscCadence();
    ConfiguredRateDrivesBothOutputs();
    ParsesTheRate();
    KytyFrequencyFeedsTheRate();
    ParsesUnpacedFlips();
    std::puts("vblank schedule tests passed");
    return 0;
}
