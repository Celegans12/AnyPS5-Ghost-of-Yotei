#include "prx/libSceAgcDriver/Core/include/SubmitPolicy.hpp"

#include <cstdint>

namespace AgcDriver::Core {

namespace {

bool reached(std::uint64_t now, std::uint64_t since, std::uint64_t deadline) {
    if (since == 0) return false;
    const std::uint64_t elapsed = now > since ? now - since : 0;
    return elapsed >= deadline;
}

bool capped(const SubmitState& state, const SubmitLimits& limits) {
    return limits.workCap != 0 && state.work >= limits.workCap;
}

}

SubmitCause AtSubmissionEnd(const SubmitState& state, const SubmitLimits& limits) {
    if (state.flip) return SubmitCause::Flip;
    if (state.interrupt) return SubmitCause::Interrupt;
    if (state.waiter) return SubmitCause::Waiter;
    if (state.queued == 0) return SubmitCause::Empty;
    if (capped(state, limits)) return SubmitCause::Cap;
    if (state.work >= limits.groupWork) return SubmitCause::Work;
    if (reached(state.now, state.oldestLabel, limits.deadline) || reached(state.now, state.heldSince, limits.deadline)) return SubmitCause::Deadline;
    return SubmitCause::Held;
}

SubmitCause AfterOp(const SubmitState& state, const SubmitLimits& limits) {
    if (state.heldSince != 0) return AtSubmissionEnd(state, limits);
    if (capped(state, limits)) return SubmitCause::Cap;
    if (reached(state.now, state.oldestLabel, limits.deadline)) return SubmitCause::Deadline;
    return SubmitCause::Held;
}

const char* SubmitCauseName(SubmitCause cause) {
    switch (cause) {
        case SubmitCause::Held: return "held";
        case SubmitCause::Flip: return "flip";
        case SubmitCause::Interrupt: return "interrupt";
        case SubmitCause::Waiter: return "waiter";
        case SubmitCause::Empty: return "empty";
        case SubmitCause::Work: return "work";
        case SubmitCause::Deadline: return "deadline";
        case SubmitCause::Cap: return "cap";
        case SubmitCause::Count: break;
    }
    return "unknown";
}

}
