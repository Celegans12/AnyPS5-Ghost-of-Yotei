#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_SUBMITPOLICY_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_SUBMITPOLICY_HPP

#include <cstdint>

namespace AgcDriver::Core {

struct SubmitState {
    std::uint64_t now = 0;
    std::uint64_t oldestLabel = 0;
    std::uint64_t heldSince = 0;
    std::uint64_t gpuLoad = 0;
    std::uint32_t work = 0;
    std::uint32_t queued = 0;
    bool waiter = false;
    bool interrupt = false;
    bool flip = false;
    bool gpuIdle = false;
};

struct SubmitLimits { std::uint64_t deadline = 1000000; std::uint32_t groupWork = 64; std::uint32_t workCap = 0; std::uint64_t idleLoad = 0; };

enum class SubmitCause : std::uint8_t { Held, Flip, Interrupt, Waiter, Empty, Work, Deadline, Cap, Idle, Count };

[[nodiscard]] SubmitCause AtSubmissionEnd(const SubmitState& state, const SubmitLimits& limits);
[[nodiscard]] SubmitCause AfterOp(const SubmitState& state, const SubmitLimits& limits);
[[nodiscard]] const char* SubmitCauseName(SubmitCause cause);

}

#endif
