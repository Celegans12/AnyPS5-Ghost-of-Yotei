#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHLATER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DISPATCHLATER_HPP

#include <cstdint>

namespace AgcDriver::DriverDetail {

struct DispatchLaterCounts {
    std::uint64_t handed = 0;
    std::uint64_t kept = 0;
    std::uint64_t keptKernels = 0;
    std::uint64_t queued = 0;
    std::uint64_t inlined = 0;
    std::uint64_t verified = 0;
    std::uint64_t differing = 0;
    std::uint64_t unverified = 0;
    std::uint64_t finisherNs = 0;
};

bool DispatchLater();
bool VerifyDispatchLater();
DispatchLaterCounts DispatchLaterCounted();

}

#endif
