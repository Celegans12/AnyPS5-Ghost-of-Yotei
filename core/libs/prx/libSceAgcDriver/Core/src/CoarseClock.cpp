#include "prx/libSceAgcDriver/Core/include/CoarseClock.hpp"
#include <algorithm>
#include <cstdint>

namespace AgcDriver::Core {

CoarseClock::CoarseClock(Read readFunction, void* contextPointer, std::uint32_t periodTicks)
    : read(readFunction), context(contextPointer), period(periodTicks), left(0), now(0) {}

bool CoarseClock::Tick() {
    if (left != 0) {
        if (period != 0) --left;
        return false;
    }
    Refresh();
    left = period != 0 ? period - 1 : 1;
    return true;
}

void CoarseClock::Refresh() {
    if (read != nullptr) now = read(context);
}

std::uint64_t CoarseClock::Now() const {
    return now;
}

std::uint64_t CoarseClock::Scaled(std::uint64_t nanoseconds) const {
    return nanoseconds * static_cast<std::uint64_t>(std::max<std::uint32_t>(period, 1u));
}

std::uint32_t CoarseClock::Period() const {
    return period;
}

}
