#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_COARSECLOCK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_COARSECLOCK_HPP

#include <cstdint>

namespace AgcDriver::Core {

class CoarseClock {
public:
    using Read = std::uint64_t (*)(void* context);
    CoarseClock(Read read, void* context, std::uint32_t period);
    [[nodiscard]] bool Tick();
    void Refresh();
    [[nodiscard]] std::uint64_t Now() const;
    [[nodiscard]] std::uint64_t Scaled(std::uint64_t nanoseconds) const;
    [[nodiscard]] std::uint32_t Period() const;

private:
    Read read;
    void* context;
    std::uint32_t period;
    std::uint32_t left;
    std::uint64_t now;
};

}

#endif
