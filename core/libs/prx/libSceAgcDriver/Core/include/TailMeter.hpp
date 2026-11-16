#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_TAILMETER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_CORE_INCLUDE_TAILMETER_HPP

#include "prx/libSceAgcDriver/Core/include/Op.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace AgcDriver::Core {

enum class TailCause : std::uint8_t { None, PlanChange, PassChange, BufferMiss, BufferRefresh, ImageMiss, InputMiss, Deferred, Legacy, Count };

class TailMeter {
public:
    explicit TailMeter(std::uint64_t thresholdNanoseconds = 10000);
    ~TailMeter();
    TailMeter(const TailMeter&) = delete;
    TailMeter& operator=(const TailMeter&) = delete;
    void Count(OpKind kind, TailCause cause);
    void Time(OpKind kind, TailCause cause, std::uint64_t nanoseconds, std::uint32_t weight);
    struct Row { std::uint64_t ops = 0, timed = 0, nanoseconds = 0, over = 0, overNanoseconds = 0; };
    [[nodiscard]] Row Counted(OpKind kind, TailCause cause) const;
    [[nodiscard]] std::string Report(std::uint64_t frames) const;
    void Reset();

private:
    struct State;
    std::unique_ptr<State> state;
};

[[nodiscard]] const char* TailCauseName(TailCause cause);

}

#endif
