#ifndef KYTY_COMMON_TIMER_H_
#define KYTY_COMMON_TIMER_H_

#include <chrono>
#include <cstdint>

namespace Common {

inline uint64_t SteadyNs() {
	return static_cast<uint64_t>(
	    std::chrono::nanoseconds(std::chrono::steady_clock::now().time_since_epoch()).count());
}

}

#endif
