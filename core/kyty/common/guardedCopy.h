#ifndef KYTY_COMMON_GUARDED_COPY_H_
#define KYTY_COMMON_GUARDED_COPY_H_

#include <cstdint>

namespace Common {

uint64_t GuardedCopy(void* destination, const void* source, uint64_t size) noexcept;

#if defined(_WIN32)
bool ResumeGuardedCopy(void* context) noexcept;
#endif

}

#endif
