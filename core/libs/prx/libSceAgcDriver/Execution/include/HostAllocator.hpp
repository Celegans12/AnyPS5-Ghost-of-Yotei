#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTALLOCATOR_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_HOSTALLOCATOR_HPP

#include <cstdint>

namespace AgcDriver {

bool HostAllocationsCounted() noexcept;
std::uint64_t HostAllocationsOnThread() noexcept;

}

#endif
