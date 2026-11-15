#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_MEMORY_MOVEDREAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_MEMORY_MOVEDREAD_HPP

#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace AgcDriver::DriverDetail {

struct MovedReadCounts {
    std::uint64_t guarded = 0;
    std::uint64_t checked = 0;
};

GuestMemory::Compare CopyMoved(std::uint64_t address, std::span<std::byte> out);
MovedReadCounts MovedReads();
std::string MovedReadsReport();

}

#endif
