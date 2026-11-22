#include "kernel/pthread.h"

#include "prx/libkernel/Time/include/Time.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>

namespace Libs::LibKernel {

uint64_t KYTY_SYSV_ABI KernelReadTsc() {
    return sceKernelReadTsc();
}

uint64_t KYTY_SYSV_ABI KernelGetTscFrequency() {
    return sceKernelGetTscFrequency();
}

uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
    return sceKernelGetProcessTime();
}

uint64_t KYTY_SYSV_ABI KernelGetProcessTimeCounter() {
    return sceKernelGetProcessTimeCounter();
}

}
