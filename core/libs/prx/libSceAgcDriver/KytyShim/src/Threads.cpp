#include "common/assert.h"
#include "kernel/guestPause.h"
#include "kernel/pthread.h"

#include "prx/libkernel/Time/include/Time.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <memory>

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

struct GuestPause::Impl {};

GuestPause::GuestPause() : m_impl(std::make_unique<Impl>()) {
    EXIT("KytyShim: stopping the guest threads while pipelines compile is not supported on Linux (GuestPause::Supported() is false)\n");
}

GuestPause::~GuestPause() = default;

bool GuestPause::Supported() noexcept {
    return false;
}

bool GuestPause::GuestStarted() {
    return false;
}

void GuestPause::Poll() {
    EXIT("KytyShim: GuestPause::Poll called, but no GuestPause can exist on Linux\n");
}

uint32_t GuestPause::Suspended() const noexcept {
    return 0;
}

}
