#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include "common/assert.h"
#include "kernel/eventQueue.h"
#include "libs/errno.h"

#include "SceTypes.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libkernel/Equeue/Equeue.hpp"
#include "prx/libkernel/KernelErrors.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace {

namespace Kyty = Libs::LibKernel::EventQueue;

static_assert(EQUEUE_OK == OK);
static_assert(SCE_KERNEL_ERROR_EBADF == Libs::LibKernel::KERNEL_ERROR_EBADF);
static_assert(SCE_KERNEL_ERROR_ENOENT == Libs::LibKernel::KERNEL_ERROR_ENOENT);
static_assert(EVFILT_VIDEO_OUT == Kyty::KERNEL_EVFILT_VIDEO_OUT);
static_assert((static_cast<std::uint64_t>(KytyShim::InterruptRoute) & 0xffffffffull) == 0);

struct Bridge {
    Kyty::KernelFilter filter;
};

Kyty::KernelEvent ToKyty(const KernelEvent& event) {
    return {event.ident, event.filter, event.flags, event.fflags, event.data, event.udata};
}

KernelEvent ToOurs(const Kyty::KernelEvent& event) {
    KernelEvent ours;
    ours.ident = event.ident;
    ours.filter = event.filter;
    ours.flags = event.flags;
    ours.fflags = event.fflags;
    ours.data = event.data;
    ours.udata = event.udata;
    return ours;
}

Kyty::KernelEqueueEvent Mirror(const KernelEqueueEvent& ours) {
    Kyty::KernelEqueueEvent mirror;
    mirror.triggered = ours.triggered;
    mirror.deadline_ns = ours.deadlineNs;
    mirror.interval_ns = ours.intervalNs;
    mirror.event = ToKyty(ours.event);
    mirror.filter = static_cast<const Bridge*>(ours.filter.data)->filter;
    for (const auto& pending : ours.pendingEvents) mirror.pending_events.push_back(ToKyty(pending));
    return mirror;
}

void Reflect(const Kyty::KernelEqueueEvent& mirror, KernelEqueueEvent& ours) {
    ours.triggered = mirror.triggered;
    ours.deadlineNs = mirror.deadline_ns;
    ours.intervalNs = mirror.interval_ns;
    ours.event = ToOurs(mirror.event);
    ours.pendingEvents.clear();
    for (const auto& pending : mirror.pending_events) ours.pendingEvents.push_back(ToOurs(pending));
}

void TriggerThroughKyty(KernelEqueueEvent* event, void* data) {
    auto mirror = Mirror(*event);
    mirror.filter.trigger_func(&mirror, data);
    Reflect(mirror, *event);
}

void ResetThroughKyty(KernelEqueueEvent* event) {
    auto mirror = Mirror(*event);
    mirror.filter.reset_func(&mirror);
    Reflect(mirror, *event);
}

void DeleteThroughKyty(KernelEqueue eq, KernelEqueueEvent* event) {
    auto mirror = Mirror(*event);
    mirror.filter.delete_event_func(eq, &mirror);
}

KernelEqueueEvent ToOurs(const Kyty::KernelEqueueEvent& event) {
    KernelEqueueEvent ours;
    ours.triggered = event.triggered;
    ours.deadlineNs = event.deadline_ns;
    ours.intervalNs = event.interval_ns;
    ours.event = ToOurs(event.event);
    for (const auto& pending : event.pending_events) ours.pendingEvents.push_back(ToOurs(pending));
    const auto& filter = event.filter;
    if (filter.data == nullptr && !filter.owner && filter.trigger_func == nullptr && filter.reset_func == nullptr && filter.delete_event_func == nullptr) return ours;
    auto bridge = std::make_shared<Bridge>(Bridge{filter});
    ours.filter.data = bridge.get();
    ours.filter.triggerFunc = filter.trigger_func != nullptr ? &TriggerThroughKyty : nullptr;
    ours.filter.resetFunc = filter.reset_func != nullptr ? &ResetThroughKyty : nullptr;
    ours.filter.deleteEventFunc = filter.delete_event_func != nullptr ? &DeleteThroughKyty : nullptr;
    ours.filter.owner = std::move(bridge);
    return ours;
}

}

namespace Libs::LibKernel::EventQueue {

class KernelEqueuePrivate {
public:
    explicit KernelEqueuePrivate(::KernelEqueueRef queue) : m_queue(std::move(queue)) {}

private:
    ::KernelEqueueRef m_queue;
};

KernelEqueueRef KernelPinEqueue(KernelEqueue eq) {
    auto queue = EqueuePin_nid_postfix(eq);
    if (!queue) return {};
    return std::make_shared<KernelEqueuePrivate>(std::move(queue));
}

int KYTY_SYSV_ABI KernelAddEvent(KernelEqueue eq, const KernelEqueueEvent& event) {
    if (eq == KytyShim::InterruptRoute) EXIT("KytyShim: an event was added to the interrupt route of the GPU, which only forwards interrupts to libSceAgcDriver's Eq\n");
    return EqueueAddEvent_nid_postfix(eq, ToOurs(event));
}

int KYTY_SYSV_ABI KernelTriggerEvent(KernelEqueue eq, uintptr_t ident, int16_t filter, void* trigger_data) {
    if (eq != KytyShim::InterruptRoute) return EqueueTriggerEvent_nid_postfix(eq, ident, filter, trigger_data);
    if (filter != KERNEL_EVFILT_GRAPHICS) EXIT("KytyShim: filter %d triggered on the GPU interrupt route, which takes graphics interrupts only\n", static_cast<int>(filter));
    AgcDriverDeliverEopInterrupt(static_cast<std::uint32_t>(ident), static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(trigger_data)));
    return OK;
}

int KYTY_SYSV_ABI KernelDeleteEvent(KernelEqueue eq, uintptr_t ident, int16_t filter) {
    if (eq == KytyShim::InterruptRoute) EXIT("KytyShim: event %llu deleted from the GPU interrupt route, which is never deleted\n", static_cast<unsigned long long>(ident));
    return EqueueDeleteEvent_nid_postfix(eq, ident, filter);
}

}
