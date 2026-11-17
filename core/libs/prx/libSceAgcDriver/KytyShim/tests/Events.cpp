#include "prx/libSceAgcDriver/KytyShim/src/Runtime.hpp"

#include "kernel/eventQueue.h"
#include "libs/errno.h"

#include "SceTypes.hpp"
#include "prx/libSceAgcDriver/Eq/include/Event.hpp"
#include "prx/libSceAgcDriver/Eq/include/Query.hpp"
#include "prx/libkernel/KernelErrors.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {

namespace Kyty = Libs::LibKernel::EventQueue;

void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_events: %s\n", what);
    std::abort();
}

void* Data(std::uintptr_t value) {
    return reinterpret_cast<void*>(value);
}

int Poll(KernelEqueue eq, KernelEvent* events, int capacity) {
    int count = 0;
    const KernelUseconds poll = 0;
    const int result = sceKernelWaitEqueue(eq, events, capacity, &count, &poll);
    if (result == SCE_KERNEL_ERROR_ETIMEDOUT) return 0;
    Require(result == 0, "sceKernelWaitEqueue failed");
    return count;
}

int Route(int event, std::uintptr_t context) {
    return Kyty::KernelTriggerEvent(KytyShim::InterruptRoute, static_cast<std::uintptr_t>(event), Kyty::KERNEL_EVFILT_GRAPHICS, Data(context));
}

void VerifyInterruptRoute() {
    KernelEqueue graphics = 0;
    KernelEqueue compute = 0;
    Require(sceKernelCreateEqueue(&graphics, "kyty-graphics") == 0, "cannot create the graphics queue");
    Require(sceKernelCreateEqueue(&compute, "kyty-compute") == 0, "cannot create the compute queue");
    int graphicsTag = 0;
    int computeTag = 0;
    const int computeEvent = KytyShim::FirstComputeInterruptEvent + 1;
    Require(sceAgcDriverAddEqEvent(graphics, KytyShim::GraphicsInterruptEvent, &graphicsTag) == 0, "sceAgcDriverAddEqEvent refused the graphics event");
    Require(sceAgcDriverAddEqEvent(compute, computeEvent, &computeTag) == 0, "sceAgcDriverAddEqEvent refused the compute event");

    KernelEvent events[4]{};
    Require(Route(KytyShim::GraphicsInterruptEvent, 0x1234567) == OK, "the route refused a graphics interrupt");
    Require(Poll(graphics, events, 4) == 1, "the graphics interrupt did not reach the guest's queue");
    Require(events[0].filter == Kyty::KERNEL_EVFILT_GRAPHICS && events[0].ident == 0 && events[0].udata == &graphicsTag, "the graphics event is not the guest's registration");
    Require(sceAgcDriverGetEqContextId(&events[0]) == 0x1234567, "the graphics event lost Kyty's interrupt context");
    Require(Poll(graphics, events, 4) == 0, "the graphics event stayed triggered after it was read");
    Require(Poll(compute, events, 4) == 0, "a graphics interrupt reached the compute queue");

    Require(Route(computeEvent, 9) == OK, "the route refused a compute interrupt");
    Require(Poll(compute, events, 4) == 1, "the compute interrupt did not reach the guest's queue");
    Require(events[0].ident == static_cast<std::uintptr_t>(computeEvent) && events[0].udata == &computeTag && sceAgcDriverGetEqContextId(&events[0]) == 9, "the compute event is wrong");
    Require(Poll(graphics, events, 4) == 0, "a compute interrupt reached the graphics queue");

    Require(Route(KytyShim::FirstComputeInterruptEvent + KytyShim::ComputeInterruptEvents - 1, 3) == OK, "the route refused an interrupt nobody waits for");
    Require(Poll(graphics, events, 4) == 0 && Poll(compute, events, 4) == 0, "an interrupt nobody waits for was delivered");

    Require(sceAgcDriverDeleteEqEvent(graphics, KytyShim::GraphicsInterruptEvent) == 0, "sceAgcDriverDeleteEqEvent failed");
    Require(Route(KytyShim::GraphicsInterruptEvent, 4) == OK, "the route refused an interrupt after the guest deleted its event");
    Require(Poll(graphics, events, 4) == 0, "a deleted event was triggered");
    Require(sceAgcDriverDeleteEqEvent(compute, computeEvent) == 0, "sceAgcDriverDeleteEqEvent failed for compute");
    Require(sceKernelDeleteEqueue(graphics) == 0 && sceKernelDeleteEqueue(compute) == 0, "cannot delete the queues");
}

struct Calls {
    int triggers = 0;
    int resets = 0;
    int deletes = 0;
};

void TriggerLikeKyty(Kyty::KernelEqueueEvent* event, void* data) {
    static_cast<Calls*>(event->filter.data)->triggers++;
    const auto value = reinterpret_cast<std::intptr_t>(data);
    if (event->triggered) {
        auto queued = event->event;
        queued.fflags = 1;
        queued.data = value;
        event->pending_events.push_back(queued);
        return;
    }
    event->event.fflags++;
    event->event.data = value;
    event->triggered = true;
}

void ResetLikeKyty(Kyty::KernelEqueueEvent* event) {
    static_cast<Calls*>(event->filter.data)->resets++;
    event->triggered = false;
    event->event.fflags = 0;
    event->event.data = 0;
}

void DeleteLikeKyty(Kyty::KernelEqueue, Kyty::KernelEqueueEvent* event) {
    static_cast<Calls*>(event->filter.data)->deletes++;
}

void VerifyKytyFilters() {
    KernelEqueue eq = 0;
    Require(sceKernelCreateEqueue(&eq, "kyty-filters") == 0, "cannot create the queue");
    Calls calls;
    auto owner = std::make_shared<int>(7);
    const std::weak_ptr<int> watched = owner;
    Kyty::KernelEqueueEvent event;
    event.event.ident = 5;
    event.event.filter = Kyty::KERNEL_EVFILT_GRAPHICS;
    event.event.udata = &calls;
    event.filter.data = &calls;
    event.filter.owner = std::move(owner);
    event.filter.trigger_func = TriggerLikeKyty;
    event.filter.reset_func = ResetLikeKyty;
    event.filter.delete_event_func = DeleteLikeKyty;
    Require(Kyty::KernelAddEvent(eq, event) == OK, "KernelAddEvent failed");
    event.filter.owner.reset();
    Require(!watched.expired(), "the queue did not keep the owner of Kyty's filter");

    Require(Kyty::KernelTriggerEvent(eq, 5, Kyty::KERNEL_EVFILT_GRAPHICS, Data(11)) == OK, "first trigger failed");
    Require(Kyty::KernelTriggerEvent(eq, 5, Kyty::KERNEL_EVFILT_GRAPHICS, Data(12)) == OK, "second trigger failed");
    KernelEvent events[4]{};
    Require(Poll(eq, events, 4) == 2, "the pending event of Kyty's trigger function was not delivered");
    Require(events[0].data == 11 && events[0].fflags == 1 && events[0].udata == &calls, "the first event is wrong");
    Require(events[1].data == 12 && events[1].fflags == 1 && events[1].udata == &calls, "the pending event is wrong");
    Require(calls.triggers == 2 && calls.resets == 2, "Kyty's trigger and reset functions were not called once per event");
    Require(Poll(eq, events, 4) == 0, "Kyty's reset function did not clear the event");

    Require(Kyty::KernelTriggerEvent(eq, 6, Kyty::KERNEL_EVFILT_GRAPHICS, nullptr) == Libs::LibKernel::KERNEL_ERROR_ENOENT, "an unknown event did not give ENOENT");
    Require(Kyty::KernelPinEqueue(eq) != nullptr, "KernelPinEqueue lost a live queue");
    Require(Kyty::KernelPinEqueue(Kyty::KERNEL_EQUEUE_INVALID) == nullptr && Kyty::KernelPinEqueue(KytyShim::InterruptRoute) == nullptr, "KernelPinEqueue pinned something that is not a queue");
    Require(Kyty::KernelDeleteEvent(eq, 5, Kyty::KERNEL_EVFILT_GRAPHICS) == OK, "KernelDeleteEvent failed");
    Require(calls.deletes == 1 && watched.expired(), "deleting the event did not call Kyty's delete function and release its owner");
    Require(Kyty::KernelDeleteEvent(eq, 5, Kyty::KERNEL_EVFILT_GRAPHICS) == Libs::LibKernel::KERNEL_ERROR_ENOENT, "a deleted event did not give ENOENT");
    Require(sceKernelDeleteEqueue(eq) == 0, "cannot delete the queue");
    Require(Kyty::KernelTriggerEvent(eq, 5, Kyty::KERNEL_EVFILT_GRAPHICS, nullptr) == Libs::LibKernel::KERNEL_ERROR_EBADF, "a deleted queue did not give EBADF");
    Require(Kyty::KernelAddEvent(eq, event) == Libs::LibKernel::KERNEL_ERROR_EBADF, "adding to a deleted queue did not give EBADF");
}

}

int main() {
    VerifyInterruptRoute();
    VerifyKytyFilters();
    std::puts("kyty_shim_events: ok");
    return 0;
}
