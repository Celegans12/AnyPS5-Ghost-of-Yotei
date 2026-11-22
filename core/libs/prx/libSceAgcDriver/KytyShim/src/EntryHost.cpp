#include "prx/libSceAgcDriver/KytyShim/src/EntryApi.hpp"
#include "prx/libSceAgcDriver/KytyShim/src/EntryPure.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/PadStatusOverlay.hpp"
#include "prx/libSceAgcDriver/Execution/include/PerformanceTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Presentation.hpp"
#include "prx/libSceAgcDriver/Execution/include/ThreadPriority.hpp"

#include <cstdio>
#include <stdexcept>

namespace {

void Note(const char* what) {
    std::fprintf(stderr, "[kyty] %s is not available in the Kyty GPU driver\n", what);
    std::fflush(stderr);
}

}

namespace AgcDriver {

std::size_t DisplayBufferSize(const DisplayBuffer& buffer) {
    return KytyShim::Entry::DisplayBufferBytes(buffer);
}

namespace GuestMemory {

void CheckRange(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable) {
    static_cast<void>(writable);
    KytyShim::Entry::CheckGuestRange(pointer, bytes, alignment);
}

}

}

extern "C" std::size_t AgcDriverDisplayBufferSize_nid_postfix(const AgcDriver::DisplayBuffer& buffer) {
    return AgcDriver::DisplayBufferSize(buffer);
}

extern "C" void AgcDriverCheckGuestMemory_nid_postfix(const void* pointer, std::size_t bytes, std::size_t alignment, bool writable) {
    AgcDriver::GuestMemory::CheckRange(pointer, bytes, alignment, writable);
}

extern "C" void AgcDriverRaiseWorkerThreadPriority_nid_postfix(const char* role) {
    AgcDriver::RaiseWorkerThreadPriority(role);
}

extern "C" void AgcDriverUseGcnShaders_nid_postfix() {
    throw std::runtime_error("PS4 (GNM) titles are not supported by the Kyty GPU driver");
}

extern "C" void AgcDriverTogglePerformanceOverlay_nid_postfix() {
    Note("the performance overlay");
}

extern "C" void AgcDriverRequestScreenshot_nid_postfix() {
    Note("the screenshot key (use the vkcount layer captures)");
}

extern "C" void AgcDriverPadStatusPoll_nid_postfix(const AgcDriver::PadStatusProvider* provider) {
    static_cast<void>(provider);
}

extern "C" void AgcDriverPadStatusShow_nid_postfix() {
    Note("the controller status overlay");
}

extern "C" bool AgcDriverTraceRecording_nid_postfix() {
    return false;
}

extern "C" AgcDriver::TraceThread* AgcDriverTraceThread_nid_postfix() {
    return nullptr;
}
