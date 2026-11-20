#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/include/Api.hpp"
#include "SceTypes.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#if !defined(__linux__) || !defined(ANYPS5_KYTY)
#error "kyty_shim_signal_safety checks the Linux signal handlers the kyty backend runs under"
#endif

namespace {

constexpr int GuestSigsegv = 11;
constexpr int GuestSigusr1 = 30;
constexpr int Marker = 4321;
constexpr int Clobbered = 8765;
constexpr std::uintptr_t Gib = std::uintptr_t{1} << 30;
constexpr std::uintptr_t FreshBucket = 0x0000000400000000ull + GuestArena::GuestAddressOffset + 40 * Gib;
constexpr int RemoteRaises = 100;
constexpr std::size_t HostTextBytes = std::size_t{1} << 20;
constexpr std::size_t GuestTextBytes = 16;
constexpr auto DeliveryLimit = std::chrono::seconds(5);

void Check(bool value, int line, const char* what) {
    if (value) return;
    std::fprintf(stderr, "kyty_shim_signal_safety: FAILED at line %d: %s\n", line, what);
    std::fflush(stderr);
    std::_Exit(1);
}
#define Require(value, what) Check((value), __LINE__, (what))

std::size_t page = 0;
unsigned char* faultPage = nullptr;
thread_local bool allocating = false;
thread_local bool faultInsideAllocation = false;
std::atomic<int> nestedAllocations{0};
std::atomic<int> faults{0};
std::atomic<bool> noteInFault{false};
std::atomic<int> deliveries{0};
std::atomic<int> legacyCalls{0};
std::atomic<int> lostErrno{0};
std::atomic<int> wrongLengths{0};
std::atomic<bool> spinInHost{false};
std::atomic<bool> stopSpinning{false};
std::atomic<bool> spinning{false};
std::vector<char> hostText;
std::vector<char> guestText;

void Touch(unsigned char* at) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile unsigned char*>(at) = 1;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

void Unprotect() {
    if (mprotect(faultPage, page, PROT_READ | PROT_WRITE) != 0) std::_Exit(3);
}

void Arm() {
    Require(mprotect(faultPage, page, PROT_READ) == 0, "cannot protect the fault page");
}

bool NotedProtected(std::uintptr_t address) {
    return GuestArena::GuestArenaMayBeHostProtected_nid_no_patch(address, 1);
}

}

void* operator new[](std::size_t bytes) {
    if (allocating) nestedAllocations.fetch_add(1, std::memory_order_relaxed);
    const bool outer = !allocating;
    allocating = true;
    if (outer && faultInsideAllocation) {
        faultInsideAllocation = false;
        Touch(faultPage);
    }
    void* memory = std::malloc(bytes != 0 ? bytes : 1);
    if (outer) allocating = false;
    if (memory == nullptr) throw std::bad_alloc();
    return memory;
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

void APS5_VABI FaultHandler(int, void*) {
    if (noteInFault.load()) GuestArena::GuestArenaNoteHostProtection_nid_no_patch(FreshBucket, page, false);
    errno = Clobbered;
    Unprotect();
    faults.fetch_add(1);
}

void APS5_VABI RaiseHandler(int, void*) {
    errno = Clobbered;
    deliveries.fetch_add(1);
}

void APS5_VABI LegacySegv(int) {
    errno = Clobbered;
    Unprotect();
    legacyCalls.fetch_add(1);
}

void RequireNoteWithoutAllocation() {
    GuestArena::GuestArenaNoteHostProtection_nid_no_patch(FreshBucket - 2 * Gib, page, false);
    Require(!NotedProtected(FreshBucket), "the fresh bucket was noted before the test");
    faults.store(0);
    nestedAllocations.store(0);
    noteInFault.store(true);
    Arm();
    faultInsideAllocation = true;
    void* memory = ::operator new[](64);
    ::operator delete[](memory);
    noteInFault.store(false);
    Require(faults.load() == 1, "the store inside operator new[] did not fault once");
    Require(nestedAllocations.load() == 0, "a host protection noted by a fault handler allocated while the faulting thread was inside the allocator");
    Require(NotedProtected(FreshBucket), "the protection noted inside the fault handler was lost");
    Require(!NotedProtected(FreshBucket + page), "the protection noted inside the fault handler covers the next page");
    GuestArena::GuestArenaNoteHostProtection_nid_no_patch(FreshBucket, page, true);
    GuestArena::GuestArenaNoteHostProtection_nid_no_patch(FreshBucket - 2 * Gib, page, true);
    Require(!NotedProtected(FreshBucket), "the protection noted inside the fault handler was not cleared");
}

void RequireFaultKeepsErrno() {
    faults.store(0);
    Arm();
    errno = Marker;
    Touch(faultPage);
    const int seen = errno;
    Require(faults.load() == 1, "the guest fault handler did not run once");
    Require(seen == Marker, "a guest fault handler changed errno of the code that faulted");
}

void* APS5_VABI RaiseSelf(void* result) {
    errno = Marker;
    const int before = deliveries.load();
    if (sceKernelRaiseException(scePthreadSelf(), GuestSigusr1) != 0) return nullptr;
    const auto raisedAt = std::chrono::steady_clock::now();
    while (deliveries.load() == before) {
        if (std::chrono::steady_clock::now() - raisedAt > DeliveryLimit) return nullptr;
    }
    *static_cast<int*>(result) = errno;
    return result;
}

void* APS5_VABI Spin(void*) {
    errno = Marker;
    spinning.store(true);
    while (!stopSpinning.load(std::memory_order_relaxed)) {
        const bool host = spinInHost.load(std::memory_order_relaxed);
        const auto& text = host ? hostText : guestText;
        if (std::strlen(text.data()) != text.size() - 1) wrongLengths.fetch_add(1);
        if (errno != Marker) {
            lostErrno.fetch_add(1);
            errno = Marker;
        }
    }
    return nullptr;
}

void RaiseAndWait(Pthread thread) {
    const int expected = deliveries.load() + 1;
    Require(sceKernelRaiseException(thread, GuestSigusr1) == 0, "sceKernelRaiseException failed");
    const auto raisedAt = std::chrono::steady_clock::now();
    while (deliveries.load() < expected) Require(std::chrono::steady_clock::now() - raisedAt < DeliveryLimit, "a raised exception was not delivered");
}

void RequireRaiseKeepsErrno() {
    Require(sceKernelInstallExceptionHandler(GuestSigusr1, reinterpret_cast<void*>(&RaiseHandler)) == 0, "cannot install the raise handler");
    int seen = 0;
    Pthread self = nullptr;
    Require(scePthreadCreate(&self, nullptr, RaiseSelf, &seen, "raise self") == 0, "cannot create the self-raising thread");
    void* result = nullptr;
    Require(scePthreadJoin(self, &result) == 0, "cannot join the self-raising thread");
    Require(result == &seen, "the exception a thread raised on itself was not delivered");
    Require(seen == Marker, "the handler of an exception a thread raised on itself changed its errno");

    hostText.assign(HostTextBytes, 'a');
    hostText.back() = '\0';
    guestText.assign(GuestTextBytes, 'a');
    guestText.back() = '\0';
    Pthread spinner = nullptr;
    Require(scePthreadCreate(&spinner, nullptr, Spin, nullptr, "keep errno") == 0, "cannot create the spinning thread");
    while (!spinning.load()) {
    }
    for (int raise = 0; raise < RemoteRaises; ++raise) RaiseAndWait(spinner);
    spinInHost.store(true);
    for (int raise = 0; raise < RemoteRaises; ++raise) RaiseAndWait(spinner);
    stopSpinning.store(true);
    Require(scePthreadJoin(spinner, nullptr) == 0, "cannot join the spinning thread");
    Require(wrongLengths.load() == 0, "a delivery at a return into guest code changed the result of the host call");
    Require(lostErrno.load() == 0, "the handler of a raised exception changed errno of the interrupted thread");
    Require(sceKernelRemoveExceptionHandler(GuestSigusr1) == 0, "cannot remove the raise handler");
}

void RequireLegacyHandlerKeepsErrno() {
    GuestSigaction action{};
    action.handler = reinterpret_cast<std::uintptr_t>(&LegacySegv);
    Require(sigaction_nid_postfix(GuestSigsegv, &action, nullptr) == 0, "cannot install the guest SIGSEGV action");
    legacyCalls.store(0);
    Arm();
    errno = Marker;
    Touch(faultPage);
    const int seen = errno;
    action.handler = 0;
    Require(sigaction_nid_postfix(GuestSigsegv, &action, nullptr) == 0, "cannot reset the guest SIGSEGV action");
    Require(legacyCalls.load() == 1, "the guest SIGSEGV action did not run once");
    Require(seen == Marker, "a guest SIGSEGV action changed errno of the code that faulted");
}

}

int main(int argc, char** argv) {
    const char* only = argc > 1 ? argv[1] : nullptr;
    const auto selected = [only](const char* name) { return only == nullptr || std::strcmp(only, name) == 0; };
    Require(only == nullptr || selected("allocation") || selected("fault-errno") || selected("raise-errno") || selected("action-errno"), "unknown case");
    page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    void* memory = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(memory != MAP_FAILED, "cannot map the fault page");
    faultPage = static_cast<unsigned char*>(memory);
    Touch(faultPage);

    Require(sceKernelInstallExceptionHandler(GuestSigsegv, reinterpret_cast<void*>(&FaultHandler)) == 0, "cannot install the fault handler");
    if (selected("allocation")) RequireNoteWithoutAllocation();
    if (selected("fault-errno")) RequireFaultKeepsErrno();
    Require(sceKernelRemoveExceptionHandler(GuestSigsegv) == 0, "cannot remove the fault handler");
    if (selected("raise-errno")) RequireRaiseKeepsErrno();
    if (selected("action-errno")) RequireLegacyHandlerKeepsErrno();
    std::printf("kyty_shim_signal_safety: ok (%d deliveries)\n", deliveries.load());
    return 0;
}
