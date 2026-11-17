#include "prx/libc/include/GuestArena.hpp"
#include "prx/libkernel/include/Api.hpp"
#include "SceTypes.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#if !defined(ANYPS5_KYTY)
#error "guest_host_write_filter_tests checks the host write filter of the kyty backend"
#endif

namespace {

constexpr int SigUsr1 = 30;
constexpr int GuestSigBlock = 1;
constexpr std::uintptr_t Page = 0x1000;
constexpr std::uintptr_t Gib = std::uintptr_t{1} << 30;
constexpr std::uintptr_t Base = 0x0000000400000000ull + GuestArena::GuestAddressOffset;
constexpr std::uintptr_t TableEnd = (std::uintptr_t{1} << 44) + GuestArena::GuestAddressOffset;
constexpr int StressRaises = 20000;

void Check(bool value, int line) {
    if (value) return;
    std::fprintf(stderr, "host write filter check failed at line %d\n", line);
    std::fflush(stderr);
    std::_Exit(1);
}
#define Require(value) Check((value), __LINE__)

std::atomic<int> prepared{0};
std::atomic<std::uintptr_t> preparedAddress{0};
std::atomic<std::size_t> preparedBytes{0};
std::atomic<int> filtered{0};
std::atomic<std::uintptr_t> filteredAddress{0};
std::atomic<std::size_t> filteredBytes{0};
std::atomic<bool> filterAnswer{false};
std::atomic<int> reentries{0};
thread_local bool holdingMappedRanges = false;

bool Prepare(std::uintptr_t address, std::size_t bytes) {
    prepared.fetch_add(1);
    preparedAddress.store(address);
    preparedBytes.store(bytes);
    if (holdingMappedRanges) reentries.fetch_add(1);
    return true;
}

bool Filter(std::uintptr_t address, std::size_t bytes) {
    filtered.fetch_add(1);
    filteredAddress.store(address);
    filteredBytes.store(bytes);
    return filterAnswer.load();
}

bool Write(std::uintptr_t address, std::size_t bytes) {
    const int before = prepared.load();
    {
        const GuestArena::HostWrite write(reinterpret_cast<void*>(address), bytes);
        Require(write.Open());
    }
    const bool reached = prepared.load() != before;
    if (reached) Require(preparedAddress.load() == address && preparedBytes.load() == bytes);
    return reached;
}

void Note(std::uintptr_t address, std::size_t bytes, bool writable) {
    GuestArena::GuestArenaNoteHostProtection_nid_no_patch(address, bytes, writable);
}

bool Protected(std::uintptr_t address, std::size_t bytes) {
    return GuestArena::GuestArenaMayBeHostProtected_nid_no_patch(address, bytes);
}

enum class Mode : int { Hold, Spin };

std::atomic<int> mode{static_cast<int>(Mode::Spin)};
std::atomic<bool> stop{false};
std::atomic<bool> holdReady{false};
std::atomic<std::uint64_t> spinIterations{0};
std::atomic<bool> handled{false};
std::atomic<int> deliveries{0};
std::atomic<int> maskFailures{0};
std::atomic<bool> noteOwnMask{false};
std::atomic<std::uintptr_t> maskAddress{0};

void APS5_VABI Handler(int, void*) {
    GuestSignalSet old{};
    if (noteOwnMask.load()) Note(reinterpret_cast<std::uintptr_t>(&old), sizeof(old), false);
    maskAddress.store(reinterpret_cast<std::uintptr_t>(&old));
    if (_sigprocmask_nid_postfix(GuestSigBlock, nullptr, &old) != 0) maskFailures.fetch_add(1);
    handled.store(true);
    deliveries.fetch_add(1);
}

void* APS5_VABI MapperThread(void*) {
    while (!stop.load()) {
        holdingMappedRanges = true;
        if (static_cast<Mode>(mode.load()) == Mode::Hold) {
            holdReady.store(true);
            while (!handled.load() && !stop.load()) std::this_thread::yield();
            holdReady.store(false);
        } else {
            volatile std::uint64_t spin = 0;
            for (std::uint64_t i = 0; i < 2000; ++i) spin = spin + i;
            spinIterations.fetch_add(1);
        }
        holdingMappedRanges = false;
        volatile std::uint64_t spin = 0;
        for (std::uint64_t i = 0; i < 200; ++i) spin = spin + i;
    }
    return nullptr;
}

void RaiseAndWait(Pthread thread) {
    const int expected = deliveries.load() + 1;
    Require(sceKernelRaiseException(thread, SigUsr1) == 0);
    while (deliveries.load() < expected) std::this_thread::yield();
}

void RaiseWhileHolding(Pthread thread) {
    holdReady.store(false);
    handled.store(false);
    mode.store(static_cast<int>(Mode::Hold));
    while (!holdReady.load()) std::this_thread::yield();
    RaiseAndWait(thread);
    mode.store(static_cast<int>(Mode::Spin));
    const auto seen = spinIterations.load();
    while (spinIterations.load() == seen) std::this_thread::yield();
}

void RequireMapperSignals(Pthread thread, bool watched) {
    reentries.store(0);
    RaiseWhileHolding(thread);
    Require(reentries.load() == (watched ? 1 : 0));
    if (watched) return;
    prepared.store(0);
    for (int raise = 0; raise < StressRaises; ++raise) RaiseAndWait(thread);
    Require(reentries.load() == 0);
    Require(prepared.load() == 0);
    Require(maskFailures.load() == 0);
}

void RequireUnfilteredGuard() {
    prepared.store(0);
    Require(Write(Base + 40, 100));
    Require(Write(Base + 7 * Page, 8));
    Require(!Write(Base, 0));
    Require(prepared.load() == 2);
    Require(Protected(Base, 1));
}

void RequireWatchFilter() {
    GuestArena::GuestArenaSetHostWriteFilter_nid_no_patch(&Filter);
    filtered.store(0);
    prepared.store(0);
    filterAnswer.store(false);
    Require(!Write(Base + 40, 100));
    Require(filtered.load() == 1 && filteredAddress.load() == Base + 40 && filteredBytes.load() == 100);
    Require(!Protected(Base + 40, 100));
    filterAnswer.store(true);
    Require(Write(Base + 3 * Page + 5, 9));
    Require(filteredAddress.load() == Base + 3 * Page + 5 && filteredBytes.load() == 9);
    Require(Protected(Base, 1));
    filterAnswer.store(false);
    Require(prepared.load() == 1);
}

void RequireProtectionTable() {
    const int filterCalls = filtered.load();
    filterAnswer.store(true);
    const std::uintptr_t protectedPage = Base + 16 * Page;
    Note(protectedPage, Page, false);
    Require(Write(protectedPage, 8));
    Require(Write(protectedPage + Page - 1, 1));
    Require(!Write(protectedPage + Page, 8));
    Require(!Write(protectedPage - 8, 8));
    Require(Write(protectedPage - 8, 9));
    Require(Write(protectedPage + Page - 1, 2));
    Require(Write(Base, 64 * Page));
    Require(!Write(Base + 17 * Page, 63 * Page));
    Require(filtered.load() == filterCalls);
    Note(protectedPage, Page, true);
    Require(!Write(protectedPage, 8));
    Require(!Protected(protectedPage, Page));

    const std::uintptr_t boundary = Base + 3 * Gib;
    const std::uintptr_t first = boundary - 3 * Page + 100;
    const std::uintptr_t last = boundary + 70 * Page - 100;
    Note(first, last - first, false);
    for (std::uintptr_t page = boundary - 3 * Page; page < boundary + 70 * Page; page += Page) Require(Protected(page, 1));
    Require(!Protected(boundary - 4 * Page, Page));
    Require(!Protected(boundary + 70 * Page, Page));
    Require(Protected(boundary - 4 * Page, Page + 1));
    Require(Protected(Base, 4 * Gib));
    Note(boundary - 3 * Page, 73 * Page, true);
    Require(!Protected(Base, 4 * Gib));
    Note(boundary + 64 * Page, Page, true);
    Require(!Protected(boundary - 4 * Page, 80 * Page));

    Note(Base + 9 * Gib + 5 * Page, 1, false);
    Require(!Protected(Base + 9 * Gib, 5 * Page));
    Require(Protected(Base + 9 * Gib, 5 * Page + 1));
    Require(Protected(Base, 10 * Gib));
    Note(Base + 9 * Gib, Gib, true);
    Require(!Protected(Base, 10 * Gib));

    Require(!Protected(TableEnd, Page));
    Require(!Protected(TableEnd - Page, 2 * Page));
}

void RequireProtectionBeyondTable() {
    Note(TableEnd + 2 * Gib, Page, false);
    Require(Protected(TableEnd, Page));
    Require(Protected(TableEnd - Page, 2 * Page));
    Require(!Protected(Base, Page));
}

}

int main() {
    std::atomic<bool> done{false};
    std::thread watchdog([&done] {
        for (int i = 0; i < 120000 && !done.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (done.load()) return;
        std::fprintf(stderr, "host write filter test froze (deliveries %d, reentries %d)\n", deliveries.load(), reentries.load());
        std::fflush(stderr);
        std::_Exit(2);
    });
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&Prepare);
    RequireUnfilteredGuard();
    RequireWatchFilter();

    Require(sceKernelInstallExceptionHandler(SigUsr1, reinterpret_cast<void*>(&Handler)) == 0);
    Pthread mapper = nullptr;
    Require(scePthreadCreate(&mapper, nullptr, MapperThread, nullptr, "map memory") == 0);

    filterAnswer.store(true);
    RequireMapperSignals(mapper, true);
    Require(filteredAddress.load() == maskAddress.load() && filteredBytes.load() == sizeof(GuestSignalSet));
    filterAnswer.store(false);
    RequireMapperSignals(mapper, false);

    RequireProtectionTable();
    filterAnswer.store(true);
    RequireMapperSignals(mapper, false);
    noteOwnMask.store(true);
    RequireMapperSignals(mapper, true);
    noteOwnMask.store(false);
    Note(maskAddress.load(), sizeof(GuestSignalSet), true);
    RequireMapperSignals(mapper, false);

    stop.store(true);
    Require(scePthreadJoin(mapper, nullptr) == 0);
    Require(sceKernelRemoveExceptionHandler(SigUsr1) == 0);
    RequireProtectionBeyondTable();
    GuestArena::GuestArenaSetHostWriteFilter_nid_no_patch(nullptr);
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
    std::printf("host write filter: %d guest signals delivered, mask copied out at %#llx\n", deliveries.load(), static_cast<unsigned long long>(maskAddress.load()));
    done.store(true);
    watchdog.join();
    return 0;
}
