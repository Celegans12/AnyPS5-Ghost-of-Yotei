#include "prx/libc/include/Api.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::size_t Page = 4096;
constexpr std::size_t Pages = 16;

void Check(bool value, int line) {
    if (value) return;
    std::fprintf(stderr, "read trap check failed at line %d\n", line);
    std::fflush(stderr);
    std::_Exit(1);
}
#define Require(value) Check((value), __LINE__)

struct Gpu {
    std::mutex mutex;
    std::condition_variable done;
    bool finished = true;
    std::atomic<int> resolutions{0};
    std::atomic<std::uintptr_t> lastBegin{0};
    std::atomic<std::uintptr_t> lastEnd{0};
};

Gpu gpu;
thread_local bool bypassThread = false;
std::byte* guest = nullptr;
std::byte* alias = nullptr;

void Resolve(std::uintptr_t begin, std::uintptr_t end) {
    gpu.resolutions.fetch_add(1, std::memory_order_relaxed);
    gpu.lastBegin.store(begin, std::memory_order_relaxed);
    gpu.lastEnd.store(end, std::memory_order_relaxed);
    std::unique_lock lock(gpu.mutex);
    gpu.done.wait(lock, [] { return gpu.finished; });
}

bool Bypass() {
    return bypassThread;
}

void StartWork() {
    const std::lock_guard lock(gpu.mutex);
    gpu.finished = false;
}

void FinishWork() {
    {
        const std::lock_guard lock(gpu.mutex);
        gpu.finished = true;
    }
    gpu.done.notify_all();
}

std::uint32_t& GuestWord(std::size_t pageIndex, std::size_t word) {
    return *reinterpret_cast<std::uint32_t*>(guest + pageIndex * Page + word * sizeof(std::uint32_t));
}

std::uint32_t& GpuWord(std::size_t pageIndex, std::size_t word) {
    return *reinterpret_cast<std::uint32_t*>(alias + pageIndex * Page + word * sizeof(std::uint32_t));
}

std::uint32_t ReadWord(std::size_t pageIndex, std::size_t word) {
    return reinterpret_cast<volatile std::uint32_t*>(guest + pageIndex * Page)[word];
}

std::uintptr_t Address(std::size_t pageIndex) {
    return reinterpret_cast<std::uintptr_t>(guest + pageIndex * Page);
}

GuestWriteWatch::ReadTrapCounts Counts() {
    GuestWriteWatch::ReadTrapCounts counts{};
    GuestWriteWatch::GuestWriteWatchReadTrapCounts_nid_postfix(&counts);
    return counts;
}

bool Trap(std::size_t pageIndex, std::size_t pages = 1, std::size_t offset = 0, std::size_t bytes = 0) {
    const auto begin = Address(pageIndex) + offset;
    return GuestWriteWatch::GuestWriteWatchTrapReads_nid_postfix(begin, bytes != 0 ? bytes : pages * Page - offset);
}

void Release(std::size_t pageIndex, std::size_t pages = 1, std::size_t offset = 0, std::size_t bytes = 0) {
    const auto begin = Address(pageIndex) + offset;
    GuestWriteWatch::GuestWriteWatchReleaseReads_nid_postfix(begin, bytes != 0 ? bytes : pages * Page - offset);
}

double Milliseconds(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since).count();
}

std::thread LandLater(std::size_t pageIndex, std::size_t word, std::uint32_t value, std::chrono::microseconds delay) {
    return std::thread([=] {
        std::this_thread::sleep_for(delay);
        GpuWord(pageIndex, word) = value;
        FinishWork();
    });
}

void RequireTrappedReadWaits() {
    GuestWord(0, 0) = 1;
    const auto before = Counts();
    StartWork();
    Require(Trap(0));
    auto landing = LandLater(0, 0, 2, std::chrono::milliseconds(40));
    const auto start = std::chrono::steady_clock::now();
    const auto value = ReadWord(0, 0);
    const auto waited = Milliseconds(start);
    landing.join();
    Require(value == 2);
    Require(waited >= 30.0);
    const auto after = Counts();
    Require(after.faults == before.faults + 1);
    Require(gpu.lastBegin.load() == Address(0) && gpu.lastEnd.load() == Address(0) + Page);
    Release(0);
    Require(Counts().holds == 0);
    Require(ReadWord(0, 0) == 2);
}

void RequireReleaseWithoutFault() {
    GuestWord(1, 3) = 10;
    StartWork();
    Require(Trap(1, 1, 12, 4));
    auto landing = LandLater(1, 3, 11, std::chrono::milliseconds(5));
    landing.join();
    const auto before = Counts();
    Release(1, 1, 12, 4);
    Require(ReadWord(1, 3) == 11);
    Require(Counts().faults == before.faults);
}

void RequireWriteWaits() {
    GuestWord(2, 0) = 1;
    GuestWord(2, 1) = 1;
    StartWork();
    Require(Trap(2, 1, 0, 4));
    auto landing = LandLater(2, 0, 5, std::chrono::milliseconds(20));
    reinterpret_cast<volatile std::uint32_t*>(guest + 2 * Page)[1] = 7;
    landing.join();
    Require(ReadWord(2, 0) == 5);
    Require(ReadWord(2, 1) == 7);
    Release(2, 1, 0, 4);
}

void RequireBypass() {
    GuestWord(3, 0) = 20;
    StartWork();
    Require(Trap(3));
    const auto before = Counts();
    std::uint32_t seen = 0;
    double waited = 0;
    std::thread([&] {
        bypassThread = true;
        const auto start = std::chrono::steady_clock::now();
        seen = ReadWord(3, 0);
        waited = Milliseconds(start);
    }).join();
    Require(seen == 20);
    Require(waited < 20.0);
    Require(Counts().bypassOpens == before.bypassOpens + 1);
    Require(Counts().faults == before.faults);
    FinishWork();
    Release(3);
}

void RequireCountedHolds() {
    GuestWord(4, 0) = 30;
    StartWork();
    Require(Trap(4));
    Require(Trap(4));
    Release(4);
    auto landing = LandLater(4, 0, 31, std::chrono::milliseconds(20));
    Require(ReadWord(4, 0) == 31);
    landing.join();
    StartWork();
    Require(Trap(4));
    auto second = LandLater(4, 0, 32, std::chrono::milliseconds(20));
    Require(ReadWord(4, 0) == 32);
    second.join();
    Release(4);
    Release(4);
    Require(Counts().holds == 0);
    StartWork();
    Require(Trap(4));
    Release(4);
    const auto before = Counts();
    Require(ReadWord(4, 0) == 32);
    Require(Counts().faults == before.faults);
    FinishWork();
}

void RequireLogicalProtection() {
    StartWork();
    Require(Trap(5, 2));
    std::uintptr_t runEnd = 0;
    Require(GuestWriteWatch::GuestWriteWatchProtection_nid_postfix(Address(5), Address(8), &runEnd) == (PROT_READ | PROT_WRITE));
    Require(runEnd == Address(7));
    Require(GuestWriteWatch::GuestWriteWatchProtection_nid_postfix(Address(7), Address(8), &runEnd) == -1);
    Require(GuestWriteWatch::GuestWriteWatchReadTrapsHeld_nid_postfix());
    FinishWork();
    Release(5, 2);
    Require(GuestWriteWatch::GuestWriteWatchProtection_nid_postfix(Address(5), Address(8), &runEnd) == (PROT_READ | PROT_WRITE));
    const auto asleep = Counts();
    Require(ReadWord(5, 0) == ReadWord(5, 0));
    Require(Counts().faults == asleep.faults && Counts().dormantOpens == asleep.dormantOpens + 1);
    for (int sweep = 0; sweep < 4; ++sweep) GuestWriteWatch::GuestWriteWatchSweepReadTraps_nid_postfix();
    Require(GuestWriteWatch::GuestWriteWatchProtection_nid_postfix(Address(5), Address(8), &runEnd) == -1);
    Require(!GuestWriteWatch::GuestWriteWatchReadTrapsHeld_nid_postfix());
}

void RequireThreads() {
    constexpr std::size_t Readers = 8;
    std::mt19937 random(7);
    for (std::uint32_t round = 0; round < 200; ++round) {
        for (std::size_t index = 0; index < Readers; ++index) GuestWord(8 + index, index) = 0xdead0000u + round;
        StartWork();
        const bool whole = round % 2 == 0;
        if (whole) {
            Require(Trap(8, Readers));
        } else {
            for (std::size_t index = 0; index < Readers; ++index) Require(Trap(8 + index, 1, index * 4, 4));
        }
        std::atomic<int> wrong{0};
        std::vector<std::thread> readers;
        for (std::size_t index = 0; index < Readers; ++index) {
            readers.emplace_back([&, index] {
                for (int repeat = 0; repeat < 50; ++repeat) {
                    if (ReadWord(8 + index, index) != 0xbeef0000u + round) wrong.fetch_add(1);
                }
            });
        }
        const auto delay = std::chrono::microseconds(random() % 2000);
        std::this_thread::sleep_for(delay);
        for (std::size_t index = 0; index < Readers; ++index) GpuWord(8 + index, index) = 0xbeef0000u + round;
        FinishWork();
        for (auto& reader : readers) reader.join();
        Require(wrong.load() == 0);
        if (whole) {
            Release(8, Readers);
        } else {
            for (std::size_t index = 0; index < Readers; ++index) Release(8 + index, 1, index * 4, 4);
        }
    }
    Require(Counts().holds == 0);
}

void RequireSystemCalls() {
    int pipeFds[2];
    Require(pipe(pipeFds) == 0);
    const char message[] = "kernel write into a trapped page";

    GuestWord(6, 0) = 40;
    StartWork();
    Require(Trap(6));
    Require(write(pipeFds[1], message, sizeof(message)) == static_cast<ssize_t>(sizeof(message)));
    errno = 0;
    Require(read(pipeFds[0], guest + 6 * Page + 64, sizeof(message)) == -1 && errno == EFAULT);
    auto landing = LandLater(6, 0, 41, std::chrono::milliseconds(20));
    {
        const GuestArena::HostWrite destination(guest + 6 * Page + 64, sizeof(message));
        Require(destination.Open());
        Require(read(pipeFds[0], guest + 6 * Page + 64, sizeof(message)) == static_cast<ssize_t>(sizeof(message)));
    }
    landing.join();
    Require(std::memcmp(guest + 6 * Page + 64, message, sizeof(message)) == 0);
    Require(ReadWord(6, 0) == 41);
    Release(6);

    StartWork();
    Require(Trap(6));
    errno = 0;
    Require(write(pipeFds[1], guest + 6 * Page + 64, sizeof(message)) == -1 && errno == EFAULT);
    auto second = LandLater(6, 0, 42, std::chrono::milliseconds(20));
    GuestArena::GuestArenaPrepareHostRead_nid_no_patch(guest + 6 * Page + 64, sizeof(message));
    Require(write(pipeFds[1], guest + 6 * Page + 64, sizeof(message)) == static_cast<ssize_t>(sizeof(message)));
    second.join();
    char echoed[sizeof(message)]{};
    Require(read(pipeFds[0], echoed, sizeof(echoed)) == static_cast<ssize_t>(sizeof(echoed)));
    Require(std::memcmp(echoed, message, sizeof(message)) == 0);
    Release(6);

    std::uint32_t copied = 0;
    StartWork();
    Require(Trap(6));
    auto third = LandLater(6, 0, 43, std::chrono::milliseconds(20));
    Require(GuestCopyIn_nid_no_patch(&copied, guest + 6 * Page, sizeof(copied)));
    third.join();
    Require(copied == 43);
    Release(6);

    StartWork();
    Require(Trap(6));
    auto fourth = LandLater(6, 0, 44, std::chrono::milliseconds(20));
    Require(GuestCopyInUnsynced_nid_no_patch(&copied, guest + 6 * Page, sizeof(copied)));
    fourth.join();
    Require(copied == 44);
    Release(6);

    const std::uint32_t stored = 45;
    StartWork();
    Require(Trap(6, 1, 0, 4));
    auto fifth = LandLater(6, 0, 46, std::chrono::milliseconds(20));
    Require(GuestCopyOut_nid_no_patch(guest + 6 * Page + 8, &stored, sizeof(stored)));
    fifth.join();
    Require(ReadWord(6, 2) == 45);
    Require(ReadWord(6, 0) == 46);
    Release(6, 1, 0, 4);

    GuestWord(6, 0) = 47;
    StartWork();
    Require(Trap(6));
    const auto protectFrom = Counts();
    Require(GuestWriteWatch::GuestWriteWatchProtect_nid_postfix(guest + 6 * Page, Page, PROT_READ) == 0);
    Require(ReadWord(6, 0) == 47);
    Require(Counts().faults == protectFrom.faults);
    Require(Counts().settleOpens == protectFrom.settleOpens + 1);
    std::uintptr_t protectedEnd = 0;
    Require(GuestWriteWatch::GuestWriteWatchProtection_nid_postfix(Address(6), Address(7), &protectedEnd) == PROT_READ);
    Require(GuestWriteWatch::GuestWriteWatchProtect_nid_postfix(guest + 6 * Page, Page, PROT_READ | PROT_WRITE) == 0);
    FinishWork();
    Release(6);

    StartWork();
    Require(Trap(6));
    auto sixth = LandLater(6, 0, 48, std::chrono::milliseconds(20));
    GuestWriteWatch::GuestWriteWatchSettleReads_nid_postfix(guest + 6 * Page, Page);
    Require(ReadWord(6, 0) == 48);
    sixth.join();
    Release(6);

    close(pipeFds[0]);
    close(pipeFds[1]);
    Require(Counts().holds == 0);
}

void RequireRefusals() {
    Require(GuestWriteWatch::GuestWriteWatchProtect_nid_postfix(guest + 15 * Page, Page, PROT_NONE) == 0);
    Require(!Trap(15));
    Require(GuestWriteWatch::GuestWriteWatchProtect_nid_postfix(guest + 15 * Page, Page, PROT_READ | PROT_WRITE) == 0);
    std::vector<std::byte> outside(2 * Page);
    const auto unaligned = (reinterpret_cast<std::uintptr_t>(outside.data()) + Page - 1) & ~(Page - 1);
    Require(!GuestWriteWatch::GuestWriteWatchTrapReads_nid_postfix(unaligned, 64));
}

void RequireWriteTraps() {
    GuestWord(13, 0) = 70;
    GuestWord(13, 1) = 71;
    StartWork();
    Require(GuestWriteWatch::GuestWriteWatchTrapWrites_nid_postfix(Address(13), 8));
    const auto before = Counts();
    Require(ReadWord(13, 0) == 70);
    Require(ReadWord(13, 1) == 71);
    Require(Counts().faults == before.faults);
    std::uintptr_t runEnd = 0;
    Require(GuestWriteWatch::GuestWriteWatchProtection_nid_postfix(Address(13), Address(14), &runEnd) == (PROT_READ | PROT_WRITE));
    auto landing = LandLater(13, 0, 72, std::chrono::milliseconds(20));
    const auto start = std::chrono::steady_clock::now();
    reinterpret_cast<volatile std::uint32_t*>(guest + 13 * Page)[1] = 73;
    Require(Milliseconds(start) >= 10.0);
    landing.join();
    Require(Counts().faults == before.faults + 1);
    Require(ReadWord(13, 0) == 72);
    Require(ReadWord(13, 1) == 73);
    GuestWriteWatch::GuestWriteWatchReleaseWrites_nid_postfix(Address(13), 8);

    StartWork();
    Require(GuestWriteWatch::GuestWriteWatchTrapWrites_nid_postfix(Address(13), 8));
    Require(Trap(13, 1, 64, 64));
    const auto both = Counts();
    std::uint32_t seen = 0;
    auto reader = std::thread([&] { seen = ReadWord(13, 0); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    GpuWord(13, 0) = 74;
    FinishWork();
    reader.join();
    Require(seen == 74);
    Require(Counts().faults == both.faults + 1);
    Release(13, 1, 64, 64);
    StartWork();
    Require(GuestWriteWatch::GuestWriteWatchTrapWrites_nid_postfix(Address(13), 8));
    Require(Trap(13, 1, 64, 64));
    Release(13, 1, 64, 64);
    const auto readable = Counts();
    Require(ReadWord(13, 0) == 74);
    Require(Counts().faults == readable.faults);
    auto last = LandLater(13, 0, 75, std::chrono::milliseconds(20));
    reinterpret_cast<volatile std::uint32_t*>(guest + 13 * Page)[3] = 76;
    last.join();
    Require(ReadWord(13, 0) == 75 && ReadWord(13, 3) == 76);
    GuestWriteWatch::GuestWriteWatchReleaseWrites_nid_postfix(Address(13), 8);
    GuestWriteWatch::GuestWriteWatchReleaseWrites_nid_postfix(Address(13), 8);
    Require(Counts().holds == 0);
}

void RequireDormantHolds() {
    GuestWord(12, 0) = 80;
    const auto start = Counts();
    StartWork();
    Require(Trap(12, 1));
    FinishWork();
    Release(12, 1);
    const auto before = Counts();
    Require(before.dormant == start.dormant + 1);
    StartWork();
    Require(Trap(12, 1));
    Require(Counts().revived == before.revived + 1);
    auto landing = LandLater(12, 0, 81, std::chrono::milliseconds(20));
    Require(ReadWord(12, 0) == 81);
    landing.join();
    Require(Counts().faults == before.faults + 1);
    Release(12, 1);
    StartWork();
    Require(Trap(12, 1));
    FinishWork();
    Release(12, 1);
    const auto opened = Counts();
    Require(ReadWord(12, 0) == 81);
    Require(Counts().faults == opened.faults && Counts().dormantOpens == opened.dormantOpens + 1);
    StartWork();
    Require(Trap(12, 1));
    Require(Counts().revived == opened.revived);
    auto again = LandLater(12, 0, 82, std::chrono::milliseconds(20));
    Require(ReadWord(12, 0) == 82);
    again.join();
    Release(12, 1);
    std::vector<std::byte> buffer(16);
    StartWork();
    Require(Trap(12, 1));
    FinishWork();
    Release(12, 1);
    Require(GuestCopyIn_nid_no_patch(buffer.data(), guest + 12 * Page, buffer.size()));
    StartWork();
    Require(Trap(12, 1));
    Require(GuestWriteWatch::GuestWriteWatchOpenIdleReads_nid_postfix(guest + 12 * Page, Page) < 0);
    FinishWork();
    Release(12, 1);
    const auto idle = Counts();
    Require(GuestWriteWatch::GuestWriteWatchOpenIdleReads_nid_postfix(guest + 12 * Page, Page) == 1);
    Require(GuestWriteWatch::GuestWriteWatchOpenIdleReads_nid_postfix(guest + 12 * Page, Page) == 0);
    Require(ReadWord(12, 0) == 82 && Counts().faults == idle.faults);
    for (int sweep = 0; sweep < 4; ++sweep) GuestWriteWatch::GuestWriteWatchSweepReadTraps_nid_postfix();
    Require(Counts().dormant == 0);
}

void RequireUnregister() {
    GuestWord(14, 0) = 60;
    StartWork();
    Require(Trap(14));
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(guest + 14 * Page, Page);
    const auto before = Counts();
    Require(ReadWord(14, 0) == 60);
    Require(Counts().faults == before.faults);
    FinishWork();
    Release(14);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(guest + 14 * Page, Page);
}

}

int main() {
    const int file = static_cast<int>(syscall(SYS_memfd_create, "read-trap", 0));
    Require(file >= 0);
    Require(ftruncate(file, static_cast<off_t>(Pages * Page)) == 0);
    void* guestView = mmap(nullptr, Pages * Page, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    void* gpuView = mmap(nullptr, Pages * Page, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    Require(guestView != MAP_FAILED && gpuView != MAP_FAILED);
    guest = static_cast<std::byte*>(guestView);
    alias = static_cast<std::byte*>(gpuView);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(guest, Pages * Page);
    Require(!GuestWriteWatch::GuestWriteWatchTrapReads_nid_postfix(Address(0), Page));
    Require(GuestWriteWatch::GuestWriteWatchSetReadTrapResolver_nid_postfix(&Resolve));
    GuestWriteWatch::GuestWriteWatchSetReadTrapBypass_nid_postfix(&Bypass);
    RequireTrappedReadWaits();
    RequireReleaseWithoutFault();
    RequireWriteWaits();
    RequireBypass();
    RequireCountedHolds();
    RequireLogicalProtection();
    RequireThreads();
    RequireSystemCalls();
    RequireRefusals();
    RequireWriteTraps();
    RequireDormantHolds();
    RequireUnregister();
    const auto counts = Counts();
    Require(counts.forcedOpens == 0);
    std::printf("read traps: %llu set, %llu faults waited, %llu resolutions, %llu opened under the bypass, %llu retries, %llu settles\n", static_cast<unsigned long long>(counts.trapped), static_cast<unsigned long long>(counts.faults), static_cast<unsigned long long>(counts.resolved), static_cast<unsigned long long>(counts.bypassOpens), static_cast<unsigned long long>(counts.retries), static_cast<unsigned long long>(counts.settles));
    return 0;
}
