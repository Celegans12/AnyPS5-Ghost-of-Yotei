#include "prx/libc/include/Api.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <ucontext.h>
#include <unistd.h>
#include <vector>

#if !defined(ANYPS5_KYTY)
#error "guest_gpu_fault_tests checks the fault chain of the kyty backend"
#endif

namespace {

constexpr int GuestSigsegv = 11;
constexpr std::uint64_t ErrorWrite = 1u << 1;
constexpr std::uint64_t ErrorFetch = 1u << 4;

enum class Access { Read, Write, Execute };

std::size_t page = 0;
std::byte* owned = nullptr;
std::size_t ownedBytes = 0;
int ownedFile = -1;
std::byte* codePage = nullptr;

std::atomic<int> calls{0};
std::atomic<int> lastSignal{0};
std::atomic<std::uintptr_t> lastAddress{0};
std::atomic<int> lastAccess{-1};
std::atomic<bool> asyncHeld{true};
std::atomic<bool> faultsUnblocked{true};
std::atomic<bool> raiseInside{false};
std::atomic<bool> declineAll{false};
std::atomic<bool> inGpuHandler{false};
std::atomic<int> usr1Count{0};
std::atomic<int> usr1DuringGpu{0};
std::atomic<int> guestCalls{0};
std::atomic<int> prepared{0};
std::atomic<std::uintptr_t> preparedAddress{0};
std::atomic<std::size_t> preparedBytes{0};
std::atomic<bool> prepareOpens{false};

void Check(bool value, int line) {
    if (value) return;
    std::fprintf(stderr, "gpu fault check failed at line %d\n", line);
    std::fflush(stderr);
    std::_Exit(1);
}
#define Require(value) Check((value), __LINE__)

pid_t Tid() {
    return static_cast<pid_t>(syscall(SYS_gettid));
}

std::byte* Page(std::size_t index) {
    return owned + index * page;
}

bool Owned(std::uintptr_t address) {
    const auto begin = reinterpret_cast<std::uintptr_t>(owned);
    const auto code = reinterpret_cast<std::uintptr_t>(codePage);
    return (address >= begin && address < begin + ownedBytes) || (codePage != nullptr && address >= code && address < code + page);
}

Access Decode(void* context) {
    const auto error = static_cast<std::uint64_t>(static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR]);
    if ((error & ErrorFetch) != 0) return Access::Execute;
    if ((error & ErrorWrite) != 0) return Access::Write;
    return Access::Read;
}

bool GpuFault(int signal, siginfo_t* info, void* context) {
    inGpuHandler.store(true);
    sigset_t current;
    pthread_sigmask(SIG_BLOCK, nullptr, &current);
    if (sigismember(&current, SIGUSR1) != 1 || sigismember(&current, SIGALRM) != 1) asyncHeld.store(false);
    const int otherFault = signal == SIGSEGV ? SIGBUS : SIGSEGV;
    for (const int synchronous : {otherFault, SIGILL, SIGFPE, SIGTRAP, SIGABRT})
        if (sigismember(&current, synchronous) != 0) faultsUnblocked.store(false);
    errno = EINTR;
    if (raiseInside.load()) syscall(SYS_tgkill, getpid(), Tid(), SIGUSR1);
    const auto address = reinterpret_cast<std::uintptr_t>(info->si_addr);
    calls.fetch_add(1);
    lastSignal.store(signal);
    lastAddress.store(address);
    const auto access = Decode(context);
    lastAccess.store(static_cast<int>(access));
    bool handled = false;
    if (!declineAll.load() && Owned(address)) {
        const auto at = address & ~(static_cast<std::uintptr_t>(page) - 1);
        if (signal == SIGBUS) {
            handled = ftruncate(ownedFile, static_cast<off_t>(ownedBytes)) == 0;
        } else {
            const int protection = access == Access::Execute ? PROT_READ | PROT_EXEC : PROT_READ | PROT_WRITE;
            handled = mprotect(reinterpret_cast<void*>(at), page, protection) == 0;
        }
    }
    inGpuHandler.store(false);
    return handled;
}

void Usr1(int, siginfo_t*, void*) {
    usr1Count.fetch_add(1);
    if (inGpuHandler.load()) usr1DuringGpu.fetch_add(1);
}

void APS5_VABI CountGuestSegv(int) {
    guestCalls.fetch_add(1);
}

void APS5_VABI CountGuestException(int, void*) {
    guestCalls.fetch_add(1);
}

void SetGuestSegv(std::uintptr_t handler) {
    GuestSigaction action{};
    action.handler = handler;
    Require(sigaction_nid_postfix(GuestSigsegv, &action, nullptr) == 0);
}

void Reset() {
    calls.store(0);
    lastSignal.store(0);
    lastAddress.store(0);
    lastAccess.store(-1);
}

sigset_t Mask() {
    sigset_t mask;
    pthread_sigmask(SIG_BLOCK, nullptr, &mask);
    return mask;
}

bool SameMask(const sigset_t& first, const sigset_t& second) {
    for (int signal = 1; signal < NSIG; ++signal)
        if (sigismember(&first, signal) != sigismember(&second, signal)) return false;
    return true;
}

void StoreWord(std::byte* at, std::uint32_t value) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    *reinterpret_cast<volatile std::uint32_t*>(at) = value;
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

std::uint32_t LoadWord(const std::byte* at) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    const std::uint32_t value = *reinterpret_cast<const volatile std::uint32_t*>(at);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return value;
}

void RequireTrackingOff() {
    using namespace GuestWriteWatch;
    Require(!GuestWriteWatchAvailable_nid_postfix());
    GuestWriteWatchRegister_nid_postfix(Page(0), 4 * page);
    Require(!GuestWriteWatchCovers_nid_postfix(reinterpret_cast<std::uintptr_t>(Page(0)), page));
    Require(!GuestWriteWatchSetReadTrapResolver_nid_postfix(+[](std::uintptr_t, std::uintptr_t) {}));
    Require(GuestWriteWatchSetReadTrapResolver_nid_postfix(nullptr));
    Require(!GuestWriteWatchTrapReads_nid_postfix(reinterpret_cast<std::uintptr_t>(Page(0)), page));
    Require(!GuestWriteWatchTrapWrites_nid_postfix(reinterpret_cast<std::uintptr_t>(Page(1)), page));
    Require(!GuestWriteWatchReadTrapsHeld_nid_postfix());
    std::uintptr_t runEnd = 0;
    Require(GuestWriteWatchProtection_nid_postfix(reinterpret_cast<std::uintptr_t>(Page(0)), reinterpret_cast<std::uintptr_t>(Page(4)), &runEnd) == -1);
    ReadTrapCounts counts{};
    GuestWriteWatchReadTrapCounts_nid_postfix(&counts);
    Require(counts.trapped == 0 && counts.refused == 0 && counts.faults == 0);
    Require(!GuestWriteWatchUnregister_nid_postfix(Page(0), 4 * page));
}

void RequireWrite() {
    Reset();
    Require(mprotect(Page(0), page, PROT_READ) == 0);
    const auto before = Mask();
    errno = 4321;
    StoreWord(Page(0) + 12, 0x1234);
    Require(errno == 4321);
    Require(calls.load() == 1);
    Require(lastSignal.load() == SIGSEGV);
    Require(lastAddress.load() == reinterpret_cast<std::uintptr_t>(Page(0) + 12));
    Require(lastAccess.load() == static_cast<int>(Access::Write));
    Require(LoadWord(Page(0) + 12) == 0x1234);
    Require(SameMask(before, Mask()));
}

void RequireRead() {
    Reset();
    StoreWord(Page(1), 77);
    Require(mprotect(Page(1), page, PROT_NONE) == 0);
    errno = 99;
    Require(LoadWord(Page(1)) == 77);
    Require(errno == 99);
    Require(calls.load() == 1);
    Require(lastAccess.load() == static_cast<int>(Access::Read));
}

void RequireExecute() {
    Reset();
    void* memory = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(memory != MAP_FAILED);
    codePage = static_cast<std::byte*>(memory);
    auto* code = codePage;
    code[0] = std::byte{0xb8};
    const std::uint32_t value = 0x5a5a1234;
    std::memcpy(code + 1, &value, sizeof(value));
    code[5] = std::byte{0xc3};
    Require(mprotect(code, page, PROT_READ) == 0);
    const auto function = reinterpret_cast<std::uint32_t (*)()>(code);
    Require(function() == value);
    Require(calls.load() == 1);
    Require(lastAccess.load() == static_cast<int>(Access::Execute));
    Require(lastAddress.load() == reinterpret_cast<std::uintptr_t>(code));
    munmap(memory, page);
    codePage = nullptr;
}

void RequireBus() {
    Reset();
    Require(ftruncate(ownedFile, static_cast<off_t>(ownedBytes - page)) == 0);
    StoreWord(Page(7) + 8, 31);
    Require(calls.load() == 1);
    Require(lastSignal.load() == SIGBUS);
    Require(LoadWord(Page(7) + 8) == 31);
}

void RequireAsyncSignalsHeld() {
    Reset();
    asyncHeld.store(true);
    faultsUnblocked.store(true);
    usr1Count.store(0);
    usr1DuringGpu.store(0);
    raiseInside.store(true);
    struct sigaction installed{};
    Require(sigaction(SIGSEGV, nullptr, &installed) == 0);
    Require(sigismember(&installed.sa_mask, SIGUSR1) == 0);
    Require(mprotect(Page(3), page, PROT_READ) == 0);
    const auto before = Mask();
    StoreWord(Page(3), 5);
    raiseInside.store(false);
    Require(calls.load() == 1);
    Require(asyncHeld.load());
    Require(faultsUnblocked.load());
    Require(usr1Count.load() == 1);
    Require(usr1DuringGpu.load() == 0);
    Require(SameMask(before, Mask()));
}

void RequireDeclined() {
    Reset();
    usr1Count.store(0);
    usr1DuringGpu.store(0);
    raiseInside.store(true);
    declineAll.store(true);
    Require(mprotect(Page(4), page, PROT_NONE) == 0);
    const auto before = Mask();
    std::array<std::byte, 64> copy{};
    errno = 2468;
    Require(!GuestCopyIn_nid_no_patch(copy.data(), Page(4), copy.size()));
    Require(errno == 2468);
    declineAll.store(false);
    raiseInside.store(false);
    Require(calls.load() == 1);
    Require(lastAccess.load() == static_cast<int>(Access::Read));
    Require(usr1Count.load() == 1);
    Require(usr1DuringGpu.load() == 0);
    Require(SameMask(before, Mask()));
    Require(GuestCopyIn_nid_no_patch(copy.data(), Page(4), copy.size()));
    Require(calls.load() == 2);
}

void RequireNoHandler() {
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(nullptr);
    Reset();
    Require(mprotect(Page(5), page, PROT_NONE) == 0);
    std::array<std::byte, 16> copy{};
    Require(!GuestCopyIn_nid_no_patch(copy.data(), Page(5), copy.size()));
    Require(calls.load() == 0);
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(&GpuFault);
    Require(GuestCopyIn_nid_no_patch(copy.data(), Page(5), copy.size()));
    Require(calls.load() == 1);
}

void RequireBeforeGuestHandlers() {
    guestCalls.store(0);
    SetGuestSegv(reinterpret_cast<std::uintptr_t>(&CountGuestSegv));
    Reset();
    Require(mprotect(Page(6), page, PROT_READ) == 0);
    StoreWord(Page(6), 1);
    Require(calls.load() == 1);
    Require(guestCalls.load() == 0);
    SetGuestSegv(0);
    Require(sceKernelInstallExceptionHandler(GuestSigsegv, reinterpret_cast<void*>(&CountGuestException)) == 0);
    Reset();
    Require(mprotect(Page(6), page, PROT_NONE) == 0);
    Require(LoadWord(Page(6)) == 1);
    Require(calls.load() == 1);
    Require(guestCalls.load() == 0);
    Require(sceKernelRemoveExceptionHandler(GuestSigsegv) == 0);
}

void RequireThreads() {
    constexpr std::size_t Threads = 8;
    constexpr std::uint32_t Rounds = 64;
    const std::size_t bytes = Threads * page;
    void* memory = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(memory != MAP_FAILED);
    auto* region = static_cast<std::byte*>(memory);
    auto* saved = owned;
    const auto savedBytes = ownedBytes;
    owned = region;
    ownedBytes = bytes;
    Reset();
    std::vector<std::thread> threads;
    std::atomic<int> wrong{0};
    for (std::size_t index = 0; index < Threads; ++index) {
        threads.emplace_back([&, index] {
            auto* at = region + index * page;
            for (std::uint32_t round = 1; round <= Rounds; ++round) {
                if (mprotect(at, page, PROT_READ) != 0) wrong.fetch_add(1);
                StoreWord(at + 4 * (round % 16), round);
                if (LoadWord(at + 4 * (round % 16)) != round) wrong.fetch_add(1);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    Require(wrong.load() == 0);
    Require(calls.load() == static_cast<int>(Threads * Rounds));
    owned = saved;
    ownedBytes = savedBytes;
    munmap(memory, bytes);
}

bool Prepare(std::uintptr_t address, std::size_t bytes) {
    prepared.fetch_add(1);
    preparedAddress.store(address);
    preparedBytes.store(bytes);
    if (!prepareOpens.load()) return false;
    const auto first = address & ~(static_cast<std::uintptr_t>(page) - 1);
    const auto last = (address + bytes + page - 1) & ~(static_cast<std::uintptr_t>(page) - 1);
    return mprotect(reinterpret_cast<void*>(first), last - first, PROT_READ | PROT_WRITE) == 0;
}

void RequireHostWrites() {
    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(&Prepare);
    prepared.store(0);
    {
        const GuestArena::HostWrite write(Page(0) + 40, 100);
        Require(write.Open());
    }
    Require(prepared.load() == 1);
    Require(preparedAddress.load() == reinterpret_cast<std::uintptr_t>(Page(0) + 40));
    Require(preparedBytes.load() == 100);
    Require(GuestArena::GuestArenaBeginHostWrite_nid_postfix(Page(1), 8));
    GuestArena::GuestArenaEndHostWrite_nid_postfix(Page(1), 8);
    Require(prepared.load() == 2);
    {
        const GuestArena::HostWrite empty(Page(0), 0);
        Require(empty.Open());
    }
    Require(prepared.load() == 2);

    int pipeEnds[2];
    Require(pipe(pipeEnds) == 0);
    const char text[] = "host write";
    Require(mprotect(Page(5), page, PROT_READ) == 0);
    Reset();
    Require(write(pipeEnds[1], text, sizeof(text)) == static_cast<ssize_t>(sizeof(text)));
    Require(read(pipeEnds[0], Page(5) + 16, sizeof(text)) == -1 && errno == EFAULT);
    Require(calls.load() == 0);
    prepareOpens.store(true);
    {
        const GuestArena::HostWrite write(Page(5) + 16, sizeof(text));
        Require(write.Open());
        Require(read(pipeEnds[0], Page(5) + 16, sizeof(text)) == static_cast<ssize_t>(sizeof(text)));
    }
    prepareOpens.store(false);
    Require(std::memcmp(Page(5) + 16, text, sizeof(text)) == 0);
    Require(calls.load() == 0);
    close(pipeEnds[0]);
    close(pipeEnds[1]);

    GuestArena::GuestArenaSetHostWritePreparer_nid_no_patch(nullptr);
    {
        const GuestArena::HostWrite write(Page(0), 4);
        Require(write.Open());
    }
    Require(prepared.load() == 3);
}

}

int main() {
    page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    ownedBytes = 8 * page;
    ownedFile = static_cast<int>(syscall(SYS_memfd_create, "gpu-fault", 0));
    Require(ownedFile >= 0);
    Require(ftruncate(ownedFile, static_cast<off_t>(ownedBytes)) == 0);
    void* memory = mmap(nullptr, ownedBytes, PROT_READ | PROT_WRITE, MAP_SHARED, ownedFile, 0);
    Require(memory != MAP_FAILED);
    owned = static_cast<std::byte*>(memory);

    struct sigaction usr1{};
    usr1.sa_sigaction = &Usr1;
    usr1.sa_flags = SA_SIGINFO;
    sigemptyset(&usr1.sa_mask);
    Require(sigaction(SIGUSR1, &usr1, nullptr) == 0);

    RequireTrackingOff();
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(&GpuFault);
    RequireWrite();
    RequireRead();
    RequireExecute();
    RequireBus();
    RequireAsyncSignalsHeld();
    RequireDeclined();
    RequireNoHandler();
    RequireBeforeGuestHandlers();
    RequireThreads();
    RequireHostWrites();
    Require(asyncHeld.load());
    Require(faultsUnblocked.load());
    GuestWriteWatch::GuestWriteWatchSetGpuFaultHandler_nid_no_patch(nullptr);
    std::puts("gpu fault chain tests passed");
    return 0;
}
