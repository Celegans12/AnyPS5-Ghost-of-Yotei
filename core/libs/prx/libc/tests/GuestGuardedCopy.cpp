#include "prx/libc/include/Api.hpp"
#include "prx/libc/include/General.hpp"
#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/specifics/linux/GuardedCopy.hpp"
#include "prx/libkernel/include/Api.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int GuestSigsegv = 11;
constexpr std::size_t Sizes[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 255, 4096, 8192};

std::size_t page = 0;
std::atomic<int> guestCalls{0};

void Check(bool value, int line) {
    if (value) return;
    std::fprintf(stderr, "guarded copy check failed at line %d\n", line);
    std::fflush(stderr);
    std::_Exit(1);
}
#define Require(value) Check((value), __LINE__)

bool In(void* destination, const void* source, std::size_t bytes) {
    errno = 12345;
    const bool copied = GuestCopyIn_nid_no_patch(destination, source, bytes);
    Require(errno == 12345);
    return copied;
}

bool InUnsynced(void* destination, const void* source, std::size_t bytes) {
    errno = 12345;
    const bool copied = GuestCopyInUnsynced_nid_no_patch(destination, source, bytes);
    Require(errno == 12345);
    return copied;
}

std::byte* Map(std::size_t pages, int protection) {
    void* memory = mmap(nullptr, pages * page, protection, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(memory != MAP_FAILED);
    return static_cast<std::byte*>(memory);
}

void Fill(std::byte* bytes, std::size_t count, unsigned seed) {
    for (std::size_t index = 0; index < count; ++index) bytes[index] = static_cast<std::byte>((index * 131 + seed) & 0xff);
}

void RequireCopies() {
    std::vector<std::byte> source(1024);
    std::vector<std::byte> destination(1024);
    Fill(source.data(), source.size(), 7);
    for (std::size_t bytes = 0; bytes <= 300; ++bytes) {
        for (std::size_t from = 0; from < 16; from += 3) {
            for (std::size_t to = 0; to < 16; to += 5) {
                std::memset(destination.data(), 0xee, destination.size());
                Require(In(destination.data() + 64 + to, source.data() + from, bytes));
                Require(std::memcmp(destination.data() + 64 + to, source.data() + from, bytes) == 0);
                for (std::size_t at = 0; at < 64 + to; ++at) Require(destination[at] == std::byte{0xee});
                for (std::size_t at = 64 + to + bytes; at < destination.size(); ++at) Require(destination[at] == std::byte{0xee});
                Require(InUnsynced(destination.data(), source.data() + from, bytes));
                Require(std::memcmp(destination.data(), source.data() + from, bytes) == 0);
            }
        }
    }
    std::vector<std::byte> large(3 * 1024 * 1024 + 13);
    std::vector<std::byte> copy(large.size());
    Fill(large.data(), large.size(), 3);
    Require(In(copy.data(), large.data(), large.size()));
    Require(std::memcmp(copy.data(), large.data(), large.size()) == 0);
}

void RequireFails(const std::byte* source, std::size_t bytes) {
    std::vector<std::byte> destination(bytes);
    Require(!In(destination.data(), source, bytes));
    Require(!InUnsynced(destination.data(), source, bytes));
}

void RequireUnreadable() {
    auto* region = Map(4, PROT_READ | PROT_WRITE);
    Fill(region, 4 * page, 11);
    Require(mprotect(region + page, page, PROT_NONE) == 0);
    Require(munmap(region + 3 * page, page) == 0);
    for (const auto bytes : Sizes) RequireFails(region + page, bytes);
    for (const std::size_t before : {1, 3, 4, 8, 16, 31, 32, 33, 4095}) {
        for (const auto bytes : Sizes) {
            RequireFails(region + page - before, before + bytes);
            RequireFails(region + 3 * page - before, before + bytes);
        }
    }
    std::vector<std::byte> destination(page);
    Require(In(destination.data(), region, page));
    Require(std::memcmp(destination.data(), region, page) == 0);
    Require(In(destination.data(), region + 2 * page, page));
    Require(std::memcmp(destination.data(), region + 2 * page, page) == 0);
    for (const std::uintptr_t address : {std::uintptr_t{1}, ~std::uintptr_t{0xfff}, std::uintptr_t{0x0000800000000000}, std::uintptr_t{0x8000000000000000}}) {
        std::uint64_t value = 0;
        Require(!In(&value, reinterpret_cast<const void*>(address), sizeof(value)));
    }
    auto* readOnly = Map(1, PROT_READ);
    Require(!In(readOnly, region, 8));
    Require(!In(readOnly + page - 4, region, 64));
    munmap(readOnly, page);
    const int file = static_cast<int>(syscall(SYS_memfd_create, "guarded-copy", 0));
    Require(file >= 0);
    Require(ftruncate(file, static_cast<off_t>(page)) == 0);
    void* mapped = mmap(nullptr, 2 * page, PROT_READ, MAP_SHARED, file, 0);
    Require(mapped != MAP_FAILED);
    auto* shared = static_cast<const std::byte*>(mapped);
    Require(In(destination.data(), shared, page));
    for (const auto bytes : Sizes) {
        RequireFails(shared + page, bytes);
        RequireFails(shared + page - 8, 8 + bytes);
    }
    munmap(mapped, 2 * page);
    close(file);
    munmap(region, 3 * page);
}

void RequireThreads() {
    auto* region = Map(2, PROT_READ | PROT_WRITE);
    Fill(region, page, 5);
    Require(mprotect(region + page, page, PROT_NONE) == 0);
    std::vector<std::thread> threads;
    std::atomic<int> failures{0};
    for (int index = 0; index < 8; ++index) {
        threads.emplace_back([&, index] {
            std::byte buffer[64];
            for (int round = 0; round < 20000; ++round) {
                const auto offset = static_cast<std::size_t>((round * 7 + index) % 64);
                if (!GuestCopyIn_nid_no_patch(buffer, region + offset, sizeof(buffer)) || std::memcmp(buffer, region + offset, sizeof(buffer)) != 0) failures.fetch_add(1);
                if (GuestCopyIn_nid_no_patch(buffer, region + page - 8 - offset % 8, sizeof(buffer))) failures.fetch_add(1);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    Require(failures.load() == 0);
    munmap(region, 2 * page);
}

int RunChild(void (*body)()) {
    std::fflush(nullptr);
    const pid_t child = fork();
    Require(child >= 0);
    if (child == 0) {
        body();
        _exit(0);
    }
    int status = 0;
    Require(waitpid(child, &status, 0) == child);
    return status;
}

void KillOnCopySyscalls() {
    sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_process_vm_readv, 2, 0),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_getpid, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    };
    sock_fprog program{static_cast<unsigned short>(std::size(filter)), filter};
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 || prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) != 0) _exit(2);
}

void CopyUnderFilter() {
    auto* region = Map(2, PROT_READ | PROT_WRITE);
    Fill(region, page, 9);
    Require(mprotect(region + page, page, PROT_NONE) == 0);
    KillOnCopySyscalls();
    std::byte buffer[256];
    for (const std::size_t bytes : {std::size_t{4}, std::size_t{8}, std::size_t{100}, sizeof(buffer)}) {
        Require(GuestCopyIn_nid_no_patch(buffer, region, bytes));
        Require(GuestCopyInUnsynced_nid_no_patch(buffer, region + 64, bytes));
        Require(!GuestCopyIn_nid_no_patch(buffer, region + page - 2, bytes));
    }
}

void RequireNoCopySyscall() {
    const int status = RunChild(CopyUnderFilter);
    Require(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void RequireCopySyscall() {
    const int status = RunChild(CopyUnderFilter);
    Require(WIFSIGNALED(status) && WTERMSIG(status) == SIGSYS);
}

void APS5_VABI CountGuestSegv(int) {
    guestCalls.fetch_add(1);
}

void APS5_VABI LeaveGuestSegv(int) {
    _exit(42);
}

void APS5_VABI LeaveGuestException(int, void*) {
    _exit(43);
}

void APS5_VABI CountGuestException(int, void*) {
    guestCalls.fetch_add(1);
}

void SetGuestSegv(std::uintptr_t handler) {
    GuestSigaction action{};
    action.handler = handler;
    Require(sigaction_nid_postfix(GuestSigsegv, &action, nullptr) == 0);
}

void RequireCopyFaultsStayInside() {
    auto* region = Map(2, PROT_READ | PROT_WRITE);
    Require(mprotect(region + page, page, PROT_NONE) == 0);
    guestCalls.store(0);
    for (const auto bytes : Sizes) RequireFails(region + page - 4, 4 + bytes);
    Require(guestCalls.load() == 0);
    munmap(region, 2 * page);
}

void Fault() {
    *reinterpret_cast<volatile int*>(std::uintptr_t{16}) = 1;
}

void FaultWithGuestHandler() {
    SetGuestSegv(reinterpret_cast<std::uintptr_t>(&LeaveGuestSegv));
    RequireCopyFaultsStayInside();
    Fault();
}

void FaultWithGuestDefault() {
    SetGuestSegv(0);
    RequireCopyFaultsStayInside();
    Fault();
}

void FaultWithGuestIgnore() {
    SetGuestSegv(1);
    RequireCopyFaultsStayInside();
    Fault();
}

void FaultWithException() {
    Require(sceKernelInstallExceptionHandler(GuestSigsegv, reinterpret_cast<void*>(&LeaveGuestException)) == 0);
    RequireCopyFaultsStayInside();
    Fault();
}

void RequireGuestHandlers() {
    SetGuestSegv(reinterpret_cast<std::uintptr_t>(&CountGuestSegv));
    RequireCopyFaultsStayInside();
    Require(raise_nid_postfix(GuestSigsegv) == 0);
    Require(guestCalls.load() == 1);
    SetGuestSegv(0);
    RequireCopyFaultsStayInside();
    SetGuestSegv(1);
    RequireCopyFaultsStayInside();
    int status = RunChild(FaultWithGuestHandler);
    Require(WIFEXITED(status) && WEXITSTATUS(status) == 42);
    status = RunChild(FaultWithGuestDefault);
    Require(WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV);
    status = RunChild(FaultWithGuestIgnore);
    Require(WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV);
    status = RunChild(FaultWithException);
    Require(WIFEXITED(status) && WEXITSTATUS(status) == 43);
    SetGuestSegv(0);
    Require(sceKernelInstallExceptionHandler(GuestSigsegv, reinterpret_cast<void*>(&CountGuestException)) == 0);
    RequireCopyFaultsStayInside();
    Require(sceKernelRemoveExceptionHandler(GuestSigsegv) == 0);
}

}

int main(int argc, char** argv) {
    Require(argc == 2);
    const std::string_view mode = argv[1];
    Require(mode == "direct" || mode == "syscall");
    const bool direct = mode == "direct";
    page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    Require(GuardedCopy::Ready() == direct);
    RequireCopies();
    RequireUnreadable();
    RequireThreads();
    if (direct) RequireNoCopySyscall();
    else RequireCopySyscall();
    RequireGuestHandlers();
    RequireCopies();
    RequireUnreadable();
    std::puts("guarded copy tests passed");
}
