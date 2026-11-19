#include "FaultReplay.hpp"

#include <array>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <sys/mman.h>
#include <unistd.h>

namespace {

namespace Replay = KytyShim::FaultReplay;

constexpr std::size_t   PageSize = 4096;
constexpr std::uint64_t Word     = 0x1122334455667788ull;
constexpr std::uint8_t  Byte     = 0xAB;

std::uint8_t*              g_guest   = nullptr;
std::uint8_t*              g_backing = nullptr;
std::atomic<int>           g_replays {0};
std::atomic<int>           g_fallbacks {0};
std::atomic<int>           g_traps {0};
std::atomic<bool>          g_raise_on_replay {false};
std::atomic<int>           g_user_traps {-1};
std::atomic<std::uint64_t> g_user_value {0};

void Require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "kyty_shim_fault_replay: %s\n", message);
    std::abort();
}

int Register(std::initializer_list<std::uint8_t> bytes) {
    std::array<std::uint8_t, 32> code {};
    std::size_t                  i = 0;
    for (const auto b : bytes) code[i++] = b;
    return Replay::ReplayRegister(code.data());
}

const void* Backing(std::uint64_t ea, std::uint64_t size) {
    const auto guest = reinterpret_cast<std::uint64_t>(g_guest);
    if (ea < guest || ea + size > guest + PageSize) return nullptr;
    return g_backing + (ea - guest);
}

void OnSegv(int, siginfo_t* info, void* native) {
    auto&      context = *static_cast<ucontext_t*>(native);
    const auto address = reinterpret_cast<std::uint64_t>(info->si_addr);
    if (Replay::Start(context, address, [](std::uint64_t, std::uint64_t) { return true; }, Backing)) {
        g_replays++;
        if (g_raise_on_replay.exchange(false)) ::raise(SIGUSR1);
        return;
    }
    g_fallbacks++;
    ::mprotect(g_guest, PageSize, PROT_READ);
}

void OnTrap(int, siginfo_t*, void* native) {
    if (!Replay::Finish(*static_cast<ucontext_t*>(native))) {
        static const char message[] = "kyty_shim_fault_replay: SIGTRAP not claimed\n";
        (void)!::write(2, message, sizeof(message) - 1);
        ::_exit(1);
    }
    g_traps++;
}

void OnUser(int) {
    g_user_traps = g_traps.load();
    g_user_value = *reinterpret_cast<volatile std::uint64_t*>(g_guest + 64);
}

void Install(int number, void (*handler)(int, siginfo_t*, void*), int blocked) {
    struct sigaction action {};
    action.sa_sigaction = handler;
    action.sa_flags     = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    if (blocked != 0) sigaddset(&action.sa_mask, blocked);
    Require(::sigaction(number, &action, nullptr) == 0, "sigaction failed");
}

struct Outcome {
    int replays;
    int fallbacks;
};

template <typename Body>
Outcome Run(Body&& body) {
    Require(::mprotect(g_guest, PageSize, PROT_NONE) == 0, "mprotect failed");
    const int replays   = g_replays;
    const int fallbacks = g_fallbacks;
    body();
    return {g_replays - replays, g_fallbacks - fallbacks};
}

std::uint64_t Address(std::size_t offset) {
    return reinterpret_cast<std::uint64_t>(g_guest + offset);
}

std::uint64_t LoadThrough(std::uint64_t address, std::uint64_t& base_after) {
    std::uint64_t value = 0;
    std::uint64_t base  = address;
    asm volatile("mov (%1), %0" : "=&r"(value), "+d"(base) : : "memory");
    base_after = base;
    return value;
}

std::uint64_t LoadSelf(std::uint64_t address) {
    asm volatile("mov (%0), %0" : "+d"(address) : : "memory");
    return address;
}

bool CompareSelf(std::uint64_t address) {
    bool equal = false;
    asm volatile("cmp (%1), %1\n\tsete %0" : "=&q"(equal) : "D"(address) : "cc", "memory");
    return equal;
}

std::uint64_t AddSelf(std::uint64_t address) {
    asm volatile("add 8(%0), %0" : "+b"(address) : : "cc", "memory");
    return address;
}

std::uint64_t LoadLowByteSelf(std::uint64_t address) {
    asm volatile("movb (%0), %%al" : "+a"(address) : : "memory");
    return address;
}

std::uint64_t LoadHighByteSelf(std::uint64_t address) {
    asm volatile("movb (%0), %%ah" : "+a"(address) : : "memory");
    return address;
}

std::uint64_t ZeroHighBitsSelf(std::uint64_t address) {
    std::uint64_t value = 0;
    asm volatile("bzhi %1, (%1), %0" : "=&a"(value) : "c"(address) : "memory");
    return value;
}

std::uint64_t ZeroHighBits(std::uint64_t address, std::uint64_t bits) {
    std::uint64_t value = 0;
    asm volatile("bzhi %2, (%1), %0" : "=&a"(value) : "c"(address), "d"(bits) : "memory");
    return value;
}

void Decoding() {
    Require(Register({0x48, 0x3B, 0x3F}) == -1, "cmp rdi, [rdi] would be replayed");
    Require(Register({0x48, 0x03, 0x5B, 0x08}) == -1, "add rbx, [rbx+8] would be replayed");
    Require(Register({0x48, 0x8B, 0x02}) == 2, "mov rax, [rdx] would not be replayed through rdx");
    Require(Register({0x48, 0x8B, 0x12}) == 2, "mov rdx, [rdx] would not be replayed");
    Require(Register({0x8A, 0x00}) == -1, "mov al, [rax] would be replayed");
    Require(Register({0x8A, 0x20}) == -1, "mov ah, [rax] would be replayed");
    Require(Register({0x40, 0x8A, 0x20}) == 0, "mov spl, [rax] would not be replayed");
    Require(Register({0x8A, 0x38}) == 0, "mov bh, [rax] would not be replayed");
    Require(Register({0x66, 0x8B, 0x00}) == -1, "mov ax, [rax] would be replayed");
    Require(Register({0x0F, 0xB6, 0x00}) == 0, "movzx eax, byte [rax] would not be replayed");
    Require(Register({0x66, 0x0F, 0xB6, 0x00}) == -1, "movzx ax, byte [rax] would be replayed");
    Require(Register({0x48, 0x0F, 0x44, 0x00}) == -1, "cmove rax, [rax] would be replayed");
    Require(Register({0x48, 0x0F, 0xA3, 0x00}) == -1, "bt [rax], rax would be replayed");
    Require(Register({0x48, 0x0F, 0xAF, 0x00}) == -1, "imul rax, [rax] would be replayed");
    Require(Register({0x48, 0x6B, 0x00, 0x05}) == 0, "imul rax, [rax], 5 would not be replayed");
    Require(Register({0x48, 0x0F, 0xBC, 0x00}) == -1, "bsf rax, [rax] would be replayed");
    Require(Register({0xF3, 0x48, 0x0F, 0xBC, 0x00}) == 0, "tzcnt rax, [rax] would not be replayed");
    Require(Register({0xF2, 0x48, 0x0F, 0x38, 0xF1, 0x00}) == -1, "crc32 rax, [rax] would be replayed");
    Require(Register({0x45, 0x3B, 0x00}) == -1, "cmp r8d, [r8] would be replayed");
    Require(Register({0x44, 0x3B, 0x00}) == 0, "cmp r8d, [rax] would not be replayed");
    Require(Register({0x41, 0x3B, 0x00}) == 8, "cmp eax, [r8] would not be replayed");
    Require(Register({0x48, 0x03, 0x04, 0x18}) == -1, "add rax, [rax+rbx] would be replayed");
    Require(Register({0x48, 0x03, 0x1C, 0x18}) == 0, "add rbx, [rax+rbx] would not be replayed through rax");
    Require(Register({0xC4, 0xE2, 0xF8, 0xF5, 0x04, 0x19}) == 1, "bzhi rax, [rcx+rbx], rax would not be replayed");
    Require(Register({0xC4, 0xE2, 0xF0, 0xF5, 0x04, 0x19}) == -1, "bzhi rax, [rcx+rbx], rcx would be replayed");
    Require(Register({0xC4, 0xE2, 0xF1, 0xF7, 0x01}) == -1, "shlx rax, [rcx], rcx would be replayed");
    Require(Register({0xC5, 0xF8, 0x10, 0x00}) == 0, "vmovups xmm0, [rax] would not be replayed");
    Require(Register({0x0F, 0xAE, 0x08}) == -1, "fxrstor [rax] would be replayed");
    Require(Register({0x0F, 0xAE, 0x28}) == -1, "xrstor [rax] would be replayed");
    Require(Register({0xDD, 0x20}) == -1, "frstor [rax] would be replayed");
    Require(Register({0x0F, 0xAE, 0x10}) == 0, "ldmxcsr [rax] would not be replayed");
    Require(Register({0xF7, 0x20}) == -1, "mul [rax] would be replayed");
    Require(Register({0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00}) == -1, "a RIP-relative load would be replayed");
}

void Replaying() {
    std::uint64_t base   = 0;
    std::uint64_t value  = 0;
    auto          result = Run([&] { value = LoadThrough(Address(64), base); });
    Require(result.replays == 1 && result.fallbacks == 0, "mov r, [rdx] was not replayed");
    Require(value == Word && base == Address(64), "mov r, [rdx] read a wrong value or left rdx shifted");

    result = Run([&] { value = LoadSelf(Address(64)); });
    Require(result.replays == 1 && result.fallbacks == 0, "mov rdx, [rdx] was not replayed");
    Require(value == Word, "mov rdx, [rdx] read a wrong value");

    bool equal = false;
    result     = Run([&] { equal = CompareSelf(Address(0)); });
    Require(result.replays == 0 && result.fallbacks == 1, "cmp rdi, [rdi] was replayed");
    Require(equal, "cmp rdi, [rdi] compared a shifted rdi");

    result = Run([&] { value = AddSelf(Address(0)); });
    Require(result.replays == 0 && result.fallbacks == 1, "add rbx, [rbx+8] was replayed");
    Require(value == Address(0) + 1, "add rbx, [rbx+8] added to a shifted rbx");

    result = Run([&] { value = LoadLowByteSelf(Address(16)); });
    Require(result.replays == 0 && result.fallbacks == 1, "mov al, [rax] was replayed");
    Require(value == ((Address(16) & ~0xFFull) | Byte), "mov al, [rax] left rax shifted");

    result = Run([&] { value = LoadHighByteSelf(Address(16)); });
    Require(result.replays == 0 && result.fallbacks == 1, "mov ah, [rax] was replayed");
    Require(value == ((Address(16) & ~0xFF00ull) | (std::uint64_t {Byte} << 8u)), "mov ah, [rax] left rax shifted");

    if (__builtin_cpu_supports("bmi2")) {
        result = Run([&] { value = ZeroHighBitsSelf(Address(64)); });
        Require(result.replays == 0 && result.fallbacks == 1, "bzhi rax, [rcx], rcx was replayed");
        const auto index = Address(64) & 0xFFu;
        Require(value == (index >= 64 ? Word : Word & ((std::uint64_t {1} << index) - 1)), "bzhi rax, [rcx], rcx read a wrong value");

        result = Run([&] { value = ZeroHighBits(Address(64), 16); });
        Require(result.replays == 1 && result.fallbacks == 0, "bzhi rax, [rcx], rdx was not replayed");
        Require(value == (Word & 0xFFFFu), "bzhi rax, [rcx], rdx read a wrong value");
    }

    const int traps   = g_traps;
    g_raise_on_replay = true;
    result            = Run([&] { value = LoadThrough(Address(64), base); });
    Require(result.replays == 2 && result.fallbacks == 0, "the read of a signal pending during a replay was not replayed after it");
    Require(value == Word && base == Address(64), "a signal pending during a replay broke the replayed read");
    Require(g_user_traps == traps + 1, "a signal pending during a replay ran before the replayed instruction");
    Require(g_user_value == Word, "a signal handler read a wrong value during a replay");
}

}

int main() {
    Decoding();
    const int file = ::memfd_create("kyty-fault-replay", 0);
    Require(file >= 0 && ::ftruncate(file, PageSize) == 0, "memfd failed");
    g_guest   = static_cast<std::uint8_t*>(::mmap(nullptr, PageSize, PROT_NONE, MAP_SHARED, file, 0));
    g_backing = static_cast<std::uint8_t*>(::mmap(nullptr, PageSize, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0));
    Require(g_guest != MAP_FAILED && g_backing != MAP_FAILED, "mmap failed");
    const auto guest = reinterpret_cast<std::uint64_t>(g_guest);
    std::memcpy(g_backing, &guest, sizeof(guest));
    const std::uint64_t one = 1;
    std::memcpy(g_backing + 8, &one, sizeof(one));
    g_backing[16] = Byte;
    std::memcpy(g_backing + 64, &Word, sizeof(Word));
    Install(SIGSEGV, OnSegv, SIGUSR1);
    Install(SIGTRAP, OnTrap, 0);
    Require(std::signal(SIGUSR1, OnUser) != SIG_ERR, "signal failed");
    Replaying();
    std::puts("kyty_shim_fault_replay: ok");
    return 0;
}
