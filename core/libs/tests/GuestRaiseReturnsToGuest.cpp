#include "prx/libkernel/include/Api.hpp"
#include "SceTypes.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

extern "C" char __executable_start[];
extern "C" char etext[];

static constexpr int SigUsr1 = 30;
static constexpr int Raises = 200;
static constexpr std::size_t RipOffset = 0xe0;
static constexpr std::size_t TextBytes = 4u << 20u;
static constexpr auto DeliveryLimit = std::chrono::milliseconds(50);
static void Require(bool value) { if (!value) std::abort(); }

static std::vector<char> text;
static std::atomic<int> deliveries{0};
static std::atomic<int> outsideGuest{0};
static std::atomic<std::uint64_t> wrongLengths{0};
static std::atomic<std::uint64_t> calls{0};
static std::atomic<bool> stop{false};
static std::atomic<bool> started{false};

static void APS5_VABI Handler(int, void* context) {
    std::uint64_t rip = 0;
    std::memcpy(&rip, static_cast<const unsigned char*>(context) + RipOffset, sizeof(rip));
    if (rip < reinterpret_cast<std::uintptr_t>(__executable_start) || rip >= reinterpret_cast<std::uintptr_t>(etext)) outsideGuest.fetch_add(1);
    deliveries.fetch_add(1);
}

static void* APS5_VABI Measuring(void*) {
    const std::size_t expected = text.size() - 1;
    started.store(true);
    while (!stop.load(std::memory_order_relaxed)) {
        if (std::strlen(text.data()) != expected) wrongLengths.fetch_add(1);
        calls.fetch_add(1, std::memory_order_relaxed);
    }
    return nullptr;
}

int main() {
    text.assign(TextBytes, 'a');
    text.back() = '\0';
    Require(sceKernelInstallExceptionHandler(SigUsr1, reinterpret_cast<void*>(&Handler)) == 0);
    Pthread thread = nullptr;
    Require(scePthreadCreate(&thread, nullptr, Measuring, nullptr, "measuring") == 0);
    while (!started.load()) std::this_thread::yield();
    auto longest = std::chrono::steady_clock::duration::zero();
    for (int raise = 1; raise <= Raises; ++raise) {
        const auto raisedAt = std::chrono::steady_clock::now();
        Require(sceKernelRaiseException(thread, SigUsr1) == 0);
        while (deliveries.load() < raise) {
            Require(std::chrono::steady_clock::now() - raisedAt < std::chrono::seconds(5));
            std::this_thread::yield();
        }
        longest = std::max(longest, std::chrono::steady_clock::now() - raisedAt);
    }
    stop.store(true);
    Require(scePthreadJoin(thread, nullptr) == 0);
    Require(sceKernelRemoveExceptionHandler(SigUsr1) == 0);
    std::printf("%d raises on a thread in host code %llu times: %d handlers outside guest code, %llu wrong results, longest delivery %.2f ms\n", Raises,
        static_cast<unsigned long long>(calls.load()), outsideGuest.load(), static_cast<unsigned long long>(wrongLengths.load()), std::chrono::duration<double, std::milli>(longest).count());
    Require(deliveries.load() == Raises);
    Require(outsideGuest.load() == 0);
    Require(wrongLengths.load() == 0);
    Require(longest < DeliveryLimit);
}
