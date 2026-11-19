#include "prx/libkernel/include/Api.hpp"
#include "SceTypes.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

extern "C" char __executable_start[];
extern "C" char etext[];

static constexpr int SigUsr1 = 30;
static constexpr std::size_t RipOffset = 0xe0;
static constexpr int ChainedRaises = 2000;
static constexpr int Rounds = 6;
static constexpr std::size_t TextBytes = 4u << 20u;
static constexpr auto HandlerSpin = std::chrono::milliseconds(5);
static constexpr auto LockedDeliveryLimit = std::chrono::seconds(1);
static void Require(bool value) { if (!value) std::abort(); }

enum class Mode : int { Measure, Lock, Locked };

static std::vector<char> text;
static std::mutex hostLock;
static std::atomic<int> entered{0};
static std::atomic<int> deliveries{0};
static std::atomic<int> outsideGuest{0};
static std::atomic<bool> spin{false};
static std::atomic<Mode> mode{Mode::Measure};
static std::atomic<bool> stop{false};
static std::atomic<bool> started{false};

static void APS5_VABI Handler(int, void* context) {
    std::uint64_t rip = 0;
    std::memcpy(&rip, static_cast<const unsigned char*>(context) + RipOffset, sizeof(rip));
    if (rip < reinterpret_cast<std::uintptr_t>(__executable_start) || rip >= reinterpret_cast<std::uintptr_t>(etext)) outsideGuest.fetch_add(1);
    entered.fetch_add(1);
    if (spin.load()) {
        const auto end = std::chrono::steady_clock::now() + HandlerSpin;
        while (std::chrono::steady_clock::now() < end) {}
    }
    deliveries.fetch_add(1);
}

static void* APS5_VABI Measuring(void*) {
    const std::size_t expected = text.size() - 1;
    started.store(true);
    while (!stop.load(std::memory_order_relaxed)) {
        if (mode.load() == Mode::Lock) {
            hostLock.lock();
            hostLock.unlock();
            mode.store(Mode::Locked);
            continue;
        }
        Require(std::strlen(text.data()) == expected);
    }
    return nullptr;
}

static bool AwaitCount(const std::atomic<int>& counter, int value, std::chrono::steady_clock::duration limit) {
    const auto start = std::chrono::steady_clock::now();
    while (counter.load() < value) {
        if (std::chrono::steady_clock::now() - start > limit) return false;
        std::this_thread::yield();
    }
    return true;
}

int main() {
    text.assign(TextBytes, 'a');
    text.back() = '\0';
    Require(sceKernelInstallExceptionHandler(SigUsr1, reinterpret_cast<void*>(&Handler)) == 0);
    Pthread thread = nullptr;
    Require(scePthreadCreate(&thread, nullptr, Measuring, nullptr, "measuring") == 0);
    while (!started.load()) std::this_thread::yield();
    for (int raise = 0; raise < ChainedRaises; ++raise) {
        const int before = deliveries.load();
        Require(sceKernelRaiseException(thread, SigUsr1) == 0);
        if (!AwaitCount(deliveries, before + 1, std::chrono::seconds(5))) {
            std::fprintf(stderr, "chained raise %d was not delivered in 5 s\n", raise + 1);
            std::abort();
        }
    }
    const int chainedOutside = outsideGuest.load();
    auto longest = std::chrono::steady_clock::duration::zero();
    for (int round = 0; round < Rounds; ++round) {
        spin.store(true);
        const int before = deliveries.load();
        Require(sceKernelRaiseException(thread, SigUsr1) == 0);
        Require(AwaitCount(entered, before + 1, std::chrono::seconds(5)));
        Require(sceKernelRaiseException(thread, SigUsr1) == 0);
        Require(AwaitCount(deliveries, before + 1, std::chrono::seconds(5)));
        spin.store(false);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        hostLock.lock();
        mode.store(Mode::Lock);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const int waiting = deliveries.load();
        const auto raisedAt = std::chrono::steady_clock::now();
        Require(sceKernelRaiseException(thread, SigUsr1) == 0);
        const bool delivered = AwaitCount(deliveries, waiting + 1, LockedDeliveryLimit);
        const auto waited = std::chrono::steady_clock::now() - raisedAt;
        hostLock.unlock();
        while (mode.load() != Mode::Locked) std::this_thread::yield();
        mode.store(Mode::Measure);
        if (!delivered) {
            std::fprintf(stderr, "round %d: a raise on a thread waiting for a held host mutex was not delivered within 1 s after a raise merged into a pending signal\n", round + 1);
            std::abort();
        }
        longest = std::max(longest, waited);
    }
    stop.store(true);
    Require(scePthreadJoin(thread, nullptr) == 0);
    Require(sceKernelRemoveExceptionHandler(SigUsr1) == 0);
    std::printf("%d chained raises delivered, %d of their handlers outside guest code; %d raises on a thread waiting for a held host mutex after a merged raise delivered, longest %.1f ms\n",
        ChainedRaises, chainedOutside, Rounds, std::chrono::duration<double, std::milli>(longest).count());
    Require(chainedOutside == 0);
}
