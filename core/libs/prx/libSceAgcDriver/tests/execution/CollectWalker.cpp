#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawPipeline.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using AgcDriver::DriverDetail::CollectWalker;
namespace GuestMemory = AgcDriver::GuestMemory;

constexpr std::uint64_t Block = 65536;
constexpr std::uint64_t Page = 4096;
constexpr std::uint64_t Blocks = 16;
constexpr std::chrono::seconds Patience{10};

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Pause {
    std::mutex mutex;
    std::condition_variable changed;
    bool armed = false;
    bool reached = false;
    bool released = false;
    bool timedOut = false;
};

Pause& PauseState() {
    static Pause pause;
    return pause;
}

std::atomic<std::uint64_t> osWalks{0};
std::atomic<bool> failNextWalk{false};
std::atomic<bool> stopReturned{false};

void Hook(int point) {
    if (point != 1) return;
    osWalks.fetch_add(1);
    if (failNextWalk.exchange(false)) throw std::runtime_error("injected walk failure");
    auto& pause = PauseState();
    std::unique_lock lock(pause.mutex);
    if (!pause.armed) return;
    pause.armed = false;
    pause.reached = true;
    pause.changed.notify_all();
    if (!pause.changed.wait_for(lock, Patience, [&] { return pause.released; })) pause.timedOut = true;
}

void ArmPause() {
    auto& pause = PauseState();
    const std::lock_guard lock(pause.mutex);
    pause.armed = true;
    pause.reached = false;
    pause.released = false;
    pause.timedOut = false;
}

bool PauseReached() {
    auto& pause = PauseState();
    std::unique_lock lock(pause.mutex);
    return pause.changed.wait_for(lock, Patience, [&] { return pause.reached; });
}

void ReleasePause() {
    auto& pause = PauseState();
    {
        const std::lock_guard lock(pause.mutex);
        pause.armed = false;
        pause.released = true;
    }
    pause.changed.notify_all();
}

bool PauseTimedOut() {
    auto& pause = PauseState();
    const std::lock_guard lock(pause.mutex);
    return pause.timedOut;
}

bool Within(std::chrono::milliseconds patience, const std::function<bool()>& done) {
    const auto until = std::chrono::steady_clock::now() + patience;
    while (!done()) {
        if (std::chrono::steady_clock::now() >= until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

bool WalkerIdle() {
    return Within(Patience, [] { return CollectWalker::Get().Idle(); });
}

std::uint64_t AllocateWatched(std::size_t bytes) {
#ifdef _WIN32
    void* block = GuestArena::GuestArenaAllocate_nid_postfix(bytes, Block);
    Require(block != nullptr, "the guest arena allocation failed");
    GuestArena::GuestArenaCommit_nid_postfix(block, bytes, PAGE_READWRITE, bytes);
#else
    void* raw = mmap(nullptr, bytes + Block, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(raw != MAP_FAILED, "cannot map the watched block");
    const auto begin = reinterpret_cast<std::uintptr_t>(raw);
    const auto aligned = (begin + Block - 1) & ~static_cast<std::uintptr_t>(Block - 1);
    if (aligned != begin) munmap(raw, aligned - begin);
    if (aligned + bytes != begin + bytes + Block) munmap(reinterpret_cast<void*>(aligned + bytes), begin + Block - aligned);
    void* block = reinterpret_cast<void*>(aligned);
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(block, bytes);
#endif
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    std::memset(block, 0x11, bytes);
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(block));
}

void Store(std::uint64_t address) {
    auto* byte = reinterpret_cast<volatile std::uint8_t*>(static_cast<std::uintptr_t>(address));
    *byte = static_cast<std::uint8_t>(*byte + 1);
}

void WalkAnswersTheEpochItWasHandedIn(std::uint64_t base) {
    auto& walker = CollectWalker::Get();
    const auto address = base + 2 * Block + 3 * Page;
    const auto probe = address + Page;
    GuestMemory::BumpCollectEpoch();
    const auto epoch = GuestMemory::HeldCollectEpoch();
    Require(GuestMemory::CollectWrites(base, Page) != 0, "a collect of the watched block failed");
    const auto before = GuestMemory::TrackerGeneration();
    Store(probe);
    const auto counted = walker.Counted();
    const auto walks = osWalks.load();
    Require(walker.Offer(epoch, address, 2 * Page), "the walker refused a range while its queue had room");
    Require(WalkerIdle(), "the walker did not finish the range it was handed");
    const auto after = walker.Counted();
    Require(after.handed == counted.handed + 1 && after.collected == counted.collected + 1 && after.dropped == counted.dropped, "the walker's counts do not show one range handed over and collected");
    const auto walked = osWalks.load();
    Require(walked > walks, "the walker did not walk the range it was handed");
    Require(!GuestMemory::UnchangedSince(probe & ~(Page - 1), Page, before), "the walker's walk did not stamp a store made before it");
    Require(GuestMemory::CollectWrites(address, 2 * Page) != 0 && osWalks.load() == walked, "a collect in the epoch the range was handed in walked it again");
    GuestMemory::BumpCollectEpoch();
    Require(GuestMemory::CollectWrites(address, 2 * Page) != 0 && osWalks.load() > walked, "a walk the walker made answered a later epoch");
}

void FullQueueLeavesRangesToTheCollector(std::uint64_t base) {
    auto& walker = CollectWalker::Get();
    const auto address = base + 4 * Block;
    GuestMemory::BumpCollectEpoch();
    const auto epoch = GuestMemory::HeldCollectEpoch();
    const auto counted = walker.Counted();
    ArmPause();
    Require(walker.Offer(epoch, address, Page), "the walker refused the first range");
    Require(PauseReached(), "the walker did not start the walk it was handed");
    std::uint64_t accepted = 1;
    std::uint64_t refused = 0;
    for (std::size_t offer = 0; offer < CollectWalker::Capacity + 8; ++offer) {
        if (walker.Offer(epoch, address, Page)) ++accepted;
        else ++refused;
    }
    const auto full = walker.Counted();
    ReleasePause();
    Require(accepted == CollectWalker::Capacity && refused == 9, "a queue behind a walk in progress did not take its capacity and refuse the rest");
    Require(full.handed - counted.handed == accepted && full.dropped - counted.dropped == refused, "the walker's counts do not show the ranges its full queue refused");
    Require(WalkerIdle(), "the walker did not empty its queue");
    Require(!PauseTimedOut(), "the paused walk timed out");
    Require(walker.Counted().collected - counted.collected == accepted, "the walker did not collect every range its queue took");
    const auto walks = osWalks.load();
    Require(GuestMemory::CollectWrites(address, Page) != 0 && osWalks.load() == walks, "a range the walker collected was walked again in its epoch");
    Require(walker.Offer(epoch, address, Page) && WalkerIdle(), "the walker's queue took no range once it had room again");
}

void CollectUnderTheGpuLockWaitsForTheWalk(std::uint64_t base) {
    auto& walker = CollectWalker::Get();
    const auto address = base + 6 * Block;
    GuestMemory::BumpCollectEpoch();
    const auto epoch = GuestMemory::HeldCollectEpoch();
    ArmPause();
    Require(walker.Offer(epoch, address, Block), "the walker refused a range while its queue had room");
    Require(PauseReached(), "the walker did not start the walk it was handed");
    const auto walks = osWalks.load();
    std::atomic<bool> finished{false};
    std::uint64_t collected = 0;
    std::thread committer([&] {
        GuestMemory::MarkProofThread();
        GuestMemory::AdoptCollectEpoch(epoch);
        const std::lock_guard gpu(GuestMemory::GpuMutex());
        collected = GuestMemory::CollectWrites(address + Page, Page);
        finished.store(true);
    });
    const bool early = Within(std::chrono::milliseconds(150), [&] { return finished.load(); });
    ReleasePause();
    committer.join();
    Require(!early, "a collect of a block the walker was walking did not wait for that walk");
    Require(!PauseTimedOut(), "the paused walk timed out");
    Require(collected != 0 && osWalks.load() == walks, "a collect that waited for the walker's walk of its block walked the block again");
    Require(WalkerIdle(), "the walker did not finish the walk a collect waited for");
}

void StoppedWalkerTakesNothing(std::uint64_t base) {
    auto& walker = CollectWalker::Get();
    const auto address = base + 8 * Block;
    Require(WalkerIdle(), "the walker still held ranges before it was stopped");
    std::thread stopper([] {
        CollectWalker::Get().Stop();
        stopReturned.store(true);
    });
    if (!Within(Patience, [] { return stopReturned.load(); })) {
        stopper.detach();
        throw std::runtime_error("stopping an idle walker did not return");
    }
    stopper.join();
    const auto counted = walker.Counted();
    GuestMemory::BumpCollectEpoch();
    Require(!walker.Offer(GuestMemory::HeldCollectEpoch(), address, Page), "a stopped walker took a range");
    Require(walker.Counted().dropped == counted.dropped + 1 && walker.Counted().handed == counted.handed, "a range offered to a stopped walker was not counted as left to the collector");
    const auto walks = osWalks.load();
    Require(GuestMemory::CollectWrites(address, Page) != 0 && osWalks.load() > walks, "a range a stopped walker refused was not walked by the thread that collected it");
}

void FailedWalkReachesTheNextOffer(std::uint64_t base) {
    auto& walker = CollectWalker::Get();
    const auto address = base + 10 * Block;
    GuestMemory::BumpCollectEpoch();
    const auto epoch = GuestMemory::HeldCollectEpoch();
    failNextWalk.store(true);
    Require(walker.Offer(epoch, address, Page), "the walker refused a range while its queue had room");
    Require(WalkerIdle(), "the walker did not give up after a failed walk");
    bool rethrown = false;
    try {
        walker.Offer(epoch, address, Page);
    } catch (const std::exception& error) {
        rethrown = std::string(error.what()).find("injected walk failure") != std::string::npos;
    }
    Require(rethrown, "the first range offered after a failed walk did not rethrow that failure");
    Require(!walker.Offer(epoch, address, Page), "a walker that failed took a range");
    const auto walks = osWalks.load();
    Require(GuestMemory::CollectWrites(address, Page) != 0 && osWalks.load() > walks, "a range whose walk failed on the walker was not walked by the thread that collected it");
    walker.Stop();
}

}

int main(int argc, char** argv) {
    try {
        if (!GuestMemory::WriteWatched()) {
            std::cout << "collect walker tests skipped: guest memory is not write watched\n";
            return 77;
        }
        const auto base = AllocateWatched(static_cast<std::size_t>(Blocks * Block));
        if (!GuestMemory::Watched(base, static_cast<std::size_t>(Blocks * Block))) {
            std::cout << "collect walker tests skipped: the test block is not write watched\n";
            return 77;
        }
        GuestMemory::SetObservationHookForTest(&Hook);
        if (argc == 2 && std::string(argv[1]) == "--failure") {
            FailedWalkReachesTheNextOffer(base);
        } else {
            WalkAnswersTheEpochItWasHandedIn(base);
            FullQueueLeavesRangesToTheCollector(base);
            CollectUnderTheGpuLockWaitsForTheWalk(base);
            StoppedWalkerTakesNothing(base);
        }
        std::cout << "collect walker tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        ReleasePause();
        std::cerr << error.what() << '\n';
        return 1;
    }
}
